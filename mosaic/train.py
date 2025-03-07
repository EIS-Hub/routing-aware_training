import os
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from jax import vmap, pmap, jit, value_and_grad, local_device_count
from jax.lax import scan, cond
import jax.numpy as np
import jax.random as random
try : 
    from mosaic.utils import calc_mosaic_stats, pruner, add_noise
    from mosaic.routing.MOSAIC import *
    from mosaic.DEEP_R import *
except ModuleNotFoundError : 
    from utils import calc_mosaic_stats, pruner, add_noise
    from routing.MOSAIC import *
    from mosaic.DEEP_R import * 
import numpy as onp
import wandb
import optax

matplotlib.use('Agg')

def train_mosaic(key, n_batch, n_inp, n_rec, n_out, thr_rec, tau_rec, 
                 lr, lr_dropstep, w_gain, grad_clip, train_dl, test_dl, val_dl,
                 model, param_initializer, sparser, decoder, 
                 noise_start_step, prune_start_step, prune_thr, 
                 noise_std, n_epochs, n_core, W_mask, lambda_con,
                 target_fr, lambda_fr, dataset_name, alpha, epsilon_adam, prune_nbr, 
                 co_profile
                 ):
    
    # Testing Loop
    shd_valid_loader = val_dl
    if dataset_name == 'shd':
        shd_test_loader = test_dl
    elif dataset_name == 'ssc':
        ssc_test_loader = test_dl
    elif dataset_name == 'all':
        shd_test_loader, ssc_test_loader = test_dl

    config=Mosaic_config(n_core, (int(onp.sqrt(n_core)), int(onp.sqrt(n_core))), int(n_rec/n_core), prune_thr, w_gain, alpha, n_out)
    config.protection_mask = protection_mask(config)
    config.co_profile = co_profile
    
    key, key_model = random.split(key, 2)
    n_devices = local_device_count()

    def net_step(net_params, x_t):
        ''' Single network inference (x_t -> yhat_t)
        '''
        net_params, z_rec = model(net_params, x_t)
        return net_params, z_rec

    @jit
    def predict(weight, X):
        """ Scans over time and return predictions. """
        _, net_const, net_dyn = param_initializer(
            key, n_inp, n_rec, n_out, n_core, thr_rec, tau_rec, w_gain
        )
        _, z_rec = scan(net_step, [weight, net_const, net_dyn], X, length=100)
        z_rec = rearrange(z_rec, 't o -> o t') 
        Yhat = decoder(z_rec[-n_out:]) 
        return Yhat, z_rec

    v_predict = vmap(predict, in_axes=(None, 0))
    p_predict = pmap(v_predict, in_axes=(None, 0))

    def loss(key, weight, W_co, X, Y, epoch):
        """ Calculates CE loss after predictions. """
        X = rearrange(X, '(d b) t i -> d b t i', d=n_devices)
        Y = rearrange(Y, '(d b) o -> d b o', d=n_devices)

        Yhat, z_rec = p_predict(np.multiply(weight, W_co), X)
        
        num_correct = np.sum(np.equal(np.argmax(Yhat, 2), np.argmax(Y, 2)))
        true_pos = np.sum(jax.nn.one_hot(np.multiply(np.argmax(Yhat, 2)+1, np.equal(np.argmax(Yhat, 2), np.argmax(Y, 2))), 21)[:, :, 1:], axis=1)
        truth = np.sum(jax.nn.one_hot(np.argmax(Y, 2), 20), axis=1)
        pos = np.sum(jax.nn.one_hot(np.argmax(Yhat, 2), 20), axis=1)

        recall = np.divide(true_pos, truth)
        precision = np.divide(true_pos, pos)

        loss_ce = -np.mean(np.sum(Yhat * Y, axis=2, dtype=np.float32))
        loss_sp = 0 #sparser(weight, W_mask, config)
        loss_fr = np.mean(target_fr - 10 * np.mean(z_rec)) ** 2 
        loss_total = loss_ce + loss_sp * lambda_con + loss_fr * lambda_fr
        output_firing_rate = np.mean(z_rec[-n_out:])
        loss_values = [num_correct, 10 * np.mean(z_rec), loss_ce, loss_sp, loss_fr, output_firing_rate, recall, precision]
        return loss_total, loss_values
 
    @jit
    def update(key, epoch, weight, W_co, X, Y, opt_state):
        value, grads = value_and_grad(loss, has_aux=True, argnums=(1))(key, weight, W_co, X, Y, epoch)
        grads = np.clip(grads, -grad_clip, grad_clip) * W_co
        updates, opt_state = solver.update(grads, opt_state, weight)
        opt_state = (optax.ScaleByAdamState(
            count = opt_state[0].count, 
            mu = opt_state[0].mu * W_co, 
            nu = opt_state[0].nu * W_co,
        ), opt_state[1])
        #updates = np.clip(updates, -0.05 * w_gain, 0.05 * w_gain )
        weight = optax.apply_updates(weight, updates * W_co)
        return weight, opt_state, value

    def one_hot(x, n_class):
        return np.array(x[:, None] == np.arange(n_class), dtype=np.float32)

    def total_correct(weight, X, Y):
        X = rearrange(X, '(d b) t i -> d b t i', d=n_devices)
        Y = rearrange(Y, '(d b) -> d b', d=n_devices)
        Yhat, _ = p_predict(weight, X)
        acc = np.sum(np.equal(np.argmax(Yhat, 2), Y[0]))
        return acc

    W_co, weight, key, _, distance_Nt_Nt= DEEP_R_init(key, config) 
    polynomial_decay = optax.polynomial_schedule(
        init_value=lr, end_value=lr/10, power = 0.5, 
        transition_steps=n_epochs, transition_begin=int(0.20 * n_epochs)
    )
    solver = optax.adam(learning_rate = polynomial_decay, eps=epsilon_adam)
    opt_state = solver.init(weight)
    
    print("init done")

    plt.matshow(np.abs(weight) > 0)
    plt.savefig("data/W_init.png")
    plt.clf()
    plt.hist(weight[weight != 0].flatten(), bins='auto')
    plt.savefig(f"data/histogram_init")
    plt.clf()

    # Training loop
    train_loss = []
    for epoch in range(n_epochs):
        t = time.time()
        acc = 0
        output_fr_acc = 0
        fr_loss_acc = 0
        recall_acu = np.zeros(shape=(1, 20))
        precision_acu = np.zeros(shape=(1, 20))

        for batch_idx, (x, y) in enumerate(train_dl):
            y = one_hot(y, n_out)
            key, _ = random.split(key)
            
            weight, opt_state, (L, [tot_correct, fr_rec, loss_ce, loss_sp, loss_fr, output_fr, recall, precision]) = update(
                key, epoch, weight, W_co, x, y, opt_state
            )
            assert np.all(np.isfinite(weight))

            train_loss.append(L)
            acc += tot_correct
            output_fr_acc += output_fr
            fr_loss_acc += loss_fr
            recall_acu = recall_acu + recall
            precision_acu = precision_acu + precision

        # Training logs
        train_acc = 100*acc/((batch_idx+1)*n_batch)
        recall_acc = 100*recall_acu/((batch_idx+1))
        precision_acc = 100*precision_acu/((batch_idx+1))
        W_co, weight, key, _, stats = DEEP_R(W_co, weight, config, key, distance_Nt_Nt)
        
        mean_weight = np.mean(np.abs(weight[W_co]))

        elapsed_time = time.time() - t
        print(f'Epoch: [{epoch}/{n_epochs}] - Loss: {L:.2f} - '
              f'Training acc: {train_acc:.2f} - t: {elapsed_time:.2f} sec - Killed: {stats[0]} - mean weights : {mean_weight:.2f}')

        wandb.log({"Train Accuracy" : train_acc, "Learning Rate" : polynomial_decay(epoch), "killed" : stats[0], "born" : stats[1], 
                   "output_firing_rate" : output_fr_acc/(batch_idx+1), "fr loss" : fr_loss_acc/(batch_idx+1),
                   "avg weights" : mean_weight
                   }, step = epoch)
     
        if(epoch%25==0):
            plt.matshow(W_co)
            plt.savefig('data/connectivity_matrix.png')
            wandb.log({"connectivity" : plt}, step = epoch)
            plt.clf()
            plt.matshow(np.abs(weight) > config.w_thr)
            plt.savefig('data/weight_matrix.png')
            wandb.log({"weights" : plt}, step = epoch)
            plt.clf()

        if(epoch%10==0): 
            acc = 0; test_acc = 0 
            for batch_idx, (x, y) in enumerate(shd_test_loader):
                acc += total_correct(weight, x, y)
                test_acc = 100*acc/((batch_idx+1)*n_batch)
            wandb.log({"Test Accuracy" : test_acc}, step = epoch)

    # SHD
    acc = 0; test_acc_shd = 0
    if dataset_name in ['shd', 'all']:
        for batch_idx, (x, y) in enumerate(shd_test_loader):
            acc += total_correct(weight, x, y)
        test_acc_shd = 100*acc/((batch_idx+1)*n_batch)
    print(f'SHD Test Accuracy: {test_acc_shd:.2f}')

    acc = 0; val_acc_shd = 0
    if dataset_name in ['shd', 'all']:
        for batch_idx, (x, y) in enumerate(shd_valid_loader):
            acc += total_correct(weight, x, y)
        val_acc_shd = 100*acc/((batch_idx+1)*n_batch)
    print(f'SHD Validation Accuracy: {val_acc_shd:.2f}')

    # SSC
    acc = 0 ; test_acc_ssc = 0
    if dataset_name in ['ssc', 'all']:
        for batch_idx, (x, y) in enumerate(ssc_test_loader):
            acc += total_correct(weight, x, y)
        test_acc_ssc = 100*acc/((batch_idx+1)*n_batch)
    print(f'SSC Test Accuracy: {test_acc_ssc:.2f}')

    return train_loss, test_acc_shd, test_acc_ssc, val_acc_shd, (W_co, weight)