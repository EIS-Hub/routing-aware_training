import jax
import jax.numpy as np 
from functools import partial

#@partial(jax.jit, static_argnames=['config'])
def DEEP_R_prune(W, W_co_n_Nt, config) :
    #TODO : NT and output protection    
    W_thr = (np.abs(W) >= config.w_thr).astype("bool")

    segment_ids = np.array([[i] * config.NT_size for i in range(config.NT_num)]).flatten()
    Co_n_Nt = jax.ops.segment_sum(np.multiply(np.ones(W_thr.shape), W_thr).T, segment_ids, num_segments=config.NT_num).T

    to_prune = (Co_n_Nt < config.Co_prune_nbr) * W_co_n_Nt * (~config.protection_mask)

    W_co_n_Nt = np.where(to_prune, 0, W_co_n_Nt).astype('bool')
    W_co_n_n = W_co_n_Nt.repeat(config.NT_size, axis=1)

    W = np.where(W_co_n_n, W, 0)
    killed = np.sum(to_prune)

    return W, W_co_n_n, W_co_n_Nt, killed


def DEEP_R_create(key, config, W, W_co_n_Nt, d_Nt_Nt):
    profile_Nt_Nt = config.co_profile
    #Compute the Nt to Nt connections
    segment_ids = np.array([[i] * config.NT_size for i in range(config.NT_num)]).flatten()
    W_co_Nt_Nt = jax.ops.segment_sum(W_co_n_Nt.astype('i2'), segment_ids)
    
    d_co_Nt_Nt = np.multiply(d_Nt_Nt, W_co_Nt_Nt)

    # Compute the connection deficit
    dmax = 2*(config.NT_shape[0] + config.NT_shape[1] -2) -1
    profile_Nt_Nt = np.pad(profile_Nt_Nt, (0, dmax - len(profile_Nt_Nt)))
    to_create = np.zeros(shape=(config.NT_num, dmax))
    for dist in range(1, dmax) : 
        to_create = to_create.at[:, dist].set(np.round(profile_Nt_Nt[dist] * config.NT_size * np.sum(d_Nt_Nt == dist, axis=1)) - np.sum(W_co_Nt_Nt * (d_Nt_Nt == dist), axis=1))
    #list values to create (for perf reason)
    list_to_create = np.nonzero(to_create)

    
    keys = jax.random.split(key, num = len(list_to_create[0]) + 1)
    for index in range(len(list_to_create[0])) :
        #get source Nt and distance to use
        source = list_to_create[0][index]
        dist = list_to_create[1][index]
        #compute the available connections from source at distance
        temp_d_n_NT = np.dot(np.logical_not(W_co_n_Nt[source*config.NT_size:(source+1)*config.NT_size, :]), np.diag(d_Nt_Nt[source, :]))
        Available = np.nonzero(temp_d_n_NT == dist)
        Available = (Available[0] + config.NT_size * source, Available[1])
        #choose connections via index
        Indexes = np.arange(len(Available[0])) 
        Indexes = jax.random.choice(keys[index], Indexes, shape=(int(to_create[source, dist]),), replace=False)
        sources = Available[0][Indexes]
        dest_NT = Available[1][Indexes]
        #plug them and random initialize weights
        W_co_n_Nt = W_co_n_Nt.at[sources, dest_NT].set(True)
        values= config.alpha * config.w_thr * jax.random.normal(keys[index], shape=(1, int(config.NT_size * to_create[source, dist])))
        values = values + np.sign(values) * config.w_thr
        
        for write_index in range(len(sources)) :
            write_NT = dest_NT[write_index] 
            W = W.at[sources[write_index], write_NT*config.NT_size:(write_NT + 1)*config.NT_size ].set(values[0, write_index*config.NT_size:(write_index+1)*config.NT_size])

    born = np.sum(to_create)

    return keys[-1], W, W_co_n_Nt, born

def DEEP_R (W_co_n_n, W, config, key, distance_Nt_Nt, inverse_map=None) :  
    segment_ids = np.array([[i] * config.NT_size for i in range(config.NT_num)]).flatten()
    W_co_n_Nt = jax.ops.segment_sum(W_co_n_n.T.astype('i2'), segment_ids, num_segments=config.NT_num).T
    W, _, W_co_n_Nt, killed = DEEP_R_prune(W, W_co_n_Nt, config)

    key, W, W_co_n_Nt, born = DEEP_R_create(key, config, W, W_co_n_Nt, distance_Nt_Nt)

    W_co = W_co_n_Nt.repeat(config.NT_size, axis=1)

    return W_co, W, key, None, [killed, born]

#full connection inside NTs. 
def create_block_diagonal_matrix(blocks):
    n = len(blocks)       
    N = blocks[0].shape[0]  
    # Create an empty matrix of size nN x nN
    result = np.zeros((n * N, n * N))
    # Insert each block into the result matrix
    for i, block in enumerate(blocks):
        start_idx = i * N
        result = result.at[start_idx:start_idx + N, start_idx:start_idx + N].set(block)
    return result

def DEEP_R_init(key, config) : 
    N = config.NT_num * config.NT_size
    
    key, W, W_co_n_NT, _ = DEEP_R_create(key, config, np.zeros((N , N)), np.zeros((N, config.NT_num), dtype='bool'), config.distance_mosaic_NT_NT())
    W_co = np.concatenate([np.tile(W_co_n_NT[:, i], (config.NT_size, 1)) for i in range(config.n_core)], axis=0).T
    #add FC in Nts
    W_co = W_co | create_block_diagonal_matrix([np.ones(shape=(config.NT_size, config.NT_size))] * config.NT_num).astype('bool')
    keys = jax.random.split(key, num=config.NT_num + 1)
    return_key = keys[-1]
    NTs = []
    for n in range(config.NT_num) : 
        tmp = jax.random.normal(keys[n], shape=(config.NT_size, config.NT_size))*config.w_thr*config.alpha
        NTs.append(tmp + config.w_thr * (tmp >= 0) - config.w_thr * (tmp <0))
    W = W + create_block_diagonal_matrix(NTs)
    D_Nt_Nt = config.distance_mosaic_NT_NT()
    return W_co, W, return_key, None, D_Nt_Nt

#creates a protection mask for the NTs and the output neurons. 
def protection_mask(config) : 
    n_rec = config.NT_size * config.NT_num
    tmp = create_block_diagonal_matrix([np.ones(shape=(config.NT_size, config.NT_size))] * config.NT_num).astype('bool') | np.zeros((n_rec, n_rec)).at[:, -config.NT_size * (config.n_out//config.NT_size + 1):].set(1).astype('bool')
    return tmp[:, ::config.NT_size]

if __name__ == "__main__" : 
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    np.set_printoptions(threshold = np.inf)
    
    from routing.MOSAIC import *
    config = Mosaic_config(25, (5, 5), 32, 0.1)
    config.n_out = 20
    config.n_core = 25
    config.n_rec = 25 * 32
    config.w_gain = 2.5
    config.w_thr = 0.1
    config.co_profile = np.array([0, 1, 0, 0.1], dtype='f2')
    config.Co_prune_nbr = 2
    config.alpha = 2    
    config.protection_mask = protection_mask(config)

    W_co_n_NT= np.zeros((800, 25))
    W = np.zeros((800, 800))
    W= W.at[0, 0:5].set(1)
    W_co_n_NT = W_co_n_NT.at[0, 0].set(True) 
    W_co_n_NT = W_co_n_NT.at[0, 5].set(True)

    from DEEP_R import distance_mosaic_NT_NT

    #_, _, _, killed = DEEP_R_prune(W, W_co_n_NT, config)
    #print(killed)
    #looks like prune works

    #_, W, W_co, born = DEEP_R_create(jax.random.PRNGKey(42), config, W, np.zeros((800, 25)), distance_mosaic_NT_NT(config))
    #print(f"borned : {born}")
    
    W_co, W, _, _, D_Nt_Nt = DEEP_R_init(jax.random.PRNGKey(42), config)

    import matplotlib.pyplot as plt 
    plt.matshow(W_co)
    plt.savefig("data/init_W_co.png")
    plt.clf()
    plt.hist(W[W != 0].flatten(), bins='auto')
    plt.savefig("data/hist_init.png")
    
    #W_co, W, key, _, stats = DEEP_R(W_co, W, config, jax.random.PRNGKey(42), config.distance_mosaic_NT_NT())
    #print(stats)
