#############################################################
# this file contains code to route SNNs on MOSAIC           #
#                                                           # 
#                                                           #
#                                                           #
#############################################################


class Mosaic_config :
    def __init__(self, NT_num, NT_shape, NT_size, w_thr, w_gain=1, alpha=1, n_out=20, prune_nbr=20) -> None:
        '''
        w_thr : Pruning threshold for DEEPR
        w_gain*alpha : std of the weight at initialization
        w_gain : distance to treshold at initialization
        alpha : distancet to threshold at training
        n_out : number of output neurons
        prune_nbr : number of minimum connections to neuron in a NT. Prune a connection if the number of neuron in a NT is less than prune_nbr
        '''
        self.w_thr = w_thr
        self.NT_num = NT_num
        self.NT_size = NT_size
        self.n_rec = self.NT_num * self.NT_size
        self.n_core=NT_num
        self.NT_shape = NT_shape
        self.full_shape = (2 * NT_shape[0] + 1, 2 * NT_shape[1] + 1)
        self.w_gain = w_gain
        self.alpha = alpha
        self.n_out = n_out
        self.Co_prune_nbr = prune_nbr

    def set_th_func(self, th_func) -> None: 
        self.th_func = th_func

    #here using : 
    #
    #   R_0 | R_1   
    #   ---------
    #     X | N_t    
    #
    # fill the constraints as a pytree following (I_NESW, O_NESW)
    def set_router_constraints(self, Nt, R_0, R_1) -> None : 
        self.constraints_Nt = Nt, 
        self.constraints_R_0 = R_0
        self.constraints_R_1 = R_1

    def distance_mosaic_NT_NT(self) :
        '''
        Give the distance between two NTs in the mosaic
        (Useful to compute the probability)
        '''
        import numpy as onp
        num_NT = self.n_core
        self.n_rec = self.NT_num * self.NT_size
        NT_size = int(self.n_rec/self.n_core)
        MOSAIC_size_nt = onp.sqrt(num_NT)

        def placement(nt_id):
            return onp.array([nt_id%MOSAIC_size_nt, nt_id//MOSAIC_size_nt])

        W_mask = onp.zeros(shape=(num_NT, num_NT))
            
        for i in range(num_NT) : 
            for j in range(num_NT) :
                source_placement = placement(i)
                dest_placement = placement(j)
                distance = onp.sum(onp.abs(dest_placement - source_placement))
                if(i == j) : 
                    W_mask[i, j] = 0
                elif(distance == 1) : 
                    W_mask[i, j] = 1 
                elif(source_placement[0] == dest_placement[0] or source_placement[1] == dest_placement[1]) :
                    W_mask[i, j] = 2 * distance + 1
                else : 
                    W_mask[i, j] = 2 * distance  - 1
        return W_mask

import numpy as np


if __name__ == "__main__" :
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    np.set_printoptions(threshold = np.inf)

    from MOSAIC import *
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


