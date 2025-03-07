#############################################################
# this file contains code to route SNNs on MOSAIC           #
#                                                           # 
#      Vienne la nuit sonne l'heure                         #
#      Les jours s'en vont je demeure                       #
#                                                           #
#                                                           #
#############################################################
# TODOs:
# -TODO: time benchmark this router, might need some optimization, and threading

import jax
import jax.numpy as np
import numpy as onp

try : 
    from mosaic.routing.MOSAIC import Mosaic_config
except ModuleNotFoundError : 
    from routing.MOSAIC import Mosaic_config

#   Placement of blocks on MOSAIC
#   inputs : 
#       - id of a Neuron Tile (NT)
#       - string : 
#           * "full" : coordinates with routers on the grid
#           * "NT" : coordinates on a NT only grid 
#       - config : Mosaic_config
#   output : 
#       - 2D coordinates 
#   
#   we follow a western reading identation :  
#
#   0 -> 1 -> 2
#    _________|
#    |
#   3 -> 4 -> 5    
#
def placement(id, str, config : Mosaic_config) : 
    original_coord = onp.array([ id//config.NT_shape[1] , id%config.NT_shape[1]])
    if(str == "NT") : 
        return original_coord
    elif(str == "full") : 
        return 2 * (original_coord) + 1 
    

#   Router constraints for a given config
#   inputs : 
#       - a mosaic config
#   output : 
#       - a 2D array of cardinal constraints (i, j, 8) (4 input, 4 outputs) we follow a (in_NESW, out_NESW) order 
#   Here we work under the assumption of symetric router
def make_router_constraints(config : Mosaic_config) : 
    #rotate_right 90 degrees
    def rotate_90(constraint_vector) : 
        output = onp.zeros(shape=(8))
        output[0] = constraint_vector[3]
        output[1:4] = constraint_vector[:3]
        output[4] = constraint_vector[7]
        output[5:] = constraint_vector[4:7]
        return output

    def symetry (constraint_vector) : 
        return onp.all(rotate_90(rotate_90(constraint_vector)) == constraint_vector)

    Nt = config.constraints_Nt[0]
    R_1 = config.constraints_R_1
    R_1_bis = rotate_90(R_1)
    R_0 = config.constraints_R_0

    assert symetry(Nt)
    assert symetry(R_1)
    assert symetry(R_0)

    n = config.NT_shape[0] * 2 + 1
    constraints = onp.zeros(shape=(n, n, 8))

    #fill our matrix
    for x in range(config.NT_shape[0]) : 
        for y in range(config.NT_shape[1]) :
            x_nt = 2 * x + 1
            y_nt = 2 * y + 1
            constraints[x_nt, y_nt, :] = Nt 
            constraints[x_nt-1, y_nt, :] = R_1
            constraints[x_nt, y_nt-1, :] = R_1_bis
            constraints[x_nt+1, y_nt, :] = R_1
            constraints[x_nt, y_nt+1, :] = R_1_bis
            constraints[x_nt+1, y_nt+1, :] = R_0
            constraints[x_nt-1, y_nt-1, :] = R_0 
            constraints[x_nt-1, y_nt+1, :] = R_0 
            constraints[x_nt+1, y_nt-1, :] = R_0 

    return constraints


#   the 1 turn routing tree compute function
#   takes coordinates of source and destination and returns a map couting the number of used input and outputs. 
#   uses a 1 turn routing scheme for simplicity, doesn't optimize for shared paths
#   inputs : 
#       - source coords
#       - dest coords list
#   output : 
#       -described 3D (i, j, 8) map 
#
# TODOs:
# -TODO: test this
# -TODO: implement a better routing algorithm
def one_turn_routing_tree(source_coord, list_dest_coords, config : Mosaic_config) : 

    def NT_distance(a, b) : 
        A = onp.array(a)
        B = onp.array(b)
        return onp.sum(onp.abs(A - B)) 

    def direction(source, dest) : 
        if(dest[0] > source[0]) : 
            return "S"
        elif(dest[0] < source[0]) : 
            return "N"
        elif(dest[1] > source[1]) : 
            return "E"
        else :
            return "W"
        
    def d_coord(dir) : 
        if(dir == "N") : 
            return ([0, 0, 4], [-1, 0, 2])
        elif(dir == "S") : 
            return ([0, 0, 6], [1, 0, 0])
        elif(dir == "E") : 
            return ([0, 0, 5], [0, 1, 3])
        elif(dir == "W") : 
            return ([0, 0, 7], [0, -1, 1])        
        
    def fill_line(start, stop, map) : 
        dir = direction(start, stop)
        distance = onp.sum(onp.abs(onp.array(stop) - onp.array(start)))
        current_position = start
        for _ in range(distance) : 
            d_out, d_in = d_coord(dir)
            map[current_position[0] + d_out[0], current_position[1] + d_out[1], d_out[2]] += 1
            map[current_position[0] + d_in[0] , current_position[1] + d_in[1] , d_in[2] ] += 1
            current_position = (current_position[0] + d_in[0], current_position[1] + d_in[1])
        return map
    
    map = onp.zeros(shape= tuple(config.full_shape) + (8,) )
    for dest_coord in list_dest_coords :
        #TODO : get rid of NT RNN ... this is bad code... 
        if((dest_coord == source_coord).all()) : 
            continue 

        #step 0 : 1D routing 
        if(NT_distance(source_coord, dest_coord) == 2) : # we are in full coords (in_NESW, out_NESW)
            dir = direction(source_coord, dest_coord) 
            if(dir == "N") : 
                map[source_coord[0]-1, source_coord[1], 2] += 1
                map[source_coord[0]-1, source_coord[1], 4] += 1
                map[source_coord[0]-2, source_coord[1], 2] += 1
                
            elif(dir == "S") :
                map[source_coord[0]+1, source_coord[1], 0] += 1
                map[source_coord[0]+1, source_coord[1], 6] += 1
                map[source_coord[0]+2, source_coord[1], 0] += 1

            elif(dir == "E") :
                map[source_coord[0], source_coord[1]+1, 3] += 1
                map[source_coord[0], source_coord[1]+1, 5] += 1
                map[source_coord[0], source_coord[1]+2, 3] += 1

            elif(dir == "W") :
                map[source_coord[0], source_coord[1]-1, 1] += 1
                map[source_coord[0], source_coord[1]-1, 7] += 1
                map[source_coord[0], source_coord[1]-2, 1] += 1

            continue 

        #step 1 : choose output router 
            #- up down if we are not  verticaly aligned 
            #- side if not 
        if(source_coord[1] == dest_coord[1]) : 
            map[source_coord[0], source_coord[1] -1, 1] += 1
            turning_point = (source_coord[0], source_coord[1] - 1)
        else : 
            if(dest_coord[0] > source_coord[0]) : 
                map[source_coord[0] + 1, source_coord[1], 0] += 1
                turning_point = (source_coord[0] + 1, source_coord[1])
            else : 
                map[source_coord[0] - 1, source_coord[1], 2] += 1
                turning_point = (source_coord[0] - 1, source_coord[1])

        #step 2 : fill horizontal line up to tunring point 
            #- if no horizontal move do nothing 
        
        if(source_coord[1] != dest_coord[1]) : 
            if(source_coord[1] > dest_coord[1]) : 
                stop_col = dest_coord[1] + 1 
            else : 
                stop_col = dest_coord[1] - 1
            stop_point = (turning_point[0], stop_col)
            map = fill_line(turning_point, stop_point, map)
            turning_point = stop_point

        #step 3 : fill vertical routers 
            #- fill until router to correct height 
        if(source_coord[1] == dest_coord[1]) : 
            left_dest = (dest_coord[0], dest_coord[1] - 1)
            map = fill_line(turning_point, left_dest, map)
            turning_point = left_dest
        else : 
            stop_point = (dest_coord[0], turning_point[1])
            map = fill_line(turning_point, stop_point, map)
            turning_point = stop_point

        #step 4 : fill last output or router, and NT input (in_NESW, out_NESW)
        dir = direction(turning_point, dest_coord)
        if(dir == "S") :
            map[turning_point[0], turning_point[1], 6] += 1
            map[turning_point[0]+1, turning_point[1], 0] += 1
        elif(dir == "N") : 
            map[turning_point[0], turning_point[1], 4] += 1
            map[turning_point[0]-1, turning_point[1], 2] += 1
        elif(dir == "E") : 
            map[turning_point[0], turning_point[1], 5] += 1
            map[turning_point[0], turning_point[1]+1, 3] += 1
        elif(dir == "W"): 
            map[turning_point[0], turning_point[1], 7] += 1
            map[turning_point[0], turning_point[1]-1, 1] += 1

    return map > 0



#   a function to generate a report on routing
#
#   returns : 
#       - a plot of the routing on MOSAIC
#       - a metric for average usage of routers
#       - a minimum router configuration (min NT, min R_1, min R_0) those are shape=(8) flowing the nomenclature : (in_NESW, out_NESW)
#
#         Nt  | R1
#       ______|______
#             |
#         R1  | R0
#
# includes 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def report_routing(router_use, router_constraints, config):
    plt.clf()
    avg_router_usage = np.average(np.divide(router_use, router_constraints))
    #print("avg router usage : " + str(avg_router_usage))

    # Calculate usage matrix
    usage_matrix = np.max(np.divide(router_use, router_constraints), axis=2)
    max_usage = max(np.max(usage_matrix), 1)  # Ensure max_usage is at least 1
    
    # Setup the colormap based on the maximum usage
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])
    boundaries = [0, 1, max_usage] if max_usage > 1 else [0, 1]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True) if max_usage > 1 else mcolors.Normalize(vmin=0, vmax=1)
    
    # Use matshow to plot the usage matrix, keeping a reference to the mappable object it returns
    mappable = plt.matshow(usage_matrix, cmap=cmap, norm=norm)
    
    # Add custom text and patches if needed
    for x in range(usage_matrix.shape[0]):
        for y in range(usage_matrix.shape[1]):
            if x % 2 != 0 and y % 2 != 0:  # Check for odd indices
                plt.text(y, x, 'NT', ha='center', va='center', color='blue')  # Marking with 'NT'
                plt.gca().add_patch(plt.Rectangle((y - 0.5, x - 0.5), 1, 1, fill=False, edgecolor='black', lw=2))

    # Attach the colorbar to the mappable object
    plt.colorbar(mappable)
    #plt.savefig("router_usage.png")

    #connect output of neuron tile to the input of surrounding Router 1
    for x in range(config.NT_shape[0]) : 
        for y in range(config.NT_shape[1]) :
            x_nt = 2 * x + 1
            y_nt = 2 * y + 1
            router_use[x_nt - 1, y_nt, 2] = config.NT_size
            router_use[x_nt + 1, y_nt, 0] = config.NT_size
            router_use[x_nt, y_nt - 1, 1] = config.NT_size
            router_use[x_nt, y_nt + 1, 3] = config.NT_size

    #extract the router usage for the Nts, R_0, R_1, R_2
    Nt_usage  = router_use[1::2, 1::2, :]
    R_0_usage = router_use[::2, ::2 , :]
    R_1_usage_0 = router_use[1::2, ::2, :]
    R_1_usage_1 = router_use[::2, 1::2, :]
    # extract the most used side on each type of router
    N_t_max = Nt_usage.max(axis=(0, 1))
    R_0_max = R_0_usage.max(axis=(0, 1))

    tmp = R_1_usage_0.max(axis=(0, 1))
    def permute_right(a):
        length = a.shape[0]
        b = np.zeros(shape=(length,))
        b = b.at[0].set(a[-1])
        b = b.at[1:].set(a[:-1])
        return b
    tmp = [permute_right(tmp[0:4]), permute_right(tmp[4:8])]
    tmp = np.concatenate(tmp)

    R_1_max = jax.numpy.stack([R_1_usage_1.max(axis=(0, 1)), tmp], axis=0).max(axis=0)

    min_router_config = [N_t_max, R_0_max, R_1_max]

    return [plt, avg_router_usage, min_router_config]



#   1-turn router on MOSAIC 
#   inputs : 
#       - Weigths of a network
#       - A Mosaic config
#   outputs : 
#       if routing is impossible : 
#           - a False, a report on the routing, the number of router out of spec
#       if correct : 
#           - a True, a report on the routing, 0    
#
# TODOs : 
# -TODO: look at jitability
def route_MOSAIC(W_th, config : Mosaic_config) : 

    #ignore inside NT connections. 
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
    W_th = np.where(create_block_diagonal_matrix([np.ones(shape=(config.NT_size, config.NT_size))] * config.NT_num).astype('bool'), np.zeros(shape=W_th.shape), W_th)

    # router scoreboard is a 2D map of MOSAIC, containing 1D information : (Num_inputs, Num_outputs)
    router_scoreboard = onp.zeros(shape=tuple(config.full_shape) + (8,))
    destination_seg_ids = onp.array([[i] * config.NT_size for i in range(config.NT_num)]).flatten()
    routing_tree_scoreboard = onp.zeros(shape=(config.NT_num * config.NT_size, 4))

    for neuron_source_id in range(W_th.shape[0]) : 
        # check if that activity leads somewhere 
        active = (np.sum(W_th[neuron_source_id, :]) >= 1) 

        if not active : 
            continue
        else : 
            
            #get the destinations ids 
            neuron_to_NTs = jax.ops.segment_sum(W_th[neuron_source_id, :], segment_ids=destination_seg_ids, num_segments=config.NT_num)
            destination_NT_ids = np.nonzero(neuron_to_NTs)[0]
            source_NT_id = neuron_source_id//config.NT_size
            source_NT_coords = placement(source_NT_id, "full", config)

            list_dest_NT_coords = []
            #TODO: use vmap to get a better implem
            for destination in destination_NT_ids : 
                list_dest_NT_coords.append(placement(destination, "full", config))
            routed = one_turn_routing_tree(source_NT_coords, list_dest_NT_coords, config) 
            used = np.sum(routed, axis=2)>0 
            num_R0 = np.sum(used[::2, ::2])
            num_Nt = np.sum(used[1::2, 1::2])
            num_R1 = np.sum(used[1::2, ::2]) + np.sum(used[::2, 1::2])

            router_scoreboard = router_scoreboard + routed
            routing_tree_scoreboard[neuron_source_id, :] = onp.array([1, num_R0, num_R1, num_Nt])

    router_constraints = make_router_constraints(config)

    constraint_violations = np.sum(router_scoreboard > router_constraints)

    report = report_routing(router_scoreboard, router_constraints, config) + [routing_tree_scoreboard]

    if(constraint_violations == 0) : 
        #print("Routing successfull") 
        return True, report, 0
    else :
        #print("Routing fail")
        #print("Current number of maxed routers : " + str(constraint_violations))
        return False, report, constraint_violations
    

#test code
if __name__ == "__main__" : 
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def th(w, config : Mosaic_config): 
        return (abs(w)>config.w_thr).astype(np.float32)
    th_vmap = jax.vmap(th, in_axes=(0, None))
    th_matrix = jax.vmap(th_vmap, in_axes=(1, None))

    config = Mosaic_config(16, (4, 4), 5, 0.1)
    config.set_router_constraints((5, 5, 5, 5, 5, 5, 5, 5), (1, 1, 1, 1, 1, 1, 1, 1), (5, 5, 5, 5, 8, 8, 8, 8))
    config.set_th_func(th_matrix)

    W = onp.zeros(shape=(80, 80))
    W[0, 2] = 1 # NT0 -> NT0
    W[0, 5] = 1 # NT0 -> NT1
    W[0, 20] = 1 # NT0 -> NT4

    from time import time

    start = time()
    res, report, _ = route_MOSAIC(W.astype('bool'), config)
    stop=time()
    print("routing duration : " + str(stop -start))
    report[0].savefig("data/routing_test.png")
    print(report[2][2])