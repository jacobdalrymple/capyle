# Name: NAME
# Dimensions: 2

# --- Set up executable path, do not edit ---
import sys
import inspect
this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')
# ---

import numpy as np
import capyle.utils as utils
from capyle.ca import Grid2D, Neighbourhood, randomise2d

def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest Fires"
    config.dimensions = 2
    config.wrap = False

    # 0 = BURNT OUT
    # 1 = BURNABLE
    # 2 = ON FIRE
    # 3 = NOT BURNABLE
    config.states = (0, 1, 2, 3)
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    config.state_colors = [(0.1, 0.1, 0.1), (0, 1, 0), (1, 0, 0), (0, 0, 1)]

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config

##Compute fireability for the grid
def is_firable(grid, N_ga,NE_ga, E_ga, SE_ga, S_ga, SW_ga, W_ga, NW_ga, N, E, S, W,NE, SE, NW, SW, grid_attribs):
    print(N_ga[1][1])

    fireable = grid == 1

    N_fire = N == 2
    E_fire = E == 2
    S_fire = S == 2
    W_fire = W == 2

    NE_fire = NE == 2
    NW_fire = NW == 2
    SE_fire = SE == 2
    SW_fire = SW == 2

    # fire_close = (N == 2) | (E == 2) | (W == 2) | (S == 2)
    # fire_far   = (NW == 2) | (NE == 2) | (SW == 2) | (SE == 2)
    
    fire_close  = N_fire | E_fire | S_fire | W_fire
    fire_far = NE_fire | NW_fire | SE_fire | SW_fire

    fire_N_grid_attribs = N_ga[N_fire]
    fire_E_grid_attribs = E_ga[E_fire] 
    fire_S_grid_attribs = S_ga[S_fire]
    fire_W_grid_attribs = W_ga[W_fire]

    fire_NE_grid_attribs = NE_ga[NE_fire]
    fire_NW_grid_attribs = NW_ga[NW_fire]
    fire_SW_grid_attribs = SW_ga[SW_fire]
    fire_SE_grid_attribs = SE_ga[SE_fire]

    #height, wind, rate_of_flam, humidity, fuel
    height_diff_N = grid_attribs[:,:,0] - fire_N_grid_attribs[:,0] 
    height_diff_S = grid_attribs[:,:,0] - fire_E_grid_attribs[:,0]
    height_diff_E = grid_attribs[:,:,0] - fire_S_grid_attribs[:,:,0]
    height_diff_W = grid_attribs[:,:,0] - fire_W_grid_attribs[:,:,0]

    print(height_diff_N)
    print(height_diff_S)
    print(height_diff_E)
    print(height_diff_W)

    # neighbour_on_fire = fire_close | fire_far
    # fire_close_grid_attribs =  N_ga
    # neighbours_on_fire_height = grid_attribs[neighbour_on_fire][:,:,3]
    #height_diff = grid_attribs[:,:,3] - neighbour_on_fire  
    

### Vectorised function to reduce fuel based on 5 property arrays given
def reduce_fuel(height, wind, rate_of_flam, humidity, fuel, ):
    # with_spare_fuel = (fuel - rate_of_flam) >= 0
    # fuel[with_spare_fuel] = np.around(fuel[with_spare_fuel] - rate_of_flam[with_spare_fuel], 3)
    fuel = (fuel - rate_of_flam).clip(min=0)
    return np.array([height, wind, rate_of_flam, humidity, fuel]).T


def transition_function(grid, neighbourstates, neighbourcounts, grid_attribs):
    """Function to apply the transition rules
    and return the new grid"""

    on_fire = grid == 2
    fireable = grid == 1

    cells_grid_attribs_on_fire = grid_attribs[on_fire]
    

    N_grid_attribs = np.roll(grid_attribs, 1)
    S_grid_attribs = np.roll(grid_attribs, -1)
    E_grid_attribs = np.rollaxis(grid_attribs, 1, 1)
    W_grid_attribs = np.rollaxis(grid_attribs, 1, -1)

    NW_grid_attribs = np.rollaxis(N_grid_attribs, 1, -1)
    NE_grid_attribs = np.rollaxis(N_grid_attribs, 1, 1)
    SW_grid_attribs = np.rollaxis(S_grid_attribs, 1, -1)
    SE_grid_attribs = np.rollaxis(S_grid_attribs, 1, 1)


    NW, N, NE, W, E, SW, S, SE = neighbourstates

    is_firable(grid, N_grid_attribs, NE_grid_attribs, E_grid_attribs, SE_grid_attribs, S_grid_attribs, SW_grid_attribs, W_grid_attribs, NW_grid_attribs, N, E, S, W,NE, SE, NW, SW,  grid_attribs)

    print("neighbourstates")
    print(neighbourstates[0][0])

    print("neighbourstates.shape")
    print(neighbourstates.shape)
    neighboursTransposed = neighbourstates.T
    print(neighboursTransposed[0][0])
    # print(N.shape)
    print("neighboursTransposed.shape")
    print(neighboursTransposed.shape)

    fire_close = (N == 2) | (E == 2) | (W == 2) | (S == 2)
    fire_far   = (NW == 2) | (NE == 2) | (SW == 2) | (SE == 2)
    neighbour_on_fire = fire_close | fire_far

    # print( [N for cell in N] )

    cells_grid_attribs_neighbours_fireable = grid_attribs[neighbour_on_fire]

    print("\n\n\n cells_grid_attribs_neighbours_fireable.shape")
    print(cells_grid_attribs_neighbours_fireable.shape)

    firable_with_on_fire_neighbours = fireable & neighbour_on_fire
    print(neighbour_on_fire.shape)
    cells_grid_attribs_fireable = grid_attribs[firable_with_on_fire_neighbours]
    
    firable_sub_set = grid[firable_with_on_fire_neighbours]
    


    grid[firable_with_on_fire_neighbours] = 2 
    
    grid_attribs[on_fire] = reduce_fuel(
        cells_grid_attribs_on_fire[:, 0], cells_grid_attribs_on_fire[:, 1], cells_grid_attribs_on_fire[:, 2], cells_grid_attribs_on_fire[:, 3], cells_grid_attribs_on_fire[:, 4])


    print(grid_attribs.shape)

    burnt_out_mask = grid_attribs[:,:,4] == 0

    grid[burnt_out_mask] = 0
    # print(not_on_fire)

    # print(not_on_fire.shape)
    # print(grid.shape)
    # print(not_on_fire)
    # grid[not_on_fire] = 0


    # print(cells_grid_attribs_fireable[0].shape)
    # red_fuel = np.vectorize(reduce_fuel,  otypes=[np.float64])

    # cells_grid_attribs_on_fire = red_fuel(*cells_grid_attribs_on_fire)
    # grid_attribs[on_fire] = cells_grid_attribs_on_fire
    # Update

    # print(neighbourstates)
    # print(neighbourstates.shape)
    # NW, N, NE, W, E, SW, S, SE = neighbourstates
    # print("NW")

    # print(NW.shape)

    # in_state_3 = (grid == 3) # cells currently in state 3
    # all_corners_above_1 = (NW > 1) & (NE > 1) & (SW > 1) & (SE > 1) # corner states > 1
    # print(all_corners_above_1.shape)
    # to_one = in_state_3 & all_corners_above_1 # union the results
    # grid[to_one] = 1

    # g = lambda x: 1 if x > 1 else round(x, 1)
    # prod = lambda x, y: x*y
    # s = lambda x: 1 if x > 0 else 0
    # l = lambda x, y, w, z: [x, y, w, z]

    # #Calculate flammability for that cell
    # add_list = lambda x, y, z, g: s(g)*(x+y+z)

    # #Edge condition catching
    # within_bounds = lambda x, y, z: True if (x+y <= 49) and (x+z <= 49) and (x+y >= 0)  and (x+z >= 0) else False

    # near_steps = np.array([[0,1], [1,0], [0,-1], [-1,0]])
    # dist_steps = np.array([[2,2], [2,-2], [-2,-2], [-2,2]])

    # #Vectorize
    # for i in range(50):
    #     for j in range(50):

    #         #TODO: More efficient way?
    #         near_attribs = [ l(*grid_attribs[ i+steps[0], i+steps[1] ], grid[ i+steps[0], i+steps[1]] ) if within_bounds(i, *steps) else [0,0,0,0] for steps in near_steps]
    #         dist_attribs = [ l(*grid_attribs[ i+steps[0], i+steps[1] ], grid[ i+steps[0], i+steps[1]] )  if within_bounds(i, *steps) else [0,0,0,0] for steps in dist_steps]

    #         near_sum = 0
    #         dist_sum = 0

    #         #How burnt the cell will be
    #         #So if sums of both = 0, then cell is 0
    #         for k in range(4):
    #             near_sum += add_list(*near_attribs[k])
    #             dist_sum += add_list(*dist_attribs[k])

    #         print((near_sum + 0.25*dist_sum))

    #         #Round to state (0 -> 1)
    #         grid[i][j] += g(near_sum + 0.25*dist_sum)

    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    grid_attribs = np.zeros((*config.grid_dims, 5))

    # 0: Height - Scalar value
    # 1: Wind/Magnitude - East to West
    # 2: Flammability
    # 3: Humidity?
    # 4: Fuel
    grid_attribs[...] = (0, 0.1, 0.1, 0, 1)

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, (transition_function, grid_attribs))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
