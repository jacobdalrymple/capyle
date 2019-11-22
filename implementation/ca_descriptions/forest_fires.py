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

    #config.states = (0,1,2,3,4,5,6,7,8,9,10)
    config.states = (0, 1, 2, 3)
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    config.state_colors = [(0, 1, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),
                           (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (0, 0, 0)]

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config


### Vectorised function to reduce fuel based on 5 property arrays given
def reduce_fuel(height, wind, rate_of_flam, humidity, fuel):

    with_spare_fuel = (fuel - rate_of_flam) >= 0
    fuel[with_spare_fuel] = np.around(fuel[with_spare_fuel] - rate_of_flam[with_spare_fuel], 3)
    return np.array([height, wind, rate_of_flam, humidity, fuel]).T


def transition_function(grid, neighbourstates, neighbourcounts, grid_attribs):
    """Function to apply the transition rules
    and return the new grid"""
    # unpack the state arrays

    #Generate a mask of those on fire
    on_fire = grid == 2

    #Generate a mask of those not on fire, but can be
    fireable = grid == 1


    #The actual cell state values that are currently on fire
    #cells_on_fire = grid[on_fire]
    #The actual cell state values that are able to be on fire, but
    #not yet
    #cells_fireable = grid[fireable]

    #The attribs for those cells on fire
    cells_grid_attribs_on_fire = grid_attribs[on_fire]
    
    #The attribs for those not on fire yet, but can be
    cells_grid_attribs_fireable = grid_attribs[fireable]

    # print("ON_FIRE")
    # print(on_fire)
    # print(on_fire.shape)

    # print("\n\n\ CELLS_ON_FIRE")
    # print(cells_on_fire)
    # print(cells_on_fire.shape)

    # print("\n\n\n\n")
    # print(grid_attribs.shape)
    #print(cells_grid_attribs_on_fire.shape)
    # print("\n\n\ Average fuel of on fire elements:")
    # print(np.average(cells_grid_attribs_on_fire[:, 4]))

    #Calculate the reduction of fuel for all those elements that
    #are currently on fire, and set all those on fire to their newly computed values
    grid_attribs[on_fire] = reduce_fuel(
        cells_grid_attribs_on_fire[:, 0], cells_grid_attribs_on_fire[:, 1], cells_grid_attribs_on_fire[:, 2], cells_grid_attribs_on_fire[:, 3], cells_grid_attribs_on_fire[:, 4])


    print(grid_attribs.shape)

    #Get a mask of all the cells that have no fuel.
    burnt_out_mask = grid_attribs[:,:,4] == 0
    #print(burnt_out_mask.shape)
    #print(grid.shape)

    #Use that to set all of those to 0, the burnt out state.
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

    # 0: Height
    # 1: Wind/Magnitude
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
