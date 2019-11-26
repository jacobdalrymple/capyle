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


near_translations = [[0, 1], [-1, 0], [0, -1], [1, 0]]
far_translations = [[1, 1], [1, -1], [-1, -1], [-1, 1]]


def within_bounds(col, row): return col < 100 and row < 100


def neighbours_on_fire(row, col, neighbours):

    NW, N, NE, W, E, SW, S, SE = neighbours
    fire_close = list()
    fire_far = list()

    if (N == 2):
        fire_close.append((row+1, col))
    if (E == 2):
        fire_close.append((row, col+1))
    if (W == 2):
        fire_close.append((row, col-1))
    if (S == 2):
        fire_close.append((row-1, col))

    if (NW == 2):
        fire_far.append((row+1, col-1))
    if (NE == 2):
        fire_far.append((row+1, col+1))
    if (SW == 2):
        fire_far.append((row-1, col-1))
    if (SE == 2):
        fire_far.append((row-1, col+1))

    return fire_close, fire_far

# Compute fireability for the grid


def ignites(cell_states, cells_attribs, neighbours):

    for row in range(len(cell_states)):
        for col in range(len(cell_states[:])):

            ignition_prob = 0

            fire_close, fire_far = neighbours_on_fire(
                row, col, neighbours[row][col])

            if len(fire_close) >= 1:
                for row1, col1 in fire_close:
                    rate_of_flam = cells_attribs[row1-1, col1-1, 2]
                    ignition_prob += 0.5 * rate_of_flam

            if len(fire_far) >= 1:
                for row2, col2 in fire_far:
                    rate_of_flam = cells_attribs[row2-1, col2-1, 2]
                    ignition_prob += 0.25 * rate_of_flam

            if ignition_prob > 0.6:
                cell_states[row][col] = 2

    return cell_states

# Vectorised function to reduce fuel based on 5 property arrays given


def reduce_fuel(height, wind, rate_of_flam, humidity, fuel):
    # with_spare_fuel = (fuel - rate_of_flam) >= 0
    # fuel[with_spare_fuel] = np.around(fuel[with_spare_fuel] - rate_of_flam[with_spare_fuel], 3)
    fuel = (fuel - rate_of_flam).clip(min=0)
    return np.array([height, wind, rate_of_flam, humidity, fuel]).T


def ignite(height, wind, rate_of_flam, humidity, fuel, on_fire_neighbours):

    wind_prob = np.interp(wind, (wind.min(), wind.max()), (0, 0.5))
    height_prob = np.interp(height, (height.min(), height.max()), (0, 0.5))

    prob = on_fire_neighbours*(wind_prob + height_prob + rate_of_flam)
    normalised_prob = np.interp(prob, (prob.min(), prob.max()), (0, 1))

    return (normalised_prob > 0.5).astype(int)+1


def transition_function(grid, neighbourstates, neighbourcounts, grid_attribs):
    """Function to apply the transition rules
    and return the new grid"""

    fireable = grid == 1

    cells_grid_attribs_fireable = grid_attribs[fireable]

    # neighbours_of_onfire_cells = neighbourcounts[fireable]
    # neighbours_of_onfire_cells[2]
    # grid = ignites(grid, grid_attribs, neighbourstates.T)
    grid[fireable] = ignite(cells_grid_attribs_fireable[:, 0],
                            cells_grid_attribs_fireable[:, 1],
                            cells_grid_attribs_fireable[:, 2],
                            cells_grid_attribs_fireable[:, 3],
                            cells_grid_attribs_fireable[:, 4],
                            neighbourcounts[2][fireable]) if grid[fireable].size > 0 else grid[fireable]
    # print("res.shape")
    # print(res.shape)

    NW, N, NE, W, E, SW, S, SE = neighbourstates

    fire_close = (N == 2) | (E == 2) | (W == 2) | (S == 2)
    fire_far = (NW == 2) | (NE == 2) | (SW == 2) | (SE == 2)
    neighbour_on_fire = fire_close | fire_far

    cells_at_fire_risk = neighbour_on_fire & fireable
    grid[cells_at_fire_risk] = 2

    on_fire = grid == 2

    cells_grid_attribs_on_fire = grid_attribs[on_fire]

    grid_attribs[on_fire] = reduce_fuel(cells_grid_attribs_on_fire[:, 0],
                                        cells_grid_attribs_on_fire[:, 1],
                                        cells_grid_attribs_on_fire[:, 2],
                                        cells_grid_attribs_on_fire[:, 3],
                                        cells_grid_attribs_on_fire[:, 4])

    burnt_out = grid_attribs[:, :, 4] == 0
    grid[burnt_out] = 0

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
