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
from math import pi

import time
from math import floor

def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest Fires Updated"
    config.dimensions = 2
    config.wrap = False
    config.num_generations = 500
    # 0 = BURNT OUT
    # 1 = DEFAULT, BURNABLE GRASS
    # 2 = DENSE FOREST
    # 3 = HIGH FLAMMABLE SCRUB
    # 4 = ON FIRE
    # 5 = BUILDINGS
    # 6 = WATER
    config.states = (0, 1, 2, 3, 4, 5, 6)
    # -------------------------------------------------------------------------

    # ---- Override the defaults below (these may be changed at anytime) ----

    config.state_colors = [(0.1, 0.1, 0.1), (0, 1, 0), (0, 0.5, 0), (0.5, 0.8, 0.5), (1, 0, 0), (0.5, 0.5, 0.5), (0, 0, 1) ]

    # ----------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config

def scale(arr, min_, max_):
    return np.interp(arr, (arr.min(),
                                    arr.max()), (min_, max_))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def cal_wind_weight(wind_spread, neighbour_states):

    wind_weight = np.linspace(0, 0, wind_spread.shape[0])

    for i in range(neighbour_states.shape[0]):
        index = neighbour_states[i] == 4
        wind_weight[index] += wind_spread[index,i]

    return wind_weight

def cal_height_weight(height, height_neighbours, neighbour_states):

    height_weight = np.linspace(1, 1, height_neighbours.shape[0])

    for i in range(neighbour_states.shape[0]):
        index = neighbour_states[i] == 4
        height_diff = height[index] - height_neighbours[index,i]
        height_weight[index] += 0.0051 * height_diff

    return height_weight


def ignite(height, rate_of_flam, humidity, fuel, wind_spread, height_neighbours, on_fire_neighbours, neighbour_states):

    wind_weight = cal_wind_weight(wind_spread, neighbour_states)
    height_weight = cal_height_weight(height, height_neighbours, neighbour_states)

    prob = 0.5 * (on_fire_neighbours > 0).astype(int) * rate_of_flam * wind_weight * height_weight * np.random.rand(height.shape[0])

    random = np.random.rand(prob.shape[0])

    return random < prob


# Vectorised function to reduce fuel based on 5 property arrays given
def reduce_fuel(height, rate_of_flam, humidity, fuel, wind_spread, height_neighbours):
    new_fuel = (fuel - rate_of_flam).clip(min=0)
    return np.array([height, rate_of_flam, humidity, new_fuel, *(wind_spread.T), *(height_neighbours.T)]).T


def transition_function(grid, neighbourstates, neighbourcounts, time_to_drop, grid_attribs ):
    """Function to apply the transition rules
    and return the new grid"""
    print(time_to_drop[0])
    if(time_to_drop[0] == 0):
        square = grid.shape[0]
        offset_y = 0
        top_y = 0
        bottom_y = top_y + 10

        left_x = 189
        right_x = left_x + 10
        print("("+str(left_x)+","+str(top_y)+")",
              "("+str(right_x)+","+str(bottom_y)+")")
        grid[top_y:bottom_y, left_x: right_x] = 6



    fireable = (grid == 1) | (grid == 2) | (grid == 3)
    on_fire = grid == 4
    cells_grid_attribs_fireable = grid_attribs[fireable]

    if cells_grid_attribs_fireable[:, 0].shape[0] > 0:
        should_be_on_fire = ignite(cells_grid_attribs_fireable[:, 0],
                                cells_grid_attribs_fireable[:, 1],
                                cells_grid_attribs_fireable[:, 2],
                                cells_grid_attribs_fireable[:, 3],
                                cells_grid_attribs_fireable[:, 4:12],
                                cells_grid_attribs_fireable[:,12:],
                                neighbourcounts[4][fireable],
                                neighbourstates[:,fireable])

        fire_cells = grid[fireable]
        fire_cells[should_be_on_fire] = 4
        grid[fireable] = fire_cells

    cells_grid_attribs_on_fire = grid_attribs[on_fire]

    grid_attribs[on_fire] = reduce_fuel(cells_grid_attribs_on_fire[:, 0],
                                        cells_grid_attribs_on_fire[:, 1],
                                        cells_grid_attribs_on_fire[:, 2],
                                        cells_grid_attribs_on_fire[:, 3],
                                        cells_grid_attribs_on_fire[:, 4:12],
                                        cells_grid_attribs_on_fire[:, 12:],)

    burnt_out = grid_attribs[:, :, 3] == 0
    grid[burnt_out] = 0

    time_to_drop[0] = time_to_drop[0] -1


    return grid

def cal_wind_spread_vectors(wind_x, wind_y):

    wind_vector = np.array([wind_x, wind_y])
    wind_mag = np.linalg.norm(wind_vector, axis=0)
    fire_vectors = np.array([[1,1], [0,1], [-1,1], [1,0], [0,-1], [1,-1], [0,-1], [-1,-1]])
    angle_diffs = np.zeros(8)

    for i in range(8):
        angle_diffs[i] = angle_between(wind_vector, fire_vectors[i])
    
    return np.exp(0.045 * wind_mag) * np.exp(wind_mag * 0.131 * (np.cos(angle_diffs) - 1))

def main():
    """ Main function that sets up, runs and saves CA"""
    config = setup(sys.argv[1:])
    wind_x = -15
    wind_y = 15
    grid_attribs = np.zeros((*config.grid_dims, 20))

    time_to_drop = np.array([295])
    # 0: Height - Scalar value
    # 1: Flammability
    # 2: Humidity?
    # 3: Fuel
    # 4-12: wind_spread_weights
    grid_attribs[:,:,0] = 0
    grid_attribs[:,:,1] = 1
    grid_attribs[:,:,2] = 0
    grid_attribs[:,:,3] = 20
    grid_attribs[:,:,4:12] = cal_wind_spread_vectors(wind_x, wind_y)
    

    config.initial_grid = np.ones( config.grid_dims)
    size_y , size_x = config.initial_grid.shape

    #Ariel drop
   
    #Pond
    config.initial_grid[ int( 0.2*size_y ) : int( 0.3*size_y), int(0.1*size_x):int(0.3*size_x)] = 6

    #Fire
    config.initial_grid[ 0, size_x-1] = 4

    #Town
    town_x_coords = [0, int(0.05*size_x)]
    town_y_coords = [int(0.95*size_y), size_y-1]
    config.initial_grid[ town_y_coords[0] : town_y_coords[1], town_x_coords[0]: town_x_coords[1]] = 5

    #Dense Forest
    d_forest_x_coords = [int(0.3*size_x), int(0.5*size_x)]
    d_forest_y_coords = [int( 0.6*size_y ), int( 0.81*size_y)]
    config.initial_grid[d_forest_y_coords[0] : d_forest_y_coords[1], d_forest_x_coords[0] : d_forest_x_coords[1]] = 2
    grid_attribs[d_forest_y_coords[0] : d_forest_y_coords[1], d_forest_x_coords[0] : d_forest_x_coords[1], 1] = 0.2
    grid_attribs[d_forest_y_coords[0] : d_forest_y_coords[1], d_forest_x_coords[0] : d_forest_x_coords[1], 3] = 30

    #Scrubland
    scrubland_x_coords = [int(0.65*size_x), int(0.7*size_x)]
    scrubland_y_coords = [int( 0.1*size_y ), int( 0.7*size_y)]
    config.initial_grid[scrubland_y_coords[0] : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1]] = 3
    grid_attribs[scrubland_y_coords[0] : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 1] = 3
    grid_attribs[scrubland_y_coords[0] : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 3] = 10

    #top slope of cayon
    grid_attribs[scrubland_y_coords[0] + 5 : scrubland_y_coords[0], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -1
    grid_attribs[scrubland_y_coords[0] + 4 : scrubland_y_coords[0], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -125.75
    grid_attribs[scrubland_y_coords[0] + 3 : scrubland_y_coords[0], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -250.5
    grid_attribs[scrubland_y_coords[0] + 2 : scrubland_y_coords[0], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -375.25
    grid_attribs[scrubland_y_coords[0] + 1 : scrubland_y_coords[0], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -500
    #bottom slope of cayon
    grid_attribs[scrubland_y_coords[1] - 1 : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -1
    grid_attribs[scrubland_y_coords[1] - 2 : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -125.75
    grid_attribs[scrubland_y_coords[1] - 3 : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -250.5
    grid_attribs[scrubland_y_coords[1] - 4 : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -375.25
    grid_attribs[scrubland_y_coords[1] - 5 : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -500
    #sides of cayon
    grid_attribs[scrubland_y_coords[0] : scrubland_y_coords[1], scrubland_x_coords[0] : scrubland_x_coords[1], 0] = -500

    transitions = [[1,1], [0,1], [-1,1], [1,0], [0,-1], [1,-1], [0,-1], [-1,-1]]
    height_neighbours = np.zeros((*config.grid_dims, 8))

    for i in range(grid_attribs.shape[0]):
        for j in range(grid_attribs.shape[1]):
            for k in range(len(transitions)):
                x_coord = j - transitions[k][0]
                y_coord = i - transitions[k][1]
                if (0 <= x_coord < 200) and (0 <= y_coord < 200):
                    height_neighbours[i][j][k] = grid_attribs[y_coord, x_coord, 0]

    grid_attribs[:,:,12:] = height_neighbours

    # Create grid object using parameters from config + transition function
    grid = Grid2D(config, (transition_function, time_to_drop, grid_attribs))

    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    # Save updated config to file
    config.save()
    # Save timeline to file
    utils.save(timeline, config.timeline_path)


if __name__ == "__main__":
    main()
