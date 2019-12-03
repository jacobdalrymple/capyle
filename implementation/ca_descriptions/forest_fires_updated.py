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

# np.set_printoptions(threshold=sys.maxsize)


def setup(args):
    """Set up the config object used to interact with the GUI"""
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest Fires Updated"
    config.dimensions = 2
    config.wrap = False
    config.num_generations = 100
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

# Vectorised function to reduce fuel based on 5 property arrays given
def reduce_fuel(height, wind_x, wind_y, rate_of_flam, humidity, fuel):
    # with_spare_fuel = (fuel - rate_of_flam) >= 0
    # fuel[with_spare_fuel] = np.around(fuel[with_spare_fuel] - rate_of_flam[with_spare_fuel], 3)
    fuel = (fuel - rate_of_flam).clip(min=0)
    return np.array([height, wind_x, wind_y, rate_of_flam, humidity, fuel]).T

def scale(arr, min_, max_):
    return np.interp(arr, (arr.min(),
                                    arr.max()), (min_, max_))
    
def cal_wind_weight(wind_x, wind_y, neighbour_states):

    wind_weight = np.linspace(1, 1, wind_x.shape[0])

    NW, N, NE, W, E, SW, S, SE = neighbour_states

    #eastward_fire = (neighbour_states[:,4] == 4) #| (neighbour_states[:,2] == 4) | (neighbour_states[:,7] == 4)
    ##westward_fire = (neighbour_states[:,0] == 4) | (neighbour_states[:,3] == 4) | (neighbour_states[:,5] == 4)
    #x_wind = wind_x == -1
    #east = eastward_fire & x_wind
    #west = westward_fire & x_wind
    wind_weight[E==4] = 5
    wind_weight[W==4] = 0.1

    return wind_weight


def ignite(height, wind_x, wind_y, rate_of_flam, humidity, fuel, on_fire_neighbours, neighbour_states):
    
    on_fire_neighbours[on_fire_neighbours == 1] = 0.2
    on_fire_neighbours[on_fire_neighbours == 2] = 0.5 
    on_fire_neighbours[on_fire_neighbours > 4] = 1  

    wind_weight = cal_wind_weight(wind_x, wind_y, neighbour_states)

    prob = on_fire_neighbours*(rate_of_flam*np.random.rand(height.shape[0])*wind_weight)#*wind_x*wind_y
    
    min_flam = np.min(rate_of_flam)
    return prob > 0.2#(0.25)#*4)

# def ignite(height, wind, rate_of_flam, humidity, fuel, on_fire_neighbours):

#     wind_prob = np.interp(wind, (wind.min(), wind.max()), (0, 0.5))
#     height_prob = np.interp(height, (height.min(), height.max()), (0, 0.5))

#     prob = on_fire_neighbours*(wind_prob + height_prob + rate_of_flam)
#     normalised_prob = np.interp(prob, (prob.min(), prob.max()), (0, 1))

#     return (normalised_prob > 0.5).astype(int)+1



def transition_function(grid, neighbourstates, neighbourcounts, grid_attribs):
    """Function to apply the transition rules
    and return the new grid"""

    fireable = (grid == 1) | (grid == 2) | (grid == 3)
    #print("fireable.shape")
    #print(fireable.shape)
    on_fire = grid == 4

    cells_grid_attribs_fireable = grid_attribs[fireable]

    # neighbours_of_onfire_cells = neighbourcounts[fireable]
    # neighbours_of_onfire_cells[2]
    # grid = ignites(grid, grid_attribs, neighbourstates.T)
    if cells_grid_attribs_fireable[:, 0].shape[0] > 0:
        should_be_on_fire = ignite(cells_grid_attribs_fireable[:, 0],
                                cells_grid_attribs_fireable[:, 1],
                                cells_grid_attribs_fireable[:, 2],
                                cells_grid_attribs_fireable[:, 3],
                                cells_grid_attribs_fireable[:, 4],
                                cells_grid_attribs_fireable[:, 5],
                                neighbourcounts[4][fireable],
                                neighbourstates[:,fireable])

        #print("should_be_on_fire.shape")
        #print(should_be_on_fire.shape)
        # print(grid[fireable]    )
        # grid[fireable] = 3  
        fire_cells = grid[fireable]
        fire_cells[should_be_on_fire] = 4
        grid[fireable] = fire_cells 
    # fireable_and_should_be_on_fire = fireable & should_be_on_fire
    # grid[fireable] = 
    # print(grid[fireable][should_be_on_fire].shape)
    # print("res.shape")
    # print(res.shape)

    # NW, N, NE, W, E, SW, S, SE = neighbourstates

    # fire_close = (N == 2) | (E == 2) | (W == 2) | (S == 2)
    # fire_far = (NW == 2) | (NE == 2) | (SW == 2) | (SE == 2)
    # neighbour_on_fire = fire_close | fire_far

    # cells_at_fire_risk = neighbour_on_fire & fireable
    # grid[cells_at_fire_risk] = 2


    cells_grid_attribs_on_fire = grid_attribs[on_fire]

    grid_attribs[on_fire] = reduce_fuel(cells_grid_attribs_on_fire[:, 0],
                                        cells_grid_attribs_on_fire[:, 1],
                                        cells_grid_attribs_on_fire[:, 2],
                                        cells_grid_attribs_on_fire[:, 3],
                                        cells_grid_attribs_on_fire[:, 4],
                                        cells_grid_attribs_on_fire[:, 5])

    burnt_out = grid_attribs[:, :, 5] == 0
    grid[burnt_out] = 0

    return grid


def main():
    """ Main function that sets up, runs and saves CA"""
    # Get the config object from set up
    config = setup(sys.argv[1:])

    # 0 = BURNT OUT
    # 1 = DEFAULT, BURNABLE GRASS
    # 2 = DENSE FOREST
    # 3 = HIGH FLAMMABLE SCRUB
    # 4 = ON FIRE
    # 5 = BUILDINGS
    # 6 = WATER


    grid_attribs = np.zeros((*config.grid_dims, 6))


    
    # 0: Height - Scalar value
    # 1: Wind/Magnitude - East to West
    # 2: Wind Mag - North to South
    # 3: Flammability
    # 4: Humidity?
    # 5: Fuel
    grid_attribs[...] = (0, -1, -1, 0.3, 0, 2)
    # grid_attribs[:,:,0 ] = np.random.randint(-5, 5, size=grid_attribs[:, 0].shape[0])
    # grid_attribs[:,:,1 ] = np.random.randint(0, 5, size=grid_attribs[:, 0].shape[0])
    # print(winds.shape)

    config.initial_grid = np.ones( config.grid_dims)
    size_y , size_x = config.initial_grid.shape
    #Pond
    config.initial_grid[ int( 0.2*size_y ) : int( 0.3*size_y), int(0.1*size_x):int(0.3*size_x)] = 6
    
    #Fire
    config.initial_grid[ 0, size_x-101] = 4
    config.initial_grid[ 0, size_x-102] = 4
    config.initial_grid[ 1, size_x-102] = 4
    config.initial_grid[ 1, size_x-103] = 4
    
    #Town
    config.initial_grid[ int(0.95*size_y) : size_y-1, 0: int(0.05*size_x)] = 5

    #Dense Forest
    config.initial_grid[ int( 0.6*size_y ) : int( 0.81*size_y), int(0.3*size_x): int(0.5*size_x)] = 2
    grid_attribs[ int( 0.6*size_y ) : int( 0.81*size_y), int(0.3*size_x): int(0.5*size_x) ] = ( 0, 1, 0.1, 0.075, 0, 3)

    #Scrubland
    config.initial_grid[ int( 0.1*size_y ) : int( 0.7*size_y), int(0.65*size_x): int(0.7*size_x)] = 3
    grid_attribs[int( 0.1*size_y ) : int( 0.7*size_y), int(0.65*size_x): int(0.7*size_x)] = (-1, 0.1, 1, 3, 0, 1)

    #grid_attribs[:, :, 1] = np.linspace(0.5, 0.5, grid_attribs[:, 0].shape[0])
    #grid_attribs[:,:,2]  = np.linspace(4,4, grid_attribs[:,0].shape[0])


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
