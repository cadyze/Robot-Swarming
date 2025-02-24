import matplotlib.pyplot as plt
import RobotSwarmingSimulator
import swarming_simulator
import pandas as pd
import os
import time
from RobotSwarmingSimulator import COLLISION_PROTOCOL, STARTING_POSITION
import json
import numpy as np

obstacles = []


def state_to_info(state, grid_size):
    orientation = state % 6
    state //= 6
    y = state // grid_size
    x = state % grid_size
    return x, y, orientation

def info_to_state(x, y, o, grid_size):
        return 6 * (x + grid_size * y) + o

def compute_mean_std(file_path):
    import numpy as np

    # Read the CSV file
    df = pd.read_csv(file_path)
    m_values = []

    # Check if 'Timesteps' column exists in the CSV
    if 'Timesteps' in df.columns:
        # Extract the 'M' values
        m_values = df['Timesteps'].values
    
    return np.mean(m_values), np.std(m_values)

def run_simulation(grid_size, num_robots, collision_protocol, laziness_prob, tracked_robot=-1, show_graph=False, starting_pos=STARTING_POSITION.FILL, sensing_range=1, generate_random_obs=False):  
    # Arena: 31x31, 61x61, 81x81
    # Number of Robots: 1, 5, 10, 20 -> 1, 3, 5, 10
    # Protocols: BREAK,  FIND NEXT AVAILABLE
    # Tracking Robots: First index, Last index, Middle Robot
    # TODO: Add robots that allign against the wall
    # XXXX: Graph the number of collision protocols called
    # TODO: Create the triangle but leave the insides unfilled by robots
    # XXXX: Save the raw data of number of iterations and collisions
    # TODO: Add spaces around robots (starting position)
    
    if tracked_robot > num_robots:
        raise Exception("ERROR: TRYING TO TRACK A ROBOT THAT DOESN'T EXIST.")
    
    # Create dynamic .csv path
    file_name = "A{}_R{}".format(grid_size, num_robots)

    if num_robots > 1:
        if generate_random_obs == False:
            file_name += "_{}".format(starting_pos.name)
        else:
            file_name += "_RR".format()

    file_name += "_LZP{}".format(str(laziness_prob * 100).replace(".", ","))

    if tracked_robot != -1:
        file_name += "_ST{}".format(0, 0, 0, grid_size)

    if num_robots > 1:
        if collision_protocol == COLLISION_PROTOCOL.WAIT_NEXT:
            file_name += "_WN"
        elif collision_protocol == COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE:
            file_name += "_FN"
    csv_path = "{}/{}.csv".format(file_name, file_name)

    # Setting up the dataframe for consistent .csv writing
    df = pd.DataFrame({
        'Timesteps': [],
        'Collisions': [],
        'Steps Waited': [],
        'Real-Time Elapsed': []
    })

    # Check if the folder exists
    if not os.path.exists(file_name):
        os.makedirs(file_name)
        print(f"Folder '{file_name}' created successfully!")
    else:
        print(f"Folder '{file_name}' already exists.")

    # Creates a new .csv if it doesn't exist
    if csv_path != "" and not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV file at {csv_path}.")


    # Save .json file holding information on paths and positions visited
    path_file_name = "{}/PathData.json".format(file_name)  # JSON file name
    position_visits_histories = []

    # Check if the file exists
    if os.path.exists(path_file_name):
        # Load the file
        with open(path_file_name, "r") as json_file:
            loaded_data = json.load(json_file)
        position_visits_histories = loaded_data.get("position_visits", [])  # Safely get the value of position_visits

    # Create the SwarmingSimulation
    RobotSwarmSimulator = RobotSwarmingSimulator.SwarmSimulator(grid_size, (grid_size // 2, grid_size // 2), obstacles, 
                                                                starting_pos, sensing_range, laziness_prob, for_math=False, rand_rob_obs=generate_random_obs)
    moves, collisions = 0, 0

    prob = 0
    math = RobotSwarmingSimulator.SwarmSimulator(g, (g // 2, g // 2), obstacles, 
                                                                STARTING_POSITION.FILL, 1, prob, for_math=True)
    HTmu, HTvariance, HTstd = math.calculate_mean_variance()
    print("PROB: {} | HTMU: {} | HTVARIANCE: {} | HTSTD {}".format(prob, HTmu[0], HTvariance[0], HTstd[0]))

    iter = 0

    # Read the CSV file
    df = pd.read_csv(csv_path)
    m_values = []

    # Check if 'Timesteps' column exists in the CSV
    if 'Timesteps' in df.columns:
        # Extract the 'M' values
        m_values = df['Timesteps'].values

    while True:
        mean, std = np.mean(m_values), np.std(m_values)
        if np.round(HTmu[0], 1) - 1 == np.round(mean, 1) and np.round(HTstd[0], 1) == np.round(std, 1):
            break

        # Check if equivalent
        if iter % 250 == 0:
            print("ITER: {} | MEAN: {} | STD: {}".format(iter, mean, std))
        iter += 1
        start_time = time.time()
        moves, collisions, steps_waited, pos_visited = RobotSwarmSimulator.start_robot_swarming(num_robots, collision_protocol, show_graph=show_graph, tracked_robot=tracked_robot)
        time_elapsed = time.time() - start_time

        # If given a .csv, write the data to it
        if csv_path != "":
            df.loc[len(df)] = [moves, collisions, steps_waited, time_elapsed]
            df.to_csv(csv_path, mode='a', header=False, index=False)
            df = df[0:0]
        
        # Append path history to path data
        position_visits_histories.append(pos_visited.tolist())
        m_values = np.append(m_values, moves)
    
    # Add the path data to the path
    with open(path_file_name, "w") as json_file:
        json.dump({"position_visits": position_visits_histories}, json_file)

# TODO: Caclculate the number of collisions waited and calculate probability to use for laziness
num_robots = 5
g = 61

# run_simulation(g, num_robots, COLLISION_PROTOCOL.WAIT_NEXT, laziness_prob=0, tracked_robot=num_robots-1, generate_random_obs=False, show_graph=True)    
run_simulation(g, 1, COLLISION_PROTOCOL.WAIT_NEXT, laziness_prob=0.0014, tracked_robot=0, generate_random_obs=False, show_graph=False)     



# prob = 0
# math = RobotSwarmingSimulator.SwarmSimulator(g, (g // 2, g // 2), obstacles, 
#                                                             STARTING_POSITION.FILL, 1, prob, for_math=True)
# HTmu, HTvariance, HTstd = math.calculate_mean_variance()
# print("PROB: {} | HTMU: {} | HTVARIANCE: {} | HTSTD {}".format(prob, HTmu[0], HTvariance[0], HTstd[0]))
# for prob in np.arange(0.0012, 0.0014, 0.0001):

# TODO: Do the laziness and number robot simulations

# TODO: XXX Check random generation to not be on top of each other 
# TODO: Create a table of all different parameter combinations, showing all of their means and STD | COLUMN HEADERS: Number of Robots, Laziness, Mean, STD., %Waited (Max of 2)
# TODO: Also add a mathematical equation to calculate 
# TODO: Calculate the mean and STD using the model then run simulations until they are nearly equal to one another (R1, LZ0)
# TODO: Add a spacing of two between all obstacles and main robot
# TODO: Scale the matrix by laziness so you can perform the mathematical calculation on the laziness transition matrices
# TODO: XXX Track the number of times an (x, y)  has been repeated hit in a single simulation - used to sense for movement traps and cycles

# TODO: Count the number (%) of the arena covered
# TODO: Searching patterns


# TODO: Extract the percent of the time is the robot being held up by obstacles, run numerous obstacles - calculate XXX
# TODO: Extract the mean and STD of starting state of the robot
# TODO: Add dyanmic obstacles argument, add positions to spawn obstacles in XXX
# TODO: Test different linear system solver - direct solver XXX
# TODO: Mowing lawn type of movement policy

# TODO: Look into function that solves next state linearly.
# TODO: Look GMRES, the matrix you provided can be replaced by a vector