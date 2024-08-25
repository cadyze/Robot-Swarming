import matplotlib.pyplot as plt
import RobotSwarmingSimulator
import swarming_simulator
import pandas as pd
import os
from RobotSwarmingSimulator import COLLISION_PROTOCOL, STARTING_POSITION

def plot_histogram(data, grid_size):
    plt.hist(data, bins='auto', edgecolor='black')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.title('Number of Moves to Find Target in a {}x{} Arena'.format(grid_size, grid_size))
    plt.show()

# border_obstacle = [ (7, 5), (7, 4), (7, 3), (6, 3), (5, 3), (4, 4), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (5, 6), (4, 6), (4, 5), (5, 4), (6, 4), (6, 5)]
# obstacle2 = [(4, 2), (3, 3), (4, 3), (5, 2)]
# grid_size = 61
# collision_protocol = RobotSwarmingSimulator.COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE
obstacles = []

def run_simulation(grid_size, num_robots, collision_protocol, tracked_robot=-1, show_graph=False, starting_pos=STARTING_POSITION.FILL, sensing_range=1):  
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
    csv_path = "A{}_R{}".format(grid_size, num_robots)
    if collision_protocol == COLLISION_PROTOCOL.WAIT_NEXT:
        csv_path += "_WN"
    elif collision_protocol == COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE:
        csv_path += "_FN"

    if tracked_robot != -1:
        csv_path += "_T{}".format(tracked_robot)

    csv_path += "_{}".format(starting_pos.name)
    csv_path += ".csv"

    # Setting up the dataframe for consistent .csv writing
    df = pd.DataFrame({
        'Timesteps': [],
        'Collisions': [],
        'Steps Waited': []
    })

    # Creates a new .csv if it doesn't exist
    if csv_path != "" and not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV file at {csv_path}.")

    # Create the SwarmingSimulation
    RobotSwarmSimulator = RobotSwarmingSimulator.SwarmSimulator(grid_size, (grid_size // 2, grid_size // 2), obstacles, 
                                                                starting_pos, sensing_range=sensing_range)
    moves, collisions = 0, 0
    for _ in range(1000):
        moves, collisions, steps_waited = RobotSwarmSimulator.start_robot_swarming(num_robots, collision_protocol, show_graph=show_graph, tracked_robot=tracked_robot)

        # If given a .csv, write the data to it
        if csv_path != "":
            df.loc[len(df)] = [moves, collisions, steps_waited]
            df.to_csv(csv_path, mode='a', header=False, index=False)
            df = df[0:0]

# run_simulation(30, 1, COLLISION_PROTOCOL.WAIT_NEXT, starting_pos=STARTING_POSITION.FILL, sensing_range=3)
run_simulation(61, 10, COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
               tracked_robot=0, starting_pos=STARTING_POSITION.FILL)
run_simulation(61, 10, COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
               tracked_robot=9, starting_pos=STARTING_POSITION.FILL)
run_simulation(61, 10, COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
               tracked_robot=9, starting_pos=STARTING_POSITION.SPACED)
run_simulation(61, 10, COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
               tracked_robot=9, starting_pos=STARTING_POSITION.EDGE)
run_simulation(61, 10, COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
               tracked_robot=0, starting_pos=STARTING_POSITION.SPACED)
run_simulation(61, 10, COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
               tracked_robot=0, starting_pos=STARTING_POSITION.EDGE)

run_simulation(61, 10, COLLISION_PROTOCOL.WAIT_NEXT,
               tracked_robot=9, starting_pos=STARTING_POSITION.FILL)
run_simulation(61, 10, COLLISION_PROTOCOL.WAIT_NEXT,
               tracked_robot=9, starting_pos=STARTING_POSITION.SPACED)
run_simulation(61, 10, COLLISION_PROTOCOL.WAIT_NEXT,
               tracked_robot=9, starting_pos=STARTING_POSITION.EDGE)
run_simulation(61, 10, COLLISION_PROTOCOL.WAIT_NEXT,
               tracked_robot=0, starting_pos=STARTING_POSITION.FILL)
run_simulation(61, 10, COLLISION_PROTOCOL.WAIT_NEXT,
               tracked_robot=0, starting_pos=STARTING_POSITION.SPACED)
run_simulation(61, 10, COLLISION_PROTOCOL.WAIT_NEXT,
               tracked_robot=0, starting_pos=STARTING_POSITION.EDGE)