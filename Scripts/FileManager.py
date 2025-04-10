import os
import pandas as pd
from RobotSwarmingSimulator import COLLISION_PROTOCOL

def getCSVFromSwarmParameters(grid_size, num_robots, generate_random_obs, laziness_prob, collision_protocol, tracked_robot):
    folder_paths = ["Data"]

    folder_paths.append("A{}".format(grid_size)) # Arena size

    # If the robots are randomly generated or not
    if num_robots > 1:
        if generate_random_obs:
            folder_paths.append("RANDOM_SPAWNS")
        else:
            folder_paths.append("NON-RANDOM_SPAWNS")

    # Laziness probabilities
    folder_paths.append("LZP_{}".format(str(round(laziness_prob * 100, 2)))) # Testing whether '.' can be used

    # Collision protocols for multi-robot swarms
    if num_robots > 1:
        if collision_protocol == COLLISION_PROTOCOL.WAIT_NEXT:
            folder_paths.append("WAIT_NEXT")
        elif collision_protocol == COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE:
            folder_paths.append("FIND_NEXT")

    # Number of robots
    folder_paths.append("R{}".format(num_robots))

    # Tracked robot
    if tracked_robot == -1:
        folder_paths.append("T_FIRST")
    else:
        folder_paths.append("T_{}".format(tracked_robot))

    path_to_folder = ""
    def create_folder(path, folder_name):
        combined_path = "{}/{}".format(path, folder_name)
        # print(combined_path)
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)
            # print("Made folder!")

    def create_nested_folders(folders):
        current_path = "."
        for folder in folders:
            create_folder(current_path, folder)
            current_path = "{}/{}".format(current_path, folder)
        print("Final Path Created: {}".format(current_path))
        return current_path

    path_to_folder = create_nested_folders(folder_paths)
    csv_path = "{}/SwarmSimulationData.csv".format(path_to_folder)    

    # Setting up the dataframe for consistent .csv writing
    df = pd.DataFrame({
        'Timesteps': [],
        'Collisions': [],
        'Steps Waited': [],
        'Real-Time Elapsed': []
    })

    # Creates a new .csv if it doesn't exist
    if csv_path != "" and not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
        print(f"Created new CSV file at {csv_path}.")
    return csv_path
        
def patchSwarmCSV(csv_path, moves, collisions, steps_waited, time_elapsed):
    if csv_path:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            # Create new CSV with headers
            header = True
        else:
            # File exists and has content
            header = False

        new_row = pd.DataFrame([[moves, collisions, steps_waited, time_elapsed]],
                            columns=['Timesteps', 'Collisions', 'Steps Waited', 'Real-Time Elapsed'])
        
        new_row.to_csv(csv_path, mode='a', header=header, index=False)