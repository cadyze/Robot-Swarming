import os
import time

import pandas as pd

from swarm_sim.enums import COLLISION_PROTOCOL


def getCSVFromSwarmParameters(grid_size, num_robots, generate_random_obs, laziness_prob, collision_protocol, tracked_robot):
    folder_paths = ["Data"]

    folder_paths.append("A{}".format(grid_size))  # Arena size

    # If the robots are randomly generated or not
    if num_robots > 1:
        if generate_random_obs:
            folder_paths.append("RANDOM_SPAWNS")
        else:
            folder_paths.append("NON-RANDOM_SPAWNS")

    # Laziness probabilities
    folder_paths.append("LZP_{}".format(str(round(laziness_prob * 100, 2))))  # Testing whether '.' can be used

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

    def create_folder(path, folder_name):
        combined_path = "{}/{}".format(path, folder_name)
        if not os.path.exists(combined_path):
            os.makedirs(combined_path)

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


def patchSwarmCSV(csv_path, moves, collisions, steps_waited, time_elapsed,
                  max_write_attempts=8):
    """Append one result row, tolerating short-lived Windows file locks.

    Spreadsheet applications and background sync tools can temporarily lock a
    CSV between simulation iterations. Retrying the append is safe because
    ``to_csv`` either opens and writes the row or raises before modifying it.
    """
    if csv_path:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            # Create new CSV with headers
            header = True
        else:
            # File exists and has content
            header = False

        new_row = pd.DataFrame([[moves, collisions, steps_waited, time_elapsed]],
                                columns=['Timesteps', 'Collisions', 'Steps Waited', 'Real-Time Elapsed'])

        for attempt in range(max_write_attempts):
            try:
                new_row.to_csv(csv_path, mode='a', header=header, index=False)
                return
            except PermissionError as error:
                if attempt == max_write_attempts - 1:
                    raise PermissionError(
                        f"Could not append to '{csv_path}' after "
                        f"{max_write_attempts} attempts. Close the CSV in Excel "
                        "or any editor/sync tool and run the simulation again."
                    ) from error

                # 0.05 + 0.10 + ... + 3.2 seconds: enough for a transient
                # sharing lock without silently hiding a persistent one.
                time.sleep(0.05 * (2 ** attempt))
