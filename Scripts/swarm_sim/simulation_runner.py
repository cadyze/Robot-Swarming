"""
Drives repeated swarm runs and appends each run's stats to the CSV path(s)
FileManager derives from the run's parameters.
"""
import time

from swarm_sim import file_manager
from swarm_sim.enums import STARTING_POSITION
from swarm_sim.swarm_simulator import SwarmSimulator


def run_simulation(grid_size, num_robots, collision_protocol, laziness_prob, tracked_robot=-1,
                    show_graph=False, starting_pos=STARTING_POSITION.FILL, sensing_range=1,
                    generate_random_obs=False, wait_all=False, obstacles=None, num_iterations=3000):
    if tracked_robot >= num_robots:
        raise ValueError("tracked_robot must be a valid robot index.")

    obstacles = obstacles or []
    simulator = SwarmSimulator(
        grid_size, (grid_size // 2, grid_size // 2), obstacles,
        starting_pos, sensing_range, laziness_prob, for_math=False,
        rand_rob_obs=generate_random_obs)

    robot_csvs = []
    if wait_all:
        for robot_ind in range(num_robots):
            robot_csvs.append(file_manager.getCSVFromSwarmParameters(
                grid_size=grid_size, num_robots=num_robots, generate_random_obs=generate_random_obs,
                laziness_prob=laziness_prob, collision_protocol=collision_protocol, tracked_robot=robot_ind))
    else:
        robot_csvs.append(file_manager.getCSVFromSwarmParameters(
            grid_size=grid_size, num_robots=num_robots, generate_random_obs=generate_random_obs,
            laziness_prob=laziness_prob, collision_protocol=collision_protocol, tracked_robot=tracked_robot))

    for iteration in range(num_iterations):
        if iteration % 250 == 0:
            print("ITER: {}".format(iteration))
        start_time = time.time()
        result = simulator.start_robot_swarming(
            num_robots, collision_protocol, show_graph=show_graph,
            tracked_robot=tracked_robot, wait_for_all=wait_all)
        time_elapsed = time.time() - start_time

        if wait_all:
            moves, collisions, steps_waited = result
            for robot_ind in range(num_robots):
                file_manager.patchSwarmCSV(
                    robot_csvs[robot_ind], moves[robot_ind], collisions[robot_ind],
                    steps_waited[robot_ind], time_elapsed)
        else:
            moves, collisions, steps_waited = result[:3]
            file_manager.patchSwarmCSV(robot_csvs[0], moves, collisions, steps_waited, time_elapsed)
