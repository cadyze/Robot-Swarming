"""
Orchestrates a single swarm run: builds the transition matrices for a grid
config, spawns robots, and steps them (via movement_controller) until the
target is found -- or, in wait_for_all mode, until every robot has found it.
"""
import random as rand

import numpy as np

import graph_swarm
from swarm_sim.arena_builder import build_arena
from swarm_sim.enums import STARTING_POSITION
from swarm_sim.hitting_time_math import calculate_mean_variance
from swarm_sim.movement_controller import get_robot_next_move
from swarm_sim.state_space import state_to_info


class SwarmSimulator:
    def __init__(self, grid_size, target_pos, obstacles, starting_pos, sensing_range, laziness_prob,
                 for_math, use_obs_pos_for_rob=False, rand_rob_obs=False):
        self.grid_size = grid_size
        self.target_pos = target_pos
        self.collisions = 0
        self.timesteps = 0
        self.num_waits = 0
        self.sensing_range = sensing_range
        self.laziness_prob = laziness_prob
        self.for_math = for_math
        self.Arena, self.DynamicObstacleArena = build_arena(
            grid_size, target_pos, obstacles, laziness_prob, sensing_range, for_math)
        self.obstacles = obstacles
        self.starting_pos = starting_pos
        self.tracked_robot = -1
        self.use_obs_pos_for_rob = use_obs_pos_for_rob
        self.rand_rob_obs = rand_rob_obs

    def state_to_info(self, state, grid_size):
        return state_to_info(state, grid_size)

    def calculate_mean_variance(self, save_dir="."):
        return calculate_mean_variance(self.Arena, save_dir=save_dir)

    def start_robot_swarming(self, num_robots, collision_protocol,
                              show_graph=True, tracked_robot=-1, wait_for_all=False):

        if wait_for_all and tracked_robot != -1:
            raise Exception("ERROR: CANNOT TRACK A ROBOT WHILE WAITING FOR ALL TO FINISH.")

        if tracked_robot >= num_robots:
            raise ValueError("tracked_robot must be a valid robot index.")

        self.tracked_robot = tracked_robot
        self.collisions = [0 for _ in range(num_robots)]
        self.num_waits = [0 for _ in range(num_robots)]

        def info_to_state(x, y, orientation):
            return 6 * (x + self.grid_size * y) + orientation

        # Instantiate the lower left corner with number of drones starting at 0, 0
        curr_coords = [(0, 0, 0)]
        curr_states = [info_to_state(0, 0, 0)]
        n_to_instantiate = num_robots - 1
        x, y = 0, 1
        layer = 1
        i = 0

        def is_there_overlap(coord, curr_coords):
            x_t, y_t = coord
            for e_coords in curr_coords:
                x2, y2, _ = e_coords
                if x_t == x2 and y_t == y2:
                    return True
            return False

        while n_to_instantiate != 0:
            # Check first if using obstacle positions
            if self.use_obs_pos_for_rob:
                x, y = self.obstacles[i]
                curr_states.append(info_to_state(x, y, 0))
                n_to_instantiate -= 1
                i += 1
            elif self.rand_rob_obs:
                x, y, o = 0, 0, 0
                # Keep trying to spawn if there's any random overlap
                while is_there_overlap((x, y), curr_coords):
                    x, y, o = state_to_info(rand.randint(0, 6 * (self.grid_size ** 2) - 1), self.grid_size)
                curr_states.append(info_to_state(x, y, o))
                curr_coords.append((x, y, o))
                n_to_instantiate -= 1
            else:
                # Instantiate a new robot depending on the starting position given
                if self.starting_pos == STARTING_POSITION.EDGE:
                    if x == 0 or y == 0:
                        curr_states.append(info_to_state(x, y, 0))
                        n_to_instantiate -= 1
                elif self.starting_pos == STARTING_POSITION.SPACED:
                    if x % 2 == 0 and y % 2 == 0:
                        curr_states.append(info_to_state(x, y, 0))
                        n_to_instantiate -= 1
                elif self.starting_pos == STARTING_POSITION.FILL:
                    curr_states.append(info_to_state(x, y, 0))
                    n_to_instantiate -= 1

                # Update for next coordinates
                if y == 0:
                    layer += 1
                    y = layer
                    x = 0
                else:
                    x += 1
                    y -= 1

        # Reverse so that the first robot is able to move out freely
        curr_states = curr_states[::-1]

        # For tracking the positions the robot has traveled to
        pos_visited = np.zeros((self.grid_size, self.grid_size))

        curr_states = curr_states[:num_robots]
        history = [[] for _ in range(num_robots)]
        is_target_found = False
        first_robot_to_target = None
        current_positions = [(0, 0) for _ in range(num_robots)]
        robots_finished = [False for _ in range(num_robots)]

        self.timesteps = [0 for _ in range(num_robots)]
        while not is_target_found:
            for robot_ind in range(num_robots):
                x, y, _ = state_to_info(curr_states[robot_ind], self.grid_size)
                history[robot_ind].append((x, y))

                if wait_for_all:
                    if x == self.target_pos[0] and y == self.target_pos[1]:
                        # All finished robots will be relocated to off the board to avoid extra collisions
                        robots_finished[robot_ind] = True
                        current_positions[robot_ind] = (-1, -1)
                    else:
                        curr_states[robot_ind], current_positions[robot_ind] = get_robot_next_move(
                            curr_states, robot_ind, self.Arena, self.grid_size, current_positions,
                            collision_protocol, self.collisions, self.num_waits, tracked_robot=self.tracked_robot)
                        self.timesteps[robot_ind] += 1
                else:
                    # Code for tracking robots
                    if tracked_robot != -1:
                        if tracked_robot == robot_ind:
                            if x == self.target_pos[0] and y == self.target_pos[1]:
                                is_target_found = True
                                first_robot_to_target = robot_ind
                            curr_states[robot_ind], current_positions[robot_ind] = get_robot_next_move(
                                curr_states, robot_ind, self.Arena, self.grid_size, current_positions,
                            collision_protocol, self.collisions, self.num_waits, tracked_robot=self.tracked_robot)
                            self.timesteps[robot_ind] += 1
                            pos_visited[x, y] += 1
                        else:
                            curr_states[robot_ind], current_positions[robot_ind] = get_robot_next_move(
                                curr_states, robot_ind, self.DynamicObstacleArena, self.grid_size, current_positions,
                                collision_protocol, self.collisions, self.num_waits, tracked_robot=self.tracked_robot)
                    else:
                        curr_states[robot_ind], current_positions[robot_ind] = get_robot_next_move(
                            curr_states, robot_ind, self.Arena, self.grid_size, current_positions,
                            collision_protocol, self.collisions, self.num_waits, tracked_robot=self.tracked_robot)
                        if x == self.target_pos[0] and y == self.target_pos[1]:
                            is_target_found = True
                            first_robot_to_target = robot_ind
                        self.timesteps[robot_ind] += 1

            # Check goals for wait for all
            if wait_for_all and all(robots_finished):
                is_target_found = True
                print("Found Target!")

        if show_graph:
            graph_swarm.graph_arena(self.grid_size, self.target_pos, history, self.obstacles, self.tracked_robot)

        if wait_for_all:
            return self.timesteps, self.collisions, self.num_waits
        if tracked_robot != -1:
            return (self.timesteps[tracked_robot], self.collisions[tracked_robot],
                    self.num_waits[tracked_robot], pos_visited)

        # In first-hit mode, return the robot whose arrival ended the run.
        return (self.timesteps[first_robot_to_target], self.collisions[first_robot_to_target],
                self.num_waits[first_robot_to_target])
