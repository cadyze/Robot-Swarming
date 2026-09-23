"""
Resolves a single robot's next move: samples a next state from its row of
the (sparse) transition matrix, and applies a collision protocol when the
sampled cell is already occupied by another robot.
"""
import numpy as np

from swarm_sim.enums import COLLISION_PROTOCOL
from swarm_sim.state_space import state_to_info


def _sample_state(probability_row):
    return np.random.choice(probability_row.indices, p=probability_row.data)


def get_robot_next_move(curr_states, robot_ind, probability_matrix, grid_size,
                         current_positions, collision_protocol, collisions, num_waits,
                         tracked_robot=-1, max_find_next_iters=100):
    row = probability_matrix[curr_states[robot_ind]]
    next_state = _sample_state(row)
    x, y, _ = state_to_info(next_state, grid_size)

    def should_count():
        if tracked_robot != -1:
            return robot_ind == tracked_robot
        return True

    if (x, y) in current_positions:
        if should_count():
            collisions[robot_ind] += 1

        if collision_protocol == COLLISION_PROTOCOL.WAIT_NEXT:
            next_state = curr_states[robot_ind]
            x, y, _ = state_to_info(next_state, grid_size)
            if should_count():
                num_waits[robot_ind] += 1
        elif collision_protocol == COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE:
            found_next = False
            for _ in range(max_find_next_iters):
                next_state = _sample_state(row)
                x, y, _ = state_to_info(next_state, grid_size)
                if (x, y) not in current_positions:
                    found_next = True
                    break

            if should_count() and not found_next:
                # No alternative was available, so this attempt becomes a wait.
                next_state = curr_states[robot_ind]
                x, y, _ = state_to_info(next_state, grid_size)
                num_waits[robot_ind] += 1
    return next_state, (x, y)
