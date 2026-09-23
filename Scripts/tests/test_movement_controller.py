import numpy as np
import pytest
from scipy.sparse import csr_matrix

from swarm_sim.enums import COLLISION_PROTOCOL
from swarm_sim.movement_controller import get_robot_next_move
from swarm_sim.state_space import info_to_state

GRID_SIZE = 3


def _row_matrix(n_states, entries):
    """entries: {state: {next_state: prob}} for a single source row (state 0)."""
    data, rows, cols = [], [], []
    for src, targets in entries.items():
        for dst, prob in targets.items():
            rows.append(src)
            cols.append(dst)
            data.append(prob)
    return csr_matrix((data, (rows, cols)), shape=(n_states, n_states))


def test_wait_next_holds_position_and_counts_collision():
    n_states = 6 * GRID_SIZE ** 2
    src = info_to_state(0, 0, 0, GRID_SIZE)
    dst = info_to_state(1, 0, 0, GRID_SIZE)
    matrix = _row_matrix(n_states, {src: {dst: 1.0}})

    curr_states = [src, dst]
    current_positions = [(0, 0), (1, 0)]
    collisions = [0, 0]
    num_waits = [0, 0]

    next_state, pos = get_robot_next_move(
        curr_states, 0, matrix, GRID_SIZE, current_positions,
        COLLISION_PROTOCOL.WAIT_NEXT, collisions, num_waits, tracked_robot=-1,
    )

    assert next_state == src
    assert pos == (0, 0)
    assert collisions == [1, 0]
    assert num_waits == [1, 0]


def test_find_next_available_resolves_to_free_cell():
    np.random.seed(42)
    n_states = 6 * GRID_SIZE ** 2
    src = info_to_state(0, 0, 0, GRID_SIZE)
    occupied = info_to_state(1, 0, 0, GRID_SIZE)
    free = info_to_state(0, 1, 1, GRID_SIZE)
    matrix = _row_matrix(n_states, {src: {occupied: 0.5, free: 0.5}})

    curr_states = [src, occupied]
    current_positions = [(0, 0), (1, 0)]
    collisions = [0, 0]
    num_waits = [0, 0]

    next_state, pos = get_robot_next_move(
        curr_states, 0, matrix, GRID_SIZE, current_positions,
        COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE, collisions, num_waits, tracked_robot=-1,
    )

    assert pos not in current_positions
    assert next_state == free
    assert collisions == [1, 0]
    assert num_waits == [0, 0]


def test_find_next_available_waits_when_every_candidate_is_occupied():
    n_states = 6 * GRID_SIZE ** 2
    src = info_to_state(0, 0, 0, GRID_SIZE)
    occupied = info_to_state(1, 0, 0, GRID_SIZE)
    matrix = _row_matrix(n_states, {src: {occupied: 1.0}})

    curr_states = [src, occupied]
    current_positions = [(0, 0), (1, 0)]
    collisions = [0, 0]
    num_waits = [0, 0]

    next_state, position = get_robot_next_move(
        curr_states, 0, matrix, GRID_SIZE, current_positions,
        COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE, collisions, num_waits, tracked_robot=-1,
    )

    assert next_state == src
    assert position == (0, 0)
    assert collisions == [1, 0]
    assert num_waits == [1, 0]


def test_tracked_robot_only_counts_for_itself():
    n_states = 6 * GRID_SIZE ** 2
    src = info_to_state(0, 0, 0, GRID_SIZE)
    dst = info_to_state(1, 0, 0, GRID_SIZE)
    matrix = _row_matrix(n_states, {src: {dst: 1.0}})

    curr_states = [src, dst]
    current_positions = [(0, 0), (1, 0)]
    collisions = [0, 0]
    num_waits = [0, 0]

    get_robot_next_move(
        curr_states, 0, matrix, GRID_SIZE, current_positions,
        COLLISION_PROTOCOL.WAIT_NEXT, collisions, num_waits, tracked_robot=1,
    )

    assert collisions == [0, 0]
    assert num_waits == [0, 0]
