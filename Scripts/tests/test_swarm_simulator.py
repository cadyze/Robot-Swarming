import json
import os

import numpy as np
import pytest

from swarm_sim.enums import COLLISION_PROTOCOL, STARTING_POSITION
from swarm_sim.swarm_simulator import SwarmSimulator

from conftest import FIXTURES_DIR

with open(os.path.join(FIXTURES_DIR, "manifest.json")) as f:
    MANIFEST = json.load(f)


def _make_sim(grid_size=5, laziness_prob=0.0, obstacles=None):
    target_pos = (grid_size // 2, grid_size // 2)
    return SwarmSimulator(
        grid_size=grid_size,
        target_pos=target_pos,
        obstacles=obstacles or [],
        starting_pos=STARTING_POSITION.FILL,
        sensing_range=1,
        laziness_prob=laziness_prob,
        for_math=False,
    )


def test_wait_for_all_terminates_and_matches_baseline_distribution():
    np.random.seed(1234)
    sim = _make_sim(grid_size=5)

    move_counts = []
    for _ in range(200):
        timesteps, collisions, waits = sim.start_robot_swarming(
            2, COLLISION_PROTOCOL.WAIT_NEXT, show_graph=False, wait_for_all=True)
        assert len(timesteps) == 2
        assert all(t > 0 for t in timesteps)
        assert all(c >= 0 for c in collisions)
        assert all(w >= 0 for w in waits)
        move_counts.extend(timesteps)

    baseline = MANIFEST["stochastic_baseline"]
    mean = float(np.mean(move_counts))
    std = float(np.std(move_counts))

    # Sampling method changed (sparse row support vs. dense np.random.choice
    # over the whole state space), so trajectories aren't bit-for-bit
    # reproducible -- this checks the distribution didn't shift meaningfully.
    assert mean == pytest.approx(baseline["mean"], rel=0.35)
    assert std == pytest.approx(baseline["std"], rel=0.5)


def test_single_robot_tracked_mode_reports_timesteps_and_full_visit_grid():
    sim = _make_sim(grid_size=5)
    timesteps, collisions, waits, pos_visited = sim.start_robot_swarming(
        1, COLLISION_PROTOCOL.WAIT_NEXT, show_graph=False, tracked_robot=0)

    assert timesteps > 0
    assert pos_visited.shape == (5, 5)
    assert pos_visited.sum() > 0


def test_untracked_non_wait_all_returns_first_hit_statistics():
    sim = _make_sim(grid_size=5)
    timesteps, collisions, waits = sim.start_robot_swarming(
        1, COLLISION_PROTOCOL.WAIT_NEXT, show_graph=False)
    assert timesteps > 0
    assert collisions >= 0
    assert waits >= 0


def test_obstacles_do_not_crash_wait_for_all():
    sim = _make_sim(grid_size=7, obstacles=[[(1, 1), (1, 2), (2, 2), (2, 1)]])
    timesteps, collisions, waits = sim.start_robot_swarming(
        2, COLLISION_PROTOCOL.WAIT_NEXT, show_graph=False, wait_for_all=True)
    assert len(timesteps) == 2
