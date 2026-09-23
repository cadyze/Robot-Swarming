import os

import pandas as pd

from swarm_sim.enums import COLLISION_PROTOCOL
from swarm_sim.simulation_runner import run_simulation


def test_wait_all_records_steps_waited_not_moves(tmp_path, monkeypatch):
    """
    Regression test for a bug (fixed) where the wait_all path wrote
    moves[robot_ind] into the "Steps Waited" CSV column instead of
    steps_waited[robot_ind].
    """
    monkeypatch.chdir(tmp_path)

    run_simulation(5, 2, COLLISION_PROTOCOL.WAIT_NEXT, laziness_prob=0,
                    show_graph=False, wait_all=True, num_iterations=5)

    csv_path = os.path.join(
        "Data", "A5", "NON-RANDOM_SPAWNS", "LZP_0", "WAIT_NEXT", "R2", "T_0",
        "SwarmSimulationData.csv")
    df = pd.read_csv(csv_path)

    assert len(df) == 5
    # "Steps Waited" must never exceed "Timesteps" for a given run; the bug
    # made them identical every time, which this also would have caught if
    # any waiting occurred, but the strict invariant below always holds.
    assert (df["Steps Waited"] <= df["Timesteps"]).all()
    # With WAIT_NEXT and 2 robots on a 5x5 grid, at least one run should hit
    # a collision/wait, and the two columns must be able to differ.
    assert not (df["Steps Waited"] == df["Timesteps"]).all()


def test_first_hit_writes_one_scalar_result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    run_simulation(5, 1, COLLISION_PROTOCOL.WAIT_NEXT, laziness_prob=0,
                   show_graph=False, wait_all=False, num_iterations=2)

    csv_path = os.path.join(
        "Data", "A5", "LZP_0", "R1", "T_FIRST", "SwarmSimulationData.csv")
    df = pd.read_csv(csv_path)

    assert len(df) == 2
    assert (df["Timesteps"] > 0).all()
