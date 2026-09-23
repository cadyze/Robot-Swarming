import os

import pandas as pd

from swarm_sim.enums import COLLISION_PROTOCOL
from swarm_sim import file_manager


def test_getCSVFromSwarmParameters_creates_expected_nested_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    csv_path = file_manager.getCSVFromSwarmParameters(
        grid_size=11, num_robots=3, generate_random_obs=False,
        laziness_prob=0.0, collision_protocol=COLLISION_PROTOCOL.WAIT_NEXT,
        tracked_robot=-1,
    )

    expected = os.path.join(
        ".", "Data", "A11", "NON-RANDOM_SPAWNS", "LZP_0.0", "WAIT_NEXT", "R3", "T_FIRST",
        "SwarmSimulationData.csv",
    ).replace("\\", "/")
    assert csv_path.replace("\\", "/") == expected
    assert os.path.isfile(csv_path)

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["Timesteps", "Collisions", "Steps Waited", "Real-Time Elapsed"]
    assert len(df) == 0


def test_getCSVFromSwarmParameters_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path1 = file_manager.getCSVFromSwarmParameters(
        grid_size=11, num_robots=1, generate_random_obs=False,
        laziness_prob=0.0, collision_protocol=COLLISION_PROTOCOL.WAIT_NEXT,
        tracked_robot=0,
    )
    file_manager.patchSwarmCSV(path1, 5, 0, 0, 0.01)

    path2 = file_manager.getCSVFromSwarmParameters(
        grid_size=11, num_robots=1, generate_random_obs=False,
        laziness_prob=0.0, collision_protocol=COLLISION_PROTOCOL.WAIT_NEXT,
        tracked_robot=0,
    )
    assert path1 == path2
    df = pd.read_csv(path2)
    assert len(df) == 1  # patch wasn't wiped out by re-requesting the same path


def test_patchSwarmCSV_appends_rows_with_header_once(tmp_path):
    csv_path = tmp_path / "data.csv"

    file_manager.patchSwarmCSV(str(csv_path), 10, 1, 2, 0.5)
    file_manager.patchSwarmCSV(str(csv_path), 12, 0, 1, 0.6)

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["Timesteps", "Collisions", "Steps Waited", "Real-Time Elapsed"]
    assert df["Timesteps"].tolist() == [10, 12]
    assert df["Collisions"].tolist() == [1, 0]


def test_patchSwarmCSV_retries_a_transient_file_lock(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    original_to_csv = pd.DataFrame.to_csv
    calls = 0

    def temporarily_locked_to_csv(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("file is temporarily locked")
        return original_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", temporarily_locked_to_csv)
    monkeypatch.setattr("swarm_sim.file_manager.time.sleep", lambda _seconds: None)

    file_manager.patchSwarmCSV(str(csv_path), 10, 1, 2, 0.5)

    assert calls == 2
    assert pd.read_csv(csv_path)["Timesteps"].tolist() == [10]
