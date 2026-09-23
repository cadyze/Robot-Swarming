import json
import os

import numpy as np
import pytest

from swarm_sim.arena_builder import build_arena
from swarm_sim.hitting_time_math import calculate_mean_variance

from conftest import FIXTURES_DIR

with open(os.path.join(FIXTURES_DIR, "manifest.json")) as f:
    MANIFEST = json.load(f)


@pytest.mark.parametrize("cfg", MANIFEST["mean_var"], ids=lambda c: c["name"])
def test_matches_pre_refactor_golden(cfg, tmp_path):
    grid_size = cfg["grid_size"]
    target_pos = (grid_size // 2, grid_size // 2)

    arena, _ = build_arena(
        grid_size=grid_size,
        target_pos=target_pos,
        obstacles=[],
        laziness_prob=cfg["laziness_prob"],
        sensing_range=1,
        for_math=True,
    )

    golden = np.load(os.path.join(FIXTURES_DIR, f"meanvar_{cfg['name']}.npz"))

    ht_mu, ht_var, ht_std = calculate_mean_variance(arena, save_dir=str(tmp_path))

    np.testing.assert_allclose(ht_mu, golden["ht_mu"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(ht_var, golden["ht_var"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(ht_std, golden["ht_std"], rtol=1e-6, atol=1e-8)

    # calculate_mean_variance persists .npy snapshots for downstream analysis scripts
    for name in ("mean_matrix.npy", "variance_matrix.npy", "std_matrix.npy"):
        assert (tmp_path / name).exists()
