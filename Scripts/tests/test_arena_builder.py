import json
import os

import numpy as np
import pytest
import scipy.sparse as sp

from swarm_sim.arena_builder import build_arena

from conftest import FIXTURES_DIR

with open(os.path.join(FIXTURES_DIR, "manifest.json")) as f:
    MANIFEST = json.load(f)


def _load_golden(name):
    data = np.load(os.path.join(FIXTURES_DIR, f"arena_{name}.npz"), allow_pickle=True)
    return data["arena"], data["dynamic_arena"]


@pytest.mark.parametrize("cfg", MANIFEST["arenas"], ids=lambda c: c["name"])
def test_matches_pre_refactor_golden(cfg):
    grid_size = cfg["grid_size"]
    target_pos = (grid_size // 2, grid_size // 2)
    obstacles = [[tuple(node) for node in loop] for loop in cfg["obstacles"]]

    arena, dynamic_arena = build_arena(
        grid_size=grid_size,
        target_pos=target_pos,
        obstacles=obstacles,
        laziness_prob=cfg["laziness_prob"],
        sensing_range=cfg["sensing_range"],
        for_math=cfg["for_math"],
    )

    assert sp.issparse(arena)
    assert sp.issparse(dynamic_arena)
    assert arena.shape == tuple(cfg["arena_shape"])

    golden_arena, golden_dynamic = _load_golden(cfg["name"])
    np.testing.assert_allclose(arena.toarray(), golden_arena, atol=1e-10)
    np.testing.assert_allclose(dynamic_arena.toarray(), golden_dynamic, atol=1e-10)


@pytest.mark.parametrize("cfg", [c for c in MANIFEST["arenas"] if c["laziness_prob"] == 0.0],
                          ids=lambda c: c["name"])
def test_rows_are_stochastic_when_not_lazy(cfg):
    """With no laziness the transition matrix must be row-stochastic (or an
    all-zero absorbing row for the extra bucket state in for_math mode)."""
    grid_size = cfg["grid_size"]
    target_pos = (grid_size // 2, grid_size // 2)
    obstacles = [[tuple(node) for node in loop] for loop in cfg["obstacles"]]

    arena, _ = build_arena(
        grid_size=grid_size,
        target_pos=target_pos,
        obstacles=obstacles,
        laziness_prob=cfg["laziness_prob"],
        sensing_range=cfg["sensing_range"],
        for_math=cfg["for_math"],
    )
    row_sums = np.asarray(arena.sum(axis=1)).ravel()
    for total in row_sums:
        assert total == pytest.approx(0.0, abs=1e-9) or total == pytest.approx(1.0, abs=1e-9)
