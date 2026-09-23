"""
One-off generator that captures the CURRENT (pre-refactor) simulator's
numeric behavior as golden fixtures, so the post-refactor sparse-matrix
implementation can be checked against it for equivalence.

Run this BEFORE refactoring, with the repo's original flat modules on
the path (RobotSwarmingSimulator, FileManager). Do not re-run it after
the refactor -- the fixtures it writes are the ground truth the new
code is judged against.
"""
import json
import os
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
sys.path.insert(0, SCRIPTS_DIR)

import RobotSwarmingSimulator as rss  # noqa: E402
from RobotSwarmingSimulator import STARTING_POSITION  # noqa: E402

ARENA_CONFIGS = [
    dict(name="grid4_base", grid_size=4, laziness_prob=0.0, sensing_range=1, obstacles=[], for_math=False),
    dict(name="grid5_base", grid_size=5, laziness_prob=0.0, sensing_range=1, obstacles=[], for_math=False),
    dict(name="grid5_lazy", grid_size=5, laziness_prob=0.1, sensing_range=1, obstacles=[], for_math=False),
    dict(name="grid5_sense2", grid_size=5, laziness_prob=0.0, sensing_range=2, obstacles=[], for_math=False),
    dict(name="grid13_sense3", grid_size=13, laziness_prob=0.0, sensing_range=3, obstacles=[], for_math=False),
    dict(name="grid7_obstacle", grid_size=7, laziness_prob=0.05, sensing_range=1,
         obstacles=[[(1, 1), (1, 2), (2, 2), (2, 1)]], for_math=False),
    dict(name="grid4_formath", grid_size=4, laziness_prob=0.0, sensing_range=1, obstacles=[], for_math=True),
    dict(name="grid5_lazy_formath", grid_size=5, laziness_prob=0.1, sensing_range=1, obstacles=[], for_math=True),
]

MEAN_VAR_CONFIGS = [
    dict(name="mv_grid4", grid_size=4, laziness_prob=0.0),
    dict(name="mv_grid5", grid_size=5, laziness_prob=0.05),
]


def build(cfg):
    grid_size = cfg["grid_size"]
    target_pos = (grid_size // 2, grid_size // 2)
    sim = rss.SwarmSimulator(
        grid_size=grid_size,
        target_pos=target_pos,
        obstacles=cfg["obstacles"],
        starting_pos=STARTING_POSITION.FILL,
        sensing_range=cfg["sensing_range"],
        laziness_prob=cfg["laziness_prob"],
        for_math=cfg["for_math"],
    )
    return sim


def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    arena_manifest = []
    for cfg in ARENA_CONFIGS:
        sim = build(cfg)
        arena_path = os.path.join(FIXTURES_DIR, f"arena_{cfg['name']}.npz")
        np.savez_compressed(
            arena_path,
            arena=sim.Arena,
            dynamic_arena=sim.DynamicObstacleArena,
        )
        row_sums = sim.Arena.sum(axis=1)
        arena_manifest.append({**cfg, "arena_shape": list(sim.Arena.shape),
                                "row_sum_min": float(row_sums.min()),
                                "row_sum_max": float(row_sums.max())})
        print(f"[arena] {cfg['name']}: shape={sim.Arena.shape}")

    mv_manifest = []
    for cfg in MEAN_VAR_CONFIGS:
        sim = build(dict(cfg, sensing_range=1, obstacles=[], for_math=True))
        ht_mu, ht_var, ht_std = sim.calculate_mean_variance()
        mv_path = os.path.join(FIXTURES_DIR, f"meanvar_{cfg['name']}.npz")
        np.savez_compressed(mv_path, ht_mu=ht_mu, ht_var=ht_var, ht_std=ht_std)
        mv_manifest.append({**cfg, "mu0": float(ht_mu[0]), "std0": float(ht_std[0])})
        print(f"[meanvar] {cfg['name']}: mu[0]={ht_mu[0]:.4f} std[0]={ht_std[0]:.4f}")

    # Loose statistical baseline for the stochastic wait-for-all pipeline
    # (this is the branch actually exercised by run_simulation/wait_all=True).
    # Exact per-run trajectories are allowed to change across implementations
    # (e.g. sparse row sampling consumes the RNG differently than dense
    # np.random.choice), so this is a distribution-shape sanity check only.
    np.random.seed(1234)
    sim = build(dict(name="stoch", grid_size=5, laziness_prob=0.0, sensing_range=1,
                      obstacles=[], for_math=False))
    move_counts = []
    for _ in range(200):
        timesteps, _collisions, _waits = sim.start_robot_swarming(
            2, rss.COLLISION_PROTOCOL.WAIT_NEXT, show_graph=False, wait_for_all=True)
        move_counts.extend(timesteps)
    stoch_summary = {
        "n_samples": len(move_counts),
        "mean": float(np.mean(move_counts)),
        "std": float(np.std(move_counts)),
        "min": int(np.min(move_counts)),
        "max": int(np.max(move_counts)),
    }
    print("[stochastic] 2-robot wait-all grid5 timesteps:", stoch_summary)

    with open(os.path.join(FIXTURES_DIR, "manifest.json"), "w") as f:
        json.dump({"arenas": arena_manifest, "mean_var": mv_manifest,
                    "stochastic_baseline": stoch_summary}, f, indent=2)

    print("Golden fixtures written to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
