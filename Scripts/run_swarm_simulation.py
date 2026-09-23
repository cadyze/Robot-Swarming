"""Command-line entry point for swarm simulations and CSV result collection."""
import argparse

from swarm_sim.enums import COLLISION_PROTOCOL
from swarm_sim.simulation_runner import run_simulation

PROTOCOLS = {
    "wait-next": COLLISION_PROTOCOL.WAIT_NEXT,
    "find-next": COLLISION_PROTOCOL.FIND_NEXT_AVAILABLE,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hex-grid swarm hitting-time simulations.")
    parser.add_argument("--grid-size", type=int, default=11,
                        help="Square arena side length (default: 11).")
    parser.add_argument("--robots", type=int, default=3,
                        help="Number of robots to simulate (default: 3).")
    parser.add_argument("--iterations", type=int, default=3000,
                        help="Independent runs to record (default: 3000).")
    parser.add_argument("--laziness", type=float, default=0.0,
                        help="Probability that a robot stays put on each step (default: 0).")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="wait-next",
                        help="Collision handling protocol (default: wait-next).")
    parser.add_argument("--tracked-robot", type=int, default=-1,
                        help="Robot index to track; omit for the first robot to reach the target.")
    parser.add_argument("--first-hit", action="store_true",
                        help="Stop when a robot reaches the target instead of waiting for every robot.")
    parser.add_argument("--random-spawns", action="store_true",
                        help="Place robots randomly rather than using the configured deterministic layout.")
    parser.add_argument("--show-graph", action="store_true",
                        help="Display the path graph for each run.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.robots < 1:
        raise SystemExit("--robots must be at least 1.")
    if args.tracked_robot >= args.robots:
        raise SystemExit("--tracked-robot must be smaller than --robots.")
    if args.tracked_robot >= 0 and not args.first_hit:
        raise SystemExit("--tracked-robot requires --first-hit.")
    run_simulation(
        grid_size=args.grid_size,
        num_robots=args.robots,
        collision_protocol=PROTOCOLS[args.protocol],
        laziness_prob=args.laziness,
        tracked_robot=args.tracked_robot,
        show_graph=args.show_graph,
        generate_random_obs=args.random_spawns,
        wait_all=not args.first_hit,
        num_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
