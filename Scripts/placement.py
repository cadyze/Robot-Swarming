import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_all_robot_data(root_dir):
    robot_data = {}
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        data_path = os.path.join(folder_path, "SwarmSimulationData.csv")
        if os.path.isfile(data_path):
            try:
                df = pd.read_csv(data_path)
                robot_index = folder_name  # e.g., "T0", "T1", etc.
                robot_data[robot_index] = df["Timesteps"].tolist()
            except Exception as e:
                print(f"Failed to load {data_path}: {e}")
    return robot_data

def calculate_placements(robot_data):
    placement_counts = defaultdict(lambda: defaultdict(int))  # robot -> placement -> count
    num_simulations = len(next(iter(robot_data.values())))  # assumes all same length

    for sim_idx in range(num_simulations):
        sim_results = []
        for robot, timesteps in robot_data.items():
            if sim_idx < len(timesteps):
                sim_results.append((robot, timesteps[sim_idx]))

        sim_results.sort(key=lambda x: x[1])  # lower timesteps = better
        for rank, (robot, _) in enumerate(sim_results, start=1):
            placement_counts[robot][rank] += 1

    return placement_counts

def save_individual_placement_charts(placement_counts, output_dir="placement_charts"):
    os.makedirs(output_dir, exist_ok=True)
    robots = sorted(placement_counts.keys())
    max_rank = max(max(ranks.keys()) for ranks in placement_counts.values())

    for rank in range(1, max_rank + 1):
        values = [placement_counts[robot].get(rank, 0) for robot in robots]

        plt.figure(figsize=(10, 5))
        plt.bar(robots, values, color="skyblue")
        plt.title(f"Number of Times Each Robot Got {rank}{ordinal_suffix(rank)} Place")
        plt.xlabel("Robot")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = os.path.join(output_dir, f"placement_{rank}.png")
        plt.savefig(filename)
        plt.close()

def ordinal_suffix(n):
    """Returns ordinal string (1st, 2nd, 3rd, etc.)"""
    return "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

# === MAIN ===
for i in range(10, 21):
    root_dir = "./Data/A31/NON-RANDOM_SPAWNS/LZP_0/WAIT_NEXT/R{}".format(i)  # Replace with your actual folder path
    robot_data = load_all_robot_data(root_dir)
    placement_counts = calculate_placements(robot_data)
    save_individual_placement_charts(placement_counts, root_dir + "/placement_charts")
