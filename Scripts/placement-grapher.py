import os
import pandas as pd
import matplotlib.pyplot as plt

def load_robot_timesteps(root_dir):
    """Loads timesteps from all T#/data.csv files into a dict."""
    robot_data = {}
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        data_path = os.path.join(folder_path, "SwarmSimulationData.csv")
        if os.path.isfile(data_path):
            try:
                df = pd.read_csv(data_path)
                robot_data[folder_name] = df["Timesteps"].tolist()
            except Exception as e:
                print(f"Error reading {data_path}: {e}")
    return robot_data

def compute_simulation_stats(robot_data):
    """Returns lists of min, max, and average timesteps per simulation index."""
    num_simulations = len(next(iter(robot_data.values())))
    min_vals, max_vals, avg_vals = [], [], []

    for i in range(num_simulations):
        timesteps = [data[i] for data in robot_data.values() if i < len(data)]
        if not timesteps:
            continue
        min_vals.append(min(timesteps))
        max_vals.append(max(timesteps))
        avg_vals.append(sum(timesteps) / len(timesteps))

    return min_vals, max_vals, avg_vals

def plot_timestep_ranges(min_vals, max_vals, avg_vals, output_path="timesteps_line_plot.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(min_vals, label="Fastest (Min) | Avg: {}".format(sum(min_vals) / len(min_vals)), color='green', marker='o')
    plt.plot(max_vals, label="Slowest (Max) | Avg: {}".format(sum(max_vals) / len(max_vals)), color='red', marker='o')
    plt.plot(avg_vals, label="Average: {}".format(sum(avg_vals) / len(avg_vals)), color='blue', marker='o')

    plt.title("Timesteps per Simulation")
    plt.xlabel("Simulation Index")
    plt.ylabel("Timesteps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved line plot to: {output_path}")

# === RUN ===
for i in range(10, 21):
    root_dir = "./Data/A31/NON-RANDOM_SPAWNS/LZP_0/WAIT_NEXT/R{}".format(i)  # Replace with your actual path
    robot_data = load_robot_timesteps(root_dir)
    min_vals, max_vals, avg_vals = compute_simulation_stats(robot_data)
    plot_timestep_ranges(min_vals, max_vals, avg_vals, root_dir + "/timesteps_line_plot.png")
