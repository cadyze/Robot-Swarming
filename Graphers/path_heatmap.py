import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def create_heatmap(combined_values, subdir):
    # Create a heatmap for the combined array
    plt.figure(figsize=(8, 6))  # Adjust the figure size
    plt.imshow(combined_values, cmap='viridis', interpolation='nearest')  # Heatmap
    plt.colorbar(label='Values')  # Add a colorbar
    plt.title("Summed Heatmap")  # Title for the heatmap
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.gca().invert_yaxis()
    plt.savefig("{}/PositionVisits.png".format(subdir), dpi=300, bbox_inches='tight')

def graph_all():
    # Define the root directory where the search will start
    root_dir = '.'

    # Walk through all subdirectories and files
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "PathData.json":
                # Construct the full file path
                file_path = os.path.join(subdir, file)

                position_histories = []
                # Load the JSON file
                with open(file_path, "r") as json_file:
                    loaded_data = json.load(json_file)
                    position_histories = loaded_data.get("position_visits")
                    combined_visits = np.sum(position_histories, axis=0)
                    print(combined_visits)
                    create_heatmap(combined_visits, subdir)


    print("Heatmaps have been saved for all PathData files.")

graph_all()