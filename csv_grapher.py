import pandas as pd
import matplotlib.pyplot as plt
def graph_multiple():
    # Example list of CSV file paths
    # csv_files = ['R10\FN\A61_R10_FN_T0_EDGE.csv', 'R10/FN/A61_R10_FN_T0_FILL.csv', 'R10/WN/A61_R10_WN_T9_EDGE.csv']
    csv_files = ['R10/FN/A61_R10_FN_T0_FILL.csv']

    # Initialize a list of colors for different datasets
    colors = ['blue', 'green', 'red']

    # Initialize an empty list to store all M values (to calculate common bins)
    all_m_values = []

    # First, gather all M values from all files to determine common bins
    for file in csv_files:
        df = pd.read_csv(file)
        all_m_values.extend(df['Timesteps'].values)

    # Define common bins using all the M values
    bins = plt.hist(all_m_values, bins=30)[1]  # This returns the bin edges

    # Clear the current plot to avoid plotting the initial histogram
    plt.clf()

    # Now plot each dataset with the same bins
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file)
        m_values = df['Timesteps'].values
        
        # Plot the histogram for this dataset with common bins
        plt.hist(m_values, bins=bins, alpha=0.5, color=colors[i], label=f'Dataset {i+1}', edgecolor='black')

    # Add labels and title
    plt.xlabel('Timesteps')
    plt.ylabel('Frequency')
    plt.title('Histogram of M Values from Multiple CSV Files')

    # Add a legend to differentiate datasets
    plt.legend()

    # Show the plot
    plt.show()

def graph_all():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    # Define the root directory where the search will start
    root_dir = '.'

    # Walk through all subdirectories and files
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(subdir, file)
                
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Check if 'M' column exists in the CSV
                if 'Timesteps' in df.columns:
                    # Extract the 'M' values
                    m_values = df['Timesteps'].values
                    
                    # Plot the histogram for this CSV file
                    plt.figure()  # Create a new figure for each CSV file
                    plt.hist(m_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
                    
                    # Add labels and title
                    plt.xlabel('Timesteps')
                    plt.ylabel('Frequency')
                    plt.title(f'Histogram of M Values from {file}')
                    
                    # Save the plot to a file (same directory as the CSV)
                    output_file = os.path.join(subdir, f'{os.path.splitext(file)[0]}_histogram.png')
                    plt.savefig(output_file)
                    
                    # Close the plot to free up memory
                    plt.close()

    print("Histograms have been saved for all CSV files.")

def compute_mean_std(file_path):
    import numpy as np

    # Read the CSV file
    df = pd.read_csv(file_path)
    m_values = []

    # Check if 'Timesteps' column exists in the CSV
    if 'Timesteps' in df.columns:
        # Extract the 'M' values
        m_values = df['Timesteps'].values
    
    return np.mean(m_values), np.std(m_values)

graph_all()
mean, std = compute_mean_std("A30_R1_WN_FILL.csv")
print("MEAN: {} | STD_DEV: {}".format(mean, std))