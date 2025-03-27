import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

def compute_mean_std_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    m_values = []

    # Check if 'Timesteps' column exists in the CSV
    if 'Timesteps' in df.columns:
        # Extract the 'M' values
        m_values = df['Timesteps'].values
    
    return np.mean(m_values), np.std(m_values), m_values

class GammaData:
    def __init__(self, csv=None, mean=None, std=None, data=None):
        if csv:
            mean, std, data = compute_mean_std_data(csv)
        self.mean = mean
        self.std = std
        self.data = data

def fit_and_plot_gamma_distributions(gamma_data_list):
    for gamma_data in gamma_data_list:
        mean = gamma_data.mean
        std = gamma_data.std
        data = gamma_data.data
        
        # Calculate shape and scale parameters
        shape = (mean / std) ** 2
        scale = std ** 2 / mean
        
        # Generate x values
        x = np.linspace(0, mean + 4 * std, 1000)
        
        # Generate gamma distribution
        gamma_dist = stats.gamma.pdf(x, shape, scale=scale)
        
        # Plot gamma distribution
        plt.plot(x, gamma_dist, label=f'Gamma Distribution (mean={mean}, std={std})')
        
        if data is not None:
            # Plot the data histogram
            plt.hist(data, bins=30, density=True, alpha=0.5, label=f'Data (mean={mean}, std={std})')
    
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Gamma Distribution Fit')
    plt.show()

# Example usage
gamma_data_list = [
    GammaData(csv='path/to/csv1.csv'),
    GammaData(mean=15, std=3, data=np.random.gamma(shape=(15 / 3) ** 2, scale=3 ** 2 / 15, size=1000))
]

fit_and_plot_gamma_distributions(gamma_data_list)
