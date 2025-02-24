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
        m_values = m_values[m_values <= 6000]
    
    return np.mean(m_values), np.std(m_values), m_values

class GammaData:
    def __init__(self, csv=None, mean=None, std=None, data=None):
        if csv:
            mean, std, data = compute_mean_std_data(csv)
        self.mean = mean
        self.std = std
        self.data = data

def fit_and_plot_gamma_distributions(gamma_data_list):
    plt.figure(figsize=(10, 6))
    for gamma_data in gamma_data_list:
        mean = gamma_data.mean
        std = gamma_data.std
        data = gamma_data.data
        shape, loc, scale = stats.gamma.fit(data, floc=100)
        # Calculate shape and scale parameters
        # shape = (mean / std) ** 2
        # scale = std ** 2 / mean
        
        # Generate x values
        x = np.linspace(0, 9000, 1000)
        
        # Generate gamma distribution
        gamma_dist = stats.gamma.pdf(x, shape, scale=scale)
        
        # Plot gamma distribution
        plt.plot(x, gamma_dist, label=f'Gamma Distribution (mean={round(mean, 2)}, std={round(std, 2)})')
        
        if data is not None:
            # Plot the data histogram
            plt.hist(data, bins=120, density=True, alpha=0.5, label=f'Data | (mean={round(mean, 2)}, std={round(std, 2)})')
    
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Gamma Distribution Fit')
    plt.show()

# Example usage
gamma_data_list = [
    GammaData(csv='./A61_R1_LZP0,0_ST0/A61_R1_LZP0,0_ST0.csv'),
    # GammaData(csv='./A61_R1_LZP0,12_ST0/A61_R1_LZP0,12_ST0.csv'),
    # GammaData(csv='./A61_R5_RR_LZ-0_ST0_WN.csv'),
    # GammaData(csv='./A61_R10_RR_LZ-0_ST0_WN.csv'),
    GammaData(mean=1835.4358446260867, std=1617.2849372380126) # G61, LZ-0, ST0
    # GammaData(mean=15, std=3, data=np.random.gamma(shape=(15 / 3) ** 2, scale=3 ** 2 / 15, size=1000))
]

fit_and_plot_gamma_distributions(gamma_data_list)
