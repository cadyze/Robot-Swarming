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
        m_values = m_values[m_values > 0]
    
    return np.mean(m_values), np.std(m_values), m_values

class GammaData:
    def __init__(self, csv=None, mean=None, std=None, data=None):
        if csv:
            mean, std, data = compute_mean_std_data(csv)
            self.csv = csv
        self.mean = mean
        self.std = std
        self.data = data

def fit_and_plot_gamma_distributions(gamma_data_list):
    plt.figure(figsize=(10, 6))
    for gamma_data in gamma_data_list:
        mean = gamma_data.mean
        std = gamma_data.std
        data = gamma_data.data

        MIN_STEPS_NEEDED = 10
        shape, loc, scale = stats.gamma.fit(data, floc=MIN_STEPS_NEEDED)
        # Calculate shape and scale parameters
        # shape = (mean / std) ** 2
        # scale = std ** 2 / mean
        
        # Generate x values
        x = np.linspace(0, 200, 1000)
        
        # Generate gamma distribution
        gamma_dist = stats.gamma.pdf(x-MIN_STEPS_NEEDED, shape, scale=scale)
        
        # Plot gamma distribution
        if not gamma_data.csv:
            plt.plot(x, gamma_dist, label=f'Gamma Distribution (mean={round(mean, 2)}, std={round(std, 2)})')
        else:
            plt.plot(x, gamma_dist, label=f'{gamma_data.csv}')

        
        if data is not None:
            # Plot the data histogram
            plt.hist(data, bins=120, density=True, alpha=0.5, label=f'Data | (mean={round(mean, 2)}, std={round(std, 2)})')
    
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Gamma Distribution Fit')
    plt.show()


def plot_gamma_cdf(data):
    """
    Fits a Gamma distribution to the given data, computes the empirical CDF,
    and plots both the empirical CDF and the fitted Gamma CDF.

    Parameters:
    - data: array-like, the dataset to analyze.
    """
    if len(data) == 0:
        print("Error: Data array is empty.")
        return
    
    # Sort the data for empirical CDF
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)  # Empirical CDF

    # Fit a Gamma distribution to the data
    shape, loc, scale = stats.gamma.fit(data, floc=0)  # Fix loc=0 for standard Gamma

    # Compute the fitted Gamma CDF
    gamma_cdf = stats.gamma.cdf(x, shape, loc, scale)

    # Perform KS test
    ks_stat, p_value = stats.kstest(data, 'gamma', args=(shape, loc, scale))

    # Plot empirical CDF vs fitted Gamma CDF
    plt.figure(figsize=(8, 6))
    plt.plot(x, gamma_cdf, label="Fitted Gamma CDF", color='deeppink')
    plt.plot(x, y, marker=".", label="Empirical CDF", color='royalblue', ms=1)

    # Labels and legend
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Probability")
    plt.title("Empirical (R1, 11x11) CDF vs. Fitted Gamma CDF")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print results
    print(f"Estimated Gamma Parameters: shape={shape:.4f}, scale={scale:.4f}")
    print(f"KS Statistic: {ks_stat:.4f}, P-Value: {p_value:.4f}")
    if p_value > 0.05:
        print("Result: The Gamma distribution is a good fit for the data.")
    else:
        print("Result: The Gamma distribution may not be a good fit for the data.")

# Example usage:
# Generate example gamma-distributed data
# example_data = np.random.gamma(shape=2, scale=2, size=3000)

# # Call function with the data
# plot_gamma_cdf(example_data)


# Example usage
gamma_data_list = [
    # GammaData(csv='./A11/MULTI_R/RR/A11_R5_RR_LZP0_ST0_WN/A11_R5_RR_LZP0_ST0_WN.csv'),
    # GammaData(csv='./A11/N_RR/A11_R5_FILL_LZP0_ST0_WN/A11_R5_FILL_LZP0_ST0_WN.csv'),
    # GammaData(csv='./A11/N_RR/A11_R26_FILL_LZP0_ST0_WN/A11_R26_FILL_LZP0_ST0_WN.csv'),
    # GammaData(csv='./A11_R13_RR_LZP0_ST0_WN/A11_R13_RR_LZP0_ST0_WN.csv'),
    GammaData(csv='./A31/RR/A31_R15_RR_LZP0_ST0_WN/A31_R15_RR_LZP0_ST0_WN.csv'),
    GammaData(csv='./A11/MULTI_R/N_RR/A11_R1_LZP0_ST0/A11_R1_LZP0_ST0.csv'),
    # GammaData(mean=1835.4358446260867, std=1617.2849372380126) # G61, LZ-0, ST0
    # GammaData(mean=15, std=3, data=np.random.gamma(shape=(15 / 3) ** 2, scale=3 ** 2 / 15, size=1000))
]

# fit_and_plot_gamma_distributions(gamma_data_list)
plot_gamma_cdf(gamma_data_list[0].data)
