import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

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
    plt.plot(x, y, marker="o", linestyle="none", label="Empirical CDF")
    plt.plot(x, gamma_cdf, linestyle="dashed", label="Fitted Gamma CDF", color='red')

    # Labels and legend
    plt.xlabel("Data Values")
    plt.ylabel("Cumulative Probability")
    plt.title("Empirical CDF vs. Fitted Gamma CDF")
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
example_data = np.random.gamma(shape=2, scale=2, size=3000)

# Call function with the data
plot_gamma_cdf(example_data)
