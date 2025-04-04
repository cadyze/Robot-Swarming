import numpy as np
from scipy import stats
import scipy.stats as stats

class KSTest:
    def __init__(self):
        pass

    @staticmethod
    def perform_ks_test(samples, population):
        # Create gamma off of the mean and var
        shape, loc, scale = stats.gamma.fit(population, floc=0)
        print(f"Fitting off of population size: {len(population)}, and samples size: {len(samples)}")
        # x = np.linspace(min_steps, 200, 1000)
        gamma_dist = lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale)
        ks_statistic, p_value = stats.kstest(samples, gamma_dist)
        return ks_statistic, p_value

    # def perform_ks_test(sr_mean, sr_var, mr_mean, mr_var, n_mr):
    #     # Given empirical population statistics for single-robot
    #     single_robot_mean = sr_mean
    #     single_robot_var = sr_var

    #     multi_robot_mean = mr_mean
    #     multi_robot_var = mr_var

    #     n_multi = n_mr  # Sample size for multi-robot system

    #     # Generate synthetic normal distribution for multi-robot samples
    #     np.random.seed(0)  # For reproducibility
    #     multi_robot_samples = np.random.normal(multi_robot_mean, np.sqrt(multi_robot_var), n_multi)

    #     # Compute the empirical CDF of the single-robot distribution (assumed normal)
    #     from scipy.stats import norm

    #     # Theoretical single-robot distribution CDF
    #     single_robot_cdf = lambda x: norm.cdf(x, loc=single_robot_mean, scale=np.sqrt(single_robot_var))

    #     # Compute KS test using empirical vs theoretical distribution
    #     ks_statistic, p_value = stats.kstest(multi_robot_samples, single_robot_cdf)

    #     # Display results
    #     print("KS-STAT: {} | p_value: {}".format(ks_statistic, p_value))