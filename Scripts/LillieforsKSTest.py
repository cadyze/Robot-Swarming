"""
KSTest.perform_ks_test fits a gamma distribution's parameters (shape, scale)
from the *same data* it then runs a KS test against. scipy.stats.kstest's
p-value assumes the reference distribution is fully specified in advance --
when parameters are instead estimated from the data, the KS statistic is
systematically pulled down (the fit was chosen to match the data), so the
textbook p-value is biased toward "looks like a good fit" and can't be
trusted. This is exactly the composite-hypothesis problem Lilliefors (1967)
corrected for the normal distribution by tabulating the null distribution of
the KS statistic via simulation. There's no equivalent published table for
gamma, so this reproduces Lilliefors' method directly via a parametric
bootstrap: repeatedly simulate data from the fitted distribution, refit it
the same way the real pipeline does, and see how extreme the KS statistic
gets by chance alone. That empirical distribution is what the observed
statistic gets compared against, instead of the standard KS table.
"""
import numpy as np
import scipy.stats as stats


class LillieforsKSTest:
    @staticmethod
    def perform_ks_test(samples, population, n_bootstrap=1000, random_state=None):
        """
        Mirrors KSTest.perform_ks_test's fit-on-population/test-on-samples
        setup, but returns a Lilliefors-corrected p-value alongside the
        naive one.

        Returns:
            ks_statistic, naive_p_value, corrected_p_value, fitted_params
        """
        shape, loc, scale = stats.gamma.fit(population, floc=0)
        gamma_cdf = lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale)
        ks_statistic, naive_p_value = stats.kstest(samples, gamma_cdf)

        rng = np.random.default_rng(random_state)
        n_population = len(population)
        n_samples = len(samples)
        boot_stats = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            # Simulate a synthetic population from the FITTED distribution,
            # then refit gamma to it exactly as the real pipeline refits to
            # its own population -- this is what makes it a "Lilliefors-type"
            # correction rather than a plain bootstrap.
            synth_population = stats.gamma.rvs(shape, loc=loc, scale=scale,
                                                size=n_population, random_state=rng)
            s_shape, s_loc, s_scale = stats.gamma.fit(synth_population, floc=0)
            synth_samples = rng.choice(synth_population, size=n_samples, replace=False)
            synth_cdf = lambda x, sh=s_shape, lo=s_loc, sc=s_scale: stats.gamma.cdf(x, sh, loc=lo, scale=sc)
            boot_stats[b], _ = stats.kstest(synth_samples, synth_cdf)

        corrected_p_value = float(np.mean(boot_stats >= ks_statistic))
        return ks_statistic, naive_p_value, corrected_p_value, (shape, loc, scale)


if __name__ == "__main__":
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from swarm_sim import csv_utils as CSVUtils  # noqa: E402
    from KSTest import KSTest  # noqa: E402

    CSV_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "A31", "RR", "A31_R15_RR_LZP0_ST0_WN", "A31_R15_RR_LZP0_ST0_WN.csv")
    SAMPLE_SIZE = 300
    N_BOOTSTRAP = 1000

    utils = CSVUtils.CSVUtils(CSV_PATH)
    utils.generate_new_sample(SAMPLE_SIZE)
    population = utils.m_values
    samples = utils.m_sample_values

    print(f"population size: {len(population)} | sample size: {len(samples)}")

    ks_stat, naive_p = KSTest.perform_ks_test(samples, population)
    print(f"\n[naive scipy.stats.kstest, existing KSTest.py]  D={ks_stat:.4f}  p={naive_p:.4f}")

    t0 = time.perf_counter()
    ks_stat2, naive_p2, corrected_p, params2 = LillieforsKSTest.perform_ks_test(
        samples, population, n_bootstrap=N_BOOTSTRAP, random_state=0)
    elapsed = time.perf_counter() - t0
    print(f"[Lilliefors-corrected, B={N_BOOTSTRAP}] D={ks_stat2:.4f}  "
          f"naive_p={naive_p2:.4f}  corrected_p={corrected_p:.4f}  ({elapsed:.1f}s)")

    if corrected_p < naive_p2:
        print(f"\nCorrection lowered the p-value by {naive_p2 - corrected_p:.4f} "
              "-- the naive test was overstating goodness of fit.")
    else:
        print("\nCorrection did not lower the p-value in this case.")
