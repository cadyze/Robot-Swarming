"""
Closed-form mean/variance of hitting times via the fundamental matrix of an
absorbing Markov chain, solved with a sparse direct solver instead of
inverting (I - Q) densely.
"""
import os

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve


def calculate_mean_variance(arena, save_dir="."):
    """
    Args:
        arena: sparse transition matrix whose last row/col is the absorbing
            "bucket" state (i.e. built with for_math=True).
        save_dir: directory .npy snapshots are written to, for downstream
            analysis scripts (SimulatorMath, notebooks).

    Returns:
        (HT_mu, HT_variance, HT_std) mean/variance/std hitting times per state.
    """
    P_sparse = csc_matrix(arena[:-1, :-1])  # Submatrix Q (non-absorbing states)
    n = P_sparse.shape[0]
    I_sparse = eye(n, format="csc")

    IMQ = I_sparse - P_sparse
    b = np.ones(n)
    HT_mu = spsolve(IMQ, b)

    HT_mu_squared = np.square(HT_mu)
    HT_variance = spsolve(IMQ, 2 * HT_mu) - HT_mu - HT_mu_squared
    HT_std = np.sqrt(HT_variance)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "mean_matrix.npy"), HT_mu)
    np.save(os.path.join(save_dir, "variance_matrix.npy"), HT_variance)
    np.save(os.path.join(save_dir, "std_matrix.npy"), HT_std)

    return HT_mu, HT_variance, HT_std
