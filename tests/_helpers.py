"""Synthetic-data helpers shared across tests."""

import numpy as np

SEED = 123


def make_simplex_data(K, J, N, seed=SEED, scale=False, noise=0.0):
    """Draw Y = W H from a simplex mixing model with known H and W.

    Pure endmembers are planted as the first K rows so the model is exactly
    identifiable from the data.
    """
    rng = np.random.default_rng(seed)
    H = rng.dirichlet(np.ones(J), size=K)                 # (K, J) row-stochastic
    W = rng.dirichlet(np.ones(K), size=N)                 # (N, K) row-stochastic
    W[:K] = np.eye(K)                                     # pure pixels present
    if scale:
        W = W * rng.uniform(0.5, 5.0, size=(N, 1))        # varying intensities
    Y = W @ H
    if noise:
        Y = Y + noise * rng.random(Y.shape)
    return Y, H, W
