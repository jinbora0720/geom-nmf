"""
GeomNMF quickstart
==================

Simulate data from a K-source simplex mixing model, fit GeomNMF, check the
recovered endmembers against the truth, and plot the source attribution.

Run with::

    pip install "geom-nmf[plot]"
    python examples/quickstart.py
"""

import numpy as np

from geom_nmf import GeomNMF, barplot_Phi, permute_to_reference

rng = np.random.default_rng(2026)

# ---------------------------------------------------------------------------
# 1. Simulate Y = W H with K sources and J features
# ---------------------------------------------------------------------------
K, J, N = 3, 6, 1000
feature_names = ["PM1", "PM2.5", "PM10", "CO", "NO2", "O3"]

H_true = rng.dirichlet(np.ones(J), size=K)               # (K, J) source profiles, rows sum to 1
W_true = rng.dirichlet(0.5 * np.ones(K), size=N)         # (N, K) mixing proportions
intensity = rng.lognormal(mean=0.0, sigma=0.5, size=(N, 1))
Y = intensity * (W_true @ H_true)                        # (N, J) observed, nonnegative

# ---------------------------------------------------------------------------
# 2. Fit GeomNMF
# ---------------------------------------------------------------------------
result = GeomNMF(Y, K=K, seed=1)

print(f"Fitted in {result.elapsed:.2f}s")
print(f"Intrinsic rank:        {result.fit_diagnostics['intrinsic_rank']} (expected K-1 = {K - 1})")
print(f"Candidates searched:   {result.fit_diagnostics['n_cand']}")
print(f"log-volume of simplex: {result.logvol:.4f}")
print(f"cond(H_aug H_aug^T):   {result.weight_diagnostics['cond_G']:.2e}")

# ---------------------------------------------------------------------------
# 3. Compare with the truth (sources are only identified up to permutation)
# ---------------------------------------------------------------------------
H_hat, mu_hat, Phi_hat, order = permute_to_reference(
    H_true, result.H_star, result.mu_tilde, result.Phi
)
err = np.abs(H_hat - H_true).max(axis=1)
print("Max abs. endmember error per source:", np.round(err, 4))

# ---------------------------------------------------------------------------
# 4. Plot the source attribution matrix Phi
# ---------------------------------------------------------------------------
barplot_Phi(
    Phi_hat,
    feature_labels=feature_names,
    title="GeomNMF source attribution (simulated data)",
    savepath="quickstart_phi.png",
)
