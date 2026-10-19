# geom-nmf

[![PyPI](https://img.shields.io/pypi/v/geom-nmf)](https://pypi.org/project/geom-nmf/)
[![Tests](https://github.com/jinbora0720/geom-nmf/actions/workflows/tests.yml/badge.svg)](https://github.com/jinbora0720/geom-nmf/actions/workflows/tests.yml)
![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Geometric non-negative matrix factorization (GeomNMF)** via maximum-volume simplex fitting.

GeomNMF assumes a nonnegative data matrix follows a simplex mixing model

```
Y ≈ W H,   W ≥ 0,   H ≥ 0,   H 1 = 1
```

and estimates the source profiles `H` (endmembers) as the vertices of the maximum-volume
(K−1)-simplex inscribed in the convex hull of the row-normalized data. It then recovers the
mixing weights `W` and a source attribution matrix `Φ` giving the share of each feature
attributable to each source. Typical applications include source apportionment of
multi-pollutant air quality data and spectral unmixing.

# Getting Started

## Installation

```sh
pip install geom-nmf
```

With optional plotting (matplotlib) and progress-bar (tqdm) support:

```sh
pip install "geom-nmf[all]"
```

| Extra      | Enables                                                      |
|------------|--------------------------------------------------------------|
| `plot`     | `barplot_Phi` (installs matplotlib)                          |
| `progress` | tqdm progress bar for the max-volume search (`verbose=True`) |
| `all`      | both                                                         |

To get the latest development version from GitHub instead:

```sh
pip install "geom-nmf[all] @ git+https://github.com/jinbora0720/geom-nmf.git"
```

## Quick example

```python
import numpy as np
from geom_nmf import GeomNMF, barplot_Phi, permute_to_reference

rng = np.random.default_rng(0)
K, J, N = 3, 6, 1000
H_true = rng.dirichlet(np.ones(J), size=K)          # (K, J) source profiles
W_true = rng.dirichlet(0.5 * np.ones(K), size=N)    # (N, K) mixing proportions
Y = rng.lognormal(size=(N, 1)) * (W_true @ H_true)  # (N, J) nonnegative data

result = GeomNMF(Y, K=3)

result.H_star      # (K, J) estimated source profiles, rows sum to 1
result.W_tilde     # (N, K) source contributions on the original scale of Y
result.mu_tilde    # (K,)   mean contribution per source
result.Phi         # (K, J) share of each feature attributable to each source
result.fit_diagnostics["intrinsic_rank"]   # should be K - 1 for noise-free data

# sources are identified up to permutation; align with a reference if you have one
H_hat, mu_hat, Phi_hat, order = permute_to_reference(H_true, result.H_star, result.mu_tilde, result.Phi)

barplot_Phi(Phi_hat, feature_labels=["PM1", "PM2.5", "PM10", "CO", "NO2", "O3"])
```

`Y` can be a NumPy array, a pandas DataFrame, or any 2-D array-like. A runnable version is in
[`examples/quickstart.py`](examples/quickstart.py).

## Public API

| Function / class        | Purpose                                                          |
|-------------------------|------------------------------------------------------------------|
| `GeomNMF`               | Full pipeline; returns a `GeomNMFResult` named tuple             |
| `nfindr`                | Standalone N-FINDR endmember extraction                          |
| `log_simplex_volume`    | Log-volume of the simplex spanned by K points                    |
| `estimate_weights`      | Recover simplex weights `W` given `Y` and `H`                    |
| `compute_Phi`           | Source attribution matrix from mean weights and `H`              |
| `permute_to_reference`  | Align estimated sources to a reference ordering (Hungarian)      |
| `barplot_Phi`           | Grouped bar plot of `Φ`, optionally with bootstrap error bars    |

## Choosing options

- **`candidate_method="exact"`** (default) uses convex-hull vertices and is globally optimal,
  but the number of hull vertices grows quickly for noisy, high-dimensional data.
- **`candidate_method="random"`** uses extremes along random projections; cap the pool with
  `n_candidates`.
- **`prune=True`** clusters candidates before the exhaustive search. It can discard the true
  extremes, so pair it with **`refine_greedy=True`**, which runs N-FINDR swaps on the full data
  and keeps the result only if it increases the simplex volume.
- A `RuntimeWarning` is raised if the exhaustive search would exceed 50 million subsets.

## Citation

GeomNMF is described in:

> Jin, B., Spada, N. J., Creamean, J., Shupe, M., & Datta, A. (2025).
> *Geometric Identification and Consistent Estimation of Source Apportionment with
> Application to Greenland Summit Aerosols.* arXiv:2510.03616.
> https://arxiv.org/abs/2510.03616

If you use geom-nmf in your work, please cite this paper:

```bibtex
@misc{jin2025geometric,
  title         = {Geometric Identification and Consistent Estimation of Source Apportionment
                   with Application to Greenland Summit Aerosols},
  author        = {Jin, Bora and Spada, Nicholas J. and Creamean, Jessie and Shupe, Matthew
                   and Datta, Abhirup},
  year          = {2025},
  eprint        = {2510.03616},
  archivePrefix = {arXiv},
  primaryClass  = {math.ST}
}
```

GitHub's **"Cite this repository"** button (from [`CITATION.cff`](CITATION.cff)) gives the same
reference in other formats.

### Applications

- Jin, B., Salmerón, B. D., McClosky, D., et al. (2025). *Source apportionment of air pollution
  burden using geometric non-negative matrix factorization and high-throughput multi-pollutant
  air sensor data in Curtis Bay, Baltimore, USA.* arXiv:2511.11833.
  https://arxiv.org/abs/2511.11833

## Contributing

Bug reports, questions, and pull requests are welcome. Please open an
[issue](https://github.com/jinbora0720/geom-nmf/issues) to report a problem or discuss an idea.
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment, run the
tests, and build the documentation.

## License

&copy; 2026 Bora Jin. Released under the [MIT License](LICENSE).
