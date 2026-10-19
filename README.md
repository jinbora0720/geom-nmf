# geom-nmf

![Tests](https://github.com/jinbora0720/geom-nmf/actions/workflows/tests.yml/badge.svg)
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

Install the latest version directly from GitHub:

```sh
pip install git+https://github.com/jinbora0720/geom-nmf.git
```

With optional plotting (matplotlib) and progress-bar (tqdm) support:

```sh
pip install "geom-nmf[all] @ git+https://github.com/jinbora0720/geom-nmf.git"
```

| Extra      | Enables                                                     |
|------------|-------------------------------------------------------------|
| `plot`     | `barplot_Phi` (installs matplotlib)                         |
| `progress` | tqdm progress bar for the max-volume search (`verbose=True`) |
| `all`      | both                                                        |

Install from a specific branch or tag:

```sh
pip install git+https://github.com/jinbora0720/geom-nmf.git@<branch-name-or-tag>
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

## Contributing

We welcome contributions! If you'd like to fix a bug, add a feature, or improve documentation, follow the steps below. Don't worry if you're new to this — it's a straightforward process once you've done it a couple of times.

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments.

### 1. Fork the repository

A "fork" is your own personal copy of the project on GitHub. To create one, click the **Fork** button in the top-right corner of the [geom-nmf repository](https://github.com/jinbora0720/geom-nmf) page.

### 2. Clone your fork

Download your fork to your local machine:

```sh
git clone https://github.com/<your-github-username>/geom-nmf.git
cd geom-nmf
```

### 3. Install Poetry

If you don't already have Poetry installed, choose the instructions for your operating system below.

**macOS**

The recommended way on macOS is via [Homebrew](https://brew.sh/). If you don't have Homebrew installed, install it first:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Poetry:

```sh
brew install poetry
```

**Linux / Windows (WSL)**

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Then follow any instructions it prints to add Poetry to your `PATH`.

**Verify the installation**

```sh
poetry --version
```

### 4. Install dependencies

This installs both the main dependencies and the development dependencies (e.g. pytest, sphinx) into an isolated virtual environment managed by Poetry:

```sh
poetry install
```

To also install the optional coverage or documentation tools:

```sh
poetry install --with coverage
poetry install --with docs
```

You only need to run this once (and again whenever dependencies change).

To activate the virtual environment (Poetry ≥ 2.0):

```sh
eval $(poetry env activate)
```

On Poetry 1.x, use `poetry shell` instead.

Your prompt will change to indicate you're inside the environment. To exit when you're done:

```sh
exit
```

Alternatively, prefix any command with `poetry run` to run it inside the environment without activating it:

```sh
poetry run pytest tests/
```

### 5. Create a branch

Never make changes directly on `main`. Instead, create a new branch with a short descriptive name:

```sh
git checkout -b my-feature-branch
```

### 6. Make your changes

Edit files, write code, fix bugs — whatever your contribution involves. Keep your changes focused on one thing per branch.

### 7. Run the tests

Before submitting, make sure all tests pass:

```sh
poetry run pytest tests/
```

If you've added new functionality, add a corresponding test in the `tests/` directory.

Common flags:

| Flag | Description |
|------|-------------|
| `-v` | Verbose output — shows each test name and pass/fail status |
| `-q` | Quiet output — minimal output, just a summary |
| `-x` | Stop after the first failure |
| `-s` | Disable output capture — allows `print()` statements to show in terminal |
| `-k "expression"` | Run only tests whose names match the expression (e.g. `-k "test_fit"`) |
| `--tb=short` | Show a shortened traceback on failures (`short`, `long`, `no`, `line`) |
| `--cov=geom_nmf` | Report test coverage for the package (requires `pytest-cov`) |

Examples:

```sh
# Run all tests verbosely
pytest -v tests/

# Stop on first failure with full traceback
pytest -x --tb=long tests/

# Run only tests matching a keyword
pytest -v -k "test_fit" tests/

# Run with coverage report
pytest --cov=geom_nmf --cov-report=term-missing tests/
```

### 8. Commit and push your changes

Stage and commit your changes with a clear message describing what you did:

```sh
git add <file(s) you changed>
git commit -m "Brief description of your change"
git push origin my-feature-branch
```

### 9. Open a Pull Request

Go to your fork on GitHub. You should see a prompt to open a Pull Request — click it. Fill in a short description of what your changes do and why. A maintainer will review your PR and may leave comments or request changes before merging.

### Tips

- Keep pull requests small and focused — one feature or fix per PR is much easier to review
- Check that your branch is up to date with `main` before opening a PR:
  ```sh
  git fetch origin
  git rebase origin/main
  ```
- To add a new dependency: `poetry add <package>` (or `poetry add --group dev <package>` for dev-only)
- If you're unsure about a change, open an issue first to discuss it before writing code

## Building Documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) from the NumPy-style
docstrings in the source code.

```sh
poetry install --with docs
poetry run make -C docs html
```

Then open `docs/_build/html/index.html` in your browser. The build treats warnings as errors
(`-W`), so broken references or malformed docstrings will fail the build.

## Building a distribution

```sh
poetry build          # writes dist/geom_nmf-<version>.tar.gz and .whl
```

&copy; 2026 Bora Jin. Released under the [MIT License](LICENSE).
