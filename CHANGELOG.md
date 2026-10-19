# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0]

### Added
- Optional extras: `plot` (matplotlib), `progress` (tqdm), and `all`.
- `barplot_Phi` now returns `(fig, ax)` and accepts `ax=`, `source_labels=`, `ylabel=`, and
  `show=` (default `True`, matching previous behavior).
- Input validation in `GeomNMF`: `Y` must be 2-D, and `K` must be an integer with
  `2 <= K <= min(n, J)`. A clear `ValueError` is raised when the data's intrinsic rank is below
  `K - 1`.
- `RuntimeWarning` when the exhaustive max-volume search exceeds
  `geom_nmf.endmembers.LARGE_SEARCH_WARN` (50M) subsets.
- `py.typed` marker (PEP 561).
- Sphinx documentation (`docs/`), runnable example (`examples/quickstart.py`), and a test suite
  covering every public function (91 tests).
- CI: Python 3.9–3.13 test matrix, documentation build, and package build checks.

### Changed
- Supported Python versions are now `>=3.9` (previously `>=3.8,<3.12`; 3.8 and 3.9 did not
  import because of the type-hint syntax).
- `pandas` is no longer required; DataFrames are still accepted as input.
- `matplotlib` and `tqdm` are now optional.
- Declared `scipy` as a dependency (it was used but missing).
- `__version__` falls back to `"0+unknown"` when the package is not installed.

### Fixed
- `GeomNMF(..., K=2)` with the default `candidate_method="exact"` crashed because QHull
  cannot handle 1-D data; 1-D hulls are now handled directly.
- `import geom_nmf` failed on a clean install because `pandas`/`matplotlib` were imported but
  not declared as dependencies.

## [0.1.0a0]

- Initial pre-release.
- Oct 19, 2026
