# Contributing to geom-nmf

Thanks for your interest in geom-nmf! Bug reports, questions, documentation fixes, and new
features are all welcome. If you're new to contributing on GitHub, don't worry — the steps
below walk through the whole process.

## Reporting bugs and asking questions

Please use the [issue tracker](https://github.com/jinbora0720/geom-nmf/issues). For a bug
report, include:

- what you ran (a small code example, ideally with simulated data),
- what you expected and what happened instead (the full error message, if any),
- your versions: `python -c "import geom_nmf, numpy, scipy, sys; print(geom_nmf.__version__, numpy.__version__, scipy.__version__, sys.version)"`.

If you're planning a larger change, please open an issue to discuss it first.

## Setting up a development environment

You need Python 3.9 or newer and git.

1. **Fork** the [repository](https://github.com/jinbora0720/geom-nmf) (the **Fork** button,
   top right) and **clone** your fork:

   ```sh
   git clone https://github.com/<your-github-username>/geom-nmf.git
   cd geom-nmf
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

3. **Install the package in editable mode** with the tools for testing and docs:

   ```sh
   pip install -e ".[all]" pytest pytest-cov pandas sphinx
   ```

   "Editable" (`-e`) means changes you make to the source code take effect immediately,
   without reinstalling.

> The repository also includes a `poetry.lock` for those who prefer
> [Poetry](https://python-poetry.org/) (`poetry install --with dev,docs --all-extras`;
> Poetry 2 needs Python 3.10+). It is optional — everything works with plain pip.

## Making changes

1. **Create a branch** with a short descriptive name (never work directly on `main`):

   ```sh
   git checkout -b fix-weight-diagnostics
   ```

2. **Make your changes.** Please:
   - write docstrings in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html),
     matching the existing code;
   - add or update tests in `tests/` for any new behavior or bug fix;
   - add a line under `## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) describing the change;
   - keep each pull request focused on one thing.

3. **Run the tests:**

   ```sh
   pytest
   ```

   Useful options:

   | Option | What it does |
   |--------|--------------|
   | `-v` | show each test name and result |
   | `-x` | stop at the first failure |
   | `-k "name"` | run only tests whose names match, e.g. `-k "nfindr"` |
   | `-s` | show `print()` output |
   | `--cov=geom_nmf --cov-report=term-missing` | report which lines are not covered by tests |

4. **Build the documentation** (if you changed docstrings or files in `docs/`):

   ```sh
   make -C docs html
   ```

   Then open `docs/_build/html/index.html`. Warnings are treated as errors, so malformed
   docstrings will fail the build.

5. **Commit and push** to your fork:

   ```sh
   git add <files you changed>
   git commit -m "Short description of the change"
   git push origin fix-weight-diagnostics
   ```

6. **Open a pull request** from your branch to `jinbora0720/geom-nmf:main`. GitHub Actions will
   automatically run the tests on Linux, macOS, and Windows and on Python 3.9–3.13, build the
   docs, and check the package build. I'll review the pull request and may suggest changes
   before merging.

Before opening a pull request, make sure your branch is up to date with `main`:

```sh
git remote add upstream https://github.com/jinbora0720/geom-nmf.git   # once
git fetch upstream
git rebase upstream/main
```

## Building the package locally

```sh
pip install build twine
python -m build                  # writes dist/geom_nmf-<version>.tar.gz and .whl
python -m twine check dist/*
```

## Releasing (maintainer)

1. Update `version` in `pyproject.toml` and `CITATION.cff`, and rename `## [Unreleased]` in
   `CHANGELOG.md` to `## [x.y.z] - YYYY-MM-DD`. Commit and push to `main`; wait for CI to pass.
2. On GitHub, create a release with tag `vx.y.z` targeting `main`, using the changelog entry as
   the release notes.
3. The **Publish to PyPI** workflow builds and uploads the release automatically.

## Support

geom-nmf is maintained by Bora Jin ([@jinbora0720](https://github.com/jinbora0720)). I aim to
respond to issues and pull requests within a couple of weeks.
