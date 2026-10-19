"""Sphinx configuration for geom-nmf."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as _version

# Allow building docs from a source checkout without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

project = "geom-nmf"
author = "Bora Jin"
copyright = "2026, Bora Jin"

try:
    release = _version("geom-nmf")
except PackageNotFoundError:  # pragma: no cover
    release = "0+unknown"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["matplotlib"]
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_theme_options = {
    "description": "Geometric NMF via maximum-volume simplex fitting",
    "github_user": "jinbora0720",
    "github_repo": "geom-nmf",
    "github_button": True,
}
