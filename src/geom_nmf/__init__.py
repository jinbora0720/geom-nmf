"""
geom_nmf
========
Geometric NMF: geometric non-negative matrix factorization via maximum-volume simplex fitting.

Public API
----------
GeomNMF                      Main pipeline.
GeomNMFResult                Named tuple returned by :func:`GeomNMF`.
nfindr                       N-FINDR endmember extraction (standalone).
log_simplex_volume           Log-volume of a simplex spanned by K points.
compute_Phi                  Source attribution matrix from source means and H.
estimate_weights             Recover W from Y and H.
permute_to_reference         Permute estimated H, mu, Phi to best match a reference ordering.
barplot_Phi                  Grouped bar plot of Phi (% attribution per source and feature).
                             Requires the optional ``plot`` extra (matplotlib).
"""

from importlib.metadata import PackageNotFoundError, version

from .core import GeomNMF, GeomNMFResult
from .endmembers import log_simplex_volume
from .nfindr import nfindr
from .plot import barplot_Phi
from .utils import permute_to_reference
from .weights import compute_Phi, estimate_weights

try:
    __version__ = version("geom-nmf")
except PackageNotFoundError:  # running from a source checkout without installing
    __version__ = "0+unknown"

__all__ = [
    "GeomNMF",
    "GeomNMFResult",
    "nfindr",
    "log_simplex_volume",
    "compute_Phi",
    "estimate_weights",
    "permute_to_reference",
    "barplot_Phi",
    "__version__",
]
