Getting started
===============

Installation
------------

.. code-block:: sh

   pip install git+https://github.com/jinbora0720/geom-nmf.git

   # with plotting support (matplotlib) and progress bars (tqdm)
   pip install "geom-nmf[all] @ git+https://github.com/jinbora0720/geom-nmf.git"

Optional extras:

============  ===========================================================
Extra         Enables
============  ===========================================================
``plot``      :func:`geom_nmf.barplot_Phi` (installs matplotlib)
``progress``  tqdm progress bar for the exhaustive search when ``verbose=True``
``all``       both of the above
============  ===========================================================

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from geom_nmf import GeomNMF, barplot_Phi

   # Y: (n, J) nonnegative array or pandas DataFrame
   result = GeomNMF(Y, K=3)

   result.H_star      # (K, J) source profiles; rows sum to 1
   result.W_tilde     # (n, K) source contributions on the original scale of Y
   result.mu_tilde    # (K,)   mean contribution of each source
   result.Phi         # (K, J) share of each feature attributable to each source

   barplot_Phi(result.Phi, feature_labels=list(df.columns))

A complete, runnable example with simulated data is in
``examples/quickstart.py``.

Choosing a candidate method
---------------------------

GeomNMF first builds a pool of candidate extreme points and then searches all
:math:`\binom{m}{K}` subsets of the :math:`m` candidates for the
maximum-volume simplex.

``candidate_method="exact"`` (default)
    Uses the convex-hull vertices (QHull). Globally optimal over the data, but
    the number of hull vertices grows quickly with the intrinsic dimension of
    noisy data.

``candidate_method="random"``
    Records the extremes of the data along ``n_directions`` random directions.
    Use ``n_candidates`` to cap the pool size.

For large candidate pools, set ``prune=True`` to cluster candidates and keep
one representative per cluster. Pruning can discard the true extremes, so
combine it with ``refine_greedy=True``, which runs N-FINDR swaps on the full
data starting from the pruned solution and keeps the result only if it
increases the simplex volume.

A :class:`RuntimeWarning` is raised when the exhaustive search would exceed
:data:`geom_nmf.endmembers.LARGE_SEARCH_WARN` subsets.

Diagnostics
-----------

``result.fit_diagnostics["intrinsic_rank"]`` should equal ``K - 1`` for
noise-free data. ``result.weight_diagnostics`` reports the conditioning of the
weight solve (``cond_G``) and which observations fell outside the fitted
simplex (``large_neg_rows``, ``not_sum1_rows``); many such rows suggest
``K`` is too small or the model is misspecified.
