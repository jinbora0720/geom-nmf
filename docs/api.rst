API reference
=============

.. currentmodule:: geom_nmf

Main pipeline
-------------

.. autofunction:: GeomNMF

.. autoclass:: GeomNMFResult
   :members:
   :undoc-members:

Building blocks
---------------

.. autofunction:: nfindr

.. autofunction:: log_simplex_volume

.. autofunction:: estimate_weights

.. autofunction:: compute_Phi

.. autofunction:: permute_to_reference

Plotting
--------

.. autofunction:: barplot_Phi

Lower-level helpers
-------------------

.. automodule:: geom_nmf.endmembers
   :members: get_hull_candidates, get_random_direction_candidates, prune_close_points, estimate_H, LARGE_SEARCH_WARN
