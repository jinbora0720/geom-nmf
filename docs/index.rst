geom-nmf
========

**geom-nmf** implements *geometric non-negative matrix factorization*
(GeomNMF): given a nonnegative data matrix :math:`Y` that follows a simplex
mixing model

.. math::

   Y \approx W H, \qquad W \ge 0,\; H \ge 0,\; H\mathbf{1} = \mathbf{1},

it estimates the source profiles :math:`H` (endmembers) as the vertices of the
maximum-volume :math:`(K-1)`-simplex inscribed in the convex hull of the
row-normalized data, then recovers the mixing weights :math:`W` and a source
attribution matrix :math:`\Phi`.

.. toctree::
   :maxdepth: 2

   getting_started
   method
   api
   changelog
