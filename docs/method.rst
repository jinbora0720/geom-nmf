Method overview
===============

Let :math:`Y \in \mathbb{R}_{\ge 0}^{n \times J}` with row sums
:math:`r_i = \sum_j Y_{ij} > 0`. :func:`geom_nmf.GeomNMF` proceeds as follows.

1. **Row-normalize.** :math:`Y^\star = \operatorname{diag}(r)^{-1} Y`. Under the
   simplex model each row of :math:`Y^\star` is a convex combination of the
   rows of :math:`H`.

2. **Project to the intrinsic affine subspace.** Drop the last (redundant)
   coordinate, center, and keep the right singular vectors with singular
   values above ``rank_tol``. The number kept is the *intrinsic rank*, which
   must be at least :math:`K-1`.

3. **Build a candidate pool** of extreme points: convex-hull vertices
   (``"exact"``) or random-projection extremes (``"random"``, computed in
   whitened coordinates). Optionally prune by clustering.

4. **Maximum-volume search.** Among all :math:`K`-subsets of candidates, choose
   the one maximizing the log-volume

   .. math::

      \log \operatorname{vol} = \sum_{i=1}^{K-1} \log s_i - \log (K-1)!,

   where :math:`s_i` are the singular values of the edge matrix. The selected
   rows of :math:`Y^\star` form :math:`\hat H^\star`. Optionally refine with
   N-FINDR (:func:`geom_nmf.nfindr`).

5. **Recover weights.** Solve :math:`[Y^\star \mid \mathbf{1}] \approx
   W^\star [\hat H^\star \mid \mathbf{1}]` with the pseudoinverse, clip negative
   weights and renormalize rows to the simplex, then rescale:
   :math:`\tilde W = \operatorname{diag}(r)\, W^\star`.

6. **Source attribution.** With :math:`\tilde\mu_k = n^{-1}\sum_i \tilde W_{ik}`,

   .. math::

      \Phi_{kj} = \frac{\tilde\mu_k \hat H^\star_{kj}}
                       {\sum_{k'} \tilde\mu_{k'} \hat H^\star_{k'j}},

   the expected share of feature :math:`j` attributable to source :math:`k`.

Sources are identified only up to permutation; use
:func:`geom_nmf.permute_to_reference` to align an estimate with a reference.
