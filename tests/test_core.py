"""Tests for the main GeomNMF pipeline (geom_nmf.core)."""

import numpy as np
import pytest

from geom_nmf import GeomNMF, GeomNMFResult, permute_to_reference

from ._helpers import SEED, make_simplex_data


def _aligned_error(H_true, res):
    H_perm, _, _, order = permute_to_reference(H_true, res.H_star, res.mu_tilde, res.Phi)
    return np.abs(H_perm - H_true).max(), order


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K,J", [(2, 5), (3, 8), (4, 6), (5, 7)])
def test_exact_recovery_with_pure_pixels(K, J):
    Y, H, _ = make_simplex_data(K, J, 300)
    res = GeomNMF(Y, K)
    err, _ = _aligned_error(H, res)
    assert err < 1e-10
    assert res.fit_diagnostics["intrinsic_rank"] == K - 1


def test_recovers_scaled_weights():
    K, J = 3, 6
    Y, H, W = make_simplex_data(K, J, 400, scale=True)
    res = GeomNMF(Y, K)
    _, order = _aligned_error(H, res)
    W_hat = res.W_tilde[:, order]
    np.testing.assert_allclose(W_hat, W, atol=1e-8)
    np.testing.assert_allclose(res.mu_tilde[order], W.mean(axis=0), atol=1e-8)
    # rows of W_tilde sum to the original row sums of Y
    np.testing.assert_allclose(res.W_tilde.sum(axis=1), Y.sum(axis=1), rtol=1e-10)


def test_result_structure(simplex_data):
    Y, _, _ = simplex_data
    res = GeomNMF(Y, 3)
    assert isinstance(res, GeomNMFResult)
    assert res.H_star.shape == (3, 8)
    assert res.W_tilde.shape == (500, 3)
    assert res.mu_tilde.shape == (3,)
    assert res.Phi.shape == (3, 8)
    assert np.isfinite(res.logvol)
    assert res.elapsed >= 0
    np.testing.assert_allclose(res.H_star.sum(axis=1), 1.0)
    np.testing.assert_allclose(res.Phi.sum(axis=0), 1.0)
    assert set(res.fit_diagnostics) == {
        "intrinsic_rank",
        "n_cand",
        "cand_idx",
        "direction_hit_counts",
        "logvol_before_refinement",
    }
    assert {"cond_G", "rank_H_aug", "large_neg_count", "not_sum1_count"} <= set(
        res.weight_diagnostics
    )
    assert res.weight_diagnostics["rank_H_aug"] == 3
    assert res.fit_diagnostics["direction_hit_counts"] is None
    assert res.fit_diagnostics["logvol_before_refinement"] is None


def test_scale_invariance(simplex_data):
    """Row normalization makes H_star invariant to per-row scaling of Y."""
    Y, _, _ = simplex_data
    rng = np.random.default_rng(0)
    res1 = GeomNMF(Y, 3)
    res2 = GeomNMF(Y * rng.uniform(1, 10, size=(Y.shape[0], 1)), 3)
    np.testing.assert_allclose(np.sort(res1.H_star, axis=0), np.sort(res2.H_star, axis=0))


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

def test_accepts_dataframe(simplex_data):
    pd = pytest.importorskip("pandas")
    Y, _, _ = simplex_data
    df = pd.DataFrame(Y, columns=[f"f{j}" for j in range(Y.shape[1])])
    res_df = GeomNMF(df, 3)
    res_np = GeomNMF(Y, 3)
    np.testing.assert_allclose(res_df.H_star, res_np.H_star, rtol=1e-12)


def test_accepts_list(simplex_data):
    Y, _, _ = simplex_data
    res = GeomNMF(Y.tolist(), 3)
    assert res.H_star.shape == (3, 8)


def test_accepts_numpy_integer_K(simplex_data):
    Y, _, _ = simplex_data
    res = GeomNMF(Y, np.int64(3))
    assert res.H_star.shape == (3, 8)


# ---------------------------------------------------------------------------
# Candidate methods and options
# ---------------------------------------------------------------------------

def test_random_candidates(simplex_data):
    Y, H, _ = simplex_data
    res = GeomNMF(Y, 3, candidate_method="random", n_directions=2000, seed=SEED)
    err, _ = _aligned_error(H, res)
    assert err < 1e-10
    counts = res.fit_diagnostics["direction_hit_counts"]
    assert counts.shape == (Y.shape[0],)
    assert counts.sum() == 2 * 2000


def test_random_candidates_n_candidates_and_n_top(simplex_data):
    Y, H, _ = simplex_data
    res = GeomNMF(
        Y, 3, candidate_method="random", n_directions=1000, n_top=3, n_candidates=10
    )
    assert res.fit_diagnostics["n_cand"] <= 10
    err, _ = _aligned_error(H, res)
    assert err < 1e-10


def test_random_is_reproducible(simplex_data):
    Y, _, _ = simplex_data
    a = GeomNMF(Y, 3, candidate_method="random", n_directions=500, seed=7)
    b = GeomNMF(Y, 3, candidate_method="random", n_directions=500, seed=7)
    np.testing.assert_array_equal(a.H_star, b.H_star)


def test_prune_reduces_candidates():
    pytest.importorskip("sklearn")
    K, J = 3, 6
    Y, _, _ = make_simplex_data(K, J, 800, noise=1e-3)
    res = GeomNMF(
        Y, K, candidate_method="random", n_directions=3000, prune=True, n_clusters=6
    )
    assert 1 <= res.fit_diagnostics["n_cand"] <= 9  # at most 1.5 * n_clusters
    assert len(res.fit_diagnostics["cand_idx"]) == res.fit_diagnostics["n_cand"]


def test_prune_with_greedy_refinement_recovers_endmembers():
    """Pruning keeps cluster representatives, which can miss the true extremes;
    greedy refinement on the full data should recover them."""
    pytest.importorskip("sklearn")
    K, J = 3, 6
    Y, H, _ = make_simplex_data(K, J, 800, noise=1e-3)
    pruned = GeomNMF(
        Y, K, candidate_method="random", n_directions=3000, prune=True, n_clusters=6
    )
    refined = GeomNMF(
        Y, K, candidate_method="random", n_directions=3000, prune=True, n_clusters=6,
        refine_greedy=True,
    )
    assert refined.logvol >= pruned.logvol
    err, _ = _aligned_error(H, refined)
    assert err < 0.01


def test_prune_skipped_when_few_candidates(simplex_data, capsys):
    Y, _, _ = simplex_data
    res = GeomNMF(Y, 3, prune=True, n_clusters=1000, verbose=True)
    assert "Pruning skipped" in capsys.readouterr().out
    assert res.fit_diagnostics["n_cand"] == len(res.fit_diagnostics["cand_idx"])


def test_refine_greedy_never_decreases_volume():
    K, J = 3, 6
    Y, _, _ = make_simplex_data(K, J, 600, noise=1e-3)
    res = GeomNMF(
        Y, K, candidate_method="random", n_directions=200, n_candidates=5,
        refine_greedy=True,
    )
    before = res.fit_diagnostics["logvol_before_refinement"]
    assert before is not None
    assert res.logvol >= before


def test_refine_greedy_skipped_for_exact(simplex_data, capsys):
    Y, _, _ = simplex_data
    res = GeomNMF(Y, 3, refine_greedy=True, verbose=True)
    assert "Greedy refinement skipped" in capsys.readouterr().out
    assert res.fit_diagnostics["logvol_before_refinement"] is None


def test_verbose_prints(simplex_data, capsys):
    Y, _, _ = simplex_data
    GeomNMF(Y, 3, verbose=True)
    out = capsys.readouterr().out
    assert "convex hull" in out.lower()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite(simplex_data, bad):
    Y, _, _ = simplex_data
    Y = Y.copy()
    Y[0, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        GeomNMF(Y, 3)


def test_rejects_negative(simplex_data):
    Y, _, _ = simplex_data
    Y = Y.copy()
    Y[1, 2] = -0.1
    with pytest.raises(ValueError, match="negative"):
        GeomNMF(Y, 3)


def test_rejects_zero_rows(simplex_data):
    Y, _, _ = simplex_data
    Y = Y.copy()
    Y[[3, 7]] = 0.0
    with pytest.raises(ValueError, match=r"all-zero row\(s\) \(indices: \[3, 7\]\)"):
        GeomNMF(Y, 3)


def test_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        GeomNMF(np.ones(10), 2)
    with pytest.raises(ValueError, match="2-D"):
        GeomNMF(np.ones((2, 3, 4)), 2)


@pytest.mark.parametrize("K", [0, 1, 9])
def test_rejects_bad_K_range(simplex_data, K):
    Y, _, _ = simplex_data
    with pytest.raises(ValueError, match="K must"):
        GeomNMF(Y, K)


@pytest.mark.parametrize("K", [2.5, "3", True])
def test_rejects_non_integer_K(simplex_data, K):
    Y, _, _ = simplex_data
    with pytest.raises(ValueError, match="integer"):
        GeomNMF(Y, K)


def test_rejects_K_greater_than_n():
    Y = np.random.default_rng(0).random((3, 10))
    with pytest.raises(ValueError, match="n=3"):
        GeomNMF(Y, 4)


def test_rejects_bad_candidate_method(simplex_data):
    Y, _, _ = simplex_data
    with pytest.raises(ValueError, match="candidate_method"):
        GeomNMF(Y, 3, candidate_method="hull")


def test_rejects_insufficient_rank():
    # data generated from K=2 sources cannot support K=4 endmembers
    Y, _, _ = make_simplex_data(2, 6, 100)
    with pytest.raises(ValueError, match="intrinsic"):
        GeomNMF(Y, 4)
