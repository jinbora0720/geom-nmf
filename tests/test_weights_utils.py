"""Tests for weight recovery, Phi, and permutation matching."""

import numpy as np
import pytest

from geom_nmf import compute_Phi, estimate_weights, permute_to_reference

from ._helpers import make_simplex_data

# ---------------------------------------------------------------------------
# compute_Phi
# ---------------------------------------------------------------------------

def test_compute_Phi_hand_example():
    mu = np.array([1.0, 3.0])
    H = np.array([[0.5, 0.5], [0.25, 0.75]])
    Phi = compute_Phi(mu, H)
    # column 0: 0.5 vs 0.75 -> 0.4, 0.6 ; column 1: 0.5 vs 2.25 -> 2/11, 9/11
    np.testing.assert_allclose(Phi, [[0.4, 2 / 11], [0.6, 9 / 11]])


def test_compute_Phi_columns_sum_to_one_and_accepts_lists():
    rng = np.random.default_rng(0)
    H = rng.dirichlet(np.ones(6), size=4)
    Phi = compute_Phi(list(rng.random(4)), H.tolist())
    np.testing.assert_allclose(Phi.sum(axis=0), 1.0)
    assert np.all(Phi >= 0)


def test_compute_Phi_zero_column_is_nan():
    H = np.array([[1.0, 0.0], [1.0, 0.0]])
    with np.errstate(invalid="ignore"):
        Phi = compute_Phi(np.array([1.0, 1.0]), H)
    assert np.all(np.isnan(Phi[:, 1]))


# ---------------------------------------------------------------------------
# estimate_weights
# ---------------------------------------------------------------------------

def test_estimate_weights_exact_recovery():
    Y, H, W = make_simplex_data(3, 7, 200)
    W_hat, diag = estimate_weights(Y, H)
    np.testing.assert_allclose(W_hat, W, atol=1e-10)
    assert diag["rank_H_aug"] == 3
    assert diag["large_neg_count"] == 0
    assert diag["not_sum1_count"] == 0
    assert diag["max_row_sum_dev_H"] < 1e-12
    assert diag["aug_resid_inf"] < 1e-10
    assert diag["I_err"] < 1e-10


def test_estimate_weights_clips_and_flags_outside_points():
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])[:2]  # K=2 edge
    # a point not on the segment between the two endmembers
    Y = np.array([[0.5, 0.5, 0.0], [1.2, -0.2, 0.0]])
    W, diag = estimate_weights(Y, H, weight_tol=1e-3)
    assert np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert diag["large_neg_rows"].tolist() == [1]
    assert diag["large_neg_count"] == 1


def test_estimate_weights_verbose(capsys):
    Y, H, _ = make_simplex_data(2, 4, 20)
    estimate_weights(Y, H, verbose=True)
    assert "H_aug" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# permute_to_reference
# ---------------------------------------------------------------------------

def test_permute_to_reference_recovers_permutation():
    rng = np.random.default_rng(3)
    K, J = 5, 6
    H_ref = rng.dirichlet(np.ones(J), size=K)
    mu = rng.random(K)
    Phi = compute_Phi(mu, H_ref)
    perm = rng.permutation(K)
    H_p, mu_p, Phi_p, order = permute_to_reference(
        H_ref, H_ref[perm] + 1e-4, mu[perm], Phi[perm]
    )
    np.testing.assert_array_equal(perm[order], np.arange(K))
    np.testing.assert_allclose(H_p, H_ref + 1e-4)
    np.testing.assert_allclose(mu_p, mu)
    np.testing.assert_allclose(Phi_p, Phi)


def test_permute_to_reference_identity():
    H = np.eye(3)
    _, _, _, order = permute_to_reference(H, H, np.ones(3), H)
    assert order.tolist() == [0, 1, 2]


@pytest.mark.parametrize("K", [2, 4])
def test_permute_output_shapes(K):
    rng = np.random.default_rng(0)
    H = rng.random((K, 3))
    out = permute_to_reference(H, H[::-1], np.arange(K), H[::-1])
    assert out[0].shape == (K, 3) and out[1].shape == (K,) and out[3].shape == (K,)
