"""Tests for candidate generation and volume search (geom_nmf.endmembers)."""

import math

import numpy as np
import pytest

from geom_nmf import endmembers
from geom_nmf.endmembers import (
    estimate_H,
    get_hull_candidates,
    get_random_direction_candidates,
    log_simplex_volume,
    prune_close_points,
)

# ---------------------------------------------------------------------------
# log_simplex_volume
# ---------------------------------------------------------------------------

def test_log_volume_right_triangle():
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    logv, r = log_simplex_volume(tri)
    assert r == 2
    assert logv == pytest.approx(math.log(0.5))


def test_log_volume_unit_cube_corner_tetrahedron():
    tet = np.vstack([np.zeros(3), np.eye(3)])
    logv, r = log_simplex_volume(tet)
    assert r == 3
    assert logv == pytest.approx(-math.log(6))


def test_log_volume_is_translation_and_rotation_invariant():
    rng = np.random.default_rng(1)
    P = rng.random((4, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    a, _ = log_simplex_volume(P)
    b, _ = log_simplex_volume(P @ Q + 5.0)
    assert a == pytest.approx(b)


def test_log_volume_in_higher_ambient_dimension():
    # a triangle embedded in R^5 has the same area as in R^2
    tri = np.zeros((3, 5))
    tri[1, 0] = 2.0
    tri[2, 3] = 3.0
    logv, r = log_simplex_volume(tri)
    assert r == 2
    assert logv == pytest.approx(math.log(3.0))


def test_log_volume_degenerate_is_neg_inf():
    collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    logv, r = log_simplex_volume(collinear)
    assert logv == -np.inf
    assert r == 1


# ---------------------------------------------------------------------------
# get_hull_candidates
# ---------------------------------------------------------------------------

def test_hull_candidates_square_with_interior():
    rng = np.random.default_rng(0)
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    interior = 0.1 + 0.8 * rng.random((50, 2))
    X = np.vstack([interior, corners])
    idx = get_hull_candidates(X)
    assert set(idx.tolist()) == {50, 51, 52, 53}


def test_hull_candidates_1d():
    x = np.array([[0.3], [-1.0], [0.0], [2.0], [0.5]])
    idx = get_hull_candidates(x)
    assert sorted(idx.tolist()) == [1, 3]
    # also accepts a flat vector
    assert sorted(get_hull_candidates(x.ravel()).tolist()) == [1, 3]


# ---------------------------------------------------------------------------
# get_random_direction_candidates
# ---------------------------------------------------------------------------

def test_random_direction_candidates_find_triangle_vertices():
    rng = np.random.default_rng(0)
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    W = rng.dirichlet(np.ones(3), size=200)
    X = np.vstack([V, W @ V])
    X = X - X.mean(axis=0)
    idx, counts = get_random_direction_candidates(X, n_directions=500, seed=1)
    assert set(idx.tolist()) == {0, 1, 2}  # only vertices can be extreme
    assert counts.sum() == 1000
    assert np.all(np.diff(counts[idx]) <= 0)  # sorted by decreasing frequency


def test_random_direction_candidates_n_candidates_and_n_top():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 3))
    idx, counts = get_random_direction_candidates(
        X, n_directions=100, n_top=4, n_candidates=7, seed=2
    )
    assert len(idx) == 7
    assert counts.sum() == 2 * 4 * 100


def test_random_direction_candidates_batches(capsys):
    # n_directions larger than the internal batch size exercises batching
    X = np.random.default_rng(0).standard_normal((40, 2))
    _, counts = get_random_direction_candidates(X, n_directions=5000, verbose=True)
    assert counts.sum() == 10000
    assert "#cands=" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# prune_close_points
# ---------------------------------------------------------------------------

def test_prune_returns_actual_points_one_per_cluster():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    X = np.vstack([c + 0.05 * rng.standard_normal((25, 2)) for c in centers])
    reps, sel = prune_close_points(X, n_clusters=4, seed=0)
    assert 1 <= len(sel) <= 6
    np.testing.assert_array_equal(reps, X[sel])
    assert sel == sorted(sel)


def test_prune_edge_cases():
    empty = np.zeros((0, 3))
    reps, sel = prune_close_points(empty)
    assert reps.shape == (0, 3) and sel == []

    one = np.ones((1, 3))
    reps, sel = prune_close_points(one)
    assert sel == [0]

    # lower > upper -> returned unchanged
    X = np.random.default_rng(0).random((5, 2))
    reps, sel = prune_close_points(X, n_clusters=10)
    assert sel == list(range(5))


# ---------------------------------------------------------------------------
# estimate_H
# ---------------------------------------------------------------------------

def test_estimate_H_selects_max_volume_subset():
    pts = np.array(
        [[0, 0], [4, 0], [0, 4], [1, 1], [2, 1], [1, 2]], dtype=float
    )
    H, logv = estimate_H(pts, 3)
    assert {tuple(r) for r in H} == {(0, 0), (4, 0), (0, 4)}
    assert logv == pytest.approx(math.log(8.0))


def test_estimate_H_all_degenerate_raises():
    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    with pytest.raises(RuntimeError, match="non-degenerate"):
        estimate_H(collinear, 3)


def test_estimate_H_too_few_candidates_raises():
    with pytest.raises(RuntimeError, match="candidate"):
        estimate_H(np.eye(2), 3)


def test_estimate_H_warns_on_huge_search(monkeypatch):
    monkeypatch.setattr(endmembers, "LARGE_SEARCH_WARN", 5)
    pts = np.vstack([np.zeros(2), np.eye(2), [[1, 1]]])
    with pytest.warns(RuntimeWarning, match="Exhaustive search"):
        estimate_H(pts, 2)
