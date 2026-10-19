"""Tests for N-FINDR endmember extraction (geom_nmf.nfindr)."""

import numpy as np
import pytest

from geom_nmf import nfindr

from ._helpers import make_simplex_data


@pytest.mark.parametrize("init", ["atgp", "random"])
@pytest.mark.parametrize("normalize", [True, False])
def test_nfindr_finds_pure_pixels(init, normalize):
    K, J, N = 4, 8, 400
    Y, H, _ = make_simplex_data(K, J, N)
    E, idx = nfindr(Y, K, init=init, normalize=normalize, seed=0, max_iter=10)
    assert E.shape == (K, J)
    assert set(idx.tolist()) == set(range(K))  # the planted pure rows
    np.testing.assert_array_equal(E, Y[idx])


def test_nfindr_with_init_idx():
    K, J, N = 3, 6, 200
    Y, _, _ = make_simplex_data(K, J, N)
    _, idx = nfindr(Y, K, init_idx=[10, 20, 30], normalize=False)
    assert set(idx.tolist()) == {0, 1, 2}


def test_nfindr_3d_cube():
    K, J = 3, 5
    Y, _, _ = make_simplex_data(K, J, 100)
    cube = Y.reshape(10, 10, J)
    E, idx = nfindr(cube, K, normalize=False)
    assert E.shape == (K, J)
    np.testing.assert_array_equal(E, cube.reshape(-1, J)[idx])


def test_nfindr_accepts_dataframe():
    pd = pytest.importorskip("pandas")
    Y, _, _ = make_simplex_data(3, 5, 80)
    E1, i1 = nfindr(pd.DataFrame(Y), 3)
    E2, i2 = nfindr(Y, 3)
    np.testing.assert_array_equal(i1, i2)


def test_nfindr_degenerate_start_fallback():
    # init with duplicated rows -> zero-volume simplex -> explicit fallback path
    K, J = 3, 5
    Y, _, _ = make_simplex_data(K, J, 60)
    Y = np.vstack([Y, Y[10], Y[10]])
    _, idx = nfindr(Y, K, init_idx=[10, 60, 61], normalize=False)
    assert set(idx.tolist()) == {0, 1, 2}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(K=1), "at least 2"),
        (dict(K=10), "J\\+1"),
        (dict(K=3, init="bogus"), "init must"),
        (dict(K=3, init_idx=[0, 1]), "exactly K"),
        (dict(K=3, init_idx=[0, 0, 1]), "duplicate"),
        (dict(K=3, init_idx=[0, 1, 999]), "lie in"),
    ],
)
def test_nfindr_validation(kwargs, match):
    Y, _, _ = make_simplex_data(3, 5, 50)
    with pytest.raises(ValueError, match=match):
        nfindr(Y, **kwargs)


def test_nfindr_rejects_bad_ndim():
    with pytest.raises(ValueError, match="2-D"):
        nfindr(np.ones(5), 2)


def test_nfindr_rejects_too_few_rows():
    with pytest.raises(ValueError, match="observations"):
        nfindr(np.random.default_rng(0).random((2, 5)), 3)
