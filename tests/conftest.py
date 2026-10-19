"""Shared fixtures for the geom_nmf test suite."""

import os

# Use a non-interactive matplotlib backend so plotting tests never block.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

from ._helpers import SEED, make_simplex_data  # noqa: F401


@pytest.fixture
def simplex_data():
    """K=3, J=8, N=500 noise-free simplex data."""
    return make_simplex_data(3, 8, 500)
