"""Tests for plotting helpers and package-level metadata."""

import numpy as np
import pytest

import geom_nmf
from geom_nmf import barplot_Phi

mpl = pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def phi():
    rng = np.random.default_rng(0)
    A = rng.random((3, 4))
    return A / A.sum(axis=0)


def test_barplot_returns_fig_ax(phi):
    fig, ax = barplot_Phi(phi, show=False)
    assert isinstance(fig, mpl.figure.Figure)
    assert len(ax.patches) == 3 * 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["Source 1", "Source 2", "Source 3"]
    assert ax.get_ylim() == (0.0, 100.0)
    heights = np.array([p.get_height() for p in ax.patches]).reshape(4, 3).T
    np.testing.assert_allclose(heights, phi * 100)


def test_barplot_custom_labels_colors_and_ax(phi):
    fig0, ax0 = plt.subplots()
    labels = ["PM1", "PM2.5", "PM10", "CO"]
    fig, ax = barplot_Phi(
        phi,
        feature_labels=labels,
        source_labels=["Traffic", "Industry", "Dust"],
        feature_colors={"PM1": "red", "CO": "black"},
        ylabel="%",
        title="T",
        ax=ax0,
        show=False,
    )
    assert fig is fig0 and ax is ax0
    assert ax.get_ylabel() == "%"
    assert ax.get_title() == "T"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == labels


def test_barplot_color_list(phi):
    _, ax = barplot_Phi(phi, feature_colors=["r", "g", "b", "k"], show=False)
    assert mpl.colors.to_hex(ax.patches[0].get_facecolor()) == "#ff0000"


def test_barplot_bootstrap_stack_with_errorbars(phi):
    rng = np.random.default_rng(1)
    stack = np.clip(phi[None] + 0.02 * rng.standard_normal((20, 3, 4)), 0, 1)
    from matplotlib.container import BarContainer

    _, ax = barplot_Phi(stack, show=False)
    bars = [c for c in ax.containers if isinstance(c, BarContainer)]
    assert len(bars) == 4
    assert all(c.errorbar is not None for c in bars)
    _, ax2 = barplot_Phi(stack, show_err=False, show=False)
    bars2 = [c for c in ax2.containers if isinstance(c, BarContainer)]
    assert len(bars2) == 4
    assert all(c.errorbar is None for c in bars2)


def test_barplot_savepath(tmp_path, phi):
    out = tmp_path / "phi.png"
    barplot_Phi(phi, savepath=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_barplot_show_true_does_not_block(phi):
    barplot_Phi(phi)  # Agg backend: plt.show() is a no-op


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(Phi=np.ones(3)), "Expected 2D"),
        (dict(Phi=np.ones((2, 3)), feature_labels=["a"]), "feature_labels"),
        (dict(Phi=np.ones((2, 3)), source_labels=["a"]), "source_labels"),
    ],
)
def test_barplot_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        barplot_Phi(show=False, **kwargs)


def test_version_and_exports():
    assert isinstance(geom_nmf.__version__, str) and geom_nmf.__version__
    for name in geom_nmf.__all__:
        assert hasattr(geom_nmf, name), name
