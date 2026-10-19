"""
geom_nmf.plot
=============
Plotting helpers for GeomNMF results.

Matplotlib is an optional dependency; install it with
``pip install "geom-nmf[plot]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "barplot_Phi requires matplotlib. Install it with "
            "`pip install \"geom-nmf[plot]\"` or `pip install matplotlib`."
        ) from exc
    return plt


def barplot_Phi(
    Phi: np.ndarray,
    title: str = "Estimated source attribution (GeomNMF)",
    feature_labels: Sequence[str] | None = None,
    feature_colors: Sequence[Any] | Mapping[str, Any] | None = None,
    show_err: bool = True,
    savepath: str | None = None,
    source_labels: Sequence[str] | None = None,
    ylabel: str = "Expected % of pollutants\nattributable to each source",
    ax: Axes | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """
    Grouped bar plot of the source attribution matrix Phi.

    Each group of bars corresponds to one source (k); within each group
    there is one bar per feature (j), with height ``Phi[k, j] * 100`` (%).
    When a bootstrap stack is provided, bars show the mean across replicates
    and optional error bars show ±1 standard deviation.

    Parameters
    ----------
    Phi : array-like
        Either:

        * shape ``(K, J)``: single Phi matrix.
        * shape ``(n_reps, K, J)``: bootstrap stack; mean and std are
          computed across the first axis.

        Values are assumed to be in [0, 1] and are multiplied by 100 for
        display.
    title : str, optional
        Figure title.  Default ``"Estimated source attribution (GeomNMF)"``.
    feature_labels : sequence of str or None, optional
        Feature labels of length J used in the legend.
        If None, defaults to ``["Feature 1", ..., "Feature J"]``.
    feature_colors : None, sequence, or mapping, optional
        Bar colors per feature.  None uses matplotlib defaults; a sequence of
        length J assigns colors by position; a mapping maps feature label to
        color.  Default None.
    show_err : bool, optional
        If True and *Phi* is 3D, draw error bars (±1 std across bootstrap
        reps).  Default True.
    savepath : str or None, optional
        File path to save the figure (200 dpi, tight bbox).  Default None.
    source_labels : sequence of str or None, optional
        Tick labels of length K for the sources.  If None, defaults to
        ``["Source 1", ..., "Source K"]``.
    ylabel : str, optional
        Y-axis label.  Default ``"Expected % of pollutants\\nattributable to each source"``.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes to draw on.  If None, a new ``(10, 5)`` figure is created.
    show : bool, optional
        If True, call ``plt.show()`` after drawing.  Set to False in scripts
        or tests where you want to keep working with the returned figure.
        Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    ax : matplotlib.axes.Axes
        The axes containing the plot.

    Raises
    ------
    ValueError
        If *Phi* is not 2D or 3D, or if *feature_labels* / *source_labels*
        have the wrong length.
    ImportError
        If matplotlib is not installed.
    """
    A = np.asarray(Phi, dtype=float)
    if A.ndim == 2:
        K, J = A.shape                                          # (K, J)
        Phi_mean = A * 100                                      # in %
        Phi_std = None
    elif A.ndim == 3:
        n_reps, K, J = A.shape                                  # (n_reps, K, J)
        Phi_mean = np.nanmean(A, axis=0) * 100                  # (K, J), in %
        Phi_std = (
            np.nanstd(A, axis=0, ddof=1 if n_reps > 1 else 0) * 100 if show_err else None
        )
    else:
        raise ValueError(f"Expected 2D (K, J) or 3D (n_reps, K, J); got shape {A.shape}")

    # feature labels
    if feature_labels is None:
        labels = [f"Feature {j+1}" for j in range(J)]
    else:
        if len(feature_labels) != J:
            raise ValueError(f"feature_labels must have length {J} (got {len(feature_labels)})")
        labels = list(feature_labels)

    if source_labels is None:
        src_labels = [f"Source {k+1}" for k in range(K)]
    else:
        if len(source_labels) != K:
            raise ValueError(f"source_labels must have length {K} (got {len(source_labels)})")
        src_labels = list(source_labels)

    plt = _import_pyplot()

    x = np.arange(K)                    # one tick per source
    width = 0.8 / max(J, 1)             # bar width within each source group

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for j in range(J):
        # build error bar kwargs only if we have usable stds
        kwargs: dict[str, Any] = {}
        if Phi_std is not None:
            yerr = Phi_std[:, j]        # (K,) std for feature j across sources
            if np.any(np.isfinite(yerr)):
                kwargs["yerr"] = yerr
                kwargs["error_kw"] = {"elinewidth": 1.2, "capsize": 3}

        # choose color for this feature
        if feature_colors is None:
            c = None
        elif isinstance(feature_colors, Mapping):
            c = feature_colors.get(labels[j], None)
        else:
            c = feature_colors[j]

        ax.bar(
            x + j * width,
            Phi_mean[:, j],            # (K,) attribution of feature j per source
            width,
            color=c,
            label=labels[j],
            **kwargs,
        )

    ax.set_xticks(x + (J - 1) * width / 2)
    ax.set_xticklabels(src_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)

    # Legend outside on the right
    ax.legend(title="", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
