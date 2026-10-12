"""Stage S7: figures. Every function returns a matplotlib ``Figure`` or a plotly ``Figure`` and
never shows or saves anything, so the same code serves the CLI, the app and scripts.

matplotlib figures are created with the object-oriented API (no ``pyplot`` state), so they can
be used from a web server without a display.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from scipy.cluster import hierarchy as hc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fastforest.metrics import METRIC_NAMES

if TYPE_CHECKING:
    from fastforest.redundancy import RedundancyResult
    from fastforest.result import FastForestResult

STAGE_LABELS = {
    "full": "all features",
    "important": "important features",
    "final": "final features",
}


def palette(n: int) -> list[str]:
    """Return ``n`` distinct hex colours (tab10 up to 10 classes, tab20 beyond)."""
    cmap = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return [to_hex(cmap(i % cmap.N)) for i in range(n)]


def _labels(y, class_names) -> np.ndarray:
    return np.asarray(class_names, dtype=object)[np.asarray(y, dtype=int)]


# ------------------------------------------------------------------ confusion matrices
def confusion_matrix_figure(
    y_true,
    y_pred,
    class_names: list[str],
    *,
    normalize: bool = False,
    title: str | None = None,
    cmap: str = "Blues",
) -> Figure:
    """Plot a confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like
        Encoded labels.
    class_names : list of str
        Display names; label ``k`` is ``class_names[k]``.
    normalize : bool, default False
        Row-normalise (recall per class) when True.
    title : str, optional
        Figure title.
    cmap : str, default "Blues"
        matplotlib colormap.
    """
    k = len(class_names)
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(k), normalize="true" if normalize else None
    )
    size = max(4.0, 0.45 * k + 2.5)
    fig = Figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax,
        cmap=cmap,
        colorbar=False,
        xticks_rotation=67.5,
        values_format=".2f" if normalize else "d",
    )
    ax.set_title(title or ("Normalised confusion matrix" if normalize else "Confusion matrix"))
    fig.tight_layout()
    return fig


def confusion_matrices(
    result: FastForestResult, *, stage: str = "final", split: str = "test"
) -> tuple[Figure, Figure]:
    """Return the raw and row-normalised confusion matrices of one stage on one split."""
    y_true = result.y[getattr(result.splits, split)]
    y_pred = result.predictions[stage][split]
    n = result.n_features[stage]
    label = f"{STAGE_LABELS[stage]} ({n}), {split} split"
    return (
        confusion_matrix_figure(y_true, y_pred, result.classes, title=f"Confusion matrix: {label}"),
        confusion_matrix_figure(
            y_true, y_pred, result.classes, normalize=True, title=f"Normalised: {label}"
        ),
    )


# ---------------------------------------------------------------------- importances
def importance_bar(
    importances: pd.Series, *, top_n: int = 30, title: str | None = None
) -> go.Figure:
    """Horizontal bar chart of the ``top_n`` most important features (largest on top)."""
    top = importances.head(top_n)
    fig = go.Figure(
        go.Bar(x=top.to_numpy()[::-1], y=[str(i) for i in top.index][::-1], orientation="h")
    )
    fig.update_layout(
        title=title or f"Top {len(top)} features",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, 22 * len(top) + 120),
        margin={"l": 10, "r": 10, "t": 50, "b": 40},
    )
    return fig


def search_history_plot(result: FastForestResult) -> go.Figure:
    """Validation metrics against the number of features at every step of the threshold search.

    The dashed line is the bound: full-model accuracy minus ``bound``.
    """
    rows = []
    for step in result.search_history:
        row = {"q": step["q"], "n_features": step["n_features"], "satisfied": step["satisfied"]}
        row.update(step["metrics"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    fig = go.Figure()
    for name in METRIC_NAMES:
        fig.add_trace(
            go.Scatter(
                x=frame["n_features"],
                y=frame[name],
                mode="lines+markers",
                name=name.replace("_", " "),
                customdata=frame["q"],
                hovertemplate="q=%{customdata}<br>%{y:.4f} with %{x} features<extra></extra>",
            )
        )
    full_acc = result.metrics["full"]["val"]["accuracy"]
    fig.add_hline(
        y=full_acc - result.params["bound"],
        line_dash="dash",
        annotation_text="accuracy bound",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Threshold search (validation split)",
        xaxis_title="Features kept",
        yaxis_title="Metric",
        xaxis_type="log",
        legend_title=None,
    )
    return fig


# ------------------------------------------------------------------------ dendrogram
def dendrogram(
    redundancy: RedundancyResult,
    *,
    font_size: int = 8,
    title: str = "Similarity dendrogram (distance = 1 − Spearman ρ)",
) -> Figure:
    """Dendrogram of the retained features with the cut height drawn as a dashed line."""
    n = len(redundancy.labels)
    height = min(60.0, max(4.0, 0.28 * n + 1.5))
    fig = Figure(figsize=(9, height))
    ax = fig.add_subplot(111)
    if redundancy.skipped:
        ax.text(
            0.5,
            0.5,
            "Fewer than 3 features: redundancy step skipped",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig
    hc.dendrogram(
        redundancy.linkage,
        labels=redundancy.labels,
        orientation="left",
        leaf_font_size=font_size,
        color_threshold=redundancy.threshold,
        ax=ax,
    )
    ax.axvline(
        redundancy.threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"cut at d = {redundancy.threshold:g}",
    )
    ax.set_xlabel("distance (1 − ρ)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- per-class distributions
def class_heatmaps(
    X: pd.DataFrame, y, class_names: list[str], *, method: str = "spearman"
) -> dict[str, go.Figure]:
    """Per-class correlation heatmaps of the final features.

    Parameters
    ----------
    X : pandas.DataFrame
        Final-feature values (all rows).
    y : array-like
        Encoded labels for the rows of ``X``.
    class_names : list of str
        Class names.
    method : {"spearman", "pearson", "kendall"}, default "spearman"
        Correlation method passed to :meth:`pandas.DataFrame.corr`.

    Returns
    -------
    dict
        Class name → plotly figure.
    """
    y = np.asarray(y, dtype=int)
    figs = {}
    for k, name in enumerate(class_names):
        sub = X.iloc[np.flatnonzero(y == k)]
        if len(sub) > 1:
            corr = sub.corr(method=method).fillna(0.0)
        else:
            corr = pd.DataFrame(0.0, index=X.columns, columns=X.columns)
        fig = px.imshow(
            corr,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title=f"{method.title()} correlation of final features: {name} (n = {len(sub)})",
        )
        fig.update_xaxes(tickangle=90)
        fig.update_layout(height=max(400, 18 * len(X.columns) + 150))
        figs[name] = fig
    return figs


def class_histograms(
    X: pd.DataFrame,
    y,
    class_names: list[str],
    *,
    bins: int = 20,
    max_features: int | None = None,
) -> dict[str, Figure]:
    """Per-class histograms of the final features (raw, un-scaled values).

    Parameters
    ----------
    X : pandas.DataFrame
        Final-feature values (all rows), columns in importance order.
    y : array-like
        Encoded labels for the rows of ``X``.
    class_names : list of str
        Class names.
    bins : int, default 20
        Histogram bins.
    max_features : int, optional
        Plot only the first ``max_features`` columns.

    Returns
    -------
    dict
        Class name → matplotlib figure.
    """
    y = np.asarray(y, dtype=int)
    columns = list(X.columns)[:max_features] if max_features else list(X.columns)
    colours = palette(len(class_names))
    ncols = min(4, max(1, len(columns)))
    nrows = max(1, math.ceil(len(columns) / ncols))
    figs = {}
    for k, name in enumerate(class_names):
        sub = X.iloc[np.flatnonzero(y == k)]
        fig = Figure(figsize=(3.2 * ncols, 2.4 * nrows + 0.6))
        axes = fig.subplots(nrows, ncols, squeeze=False)
        for i, col in enumerate(columns):
            ax = axes[i // ncols][i % ncols]
            ax.hist(sub[col].to_numpy(dtype=float), bins=bins, color=colours[k])
            ax.set_title(str(col), fontsize=9)
            ax.tick_params(labelsize=7)
        for j in range(len(columns), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.suptitle(f"Final feature distributions: {name} (n = {len(sub)})")
        fig.tight_layout()
        figs[name] = fig
    return figs


# ------------------------------------------------------------------------ embeddings
def _scatter(emb: np.ndarray, y, class_names, *, axis_prefix: str, title: str) -> go.Figure:
    labels = _labels(y, class_names)
    cols = [f"{axis_prefix}{i + 1}" for i in range(emb.shape[1])]
    frame = pd.DataFrame(emb, columns=cols)
    frame["class"] = labels
    order = {"class": list(class_names)}
    colours = dict(zip(class_names, palette(len(class_names)), strict=True))
    if emb.shape[1] >= 3:
        fig = px.scatter_3d(
            frame,
            x=cols[0],
            y=cols[1],
            z=cols[2],
            color="class",
            category_orders=order,
            color_discrete_map=colours,
            title=title,
        )
        fig.update_traces(marker={"size": 4, "line": {"width": 0.5, "color": "white"}})
    else:
        fig = px.scatter(
            frame,
            x=cols[0],
            y=cols[1] if len(cols) > 1 else None,
            color="class",
            category_orders=order,
            color_discrete_map=colours,
            title=title,
        )
        fig.update_traces(marker={"size": 7, "line": {"width": 0.5, "color": "white"}})
    fig.update_layout(legend_title=None)
    return fig


def lda_available_components(n_classes: int, n_features: int) -> int:
    """Maximum number of LDA components: ``min(n_classes - 1, n_features)``."""
    return max(0, min(n_classes - 1, n_features))


def lda_embedding(X: pd.DataFrame, y, *, n_components: int = 2) -> np.ndarray:
    """Project ``X`` onto its first ``n_components`` linear discriminants.

    Raises
    ------
    ValueError
        If ``n_components`` exceeds :func:`lda_available_components` (2-D needs at least 3
        classes, 3-D at least 4; bug B8).
    """
    y = np.asarray(y, dtype=int)
    available = lda_available_components(len(np.unique(y)), X.shape[1])
    if n_components > available:
        raise ValueError(
            f"LDA with {n_components} components needs at least {n_components + 1} classes and "
            f"{n_components} features; only {available} component(s) are available"
        )
    return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(
        X.to_numpy(dtype=float), y
    )


def lda_scatter(
    X: pd.DataFrame, y, class_names: list[str], *, n_components: int = 2, title: str | None = None
) -> go.Figure:
    """LDA scatter plot of the final features (2-D for K >= 3 classes, 3-D for K >= 4)."""
    emb = lda_embedding(X, y, n_components=n_components)
    return _scatter(emb, y, class_names, axis_prefix="LD", title=title or f"LDA ({n_components}-D)")


def umap_embedding(
    X: pd.DataFrame,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int | None = None,
) -> np.ndarray:
    """Compute a UMAP embedding of ``X`` (seeded, single-threaded when a seed is given).

    ``n_neighbors`` must be at least 2 (bug B9) and is capped at ``n_samples - 1``.
    """
    import umap  # heavy import (numba); keep it lazy

    n = len(X)
    if n_neighbors < 2:
        raise ValueError("UMAP needs n_neighbors >= 2")
    n_neighbors = min(int(n_neighbors), max(2, n - 1))
    kwargs = {}
    if random_state is not None:
        kwargs["n_jobs"] = 1  # UMAP ignores n_jobs > 1 with a seed and would warn
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        **kwargs,
    )
    return np.asarray(reducer.fit_transform(X.to_numpy(dtype=float)))


def umap_scatter(
    X: pd.DataFrame,
    y,
    class_names: list[str],
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = None,
    title: str | None = None,
) -> go.Figure:
    """UMAP scatter plot (2-D or 3-D) of the final features coloured by class."""
    emb = umap_embedding(
        X,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return _scatter(
        emb,
        y,
        class_names,
        axis_prefix="UMAP",
        title=title or f"UMAP ({n_components}-D, n_neighbors = {n_neighbors})",
    )
