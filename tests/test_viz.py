"""Stage S7 figure functions return figure objects and respect B8/B9."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure

from fastforest import viz
from fastforest.redundancy import RedundancyResult


def test_confusion_matrices(sample_result):
    raw, norm = viz.confusion_matrices(sample_result, stage="final", split="test")
    assert isinstance(raw, Figure) and isinstance(norm, Figure)
    val_raw, _ = viz.confusion_matrices(sample_result, stage="full", split="val")
    assert isinstance(val_raw, Figure)
    fig = viz.confusion_matrix_figure([0, 1], [0, 1], ["a", "b"], normalize=True, title="t")
    assert fig.axes[0].get_title() == "t"


def test_importance_bar_puts_largest_on_top(sample_result):
    imp = sample_result.importances["full"]
    fig = viz.importance_bar(imp, top_n=10)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].y) == 10
    assert fig.data[0].y[-1] == imp.index[0]
    assert fig.data[0].x[-1] == pytest.approx(imp.iloc[0])


def test_search_history_plot(sample_result):
    fig = viz.search_history_plot(sample_result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4


def test_dendrogram_and_skipped_variant(sample_result):
    assert isinstance(viz.dendrogram(sample_result.redundancy), Figure)
    skipped = RedundancyResult(["a", "b"], [], ["a", "b"], [], 0.01)
    fig = viz.dendrogram(skipped)
    assert isinstance(fig, Figure)
    assert "skipped" in fig.axes[0].texts[0].get_text()


def test_class_heatmaps_and_histograms(sample_result):
    X, y, classes = sample_result.X_final, sample_result.y, sample_result.classes
    heatmaps = viz.class_heatmaps(X, y, classes)
    assert list(heatmaps) == classes
    assert all(isinstance(f, go.Figure) for f in heatmaps.values())
    hists = viz.class_histograms(X, y, classes, max_features=5)
    assert list(hists) == classes
    fig = hists[classes[0]]
    assert isinstance(fig, Figure)
    assert sum(ax.axison for ax in fig.axes) == 5
    # a class with a single row still produces a heatmap
    single = viz.class_heatmaps(X.iloc[:1], np.array([0]), ["only"])
    assert isinstance(single["only"], go.Figure)


def test_lda_component_rules_b8(sample_result):
    assert viz.lda_available_components(3, 10) == 2
    assert viz.lda_available_components(4, 10) == 3
    assert viz.lda_available_components(2, 10) == 1
    assert viz.lda_available_components(5, 2) == 2
    X, y, classes = sample_result.X_final, sample_result.y, sample_result.classes
    assert isinstance(viz.lda_scatter(X, y, classes, n_components=2), go.Figure)
    assert isinstance(viz.lda_scatter(X, y, classes, n_components=3), go.Figure)
    three_classes = np.asarray(y) < 3
    with pytest.raises(ValueError, match="components"):
        viz.lda_embedding(X[three_classes], y[three_classes], n_components=3)
    two_classes = np.asarray(y) < 2
    with pytest.raises(ValueError):
        viz.lda_scatter(X[two_classes], y[two_classes], classes[:2], n_components=2)


def test_umap_rejects_n_neighbors_below_two_b9():
    X = pd.DataFrame(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="n_neighbors"):
        viz.umap_embedding(X, n_neighbors=1)


def test_umap_embedding_and_scatter():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    y = np.repeat([0, 1, 2], 20)
    emb = viz.umap_embedding(X, n_components=2, n_neighbors=200, random_state=0)
    assert emb.shape == (60, 2)  # n_neighbors is capped at n - 1
    fig = viz.umap_scatter(X, y, ["x", "y", "z"], n_components=3, n_neighbors=5, random_state=0)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3


def test_palette_sizes():
    assert len(viz.palette(3)) == 3
    assert len(set(viz.palette(15))) == 15
