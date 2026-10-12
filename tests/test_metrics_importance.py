"""Metrics, the bound test and importance ranking."""

import numpy as np
import pandas as pd
import pytest

from fastforest.importance import compute_importance, sort_importances
from fastforest.metrics import (
    METRIC_NAMES,
    compute_metrics,
    criterion_metrics,
    metric_drops,
    within_bound,
)
from fastforest.reduce import make_forest


def test_compute_metrics_perfect_and_imperfect():
    perfect = compute_metrics([0, 1, 2, 2], [0, 1, 2, 2])
    assert set(perfect) == set(METRIC_NAMES)
    assert all(v == 1.0 for v in perfect.values())
    m = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert m["accuracy"] == 0.5
    assert m["macro_recall"] == 0.5
    assert m["macro_precision"] == 0.25  # zero_division=0 for the never-predicted class


def test_criterion_metrics():
    assert criterion_metrics("accuracy") == ("accuracy",)
    assert criterion_metrics("all") == ("accuracy", "macro_precision", "macro_recall")
    assert criterion_metrics(["macro_f1"]) == ("macro_f1",)
    with pytest.raises(ValueError):
        criterion_metrics("bogus")
    with pytest.raises(ValueError):
        criterion_metrics(["nope"])
    with pytest.raises(ValueError):
        criterion_metrics([])


def test_within_bound_semantics():
    ref = {"accuracy": 0.9, "macro_precision": 0.8, "macro_recall": 0.7, "macro_f1": 0.75}
    cand = {"accuracy": 0.89, "macro_precision": 0.5, "macro_recall": 0.7, "macro_f1": 0.75}
    assert within_bound(ref, cand, 0.0101, "accuracy")
    assert not within_bound(ref, cand, 0.0099, "accuracy")
    assert not within_bound(ref, cand, 0.0101, "all")  # precision dropped by 0.3
    better = {k: v + 0.1 for k, v in ref.items()}
    assert within_bound(ref, better, 0.0)
    assert metric_drops(ref, cand)["macro_precision"] == pytest.approx(0.3)


def test_sort_importances_descending_with_stable_ties():
    s = sort_importances([0.2, 0.5, 0.2, 0.0], ["a", "b", "c", "d"])
    assert s.index.tolist() == ["b", "a", "c", "d"]
    assert s.tolist() == [0.5, 0.2, 0.2, 0.0]
    with pytest.raises(ValueError):
        sort_importances([1.0], ["a", "b"])


def test_compute_importance_mdi_and_permutation(small_synthetic):
    X, y, groups = small_synthetic
    model = make_forest(n_estimators=30, random_state=0).fit(X, y)
    mdi = compute_importance(model, X)
    assert mdi.index.tolist()[0] in groups["informative"]
    assert np.isclose(mdi.sum(), 1.0)
    perm = compute_importance(model, X, y, method="permutation", n_repeats=2, random_state=0)
    assert set(perm.index) == set(X.columns)
    assert perm.iloc[0] >= perm.iloc[-1]
    assert isinstance(perm, pd.Series)
    with pytest.raises(ValueError, match="requires y"):
        compute_importance(model, X, method="permutation")
    with pytest.raises(ValueError, match="method"):
        compute_importance(model, X, method="shap")
