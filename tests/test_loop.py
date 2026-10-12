"""Stage S4 termination (bug B2): the search ends by counting quantile steps."""

import numpy as np
import pandas as pd

from fastforest.importance import compute_importance
from fastforest.metrics import compute_metrics
from fastforest.reduce import make_forest, quantile_levels, quantile_threshold_search
from fastforest.split import make_splits


def test_quantile_levels_default_schedule():
    levels = quantile_levels()
    assert len(levels) == 100
    assert levels[0] == 0.99 and levels[-1] == 0.0
    assert all(b < a for a, b in zip(levels[:-1], levels[1:], strict=True))
    assert min(levels) >= 0.0


def test_quantile_levels_custom_and_invalid():
    assert quantile_levels(0.9, 0.1) == [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    assert quantile_levels(0.95, 0.05)[-1] == 0.0
    import pytest

    with pytest.raises(ValueError):
        quantile_levels(0.5, 0.6)


def _setup(small_synthetic):
    X, y, _ = small_synthetic
    s = make_splits(y, random_state=0)
    X_train, X_val = X.iloc[s.train], X.iloc[s.val]
    y_train, y_val = y[s.train], y[s.val]
    forest = lambda: make_forest(n_estimators=20, random_state=0)  # noqa: E731
    full = forest().fit(X_train, y_train)
    imp = compute_importance(full, X_val)
    metrics_full = compute_metrics(y_val, full.predict(X_val))
    return X_train, y_train, X_val, y_val, imp, metrics_full, forest


def test_search_terminates_when_bound_is_never_satisfied(small_synthetic):
    X_train, y_train, X_val, y_val, imp, _, forest = _setup(small_synthetic)
    unreachable = {
        name: 2.0 for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    }
    res = quantile_threshold_search(
        X_train, y_train, X_val, y_val, imp, unreachable, bound=0.0, forest_factory=forest
    )
    positive = imp[imp > 0].index.tolist()
    assert res.features == positive
    assert res.q_final is None
    assert res.bound_satisfied is False
    assert res.history[-1].q is None
    assert len(res.history) <= 101
    assert all(step.satisfied is False for step in res.history)


def test_search_stops_at_first_satisfying_quantile(small_synthetic):
    X_train, y_train, X_val, y_val, imp, _, forest = _setup(small_synthetic)
    trivial = {name: 0.0 for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    res = quantile_threshold_search(
        X_train, y_train, X_val, y_val, imp, trivial, bound=0.0, forest_factory=forest
    )
    assert res.q_final == 0.99
    assert res.bound_satisfied
    assert len(res.history) == 1
    assert res.history[0].n_features == len(res.features)


def test_search_reuses_models_for_identical_feature_sets(small_synthetic):
    X_train, y_train, X_val, y_val, imp, metrics_full, forest = _setup(small_synthetic)
    tied = pd.Series(np.where(imp.to_numpy() > 0, 1.0, 0.0), index=imp.index)
    unreachable = {name: 2.0 for name in metrics_full}
    res = quantile_threshold_search(
        X_train, y_train, X_val, y_val, tied, unreachable, bound=0.0, forest_factory=forest
    )
    # with all positive importances tied, "importance > quantile" is empty at every q,
    # so only the fallback is evaluated
    assert len(res.history) == 1 and res.history[0].retrained


def test_search_with_callback_and_explicit_metric_list(small_synthetic):
    X_train, y_train, X_val, y_val, imp, metrics_full, forest = _setup(small_synthetic)
    seen = []
    res = quantile_threshold_search(
        X_train,
        y_train,
        X_val,
        y_val,
        imp,
        metrics_full,
        bound=0.05,
        criterion=("accuracy", "macro_f1"),
        forest_factory=forest,
        callback=lambda event: seen.append(event["step"].q),
    )
    assert seen == [step.q for step in res.history]
    assert res.bound_satisfied


def test_no_positive_importance_raises(small_synthetic):
    import pytest

    X_train, y_train, X_val, y_val, imp, metrics_full, forest = _setup(small_synthetic)
    with pytest.raises(ValueError, match="positive importance"):
        quantile_threshold_search(
            X_train, y_train, X_val, y_val, imp * 0.0, metrics_full, forest_factory=forest
        )
