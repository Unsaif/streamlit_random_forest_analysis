"""Stage S5: Spearman clustering; regression tests for bugs B1, B4 and B5."""

import numpy as np
import pandas as pd
import pytest

from fastforest.redundancy import find_redundant, spearman_distance


def correlated_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return pd.DataFrame(
        {
            "a": base,
            "b": base.copy(),  # exact duplicate, rho = 1
            "c": base + rng.normal(scale=0.02, size=n),  # near copy, rho > 0.99
            "d": rng.normal(size=n),  # unrelated
            "e": -base,  # perfectly anti-correlated: distance 2, never merged
        }
    )


def test_duplicate_and_near_copy_cluster_keeps_most_important_member():
    X = correlated_frame()
    importances = pd.Series({"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.2, "e": 0.05})
    res = find_redundant(X, importances)
    assert res.clusters == [["b", "c", "a"]]
    assert res.keep == ["b", "d", "e"]
    assert res.drop == ["c", "a"]
    assert not res.skipped
    assert res.linkage is not None and res.distance is not None


def test_b1_regression_alphabetically_last_feature_is_not_kept_by_name():
    """The old code used ``max(dict)``, which returned the alphabetically last name."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=200)
    X = pd.DataFrame({"zzz": base, "aaa": base.copy(), "mmm": rng.normal(size=200)})
    importances = pd.Series({"zzz": 0.1, "aaa": 0.9, "mmm": 0.5})
    res = find_redundant(X, importances)
    assert res.keep == ["aaa", "mmm"]
    assert res.drop == ["zzz"]


def test_ties_broken_by_column_order():
    rng = np.random.default_rng(2)
    base = rng.normal(size=100)
    X = pd.DataFrame({"second": base, "first": base.copy(), "other": rng.normal(size=100)})
    importances = pd.Series({"second": 0.4, "first": 0.4, "other": 0.2})
    res = find_redundant(X, importances)
    assert res.clusters == [["second", "first"]]


@pytest.mark.parametrize("n_cols", [1, 2])
def test_b4_b5_fewer_than_three_features_is_skipped_not_crashing(n_cols):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, n_cols)), columns=[f"f{i}" for i in range(n_cols)])
    importances = pd.Series(np.linspace(0.5, 0.1, n_cols), index=X.columns)
    res = find_redundant(X, importances)
    assert res.skipped
    assert res.drop == [] and res.keep == list(X.columns) and res.clusters == []


def test_constant_column_does_not_crash():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"const": np.ones(80), "x": rng.normal(size=80), "y": rng.normal(size=80)})
    importances = pd.Series({"const": 0.0, "x": 0.5, "y": 0.5})
    res = find_redundant(X, importances)
    assert res.drop == []


def test_threshold_controls_merging():
    rng = np.random.default_rng(3)
    base = rng.normal(size=500)
    X = pd.DataFrame(
        {"a": base, "b": base + rng.normal(scale=0.3, size=500), "c": rng.normal(size=500)}
    )
    importances = pd.Series({"a": 0.6, "b": 0.3, "c": 0.1})
    rho = X[["a", "b"]].corr(method="spearman").iloc[0, 1]
    assert 0.9 < rho < 0.99
    assert find_redundant(X, importances, cluster_threshold=0.01).drop == []
    assert find_redundant(X, importances, cluster_threshold=0.2).drop == ["b"]


def test_missing_importance_raises():
    X = correlated_frame()
    with pytest.raises(ValueError, match="importances missing"):
        find_redundant(X, pd.Series({"a": 1.0}))


def test_spearman_distance_properties():
    X = correlated_frame()
    d = spearman_distance(X)
    arr = d.to_numpy()
    assert d.shape == (5, 5)
    assert np.allclose(arr, arr.T)
    assert np.allclose(np.diag(arr), 0.0)
    assert arr.min() >= 0.0 and arr.max() <= 2.0
    assert d.loc["a", "b"] == 0.0
    assert d.loc["a", "e"] == pytest.approx(2.0)
    assert spearman_distance(X[["a", "b"]]).loc["a", "b"] == 0.0
    assert spearman_distance(X[["a"]]).shape == (1, 1)
