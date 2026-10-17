"""End-to-end behaviour of :class:`fastforest.FastForest` on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from fastforest import FastForest
from fastforest.metrics import METRIC_NAMES, compute_metrics, within_bound
from fastforest.result import SPLITS, STAGES
from fastforest.split import make_splits
from tests.conftest import synthetic


def necessary_features_dataset(n: int = 1000, n_noise: int = 30, seed: int = 0):
    """Three Gaussian features whose arg-max is the class, plus pure-noise columns.

    Each of the three signal features is necessary: dropping any one of them costs far more
    than one percentage point of accuracy.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3 + n_noise))
    y = np.argmax(X[:, :3], axis=1)
    columns = ["need0", "need1", "need2"] + [f"noise{i}" for i in range(n_noise)]
    return pd.DataFrame(X, columns=columns), y


@pytest.fixture(scope="module")
def plan_result():
    """The dataset prescribed by the plan: 1,000 x 200, 10 informative, 10 redundant, 3 classes."""
    X, y, groups = synthetic()
    return FastForest(bound=0.01, random_state=0).fit(X, y), groups


def test_noise_dropped_and_only_signal_selected(plan_result):
    result, groups = plan_result
    noise = set(groups["noise"])
    signal = set(groups["informative"]) | set(groups["redundant"])
    assert len(noise - set(result.features_final)) >= 0.8 * len(noise)
    assert set(result.features_final) <= signal
    assert set(result.features_important) <= signal
    assert result.bound_satisfied and result.final_bound_satisfied
    assert 0 < len(result.features_final) < len(result.features_full)


def test_every_necessary_feature_is_retained():
    X, y = necessary_features_dataset()
    result = FastForest(bound=0.01, random_state=0).fit(X, y)
    assert set(result.features_final) >= {"need0", "need1", "need2"}
    assert not any(f.startswith("noise") for f in result.features_final)
    assert result.q_final is not None


def test_bound_holds_on_validation_for_every_stage(plan_result):
    result, _ = plan_result
    reference = result.metrics["full"]["val"]
    for stage in ("important", "final"):
        assert within_bound(reference, result.metrics[stage]["val"], 0.01, "accuracy")


def test_metrics_reported_for_every_stage_and_split(plan_result):
    result, _ = plan_result
    assert set(result.metrics) == set(STAGES)
    for stage in STAGES:
        assert set(result.metrics[stage]) == set(SPLITS)
        for split in SPLITS:
            assert set(result.metrics[stage][split]) == set(METRIC_NAMES)
            assert all(0.0 <= v <= 1.0 for v in result.metrics[stage][split].values())
        assert len(result.predictions[stage]["test"]) == len(result.splits.test)
    assert result.summary().shape == (9, 3 + len(METRIC_NAMES))


def test_deterministic_with_fixed_seed():
    X, y, _ = synthetic(n_samples=300, n_features=40, n_informative=5, n_redundant=2)
    a = FastForest(n_estimators=30, random_state=3).fit(X, y)
    b = FastForest(n_estimators=30, random_state=3).fit(X, y)
    c = FastForest(n_estimators=30, random_state=4).fit(X, y)
    assert a.features_final == b.features_final
    assert a.metrics == b.metrics
    assert a.splits.to_dict() == b.splits.to_dict()
    assert a.splits.to_dict() != c.splits.to_dict()


def test_target_column_name_and_premade_splits_are_accepted():
    X, y, _ = synthetic(n_samples=300, n_features=40, n_informative=5, n_redundant=2)
    frame = X.assign(label=[f"c{v}" for v in y])
    splits = make_splits(y, random_state=11)
    a = FastForest(n_estimators=30, random_state=0).fit(frame, "label", splits=splits)
    b = FastForest(n_estimators=30, random_state=0).fit(X, frame["label"], splits=tuple(splits))
    assert a.classes == ["c0", "c1", "c2"]
    assert a.splits.to_dict() == b.splits.to_dict() == splits.to_dict()
    assert a.features_final == b.features_final
    assert "label" not in a.features_full
    assert a.X_final.shape == (300, len(a.features_final))
    assert a.y.tolist() == y.tolist()


def test_index_split_and_numpy_input():
    X, y, _ = synthetic(n_samples=200, n_features=20, n_informative=4, n_redundant=0)
    test_idx = list(range(0, 200, 5))
    result = FastForest(n_estimators=20, split="index", random_state=0).fit(
        X.to_numpy(), y, test_idx=test_idx
    )
    assert result.splits.test.tolist() == test_idx
    assert result.features_full == [str(i) for i in range(20)]


def test_permutation_importance_and_all_criterion_without_enforcement():
    X, y, _ = synthetic(n_samples=300, n_features=30, n_informative=5, n_redundant=3)
    result = FastForest(
        n_estimators=30,
        importance="permutation",
        n_repeats=2,
        criterion="all",
        bound=0.05,
        enforce_bound=False,
        add_missing_indicator=True,
        random_state=0,
    ).fit(X, y)
    assert result.params["importance"] == "permutation"
    assert result.features_final
    assert result.reinserted == []
    assert set(result.importances["full"].index) == set(X.columns)


def test_callback_receives_stage_events_and_search_steps():
    X, y, _ = synthetic(n_samples=300, n_features=30, n_informative=5, n_redundant=0)
    events = []
    FastForest(n_estimators=20, random_state=0).fit(X, y, callback=events.append)
    stages = [e["stage"] for e in events]
    for expected in ("split", "preprocess", "full_model", "importance", "search", "done"):
        assert expected in stages
    search_steps = [e["step"] for e in events if e["stage"] == "search"]
    assert search_steps and search_steps[-1].satisfied
    assert all("elapsed" in e for e in events if e["stage"] != "search")


def test_categorical_and_missing_values_run_end_to_end():
    X, y, _ = synthetic(n_samples=300, n_features=20, n_informative=4, n_redundant=0)
    rng = np.random.default_rng(0)
    X = X.copy()
    X.loc[rng.choice(300, 30, replace=False), "inf0"] = np.nan
    X["colour"] = rng.choice(["red", "green", "blue"], size=300)
    X["flag"] = rng.choice([True, False], size=300)
    result = FastForest(n_estimators=20, random_state=0).fit(X, y)
    assert result.preprocessor.categorical_columns_ == ["colour", "flag"]
    assert not result.X_final.isna().any().any()
    json.dumps(result.to_dict())


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        FastForest(criterion="bogus")
    with pytest.raises(ValueError):
        FastForest(importance="shap")
    with pytest.raises(ValueError):
        FastForest(split="sideways")
    with pytest.raises(ValueError):
        FastForest(bound=-1)
    with pytest.raises(ValueError):
        FastForest(q_start=0.5, q_step=0.6)


def test_invalid_inputs_raise():
    X, y, _ = synthetic(n_samples=100, n_features=10, n_informative=3, n_redundant=0)
    ff = FastForest(n_estimators=10, random_state=0)
    with pytest.raises(KeyError):
        ff.fit(X, "missing_column")
    with pytest.raises(ValueError, match="rows"):
        ff.fit(X, y[:-1])
    with pytest.raises(ValueError, match="two classes"):
        ff.fit(X, np.zeros(100, dtype=int))
    with pytest.raises(ValueError, match="duplicate"):
        ff.fit(pd.DataFrame(X.to_numpy(), columns=["a"] * 10), y)
    with pytest.raises(ValueError, match="no feature"):
        ff.fit(X.iloc[:, :0], y)


# ----------------------------------------------------------------- bound enforcement (D6)
class KeyClassifier:
    """Predicts ``X['key'] > 0`` when ``key`` is present, otherwise always class 0."""

    def fit(self, X, y):
        self.columns_ = list(X.columns)
        return self

    def predict(self, X):
        if "key" in self.columns_:
            return (X["key"].to_numpy() > 0).astype(int)
        return np.zeros(len(X), dtype=int)


class StubForest(FastForest):
    def _forest(self):
        return KeyClassifier()


def _enforcement_setup():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["key", "dup", "other"])
    y = (X["key"] > 0).astype(int).to_numpy()
    X_train, X_val, y_train, y_val = X.iloc[:60], X.iloc[60:], y[:60], y[60:]
    features_important = ["key", "dup", "other"]
    imp = pd.Series({"key": 0.9, "dup": 0.05, "other": 0.05})
    model_important = KeyClassifier().fit(X_train, y_train)
    perfect = compute_metrics(y_val, model_important.predict(X_val))
    return X_train, y_train, X_val, y_val, features_important, imp, model_important, perfect


def test_bound_enforcement_reinserts_dropped_feature_in_importance_order():
    X_train, y_train, X_val, y_val, important, imp, model, perfect = _enforcement_setup()
    ff = StubForest(bound=0.0, enforce_bound=True)
    features, final_model, metrics, reinserted = ff._final_model(
        X_train, y_train, X_val, y_val, important, ["key"], imp, perfect, model, perfect
    )
    assert reinserted == ["key"]
    assert features == important  # original order restored
    assert final_model is model  # identical feature set reuses the S4 model
    assert metrics["accuracy"] == 1.0


def test_bound_enforcement_off_keeps_the_reduced_set():
    X_train, y_train, X_val, y_val, important, imp, model, perfect = _enforcement_setup()
    ff = StubForest(bound=0.0, enforce_bound=False)
    features, final_model, metrics, reinserted = ff._final_model(
        X_train, y_train, X_val, y_val, important, ["key"], imp, perfect, model, perfect
    )
    assert reinserted == []
    assert features == ["dup", "other"]
    assert metrics["accuracy"] < 1.0


def test_nothing_dropped_returns_the_s4_model_unchanged():
    X_train, y_train, X_val, y_val, important, imp, model, perfect = _enforcement_setup()
    ff = StubForest()
    features, final_model, metrics, reinserted = ff._final_model(
        X_train, y_train, X_val, y_val, important, [], imp, perfect, model, perfect
    )
    assert (features, final_model, metrics, reinserted) == (important, model, perfect, [])
