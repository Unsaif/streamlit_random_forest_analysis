"""Stages S2–S6: full model, importance ranking, quantile-threshold search, redundancy removal
and the final model with bound enforcement, orchestrated by :class:`FastForest`."""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fastforest.importance import IMPORTANCE_METHODS, compute_importance
from fastforest.metrics import compute_metrics, criterion_metrics, within_bound
from fastforest.preprocess import Preprocessor, encode_labels
from fastforest.redundancy import find_redundant
from fastforest.result import FastForestResult
from fastforest.split import SPLIT_METHODS, Splits, make_splits

logger = logging.getLogger("fastforest")

Callback = Callable[[dict[str, Any]], None]
"""Progress callback: receives a dict with at least ``stage`` and ``message``."""


def make_forest(
    *,
    n_estimators: int = 100,
    max_features: float | int | str = 0.5,
    min_samples_leaf: int = 5,
    random_state: int | None = None,
    n_jobs: int | None = -1,
    oob_score: bool = True,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Return an unfitted random forest with the FastForest hyper-parameters (stage S2).

    Parameters
    ----------
    n_estimators : int, default 100
        Number of trees.
    max_features : float, int or str, default 0.5
        Features considered per split (scikit-learn semantics).
    min_samples_leaf : int, default 5
        Minimum samples per leaf.
    random_state : int, optional
        Seed; the same seed is reused for every retrained forest in a run.
    n_jobs : int, default -1
        Parallel workers.
    oob_score : bool, default True
        Record the out-of-bag score.
    **kwargs
        Passed through to :class:`sklearn.ensemble.RandomForestClassifier`.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        oob_score=oob_score,
        **kwargs,
    )


def quantile_levels(q_start: float = 0.99, q_step: float = 0.01) -> list[float]:
    """Return the quantile levels ``q_start, q_start - q_step, …, 0.0`` (bug B2 fix).

    The sequence is built by counting steps, never by comparing floats to zero.
    """
    if not (0 < q_step <= q_start <= 1):
        raise ValueError("require 0 < q_step <= q_start <= 1")
    n_steps = int(round(q_start / q_step))
    levels = []
    for i in range(n_steps + 1):
        level = round(q_start - i * q_step, 10)
        levels.append(0.0 if level <= 0 else level)
    if levels[-1] != 0.0:
        levels.append(0.0)
    return levels


@dataclass
class SearchStep:
    """One evaluated quantile of the threshold search.

    ``q`` is None for the fallback model that keeps every positive-importance feature.
    ``retrained`` is False when the feature set was identical to the previous step and the
    previous model was reused.
    """

    q: float | None
    threshold: float
    n_features: int
    metrics: dict[str, float]
    satisfied: bool
    retrained: bool


@dataclass
class SearchResult:
    """Outcome of :func:`quantile_threshold_search`."""

    features: list[str]
    model: RandomForestClassifier
    metrics_val: dict[str, float]
    q_final: float | None
    bound_satisfied: bool
    history: list[SearchStep]


def quantile_threshold_search(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    importances: pd.Series,
    metrics_full: dict[str, float],
    *,
    bound: float = 5e-4,
    criterion="accuracy",
    q_start: float = 0.99,
    q_step: float = 0.01,
    forest_factory: Callable[[], RandomForestClassifier] | None = None,
    callback: Callback | None = None,
) -> SearchResult:
    """Find the smallest importance-thresholded feature set within ``bound`` of the full model.

    Let ``I+`` be the features with importance > 0. For ``q = q_start, q_start - q_step, …, 0``
    the features with importance above the ``q``-quantile of ``I+`` are kept, a forest is
    retrained on them and evaluated on the validation split, and the search stops at the first
    ``q`` for which every metric in ``criterion`` drops by at most ``bound``. If no ``q``
    succeeds, all of ``I+`` is kept and the bound is re-checked on that model.

    Parameters
    ----------
    X_train, X_val : pandas.DataFrame
        Preprocessed training and validation features (all columns).
    y_train, y_val : array-like
        Encoded labels.
    importances : pandas.Series
        Importance per feature (any order; see :func:`fastforest.importance.compute_importance`).
    metrics_full : dict
        Validation metrics of the full model (the reference for the bound).
    bound : float, default 5e-4
        Maximum allowed drop, absolute on the [0, 1] scale.
    criterion : {"accuracy", "all"} or iterable of str, default "accuracy"
        Metrics the bound applies to.
    q_start, q_step : float
        First quantile and step size.
    forest_factory : callable, optional
        Returns a fresh unfitted forest; defaults to :func:`make_forest`.
    callback : callable, optional
        Called after every step with ``{"stage": "search", "step": SearchStep}``.

    Returns
    -------
    SearchResult
        Selected features (descending importance), fitted model, validation metrics, final
        quantile (None for the fallback), whether the bound held, and the step history.
    """
    factory = forest_factory or make_forest
    criterion_names = criterion_metrics(criterion)
    imp = pd.Series(importances, dtype=np.float64)
    imp = imp.iloc[np.lexsort((np.arange(len(imp)), -imp.to_numpy()))]
    positive = imp[imp > 0]
    if positive.empty:
        raise ValueError("no feature has positive importance; nothing can be selected")
    names = imp.index.to_numpy()
    values = imp.to_numpy()

    history: list[SearchStep] = []
    cache: dict[str, Any] = {"features": None, "model": None, "metrics": None}

    def evaluate(features: list[str]) -> tuple[RandomForestClassifier, dict[str, float], bool]:
        if features == cache["features"]:
            return cache["model"], cache["metrics"], False
        model = factory().fit(X_train[features], y_train)
        metrics = compute_metrics(y_val, model.predict(X_val[features]))
        cache.update(features=features, model=model, metrics=metrics)
        return model, metrics, True

    def record(q, threshold, features, model, metrics, retrained) -> SearchStep:
        ok = within_bound(metrics_full, metrics, bound, criterion_names)
        step = SearchStep(q, float(threshold), len(features), metrics, ok, retrained)
        history.append(step)
        logger.info(
            "search q=%s: %d features, accuracy %.4f, %s",
            "all" if q is None else f"{q:.2f}",
            len(features),
            metrics["accuracy"],
            "within bound" if ok else "outside bound",
        )
        if callback is not None:
            callback({"stage": "search", "message": "threshold search", "step": step})
        return step

    for q in quantile_levels(q_start, q_step):
        threshold = float(positive.quantile(q))
        features = names[values > threshold].tolist()
        if not features:
            continue
        model, metrics, retrained = evaluate(features)
        step = record(q, threshold, features, model, metrics, retrained)
        if step.satisfied:
            return SearchResult(features, model, metrics, q, True, history)

    features = positive.index.tolist()
    model, metrics, retrained = evaluate(features)
    step = record(None, 0.0, features, model, metrics, retrained)
    return SearchResult(features, model, metrics, None, step.satisfied, history)


def _resolve_xy(X, y) -> tuple[pd.DataFrame, np.ndarray]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, str):
        if y not in X.columns:
            raise KeyError(f"target column {y!r} not found in X")
        target = X[y].to_numpy()
        X = X.drop(columns=[y])
    else:
        target = np.asarray(y)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            target = y.to_numpy()
    target = target.ravel()
    if len(target) != len(X):
        raise ValueError(f"X has {len(X)} rows but y has {len(target)} values")
    if X.shape[1] == 0:
        raise ValueError("X has no feature columns")
    if any(not isinstance(c, str) for c in X.columns):
        X = X.rename(columns=str)
    if X.columns.has_duplicates:
        raise ValueError("X has duplicate column names")
    return X, target


class FastForest:
    """Random-forest feature reduction (stages S0–S6 of the FastForest method).

    Parameters
    ----------
    bound : float, default 5e-4
        Maximum allowed drop of each criterion metric relative to the full model, absolute on
        the [0, 1] scale (``5e-4`` is 0.05 percentage points).
    criterion : {"accuracy", "all"}, default "accuracy"
        Metrics the bound applies to: accuracy only, or accuracy, macro precision and macro
        recall together.
    n_estimators : int, default 100
        Trees per forest.
    max_features : float, int or str, default 0.5
        Features considered per split.
    min_samples_leaf : int, default 5
        Minimum samples per leaf.
    importance : {"mdi", "permutation"}, default "mdi"
        Mean decrease in impurity, or permutation importance on the validation split.
    n_repeats : int, default 5
        Permutations per feature when ``importance="permutation"``.
    q_start, q_step : float, default 0.99 and 0.01
        Quantile schedule of the threshold search.
    cluster_threshold : float, default 0.01
        Cut height (``1 - rho``) for redundancy clustering; 0.01 merges ``rho >= 0.99``.
    enforce_bound : bool, default True
        After redundancy removal, re-insert dropped features in importance order until the
        validation bound holds again (decision D6).
    split : {"stratified", "random", "index"}, default "stratified"
        Split method; see :func:`fastforest.split.make_splits`.
    val_size, test_size : float, default 0.2
        Validation and test fractions.
    max_card, categorical_low_cardinality, add_missing_indicator
        Passed to :class:`fastforest.preprocess.Preprocessor`.
    n_jobs : int, default -1
        Parallel workers for forests and permutation importance.
    random_state : int, optional
        Seed for every stochastic step (splits, forests, permutation importance).

    Examples
    --------
    >>> from fastforest import FastForest, load_sample
    >>> X, y = load_sample()
    >>> result = FastForest(n_estimators=50, random_state=0).fit(X, y)
    >>> len(result.features_final) < len(result.features_full)
    True
    """

    def __init__(
        self,
        *,
        bound: float = 5e-4,
        criterion: str = "accuracy",
        n_estimators: int = 100,
        max_features: float | int | str = 0.5,
        min_samples_leaf: int = 5,
        importance: str = "mdi",
        n_repeats: int = 5,
        q_start: float = 0.99,
        q_step: float = 0.01,
        cluster_threshold: float = 0.01,
        enforce_bound: bool = True,
        split: str = "stratified",
        val_size: float = 0.2,
        test_size: float = 0.2,
        max_card: int = 20,
        categorical_low_cardinality: bool = False,
        add_missing_indicator: bool = False,
        n_jobs: int | None = -1,
        random_state: int | None = None,
    ) -> None:
        criterion_metrics(criterion)
        if importance not in IMPORTANCE_METHODS:
            raise ValueError(f"importance must be one of {IMPORTANCE_METHODS}")
        if split not in SPLIT_METHODS:
            raise ValueError(f"split must be one of {SPLIT_METHODS}")
        if bound < 0:
            raise ValueError("bound must be non-negative")
        quantile_levels(q_start, q_step)
        self.bound = bound
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.importance = importance
        self.n_repeats = n_repeats
        self.q_start = q_start
        self.q_step = q_step
        self.cluster_threshold = cluster_threshold
        self.enforce_bound = enforce_bound
        self.split = split
        self.val_size = val_size
        self.test_size = test_size
        self.max_card = max_card
        self.categorical_low_cardinality = categorical_low_cardinality
        self.add_missing_indicator = add_missing_indicator
        self.n_jobs = n_jobs
        self.random_state = random_state

    def get_params(self) -> dict[str, Any]:
        """Return the constructor parameters as a dict."""
        return {
            "bound": self.bound,
            "criterion": self.criterion,
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "importance": self.importance,
            "n_repeats": self.n_repeats,
            "q_start": self.q_start,
            "q_step": self.q_step,
            "cluster_threshold": self.cluster_threshold,
            "enforce_bound": self.enforce_bound,
            "split": self.split,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "max_card": self.max_card,
            "categorical_low_cardinality": self.categorical_low_cardinality,
            "add_missing_indicator": self.add_missing_indicator,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def _forest(self) -> RandomForestClassifier:
        return make_forest(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _importance(self, model, X, y) -> pd.Series:
        return compute_importance(
            model,
            X,
            y,
            method=self.importance,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    # ---------------------------------------------------------------------- fit
    def fit(
        self,
        X,
        y,
        *,
        splits: Splits | tuple | None = None,
        test_idx=None,
        callback: Callback | None = None,
    ) -> FastForestResult:
        """Run the full pipeline and return a :class:`~fastforest.result.FastForestResult`.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature table. If ``y`` is a column name, that column is taken from ``X``.
        y : array-like or str
            Class labels, or the name of the target column in ``X``.
        splits : Splits or tuple of three index arrays, optional
            Pre-made positional ``(train, val, test)`` indices; overrides ``split``,
            ``val_size`` and ``test_size`` so that several methods can share one split.
        test_idx : array-like of int, optional
            Test rows for ``split="index"``.
        callback : callable, optional
            Progress callback receiving dicts with ``stage``, ``message``, ``elapsed`` and, for
            search steps, ``step`` (a :class:`SearchStep`).

        Returns
        -------
        FastForestResult
        """
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def emit(stage: str, message: str) -> None:
            logger.info(message)
            if callback is not None:
                callback({"stage": stage, "message": message, "elapsed": time.perf_counter() - t0})

        def lap(name: str, start: float) -> float:
            now = time.perf_counter()
            timings[name] = now - start
            return now

        X, target = _resolve_xy(X, y)
        y_enc, classes = encode_labels(target)
        if len(classes) < 2:
            raise ValueError("the target must have at least two classes")

        # S1 ---------------------------------------------------------------------------------
        t = time.perf_counter()
        if splits is None:
            splits = make_splits(
                y_enc,
                method=self.split,
                val_size=self.val_size,
                test_size=self.test_size,
                test_idx=test_idx,
                random_state=self.random_state,
            )
        elif not isinstance(splits, Splits):
            splits = Splits.from_arrays(*splits, n_samples=len(y_enc))
        y_train, y_val, y_test = y_enc[splits.train], y_enc[splits.val], y_enc[splits.test]
        t = lap("split", t)
        sizes = splits.sizes
        emit(
            "split",
            f"Split {len(y_enc)} rows into {sizes['train']} train / {sizes['val']} validation / "
            f"{sizes['test']} test",
        )

        # S0 ---------------------------------------------------------------------------------
        preprocessor = Preprocessor(
            max_card=self.max_card,
            categorical_low_cardinality=self.categorical_low_cardinality,
            add_missing_indicator=self.add_missing_indicator,
        ).fit(X.iloc[splits.train])
        X_train = preprocessor.transform(X.iloc[splits.train])
        X_val = preprocessor.transform(X.iloc[splits.val])
        X_test = preprocessor.transform(X.iloc[splits.test])
        features_full = list(X_train.columns)
        t = lap("preprocess", t)
        emit(
            "preprocess",
            f"Preprocessed {len(features_full)} features "
            f"({len(preprocessor.categorical_columns_)} categorical)",
        )

        # S2 ---------------------------------------------------------------------------------
        model_full = self._forest().fit(X_train, y_train)
        pred_val_full = model_full.predict(X_val)
        metrics_full_val = compute_metrics(y_val, pred_val_full)
        t = lap("full_model", t)
        emit(
            "full_model",
            f"Full model: validation accuracy {metrics_full_val['accuracy']:.4f} "
            f"with {len(features_full)} features",
        )

        # S3 ---------------------------------------------------------------------------------
        imp_full = self._importance(model_full, X_val, y_val)
        t = lap("importance", t)
        emit("importance", f"Ranked features by {self.importance} importance")

        # S4 ---------------------------------------------------------------------------------
        search = quantile_threshold_search(
            X_train,
            y_train,
            X_val,
            y_val,
            imp_full,
            metrics_full_val,
            bound=self.bound,
            criterion=self.criterion,
            q_start=self.q_start,
            q_step=self.q_step,
            forest_factory=self._forest,
            callback=callback,
        )
        features_important = list(search.features)
        model_important = search.model
        t = lap("search", t)
        emit(
            "search_done",
            f"Threshold search kept {len(features_important)} features "
            f"(q = {'all positive' if search.q_final is None else f'{search.q_final:.2f}'}, "
            f"bound {'satisfied' if search.bound_satisfied else 'NOT satisfied'})",
        )

        # S5 ---------------------------------------------------------------------------------
        imp_important = self._importance(model_important, X_val[features_important], y_val)
        redundancy = find_redundant(
            X_train[features_important], imp_important, cluster_threshold=self.cluster_threshold
        )
        t = lap("redundancy", t)
        emit(
            "redundancy",
            "Redundancy step skipped (fewer than 3 features)"
            if redundancy.skipped
            else f"Redundancy: {len(redundancy.clusters)} clusters, "
            f"{len(redundancy.drop)} features dropped",
        )

        # S6 ---------------------------------------------------------------------------------
        features_final, model_final, metrics_final_val, reinserted = self._final_model(
            X_train,
            y_train,
            X_val,
            y_val,
            features_important,
            redundancy.drop,
            imp_important,
            metrics_full_val,
            model_important,
            search.metrics_val,
        )
        final_ok = within_bound(metrics_full_val, metrics_final_val, self.bound, self.criterion)
        t = lap("final_model", t)
        emit(
            "final_model",
            f"Final model: {len(features_final)} features, validation accuracy "
            f"{metrics_final_val['accuracy']:.4f}"
            + (f", {len(reinserted)} re-inserted to hold the bound" if reinserted else ""),
        )

        # Test evaluation (once per model) ---------------------------------------------------
        predictions = {
            "full": {"val": pred_val_full, "test": model_full.predict(X_test)},
            "important": {
                "val": model_important.predict(X_val[features_important]),
                "test": model_important.predict(X_test[features_important]),
            },
            "final": {
                "val": model_final.predict(X_val[features_final]),
                "test": model_final.predict(X_test[features_final]),
            },
        }
        metrics = {
            stage: {
                "val": compute_metrics(y_val, preds["val"]),
                "test": compute_metrics(y_test, preds["test"]),
            }
            for stage, preds in predictions.items()
        }
        X_final = preprocessor.transform(X, columns=features_final)
        t = lap("test_eval", t)
        timings["total"] = time.perf_counter() - t0
        emit(
            "done",
            f"Done: test accuracy {metrics['final']['test']['accuracy']:.4f} with "
            f"{len(features_final)} of {len(features_full)} features",
        )

        reinserted_set = set(reinserted)
        return FastForestResult(
            features_full=features_full,
            features_important=features_important,
            features_final=features_final,
            dropped_redundant=[f for f in redundancy.drop if f not in reinserted_set],
            reinserted=reinserted,
            clusters=redundancy.clusters,
            importances={"full": imp_full, "important": imp_important},
            metrics=metrics,
            bound_satisfied=search.bound_satisfied,
            final_bound_satisfied=final_ok,
            q_final=search.q_final,
            search_history=[dataclasses.asdict(step) for step in search.history],
            timings=timings,
            params=self.get_params(),
            classes=classes,
            splits=splits,
            y=y_enc,
            X_final=X_final,
            predictions=predictions,
            models={"full": model_full, "important": model_important, "final": model_final},
            preprocessor=preprocessor,
            redundancy=redundancy,
        )

    def _final_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        features_important: list[str],
        dropped: list[str],
        imp_important: pd.Series,
        metrics_full_val: dict[str, float],
        model_important: RandomForestClassifier,
        metrics_important_val: dict[str, float],
    ) -> tuple[list[str], RandomForestClassifier, dict[str, float], list[str]]:
        """Stage S6: retrain on the non-redundant features and enforce the bound (D6).

        Features are always kept in ``features_important`` order, so re-inserting every dropped
        feature reproduces the S4 model exactly.
        """
        if not dropped:
            return list(features_important), model_important, metrics_important_val, []

        dropped_set = set(dropped)
        features = [f for f in features_important if f not in dropped_set]

        def fit_eval(feats: list[str]):
            if feats == features_important:
                return model_important, metrics_important_val
            model = self._forest().fit(X_train[feats], y_train)
            return model, compute_metrics(y_val, model.predict(X_val[feats]))

        model, metrics = fit_eval(features)
        reinserted: list[str] = []
        if self.enforce_bound:
            position = {name: i for i, name in enumerate(features_important)}
            candidates = sorted(dropped, key=lambda n: (-float(imp_important[n]), position[n]))
            for name in candidates:
                if within_bound(metrics_full_val, metrics, self.bound, self.criterion):
                    break
                reinserted.append(name)
                keep = set(features) | {name}
                features = [f for f in features_important if f in keep]
                model, metrics = fit_eval(features)
                logger.info(
                    "bound enforcement: re-inserted %s (%d features, accuracy %.4f)",
                    name,
                    len(features),
                    metrics["accuracy"],
                )
        return features, model, metrics, reinserted
