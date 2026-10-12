"""Feature-selection methods compared in the benchmark (PUBLICATION_PLAN.md §6.2).

Every method has the signature ``select(X, y, splits, *, seed, params, rf_params, n_jobs,
context)`` and returns a :class:`Selection`. Methods may use the training split and, where
the method needs it, the validation split; the test split is never touched here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from fastforest import FastForest
from fastforest.split import Splits


@dataclass
class Selection:
    features: list[str]
    extra: dict = field(default_factory=dict)


def make_rf(rf_params: dict, seed: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(**rf_params, random_state=seed, n_jobs=n_jobs)


def _train(X: pd.DataFrame, y: np.ndarray, splits: Splits):
    return X.iloc[splits.train], y[splits.train]


def _val(X: pd.DataFrame, y: np.ndarray, splits: Splits):
    return X.iloc[splits.val], y[splits.val]


# ------------------------------------------------------------------------- FastForest
def _fastforest(importance: str):
    def select(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
        ff_params = dict(context["fastforest"])
        ff = FastForest(
            bound=ff_params["bound"],
            criterion=ff_params["criterion"],
            cluster_threshold=ff_params["cluster_threshold"],
            enforce_bound=ff_params["enforce_bound"],
            n_estimators=rf_params["n_estimators"],
            max_features=rf_params["max_features"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            importance=importance,
            n_repeats=params.get("n_repeats", 5),
            n_jobs=n_jobs,
            random_state=seed,
        )
        result = ff.fit(X, y, splits=splits)
        return Selection(
            list(result.features_final),
            {
                "n_important": result.n_features["important"],
                "q_final": result.q_final,
                "bound_satisfied": result.bound_satisfied,
                "final_bound_satisfied": result.final_bound_satisfied,
                "n_dropped_redundant": len(result.dropped_redundant),
                "n_reinserted": len(result.reinserted),
                "timings": result.timings,
                "val_accuracy_final": result.metrics["final"]["val"]["accuracy"],
            },
        )

    return select


# ------------------------------------------------------------------------------ Boruta
def boruta(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    from boruta import BorutaPy

    X_train, y_train = _train(X, y, splits)
    estimator = make_rf(rf_params, seed, n_jobs)
    selector = BorutaPy(
        estimator,
        n_estimators=rf_params["n_estimators"],  # S2 hyper-parameters, not Boruta's "auto" rule
        perc=params.get("perc", 100),
        alpha=params.get("alpha", 0.05),
        max_iter=params.get("max_iter", 100),
        random_state=seed,
        verbose=0,
    )
    selector.fit(X_train.to_numpy(), y_train)
    support = np.asarray(selector.support_, dtype=bool)
    tentative = np.asarray(selector.support_weak_, dtype=bool)
    return Selection(
        X.columns[support].tolist(),
        {
            "n_tentative": int(tentative.sum()),
            "n_iterations": int(getattr(selector, "_iter", 0) or 0),
        },
    )


# ------------------------------------------------------------------------------- RFECV
def rfecv(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    X_train, y_train = _train(X, y, splits)
    cv = StratifiedKFold(params.get("cv", 5), shuffle=True, random_state=seed)
    selector = RFECV(
        make_rf(rf_params, seed, n_jobs),
        step=params.get("step", 0.1),
        cv=cv,
        scoring="accuracy",
        min_features_to_select=1,
        n_jobs=1,
    )
    selector.fit(X_train, y_train)
    scores = selector.cv_results_.get("mean_test_score")
    return Selection(
        X.columns[selector.support_].tolist(),
        {
            "n_steps": int(len(scores)) if scores is not None else None,
            "best_cv_accuracy": float(np.max(scores)) if scores is not None else None,
        },
    )


# ---------------------------------------------------------- permutation-importance > 0
def perm_threshold(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    X_train, y_train = _train(X, y, splits)
    X_val, y_val = _val(X, y, splits)
    model = make_rf(rf_params, seed, n_jobs).fit(X_train, y_train)
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=params.get("n_repeats", 5),
        random_state=seed,
        n_jobs=n_jobs,
    )
    keep = result.importances_mean > 0
    return Selection(X.columns[keep].tolist(), {"n_positive": int(keep.sum())})


# ------------------------------------------------------------------- L1-logistic (LASSO)
def l1_logistic(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    X_train, y_train = _train(X, y, splits)
    scaler = StandardScaler().fit(X_train)
    Z = scaler.transform(X_train)
    cv = StratifiedKFold(params.get("cv", 5), shuffle=True, random_state=seed)
    model = LogisticRegressionCV(
        Cs=params.get("Cs", 8),
        cv=cv,
        penalty="l1",
        solver="saga",
        scoring="accuracy",
        max_iter=params.get("max_iter", 500),
        tol=params.get("tol", 1e-3),
        n_jobs=n_jobs,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(Z, y_train)
        # One-standard-error rule (as glmnet's lambda.1se): the strongest regularisation whose
        # mean CV accuracy is within one standard error of the best. The CV-best C on its own
        # tends to keep almost every feature.
        scores = next(iter(model.scores_.values()))  # (n_folds, n_Cs); identical per class
        mean = scores.mean(axis=0)
        se = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
        best = int(np.argmax(mean))
        candidates = np.flatnonzero(mean >= mean[best] - se[best])
        c_cv_best = float(np.atleast_1d(model.C_)[0])
        c_1se = float(model.Cs_[candidates.min()])
        if params.get("rule", "1se") == "1se" and c_1se != c_cv_best:
            model = LogisticRegression(
                C=c_1se,
                penalty="l1",
                solver="saga",
                max_iter=params.get("max_iter", 500),
                tol=params.get("tol", 1e-3),
                random_state=seed,
            ).fit(Z, y_train)
    n_convergence_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    nonzero = np.any(np.abs(model.coef_) > 0, axis=0)
    return Selection(
        X.columns[nonzero].tolist(),
        {
            "C_cv_best": c_cv_best,
            "C_1se": c_1se,
            "C_used": float(getattr(model, "C", c_cv_best)),
            "rule": params.get("rule", "1se"),
            "cv_accuracy_best": float(mean[best]),
            "n_convergence_warnings": int(n_convergence_warnings),
        },
    )


# ----------------------------------------------------------------------------- controls
def full(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    return Selection(list(X.columns), {})


def random(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    size = context.get("reference_n_features")
    if size is None:
        raise RuntimeError("the random control needs the fastforest_mdi result of the same seed")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(X.columns), size=min(size, len(X.columns)), replace=False)
    return Selection(X.columns[np.sort(chosen)].tolist(), {"reference_n_features": int(size)})


METHODS: dict[str, Callable[..., Selection]] = {
    "fastforest_mdi": _fastforest("mdi"),
    "fastforest_perm": _fastforest("permutation"),
    "boruta": boruta,
    "rfecv": rfecv,
    "perm_threshold": perm_threshold,
    "l1_logistic": l1_logistic,
    "full": full,
    "random": random,
}
