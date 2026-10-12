"""Feature-selection methods compared in the benchmark (PUBLICATION_PLAN.md §6.2).

Every method has the signature ``select(X, y, splits, *, seed, params, rf_params, n_jobs,
context)`` and returns a :class:`Selection`. Methods may use the training split and, where
the method needs it, the validation split; the test split is never touched here.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import parse_version

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


def _l1_kwargs(cv: bool) -> dict:
    """Pure-L1 penalty in the form the installed scikit-learn expects.

    ``penalty="l1"`` is deprecated since 1.8 (removed in 1.10) in favour of ``l1_ratio=1``;
    older releases only honour ``l1_ratio`` together with ``penalty="elasticnet"``, which is
    the same objective at ``l1_ratio=1``.
    """
    import sklearn

    if parse_version(sklearn.__version__) >= parse_version("1.8"):
        return {"l1_ratios": [1.0]} if cv else {"l1_ratio": 1.0}
    return {"penalty": "elasticnet", "l1_ratios": [1.0]} if cv else {"penalty": "l1"}


def _perm_cache_path(context: dict, seed: int) -> Path | None:
    cache_dir = context.get("cache_dir")
    return Path(cache_dir) / f"perm_importance_seed{seed}.csv" if cache_dir else None


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
        cache = _perm_cache_path(context, seed)
        if importance == "permutation" and cache is not None:
            # The S2 forest and its permutation importance on the validation split are exactly
            # what the perm_threshold baseline computes; share them (see perm_threshold).
            cache.parent.mkdir(parents=True, exist_ok=True)
            result.importances["full"].rename_axis("feature").to_csv(cache)
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
        {"n_tentative": int(tentative.sum()), "n_rejected": int((~support & ~tentative).sum())},
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
    """Keep features with permutation importance > 0 on the validation split.

    The S2 forest on the training split and its permutation importance (same seed, same
    ``n_repeats``) are identical to stage S2/S3 of ``fastforest_perm``; when that run wrote its
    importances to the cache they are reused instead of recomputed (results are identical,
    permutation importance is seeded per feature independently of ``n_jobs``).
    """
    cache = _perm_cache_path(context, seed)
    if cache is not None and cache.exists():
        cached = pd.read_csv(cache, index_col=0)["importance"]
        if set(cached.index) == set(X.columns):
            keep = cached.reindex(X.columns).to_numpy() > 0
            return Selection(
                X.columns[keep].tolist(), {"n_positive": int(keep.sum()), "from_cache": True}
            )
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
    return Selection(X.columns[keep].tolist(), {"n_positive": int(keep.sum()), "from_cache": False})


# ------------------------------------------------------------------- L1-logistic (LASSO)
def _one_se_choice(mean: np.ndarray, se: np.ndarray, Cs: np.ndarray) -> tuple[int, int]:
    """Indices of the CV-best C and of the strongest C within one SE of it (``Cs`` ascending)."""
    best = int(np.argmax(mean))
    candidates = np.flatnonzero(mean >= mean[best] - se[best])
    return best, int(candidates.min())


def _l1_estimator(solver: str, C: float, params: dict, seed: int, n_jobs: int | None = 1):
    kwargs = dict(
        C=C,
        solver=solver,
        max_iter=params.get("max_iter", 500),
        tol=params.get("tol", 1e-3),
        random_state=seed,
        **_l1_kwargs(cv=False),
    )
    if solver == "liblinear":  # no native multiclass since scikit-learn 1.8
        return OneVsRestClassifier(LogisticRegression(**kwargs), n_jobs=n_jobs)
    return LogisticRegression(**kwargs)


def _cv_score(solver, C, params, seed, Z, y, train_idx, val_idx) -> float:
    est = _l1_estimator(solver, C, params, seed).fit(Z[train_idx], y[train_idx])
    return float(est.score(Z[val_idx], y[val_idx]))


def _coef(model) -> np.ndarray:
    if hasattr(model, "estimators_"):
        return np.vstack([e.coef_ for e in model.estimators_])
    return model.coef_


def l1_logistic(X, y, splits, *, seed, params, rf_params, n_jobs, context) -> Selection:
    """L1-regularised logistic regression; keep features with a non-zero coefficient.

    C is chosen by stratified 5-fold CV on the training split with the one-standard-error rule
    (the strongest regularisation whose mean CV accuracy is within one SE of the best, as
    glmnet's ``lambda.1se``); ``rule: cv`` uses the CV-best C instead. ``solver`` is ``saga``
    (multinomial; ``dtype: float32`` halves its memory traffic) or ``liblinear`` (one-versus-rest,
    classes fitted in parallel).
    """
    X_train, y_train = _train(X, y, splits)
    dtype = np.float32 if params.get("dtype") == "float32" else np.float64
    Z = StandardScaler().fit_transform(X_train).astype(dtype, copy=False)
    n_folds = params.get("cv", 5)
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    solver = params.get("solver", "saga")
    Cs = np.logspace(params.get("log_c_min", -4), params.get("log_c_max", 4), params.get("Cs", 8))
    rule = params.get("rule", "1se")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if solver == "saga":
            cv_model = LogisticRegressionCV(
                Cs=Cs,
                cv=cv,
                solver="saga",
                scoring="accuracy",
                max_iter=params.get("max_iter", 500),
                tol=params.get("tol", 1e-3),
                n_jobs=n_jobs,
                random_state=seed,
                **_l1_kwargs(cv=True),
            ).fit(Z, y_train)
            scores = np.asarray(next(iter(cv_model.scores_.values())))  # identical per class
            if scores.ndim == 3:  # (n_folds, n_Cs, n_l1_ratios) when l1_ratios is given
                scores = scores[:, :, 0]
            Cs = np.asarray(cv_model.Cs_)
        else:  # one single-threaded one-versus-rest fit per (fold, C), folds x Cs in parallel
            folds = list(cv.split(Z, y_train))
            jobs = [(k, j) for k in range(len(folds)) for j in range(len(Cs))]
            results = Parallel(n_jobs=n_jobs)(
                delayed(_cv_score)(solver, float(Cs[j]), params, seed, Z, y_train, *folds[k])
                for k, j in jobs
            )
            scores = np.zeros((n_folds, len(Cs)))
            for (k, j), score in zip(jobs, results, strict=True):
                scores[k, j] = score
            cv_model = None
        mean = scores.mean(axis=0)
        se = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
        best, one_se = _one_se_choice(mean, se, Cs)
        chosen = one_se if rule == "1se" else best
        if cv_model is not None and chosen == best:
            model = cv_model  # already refitted on the full training split at the best C
        else:
            model = _l1_estimator(solver, float(Cs[chosen]), params, seed, n_jobs=n_jobs).fit(
                Z, y_train
            )
    n_convergence_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    nonzero = np.any(np.abs(_coef(model)) > 0, axis=0)
    return Selection(
        X.columns[nonzero].tolist(),
        {
            "solver": solver,
            "rule": rule,
            "C_cv_best": float(Cs[best]),
            "C_1se": float(Cs[one_se]),
            "C_used": float(Cs[chosen]),
            "cv_accuracy_best": float(mean[best]),
            "cv_accuracy_used": float(mean[chosen]),
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
