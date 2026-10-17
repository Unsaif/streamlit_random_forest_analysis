"""Stage S3: feature importance (mean decrease in impurity or permutation importance)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

IMPORTANCE_METHODS = ("mdi", "permutation")


def sort_importances(values, names) -> pd.Series:
    """Return importances as a Series sorted descending, ties broken by column order."""
    values = np.asarray(values, dtype=np.float64)
    names = list(names)
    if len(values) != len(names):
        raise ValueError("values and names must have the same length")
    order = np.lexsort((np.arange(len(values)), -values))
    return pd.Series(values[order], index=[names[i] for i in order], name="importance")


def compute_importance(
    model,
    X: pd.DataFrame,
    y=None,
    *,
    method: str = "mdi",
    n_repeats: int = 5,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> pd.Series:
    """Rank the features of a fitted forest by importance.

    Parameters
    ----------
    model : fitted estimator
        A fitted ``RandomForestClassifier`` (or any estimator exposing
        ``feature_importances_`` for ``method="mdi"``).
    X : pandas.DataFrame
        Features the model was trained on, in the same column order. For permutation
        importance this should be the *validation* split.
    y : array-like, optional
        Labels for ``X``; required for ``method="permutation"``.
    method : {"mdi", "permutation"}, default "mdi"
        Mean decrease in impurity, or permutation importance (mean over ``n_repeats``).
    n_repeats : int, default 5
        Permutations per feature (permutation importance only).
    random_state : int, optional
        Seed for the permutations.
    n_jobs : int, optional
        Parallel workers for permutation importance.

    Returns
    -------
    pandas.Series
        Importance per feature, sorted descending; ties keep the original column order.
    """
    if method not in IMPORTANCE_METHODS:
        raise ValueError(f"method must be one of {IMPORTANCE_METHODS}, got {method!r}")
    if method == "mdi":
        values = model.feature_importances_
    else:
        if y is None:
            raise ValueError("permutation importance requires y")
        result = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )
        values = result.importances_mean
    return sort_importances(values, X.columns)
