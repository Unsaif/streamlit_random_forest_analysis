"""Stage S0: column typing, imputation and encoding.

:class:`Preprocessor` is fitted on the training split only and then applied to every split, so
no statistic computed from validation or test rows reaches the model. Trees are scale-invariant,
so continuous columns are *not* scaled; histograms of the final features therefore show the raw
values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

MISSING_CATEGORY = "(missing)"
"""Category assigned to missing values of categorical columns."""

UNKNOWN_CATEGORY_CODE = -1
"""Code assigned to categories that were not seen during fitting."""


def split_columns(
    X: pd.DataFrame, *, max_card: int = 20, categorical_low_cardinality: bool = False
) -> tuple[list[str], list[str]]:
    """Split the columns of ``X`` into continuous and categorical groups.

    Numeric columns are continuous; object, string, category and boolean columns are
    categorical. Numeric columns with at most ``max_card`` distinct non-missing values are
    treated as categorical only when ``categorical_low_cardinality`` is True (the legacy fastai
    behaviour); the default keeps them continuous, which is what tree models need.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature table.
    max_card : int, default 20
        Cardinality threshold used when ``categorical_low_cardinality`` is True.
    categorical_low_cardinality : bool, default False
        Whether to apply the ``max_card`` rule to numeric columns.

    Returns
    -------
    continuous, categorical : list of str
        Column names, each list in the original column order.
    """
    continuous: list[str] = []
    categorical: list[str] = []
    for name in X.columns:
        col = X[name]
        if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
            categorical.append(name)
        elif categorical_low_cardinality and col.nunique(dropna=True) <= max_card:
            categorical.append(name)
        else:
            continuous.append(name)
    return continuous, categorical


def _check_frame(X) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if X.columns.has_duplicates:
        dupes = X.columns[X.columns.duplicated()].unique().tolist()
        raise ValueError(f"X has duplicate column names: {dupes[:5]}")
    return X


def _as_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Return categorical columns as plain strings with an explicit missing category."""
    out = frame.astype(object)
    out = out.where(out.notna(), MISSING_CATEGORY)
    return out.astype(str)


def _impute(col: pd.Series, fill_value: float) -> np.ndarray:
    """Return ``col`` as a NumPy array with missing values replaced by ``fill_value``."""
    if isinstance(col.dtype, pd.api.extensions.ExtensionDtype):
        return col.astype("float64").fillna(fill_value).to_numpy()
    arr = col.to_numpy()
    if np.issubdtype(arr.dtype, np.floating):
        mask = np.isnan(arr)
        if mask.any():
            arr = np.where(mask, fill_value, arr)
    return arr


class Preprocessor:
    """Impute and encode a feature table for tree models (stage S0).

    Parameters
    ----------
    max_card : int, default 20
        See :func:`split_columns`.
    categorical_low_cardinality : bool, default False
        See :func:`split_columns`.
    add_missing_indicator : bool, default False
        If True, every continuous column that has missing values in the training data gets an
        extra ``<name>_na`` indicator column (1 where the value was missing). The legacy fastai
        pipeline always did this, which allowed indicator columns to be "selected" as features.

    Attributes
    ----------
    feature_names_in_ : list of str
        Input columns, in order.
    feature_names_out_ : list of str
        Output columns: the input columns in order, followed by any indicator columns.
    continuous_columns_, categorical_columns_ : list of str
        Column typing decided on the training data.
    medians_ : dict
        Training median per continuous column (0.0 when a column is entirely missing).
    categories_ : dict
        Sorted category list per categorical column; codes are positions in that list.
    columns_with_missing_ : list of str
        Continuous columns that had missing values in the training data.

    Notes
    -----
    Continuous columns keep their dtype when nothing has to be imputed, so integer presence
    tables stay integer and large inputs are not up-cast to float64. Categorical codes are
    ``int64``; unseen categories become ``-1`` and missing values form their own category.
    """

    def __init__(
        self,
        *,
        max_card: int = 20,
        categorical_low_cardinality: bool = False,
        add_missing_indicator: bool = False,
    ) -> None:
        self.max_card = max_card
        self.categorical_low_cardinality = categorical_low_cardinality
        self.add_missing_indicator = add_missing_indicator

    # ------------------------------------------------------------------ fitting
    def fit(self, X: pd.DataFrame) -> Preprocessor:
        """Learn column types, medians and category lists from the training table.

        Parameters
        ----------
        X : pandas.DataFrame
            Training features (no target column).

        Returns
        -------
        Preprocessor
            ``self``.
        """
        X = _check_frame(X)
        continuous, categorical = split_columns(
            X,
            max_card=self.max_card,
            categorical_low_cardinality=self.categorical_low_cardinality,
        )
        self.feature_names_in_ = list(X.columns)
        self.continuous_columns_ = continuous
        self.categorical_columns_ = categorical

        if continuous:
            numeric = X if len(continuous) == len(X.columns) else X[continuous]
            medians = numeric.median(axis=0)
            has_missing = numeric.isna().any(axis=0)
            self.medians_ = {
                name: (0.0 if pd.isna(medians[name]) else float(medians[name]))
                for name in continuous
            }
            self.columns_with_missing_ = [name for name in continuous if bool(has_missing[name])]
        else:
            self.medians_ = {}
            self.columns_with_missing_ = []

        self.missing_indicator_columns_ = (
            [f"{name}_na" for name in self.columns_with_missing_]
            if self.add_missing_indicator
            else []
        )

        if categorical:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=UNKNOWN_CATEGORY_CODE,
                dtype=np.int64,
            )
            encoder.fit(_as_strings(X[categorical]))
            self.categories_ = {
                name: list(cats)
                for name, cats in zip(categorical, encoder.categories_, strict=True)
            }
        else:
            self.categories_ = {}

        self.feature_names_out_ = self.feature_names_in_ + self.missing_indicator_columns_
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("Preprocessor is not fitted; call fit() first")

    # -------------------------------------------------------------- transforming
    def transform(self, X: pd.DataFrame, *, columns: list[str] | None = None) -> pd.DataFrame:
        """Apply the fitted imputation and encoding.

        Parameters
        ----------
        X : pandas.DataFrame
            Table with (at least) the input columns needed for ``columns``.
        columns : list of str, optional
            Output columns to produce (a subset of ``feature_names_out_``). Default: all. Use
            this to transform only the selected features of a large table without materialising
            every column.

        Returns
        -------
        pandas.DataFrame
            Numeric table with the requested columns, same index as ``X``.
        """
        self._check_fitted()
        X = _check_frame(X)
        requested = list(self.feature_names_out_) if columns is None else list(columns)
        out_set = set(self.feature_names_out_)
        unknown = [c for c in requested if c not in out_set]
        if unknown:
            raise ValueError(f"unknown output columns requested: {unknown[:5]}")

        indicator_source = dict(
            zip(self.missing_indicator_columns_, self.columns_with_missing_, strict=True)
        )
        needed = {indicator_source.get(c, c) for c in requested}
        missing = [c for c in self.feature_names_in_ if c in needed and c not in X.columns]
        if missing:
            raise ValueError(f"columns missing from input: {missing[:5]}")

        categorical = set(self.categorical_columns_)
        parts: dict[str, np.ndarray] = {}
        for name in requested:
            if name in indicator_source:
                parts[name] = X[indicator_source[name]].isna().to_numpy().astype(np.int64)
            elif name in categorical:
                values = _as_strings(X[[name]])[name]
                codes = pd.Categorical(values, categories=self.categories_[name]).codes
                parts[name] = np.asarray(codes, dtype=np.int64)
            else:
                parts[name] = _impute(X[name], self.medians_[name])
        return pd.DataFrame(parts, index=X.index, columns=requested)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``X`` and return the transformed table."""
        return self.fit(X).transform(X)


def encode_labels(y) -> tuple[np.ndarray, list[str]]:
    """Encode class labels as integers ``0 … K-1`` in sorted class order.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels. Missing values are not allowed.

    Returns
    -------
    codes : numpy.ndarray of int64
        Encoded labels.
    classes : list of str
        Class names in sorted order; ``classes[k]`` is the label encoded as ``k``.
    """
    series = pd.Series(y).reset_index(drop=True)
    if series.isna().any():
        raise ValueError("the target column contains missing values")
    encoder = LabelEncoder()
    try:
        codes = encoder.fit_transform(series.to_numpy())
    except TypeError:  # mixed types cannot be sorted; fall back to strings
        codes = encoder.fit_transform(series.astype(str).to_numpy())
    classes = [str(c) for c in encoder.classes_]
    return np.asarray(codes, dtype=np.int64), classes
