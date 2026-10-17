"""Stage S1: seeded train/validation/test splits."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_METHODS = ("stratified", "random", "index")


@dataclass(frozen=True)
class Splits:
    """Positional row indices of the three splits.

    Attributes
    ----------
    train, val, test : numpy.ndarray of int
        Sorted, disjoint positional indices.
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __iter__(self):
        yield self.train
        yield self.val
        yield self.test

    @property
    def sizes(self) -> dict[str, int]:
        """Number of rows in each split."""
        return {
            "train": int(len(self.train)),
            "val": int(len(self.val)),
            "test": int(len(self.test)),
        }

    def to_dict(self) -> dict[str, list[int]]:
        """Return the indices as plain lists (JSON-serialisable)."""
        return {"train": self.train.tolist(), "val": self.val.tolist(), "test": self.test.tolist()}

    @classmethod
    def from_arrays(cls, train, val, test, *, n_samples: int | None = None) -> Splits:
        """Build and validate a :class:`Splits` from three index arrays.

        Parameters
        ----------
        train, val, test : array-like of int
            Positional indices.
        n_samples : int, optional
            If given, indices must lie in ``[0, n_samples)``.

        Raises
        ------
        ValueError
            If a split is empty, has duplicates, overlaps another split or is out of range.
        """
        parts = [np.asarray(p, dtype=np.int64).ravel() for p in (train, val, test)]
        for name, p in zip(("train", "val", "test"), parts, strict=True):
            if p.size == 0:
                raise ValueError(f"the {name} split is empty")
            if n_samples is not None and (p.min() < 0 or p.max() >= n_samples):
                raise ValueError(f"{name} indices out of range: must lie in [0, {n_samples})")
            if len(np.unique(p)) != len(p):
                raise ValueError(f"the {name} split contains duplicate indices")
        together = np.concatenate(parts)
        if len(np.unique(together)) != len(together):
            raise ValueError("train, val and test indices overlap")
        return cls(*(np.sort(p) for p in parts))


def _stratified_or_warn(indices, y, *, test_size, random_state, stratify: bool):
    if stratify:
        try:
            return train_test_split(
                indices, test_size=test_size, stratify=y, random_state=random_state
            )
        except ValueError as err:
            warnings.warn(
                f"stratified split failed ({err}); falling back to a random split",
                stacklevel=3,
            )
    return train_test_split(indices, test_size=test_size, random_state=random_state)


def make_splits(
    y,
    *,
    method: str = "stratified",
    val_size: float = 0.2,
    test_size: float = 0.2,
    test_idx=None,
    random_state: int | None = None,
) -> Splits:
    """Split row positions into train, validation and test sets (stage S1).

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels (used for stratification and to get the number of rows).
    method : {"stratified", "random", "index"}, default "stratified"
        ``"stratified"``: class-stratified ``train/val/test`` split. ``"random"``: the same
        without stratification. ``"index"``: ``test_idx`` are the test rows and the remaining
        rows are split 75/25 into train and validation (stratified when possible).
    val_size, test_size : float, default 0.2
        Fractions of all rows (ignored for ``method="index"``, where ``test_idx`` fixes the
        test set and the remainder is split 75/25).
    test_idx : array-like of int, optional
        Positional test indices for ``method="index"``.
    random_state : int, optional
        Seed for the splitter.

    Returns
    -------
    Splits
        Sorted, disjoint positional indices.
    """
    if method not in SPLIT_METHODS:
        raise ValueError(f"method must be one of {SPLIT_METHODS}, got {method!r}")
    y = np.asarray(y)
    n = len(y)
    if n < 3:
        raise ValueError("at least 3 rows are needed to form three splits")
    indices = np.arange(n)

    if method == "index":
        if test_idx is None:
            raise ValueError("method='index' requires test_idx")
        test = np.unique(np.asarray(test_idx, dtype=np.int64).ravel())
        if test.size == 0 or test.min() < 0 or test.max() >= n:
            raise ValueError(f"test_idx must be non-empty positions in [0, {n})")
        rest = np.setdiff1d(indices, test)
        if rest.size < 2:
            raise ValueError("test_idx leaves fewer than 2 rows for training and validation")
        train, val = _stratified_or_warn(
            rest, y[rest], test_size=0.25, random_state=random_state, stratify=True
        )
        return Splits.from_arrays(train, val, test, n_samples=n)

    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size and test_size must be in (0, 1) and sum to less than 1")
    stratify = method == "stratified"
    n_test = max(1, int(round(test_size * n)))
    n_val = max(1, int(round(val_size * n)))
    if n_test + n_val >= n:
        raise ValueError("val_size and test_size leave no rows for training")
    rest, test = _stratified_or_warn(
        indices, y, test_size=n_test, random_state=random_state, stratify=stratify
    )
    train, val = _stratified_or_warn(
        rest, y[rest], test_size=n_val, random_state=random_state, stratify=stratify
    )
    return Splits.from_arrays(train, val, test, n_samples=n)
