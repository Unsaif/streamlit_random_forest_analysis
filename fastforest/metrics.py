"""Classification metrics and the bound test used by the threshold search."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

METRIC_NAMES = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
"""Metrics recorded for every model and split."""

CRITERIA = {
    "accuracy": ("accuracy",),
    "all": ("accuracy", "macro_precision", "macro_recall"),
}
"""Named metric sets the bound can be applied to."""


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy, macro precision, macro recall and macro F1 on the [0, 1] scale.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted encoded labels.

    Returns
    -------
    dict
        Keys are :data:`METRIC_NAMES`. Precision and recall use ``zero_division=0``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def criterion_metrics(criterion: str | Iterable[str]) -> tuple[str, ...]:
    """Resolve a criterion name or explicit metric list to a tuple of metric names.

    Parameters
    ----------
    criterion : {"accuracy", "all"} or iterable of str
        ``"accuracy"`` uses accuracy only; ``"all"`` uses accuracy, macro precision and
        macro recall. An iterable selects metrics from :data:`METRIC_NAMES` directly.
    """
    if isinstance(criterion, str):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {tuple(CRITERIA)}, got {criterion!r}")
        return CRITERIA[criterion]
    names = tuple(criterion)
    unknown = [m for m in names if m not in METRIC_NAMES]
    if unknown or not names:
        raise ValueError(f"unknown metrics {unknown}; choose from {METRIC_NAMES}")
    return names


def metric_drops(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    """Return ``reference - candidate`` for every metric (positive means the candidate is worse)."""
    return {name: reference[name] - candidate[name] for name in METRIC_NAMES}


def within_bound(
    reference: dict[str, float],
    candidate: dict[str, float],
    bound: float,
    criterion: str | Iterable[str] = "accuracy",
) -> bool:
    """Test whether ``candidate`` is within ``bound`` of ``reference`` on the chosen metrics.

    The test is ``reference[m] - candidate[m] <= bound`` for every metric ``m`` in the
    criterion; a candidate that beats the reference always passes.
    """
    return all(reference[m] - candidate[m] <= bound for m in criterion_metrics(criterion))
