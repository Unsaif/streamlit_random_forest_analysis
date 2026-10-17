"""Container for the output of :meth:`fastforest.FastForest.fit`."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fastforest._version import __version__
from fastforest.metrics import METRIC_NAMES
from fastforest.preprocess import Preprocessor
from fastforest.redundancy import RedundancyResult
from fastforest.split import Splits

STAGES = ("full", "important", "final")
"""Model stages: all features (S2), after the threshold search (S4), after redundancy (S6)."""

SPLITS = ("train", "val", "test")


def _jsonable(obj: Any) -> Any:
    """Recursively convert NumPy / pandas / dataclass objects to JSON-serialisable types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


@dataclass
class FastForestResult:
    """Everything produced by one FastForest run.

    Attributes
    ----------
    features_full, features_important, features_final : list of str
        Feature sets after preprocessing (S0), the threshold search (S4) and redundancy removal
        plus bound enforcement (S6). ``features_important`` and ``features_final`` are in
        descending importance order.
    dropped_redundant : list of str
        Features removed in S5 that stayed removed.
    reinserted : list of str
        Redundant features put back by bound enforcement (S6), in the order re-inserted.
    clusters : list of list of str
        Redundancy clusters (kept member first).
    importances : dict of pandas.Series
        ``"full"``: importance of every feature in the full model; ``"important"``: importance
        in the S4 model (used by redundancy removal and bound enforcement).
    metrics : dict
        ``metrics[stage][split][metric]`` for stage in :data:`STAGES`, split in
        ``("train", "val", "test")`` and metric in :data:`fastforest.metrics.METRIC_NAMES`.
        Only validation metrics drive any decision; test metrics are computed once at the end.
    bound_satisfied : bool
        Whether the S4 model satisfied the bound on the validation split.
    final_bound_satisfied : bool
        Whether the final (S6) model satisfies the bound on the validation split.
    q_final : float or None
        Quantile at which the search stopped; None when every positive-importance feature was
        kept because no quantile satisfied the bound.
    search_history : list of dict
        One entry per evaluated quantile (``q``, ``threshold``, ``n_features``, ``metrics``,
        ``satisfied``, ``retrained``).
    timings : dict
        Wall-clock seconds per stage and ``"total"``.
    params : dict
        Constructor parameters of the :class:`~fastforest.FastForest` that produced the result.
    classes : list of str
        Class names; label ``k`` in ``y`` is ``classes[k]``.
    splits : Splits
        Row indices of the three splits.
    y : numpy.ndarray
        Encoded labels for every row.
    X_final : pandas.DataFrame
        Preprocessed values of the final features for every row (input for the visualisations).
    predictions : dict
        ``predictions[stage][split]`` predicted labels.
    models : dict
        Fitted ``RandomForestClassifier`` per stage.
    preprocessor : Preprocessor
        Fitted stage-S0 transformer.
    redundancy : RedundancyResult
        Full redundancy output including the linkage matrix for the dendrogram.
    """

    features_full: list[str]
    features_important: list[str]
    features_final: list[str]
    dropped_redundant: list[str]
    reinserted: list[str]
    clusters: list[list[str]]
    importances: dict[str, pd.Series] = field(repr=False)
    metrics: dict[str, dict[str, dict[str, float]]]
    bound_satisfied: bool
    final_bound_satisfied: bool
    q_final: float | None
    search_history: list[dict[str, Any]] = field(repr=False)
    timings: dict[str, float]
    params: dict[str, Any]
    classes: list[str]
    splits: Splits = field(repr=False)
    y: np.ndarray = field(repr=False)
    X_final: pd.DataFrame = field(repr=False)
    predictions: dict[str, dict[str, np.ndarray]] = field(repr=False)
    models: dict[str, Any] = field(repr=False)
    preprocessor: Preprocessor = field(repr=False)
    redundancy: RedundancyResult = field(repr=False)

    # ----------------------------------------------------------------- summaries
    @property
    def n_samples(self) -> int:
        """Number of rows."""
        return int(len(self.y))

    @property
    def n_features(self) -> dict[str, int]:
        """Feature count per stage."""
        return {
            "full": len(self.features_full),
            "important": len(self.features_important),
            "final": len(self.features_final),
        }

    @property
    def reduction(self) -> float:
        """Fraction of features removed, ``1 - n_final / n_full``."""
        return 1.0 - len(self.features_final) / len(self.features_full)

    def summary(self) -> pd.DataFrame:
        """Return a table with one row per stage and split and one column per metric."""
        rows = []
        for stage in STAGES:
            for split in SPLITS:
                row = {"stage": stage, "split": split, "n_features": self.n_features[stage]}
                row.update(self.metrics[stage][split])
                rows.append(row)
        return pd.DataFrame(rows, columns=["stage", "split", "n_features", *METRIC_NAMES])

    def importances_frame(self) -> pd.DataFrame:
        """Return every feature with its importances and selection flags."""
        full = self.importances["full"]
        important = self.importances["important"]
        selected_important = set(self.features_important)
        selected_final = set(self.features_final)
        dropped = set(self.dropped_redundant)
        frame = pd.DataFrame(
            {
                "feature": full.index,
                "rank_full": np.arange(1, len(full) + 1),
                "importance_full": full.to_numpy(),
                "importance_important": important.reindex(full.index).to_numpy(),
                "selected_important": [f in selected_important for f in full.index],
                "selected_final": [f in selected_final for f in full.index],
                "dropped_redundant": [f in dropped for f in full.index],
            }
        )
        return frame.reset_index(drop=True)

    def features_frame(self) -> pd.DataFrame:
        """Return the final features (in full-model importance order) with their importances.

        ``importance_full`` is the importance in the model on all features (the ranking the
        threshold search used); ``importance_important`` is the importance in the S4 model
        (used for redundancy removal and bound enforcement).
        """
        full = self.importances["full"]
        important = self.importances["important"]
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(self.features_final) + 1),
                "feature": self.features_final,
                "importance_full": full.reindex(self.features_final).to_numpy(),
                "importance_important": important.reindex(self.features_final).to_numpy(),
            }
        )

    # --------------------------------------------------------------- serialising
    def to_dict(self, *, include_splits: bool = False) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no fitted models or data).

        Parameters
        ----------
        include_splits : bool, default False
            Also include the row indices of the three splits.
        """
        payload: dict[str, Any] = {
            "fastforest_version": __version__,
            "params": self.params,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "reduction": self.reduction,
            "classes": self.classes,
            "split_sizes": self.splits.sizes,
            "features_important": self.features_important,
            "features_final": self.features_final,
            "dropped_redundant": self.dropped_redundant,
            "reinserted": self.reinserted,
            "clusters": self.clusters,
            "bound_satisfied": self.bound_satisfied,
            "final_bound_satisfied": self.final_bound_satisfied,
            "q_final": self.q_final,
            "metrics": self.metrics,
            "search_history": self.search_history,
            "timings": self.timings,
        }
        if include_splits:
            payload["splits"] = self.splits.to_dict()
        return _jsonable(payload)

    def to_json(
        self, path: str | Path | None = None, *, indent: int = 2, include_splits: bool = False
    ) -> str:
        """Serialise :meth:`to_dict` to JSON, optionally writing it to ``path``."""
        text = json.dumps(self.to_dict(include_splits=include_splits), indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``features.csv``, ``metrics.json``, ``importances.csv`` and ``clusters.json``.

        Parameters
        ----------
        out_dir : path-like
            Directory to create or reuse.

        Returns
        -------
        dict
            Name → path of every file written.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "features": out / "features.csv",
            "metrics": out / "metrics.json",
            "importances": out / "importances.csv",
            "clusters": out / "clusters.json",
        }
        self.features_frame().to_csv(paths["features"], index=False)
        self.to_json(paths["metrics"])
        self.importances_frame().to_csv(paths["importances"], index=False)
        clusters = {
            "threshold": self.redundancy.threshold,
            "skipped": self.redundancy.skipped,
            "clusters": self.clusters,
            "dropped_redundant": self.dropped_redundant,
            "reinserted": self.reinserted,
        }
        paths["clusters"].write_text(json.dumps(_jsonable(clusters), indent=2), encoding="utf-8")
        return paths
