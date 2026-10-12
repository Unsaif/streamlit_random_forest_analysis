"""Stage S5: redundancy removal by Spearman-correlation clustering.

Pairwise Spearman correlation between the retained features is turned into the distance
``d = 1 - rho``, clustered with average linkage, and cut at ``cluster_threshold`` (default 0.01,
i.e. ``rho >= 0.99``). In every cluster the feature with the highest importance is kept.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy as hc
from scipy.spatial.distance import squareform
from scipy.stats import ConstantInputWarning, spearmanr

MIN_FEATURES = 3
"""Below this many features the redundancy step is skipped (bugs B4/B5)."""


@dataclass
class RedundancyResult:
    """Outcome of :func:`find_redundant`.

    Attributes
    ----------
    labels : list of str
        Features that were examined, in input order.
    clusters : list of list of str
        Clusters with two or more members, each ordered by importance (kept feature first).
    keep, drop : list of str
        Features kept and dropped.
    threshold : float
        Distance at which the dendrogram was cut.
    linkage : numpy.ndarray or None
        SciPy linkage matrix (None when the step was skipped).
    distance : pandas.DataFrame or None
        Pairwise distance matrix ``1 - rho`` (None when the step was skipped).
    """

    labels: list[str]
    clusters: list[list[str]]
    keep: list[str]
    drop: list[str]
    threshold: float
    linkage: np.ndarray | None = field(default=None, repr=False)
    distance: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def skipped(self) -> bool:
        """True when fewer than :data:`MIN_FEATURES` features were available."""
        return self.linkage is None


def spearman_distance(X: pd.DataFrame) -> pd.DataFrame:
    """Return the pairwise distance ``1 - rho`` between the columns of ``X``.

    NaN correlations (constant columns) are treated as zero correlation, the matrix is
    symmetrised, distances are clipped to ``[0, 2]`` and the diagonal is set to zero. A perfect
    negative correlation is distance 2, so anti-correlated features are not merged.
    """
    values = X.to_numpy(dtype=np.float64)
    p = values.shape[1]
    if p == 1:
        rho = np.ones((1, 1))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            stat = spearmanr(values, axis=0).statistic
        if p == 2:  # scipy returns a scalar for two columns
            r = float(np.nan_to_num(stat, nan=0.0))
            rho = np.array([[1.0, r], [r, 1.0]])
        else:
            rho = np.asarray(stat, dtype=np.float64)
    rho = np.nan_to_num(rho, nan=0.0)
    rho = (rho + rho.T) / 2.0
    dist = np.clip(1.0 - rho, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=X.columns, columns=X.columns)


def find_redundant(
    X: pd.DataFrame,
    importances: pd.Series,
    *,
    cluster_threshold: float = 0.01,
    linkage_method: str = "average",
) -> RedundancyResult:
    """Cluster correlated features and keep the most important member of each cluster.

    Parameters
    ----------
    X : pandas.DataFrame
        Training rows of the retained features (string column names).
    importances : pandas.Series
        Importance per feature from the model trained on these features (stage S4). Every
        column of ``X`` must be present.
    cluster_threshold : float, default 0.01
        Cut height on the ``1 - rho`` scale; 0.01 merges features with ``rho >= 0.99``.
    linkage_method : str, default "average"
        Linkage passed to :func:`scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    RedundancyResult
        Clusters, kept and dropped features, and the linkage for plotting. When ``X`` has fewer
        than :data:`MIN_FEATURES` columns nothing is dropped and ``linkage`` is None.
    """
    labels = [str(c) for c in X.columns]
    if len(labels) < MIN_FEATURES:
        return RedundancyResult(labels, [], list(labels), [], float(cluster_threshold))

    imp = pd.Series(importances).reindex(X.columns)
    if imp.isna().any():
        missing = imp.index[imp.isna()].tolist()
        raise ValueError(f"importances missing for columns {missing[:5]}")

    distance = spearman_distance(X)
    linkage = hc.linkage(squareform(distance.to_numpy(), checks=False), method=linkage_method)
    assignments = hc.fcluster(linkage, t=cluster_threshold, criterion="distance")

    position = {name: i for i, name in enumerate(labels)}
    groups: dict[int, list[str]] = {}
    for name, cluster_id in zip(labels, assignments, strict=True):
        groups.setdefault(int(cluster_id), []).append(name)

    clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda n: (-float(imp[n]), position[n]))
        clusters.append(members)
    clusters.sort(key=lambda members: position[members[0]])

    drop = [name for members in clusters for name in members[1:]]
    dropped = set(drop)
    keep = [name for name in labels if name not in dropped]
    return RedundancyResult(
        labels, clusters, keep, drop, float(cluster_threshold), linkage, distance
    )
