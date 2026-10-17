"""FastForest: random-forest feature reduction for high-dimensional tabular data.

The pipeline trains a random forest on all features, searches over importance quantiles for the
smallest feature subset whose validation metrics stay within a bound of the full model, removes
redundant features by Spearman-correlation clustering, and evaluates the final model once on a
held-out test split.

Examples
--------
>>> from fastforest import FastForest, load_sample
>>> X, y = load_sample()
>>> result = FastForest(random_state=0).fit(X, y)
>>> result.features_final  # doctest: +SKIP
['...']
"""

from fastforest._version import __version__
from fastforest.datasets import load_sample, read_table
from fastforest.reduce import FastForest, make_forest, quantile_threshold_search
from fastforest.result import FastForestResult

__all__ = [
    "FastForest",
    "FastForestResult",
    "__version__",
    "load_sample",
    "make_forest",
    "quantile_threshold_search",
    "read_table",
]
