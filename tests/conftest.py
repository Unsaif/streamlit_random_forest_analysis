"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fastforest import FastForest, load_sample


def synthetic(
    *,
    n_samples: int = 1000,
    n_features: int = 200,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_classes: int = 3,
    random_state: int = 0,
    **kwargs,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, list[str]]]:
    """``make_classification`` with named columns: ``inf*``, ``red*`` then ``noise*``."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        shuffle=False,
        random_state=random_state,
        **kwargs,
    )
    n_noise = n_features - n_informative - n_redundant
    groups = {
        "informative": [f"inf{i}" for i in range(n_informative)],
        "redundant": [f"red{i}" for i in range(n_redundant)],
        "noise": [f"noise{i}" for i in range(n_noise)],
    }
    columns = groups["informative"] + groups["redundant"] + groups["noise"]
    return pd.DataFrame(X, columns=columns), y, groups


@pytest.fixture(scope="session")
def sample_result():
    """A small, fast FastForest run on the bundled sample, shared across tests."""
    X, y = load_sample()
    return FastForest(n_estimators=20, random_state=0).fit(X, y)


@pytest.fixture(scope="session")
def small_synthetic():
    X, y, groups = synthetic(n_samples=300, n_features=30, n_informative=5, n_redundant=0)
    return X, y, groups
