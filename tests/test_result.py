"""FastForestResult summaries and serialisation."""

import json

import numpy as np
import pandas as pd

from fastforest.metrics import METRIC_NAMES
from fastforest.result import _jsonable


def test_counts_and_reduction(sample_result):
    n = sample_result.n_features
    assert n["full"] == 500 and n["important"] >= n["final"] >= 1
    assert sample_result.n_samples == 1000
    assert 0.0 < sample_result.reduction < 1.0
    assert sample_result.classes == ["Gut", "Nasal_cavity", "Skin", "Vagina"]


def test_summary_table(sample_result):
    summary = sample_result.summary()
    assert summary.shape == (9, 3 + len(METRIC_NAMES))
    assert summary["stage"].tolist()[:3] == ["full", "full", "full"]
    assert summary["split"].tolist()[:3] == ["train", "val", "test"]
    assert summary.loc[summary["stage"] == "final", "n_features"].unique().tolist() == [
        sample_result.n_features["final"]
    ]


def test_importances_frame_flags_are_consistent(sample_result):
    frame = sample_result.importances_frame()
    assert len(frame) == 500
    assert frame["rank_full"].tolist() == list(range(1, 501))
    assert frame["importance_full"].is_monotonic_decreasing
    assert frame["selected_important"].sum() == sample_result.n_features["important"]
    assert frame["selected_final"].sum() == sample_result.n_features["final"]
    assert not (frame["selected_final"] & frame["dropped_redundant"]).any()
    assert (frame["selected_final"] <= frame["selected_important"]).all()
    assert frame.loc[~frame["selected_important"], "importance_important"].isna().all()


def test_features_frame(sample_result):
    frame = sample_result.features_frame()
    assert frame.columns.tolist() == ["rank", "feature", "importance_full", "importance_important"]
    assert frame["feature"].tolist() == sample_result.features_final
    assert frame["importance_full"].notna().all()


def test_to_dict_is_json_serialisable(sample_result):
    payload = sample_result.to_dict(include_splits=True)
    text = json.dumps(payload)
    back = json.loads(text)
    assert back["n_features"] == sample_result.n_features
    assert back["classes"] == sample_result.classes
    assert set(back["metrics"]) == {"full", "important", "final"}
    assert sorted(back["splits"]) == ["test", "train", "val"]
    assert back["split_sizes"] == {"train": 600, "val": 200, "test": 200}
    assert "splits" not in sample_result.to_dict()
    assert back["search_history"][-1]["satisfied"] is True


def test_to_json_and_save(tmp_path, sample_result):
    text = sample_result.to_json(tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text()) == json.loads(text)
    paths = sample_result.save(tmp_path / "out")
    assert sorted(p.name for p in paths.values()) == [
        "clusters.json",
        "features.csv",
        "importances.csv",
        "metrics.json",
    ]
    assert all(p.exists() for p in paths.values())
    clusters = json.loads(paths["clusters"].read_text())
    assert clusters["clusters"] == sample_result.clusters
    assert len(pd.read_csv(paths["features"])) == sample_result.n_features["final"]


def test_jsonable_converts_numpy_and_pandas():
    out = _jsonable(
        {
            "a": np.int64(3),
            "b": np.float32(0.5),
            "c": np.bool_(True),
            "d": np.arange(2),
            "e": pd.Series([1.0], index=["x"]),
            "f": (1, 2),
        }
    )
    assert out == {"a": 3, "b": 0.5, "c": True, "d": [0, 1], "e": {"x": 1.0}, "f": [1, 2]}
    json.dumps(out)
