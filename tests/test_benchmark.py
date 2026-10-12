"""The benchmark runner and summariser work end to end on a tiny configuration."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import main as run_main  # noqa: E402
from benchmarks.summarize import main as summarize_main  # noqa: E402
from fastforest.datasets import sample_path  # noqa: E402

pytest.importorskip("boruta")


def test_benchmark_end_to_end(tmp_path):
    config = {
        "output_dir": str(tmp_path / "results"),
        "prepared_dir": str(tmp_path / "prepared"),
        "seeds": [0, 1],
        "split": {"val_size": 0.2, "test_size": 0.2},
        "comparison_max_rows": 400,
        "min_class_size": 10,
        "n_jobs": 2,
        "timeout_hours": 1,
        "final_model": {"n_estimators": 10, "max_features": 0.5, "min_samples_leaf": 5},
        "fastforest": {
            "bound": 0.02,
            "criterion": "accuracy",
            "cluster_threshold": 0.01,
            "enforce_bound": True,
        },
        "methods": {
            "fastforest_mdi": {"importance": "mdi"},
            "l1_logistic": {"cv": 3, "Cs": 4, "max_iter": 100},
            "full": {},
            "random": {},
        },
        "datasets": {
            "sample": {
                "name": "bundled sample",
                "source": "test",
                "kind": "csv",
                "path": str(sample_path()),
                "target": "Body site",
                "feature_type": "continuous",
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    assert run_main(["--config", str(config_path)]) == 0
    runs = sorted((tmp_path / "results" / "runs").glob("*/*/seed*.json"))
    assert len(runs) == 8  # 4 methods x 2 seeds
    record = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "fastforest_mdi" / "seed0.json").read_text()
    )
    assert record["status"] == "ok"
    assert 0 < record["n_features"] < record["n_features_total"]
    assert record["n_rows"] == 400
    assert set(record["metrics_test"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }
    random_record = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "random" / "seed0.json").read_text()
    )
    assert random_record["n_features"] == record["n_features"]

    # a second invocation skips everything
    assert run_main(["--config", str(config_path)]) == 0

    assert summarize_main(["--config", str(config_path)]) == 0
    summary_dir = tmp_path / "results" / "summary"
    summary = pd.read_csv(summary_dir / "summary.csv")
    assert set(summary["method"]) == {"fastforest_mdi", "l1_logistic", "full", "random"}
    assert (summary["n_runs"] == 2).all()
    stability = pd.read_csv(summary_dir / "stability.csv")
    assert stability.loc[stability["method"] == "full", "jaccard_mean"].item() == 1.0
    datasets = pd.read_csv(summary_dir / "datasets.csv")
    assert datasets.loc[0, "rows_comparison"] == 400 and datasets.loc[0, "subsampled"]
    env = json.loads((tmp_path / "results" / "env.json").read_text())
    assert env["config_hash"] and env["versions"]["fastforest"]
    assert (summary_dir / "fig_reduction_vs_accuracy.png").exists()
    assert (summary_dir / "per_run.csv").exists()
