"""The benchmark runner and summariser work end to end on a tiny configuration."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

yaml = pytest.importorskip("yaml")  # the benchmark needs the `bench` extra
pytest.importorskip("boruta")
psutil = pytest.importorskip("psutil")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import main as run_main  # noqa: E402
from benchmarks.summarize import main as summarize_main  # noqa: E402
from fastforest.datasets import sample_path  # noqa: E402


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
            "fastforest_perm": {"importance": "permutation", "n_repeats": 2},
            "perm_threshold": {"n_repeats": 2},
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
    assert len(runs) == 12  # 6 methods x 2 seeds
    record = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "fastforest_mdi" / "seed0.json").read_text()
    )
    assert record["status"] == "ok"
    assert 0 < record["n_features"] < record["n_features_total"]
    run_log = tmp_path / "results" / "runs" / "sample" / "fastforest_mdi" / "seed0.log"
    assert "Threshold search kept" in run_log.read_text()  # worker output goes to its own log
    assert record["n_rows"] == 400
    assert len(record["load_avg_start"]) == 3 and len(record["load_avg_end"]) == 3
    assert set(record["metrics_test"]) == {
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    }
    perm = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "perm_threshold" / "seed0.json").read_text()
    )
    ff_perm = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "fastforest_perm" / "seed0.json").read_text()
    )
    assert perm["extra"]["from_cache"] is True
    assert perm["n_features"] == ff_perm["extra"]["n_important"]  # same positive-importance set
    random_record = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "random" / "seed0.json").read_text()
    )
    assert random_record["n_features"] == record["n_features"]

    # a second invocation skips everything
    assert run_main(["--config", str(config_path)]) == 0

    assert summarize_main(["--config", str(config_path)]) == 0
    summary_dir = tmp_path / "results" / "summary"
    summary = pd.read_csv(summary_dir / "summary.csv")
    assert set(summary["method"]) == {
        "fastforest_mdi",
        "fastforest_perm",
        "perm_threshold",
        "l1_logistic",
        "full",
        "random",
    }
    assert (summary["n_runs"] == 2).all()
    stability = pd.read_csv(summary_dir / "stability.csv")
    assert stability.loc[stability["method"] == "full", "jaccard_mean"].item() == 1.0
    datasets = pd.read_csv(summary_dir / "datasets.csv")
    assert datasets.loc[0, "rows_comparison"] == 400 and datasets.loc[0, "subsampled"]
    env = json.loads((tmp_path / "results" / "env.json").read_text())
    assert env["config_hash"] and env["versions"]["fastforest"]
    assert (summary_dir / "fig_reduction_vs_accuracy.png").exists()
    assert (summary_dir / "per_run.csv").exists()


def test_timeout_kills_the_worker_and_its_children(tmp_path):
    """Regression: a timed-out run must not leave orphaned joblib workers computing."""
    started = time.time()
    config = {
        "output_dir": str(tmp_path / "results"),
        "prepared_dir": str(tmp_path / "prepared"),
        "seeds": [0],
        "split": {"val_size": 0.2, "test_size": 0.2},
        "comparison_max_rows": 1000,
        "min_class_size": 10,
        "n_jobs": 2,
        "timeout_hours": 8 / 3600,  # 8 s: inside the permutation-importance step
        "final_model": {"n_estimators": 10, "max_features": 0.5, "min_samples_leaf": 5},
        "fastforest": {
            "bound": 0.02,
            "criterion": "accuracy",
            "cluster_threshold": 0.01,
            "enforce_bound": True,
        },
        "methods": {"fastforest_perm": {"importance": "permutation", "n_repeats": 50}},
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
    record = json.loads(
        (tmp_path / "results" / "runs" / "sample" / "fastforest_perm" / "seed0.json").read_text()
    )
    assert record["status"] == "timeout"

    time.sleep(1)
    orphans = []
    for proc in psutil.process_iter(["cmdline", "create_time", "ppid"]):
        cmdline = " ".join(proc.info["cmdline"] or [])
        if (proc.info["create_time"] or 0) >= started and proc.info["ppid"] == 1:
            if "joblib.externals.loky" in cmdline or "run_benchmark.py" in cmdline:
                orphans.append(cmdline[:80])
    assert not orphans, orphans


def test_large_run_logs_are_compressed(tmp_path):
    from benchmarks import run_benchmark as rb

    run_log = tmp_path / "seed0.log"
    run_log.write_bytes(b"x" * (rb.RUN_LOG_GZIP_BYTES + 1))
    rb._compress_if_large(run_log)
    assert not run_log.exists() and (tmp_path / "seed0.log.gz").exists()
    small = tmp_path / "seed1.log"
    small.write_text("fine")
    rb._compress_if_large(small)
    assert small.read_text() == "fine"


def test_recovery_table_precision_and_recall():
    from benchmarks.summarize import recovery_table

    cfg = {"datasets": {"toy": {"relevant_features": ["a", "b", "c", "d"]}, "other": {}}}
    runs = [
        {"dataset": "toy", "method": "m1", "seed": 0, "status": "ok", "features": ["a", "b", "x"]},
        {"dataset": "toy", "method": "m2", "seed": 0, "status": "ok", "features": []},
        {"dataset": "toy", "method": "m3", "seed": 0, "status": "error"},
        {"dataset": "other", "method": "m1", "seed": 0, "status": "ok", "features": ["a"]},
    ]
    rec = recovery_table(runs, cfg)
    assert list(rec["method"]) == ["m1", "m2"]
    first = rec.iloc[0]
    assert first["n_relevant_selected"] == 2
    assert first["precision"] == pytest.approx(2 / 3)
    assert first["recall"] == pytest.approx(0.5)
    assert pd.isna(rec.iloc[1]["precision"]) and rec.iloc[1]["recall"] == 0
