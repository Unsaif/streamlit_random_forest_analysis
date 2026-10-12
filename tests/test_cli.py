"""The ``fastforest reduce`` command."""

import json

import pytest

from fastforest import __version__
from fastforest.cli import environment_info, main
from fastforest.datasets import sample_path
from tests.conftest import synthetic


def test_reduce_writes_outputs_and_figures(tmp_path, capsys):
    out = tmp_path / "out"
    code = main(
        [
            "reduce",
            str(sample_path()),
            "--target",
            "Body site",
            "--out",
            str(out),
            "--n-estimators",
            "20",
            "--no-umap",
            "--quiet",
        ]
    )
    assert code == 0
    for name in ("features.csv", "metrics.json", "importances.csv", "clusters.json"):
        assert (out / name).exists()
    payload = json.loads((out / "metrics.json").read_text())
    assert payload["environment"]["fastforest"] == __version__
    assert "reduce" in payload["command"]
    assert payload["input"]["n_rows"] == 1000
    assert payload["n_features"]["final"] >= 1
    assert "splits" not in payload
    figures = sorted(p.name for p in (out / "figures").iterdir())
    assert "dendrogram.png" in figures
    assert "lda_2d.html" in figures and "lda_3d.html" in figures
    assert "importance_top30.html" in figures and "search_history.html" in figures
    assert sum(f.startswith("heatmap_") for f in figures) == 4
    assert sum(f.startswith("histograms_") for f in figures) == 4
    assert sum(f.startswith("confusion_") for f in figures) == 6
    assert not any(f.startswith("umap") for f in figures)
    assert "FastForest kept" in capsys.readouterr().out


def test_reduce_without_figures_with_index_col_and_splits(tmp_path):
    X, y, _ = synthetic(n_samples=200, n_features=15, n_informative=3, n_redundant=0)
    table = tmp_path / "data.tsv"
    X.assign(label=y).to_csv(table, sep="\t", index=True, index_label="id")
    out = tmp_path / "out"
    code = main(
        [
            "reduce",
            str(table),
            "--target",
            "label",
            "--out",
            str(out),
            "--n-estimators",
            "10",
            "--no-figures",
            "--include-splits",
            "--index-col",
            "0",
            "--split",
            "random",
            "--method",
            "all",
            "--bound",
            "0.05",
            "--no-enforce-bound",
            "--quiet",
        ]
    )
    assert code == 0
    assert not (out / "figures").exists()
    payload = json.loads((out / "metrics.json").read_text())
    assert payload["n_features"]["full"] == 15
    assert sorted(payload["splits"]) == ["test", "train", "val"]
    assert payload["params"]["criterion"] == "all"
    assert payload["params"]["enforce_bound"] is False


def test_missing_target_column_returns_2(tmp_path):
    code = main(
        ["reduce", str(sample_path()), "--target", "nope", "--out", str(tmp_path), "--quiet"]
    )
    assert code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_environment_info_keys():
    info = environment_info()
    assert {"fastforest", "python", "platform", "numpy", "pandas", "scikit-learn"} <= set(info)
