"""Turn benchmark run files into the tables and figures of the paper (PUBLICATION_PLAN.md §6.4–6.5).

    python benchmarks/summarize.py --config benchmarks/config.yaml

Writes under ``<output_dir>/summary/``: ``per_run.csv`` (one row per dataset x method x seed),
``summary.csv`` (mean ± SD per dataset x method), ``stability.csv`` (mean pairwise Jaccard of the
selected sets across seeds), ``datasets.csv`` (table T1), ``scalability.csv`` (table T3),
``env.json`` and the figures F3 (reduction vs test accuracy) and F4 (stability).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.run_benchmark import load_config  # noqa: E402

METHOD_LABELS = {
    "fastforest_mdi": "FastForest (MDI)",
    "fastforest_perm": "FastForest (permutation)",
    "boruta": "Boruta",
    "rfecv": "RFECV",
    "perm_threshold": "Permutation > 0",
    "l1_logistic": "L1 logistic",
    "full": "No selection",
    "random": "Random subset",
}
METHOD_ORDER = list(METHOD_LABELS)


def environment(cfg: dict) -> dict:
    import scipy
    import sklearn

    import fastforest

    chip = platform.processor()
    if platform.system() == "Darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    try:
        import psutil

        ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:  # pragma: no cover
        ram_gb = None
    try:
        import boruta

        boruta_version = getattr(boruta, "__version__", "installed")
    except ImportError:  # pragma: no cover
        boruta_version = None
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        "config_path": cfg["_config_path"],
        "config_hash": cfg["_config_hash"],
        "seeds": cfg["seeds"],
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "chip": chip,
            "cpu_count": os.cpu_count(),
            "ram_gb": ram_gb,
        },
        "versions": {
            "python": platform.python_version(),
            "fastforest": fastforest.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "boruta": boruta_version,
        },
    }


def collect_runs(output_dir: Path, *, scalability: bool = False) -> list[dict]:
    sub = "runs_scalability" if scalability else "runs"
    return [json.loads(p.read_text()) for p in sorted((output_dir / sub).glob("*/*/seed*.json"))]


def per_run_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {
            "dataset": r["dataset"],
            "method": r["method"],
            "seed": r["seed"],
            "status": r.get("status"),
            "n_rows": r.get("n_rows"),
            "n_features_total": r.get("n_features_total"),
            "n_classes": r.get("n_classes"),
            "n_features": r.get("n_features"),
            "reduction": r.get("reduction"),
            "select_seconds": r.get("select_seconds"),
            "eval_seconds": r.get("eval_seconds"),
            "peak_rss_mb": r.get("peak_rss_mb"),
        }
        for split in ("test", "val"):
            for k, v in (r.get(f"metrics_{split}") or {}).items():
                row[f"{split}_{k}"] = v
        extra = r.get("extra") or {}
        for key in (
            "q_final",
            "bound_satisfied",
            "n_important",
            "n_tentative",
            "n_convergence_warnings",
        ):
            if key in extra:
                row[key] = extra[key]
        rows.append(row)
    # perm_threshold reuses the forest and permutation importances computed by fastforest_perm
    # (stages S2 + S3, identical by construction), so its own wall-clock time is ~0 s; charge it
    # the time of that shared computation instead.
    shared = {
        (r["dataset"], r["seed"]): sum(
            (r.get("extra") or {}).get("timings", {}).get(k, 0.0)
            for k in ("full_model", "importance")
        )
        for r in runs
        if r["method"] == "fastforest_perm" and r.get("status") == "ok"
    }
    for row, r in zip(rows, runs, strict=True):
        if r["method"] == "perm_threshold" and (r.get("extra") or {}).get("from_cache"):
            key = (r["dataset"], r["seed"])
            if key in shared:
                row["select_seconds"] = shared[key]
                row["select_seconds_from"] = "fastforest_perm S2+S3"
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["method"] = pd.Categorical(
            frame["method"], METHOD_ORDER + sorted(set(frame["method"]) - set(METHOD_ORDER))
        )
        frame = frame.sort_values(["dataset", "method", "seed"]).reset_index(drop=True)
    return frame


def mean_sd(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    groups = frame.groupby(["dataset", "method"], observed=True)
    out = groups.size().rename("n_runs").to_frame()
    for col in columns:
        if col in frame:
            out[f"{col}_mean"] = groups[col].mean()
            out[f"{col}_sd"] = groups[col].std(ddof=1)
    return out.reset_index()


def stability_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    sets: dict[tuple[str, str], list[set[str]]] = {}
    for r in runs:
        if r.get("status") == "ok" and r.get("features") is not None:
            sets.setdefault((r["dataset"], r["method"]), []).append(set(r["features"]))
    for (dataset, method), selected in sets.items():
        pairs = list(itertools.combinations(selected, 2))
        jaccard = [(len(a & b) / len(a | b)) if (a | b) else 1.0 for a, b in pairs]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_runs": len(selected),
                "jaccard_mean": float(np.mean(jaccard)) if jaccard else np.nan,
                "jaccard_sd": float(np.std(jaccard, ddof=1)) if len(jaccard) > 1 else np.nan,
                "n_features_union": len(set().union(*selected)),
                "n_features_core": len(set.intersection(*selected)),
            }
        )
    return pd.DataFrame(rows)


def datasets_table(cfg: dict, prepared_dir: Path) -> pd.DataFrame:
    rows = []
    for name in cfg["datasets"]:
        meta_path = prepared_dir / f"{name}.json"
        if not meta_path.exists():
            continue
        m = json.loads(meta_path.read_text())
        rows.append(
            {
                "dataset": name,
                "name": m["name"],
                "source": m["source"],
                "target": m["target"],
                "rows_full": m["n_rows_full"],
                "rows_comparison": m["n_rows_comparison"],
                "features": m["n_features"],
                "classes_full": m["n_classes_full"],
                "classes_comparison": m["n_classes_comparison"],
                "feature_type": m["feature_type"],
                "subsampled": m["subsampled"],
            }
        )
    return pd.DataFrame(rows)


def scalability_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        extra = r.get("extra") or {}
        rows.append(
            {
                "dataset": r["dataset"],
                "method": r["method"],
                "seed": r["seed"],
                "status": r.get("status"),
                "rows": r.get("n_rows"),
                "features": r.get("n_features_total"),
                "classes": r.get("n_classes"),
                "features_final": r.get("n_features"),
                "reduction": r.get("reduction"),
                "select_seconds": r.get("select_seconds"),
                "peak_rss_mb": r.get("peak_rss_mb"),
                "test_accuracy": (r.get("metrics_test") or {}).get("accuracy"),
                "test_macro_f1": (r.get("metrics_test") or {}).get("macro_f1"),
                **{f"t_{k}": v for k, v in (extra.get("timings") or {}).items()},
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ figures
def _dataset_order(present, cfg: dict | None) -> list[str]:
    """Datasets in config order (falls back to first appearance)."""
    present = list(dict.fromkeys(present))
    if cfg:
        ordered = [d for d in cfg["datasets"] if d in present]
        # multi-seed tables first (config order within each group), single-seed tables after
        n_seeds = {d: len(cfg["datasets"][d].get("seeds", cfg["seeds"])) for d in ordered}
        ordered.sort(key=lambda d: -n_seeds[d])
        return ordered + [d for d in present if d not in ordered]
    return present


def _dataset_label(name: str, cfg: dict | None) -> str:
    if cfg and name in cfg["datasets"]:
        return cfg["datasets"][name].get("name", name)
    return name


def figure_reduction_vs_accuracy(per_run: pd.DataFrame, cfg: dict | None = None) -> Figure:
    from matplotlib.ticker import LogLocator, NullFormatter

    ok = per_run[per_run["status"] == "ok"]
    datasets = _dataset_order(ok["dataset"], cfg)
    ncols = min(3, max(1, len(datasets)))
    nrows = int(np.ceil(len(datasets) / ncols))
    fig = Figure(figsize=(4.2 * ncols, 3.6 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    markers = dict(zip(METHOD_ORDER, "o s ^ v D P X *".split(), strict=False))
    for i, dataset in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = ok[ok["dataset"] == dataset]
        for method, grp in sub.groupby("method", observed=True):
            acc = 100 * grp["test_accuracy"]
            ax.errorbar(
                grp["n_features"].mean(),
                acc.mean(),
                xerr=grp["n_features"].std(ddof=1) if len(grp) > 1 else None,
                yerr=acc.std(ddof=1) if len(grp) > 1 else None,
                fmt=markers.get(str(method), "o"),
                capsize=2,
                label=METHOD_LABELS.get(str(method), str(method)),
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("features retained (log scale)")
        ax.set_ylabel("test accuracy (%)")
        n_seeds = sub["seed"].nunique()
        ax.set_title(
            f"{_dataset_label(dataset, cfg)}\n({n_seeds} seed{'s' if n_seeds > 1 else ''})",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    for j in range(len(datasets), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def figure_stability(stability: pd.DataFrame, cfg: dict | None = None) -> Figure:
    stability = stability[stability["n_runs"] > 1]
    datasets = _dataset_order(stability["dataset"], cfg)
    # "no selection" is 1 by definition; leave it out
    methods = [m for m in METHOD_ORDER if m in set(stability["method"]) and m != "full"]
    fig = Figure(figsize=(1.6 * len(datasets) + 4, 3.8))
    ax = fig.add_subplot(111)
    width = 0.8 / max(1, len(methods))
    x = np.arange(len(datasets))
    for k, method in enumerate(methods):
        sub = stability[stability["method"] == method].set_index("dataset").reindex(datasets)
        ax.bar(
            x + (k - len(methods) / 2 + 0.5) * width,
            sub["jaccard_mean"],
            width,
            yerr=sub["jaccard_sd"],
            capsize=2,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_dataset_label(d, cfg) for d in datasets], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("mean pairwise Jaccard")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks" / "config.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    output_dir = ROOT / cfg["output_dir"]
    prepared_dir = ROOT / cfg["prepared_dir"]
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(output_dir)
    per_run = per_run_table(runs)
    per_run.to_csv(summary_dir / "per_run.csv", index=False)
    if not per_run.empty:
        ok = per_run[per_run["status"] == "ok"]
        summary = mean_sd(
            ok,
            [
                "n_features",
                "reduction",
                "test_accuracy",
                "test_macro_f1",
                "test_macro_precision",
                "test_macro_recall",
                "select_seconds",
                "peak_rss_mb",
            ],
        )
        failures = (
            per_run[per_run["status"] != "ok"].groupby(["dataset", "method"], observed=True).size()
        )
        summary["n_failed"] = [
            failures.get((d, m), 0)
            for d, m in zip(summary["dataset"], summary["method"], strict=True)
        ]
        summary.to_csv(summary_dir / "summary.csv", index=False)
        stability = stability_table(runs)
        stability.to_csv(summary_dir / "stability.csv", index=False)
        fig = figure_reduction_vs_accuracy(ok, cfg)
        fig.savefig(summary_dir / "fig_reduction_vs_accuracy.png", dpi=200, bbox_inches="tight")
        fig.savefig(summary_dir / "fig_reduction_vs_accuracy.pdf", bbox_inches="tight")
        if not stability[stability["n_runs"] > 1].empty:
            figure_stability(stability, cfg).savefig(
                summary_dir / "fig_stability.png", dpi=200, bbox_inches="tight"
            )
    datasets_table(cfg, prepared_dir).to_csv(summary_dir / "datasets.csv", index=False)
    scal = collect_runs(output_dir, scalability=True)
    if scal:
        scalability_table(scal).to_csv(summary_dir / "scalability.csv", index=False)
    (output_dir / "env.json").write_text(json.dumps(environment(cfg), indent=2))
    print(f"{len(runs)} comparison runs, {len(scal)} scalability runs -> {summary_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
