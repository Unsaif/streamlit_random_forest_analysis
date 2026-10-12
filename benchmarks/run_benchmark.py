"""Run the FastForest benchmark (PUBLICATION_PLAN.md §6.3).

    python benchmarks/run_benchmark.py --config benchmarks/config.yaml
    python benchmarks/run_benchmark.py --config benchmarks/config.yaml --datasets body_site \
        --seeds 0 1
    python benchmarks/run_benchmark.py --config benchmarks/config.yaml --scalability

Every (dataset, method, seed) run executes in its own subprocess so that its peak resident
memory can be measured and a failure cannot take the whole benchmark down. Results are written
to ``<output_dir>/runs/<dataset>/<method>/seed<k>.json`` and existing results are skipped, so
an interrupted benchmark resumes where it stopped. ``benchmarks/summarize.py`` turns the run
files into the tables and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.datasets import load_prepared, prepare_dataset, prepared_paths  # noqa: E402
from benchmarks.methods import METHODS, make_rf  # noqa: E402

log = logging.getLogger("benchmark")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return cfg


def run_path(output_dir: Path, dataset: str, method: str, seed: int, *, scalability: bool) -> Path:
    sub = "runs_scalability" if scalability else "runs"
    return output_dir / sub / dataset / method / f"seed{seed}.json"


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if platform.system() == "Darwin" else rss / 1e3


# ------------------------------------------------------------------------------ worker
def worker(spec_path: Path) -> int:
    """Run one (dataset, method, seed) and write its result file."""
    from sklearn.preprocessing import LabelEncoder

    from fastforest.metrics import compute_metrics
    from fastforest.split import make_splits

    spec = json.loads(spec_path.read_text())
    out = Path(spec["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    X, y_raw = load_prepared(Path(spec["data"]))
    encoder = LabelEncoder().fit(y_raw)
    y = encoder.transform(y_raw)
    splits = make_splits(
        y,
        method="stratified",
        val_size=spec["split"]["val_size"],
        test_size=spec["split"]["test_size"],
        random_state=spec["seed"],
    )
    load_seconds = time.perf_counter() - t_load

    method = METHODS[spec["method"]]
    context = {
        "fastforest": spec["fastforest"],
        "reference_n_features": spec.get("reference_n_features"),
    }
    t0 = time.perf_counter()
    status = "ok"
    error = None
    try:
        selection = method(
            X,
            y,
            splits,
            seed=spec["seed"],
            params=spec["params"],
            rf_params=spec["rf_params"],
            n_jobs=spec["n_jobs"],
            context=context,
        )
    except Exception as exc:
        log.exception("%s failed", spec["method"])
        status, error, selection = "error", f"{type(exc).__name__}: {exc}", None
    select_seconds = time.perf_counter() - t0

    record = {
        "dataset": spec["dataset"],
        "method": spec["method"],
        "seed": spec["seed"],
        "scalability": spec["scalability"],
        "status": status,
        "error": error,
        "n_rows": int(len(y)),
        "n_features_total": int(X.shape[1]),
        "n_classes": int(len(encoder.classes_)),
        "split_sizes": splits.sizes,
        "load_seconds": load_seconds,
        "select_seconds": select_seconds,
        "params": spec["params"],
    }
    if selection is not None:
        features = [c for c in X.columns if c in set(selection.features)]  # dataset column order
        t1 = time.perf_counter()
        X_train, y_train = X.iloc[splits.train], y[splits.train]
        X_test, y_test = X.iloc[splits.test], y[splits.test]
        X_val, y_val = X.iloc[splits.val], y[splits.val]
        if features:
            model = make_rf(spec["rf_params"], spec["seed"], spec["n_jobs"])
            model.fit(X_train[features], y_train)
            pred_test = model.predict(X_test[features])
            pred_val = model.predict(X_val[features])
        else:  # nothing selected: majority-class predictor
            majority = int(np.bincount(y_train).argmax())
            pred_test = np.full(len(y_test), majority)
            pred_val = np.full(len(y_val), majority)
        record.update(
            {
                "n_features": len(features),
                "reduction": 1.0 - len(features) / X.shape[1],
                "features": features,
                "metrics_test": compute_metrics(y_test, pred_test),
                "metrics_val": compute_metrics(y_val, pred_val),
                "eval_seconds": time.perf_counter() - t1,
                "extra": selection.extra,
            }
        )
    record["peak_rss_mb"] = peak_rss_mb()
    record["wall_seconds_total"] = time.perf_counter() - t_load
    out.write_text(json.dumps(record, indent=1, default=_json_default))
    return 0 if status == "ok" else 1


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


import numpy as np  # noqa: E402  (used by worker; imported late to keep the CLI start fast)


# ------------------------------------------------------------------------------ driver
def launch(spec: dict, spec_path: Path, timeout_hours: float) -> str:
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=1))
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)]
    try:
        proc = subprocess.run(cmd, timeout=timeout_hours * 3600, check=False)
    except subprocess.TimeoutExpired:
        Path(spec["out"]).parent.mkdir(parents=True, exist_ok=True)
        Path(spec["out"]).write_text(
            json.dumps(
                {
                    "dataset": spec["dataset"],
                    "method": spec["method"],
                    "seed": spec["seed"],
                    "scalability": spec["scalability"],
                    "status": "timeout",
                    "timeout_hours": timeout_hours,
                },
                indent=1,
            )
        )
        return "timeout"
    return "ok" if proc.returncode == 0 else "error"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks" / "config.yaml")
    parser.add_argument("--datasets", nargs="*", help="subset of dataset names")
    parser.add_argument("--methods", nargs="*", help="subset of method names")
    parser.add_argument("--seeds", nargs="*", type=int, help="subset of seeds")
    parser.add_argument("--scalability", action="store_true", help="FastForest on the full rows")
    parser.add_argument("--prepare-only", action="store_true", help="prepare datasets and stop")
    parser.add_argument("--force", action="store_true", help="re-run even if a result exists")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    if args.worker:
        return worker(args.worker)

    cfg = load_config(args.config)
    output_dir = ROOT / cfg["output_dir"]
    prepared_dir = ROOT / cfg["prepared_dir"]
    datasets = args.datasets or list(cfg["datasets"])
    methods = args.methods or list(cfg["methods"])
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown methods {unknown}; known: {sorted(METHODS)}")
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    started = time.perf_counter()
    counts = {"ok": 0, "error": 0, "timeout": 0, "skipped": 0}

    for name in datasets:
        dcfg = cfg["datasets"][name]
        meta = prepare_dataset(
            name,
            dcfg,
            root=ROOT,
            prepared_dir=prepared_dir,
            comparison_max_rows=cfg["comparison_max_rows"],
            min_class_size=cfg["min_class_size"],
        )
        if args.prepare_only:
            continue
        paths = prepared_paths(prepared_dir, name)
        if args.scalability:
            if not meta.get("scalability"):
                continue
            data_path = paths["full"]
            run_methods = [m for m in methods if m.startswith("fastforest")]
            run_seeds = seeds[: dcfg.get("scalability_seeds", cfg.get("scalability_seeds", 1))]
        else:
            data_path = paths["comparison"]
            run_methods = [m for m in methods if m in dcfg.get("methods", methods)]
            run_seeds = [s for s in seeds if s in dcfg.get("seeds", seeds)]

        for seed in run_seeds:
            for method in run_methods:
                out = run_path(output_dir, name, method, seed, scalability=args.scalability)
                if out.exists() and not args.force:
                    counts["skipped"] += 1
                    continue
                reference = None
                if method == "random":
                    ref = run_path(
                        output_dir, name, "fastforest_mdi", seed, scalability=args.scalability
                    )
                    if not ref.exists():
                        log.warning(
                            "%s seed %d: random control skipped (no fastforest_mdi result)",
                            name,
                            seed,
                        )
                        continue
                    reference = json.loads(ref.read_text()).get("n_features")
                spec = {
                    "dataset": name,
                    "method": method,
                    "seed": seed,
                    "scalability": args.scalability,
                    "data": str(data_path),
                    "out": str(out),
                    "split": cfg["split"],
                    "rf_params": cfg["final_model"],
                    "fastforest": cfg["fastforest"],
                    "params": cfg["methods"].get(method, {}),
                    "n_jobs": cfg.get("n_jobs", -1),
                    "reference_n_features": reference,
                }
                spec_path = output_dir / "specs" / f"{name}__{method}__seed{seed}.json"
                t0 = time.perf_counter()
                status = launch(spec, spec_path, cfg.get("timeout_hours", 12))
                counts[status] += 1
                log.info(
                    "%s | %-16s | seed %d | %-7s | %.1f s",
                    name,
                    method,
                    seed,
                    status,
                    time.perf_counter() - t0,
                )

    log.info("done in %.1f min: %s", (time.perf_counter() - started) / 60, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
