"""Command-line interface.

``fastforest reduce data.csv --target "Body site" --out results/demo`` runs the pipeline and
writes ``features.csv``, ``metrics.json``, ``importances.csv``, ``clusters.json`` and a
``figures/`` directory (PNG for matplotlib figures, HTML for plotly figures).
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from fastforest import __version__, viz
from fastforest.datasets import read_table
from fastforest.reduce import FastForest
from fastforest.result import FastForestResult

logger = logging.getLogger("fastforest")


def environment_info() -> dict[str, str]:
    """Versions and platform details recorded alongside every run."""
    return {
        "fastforest": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "x"


def write_figures(
    result: FastForestResult,
    out_dir: Path,
    *,
    umap: bool = True,
    umap_neighbors: int = 15,
    random_state: int | None = None,
) -> list[Path]:
    """Write every stage-S7 figure for ``result`` into ``out_dir``.

    Returns the list of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save_mpl(fig, name: str) -> None:
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        written.append(path)

    def save_plotly(fig, name: str) -> None:
        path = out_dir / f"{name}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        written.append(path)

    for stage in ("full", "important", "final"):
        raw, norm = viz.confusion_matrices(result, stage=stage, split="test")
        save_mpl(raw, f"confusion_{stage}_test")
        save_mpl(norm, f"confusion_{stage}_test_normalised")
    save_plotly(viz.importance_bar(result.importances["full"]), "importance_top30")
    save_plotly(viz.search_history_plot(result), "search_history")
    save_mpl(viz.dendrogram(result.redundancy), "dendrogram")

    X, y, classes = result.X_final, result.y, result.classes
    for name, fig in viz.class_heatmaps(X, y, classes).items():
        save_plotly(fig, f"heatmap_{_slug(name)}")
    for name, fig in viz.class_histograms(X, y, classes).items():
        save_mpl(fig, f"histograms_{_slug(name)}")

    k, p = len(classes), X.shape[1]
    for dims in (2, 3):
        if viz.lda_available_components(k, p) >= dims:
            save_plotly(viz.lda_scatter(X, y, classes, n_components=dims), f"lda_{dims}d")
        else:
            logger.info("LDA %d-D skipped: needs at least %d classes", dims, dims + 1)
    if umap:
        for dims in (2, 3):
            save_plotly(
                viz.umap_scatter(
                    X,
                    y,
                    classes,
                    n_components=dims,
                    n_neighbors=umap_neighbors,
                    random_state=random_state,
                ),
                f"umap_{dims}d",
            )
    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``fastforest`` command."""
    parser = argparse.ArgumentParser(
        prog="fastforest",
        description="Random-forest feature reduction for high-dimensional tabular data.",
    )
    parser.add_argument("--version", action="version", version=f"fastforest {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("reduce", help="run the FastForest pipeline on a table")
    p.add_argument("table", type=Path, help="CSV / TSV / Parquet file with one target column")
    p.add_argument("--target", required=True, help="name of the class column")
    p.add_argument("--out", type=Path, default=Path("results/fastforest"), help="output directory")
    p.add_argument("--bound", type=float, default=5e-4, help="allowed metric drop (default 5e-4)")
    p.add_argument(
        "--method",
        choices=("accuracy", "all"),
        default="accuracy",
        help="bound on accuracy only, or on accuracy, precision and recall",
    )
    p.add_argument("--n-estimators", type=int, default=100, help="trees per forest")
    p.add_argument("--max-features", type=float, default=0.5, help="features per split")
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument(
        "--importance", choices=("mdi", "permutation"), default="mdi", help="importance method"
    )
    p.add_argument("--cluster-threshold", type=float, default=0.01, help="redundancy cut (1-rho)")
    p.add_argument("--no-enforce-bound", action="store_true", help="disable D6 bound enforcement")
    p.add_argument("--split", choices=("stratified", "random"), default="stratified")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0, help="random_state for every stochastic step")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--sep", default=None, help="delimiter (inferred from the extension)")
    p.add_argument("--index-col", default=None, help="column to use as row index")
    p.add_argument("--no-figures", action="store_true", help="skip all figures")
    p.add_argument("--no-umap", action="store_true", help="skip the UMAP figures")
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--include-splits", action="store_true", help="store split indices in JSON")
    p.add_argument("--quiet", action="store_true", help="only print warnings")
    p.set_defaults(func=cmd_reduce)
    return parser


def cmd_reduce(args: argparse.Namespace) -> int:
    """Run ``fastforest reduce``."""
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.perf_counter()
    index_col = args.index_col
    if isinstance(index_col, str) and index_col.isdigit():
        index_col = int(index_col)
    frame = read_table(args.table, sep=args.sep, index_col=index_col)
    if args.target not in frame.columns:
        logger.error(
            "target column %r not found; columns: %s", args.target, list(frame.columns)[:10]
        )
        return 2
    logger.info("Read %s: %d rows x %d columns", args.table, frame.shape[0], frame.shape[1] - 1)

    ff = FastForest(
        bound=args.bound,
        criterion=args.method,
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        importance=args.importance,
        cluster_threshold=args.cluster_threshold,
        enforce_bound=not args.no_enforce_bound,
        split=args.split,
        val_size=args.val_size,
        test_size=args.test_size,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    result = ff.fit(frame, args.target)

    out = Path(args.out)
    paths = result.save(out)
    payload = result.to_dict(include_splits=args.include_splits)
    payload["command"] = " ".join(sys.argv)
    payload["input"] = {"path": str(args.table), "target": args.target, "n_rows": int(len(frame))}
    payload["environment"] = environment_info()
    paths["metrics"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.no_figures:
        figures = write_figures(
            result,
            out / "figures",
            umap=not args.no_umap,
            umap_neighbors=args.umap_neighbors,
            random_state=args.seed,
        )
        logger.info("Wrote %d figures to %s", len(figures), out / "figures")

    n = result.n_features
    test = result.metrics["final"]["test"]
    print(
        f"FastForest kept {n['final']} of {n['full']} features "
        f"({result.reduction:.1%} reduction); test accuracy {test['accuracy']:.4f}, "
        f"macro F1 {test['macro_f1']:.4f}; bound "
        f"{'satisfied' if result.final_bound_satisfied else 'NOT satisfied'}. "
        f"Outputs in {out} ({time.perf_counter() - started:.1f} s)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``fastforest`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
