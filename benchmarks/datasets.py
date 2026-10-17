"""Dataset loading and preparation for the benchmark (PUBLICATION_PLAN.md §6.1).

Each dataset is prepared once into ``<prepared_dir>/<name>.parquet`` (the comparison table,
stratified-subsampled when larger than ``comparison_max_rows``) plus ``<name>.full.parquet`` for
scalability runs, and ``<name>.json`` with the sizes and class counts that feed table T1.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

log = logging.getLogger("benchmark")

TARGET = "__target__"
INDEX_COLUMN = "__index_level_0__"
METADATA_CACHE = "metadata_cache.parquet"


def read_presence_parquet(path: Path, batch_size: int = 8192) -> pd.DataFrame:
    """Stream a reaction-presence Parquet file into a ``uint8`` DataFrame indexed by genome ID.

    The files store 0/1 values as int64 (10 GB in memory for the combined set); streaming row
    batches and casting to uint8 keeps the peak well below that.
    """
    reader = pq.ParquetFile(path)
    names = [n for n in reader.schema_arrow.names if n != INDEX_COLUMN]
    blocks: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for batch in reader.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        if INDEX_COLUMN in frame.columns:
            ids.append(frame.pop(INDEX_COLUMN).to_numpy())
        else:
            ids.append(frame.index.to_numpy())
        values = frame[names].to_numpy()
        if values.max() > 255 or values.min() < 0:
            raise ValueError(f"{path}: values outside uint8 range")
        blocks.append(values.astype(np.uint8))
    data = np.concatenate(blocks)
    index = pd.Index(np.concatenate(ids).astype(str), name="genome")
    return pd.DataFrame(data, columns=names, index=index)


def load_metadata(sources: list[dict], cache_dir: Path) -> pd.DataFrame:
    """Concatenate metadata workbooks into one table indexed by genome ID (cached as Parquet)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in sources:
        path = Path(source["path"])
        cache = cache_dir / f"{path.stem}.parquet"
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            frame = pd.read_parquet(cache)
        else:
            log.info("reading %s (cached afterwards)", path)
            frame = pd.read_excel(path)
            frame = frame.astype({c: str for c in frame.columns if frame[c].dtype == object})
            frame.to_parquet(cache)
        key = source["key"]
        frame = frame.rename(columns={key: "genome"}).set_index("genome")
        frame.index = frame.index.astype(str)
        frames.append(frame)
    meta = pd.concat(frames, axis=0, join="outer")
    meta = meta[~meta.index.duplicated(keep="first")]
    return meta


def load_raw(cfg: dict, root: Path, cache_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset's features and labels as described by its config entry."""
    kind = cfg["kind"]
    if kind == "csv":
        frame = pd.read_csv(root / cfg["path"])
        y = frame.pop(cfg["target"]).astype(str)
        return frame, y
    if kind == "lab_parquet":
        X = read_presence_parquet(root / cfg["path"])
        sources = [{"path": root / m["path"], "key": m["key"]} for m in cfg["metadata"]]
        meta = load_metadata(sources, cache_dir)
        y = meta[cfg["target"]].reindex(X.index)
        y = y.replace({"nan": np.nan, "None": np.nan, "": np.nan})
        mask = y.notna().to_numpy()
        log.info(
            "%s: %d of %d genomes have a %s label", cfg["path"], mask.sum(), len(y), cfg["target"]
        )
        return X.iloc[np.flatnonzero(mask)], y[mask].astype(str)
    if kind == "openml":
        from scipy import sparse
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(
            data_id=cfg["openml_id"],
            data_home=root / cfg.get("data_home", "benchmarks/data/openml"),
            as_frame=False,
            parser="auto",
        )
        data = bunch.data.toarray() if sparse.issparse(bunch.data) else np.asarray(bunch.data)
        X = pd.DataFrame(data, columns=[str(c) for c in bunch.feature_names])
        y = pd.Series(np.asarray(bunch.target).astype(str), name=cfg.get("target", "target"))
        log.info("OpenML %s: %d rows x %d features", cfg["openml_id"], *X.shape)
        return X, y
    raise ValueError(f"unknown dataset kind {kind!r}")


def filter_classes(X: pd.DataFrame, y: pd.Series, min_class_size: int):
    """Drop classes with fewer than ``min_class_size`` rows."""
    counts = y.value_counts()
    keep = counts.index[counts >= min_class_size]
    dropped = counts[counts < min_class_size].to_dict()
    mask = y.isin(keep).to_numpy()
    return X.iloc[np.flatnonzero(mask)], y[mask], dropped


def stratified_subsample(X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int = 0):
    """Return a class-stratified subsample of at most ``max_rows`` rows (seeded)."""
    if len(y) <= max_rows:
        return X, y
    keep, _ = train_test_split(
        np.arange(len(y)), train_size=max_rows, stratify=y.to_numpy(), random_state=seed
    )
    keep = np.sort(keep)
    return X.iloc[keep], y.iloc[keep]


def prepared_paths(prepared_dir: Path, name: str) -> dict[str, Path]:
    return {
        "comparison": prepared_dir / f"{name}.parquet",
        "full": prepared_dir / f"{name}.full.parquet",
        "meta": prepared_dir / f"{name}.json",
    }


def prepare_dataset(
    name: str,
    cfg: dict,
    *,
    root: Path,
    prepared_dir: Path,
    comparison_max_rows: int,
    min_class_size: int,
    force: bool = False,
) -> dict:
    """Prepare (or reuse) the comparison and scalability tables of one dataset."""
    paths = prepared_paths(prepared_dir, name)
    if paths["meta"].exists() and paths["comparison"].exists() and not force:
        return json.loads(paths["meta"].read_text())

    prepared_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_raw(cfg, root, prepared_dir / "cache")
    n_labelled = len(y)
    X, y, dropped = filter_classes(X, y, min_class_size)
    meta = {
        "name": cfg.get("name", name),
        "source": cfg.get("source", ""),
        "target": cfg["target"],
        "feature_type": cfg.get("feature_type", ""),
        "n_rows_labelled": int(n_labelled),
        "n_rows_full": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_classes_full": int(y.nunique()),
        "classes_dropped": {str(k): int(v) for k, v in dropped.items()},
        "scalability": bool(cfg.get("scalability", False)),
    }
    if meta["scalability"]:
        full = X.copy()
        full[TARGET] = y.to_numpy()
        full.to_parquet(paths["full"])
        del full

    Xc, yc = stratified_subsample(X, y, comparison_max_rows, seed=0)
    Xc, yc, dropped_after = filter_classes(Xc, yc, min_class_size)
    meta.update(
        {
            "subsampled": bool(len(yc) < len(y)),
            "n_rows_comparison": int(len(yc)),
            "n_classes_comparison": int(yc.nunique()),
            "class_counts_comparison": {str(k): int(v) for k, v in yc.value_counts().items()},
            "classes_dropped_after_subsample": {str(k): int(v) for k, v in dropped_after.items()},
        }
    )
    comparison = Xc.copy()
    comparison[TARGET] = yc.to_numpy()
    comparison.to_parquet(paths["comparison"])
    paths["meta"].write_text(json.dumps(meta, indent=2))
    log.info(
        "%s: %d rows x %d features, %d classes (comparison %d rows, %d classes)",
        name,
        meta["n_rows_full"],
        meta["n_features"],
        meta["n_classes_full"],
        meta["n_rows_comparison"],
        meta["n_classes_comparison"],
    )
    return meta


def load_prepared(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a prepared table and split off the label column."""
    frame = pd.read_parquet(path)
    y = frame.pop(TARGET).astype(str)
    return frame, y
