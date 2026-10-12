"""Build the bundled subsample of ``reaction_abundance_body_site.csv`` (decision D8).

The full dataset (1,957 genomes x 8,306 reactions, ~100 MB) is too large for a pip package. This
script writes a stratified 1,000-row subsample with 500 reactions: the 100 most important in a
seeded random forest on the full data plus 400 drawn at random from the rest (constant columns
excluded), rounded to four significant digits and gzip-compressed (about 1 MB).

Usage::

    python scripts/make_sample_dataset.py --source reaction_abundance_body_site.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fastforest.datasets import SAMPLE_FILENAME, SAMPLE_TARGET
from fastforest.preprocess import encode_labels
from fastforest.reduce import make_forest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=Path("reaction_abundance_body_site.csv"))
    parser.add_argument("--target", default=SAMPLE_TARGET)
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument("--n-top", type=int, default=100)
    parser.add_argument("--n-random", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("fastforest/data") / SAMPLE_FILENAME)
    args = parser.parse_args()

    frame = pd.read_csv(args.source)
    X = frame.drop(columns=[args.target])
    y = frame[args.target]
    X = X.loc[:, X.nunique() > 1]
    print(f"source: {frame.shape[0]} rows, {X.shape[1]} non-constant features")

    y_enc, classes = encode_labels(y)
    forest = make_forest(random_state=args.seed).fit(X, y_enc)
    importance = pd.Series(forest.feature_importances_, index=X.columns)
    top = importance.sort_values(ascending=False, kind="stable").index[: args.n_top].tolist()

    rng = np.random.default_rng(args.seed)
    rest = [c for c in X.columns if c not in set(top)]
    random_cols = rng.choice(rest, size=args.n_random, replace=False).tolist()
    chosen = set(top) | set(random_cols)
    keep = [c for c in X.columns if c in chosen]  # original column order hides which are "top"

    rows, _ = train_test_split(
        np.arange(len(frame)), train_size=args.n_rows, stratify=y, random_state=args.seed
    )
    sample = frame.iloc[np.sort(rows)][keep + [args.target]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.out, index=False, float_format="%.4g", compression="gzip")

    size_mb = args.out.stat().st_size / 1e6
    n_rows, n_cols = sample.shape
    print(f"wrote {args.out}: {n_rows} rows x {n_cols - 1} features, {size_mb:.2f} MB")
    print(sample[args.target].value_counts().to_string())


if __name__ == "__main__":
    main()
