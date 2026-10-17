"""Bundled sample data and table loading helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pandas as pd

SAMPLE_TARGET = "Body site"
"""Target column of the bundled sample dataset."""

SAMPLE_FILENAME = "reaction_abundance_body_site_sample.csv.gz"

TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".parquet", ".pq", ".csv.gz", ".tsv.gz", ".txt.gz")


def sample_path() -> Path:
    """Path of the bundled sample dataset inside the installed package."""
    return Path(str(resources.files("fastforest").joinpath("data", SAMPLE_FILENAME)))


def load_sample(*, return_X_y: bool = True):
    """Load the bundled subsample of the body-site reaction-abundance dataset.

    The sample has 1,000 stratified rows and 500 of the 8,306 reactions (100 most important in a
    forest on the full data plus 400 chosen at random), rounded to four significant digits.
    See ``fastforest/data/README.md`` and ``scripts/make_sample_dataset.py``.

    Parameters
    ----------
    return_X_y : bool, default True
        Return ``(X, y)`` when True, the whole table otherwise.

    Returns
    -------
    (pandas.DataFrame, pandas.Series) or pandas.DataFrame
    """
    frame = pd.read_csv(sample_path())
    if return_X_y:
        return frame.drop(columns=[SAMPLE_TARGET]), frame[SAMPLE_TARGET]
    return frame


def read_table(
    source,
    *,
    name: str | None = None,
    sep: str | None = None,
    index_col: int | str | None = None,
) -> pd.DataFrame:
    """Read a CSV / TSV / Parquet table from a path or file-like object.

    Parameters
    ----------
    source : path-like or file-like
        File to read. Uploaded files (with a ``.name`` attribute) are supported.
    name : str, optional
        File name used to infer the format when ``source`` is file-like.
    sep : str, optional
        Delimiter; inferred from the extension when omitted (``.tsv``/``.txt`` → tab).
    index_col : int or str, optional
        Column to use as the row index.
    """
    filename = name or getattr(source, "name", None) or str(source)
    lower = str(filename).lower()
    if lower.endswith((".parquet", ".pq")):
        frame = pd.read_parquet(source)
        if index_col is not None:
            frame = frame.set_index(index_col)
        return frame
    if sep is None:
        base = lower[:-3] if lower.endswith(".gz") else lower
        sep = "\t" if base.endswith((".tsv", ".txt")) else ","
    compression = "gzip" if lower.endswith(".gz") else None
    return pd.read_csv(source, sep=sep, index_col=index_col, compression=compression)
