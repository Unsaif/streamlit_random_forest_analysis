# Changelog

All notable changes to FastForest are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased] — 1.0.0.dev0 (Phase 1: package refactor)

### Added
- `fastforest` Python package: `preprocess`, `split`, `importance`, `metrics`, `reduce`,
  `redundancy`, `result`, `viz`, `datasets`, `cli` modules; `FastForest` estimator and
  `FastForestResult` container; every stochastic step takes `random_state`.
- `fastforest reduce` command-line interface writing `features.csv`, `metrics.json`,
  `importances.csv`, `clusters.json` and figures (PNG for matplotlib, HTML for plotly).
- Bundled subsample of the body-site dataset (`fastforest.load_sample()`), built by
  `scripts/make_sample_dataset.py`.
- `pyproject.toml` with `app`, `bench` and `dev` extras; `LICENSE` (GPL-3.0-or-later),
  `CITATION.cff`, `CONTRIBUTING.md`, this changelog; Dockerfile on `python:3.11-slim`.
- Streamlit app rewritten as a thin layer (`app/streamlit_app.py`) that calls the library and
  renders returned figures; adds seed, split-size, importance-method and redundancy-cut inputs.
- Three-way split: the app shows validation metrics for every stage and test metrics at the end.

### Behaviour changes (relative to the 2021 `random_forest_analysis.py`)
All of these follow `PUBLICATION_PLAN.md` §3 and decisions D2–D6.
- **Preprocessing (D2, B12):** scikit-learn replaces fastai. Continuous columns are median-imputed
  and *not* normalised (trees are scale-invariant; histograms now show raw values). Numeric
  columns with ≤ 20 distinct values stay continuous unless `categorical_low_cardinality=True`.
  No `_na` indicator columns are added unless `add_missing_indicator=True`. Categorical columns
  are ordinal-encoded with unseen categories mapped to −1. Labels are encoded with
  `LabelEncoder` (sorted class order, B7).
- **Trees (D4):** `n_estimators` default 100 (was 40).
- **Splits (D5, B3):** stratified 60/20/20 train/validation/test (was 80/20). The threshold
  search uses the validation split only; the test split is evaluated once at the end. The
  `index` split method puts user-supplied rows in the test set and splits the remainder 75/25.
- **Bound enforcement (D6):** after redundancy removal, dropped features are re-inserted in
  importance order until the validation bound holds again (`enforce_bound=True`).
- **Threshold search termination (B2):** iterates over an explicit list of quantiles
  0.99, 0.98, …, 0.00. If no quantile satisfies the bound, every feature with positive
  importance is kept, and the bound is re-checked on that model and reported in
  `bound_satisfied`.
- **Redundancy clustering:** the Spearman matrix is no longer rounded to four decimals before
  clustering; NaN correlations (constant columns) are treated as distance 1.
- **Per-class heatmaps** use Spearman correlation (spec S7); the old app used Pearson.
- **UMAP:** `n_neighbors` default 15 (was 250); minimum 2 (B9); seeded (B6).
- **Determinism (B6):** a single `random_state` seeds every forest, split, permutation
  importance and UMAP.

### Fixed
- B1: redundancy removal kept the alphabetically last feature of a cluster
  (`max(dict)`), not the most important one.
- B2: float-equality loop termination; quantile could go negative and raise.
- B3: stopping rule and reported metrics used the same validation split.
- B4/B5: Spearman on two columns returned a scalar and crashed `squareform`; the row-count
  check meant to be a column-count check. The redundancy step is now skipped below 3 features.
- B6: no `random_state` anywhere.
- B7: class names relied on fastai's sorted encoding.
- B8: LDA 2-D requires K ≥ 3 classes and 3-D requires K ≥ 4; both conditions implemented.
- B9: UMAP slider allowed `n_neighbors = 0`.
- B10: `@st.cache` replaced by `st.cache_data`.
- B11: global `warnings.filterwarnings("ignore")` removed.
- B12: see preprocessing above.

### Removed
- fastai, torch, torchvision, Pillow, fpdf, dtreeviz, seaborn and the 2021 `requirements.txt`
  freeze. `main.py` and `random_forest_analysis.py` are replaced by `app/streamlit_app.py` and
  the package.
