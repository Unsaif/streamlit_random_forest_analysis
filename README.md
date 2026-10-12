# FastForest

[![Web app](https://img.shields.io/badge/web%20app-fastforest.vmh.life-2b7a4b)](https://fastforest.vmh.life)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

**FastForest** reduces a high-dimensional labelled table to the few features a random forest
needs to classify it, without losing accuracy. It trains a forest on all features, searches over
importance thresholds for the smallest subset whose validation metrics stay within a bound of the
full model, removes redundant features by Spearman-correlation clustering, and reports the final
model on a held-out test split. It ships as a Python library, a command-line tool and a Streamlit
web app.

It was built for metabolic-reconstruction data (thousands of reactions per genome) but works on
any table with one class column.

## Installation

```bash
pip install git+https://github.com/ThieleLab/streamlit_random_forest_analysis.git
```

Optional extras: `[app]` for the Streamlit app, `[bench]` for the benchmark scripts, `[dev]`
for tests and linting. For development:

```bash
git clone https://github.com/ThieleLab/streamlit_random_forest_analysis.git
cd streamlit_random_forest_analysis
pip install -e ".[app,dev]"
```

Python 3.10 or newer. Installing takes about a minute; there is no deep-learning dependency.

## 30-second example

```python
from fastforest import FastForest, load_sample

X, y = load_sample()  # 1,000 genomes x 500 reactions, 4 body sites
ff = FastForest(bound=5e-4, criterion="accuracy", n_estimators=100, random_state=0)
result = ff.fit(X, y)

result.features_final  # the selected features, most important first
result.metrics["final"]["test"]  # accuracy, macro precision / recall / F1 on the test split
result.summary()  # one row per stage and split
result.to_json("fastforest_result.json")
```

Figures are plain matplotlib / plotly objects:

```python
from fastforest import viz

raw, normalised = viz.confusion_matrices(result, stage="final", split="test")
viz.importance_bar(result.importances["full"]).show()
viz.dendrogram(result.redundancy).savefig("dendrogram.png")
viz.lda_scatter(result.X_final, result.y, result.classes).show()
viz.umap_scatter(result.X_final, result.y, result.classes, random_state=0).show()
```

## Command line

```bash
fastforest reduce reaction_abundance_body_site.csv --target "Body site" --out results/demo
```

writes `features.csv`, `metrics.json` (parameters, metrics, timings, environment),
`importances.csv`, `clusters.json` and a `figures/` directory. Useful options: `--bound 5e-4`,
`--method accuracy|all`, `--importance mdi|permutation`, `--n-estimators 100`, `--seed 0`,
`--no-umap`, `--no-figures`. Run `fastforest reduce --help` for the full list. CSV, TSV and
Parquet inputs are accepted.

## Web app

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

or use the hosted instance at <https://fastforest.vmh.life>. Upload a table (or pick the bundled
sample), choose the class column, set the bound and seed in the sidebar and press *Run*. The app
shows validation metrics for every stage, test metrics at the end, confusion matrices, the
threshold search, the redundancy dendrogram, per-class heatmaps and histograms, LDA and UMAP, and
lets you download the final feature list.

With Docker:

```bash
docker compose up --build        # then open http://localhost:8501
```

## How it works

| Stage | What happens | Default |
| --- | --- | --- |
| S0 Preprocess | median imputation, ordinal encoding of categoricals; fitted on the training split only; no scaling | |
| S1 Split | stratified train / validation / test | 60 / 20 / 20, seeded |
| S2 Full model | `RandomForestClassifier(max_features=0.5, min_samples_leaf=5)` on all features | 100 trees |
| S3 Importance | mean decrease in impurity, or permutation importance on the validation split | MDI |
| S4 Threshold search | keep features above the *q*-quantile of the positive importances for *q* = 0.99, 0.98, …, 0; retrain; stop when validation metrics are within the bound of S2 | bound 5 × 10⁻⁴ on accuracy |
| S5 Redundancy | Spearman ρ between kept features, average-linkage clustering, cut at 1 − ρ = 0.01; keep the most important member of each cluster | ρ ≥ 0.99 |
| S6 Final model | retrain; if the bound breaks, re-insert dropped features in importance order until it holds; evaluate once on the test split | enforce on |
| S7 Figures | confusion matrices, importances, dendrogram, per-class heatmaps and histograms, LDA (2-D for ≥ 3 classes, 3-D for ≥ 4), UMAP | |

Every stochastic step takes `random_state`, so a run is reproducible from its `metrics.json`.

## Repository layout

| Path | Contents |
| --- | --- |
| `fastforest/` | the library: `preprocess`, `split`, `importance`, `metrics`, `reduce`, `redundancy`, `result`, `viz`, `cli`, `datasets`, bundled sample data |
| `app/streamlit_app.py` | the Streamlit app (thin; calls the library) |
| `tests/` | pytest suite |
| `benchmarks/` | evaluation for the paper |
| `scripts/` | one-off helpers (e.g. building the sample dataset) |
| `docs/` | notes and the paper sources |
| `reaction_abundance_body_site.csv` | the full example dataset (1,957 genomes × 8,306 reactions) |

## Citation

If FastForest is useful in your work, please cite it (see `CITATION.cff`). An earlier version of
the method produced the taxonomic classifier in Heinken, Hulshof *et al.*, *APOLLO: a
genome-scale metabolic reconstruction resource of 247,092 diverse human microbes spanning
multiple continents, age groups, and body sites*, Cell Systems 2025,
<https://doi.org/10.1016/j.cels.2025.101196>.

## Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and questions go to the
[issue tracker](https://github.com/ThieleLab/streamlit_random_forest_analysis/issues).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
