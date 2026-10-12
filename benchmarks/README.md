# FastForest benchmark

Evaluation protocol for the paper (plan §6). Everything is driven by `config.yaml`; run from
the repository root with the `bench` extra installed (`pip install -e ".[bench]"`).

```bash
python benchmarks/run_benchmark.py --config benchmarks/config.yaml            # all datasets, methods, seeds
python benchmarks/run_benchmark.py --config benchmarks/config.yaml --scalability   # FastForest on the full rows
python benchmarks/summarize.py    --config benchmarks/config.yaml            # tables, figures, env.json
```

Useful flags: `--datasets body_site rp_90k_phylum`, `--methods fastforest_mdi rfecv`,
`--seeds 0 1`, `--prepare-only`, `--force`.

## Protocol

For every dataset and seed, rows are split 60/20/20 (stratified, seeded) and the split is shared
by every method. Each method selects features from the training split, plus the validation split
where the method needs one (FastForest's threshold search, the permutation-importance threshold).
A fresh random forest with the S2 hyper-parameters (100 trees, `max_features=0.5`,
`min_samples_leaf=5`) is then trained on the training rows of the selected features and
evaluated **once** on the test split. Recorded per run: number of features, test accuracy and
macro precision/recall/F1, selection wall-clock seconds, peak resident memory of the worker
process, and method-specific details (FastForest's stopping quantile, Boruta's tentative count,
the L1 regularisation chosen, …).

Datasets with more than `comparison_max_rows` (20,000) labelled rows are stratified-subsampled
once (seed 0) for the method comparison; FastForest alone also runs on the full rows for the
scalability table. Classes with fewer than `min_class_size` (10) rows are dropped before and
after subsampling.

## Methods

| key | method |
| --- | --- |
| `fastforest_mdi` | FastForest, mean-decrease-in-impurity importance (the default) |
| `fastforest_perm` | FastForest, permutation importance on the validation split |
| `boruta` | Boruta (`BorutaPy`) with the S2 forest, confirmed features only |
| `rfecv` | scikit-learn `RFECV`, S2 forest, `step=0.1`, 5-fold stratified CV on train |
| `perm_threshold` | S2 forest on train; keep features with permutation importance > 0 on validation |
| `l1_logistic` | L1-regularised logistic regression; C by 5-fold CV with the one-standard-error rule; keep non-zero coefficients |
| `full` | control: no selection |
| `random` | control: a random subset the size of `fastforest_mdi`'s final set for the same seed |

## Layout of the results

```
results/
  runs/<dataset>/<method>/seed<k>.json        one file per run (resumable: existing files are skipped)
  runs_scalability/<dataset>/<method>/seed<k>.json
  specs/                                      the exact worker inputs
  summary/per_run.csv  summary.csv  stability.csv  datasets.csv  scalability.csv
  summary/fig_reduction_vs_accuracy.{png,pdf}  fig_stability.png
  env.json                                    hardware, versions, command, config hash
benchmarks/data/
  lab/        the APOLLO reaction-presence Parquet files and metadata workbooks (not in git)
  prepared/   prepared tables and per-dataset JSON metadata (not in git)
```
