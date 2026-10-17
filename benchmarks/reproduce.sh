#!/usr/bin/env bash
# Reproduce every benchmark table and figure from the raw inputs (PUBLICATION_PLAN.md §6).
#
#   bash benchmarks/reproduce.sh [config]          default config: benchmarks/config.yaml
#
# Inputs: reaction_abundance_body_site.csv (repository root) and the APOLLO reaction-presence
# Parquet files plus metadata workbooks in benchmarks/data/lab/ (not in git; see
# benchmarks/README.md). Finished runs are skipped, so an interrupted reproduction resumes where
# it stopped. Runs at the lowest CPU priority; time the scalability step on an idle machine.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-benchmarks/config.yaml}"
PYTHON="${PYTHON:-python}"
nice -n 19 "$PYTHON" benchmarks/run_benchmark.py --config "$CONFIG"                 # T2, F3, F4
nice -n 19 "$PYTHON" benchmarks/run_benchmark.py --config "$CONFIG" --scalability   # T3
"$PYTHON" benchmarks/summarize.py --config "$CONFIG"                               # tables, figures, env
