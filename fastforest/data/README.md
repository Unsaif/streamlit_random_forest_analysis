# Bundled sample dataset

`reaction_abundance_body_site_sample.csv.gz` is a subsample of the body-site reaction-abundance
dataset used in the FastForest paper (APOLLO genome-scale reconstructions; Heinken, Hulshof et al.,
Cell Systems 2025). The full table (1,957 genomes × 8,306 reactions, ~100 MB) lives in the
repository root as `reaction_abundance_body_site.csv` and will be deposited on Zenodo.

The sample is produced by `scripts/make_sample_dataset.py` with seed 0:

- 1,000 rows, stratified by `Body site` (Gut, Skin, Nasal_cavity, Vagina);
- 500 reactions: the 100 with the highest mean-decrease-in-impurity importance in a 100-tree
  forest on the full data, plus 400 drawn at random from the remaining non-constant reactions;
  columns keep their original order;
- values rounded to four significant digits.

It exists for tests, the demo in the Streamlit app and the 30-second example in the README. Do not
use it for benchmark numbers.
