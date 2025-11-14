# FastForest

Check out the webapp here: [FastForest](https://fastforest.vmh.life)

FastForest is a Streamlit application that helps you explore high-dimensional tabular datasets with a focus on feature reduction using random forests. It trains a baseline model, identifies the most informative features, removes redundant ones, and visualizes the resulting latent space with LDA and UMAP. The app ships with an example metabolomics dataset (`reaction_abundance_body_site.csv`) so you can see the full workflow end-to-end.

## Highlights

- **Automated feature pruning** – trains iterative random forest models until accuracy (and optionally precision/recall) remain within a configurable bound of the baseline run.
- **Redundancy detection** – drops correlated features via Spearman clustering to further shrink the feature set.
- **Visual diagnostics** – confusion matrices, feature-importance histograms, LDA 2D/3D plots, and UMAP embeddings help you understand class separability.
- **Flexible data splitting** – choose stratified, random, or index-based train/validation splits directly from the sidebar.
- **Exports** – download the curated feature list or the sample dataset for reference.

## Repository Layout

| Path | Description |
| --- | --- |
| `main.py` | Streamlit UI entrypoint; handles sidebar inputs, file upload, sample data download, and result display. |
| `random_forest_analysis.py` | Core analysis pipeline: preprocessing via fastai, model training, feature filtering, clustering, LDA/UMAP charts. |
| `reaction_abundance_body_site.csv` | Example dataset (body-site classification) used when no file is uploaded. |
| `requirements.txt` | Locked Python dependencies for local installs and the Docker image. |
| `Dockerfile`, `docker-compose.yml` | Containerized deployment assets exposing the app on port 8501. |

## Getting Started (Local Python)

Prerequisites:

- Python 3.8+ (matching the version in the Dockerfile)
- `pip`
- Recommended: a virtual environment (`python3 -m venv .venv && source .venv/bin/activate`)

```bash
pip install -r requirements.txt
streamlit run main.py
```

Streamlit defaults to `http://localhost:8501`. Keep the terminal open to see Streamlit logs.

## Running with Docker

Build and run the container manually:

Use the provided compose file (auto-rebuilds and adds a health check):

```bash
docker-compose up --build
```

Navigate to `http://localhost:8501` once the health check reports `healthy`.

## Using the App

1. **Upload data** – provide a CSV/TSV with one categorical dependent variable column. Alternatively download the sample file from the homepage to see the required schema.
2. **Set the dependent variable** – enter the column name exactly as it appears (e.g., `Body site` for the sample data).
3. **Tune sidebar knobs** (defaults usually work):
   - `Reduction Method`: stop once accuracy matches the baseline, or enforce accuracy/precision/recall simultaneously.
   - `Bound (10^-4)`: acceptable metric drift (e.g., `5` represents `5 × 10⁻⁴`).
   - `UMAP`: disable if you want faster results on very large datasets.
   - `n_neighbours (UMAP)`: controls how much global vs. local structure UMAP preserves.
   - `Train-Test split`: pick stratified, random, or supply explicit indices.
4. **Review outputs** – the app walks through baseline metrics, feature-importance plots, reduced-model metrics, redundancy pruning details, LDA/UMAP visualizations, and finally lists the surviving features.
5. **Export results** – download the final feature list as `features.csv` for downstream modeling.

## Contributing

Issues and pull requests are welcome. 

