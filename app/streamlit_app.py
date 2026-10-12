"""FastForest Streamlit front end.

This module only collects parameters, calls :class:`fastforest.FastForest` and renders the
figures returned by :mod:`fastforest.viz`. No analysis logic lives here.
"""

from __future__ import annotations

import hashlib
import io
import json

import pandas as pd
import streamlit as st

from fastforest import FastForest, __version__, read_table, viz
from fastforest.datasets import SAMPLE_TARGET, load_sample

STAGE_TITLES = {
    "full": "Model on all features",
    "important": "Model on the important features",
    "final": "Final model",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Precision (macro)",
    "macro_recall": "Recall (macro)",
    "macro_f1": "F1 (macro)",
}
MAX_HISTOGRAM_FEATURES = 30

st.set_page_config(
    page_title="FastForest",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------- caching
@st.cache_data(show_spinner=False)
def sample_frame() -> pd.DataFrame:
    return load_sample(return_X_y=False)


@st.cache_data(show_spinner=False)
def frame_to_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner="Reading file…")
def read_uploaded(data: bytes, name: str) -> pd.DataFrame:
    return read_table(io.BytesIO(data), name=name)


# ----------------------------------------------------------------------------- sidebar
def sidebar_options() -> dict:
    st.sidebar.header("Parameters")
    opts: dict = {}
    opts["criterion"] = st.sidebar.selectbox(
        "Reduction criterion",
        ["accuracy", "all"],
        format_func=lambda v: "accuracy" if v == "accuracy" else "accuracy, precision and recall",
        help="Metrics that must stay within the bound of the model on all features.",
    )
    bound_e4 = st.sidebar.slider(
        "Bound (× 10⁻⁴)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Maximum allowed drop of each criterion metric, absolute on the [0, 1] scale. "
        "5 means 5 × 10⁻⁴ = 0.05 percentage points.",
    )
    opts["bound"] = bound_e4 * 1e-4
    opts["importance"] = st.sidebar.selectbox(
        "Feature importance",
        ["mdi", "permutation"],
        format_func=lambda v: "mean decrease in impurity" if v == "mdi" else "permutation",
    )
    opts["n_estimators"] = int(
        st.sidebar.number_input(
            "Trees per forest", min_value=10, max_value=2000, value=100, step=10
        )
    )
    opts["random_state"] = int(
        st.sidebar.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=0, step=1)
    )
    opts["split"] = st.sidebar.selectbox(
        "Split",
        ["stratified", "random", "index"],
        help="stratified / random 60-20-20 train-validation-test, or 'index': you supply the "
        "test rows and the rest is split 75/25.",
    )
    opts["val_size"] = st.sidebar.slider("Validation fraction", 0.05, 0.5, 0.2, 0.05)
    opts["test_size"] = st.sidebar.slider("Test fraction", 0.05, 0.5, 0.2, 0.05)
    opts["cluster_threshold"] = float(
        st.sidebar.number_input(
            "Redundancy cut (1 − ρ)",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.005,
            format="%.3f",
            help="Features whose Spearman correlation is at least 1 − cut are clustered; only "
            "the most important member of each cluster is kept.",
        )
    )
    opts["enforce_bound"] = st.sidebar.toggle("Enforce bound after redundancy removal", True)
    opts["umap"] = st.sidebar.toggle("UMAP", True)
    opts["umap_neighbors"] = (
        int(st.sidebar.slider("n_neighbors (UMAP)", min_value=2, max_value=500, value=15))
        if opts["umap"]
        else 15
    )
    st.sidebar.caption(f"FastForest {__version__}")
    return opts


# ---------------------------------------------------------------------------- pipeline
def run_pipeline(frame: pd.DataFrame, target: str, opts: dict, test_idx):
    ff = FastForest(
        bound=opts["bound"],
        criterion=opts["criterion"],
        n_estimators=opts["n_estimators"],
        importance=opts["importance"],
        cluster_threshold=opts["cluster_threshold"],
        enforce_bound=opts["enforce_bound"],
        split=opts["split"],
        val_size=opts["val_size"],
        test_size=opts["test_size"],
        random_state=opts["random_state"],
    )
    with st.status("Running FastForest…", expanded=True) as status:
        progress = st.progress(0.0, text="Starting")

        def callback(event: dict) -> None:
            if event["stage"] == "search":
                step = event["step"]
                frac = 1.0 if step.q is None else min(1.0, (0.99 - step.q) / 0.99)
                label = "all positive" if step.q is None else f"q = {step.q:.2f}"
                progress.progress(
                    frac,
                    text=f"Threshold search: {label}, {step.n_features} features, "
                    f"validation accuracy {step.metrics['accuracy']:.4f}",
                )
            else:
                status.write(f"{event['message']}  ·  {event['elapsed']:.1f} s")

        result = ff.fit(
            frame.drop(columns=[target]), frame[target], test_idx=test_idx, callback=callback
        )
        progress.progress(1.0, text="Done")
        status.update(
            label=f"Finished in {result.timings['total']:.1f} s", state="complete", expanded=False
        )
    return result


# --------------------------------------------------------------------------- rendering
def render_metrics(result, stage: str, split: str, *, reference: dict | None = None) -> None:
    metrics = result.metrics[stage][split]
    cols = st.columns(len(METRIC_LABELS))
    for col, (name, label) in zip(cols, METRIC_LABELS.items(), strict=True):
        delta = None
        if reference is not None:
            delta = f"{(metrics[name] - reference[name]) * 100:+.2f} pp"
        col.metric(label, f"{metrics[name] * 100:.2f}%", delta)


def render_confusions(result, stage: str) -> None:
    raw, norm = viz.confusion_matrices(result, stage=stage, split="test")
    left, right = st.columns(2)
    left.pyplot(raw)
    right.pyplot(norm)


def render_results(result, opts: dict) -> None:
    n = result.n_features
    full_val = result.metrics["full"]["val"]
    sizes = result.splits.sizes

    st.header("Dataset")
    cols = st.columns(5)
    cols[0].metric("Rows", result.n_samples)
    cols[1].metric("Features", n["full"])
    cols[2].metric("Classes", len(result.classes))
    cols[3].metric(
        "Train / validation / test", f"{sizes['train']} / {sizes['val']} / {sizes['test']}"
    )
    cols[4].metric("Total time", f"{result.timings['total']:.1f} s")

    # ---- S2
    st.header(f"1. {STAGE_TITLES['full']} ({n['full']})")
    st.caption("Validation split. Confusion matrices are on the test split.")
    render_metrics(result, "full", "val")
    render_confusions(result, "full")
    with st.expander("Notes"):
        st.markdown(
            "Confusion matrices list the predicted label along the bottom and the true label on "
            "the left. Accuracy, precision and recall are all derived from them. Precision and "
            "recall are macro-averaged over classes."
        )
    st.plotly_chart(viz.importance_bar(result.importances["full"]))
    with st.expander("Notes"):
        st.markdown(
            "Mean decrease in impurity measures how much each feature reduces node impurity, "
            "weighted by the fraction of samples reaching the node. Rankings vary between runs "
            "with different seeds, but truly informative features keep appearing near the top. "
            "Permutation importance (sidebar) measures the accuracy lost when a feature is "
            "shuffled on the validation split and is less biased towards high-cardinality "
            "features."
        )

    # ---- S4
    st.header(f"2. {STAGE_TITLES['important']} ({n['important']})")
    cols = st.columns(4)
    cols[0].metric(
        "Features kept", n["important"], f"{n['important'] - n['full']}", delta_color="off"
    )
    cols[1].metric("Reduction", f"{(1 - n['important'] / n['full']) * 100:.1f}%")
    cols[2].metric(
        "Stopping quantile", "all > 0" if result.q_final is None else f"{result.q_final:.2f}"
    )
    cols[3].metric("Bound satisfied", "yes" if result.bound_satisfied else "no")
    if not result.bound_satisfied:
        st.warning(
            "No importance threshold kept the validation metrics within the bound; every "
            "feature with positive importance was kept."
        )
    render_metrics(result, "important", "val", reference=full_val)
    st.plotly_chart(viz.search_history_plot(result))
    render_confusions(result, "important")
    with st.expander("Notes"):
        st.markdown(
            "Starting from the 99th percentile of the positive importances, the threshold is "
            "lowered one percentile at a time; at each step a new forest is trained on the "
            "features above the threshold and scored on the validation split. The search stops "
            "at the first threshold whose metrics are within the bound of the model on all "
            "features. Deltas are relative to that model."
        )

    # ---- S5
    st.header("3. Redundancy removal")
    red = result.redundancy
    if red.skipped:
        st.info("Fewer than 3 features remained, so the redundancy step was skipped.")
    else:
        cols = st.columns(3)
        cols[0].metric("Clusters (ρ ≥ %.3g)" % (1 - red.threshold), len(result.clusters))
        cols[1].metric("Dropped as redundant", len(result.dropped_redundant))
        cols[2].metric("Re-inserted to hold the bound", len(result.reinserted))
        st.pyplot(viz.dendrogram(red))
        if result.clusters:
            table = pd.DataFrame(
                {
                    "kept": [c[0] for c in result.clusters],
                    "dropped": [", ".join(c[1:]) for c in result.clusters],
                }
            )
            st.dataframe(table, hide_index=True)
        if result.reinserted:
            st.info(
                "Re-inserted in importance order because the validation metrics left the "
                "bound: " + ", ".join(result.reinserted)
            )
    with st.expander("Notes"):
        st.markdown(
            "Features are clustered by Spearman rank correlation (distance 1 − ρ, average "
            "linkage). Within a cluster that merges below the cut only the most important "
            "feature is kept. If the validation metrics then leave the bound, dropped features "
            "are re-inserted in importance order until the bound holds again."
        )

    # ---- S6
    st.header(f"4. {STAGE_TITLES['final']} ({n['final']})")
    cols = st.columns(3)
    cols[0].metric("Final features", n["final"], f"{n['final'] - n['full']}", delta_color="off")
    cols[1].metric("Total reduction", f"{result.reduction * 100:.1f}%")
    cols[2].metric("Bound satisfied", "yes" if result.final_bound_satisfied else "no")
    st.subheader("Validation split")
    render_metrics(result, "final", "val", reference=full_val)
    st.subheader("Test split (evaluated once, after all choices were made)")
    render_metrics(result, "final", "test", reference=result.metrics["full"]["test"])
    render_confusions(result, "final")
    summary = result.summary()
    for name in METRIC_LABELS:
        summary[name] = (summary[name] * 100).round(2)
    st.dataframe(summary.rename(columns=METRIC_LABELS), hide_index=True)
    st.caption("All metrics in percent.")

    # ---- distributions
    st.header("5. Final features per class")
    X, y, classes = result.X_final, result.y, result.classes
    heatmaps = viz.class_heatmaps(X, y, classes)
    histograms = viz.class_histograms(X, y, classes, max_features=MAX_HISTOGRAM_FEATURES)
    if n["final"] > MAX_HISTOGRAM_FEATURES:
        st.caption(f"Histograms show the {MAX_HISTOGRAM_FEATURES} most important final features.")
    columns = st.columns(2)
    for i, name in enumerate(classes):
        with columns[i % 2]:
            st.plotly_chart(heatmaps[name])
    columns = st.columns(2)
    for i, name in enumerate(classes):
        with columns[i % 2]:
            st.pyplot(histograms[name])
    with st.expander("Notes"):
        st.markdown(
            "Heatmaps show the Spearman correlation between the final features within one "
            "class; red is positive, blue negative. Histograms show the raw feature values per "
            "class (categorical columns appear as integer codes)."
        )

    # ---- LDA
    st.header("6. Linear discriminant analysis")
    available = viz.lda_available_components(len(classes), n["final"])
    if available >= 2:
        st.plotly_chart(viz.lda_scatter(X, y, classes, n_components=2))
    if available >= 3:
        st.plotly_chart(viz.lda_scatter(X, y, classes, n_components=3))
    if available < 2:
        st.info("A 2-D LDA plot needs at least 3 classes (and 2 features); 3-D needs 4 classes.")
    with st.expander("Notes"):
        st.markdown(
            "LDA projects the final features onto the directions that best separate the "
            "classes (at most K − 1 for K classes). Clear clusters mean the final features "
            "separate the classes linearly; overlaps point to the samples the model struggles "
            "with."
        )

    # ---- UMAP
    if opts["umap"]:
        st.header("7. UMAP")
        with st.spinner("Computing UMAP embeddings…"):
            for dims in (2, 3):
                st.plotly_chart(
                    viz.umap_scatter(
                        X,
                        y,
                        classes,
                        n_components=dims,
                        n_neighbors=opts["umap_neighbors"],
                        random_state=opts["random_state"],
                    )
                )
        with st.expander("Notes"):
            st.markdown(
                "UMAP arranges samples by similarity without using the labels. If the final "
                "features carry the class structure, the classes separate in the embedding. "
                "Larger n_neighbors emphasises global structure, smaller values local structure."
            )

    # ---- outputs
    st.header("Final features")
    st.dataframe(result.features_frame(), hide_index=True)
    left, right = st.columns(2)
    left.download_button(
        "Download features.csv",
        frame_to_csv(result.features_frame()),
        "features.csv",
        "text/csv",
        key="download-features",
    )
    right.download_button(
        "Download result.json",
        json.dumps(result.to_dict(), indent=2).encode("utf-8"),
        "fastforest_result.json",
        "application/json",
        key="download-json",
    )


# -------------------------------------------------------------------------------- page
def main() -> None:
    st.title("FastForest")
    st.markdown("### *Feature reduction for high-dimensional tabular data with random forests*")
    with st.expander("How it works"):
        st.markdown(
            """
            1. **Split** the rows into training, validation and test sets (stratified by class).
            2. **Train** a random forest on all features and rank them by importance.
            3. **Search** for the smallest importance threshold whose forest keeps the
               validation metrics within a bound of the full model.
            4. **Remove redundancy**: cluster the kept features by Spearman correlation and keep
               the most important member of each cluster; re-insert features if the bound breaks.
            5. **Evaluate once on the test split**, which no step above has seen, and visualise
               the final features with per-class heatmaps and histograms, LDA and UMAP.
            """
        )
    opts = sidebar_options()

    source = st.radio("Data", ["Upload a file", "Use the bundled sample dataset"], horizontal=True)
    frame = None
    data_key = None
    if source == "Upload a file":
        upload = st.file_uploader(
            "CSV, TSV or Parquet with one class column", type=["csv", "tsv", "txt", "parquet", "gz"]
        )
        if upload is not None:
            data = upload.getvalue()
            frame = read_uploaded(data, upload.name)
            data_key = hashlib.sha1(data).hexdigest()
    else:
        frame = sample_frame()
        data_key = "sample"
        st.caption(
            "Reaction abundances of 1,000 microbial genomes across four body sites: a 500-feature "
            "subsample of the dataset used in the FastForest paper. Target column: "
            f"**{SAMPLE_TARGET}**."
        )
        st.download_button(
            "Download the sample as CSV",
            frame_to_csv(frame),
            "reaction_abundance_body_site_sample.csv",
            "text/csv",
            key="download-sample",
        )

    if frame is None:
        st.info("Upload a file or pick the sample dataset to get started.")
        return

    st.write(f"{frame.shape[0]:,} rows × {frame.shape[1] - 1:,} features + target")
    columns = list(frame.columns)
    default = columns.index(SAMPLE_TARGET) if SAMPLE_TARGET in columns else len(columns) - 1
    target = st.selectbox("Target (class) column", columns, index=default)

    test_idx = None
    if opts["split"] == "index":
        raw = st.text_input("Test row positions (comma-separated, 0-based)")
        try:
            test_idx = [int(x) for x in raw.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            st.error("Test row positions must be integers.")
            return
        if not test_idx:
            st.info("Enter the test rows to continue.")
            return

    config_key = json.dumps(
        {"data": data_key, "target": target, "opts": opts, "test_idx": test_idx}, sort_keys=True
    )
    run = st.button("Run FastForest", type="primary")
    if run:
        try:
            result = run_pipeline(frame, target, opts, test_idx)
        except ValueError as err:
            st.error(str(err))
            return
        st.session_state["result"] = result
        st.session_state["result_key"] = config_key

    if st.session_state.get("result_key") == config_key and "result" in st.session_state:
        render_results(st.session_state["result"], opts)
    elif "result" in st.session_state:
        st.info("Parameters or data changed. Press *Run FastForest* to recompute.")


main()
