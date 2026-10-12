"""Phase 1 smoke test: the bundled sample runs end to end. The full suite comes in Phase 2."""

from fastforest import FastForest, load_sample
from fastforest.metrics import METRIC_NAMES
from fastforest.result import STAGES


def test_sample_pipeline_runs():
    X, y = load_sample()
    result = FastForest(n_estimators=20, random_state=0).fit(X, y)

    assert result.features_final
    assert len(result.features_final) <= len(result.features_important) <= len(result.features_full)
    assert set(result.metrics) == set(STAGES)
    for stage in STAGES:
        for split in ("val", "test"):
            metrics = result.metrics[stage][split]
            assert set(metrics) == set(METRIC_NAMES)
            assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert result.X_final.shape == (len(y), len(result.features_final))
    assert "total" in result.timings
