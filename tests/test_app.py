"""Streamlit AppTest smoke test on the bundled sample."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_runs_end_to_end_on_bundled_sample():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception

    at.number_input[0].set_value(20)  # trees per forest
    at.toggle[1].set_value(False)  # UMAP off (slow numba compile)
    at.radio[0].set_value("Use the bundled sample dataset")
    at.run()
    assert not at.exception
    target = next(s for s in at.selectbox if s.label == "Target (class) column")
    assert target.value == "Body site"

    run_button = next(b for b in at.button if b.label == "Run FastForest")
    run_button.click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    headers = [h.value for h in at.header]
    assert headers[0] == "Dataset"
    assert any(h.startswith("1. Model on all features (500)") for h in headers)
    assert any(h.startswith("4. Final model") for h in headers)
    assert "6. Linear discriminant analysis" in headers
    assert "7. UMAP" not in headers
    assert "Final features" in headers
    assert not at.error and not at.warning
    assert len(at.dataframe) >= 2
