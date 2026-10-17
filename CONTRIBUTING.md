# Contributing to FastForest

Thank you for considering a contribution. Bug reports, feature requests and pull requests are
welcome on the [issue tracker](https://github.com/ThieleLab/streamlit_random_forest_analysis/issues).

## Development setup

```bash
git clone https://github.com/ThieleLab/streamlit_random_forest_analysis.git
cd streamlit_random_forest_analysis
python -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.12 .venv
pip install -e ".[app,bench,dev]"
```

Python 3.10 or newer is required.

## Checks that must pass

```bash
ruff check . && ruff format --check .
pytest -q
```

Run these before every commit. Continuous integration runs the same commands on Python 3.10,
3.11 and 3.12.

## Ground rules

- The core package `fastforest/` never imports `streamlit`. User-interface code lives in `app/`.
  Figure functions return matplotlib or plotly figure objects and never call `st.*` or
  `plt.show()`.
- Every public function has a NumPy-style docstring. Every bug fix comes with a regression test
  in `tests/`.
- Every stochastic step takes a `random_state`.
- Do not change the algorithm's behaviour silently. Any deviation from the documented method
  is listed in `CHANGELOG.md` under "Behaviour changes" and mentioned in the pull request.
- No Jupyter notebooks in the repository; put scripts in `scripts/` or `benchmarks/`.

## Pull requests

Work on a branch, keep commits small with clear messages, and open a pull request against
`master`. Describe what changed and why, and reference the issue it closes if there is one.

## Reporting bugs

Please include the FastForest version (`fastforest --version`), Python version, operating
system, the exact command or code, and the full error message. A minimal dataset that
reproduces the problem is the most helpful thing you can attach.

## Code of conduct

Be kind and constructive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
