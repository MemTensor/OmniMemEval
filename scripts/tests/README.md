# OmniMemEval Tests

The public test suite covers both evaluation tracks, shared utilities, memory
adapter configuration, pipeline contracts, and integration smoke helpers.

## Offline Tests

From the repository root, use Python 3.11 or 3.12 and install the lightweight
test dependencies. These tests use local fixtures and mocks; they do not
require backend credentials, downloaded benchmark datasets, or model services.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

`pytest.ini` selects the unit and pipeline-contract suites. Pytest runs both
the existing `unittest.TestCase` tests and the function-style AgentBench tests;
`unittest discover` alone does not execute the latter.

To run one group or inspect collection:

```bash
python -m pytest -q scripts/tests/unit
python -m pytest -q scripts/tests/pipeline
python -m pytest -q scripts/tests/unit/test_agentbench_*.py
python -m pytest --collect-only -q
```

## Repository Checks

GitHub Actions runs the offline suites on Python 3.11 and 3.12, compiles Python
files, checks each tracked shell script, and checks relative Markdown file
links. The link check does not validate external URLs or heading anchors.

```bash
python -m compileall -q scripts data
git ls-files -z '*.sh' | xargs -0 -n 1 bash -n
python scripts/check_doc_links.py
```

## Integration Smoke Tests

Install the relevant track's runtime dependencies and configure a dedicated
test account before running integration smoke tests. These call real backend
APIs and may create and delete memory records; they are not run by CI.

```bash
conda run -n omnimemeval python scripts/tests/integration/smoke_clients.py --lib memos --env .env.memos
```
