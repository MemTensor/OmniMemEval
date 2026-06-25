# OmniMemEval Tests

The public test suite covers the LoCoMo and LongMemEval pipelines, shared
utilities, memory adapter configuration, and integration smoke helpers.

```bash
conda run -n omnimemeval python -m unittest discover -s scripts/tests -p 'test_*.py'
conda run -n omnimemeval python -m unittest discover -s scripts/tests/unit -p 'test_*.py'
conda run -n omnimemeval python -m unittest discover -s scripts/tests/pipeline -p 'test_*.py'
```

Integration smoke tests require real backend credentials:

```bash
conda run -n omnimemeval python scripts/tests/integration/smoke_clients.py --lib memos --env .env.memos
```
