# LCM-X

The in-process `lcmx` adapter supports [LCM-X v0.28.0](https://github.com/electricsheephq/lcm-x/tree/709099b3a397a7893a7abae8802ebe7bf605f79e)
(commit `709099b3a397a7893a7abae8802ebe7bf605f79e`). The 2026-10-18 runs used
v0.28.0-rc1 (`559b2cb92031d94869f0c43651a85169fe1f45ab`); v0.28.0 adds only
a release-notes file to that tree, so the executable code is identical.

## Setup

From the OmniMemEval repository root, use Python 3.11 or newer:
```bash
git clone https://github.com/electricsheephq/lcm-x.git && git -C lcm-x checkout v0.28.0
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_user_memory.txt
python -m pip install -r lcm-x/requirements-semantic.txt
cp env_examples/.env.lcmx .env.lcmx
```

LCM-X is a source-imported plugin, not a pip-installable package. Its
[`requirements-semantic.txt`](https://github.com/electricsheephq/lcm-x/blob/709099b3a397a7893a7abae8802ebe7bf605f79e/requirements-semantic.txt)
requires `fastembed>=0.8.0,<1.0`. The adapter imports `benchmarking.longmemeval`
from `LCMX_REPO`; that module supplies the standalone `hermes_lcm` package alias,
so no Hermes host installation or plugin registration is needed.

Edit `.env.lcmx`: replace the checkout and store placeholders with absolute paths
to your `lcm-x` checkout and a writable, dedicated store root. Configure the
answer/judge endpoints and credentials locally for benchmark runs; the adapter
smoke below does not call either model. Never commit the populated environment.

Prepare the FastEmbed model once with network access, using the same cache for
download and execution (the template enables `HF_HUB_OFFLINE=1`):

```bash
export FASTEMBED_CACHE_PATH="$HOME/.cache/fastembed"
HF_HUB_OFFLINE=0 python -c 'import os; from fastembed import TextEmbedding; list(TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=os.environ["FASTEMBED_CACHE_PATH"], threads=2).embed(["cache warmup"]))'
set -a; source .env.lcmx; set +a
```

## Offline adapter smoke

Use a disposable store root; this command ingests one user, searches, then deletes
that user's store. It needs the prepared embedding cache, but no LLM credentials:

```bash
python - <<'PY'
import os
import tempfile
from scripts.client_factory.registry import create_client

with tempfile.TemporaryDirectory() as root:
    os.environ["LCMX_DB_ROOT"] = root
    client = create_client("lcmx")
    try:
        client.add([{"role": "user", "content": "My favorite drink is green tea."}], "smoke-1")
        assert "green tea" in client.search("favorite drink", "smoke-1", 20)
    finally:
        client.delete("smoke-1")
    assert client.search("favorite drink", "smoke-1", 20) == ""
print("add=1 search=1 delete=1")
PY
python -m unittest -v scripts.tests.unit.test_lcmx_client
```

User IDs must fullmatch `[A-Za-z0-9][A-Za-z0-9_.-]{0,199}`. Unsupported IDs
raise `ValueError` before user-specific filesystem or lock access, including on
empty adds. Each accepted ID names its own store directory and lock unchanged.

## Reported run configuration

The template explicitly records `LCMX_CONFIG=fleet`, `LCMX_CHUNK_POLICY=full`,
`LCMX_DELIVERY=full`, `LCMX_EMBED_PROVIDER=fastembed`,
`LCMX_EMBED_MODEL=BAAI/bge-small-en-v1.5`, `HF_HUB_OFFLINE=1`, `TOPK=20`, and
`LLM_WORKERS=10`. The adapter's default chunk policy remains `conversational`;
load the template to reproduce `full`. The answer model was
`gpt-4.1-mini-2025-04-14`; the judge was `gpt-4o-mini-2024-07-18`.

Run each suite from the repository root after configuring `.env.lcmx`:

```bash
./scripts/run_lme_eval.sh --lib lcmx --env .env.lcmx --version board --workers 2 --llm-workers 10 --top-k 20
./scripts/run_locomo_eval.sh --lib lcmx --env .env.lcmx --version board-rc1-20261018 --workers 2 --llm-workers 10 --top-k 20
./scripts/run_halumem_eval.sh --lib lcmx --env .env.lcmx --version board-rc1-20261018 --workers 2 --llm-workers 10 --top-k 20
./scripts/run_pmv2_eval.sh --lib lcmx --env .env.lcmx --version board-rc1-20261018 --workers 2 --llm-workers 10 --top-k 20
./scripts/run_beam_eval.sh --lib lcmx --env .env.lcmx --version board-rc1-100k-20261018 --workers 2 --llm-workers 10 --top-k 20 --scale 100k
```

These select LongMemEval-S, LoCoMo, HaluMem medium, PersonaMem v2, and BEAM 100K. `--version`
labels output directories; use a fresh label and store root for each run.
Benchmarks require network access for datasets and answer/judge API calls;
`HF_HUB_OFFLINE=1` refers to the prepared local embedding model.
