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
empty adds. Storage directories and locks use the same filesystem-safe key:
the first 24 characters of the ID in lowercase, a hyphen, and the first 32 hex
characters of SHA-256 of the original ID's UTF-8 bytes. The digest preserves
case distinctions: `CaseUser` and `caseuser` remain separate even on a
case-insensitive filesystem. Delete operations use this same mapping.

**Compatibility:** stores created by earlier adapter versions under raw user
IDs are not reused by this mapping. Re-ingest the data with the updated adapter;
there is no automatic migration of those stores.

## Result provenance

The contributor-run results were produced by the LCM-X maintainers on
**2026-10-18**, using OmniMemEval harness commit
[`162640314863db64b5af62138b5b81776516d1ff`](https://github.com/electricsheephq/OmniMemEval/tree/162640314863db64b5af62138b5b81776516d1ff)
on upstream base `a5c8b4ba7d9566371e7fca2b20731c2214971149`.
The [public evidence release](https://github.com/electricsheephq/OmniMemEval/releases/tag/lcmx-v0.28.0-results-20261018)
provides the [results archive](https://github.com/electricsheephq/OmniMemEval/releases/download/lcmx-v0.28.0-results-20261018/lcmx-omnimemeval-results-20261018.tar.gz),
whose SHA-256 is `af4eebcb0909a86b5eb7969c4f167967b0d53b17559e7855e1cff58255b2682a`.
Its `MANIFEST.json` records the run date, commits, settings, commands, dataset
hashes and output hashes. The archive includes per-question outputs, stage
status files, answer-token receipts and the offline recomputation script.

The template requests the dated answer model `gpt-4.1-mini-2025-04-14` and judge
`gpt-4o-mini-2024-07-18` through a direct OpenAI endpoint. The reported runs
requested those dated models through **OpenRouter**, and the answer usage
receipts identify the resolved model as `openai/gpt-4.1-mini`. The receipts
establish that reported provider/model identifier and recorded token usage;
they do not establish the exact dated model revision or equivalence to a
direct OpenAI run. This limitation does not establish that a different model
was used.

### PersonaMem v2 snapshot

The archive's original manifest left this dataset unpinned. The complete input
snapshot has since been identified as
[`bowen-upenn/PersonaMem-v2` at `ed956dea41521fc4499acbc63f966e0fd3c053ba`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/tree/ed956dea41521fc4499acbc63f966e0fd3c053ba).
The [supplemental snapshot manifest](./lcmx-personamem-v2-manifest.json) pins the
CSV and all 200 referenced 32K chat-history files with individual SHA-256 hashes
and sizes. The CSV SHA-256 is
`95f2a8a324aab7baf2af937feae12731369e2abf7cad5ab3e170594cb25a3e52`.

All 201 retained inputs match their download metadata and the public pinned
revision's content hashes; they were downloaded on 2026-10-14, before the run.
All 5,000 saved response questions, categories, topics, options and gold labels
match the CSV, and the saved response file matches the public archive's hash.
The reproduced row is retained with this retrospective pin; the original
archive is unchanged. This identifies the inputs, rather than claiming a new
benchmark run. To reproduce the snapshot, fetch the CSV and the chat histories
from that exact revision and verify them against the supplemental manifest.

## Reported run configuration

The template explicitly records `LCMX_CONFIG=fleet`, `LCMX_CHUNK_POLICY=full`,
`LCMX_DELIVERY=full`, `LCMX_EMBED_PROVIDER=fastembed`,
`LCMX_EMBED_MODEL=BAAI/bge-small-en-v1.5`, `HF_HUB_OFFLINE=1`, `TOPK=20`, and
`LLM_WORKERS=10`. The adapter's default chunk policy remains `conversational`;
load the template to reproduce `full`. The requested answer model was
`gpt-4.1-mini-2025-04-14`; the requested judge was `gpt-4o-mini-2024-07-18`.
See the endpoint and model-resolution limitation above.

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
