# OmniMemEval

[中文版](./README_zh.md)

OmniMemEval is a standardized evaluation framework for memory system APIs. It is
designed to support multiple memory benchmarks through a shared evaluation
pipeline and a common adapter layer for mainstream memory backends. Users can
switch memory backends with `--lib` and compare mainstream memory products,
self-hosted memory frameworks, and custom adapters under the same benchmark
flow. The current open-source release includes LoCoMo and LongMemEval
pipelines, with additional benchmark support planned.

Current benchmark coverage:

- [LoCoMo](#locomo): long-conversation QA with multi-hop and temporal recall.
- [LongMemEval](#longmemeval): 500 long-term memory questions across sessions.

## Pipeline

OmniMemEval benchmark pipelines use the same staged flow:

```text
┌──────────────────┐
│ Benchmark Data   │
│ dataset-specific │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      add()       ┌──────────────────┐
│ 1. Ingest        ├─────────────────▶│ Memory Backend   │
│ conversations    │                  │ selected by --lib│
└────────┬─────────┘                  └────────┬─────────┘
         │                                     │
         ▼                                     │ search()
┌──────────────────┐                           │
│ 2. Search        │◀──────────────────────────┘
│ retrieve context │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      ANSWER LLM
│ 3. Answer        ├─────────────────▶ generated answers
│ generation       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      EVAL LLM / NLP
│ 4. Evaluation    ├─────────────────▶ judged records
│ LLM-as-Judge     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Metrics       │
│ accuracy/latency │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Report        │
│ markdown/results │
└──────────────────┘
```

- Ingest calls the selected memory client `add()`.
- Search calls the selected memory client `search()`.
- Answer generation uses an OpenAI-compatible ANSWER model.
- Evaluation uses an OpenAI-compatible EVAL model for LLM-as-Judge plus NLP metrics.
- Metrics and reports are written under `results/<benchmark>/<LIB>-<VERSION>/`.

The shell runners and Python stages support checkpoint/resume so interrupted
runs can continue from the last completed step.

## Quick Start

### 1. Create Environment

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
pip install -r requirements.txt
```

### 2. Configure Credentials

Start from a product-specific template:

```bash
cp env_examples/.env.memos .env.memos
```

Fill in the required memory product credentials and the OpenAI-compatible
ANSWER/EVAL LLM settings:

- `ANSWER_MODEL`, `ANSWER_API_KEY`, `ANSWER_BASE_URL`
- `EVAL_MODEL`, `EVAL_API_KEY`, `EVAL_BASE_URL`
- Product-specific memory credentials such as `MEMOS_API_KEY` or `MEM0_API_KEY`

See [env_examples/README.md](./env_examples/README.md) and
[env_examples/PARAMETERS.md](./env_examples/PARAMETERS.md).

### 3. Prepare Data

```bash
# LoCoMo
python data/locomo/prepare_locomo.py

# LongMemEval S
python data/longmemeval/prepare_longmemeval.py
```

Benchmark data is downloaded on demand and is not committed to this repository.
LoCoMo is licensed under CC BY-NC 4.0, and LongMemEval is MIT-licensed; see
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) and the dataset README files.

### 4. Run Evaluations

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

Useful shared options:

| Option | Purpose |
|--------|---------|
| `--version <name>` | Result directory suffix. Defaults to `omnimemeval_<date>`. |
| `--from-step N` / `--to-step N` | Run a subset of pipeline steps. |
| `--replay <result_dir>` | Recompute later stages from an existing result directory. |
| `--top-k N` | Search result count. Overrides `TOPK` from the env file. |
| `--llm-workers N` | Concurrent answer/eval LLM workers. |
| `--allow-empty-search 1` | Allow successful runs with no raw memory returned. |
| `--skip-failed-search 1` | Mark failed search items as skipped instead of failing the step. |
| `--skip-failed-answer 1` | Mark failed answer items as skipped instead of failing the step. |
| `--skip-failed-judge 1` | Mark failed judge items as skipped instead of failing the step. |

LongMemEval also supports per-conversation streaming:

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --streaming 1
```

Streaming performs add, search, save, and delete for each conversation before
moving to the next one. It supports `--start-idx`, `--end-idx`,
`--restart-unit`, `--no-resume`, and `--skip-failed-streaming`.

Minimal smoke commands:

```bash
# LoCoMo: run ingestion and search only
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --version smoke_locomo --to-step 2

# LongMemEval: run one streaming conversation through search only
./scripts/run_lme_eval.sh --lib memos --env .env.memos --version smoke_lme \
  --streaming 1 --start-idx 0 --end-idx 0 --to-step 2
```

Replay later stages from an existing result directory:

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --replay results/locomo/{LIB}-{VERSION}/
./scripts/run_lme_eval.sh --lib memos --env .env.memos --replay results/lme/{LIB}-{VERSION}/
```

## Benchmark Results

See [docs/benchmark-results.md](./docs/benchmark-results.md) for the public
LoCoMo and LongMemEval result snapshot reproduced under OmniMemEval's shared
evaluation setup. The document includes reproduced scores, context-token
metrics, deployment notes, published reference scores, and reproduction
commands.

## Supported Memory Backends

The public adapter layer exposes a common `add()` / `search()` / `delete()`
interface for mainstream memory products and self-hosted memory frameworks:

Use `--lib` to run the same benchmark against different memory solutions
without changing the benchmark stages, prompt flow, or metric calculation.

| `--lib` | Adapter |
|---------|---------|
| `memos` | MemOS |
| `mem0` | Mem0 |
| `zep` | Zep |
| `supermemory` | Supermemory |
| `everos` | EverOS |
| `letta` | Letta |
| `hindsight` | Hindsight |
| `graphiti` | Zep Graphiti local/self-hosted |
| `cognee` | Cognee |
| `viking` | Viking Memory |
| `memori` | Memori |
| `memmachine` | MemMachine |
| `memorylake` | MemoryLake |
| `backboard` | Backboard.io |
| `mem9` | mem9 |

## Benchmarks

<a id="locomo"></a>
### LoCoMo

LoCoMo evaluates long-conversation memory with multi-hop, temporal, and
open-domain QA. Data and license notes live in
[data/locomo/README.md](./data/locomo/README.md).

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

Results: `results/locomo/{LIB}-{VERSION}/`

Replay later stages:

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --replay results/locomo/{LIB}-{VERSION}/
```

<a id="longmemeval"></a>
### LongMemEval

LongMemEval evaluates long-term memory across sessions. OmniMemEval loads
`longmemeval_s_cleaned.json` through a shared loader that removes known bad
special tokens and applies the same cleaned data to ingestion and search.

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

Results: `results/lme/{LIB}-{VERSION}/`

Replay later stages:

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --replay results/lme/{LIB}-{VERSION}/
```

## Cleanup

To delete backend memory created by a run:

```bash
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme --dry-run
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme --yes
```

`--dry-run` prints target ids without deleting data. Destructive deletion
requires `--yes`.

## Project Layout

```text
OmniMemEval/
├── data/
│   ├── locomo/
│   └── longmemeval/
├── docs/
│   └── benchmark-results.md
├── env_examples/
├── scripts/
│   ├── client_factory/
│   ├── locomo/
│   ├── longmemeval/
│   ├── tests/
│   ├── utils/
│   ├── run_locomo_eval.sh
│   ├── run_lme_eval.sh
│   └── run_memory_clear.sh
├── README.md
├── README_zh.md
├── THIRD_PARTY_NOTICES.md
└── requirements.txt
```

## Verification

```bash
bash -n scripts/_experiment_utils.sh scripts/run_locomo_eval.sh scripts/run_lme_eval.sh scripts/run_memory_clear.sh
conda run -n omnimemeval python -m compileall -q scripts
conda run -n omnimemeval python -m unittest discover -s scripts/tests -p 'test_*.py'
```

## License

See [LICENSE](./LICENSE). Third-party benchmark data keeps its upstream license;
the OmniMemEval code license does not relicense external datasets. See
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
