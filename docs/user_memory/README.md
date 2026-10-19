# User Memory Evaluation

[中文版](./README_zh.md)

[Documentation index](../README.md)

This document covers the User Memory Evaluation track in OmniMemEval. This track evaluates memory backend APIs through a shared benchmark pipeline and a common adapter layer. Users can switch memory backends with `--lib` and compare mainstream memory products, self-hosted memory frameworks, and custom adapters under the same benchmark flow.

The adapter layer covers 14 mainstream memory solutions through 15 adapter entries, including MemOS, Mem0, Zep/Graphiti, Supermemory, EverOS, Letta, Hindsight, Cognee, Viking Memory, Memori, MemMachine, MemoryLake, Backboard.io, and mem9. This track supports five benchmark tasks: LoCoMo, LongMemEval, BEAM, PersonaMem v2, and HaluMem. These tasks cover complementary
long-term memory capabilities, including conversation recall, cross-session
updates, large-scale retrieval, personalization, and robustness under
hallucination, conflict, and dynamic-update scenarios.

Benchmark coverage:

- [LoCoMo](#locomo): long-conversation QA for multi-hop recall, temporal
  reasoning, and open-domain memory use.
- [LongMemEval](#longmemeval): cross-session long-term memory with knowledge
  updates, temporal reasoning, and preference questions.
- [BEAM](#beam): large-scale memory retrieval from 128K to 10M token contexts.
- [PersonaMem v2](#personamem-v2): personalized memory evaluation focused on
  preferences, sensitive information, and user-specific behavior.
- [HaluMem](#halumem): robustness evaluation for memory hallucination, boundary
  detection, conflicts, multi-hop inference, and dynamic updates.

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
pip install -r requirements_user_memory.txt
```

AgentBench dependencies live in `requirements_agentbench.txt` and should be installed in a separate Agent Memory environment.

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

See [env_examples/README.md](../../env_examples/README.md) and
[env_examples/PARAMETERS.md](../../env_examples/PARAMETERS.md).

### 3. Prepare Data

```bash
# LoCoMo
python data/locomo/prepare_locomo.py

# LongMemEval S
python data/longmemeval/prepare_longmemeval.py

# BEAM 100K
python data/beam/prepare_beam.py

# PersonaMem v2
python data/personamem_v2/prepare_personamem.py

# HaluMem Medium
python data/halumem/prepare_halumem.py
```

Benchmark data is downloaded on demand and is not committed to this repository.
See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) and the dataset README
files for upstream dataset licenses and redistribution notes.

### 4. Run Evaluations

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
./scripts/run_beam_eval.sh --lib memos --env .env.memos
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
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

Streaming mode is available for LongMemEval, BEAM, PersonaMem v2, and
HaluMem. In streaming mode, OmniMemEval runs add, search, save, and delete for
each benchmark unit before moving to the next unit. Use `--streaming 1` on the
corresponding runner:

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --streaming 1
./scripts/run_beam_eval.sh --lib memos --env .env.memos --streaming 1
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --streaming 1
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --streaming 1
```

Streaming runs support `--start-idx`, `--end-idx`, `--restart-unit`,
`--no-resume`, and `--skip-failed-streaming`.

Minimal smoke commands:

```bash
# LoCoMo: run ingestion and search only
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --version smoke_locomo --to-step 2

# LongMemEval: run one streaming conversation through search only
./scripts/run_lme_eval.sh --lib memos --env .env.memos --version smoke_lme \
  --streaming 1 --start-idx 0 --end-idx 0 --to-step 2

# BEAM: run ingestion and search only on the default 100K scale
./scripts/run_beam_eval.sh --lib memos --env .env.memos --version smoke_beam --to-step 2

# PersonaMem v2: run ingestion and search only
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --version smoke_pmv2 --to-step 2

# HaluMem: run ingestion and search only
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --version smoke_hm --to-step 2
```

Replay later stages from an existing result directory:

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos --replay results/locomo/{LIB}-{VERSION}/
./scripts/run_lme_eval.sh --lib memos --env .env.memos --replay results/lme/{LIB}-{VERSION}/
./scripts/run_beam_eval.sh --lib memos --env .env.memos --replay results/beam/{LIB}-{VERSION}/
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --replay results/pmv2/{LIB}-{VERSION}/
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --replay results/halumem/{LIB}-{VERSION}/
```

## Benchmark Results

See [benchmark results](./results.md) for the public
result snapshot reproduced under OmniMemEval's shared evaluation setup. The
document includes reproduced scores, context-token metrics, deployment notes,
published reference scores, and reproduction commands for the currently public
benchmark pipelines.

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
[data/locomo/README.md](../../data/locomo/README.md).

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
special tokens and applies the same cleaned data to ingestion and search. Data
and license notes live in
[data/longmemeval/README.md](../../data/longmemeval/README.md).

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

Results: `results/lme/{LIB}-{VERSION}/`

Replay later stages:

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --replay results/lme/{LIB}-{VERSION}/
```

<a id="beam"></a>
### BEAM

BEAM evaluates long-term memory at 128K, 500K, 1M, and 10M token scales with
per-nugget LLM-as-Judge scoring. Data and license notes live in
[data/beam/README.md](../../data/beam/README.md).

```bash
./scripts/run_beam_eval.sh --lib memos --env .env.memos
```

Results: `results/beam/{LIB}-{VERSION}/`

Replay later stages:

```bash
./scripts/run_beam_eval.sh --lib memos --env .env.memos --replay results/beam/{LIB}-{VERSION}/
```

<a id="personamem-v2"></a>
### PersonaMem v2

PersonaMem v2 evaluates personalized memory and preference-aware multiple-choice
QA. Data and license notes live in
[data/personamem_v2/README.md](../../data/personamem_v2/README.md).

```bash
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
```

Results: `results/pmv2/{LIB}-{VERSION}/`

Replay later stages:

```bash
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos --replay results/pmv2/{LIB}-{VERSION}/
```

<a id="halumem"></a>
### HaluMem

HaluMem evaluates memory hallucination, conflict handling, dynamic updates, and
memory-boundary robustness. Data and license notes live in
[data/halumem/README.md](../../data/halumem/README.md).

```bash
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
```

Results: `results/halumem/{LIB}-{VERSION}/`

Replay later stages:

```bash
./scripts/run_halumem_eval.sh --lib memos --env .env.memos --replay results/halumem/{LIB}-{VERSION}/
```

## Cleanup

To delete backend memory created by a run:

```bash
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme,beam,pmv2,hm --dry-run
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version <name> --datasets locomo,lme,beam,pmv2,hm --yes
```

`--dry-run` prints target ids without deleting data. Destructive deletion
requires `--yes`. Use repeatable `--beam-scale` and `--halumem-variant` when
clearing non-default BEAM or HaluMem datasets.

## Project Layout

```text
OmniMemEval/
├── data/
│   ├── beam/
│   ├── halumem/
│   ├── locomo/
│   ├── longmemeval/
│   └── personamem_v2/
├── docs/
│   └── user_memory/
│       ├── README.md
│       ├── README_zh.md
│       └── results.md
├── env_examples/
├── scripts/
│   ├── client_factory/
│   ├── beam/
│   ├── halumem/
│   ├── locomo/
│   ├── longmemeval/
│   ├── personamem_v2/
│   ├── tests/
│   ├── utils/
│   ├── run_beam_eval.sh
│   ├── run_halumem_eval.sh
│   ├── run_locomo_eval.sh
│   ├── run_lme_eval.sh
│   ├── run_pmv2_eval.sh
│   └── run_memory_clear.sh
├── README.md
├── README_zh.md
├── THIRD_PARTY_NOTICES.md
├── requirements_user_memory.txt  # User Memory dependencies
└── requirements_agentbench.txt   # AgentBench dependencies
```

## Verification

```bash
git ls-files -z '*.sh' | xargs -0 -n 1 bash -n
conda run -n omnimemeval python -m compileall -q scripts data
conda run -n omnimemeval python -m pip install -r requirements-dev.txt
conda run -n omnimemeval python -m pytest -q
```

See the [test guide](../../scripts/tests/README.md) for offline suites, CI checks,
and backend integration smoke tests.

## License

See [LICENSE](../../LICENSE). Third-party benchmark data keeps its upstream license;
the OmniMemEval code license does not relicense external datasets. See
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
