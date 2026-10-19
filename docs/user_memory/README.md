# User Memory Evaluation

[中文版](./README_zh.md) · [Documentation index](../README.md) · [Results](./results.md)

## Overview

This track evaluates memory backend APIs through a shared benchmark pipeline
and adapter layer. Select a backend with `--lib` to evaluate memory products,
self-hosted frameworks, or custom adapters under the same benchmark flow.

There are 15 adapter entries covering 14 memory solutions (Zep and Graphiti
have separate entries). Supported benchmarks are LoCoMo, LongMemEval, BEAM,
PersonaMem v2, and HaluMem.

Read this guide in order for a first run. For an existing setup, jump to
[advanced runs](#advanced-runs), [outputs and results](#outputs-and-results),
or [extensions and contribution](#extensions-and-contribution).
Run all commands from the repository root.

## Environment and configuration

<a id="1-create-environment"></a>
### Install dependencies

Use a dedicated User Memory environment:

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
python -m pip install -r requirements_user_memory.txt
```

Install Agent Memory dependencies in a separate environment, as described in the
[Agent Memory guide](../agent_memory/README.md).

<a id="2-configure-credentials"></a>
### Configure credentials

Copy the template for the backend you want to evaluate, then fill in its values.
For example:

```bash
cp env_examples/.env.memos .env.memos
```

Configure the answer model, judge model where applicable, and backend credentials:

- `ANSWER_MODEL`, `ANSWER_API_KEY`, `ANSWER_BASE_URL`
- `EVAL_MODEL`, `EVAL_API_KEY`, `EVAL_BASE_URL` for LLM judging
- Backend credentials such as `MEMOS_API_KEY` or `MEM0_API_KEY`

See the [template index](../../env_examples/README.md) and
[parameter reference](../../env_examples/PARAMETERS.md) for backend-specific
settings, defaults, and benchmark applicability. PersonaMem v2 uses option
matching, not an LLM judge. Do not commit filled credential files.

<a id="3-prepare-data"></a>
## Data preparation

Download only the benchmarks you plan to run:

```bash
# LoCoMo
python data/locomo/prepare_locomo.py

# LongMemEval-S
python data/longmemeval/prepare_longmemeval.py

# BEAM 100K
python data/beam/prepare_beam.py

# PersonaMem v2
python data/personamem_v2/prepare_personamem.py

# HaluMem-Medium
python data/halumem/prepare_halumem.py
```

Data is downloaded on demand and is not committed to this repository. See the
[benchmark descriptions](#benchmarks), dataset READMEs, and
[third-party notices](../../THIRD_PARTY_NOTICES.md) for data preparation and licenses.

## Quick start

After configuring the backend and preparing LongMemEval data, check ingestion
and search for one conversation:

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos --version smoke_lme \
  --streaming 1 --start-idx 0 --end-idx 0 --to-step 2
```

This calls a real backend and may incur charges. The index range selects the
first evaluation conversation. Streaming mode adds, searches, saves results,
and deletes that unit's backend memory. `--to-step 2` stops before answer
generation and judging; this check does not produce a full benchmark score.
Use a dedicated evaluation account.

For complete scored runs, see [advanced runs](#advanced-runs). CLI options are
also available through each runner's `--help`.

## Benchmarks

### LoCoMo

Long-conversation QA covering multi-hop recall, temporal reasoning, and
open-domain memory use. The reproduced results exclude adversarial questions.
See [data and license notes](../../data/locomo/README.md).

### LongMemEval

Cross-session long-term memory covering knowledge updates, temporal reasoning,
and preferences. The shared loader uses `longmemeval_s_cleaned.json`, removing
known bad special tokens consistently for ingestion and search.
See [data and license notes](../../data/longmemeval/README.md).

### BEAM

Large-scale long-memory QA. The runner supports 100K (approximately 128K
tokens), 500K, 1M, and 10M scales, with per-nugget LLM judging.
See [data and license notes](../../data/beam/README.md).

<a id="personamem-v2"></a>
### PersonaMem v2

Personalized memory and preference-aware multiple-choice QA. Scoring matches the
selected option against the gold answer without an LLM judge.
See [data and license notes](../../data/personamem_v2/README.md).

### HaluMem

Memory robustness covering hallucinations, memory boundaries, conflicts,
multi-hop inference, and dynamic updates.
See [data and license notes](../../data/halumem/README.md).

## Advanced runs

<a id="4-run-evaluations"></a>
### Full benchmark runs

Each command below evaluates its benchmark's configured dataset, not a small
smoke subset. Prepare the corresponding data and credentials first.

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
./scripts/run_beam_eval.sh --lib memos --env .env.memos
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
```

Replace `memos` and the env file with the desired backend. Set `--version <name>`
to isolate run artifacts. Shared defaults and benchmark-specific options are
maintained in the [parameter reference](../../env_examples/PARAMETERS.md).

<a id="pipeline"></a>
### Pipeline and scoring

The standard retrieval-based flow is:

```text
Dataset → Ingest (add) → Search → Answer → Evaluate → Metrics → Report
             ↕             ↕
               Memory backend
```

- Ingestion and search call the selected adapter's `add()` and `search()`.
- Answer generation normally uses the configured OpenAI-compatible answer model.
  Backend-specific direct-answer modes may replace or skip that stage; disclose
  the selected mode when reporting results.
- LoCoMo, LongMemEval, and HaluMem use LLM judging, with additional metrics where
  implemented. BEAM uses LLM-based nugget scoring. PersonaMem v2 uses option matching.
- Runners support stage checkpoints and replay. A partial run is not a complete
  benchmark result.

### Streaming and partial runs

LongMemEval, BEAM, PersonaMem v2, and HaluMem support `--streaming 1`.
For each unit, streaming runs add, search, save, and delete before proceeding.
It supports `--start-idx`, `--end-idx`, `--restart-unit`, `--no-resume`,
and `--skip-failed-streaming`; check the selected runner's `--help` for semantics.

Use `--to-step 2` for ingestion/search checks. This limits stages, not dataset
size; without an explicit supported subset selector, it can still process the
whole dataset. The [quick start](#quick-start) combines both limits for LongMemEval.

### Replay

Resume later stages in an existing run by using the same backend/version and
keeping the prerequisite artifacts. For example, to regenerate answers onward:

```bash
./scripts/run_lme_eval.sh --lib memos --env .env.memos \
  --version existing_run --from-step 3
```

Replace `existing_run` with your saved run version. Explicit `--from-step`
forces that stage and subsequent stages to run again.

By contrast, `--replay` loads a saved experiment configuration and creates a new
timestamped run version. It does not copy earlier-stage artifacts into the new
directory. To repeat a complete run:

```bash
./scripts/run_lme_eval.sh --env .env.memos \
  --replay results/lme/{LIB}-{VERSION}/
```

Other runners accept `--replay` with their corresponding result directory.
Replace `{LIB}-{VERSION}` with the saved backend key and run version. Use the
`--env FILE --replay DIR` ordering shown above; the backend key and run settings
are restored from the saved configuration, so do not add `--lib`.
Only use trusted configurations: replay sources `experiment_config.sh` as shell code.
Record any configuration changes and disclose checkpoint reuse or re-evaluation
when submitting scores.

<a id="cleanup"></a>
### Cleanup

Preview the evaluation user IDs before deleting backend memory from a run:

```bash
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version '<run-version>' --datasets locomo,lme,beam,pmv2,hm --dry-run
./scripts/run_memory_clear.sh --lib memos --env .env.memos --version '<run-version>' --datasets locomo,lme,beam,pmv2,hm --yes
```

Replace `<run-version>` with the intended run version. `--dry-run` does not
delete data; actual deletion requires `--yes`.
For non-default BEAM or HaluMem datasets, use repeatable `--beam-scale`
or `--halumem-variant` options as appropriate. Verify the account and run version.

<a id="benchmark-results"></a>
## Outputs and results

Artifacts are written under these directories, using the backend key and run version:

| Benchmark | Result directory |
|---|---|
| LoCoMo | `results/locomo/{LIB}-{VERSION}/` |
| LongMemEval | `results/lme/{LIB}-{VERSION}/` |
| BEAM | `results/beam/{LIB}-{VERSION}/` |
| PersonaMem v2 | `results/pmv2/{LIB}-{VERSION}/` |
| HaluMem | `results/halumem/{LIB}-{VERSION}/` |

See [benchmark results](./results.md) for the public snapshot, evaluation
configuration, aggregation rules, and context-token definitions.
Reproduced Results and Published Reference Results have different evidence
and comparability requirements; do not mix them.

<a id="supported-memory-backends"></a>
## Extensions and contribution

The adapter layer provides common add/search operations and user-scoped cleanup.
Supported backend keys are:

| `--lib` | Backend |
|---|---|
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

Follow the [adapter checklist](../../CONTRIBUTING.md#adding-a-user-memory-adapter)
for integration, configuration, isolation, and testing, and the
[result submission requirements](../../CONTRIBUTING.md#submitting-benchmark-results)
for published or reproduced scores.

<a id="project-layout"></a>
The root [repository layout](../../README.md#repository-layout) describes where
code, configuration, and data preparation live.

<a id="verification"></a>
## Verification and licenses

Use a separate development environment with Python 3.11 or 3.12 for offline checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/check_doc_links.py
```

The [test guide](../../scripts/tests/README.md) distinguishes offline tests,
repository checks, and real-backend integration tests. Offline CI does not
validate benchmark scores or call live services.

<a id="license"></a>
See [LICENSE](../../LICENSE) for the code license. External datasets keep their
upstream licenses; see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
