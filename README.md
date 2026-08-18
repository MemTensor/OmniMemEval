# OmniMemEval

[中文版](./README_zh.md)

OmniMemEval is an evaluation framework for memory systems and memory-augmented agents. It provides two complementary evaluation tracks:

| Track | Evaluation Target | Benchmarks / Domains | Documentation |
|---|---|---|---|
| User Memory Evaluation | Memory backend APIs exposed through `add()` and `search()` | LoCoMo, LongMemEval, BEAM, PersonaMem v2, HaluMem | [docs/user_memory/README.md](./docs/user_memory/README.md) |
| Agent Memory Evaluation | Agent runtimes equipped with memory plugins | AgentBench domains: reasoning, information retrieval, knowledge work, code implementation, software engineering | [docs/agent_memory/README.md](./docs/agent_memory/README.md) |

## Evaluation Tracks

### User Memory Evaluation

User Memory Evaluation measures the capability of memory backend systems through a standardized API adapter layer. The same benchmark pipeline can be run against mainstream memory products, self-hosted memory frameworks, or custom adapters by selecting a backend with `--lib`.

The pipeline covers ingestion, retrieval, answer generation, LLM-as-Judge evaluation, metric aggregation, and report generation. Results are written under `results/<benchmark>/<LIB>-<VERSION>/`.

Start here: [docs/user_memory/README.md](./docs/user_memory/README.md)

### Agent Memory Evaluation

Agent Memory Evaluation measures the task performance of an agent runtime after a memory plugin is installed. The current implementation is based on AgentBench and evaluates OpenClaw across five task domains. The memory-plugin protocol includes memory cleanup, training, memory settling, backup, restore, and test execution.

Results are written under `results/agentbench/`.

Start here: [docs/agent_memory/README.md](./docs/agent_memory/README.md)

## Installation

User Memory Evaluation and Agent Memory Evaluation use separate environments. For User Memory Evaluation, create the base Python environment from the repository root:

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
pip install -r requirements_user_memory.txt
```

Agent Memory Evaluation recommends a separate `agentmem` environment because AgentBench domains require OpenClaw and additional domain dependencies:

```bash
conda create -n agentmem python=3.12 -y
conda activate agentmem
python -m pip install -U pip
pip install -r requirements_agentbench.txt
```

See [docs/agent_memory/README.md](./docs/agent_memory/README.md) for OpenClaw, system packages, and domain-specific setup.

Each evaluation track may require additional dependencies:

- User Memory Evaluation: memory backend credentials, ANSWER/EVAL LLM credentials, and benchmark-specific data preparation. See [docs/user_memory/README.md](./docs/user_memory/README.md).
- Agent Memory Evaluation: OpenClaw CLI, AgentBench data, and domain-specific dependencies such as Docker, LiveCodeBench, and BrowseComp-Plus indexing. See [docs/agent_memory/README.md](./docs/agent_memory/README.md).

## Quick Start

### User Memory Evaluation

```bash
cp env_examples/.env.memos .env.memos
python data/locomo/prepare_locomo.py
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

### Agent Memory Evaluation

```bash
cp env_examples/.env.agent .env.agent
python scripts/agentbench/prepare_data.py download --domains reasoning
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version smoke_agentbench \
  --trials 1 \
  --parallel 1
```

## Results

- User Memory public result snapshot: [docs/benchmark-results.md](./docs/benchmark-results.md)
- AgentBench evaluation results: [docs/agent_memory/eval_res.md](./docs/agent_memory/eval_res.md)

## Repository Layout

```text
configs/
  agentbench/                 # Agent Memory Evaluation configs
data/
  locomo/ longmemeval/ beam/  # User Memory benchmark data preparation
  personamem_v2/ halumem/
docs/
  user_memory/                # User Memory Evaluation guides (EN/ZH)
  agent_memory/               # Agent Memory Evaluation guides and results (EN/ZH)
  benchmark-results.md        # User Memory public result snapshot
env_examples/                 # Environment templates and parameter docs
scripts/
  agentbench/                 # AgentBench runner implementation
  client_factory/             # User Memory backend adapters
  locomo/ longmemeval/ beam/  # User Memory benchmark pipelines
  personamem_v2/ halumem/
  tests/                      # Test suite
  utils/                      # Shared utilities
requirements_user_memory.txt  # User Memory Evaluation dependencies
requirements_agentbench.txt   # Agent Memory Evaluation dependencies
```

## License

See [LICENSE](./LICENSE). Third-party benchmark data keeps its upstream license; the OmniMemEval code license does not relicense external datasets. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
