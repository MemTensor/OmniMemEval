# OmniMemEval

[中文版](./README_zh.md)

OmniMemEval is an evaluation framework for memory systems and memory-augmented agents. It provides two complementary evaluation tracks:

| Track | Evaluation Target | Benchmarks / Domains | Documentation |
|---|---|---|---|
| User Memory Evaluation | Memory backend APIs exposed through `add()` and `search()` | LoCoMo, LongMemEval, BEAM, PersonaMem v2, HaluMem | [docs/user_memory/README.md](./docs/user_memory/README.md) |
| Agent Memory Evaluation | Agent runtimes equipped with memory plugins | AgentBench domains: reasoning, information retrieval, knowledge work, code implementation, software engineering | [docs/agent_memory/README.md](./docs/agent_memory/README.md) |

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
```

Fill in the memory backend credentials and `ANSWER_*` / `EVAL_*` model settings
in `.env.memos` before continuing. Then prepare data and run the full LoCoMo
evaluation (this calls the configured backend and model APIs):

```bash
python data/locomo/prepare_locomo.py
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

### Agent Memory Evaluation

```bash
cp env_examples/.env.agent .env.agent
```

Complete the OpenClaw setup and credentials in the
[Agent Memory guide](./docs/agent_memory/README.md), then run one reasoning
task as a smoke check:

```bash
mkdir -p data/agentbench
hf download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --test-split 1 \
  --version smoke_agentbench \
  --trials 1 \
  --parallel 1
```

## Results

- User Memory public result snapshot: [docs/user_memory/results.md](./docs/user_memory/results.md)
- Agent Memory evaluation results: [docs/agent_memory/results.md](./docs/agent_memory/results.md)

Documentation index: [docs/README.md](./docs/README.md)

## Repository Layout

```text
configs/
  agentbench/                 # Agent Memory Evaluation configs
data/
  locomo/ longmemeval/ beam/  # User Memory benchmark data preparation
  personamem_v2/ halumem/
docs/
  README.md / README_zh.md    # Documentation index
  user_memory/                # User Memory guides and results
  agent_memory/               # Agent Memory guides, architecture, and results
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
requirements-dev.txt          # Offline tests and development checks
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the branch/PR workflow, adapter
integration, and benchmark-result submission requirements. The
[test guide](./scripts/tests/README.md) documents offline checks and CI.

## License

See [LICENSE](./LICENSE). Third-party benchmark data keeps its upstream license; the OmniMemEval code license does not relicense external datasets. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
