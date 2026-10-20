# Agent Memory Evaluation

[中文版](./README_zh.md) · [Documentation index](../README.md) · [Results](./results.md)

## Overview

This track evaluates an agent runtime with or without a memory plugin using
the AgentBench module in OmniMemEval. The current public runtime adapter covers
OpenClaw across five task domains.

There are two evaluation setups:

- Baseline: `test_only` or `train_then_test` without the target memory plugin.
- Memory plugin: `memory_train_backup_test`, including cleanup, training,
  settling, backup, restore, and testing.

Read this guide in order for a first run. For an existing setup, jump to
[advanced runs](#advanced-runs), [outputs and results](#outputs-and-results),
or [extensions and contribution](#extensions-and-contribution).
Run all commands from the repository root.

<a id="environment"></a>
## Environment and configuration

### Install dependencies

Use a dedicated Agent Memory environment, separate from User Memory:

```bash
conda create -n agentmem python=3.12 -y
conda activate agentmem
python -m pip install -U pip
python -m pip install -r requirements_agentbench.txt
npm install -g openclaw
```

Configure a valid OpenClaw model/provider, then verify that the CLI is available:

```bash
openclaw --version
openclaw agent --help
```

### Configure credentials

Copy the Agent Memory template, then fill in the settings for the selected domain:

```bash
cp env_examples/.env.agent .env.agent
```

AgentBench loads the project `.env.agent` by default; `--env FILE` loads an
additional env file. Keep these credentials separate from User Memory backend
files such as `.env.memos`.

| Purpose | Configuration |
|---|---|
| Agent model/provider | `configs/agentbench/agents/openclaw.yaml`, using `LLM_BASE_URL` and `LLM_API_KEY` |
| Judge | `JUDGE_MODEL`, `JUDGE_API_BASE`, `JUDGE_API_KEY` |
| Information retrieval embeddings | `IR_EMBEDDING_ENDPOINT`, `IR_EMBEDDING_MODEL`, `IR_EMBEDDING_API_KEY` |
| External evaluation service | `EVALUATION_API_BASE`, `EVALUATION_API_KEY`, `EVALUATION_MODEL_OWNER`, `EVALUATION_MODEL_NAME`, `EVALUATION_TIMEOUT`, `EVALUATION_MAX_RETRIES` |

Use the [env template](../../env_examples/.env.agent) and selected
`configs/agentbench/domains/*.yaml` as the field reference. Provider fields
not configured through the agent YAML/env settings can fall back to values
resolvable from `~/.openclaw/openclaw.json`. Do not commit filled credential files.

### Prepare a memory plugin when needed

A baseline run does not require the target memory plugin. For
`memory_train_backup_test`, install and configure that plugin first.
For example, the MemOS local plugin provides this installer:

```bash
curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/main/apps/memos-local-plugin/install.sh | bash
```

Review upstream installers before executing them. Complete the generated plugin
configuration, required credentials, local paths, and service endpoints, then
confirm the plugin is enabled in OpenClaw.

Memory-plugin runs enforce `runtime.home_mode: isolated_copy`, but
`home_links` can still link real plugin data into the temporary OpenClaw home.
Lifecycle clear/restore commands can delete or overwrite that data and restart
services. Inspect the selected plugin YAML and use dedicated evaluation data,
not personal memory.

<a id="data-source"></a>
## Data preparation

AgentBench uses the `EverMind-AI/EvoAgentBench` dataset. Dataset ownership,
splits, and underlying benchmark definitions follow the upstream project;
OmniMemEval provides runner integration, configuration, lifecycle management,
and result organization.

- [Hugging Face dataset](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)
- [Upstream project](https://github.com/EverMind-AI/EvoAgentBench)

| Domain | Benchmark | Train | Test |
|---|---|---:|---:|
| `information_retrieval` | BrowseCompPlus | 154 | 65 |
| `reasoning` | OmniMath | 478 | 100 |
| `software_engineering` | SWE-Bench | 101 | 26 |
| `code_implementation` | LiveCodeBench | 97 | 39 |
| `knowledge_work` | GDPVal | 87 | 58 |

### Download the dataset

```bash
mkdir -p data/agentbench
hf download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
```

Expected layout:

```text
data/agentbench/
  BrowseComp-Plus/
  Reasoning & Problem Decomposition/
  gdpval/
  livecode/
  swebench/
```

The `hf` CLI is provided by `huggingface_hub`; see the
[upstream CLI guide](https://huggingface.co/docs/huggingface_hub/en/guides/cli).
If using custom data locations, update the corresponding paths in
`configs/agentbench/domains/*.yaml`.

### Prepare domain-specific dependencies

Install only the additional tools needed by the selected domains.
LiveCodeBench uses its official verifier package:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git ./LiveCodeBench
python -m pip install --no-deps -e ./LiveCodeBench
```

BrowseComp-Plus dense retrieval requires preprocessing, an index, and embedding
service configuration:

```bash
python scripts/agentbench/utils/browsecomp-plus-tools/setup_data.py \
  --output-dir ./data/agentbench/BrowseComp-Plus \
  --skip-index
python scripts/agentbench/utils/browsecomp-plus-tools/build_dense_index.py \
  --config configs/agentbench/domains/information_retrieval.yaml
```

SWE-Bench requires Docker and network access to PyPI/GitHub from the evaluation
containers. GDPVal requires PDF/Office processing utilities. On Debian/Ubuntu,
install the required system packages with administrator privileges:

```bash
apt-get update
apt-get install -y docker.io poppler-utils libreoffice openjdk-21-jdk
systemctl start docker
```

Java is relevant to Pyserini/BM25 retrieval; a setup using only a prebuilt dense
index may not need it. These are domain-specific requirements, not prerequisites
for the reasoning quick start below.

## Quick start

After preparing the environment, credentials, and dataset, run one reasoning task:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version smoke_agentbench \
  --test-split 1 \
  --trials 1 \
  --parallel 1
```

This invokes real model/judge services and may incur charges. For reasoning,
`--test-split 1` selects the first test question; omitting it selects the full
test split. `--parallel` controls concurrency, not task count. A smoke run is
not a full benchmark result.

For complete protocols, see [advanced runs](#advanced-runs).
The runner's `--help` lists CLI options.

## Advanced runs

<a id="protocols"></a>
### Evaluation protocols

The following commands use complete configured splits.

#### `test_only`

Run the test split:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw --domain reasoning --protocol test_only \
  --version baseline_reasoning_test --trials 1 --parallel 1
```

#### `train_then_test`

Run training followed by testing, without memory-plugin lifecycle commands:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw --domain reasoning --protocol train_then_test \
  --version train_then_test_reasoning --trials 1 --parallel 1
```

#### `memory_train_backup_test`

Evaluate a memory plugin separately from the baseline:

1. Set the plugin to train mode and clear memory.
2. Run training.
3. Send verifier feedback as the next turn in the same OpenClaw session.
4. If enabled, submit plugin-side structured feedback with the same session/episode semantics.
5. Wait for memory settling/evolution.
6. Back up domain memory.
7. Restore the domain backup before every test run, even when test writes are disabled.
8. Run testing and write results under `results/agentbench/`.

Single domain:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw --domain reasoning --protocol memory_train_backup_test \
  --memory-plugin memos --version memos_reasoning_eval \
  --test-runs 1 --trials 1 --parallel 1
```

All five domains:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos --version memos_5domain_eval \
  --test-runs 1 --trials 1 --parallel 1
```

Domain subset:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos --domains reasoning,code_implementation \
  --version memos_reasoning_code_eval
```

### Memory plugin configuration

Lifecycle configs live under `configs/agentbench/memory_plugins/`:
`memos.yaml`, `everos.yaml`, `openviking.yaml`, `supermemory.yaml`,
and `hindsight.yaml`.

Each config declares:

- `plugin`: label used in result directory names.
- `backup_dir` and `backup_file_template`: backup location.
- `settle_seconds` or `commands.wait_settle`: post-training settling behavior.
- `home_links`: plugin paths linked into the isolated OpenClaw home.
- `modes.train/test` or `commands.set_mode_*`: read/write modes.
- `commands.clear/backup/restore`: lifecycle operations.
- `feedback`: training feedback and optional plugin-side structured feedback.

For example, `memos.yaml` enables structured feedback:

```yaml
feedback:
  enabled: true
  timeout: 300
  memos_structured_submit: true
  memos_submit_timeout: 900
```

Other plugin configs use ordinary training feedback by default without this
structured submit. Check the selected YAML rather than assuming identical
plugin behavior.

<a id="results"></a>
## Outputs and results

For the memory-plugin protocol, the output layout is:

```text
results/agentbench/<profile>-<version>-<domain>/
  experiment_config.json
  memory_lifecycle.log
  memory_lifecycle.json
  train/
    summary.json
    report.md
    <task>__trial_1/
      result.json
      response.txt
      session.jsonl
      verifier/
  test_run_1/
    summary.json
    report.md
    <task>__trial_1/
      result.json
      response.txt
      session.jsonl
      verifier/
```

Key fields:

- `summary.json`: `pass@1`, average reward, and average runtime.
- `result.json.agent_result`: completion status, runtime, and session recovery metadata.
- `result.json.feedback_result`: same-session training feedback status.
- `result.json.memos_feedback_result`: plugin-side structured feedback status when enabled.
- `memory_lifecycle.json`: clear/backup/restore events.

See [evaluation results](./results.md) for published setups and tables. The
snapshot also includes Hermes results; the current public runtime adapter covers
OpenClaw, not a Hermes reproduction path. Model, judge, and embedding service
availability can affect per-sample rewards.

## Extensions and contribution

The [architecture and extension contracts](./architecture.md) describe the
runner, session identity, result fields, and agent/domain extension points.
Agent Memory adapters are separate from the User Memory
`scripts/client_factory/` adapter layer.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for PR workflow and evidence
requirements. The root [repository layout](../../README.md#repository-layout)
describes where code, configuration, and data preparation live.

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

See [LICENSE](../../LICENSE) for the code license. External datasets keep their
upstream licenses; see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
