# Agent Memory Evaluation

[中文版](./README_zh.md)

Agent Memory Evaluation is the AgentBench-based evaluation track in OmniMemEval. It measures the task performance of an agent runtime after a memory plugin is installed. The current implementation evaluates OpenClaw and Hermes across five task domains and supports both plain agent runs and memory-plugin lifecycle runs.

Results are recorded in [eval_res.md](./eval_res.md).

Product-specific configuration and execution guides:

- [Mem0](./products/mem0.md)
- [EverOS](./products/everos.md)
- [Hindsight](./products/hindsight.md)
- [OpenViking](./products/openviking.md)
- [Supermemory](./products/supermemory.md)

## Evaluation Protocols

- plain AgentBench: runs `test_only` or `train_then_test` without enabling the target memory plugin.
- memory plugin AgentBench: runs the `memory_train_backup_test` protocol, including memory cleanup, train execution, memory settling, backup, restore, and test execution.

## Data Source

AgentBench uses the `EvoAgentBench` dataset published by EverMind (Hugging Face/GitHub namespace: `EverMind-AI`). Dataset ownership, task splits, and upstream benchmark definitions follow the EverMind-AI/EvoAgentBench dataset and project documentation. OmniMemEval only provides runner integration, configuration, lifecycle management, and result organization.

References:

- Hugging Face dataset: `EverMind-AI/EvoAgentBench`
- GitHub project: `EverMind-AI/EvoAgentBench`
- Underlying benchmarks: BrowseCompPlus, OmniMath, SWE-Bench, LiveCodeBench, GDPVal

Dataset size:

| Domain | Benchmark | Train | Test |
|---|---|---:|---:|
| `information_retrieval` | BrowseCompPlus | 154 | 65 |
| `reasoning` | OmniMath | 478 | 100 |
| `software_engineering` | SWE-Bench | 101 | 26 |
| `code_implementation` | LiveCodeBench | 97 | 39 |
| `knowledge_work` | GDPVal | 87 | 58 |

## Environment

Agent Memory Evaluation uses a separate conda environment from the User Memory track:

```bash
conda create -n agentmem python=3.12 -y
conda activate agentmem
python -m pip install -U pip
pip install -r requirements_agentbench.txt
```

Do not reuse the User Memory environment for AgentBench. User Memory and AgentBench dependencies are maintained in separate requirements files.

Install OpenClaw CLI and make sure it is available in `PATH`:

```bash
npm install -g openclaw
```

To evaluate Hermes, install the Hermes CLI using the deployment method for
your environment and make sure the current shell can invoke it:

```bash
hermes --version
hermes chat --help
```

For memory-plugin lifecycle evaluations, install the target memory plugin before running AgentBench. For example, install the MemOS local plugin with:

```bash
# Valid values are openclaw, hermes, and all.
TARGET_AGENT=hermes
curl -fsSL https://raw.githubusercontent.com/MemTensor/MemOS/main/apps/memos-local-plugin/install.sh \
  | bash -s -- --agent "$TARGET_AGENT"
```

After installation, configure the target runtime and memory plugin before
running validation. The exact settings depend on the plugin under evaluation.
For the MemOS local plugin, OpenClaw uses `~/.openclaw/memos-plugin/` and
Hermes uses `~/.hermes/memos-plugin/`. Complete the LLM, embedding, and
credential settings in the corresponding `config.yaml`. A Hermes installation
should also create the `~/.hermes/plugins/memory/memtensor` provider. The
OmniMemEval Hermes MemOS profile selects an evaluation-specific provider only
inside its isolated runtime configuration; it does not rewrite the user's
global Hermes configuration.

At minimum, verify the following before running AgentBench:

- The target OpenClaw or Hermes runtime has a valid model/provider configuration.
- The memory plugin is installed and discoverable by the target runtime.
- Required plugin credentials, local paths, or service endpoints are configured.
- The AgentBench env file `.env.agent` is prepared as described below.

After installation and configuration are complete, verify the CLI for the
target runtime:

```bash
openclaw --version
openclaw agent --help

hermes --version
hermes chat --help
```

Create the Agent Memory env file from the template:

```bash
cp env_examples/.env.agent .env.agent
```

AgentBench loads `.env.agent` by default. `--env FILE` can be used to supply an additional env file. This keeps Agent Memory credentials separate from User Memory backend env files such as `.env.memos` and `.env.mem0`.

Runtime model settings are configured in:

```text
configs/agentbench/agents/{openclaw,hermes}.yaml
```

Common `.env.agent` fields:

```bash
LLM_BASE_URL=...
LLM_API_KEY=...

JUDGE_MODEL=...
JUDGE_API_BASE=...
JUDGE_API_KEY=...

IR_EMBEDDING_ENDPOINT=...
IR_EMBEDDING_MODEL=...
IR_EMBEDDING_API_KEY=...

EVALUATION_API_BASE=...
EVALUATION_API_KEY=...
EVALUATION_MODEL_OWNER=...
EVALUATION_MODEL_NAME=...
EVALUATION_TIMEOUT=240
EVALUATION_MAX_RETRIES=3
```

Model/provider credentials from `.env.agent` are injected into the selected
agent configuration. If OpenClaw provider fields are not set there, its adapter
falls back to values resolvable from `~/.openclaw/openclaw.json`. Hermes starts
from `~/.hermes/config.yaml` and then applies the settings in
`configs/agentbench/agents/hermes.yaml` inside the isolated evaluation home.

## Data Preparation

Run the following commands from the repository root:

```bash
mkdir -p data/agentbench
huggingface-cli download EverMind-AI/EvoAgentBench \
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

Domain data paths are configured under `configs/agentbench/domains/*.yaml`. If a custom data location is used, update the relevant domain yaml fields.

LiveCodeBench requires its official verifier package:

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench.git ./LiveCodeBench
pip install --no-deps -e ./LiveCodeBench
```

BrowseComp-Plus dense retrieval requires preprocessing, an index, and embedding service configuration:

```bash
python scripts/agentbench/utils/browsecomp-plus-tools/setup_data.py \
  --output-dir ./data/agentbench/BrowseComp-Plus \
  --skip-index
python scripts/agentbench/utils/browsecomp-plus-tools/build_dense_index.py \
  --config configs/agentbench/domains/information_retrieval.yaml
```

SWE-Bench requires Docker. GDPVal requires PDF/Office processing utilities. On Debian/Ubuntu:

```bash
apt-get update
apt-get install -y docker.io poppler-utils libreoffice openjdk-21-jdk
systemctl start docker
```

## Quick Start

Reasoning smoke run:

```bash
cp env_examples/.env.agent .env.agent
mkdir -p data/agentbench
huggingface-cli download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version smoke_agentbench \
  --trials 1 \
  --parallel 1
```

MemOS lifecycle smoke run:

```bash
./scripts/run_agentbench_memos_5domain_1each_smoke.sh \
  --domains reasoning \
  --version memos_reasoning_1each_smoke \
  --tasks-per-split 1 \
  --trials 1 \
  --test-runs 1 \
  --parallel 1 \
  --settle-seconds 0
```

The default OpenClaw MemOS smoke runs the full `memory_train_backup_test`
lifecycle on a small task subset: clear memory, train, send verifier feedback,
submit MemOS structured feedback, backup, restore, test, and finalize. Use this
before a full five-domain run to validate local OpenClaw, MemOS, model, judge,
and embedding configuration. Hermes + MemOS does not perform the additional
structured submit by default; its flow is described below.

### Hermes + MemOS Plugin Evaluation

This section uses Hermes + MemOS to illustrate the complete flow. Installation
and configuration details for other Hermes memory providers such as Hindsight,
OpenViking, and Supermemory are documented in their product guides. They use
the same runner entry points and `memory_train_backup_test` protocol.

#### 1. Check prerequisites

The default Hermes home is `~/.hermes`. Set `HERMES_HOME` before starting the
evaluation if a different directory is used. The Hermes + MemOS lifecycle
requires the Hermes configuration, the MemOS bridge, and several system
commands:

```bash
command -v hermes
command -v node
command -v sqlite3
command -v timeout
command -v curl

test -f "${HERMES_HOME:-$HOME/.hermes}/config.yaml"
test -f "${HERMES_HOME:-$HOME/.hermes}/memos-plugin/config.yaml"
test -f "${HERMES_HOME:-$HOME/.hermes}/memos-plugin/dist/bridge.cjs"
test -e "${HERMES_HOME:-$HOME/.hermes}/plugins/memory/memtensor"
```

`~/.hermes/memos-plugin/config.yaml` configures MemOS itself. The repository
`.env.agent` and `configs/agentbench/agents/hermes.yaml` configure the primary
Hermes model used by this AgentBench run. These settings have different roles,
and all referenced services and credentials must be usable.

#### 2. Run a single-domain smoke test

Start with one reasoning train item and one reasoning test item:

```bash
./scripts/run_agentbench_memos_5domain_1each_smoke.sh \
  --agent hermes \
  --domains reasoning \
  --version hermes_memos_reasoning_1each_smoke \
  --tasks-per-split 1 \
  --trials 1 \
  --test-runs 1 \
  --parallel 1 \
  --settle-seconds 0
```

To debug specific samples, select explicit train and test tasks:

```bash
./scripts/run_agent_eval.sh \
  --agent hermes \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version hermes_memos_reasoning_debug \
  --train-task omni_35 \
  --test-task omni_2080 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

#### 3. Start the full evaluation

Run a complete train/test cycle for one domain:

```bash
./scripts/run_agent_eval.sh \
  --agent hermes \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version hermes_memos_reasoning_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Run all five domains sequentially:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin memos \
  --version hermes_memos_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Use `--domains reasoning,code_implementation` to run only a subset. Complete
the single-domain smoke first. Domain-specific verifier, Docker, retrieval
index, and document-tool dependencies still follow the Data Preparation
section.

#### 4. Hermes + MemOS execution semantics

The workflow does not train and test directly against the user's global MemOS
database:

1. The lifecycle first backs up the existing Hermes MemOS SQLite database and
   creates run-scoped `HERMES_HOME` and `MEMOS_HOME` directories under the
   result directory. It does not clear or overwrite the original
   `~/.hermes/memos-plugin/data/memos.db`.
2. `profiles/hermes/memos.yaml` creates a temporary Hermes home for each task,
   loads the `omnimemeval_memos` evaluation provider, and writes mutable state
   to the run-scoped MemOS SQLite database.
3. After a train task is verified, verifier feedback is sent to the same real
   Hermes session through `hermes chat --resume`. Hermes MemOS captures that
   normal conversation turn, so `structured_submit` defaults to `false`. Do
   not add `--memos-structured-feedback` to this flow.
4. Every successful train call must produce a closed, non-empty episode in the
   MemOS database. A successful Hermes CLI exit without durable capture is
   treated as a technical failure.
5. After training, the lifecycle waits for episodes to close and drains the
   embedding and evolution queues. It creates the domain training backup only
   after the SQLite integrity check succeeds.
6. The same training backup is restored before every test run. After testing,
   finalize, checkpoint, and cleanup prevent accidental state sharing across
   domains or repeated test runs.

Example result layout:

```text
results/agentbench/hermes-memos-hermes_memos_reasoning_eval-reasoning/
  experiment_config.json
  memory_lifecycle.log
  memory_lifecycle.json
  memory_backups/
    user-global-hermes-*.sqlite3
    memos-hermes-reasoning-*.sqlite3
  train/
  test_run_1/
```

For Hermes + MemOS troubleshooting, inspect these locations first:

- `memory_lifecycle.log`: validate, clear, settle, backup, restore, and cleanup
  status.
- `memory_lifecycle.json`: command, timing, and exit status for lifecycle
  stages.
- `train/<task>__trial_1/result.json` under `agent_result.memos_capture`: the
  durable Hermes session/episode capture record.
- `<run_dir>/runtime/hermes/memos-plugin/logs/` while the run is active: bridge,
  embedding, and evolution errors. Cleanup removes this isolated directory, so
  copy these logs while diagnosing an active failure if they must be retained.
- The Hermes viewer uses port `18800` by default. With
  `MEMOS_START_DAEMON=auto`, an existing user daemon is left untouched and the
  evaluation continues headlessly rather than stopping that process.

## Protocols

### `test_only`

Runs the test split.

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version baseline_reasoning_test \
  --trials 1 \
  --parallel 1
```

### `train_then_test`

Runs train split followed by test split. This protocol does not execute memory plugin lifecycle commands.

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol train_then_test \
  --version train_then_test_reasoning \
  --trials 1 \
  --parallel 1
```

### `memory_train_backup_test`

This protocol evaluates memory plugins separately from baseline results:

1. Set plugin to train mode.
2. Clear memory.
3. Run train split.
4. Submit verifier feedback as the next turn in the same agent session
   (OpenClaw or Hermes).
5. If structured feedback is enabled in plugin config, submit plugin-side explicit feedback while preserving the same session/episode semantics.
6. Wait for memory settling/evolution.
7. Back up domain memory.
8. Restore the domain backup before every test run.
9. Run test split and write results to `results/agentbench`.

Single domain:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version memos_reasoning_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

For targeted lifecycle debugging, select explicit train/test tasks:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version memos_reasoning_debug \
  --train-task omni_35 \
  --test-task omni_2080 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Five domains:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos \
  --version memos_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Domain subset:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos \
  --domains reasoning,code_implementation \
  --version memos_reasoning_code_eval
```

## Agent, Profile, and Memory Lifecycle Configuration

Configuration is split into three layers:

```text
configs/agentbench/
  agents/{openclaw,hermes}.yaml
  profiles/openclaw/{plain,memos,everos,hindsight,openviking,supermemory}.yaml
  profiles/hermes/{plain,memos,hindsight,openviking,supermemory}.yaml
  memory_plugins/memos/lifecycle/{openclaw,hermes}.yaml
  memory_plugins/{everos,hindsight,openviking,supermemory}/lifecycle/openclaw.yaml
  memory_plugins/{hindsight,openviking,supermemory}/lifecycle/hermes.yaml
```

`agents` owns runtime/model defaults, `profiles/<agent>` owns the runtime-specific
plain or memory integration, and lifecycle configs own clear, settle, backup, and
restore protocol stages, including safety snapshot and cleanup. Memory protocols
infer `--profile` from `--memory-plugin`; mismatched
agent/plugin declarations fail before tasks start.

Hermes profiles do not replace the user's global `platform_toolsets.cli`.
The adapter inherits that list, removes only `web` outside `knowledge_work`,
and narrows `information_retrieval` to its local search MCP tool only.
Non-MemOS product profiles also do not select or enable a memory provider or
OpenClaw plugin slot. The evaluation user must configure the intended product;
the lifecycle validates that selection before any destructive stage.

Each config declares:

- `plugin`: plugin label used in result directory names.
- `backup_dir` and `backup_file_template`: memory backup location.
- `settle_seconds` or `commands.wait_settle`: post-train memory settling logic.
- `modes.train/test` or `commands.set_mode_*`: plugin read/write mode.
- `commands.clear/backup/restore`: memory cleanup, backup, and restore commands.
- `execution`: optional lifecycle execution strategy. OpenClaw MemOS uses
  `capture_mode: manual_after_feedback` so task execution can run in parallel
  while memory writes are submitted explicitly after verifier feedback. Hermes
  MemOS instead captures the task and resumed feedback turn automatically
  through its memory provider.
- `max_parallel`: optional cap for plugins that cannot safely run multiple
  agent processes against the same local runtime.
- `feedback`: train feedback settings and optional plugin-side structured feedback.
  Use `structured_submit`, `backend`, and `submit_timeout` for generic plugin
  feedback. `backend: memos` uses the MemOS bridge submitter.

For MemOS, `structured_submit: true` calls the MemOS bridge after the verifier
feedback turn. This step can depend on the MemOS embedding/LLM configuration and
may take longer than the agent call itself; tune `submit_timeout` if the service
is slow.

## Results

Each domain writes outputs under:

```text
results/agentbench/<profile>-<version>-<domain>/
  experiment_config.json
  memory_lifecycle.log
  memory_lifecycle.json
  train/
  test_run_1/
```

Key files:

- `summary.json`: `pass@1`, average reward, and average runtime.
- `result.json.agent_result`: agent completion status and session recovery metadata.
- `result.json.feedback_result`: train feedback turn status.
- `result.json.plugin_feedback_result`: plugin-side structured feedback status when enabled.
- `result.json.memos_feedback_result`: compatibility alias for MemOS structured feedback.
- `memory_lifecycle.json`: clear/backup/restore lifecycle events.

Recent evaluation results are recorded in [eval_res.md](./eval_res.md). Model service, external judge service, and embedding service status may affect per-sample rewards.

## Troubleshooting

- If a smoke run stalls after the agent response is written, check whether the
  domain verifier is waiting on an external judge service. The reasoning domain
  uses `verify_mode: llm` by default.
- To validate the lifecycle mechanics without the external reasoning judge,
  create a temporary copy of `configs/agentbench/domains/reasoning.yaml` with
  `verify_mode: exact`, then pass it with `--domain-config`. Do not use this for
  official evaluation results.
- If a run stalls during MemOS structured feedback, inspect
  `result.json.plugin_feedback_result`, `memory_lifecycle.log`, and
  `~/.openclaw/memos-plugin/data/memos.db`. A successful MemOS submit should
  create an episode, task/feedback traces, and one explicit feedback row.
- After an interrupted run, confirm no OpenClaw/MemOS local runtime is still
  listening on the MemOS viewer port:

```bash
lsof -nP -iTCP:18799 -sTCP:LISTEN
```
