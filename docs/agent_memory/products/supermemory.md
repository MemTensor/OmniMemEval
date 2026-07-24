# Supermemory Evaluation Configuration and Execution

[中文](./supermemory_zh.md)

This guide explains how to evaluate OpenClaw/Hermes + Supermemory with
OmniMemEval AgentBench. Follow the
[Supermemory plugins page](https://supermemory.ai/plugins/),
[OpenClaw plugin project](https://github.com/supermemoryai/openclaw-supermemory),
and [Hermes Memory Providers guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers/)
to install the product.

## Support Matrix

| Agent | Status | Lifecycle configuration |
|---|---|---|
| OpenClaw | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/supermemory/lifecycle/openclaw.yaml` |
| Hermes | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/supermemory/lifecycle/hermes.yaml` |

Supermemory may use its cloud API or a compatible self-hosted service. In both
cases, the evaluation host must be able to reach the configured API endpoint.

## Pre-Evaluation Configuration

### 1. Prepare the shared environment

Prepare `.env.agent`, model services, and EvoAgentBench data by following the
[AgentBench documentation](../README.md). Do not write a Supermemory API key to
the repository or result directory.

### 2. Configure OpenClaw

The framework does not install, enable, or select the plugin. The user
configuration must satisfy:

- `plugins.enabled` is not `false`.
- If `plugins.allow` is present, it includes `openclaw-supermemory`.
- `plugins.slots.memory` is `openclaw-supermemory`.
- `plugins.entries.openclaw-supermemory` exists and is enabled.
- A usable API key and base URL are available from plugin configuration or the
  environment.
- The `supermemory` Node SDK can be resolved from
  `~/.openclaw/npm/node_modules`.

Example checks:

```bash
openclaw supermemory status
openclaw config get plugins.slots.memory
test -d ~/.openclaw/npm/node_modules
```

During evaluation, the framework forces the isolated configuration to use:

```text
containerTag = omnimemeval_<run_id>
train: autoRecall=true, autoCapture=true
test:  autoRecall=true, autoCapture=false
```

The user's normal container is not cleared. The lifecycle refuses to manage a
container whose tag does not start with `omnimemeval_`.

### 3. Configure Hermes

The user must select Supermemory before evaluation:

```bash
hermes memory status
hermes config set memory.provider supermemory
```

The deployment must satisfy:

- `memory.provider: supermemory` is set in `~/.hermes/config.yaml`.
- `~/.hermes/supermemory.json` exists and contains the endpoint/container
  configuration.
- The API key is available from the Hermes `.env`, process environment, or the
  deployment's protected credential file.

Example checks:

```bash
hermes memory status
test -f ~/.hermes/supermemory.json
```

The Hermes lifecycle copies `supermemory.json` and changes `container_tag` in
the isolated copy to the current run's `omnimemeval_<run_id>`.

If a self-hosted endpoint or credential source differs from the default
implementation, copy the lifecycle YAML and, if necessary, adapt the
configuration lookup in `hermes_supermemory_ctl.py`. Select the custom
lifecycle with `--memory-plugin-config`.

## Two-Sample End-to-End Validation

OpenClaw:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin supermemory \
  --version openclaw_supermemory_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Hermes:

```bash
./scripts/run_agent_eval.sh \
  --agent hermes \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin supermemory \
  --version hermes_supermemory_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

The lifecycle:

1. Validates that the user selected Supermemory.
2. Creates a run-scoped Agent home and container tag.
3. Clears the current run's container.
4. Enables automatic recall and capture during training.
5. Waits for extraction, chunking, embedding, and indexing to finish.
6. Exports run-scoped memory and creates the train backup.
7. Imports the backup before testing and waits for indexing again.
8. Disables automatic capture during testing while keeping recall enabled.
9. Deletes the run-scoped container contents and isolated home during cleanup.

OpenClaw backups primarily contain document JSONL. Hermes backups contain both
documents and generated memories, and restore generated memories first.

## Full Evaluation

Run all five domains with OpenClaw:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin supermemory \
  --version openclaw_supermemory_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Run all five domains with Hermes:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin supermemory \
  --version hermes_supermemory_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

## Result Validation

Result directory format:

```text
results/agentbench/<agent>-supermemory-<version>-<domain>/
```

Check the following:

- Every lifecycle stage returns zero.
- `wait_settle` reports no `failed`, `error`, or `cancelled` documents.
- The OpenClaw backup's `manifest.json` has the expected `document_count`.
- The Hermes backup reports the expected document and memory counts.
- Restore and indexing complete before the test split starts.
- The run's `omnimemeval_<run_id>` container is empty or removed after cleanup.

## Troubleshooting

- **HTTP 401/403:** verify that the plugin API key matches the actual
  Supermemory service. For self-hosted deployments, updating only the Agent
  configuration may leave a stale key in the service environment.
- **The run container is empty after training:** verify that the plugin and
  lifecycle use the same run-scoped container tag. The framework overwrites the
  tag in the isolated configuration.
- **Document status is `failed`:** inspect the self-hosted memory-extraction
  LLM, embedding model, and queue worker. The framework fails closed instead of
  treating failed documents as settled.
- **Testing creates new memory:** verify that `set_mode_test` ran and set
  `autoCapture=false`; the mode transition is recorded in
  `memory_lifecycle.log`.
