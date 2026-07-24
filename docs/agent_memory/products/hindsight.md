# Hindsight Evaluation Configuration and Execution

[中文](./hindsight_zh.md)

This guide explains how to evaluate OpenClaw/Hermes + Hindsight with
OmniMemEval AgentBench. Follow the
[Hindsight OpenClaw integration guide](https://hindsight.vectorize.io/sdks/integrations/openclaw)
and the
[Hermes Memory Providers guide](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory-providers/)
to install the product. This document focuses on evaluation configuration and
execution.

## Support Matrix

| Agent | Status | Lifecycle configuration |
|---|---|---|
| OpenClaw | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/hindsight/lifecycle/openclaw.yaml` |
| Hermes | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/hindsight/lifecycle/hermes.yaml` |

## Pre-Evaluation Configuration

### 1. Prepare the shared environment

Prepare `.env.agent`, model services, and EvoAgentBench data by following the
[AgentBench documentation](../README.md). The evaluation framework does not
install or start a Hindsight deployment that the user has not configured.

### 2. Configure OpenClaw

The user must enable Hindsight manually. The framework validates the selection
but does not select the plugin. The current lifecycle requires:

- `plugins.enabled` is not `false`.
- If `plugins.allow` is present, it includes `hindsight-openclaw`.
- `plugins.slots.memory` is `hindsight-openclaw`.
- `plugins.entries.hindsight-openclaw` exists and is enabled.
- The Hindsight API is available at `http://127.0.0.1:9077` by default.
- For the validated embedded deployment, the instance description is at
  `/var/lib/hindsight-openclaw/.pg0/instances/hindsight/instance.json`.

Example checks:

```bash
openclaw config get plugins.slots.memory
curl -fsS http://127.0.0.1:9077/health
command -v hindsight-admin
```

During evaluation, the framework copies the OpenClaw configuration and assigns
a fixed run-scoped bank in the isolated copy:

```text
omnimemeval-<run_id>
```

Existing user banks are not cleared or modified.

### 3. Configure Hermes

The user must select Hindsight before evaluation:

```bash
hermes memory status
hermes config set memory.provider hindsight
```

The deployment must satisfy:

- `memory.provider: hindsight` is set in `~/.hermes/config.yaml`.
- `~/.hermes/hindsight/config.json` exists and contains a usable mode and
  credentials.
- The Hindsight API is available at `http://127.0.0.1:9177` by default.
- `hindsight-admin` is available to the user running the evaluation.

Example checks:

```bash
hermes memory status
test -f ~/.hermes/hindsight/config.json
curl -fsS http://127.0.0.1:9177/health
command -v hindsight-admin
```

If the API endpoint or configuration path differs, copy the corresponding
lifecycle YAML, update its `env`, and select it with
`--memory-plugin-config`.

## Two-Sample End-to-End Validation

OpenClaw:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin hindsight \
  --version openclaw_hindsight_reasoning2_smoke \
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
  --memory-plugin hindsight \
  --version hermes_hindsight_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

The lifecycle:

1. Validates that the user selected Hindsight.
2. Creates an isolated Agent home and points the provider at a run-scoped bank.
3. Deletes and recreates only that run-scoped bank.
4. Waits for Hindsight operations to remain idle after training.
5. Backs up the bank with `hindsight-admin export-bank`.
6. Restores it before testing with `import-bank`.
7. Deletes the run-scoped bank and isolated home after testing.

Any operation in a `failed`, `error`, or `cancelled` state fails the evaluation.

## Full Evaluation

Set `AGENT` to `openclaw` or `hermes`:

```bash
AGENT=openclaw
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent "$AGENT" \
  --memory-plugin hindsight \
  --version "${AGENT}_hindsight_5domain_eval" \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Run a domain subset:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin hindsight \
  --domains reasoning,code_implementation \
  --version hermes_hindsight_reasoning_code_eval
```

## Result Validation

Result directory format:

```text
results/agentbench/<agent>-hindsight-<version>-<domain>/
```

Check the following:

- Every lifecycle command has `returncode: 0` in `memory_lifecycle.json`.
- `memory_lifecycle.log` records bank idle, backup, and restore success.
- The backup's `manifest.json` reports the expected `has_bank` value.
- Train and test `summary.json` files exist, and infrastructure failures are
  not counted as task-quality failures.
- The run's `omnimemeval-<run_id>` bank no longer exists after cleanup.

## Troubleshooting

- **Hermes reports a provider mismatch:** manually set
  `memory.provider: hindsight` in the user configuration. The framework will
  not change it.
- **`hindsight-admin` cannot be found:** make it available in the Hindsight
  daemon environment or system `PATH`. The controller also checks locations
  adjacent to the daemon executable.
- **Bank export cannot connect to the database:** verify the local pg0 instance
  file, daemon user, and database port.
- **Operations never settle:** inspect Hindsight LLM/embedding credentials and
  service logs. Do not shorten the timeout to bypass incomplete memory writes.
