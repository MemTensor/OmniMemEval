# OpenViking Evaluation Configuration and Execution

[中文](./openviking_zh.md)

This guide explains how to evaluate OpenClaw/Hermes + OpenViking with
OmniMemEval AgentBench. Follow the
[official OpenViking project](https://github.com/volcengine/OpenViking),
[OpenClaw plugin guide](https://github.com/volcengine/OpenViking/blob/main/examples/openclaw-plugin/INSTALL.md),
and [Hermes integration guide](https://docs.openviking.ai/en/agent-integrations/05-hermes)
to install the product.

## Support Matrix

| Agent | Status | Lifecycle configuration |
|---|---|---|
| OpenClaw | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/openviking/lifecycle/openclaw.yaml` |
| Hermes | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/openviking/lifecycle/hermes.yaml` |

## Pre-Evaluation Configuration

### 1. Prepare the shared environment

Prepare `.env.agent`, model services, and evaluation data by following the
[AgentBench documentation](../README.md). The evaluation user must install and
validate the OpenViking service and Agent integration before running the
framework.

### 2. Configure OpenClaw

In the current OpenClaw integration, OpenViking owns the `contextEngine` slot,
not the memory slot. The user configuration must satisfy:

- `plugins.enabled` is not `false`.
- If `plugins.allow` is present, it includes `openviking`.
- `plugins.slots.contextEngine` is `openviking`.
- `plugins.entries.openviking` exists and is enabled.
- The plugin directory is `~/.openclaw/extensions/openviking` by default.

Example checks:

```bash
openclaw config get plugins.slots.contextEngine
openclaw config get plugins.entries.openviking.config
test -d ~/.openclaw/extensions/openviking
curl -fsS http://127.0.0.1:1933/health
```

The default OpenClaw lifecycle also expects:

| Item | Default |
|---|---|
| OpenViking server | `/opt/openviking/venv/bin/openviking-server` |
| User configuration | `~/.openviking/ov.conf` |
| User data | `/opt/openviking/data` |
| API | `http://127.0.0.1:1933` |

The framework copies `ov.conf` and changes `storage.workspace` in the isolated
copy to the current run's data directory.

### 3. Configure Hermes

The user must select OpenViking manually:

```bash
hermes memory status
hermes config set memory.provider openviking
```

The default Hermes lifecycle uses these deployment conventions:

| Item | Default |
|---|---|
| User OpenViking configuration directory | `~/.openviking` |
| User service container | `openviking` |
| Evaluation container | `omnimemeval-openviking-<run_id>` |
| Image | `ghcr.io/volcengine/openviking:latest` |
| API | `http://127.0.0.1:1933` |

Run these checks before evaluation:

```bash
hermes memory status
docker inspect openviking >/dev/null
test -f ~/.openviking/ov.conf
curl -fsS http://127.0.0.1:1933/health
```

The user's source container is paused during evaluation. The framework starts
an evaluation container with isolated configuration and data, then restores
the source container during cleanup. Do not run multiple OpenViking evaluations
that compete for port `1933` on the same host.

If the directory, container name, image, or port differs, copy the corresponding
lifecycle YAML, update its `env`, and pass the custom file with
`--memory-plugin-config`.

## Two-Sample End-to-End Validation

OpenClaw:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin openviking \
  --version openclaw_openviking_reasoning2_smoke \
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
  --memory-plugin openviking \
  --version hermes_openviking_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

The lifecycle:

1. Validates the user's OpenViking provider or slot selection.
2. Snapshots the user's configuration and data.
3. Creates a run-scoped Agent home, OpenViking configuration, and data path.
4. Clears evaluation data and starts the isolated service.
5. Polls `/api/v1/tasks` after training and requires 30 seconds of stable idle.
6. Stops the isolated service before backing up its data.
7. Restores the backup, restarts the service, and executes the test split.
8. Deletes isolated state and restores the user's original service.

A task in a `failed`, `error`, or `cancelled` state fails the lifecycle.

## Full Evaluation

Run all five domains with OpenClaw:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin openviking \
  --version openclaw_openviking_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Run all five domains with Hermes:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent hermes \
  --memory-plugin openviking \
  --version hermes_openviking_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

## Result Validation

Result directory format:

```text
results/agentbench/<agent>-openviking-<version>-<domain>/
```

Check the following:

- Every lifecycle stage returns zero.
- `wait_settle` reports a stably idle task queue.
- The backup contains run-scoped OpenViking data, not only configuration files.
- The restored service passes its health check before the test split starts.
- The evaluation container is removed and the user's source service is healthy
  after cleanup.

## Troubleshooting

- **OpenClaw fails slot validation:** select
  `plugins.slots.contextEngine`, not `plugins.slots.memory`, for OpenViking.
- **Hermes returns HTTP 401/403:** verify the API key in the run-scoped `.env`
  and any account/user settings required by a multi-tenant deployment. The
  framework passes authorization and tenant headers.
- **Backup reports database corruption or lock files:** the server/container
  must stop before archiving. Do not bypass the lifecycle stop step.
- **The task queue does not settle:** inspect OpenViking embedding, vector
  database, and LLM configuration together with the service/container logs.
