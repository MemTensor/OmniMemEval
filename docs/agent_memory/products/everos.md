# EverOS Evaluation Configuration and Execution

[中文](./everos_zh.md)

This guide explains how to evaluate OpenClaw + EverOS with OmniMemEval
AgentBench. Follow the [official EverOS project](https://github.com/EverMind-AI/EverOS)
and the documentation for your OpenClaw connector to install the product. This
guide focuses on evaluation-framework configuration and execution.

## Support Matrix

| Agent | Status | Lifecycle configuration |
|---|---|---|
| OpenClaw | Supported and validated on a deployed host | `configs/agentbench/memory_plugins/everos/lifecycle/openclaw.yaml` |
| Hermes | Not currently provided | — |

## Pre-Evaluation Configuration

### 1. Prepare the shared environment

Prepare the AgentBench environment, `.env.agent`, and EvoAgentBench data from
the [AgentBench documentation](../README.md).

If the data is already stored in another project, link it to the location
expected by this repository:

```bash
mkdir -p data
ln -s /path/to/EvoAgentBench/data data/agentbench
```

After linking it, directories such as
`data/agentbench/Reasoning & Problem Decomposition/` must be available.

### 2. Select EverOS manually

The evaluation framework does not change the OpenClaw plugin selection. Before
running an evaluation, enable the installed EverOS/EverMemOS OpenClaw connector
in the user's `~/.openclaw/openclaw.json`, then verify that a normal OpenClaw
session can write to and recall from EverOS.

The current profile links these plugin directories into the isolated evaluation
home:

```text
~/.openclaw/plugins/evermemos-openclaw-plugin
~/.openclaw/npm
```

If the connector is installed elsewhere, copy
`configs/agentbench/profiles/openclaw/everos.yaml`, update `home_links`, and
select the custom profile with `--profile-config`.

### 3. Verify lifecycle dependencies

The default lifecycle expects the following deployment:

| Item | Default |
|---|---|
| User EverOS root | `~/.everos` |
| EverOS API | `http://127.0.0.1:8000` |
| EverMemOS compatibility API | `http://127.0.0.1:1995` |
| Compatibility service directory | `/opt/everos-compat` |
| Compatibility service entry point | `uvicorn evermemos_v0_compat:app` |

Run these checks before evaluation:

```bash
command -v everos
command -v uvicorn
test -f ~/.everos/everos.toml
test -d ~/.openclaw/plugins/evermemos-openclaw-plugin
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:1995/health
```

If the ports, directories, or service entry points differ, copy
`configs/agentbench/memory_plugins/everos/lifecycle/openclaw.yaml`, update its
`env` and commands, and pass the custom file with
`--memory-plugin-config`.

## Two-Sample End-to-End Validation

Start with two reasoning train samples and two reasoning test samples:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin everos \
  --version everos_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

The framework performs the following sequence:

1. Snapshot the user's OpenClaw and EverOS state.
2. Create a run-scoped OpenClaw home and EverOS root.
3. Clear run-scoped memory and start the isolated services.
4. Execute the train samples and verifier feedback turns.
5. Call EverOS flush and cascade sync, then wait for persistence.
6. Back up the domain's Markdown memory and indexes.
7. Restore the train backup and execute the test samples.
8. Remove run-scoped state and restart the user's original EverOS services.

## Full Evaluation

Run one complete domain:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin everos \
  --version everos_reasoning_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Run all five domains sequentially:

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin everos \
  --version everos_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

## Result Validation

Results are written to:

```text
results/agentbench/openclaw-everos-<version>-<domain>/
```

Check at least the following:

- Validate, clear, wait, backup, restore, and cleanup succeeded in
  `memory_lifecycle.json`.
- Flush, cascade sync, and both health checks succeeded in
  `memory_lifecycle.log`.
- The EverOS train backup under `memory_backups/` is not an empty archive.
- Both `train/summary.json` and `test_run_1/summary.json` exist.
- The user's services on ports `8000` and `1995` are healthy after cleanup.

## Troubleshooting

- **Plugin directory is missing:** update `home_links` in the profile. Do not
  link mutable user memory data into the run-scoped home.
- **Memory settling does not finish:** inspect the EverOS server log,
  compatibility-service log, and the flush/cascade responses.
- **The backup contains no train memory:** make sure the connector uses the
  same session, app, and project configuration as the lifecycle and that train
  turns reach the compatibility API.
- **A service is terminated unexpectedly:** do not run multiple EverOS
  evaluations that compete for ports `8000` and `1995` on the same host.
