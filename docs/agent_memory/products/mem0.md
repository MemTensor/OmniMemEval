# Mem0 Evaluation Configuration and Execution

[中文](./mem0_zh.md)

This guide explains how to evaluate OpenClaw/Hermes + Mem0 with OmniMemEval
AgentBench. Install Mem0 and the relevant Agent integration by following their
official documentation; this document focuses on framework configuration,
isolation, and execution.

## Support Matrix

| Agent | Supported Mem0 backend | Lifecycle configuration |
|---|---|---|
| OpenClaw | `open-source`/OSS with a Qdrant server | `configs/agentbench/memory_plugins/mem0/lifecycle/openclaw.yaml` |
| Hermes | OSS with a local embedded Qdrant path | `configs/agentbench/memory_plugins/mem0/lifecycle/hermes.yaml` |

The built-in lifecycle does not currently support Mem0 Cloud, PGVector, or
other Agent integrations. For another backend, copy the lifecycle YAML and
controller, then select the custom file with `--memory-plugin-config`.

## Pre-Evaluation Configuration

### Shared environment

Prepare `.env.agent`, model services, and EvoAgentBench data by following the
[AgentBench guide](../README.md). Keep Mem0 LLM, embedding, and vector-store
credentials in protected Agent configuration or environment files, not in the
repository.

### OpenClaw

The framework does not install, enable, or select the plugin. Configure:

- `plugins.slots.memory: openclaw-mem0`.
- An enabled `plugins.entries.openclaw-mem0` entry.
- `plugins.allow` includes `openclaw-mem0` when the allowlist is present.
- Plugin `mode` is `open-source` or `oss`.
- `oss.vectorStore.provider` is `qdrant`, with a reachable URL and collection.
- The `mem0ai` OSS SDK used by the plugin is resolvable from
  `~/.openclaw/npm/node_modules`.

Example checks:

```bash
openclaw config get plugins.slots.memory
openclaw mem0 status --json
curl -fsS http://127.0.0.1:6333/collections >/dev/null
```

The lifecycle copies the user configuration to a run-scoped home and applies:

```text
userId = omnimemeval:<run_id>
train: autoRecall=false, autoCapture=false, skills.triage.enabled=false
test:  autoRecall=true,  autoCapture=false, skills.triage.enabled=false
```

After training, the framework explicitly submits each task's expected answer,
agent answer, verifier feedback, and agent reflection through the official
Mem0 OSS SDK used by the plugin, with `infer=false`, and waits for completion.
This avoids both skills mode not auto-capturing and fire-and-forget writes being
lost when a one-shot OpenClaw process exits.

### Hermes

Select Mem0 manually:

```bash
hermes memory status
hermes config set memory.provider mem0
```

Verify:

- `memory.provider: mem0` is set in `~/.hermes/config.yaml`.
- `~/.hermes/mem0.json` exists and uses `mode: oss`.
- `oss.vector_store.provider` is `qdrant` with a local `path`.
- The Hermes Python environment can import `mem0` and `qdrant_client`.

The lifecycle copies `mem0.json` and rewrites `user_id`, `agent_id`, the Qdrant
path, and `MEM0_DIR` for the current run. After training, it uses the Mem0 SDK
from the Hermes environment to store verified feedback with `infer=False`.
This avoids losing writes performed by Hermes' background synchronization
thread when its CLI exits.

Embedded Qdrant exclusively locks its storage directory, so the Hermes Mem0
lifecycle automatically caps `--parallel` at `1`.

## Two-Sample End-to-End Validation

OpenClaw:

```bash
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol memory_train_backup_test \
  --memory-plugin mem0 \
  --version openclaw_mem0_reasoning2_smoke \
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
  --memory-plugin mem0 \
  --version hermes_mem0_reasoning2_smoke \
  --train-task omni_35,omni_159 \
  --test-task omni_2080,omni_2185 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

The lifecycle:

1. Validates that the user selected Mem0 and a supported Qdrant mode.
2. Creates run-scoped Agent homes, identities, and data paths.
3. Clears only the current `omnimemeval:<run_id>` memory.
4. Runs training and the verifier-feedback turn.
5. Explicitly stores verified feedback and rejects an empty backup.
6. Waits for stable memory counts and creates the training backup.
7. Restores the exact training snapshot before testing.
8. Enables recall without automatic writes during testing.
9. Deletes run-scoped data and isolated homes during cleanup.

OpenClaw backups contain complete points, payloads, and vectors for the run
user in the primary and `_entities` collections. Hermes backups contain the
run-owned embedded Qdrant directory and history data under `MEM0_DIR`.

## Full Five-Domain Evaluation

```bash
./scripts/run_agentbench_memory_train_backup_test.sh \
  --agent openclaw \
  --memory-plugin mem0 \
  --version openclaw_mem0_5domain_eval \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

Change `--agent` to `hermes` for Hermes. Its parallelism is capped at one.

## Result Validation

Result directory:

```text
results/agentbench/<agent>-mem0-<version>-<domain>/
```

Check:

- Explicit ingest reports more than zero memories in `memory_lifecycle.log`.
- Backup, restore, and test stages all succeed.
- The restored OpenClaw primary collection has the backed-up run point count.
- The restored Hermes `memory_count` matches its manifest.
- Cleanup removes the current run data without modifying the user's original
  Mem0 namespace.

## Troubleshooting

- **Training succeeds but backup is empty:** inspect explicit-ingest logs.
  Enabling skills triage or `autoCapture` alone does not prove that a one-shot
  CLI completed its write.
- **OpenClaw recall times out:** inspect embedding latency, model
  configuration, and Qdrant connectivity. A recall timeout means no memory was
  injected for that turn.
- **Hermes reports a Qdrant storage lock:** do not run multiple Hermes trials
  against the same lifecycle directory; the framework caps `--parallel` at 1.
- **Dimension or collection mismatch:** the OpenClaw embedding dimension must
  match the Qdrant collection, and the lifecycle must point to the collection
  actually used by the plugin.
