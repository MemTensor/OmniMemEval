# Agent Memory Evaluation Architecture

[中文版](./architecture_zh.md) · [Documentation index](../README.md) ·
[Guide](./README.md) · [Results](./results.md)

This document describes the current Agent Memory Evaluation architecture and
its public extension contracts. Operational setup and evaluation commands remain
in the [Agent Memory Evaluation guide](./README.md).

## Architecture Overview

Agent Memory Evaluation evaluates the following combination:

```text
Agent Runtime + Optional Memory Plugin + Task Domain + Verifier
```

The AgentBench entrypoint owns protocol and memory-lifecycle ordering, while the
runner owns task, trial, retry, and feedback execution. Runtime-specific
behavior is implemented by an `AgentAdapter`; task loading, prompt construction,
environment setup, and verification are implemented by a `DomainAdapter`.
Memory-plugin lifecycle actions are generally configured under
`configs/agentbench/memory_plugins/`.

## Core Components

### `AgentAdapter`

[`AgentAdapter`](../../scripts/agentbench/agents/base.py) defines how the runner
drives an agent runtime. It builds session metadata, invokes the runtime,
collects session artifacts and token usage, and classifies retryable runtime
outcomes.

The current public agent registry contains the
[`OpenClawAgentAdapter`](../../scripts/agentbench/agents/openclaw.py).

### `DomainAdapter`

[`DomainAdapter`](../../scripts/agentbench/domains/base.py) defines how a task
domain loads tasks, prepares per-task resources, builds prompts, verifies agent
output, and cleans up task state.

The current domain registry covers reasoning, information retrieval, knowledge
work, code implementation, and software engineering.

### Runner and Memory Lifecycle

[`run_agent_eval.py`](../../scripts/agentbench/run_agent_eval.py) selects the
evaluation protocol and owns phase and memory-lifecycle ordering.

[`runner.py`](../../scripts/agentbench/runner.py) executes tasks and trials,
writes per-trial artifacts, applies retry policy, and generates phase summaries.

[`memory_lifecycle.py`](../../scripts/agentbench/memory_lifecycle.py) executes
the configured plugin lifecycle commands. The
`memory_train_backup_test` protocol owns the ordering of validation, mode
selection, cleanup, training, settling, backup, restore, and test execution.
Optional structured feedback integrations remain explicit in the runner and
plugin configuration.

## Evaluation Execution Flow

For each task trial, the runner follows this sequence:

```text
domain.setup
agent.prepare_task
domain.build_prompt
agent.call
domain.verify
optional train feedback
domain.record_agent_outcome
domain.cleanup
agent.collect_session
agent.cleanup_task
write result.json
```

Domain-level initialization and finalization wrap the complete phase. A phase
summary is generated after all selected tasks and trials complete.

## Session Identity Contract

[`SessionSpec`](../../scripts/agentbench/session.py) separates identifiers used
for different purposes:

- `cli_session_id`: a sanitized identifier accepted by the agent CLI.
- `semantic_session_id`: an identifier that preserves benchmark phase, domain,
  split, task, and trial semantics.
- `source_ref`: a stable benchmark source reference.
- `metadata`: structured benchmark, phase, domain, split, task, trial, and agent
  metadata.
- `agent_session_ref`: an optional runtime-specific session reference.
- `openclaw_session_key`: the OpenClaw session key when OpenClaw is used.
- `openclaw_gateway_session_id`: the corresponding OpenClaw gateway identifier.

The OpenClaw adapter passes `cli_session_id` to `openclaw agent --session-id`
and exposes the complete serialized `SessionSpec` through
`OMNIMEMEVAL_AGENT_CONTEXT` by default. This keeps CLI-safe identifiers aligned
with plugin and result metadata.

## Result and Metric Contract

Each trial writes a `result.json` containing task, phase, split, trial, session,
agent result, verifier result, feedback result, token usage, timestamps, and
exception information. Supporting artifacts can include `response.txt`,
`session.jsonl`, and verifier output.

Each phase writes `summary.json` and `report.md`. The summary includes task and
trial counts, `pass@1`, configured `pass@n`, average pass rate, reward, elapsed
time, turns, tokens, failure classes, infrastructure-excluded statistics, and
domain-specific metrics.

## Extension Points

- Add an agent runtime by implementing `AgentAdapter` and registering it in
  `scripts/agentbench/agents/__init__.py`.
- Add a task domain by implementing `DomainAdapter` and registering it in
  `scripts/agentbench/domains/__init__.py`.
- Add or update memory-plugin lifecycle behavior through
  `configs/agentbench/memory_plugins/` and explicit integration code when a
  generic command lifecycle is insufficient.

Agent Memory Evaluation remains separate from the User Memory
`scripts/client_factory/` adapter layer because the two tracks evaluate
different system boundaries.
