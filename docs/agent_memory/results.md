<a id="agentbench-evaluation-results"></a>
# Agent Memory Evaluation Results

[中文版](./results_zh.md) · [Documentation index](../README.md) ·
[Guide](./README.md) · [Architecture](./architecture.md)

These tables record OpenClaw and Hermes evaluation setups. The current public
Agent Memory runner guide and registered runtime adapter cover OpenClaw.
This page is the canonical source for numeric result tables; the Chinese page
provides translated explanations and links to the same tables.

## Metrics

| Metric | Meaning |
|---|---|
| Acc (%) | Average single-run pass rate across 3 independent runs of the same task |
| Avg turns | Average model response turns per task |
| Avg chars | Average response-text character count per task |

Acc measures single-run task completion, not pass-at-3. Numeric cells are shown
without ranking or statistical-significance markers; no significance test is
documented in this snapshot. Service availability can affect per-sample rewards.

## Data and Evaluation Setup

Dataset source and split logic:

Reference: [https://huggingface.co/datasets/EverMind-AI/EvoAgentBench](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)

Evaluated agents: OpenClaw and Hermes with the corresponding product plugin or integration.

Baseline refers to the corresponding evaluated agent running without any plugin.

The OpenClaw version used for OpenClaw evaluation is 2026.5.7.

The Hermes version used for Hermes evaluation is Hermes Agent v0.18.0 (2026.7.1).

Products not marked as cloud services use locally deployed services and their corresponding agent plugins or integrations.

MemOS uses Memos-Local-Plugin version 2.0.8.

OpenClaw and Hermes answer model configuration: qwen3.6-flash in no_thinking mode.

Evaluation judge model: qwen3.6-flash in thinking mode.

## Results

### OpenClaw

| Method | BrowseComp-Plus Acc (%) | BrowseComp-Plus Avg turns | OmniMath Acc (%) | OmniMath Avg chars | SWE-Bench Acc (%) | SWE-Bench Avg turns | LiveCodeBench Acc (%) | LiveCodeBench Avg turns | GDPVal Acc (%) | GDPVal Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | 18.46 | 35.1 | 52 | 5658.6 | 26.92 | 58.8 | 51.28 | 23.9 | 34.48 | 17.2 |
| **Mem0** | 13.33 | 36.4 | 53 | 6090.53 | 26.92 | 55.78 | 60.68 | 6.27 | 41.38 | 17.28 |
| **EverOS** | 18.98 | 30.8 | 54.67 | 6085.87 | 32.05 | 54.77 | 59.83 | 7.2 | 38.51 | 16.97 |
| **Supermemory** | 14.36 | 32.03 | 58 | 5632.1 | 30.7 | 51.37 | 59.83 | 6.83 | 43.10 | 16.70 |
| **Hindsight** | 16.92 | 35.6 | 55 | 6177 | 38.46 | 56.7 | 71.79 | 14.2 | 50.00 | 17.5 |
| **mem9**<br>**(Cloud service)** | 12.31 | 31.6 | 54.33 | 5880.9 | 30.77 | 59.1 | 64.1 | 8.23 | 47.7 | 15.8 |
| **Memori**<br>**(Cloud service)** | 16.92 | 36.3 | 52 | 6189.2 | 34.62 | 43.1 | 51.28 | 9.3 | 37.93 | 15.8 |
| **MemOS** | 23.85 | 43.3 | 61 | 5164.2 | 38.46 | 49.55 | 64.96 | 9.43 | 62.07 | 23.93 |

### Hermes

| Method | BrowseComp-Plus Acc (%) | BrowseComp-Plus Avg turns | OmniMath Acc (%) | OmniMath Avg chars | SWE-Bench Acc (%) | SWE-Bench Avg turns | LiveCodeBench Acc (%) | LiveCodeBench Avg turns | GDPVal Acc (%) | GDPVal Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | 18.97 | 43.23 | 62 | 10872.8 | 37.18 | 84.93 | 52.99 | 22.27 | 54.6 | 62.57 |
| **Mem0** | 17.95 | 42.73 | 64.33 | 11740 | 34.62 | 94.37 | 57.26 | 25.63 | 56.9 | 57.73 |
| **Supermemory** | 18.46 | 41.03 | 64.00 | 10675.90 | 35.90 | 92.23 | 66.67 | 19.73 | 57.47 | 58.60 |
| **Hindsight** | 21.54 | 48.33 | 64.00 | 12089.63 | 28.20 | 84.47 | 61.54 | 30.07 | 57.47 | 59.67 |
| **Viking** | 21.54 | 39.33 | 58.67 | 12514.20 | 35.90 | 88.70 | 60.68 | 27.77 | 58.05 | 54.27 |
| **mem9**<br>**(Cloud service)** | 18.46 | 40.13 | 66 | 12419.4 | 39.74 | 98.4 | 58.97 | 18.77 | 55.75 | 53.17 |
| **Memori**<br>**(Cloud service)** | 14.87 | 40.13 | 64.33 | 14811 | 28.21 | 91.5 | 54.7 | 22.37 | 55.75 | 53.17 |
| **MemOS** | 20.75 | 36.20 | 72.67 | 5915.7 | 52.56 | 76.37 | 64.1 | 25.63 | 55.17 | 58.6 |

## Contributing Results

Include the evaluated runtime/plugin versions, protocol, task scope, run/trial
counts, configuration, failure handling, aggregation rules, and stable public
artifacts sufficient to recompute the reported metrics. See the
[result evidence requirements](../../CONTRIBUTING.md#submitting-benchmark-results).
An entry in this historical snapshot does not imply that its complete setup is
supported by the current public runner.
