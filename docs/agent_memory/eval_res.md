# AgentBench Evaluation Results

[Chinese](./eval_res_zh.md)

## Metrics

> Acc: the average single-run pass rate across 3 independent runs of the same task, used to measure stable task completion in one run.

> Avg turns: the average number of model response turns triggered by the agent per task, used to measure interaction and reasoning depth.

> Avg chars: the average character length of the agent's response text, used to measure output length.

## Data And Evaluation Setup

Dataset source and split logic:

Reference: [https://huggingface.co/datasets/EverMind-AI/EvoAgentBench](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench)

Evaluated agent: OpenClaw with the corresponding product plugin.

Baseline refers to OpenClaw running without any plugin.

The OpenClaw version used for evaluation is 2026.5.7.

Products not marked as cloud services use locally deployed services and their corresponding OpenClaw plugins.

MemOS uses Memos-Local-Plugin version 2.0.8.

OpenClaw answer model configuration: qwen3.6-flash in no_thinking mode.

Evaluation judge model: qwen3.6-flash in thinking mode.

## Results

| **Method** | BrowseComp-Plus Acc | BrowseComp-Plus Avg turns | OmniMath <br>Acc | OmniMath Cost <br>Avg chars (unit: k tokens) | SWE-Bench <br>Acc | SWE-Bench <br>Avg turns | LiveCodeBench <br>Acc | LiveCodeBench Avg turns | GDPVal <br>Acc | GDPVal <br>Avg turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | 18.46 | 35.1 | 52 | 5658.6 | 26.92 | 58.8 | 51.28 | 23.9 | 34.48 | 17.2 |
| **Mem0** | 13.33 | 36.4 | 53 | 6090.53 | 26.92 | 55.78 | 60.68 | 6.27 | 41.38 | 17.28 |
| **EverOS** | 18.98 | 30.8 | 54.67 | 6085.87 | 32.05 | 54.77 | 59.83 | 7.2 | 38.51 | 16.97 |
| **Supermemory** | 14.36 | 32.03 | 58 | 5632.1 | 30.7 | 51.37 | 59.83 | 6.83 | 43.10 | 16.70 |
| **Hindsight** | 16.92 | 35.6 | 55 | 6177 | 38.46 | 56.7 | 71.79 | 14.2 | 50.00 | 17.5 |
| **mem9**<br>**(Cloud service)** | 12.31 | 31.6 | 54.33 | 5880.9 | 30.77 | 59.1 | 64.1 | 8.23 | 47.7 | 15.8 |
| **Memori**<br>**(Cloud service)** | 16.92 | 36.3 | 52 | 6189.2 | 34.62 | 43.1 | 51.28 | 9.3 | 37.93 | 15.8 |
| **MemOS** | **23.85*** | 43.3 | **61*** | 5164.2 | **38.46*** | 49.55 | <u>64.96</u> | 9.43 | **62.07*** | 23.93 |
