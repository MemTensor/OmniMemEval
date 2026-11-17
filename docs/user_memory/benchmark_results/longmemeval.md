# LongMemEval Results

[Results overview](../results.md) · [Evaluation setup](../results.md#evaluation-setup) ·
[Reproduction notes](../results.md#reproduction-notes) · [中文说明](../results_zh.md)

LongMemEval evaluates long-term interactive memory across sessions. The
OmniMemEval public pipeline uses the cleaned LongMemEval-S data by default.
The `Overall` score is the question-count-weighted mean across all 500
questions; it is not the unweighted mean of the six category percentages.

| Category | Count | Description |
| --- | ---: | --- |
| single-session-user | 70 | User fact extraction from one historical session |
| single-session-assistant | 56 | Assistant-provided information extraction from one historical session |
| single-session-preference | 30 | User preference inference from one historical session |
| temporal-reasoning | 133 | Time-aware reasoning over session timestamps |
| multi-session | 133 | Reasoning over information from multiple sessions |
| knowledge-update | 78 | Selecting the latest valid answer after information changes |

## Reproduced Results

[LCM-X](../lcmx.md#result-provenance), [Maasv](../maasv.md#result-provenance),
[cachly](../cachly.md#result-provenance) and [ThreadAtlas](../threadatlas.md#result-provenance) rows are contributor-run.
LCM-X used OpenRouter; its exact dated answer-model revision is unconfirmed.
See [run notes and limitations](../results.md#contributor-run-notes) before
comparing these rows.

| Backend | Deployment | SS-User | SS-Asst | SS-Pref | Temp. Reas | Multi-S | Know. Upd | Overall | Context Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 | cloud | 8.57 | 85.71 | 96.67 | 51.13 | 50.38 | 79.49 | 56.00 | 856 |
| graphiti-zep | local/self-hosted | 94.29 | 100.00 | 86.67 | 74.44 | 67.67 | 79.49 | 79.80 | 117,106 |
| Supermemory | cloud | 87.14 | 41.07 | 65.56 | 68.42 | 60.15 | 71.37 | 66.07 | 6,635 |
| Viking | cloud | 75.24 | 46.43 | 96.67 | 55.39 | 57.89 | 60.26 | 61.07 | 2,291 |
| Cognee | local/self-hosted | 67.14 | 60.71 | 83.33 | 47.37 | 37.59 | 51.28 | 51.80 | 10,305 |
| Letta | cloud | 95.71 | 98.21 | 76.67 | 69.42 | 65.41 | 82.05 | 77.67 | 49,431 |
| Hindsight | local/self-hosted | 82.86 | 14.29 | 96.67 | 82.71 | 71.43 | 78.21 | 72.20 | 29,755 |
| Memori | cloud | 84.14 | 1.79 | 23.33 | 3.76 | 18.80 | 6.41 | 20.80 | 2,779 |
| EverOS | local/self-hosted | 91.43 | 89.29 | 96.67 | 81.95 | 66.17 | 79.49 | 80.40 | 12,379 |
| MemMachine | local/self-hosted | 75.71 | 96.43 | 83.33 | 55.64 | 39.85 | 75.64 | 63.60 | 2,803 |
| mem9 | cloud | 95.71 | 94.64 | 56.67 | 77.44 | 62.41 | 85.90 | 78.00 | 3,805 |
| MemOS | cloud | 100.00 | 100.00 | 100.00 | 89.47 | 78.95 | 84.62 | 89.20 | 4,151 |
| LCM-X | local/self-hosted | 95.71 | 94.64 | 96.67 | 85.71 | 75.19 | 92.31 | 87.00 | 5,164 |
| Maasv | local/self-hosted | 98.57 | 98.21 | 93.33 | 89.47 | 86.47 | 89.74 | 91.20 | 3,999 |
| cachly | local/self-hosted | 100.00 | 98.21 | 86.67 | 87.97 | 83.46 | 88.46 | 89.60 | 4,564 |
| ThreadAtlas | local/self-hosted | 100.00 | 96.43 | 93.33 | 84.21 | 77.44 | 87.18 | 87.00 | 4,351 |

## Published Reference Results

These externally reported scores may use different data, models, prompts or
budgets. They are not directly comparable to the reproduced results above.

| Backend | SS-User | SS-Asst | SS-Pref | Temp. Reas | Multi-S | Know. Upd | Overall | Context Tokens | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mem0 | 98.6 | 98.2 | 96.7 | 93.6 | 88.0 | 97.0 | 94.4 | 6,787 | [mem0.ai research](https://mem0.ai/research) |
| Zep | 94.3 | 96.4 | 90.0 | 90.2 | 83.5 | 93.6 | 90.2 | 4,408 | [getzep research](https://www.getzep.com/research/) |
| Supermemory | 97.0 | 100.0 | 90.0 | 91.0 | 93.0 | 99.0 | 95.0 | - | [Supermemory LongMemBench](https://supermemory.ai/research/longmembench/) |
| Hindsight | - | - | - | - | - | - | 94.6 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| EverOS | 97.14 | 85.71 | 93.33 | 77.44 | 73.68 | 89.74 | 83.0 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |
| Backboard.io | 97.1 | 98.2 | 90.0 | 91.7 | 91.7 | 93.6 | 93.4 | - | [Backboard LongMemEval repo](https://github.com/Backboard-io/Backboard-longmemEval-results) |
| CPersona (pro) | 97.14 | 94.64 | 96.67 | 86.47 | 85.71 | 88.46 | 89.60 | 4,262 | [CPersona run records](https://github.com/Cloto-dev/cpersona-omnimemeval-2.6.11) (this harness at `0b1ea8d`; answers `gpt-4.1-mini-2025-04-14`, judge `gpt-4o-mini`) |
| CPersona (lite) | 95.71 | 83.93 | 90.00 | 80.45 | 74.44 | 82.05 | 82.20 | 993 | [CPersona run records](https://github.com/Cloto-dev/cpersona-omnimemeval-2.6.11) (this harness at `0b1ea8d`; answers `gpt-4.1-mini-2025-04-14`, judge `gpt-4o-mini`) |
| Hojo MemoryEngine | 98.57 | 78.57 | 83.33 | 78.95 | 68.42 | 88.46 | 80.60 | - | [Hojo evaluation details](../hojo_reference_results.md#longmemeval-s) |
| ThreadAtlas | 100.00 | 96.43 | 86.67 | 83.46 | 79.70 | 89.74 | 87.40 | 4,332 | [ThreadAtlas run records](https://github.com/jhffmn82/threadatlas/tree/dff7388606c3979c410195e28ce0b59cf4faa97a/omnimemeval) (contributor-run, 500 questions; `gpt-5.6-luna` memory reading and prebuilt sessions from the original dataset; [methodology](../threadatlas.md#version-deployment-and-settings-of-the-reported-runs)) |
