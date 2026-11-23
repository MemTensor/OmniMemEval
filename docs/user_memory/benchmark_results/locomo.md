# LoCoMo Results

[Results overview](../results.md) · [Evaluation setup](../results.md#evaluation-setup) ·
[Reproduction notes](../results.md#reproduction-notes) · [中文说明](../results_zh.md)

LoCoMo evaluates long-conversation memory with multi-hop, temporal, and
open-domain question answering. The reproduced evaluation excludes category 5
adversarial questions and covers 1,540 questions.

| Category | Count | Description |
| --- | ---: | --- |
| Single-Hop | 841 | Direct fact extraction from one evidence source |
| Multi-Hop | 282 | Reasoning over multiple conversation turns |
| Temporal | 321 | Time-aware retrieval and temporal reasoning |
| Open-Domain | 96 | Open-ended reasoning over multiple pieces of evidence |

## Reproduced Results

[LCM-X](../lcmx.md#result-provenance), [Maasv](../maasv.md#result-provenance),
[cachly](../cachly.md#result-provenance), [ThreadAtlas](../threadatlas.md#result-provenance) and
[CPersona](../cpersona.md#result-provenance) rows are contributor-run.
LCM-X used OpenRouter; its exact dated answer-model revision is unconfirmed.
See [run notes and limitations](../results.md#contributor-run-notes) before
comparing these rows.

| Backend | Deployment | Single-Hop | Multi-Hop | Temporal | Open-Domain | Overall | Context Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 | cloud | 81.09 | 76.12 | 77.15 | 54.17 | 77.68 | 17,395 |
| Zep | cloud | 65.36 | 68.79 | 55.56 | 63.54 | 63.83 | 1,862 |
| Supermemory | cloud | 75.39 | 77.07 | 67.60 | 66.67 | 73.53 | 15,238 |
| Viking | cloud | 78.04 | 73.29 | 48.81 | 50.00 | 69.33 | 5,964 |
| Cognee | cloud | 87.99 | 78.84 | 81.83 | 63.19 | 83.48 | 32,532 |
| Letta | cloud | 87.99 | 76.24 | 53.48 | 63.54 | 77.12 | 14,188 |
| Hindsight | cloud | 88.98 | 78.84 | 73.52 | 58.33 | 81.99 | 24,683 |
| Memori | cloud | 47.32 | 44.09 | 22.53 | 43.75 | 41.34 | 8,139 |
| EverOS | cloud | 86.80 | 77.78 | 84.11 | 57.29 | 82.75 | 8,559 |
| MemMachine | local/self-hosted | 83.47 | 53.19 | 71.96 | 57.29 | 73.90 | 2,577 |
| mem9 | cloud | 79.27 | 62.88 | 73.62 | 55.90 | 73.64 | 1,597 |
| MemoryLake | cloud | 70.87 | 75.30 | 79.75 | 54.17 | 72.49 | 5,202 |
| Backboard.io | cloud | 25.09 | 22.34 | 13.40 | 29.17 | 22.40 | 1,198 |
| MemOS | cloud | 92.51 | 88.65 | 85.05 | 69.79 | 88.83 | 5,400 |
| LCM-X | local/self-hosted | 84.30 | 66.31 | 79.75 | 64.58 | 78.83 | 1,659 |
| Maasv | local/self-hosted | 91.80 | 82.98 | 85.98 | 71.88 | 87.73 | 1,551 |
| cachly | local/self-hosted | 95.01 | 87.94 | 86.29 | 76.04 | 90.71 | 6,673 |
| ThreadAtlas | local/self-hosted | 93.22 | 85.11 | 86.29 | 70.83 | 88.90 | 6,213 |
| CPersona | local/self-hosted | 88.70 | 78.01 | 83.18 | 63.54 | 84.03 | 4,441 |

## Published Reference Results

These externally reported scores may use different data, models, prompts or
budgets. They are not directly comparable to the reproduced results above.

| Backend | Single-Hop | Multi-Hop | Temporal | Open-Domain | Overall | Context Tokens | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mem0 | 94.6 | 95.4 | 92.5 | 82.3 | 92.5 | 6,956 | [mem0.ai research](https://mem0.ai/research) |
| Zep | 96.4 | 94.0 | 95.6 | 79.2 | 94.7 | 5,760 | [getzep research](https://www.getzep.com/research/) |
| Supermemory | - | - | - | - | 77.1 | - | [Supermemory issue 795](https://github.com/supermemoryai/supermemory/issues/795) |
| Letta | - | - | - | - | 74.0 | - | [Letta benchmark blog](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| Hindsight | - | - | - | - | 92.0 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| Memori | 87.87 | 72.70 | 80.37 | 63.54 | 81.95 | 1,294 | [Memori benchmark](https://memorilabs.ai/docs/memori-cloud/benchmark/results/) |
| EverOS | 96.67 | 91.84 | 89.72 | 76.04 | 93.05 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |
| mem9 | 89.71 | 83.16 | 89.25 | 64.58 | 86.85 | - | [mem9](https://mem9.ai/) |
| MemoryLake | 96.79 | 91.84 | 91.28 | 85.42 | 94.03 | - | [MemoryLake benchmark](https://www.memorylake.ai/products/compare/benchmarks) |
| Backboard.io | 89.36 | 75.00 | 91.90 | 91.20 | 90.00 | - | [Backboard LoCoMo repo](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) |
| past.dev | 95.01 | 93.26 | 93.46 | 75.00 | 93.12 | - | [past.dev benchmarks](https://past.dev/benchmarks) (community-corrected answer keys) |
| CPersona (pro) | 88.70 | 78.01 | 83.18 | 63.54 | 84.03 | 4,441 | [CPersona run records](https://github.com/Cloto-dev/cpersona-omnimemeval-2.6.11) (this harness at `0b1ea8d`; the 1,540-question non-adversarial subset, categories 1–4; answers `gpt-4.1-mini-2025-04-14`, judge `gpt-4o-mini`) |
| CPersona (lite) | 73.84 | 51.06 | 72.59 | 54.17 | 68.18 | 1,485 | [CPersona run records](https://github.com/Cloto-dev/cpersona-omnimemeval-2.6.11) (this harness at `0b1ea8d`; the 1,540-question non-adversarial subset, categories 1–4; answers `gpt-4.1-mini-2025-04-14`, judge `gpt-4o-mini`) |
| Hojo MemoryEngine | 92.98 | 81.91 | 85.36 | 72.92 | 88.12 | - | [Hojo evaluation details: 1,540-question non-adversarial subset](../hojo_reference_results.md#locomo) |
| ThreadAtlas | 94.17 | 83.69 | 86.29 | 71.88 | 89.22 | 6,190 | [ThreadAtlas run records](https://github.com/jhffmn82/threadatlas/tree/dff7388606c3979c410195e28ce0b59cf4faa97a/omnimemeval) (contributor-run, 1,540 non-adversarial questions; memory reading uses `gpt-5.6-luna`, not the shared baseline; [methodology](../threadatlas.md#version-deployment-and-settings-of-the-reported-runs)) |
