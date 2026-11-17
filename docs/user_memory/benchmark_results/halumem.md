# HaluMem Results

[Results overview](../results.md) · [Evaluation setup](../results.md#evaluation-setup) ·
[Reproduction notes](../results.md#reproduction-notes) · [中文说明](../results_zh.md)

HaluMem evaluates hallucination robustness in memory systems, including fact
recall, boundary detection, conflicting memory handling, generalization,
multi-hop inference, and dynamic updates.

| Question Type | Count | Description |
| --- | ---: | --- |
| Basic Fact Recall | 746 | Extracting concrete facts from memory |
| Memory Boundary | 828 | Recognizing questions outside known memory rather than fabricating answers |
| Memory Conflict | 769 | Resolving conflicting memory records |
| Generalization & Application | 746 | Reasoning and applying known memories |
| Multi-hop Inference | 198 | Connecting multiple memories for inference |
| Dynamic Update | 180 | Answering with the latest valid state after memory updates |

## Reproduced Results

[LCM-X](../lcmx.md#result-provenance), [Maasv](../maasv.md#result-provenance),
and [ThreadAtlas](../threadatlas.md#result-provenance) rows are contributor-run.
LCM-X used OpenRouter; its exact dated answer-model revision is unconfirmed.
See [run notes and limitations](../results.md#contributor-run-notes) before
comparing these rows.

| Backend | Deployment | Basic Fact | Boundary | Conflict | Generalization | Multi-Hop | Dynamic Update | Overall | Context Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 | cloud | 56.30 | 91.91 | 74.77 | 78.42 | 68.69 | 42.22 | 73.64 | 803 |
| graphiti-zep | local/self-hosted | 69.44 | 89.13 | 88.43 | 84.05 | 79.80 | 62.22 | 81.71 | 5,404 |
| Supermemory | cloud | 25.20 | 97.46 | 44.86 | 49.87 | 41.92 | 16.11 | 52.61 | 1,672 |
| Viking | cloud | 57.91 | 92.51 | 80.49 | 84.72 | 74.75 | 47.78 | 77.39 | 3,196 |
| Cognee | local/self-hosted | 52.41 | 92.51 | 74.64 | 78.28 | 69.70 | 35.56 | 72.60 | 8,981 |
| Letta | cloud | 85.92 | 83.45 | 85.70 | 90.62 | 84.34 | 71.11 | 85.43 | 45,349 |
| Hindsight | local/self-hosted | 78.42 | 87.45 | 86.61 | 86.19 | 81.31 | 73.89 | 83.99 | 14,798 |
| Memori | cloud | 20.91 | 93.24 | 41.35 | 53.35 | 24.24 | 11.11 | 49.38 | 3,275 |
| EverOS | local/self-hosted | 87.80 | 88.89 | 89.99 | 90.48 | 86.36 | 80.56 | 88.66 | 10,824 |
| MemMachine | local/self-hosted | 20.38 | 97.42 | 34.72 | 45.31 | 26.77 | 11.11 | 47.02 | 1,093 |
| mem9 | cloud | 54.56 | 94.20 | 68.79 | 80.16 | 70.71 | 38.89 | 72.80 | 893 |
| MemOS | cloud | 69.03 | 91.30 | 86.48 | 83.51 | 74.24 | 55.00 | 80.91 | 1,187 |
| LCM-X | local/self-hosted | 73.06 | 89.73 | 85.96 | 82.31 | 78.79 | 62.78 | 81.68 | 1,985 |
| Maasv | local/self-hosted | 71.72 | 91.91 | 87.52 | 82.17 | 79.80 | 62.78 | 82.29 | 2,241 |
| ThreadAtlas | local/self-hosted | 75.34 | 92.63 | 88.43 | 86.06 | 78.28 | 55.56 | 83.82 | 3,052 |

## Published Reference Results

These externally reported scores may use different data, models, prompts or
budgets. They are not directly comparable to the reproduced results above.

| Backend | Overall | Context Tokens | Source |
| --- | ---: | ---: | --- |
| EverOS | 93.04 | - | [EverMind](https://evermind.ai/) |
| Hojo MemoryEngine | 77.33 | - | [Hojo evaluation details](../hojo_reference_results.md#halumem-medium) |
