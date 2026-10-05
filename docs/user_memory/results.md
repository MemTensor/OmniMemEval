# Benchmark Results

[User Memory Evaluation guide](./README.md) ·
[Documentation index](../README.md)

This document provides a public result snapshot for OmniMemEval's current
benchmark pipelines. The reproduced scores were generated under one evaluation
harness so that memory backends are compared with the same data, prompts,
answer model, judge model, and metric logic within each benchmark.

These results are intended to make comparison and reproduction easier. For each
backend, the adapter and run configuration were prepared according to the
product's public documentation, API reference, and available benchmark guidance.
They are not a claim that every adapter has reached a globally optimal
product-specific configuration. Contributions that improve an adapter's
documented setup or default parameters are welcome.

## Evaluation Setup

All reproduced runs used the same baseline evaluation configuration:

| Component | Configuration |
| --- | --- |
| Benchmarks | LoCoMo, LongMemEval, BEAM, PersonaMem v2, HaluMem |
| Memory service model | `gpt-4.1-mini-2025-04-14` where the backend requires a model setting |
| Answer model | `gpt-4.1-mini-2025-04-14` |
| Judge model | `gpt-4o-mini-2024-07-18` |
| Primary metric | LLM-as-a-judge accuracy for LoCoMo, LongMemEval, and HaluMem; nugget score for BEAM; rule-matching accuracy for PersonaMem v2 |
| Efficiency metric | Average answer-stage context tokens |

The reported `Context Tokens` value is the average number of tokens sent to the
answer model per question, including the answer prompt and the retrieved context
rendered by the memory backend. Lower context tokens indicate better token
efficiency when accuracy is comparable.

Rows marked `local/self-hosted` were evaluated through a local or self-hosted
service deployment because the managed cloud service was unavailable,
insufficient for the full run, or not the recommended evaluation route at the
time of testing.

Published reference scores are included only as external context. They may use
different models, prompts, retrieval settings, context budgets, data versions,
or judge implementations, and should not be treated as directly comparable to
the reproduced OmniMemEval scores.

## Result Summary

| Backend | LoCoMo | LongMemEval | BEAM 100K | BEAM 10M | PersonaMem v2 | HaluMem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 | 77.68 | 56.00 | 70.41 | 43.33 | 36.76 | 73.64 |
| Zep / Graphiti | 63.83 | 79.80 | 68.70 | 56.11 | 32.34 | 81.71 |
| Supermemory | 73.53 | 66.07 | 65.98 | 52.49 | 39.64 | 52.61 |
| Viking | 69.33 | 61.07 | 70.76 | 58.14 | 30.80 | 77.39 |
| Cognee | 83.48 | 51.80 | 59.30 | 56.02 | 26.46 | 72.60 |
| Letta | 77.12 | 77.67 | 69.22 | 52.30 | 35.12 | 85.43 |
| Hindsight | 81.99 | 72.20 | 70.22 | 59.75 | 37.98 | 83.99 |
| Memori | 41.34 | 20.80 | - | - | 33.16 | 49.38 |
| EverOS | 82.75 | 80.40 | 58.64 | 47.73 | 35.94 | 88.66 |
| MemMachine | 73.90 | 63.60 | 64.80 | 51.90 | 34.14 | 47.02 |
| mem9 | 73.64 | 78.00 | 65.75 | 57.30 | 30.76 | 72.80 |
| MemoryLake | 72.49 | - | - | - | - | - |
| Backboard.io | 22.40 | - | - | - | - | - |
| MemOS | 88.83 | 89.20 | 66.87 | 56.75 | 40.58 | 80.91 |

A dash (`-`) means that a reproduced result is not included in this snapshot.
For these missing cells, full runs were not completed under the same evaluation
setup because of account/API access, service availability, benchmark support,
or run-cost constraints. Partial or non-comparable runs are excluded rather than
mixed into the reproduced result tables.

## LoCoMo

LoCoMo evaluates long-conversation memory with multi-hop, temporal, and
open-domain question answering. The reproduced evaluation excludes category 5
adversarial questions and covers 1,540 questions.

| Category | Count | Description |
| --- | ---: | --- |
| Single-Hop | 841 | Direct fact extraction from one evidence source |
| Multi-Hop | 282 | Reasoning over multiple conversation turns |
| Temporal | 321 | Time-aware retrieval and temporal reasoning |
| Open-Domain | 96 | Open-ended reasoning over multiple pieces of evidence |

### Reproduced Results

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

### Published Reference Results

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

## LongMemEval

LongMemEval evaluates long-term interactive memory across sessions. The
OmniMemEval public pipeline uses the cleaned LongMemEval-S data by default.

| Category | Count | Description |
| --- | ---: | --- |
| single-session-user | 70 | User fact extraction from one historical session |
| single-session-assistant | 56 | Assistant-provided information extraction from one historical session |
| single-session-preference | 30 | User preference inference from one historical session |
| temporal-reasoning | 133 | Time-aware reasoning over session timestamps |
| multi-session | 133 | Reasoning over information from multiple sessions |
| knowledge-update | 78 | Selecting the latest valid answer after information changes |

### Reproduced Results

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

### Published Reference Results

| Backend | SS-User | SS-Asst | SS-Pref | Temp. Reas | Multi-S | Know. Upd | Overall | Context Tokens | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mem0 | 98.6 | 98.2 | 96.7 | 93.6 | 88.0 | 97.0 | 94.4 | 6,787 | [mem0.ai research](https://mem0.ai/research) |
| Zep | 94.3 | 96.4 | 90.0 | 90.2 | 83.5 | 93.6 | 90.2 | 4,408 | [getzep research](https://www.getzep.com/research/) |
| Supermemory | 97.0 | 100.0 | 90.0 | 91.0 | 93.0 | 99.0 | 95.0 | - | [Supermemory LongMemBench](https://supermemory.ai/research/longmembench/) |
| Hindsight | - | - | - | - | - | - | 94.6 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| EverOS | 97.14 | 85.71 | 93.33 | 77.44 | 73.68 | 89.74 | 83.0 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |
| Backboard.io | 97.1 | 98.2 | 90.0 | 91.7 | 91.7 | 93.6 | 93.4 | - | [Backboard LongMemEval repo](https://github.com/Backboard-io/Backboard-longmemEval-results) |

## BEAM

BEAM evaluates long-term memory at different context scales. The public
OmniMemEval runner supports 100K, 500K, 1M, and 10M scales; the reproduced
snapshot below reports the 100K and 10M scales from the current result set.

BEAM uses nugget score rather than binary accuracy. Nugget score measures how
well a generated answer covers the atomic reference facts, with 1.0 for fully
covered, 0.5 for partially covered, and 0.0 for incorrect or missing evidence.

| Scale | Questions | Description |
| --- | ---: | --- |
| 100K | 400 | Baseline long-memory scale, approximately 128K tokens per conversation |
| 10M | 200 | Extreme long-memory scale, approximately 10M tokens per conversation |

### Reproduced Results

| Backend | Deployment | 100K Nugget Score | 100K Context Tokens | 10M Nugget Score | 10M Context Tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Mem0 | cloud | 70.41 +/- 36.36 | 1,055 | 43.33 +/- 41.99 | 1,086 |
| graphiti-zep | local/self-hosted | 68.70 +/- 36.46 | 8,667 | 56.11 +/- 40.70 | 176,211 |
| Supermemory | cloud | 65.98 +/- 37.10 | 6,294 | 52.49 +/- 40.44 | 6,574 |
| Viking | cloud | 70.76 +/- 35.23 | 2,023 | 58.14 +/- 39.57 | 2,080 |
| Cognee | local/self-hosted | 59.30 +/- 39.44 | 33,065 | 56.02 +/- 41.33 | 40,914 |
| Letta | cloud | 69.22 +/- 36.76 | 82,013 | 52.30 +/- 41.06 | 49,786 |
| Hindsight | local/self-hosted | 70.22 +/- 36.44 | 24,085 | 59.75 +/- 39.63 | 23,815 |
| EverOS | local/self-hosted | 58.64 +/- 39.36 | 7,393 | 47.73 +/- 41.66 | 11,657 |
| MemMachine | local/self-hosted | 64.80 +/- 37.62 | 5,413 | 51.90 +/- 40.46 | 5,448 |
| mem9 | cloud | 65.75 +/- 38.92 | 5,372 | 57.30 +/- 39.03 | 4,947 |
| MemOS | cloud | 66.87 +/- 37.37 | 1,636 | 56.75 +/- 39.17 | 1,558 |

### Published Reference Results

| Backend | 100K Nugget Score | 100K Context Tokens | 10M Nugget Score | 10M Context Tokens | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Mem0 | 64.1 | 6,719 | 48.6 | 6,914 | [mem0.ai research](https://mem0.ai/research) |
| Hindsight | 75.0 | - | 64.1 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| past.dev | 92.08 | - | 85.03 | - | [past.dev BEAM leaderboard](https://past.dev/benchmarks/beam) |

## PersonaMem v2

PersonaMem v2 evaluates personalized memory and preference-aware multiple-choice
question answering. Accuracy is computed by matching the selected option against
the gold answer; repeated runs are averaged.

| Preference Type | Count | Description |
| --- | ---: | --- |
| ask_to_forget | 1,048 | Preferences that the user later asks the system to forget |
| neutral_preferences | 858 | Neutral personal preferences |
| anti_stereotypical_pref | 855 | Preferences that go against stereotypes |
| therapy_background | 627 | Therapy or mental-health background preferences |
| health_and_medical_conditions | 568 | Health and medical-condition preferences |
| stereotypical_pref | 533 | Stereotypical preference cases |
| sensitive_info | 511 | Sensitive personal information cases |

### Reproduced Results

| Backend | Deployment | Anti-Stereotypical | Ask-to-Forget | Health/Medical | Neutral | Sensitive | Stereotypical | Therapy | Overall | Context Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 | cloud | 41.40 | 24.33 | 38.38 | 39.28 | 30.72 | 45.78 | 43.54 | 36.76 | 1,388 |
| graphiti-zep | local/self-hosted | 28.19 | 29.77 | 28.70 | 31.93 | 30.92 | 37.90 | 42.58 | 32.34 | 3,645 |
| Supermemory | cloud | 42.34 | 31.97 | 42.96 | 42.42 | 30.72 | 49.72 | 40.67 | 39.64 | 4,473 |
| Viking | cloud | 31.58 | 25.38 | 29.23 | 29.49 | 33.66 | 36.59 | 34.77 | 30.80 | 1,688 |
| Cognee | local/self-hosted | 18.83 | 41.13 | 16.90 | 18.65 | 32.88 | 16.51 | 34.93 | 26.46 | 10,189 |
| Letta | cloud | 31.46 | 38.93 | 32.39 | 34.85 | 27.40 | 39.40 | 39.23 | 35.12 | 30,903 |
| Hindsight | local/self-hosted | 33.57 | 40.65 | 36.62 | 38.11 | 36.01 | 42.78 | 38.12 | 37.98 | 15,926 |
| Memori | cloud | 30.18 | 27.48 | 32.39 | 37.53 | 33.46 | 36.77 | 38.12 | 33.16 | 3,109 |
| EverOS | cloud | 33.57 | 31.49 | 34.68 | 37.76 | 33.27 | 44.47 | 40.19 | 35.94 | 6,572 |
| MemMachine | local/self-hosted | 28.19 | 45.04 | 32.22 | 28.44 | 29.75 | 32.46 | 38.60 | 34.14 | 1,988 |
| mem9 | cloud | 32.28 | 22.90 | 32.57 | 34.50 | 25.05 | 38.27 | 33.33 | 30.76 | 2,045 |
| MemOS | cloud | 33.80 | 57.82 | 32.57 | 35.66 | 36.59 | 39.40 | 39.23 | 40.58 | 1,908 |

### Published Reference Results

| Backend | Overall | Context Tokens | Source |
| --- | ---: | ---: | --- |
| EverOS | 53.25 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |

## HaluMem

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

### Reproduced Results

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

### Published Reference Results

| Backend | Overall | Context Tokens | Source |
| --- | ---: | ---: | --- |
| EverOS | 93.04 | - | [EverMind](https://evermind.ai/) |

## Reproduction Notes

To reproduce a run, configure one of the templates under `env_examples/`, then
run the corresponding benchmark script:

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
./scripts/run_beam_eval.sh --lib memos --env .env.memos
./scripts/run_pmv2_eval.sh --lib memos --env .env.memos
./scripts/run_halumem_eval.sh --lib memos --env .env.memos
```

Replace `memos` with another adapter key to evaluate a different backend under
the same benchmark pipeline. Use `--version <name>` to isolate result
directories and make comparisons explicit.

Result artifacts are written under:

```text
results/locomo/{LIB}-{VERSION}/
results/lme/{LIB}-{VERSION}/
results/beam/{LIB}-{VERSION}/
results/pmv2/{LIB}-{VERSION}/
results/halumem/{LIB}-{VERSION}/
```

Benchmark datasets are downloaded on demand and are not committed to this
repository. See [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for dataset
license information.
