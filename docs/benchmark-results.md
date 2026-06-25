# Benchmark Results

This document provides a public result snapshot for OmniMemEval's current
LoCoMo and LongMemEval evaluation pipelines. The reproduced scores were
generated under one evaluation harness so that memory backends are compared
with the same data, prompts, answer model, judge model, and metric logic.

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
| Benchmarks | LoCoMo, LongMemEval |
| Memory service model | `gpt-4.1-mini-2025-04-14` where the backend requires a model setting |
| Answer model | `gpt-4.1-mini-2025-04-14` |
| Judge model | `gpt-4o-mini-2024-07-18` |
| Judge metric | LLM-as-a-judge accuracy |
| Efficiency metric | Average answer-stage context tokens |

For LoCoMo and LongMemEval, the reported `Context Tokens` value is the average
number of tokens sent to the answer model per question, including the answer
prompt and the retrieved context rendered by the memory backend. Lower context
tokens indicate better token efficiency when accuracy is comparable.

Rows marked `local/self-hosted` were evaluated through a local or self-hosted
service deployment because the managed cloud service was unavailable,
insufficient for the full run, or not the recommended evaluation route at the
time of testing.

Published reference scores are included only as external context. They may use
different models, prompts, retrieval settings, context budgets, data versions,
or judge implementations, and should not be treated as directly comparable to
the reproduced OmniMemEval scores.

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
| Viking | cloud | 78.04 | 73.29 | 48.81 | 50.00 | 69.33 | 5,964 |
| Letta | cloud | 87.99 | 76.24 | 53.48 | 63.54 | 77.12 | 14,188 |
| Supermemory | cloud | 75.39 | 77.07 | 67.60 | 66.67 | 73.53 | 15,238 |
| Cognee | cloud | 87.99 | 78.84 | 81.83 | 63.19 | 83.48 | 32,532 |
| Memori | cloud | 47.32 | 44.09 | 22.53 | 43.75 | 41.34 | 8,139 |
| Hindsight | cloud | 88.98 | 78.84 | 73.52 | 58.33 | 81.99 | 24,683 |
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
| Letta | - | - | - | - | 74.0 | - | [Letta benchmark blog](https://www.letta.com/blog/benchmarking-ai-agent-memory) |
| Supermemory | - | - | - | - | 77.1 | - | [Supermemory issue 795](https://github.com/supermemoryai/supermemory/issues/795) |
| Memori | 87.87 | 72.70 | 80.37 | 63.54 | 81.95 | 1,294 | [Memori benchmark](https://memorilabs.ai/docs/memori-cloud/benchmark/results/) |
| Hindsight | - | - | - | - | 92.0 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| EverOS | 96.67 | 91.84 | 89.72 | 76.04 | 93.05 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |
| mem9 | 89.71 | 83.16 | 89.25 | 64.58 | 86.85 | - | [mem9](https://mem9.ai/) |
| MemoryLake | 96.79 | 91.84 | 91.28 | 85.42 | 94.03 | - | [MemoryLake benchmark](https://www.memorylake.ai/products/compare/benchmarks) |
| Backboard.io | 89.36 | 75.00 | 91.90 | 91.20 | 90.00 | - | [Backboard LoCoMo repo](https://github.com/Backboard-io/Backboard-Locomo-Benchmark) |

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
| Viking | cloud | 75.24 | 46.43 | 96.67 | 55.39 | 57.89 | 60.26 | 61.07 | 2,291 |
| Letta | cloud | 95.71 | 98.21 | 76.67 | 69.42 | 65.41 | 82.05 | 77.67 | 49,431 |
| Supermemory | cloud | 87.14 | 41.07 | 65.56 | 68.42 | 60.15 | 71.37 | 66.07 | 6,635 |
| Cognee | local/self-hosted | 67.14 | 60.71 | 83.33 | 47.37 | 37.59 | 51.28 | 51.80 | 10,305 |
| Memori | cloud | 84.14 | 1.79 | 23.33 | 3.76 | 18.80 | 6.41 | 20.80 | 2,779 |
| Hindsight | local/self-hosted | 82.86 | 14.29 | 96.67 | 82.71 | 71.43 | 78.21 | 72.20 | 29,755 |
| EverOS | local/self-hosted | 91.43 | 89.29 | 96.67 | 81.95 | 66.17 | 79.49 | 80.40 | 12,379 |
| MemMachine | local/self-hosted | 75.71 | 96.43 | 83.33 | 55.64 | 39.85 | 75.64 | 63.60 | 2,803 |
| mem9 | cloud | 95.71 | 94.64 | 56.67 | 77.44 | 62.41 | 85.90 | 78.00 | 3,805 |
| MemOS | cloud | 100.00 | 100.00 | 100.00 | 89.47 | 78.95 | 84.62 | 89.20 | 4,151 |

MemoryLake and Backboard.io are not included in the reproduced LongMemEval
table because full runs were not completed under the same evaluation setup due
to account/API access and run-cost constraints.

### Published Reference Results

| Backend | SS-User | SS-Asst | SS-Pref | Temp. Reas | Multi-S | Know. Upd | Overall | Context Tokens | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mem0 | 98.6 | 98.2 | 96.7 | 93.6 | 88.0 | 97.0 | 94.4 | 6,787 | [mem0.ai research](https://mem0.ai/research) |
| Zep | 94.3 | 96.4 | 90.0 | 90.2 | 83.5 | 93.6 | 90.2 | 4,408 | [getzep research](https://www.getzep.com/research/) |
| Supermemory | 97.0 | 100.0 | 90.0 | 91.0 | 93.0 | 99.0 | 95.0 | - | [Supermemory LongMemBench](https://supermemory.ai/research/longmembench/) |
| Hindsight | - | - | - | - | - | - | 94.6 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| EverOS | 97.14 | 85.71 | 93.33 | 77.44 | 73.68 | 89.74 | 83.0 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |
| Backboard.io | 97.1 | 98.2 | 90.0 | 91.7 | 91.7 | 93.6 | 93.4 | - | [Backboard LongMemEval repo](https://github.com/Backboard-io/Backboard-longmemEval-results) |

## Reproduction Notes

To reproduce a run, configure one of the templates under `env_examples/`, then
run the corresponding benchmark script:

```bash
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
./scripts/run_lme_eval.sh --lib memos --env .env.memos
```

Replace `memos` with another adapter key to evaluate a different backend under
the same benchmark pipeline. Use `--version <name>` to isolate result
directories and make comparisons explicit.

Result artifacts are written under:

```text
results/locomo/{LIB}-{VERSION}/
results/lme/{LIB}-{VERSION}/
```

Benchmark datasets are downloaded on demand and are not committed to this
repository. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for dataset
license information.
