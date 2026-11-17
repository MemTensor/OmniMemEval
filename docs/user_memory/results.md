<a id="benchmark-results"></a>
# User Memory Evaluation Results

[中文版](./results_zh.md) · [Documentation index](../README.md) ·
[Guide](./README.md)

This page provides the cross-benchmark summary and shared evaluation setup.
The linked benchmark pages contain the detailed numeric tables; the Chinese
page provides translated explanations and links to those tables.

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

The baseline requested evaluation configuration is below. Contributor-run
endpoint and model-resolution limitations are disclosed alongside their rows.

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

| Backend | [LoCoMo](./benchmark_results/locomo.md) | [LongMemEval](./benchmark_results/longmemeval.md) | [BEAM 100K](./benchmark_results/beam.md) | [BEAM 10M](./benchmark_results/beam.md) | [PersonaMem v2](./benchmark_results/personamem-v2.md) | [HaluMem](./benchmark_results/halumem.md) |
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
| LCM-X | 78.83 | 87.00 | 71.84 | 63.32 | 33.96 | 81.68 |
| Maasv | 87.73 | 91.20 | 76.38 | 65.09 | 39.86 | 82.29 |
| cachly | 90.71 | 89.60 | 73.04 | - | - | - |
| ThreadAtlas | 88.90 | 87.00 | 73.12 | - | 36.92 | 83.82 |

### Contributor-run notes

These rows were run by contributors. Versions, configurations, evidence and
run history are maintained in the linked backend documentation.

| Backend / run author | Notes and provenance |
| --- | --- |
| [LCM-X](./lcmx.md#result-provenance) / LCM-X maintainers | OpenRouter receipts do not confirm the exact dated answer-model revision or equivalence to direct OpenAI calls; they do not establish a different model either. [BEAM 10M](./lcmx.md#beam-10m-run-2026-10-22) is a separate run. |
| [Maasv](./maasv.md#result-provenance) / Maasv team | Direct OpenAI dated models; self-hosted, closed-source backend. Public records support artifact arithmetic and answer/judge replay. |
| [cachly](./cachly.md#result-provenance) / Heinrich Neb | Direct OpenAI dated models; private reranker and disclosed benchmark-driven tuning. LongMemEval replaces an earlier run affected by a time-zone fault. |
| [ThreadAtlas](./threadatlas.md#result-provenance) / Justin Hoffman | Baseline memory reading uses the dated model, with every session read during the run. LoCoMo uses the latest of three full runs, not their maximum. Earlier reference runs used different memory-reading settings. |

A dash (`-`) means that a reproduced result is not included in this snapshot.
For these missing cells, full runs were not completed under the same evaluation
setup because of account/API access, service availability, benchmark support,
or run-cost constraints. Partial or non-comparable runs are excluded rather than
mixed into the reproduced result tables.

## LoCoMo

<a id="reproduced-results"></a>
<a id="published-reference-results"></a>

[Scope, metrics and detailed results](./benchmark_results/locomo.md) ·
[Reproduced Results](./benchmark_results/locomo.md#reproduced-results) ·
[Published Reference Results](./benchmark_results/locomo.md#published-reference-results)

## LongMemEval

<a id="reproduced-results-1"></a>
<a id="published-reference-results-1"></a>

[Scope, metrics and detailed results](./benchmark_results/longmemeval.md) ·
[Reproduced Results](./benchmark_results/longmemeval.md#reproduced-results) ·
[Published Reference Results](./benchmark_results/longmemeval.md#published-reference-results)

## BEAM

<a id="reproduced-results-2"></a>
<a id="published-reference-results-2"></a>

[Scope, metrics and detailed results](./benchmark_results/beam.md) ·
[Reproduced Results](./benchmark_results/beam.md#reproduced-results) ·
[Published Reference Results](./benchmark_results/beam.md#published-reference-results)

## PersonaMem v2

<a id="reproduced-results-3"></a>
<a id="published-reference-results-3"></a>

[Scope, metrics and detailed results](./benchmark_results/personamem-v2.md) ·
[Reproduced Results](./benchmark_results/personamem-v2.md#reproduced-results) ·
[Published Reference Results](./benchmark_results/personamem-v2.md#published-reference-results)

## HaluMem

<a id="reproduced-results-4"></a>
<a id="published-reference-results-4"></a>

[Scope, metrics and detailed results](./benchmark_results/halumem.md) ·
[Reproduced Results](./benchmark_results/halumem.md#reproduced-results) ·
[Published Reference Results](./benchmark_results/halumem.md#published-reference-results)

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

## Contributing Results

See the [result submission requirements](../../CONTRIBUTING.md#submitting-benchmark-results)
for required evidence, aggregation details, and the distinction between
Published Reference Results and Reproduced Results.
