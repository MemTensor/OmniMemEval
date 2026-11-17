# BEAM Results

[Results overview](../results.md) · [Evaluation setup](../results.md#evaluation-setup) ·
[Reproduction notes](../results.md#reproduction-notes) · [中文说明](../results_zh.md)

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

## Reproduced Results

[LCM-X](../lcmx.md#result-provenance), [Maasv](../maasv.md#result-provenance),
[cachly](../cachly.md#result-provenance) and [ThreadAtlas](../threadatlas.md#result-provenance) rows are contributor-run.
LCM-X used OpenRouter; its exact dated answer-model revision is unconfirmed.
See [run notes and limitations](../results.md#contributor-run-notes) before
comparing these rows.

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
| LCM-X | local/self-hosted | 71.84 +/- 34.86 | 11,926 | 63.32 +/- 40.02 | 9,625 |
| Maasv | local/self-hosted | 76.38 +/- 32.50 | 8,526 | 65.09 +/- 38.85 | 8,656 |
| cachly | local/self-hosted | 73.04 +/- 34.94 | 5,189 | - | - |
| ThreadAtlas | local/self-hosted | 73.12 +/- 34.41 | 5,664 | - | - |

The LCM-X 10M cell is a single contributor run (200 questions, 2026-10-22); the
+/- figure is the per-question standard deviation printed by the harness. See
[LCM-X BEAM 10M run](../lcmx.md#beam-10m-run-2026-10-22).

The cachly 100K cell is a single contributor run (400 questions, 2026-11-13).
cachly's four earlier BEAM 100K runs (70.28 to 72.52) are listed in
[cachly BEAM 100K provenance](../cachly.md#beam-100k).

The ThreadAtlas 100K cell is a single contributor run (400 questions,
2026-11-16); the +/- figure is the per-question standard deviation printed by
the harness. See [ThreadAtlas result provenance](../threadatlas.md#result-provenance).

## Published Reference Results

These externally reported scores may use different data, models, prompts or
budgets. They are not directly comparable to the reproduced results above.

| Backend | 100K Nugget Score | 100K Context Tokens | 10M Nugget Score | 10M Context Tokens | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Mem0 | 64.1 | 6,719 | 48.6 | 6,914 | [mem0.ai research](https://mem0.ai/research) |
| Hindsight | 75.0 | - | 64.1 | - | [Hindsight Benchmarks](https://benchmarks.hindsight.vectorize.io/) |
| past.dev | 92.08 | - | 85.03 | - | [past.dev BEAM leaderboard](https://past.dev/benchmarks/beam) |
| OpenWay.ai | 75.07 | 5,366 | - | - | [OpenWay.ai BEAM-100K runs](https://github.com/andrei2084k-prog/openway.ai-beam-100k) (OmniMemEval harness at `0b1ea8d`; latest run, on a newer version of OpenWay.ai) |
| Hojo MemoryEngine | 69.58 | - | - | - | [100K evaluation details](../hojo_reference_results.md#beam-100k); [10M status](../hojo_reference_results.md#beam-10m) |
