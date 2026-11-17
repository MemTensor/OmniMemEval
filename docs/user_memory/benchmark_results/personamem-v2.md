# PersonaMem v2 Results

[Results overview](../results.md) · [Evaluation setup](../results.md#evaluation-setup) ·
[Reproduction notes](../results.md#reproduction-notes) · [中文说明](../results_zh.md)

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

## Reproduced Results

[LCM-X](../lcmx.md#result-provenance), [Maasv](../maasv.md#result-provenance),
and [ThreadAtlas](../threadatlas.md#result-provenance) rows are contributor-run.
LCM-X used OpenRouter; its exact dated answer-model revision is unconfirmed.
See [run notes and limitations](../results.md#contributor-run-notes) before
comparing these rows.

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
| LCM-X | local/self-hosted | 32.98 | 32.82 | 34.15 | 31.59 | 32.68 | 39.21 | 36.84 | 33.96 | 3,534 |
| Maasv | local/self-hosted | 36.14 | 53.24 | 34.33 | 35.66 | 33.46 | 41.28 | 37.32 | 39.86 | 2,930 |
| ThreadAtlas | local/self-hosted | 30.29 | 47.42 | 36.09 | 34.15 | 31.51 | 36.96 | 37.32 | 36.92 | 4,268 |

## Published Reference Results

These externally reported scores may use different data, models, prompts or
budgets. They are not directly comparable to the reproduced results above.

| Backend | Overall | Context Tokens | Source |
| --- | ---: | ---: | --- |
| EverOS | 53.25 | - | [EverMemOS paper](https://arxiv.org/abs/2601.02163) |
| Hojo MemoryEngine | 35.14 | - | [Hojo evaluation details](../hojo_reference_results.md#personamem-v2) |
