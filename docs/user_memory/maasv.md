# Maasv

The `maasv` adapter evaluates Maasv through a self-hosted Maasv service over
HTTP. Maasv is not open source.

## Deployment and configuration

| Setting | Value |
| --- | --- |
| Backend | Maasv v0.4.48, local/self-hosted, one store per evaluation user |
| Ingest | Messages sent in ordered batches (`MAASV_BATCH_SIZE`, default 50) |
| Search | `top_k` from the runner (default 20); the service returns the results oldest to newest, each rendered as `[YYYY-MM-DD HH:MM · #n] <memory text>`, after the two-sentence header the Maasv engine adds ("Memories run oldest to newest. A later statement may revise an earlier one; if two statements conflict, say so and say which is later."), visible in every run's `search_context` |
| LoCoMo store layout | One store per conversation: all turns once, under the speaker A user ID, searched once per question (`shared_conv_search`, as for EverOS, Cognee, Letta, Hindsight and Backboard.io) |
| Memory service model | none (the Maasv service made no LLM calls in these runs) |

## Running

Start the Maasv service, then run the benchmark with one worker so requests
arrive grouped by user:

```bash
cp env_examples/.env.maasv .env.maasv   # fill ANSWER_* and EVAL_* credentials
./scripts/run_locomo_eval.sh --lib maasv --env .env.maasv --workers 1 --version <name>
./scripts/run_beam_eval.sh --lib maasv --env .env.maasv --workers 1 --scale 100k --version <name>
```

`MAASV_BASE_URL` (default `http://127.0.0.1:8765`) points the adapter at the
service. The offline adapter tests are `scripts/tests/unit/test_maasv_client.py`.

## Result provenance

All runs on Maasv v0.4.48, 2026-10-31 to 2026-11-01, by the Maasv team,
harness `7c79d46` plus the `maasv` adapter (no pipeline, prompt or metric
changes), `gpt-4.1-mini-2025-04-14` answering and `gpt-4o-mini-2024-07-18`
judging through the direct OpenAI API, one judge run each. They replace the
Maasv v0.4.46 runs of 2026-10-29 to 2026-10-31, whose records stay published
([omnimemeval-results](https://github.com/maasv-ai/omnimemeval-results)), as do
the earlier LoCoMo and BEAM 100K runs that kept every evaluation user in one
shared Maasv store, marked superseded.

### LoCoMo

| Item | Value |
| --- | --- |
| Dataset | `locomo10.json`, SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; original answer key; categories 1 to 4 |
| Coverage | 1,540 of 1,540 questions searched, answered and judged; 0 failed, 0 skipped; denominator 1,540 |
| Score | 87.73 (1,351 / 1,540) |
| Context tokens | 2,388,178 answer prompt tokens over 1,540 calls (1,550.8 per question); 47,137 completion tokens |
| Command | `run_locomo_eval.sh --lib maasv --env .env.maasv --workers 1 --version maasv_v0448_locomo_r2` (one invocation, non-streaming, 2026-10-31 22:21 to 22:36 ET) |
| Earlier run | Run 1 on v0.4.48 (86.30) is not published: the local Maasv service used for it did not return results the way the shipped service does |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/locomo-20261031-r2

### BEAM 100K

| Item | Value |
| --- | --- |
| Dataset | BEAM 100K from `prepare_beam.py --scale 100k`: 20 conversations, 5,732 messages, 400 questions |
| Coverage | 400 of 400 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed conversations at ingestion |
| Score | Nugget score 76.38 ± 32.50 (per-question SD) |
| Context tokens | 3,410,257 answer prompt tokens over 400 calls (8,525.6 per question); 219,315 completion tokens |
| Command | `run_beam_eval.sh --lib maasv --env .env.maasv --workers 1 --scale 100k --version maasv_v0448_beam100k_r2` (one invocation, non-streaming, 2026-10-31 22:36 to 22:48 ET) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/beam100k-20261031-r2

### BEAM 10M

| Item | Value |
| --- | --- |
| Dataset | BEAM 10M from `prepare_beam.py --scale 10m`: 10 conversations, 208,696 messages, 200 questions |
| Coverage | 200 of 200 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed conversations at ingestion |
| Score | Nugget score 65.09 ± 38.85 (per-question SD) |
| Context tokens | 1,731,213 answer prompt tokens over 200 calls (8,656.1 per question); 119,439 completion tokens |
| Command | `run_beam_eval.sh --lib maasv --env .env.maasv --workers 1 --scale 10m --version maasv_v0448_beam10m_r2` (one invocation, non-streaming, 2026-11-01 17:07 to 20:38 ET) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/beam10m-20261101-r2

### HaluMem-Medium

| Item | Value |
| --- | --- |
| Dataset | `HaluMem-Medium.jsonl` from `prepare_halumem.py`, SHA-256 `486fbc130a5c8781a2af27ffa508a1d7855245137aa449c193ac4d29c45634e7`: 20 users, 1,387 sessions, 60,146 messages, 3,467 questions |
| Coverage | 3,467 of 3,467 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed users at ingestion |
| Score | 82.29 (2,853 / 3,467) |
| Context tokens | 7,767,984 answer prompt tokens over 3,467 calls (2,240.5 per question); 280,960 completion tokens |
| Commands | `run_halumem_eval.sh --lib maasv --env .env.maasv --workers 1 --variant medium --version maasv_v0448_halumem_r3` (ingestion and search, 2026-11-01 09:08 to 10:59 ET, stopped at the start of the answer stage because the answer API key had stopped working; no answer was kept), then the same with `--from-step 3` (answers, judge and report, 11:34 to 11:43 ET) |
| Earlier run | Run 2 on v0.4.48 (2026-10-31) is not published: the answer API account stopped accepting calls during its answer stage |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/halumem-20261101-r3

### PersonaMem v2

| Item | Value |
| --- | --- |
| Dataset | `benchmark/text/benchmark.csv` from `prepare_personamem.py`, SHA-256 `95f2a8a324aab7baf2af937feae12731369e2abf7cad5ab3e170594cb25a3e52`, plus 200 `chat_history_32k` files: 200 personas, 5,000 questions |
| Coverage | 5,000 of 5,000 questions searched and answered; 0 failed, 0 skipped |
| Score | 39.86 (1,993 / 5,000), option matching |
| Context tokens | 14,648,094 answer prompt tokens over 5,000 calls (2,929.6 per question); 23,967 completion tokens |
| Command | `run_pmv2_eval.sh --lib maasv --env .env.maasv --workers 1 --version maasv_v0448_pmv2_r2` (one invocation, non-streaming, 2026-11-01 11:43 to 13:01 ET) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/pmv2-20261101-r2

### LongMemEval-S

| Item | Value |
| --- | --- |
| Dataset | `longmemeval_s_cleaned.json` from `prepare_longmemeval.py --variant s`, SHA-256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`: 500 questions, 23,867 sessions, 246,750 messages |
| Coverage | 500 of 500 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed users at ingestion |
| Score | 91.20 (456 / 500) |
| Context tokens | 1,999,686 answer prompt tokens over 500 calls (3,999.4 per question); 82,519 completion tokens |
| Command | `run_lme_eval.sh --lib maasv --env .env.maasv --workers 1 --version maasv_v0448_lme_r2` (one invocation, non-streaming, 2026-11-01 13:01 to 17:07 ET) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/lme-20261101-r2

The records support recomputing the aggregates and replaying the answer and
judge stages.
