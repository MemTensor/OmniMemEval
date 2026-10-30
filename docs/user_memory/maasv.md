# Maasv

The `maasv` adapter evaluates Maasv through a self-hosted Maasv service over
HTTP. Maasv is not open source.

## Deployment and configuration

| Setting | Value |
| --- | --- |
| Backend | Maasv v0.4.46, local/self-hosted, one store per evaluation user |
| Ingest | Messages sent in ordered batches (`MAASV_BATCH_SIZE`, default 50) |
| Search | `top_k` from the runner (default 20); each result rendered as `[YYYY-MM-DD HH:MM] <memory text>` |
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

All three runs on 2026-10-29/30 by the Maasv team, harness `7c79d46` plus the `maasv`
adapter (no pipeline, prompt or metric changes), `gpt-4.1-mini-2025-04-14`
answering and `gpt-4o-mini-2024-07-18` judging through the direct OpenAI API,
one judge run each. An earlier run of LoCoMo and BEAM 100K kept every evaluation user in one
shared Maasv store, which reduced retrieval quality; those records stay
published, marked superseded.

### LoCoMo

| Item | Value |
| --- | --- |
| Dataset | `locomo10.json`, SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; original answer key; categories 1–4 |
| Coverage | 1,540 of 1,540 questions searched, answered and judged; 0 failed, 0 skipped; denominator 1,540 |
| Score | 86.10 (1,326 / 1,540) |
| Context tokens | 2,267,884 answer prompt tokens over 1,540 calls (1,472.7 per question); 46,454 completion tokens |
| Commands | `run_locomo_eval.sh --lib maasv --env .env.maasv --workers 1 --version maasv_v0446_r2`, then the same with `--from-step 4` (see below) |
| Judge re-run | In the first invocation one judge reply was not valid JSON, so the judge stage failed one conversation (81 questions). The second invocation re-ran the judge stage for all 1,540 answers; the reported score is from that pass. |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/locomo-20261029-r2

### BEAM 100K

| Item | Value |
| --- | --- |
| Dataset | BEAM 100K from `prepare_beam.py --scale 100k`: 20 conversations, 5,732 messages, 400 questions |
| Coverage | 400 of 400 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed conversations at ingestion |
| Score | Nugget score 75.43 ± 32.08 (per-question SD) |
| Context tokens | 3,367,740 answer prompt tokens over 400 calls (8,419.4 per question) |
| Command | `run_beam_eval.sh --lib maasv --env .env.maasv --workers 1 --scale 100k --version maasv_v0446_beam100k_r2` (one invocation) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/beam100k-20261029-r2

### BEAM 10M

| Item | Value |
| --- | --- |
| Dataset | BEAM 10M from `prepare_beam.py --scale 10m`: 10 conversations, 208,696 messages, 200 questions |
| Coverage | 200 of 200 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed conversations at ingestion |
| Score | Nugget score 61.69 ± 38.10 (per-question SD) |
| Context tokens | 1,707,130 answer prompt tokens over 200 calls (8,535.6 per question) |
| Command | `run_beam_eval.sh --lib maasv --env .env.maasv --workers 1 --scale 10m --version maasv_v0446_beam10m_r2` (one invocation, non-streaming, 2026-10-29 20:41 to 2026-10-30 08:16 ET) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/beam10m-20261030-r2

The records support recomputing the aggregates and replaying the answer and
judge stages.
