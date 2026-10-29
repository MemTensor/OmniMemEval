# Maasv

The `maasv` adapter evaluates Maasv through a self-hosted Maasv service over
HTTP. Maasv is not open source.

## Deployment and configuration

| Setting | Value |
| --- | --- |
| Backend | Maasv v0.4.46, local/self-hosted |
| Search | `top_k` from the runner (default 20); each result rendered as `[YYYY-MM-DD HH:MM] <memory text>` |
| LoCoMo store layout | One store per conversation: all turns once, under the speaker A user ID, searched once per question (`shared_conv_search`, as for EverOS, Cognee, Letta, Hindsight and Backboard.io) |
| Memory service model | none (the Maasv service made no LLM calls in this run) |

## Running

Start the Maasv service, then run the benchmark:

```bash
cp env_examples/.env.maasv .env.maasv   # fill ANSWER_* and EVAL_* credentials
./scripts/run_locomo_eval.sh --lib maasv --env .env.maasv --version <name>
```

`MAASV_BASE_URL` (default `http://127.0.0.1:8765`) points the adapter at the
service. The offline adapter tests are `scripts/tests/unit/test_maasv_client.py`.

## Result provenance

LoCoMo, 2026-10-29, run by the Maasv team:

| Item | Value |
| --- | --- |
| Harness | OmniMemEval `7c79d46` plus the `maasv` adapter commit (no pipeline, prompt or metric changes) |
| Dataset | `locomo10.json`, SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`; original answer key; categories 1–4 |
| Coverage | 1,540 of 1,540 questions searched, answered and judged; 0 failed, 0 skipped, 0 timed out; denominator 1,540 |
| Answer model | `gpt-4.1-mini-2025-04-14`, direct OpenAI API |
| Judge model | `gpt-4o-mini-2024-07-18`, direct OpenAI API; one judge run (`--num-runs 1`) |
| Prompts | Harness defaults at `7c79d46` |
| Context tokens | 2,251,148 answer prompt tokens over 1,540 calls (1,461.8 per question); 45,718 completion tokens |
| Checkpoints | Steps 1–2 and steps 3–6 ran as two invocations on the same run version; no stage was re-run and no records were re-evaluated |
| Commands | `run_locomo_eval.sh --lib maasv --env .env.maasv --version maasv_v0446_r1 --to-step 2`, then the same with `--from-step 3` |

Per-question search, answer and judge records, the run configuration and the
generated report are published at: https://github.com/maasv-ai/omnimemeval-results/releases/tag/locomo-20261029.

The records support recomputing the aggregate and replaying the answer and
judge stages.

BEAM 100K, 2026-10-29, run by the Maasv team:

| Item | Value |
| --- | --- |
| Harness | OmniMemEval `7c79d46` plus the `maasv` adapter (no pipeline, prompt or metric changes) |
| Dataset | BEAM 100K from `prepare_beam.py --scale 100k`: 20 conversations, 5,732 messages, 400 questions |
| Coverage | 400 of 400 questions searched, answered and judged; 0 failed, 0 skipped; 0 failed conversations at ingestion |
| Answer / judge | `gpt-4.1-mini-2025-04-14` / `gpt-4o-mini-2024-07-18`, direct OpenAI API; one judge run |
| Score | Nugget score 74.83 ± 32.87 (per-question SD) |
| Context tokens | 3,470,215 answer prompt tokens over 400 calls (8,675.5 per question) |
| Command | `run_beam_eval.sh --lib maasv --env .env.maasv --version maasv_v0446_beam100k_r1 --scale 100k` (one invocation) |

Records: https://github.com/maasv-ai/omnimemeval-results/releases/tag/beam100k-20261029.
