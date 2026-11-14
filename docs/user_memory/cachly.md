# cachly

cachly is a persistent memory service for AI agents (the "Brain"). It stores
what an agent learned as text records, indexes them with embeddings in Valkey,
and returns the most relevant records for a query.

The `cachly` adapter talks to a self-hosted add/search service that follows the
AML contract `memory-api-v1.1` and sits in front of a cachly Brain (Valkey plus
an embedding service). The adapter is an HTTP client only; it does not start
or embed anything itself.

- Source: <https://github.com/cachly-dev/cachly-mcp> (Apache-2.0). Verified
  against `@cachly-dev/mcp-server` 0.10.218 (LoCoMo) and the service of
  0.10.253 (LongMemEval, BEAM 100K).
- Service code for the add/search endpoints:
  <https://github.com/cachly-dev/cachly-mcp/tree/main/aml-service> (contract,
  start and settings in its README).

## Setup

From the OmniMemEval repository root, use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_user_memory.txt
cp env_examples/.env.cachly .env.cachly
```

Start the service in front of a cachly Brain (Valkey and the embedding service
must be reachable by the service). Then edit `.env.cachly`:

| Variable | Meaning |
| --- | --- |
| `CACHLY_AML_URL` | Base URL of the service, for example `http://127.0.0.1:8780`. Required. |
| `CACHLY_AML_KEY` | Bearer key of the service. Required. Keep it out of version control. |

Configure the answer and judge endpoints and keys locally for benchmark runs.
The adapter smoke below calls neither model. Never commit the populated file.

## Service contract

All calls send `Authorization: Bearer <CACHLY_AML_KEY>` and JSON bodies.

| Route | Request | Response |
| --- | --- | --- |
| `POST /v1/memory/add` | `request_id`, `user_id`, `session_id`, `messages[]` (`role`, `content`, optional `timestamp` in Unix ms) | `success`, `request_id`; returns once the records are searchable |
| `POST /v1/memory/search` | `query`, `user_id`, `top_k` | `data[]`, each with `content` |
| `POST /v1/memory/delete` | `user_id` | `deleted` (`true` or `false`); an unknown user is not an error |

Behaviour of the adapter:

- Each `user_id` has its own isolated store; a search only sends the `user_id`
  it was asked for.
- A message with a speaker name is sent as `Name: text`, unless the text
  already starts with `Name:`. The role is `user` or `assistant`.
- `chat_time` (LoCoMo format `1:56 pm on 8 May, 2023`, ISO strings, or numbers)
  becomes the Unix-millisecond `timestamp`. Unparseable times are omitted.
- `request_id` is `omni:` plus the first 32 hex characters of the SHA-256 of
  `[user_id, messages]`, so a repeated add of the same data carries the same
  id. `session_id` equals `user_id`.
- Search returns the `content` of each hit, one per line, in the order the
  service returns them. When the service marks its response
  `"ordnung": "zeit"` (records oldest first), the first line is
  `(Memories in chronological order, oldest first; when two memories
  disagree, the later one is the current state.)`. The service of cachly
  0.10.253 does this by default; `AML_ZEITORDNUNG=aus` in the service
  environment returns the records in rank order without the mark.
- HTTP 4xx (except 429) raises at once. 429, 5xx and connection errors are
  retried by the shared `_retry` helper.
- The service returns up to `top_k` records; cachly caps a single search at 20.

LoCoMo: cachly keeps one shared store per conversation, like Supermemory,
Backboard and Hindsight. Each session is added once, under the speaker A
`user_id`, and both speakers are answered from that store (`shared_conv_search`).
The answer step then merges the two identical speaker sections into one block.
A message that shares a photo keeps the photo description LoCoMo ships with it
(`blip_caption`), appended as ` [image: ...]`, as the Hindsight client does.

## Non-default settings

| Setting | Value | Why |
| --- | --- | --- |
| `CACHLY_ADD_TIMEOUT` | `3600` | An add waits until all embeddings are written; long conversations take minutes. |
| `CACHLY_SEARCH_TIMEOUT` | `120` | Seconds per search call. |
| `TOPK` | `20` | Records requested per query. |
| `LLM_WORKERS` | `10` | Runner default. |

Answer model `gpt-4.1-mini-2025-04-14` and judge model
`gpt-4o-mini-2024-07-18` are set in the template, as in the Evaluation Setup.
The embedding models and Valkey settings belong to the service, not to this
adapter. All reported runs embed with `bge-m3` (query and records) and
`qwen3-embedding:0.6b` (second record vector), both served by Ollama: on CPU
behind the cachly API for LoCoMo, on a rented GPU behind a local stand-in for
the same API for LongMemEval and BEAM 100K. When the service has a cachly API token, search also asks the
API's cross-encoder reranker (`POST /api/v1/rerank`) to reorder the top 25
candidates, with a 2,500 ms limit; `CACHLY_LESER=off` in the service
environment switches it off.

## Adapter smoke

Against a running service, this ingests one user, searches, and deletes it:

```bash
python - <<'PY'
from scripts.client_factory.registry import create_client

client = create_client("cachly")
try:
    client.add([{"role": "user", "content": "My favorite drink is green tea."}], "smoke-1")
    assert "green tea" in client.search("favorite drink", "smoke-1", 20)
finally:
    client.delete("smoke-1")
print("add=1 search=1 delete=1")
PY
python -m pytest scripts/tests/unit/test_cachly_client.py -q
```

The unit tests need no service: they patch `requests`.

Run a suite from the repository root after configuring `.env.cachly`:

```bash
./scripts/run_locomo_eval.sh --lib cachly --env .env.cachly
```

## Result provenance

The cachly rows in [results.md](./results.md) were produced by Heinrich Neb
(cachly), one run per suite, with the answer and judge models of the template
through the OpenAI API.

### LoCoMo

Ingestion and search ran from 2026-10-29 22:51 to 2026-10-30
04:48 UTC, answer and judge stages from 04:48 to 05:04 UTC, with OmniMemEval
harness commit `f7476c5502115542eca9971195bf8c5ca97b6256` and
`@cachly-dev/mcp-server` 0.10.218 (cachly-mcp commit
`5ecd1a67ca54391f552841a7c4b0821d788917fe`).

The [public evidence release](https://github.com/cachly-dev/OmniMemEval/releases/tag/cachly-locomo-results-20261030)
provides the [results archive](https://github.com/cachly-dev/OmniMemEval/releases/download/cachly-locomo-results-20261030/cachly-omnimemeval-locomo-20261030.tar.gz),
whose SHA-256 is `fae88340204ed2a2986252d27bc2f4875b310b4bd05248cce44b695f74cf5045`. It holds the per-question records, stage
status files, answer-token receipts, the run settings, a recompute script and
a `MANIFEST.json` with file hashes. Search, answer and judge each report 1,540
successes and no skipped records; the run was done once.

- **Reranker.** Search asked the cachly API's cross-encoder reranker (model
  `leser-klein-voll`, trained by cachly, not public) to reorder the top 25
  candidates, with a 2,500 ms limit; how many searches it reordered was not
  recorded. A later search of the same stores with it off shared 17.97 of 20
  chunks per question on average (115 contexts identical); the archive lists
  each question. `CACHLY_LESER=off` switches it off.
- **Adapter as run.** The run used a working copy of the harness with this
  adapter. Its client file also carried a local development transport, which
  is off when `CACHLY_AML_URL` is set. The archive's `check_requests.py` shows
  that the 1,812 add and search requests of the run's client and of this
  adapter are identical.
- **Context Tokens** is answer prompt tokens divided by answer calls
  (10,275,728 / 1,540). The pipeline's `grades.json` value (12,760) counts the
  search context before the answer step merges the two identical speaker
  sections.
- **Earlier runs.** Complete runs with earlier configurations scored 85.58
  (one store per speaker), 88.57 (shared store, top-k 14) and 88.70 (top-k
  20). The last changes, photo descriptions and date spans for phrases such as
  "last week", came from reading lost LoCoMo questions; LoCoMo has no separate
  development split. The top-k 14 and top-k 20 runs disagreed on 71
  questions, yet their overall scores differ by 0.13 points.
- **Time order.** This run predates the time order: its service returned
  records in rank order without the `ordnung` field, so its contexts carry no
  time-order line. LoCoMo was not run again with the time order.

### LongMemEval

Ingestion and search ran from 2026-11-13 21:51 to 2026-11-14 04:35 UTC in
streaming mode (per question: delete, add, search, delete; eight strands side
by side), answer and judge stages from 04:36 to 04:43 UTC. The service was
`aml-service/dienst.mjs` as published in cachly-mcp commit `2e1b0e6501c3`
(`@cachly-dev/mcp-server` 0.10.253). The run used a copy taken from
cachly-dev/cachly `fbfead1e` before the time order was merged; it differs
only in the default of `AML_ZEITORDNUNG`, which the run set to `1`.

The [public evidence release](https://github.com/cachly-dev/OmniMemEval/releases/tag/cachly-lme-beam-results-20261114)
provides the [LongMemEval archive](https://github.com/cachly-dev/OmniMemEval/releases/download/cachly-lme-beam-results-20261114/cachly-omnimemeval-lme-20261114.tar.gz),
whose SHA-256 is `3b429da8687d6aea0cf6aee6378b3873661aeb9ad148762f0d4d7f77839fb23d`.
It holds the per-question records, stage status files, the streaming log of
each strand, answer-token receipts, the run settings, the service file as run,
a recompute script, a request check and a `MANIFEST.json` with file hashes.
Search, answer and judge each report 500 successes and no skipped records.

- **Time order.** The service returns the top 20 records oldest first and
  marks the response, so all 500 contexts start with the time-order line.
- **Embeddings and reranker.** `bge-m3` and `qwen3-embedding:0.6b` on Ollama
  on a rented GPU (vast.ai, RTX 4080 SUPER, Czechia), reached through a local
  stand-in for the cachly embedding API. The reranker (an int8 ONNX build of
  cachly's model, not public) ran on a private host; how many searches it
  reordered was not recorded, and no search with it off was run.
- **Adapter as run.** `check_requests.py lme` drives the client as run and
  this adapter through the harness's own code for all 500 questions: the
  25,367 requests (23,867 add, 500 search, 1,000 delete) are identical. The
  add count equals the count in the service log.
- **Context Tokens** is answer prompt tokens divided by answer calls
  (2,270,681 / 500). The pipeline's `grades.json` value (4,330) counts the
  search context alone.
- **Earlier attempts.** LongMemEval was answered and judged once. A
  10-question probe and a first full attempt without time order (stopped
  after 101 questions) were never answered or judged. The time order came
  from BEAM 100K; no LongMemEval answers were read to make changes.
- **Uncertainty.** One question is 0.2 points; the binomial standard error of
  a score near 90 % over 500 questions is about 1.35 points.

### BEAM 100K

Ingestion and search ran on 2026-11-13 from 19:52 to 20:12 UTC, answer and
judge stages from 20:12 to 20:32 UTC, with the same service, embeddings and
reranker as the LongMemEval run (one store per conversation, not streaming).

The same release provides the [BEAM 100K archive](https://github.com/cachly-dev/OmniMemEval/releases/download/cachly-lme-beam-results-20261114/cachly-omnimemeval-beam100k-20261113.tar.gz),
whose SHA-256 is `f32a8f51ac21db15925957b44abc8951625ab87df1864ea24b5ba7c1edbb427b`,
with the same kinds of files and the aggregates of the earlier runs below.
Search, answer and judge each report 400 successes and no skipped records;
`check_requests.py beam` shows that the 490 requests (90 add, 400 search) of
the client as run and of this adapter are identical.

All five complete BEAM 100K runs, on 2026-11-13, with the same models and
top-k 20:

| Run | Configuration | Nugget Score |
| --- | --- | ---: |
| beam-g1 | records in rank order, no time-order line | 70.28 |
| beam-k0 | beam-g1's contexts unchanged; answer and judge again | 70.68 |
| beam-z1 | beam-g1's contexts re-sorted oldest first by a script | 72.52 |
| beam-z2 | as beam-z1, plus the time-order line | 71.94 |
| beam-g2 | time order in the service, new stores and search (reported) | 73.04 |

beam-k0 shows the noise of the answer and judge stages alone: with the same
contexts as beam-g1, 90 of 400 questions got a different score. The time
order was designed after reading the beam-g1 results, whose weakest
dimensions were event ordering (31.07) and knowledge update (52.50). BEAM has
no separate development split, so the gain on BEAM is not an independent
test.
