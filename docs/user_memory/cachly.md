# cachly

cachly is a persistent memory service for AI agents (the "Brain"). It stores
what an agent learned as text records, indexes them with embeddings in Valkey,
and returns the most relevant records for a query.

The `cachly` adapter talks to a self-hosted add/search service that follows the
AML contract `memory-api-v1.1` and sits in front of a cachly Brain (Valkey plus
an embedding service). The adapter is an HTTP client only; it does not start
or embed anything itself.

- Source: <https://github.com/cachly-dev/cachly-mcp> (Apache-2.0). Verified
  against `@cachly-dev/mcp-server` 0.10.218.
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
  service returns them.
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
adapter. Reported runs embed through the cachly API with `bge-m3` (query and
records) and `qwen3-embedding:0.6b` (second record vector), both served by
Ollama on CPU. When the service has a cachly API token, search also asks the
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

The LoCoMo row in [results.md](./results.md) was produced by Heinrich Neb
(cachly). Ingestion and search ran from 2026-10-29 22:51 to 2026-10-30
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
