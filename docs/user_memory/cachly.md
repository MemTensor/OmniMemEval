# cachly

cachly is a persistent memory service for AI agents (the "Brain"). It stores
what an agent learned as text records, indexes them with embeddings in Valkey,
and returns the most relevant records for a query.

The `cachly` adapter talks to a self-hosted add/search service that follows the
AML contract `memory-api-v1.1` and sits in front of a cachly Brain (Valkey plus
an embedding service). The adapter is an HTTP client only; it does not start
or embed anything itself.

- Source and release: TODO(maintainer): public repository link and the cachly
  release or commit this adapter was verified against.
- Service code for the add/search endpoints: TODO(maintainer): public link.

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

## Non-default settings

| Setting | Value | Why |
| --- | --- | --- |
| `CACHLY_ADD_TIMEOUT` | `3600` | An add waits until all embeddings are written; long conversations take minutes. |
| `CACHLY_SEARCH_TIMEOUT` | `120` | Seconds per search call. |
| `TOPK` | `20` | Records requested per query. |
| `LLM_WORKERS` | `10` | Runner default. |

Answer model `gpt-4.1-mini-2025-04-14` and judge model
`gpt-4o-mini-2024-07-18` are set in the template, as in the Evaluation Setup.
The embedding model and Valkey settings belong to the service, not to this
adapter: TODO(maintainer): state the embedding model and service version used
for reported runs.

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

No reported results are listed yet.
