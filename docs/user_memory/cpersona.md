# CPersona

The `cpersona` adapter reaches a self-hosted [CPersona](https://github.com/Cloto-dev/cpersona) server
over MCP streamable HTTP. CPersona stores conversation records and recalls them with hybrid vector and
full-text search, then answers a search with quotes from the stored records. It calls no generative
model to store or to search, so the baseline's memory-service model does not apply. Each harness user
is a CPersona agent, and every call names it.

## How it answers the harness

- `add(messages, user_id, session_key=None)`: one call is one session, stored as one record: a
  `Session date:` line, then one paragraph per turn prefixed by its role, stamped with the session's
  time. The harness's session key becomes the record id, so a session stored twice is skipped as a
  duplicate. A write the server rejects or truncates raises an error.
- `search(query, user_id, top_k)`: CPersona's `reconstruct` tool in the mode `CPERSONA_SEARCH_MODE`
  names. Each returned item becomes one block: the record's time, then the text the item quotes. The
  harness's `top_k` is not passed; the mode decides how many items come back (`pro`: a 15-item
  window plus up to 5 reserved places, the response capped at 5,000 tokens; `lite`: a 10-item
  window plus up to 3 reserved places, quotes limited to 2,800 characters).
- `delete(user_id)`: CPersona's `delete_agent_data` for that agent. Other agents are untouched.
- LongMemEval goes through `generic_text_search`. For LoCoMo, each session is stored once, as the
  harness's speaker-A view (which carries both speakers' turns), in one store per conversation, and
  searched with `shared_conv_search`, as for EverOS and Supermemory.

## Setup

The adapter needs the `mcp` package (2.x) in the harness's environment:

```bash
python -m pip install "mcp>=2,<3"
cp env_examples/.env.cpersona .env.cpersona
```

The server runs separately. The submitted rows used CPersona 2.6.11 on Python 3.11:

```bash
python3.11 -m venv cpersona-venv
cpersona-venv/bin/pip install "cpersona==2.6.11"
```

Start it with `cpersona-venv/bin/python -m cpersona.server` and this environment (every setting not
listed was at its default in the submitted runs):

```bash
TZ=UTC
CPERSONA_TRANSPORT=streamable-http
CPERSONA_HTTP_HOST=127.0.0.1
CPERSONA_HTTP_PORT=8501
CPERSONA_AUTH_TOKEN="<a bearer token, also set in .env.cpersona>"
CPERSONA_DB_PATH="<a new, empty database file>"
CPERSONA_EMBEDDING_MODE=http
CPERSONA_EMBEDDING_URL="http://127.0.0.1:<embedding port>/embed"
CPERSONA_EMBEDDING_MODEL=bge-m3
CPERSONA_MAX_CONTENT_LENGTH=80000
```

`CPERSONA_MAX_CONTENT_LENGTH=80000` (default 16,000) lets every LongMemEval and LoCoMo session be
stored whole. The embedding endpoint is any HTTP server that answers `POST /embed` with body
`{"texts": [...]}` by `{"embeddings": [[...], ...]}`, one vector per text. The submitted runs used
`BAAI/bge-m3` through sentence-transformers, normalized, with `max_seq_length` 512, served by a local
process; see [Result provenance](#result-provenance) for versions.

## Offline adapter tests

```bash
python -m unittest scripts/tests/unit/test_cpersona_client.py
```

## Adapter smoke

Against a running server, with `CPERSONA_MCP_URL` and `CPERSONA_AUTH_TOKEN` exported, this stores
one session for each of two users, checks that a search of one never returns the other's record, and
deletes both. It calls neither the answer nor the judge model.

```bash
python - <<'PY'
from scripts.client_factory.registry import create_client

client = create_client("cpersona")
client.add([{"role": "user", "content": "I adopted a grey cat named Pumpernickel.",
             "chat_time": "2023-05-08T13:56:00"}], "smoke-a")
client.add([{"role": "user", "content": "I run a marathon in Kyoto every October.",
             "chat_time": "2023-07-01T07:00:00"}], "smoke-b")
try:
    a = "\n".join(client.search("What is my cat's name?", "smoke-a", 20))
    assert "Pumpernickel" in a and "Kyoto" not in a
finally:
    client.delete("smoke-a")
b = "\n".join(client.search("Where do I run a marathon?", "smoke-b", 20))
assert "Kyoto" in b and "Pumpernickel" not in b
client.delete("smoke-b")
print("add=2 search=2 delete=2")
PY
```

Run on 2026-11-27 against CPersona 2.6.11 installed from PyPI, with the server settings of the runs
below and the harness on mcp 2.2.0: the snippet above passed, and a longer check of the same client
passed all 19 of its checks (registry creation; three sessions stored and a repeated session key
skipped; `pro` and `lite` searches each returning the user's own record, led by its time, and never
the other user's; cleanup through the harness's `delete_user_data` removing one user's records and
leaving the other's).

## Runs

CPersona builds a stored record's search structures on a background queue after the store returns,
so the submitted runs did not start searching until that queue was empty. Run the stages one at a
time, under `OMNIMEMEVAL_ENV_FILE=.env.cpersona`:

```bash
python scripts/longmemeval/lme_ingestion.py --lib cpersona --version my_run --workers 2
# wait until the server's queue is empty (get_queue_status reports no pending task)
python scripts/longmemeval/lme_search.py    --lib cpersona --version my_run --workers 2
python scripts/longmemeval/lme_responses.py --lib cpersona --version my_run --llm-workers 3
python scripts/longmemeval/lme_eval.py      --lib cpersona --version my_run --num_runs 1 --llm-workers 5
python scripts/longmemeval/lme_metric.py    --lib cpersona --version my_run

python scripts/locomo/locomo_ingestion.py --lib cpersona --version my_run --workers 2
# wait as above
python scripts/locomo/locomo_search.py    --lib cpersona --version my_run --workers 1
python scripts/locomo/locomo_responses.py --lib cpersona --version my_run
python scripts/locomo/locomo_eval.py      --lib cpersona --version my_run --num_runs 1
python scripts/locomo/locomo_metric.py    --lib cpersona --version my_run
```

The exact commands of the submitted runs, including the two helper scripts that waited for the queue
and calibrated the LoCoMo users, are in the
[run records](https://github.com/Cloto-dev/cpersona-omnimemeval-2.6.11/blob/cc249da5060769db79a5fbae6425467a5318c987/README.md#commands). The calibration
steps are described under [What differs from a clean run](#what-differs-from-a-clean-run).

## Result provenance

The CPersona rows under **Reproduced Results** linked from the [results overview](./results.md) were
produced by the CPersona project (Cloto-dev) on 2026-10-19. Each cell is one full run in `pro` mode.
The same runs are listed under **Published Reference Results** with their `lite` counterparts.

| | |
|---|---|
| Harness | OmniMemEval at commit [`0b1ea8d`](https://github.com/MemTensor/OmniMemEval/tree/0b1ea8d) on Python 3.12.13 (mcp 2.2.0, openai 3.23.0, tiktoken 0.14.0), with this adapter's harness entries applied as a patch and its client added. See [Harness commit and adapter](#harness-commit-and-adapter) |
| Backend | CPersona 2.6.11 (git tag [`v2.6.11`](https://github.com/Cloto-dev/cpersona/tree/v2.6.11), also on PyPI), self-hosted on one Apple silicon machine (macOS), Python 3.11.15, the settings under [Setup](#setup) |
| Search | `reconstruct` in `pro` mode with that mode's default count (`CPERSONA_SEARCH_MODE=pro`); `TOPK=20` in the env file, not passed to CPersona |
| Answer and judge | answers by `gpt-4.1-mini`, recorded as `gpt-4.1-mini-2025-04-14` in `token_usage_answer.json`; one judge run by `gpt-4o-mini`. The harness does not record the judge's resolved snapshot; one API call on the run day showed the alias resolving to `gpt-4o-mini-2024-07-18`. The alias is the `EVAL_MODEL` value of 15 of the 19 backend templates under `env_examples/` at the base commit. The harness's own prompts and metric |
| Embeddings | `BAAI/bge-m3` (revision `5617a9f61b028005a4858fdac845db406aefb181`) through sentence-transformers 5.6.0 (transformers 5.12.1, torch 2.12.1) on the Apple GPU (`mps`), batch 32, `max_seq_length` 512, normalized, served over HTTP by a local process that cached each text's vector under the text's SHA-256 |
| Data | LongMemEval-S cleaned (`longmemeval_s_cleaned.json`, sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`), all 500 questions; LoCoMo (`locomo10.json`, sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`), the 1,540 questions of categories 1–4. Answer keys unmodified |

| Benchmark | Sessions stored | Questions | Right | Score | Answer prompt tokens / calls | Context Tokens |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval | 23,867 | 500 | 448 | 89.60 | 2,131,237 / 500 | 4,262 |
| LoCoMo | 272 | 1,540 | 1,294 | 84.03 | 6,839,071 / 1,540 | 4,441 |

The [run records](https://github.com/Cloto-dev/cpersona-omnimemeval-2.6.11/tree/cc249da5060769db79a5fbae6425467a5318c987) (commit `cc249da`) hold, for each run, the
harness's own output files (per-question search results, responses, judged answers, grades, stage
status files and answer-token receipts), the sessions recorded as stored, the server's and the
harness's settings with credentials removed, and `verify/recompute.py`, which recomputes every number
above from those files and, given the datasets, checks their SHA-256 and every record's question,
reference answer and category against them.

- **Coverage.** In every run each item's recorded status is `success` at the search, answer and eval
  stages; no item failed, timed out or was skipped. The denominators are all 500 and all 1,540
  questions.
- **Context Tokens** is answer prompt tokens divided by answer calls, none estimated.
- **Repetitions.** One run per benchmark and mode. No run is combined with another.

### Harness commit and adapter

The runs used OmniMemEval `0b1ea8d` with the
[harness patch](https://github.com/Cloto-dev/cpersona/blob/de079043ba760be7080e18947fb5b45c8bcb5785/benchmarks/omnimemeval/omnimemeval-0b1ea8d-2_6_11.patch)
applied and the
[client](https://github.com/Cloto-dev/cpersona/blob/de079043ba760be7080e18947fb5b45c8bcb5785/benchmarks/omnimemeval/cpersona_client.py)
added under `scripts/client_factory/`.

- Between `0b1ea8d` and the commit this adapter is based on, nothing under `scripts/longmemeval/`
  changed. Under `scripts/locomo/` and `scripts/utils/`, the changes are other adapters' entries and
  the image-caption fix for Zep graph mode and Graphiti; the speaker-A messages that CPersona's
  LoCoMo branch stores are built by unchanged code.
- The harness entries here are those of the patch. The one addition, `cpersona` in the libraries that
  receive `chat_time`, gives the behavior the runs got through the harness's fallback for an unlisted
  library.
- The client here differs from the one the runs used in four ways that do not change what is stored
  or searched: the cleanup method is `delete(user_id)`, which the harness's cleanup calls (the runs
  started from empty databases and never called cleanup); the `mcp` imports moved into the call, so the
  offline tests need neither a server nor the package; the reading of a tool result moved into
  `parse_result`, unchanged; and `search` gives `top_k` a default and accepts extra keyword arguments,
  still not passing them.

### What differs from a clean run

1. **Calibration (LongMemEval).** CPersona calibrates a similarity threshold, per user or for the
   whole store. Calibrating one user costs about 200 recalls; a per-user calibration on this store had
   not finished its first user after 630 seconds and was stopped. Both modes therefore reused the
   calibration file of an earlier LongMemEval-S store built by the project from the same data and
   embedding model (published with the records, sha256
   `1c13934a7cf245367c7362083380c6853bf121b922cb984f8811aac6bdc90c01`): a store-wide threshold of
   0.3628 and no per-user entry for any of these users. The server restored it at startup in both
   modes.
2. **Calibration (LoCoMo).** Each of the 10 users was calibrated once before the store was copied
   for the two modes. On these small stores (19 to 32 records a user) the server cannot calibrate the
   score threshold that admits a search result, so it recalibrates the whole store at every start while it serves, and the searches
   ran beside that. A later rerun that waited for the startup calibration to finish returned the same
   context for every one of the 1,540 questions in both modes; it is with the records. Its answers and
   judgments were not run.
3. **Ingestion (LongMemEval).** A first ingestion with four workers lost 339 embedding calls to a
   refused connection on the local embedding server and was discarded. The second, with two workers,
   stored all 23,867 sessions without an embedding failure; the records are from the second.

### Development on these questions

CPersona's response modes were developed on LongMemEval-S itself. 100 of the 500 questions are a
development set (listed in the records, seed 20261003): the shape of `pro` was chosen there by judged
accuracy. The other 400 served the comparisons, fixed before each run, that decided earlier CPersona releases,
and some exploration; in all, the project judged 32 runs of all or part of the 500 questions between
2026-10-03 and 2026-10-13. On the 400 questions outside the development set, the `pro` run scores
88.75 (355 of 400), against 93.00 on the 100 development questions. LoCoMo's questions were searched
once before, on 2026-10-13, with no answers or judging.
