# ThreadAtlas

The in-process `threadatlas` adapter runs [ThreadAtlas](https://github.com/jhffmn82/threadatlas), a pipeline
that reads documents and chats into dated facts, each tied to the words it was read from, and retrieves
them with the source passages beside them. Each user has a store of their own.

## How it answers the harness

- `add(messages, user_id)`: one call is one session. ThreadAtlas renders the session as a document of
  dated turns (Step 0) and reads its facts with `gpt-5.6-luna`, or the model `THREADATLAS_READ_MODEL`
  names (Step 1), checking every fact against the passage it was read from. A session that ThreadAtlas
  has already read, word for word and at the same minute, is taken from its prebuilt store instead of
  being read again; every session of `longmemeval_s_cleaned.json` is such a session, since the store was
  built from `longmemeval_s.json`. Either way the session is written into the user's own store: its
  facts, the chat's own text in 256-token chunks, keyword rows and a vector per record.
- `search(query, user_id, top_k)`: opens the user's store, which holds that user's sessions alone.
  Retrieval fuses vector and keyword search by rank, walks two hops through the graph, and returns a
  header that says how to read the records, then up to `top_k - 1` records with their source passages.
- `delete(user_id)`: removes that user's folder; other users are untouched.

The answer and judge models are the harness's own. ThreadAtlas makes no model call at search time.

## Setup

```bash
git clone https://github.com/jhffmn82/threadatlas.git
python -m pip install -r requirements_user_memory.txt
python -m pip install numpy fastembed tiktoken
cp env_examples/.env.threadatlas .env.threadatlas
```

Edit `.env.threadatlas`: `THREADATLAS_REPO` is the checkout, `THREADATLAS_STATE_DIR` a writable folder for
the users' stores. The prebuilt store (`threadatlas.sqlite`, `threadatlas.npy`), Step 0's export
(`documents.jsonl`, `units.jsonl`) and `longmemeval_s.json` are the inputs that let LongMemEval sessions be
found instead of read again; their locations are listed with the submitted results. Without them every
session is read fresh, which needs `OPENAI_API_KEY`.

## Offline adapter tests

```bash
python -m unittest scripts/tests/unit/test_threadatlas_client.py
```

## Runs

```bash
./scripts/run_lme_eval.sh --lib threadatlas --env .env.threadatlas
./scripts/run_locomo_eval.sh --lib threadatlas --env .env.threadatlas
```

Reported runs use `TOPK=20`, `gpt-4.1-mini-2025-04-14` as the answer model and `gpt-4o-mini-2024-07-18` as
the judge, with every other setting at the harness default.

## Version, deployment and settings of the reported runs

| | |
|---|---|
| Backend | `threadatlas_memory`, published at [jhffmn82/threadatlas@a3337c5f](https://github.com/jhffmn82/threadatlas/tree/a3337c5f/threadatlas_memory). Each run used the copy in the Kaggle dataset [`jhffmn/threadatlas-omni-kit`](https://www.kaggle.com/datasets/jhffmn/threadatlas-omni-kit): version 2 for LoCoMo with `gpt-5.6-luna`, version 3 for LongMemEval, version 4 for LoCoMo with `gpt-4.1-mini-2025-04-14`. Version 3 added a database connection per thread and version 4 the `THREADATLAS_READ_MODEL` setting; the published package is version 4 with comments updated |
| Deployment | local, in-process, one store per user under `THREADATLAS_STATE_DIR` |
| Store | `threadatlas.sqlite` and `threadatlas.npy` from the public Kaggle dataset [`jhffmn/threadatlas-global-chunks`](https://www.kaggle.com/datasets/jhffmn/threadatlas-global-chunks) |
| Reading model | `gpt-5.6-luna`, the default; one LoCoMo run set `THREADATLAS_READ_MODEL="gpt-4.1-mini-2025-04-14"` |
| Retrieval | the defaults: two hops from 40 hits of each search (`THREADATLAS_HOPS`, `THREADATLAS_HITS`) |
| Workers | `--workers 10` for LoCoMo and `--workers 4` for LongMemEval; `LLM_WORKERS=10` |

The runs, their records and their exact commands are in
[jhffmn82/threadatlas/omnimemeval](https://github.com/jhffmn82/threadatlas/tree/dff7388606c3979c410195e28ce0b59cf4faa97a/omnimemeval).

## Smoke run

A bounded real-backend check before a full run: the first 20 LongMemEval users and the first LoCoMo
conversation, through every step.

```bash
python - <<'PY'
import json
for path, keep in (("data/longmemeval/longmemeval_s_cleaned.json", 20), ("data/locomo/locomo10.json", 1)):
    rows = json.load(open(path, encoding="utf-8"))
    json.dump(rows[:keep], open(path, "w", encoding="utf-8"))
PY
./scripts/run_lme_eval.sh --lib threadatlas --env .env.threadatlas --version smoke --workers 4
./scripts/run_locomo_eval.sh --lib threadatlas --env .env.threadatlas --version smoke --workers 10
```

This shortens the prepared data files in place; run the prepare scripts again before a full run. The
Kaggle notebook in the results repository does the same with `RUN = "smoke"`.

These commands ran as written on 2026-11-07 on a Kaggle CPU notebook (Python 3.13.18, kit version 5,
`gpt-5.6-luna`): every record finished, LongMemEval 20 of 20 and LoCoMo 141 of 152 judged right. The
[records and installed package versions](https://github.com/jhffmn82/threadatlas/tree/dff7388606c3979c410195e28ce0b59cf4faa97a/omnimemeval/smoke)
are with the results.
