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

ThreadAtlas has two sets of scores in [results.md](./results.md).

- **Reproduced Results** hold the baseline runs of 2026-11-16. Every session is read during the run by
  `add()` with `gpt-4.1-mini-2025-04-14`, nothing is prepared beforehand, and a user's store holds that
  user's sessions alone. See [Result provenance](#result-provenance).
- **Published Reference Results** hold the runs of 2026-11-07. Memory reading used `gpt-5.6-luna`, and the
  LongMemEval sessions had been read before the run by ThreadAtlas's full pipeline (extraction, ingestion
  and its global merge) from the original dataset. See
  [Version, deployment and settings of the reported runs](#version-deployment-and-settings-of-the-reported-runs).

## Setup

```bash
git clone https://github.com/jhffmn82/threadatlas.git
git -C threadatlas checkout 015143cdb995497022b88878c8ebc8c5223d1eee
python -m pip install -r requirements_user_memory.txt
python -m pip install numpy fastembed tiktoken
cp env_examples/.env.threadatlas .env.threadatlas
```

Edit `.env.threadatlas`: `THREADATLAS_REPO` is the checkout, `THREADATLAS_STATE_DIR` a writable folder for
the users' stores.

For the baseline setting, set `THREADATLAS_READ_MODEL="gpt-4.1-mini-2025-04-14"` and remove the lines for
`THREADATLAS_STORE_DIR`, `THREADATLAS_STEP0_DIR` and `THREADATLAS_SOURCE`. Every session is then read during
the run, which needs `OPENAI_API_KEY`. `THREADATLAS_EMBED_DEVICE="cuda"` runs the embedding model on a GPU
and needs `fastembed-gpu` in place of `fastembed`.

The prebuilt store (`threadatlas.sqlite`, `threadatlas.npy`), Step 0's export
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
./scripts/run_beam_eval.sh --lib threadatlas --env .env.threadatlas --scale 100k
./scripts/run_pmv2_eval.sh --lib threadatlas --env .env.threadatlas
./scripts/run_halumem_eval.sh --lib threadatlas --env .env.threadatlas --variant medium
```

Reported runs use `TOPK=20`, `gpt-4.1-mini-2025-04-14` as the answer model and `gpt-4o-mini-2024-07-18` as
the judge, with every other setting at the harness default.

## Result provenance

The ThreadAtlas rows under **Reproduced Results** in [results.md](./results.md) were produced by Justin Hoffman
(ThreadAtlas) on 2026-11-16. Each cell is one full run of the unchanged pipeline and prompts.

| | |
|---|---|
| Harness | OmniMemEval at commit [`5e8cdc115b2eff391e72192dbfe3786ebadc9005`](https://github.com/MemTensor/OmniMemEval/tree/5e8cdc115b2eff391e72192dbfe3786ebadc9005), unchanged: every run's `experiment_config.sh` records `git_dirty=0` |
| Backend | `threadatlas_memory` at [jhffmn82/threadatlas@015143c](https://github.com/jhffmn82/threadatlas/tree/015143cdb995497022b88878c8ebc8c5223d1eee/threadatlas_memory), unchanged, through the `threadatlas` adapter of that harness commit |
| Memory reading | `gpt-4.1-mini-2025-04-14` through the OpenAI API (`THREADATLAS_READ_MODEL`). Every session was read during the run by `add()`. No store was built beforehand: `THREADATLAS_STORE_DIR`, `THREADATLAS_STEP0_DIR` and `THREADATLAS_SOURCE` were not set |
| Answer and judge | `gpt-4.1-mini-2025-04-14` and `gpt-4o-mini-2024-07-18` through the OpenAI API; `TOPK=20`, `LLM_WORKERS=10`, `NUM_RUNS=1` |
| Embeddings | `BAAI/bge-small-en-v1.5` through fastembed, on the machine running the harness |
| Retrieval | the defaults: two hops from 40 hits of each search |
| Deployment | local, in-process, one store per user under `THREADATLAS_STATE_DIR` |
| Machine | a Kaggle notebook, one notebook version per benchmark. LongMemEval ran the embedding model on a T4 GPU (`THREADATLAS_EMBED_DEVICE="cuda"`); the other four ran on CPU |
| Data | prepared by the harness's own prepare scripts; each run's `provenance.json` holds the SHA-256 of its prepared file |

| Benchmark | Started and reported (UTC) | `--workers` | Stores | `add()` calls | Reading calls | Questions | Score | Context Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 06:00 to 06:41 | 10 | 20 | 544 | 1,597 | 1,540 | 88.90 | 6,213 |
| BEAM 100K | 07:13 to 08:48 | 20 | 20 | 90 | 735 | 400 | 73.12 | 5,664 |
| PersonaMem v2 | 07:16 to 12:04 | 20 | 200 | 2,426 | 8,168 | 5,000 | 36.92 | 4,268 |
| LongMemEval | 07:18 to 11:18 | 60 | 500 | 23,867 | 85,253 | 500 | 87.00 | 4,351 |
| HaluMem-Medium | 07:25 to 11:38 | 20 | 20 | 1,387 | 4,716 | 3,467 | 83.82 | 3,052 |

`--workers` is the number of users whose sessions are added side by side. One user's sessions are added in order.

The [run records](https://github.com/jhffmn82/threadatlas/tree/9830056802a60eb1139f60d8cbf4b3df3fe80d8d/omnimemeval/baseline) hold, for
each run, the per-question search results, responses and judged answers, the grades, the stage status files, the
answer-token receipts, the harness's settings and report, the logs, the installed packages, and every reading call
with its token counts. The notebook that ran them and a `SHA256SUMS` of every file are with them.

- **Coverage.** In every run each stage reports success for every question, with no failed or skipped record.
  PersonaMem v2 is scored by rule matching, so it has a search and an answer stage and no judge stage.
- **Arithmetic.** The judged files give LoCoMo 1,369 of 1,540, LongMemEval 435 of 500 and HaluMem 2,906 of
  3,467 right. BEAM's mean nugget score over its 400 questions is 0.7312, with a per-question standard deviation
  of 0.3441. PersonaMem v2's responses file has 1,846 of 5,000 right.
- **Context Tokens** is answer prompt tokens divided by answer calls, as each run's report prints it.
- **Harness commit.** `main` has two commits after `5e8cdc1`: the cachly adapter (`e2bb276`) and a LoCoMo
  image-caption fix for Zep graph mode and Graphiti (`1619178`). ThreadAtlas's LoCoMo sessions go through the
  default two-speaker branch of `locomo_ingestion.py` and its searches through `generic_text_search`. Neither
  commit changes those.
- **LoCoMo has three full runs in this setting.** The cell is the last one, the run on the package revision
  that every other cell used.

  | Run (UTC) | Harness | Package | Right of 1,540 | Overall |
  |---|---|---|---:|---:|
  | 2026-11-07 | `7c79d46` with the adapter's files | published as `a3337c5f` | 1,362 | 88.44 |
  | 2026-11-16 04:20 | `5e8cdc1` | `18df19c` | 1,381 | 89.68 |
  | 2026-11-16 06:00 | `5e8cdc1` | `015143c` | 1,369 | 88.90 |

  The first is the run already listed in the [records of 2026-11-07](https://github.com/jhffmn82/threadatlas/tree/dff7388606c3979c410195e28ce0b59cf4faa97a/omnimemeval).
  Between `18df19c` and `015143c` the package gained retries on rate limits, a lock so that one batch is embedded
  at a time, the GPU option and an "undated" label for a session added with no date. None of these changes how a
  dated session is read or retrieved. The three scores span 1.24 points. No run is combined with another.
- **The other four cells** are each the only full run of that benchmark in this setting.
- **Short runs.** Before the full runs, the first users of four benchmarks were run once to time them: BEAM 100K
  (1 of 20 conversations), HaluMem (1 of 20 users), PersonaMem v2 (2 of 200 personas) and LongMemEval (20 of 500
  users). They are partial runs, not scores. Their reports are with the records.

## Version, deployment and settings of the reported runs

These are the runs of 2026-11-07, listed under **Published Reference Results**.

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

**Warning:** the following commands shorten prepared dataset files in place. Run them only in a
disposable checkout with copied (not linked) datasets, never against your only prepared copy.

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

Run the prepare scripts again before a full run in that checkout. The Kaggle notebook in the results
repository does the same with `RUN = "smoke"`.

These commands ran as written on 2026-11-07 on a Kaggle CPU notebook (Python 3.13.18, kit version 5,
`gpt-5.6-luna`): every record finished, LongMemEval 20 of 20 and LoCoMo 141 of 152 judged right. The
[records and installed package versions](https://github.com/jhffmn82/threadatlas/tree/dff7388606c3979c410195e28ce0b59cf4faa97a/omnimemeval/smoke)
are with the results.
