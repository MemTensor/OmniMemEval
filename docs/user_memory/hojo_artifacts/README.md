# Hojo reference result artifacts

These are question-level score records for the five included published reference
rows. The source conversation datasets, model inputs, generated answers, service
identifiers, credentials and environment snapshots are not redistributed.

| File | Questions | Coverage key |
| --- | ---: | --- |
| [locomo.jsonl](./locomo.jsonl) | 1,540 | Conversation/question position in retained output; SHA-256 of the public question text |
| [longmemeval_s.jsonl](./longmemeval_s.jsonl) | 500 | Dataset indexes 0–499; question SHA-256 |
| [beam100k.jsonl](./beam100k.jsonl) | 400 | Record position, question index, dimension and question SHA-256 |
| [personamem_v2.jsonl](./personamem_v2.jsonl) | 5,000 | Dataset row indexes 0–4,999; question SHA-256 |
| [halumem.jsonl](./halumem.jsonl) | 3,467 | Export record ID, retained session/question positions and question SHA-256 |

`question_sha256` hashes UTF-8 question text exactly as retained. It allows a
reviewer with the matching public dataset to check coverage without including
another copy of its questions here. Categories and exact scores are retained.
The verifier checks record counts, unique IDs, binary scoring, numerical
BEAM rubric aggregation, file hashes, category counts and overall means. It
also checks contiguous dataset indexes for LongMemEval-S and PersonaMem v2.
It does not load the official datasets or independently check every question
hash against them; record-count verification is not a dataset coverage audit.
Event-ordering records retain their final Kendall tau-b branch score; the
score-only export does not independently rerun the ordering judge.

## Recompute all aggregates

Only Python's standard library is required; no API calls are made:

```bash
python3 docs/user_memory/hojo_artifacts/verify_results.py
```

[summary.json](./summary.json) contains unrounded scores, per-category results,
effective configuration fields, dataset hashes where recorded, the SHA-256 of
each original judged file and the SHA-256 of each public score file. The original
judged-file hashes identify privately retained originals; those files also
contain answers and service identifiers and are not included here.

To regenerate these exports from the original result tree:

```bash
python3 docs/user_memory/hojo_artifacts/prepare_results.py \
  --source-root /path/to/retained/results \
  --service-source /path/to/MemoryEngine/source
```

## Harness and deployment provenance

The [harness snapshot](./harness/scripts/) includes answer prompts, judge
prompts, ingestion/retrieval behavior and scoring logic for the standalone
LongMemEval, HaluMem, BEAM and PersonaMem runners. Its
[source manifest](./harness/source_manifest.json) records source commit
`8d811afe207ec12d698a328f370128598be9b54f` and per-file hashes, with one
explicit per-file source override: the BEAM runner is taken from
`e983688562daf22cece0fc050bea76d1a4aee4a8`, whose prompt and retrieval
profiles match the recorded 69.58% run. This identifies the supplied runner
source, not a verified historical deployment commit. It is a
reference snapshot from the retained source checkout, not a claim that every
historical service deployment used that commit. Exact deployment image/commit
metadata was not retained in the result reports. The LoCoMo checkout was based
on OmniMemEval commit `0b1ea8d28aa2d3e03ac4a6aee17b3006a131da7d` with local
Hojo integration changes; a clean historical harness revision was not captured.

Runs are identified by the version strings and dataset hashes in `summary.json`.
No deployment version is inferred from the score or from a newer checkout.
The separate adapter contribution is not part of this documentation change.

## Running the reference harness

These commands require a separately deployed MemoryEngine V2 API, the matching
benchmark data and private credentials. They create new evaluations and incur
API usage; they are distinct from the offline aggregate verifier above.
Choose a new `OMNIMEMEVAL_*_VERSION` and result directory for each run.

The standalone scripts require Python, `httpx`, and the `openai` Python package.
Configure the following variables privately; do not commit their values:

```text
OMNIMEMEVAL_MEMORY_BASE_URL
OMNIMEMEVAL_MEMORY_API_KEY
OMNIMEMEVAL_TENANT_ID
OMNIMEMEVAL_ORG_ID
OMNIMEMEVAL_ANSWER_BASE_URL
OMNIMEMEVAL_ANSWER_API_KEY
OMNIMEMEVAL_EVAL_BASE_URL
OMNIMEMEVAL_EVAL_API_KEY
```

Dataset files are obtained separately under their original licenses. For
LongMemEval set `OMNIMEMEVAL_LME_DATA_PATH` to the matching cleaned gzip file.
For BEAM set `OMNIMEMEVAL_BEAM100K_DATA_PATH` to the matching 100K gzip file.
For PersonaMem set `OMNIMEMEVAL_PERSONAMEM_DATA_PATH` to the downloaded dataset
root. HaluMem's runner reads `harness/data/omnimemeval/HaluMem-Medium.jsonl`
relative to this snapshot; put the matching downloaded file there locally.

The following examples show the relevant full-run settings. Their dataset
hashes, models and effective retrieval/profile fields must match `summary.json`
when comparing with these reference records. The scripts expose their prompt
templates and scoring logic; their source provenance is documented above.
These materials do not claim shared-baseline reproduction or guarantee the
same score from a new evaluation against a changed service or model endpoint.
The LongMemEval example runs serially; the retained run configuration records
six workers. The example keeps the same scoring policy, not the same scheduling.

```bash
cd docs/user_memory/hojo_artifacts
export OMNIMEMEVAL_ANSWER_MODEL=gpt-4.1-mini
export OMNIMEMEVAL_EVAL_MODEL=gpt-4o-mini
export HOJO_INDEX_ASSISTANT_MESSAGES=1

# LongMemEval-S: use an absolute output directory and a new version.
OMNIMEMEVAL_LME_VERSION=reference_lme_new \
OMNIMEMEVAL_LME_OUTPUT_DIR=/absolute/path/to/new-lme-output \
OMNIMEMEVAL_LME_START_INDEX=0 OMNIMEMEVAL_LME_CASE_LIMIT=500 \
OMNIMEMEVAL_LME_TOP_K=20 OMNIMEMEVAL_LME_CANDIDATE_LIMIT=200 \
OMNIMEMEVAL_LME_MIN_SCORE=0.5 OMNIMEMEVAL_LME_MAX_TOKENS=1500 \
OMNIMEMEVAL_LME_TIMEOUT_SECONDS=900 \
python3 harness/scripts/run_omnimemeval_lme.py

# HaluMem-Medium
OMNIMEMEVAL_HALUMEM_VERSION=reference_hm_new \
OMNIMEMEVAL_HALUMEM_OUTPUT_DIR=/absolute/path/to/new-halumem-output \
OMNIMEMEVAL_HALUMEM_START_INDEX=0 OMNIMEMEVAL_HALUMEM_QUESTION_LIMIT=3467 \
OMNIMEMEVAL_HALUMEM_WORKERS=4 OMNIMEMEVAL_HALUMEM_TOP_K=20 \
OMNIMEMEVAL_HALUMEM_CANDIDATE_LIMIT=200 OMNIMEMEVAL_HALUMEM_MIN_SCORE=0.5 \
OMNIMEMEVAL_HALUMEM_MAX_TOKENS=1500 OMNIMEMEVAL_HALUMEM_TIMEOUT_SECONDS=900 \
python3 harness/scripts/run_omnimemeval_halumem.py

# BEAM 100K: the pinned runner matches the recorded v3 prompt/retrieval profiles.
OMNIMEMEVAL_BEAM100K_VERSION=reference_beam_new \
OMNIMEMEVAL_BEAM100K_OUTPUT_DIR=/absolute/path/to/new-beam-output \
OMNIMEMEVAL_BEAM100K_START_INDEX=0 OMNIMEMEVAL_BEAM100K_CONVERSATION_LIMIT=20 \
OMNIMEMEVAL_BEAM100K_WORKERS=6 OMNIMEMEVAL_BEAM100K_TOP_K=20 \
OMNIMEMEVAL_BEAM100K_CANDIDATE_LIMIT=200 OMNIMEMEVAL_BEAM100K_MIN_SCORE=0.5 \
OMNIMEMEVAL_BEAM100K_MAX_TOKENS=1500 \
OMNIMEMEVAL_BEAM100K_CONTEXT_PROFILE=code_compact_v1 \
OMNIMEMEVAL_BEAM100K_EVIDENCE_PROFILE=user_grounded_v1 \
python3 harness/scripts/run_omnimemeval_beam100k.py

# PersonaMem v2: set PERSONAMEM_ANSWER_MODEL to match its recorded configuration.
OMNIMEMEVAL_ANSWER_MODEL="${PERSONAMEM_ANSWER_MODEL:?Set PERSONAMEM_ANSWER_MODEL}" \
OMNIMEMEVAL_PERSONAMEM_VERSION=reference_pm_new \
OMNIMEMEVAL_PERSONAMEM_OUTPUT_DIR=/absolute/path/to/new-personamem-output \
OMNIMEMEVAL_PERSONAMEM_START_INDEX=0 OMNIMEMEVAL_PERSONAMEM_QUESTION_LIMIT=5000 \
OMNIMEMEVAL_PERSONAMEM_WORKERS=4 OMNIMEMEVAL_PERSONAMEM_TOP_K=20 \
OMNIMEMEVAL_PERSONAMEM_CANDIDATE_LIMIT=200 OMNIMEMEVAL_PERSONAMEM_MIN_SCORE=0.5 \
OMNIMEMEVAL_PERSONAMEM_MAX_TOKENS=1500 \
python3 harness/scripts/run_omnimemeval_personamem_v2.py
```

The LoCoMo reference used the local Hojo integration's `run_locomo_eval.sh`
pipeline with the 1,540-question non-adversarial scope, one judge run and the
effective retrieval settings retained in `summary.json`. Its API-reported
answer/judge model snapshots are recorded in `summary.json`.
The public score export supports aggregate verification; the unavailable clean
historical integration revision remains a reproduction limitation.

Final question execution completed successfully, but the retained files do not
establish that every claim was indexed or later backfilled. These scores
describe the observed deployments, including that limitation.
