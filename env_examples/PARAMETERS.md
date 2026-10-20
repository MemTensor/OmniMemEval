# OmniMemEval Parameter Reference

This file documents the public parameters used by the LoCoMo, LongMemEval,
BEAM, PersonaMem v2, and HaluMem pipelines. Product-specific templates live in
this directory as `.env.<client>`.

Unless a product-specific table is present below, the corresponding
`.env.<client>` template is the reference for that product's public settings.

## Runner Parameters

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `--lib` | required | all | Memory adapter key from `client_factory.registry`. |
| `--env` | required | all | Dotenv file with memory and LLM credentials. |
| `--version` | `omnimemeval_<date>` | all | Result directory suffix. |
| `--from-step` | `1` | all | First pipeline step to execute. |
| `--to-step` | final step | all | Last pipeline step to execute. |
| `--replay` | unset | all | Existing result directory for recomputing later stages. |
| `--workers` | `2` | ingestion/search | Memory API worker count. |
| `--top-k` | `TOPK` or `20` | search | Number of search results requested from the memory backend. |
| `--llm-workers` | `LLM_WORKERS` or `10` | answer/eval | Concurrent LLM workers. |
| `--num-runs` | `1` | all five User Memory benchmarks | Number of answer repetitions for PersonaMem v2, or judge repetitions for LoCoMo, LongMemEval, BEAM, and HaluMem. |
| `--save-model-input` | `0` | answer | Save response-stage model input for inspection. |
| `--allow-empty-search` | `1` | search | Allow successful records when no raw memories are returned. |
| `--skip-failed-search` | `0` | search | Mark failed search records as skipped instead of failing the step. |
| `--skip-failed-answer` | `0` | answer | Mark failed answer records as skipped instead of failing the step. |
| `--skip-failed-judge` | `0` | eval | Mark failed judge records as skipped instead of failing the step. |
| `--clear` | `0` | ingestion | Delete existing backend memory for the selected run before ingestion. |
| `--notify` | `0` | report | Send optional DingTalk notification when configured. |
| `--wait-after-ingest` | `WAIT_AFTER_INGEST` or `0` | ingestion/search | Seconds to wait after ingestion before search. |

## Benchmark-Specific Parameters

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `--scale` | `100k` | BEAM | BEAM scale: `100k`, `500k`, `1m`, `10m`, or `all`. |
| `--variant` | `medium` | HaluMem | HaluMem dataset variant: `medium` or `long`. |
| `--allow-missing-data` | `0` | PersonaMem v2 | Explicitly evaluate the available subset if chat histories are missing. |

## Streaming Parameters

LongMemEval, BEAM, PersonaMem v2, and HaluMem support streaming mode. Streaming
runs add, search, save, and delete for each unit before moving to the next unit.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--streaming` | `0` | Enable per-unit streaming mode. |
| `--start-idx` | `0` | First unit index. |
| `--end-idx` | dataset end | Last unit index, inclusive. |
| `--restart-unit` | `0` | Re-run a completed unit inside the selected range. |
| `--no-resume` | `0` | Ignore streaming completed-unit checkpoint. |
| `--skip-failed-streaming` | `0` | Mark failed streaming units as skipped instead of failing the streaming step. |

## Cleanup Parameters

`./scripts/run_memory_clear.sh` deletes backend memory written by an evaluation
run. It is destructive only when `--yes` is provided; use `--dry-run` first to
inspect target ids.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--datasets` | `all` | Comma-separated benchmark keys: `locomo,lme,beam,pmv2,hm`. |
| `--beam-scale` | `100k` | BEAM scale to clear. Repeat for multiple scales. |
| `--halumem-variant` | `medium` | HaluMem variant to clear: `medium` or `long`. |
| `--dry-run` | unset | Print target ids without deleting backend data. |
| `--yes` | unset | Confirm destructive deletion. Required unless `--dry-run` is set. |

## Shared Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_WORKERS` | `10` | Default answer/eval worker count. |
| `TOPK` | `20` | Default memory search result count. |
| `ANSWER_MODEL` | required | OpenAI-compatible answer model name. |
| `ANSWER_API_KEY` | required | Answer model API key. |
| `ANSWER_BASE_URL` | required | Answer model endpoint. |
| `EVAL_MODEL` | required | OpenAI-compatible judge model name. |
| `EVAL_API_KEY` | required | Judge model API key. |
| `EVAL_BASE_URL` | required | Judge model endpoint. |
| `LLM_MAX_RETRIES` | `3` | Global LLM retry count. |
| `LLM_TIMEOUT_SECONDS` | `120` | Global LLM request timeout in seconds. |
| `OMNIMEMEVAL_MEMORY_MAX_RETRIES` | `3` | HTTP memory client retry count. |
| `OMNIMEMEVAL_MEMORY_SDK_MAX_RETRIES` | `3` | SDK memory client retry count. |
| `OMNIMEMEVAL_NLTK_INDEX_URL` | unset | Optional NLTK index mirror. |
| `OMNIMEMEVAL_NLTK_GITHUB_PROXY` | unset | Optional proxy prefix for NLTK GitHub downloads. |
| `HF_ENDPOINT` | unset | Optional Hugging Face endpoint mirror. |
| `DINGTALK_ACCESS_TOKEN` | unset | Optional DingTalk robot access token. |
| `DINGTALK_SECRET` | unset | Optional DingTalk signing secret. |

## Benchmark Data

| Benchmark | Data path | Notes |
|-----------|-----------|-------|
| LoCoMo | `data/locomo/locomo10.json` | Download with `python data/locomo/prepare_locomo.py`; see its dataset README for license notes. |
| LongMemEval | `data/longmemeval/longmemeval_s_cleaned.json` | Download with `python data/longmemeval/prepare_longmemeval.py`. |
| BEAM | `data/beam/beam_100k.json` | Default 100K scale; prepare script can download additional scales. |
| PersonaMem v2 | `data/personamem_v2/benchmark/text/benchmark.csv` and `data/personamem_v2/data/chat_history_32k/` | Download with `python data/personamem_v2/prepare_personamem.py`. |
| HaluMem | `data/halumem/HaluMem-Medium.jsonl` | Default medium variant; prepare script can download the long variant. |

## Product Templates

Each `.env.<client>` file contains the memory product variables required by
that adapter plus the shared ANSWER/EVAL variables. Templates intentionally use
neutral placeholders and must not contain real credentials.
