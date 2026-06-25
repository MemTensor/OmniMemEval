# OmniMemEval Parameter Reference

This file documents the public parameters used by the LoCoMo and LongMemEval
pipelines. Product-specific templates live in this directory as `.env.<client>`.

## Runner Parameters

| Parameter | Default | Applies to | Description |
|-----------|---------|------------|-------------|
| `--lib` | required | all | Memory adapter key from `client_factory.registry`. |
| `--env` | required | all | Dotenv file with memory and LLM credentials. |
| `--version` | `omnimemeval_<date>` | all | Result directory suffix. |
| `--from-step` | `1` | all | First pipeline step to execute. |
| `--to-step` | final step | all | Last pipeline step to execute. |
| `--replay` | unset | all | Existing result directory for recomputing later stages. |
| `--top-k` | `TOPK` or `20` | search | Number of search results requested from the memory backend. |
| `--llm-workers` | `LLM_WORKERS` or `10` | answer/eval | Concurrent LLM workers. |
| `--allow-empty-search` | `1` | search | Allow successful records when no raw memories are returned. |
| `--skip-failed-search` | `0` | search | Mark failed search records as skipped instead of failing the step. |
| `--skip-failed-answer` | `0` | answer | Mark failed answer records as skipped instead of failing the step. |
| `--skip-failed-judge` | `0` | eval | Mark failed judge records as skipped instead of failing the step. |
| `--notify` | `0` | report | Send optional DingTalk notification when configured. |

LongMemEval streaming-only parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--streaming` | `0` | Run add-search-delete per conversation. |
| `--start-idx` | `0` | First conversation index. |
| `--end-idx` | dataset end | Last conversation index, inclusive. |
| `--restart-unit` | `0` | Re-run a completed unit inside the selected range. |
| `--no-resume` | `0` | Ignore streaming completed-unit checkpoint. |
| `--skip-failed-streaming` | `0` | Mark failed streaming units as skipped instead of failing the streaming step. |
| `--wait-after-ingest` | `WAIT_AFTER_INGEST` or `0` | Seconds to wait after ingestion before search. |

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
| LoCoMo | `data/locomo/locomo10.json` | Included with upstream license notes. |
| LongMemEval | `data/longmemeval/longmemeval_s_cleaned.json` | Download with `python data/longmemeval/prepare_longmemeval.py`. |

## Product Templates

Each `.env.<client>` file contains the memory product variables required by
that adapter plus the shared ANSWER/EVAL variables. Templates intentionally use
neutral placeholders and must not contain real credentials.
