# ============================================================
# MemEval — Zep (Cloud Graph API)
# Usage: ./scripts/run_locomo_eval.sh --lib zep --env .env.zep
# ============================================================

# ─── Zep API ─────────────────────────────────────────────────
ZEP_API_KEY="<YOUR_ZEP_API_KEY>"                   # ⚠️ REQUIRED
ZEP_BASE_URL="https://api.getzep.com"
ZEP_QPS=5
ZEP_BATCH_SIZE=20
ZEP_MAX_BATCH_CHARS=40000
ZEP_MESSAGE_MAX_CHARS=4096
ZEP_WAIT_FOR_INGESTION="true"
ZEP_INGEST_TIMEOUT_SECONDS=300
ZEP_INGEST_POLL_INTERVAL=1
# Optional write-time controls:
# ZEP_IGNORE_ROLES=""
# ZEP_RETURN_CONTEXT=""

# Group mode (official LoCoMo config uses graph.add + group_id search)
# Set "true" for LoCoMo multi-speaker eval; "false" for single-user benchmarks
ZEP_USE_GROUP="true"

# Runner defaults
LLM_WORKERS=10

# Search parameters
TOPK=20
# Best-performance preset: edges=cross_encoder + nodes=rrf
ZEP_SEARCH_SCOPES="edges,nodes"
ZEP_EDGES_RERANKER="cross_encoder"
ZEP_NODES_RERANKER="rrf"
# ZEP_EPISODES_RERANKER="rrf"
ZEP_QUERY_MAX_CHARS=400
# Optional advanced parameters:
# ZEP_RERANKER="rrf"
# ZEP_MMR_LAMBDA=0.5
# ZEP_SEARCH_FILTERS_JSON='{"node_labels":["Person"]}'
# ZEP_MAX_CHARACTERS=2000
# ZEP_RETURN_RAW_RESULTS="false"
# ZEP_DISABLE_DEFAULT_ONTOLOGY="false"

# ─── ANSWER (Answer Generation) ─────────────────────────────
ANSWER_MODEL="gpt-4.1-mini"
ANSWER_BASE_URL="https://api.openai.com/v1"        # ⚠️ REQUIRED — set your LLM endpoint
ANSWER_API_KEY="<YOUR_ANSWER_API_KEY>"             # ⚠️ REQUIRED

# ─── EVAL (LLM-as-Judge Scoring) ────────────────────────────
EVAL_MODEL="gpt-4o-mini"
EVAL_BASE_URL="https://api.openai.com/v1"          # ⚠️ REQUIRED — set your LLM endpoint
EVAL_API_KEY="<YOUR_EVAL_API_KEY>"                 # ⚠️ REQUIRED

# ─── Optional Shared Controls ───────────────────────────────
# Memory API retry controls. HTTP retry count includes the initial attempt.
# MEMEVAL_MEMORY_MAX_RETRIES=8
# MEMEVAL_MEMORY_SDK_MAX_RETRIES=8

# Long add-call progress logging. Set to 0 to disable periodic heartbeat logs.
# MEMEVAL_ADD_HEARTBEAT_SECONDS=30

# Global LLM retry/timeout controls. ANSWER_* and EVAL_* override LLM_*.
# LLM_MAX_RETRIES=4
# LLM_TIMEOUT_SECONDS=600
# LLM_RETRY_BASE_SECONDS=1
# LLM_RETRY_MAX_SECONDS=60
# ANSWER_MAX_RETRIES=4
# ANSWER_TIMEOUT_SECONDS=600
# ANSWER_RETRY_BASE_SECONDS=1
# ANSWER_RETRY_MAX_SECONDS=60
# EVAL_MAX_RETRIES=4
# EVAL_TIMEOUT_SECONDS=600
# EVAL_RETRY_BASE_SECONDS=1
# EVAL_RETRY_MAX_SECONDS=60

# Optional metric resource download/cache overrides.
# HF_ENDPOINT="https://huggingface.co"
# HF_HOME="~/.cache/huggingface"
# MEMEVAL_NLTK_INDEX_URL=
# MEMEVAL_NLTK_GITHUB_PROXY=

# Optional global add batching fallback; product-specific *_MAX_BATCH_CHARS wins.
# MAX_BATCH_CHARS=0

# ─── Notification (Optional) ────────────────────────────────
# DINGTALK_ACCESS_TOKEN="<YOUR_DINGTALK_ACCESS_TOKEN>"
# DINGTALK_SECRET="<YOUR_DINGTALK_SECRET>"
