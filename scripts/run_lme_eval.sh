#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

SCRIPT_NAME="run_lme_eval.sh"
source "$SCRIPT_DIR/_experiment_utils.sh"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_lme_eval.sh --lib <memory-lib> --env <env-file> [options]

Examples:
  ./scripts/run_lme_eval.sh --lib hindsight --env .env.hindsight
  ./scripts/run_lme_eval.sh --lib everos --env .env.everos --streaming 1
  ./scripts/run_lme_eval.sh --lib hindsight --env .env.hindsight --version hs_recall_v1 --workers 5
  ./scripts/run_lme_eval.sh --lib hindsight --env .env.hindsight --wait-after-ingest 120
  ./scripts/run_lme_eval.sh --lib memos --env .env.memos --to-step 2
  ./scripts/run_lme_eval.sh --lib backboard --env .env.backboard --workers 3
  ./scripts/run_lme_eval.sh --env .env.hindsight --replay results/lme/hindsight-hs_recall_v1/

Options:
  --lib <name>            Memory product key, e.g. hindsight, memos, zep, mem0, backboard.
  --version <name>        Result version suffix. Default: omnimemeval_{yyyymmdd}.
  --workers <n>           Worker count for memory API (ingestion/search). Default: 2.
  --llm-workers <n>      Max concurrent LLM API calls (response/eval). Default: 10.
  --top-k <n>           Search top-k. Default: 20.
  --num-runs <n>          LLM-as-Judge runs per question. Default: 1.
  --save-model-input <0|1> Save response-stage model_input. Default: 0.
  --allow-empty-search <0|1> Allow successful searches with no raw memories. Default: 1.
  --skip-failed-search <0|1> Explicitly skip failed search calls. Default: 0.
  --skip-failed-answer <0|1> Explicitly skip failed answer calls. Default: 0.
  --skip-failed-judge <0|1> Explicitly skip failed judge calls. Default: 0.
  --skip-failed-streaming <0|1> Streaming mode: explicitly skip failed units. Default: 0.
  --clear <0|1>           Clear existing memories before ingestion. Default: 0.
  --notify <0|1>          Send report notification. Default: 0.
  --wait-after-ingest <s> Seconds to wait after ingestion for async processing. Default: 0.
                          (Hindsight needs time for fact extraction; recommend 60-180s)
  --streaming <0|1>       Use add-search-delete streaming mode. Default: 0.
  --start-idx <n>         Streaming mode: first conversation index. Default: 0.
  --end-idx <n>           Streaming mode: last conversation index. Default: all.
  --restart-unit <0|1>    Streaming mode: delete and discard unit checkpoints before reprocessing. Default: 0.
  --no-resume <0|1>       Streaming mode: ignore streaming_completed.txt. Default: 0.
  --env <file>            Required. Load environment variables from this file.
  --from-step <n>         Start from pipeline step n.
  --to-step <n>           Stop after pipeline step n.
  --replay <dir>          Re-run from a saved experiment_config.sh.
  -h, --help              Show this help.

Pipeline steps:
  1 Memory Ingestion      — Store conversations into the memory system
  2 Memory Search         — Retrieve relevant context for each question
  3 Answer Generation     — Generate answers using retrieved context (or reflect)
  4 LLM-as-Judge          — Grade answers against gold standard
  5 Metric Calculation    — Compute accuracy, NLP metrics, latency stats
  6 Report Generation     — Generate markdown report and optional notification

Hindsight-specific notes:
  Set HINDSIGHT_MODE=recall (default) or HINDSIGHT_MODE=reflect in .env.hindsight.
  - recall: Recall API retrieves memories → external LLM generates answer (step 3)
  - reflect: Reflect API generates answer directly → step 3 uses reflect output

Backboard-specific notes:
  Reference: https://github.com/Backboard-io/Backboard-longmemEval-results
  Set BACKBOARD_EVAL_MODE=reflect (default) or BACKBOARD_EVAL_MODE=rag in .env.backboard.
  - reflect: Backboard's built-in LLM answers directly with memory="Auto" + send_to_llm=true
             (matches official benchmark approach; step 3 is auto-skipped)
  - rag:     Retrieve memories → external ANSWER_MODEL generates answer (step 3)

  Key env variables:
    BACKBOARD_INGEST_MODE   — per_turn (official, accurate) or batch (fast, lower quality)
    BACKBOARD_MEMORY_MODE   — lite (Memory Lite) or pro (Memory Pro, higher precision)
    BACKBOARD_SEARCH_MODE   — message (Readonly retrieval pipeline) or api (basic search)
    BACKBOARD_WAIT_FOR_MEMORY — true (poll memory ops to completion; recommended)
    BACKBOARD_LLM_PROVIDER  — LLM provider for reflect mode (openai, anthropic, google)
    BACKBOARD_LLM_MODEL_NAME — Model name for reflect mode (e.g. gpt-4.1-mini)

  Recommended settings (matching official benchmark):
    --workers 3             (Backboard API rate-limits aggressively)
    --wait-after-ingest 0   (operations are polled individually when WAIT_FOR_MEMORY=true)
EOF
}

show_help_if_requested "$@"
extract_env_arg "$@"
set -- "${_REMAINING_ARGS[@]}"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

# ─── Replay or Normal mode ────────────────────────────────────────────────────
if try_replay "$@"; then
    shift 2
else
    LIB=""
    VERSION="omnimemeval_$(date +%Y%m%d)"
    WORKERS=2
    LLM_WORKERS=10
    _env_llm_workers=$(grep -E '^LLM_WORKERS=' "$OMNIMEMEVAL_ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'" || true)
    [[ -n "$_env_llm_workers" ]] && LLM_WORKERS="$_env_llm_workers"
    TOPK="${TOPK:-20}"
    _env_topk=$(grep -E '^TOPK=' "$OMNIMEMEVAL_ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'" || true)
    [[ -n "$_env_topk" ]] && TOPK="$_env_topk"
    NUM_RUNS=1
    SAVE_MODEL_INPUT=0
    ALLOW_EMPTY_SEARCH=1
    SKIP_FAILED_SEARCH=0
    SKIP_FAILED_ANSWER=0
    SKIP_FAILED_JUDGE=0
    SKIP_FAILED_STREAMING=0
    CLEAR=0
    NOTIFY=0
    WAIT_AFTER_INGEST=0
    STREAMING=0
    START_IDX=0
    END_IDX=""
    RESTART_UNIT=0
    NO_RESUME=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --lib)
                LIB="${2:?--lib requires a value}"
                shift 2
                ;;
            --version)
                VERSION="${2:?--version requires a value}"
                shift 2
                ;;
            --workers)
                WORKERS="${2:?--workers requires a number}"
                shift 2
                ;;
            --llm-workers)
                LLM_WORKERS="${2:?--llm-workers requires a number}"
                shift 2
                ;;
            --top-k)
                TOPK="${2:?--top-k requires a number}"
                shift 2
                ;;
            --num-runs)
                NUM_RUNS="${2:?--num-runs requires a number}"
                shift 2
                ;;
            --save-model-input|--save_model_input)
                SAVE_MODEL_INPUT="${2:?--save-model-input requires 0 or 1}"
                shift 2
                ;;
            --allow-empty-search|--allow_empty_search)
                ALLOW_EMPTY_SEARCH="${2:?--allow-empty-search requires 0 or 1}"
                shift 2
                ;;
            --skip-failed-search|--skip_failed_search)
                SKIP_FAILED_SEARCH="${2:?--skip-failed-search requires 0 or 1}"
                shift 2
                ;;
            --skip-failed-answer|--skip_failed_answer)
                SKIP_FAILED_ANSWER="${2:?--skip-failed-answer requires 0 or 1}"
                shift 2
                ;;
            --skip-failed-judge|--skip_failed_judge)
                SKIP_FAILED_JUDGE="${2:?--skip-failed-judge requires 0 or 1}"
                shift 2
                ;;
            --skip-failed-streaming|--skip_failed_streaming)
                SKIP_FAILED_STREAMING="${2:?--skip-failed-streaming requires 0 or 1}"
                shift 2
                ;;
            --clear)
                CLEAR="${2:?--clear requires 0 or 1}"
                shift 2
                ;;
            --notify)
                NOTIFY="${2:?--notify requires 0 or 1}"
                shift 2
                ;;
            --wait-after-ingest)
                WAIT_AFTER_INGEST="${2:?--wait-after-ingest requires seconds}"
                shift 2
                ;;
            --streaming)
                STREAMING="${2:?--streaming requires 0 or 1}"
                shift 2
                ;;
            --start-idx)
                START_IDX="${2:?--start-idx requires a number}"
                shift 2
                ;;
            --end-idx)
                END_IDX="${2:?--end-idx requires a number}"
                shift 2
                ;;
            --restart-unit)
                RESTART_UNIT="${2:?--restart-unit requires 0 or 1}"
                shift 2
                ;;
            --no-resume)
                NO_RESUME="${2:?--no-resume requires 0 or 1}"
                shift 2
                ;;
            --replay)
                echo "Error: --replay must be followed by a results directory"
                exit 1
                ;;
            *)
                echo "Error: unknown argument: $1"
                echo ""
                usage
                exit 1
                ;;
        esac
    done

    if [[ -z "$LIB" ]]; then
        echo "Error: --lib is required in normal mode"
        echo ""
        usage
        exit 1
    fi
fi
SAVE_MODEL_INPUT="${SAVE_MODEL_INPUT:-0}"
ALLOW_EMPTY_SEARCH="${ALLOW_EMPTY_SEARCH:-1}"
SKIP_FAILED_SEARCH="${SKIP_FAILED_SEARCH:-0}"
SKIP_FAILED_ANSWER="${SKIP_FAILED_ANSWER:-0}"
SKIP_FAILED_JUDGE="${SKIP_FAILED_JUDGE:-0}"
SKIP_FAILED_STREAMING="${SKIP_FAILED_STREAMING:-0}"
STREAMING="${STREAMING:-0}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
RESTART_UNIT="${RESTART_UNIT:-0}"
NO_RESUME="${NO_RESUME:-0}"
require_positive_int "--workers" "$WORKERS"
require_positive_int "--llm-workers" "$LLM_WORKERS"
require_positive_int "--top-k" "$TOPK"
require_positive_int "--num-runs" "$NUM_RUNS"
require_nonnegative_int "--start-idx" "$START_IDX"
if [[ -n "$END_IDX" ]]; then
    require_nonnegative_int "--end-idx" "$END_IDX"
fi
require_binary_flag "--save-model-input" "$SAVE_MODEL_INPUT"
require_binary_flag "--allow-empty-search" "$ALLOW_EMPTY_SEARCH"
require_binary_flag "--skip-failed-search" "$SKIP_FAILED_SEARCH"
require_binary_flag "--skip-failed-answer" "$SKIP_FAILED_ANSWER"
require_binary_flag "--skip-failed-judge" "$SKIP_FAILED_JUDGE"
require_binary_flag "--skip-failed-streaming" "$SKIP_FAILED_STREAMING"
require_binary_flag "--clear" "$CLEAR"
require_binary_flag "--notify" "$NOTIFY"
require_binary_flag "--streaming" "$STREAMING"
require_binary_flag "--restart-unit" "$RESTART_UNIT"
require_binary_flag "--no-resume" "$NO_RESUME"
require_nonnegative_seconds "--wait-after-ingest" "$WAIT_AFTER_INGEST"

RESULTS_DIR="$PROJECT_DIR/results/lme/${LIB}-${VERSION}"
PARAMS_BLOCK="LIB=\"$LIB\"
VERSION=\"$VERSION\"
WORKERS=$WORKERS
LLM_WORKERS=$LLM_WORKERS
TOPK=$TOPK
NUM_RUNS=$NUM_RUNS
SAVE_MODEL_INPUT=$SAVE_MODEL_INPUT
ALLOW_EMPTY_SEARCH=$ALLOW_EMPTY_SEARCH
SKIP_FAILED_SEARCH=$SKIP_FAILED_SEARCH
SKIP_FAILED_ANSWER=$SKIP_FAILED_ANSWER
SKIP_FAILED_JUDGE=$SKIP_FAILED_JUDGE
SKIP_FAILED_STREAMING=$SKIP_FAILED_STREAMING
CLEAR=$CLEAR
NOTIFY=$NOTIFY
WAIT_AFTER_INGEST=$WAIT_AFTER_INGEST
STREAMING=$STREAMING
START_IDX=$START_IDX
END_IDX=\"$END_IDX\"
RESTART_UNIT=$RESTART_UNIT
NO_RESUME=$NO_RESUME
FROM_STEP=$_FROM_STEP
TO_STEP=$_TO_STEP"

save_experiment_config

# ─── Run evaluation pipeline ─────────────────────────────────────────────────

CLEAR_FLAG=""
NOTIFY_FLAG=""
if [[ "$CLEAR" == "1" ]]; then
    CLEAR_FLAG="--clear"
fi
if [[ "$NOTIFY" == "1" ]]; then
    NOTIFY_FLAG="--notify"
fi

# Detect reflect-mode systems where step 3 is a no-op (answer comes from step 2)
_REFLECT_MODE=false
if [[ "$LIB" == "backboard" ]]; then
    _bb_eval_mode=$(grep -E '^BACKBOARD_EVAL_MODE=' "$OMNIMEMEVAL_ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'" | tr '[:upper:]' '[:lower:]')
    if [[ "${_bb_eval_mode:-reflect}" == "reflect" ]]; then
        _REFLECT_MODE=true
    fi
elif [[ "$LIB" == "hindsight" ]]; then
    _hs_mode=$(grep -E '^HINDSIGHT_MODE=' "$OMNIMEMEVAL_ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'" | tr '[:upper:]' '[:lower:]')
    if [[ "${_hs_mode}" == "reflect" ]]; then
        _REFLECT_MODE=true
    fi
fi

echo "LongMemEval config:"
echo "  lib=$LIB"
echo "  version=$VERSION"
echo "  workers=$WORKERS"
echo "  llm_workers=$LLM_WORKERS"
echo "  top_k=$TOPK"
echo "  num_runs=$NUM_RUNS"
echo "  save_model_input=$SAVE_MODEL_INPUT"
echo "  allow_empty_search=$ALLOW_EMPTY_SEARCH"
echo "  skip_failed_search=$SKIP_FAILED_SEARCH"
echo "  skip_failed_answer=$SKIP_FAILED_ANSWER"
echo "  skip_failed_judge=$SKIP_FAILED_JUDGE"
echo "  skip_failed_streaming=$SKIP_FAILED_STREAMING"
echo "  clear=$CLEAR"
echo "  notify=$NOTIFY"
echo "  wait_after_ingest=${WAIT_AFTER_INGEST}s"
echo "  streaming=$STREAMING"
if [[ "$STREAMING" == "1" ]]; then
    echo "  streaming_range=${START_IDX}-${END_IDX:-end}"
    echo "  streaming_restart_unit=$RESTART_UNIT"
    echo "  streaming_no_resume=$NO_RESUME"
fi
echo "  steps=${_FROM_STEP}-${_TO_STEP}"
if [[ "$_REFLECT_MODE" == "true" ]]; then
    echo "  mode=reflect (answer generated during search; step 3 skipped)"
fi

pipeline_start 6

if [[ "$STREAMING" == "1" ]]; then
    STREAM_ARGS=(
        --lib "$LIB"
        --env "$OMNIMEMEVAL_ENV_FILE"
        --version "$VERSION"
        --top-k "$TOPK"
        --allow-empty-search "$ALLOW_EMPTY_SEARCH"
        --start-idx "$START_IDX"
        --wait-after-ingest "$WAIT_AFTER_INGEST"
    )
    if [[ -n "$END_IDX" ]]; then
        STREAM_ARGS+=(--end-idx "$END_IDX")
    fi
    if [[ "$NO_RESUME" == "1" ]]; then
        STREAM_ARGS+=(--no-resume)
    fi
    if [[ "$RESTART_UNIT" == "1" || "$CLEAR" == "1" ]]; then
        STREAM_ARGS+=(--restart-unit)
    fi
    if [[ "$SKIP_FAILED_STREAMING" == "1" ]]; then
        STREAM_ARGS+=(--skip-failed-streaming)
    fi

    run_step "Memory Streaming Add/Search/Delete" \
        python scripts/longmemeval/lme_streaming.py "${STREAM_ARGS[@]}"

    run_step "Memory Search" \
        bash -c 'echo "Search results were generated by streaming step."'
else
    run_step "Memory Ingestion" \
        python scripts/longmemeval/lme_ingestion.py \
            --lib "$LIB" \
            --version "$VERSION" \
            --workers "$WORKERS" \
            $CLEAR_FLAG

    if is_positive_seconds "$WAIT_AFTER_INGEST"; then
        echo "⏳ Waiting ${WAIT_AFTER_INGEST}s for async memory processing (fact extraction)..."
        sleep "$WAIT_AFTER_INGEST"
        echo "✅ Wait complete, proceeding to search"
    fi

    run_step "Memory Search" \
        python scripts/longmemeval/lme_search.py \
            --lib "$LIB" \
            --version "$VERSION" \
            --top-k "$TOPK" \
            --workers "$WORKERS" \
            --allow-empty-search "$ALLOW_EMPTY_SEARCH" \
            --skip-failed-search "$SKIP_FAILED_SEARCH"
fi

if [[ "$_REFLECT_MODE" == "true" ]]; then
    run_step "Answer Generation (reflect passthrough)" \
        python scripts/longmemeval/lme_responses.py \
            --lib "$LIB" \
            --version "$VERSION" \
            --llm-workers "$LLM_WORKERS" \
            --save-model-input "$SAVE_MODEL_INPUT" \
            --skip-failed-answer "$SKIP_FAILED_ANSWER"
else
    run_step "Answer Generation" \
        python scripts/longmemeval/lme_responses.py \
            --lib "$LIB" \
            --version "$VERSION" \
            --llm-workers "$LLM_WORKERS" \
            --save-model-input "$SAVE_MODEL_INPUT" \
            --skip-failed-answer "$SKIP_FAILED_ANSWER"
fi

run_step "LLM-as-Judge Evaluation" \
    python scripts/longmemeval/lme_eval.py \
        --lib "$LIB" \
        --version "$VERSION" \
        --llm-workers "$LLM_WORKERS" \
        --num_runs "$NUM_RUNS" \
        --skip-failed-judge "$SKIP_FAILED_JUDGE"

run_step "Metric Calculation" \
    python scripts/longmemeval/lme_metric.py --lib "$LIB" --version "$VERSION"

run_step "Report Generation" \
    python scripts/longmemeval/lme_report.py --lib "$LIB" --version "$VERSION" $NOTIFY_FLAG

pipeline_summary
