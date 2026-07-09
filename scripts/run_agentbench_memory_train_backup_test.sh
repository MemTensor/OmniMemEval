#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  DEFAULT_PYTHON="$CONDA_PREFIX/bin/python"
else
  DEFAULT_PYTHON="$(command -v python || command -v python3 || true)"
fi
DEFAULT_VERSION="memory_train_backup_test_$(date +%Y%m%d_%H%M%S)"

AGENT="openclaw"
MEMORY_PLUGIN=""
MEMORY_PLUGIN_CONFIG=""
VERSION="$DEFAULT_VERSION"
TRIALS="1"
TEST_RUNS="1"
FEEDBACK_TIMEOUT=""
MEMOS_FEEDBACK_TIMEOUT=""
TRAIN_FEEDBACK=""
MEMOS_STRUCTURED_FEEDBACK=""
PASS_AT=""
PARALLEL="1"
RESULTS_DIR="$PROJECT_DIR/results/agentbench"
ENV_FILE=""
FORCE=0
CONTINUE_ON_ERROR=0
DOMAINS=(
  "software_engineering"
  "information_retrieval"
  "knowledge_work"
  "code_implementation"
  "reasoning"
)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_agentbench_memory_train_backup_test.sh --memory-plugin NAME [options]

Runs each selected domain with the AgentBench memory lifecycle protocol:
  1. set plugin train mode
  2. clear memory
  3. run train split
  4. wait for memory settling / evolution
  5. backup domain memory
  6. before every test run, restore that domain backup
  7. run test split

Options:
  --agent NAME                 Agent runtime name. Default: openclaw
  --memory-plugin NAME         Config name under configs/agentbench/memory_plugins
  --memory-plugin-config FILE  Explicit memory lifecycle YAML
  --version TAG                Result version tag. Default: timestamped
  --trials N / --runs N        Trials per task within each phase. Default: 1
  --test-runs N                Full test split runs per domain. Each run restores memory first. Default: 1
  --feedback-timeout N         Override verifier feedback turn timeout from plugin config
  --train-feedback             Force verifier feedback turn during train
  --no-train-feedback          Disable verifier feedback turn during train
  --memos-structured-feedback  Force MemOS explicit feedback.submit
  --no-memos-structured-feedback
                               Disable MemOS explicit feedback.submit
  --memos-feedback-timeout N   Override MemOS feedback.submit / episode.close timeout from plugin config
  --pass-at N                  Compute pass@n. Default: same as --trials
  --parallel N                 Per-domain task parallelism. Default: 1
  --results-dir DIR            Override results directory. Default: <project>/results/agentbench
  --env FILE                   Extra env file passed to the runner
  --force                      Re-run completed trials
  --continue-on-error          Continue remaining domains after one fails
  --domains CSV                Override domain list, comma-separated
  -h, --help                   Show this help

Environment:
  PYTHON=/path/to/python       Override Python executable
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --memory-plugin)
      MEMORY_PLUGIN="$2"
      shift 2
      ;;
    --memory-plugin-config)
      MEMORY_PLUGIN_CONFIG="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --trials|--runs)
      TRIALS="$2"
      shift 2
      ;;
    --test-runs)
      TEST_RUNS="$2"
      shift 2
      ;;
    --feedback-timeout)
      FEEDBACK_TIMEOUT="$2"
      shift 2
      ;;
    --train-feedback)
      TRAIN_FEEDBACK=1
      shift
      ;;
    --memos-feedback-timeout)
      MEMOS_FEEDBACK_TIMEOUT="$2"
      shift 2
      ;;
    --memos-structured-feedback)
      MEMOS_STRUCTURED_FEEDBACK="1"
      shift
      ;;
    --no-memos-structured-feedback)
      MEMOS_STRUCTURED_FEEDBACK="0"
      shift
      ;;
    --no-train-feedback)
      TRAIN_FEEDBACK=0
      shift
      ;;
    --pass-at)
      PASS_AT="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --env)
      ENV_FILE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --domains)
      IFS=',' read -r -a DOMAINS <<< "$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MEMORY_PLUGIN" && -z "$MEMORY_PLUGIN_CONFIG" ]]; then
  echo "ERROR: --memory-plugin or --memory-plugin-config is required." >&2
  exit 2
fi

PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  else
    PYTHON="$(command -v python || command -v python3)"
  fi
fi

BASE_ARGS=(
  "--agent" "$AGENT"
  "--protocol" "memory_train_backup_test"
  "--version" "$VERSION"
  "--trials" "$TRIALS"
  "--test-runs" "$TEST_RUNS"
  "--parallel" "$PARALLEL"
)
if [[ -n "$FEEDBACK_TIMEOUT" ]]; then
  BASE_ARGS+=("--feedback-timeout" "$FEEDBACK_TIMEOUT")
fi
if [[ -n "$MEMOS_FEEDBACK_TIMEOUT" ]]; then
  BASE_ARGS+=("--memos-feedback-timeout" "$MEMOS_FEEDBACK_TIMEOUT")
fi
if [[ "$TRAIN_FEEDBACK" == "1" ]]; then
  BASE_ARGS+=("--train-feedback")
elif [[ "$TRAIN_FEEDBACK" == "0" ]]; then
  BASE_ARGS+=("--no-train-feedback")
fi
if [[ "$MEMOS_STRUCTURED_FEEDBACK" == "1" ]]; then
  BASE_ARGS+=("--memos-structured-feedback")
elif [[ "$MEMOS_STRUCTURED_FEEDBACK" == "0" ]]; then
  BASE_ARGS+=("--no-memos-structured-feedback")
fi

if [[ -n "$MEMORY_PLUGIN" ]]; then
  BASE_ARGS+=("--memory-plugin" "$MEMORY_PLUGIN")
fi
if [[ -n "$MEMORY_PLUGIN_CONFIG" ]]; then
  BASE_ARGS+=("--memory-plugin-config" "$MEMORY_PLUGIN_CONFIG")
fi
if [[ -n "$PASS_AT" ]]; then
  BASE_ARGS+=("--pass-at" "$PASS_AT")
fi
BASE_ARGS+=("--results-dir" "$RESULTS_DIR")
if [[ -n "$ENV_FILE" ]]; then
  BASE_ARGS+=("--env" "$ENV_FILE")
fi
if [[ "$FORCE" -eq 1 ]]; then
  BASE_ARGS+=("--force")
fi

LOG_DIR="$PROJECT_DIR/results/agentbench/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${AGENT}-${MEMORY_PLUGIN:-custom}-${VERSION}-memory-train-backup-test.log"

echo "AgentBench memory train/backup/test run"
echo "  project:       $PROJECT_DIR"
echo "  python:        $PYTHON"
echo "  agent:         $AGENT"
echo "  memory plugin: ${MEMORY_PLUGIN:-$MEMORY_PLUGIN_CONFIG}"
echo "  version:       $VERSION"
echo "  trials:        $TRIALS"
echo "  test runs:     $TEST_RUNS"
echo "  train feedback:${TRAIN_FEEDBACK:-plugin-config}"
echo "  feedback timeout: ${FEEDBACK_TIMEOUT:-plugin-config}"
echo "  memos feedback timeout: ${MEMOS_FEEDBACK_TIMEOUT:-plugin-config}"
echo "  domains:       ${DOMAINS[*]}"
echo "  results:       $RESULTS_DIR"
echo "  log:           $LOG_FILE"
echo

FAILED=()
for domain in "${DOMAINS[@]}"; do
  domain="$(echo "$domain" | xargs)"
  [[ -z "$domain" ]] && continue

  echo "=== $(date --iso-8601=seconds) domain=$domain start ===" | tee -a "$LOG_FILE"
  set +e
  PYTHON="$PYTHON" ./scripts/run_agent_eval.sh \
    "${BASE_ARGS[@]}" \
    "--domain" "$domain" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "=== $(date --iso-8601=seconds) domain=$domain failed status=$status ===" | tee -a "$LOG_FILE"
    FAILED+=("$domain")
    if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then
      exit "$status"
    fi
  else
    echo "=== $(date --iso-8601=seconds) domain=$domain done ===" | tee -a "$LOG_FILE"
  fi
  echo | tee -a "$LOG_FILE"
done

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed domains: ${FAILED[*]}" >&2
  exit 1
fi

echo "All selected domains completed."
