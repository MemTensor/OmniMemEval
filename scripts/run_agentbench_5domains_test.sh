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
DEFAULT_VERSION="all5_test_$(date +%Y%m%d_%H%M%S)"

AGENT="openclaw"
VERSION="$DEFAULT_VERSION"
TRIALS="1"
PASS_AT=""
PARALLEL="1"
RESULTS_DIR=""
REPORT_DIR=""
REPORT_NAME=""
PLUGIN_LABEL=""
ENV_FILE=""
FORCE=0
CONTINUE_ON_ERROR=0
DOMAINS=(
  "reasoning"
  "information_retrieval"
  "knowledge_work"
  "code_implementation"
  "software_engineering"
)

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_agentbench_5domains_test.sh [options]

Runs AgentBench's five domains in test_only mode:
  reasoning
  information_retrieval
  knowledge_work
  code_implementation
  software_engineering

Defaults:
  Python: python3, or PYTHON=/path/to/python
  Agent: openclaw
  Trials: 1
  Parallel: 1
  Protocol: test_only

Options:
  --agent NAME              Agent runtime name. Default: openclaw
  --version TAG             Result version tag. Default: all5_test_<timestamp>
  --trials N / --runs N     Run each task N times. Default: 1
  --pass-at N               Compute pass@n. Default: same as --trials
  --parallel N              Per-domain task parallelism. Default: 1
  --results-dir DIR         Override results directory
  --report-dir DIR          Override aggregate report directory. Default: <results-dir>/reports
  --report-name NAME        Override aggregate report file prefix. Default: <version>
  --plugin LABEL            Optional memory plugin label recorded in the report
  --env FILE                Extra env file passed to the runner
  --force                   Re-run completed trials
  --continue-on-error       Continue remaining domains after one fails
  --domains CSV             Override domain list, comma-separated
  -h, --help                Show this help

Environment:
  PYTHON=/path/to/python    Override Python executable

Example:
  ./scripts/run_agentbench_5domains_test.sh --version openclaw_full_test --trials 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      AGENT="$2"
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
    --report-dir)
      REPORT_DIR="$2"
      shift 2
      ;;
    --report-name)
      REPORT_NAME="$2"
      shift 2
      ;;
    --plugin)
      PLUGIN_LABEL="$2"
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

PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found or not executable: $PYTHON" >&2
  echo "Set PYTHON=/path/to/python or create the agentmem conda environment." >&2
  exit 1
fi

if [[ -z "$RESULTS_DIR" ]]; then
  RESULTS_DIR="$PROJECT_DIR/results/agentbench"
fi
if [[ -z "$REPORT_DIR" ]]; then
  REPORT_DIR="$RESULTS_DIR/reports"
fi
if [[ -z "$REPORT_NAME" ]]; then
  REPORT_NAME="$VERSION"
fi

BASE_ARGS=(
  "--agent" "$AGENT"
  "--protocol" "test_only"
  "--version" "$VERSION"
  "--trials" "$TRIALS"
  "--parallel" "$PARALLEL"
  "--results-dir" "$RESULTS_DIR"
)

if [[ -n "$PASS_AT" ]]; then
  BASE_ARGS+=("--pass-at" "$PASS_AT")
fi
if [[ -n "$ENV_FILE" ]]; then
  BASE_ARGS+=("--env" "$ENV_FILE")
fi
if [[ "$FORCE" -eq 1 ]]; then
  BASE_ARGS+=("--force")
fi

LOG_DIR="$PROJECT_DIR/results/agentbench/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${AGENT}-${VERSION}-5domains-test.log"

echo "AgentBench 5-domain test run"
echo "  project: $PROJECT_DIR"
echo "  python:  $PYTHON"
echo "  agent:   $AGENT"
echo "  version: $VERSION"
echo "  trials:  $TRIALS"
echo "  domains: ${DOMAINS[*]}"
echo "  results: $RESULTS_DIR"
echo "  report:  $REPORT_DIR/${REPORT_NAME}_report.{json,md}"
if [[ -n "$PLUGIN_LABEL" ]]; then
  echo "  plugin:  $PLUGIN_LABEL"
fi
echo "  log:     $LOG_FILE"
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
      echo "Stopping after failure. Re-run with --continue-on-error to keep going." >&2
      exit "$status"
    fi
  else
    echo "=== $(date --iso-8601=seconds) domain=$domain done ===" | tee -a "$LOG_FILE"
  fi
  echo | tee -a "$LOG_FILE"
done

echo "=== $(date --iso-8601=seconds) aggregate report start ===" | tee -a "$LOG_FILE"
PYTHONPATH="$PROJECT_DIR/scripts:${PYTHONPATH:-}" "$PYTHON" - \
  "$RESULTS_DIR" "$REPORT_DIR" "$AGENT" "$VERSION" "$TRIALS" "$PARALLEL" "$REPORT_NAME" "$PLUGIN_LABEL" \
  "${DOMAINS[@]}" <<'PY' 2>&1 | tee -a "$LOG_FILE"
import sys
from pathlib import Path

from agentbench.report import generate_all_domains_report

results_root = Path(sys.argv[1])
report_dir = Path(sys.argv[2])
agent = sys.argv[3]
version = sys.argv[4]
trials = int(sys.argv[5])
parallel = int(sys.argv[6])
report_name = sys.argv[7]
plugin = sys.argv[8] or None
domains = [item.strip() for item in sys.argv[9:] if item.strip()]

generate_all_domains_report(
    results_root=results_root,
    report_dir=report_dir,
    agent=agent,
    version=version,
    domains=domains,
    trials=trials,
    parallel=parallel,
    report_name=report_name,
    plugin=plugin,
)
print(f"Report written to {report_dir / (report_name + '_report.md')}")
print(f"Machine-readable report written to {report_dir / (report_name + '_report.json')}")
PY
report_status=${PIPESTATUS[0]}
if [[ "$report_status" -ne 0 ]]; then
  echo "Aggregate report failed status=$report_status" | tee -a "$LOG_FILE"
  exit "$report_status"
fi
echo "=== $(date --iso-8601=seconds) aggregate report done ===" | tee -a "$LOG_FILE"

echo "All five domain test runs completed." | tee -a "$LOG_FILE"
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Completed with failures: ${FAILED[*]}" | tee -a "$LOG_FILE"
  exit 1
fi
