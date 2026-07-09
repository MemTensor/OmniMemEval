#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DEFAULT_PYTHON="/root/miniconda3/envs/agentmem/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    DEFAULT_PYTHON="$CONDA_PREFIX/bin/python"
  else
    DEFAULT_PYTHON="$(command -v python || command -v python3 || true)"
  fi
fi

AGENT="openclaw"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
MEMORY_PLUGIN_CONFIG="$PROJECT_DIR/configs/agentbench/memory_plugins/memos.yaml"
VERSION="memos_5domain_1each_smoke_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJECT_DIR/results/agentbench"
TRAIN_SPLIT="train"
TEST_SPLIT="test"
TASKS_PER_SPLIT="1"
TRIALS="1"
TEST_RUNS="1"
PARALLEL="1"
MAX_RETRIES="2"
SETTLE_SECONDS="0"
AGENT_TIMEOUT=""
VERIFY_TIMEOUT=""
ENV_FILE=""
FORCE=0
CONTINUE_ON_ERROR=0
TRAIN_FEEDBACK=""
MEMOS_STRUCTURED_FEEDBACK=""
MEMOS_FEEDBACK_TIMEOUT=""
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
  ./scripts/run_agentbench_memos_5domain_1each_smoke.sh [options]

Runs the MemOS memory_train_backup_test protocol across five AgentBench domains,
with one train item and one test item per domain.

Defaults are smoke-test oriented:
  --train-split train
  --test-split test
  --tasks-per-split 1
  --parallel 1
  --settle-seconds 0
  PYTHON=/root/miniconda3/envs/agentmem/bin/python when available

Options:
  --agent NAME                 Agent runtime name. Default: openclaw
  --python FILE                Python executable. Default: PYTHON env or agentmem env
  --memory-plugin-config FILE  MemOS lifecycle YAML. Default: configs/.../memos.yaml
  --version TAG                Result version tag. Default: timestamped
  --results-dir DIR            Result directory. Default: results/agentbench
  --train-split SPLIT          Train split selector. Default: train
  --test-split SPLIT           Test split selector. Default: test
  --tasks-per-split N          Pick first N task ids from each split. Default: 1
  --trials N / --runs N        Trials per task. Default: 1
  --test-runs N                Test repetitions. Default: 1
  --parallel N                 Per-domain task parallelism. Default: 1
  --max-retries N              Agent retries per task. Default: 2
  --settle-seconds N           Override memory settle wait in a temporary config. Default: 0
  --agent-timeout N            Override domain agent_timeout in a temporary config
  --verify-timeout N           Override domain verify_timeout in a temporary config
  --env FILE                   Extra env file passed to the runner
  --force                      Re-run completed trials
  --continue-on-error          Continue remaining domains after one domain fails
  --train-feedback             Force train feedback turn
  --no-train-feedback          Disable train feedback turn
  --memos-structured-feedback  Force MemOS structured feedback submit
  --no-memos-structured-feedback
  --memos-feedback-timeout N   Override MemOS feedback timeout
  --domains CSV                Override domains, comma-separated
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
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
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --train-split)
      TRAIN_SPLIT="$2"
      shift 2
      ;;
    --test-split)
      TEST_SPLIT="$2"
      shift 2
      ;;
    --tasks-per-split)
      TASKS_PER_SPLIT="$2"
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
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --settle-seconds)
      SETTLE_SECONDS="$2"
      shift 2
      ;;
    --agent-timeout)
      AGENT_TIMEOUT="$2"
      shift 2
      ;;
    --verify-timeout)
      VERIFY_TIMEOUT="$2"
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
    --train-feedback)
      TRAIN_FEEDBACK=1
      shift
      ;;
    --no-train-feedback)
      TRAIN_FEEDBACK=0
      shift
      ;;
    --memos-structured-feedback)
      MEMOS_STRUCTURED_FEEDBACK=1
      shift
      ;;
    --no-memos-structured-feedback)
      MEMOS_STRUCTURED_FEEDBACK=0
      shift
      ;;
    --memos-feedback-timeout)
      MEMOS_FEEDBACK_TIMEOUT="$2"
      shift 2
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

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$MEMORY_PLUGIN_CONFIG" ]]; then
  echo "ERROR: memory plugin config not found: $MEMORY_PLUGIN_CONFIG" >&2
  exit 2
fi

LOG_DIR="$RESULTS_DIR/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/openclaw-memos-${VERSION}-5domain-1each-smoke.log"
RUN_CONFIG_DIR="$LOG_DIR/openclaw-memos-${VERSION}-5domain-1each-smoke"
mkdir -p "$RUN_CONFIG_DIR"
RUN_MEMORY_PLUGIN_CONFIG="$RUN_CONFIG_DIR/memos.lifecycle.yaml"
TASK_MANIFEST="$RUN_CONFIG_DIR/tasks.jsonl"
: > "$TASK_MANIFEST"

"$PYTHON" - "$MEMORY_PLUGIN_CONFIG" "$RUN_MEMORY_PLUGIN_CONFIG" "$SETTLE_SECONDS" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
settle_seconds = int(sys.argv[3])

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["settle_seconds"] = settle_seconds
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

BASE_ARGS=(
  "--agent" "$AGENT"
  "--protocol" "memory_train_backup_test"
  "--memory-plugin-config" "$RUN_MEMORY_PLUGIN_CONFIG"
  "--version" "$VERSION"
  "--trials" "$TRIALS"
  "--test-runs" "$TEST_RUNS"
  "--train-split" "$TRAIN_SPLIT"
  "--test-split" "$TEST_SPLIT"
  "--parallel" "$PARALLEL"
  "--max-retries" "$MAX_RETRIES"
  "--results-dir" "$RESULTS_DIR"
)

if [[ -n "$ENV_FILE" ]]; then
  BASE_ARGS+=("--env" "$ENV_FILE")
fi
if [[ "$FORCE" == "1" ]]; then
  BASE_ARGS+=("--force")
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
if [[ -n "$MEMOS_FEEDBACK_TIMEOUT" ]]; then
  BASE_ARGS+=("--memos-feedback-timeout" "$MEMOS_FEEDBACK_TIMEOUT")
fi

{
  echo "Started: $(date --iso-8601=seconds)"
  echo "Project: $PROJECT_DIR"
  echo "Python: $PYTHON"
  echo "Version: $VERSION"
  echo "Domains: ${DOMAINS[*]}"
  echo "Train split: $TRAIN_SPLIT"
  echo "Test split: $TEST_SPLIT"
  echo "Tasks per split: $TASKS_PER_SPLIT"
  echo "Parallel: $PARALLEL"
  echo "Max retries: $MAX_RETRIES"
  echo "Settle seconds: $SETTLE_SECONDS"
  if [[ -n "$AGENT_TIMEOUT" ]]; then
    echo "Agent timeout override: $AGENT_TIMEOUT"
  fi
  if [[ -n "$VERIFY_TIMEOUT" ]]; then
    echo "Verify timeout override: $VERIFY_TIMEOUT"
  fi
  echo "Lifecycle config: $RUN_MEMORY_PLUGIN_CONFIG"
  echo "Task manifest: $TASK_MANIFEST"
  echo
} | tee "$LOG_FILE"

resolve_domain_tasks() {
  "$PYTHON" - "$PROJECT_DIR" "$1" "$TRAIN_SPLIT" "$TEST_SPLIT" "$TASKS_PER_SPLIT" <<'PY'
import json
import sys
from pathlib import Path

import yaml

project = Path(sys.argv[1])
domain = sys.argv[2]
train_split = sys.argv[3]
test_split = sys.argv[4]
limit = int(sys.argv[5])

cfg_path = project / "configs" / "agentbench" / "domains" / f"{domain}.yaml"
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project / path
    return path

def first_from_split_file(split_file: Path, split: str) -> list[str]:
    raw = json.loads(split_file.read_text(encoding="utf-8"))
    if split in raw and isinstance(raw[split], list):
        return [str(item) for item in raw[split][:limit]]
    values: list[str] = []
    clusters = raw.get("clusters") if isinstance(raw, dict) else None
    if isinstance(clusters, dict):
        for cluster in clusters.values():
            if isinstance(cluster, dict):
                values.extend(str(item) for item in cluster.get(split, []))
            elif isinstance(cluster, list) and split == "all":
                values.extend(str(item) for item in cluster)
            if len(values) >= limit:
                break
    if not values:
        raise SystemExit(f"no task ids found for domain={domain} split={split}")
    return values[:limit]

def first_from_reasoning(split: str) -> list[str]:
    data_dir = resolve_path(cfg.get("data_dir"))
    path = data_dir / f"{split}.jsonl"
    ids: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if "_idx" in row:
                ids.append(f"omni_{row['_idx']}")
            elif "id" in row:
                ids.append(str(row["id"]))
            if len(ids) >= limit:
                break
    if not ids:
        raise SystemExit(f"no reasoning task ids found for split={split}")
    return ids

if domain == "reasoning":
    train_ids = first_from_reasoning(train_split)
    test_ids = first_from_reasoning(test_split)
else:
    split_file = resolve_path(cfg.get("split_file"))
    if not split_file or not split_file.exists():
        raise SystemExit(f"domain={domain} has no split_file for split-aware smoke selection")
    train_ids = first_from_split_file(split_file, train_split)
    test_ids = first_from_split_file(split_file, test_split)

print(",".join(train_ids))
print(",".join(test_ids))
PY
}

FAILED=()
for domain in "${DOMAINS[@]}"; do
  echo "===== DOMAIN: $domain =====" | tee -a "$LOG_FILE"
  TASK_SELECTION_FILE="$(mktemp "${TMPDIR:-/tmp}/memos-1each-${domain}.XXXXXX.tasks")"
  if resolve_domain_tasks "$domain" > "$TASK_SELECTION_FILE" 2>> "$LOG_FILE"; then
    :
  else
    status=$?
    rm -f "$TASK_SELECTION_FILE"
    echo "===== DOMAIN FAILED: $domain task selection rc=$status =====" | tee -a "$LOG_FILE"
    FAILED+=("$domain")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$status"
    fi
    echo | tee -a "$LOG_FILE"
    continue
  fi
  mapfile -t domain_tasks < "$TASK_SELECTION_FILE"
  rm -f "$TASK_SELECTION_FILE"
  if [[ "${#domain_tasks[@]}" -lt 2 || -z "${domain_tasks[0]}" || -z "${domain_tasks[1]}" ]]; then
    echo "===== DOMAIN FAILED: $domain task selection returned incomplete output =====" | tee -a "$LOG_FILE"
    FAILED+=("$domain")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
    echo | tee -a "$LOG_FILE"
    continue
  fi
  TRAIN_TASK="${domain_tasks[0]}"
  TEST_TASK="${domain_tasks[1]}"
  printf '{"domain":"%s","train_split":"%s","train_task":"%s","test_split":"%s","test_task":"%s"}\n' \
    "$domain" "$TRAIN_SPLIT" "$TRAIN_TASK" "$TEST_SPLIT" "$TEST_TASK" >> "$TASK_MANIFEST"
  echo "Selected train task(s): $TRAIN_TASK" | tee -a "$LOG_FILE"
  echo "Selected test task(s): $TEST_TASK" | tee -a "$LOG_FILE"

  DOMAIN_ARGS=("--domain" "$domain" "--train-task" "$TRAIN_TASK" "--test-task" "$TEST_TASK")
  if [[ -n "$AGENT_TIMEOUT" || -n "$VERIFY_TIMEOUT" ]]; then
    RUN_DOMAIN_CONFIG="$RUN_CONFIG_DIR/${domain}.yaml"
    "$PYTHON" - "$PROJECT_DIR" "$domain" "$RUN_DOMAIN_CONFIG" "$AGENT_TIMEOUT" "$VERIFY_TIMEOUT" <<'PY'
import sys
from pathlib import Path

import yaml

project = Path(sys.argv[1])
domain = sys.argv[2]
dst = Path(sys.argv[3])
agent_timeout = sys.argv[4]
verify_timeout = sys.argv[5]

src = project / "configs" / "agentbench" / "domains" / f"{domain}.yaml"
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
if agent_timeout:
    cfg["agent_timeout"] = int(agent_timeout)
if verify_timeout:
    cfg["verify_timeout"] = int(verify_timeout)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
    DOMAIN_ARGS+=("--domain-config" "$RUN_DOMAIN_CONFIG")
  fi

  if "$PYTHON" scripts/agentbench/run_agent_eval.py "${BASE_ARGS[@]}" "${DOMAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    echo "===== DOMAIN OK: $domain =====" | tee -a "$LOG_FILE"
  else
    status=${PIPESTATUS[0]}
    echo "===== DOMAIN FAILED: $domain rc=$status =====" | tee -a "$LOG_FILE"
    FAILED+=("$domain")
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$status"
    fi
  fi
  echo | tee -a "$LOG_FILE"
done

echo "Finished: $(date --iso-8601=seconds)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "Failed domains: ${FAILED[*]}" | tee -a "$LOG_FILE"
  exit 1
fi

echo "All domains completed." | tee -a "$LOG_FILE"
