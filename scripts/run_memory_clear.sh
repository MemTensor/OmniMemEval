#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/_experiment_utils.sh"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_memory_clear.sh --env <file> --lib <memory-lib> --version <name> [options]

Options:
  --lib <name>                 Memory product key.
  --version <name>             Evaluation version suffix to delete.
  --datasets <list|all>        Comma-separated: locomo,lme. Default: all.
  --dry-run                    Print target ids without deleting backend data.
  --yes                        Confirm destructive deletion. Required unless --dry-run is set.
  --env <file>                 Load environment variables from this file.
  -h, --help                   Show this help.
EOF
}

show_help_if_requested "$@"
extract_env_arg "$@"
set -- "${_REMAINING_ARGS[@]}"

exec python3 "$SCRIPT_DIR/cleanup_eval_memory.py" --env "$MEMEVAL_ENV_FILE" "$@"
