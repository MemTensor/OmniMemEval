#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_agent_eval.sh [options]

Examples:
  ./scripts/run_agent_eval.sh \
    --agent openclaw \
    --domain reasoning \
    --protocol test_only \
    --version smoke \
    --task omni_1

  ./scripts/run_agent_eval.sh \
    --agent openclaw \
    --domain reasoning \
    --protocol train_then_test \
    --version plugin_v1

Options are passed through to scripts/agentbench/run_agent_eval.py.

Common options:
  --agent-config YAML          Optional. Defaults to configs/agentbench/agents/<agent>.yaml.
  --profile NAME               Runtime profile, such as plain or memos. Defaults to plain,
                               or to --memory-plugin for memory lifecycle protocols.
  --profile-config YAML        Explicit runtime profile YAML.
  --trials N / --runs N       Run each task N times and report per-trial plus average metrics.
  --pass-at N                 Compute pass@n from the first N trials. Defaults to --trials.
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

PYTHON="${PYTHON:-python}"
PYTHONPATH="$PROJECT_DIR/scripts:${PYTHONPATH:-}" \
    "$PYTHON" "$PROJECT_DIR/scripts/agentbench/run_agent_eval.py" "$@"
