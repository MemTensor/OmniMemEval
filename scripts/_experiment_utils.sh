#!/bin/bash
# Shared experiment snapshot & replay utilities.
# Source this file from run_*_eval.sh scripts.
#
# Required variables before sourcing:
#   PROJECT_DIR   — repository root
#   RESULTS_DIR   — directory for this experiment's results
#   SCRIPT_NAME   — e.g. "run_lme_eval.sh"
#   PARAMS_BLOCK  — multi-line string of "KEY=VALUE" lines to save/restore

# ─── Env file selection ──────────────────────────────────────────────────────
# Call: extract_env_arg "$@"; set -- "${_REMAINING_ARGS[@]}"
# Extracts --env <file> from argument list, exports MEMEVAL_ENV_FILE.
# In replay mode, --env may be inferred from the saved config when available.
# Must be called BEFORE try_replay.

_REMAINING_ARGS=()
_FROM_STEP_EXPLICIT=false
_TO_STEP_EXPLICIT=false
_REPLAY_REQUESTED=false

show_help_if_requested() {
    for arg in "$@"; do
        case "$arg" in
            -h|--help)
                usage
                exit 0
                ;;
        esac
    done
}

extract_env_arg() {
    MEMEVAL_ENV_FILE=""
    _FROM_STEP=1
    _TO_STEP=999
    _FROM_STEP_EXPLICIT=false
    _TO_STEP_EXPLICIT=false
    _REPLAY_REQUESTED=false
    _REMAINING_ARGS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --env)
                if [[ -z "${2:-}" ]]; then
                    echo "Error: --env requires a file path"
                    exit 1
                fi
                MEMEVAL_ENV_FILE="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
                if [[ ! -f "$MEMEVAL_ENV_FILE" ]]; then
                    echo "Error: env file not found: $MEMEVAL_ENV_FILE"
                    exit 1
                fi
                export MEMEVAL_ENV_FILE
                shift 2
                ;;
            --from-step)
                _FROM_STEP="${2:?--from-step requires a number}"
                _FROM_STEP_EXPLICIT=true
                shift 2
                ;;
            --to-step)
                _TO_STEP="${2:?--to-step requires a number}"
                _TO_STEP_EXPLICIT=true
                shift 2
                ;;
            --replay)
                _REPLAY_REQUESTED=true
                _REMAINING_ARGS+=("$1")
                shift
                ;;
            *)
                _REMAINING_ARGS+=("$1")
                shift
                ;;
        esac
    done
    if [[ -z "$MEMEVAL_ENV_FILE" ]]; then
        if [[ "$_REPLAY_REQUESTED" == "true" ]]; then
            return 0
        fi
        echo "Error: --env <file> is required (e.g. --env .env.mem0)"
        exit 1
    fi
    echo "Using env file: $MEMEVAL_ENV_FILE"
}

# ─── Argument validation helpers ─────────────────────────────────────────────

require_binary_flag() {
    local name="$1"
    local value="$2"
    if [[ "$value" != "0" && "$value" != "1" ]]; then
        echo "Error: $name must be 0 or 1, got: $value"
        exit 1
    fi
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $name must be a positive integer, got: $value"
        exit 1
    fi
}

require_nonnegative_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: $name must be a non-negative integer, got: $value"
        exit 1
    fi
}

require_nonnegative_seconds() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]]; then
        echo "Error: $name must be a non-negative number of seconds, got: $value"
        exit 1
    fi
}

is_positive_seconds() {
    local value="$1"
    awk -v value="$value" 'BEGIN { exit !(value + 0 > 0) }'
}

# ─── Pipeline progress tracking ──────────────────────────────────────────────

_PIPELINE_START=0
_PIPELINE_STEP=0
_PIPELINE_TOTAL=0
_STEP_TIMES=()

_fmt_duration() {
    local t=$1
    local h=$((t / 3600)) m=$(((t % 3600) / 60)) s=$((t % 60))
    if   (( h > 0 )); then printf "%dh%02dm%02ds" $h $m $s
    elif (( m > 0 )); then printf "%dm%02ds" $m $s
    else printf "%ds" $s
    fi
}

validate_step_range() {
    local total="$1"
    if ! [[ "$total" =~ ^[0-9]+$ ]] || (( total < 1 )); then
        echo "Error: pipeline total steps must be a positive integer, got: $total"
        exit 1
    fi
    if ! [[ "$_FROM_STEP" =~ ^[0-9]+$ && "$_TO_STEP" =~ ^[0-9]+$ ]]; then
        echo "Error: --from-step and --to-step must be positive integers"
        exit 1
    fi
    if [[ "$_TO_STEP_EXPLICIT" != "true" ]] && (( _TO_STEP > total )); then
        _TO_STEP=$total
    fi
    if (( _FROM_STEP < 1 || _TO_STEP < 1 || _FROM_STEP > total || _TO_STEP > total || _FROM_STEP > _TO_STEP )); then
        echo "Error: invalid step range ${_FROM_STEP}-${_TO_STEP}; valid ${SCRIPT_NAME:-pipeline} steps are 1-${total}"
        exit 1
    fi
}

pipeline_start() {
    _PIPELINE_TOTAL=$1
    validate_step_range "$_PIPELINE_TOTAL"
    _PIPELINE_STEP=0
    _PIPELINE_START=$(date +%s)
    _STEP_TIMES=()

    # When --from-step is explicitly set, clear step markers from that step onwards
    # so those steps will be forced to re-run
    if [[ "$_FROM_STEP_EXPLICIT" == "true" ]] && [[ -d "$RESULTS_DIR" ]]; then
        for (( _s = _FROM_STEP; _s <= _PIPELINE_TOTAL; _s++ )); do
            rm -f "$RESULTS_DIR/.step_${_s}_done"
        done
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    printf "║  %-64s║\n" "Pipeline: $SCRIPT_NAME"
    if (( _FROM_STEP > 1 || _TO_STEP < _PIPELINE_TOTAL )); then
        printf "║  %-64s║\n" "Steps: ${_FROM_STEP}-${_TO_STEP} of ${_PIPELINE_TOTAL}  |  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        printf "║  %-64s║\n" "Steps: $_PIPELINE_TOTAL  |  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

run_step() {
    local step_name="$1"; shift
    _PIPELINE_STEP=$((_PIPELINE_STEP + 1))

    if (( _PIPELINE_STEP < _FROM_STEP || _PIPELINE_STEP > _TO_STEP )); then
        echo ""
        echo "  [$_PIPELINE_STEP/$_PIPELINE_TOTAL] $step_name  ⏭ skipped"
        _STEP_TIMES+=("0")
        return 0
    fi

    # Auto-skip if this step already completed (marker file exists)
    local marker="$RESULTS_DIR/.step_${_PIPELINE_STEP}_done"
    if [[ -f "$marker" ]]; then
        echo ""
        echo "  [$_PIPELINE_STEP/$_PIPELINE_TOTAL] $step_name  ✅ already done, skipping  (use --from-step $_PIPELINE_STEP to force re-run)"
        _STEP_TIMES+=("0")
        return 0
    fi

    local step_start
    step_start=$(date +%s)

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  [$_PIPELINE_STEP/$_PIPELINE_TOTAL] $step_name"
    echo "────────────────────────────────────────────────────────────────────"

    local rc=0
    if "$@"; then
        rc=0
    else
        rc=$?
    fi
    local step_end
    step_end=$(date +%s)
    local step_elapsed=$((step_end - step_start))
    local pipeline_elapsed=$((step_end - _PIPELINE_START))
    _STEP_TIMES+=("$step_elapsed")

    if [ $rc -ne 0 ]; then
        echo ""
        echo "  ✗ FAILED after $(_fmt_duration $step_elapsed)"
        exit $rc
    fi

    # Mark step as completed
    touch "$marker"

    echo ""
    echo "  ✓ Step done in $(_fmt_duration $step_elapsed)  |  Pipeline: $_PIPELINE_STEP/$_PIPELINE_TOTAL  |  Elapsed: $(_fmt_duration $pipeline_elapsed)"
}

pipeline_summary() {
    local end_time
    end_time=$(date +%s)
    local total_elapsed=$((end_time - _PIPELINE_START))

    if (( _TO_STEP >= _PIPELINE_TOTAL )); then
        # Full pipeline completed — clear step markers so next run starts fresh.
        for (( _s = 1; _s <= _PIPELINE_TOTAL; _s++ )); do
            rm -f "$RESULTS_DIR/.step_${_s}_done"
        done
    else
        echo ""
        echo "Partial pipeline run completed through step $_TO_STEP; step markers are kept for resume."
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    printf "║  %-64s║\n" "Pipeline Complete!"
    printf "║  %-64s║\n" "Total time: $(_fmt_duration $total_elapsed)  |  Steps: $_PIPELINE_TOTAL"
    printf "║  %-64s║\n" "Results: $RESULTS_DIR"
    echo "╠══════════════════════════════════════════════════════════════════╣"
    printf "║  %-64s║\n" "Step breakdown:"
    for i in "${!_STEP_TIMES[@]}"; do
        local sn=$((i+1))
        if (( sn < _FROM_STEP || sn > _TO_STEP )); then
            printf "║    Step %d: %-58s║\n" "$sn" "skipped"
        else
            printf "║    Step %d: %-58s║\n" "$sn" "$(_fmt_duration ${_STEP_TIMES[$i]})"
        fi
    done
    echo "╠══════════════════════════════════════════════════════════════════╣"
    local _env_flag=" --env $(basename "$MEMEVAL_ENV_FILE")"
    printf "║  %-64s║\n" "Reproduce: ./scripts/$SCRIPT_NAME$_env_flag --replay $RESULTS_DIR"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

# ─── Replay mode ─────────────────────────────────────────────────────────────
# Call: try_replay "$@"
# Sets variables from saved config and returns 0, or does nothing and returns 1.
try_replay() {
    if [[ "${1:-}" != "--replay" ]]; then
        return 1
    fi

    local replay_dir="${2:?Usage: $SCRIPT_NAME --replay <results_dir>}"
    local config_file="$replay_dir/experiment_config.sh"
    if [[ ! -f "$config_file" ]]; then
        echo "Error: $config_file not found"
        exit 1
    fi

    echo "=== Replay mode: loading config from $replay_dir ==="

    # Remember the original VERSION before sourcing overrides it
    source "$config_file"
    local orig_version="$VERSION"

    if [[ -z "${MEMEVAL_ENV_FILE:-}" ]]; then
        local env_basename="${ENV_FILE_BASENAME:-}"
        local candidate_env=""
        if [[ -n "$env_basename" && -f "$PROJECT_DIR/$env_basename" ]]; then
            candidate_env="$PROJECT_DIR/$env_basename"
        fi
        if [[ -z "$candidate_env" ]]; then
            echo "Error: replay requires --env <file> with live credentials."
            echo "       snapshot_eval.env masks secrets and cannot be used as the runtime env."
            echo "       Use: ./scripts/$SCRIPT_NAME --env <file> --replay $replay_dir"
            exit 1
        fi
        MEMEVAL_ENV_FILE="$(cd "$(dirname "$candidate_env")" && pwd)/$(basename "$candidate_env")"
        export MEMEVAL_ENV_FILE
        echo "Using inferred env file: $MEMEVAL_ENV_FILE"
    fi

    # Strip any existing _replay_* suffix to avoid nesting, then append new one
    local base_version="${orig_version%%_replay_*}"
    VERSION="${base_version}_replay_$(date '+%Y%m%d_%H%M%S')"

    # Print all restored params (skip comments and internal vars)
    echo "  Loaded params:"
    grep -E '^[A-Z_]+=' "$config_file" | grep -v '^RUN_TIMESTAMP\|^git_' | while IFS= read -r line; do
        echo "    $line"
    done
    echo "  Original run: $RUN_TIMESTAMP"
    echo ""
    echo "  Replay VERSION: $VERSION  (original results will NOT be overwritten)"

    # Show .env diffs
    echo ""
    echo "─── env differences (snapshot vs current) ───"
    local snap_file="$replay_dir/snapshot_eval.env"
    local current_file="$MEMEVAL_ENV_FILE"
    if [[ -f "$snap_file" ]]; then
        if diff -u "$snap_file" "$current_file" > /dev/null 2>&1; then
            echo "  No differences."
        else
            diff -u --label "snapshot (snapshot_eval.env)" "$snap_file" \
                    --label "current (.env)" "$current_file" || true
        fi
    fi

    echo ""
    echo "Press Enter to continue or Ctrl+C to abort."
    read -r
    return 0
}

# ─── Save experiment config snapshot ─────────────────────────────────────────
# Call: save_experiment_config
# Uses: RESULTS_DIR, PROJECT_DIR, SCRIPT_NAME, PARAMS_BLOCK
save_experiment_config() {
    local dest="$RESULTS_DIR"
    local timestamp
    local env_basename
    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    env_basename="$(basename "$MEMEVAL_ENV_FILE")"

    mkdir -p "$dest"

    # 1) Script parameters (sourceable shell file)
    {
        echo "# Experiment configuration — auto-generated, do not edit"
        echo "# To reproduce: ./scripts/$SCRIPT_NAME --env $env_basename --replay $dest"
        echo "RUN_TIMESTAMP=\"$timestamp\""
        printf 'ENV_FILE_BASENAME=%q\n' "$env_basename"
        echo "$PARAMS_BLOCK"
    } > "$dest/experiment_config.sh"

    # 2) Git info
    {
        echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
        echo "git_branch=$(git -C "$PROJECT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
        echo "git_dirty=$(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    } >> "$dest/experiment_config.sh"

    # 3) Snapshot of env file (mask sensitive values)
    sed -E 's/(.*(_KEY|_SECRET|_TOKEN|_PASSWORD)=).*/\1***/' "$MEMEVAL_ENV_FILE" > "$dest/snapshot_eval.env"

    echo "Experiment config saved to $dest/"
    echo "  - experiment_config.sh  (script params, sourceable)"
    echo "  - snapshot_eval.env     (env copy)"
    echo ""
}
