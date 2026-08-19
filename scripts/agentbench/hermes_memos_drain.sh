#!/usr/bin/env bash
# Keep one run-scoped Hermes bridge alive until MemOS has durably finished all
# episode recovery, evolution, and embedding work. The bridge stdin must remain
# open: a plain `</dev/null` makes the bridge shut down immediately after init.
set -euo pipefail

plugin="${MEMOS_PLUGIN_HOME:?run-scoped MEMOS_PLUGIN_HOME is required}"
db="${MEMOS_DB:?run-scoped MEMOS_DB is required}"
timeout_seconds="${MEMOS_RECONCILE_TIMEOUT:-${MEMOS_FINALIZE_TIMEOUT:-1800}}"
term_timeout="${MEMOS_DAEMON_TERM_TIMEOUT:-300}"
quiet_polls="${MEMOS_RECONCILE_QUIET_POLLS:-3}"

case "$timeout_seconds" in ''|*[!0-9]*) echo "invalid Hermes MemOS reconcile timeout: $timeout_seconds" >&2; exit 1;; esac
case "$term_timeout" in ''|*[!0-9]*) echo "invalid Hermes MemOS terminate timeout: $term_timeout" >&2; exit 1;; esac
case "$quiet_polls" in ''|*[!0-9]*) echo "invalid Hermes MemOS quiet poll count: $quiet_polls" >&2; exit 1;; esac
[ "$timeout_seconds" -gt 0 ] || { echo "Hermes MemOS reconcile timeout must be positive" >&2; exit 1; }
[ "$quiet_polls" -gt 0 ] || { echo "Hermes MemOS quiet poll count must be positive" >&2; exit 1; }
command -v sqlite3 >/dev/null || { echo "sqlite3 is required to drain Hermes MemOS" >&2; exit 1; }
command -v node >/dev/null || { echo "node is required to drain Hermes MemOS" >&2; exit 1; }
[ -s "$db" ] || { echo "Hermes MemOS DB is missing or empty: $db" >&2; exit 1; }

required="$(sqlite3 -cmd '.timeout 30000' "$db" "SELECT count(*) FROM sqlite_master WHERE type='table' AND name IN ('episodes','traces','embedding_retry_queue','evolution_jobs');")"
[ "$required" = "4" ] || { echo "Hermes MemOS DB is missing required pipeline tables" >&2; exit 1; }

bridge="$plugin/dist/bridge.cjs"
[ -f "$bridge" ] || { echo "Hermes MemOS reconciliation bridge is missing: $bridge" >&2; exit 1; }
mkdir -p "$plugin/logs" "$plugin/daemon"
fifo="$plugin/daemon/reconcile-stdin.$$"
log="$plugin/logs/settle-reconcile.log"
bridge_pid=""

runtime_pids() {
  local proc pid exe args
  for proc in /proc/[0-9]*; do
    pid="${proc##*/}"
    [ "$pid" != "$$" ] && [ "$pid" != "$BASHPID" ] && [ "$pid" != "$PPID" ] || continue
    [ -r "$proc/cmdline" ] || continue
    # A lifecycle shell's `sh -c` argv contains the full heredoc, including
    # the text "bridge.cjs --agent=hermes". Require the actual executable to
    # be Node so matching can never terminate this helper or an ancestor shell.
    exe="$(readlink -f "$proc/exe" 2>/dev/null || true)"
    [ "${exe##*/}" = "node" ] || continue
    args="$(tr '\0' ' ' 2>/dev/null <"$proc/cmdline" || true)"
    case "$args" in
      *runtime-daemon.js*--agent=hermes*|*runtime-stdio-proxy.js*--agent=hermes*|*bridge.cjs*--agent=hermes*|*bridge.cts*--agent=hermes*) ;;
      *) continue;;
    esac
    grep -zFxq -- "MEMOS_PLUGIN_HOME=$plugin" "$proc/environ" 2>/dev/null || continue
    printf '%s\n' "$pid"
  done
}

stop_existing_runtime() {
  local deadline pid
  for pid in $(runtime_pids); do kill -TERM "$pid" 2>/dev/null || true; done
  deadline=$((SECONDS + term_timeout))
  while [ "$SECONDS" -lt "$deadline" ]; do
    [ -z "$(runtime_pids)" ] && return 0
    sleep 1
  done
  for pid in $(runtime_pids); do kill -KILL "$pid" 2>/dev/null || true; done
  [ -z "$(runtime_pids)" ] || {
    echo "run-scoped Hermes bridges survived reconciliation handoff" >&2
    return 1
  }
}

stop_bridge() {
  local deadline
  if [ -n "$bridge_pid" ] && kill -0 "$bridge_pid" 2>/dev/null; then
    kill -TERM "$bridge_pid" 2>/dev/null || true
    deadline=$((SECONDS + term_timeout))
    while kill -0 "$bridge_pid" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
      sleep 1
    done
    if kill -0 "$bridge_pid" 2>/dev/null; then
      kill -KILL "$bridge_pid" 2>/dev/null || true
    fi
  fi
  [ -z "$bridge_pid" ] || wait "$bridge_pid" 2>/dev/null || true
  exec 9>&- 2>/dev/null || true
  rm -f "$fifo"
}
trap stop_bridge EXIT INT TERM

# The restore stage may have a run-scoped viewer daemon. Hand ownership to one
# drain bridge before polling so the evaluation never runs duplicate workers.
stop_existing_runtime
rm -f "$fifo"
mkfifo -m 600 "$fifo"
# Opening both ends in this shell prevents an EOF while the worker drains. The
# descriptor is never written to and is closed by stop_bridge.
exec 9<>"$fifo"
node "$bridge" --agent=hermes --no-viewer <"$fifo" >>"$log" 2>&1 &
bridge_pid=$!

deadline=$((SECONDS + timeout_seconds))
stable=0
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! kill -0 "$bridge_pid" 2>/dev/null; then
    echo "Hermes MemOS reconciliation bridge exited before the pipeline became idle" >&2
    tail -n 80 "$log" >&2 || true
    exit 1
  fi

  evolution_active="$(sqlite3 -cmd '.timeout 30000' "$db" "SELECT count(*) FROM evolution_jobs WHERE status IN ('queued','leased','failed');")"
  evolution_dead="$(sqlite3 -cmd '.timeout 30000' "$db" "SELECT count(*) FROM evolution_jobs WHERE status='dead_letter';")"
  embedding_active="$(sqlite3 -cmd '.timeout 30000' "$db" "SELECT count(*) FROM embedding_retry_queue WHERE status IN ('pending','in_progress');")"
  embedding_failed="$(sqlite3 -cmd '.timeout 30000' "$db" "SELECT count(*) FROM embedding_retry_queue WHERE status='failed';")"
  # A recent empty topic is intentionally kept open across clean session
  # closes/restarts so a later related turn can resume it.  It is quiescent,
  # not unfinished background work.  Open episodes carrying traces/reward are
  # still blocking: startup recovery must close and evolve those first.
  blocking_open_episodes="$(sqlite3 -cmd '.timeout 30000' "$db" "
    SELECT count(*)
      FROM episodes
     WHERE status='open'
       AND (
         json_array_length(CASE WHEN json_valid(trace_ids_json) THEN trace_ids_json ELSE '[]' END) > 0
         OR r_task IS NOT NULL
         OR COALESCE(
              CASE WHEN json_valid(meta_json) THEN json_extract(meta_json, '$.topicState') END,
              ''
            ) NOT IN ('paused', 'interrupted')
       );
  ")"

  failures=$((evolution_dead + embedding_failed))
  if [ "$failures" -gt 0 ]; then
    echo "Hermes MemOS has $failures terminal background failures" >&2
    sqlite3 -cmd '.timeout 30000' "$db" "
      SELECT 'evolution', id, job_type, attempts, max_attempts, COALESCE(last_error, '')
        FROM evolution_jobs WHERE status='dead_letter';
      SELECT 'embedding', id, target_kind, attempts, max_attempts, COALESCE(last_error, '')
        FROM embedding_retry_queue WHERE status='failed';
    " >&2
    exit 1
  fi

  if [ "$evolution_active" -eq 0 ] && [ "$embedding_active" -eq 0 ] && [ "$blocking_open_episodes" -eq 0 ]; then
    stable=$((stable + 1))
    [ "$stable" -ge "$quiet_polls" ] && break
  else
    stable=0
  fi
  sleep 1
done

[ "$stable" -ge "$quiet_polls" ] || {
  echo "Hermes MemOS pipeline did not become idle within ${timeout_seconds}s: evolution_active=$evolution_active embedding_active=$embedding_active blocking_open_episodes=$blocking_open_episodes" >&2
  exit 1
}
test "$(sqlite3 "$db" 'PRAGMA quick_check;')" = "ok"

# A graceful SIGTERM closes the bridge transport/core only after the queue is
# already stable, so shutdown cannot abandon a live model or embedding call.
stop_bridge
bridge_pid=""
shutdown_deadline=$((SECONDS + term_timeout))
while [ "$SECONDS" -lt "$shutdown_deadline" ]; do
  [ -z "$(runtime_pids)" ] && break
  sleep 1
done
[ -z "$(runtime_pids)" ] || {
  echo "Hermes MemOS shared runtime did not stop after a clean drain" >&2
  exit 1
}

# Failed or timed-out attempts can open a topic before producing any capture.
# Once every writer has stopped, discard only episodes that contain no trace
# and no task reward.  They carry no usable memory and must not pollute retry
# session accounting or the training backup.
sqlite3 -cmd '.timeout 30000' "$db" "
  DELETE FROM episodes
   WHERE trace_ids_json = '[]'
     AND r_task IS NULL;
"
test "$(sqlite3 "$db" 'PRAGMA quick_check;')" = "ok"
trap - EXIT INT TERM
