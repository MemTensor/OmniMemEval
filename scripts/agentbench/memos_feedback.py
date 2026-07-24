from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

from agentbench.feedback import FEEDBACK_PREFIX
from agentbench.session import SessionSpec
from agentbench.session_capture import build_task_feedback_turns

MANUAL_MEMOS_SOURCE = "omnimemeval_agentbench_feedback"


def _openclaw_runtime_socket_path(memos_home: Path) -> Path:
    """Mirror the plugin's short, run-home-scoped Unix socket path."""
    root = str(memos_home.expanduser().resolve())
    digest = hashlib.sha256(root.encode()).hexdigest()[:24]
    uid = os.getuid() if hasattr(os, "getuid") else "nouid"
    return Path(tempfile.gettempdir()) / f"memos-openclaw-{uid}-{digest}.sock"


class SharedOpenClawRuntimeClient:
    """Small JSON-RPC client for the existing run-scoped MemOS owner.

    This client deliberately cannot launch ``bridge.mjs`` or a second
    MemoryCore. A missing owner socket is a technical failure: silently
    falling back to another process would reintroduce concurrent evolution
    against the same SQLite database.
    """

    def __init__(self, socket_path: Path, *, connect_timeout: float = 5.0):
        self.socket_path = Path(socket_path)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._reader = None
        self._next_id = 1
        self._lock = threading.Lock()
        deadline = time.monotonic() + max(0.0, connect_timeout)
        while True:
            try:
                self._socket.connect(str(self.socket_path))
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    self._socket.close()
                    raise RuntimeError(
                        f"MemOS shared runtime socket unavailable: {self.socket_path}: {exc}"
                    ) from exc
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        self._reader = self._socket.makefile("r", encoding="utf-8")

    def request(self, method: str, params=None, *, timeout: float = 180.0):
        with self._lock:
            if self._reader is None:
                raise RuntimeError("MemOS shared runtime client is closed")
            request_id = self._next_id
            self._next_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            self._socket.settimeout(max(0.001, float(timeout)))
            try:
                self._socket.sendall((json.dumps(payload) + "\n").encode())
                line = self._reader.readline()
            except (OSError, TimeoutError) as exc:
                self.close()
                raise RuntimeError(
                    f"MemOS shared runtime RPC {method} failed: {exc}"
                ) from exc
            if not line:
                self.close()
                raise RuntimeError(
                    f"MemOS shared runtime closed before replying to {method}"
                )
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise RuntimeError(
                    f"MemOS shared runtime returned invalid JSON for {method}"
                ) from exc
            if response.get("id") != request_id:
                self.close()
                raise RuntimeError(
                    f"MemOS shared runtime response id mismatch for {method}"
                )
            error = response.get("error")
            if error:
                raise RuntimeError(
                    f"MemOS shared runtime RPC {method} failed: "
                    f"{error.get('message', error)}"
                )
            return response.get("result")

    def close(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        try:
            self._socket.close()
        except OSError:
            pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _feedback_polarity(verifier_result: dict) -> str:
    try:
        reward = float(verifier_result.get("reward", 0.0))
    except (TypeError, ValueError):
        reward = 0.0
    if reward > 0:
        return "positive"
    if reward < 0:
        return "negative"
    if verifier_result.get("feedback") or verifier_result.get("error"):
        return "negative"
    return "neutral"


def _feedback_magnitude(verifier_result: dict) -> float:
    try:
        reward = abs(float(verifier_result.get("reward", 0.0)))
    except (TypeError, ValueError):
        reward = 0.0
    return max(0.2, min(1.0, reward or 1.0))


def _memos_namespace() -> dict:
    return {
        "agentKind": "openclaw",
        "profileId": "main",
    }


def _openclaw_home_from_session_file(session_file: Path | None) -> Path:
    if session_file:
        for parent in session_file.resolve().parents:
            if parent.name == ".openclaw":
                return parent
    return Path(os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))).expanduser()


def _memos_db_path(openclaw_home: Path) -> Path:
    return openclaw_home / "memos-plugin" / "data" / "memos.db"


def _memos_runtime_home(openclaw_home: Path) -> Path:
    configured = os.environ.get("MEMOS_PLUGIN_HOME") or os.environ.get("MEMOS_HOME")
    return Path(configured).expanduser() if configured else openclaw_home / "memos-plugin"


def _connect_shared_openclaw_runtime(
    openclaw_home: Path,
    *,
    timeout: float,
) -> SharedOpenClawRuntimeClient:
    runtime_home = _memos_runtime_home(openclaw_home).resolve()
    client = SharedOpenClawRuntimeClient(
        _openclaw_runtime_socket_path(runtime_home),
        connect_timeout=min(5.0, max(0.1, timeout)),
    )
    try:
        health = client.request("core.health", None, timeout=min(5.0, max(0.1, timeout)))
        health_data = health if isinstance(health, dict) else {}
        actual_db = Path(str(health_data.get("paths", {}).get("db", ""))).resolve()
        expected_db = (runtime_home / "data" / "memos.db").resolve()
        if (
            not isinstance(health, dict)
            or health.get("ok") is not True
            or health.get("agent") != "openclaw"
            or actual_db != expected_db
        ):
            raise RuntimeError(
                "MemOS shared runtime identity mismatch: "
                f"agent={health_data.get('agent')} "
                f"db={actual_db} expected={expected_db}"
            )
        return client
    except BaseException:
        client.close()
        raise


def _memos_plugin_root(openclaw_home: Path) -> Path:
    candidates = []
    if os.environ.get("MEMOS_PLUGIN_ROOT"):
        candidates.append(Path(os.environ["MEMOS_PLUGIN_ROOT"]).expanduser())
    candidates.extend([
        openclaw_home / "extensions" / "memos-local-plugin",
        Path.home() / ".openclaw" / "extensions" / "memos-local-plugin",
        Path("/root/gyh/MemOS/apps/memos-local-plugin"),
    ])
    for candidate in candidates:
        if (candidate / "bridge.cts").exists():
            return candidate.resolve()
    checked = ", ".join(str(item) for item in candidates)
    raise RuntimeError(f"Cannot locate memos-local-plugin bridge.cts; checked: {checked}")


def _text_from_content(content, field: str = "text") -> str:
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == field and item.get(field):
            parts.append(str(item[field]))
    return "\n".join(parts)


def _iter_session_records(session_file: Path):
    buffer = ""
    try:
        with session_file.open() as f:
            for line in f:
                buffer += line.rstrip("\n")
                try:
                    yield json.loads(buffer)
                    buffer = ""
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def _extract_turns_from_session(session_file: Path | None) -> list[dict]:
    turns = []
    current = None
    tool_calls_by_id = {}
    if not session_file or not session_file.exists():
        return turns

    for rec in _iter_session_records(session_file):
        if rec.get("type") != "message":
            continue
        msg = rec.get("message") or {}
        role = msg.get("role")
        if role == "user":
            if current is not None:
                turns.append(current)
            current = {
                "userText": _text_from_content(msg.get("content", [])),
                "agentText": "",
                "toolCalls": [],
            }
            tool_calls_by_id = {}
            continue
        if current is None:
            continue
        if role == "assistant":
            text = _text_from_content(msg.get("content", []))
            if text:
                current["agentText"] = text
            for item in msg.get("content", []):
                if not isinstance(item, dict) or item.get("type") != "toolCall":
                    continue
                tool_call = {
                    "name": str(item.get("name") or ""),
                    "input": item.get("arguments") if item.get("arguments") is not None else {},
                    "output": "",
                    "errorCode": None,
                    "toolCallId": item.get("id"),
                    "startedAt": msg.get("timestamp") or _now_ms(),
                    "endedAt": msg.get("timestamp") or _now_ms(),
                }
                current["toolCalls"].append(tool_call)
                if item.get("id"):
                    tool_calls_by_id[item["id"]] = tool_call
        elif role == "toolResult":
            tool_call = tool_calls_by_id.get(msg.get("toolCallId"))
            if tool_call is None:
                tool_call = {
                    "name": str(msg.get("toolName") or "tool"),
                    "input": {},
                    "output": "",
                    "errorCode": None,
                    "toolCallId": msg.get("toolCallId"),
                    "startedAt": msg.get("timestamp") or _now_ms(),
                    "endedAt": msg.get("timestamp") or _now_ms(),
                }
                current["toolCalls"].append(tool_call)
            tool_call["output"] = _text_from_content(msg.get("content", [])) or msg.get("details", {}).get("aggregated", "")
            tool_call["endedAt"] = msg.get("timestamp") or _now_ms()
            if msg.get("isError"):
                tool_call["errorCode"] = "tool_error"
    if current is not None:
        turns.append(current)
    return turns


def _episode_id_from_turn_start(result: dict | None) -> str:
    if not isinstance(result, dict):
        return ""
    episode_id = result.get("episodeId")
    if isinstance(episode_id, str):
        return episode_id
    query = result.get("query")
    if isinstance(query, dict):
        episode_id = query.get("episodeId")
        if isinstance(episode_id, str):
            return episode_id
    return ""


def _trace_id_from_turn_end(result: dict | None) -> str:
    if not isinstance(result, dict):
        return ""
    trace_id = result.get("traceId")
    if isinstance(trace_id, str):
        return trace_id
    trace_ids = result.get("traceIds")
    if isinstance(trace_ids, list) and trace_ids:
        last = trace_ids[-1]
        return last if isinstance(last, str) else ""
    return ""


def _turn_start_payload(full_session_id: str, prompt: str, hints: dict, ts: int) -> dict:
    return {
        "agent": "openclaw",
        "namespace": _memos_namespace(),
        "sessionId": full_session_id,
        "userText": prompt,
        "skipRetrieval": True,
        "contextHints": hints,
        "ts": ts,
    }


def _turn_end_payload(full_session_id: str, episode_id: str, turn: dict, hints: dict, ts: int) -> dict:
    return {
        "agent": "openclaw",
        "namespace": _memos_namespace(),
        "sessionId": full_session_id,
        "episodeId": episode_id,
        "agentText": turn.get("agentText") or "",
        "toolCalls": turn.get("toolCalls") or [],
        "contextHints": hints,
        "ts": ts,
    }


def _trace_ids_json(conn: sqlite3.Connection, episode_id: str) -> str:
    trace_ids = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM traces WHERE episode_id = ? ORDER BY ts, turn_id, id",
            (episode_id,),
        )
    ]
    return json.dumps(trace_ids, ensure_ascii=False)


def _mark_feedback_repair(meta_json: str, moved_episode_id: str, trace_count: int) -> str:
    try:
        meta = json.loads(meta_json or "{}")
    except json.JSONDecodeError:
        meta = {"previousMetaJson": meta_json}
    repairs = meta.setdefault("omnimemevalFeedbackRepairs", [])
    if isinstance(repairs, list):
        repairs.append({
            "movedEpisodeId": moved_episode_id,
            "traceCount": trace_count,
            "reason": "feedback_bootstrap_repair",
        })
    return json.dumps(meta, ensure_ascii=False)


def _repair_feedback_bootstrap_episode(db_path: Path, full_session_id: str) -> dict:
    repaired = 0
    moved_traces = 0
    skipped = []
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT e.id, e.started_at, e.meta_json,
                   MIN(t.ts) AS first_trace_ts,
                   SUM(CASE WHEN t.user_text LIKE ? THEN 1 ELSE 0 END) AS feedback_traces,
                   COUNT(t.id) AS trace_count
            FROM episodes e
            JOIN traces t ON t.episode_id = e.id
            WHERE e.session_id = ?
            GROUP BY e.id
            ORDER BY e.started_at, first_trace_ts, e.id
            """,
            (f"{FEEDBACK_PREFIX}%", full_session_id),
        ).fetchall()

        for episode_id, started_at, meta_json, _first_trace_ts, feedback_traces, trace_count in rows:
            if not feedback_traces:
                continue
            meta = meta_json or ""
            bootstrapped_feedback = (
                f'"initialUserText":"{FEEDBACK_PREFIX}' in meta
                or (feedback_traces == trace_count and trace_count == 1)
            )
            if not bootstrapped_feedback:
                continue

            target = conn.execute(
                """
                SELECT e.id, e.meta_json
                FROM episodes e
                JOIN traces t ON t.episode_id = e.id
                WHERE e.session_id = ?
                  AND e.id <> ?
                  AND e.started_at <= ?
                  AND t.user_text NOT LIKE ?
                GROUP BY e.id
                ORDER BY e.started_at DESC, MAX(t.ts) DESC
                LIMIT 1
                """,
                (full_session_id, episode_id, started_at, f"{FEEDBACK_PREFIX}%"),
            ).fetchone()
            if target is None:
                skipped.append({"episodeId": episode_id, "reason": "no_previous_episode"})
                continue

            target_id, target_meta = target
            trace_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM traces WHERE episode_id = ? ORDER BY ts, turn_id, id",
                    (episode_id,),
                )
            ]
            if not trace_ids:
                continue

            conn.executemany(
                "UPDATE traces SET episode_id = ? WHERE id = ?",
                [(target_id, trace_id) for trace_id in trace_ids],
            )
            try:
                conn.execute(
                    "UPDATE feedback SET episode_id = ? WHERE episode_id = ?",
                    (target_id, episode_id),
                )
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "UPDATE episodes SET trace_ids_json = ?, meta_json = ? WHERE id = ?",
                (
                    _trace_ids_json(conn, target_id),
                    _mark_feedback_repair(target_meta, episode_id, len(trace_ids)),
                    target_id,
                ),
            )
            remaining = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()[0]
            if remaining == 0:
                conn.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            else:
                conn.execute(
                    "UPDATE episodes SET trace_ids_json = ? WHERE id = ?",
                    (_trace_ids_json(conn, episode_id), episode_id),
                )
            repaired += 1
            moved_traces += len(trace_ids)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"repaired": repaired, "moved_traces": moved_traces, "skipped": skipped}


def _move_trace_to_episode(db_path: Path, trace_id: str, target_episode_id: str) -> bool:
    if not trace_id or not target_episode_id:
        return False
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        row = conn.execute(
            "SELECT episode_id FROM traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
        if not row:
            return False
        source_episode_id = row[0]
        if source_episode_id == target_episode_id:
            return False

        conn.execute(
            "UPDATE traces SET episode_id = ? WHERE id = ?",
            (target_episode_id, trace_id),
        )
        conn.execute(
            "UPDATE episodes SET trace_ids_json = ? WHERE id = ?",
            (_trace_ids_json(conn, target_episode_id), target_episode_id),
        )
        remaining = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE episode_id = ?",
            (source_episode_id,),
        ).fetchone()[0]
        if remaining == 0:
            conn.execute("DELETE FROM episodes WHERE id = ?", (source_episode_id,))
        else:
            conn.execute(
                "UPDATE episodes SET trace_ids_json = ? WHERE id = ?",
                (_trace_ids_json(conn, source_episode_id), source_episode_id),
            )
        conn.commit()
        return True
    finally:
        conn.close()


def _find_feedback_trace(db_path: Path, full_session_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() <= deadline:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            row = conn.execute(
                """
                SELECT t.id, t.episode_id, t.ts
                FROM traces t
                WHERE t.session_id = ?
                  AND t.user_text LIKE ?
                ORDER BY t.ts DESC, t.turn_id DESC
                LIMIT 1
                """,
                (full_session_id, f"{FEEDBACK_PREFIX}%"),
            ).fetchone()
            if row:
                return {"trace_id": row[0], "episode_id": row[1], "trace_ts": row[2]}
            count = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE session_id = ?",
                (full_session_id,),
            ).fetchone()[0]
            last = {"trace_count": int(count)}
        finally:
            conn.close()
        time.sleep(1)
    return {"error": "feedback_trace_not_found", **(last or {})}


def _normalize_trace_ref(trace: dict | None) -> dict:
    if not isinstance(trace, dict):
        return {"episode_id": "", "trace_id": ""}
    return {
        "episode_id": trace.get("episode_id") or trace.get("episodeId") or "",
        "trace_id": trace.get("trace_id") or trace.get("traceId") or "",
        **{key: value for key, value in trace.items() if key not in {"episode_id", "episodeId", "trace_id", "traceId"}},
    }


def _close_structured_feedback_episode(
    client,
    *,
    episode_id: str,
    capture: dict,
    timeout: float,
) -> dict:
    """Close only episodes opened by this temporary bridge client.

    With automatic OpenClaw capture, the shared runtime daemon owns the
    episode lifecycle.  A short-lived feedback bridge can persist feedback to
    that episode through SQLite, but it cannot close the daemon's in-memory
    episode and would incorrectly raise ``episode_not_found``.  Manual capture
    creates/reopens the episode in this bridge, so that path still closes it
    normally and waits for its pipeline to drain.
    """
    if capture.get("status") != "captured":
        return {"ok": True, "owner": "shared_runtime"}
    return client.request(
        "episode.close",
        {"episodeId": episode_id},
        timeout=timeout,
    )


def _context_hints(
    *,
    session: SessionSpec,
    domain_name: str,
    task: dict,
    env_info: dict,
    phase_dir: Path,
) -> dict:
    return {
        "source": MANUAL_MEMOS_SOURCE,
        "domain": domain_name,
        "taskName": task.get("name"),
        "repo": task.get("repo") or env_info.get("repo"),
        "cluster": task.get("cluster") or env_info.get("cluster"),
        "phaseDir": str(phase_dir),
        "trialContainer": env_info.get("container_name"),
        "sessionKey": session.openclaw_session_key,
        "namespace": _memos_namespace(),
    }


def _manual_capture_feedback_trace(
    *,
    client,
    full_session_id: str,
    session_file: Path | None,
    session: SessionSpec,
    domain_name: str,
    task: dict,
    env_info: dict,
    phase_dir: Path,
    timeout: float,
    task_prompt: str | None = None,
    agent_result: dict | None = None,
    feedback_prompt: str | None = None,
    feedback_result: dict | None = None,
) -> dict:
    if task_prompt is not None and feedback_prompt is not None:
        task_turn, feedback_turn, turn_count = build_task_feedback_turns(
            session_file=session_file,
            task_prompt=task_prompt,
            agent_result=agent_result or {},
            feedback_prompt=feedback_prompt,
            feedback_result=feedback_result or {},
            feedback_prefix=FEEDBACK_PREFIX,
        )
    else:
        turns = _extract_turns_from_session(session_file)
        if len(turns) < 2:
            return {
                "status": "error",
                "error": "session_missing_task_or_feedback_turn",
                "turns": len(turns),
            }
        task_turn = turns[0]
        feedback_turn = None
        for turn in turns[1:]:
            if str(turn.get("userText", "")).startswith(FEEDBACK_PREFIX):
                feedback_turn = turn
                break
        if feedback_turn is None:
            feedback_turn = turns[1]
        turn_count = len(turns)

    hints = _context_hints(
        session=session,
        domain_name=domain_name,
        task=task,
        env_info=env_info,
        phase_dir=phase_dir,
    )
    client.request(
        "session.open",
        {
            "agent": "openclaw",
            "namespace": _memos_namespace(),
            "sessionId": full_session_id,
            "meta": hints,
        },
        timeout=min(timeout, 60.0),
    )
    task_start = client.request(
        "turn.start",
        _turn_start_payload(full_session_id, task_turn.get("userText", ""), hints, _now_ms()),
        timeout=timeout,
    )
    episode_id = _episode_id_from_turn_start(task_start)
    if not episode_id:
        return {"status": "error", "error": "memos_task_turn_start_missing_episode"}
    task_end = client.request(
        "turn.end",
        _turn_end_payload(full_session_id, episode_id, task_turn, hints, _now_ms()),
        timeout=timeout,
    )
    task_trace_id = _trace_id_from_turn_end(task_end)

    feedback_start = client.request(
        "turn.start",
        _turn_start_payload(full_session_id, feedback_turn.get("userText", ""), hints, _now_ms()),
        timeout=timeout,
    )
    feedback_episode_id = _episode_id_from_turn_start(feedback_start)
    if not feedback_episode_id:
        return {"status": "error", "error": "memos_feedback_turn_start_missing_episode"}
    feedback_end = client.request(
        "turn.end",
        _turn_end_payload(full_session_id, feedback_episode_id, feedback_turn, hints, _now_ms()),
        timeout=timeout,
    )
    feedback_trace_id = _trace_id_from_turn_end(feedback_end)
    if not feedback_trace_id:
        return {"status": "error", "error": "memos_feedback_turn_end_missing_trace"}

    return {
        "status": "captured",
        "episode_id": episode_id,
        "task_trace_id": task_trace_id,
        "feedback_episode_id": feedback_episode_id,
        "feedback_trace_id": feedback_trace_id,
        "same_episode": feedback_episode_id == episode_id,
        "turns": turn_count,
    }


def submit_memos_feedback_artifact(
    *,
    session: SessionSpec,
    session_file: Path | None,
    task_prompt: str,
    agent_result: dict,
    feedback_prompt: str,
    feedback_result: dict,
    verifier_result: dict,
    domain_name: str,
    task: dict,
    env_info: dict,
    phase_dir: Path,
    timeout: float = 900.0,
) -> dict:
    if not session.openclaw_gateway_session_id:
        return {"status": "skipped", "reason": "missing_openclaw_gateway_session_id"}

    openclaw_home = _openclaw_home_from_session_file(session_file)
    db_path = _memos_db_path(openclaw_home)

    full_session_id = session.openclaw_gateway_session_id
    plugin_root = _memos_plugin_root(openclaw_home)
    client = _connect_shared_openclaw_runtime(openclaw_home, timeout=timeout)
    try:
        capture = _manual_capture_feedback_trace(
            client=client,
            full_session_id=full_session_id,
            session_file=session_file,
            session=session,
            domain_name=domain_name,
            task=task,
            env_info=env_info,
            phase_dir=phase_dir,
            timeout=timeout,
            task_prompt=task_prompt,
            agent_result=agent_result,
            feedback_prompt=feedback_prompt,
            feedback_result=feedback_result,
        )
        if capture.get("status") != "captured":
            return {
                "status": "error",
                "session_id": full_session_id,
                "db_path": str(db_path),
                "capture": capture,
                "plugin_root": str(plugin_root),
            }
        if capture.get("feedback_episode_id") != capture.get("episode_id"):
            moved = _move_trace_to_episode(
                db_path,
                capture.get("feedback_trace_id", ""),
                capture.get("episode_id", ""),
            )
            capture["feedback_trace_moved"] = moved
            if moved:
                capture["feedback_episode_id"] = capture["episode_id"]
        trace_id = capture.get("feedback_trace_id")
        episode_id = capture.get("episode_id")
        if not trace_id or not episode_id:
            return {
                "status": "error",
                "session_id": full_session_id,
                "db_path": str(db_path),
                "capture": capture,
                "error": "feedback_trace_missing_episode_or_trace_id",
                "plugin_root": str(plugin_root),
            }
        submit_result = client.request(
            "feedback.submit",
            {
                "episodeId": episode_id,
                "traceId": trace_id,
                "channel": "explicit",
                "polarity": _feedback_polarity(verifier_result),
                "magnitude": _feedback_magnitude(verifier_result),
                "rationale": feedback_prompt,
                "raw": {
                    "source": MANUAL_MEMOS_SOURCE,
                    "verifier": verifier_result,
                    "taskName": task.get("name"),
                    "domain": domain_name,
                    "phaseDir": str(phase_dir),
                    "feedbackResponse": feedback_result.get("response", ""),
                    "session": session.to_dict(),
                },
                "ts": _now_ms(),
            },
            timeout=timeout,
        )
        close_result = client.request(
            "episode.close",
            {"episodeId": episode_id},
            timeout=timeout,
        )
    finally:
        client.close()

    return {
        "status": "submitted",
        "session_id": full_session_id,
        "episode_id": episode_id,
        "feedback_trace_id": trace_id,
        "feedback_submit_id": submit_result.get("id"),
        "episode_close": close_result,
        "capture": capture,
        "db_path": str(db_path),
        "plugin_root": str(plugin_root),
    }


def submit_memos_structured_feedback(
    *,
    session: SessionSpec,
    session_file: Path | None,
    feedback_prompt: str,
    feedback_result: dict,
    verifier_result: dict,
    domain_name: str,
    task: dict,
    env_info: dict,
    phase_dir: Path,
    timeout: float = 900.0,
) -> dict:
    if not session.openclaw_gateway_session_id:
        return {"status": "skipped", "reason": "missing_openclaw_gateway_session_id"}

    openclaw_home = _openclaw_home_from_session_file(session_file)
    db_path = _memos_db_path(openclaw_home)
    if not db_path.exists():
        return {"status": "skipped", "reason": "memos_db_missing", "db_path": str(db_path)}

    full_session_id = session.openclaw_gateway_session_id
    plugin_root = _memos_plugin_root(openclaw_home)
    client = _connect_shared_openclaw_runtime(openclaw_home, timeout=timeout)
    try:
        # The OpenClaw hook watchdog may return before the daemon-owned
        # turn.end finishes. Wait for that canonical write; replaying the
        # transcript would enqueue the same memory evolution a second time.
        # The shared runtime is the only writer/processor for this path, so we
        # also never repair its live SQLite rows from the evaluator process.
        repair = {"status": "not_needed", "owner": "shared_runtime"}
        capture = {"status": "not_needed", "owner": "shared_runtime"}
        trace = _find_feedback_trace(
            db_path,
            full_session_id,
            timeout=min(300.0, max(5.0, timeout)),
        )
        if trace.get("error"):
            return {
                "status": "error",
                "session_id": full_session_id,
                "db_path": str(db_path),
                "repair": repair,
                "capture": capture,
                "error": "feedback_trace_not_found",
                "trace_lookup": trace,
                "plugin_root": str(plugin_root),
            }
        trace = _normalize_trace_ref(trace)
        if not trace["episode_id"] or not trace["trace_id"]:
            return {
                "status": "error",
                "session_id": full_session_id,
                "db_path": str(db_path),
                "repair": repair,
                "capture": capture,
                "trace": trace,
                "error": "feedback_trace_missing_episode_or_trace_id",
                "plugin_root": str(plugin_root),
            }
        submit_result = client.request(
            "feedback.submit",
            {
                "episodeId": trace["episode_id"],
                "traceId": trace["trace_id"],
                "channel": "explicit",
                "polarity": _feedback_polarity(verifier_result),
                "magnitude": _feedback_magnitude(verifier_result),
                "rationale": feedback_prompt,
                "raw": {
                    "source": MANUAL_MEMOS_SOURCE,
                    "verifier": verifier_result,
                    "taskName": task.get("name"),
                    "domain": domain_name,
                    "phaseDir": str(phase_dir),
                    "feedbackResponse": feedback_result.get("response", ""),
                    "session": session.to_dict(),
                },
                "ts": _now_ms(),
            },
            timeout=timeout,
        )
        close_result = _close_structured_feedback_episode(
            client,
            episode_id=trace["episode_id"],
            capture=capture,
            timeout=timeout,
        )
    finally:
        client.close()

    return {
        "status": "submitted",
        "session_id": full_session_id,
        "episode_id": trace["episode_id"],
        "feedback_trace_id": trace["trace_id"],
        "feedback_submit_id": submit_result.get("id"),
        "episode_close": close_result,
        "repair": repair,
        "capture": capture,
        "db_path": str(db_path),
        "plugin_root": str(plugin_root),
    }
