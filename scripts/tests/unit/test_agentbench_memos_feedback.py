import json
import socket
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.memos_feedback import (
    SharedOpenClawRuntimeClient,
    _close_structured_feedback_episode,
    _normalize_trace_ref,
    _openclaw_runtime_socket_path,
)
from agentbench.session import SessionSpec
import agentbench.memos_feedback as memos_feedback


def test_normalize_trace_ref_accepts_snake_and_camel_case_ids():
    assert _normalize_trace_ref({"episode_id": "ep1", "trace_id": "tr1"}) == {
        "episode_id": "ep1",
        "trace_id": "tr1",
    }
    assert _normalize_trace_ref({"episodeId": "ep2", "traceId": "tr2"}) == {
        "episode_id": "ep2",
        "trace_id": "tr2",
    }


class _FakeBridgeClient:
    def __init__(self):
        self.requests = []

    def request(self, method, params, *, timeout):
        self.requests.append((method, params, timeout))
        return {"ok": True}


def test_structured_feedback_does_not_close_shared_runtime_episode():
    client = _FakeBridgeClient()

    result = _close_structured_feedback_episode(
        client,
        episode_id="ep-shared",
        capture={"status": "not_needed"},
        timeout=30,
    )

    assert result == {"ok": True, "owner": "shared_runtime"}
    assert client.requests == []


def test_structured_feedback_closes_episode_created_by_temporary_bridge():
    client = _FakeBridgeClient()

    result = _close_structured_feedback_episode(
        client,
        episode_id="ep-manual",
        capture={"status": "captured"},
        timeout=30,
    )

    assert result == {"ok": True}
    assert client.requests == [
        ("episode.close", {"episodeId": "ep-manual"}, 30),
    ]


def test_openclaw_runtime_socket_path_matches_plugin_hash(tmp_path):
    home = tmp_path / "memos-plugin"
    digest = __import__("hashlib").sha256(str(home).encode()).hexdigest()[:24]

    assert _openclaw_runtime_socket_path(home) == Path(
        f"/tmp/memos-openclaw-{__import__('os').getuid()}-{digest}.sock"
    )


def test_shared_runtime_client_uses_existing_socket_without_spawning_bridge(tmp_path):
    socket_path = tmp_path / "runtime.sock"
    ready = threading.Event()
    received = []

    def serve():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(socket_path))
            server.listen(1)
            ready.set()
            conn, _ = server.accept()
            with conn, conn.makefile("r", encoding="utf-8") as reader:
                request = json.loads(reader.readline())
                received.append(request)
                conn.sendall(
                    (json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}) + "\n").encode()
                )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)

    client = SharedOpenClawRuntimeClient(socket_path, connect_timeout=1)
    try:
        assert client.request("feedback.submit", {"episodeId": "ep1"}, timeout=1) == {
            "ok": True
        }
    finally:
        client.close()
    thread.join(timeout=2)

    assert received == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "feedback.submit",
            "params": {"episodeId": "ep1"},
        }
    ]


def test_shared_runtime_client_fails_when_owner_socket_is_missing(tmp_path):
    missing = tmp_path / "missing.sock"

    try:
        SharedOpenClawRuntimeClient(missing, connect_timeout=0.01)
    except RuntimeError as exc:
        assert "shared runtime socket" in str(exc)
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing shared runtime socket must fail")


def test_structured_feedback_never_replays_turns_when_shared_capture_is_late(
    tmp_path, monkeypatch
):
    openclaw_home = tmp_path / ".openclaw"
    session_file = openclaw_home / "agents" / "main" / "sessions" / "session.jsonl"
    session_file.parent.mkdir(parents=True)
    session_file.write_text("", encoding="utf-8")
    db = openclaw_home / "memos-plugin" / "data" / "memos.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"sqlite")

    client = _FakeBridgeClient()
    client.closed = False
    client.close = lambda: setattr(client, "closed", True)
    monkeypatch.setattr(
        memos_feedback,
        "_connect_shared_openclaw_runtime",
        lambda *_args, **_kwargs: client,
    )
    monkeypatch.setattr(
        memos_feedback,
        "_find_feedback_trace",
        lambda *_args, **_kwargs: {"error": "feedback_trace_not_found"},
    )
    monkeypatch.setattr(
        memos_feedback,
        "_manual_capture_feedback_trace",
        lambda **_kwargs: pytest.fail("shared runtime turns must never be replayed"),
    )
    monkeypatch.setattr(
        memos_feedback,
        "_repair_feedback_bootstrap_episode",
        lambda *_args, **_kwargs: pytest.fail("daemon-owned DB must not be repaired directly"),
    )

    result = memos_feedback.submit_memos_structured_feedback(
        session=SessionSpec(
            cli_session_id="session",
            semantic_session_id="session",
            source_ref="test",
            openclaw_gateway_session_id="openclaw::main::agent:main:explicit:session",
        ),
        session_file=session_file,
        feedback_prompt="feedback",
        feedback_result={"response": "reflection"},
        verifier_result={"reward": 1.0},
        domain_name="reasoning",
        task={"name": "task"},
        env_info={},
        phase_dir=tmp_path,
        timeout=10,
    )

    assert result["status"] == "error"
    assert result["error"] == "feedback_trace_not_found"
    assert client.requests == []
    assert client.closed is True
