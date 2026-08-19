import importlib.util
import json
import os
import signal
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.agents import create_agent
from agentbench.agents.hermes import HermesAgentAdapter
from agentbench.config import load_yaml


def _write_global_hermes_config(home: Path, config: dict | None = None) -> None:
    config_dir = home / ".hermes"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        "model:\n  default: base-model\n" if config is None else json.dumps(config),
        encoding="utf-8",
    )


def _write_fake_hermes(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
args_file = home / "args.jsonl"
args_file.parent.mkdir(parents=True, exist_ok=True)
args_file.open("a", encoding="utf-8").write(json.dumps(sys.argv[1:]) + "\\n")
(home / "cwd.txt").write_text(os.getcwd(), encoding="utf-8")

resume = None
query = ""
for i, arg in enumerate(sys.argv):
    if arg == "--resume" and i + 1 < len(sys.argv):
        resume = sys.argv[i + 1]
    if arg == "--query" and i + 1 < len(sys.argv):
        query = sys.argv[i + 1]

session_id = resume or "hermes-session-1"
db = home / "state.db"
conn = sqlite3.connect(db)
conn.execute(
    "CREATE TABLE IF NOT EXISTS sessions ("
    "id TEXT PRIMARY KEY, source TEXT, started_at REAL, "
    "input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER, "
    "cache_write_tokens INTEGER, reasoning_tokens INTEGER)"
)
conn.execute(
    "CREATE TABLE IF NOT EXISTS messages ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, "
    "tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, finish_reason TEXT, "
    "reasoning TEXT, reasoning_content TEXT, timestamp REAL)"
)
conn.execute(
    "INSERT OR IGNORE INTO sessions VALUES (?, 'cli', ?, 11, 7, 0, 0, 0)",
    (session_id, time.time()),
)
conn.execute(
    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', ?, ?)",
    (session_id, query, time.time()),
)
conn.execute(
    "INSERT INTO messages (session_id, role, content, finish_reason, timestamp) "
    "VALUES (?, 'assistant', ?, 'stop', ?)",
    (session_id, "answer:" + query[:16], time.time()),
)
conn.commit()
conn.close()
print("answer:" + query[:16])
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_hanging_process_tree(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
child = subprocess.Popen([
    sys.executable,
    "-c",
    "import time; time.sleep(60)",
])
(home / "tree-pids.json").write_text(json.dumps({
    "parent": os.getpid(),
    "parent_pgid": os.getpgrp(),
    "child": child.pid,
    "child_pgid": os.getpgid(child.pid),
}), encoding="utf-8")

def shutdown(_signum, _frame):
    child.wait(timeout=5)
    raise SystemExit(0)

signal.signal(signal.SIGTERM, shutdown)
while True:
    time.sleep(1)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_memos_capture(
    db: Path,
    session_id: str = "hermes-session-1",
    *,
    status: str = "closed",
) -> None:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            "CREATE TABLE episodes (id TEXT PRIMARY KEY, session_id TEXT, status TEXT, trace_ids_json TEXT);"
            "CREATE TABLE traces (id TEXT PRIMARY KEY, session_id TEXT, episode_id TEXT);"
        )
        conn.execute(
            "INSERT INTO episodes VALUES ('episode-1', ?, ?, '[\"trace-1\"]')",
            (session_id, status),
        )
        conn.execute(
            "INSERT INTO traces VALUES ('trace-1', ?, 'episode-1')",
            (session_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _load_memos_overlay(monkeypatch):
    class FakeMemTensorProvider:
        def __init__(self) -> None:
            self._bridge = None
            self._session_id = "hermes-session-1"
            self._episode_id = ""
            self._agent_identity = "hermes"
            self._last_trace_id = ""

        def _runtime_namespace(self):
            return {"agentKind": "hermes", "profileId": "default"}

        def _host_runtime_context(self):
            return {}

        def _turn_start(self, query: str, *, session_id: str = ""):
            return self._bridge.request(
                "turn.start", {"query": query, "sessionId": session_id}
            )

        def sync_turn(self, user: str, assistant: str, *, session_id: str = ""):
            if user and not self._episode_id:
                self._turn_start(user, session_id=session_id)
            return self._turn_end(user, assistant, [], int(time.time() * 1000))

    plugins = types.ModuleType("plugins")
    memory = types.ModuleType("plugins.memory")
    memtensor = types.ModuleType("plugins.memory.memtensor")
    memtensor.MemTensorProvider = FakeMemTensorProvider
    monkeypatch.setitem(sys.modules, "plugins", plugins)
    monkeypatch.setitem(sys.modules, "plugins.memory", memory)
    monkeypatch.setitem(sys.modules, "plugins.memory.memtensor", memtensor)

    name = "_test_omnimemeval_memos_overlay"
    source = (
        ROOT
        / "scripts"
        / "agentbench"
        / "integrations"
        / "omnimemeval_memos"
        / "__init__.py"
    )
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _process_is_live(pid: int) -> bool:
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()[2]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state != "Z"


def _materialize_hermes_config(
    tmp_path: Path,
    monkeypatch,
    *,
    domain: str,
    env_info: dict | None = None,
) -> dict:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_WEB_TOOL_DOMAINS", raising=False)
    _write_global_hermes_config(tmp_path, {
        "platform_toolsets": {
            "cli": ["web", "terminal", "file", "code_execution", "memory"],
        },
    })

    agent = HermesAgentAdapter({"runtime": {"home_mode": "isolated_copy"}})
    session = agent.build_session_spec(
        phase="test",
        domain=domain,
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )
    prepared_env = dict(env_info or {})
    prepared_env.setdefault("workspace_dir", str(tmp_path / "workspace"))
    agent.prepare_task({"name": "omni_1"}, prepared_env, session)
    return yaml.safe_load(
        (Path(agent._temp_home) / "config.yaml").read_text(encoding="utf-8")
    )


def test_hermes_agent_is_registered():
    agent = create_agent("hermes", {"command": "hermes"})
    assert isinstance(agent, HermesAgentAdapter)


def test_default_hermes_config_shape():
    cfg = load_yaml(ROOT / "configs" / "agentbench" / "agents" / "hermes.yaml")
    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "hermes" / "plain.yaml")
    from agentbench.run_agent_eval import _compose_agent_config
    agent = _compose_agent_config(cfg["agent"], profile)
    assert cfg["kind"] == "agent"
    assert agent["name"] == "hermes"
    assert agent["runtime"]["home_mode"] == "isolated_copy"
    assert agent["memory"]["memory_enabled"] is False
    assert agent["memory"]["user_profile_enabled"] is False


def test_hermes_profiles_do_not_override_global_cli_toolsets():
    for profile_name in ("plain", "memos"):
        profile = load_yaml(
            ROOT / "configs" / "agentbench" / "profiles" / "hermes" / f"{profile_name}.yaml"
        )
        patch = profile.get("agent_patch", {}).get("hermes_config_patch", {})
        assert "platform_toolsets" not in patch


def test_hermes_memos_test_uses_normal_writable_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    global_config = {
        "memory": {
            "memory_enabled": True,
            "user_profile_enabled": True,
            "provider": "global-provider",
        }
    }
    _write_global_hermes_config(tmp_path, global_config)
    (tmp_path / ".hermes" / "memos-plugin").mkdir()

    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "hermes" / "memos.yaml")
    agent = HermesAgentAdapter(profile["agent_patch"])
    session = agent.build_session_spec(
        phase="test_run_1",
        domain="reasoning",
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(tmp_path / "workspace")},
        session,
    )

    try:
        temp_home = Path(agent._temp_home)
        temp_config = yaml.safe_load((temp_home / "config.yaml").read_text(encoding="utf-8"))
        writable_provider = temp_home / "plugins" / "omnimemeval_memos"

        assert temp_config["memory"] == {
            "memory_enabled": False,
            "user_profile_enabled": False,
            "provider": "omnimemeval_memos",
        }
        assert writable_provider.is_symlink()
        assert (writable_provider / "__init__.py").exists()
        manifest = yaml.safe_load(
            (writable_provider / "plugin.yaml").read_text(encoding="utf-8")
        )
        assert manifest["name"] == "omnimemeval_memos"
        assert manifest["kind"] == "exclusive"
        assert not (temp_home / "plugins" / "omnimemeval_memos_readonly").exists()
        assert json.loads(
            (tmp_path / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        ) == global_config
    finally:
        agent.cleanup_task()


def test_hermes_memos_train_keeps_writable_provider_in_temp_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _write_global_hermes_config(tmp_path, {})
    (tmp_path / ".hermes" / "memos-plugin").mkdir()

    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "hermes" / "memos.yaml")
    agent = HermesAgentAdapter(profile["agent_patch"])
    session = agent.build_session_spec(
        phase="train",
        domain="reasoning",
        split="train",
        task={"name": "omni_1"},
        trial=1,
    )
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(tmp_path / "workspace")},
        session,
    )

    try:
        temp_home = Path(agent._temp_home)
        temp_config = yaml.safe_load((temp_home / "config.yaml").read_text(encoding="utf-8"))
        writable_provider = temp_home / "plugins" / "omnimemeval_memos"
        assert temp_config["memory"]["provider"] == "omnimemeval_memos"
        assert writable_provider.is_symlink()
        assert (writable_provider / "__init__.py").exists()
        assert not (temp_home / "plugins" / "omnimemeval_memos_readonly").exists()
    finally:
        agent.cleanup_task()


def test_hermes_removes_only_web_outside_knowledge_work(tmp_path, monkeypatch):
    config = _materialize_hermes_config(
        tmp_path,
        monkeypatch,
        domain="reasoning",
    )

    assert config["platform_toolsets"]["cli"] == [
        "terminal",
        "file",
        "code_execution",
        "memory",
    ]


def test_hermes_keeps_global_cli_toolsets_for_knowledge_work(tmp_path, monkeypatch):
    config = _materialize_hermes_config(
        tmp_path,
        monkeypatch,
        domain="knowledge_work",
    )

    assert config["platform_toolsets"]["cli"] == [
        "web",
        "terminal",
        "file",
        "code_execution",
        "memory",
    ]


def test_hermes_workspace_cannot_be_overridden_by_profile_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    _write_global_hermes_config(tmp_path, {"terminal": {"cwd": "/global"}})
    workspace = tmp_path / "trial" / "workspace"
    agent = HermesAgentAdapter({
        "hermes_config_patch": {"terminal": {"cwd": "/profile"}},
    })
    session = agent.build_session_spec(
        phase="test", domain="reasoning", split="test",
        task={"name": "omni_1"}, trial=1,
    )
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(workspace)},
        session,
    )
    try:
        config = yaml.safe_load(
            (Path(agent._temp_home) / "config.yaml").read_text(encoding="utf-8")
        )
        assert config["terminal"]["cwd"] == str(workspace.resolve())
    finally:
        agent.cleanup_task()


def test_hermes_information_retrieval_is_search_only(tmp_path, monkeypatch):
    config = _materialize_hermes_config(
        tmp_path,
        monkeypatch,
        domain="information_retrieval",
        env_info={
            "mcp_servers": {
                "bcp-search": {
                    "type": "sse",
                    "url": "http://localhost:9100/mcp",
                },
            },
            "disabled_tools": ["read_file", "write_file", "exec", "web_search"],
        },
    )

    assert config["platform_toolsets"]["cli"] == ["bcp-search"]
    assert config["mcp_servers"]["bcp-search"] == {
        "url": "http://localhost:9100/mcp",
        "transport": "sse",
        "enabled": True,
    }


def test_hermes_base_config_inherits_global_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_hermes_config(tmp_path)

    base = load_yaml(ROOT / "configs" / "agentbench" / "agents" / "hermes.yaml")["agent"]
    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "hermes" / "plain.yaml")
    from agentbench.run_agent_eval import _compose_agent_config
    cfg = _compose_agent_config(base, profile)
    agent = HermesAgentAdapter(cfg)
    session = agent.build_session_spec(
        phase="test",
        domain="reasoning",
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )
    workspace = tmp_path / "workspace"
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(workspace)},
        session,
    )

    temp_config = yaml.safe_load((Path(agent._temp_home) / "config.yaml").read_text(encoding="utf-8"))
    assert temp_config["model"]["default"] == "base-model"
    assert "custom_providers" not in temp_config
    assert temp_config["agent"]["reasoning_effort"] == "none"


def test_hermes_feedback_resumes_real_hermes_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_hermes_config(tmp_path)
    fake_hermes = tmp_path / "hermes"
    _write_fake_hermes(fake_hermes)

    agent = HermesAgentAdapter({"command": str(fake_hermes)})
    session = agent.build_session_spec(
        phase="train",
        domain="reasoning",
        split="train",
        task={"name": "omni_1"},
        trial=1,
    )
    workspace = tmp_path / "workspace"
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(workspace)},
        session,
    )

    first = agent.call("task prompt", session, timeout=5)
    second = agent.call("Verifier feedback for the previous attempt.", session, timeout=5)

    assert first["completion_status"] == "completed"
    assert first["hermes_session_id"] == "hermes-session-1"
    assert second["completion_status"] == "completed"
    assert second["hermes_session_id"] == "hermes-session-1"

    args_lines = (Path(agent._temp_home) / "args.jsonl").read_text(encoding="utf-8").splitlines()
    first_args = json.loads(args_lines[0])
    second_args = json.loads(args_lines[1])
    assert "--resume" not in first_args
    assert second_args[second_args.index("--resume") + 1] == "hermes-session-1"
    assert (Path(agent._temp_home) / "cwd.txt").read_text(encoding="utf-8") == str(
        workspace.resolve()
    )

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    stats = agent.collect_session(session, trial_dir)
    assert stats["turns"] == 2
    assert stats["input"] == 11
    assert stats["output"] == 7
    assert (trial_dir / "session.jsonl").exists()


def test_hermes_train_requires_durable_memos_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    memos_db = tmp_path / "run-memos" / "data" / "memos.db"
    monkeypatch.setenv("MEMOS_DB", str(memos_db))
    _write_global_hermes_config(tmp_path)
    fake_hermes = tmp_path / "hermes"
    _write_fake_hermes(fake_hermes)

    agent = HermesAgentAdapter({
        "command": str(fake_hermes),
        "runtime": {
            "verify_memos_capture": True,
            "memos_capture_verify_timeout_seconds": 0,
        },
    })
    session = agent.build_session_spec(
        phase="train", domain="reasoning", split="train",
        task={"name": "omni_1"}, trial=1,
    )
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(tmp_path / "workspace")},
        session,
    )

    try:
        with pytest.raises(RuntimeError, match="capture was not persisted"):
            agent.call("task prompt", session, timeout=5)
    finally:
        agent.cleanup_task()


def test_hermes_test_requires_durable_memos_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    memos_db = tmp_path / "run-memos" / "data" / "memos.db"
    monkeypatch.setenv("MEMOS_DB", str(memos_db))
    _write_global_hermes_config(tmp_path)
    fake_hermes = tmp_path / "hermes"
    _write_fake_hermes(fake_hermes)

    agent = HermesAgentAdapter({
        "command": str(fake_hermes),
        "runtime": {
            "verify_memos_capture": True,
            "memos_capture_verify_timeout_seconds": 0,
        },
    })
    session = agent.build_session_spec(
        phase="test_run_1", domain="reasoning", split="test",
        task={"name": "omni_2"}, trial=1,
    )
    agent.prepare_task(
        {"name": "omni_2"},
        {"workspace_dir": str(tmp_path / "workspace-test")},
        session,
    )

    try:
        with pytest.raises(RuntimeError, match="capture was not persisted"):
            agent.call("test prompt", session, timeout=5)
    finally:
        agent.cleanup_task()


def test_hermes_train_reports_verified_memos_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    memos_db = tmp_path / "run-memos" / "data" / "memos.db"
    monkeypatch.setenv("MEMOS_DB", str(memos_db))
    _write_memos_capture(memos_db)
    _write_global_hermes_config(tmp_path)
    fake_hermes = tmp_path / "hermes"
    _write_fake_hermes(fake_hermes)

    agent = HermesAgentAdapter({
        "command": str(fake_hermes),
        "runtime": {"verify_memos_capture": True},
    })
    session = agent.build_session_spec(
        phase="train", domain="reasoning", split="train",
        task={"name": "omni_1"}, trial=1,
    )
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(tmp_path / "workspace")},
        session,
    )
    try:
        result = agent.call("task prompt", session, timeout=5)
    finally:
        agent.cleanup_task()

    assert result["memos_capture"] == {
        "verified": True,
        "session_id": "hermes-session-1",
        "captured_episodes": 1,
        "closed_episodes": 1,
        "open_episodes": 0,
        "traces": 1,
    }


def test_hermes_capture_gate_accepts_open_episode_with_durable_trace(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    memos_db = tmp_path / "run-memos" / "data" / "memos.db"
    monkeypatch.setenv("MEMOS_DB", str(memos_db))
    _write_memos_capture(memos_db, status="open")
    _write_global_hermes_config(tmp_path)
    fake_hermes = tmp_path / "hermes"
    _write_fake_hermes(fake_hermes)

    agent = HermesAgentAdapter({
        "command": str(fake_hermes),
        "runtime": {"verify_memos_capture": True},
    })
    session = agent.build_session_spec(
        phase="train", domain="reasoning", split="train",
        task={"name": "omni_1"}, trial=1,
    )
    agent.prepare_task(
        {"name": "omni_1"},
        {"workspace_dir": str(tmp_path / "workspace-open")},
        session,
    )
    try:
        result = agent.call("task prompt", session, timeout=5)
    finally:
        agent.cleanup_task()

    assert result["memos_capture"] == {
        "verified": True,
        "session_id": "hermes-session-1",
        "captured_episodes": 1,
        "closed_episodes": 0,
        "open_episodes": 1,
        "traces": 1,
    }


def test_hermes_memos_retrieval_timeout_degrades_but_capture_continues(monkeypatch):
    module = _load_memos_overlay(monkeypatch)

    class RpcTimeout(RuntimeError):
        code = "timeout"

    class FakeBridge:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, params=None, *, timeout=30.0):
            self.calls.append((method, params, timeout))
            if method == "turn.start":
                raise RpcTimeout("turn.start did not respond within 30s")
            if method == "turn.end":
                return {"episodeId": "episode-real", "traceId": "trace-real"}
            return {}

    provider = module.OmniMemEvalMemOSProvider()
    provider._bridge = FakeBridge()

    with pytest.raises(RpcTimeout):
        provider._turn_start("task", session_id="hermes-session-1")
    assert provider._has_unresolved_episode()

    trace_id = provider.sync_turn(
        "task", "answer", session_id="hermes-session-1"
    )

    assert trace_id == "trace-real"
    assert provider._episode_id == "episode-real"
    assert provider._last_trace_id == "trace-real"
    assert [method for method, _, _ in provider._bridge.calls] == [
        "turn.start",
        "turn.end",
    ]
    turn_end_payload = provider._bridge.calls[-1][1]
    assert turn_end_payload["episodeId"] == ""


def test_hermes_memos_non_timeout_does_not_claim_an_unresolved_episode(monkeypatch):
    module = _load_memos_overlay(monkeypatch)

    class ClosedBridge:
        def request(self, method, params=None, *, timeout=30.0):
            raise RuntimeError("bridge closed")

    provider = module.OmniMemEvalMemOSProvider()
    provider._bridge = ClosedBridge()

    with pytest.raises(RuntimeError, match="bridge closed"):
        provider._turn_start("task", session_id="hermes-session-1")

    assert provider._episode_id == ""
    assert not provider._has_unresolved_episode()


def test_hermes_memos_turn_end_timeout_does_not_issue_a_second_write(monkeypatch):
    module = _load_memos_overlay(monkeypatch)

    class RpcTimeout(RuntimeError):
        code = "timeout"

    class FakeBridge:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, params=None, *, timeout=30.0):
            self.calls.append((method, params, timeout))
            raise RpcTimeout("turn.end did not respond before the long RPC deadline")

    provider = module.OmniMemEvalMemOSProvider()
    provider._bridge = FakeBridge()
    provider._episode_id = "episode-real"

    with pytest.raises(RpcTimeout):
        provider._turn_end(
            "task",
            "final answer",
            [{"name": "exec", "result": "x" * 1000}],
            int(time.time() * 1000),
        )

    assert len(provider._bridge.calls) == 1
    method, payload, timeout = provider._bridge.calls[0]
    assert method == "turn.end"
    assert len(payload["toolCalls"]) == 1
    assert payload["requestId"].startswith("hermes-turn-")
    assert timeout == 75.0


def test_hermes_memos_turn_end_uses_a_stable_request_id(monkeypatch):
    module = _load_memos_overlay(monkeypatch)

    class FakeBridge:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, params=None, *, timeout=30.0):
            self.calls.append((method, params, timeout))
            return {"episodeId": "episode-real", "traceId": "trace-real"}

    provider = module.OmniMemEvalMemOSProvider()
    provider._bridge = FakeBridge()
    provider._episode_id = "episode-real"
    ts_ms = int(time.time() * 1000)

    provider._turn_end("task", "answer", [], ts_ms)
    provider._turn_end("task", "answer", [], ts_ms)

    first_payload = provider._bridge.calls[0][1]
    second_payload = provider._bridge.calls[1][1]
    assert first_payload["requestId"] == second_payload["requestId"]
    assert provider._bridge.calls[0][2] == 75.0
    assert provider._bridge.calls[1][2] == 75.0


def test_hermes_memos_compacts_oversized_tool_payload_before_first_request(monkeypatch):
    module = _load_memos_overlay(monkeypatch)

    class FakeBridge:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, params=None, *, timeout=30.0):
            self.calls.append((method, params, timeout))
            return {"episodeId": "episode-real", "traceId": "trace-compact"}

    provider = module.OmniMemEvalMemOSProvider()
    provider._bridge = FakeBridge()
    provider._episode_id = "episode-real"
    tool_calls = [
        {"name": "search", "result": "document"}
        for _ in range(provider._TOOL_CALL_COUNT_LIMIT + 1)
    ]

    trace_id = provider._turn_end(
        "task", "final answer", tool_calls, int(time.time() * 1000)
    )

    assert trace_id == "trace-compact"
    assert len(provider._bridge.calls) == 1
    payload = provider._bridge.calls[0][1]
    assert payload["toolCalls"] == []
    assert payload["agentText"] == "final answer"
    assert payload["contextHints"]["omnimemevalCaptureFallback"] == (
        "oversized_tool_transcript"
    )
    assert payload["contextHints"]["omnimemevalDroppedToolCalls"] == len(tool_calls)


def test_hermes_memos_stamps_trial_identity_in_episode_context(monkeypatch):
    context = {
        "semantic_session_id": "omnimemeval:test_run_1:reasoning:test:omni_2:trial:1",
        "metadata": {
            "phase": "test_run_1",
            "domain": "reasoning",
            "split": "test",
            "task": "omni_2",
            "trial": 1,
        },
    }
    monkeypatch.setenv("OMNIMEMEVAL_AGENT_CONTEXT", json.dumps(context))
    module = _load_memos_overlay(monkeypatch)

    class FakeBridge:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, params=None, *, timeout=30.0):
            self.calls.append((method, params, timeout))
            return {"episodeId": "episode-real", "traceId": "trace-real"}

    provider = module.OmniMemEvalMemOSProvider()
    provider._bridge = FakeBridge()
    provider._session_id = "hermes-session-2"
    provider._episode_id = "episode-real"
    provider._turn_end("task", "answer", [], int(time.time() * 1000))

    hints = provider._bridge.calls[0][1]["contextHints"]
    assert hints["omnimemevalTrialKey"] == context["semantic_session_id"]
    assert hints["omnimemevalPhase"] == "test_run_1"
    assert hints["omnimemevalDomain"] == "reasoning"
    assert hints["omnimemevalTask"] == "omni_2"


def test_hermes_timeout_terminates_entire_process_group(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_hermes_config(tmp_path)
    fake_hermes = tmp_path / "hanging-hermes"
    _write_hanging_process_tree(fake_hermes)

    agent = HermesAgentAdapter({
        "command": str(fake_hermes),
        "runtime": {
            "cli_timeout_grace_seconds": 0,
            "cli_terminate_grace_seconds": 0.5,
        },
    })
    session = agent.build_session_spec(
        phase="test",
        domain="reasoning",
        split="test",
        task={"name": "omni_timeout"},
        trial=1,
    )
    agent.prepare_task(
        {"name": "omni_timeout"},
        {"workspace_dir": str(tmp_path / "workspace-timeout")},
        session,
    )

    pids: dict[str, int] = {}
    try:
        result = agent.call("hang", session, timeout=0.3)
        pids = json.loads(
            (Path(agent._temp_home) / "tree-pids.json").read_text(encoding="utf-8")
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(
            _process_is_live(pids[key]) for key in ("parent", "child")
        ):
            time.sleep(0.05)
        live_pids = [
            pids[key]
            for key in ("parent", "child")
            if _process_is_live(pids[key])
        ]
    finally:
        for key in ("parent", "child"):
            pid = pids.get(key)
            if pid and _process_is_live(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        agent.cleanup_task()

    assert result["completion_status"] == "timeout"
    assert pids["parent_pgid"] == pids["parent"]
    assert pids["child_pgid"] == pids["parent"]
    assert live_pids == []


def test_hermes_home_links_are_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_hermes_config(tmp_path)
    agent = HermesAgentAdapter({"runtime": {"home_links": ["/absolute/not-allowed"]}})

    session = agent.build_session_spec(
        phase="test",
        domain="reasoning",
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )
    try:
        agent.prepare_task(
            {"name": "omni_1"},
            {"workspace_dir": str(tmp_path / "workspace")},
            session,
        )
    except RuntimeError as exc:
        assert "home_links must be relative" in str(exc)
    else:
        raise AssertionError("absolute Hermes home_links should fail")


def test_memos_hermes_lifecycle_does_not_reference_openclaw():
    path = ROOT / "configs" / "agentbench" / "memory_plugins" / "memos" / "lifecycle" / "hermes.yaml"
    text = path.read_text(encoding="utf-8").lower()
    config = load_yaml(path)
    assert config["plugin"] == "memos"
    assert config["agent"] == "hermes"
    assert ".openclaw" not in text
    assert "--agent=openclaw" not in text
    assert "--agent=hermes" in text
