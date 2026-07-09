import json
import os
import sqlite3
import sys
from pathlib import Path

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


def test_hermes_agent_is_registered():
    agent = create_agent("hermes", {"command": "hermes"})
    assert isinstance(agent, HermesAgentAdapter)


def test_default_hermes_config_shape():
    cfg = load_yaml(ROOT / "configs" / "agentbench" / "agents" / "hermes.yaml")
    agent = cfg["agent"]
    assert agent["name"] == "hermes"
    assert agent["profile"] == "hermes"
    assert agent["runtime"]["home_mode"] == "isolated_copy"
    assert agent["memory"]["memory_enabled"] is False
    assert agent["memory"]["user_profile_enabled"] is False


def test_hermes_provider_extra_body_disables_qwen_thinking(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    _write_global_hermes_config(tmp_path)

    cfg = load_yaml(ROOT / "configs" / "agentbench" / "agents" / "hermes_plain.yaml")["agent"]
    agent = HermesAgentAdapter(cfg)
    session = agent.build_session_spec(
        phase="test",
        domain="reasoning",
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )
    agent.prepare_task({"name": "omni_1"}, {}, session)

    temp_config = yaml.safe_load((Path(agent._temp_home) / "config.yaml").read_text(encoding="utf-8"))
    provider = next(
        item for item in temp_config["custom_providers"] if item["name"] == "qwen3.6-flash"
    )

    assert temp_config["agent"]["reasoning_effort"] == "none"
    assert provider["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


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
    agent.prepare_task({"name": "omni_1"}, {}, session)

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

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    stats = agent.collect_session(session, trial_dir)
    assert stats["turns"] == 2
    assert stats["input"] == 11
    assert stats["output"] == 7
    assert (trial_dir / "session.jsonl").exists()


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
        agent.prepare_task({"name": "omni_1"}, {}, session)
    except RuntimeError as exc:
        assert "home_links must be relative" in str(exc)
    else:
        raise AssertionError("absolute Hermes home_links should fail")


def test_memos_hermes_lifecycle_does_not_reference_openclaw():
    path = ROOT / "configs" / "agentbench" / "memory_plugins" / "memos_hermes.yaml"
    text = path.read_text(encoding="utf-8").lower()
    assert ".openclaw" not in text
    assert "--agent=openclaw" not in text
    assert "--agent=hermes" in text
