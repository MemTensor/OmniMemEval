import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.agents.openclaw import OpenClawAgentAdapter
from agentbench.config import load_yaml
from agentbench.run_agent_eval import _compose_agent_config


def _write_global_config(home: Path, config: dict):
    config_dir = home / ".openclaw"
    config_dir.mkdir(parents=True)
    (config_dir / "openclaw.json").write_text(json.dumps(config), encoding="utf-8")


def test_openclaw_config_filters_memory_plugins_but_keeps_brave(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_config(tmp_path, {
        "plugins": {
            "enabled": True,
            "allow": ["brave", "memos-local-plugin", "mem0", "everos", "memmy-memory"],
            "entries": {
                "brave": {"enabled": True},
                "memos-local-plugin": {"enabled": True},
                "mem0": {"enabled": True},
                "everos": {"enabled": True},
                "memmy-memory": {"enabled": True},
            },
            "slots": {"memory": "memos-local-plugin", "other": "brave"},
        },
        "tools": {
            "alsoAllow": ["brave_search", "memos_search", "mem0_search", "everos_remember", "memmy_get"]
        },
    })

    agent = OpenClawAgentAdapter({
        "runtime": {
            "disabled_plugin_names": ["memos-local-plugin", "mem0", "everos", "memmy-memory"],
            "disabled_tool_prefixes": ["memos_", "mem0_", "everos_", "memmy_"],
        }
    })

    config = agent._build_openclaw_config()

    assert config["plugins"]["allow"] == ["brave"]
    assert set(config["plugins"]["entries"]) == {"brave"}
    assert config["plugins"]["slots"] == {"other": "brave"}
    assert config["tools"]["alsoAllow"] == ["brave_search"]


def test_default_openclaw_config_selectively_disables_memory_plugins():
    base = load_yaml(ROOT / "configs" / "agentbench" / "agents" / "openclaw.yaml")["agent"]
    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "openclaw" / "plain.yaml")
    runtime = _compose_agent_config(base, profile)["runtime"]

    assert runtime["disable_plugins"] is False
    assert "memos-local-plugin" in runtime["disabled_plugin_names"]
    assert "mem0" in runtime["disabled_plugin_names"]
    assert "everos" in runtime["disabled_plugin_names"]
    assert "memos_" in runtime["disabled_tool_prefixes"]


def test_openclaw_config_still_supports_disabling_all_plugins(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_config(tmp_path, {
        "plugins": {"allow": ["brave"]},
        "mcp": {"servers": {"global-search": {"command": "search"}}},
    })

    agent = OpenClawAgentAdapter({"runtime": {"disable_plugins": True}})

    config = agent._build_openclaw_config()

    assert "plugins" not in config
    assert "mcp" not in config


def test_openclaw_config_mcp_only_keeps_only_task_mcp_and_denies_other_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_config(tmp_path, {
        "plugins": {"allow": ["brave"], "entries": {"brave": {"enabled": True}}},
        "mcp": {"servers": {"global-search": {"command": "search"}}},
        "tools": {"deny": ["web_search"]},
    })
    agent = OpenClawAgentAdapter({"runtime": {"disable_plugins": False}})
    agent._task_env_info = {
        "mcp_only": True,
        "mcp_servers": {
            "bcp-search": {"command": "python", "args": ["server.py"]},
        },
        "disabled_tools": ["exec", "read_file"],
    }

    config = agent._build_openclaw_config()

    assert "plugins" not in config
    assert set(config["mcp"]["servers"]) == {"bcp-search"}
    assert set(config["tools"]["deny"]) == {"web_search", "exec", "read_file"}


def test_isolated_home_links_expose_global_plugin_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_config(tmp_path, {
        "plugins": {"allow": ["memos-local-plugin"], "entries": {"memos-local-plugin": {"enabled": True}}},
    })
    (tmp_path / ".openclaw" / "extensions" / "memos-local-plugin").mkdir(parents=True)
    (tmp_path / ".openclaw" / "memos-plugin").mkdir()
    (tmp_path / ".openclaw" / "npm").mkdir()

    agent = OpenClawAgentAdapter({
        "model": "dashscope/qwen3.6-flash",
        "providers": {
            "dashscope": {
                "baseUrl": "https://example.test/v1",
                "apiKey": "secret",
                "models": [{"id": "qwen3.6-flash", "params": {"extra_body": {"enable_thinking": False}}}],
            },
        },
        "runtime": {
            "home_mode": "isolated_copy",
            "disabled_plugin_names": [],
            "home_links": ["extensions/memos-local-plugin", "memos-plugin", "npm"],
        },
    })

    agent._ensure_temp_config()

    temp_home = Path(agent._temp_home)
    config_dir = temp_home / ".openclaw"
    written = json.loads((config_dir / "openclaw.json").read_text(encoding="utf-8"))
    assert written["agents"]["defaults"]["model"]["primary"] == "dashscope/qwen3.6-flash"
    assert written["models"]["providers"]["dashscope"]["baseUrl"] == "https://example.test/v1"
    assert written["models"]["providers"]["dashscope"]["apiKey"] == "secret"
    assert written["plugins"]["entries"]["memos-local-plugin"]["enabled"] is True
    assert (config_dir / "extensions" / "memos-local-plugin").is_symlink()
    assert (config_dir / "memos-plugin").is_symlink()
    assert (config_dir / "npm").is_symlink()


def test_isolated_home_links_can_use_run_scoped_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    source = tmp_path / "run" / "openclaw"
    source.mkdir(parents=True)
    (source / "openclaw.json").write_text(
        json.dumps({"plugins": {"enabled": True}}),
        encoding="utf-8",
    )
    (source / "memos-plugin").mkdir()
    monkeypatch.setenv("OMNIMEMEVAL_OPENCLAW_HOME_SOURCE", str(source))

    agent = OpenClawAgentAdapter({
        "runtime": {
            "home_mode": "isolated_copy",
            "home_links": ["memos-plugin"],
        },
    })
    agent._ensure_temp_config()
    try:
        config_dir = Path(agent._temp_home) / ".openclaw"
        link = config_dir / "memos-plugin"
        assert link.is_symlink()
        assert link.resolve() == (source / "memos-plugin").resolve()
        assert json.loads((config_dir / "openclaw.json").read_text())["plugins"]["enabled"] is True
    finally:
        agent.cleanup_task()


def test_memos_profile_enables_only_selected_memory_integration():
    cfg = {
        "runtime": {
            "disable_plugins": False,
            "disabled_plugin_names": ["memos-local-plugin", "mem0"],
            "disabled_tool_prefixes": ["memos_", "mem0_"],
        }
    }

    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "openclaw" / "memos.yaml")
    prepared = _compose_agent_config(cfg, profile)

    assert prepared["runtime"]["disable_plugins"] is False
    assert prepared["runtime"]["disabled_plugin_names"] == []
    assert prepared["runtime"]["disabled_tool_prefixes"] == []
    assert prepared["runtime"]["home_mode"] == "isolated_copy"
    assert prepared["runtime"]["home_links"] == ["extensions/memos-local-plugin", "memos-plugin", "npm"]
    assert prepared["runtime"]["transport"] == "gateway"
    assert prepared["openclaw_config_patch"]["plugins"]["entries"]["memos-local-plugin"]
    assert cfg["runtime"]["disabled_plugin_names"] == ["memos-local-plugin", "mem0"]


def test_memos_test_config_is_retrieval_only_and_does_not_modify_global_config(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    global_config = {
        "plugins": {
            "enabled": True,
            "allow": ["brave", "memos-local-plugin"],
            "entries": {
                "brave": {"enabled": True},
                "memos-local-plugin": {
                    "enabled": True,
                    "config": {
                        "memory_search": {"enabled": False},
                        "memory_add": {"enabled": True},
                    },
                },
            },
        },
        "tools": {"alsoAllow": ["brave_search", "memos_search"]},
        "mcp": {"servers": {"global-search": {"command": "global"}}},
    }
    _write_global_config(tmp_path, global_config)
    config_dir = tmp_path / ".openclaw"
    (config_dir / "extensions" / "memos-local-plugin").mkdir(parents=True)
    (config_dir / "memos-plugin").mkdir()
    (config_dir / "npm").mkdir()

    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "openclaw" / "memos.yaml")
    agent = OpenClawAgentAdapter(profile["agent_patch"])
    session = agent.build_session_spec(
        phase="test_run_1",
        domain="information_retrieval",
        split="test",
        task={"name": "ir_1"},
        trial=1,
    )
    agent.prepare_task(
        {"name": "ir_1"},
        {
            "workspace_dir": str(tmp_path / "workspace-ir"),
            "mcp_only": True,
            "mcp_servers": {"bcp-search": {"command": "python", "args": ["server.py"]}},
            "disabled_tools": ["exec", "read_file"],
        },
        session,
    )

    try:
        temp_config_path = Path(agent._temp_home) / ".openclaw" / "openclaw.json"
        temp_config = json.loads(temp_config_path.read_text(encoding="utf-8"))
        entry = temp_config["plugins"]["entries"]["memos-local-plugin"]

        assert set(temp_config["plugins"]["entries"]) == {"memos-local-plugin"}
        assert set(temp_config["mcp"]["servers"]) == {"bcp-search"}
        assert entry["config"]["memory_search"]["enabled"] is True
        assert entry["config"]["memory_add"]["enabled"] is False
        assert "alsoAllow" not in temp_config["tools"]
        assert "memos_search" in temp_config["tools"]["deny"]
        assert json.loads(
            (config_dir / "openclaw.json").read_text(encoding="utf-8")
        ) == global_config
    finally:
        agent.cleanup_task()


def test_memos_train_temp_config_disables_automatic_reads_and_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_config(tmp_path, {})
    config_dir = tmp_path / ".openclaw"
    (config_dir / "extensions" / "memos-local-plugin").mkdir(parents=True)
    (config_dir / "memos-plugin").mkdir()
    (config_dir / "npm").mkdir()

    profile = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "openclaw" / "memos.yaml")
    agent = OpenClawAgentAdapter(profile["agent_patch"])
    session = agent.build_session_spec(
        phase="train",
        domain="reasoning",
        split="train",
        task={"name": "train_1"},
        trial=1,
    )
    agent.prepare_task(
        {"name": "train_1"},
        {"workspace_dir": str(tmp_path / "workspace-train")},
        session,
    )

    try:
        temp_config = json.loads(
            (Path(agent._temp_home) / ".openclaw" / "openclaw.json").read_text(encoding="utf-8")
        )
        plugin_config = temp_config["plugins"]["entries"]["memos-local-plugin"]["config"]
        assert plugin_config["memory_search"]["enabled"] is False
        assert plugin_config["memory_add"]["enabled"] is False
    finally:
        agent.cleanup_task()


def test_memos_openclaw_lifecycle_does_not_mutate_config_modes():
    lifecycle = load_yaml(
        ROOT
        / "configs"
        / "agentbench"
        / "memory_plugins"
        / "memos"
        / "lifecycle"
        / "openclaw.yaml"
    )

    assert "modes" not in lifecycle


def test_openclaw_workspace_cannot_be_overridden_by_global_or_profile_config(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_global_config(tmp_path, {
        "agents": {
            "defaults": {"workspace": "/global/default"},
            "list": [{"id": "main", "default": True, "workspace": "/global/main"}],
        }
    })
    workspace = tmp_path / "trial" / "workspace"
    agent = OpenClawAgentAdapter({
        "openclaw_config_patch": {
            "agents": {"defaults": {"workspace": "/profile/default"}},
        }
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
        config = json.loads(
            (Path(agent._temp_home) / ".openclaw" / "openclaw.json").read_text(
                encoding="utf-8"
            )
        )
        resolved = str(workspace.resolve())
        assert config["agents"]["defaults"]["workspace"] == resolved
        assert config["agents"]["list"][0]["workspace"] == resolved
    finally:
        agent.cleanup_task()


def test_openclaw_call_recovers_when_cli_does_not_exit_after_session_response(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
    session_dir.mkdir(parents=True)
    script = tmp_path / "fake_openclaw.py"
    script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

home = Path(os.environ["FAKE_OPENCLAW_HOME"])
session_id = sys.argv[1]
session_file = home / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
session_file.parent.mkdir(parents=True, exist_ok=True)
entry = {
    "type": "message",
    "message": {
        "role": "assistant",
        "stopReason": "stop",
        "content": [{"type": "text", "text": "final answer from session"}],
    },
}
session_file.write_text(json.dumps(entry) + "\\n", encoding="utf-8")
print("cli still alive", flush=True)
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )

    agent = OpenClawAgentAdapter({
        "runtime": {
            "session_completion_poll_interval": 0.05,
            "session_completion_exit_grace_seconds": 0.05,
            "session_completion_terminate_grace_seconds": 0.5,
        }
    })
    agent._temp_home = str(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent._workspace_dir = str(workspace)
    session = agent.build_session_spec(
        phase="test_run_1",
        domain="reasoning",
        split="test",
        task={"name": "omni_1"},
        trial=1,
    )

    monkeypatch.setattr(
        agent,
        "_build_cli_cmd",
        lambda prompt, call_session, timeout: [
            sys.executable,
            str(script),
            call_session.cli_session_id,
        ],
    )
    monkeypatch.setattr(
        agent,
        "_get_subprocess_env",
        lambda call_session: {
            **os.environ,
            "FAKE_OPENCLAW_HOME": str(tmp_path),
        },
    )

    start = time.time()
    result = agent.call("prompt", session, timeout=30)

    assert time.time() - start < 3
    assert result["completion_status"] == "completed"
    assert result["response"] == "final answer from session"
    assert result["cli_session_recovered"] is True
    assert result["cli_terminated_after_session_completion"] is True


def test_openclaw_call_ignores_completed_assistant_from_previous_turn(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
    session_dir.mkdir(parents=True)
    agent = OpenClawAgentAdapter({
        "runtime": {
            "session_completion_poll_interval": 0.05,
            "session_completion_exit_grace_seconds": 0.05,
            "session_completion_terminate_grace_seconds": 0.5,
        }
    })
    agent._temp_home = str(tmp_path)
    session = agent.build_session_spec(
        phase="train",
        domain="reasoning",
        split="train",
        task={"name": "omni_1"},
        trial=1,
    )
    session_file = session_dir / f"{session.cli_session_id}.jsonl"
    old_entry = {
        "type": "message",
        "message": {
            "role": "assistant",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "old answer"}],
        },
    }
    session_file.write_text(json.dumps(old_entry) + "\n", encoding="utf-8")

    script = tmp_path / "fake_openclaw_append.py"
    script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

home = Path(os.environ["FAKE_OPENCLAW_HOME"])
session_id = sys.argv[1]
session_file = home / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
time.sleep(0.2)
entry = {
    "type": "message",
    "message": {
        "role": "assistant",
        "stopReason": "stop",
        "content": [{"type": "text", "text": "new feedback answer"}],
    },
}
with session_file.open("a", encoding="utf-8") as f:
    f.write(json.dumps(entry) + "\\n")
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent,
        "_build_cli_cmd",
        lambda prompt, call_session, timeout: [
            sys.executable,
            str(script),
            call_session.cli_session_id,
        ],
    )
    monkeypatch.setattr(
        agent,
        "_get_subprocess_env",
        lambda call_session: {
            **os.environ,
            "FAKE_OPENCLAW_HOME": str(tmp_path),
        },
    )

    result = agent.call("feedback", session, timeout=30)

    assert result["completion_status"] == "completed"
    assert result["response"] == "new feedback answer"
    assert result["cli_session_recovered"] is True


def test_openclaw_call_does_not_recover_from_tool_use_intermediate_message(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
    session_dir.mkdir(parents=True)
    script = tmp_path / "fake_openclaw_tool_use_then_stop.py"
    script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

home = Path(os.environ["FAKE_OPENCLAW_HOME"])
session_id = sys.argv[1]
session_file = home / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
session_file.parent.mkdir(parents=True, exist_ok=True)
tool_use = {
    "type": "message",
    "message": {
        "role": "assistant",
        "stopReason": "toolUse",
        "content": [{"type": "text", "text": "intermediate tool plan"}],
    },
}
final = {
    "type": "message",
    "message": {
        "role": "assistant",
        "stopReason": "stop",
        "content": [{"type": "text", "text": "final after tools"}],
    },
}
with session_file.open("w", encoding="utf-8") as f:
    f.write(json.dumps(tool_use) + "\\n")
    f.flush()
    time.sleep(0.3)
    f.write(json.dumps(final) + "\\n")
    f.flush()
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )

    agent = OpenClawAgentAdapter({
        "runtime": {
            "session_completion_poll_interval": 0.05,
            "session_completion_exit_grace_seconds": 0.05,
            "session_completion_terminate_grace_seconds": 0.5,
        }
    })
    agent._temp_home = str(tmp_path)
    session = agent.build_session_spec(
        phase="test_run_1",
        domain="code_implementation",
        split="test",
        task={"name": "2757"},
        trial=1,
    )

    monkeypatch.setattr(
        agent,
        "_build_cli_cmd",
        lambda prompt, call_session, timeout: [
            sys.executable,
            str(script),
            call_session.cli_session_id,
        ],
    )
    monkeypatch.setattr(
        agent,
        "_get_subprocess_env",
        lambda call_session: {
            **os.environ,
            "FAKE_OPENCLAW_HOME": str(tmp_path),
        },
    )

    result = agent.call("prompt", session, timeout=30)

    assert result["completion_status"] == "completed"
    assert result["response"] == "final after tools"
    assert result["cli_session_recovered"] is True


def test_openclaw_call_stops_when_trajectory_records_idle_timeout(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / ".openclaw" / "agents" / "main" / "sessions"
    session_dir.mkdir(parents=True)
    script = tmp_path / "fake_openclaw_trajectory_error.py"
    script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

home = Path(os.environ["FAKE_OPENCLAW_HOME"])
session_id = sys.argv[1]
session_file = home / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
trajectory_file = session_file.with_name(f"{session_id}.trajectory.jsonl")
session_file.parent.mkdir(parents=True, exist_ok=True)
session_entry = {
    "type": "message",
    "message": {
        "role": "assistant",
        "stopReason": "toolUse",
        "content": [{"type": "text", "text": "partial tool-use response"}],
    },
}
ended = {
    "type": "session.ended",
    "data": {
        "status": "error",
        "timedOut": True,
        "idleTimedOut": True,
        "promptError": "LLM idle timeout (120s)",
    },
}
session_file.write_text(json.dumps(session_entry) + "\\n", encoding="utf-8")
trajectory_file.write_text(json.dumps(ended) + "\\n", encoding="utf-8")
time.sleep(30)
""".lstrip(),
        encoding="utf-8",
    )

    agent = OpenClawAgentAdapter({
        "runtime": {
            "session_completion_poll_interval": 0.05,
            "session_completion_terminate_grace_seconds": 0.5,
        }
    })
    agent._temp_home = str(tmp_path)
    session = agent.build_session_spec(
        phase="test_run_1",
        domain="software_engineering",
        split="test",
        task={"name": "django__django-16819"},
        trial=1,
    )

    monkeypatch.setattr(
        agent,
        "_build_cli_cmd",
        lambda prompt, call_session, timeout: [
            sys.executable,
            str(script),
            call_session.cli_session_id,
        ],
    )
    monkeypatch.setattr(
        agent,
        "_get_subprocess_env",
        lambda call_session: {
            **os.environ,
            "FAKE_OPENCLAW_HOME": str(tmp_path),
        },
    )

    start = time.time()
    result = agent.call("prompt", session, timeout=30)

    assert time.time() - start < 3
    assert result["completion_status"] == "timeout"
    assert result["method"] == "cli_trajectory"
    assert result["response"] == "partial tool-use response"
    assert result["cli_terminated_after_trajectory_end"] is True
