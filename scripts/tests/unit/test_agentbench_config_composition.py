import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.config import load_yaml
from agentbench.memory_lifecycle import CommandMemoryLifecycle
from agentbench.run_agent_eval import (
    _compose_agent_config,
    _default_memory_plugin_config,
    _load_profile_config,
    _validate_config_identity,
)


def _args(agent, memory_plugin=None, profile=None, profile_config=None):
    return argparse.Namespace(
        agent=agent,
        memory_plugin=memory_plugin,
        profile=profile,
        profile_config=profile_config,
    )


@pytest.mark.parametrize("agent", ["openclaw", "hermes"])
def test_memos_resolves_runtime_specific_profile_and_lifecycle(agent):
    lifecycle_path = _default_memory_plugin_config(agent, "memos")
    lifecycle = load_yaml(lifecycle_path)
    profile_path, profile, profile_name = _load_profile_config(
        _args(agent, memory_plugin="memos"), lifecycle
    )

    assert lifecycle_path == ROOT / "configs" / "agentbench" / "memory_plugins" / "memos" / "lifecycle" / f"{agent}.yaml"
    assert lifecycle["plugin"] == "memos"
    assert lifecycle["agent"] == agent
    assert lifecycle["env"]["MEMOS_HOME"] == lifecycle["env"]["MEMOS_PLUGIN_HOME"]
    assert lifecycle["env"]["MEMOS_DB"].startswith(lifecycle["env"]["MEMOS_HOME"])
    assert profile_path == ROOT / "configs" / "agentbench" / "profiles" / agent / "memos.yaml"
    assert profile_name == "memos"
    assert profile["agent"] == agent


def test_explicit_lifecycle_infers_profile_from_plugin():
    lifecycle = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    path, _, name = _load_profile_config(_args("hermes"), lifecycle)

    assert name == "memos"
    assert path.name == "memos.yaml"
    assert path.parent.name == "hermes"


def test_rejects_runtime_mismatch_before_execution():
    lifecycle = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    with pytest.raises(SystemExit, match="targets agent 'hermes'"):
        _validate_config_identity(
            lifecycle,
            kind="memory_lifecycle",
            agent="openclaw",
            plugin="memos",
        )


def test_profiles_keep_runtime_specific_patches_separate():
    openclaw = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "openclaw" / "memos.yaml")
    hermes = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "hermes" / "memos.yaml")
    openclaw_config = _compose_agent_config({}, openclaw)
    hermes_config = _compose_agent_config({}, hermes)

    assert "openclaw_config_patch" in openclaw_config
    assert "hermes_config_patch" not in openclaw_config
    assert "hermes_config_patch" in hermes_config
    assert "openclaw_config_patch" not in hermes_config
    assert hermes_config["runtime"]["home_links"] == ["memos-plugin"]
    assert hermes_config["runtime"]["train_memory_provider"] == "omnimemeval_memos"
    assert hermes_config["runtime"]["verify_memos_capture"] is True
    assert hermes_config["memory"]["memory_enabled"] is False
    assert hermes_config["memory"]["user_profile_enabled"] is False


def test_memos_lifecycles_use_private_sqlite_only_backups():
    required = {
        "prepare_global_snapshot",
        "clear",
        "wait_settle",
        "backup",
        "restore",
        "cleanup",
    }
    for agent in ("openclaw", "hermes"):
        config = load_yaml(_default_memory_plugin_config(agent, "memos"))
        assert required <= config["commands"].keys()
        assert config["backup_file_template"].endswith(".sqlite3")
        assert config["global_backup_file_template"].endswith(".sqlite3")
        assert config["backup_dir"] == "@run_dir@/memory_backups"
        assert "tar " not in config["commands"]["backup"]
        assert ".auth.json" not in config["commands"]["backup"]
        assert "config.yaml" not in config["commands"]["backup"]
        assert "pgrep" not in "\n".join(config["commands"].values())


def test_hermes_memos_clear_does_not_terminate_its_lifecycle_shell(tmp_path, monkeypatch):
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    hermes_home = tmp_path / "user-hermes"
    plugin_home = hermes_home / "memos-plugin"
    (plugin_home / "dist").mkdir(parents=True)
    (plugin_home / "dist" / "bridge.cjs").write_text("", encoding="utf-8")
    (plugin_home / "config.yaml").write_text("version: 1\n", encoding="utf-8")
    (hermes_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config["backup_dir"] = str(tmp_path / "backups")
    config["env"].update({
        "MEMOS_DAEMON_TERM_TIMEOUT": "1",
        "MEMOS_START_DAEMON": "0",
        "MEMOS_PREWARM": "0",
        "MEMOS_FINALIZE_TIMEOUT": "0",
    })
    run_dir = tmp_path / "run"
    lifecycle = CommandMemoryLifecycle(
        config=config,
        project_dir=ROOT,
        run_dir=run_dir,
        run_id="clear-regression",
        version="test",
    )

    lifecycle.activate_runtime_env()
    try:
        lifecycle.validate("reasoning")
        snapshot = lifecycle.prepare_global_snapshot("reasoning")
        lifecycle.clear("reasoning")

        assert snapshot.stat().st_mode & 0o777 == 0o600
        assert (run_dir / "runtime" / "hermes" / "memos-plugin" / "data").is_dir()

        with pytest.raises(RuntimeError, match="stage=wait_settle"):
            lifecycle.wait_settle("reasoning", expected_trials=1)
        assert "Hermes MemOS DB is missing or empty" in lifecycle.log_file.read_text()

        db = run_dir / "runtime" / "hermes" / "memos-plugin" / "data" / "memos.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(
                "CREATE TABLE episodes (id TEXT, status TEXT, trace_ids_json TEXT);"
                "CREATE TABLE traces (id TEXT);"
                "CREATE TABLE api_logs (id TEXT);"
                "CREATE TABLE embedding_retry_queue (id TEXT, status TEXT);"
            )
            conn.commit()
        finally:
            conn.close()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="stage=wait_settle"):
            lifecycle.wait_settle("reasoning", expected_trials=1)
        assert time.monotonic() - started < 3
        assert "capture bridges have exited but episode gate is unsatisfied" in lifecycle.log_file.read_text()

        old_path = os.environ["PATH"]
        no_sqlite_bin = tmp_path / "no-sqlite-bin"
        no_sqlite_bin.mkdir()
        (no_sqlite_bin / "bash").symlink_to("/usr/bin/bash")
        monkeypatch.setenv("PATH", str(no_sqlite_bin))
        with pytest.raises(RuntimeError, match="stage=wait_settle"):
            lifecycle.wait_settle("reasoning", expected_trials=1)
        monkeypatch.setenv("PATH", old_path)
        assert "sqlite3 is required to verify" in lifecycle.log_file.read_text()
    finally:
        lifecycle.cleanup("reasoning")
        lifecycle.restore_runtime_env()
