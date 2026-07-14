import argparse
import sys
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
    assert hermes_config["memory"]["memory_enabled"] is False
    assert hermes_config["memory"]["user_profile_enabled"] is False


def test_hermes_memos_clear_does_not_terminate_its_lifecycle_shell(tmp_path):
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    hermes_home = tmp_path / ".hermes"
    plugin_home = hermes_home / "memos-plugin"
    (plugin_home / "dist").mkdir(parents=True)
    (plugin_home / "dist" / "bridge.cjs").write_text("", encoding="utf-8")
    config["backup_dir"] = str(tmp_path / "backups")
    config["env"].update({
        "HERMES_HOME": str(hermes_home),
        "MEMOS_PLUGIN_HOME": str(plugin_home),
        "MEMOS_DB": str(plugin_home / "data" / "memos.db"),
        "MEMOS_DAEMON_TERM_TIMEOUT": "1",
        "MEMOS_DAEMON_START_WAIT_SECONDS": "0",
        "MEMOS_CLEAR_SETTLE_SECONDS": "0",
    })
    lifecycle = CommandMemoryLifecycle(
        config=config,
        project_dir=ROOT,
        run_dir=tmp_path / "run",
        run_id="clear-regression",
        version="test",
    )

    lifecycle.validate("reasoning")
    lifecycle.clear("reasoning")

    assert (plugin_home / "data").is_dir()
