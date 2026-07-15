import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.memory_lifecycle import CommandMemoryLifecycle
from agentbench.run_agent_eval import _finish_memory_lifecycle


def _config(tmp_path: Path, *, agent: str = "hermes") -> dict:
    home_name = "hermes" if agent == "hermes" else "openclaw"
    home_key = "HERMES_HOME" if agent == "hermes" else "OPENCLAW_HOME"
    return {
        "kind": "memory_lifecycle",
        "plugin": "fake-memory",
        "agent": agent,
        "backup_dir": str(tmp_path / "backups"),
        "backup_file_template": "@backup_dir@/trained-@run_id@.db",
        "global_backup_file_template": "@backup_dir@/global-@run_id@.db",
        "env": {
            home_key: f"@run_dir@/runtime/{home_name}",
            "PLUGIN_HOME": f"${home_key}/plugin",
        },
        "commands": {
            "prepare_global_snapshot": 'printf global > "@global_backup_file@"',
            "clear": "true",
            "wait_settle": 'printf %s "$OMNIMEMEVAL_EXPECTED_TRIALS" > "@run_dir@/expected"',
            "backup": 'printf trained > "@backup_file@"',
            "restore": "true",
            "cleanup": "true",
            "finalize": "true",
        },
    }


def _lifecycle(tmp_path: Path, config: dict | None = None) -> CommandMemoryLifecycle:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return CommandMemoryLifecycle(
        config=config or _config(tmp_path),
        project_dir=ROOT,
        run_dir=run_dir,
        run_id="run-1",
        version="version-1",
    )


@pytest.mark.parametrize(
    "missing_stage",
    ["prepare_global_snapshot", "clear", "backup", "restore", "cleanup"],
)
def test_required_lifecycle_stage_must_be_present_and_non_empty(tmp_path, missing_stage):
    config = _config(tmp_path)
    config["commands"][missing_stage] = []

    with pytest.raises(ValueError, match=missing_stage):
        _lifecycle(tmp_path, config)


def test_global_snapshot_uses_an_independent_file_and_expected_trials_env(tmp_path):
    lifecycle = _lifecycle(tmp_path)

    global_file = lifecycle.prepare_global_snapshot("reasoning")
    trained_file = lifecycle.backup("reasoning")
    lifecycle.wait_settle("reasoning", expected_trials=5)

    assert global_file.name == "global-run-1.db"
    assert trained_file.name == "trained-run-1.db"
    assert global_file.read_text(encoding="utf-8") == "global"
    assert trained_file.read_text(encoding="utf-8") == "trained"
    assert (tmp_path / "run" / "expected").read_text(encoding="utf-8") == "5"


def test_runtime_env_is_rendered_for_adapters_and_restored_exactly(tmp_path, monkeypatch):
    original_home = tmp_path / "custom-hermes"
    monkeypatch.setenv("HERMES_HOME", str(original_home))
    monkeypatch.delenv("PLUGIN_HOME", raising=False)
    monkeypatch.delenv("OMNIMEMEVAL_ORIGINAL_HERMES_HOME", raising=False)
    lifecycle = _lifecycle(tmp_path)

    activated = lifecycle.activate_runtime_env()

    run_home = tmp_path / "run" / "runtime" / "hermes"
    assert activated["HERMES_HOME"] == str(run_home)
    assert os.environ["HERMES_HOME"] == str(run_home)
    assert os.environ["PLUGIN_HOME"] == str(run_home / "plugin")
    assert os.environ["OMNIMEMEVAL_ORIGINAL_HERMES_HOME"] == str(original_home)

    lifecycle.restore_runtime_env()

    assert os.environ["HERMES_HOME"] == str(original_home)
    assert "PLUGIN_HOME" not in os.environ
    assert "OMNIMEMEVAL_ORIGINAL_HERMES_HOME" not in os.environ


def test_finalize_writes_manifest_even_when_finalize_command_fails(tmp_path):
    config = _config(tmp_path)
    config["commands"]["finalize"] = "exit 23"
    lifecycle = _lifecycle(tmp_path, config)

    with pytest.raises(RuntimeError, match="returncode=23"):
        lifecycle.finalize("reasoning")

    manifest = json.loads(lifecycle.manifest_file.read_text(encoding="utf-8"))
    assert manifest["plugin"] == "fake-memory"
    assert manifest["run_id"] == "run-1"


def test_wait_settle_rejects_invalid_expected_trial_count(tmp_path):
    lifecycle = _lifecycle(tmp_path)

    with pytest.raises(ValueError, match="expected_trials"):
        lifecycle.wait_settle("reasoning", expected_trials=0)


class _FailingFinalizerLifecycle:
    def __init__(self):
        self.calls = []

    def finalize(self, domain):
        self.calls.append(("finalize", domain))
        raise RuntimeError("finalize failed")

    def cleanup(self, domain, snapshot):
        self.calls.append(("cleanup", domain, snapshot))

    def restore_runtime_env(self):
        self.calls.append(("restore_runtime_env",))


def test_lifecycle_teardown_attempts_cleanup_after_finalize_failure(tmp_path):
    lifecycle = _FailingFinalizerLifecycle()
    snapshot = tmp_path / "global.db"

    with pytest.raises(RuntimeError, match="finalize failed"):
        _finish_memory_lifecycle(
            lifecycle,
            domain="reasoning",
            global_snapshot=snapshot,
            runtime_env_active=True,
            primary_error=None,
        )

    assert lifecycle.calls == [
        ("finalize", "reasoning"),
        ("cleanup", "reasoning", snapshot),
        ("restore_runtime_env",),
    ]


def test_lifecycle_teardown_does_not_hide_primary_interrupt(tmp_path):
    lifecycle = _FailingFinalizerLifecycle()
    primary = KeyboardInterrupt()

    _finish_memory_lifecycle(
        lifecycle,
        domain="reasoning",
        global_snapshot=tmp_path / "global.db",
        runtime_env_active=True,
        primary_error=primary,
    )

    assert [call[0] for call in lifecycle.calls] == [
        "finalize",
        "cleanup",
        "restore_runtime_env",
    ]


def test_lifecycle_teardown_cleans_run_state_when_snapshot_preparation_failed():
    lifecycle = _FailingFinalizerLifecycle()

    _finish_memory_lifecycle(
        lifecycle,
        domain="reasoning",
        global_snapshot=None,
        runtime_env_active=True,
        primary_error=RuntimeError("snapshot failed"),
    )

    assert lifecycle.calls == [
        ("finalize", "reasoning"),
        ("cleanup", "reasoning", None),
        ("restore_runtime_env",),
    ]
