import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.memory_lifecycle import CommandMemoryLifecycle
from agentbench.run_agent_eval import (
    _finish_memory_lifecycle,
    _settle_and_audit_memory_phase,
)


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
        "resume_file_template": "@backup_dir@/resume-@run_id@.db",
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
            "checkpoint": 'printf checkpoint > "@backup_file@"',
            "cleanup": "true",
            "finalize": "true",
        },
    }


def _lifecycle(
    tmp_path: Path,
    config: dict | None = None,
    *,
    parallel: int = 1,
) -> CommandMemoryLifecycle:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return CommandMemoryLifecycle(
        config=config or _config(tmp_path),
        project_dir=ROOT,
        run_dir=run_dir,
        run_id="run-1",
        version="version-1",
        parallel=parallel,
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


def test_lifecycle_python_token_uses_the_framework_interpreter(tmp_path):
    config = _config(tmp_path)
    config["commands"]["validate"] = (
        '"@python@" -c \'import pathlib,sys; '
        'pathlib.Path(sys.argv[1]).write_text(sys.executable)\' '
        '"@run_dir@/python-path"'
    )
    lifecycle = _lifecycle(tmp_path, config)

    lifecycle.validate("reasoning")

    assert (tmp_path / "run" / "python-path").read_text() == sys.executable


def test_checkpoint_uses_stable_resume_file(tmp_path):
    lifecycle = _lifecycle(tmp_path)

    checkpoint = lifecycle.checkpoint("reasoning")

    assert checkpoint == tmp_path / "backups" / "resume-run-1.db"
    assert checkpoint.read_text(encoding="utf-8") == "checkpoint"


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
    assert os.environ["OMNIMEMEVAL_PARALLEL"] == "1"

    lifecycle.restore_runtime_env()

    assert os.environ["HERMES_HOME"] == str(original_home)
    assert "PLUGIN_HOME" not in os.environ
    assert "OMNIMEMEVAL_ORIGINAL_HERMES_HOME" not in os.environ
    assert "OMNIMEMEVAL_PARALLEL" not in os.environ


def test_runtime_env_exposes_requested_parallelism(tmp_path):
    lifecycle = _lifecycle(tmp_path, parallel=5)

    assert lifecycle.runtime_env()["OMNIMEMEVAL_PARALLEL"] == "5"


def test_phase_session_audit_requires_exact_trial_to_session_mapping(tmp_path):
    db_path = tmp_path / "memos.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "CREATE TABLE episodes ("
        "id TEXT PRIMARY KEY, session_id TEXT, status TEXT, trace_ids_json TEXT, meta_json TEXT);"
        "CREATE TABLE traces (id TEXT PRIMARY KEY, session_id TEXT, episode_id TEXT);"
    )
    expected = {
        "omnimemeval:train:reasoning:train:task-a:trial:1": "session-a",
        "omnimemeval:train:reasoning:train:task-b:trial:1": "session-b",
    }
    for index, (trial_key, session_id) in enumerate(expected.items()):
        episode_id = f"episode-{index}"
        trace_id = f"trace-{index}"
        conn.execute(
            "INSERT INTO episodes VALUES (?, ?, 'closed', ?, ?)",
            (
                episode_id,
                session_id,
                json.dumps([trace_id]),
                json.dumps({"contextHints": {"omnimemevalTrialKey": trial_key}}),
            ),
        )
        conn.execute("INSERT INTO traces VALUES (?, ?, ?)", (trace_id, session_id, episode_id))
    conn.commit()
    conn.close()

    lifecycle = _lifecycle(tmp_path)
    lifecycle.audit_phase_sessions(db_path, expected)

    with pytest.raises(RuntimeError, match="session audit"):
        lifecycle.audit_phase_sessions(
            db_path,
            {**expected, "omnimemeval:train:reasoning:train:task-c:trial:1": "session-c"},
        )

    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE episodes SET status = 'open' WHERE session_id = 'session-a'")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="status=open"):
        lifecycle.audit_phase_sessions(db_path, expected)


def test_memory_phase_settles_before_audit_with_retained_sessions(tmp_path):
    phase_dir = tmp_path / "test_run_1"
    trial_dir = phase_dir / "task-a__trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps({
            "session": {
                "semantic_session_id": (
                    "omnimemeval:test_run_1:reasoning:test:task-a:trial:1"
                ),
            },
            "agent_result": {"hermes_session_id": "session-test"},
        }),
        encoding="utf-8",
    )

    events = []

    class FakeLifecycle:
        def runtime_env(self):
            return {"MEMOS_DB": str(tmp_path / "memos.db")}

        def wait_settle(self, domain, *, expected_trials):
            events.append(("settle", domain, expected_trials))

        def audit_phase_sessions(self, db_path, expected):
            events.append(("audit", db_path, expected))

    captured = _settle_and_audit_memory_phase(
        lifecycle=FakeLifecycle(),
        execution_config={"phase_session_audit": True},
        domain="reasoning",
        phase_dir=phase_dir,
        phase_trials=1,
        retained_trials=5,
    )

    assert captured == 1
    assert events == [
        ("settle", "reasoning", 6),
        (
            "audit",
            str(tmp_path / "memos.db"),
            {
                "omnimemeval:test_run_1:reasoning:test:task-a:trial:1": (
                    "session-test"
                ),
            },
        ),
    ]


def test_memory_phase_excludes_skipped_trial_from_settle_and_audit(tmp_path):
    phase_dir = tmp_path / "train"
    trial_dir = phase_dir / "task-a__trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps({
            "trial_status": "skipped",
            "skip_reason": "retries_exhausted:timeout",
            "agent_result": {"completion_status": "timeout"},
        }),
        encoding="utf-8",
    )
    events = []

    class FakeLifecycle:
        def runtime_env(self):
            return {"MEMOS_DB": str(tmp_path / "memos.db")}

        def wait_settle(self, domain, *, expected_trials):
            events.append(("settle", domain, expected_trials))

        def audit_phase_sessions(self, db_path, expected):
            events.append(("audit", db_path, expected))

    captured = _settle_and_audit_memory_phase(
        lifecycle=FakeLifecycle(),
        execution_config={"phase_session_audit": True},
        domain="reasoning",
        phase_dir=phase_dir,
        phase_trials=1,
        retained_trials=0,
    )

    assert captured == 0
    assert events == [("settle", "reasoning", 0)]


def test_memory_phase_records_non_blocking_session_audit_warning(tmp_path):
    phase_dir = tmp_path / "train"
    trial_dir = phase_dir / "task-a__trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps({
            "session": {
                "semantic_session_id": (
                    "omnimemeval:train:reasoning:train:task-a:trial:1"
                ),
            },
            "agent_result": {"hermes_session_id": "session-a"},
        }),
        encoding="utf-8",
    )
    events = []

    class FakeLifecycle:
        def runtime_env(self):
            return {"MEMOS_DB": str(tmp_path / "memos.db")}

        def wait_settle(self, domain, *, expected_trials):
            events.append(("settle", domain, expected_trials))

        def audit_phase_sessions(self, db_path, expected):
            events.append(("audit", db_path, expected))
            raise RuntimeError("extra retry session")

    captured = _settle_and_audit_memory_phase(
        lifecycle=FakeLifecycle(),
        execution_config={"phase_session_audit": "warn"},
        domain="reasoning",
        phase_dir=phase_dir,
        phase_trials=1,
    )

    assert captured == 1
    assert [event[0] for event in events] == ["settle", "audit"]
    warning = json.loads(
        (phase_dir / "memory_session_audit_warning.json").read_text(encoding="utf-8")
    )
    assert warning["status"] == "warning"
    assert "extra retry session" in warning["message"]


def test_relative_run_dir_is_normalized_before_rendering_runtime_env(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    relative_run_dir = Path("results") / "relative-run"
    lifecycle = CommandMemoryLifecycle(
        config=_config(tmp_path),
        project_dir=project_dir,
        run_dir=relative_run_dir,
        run_id="run-1",
        version="version-1",
    )

    assert lifecycle.run_dir == project_dir / relative_run_dir
    assert lifecycle.runtime_env()["HERMES_HOME"] == str(
        project_dir / relative_run_dir / "runtime" / "hermes"
    )
    assert lifecycle.log_file == project_dir / relative_run_dir / "memory_lifecycle.log"


def test_finalize_writes_manifest_even_when_finalize_command_fails(tmp_path):
    config = _config(tmp_path)
    config["commands"]["finalize"] = "exit 23"
    lifecycle = _lifecycle(tmp_path, config)

    with pytest.raises(RuntimeError, match="returncode=23"):
        lifecycle.finalize("reasoning")

    manifest = json.loads(lifecycle.manifest_file.read_text(encoding="utf-8"))
    assert manifest["plugin"] == "fake-memory"
    assert manifest["run_id"] == "run-1"


def test_wait_settle_accepts_zero_and_rejects_negative_expected_trial_count(tmp_path):
    lifecycle = _lifecycle(tmp_path)

    lifecycle.wait_settle("reasoning", expected_trials=0)
    assert (tmp_path / "run" / "expected").read_text(encoding="utf-8") == "0"

    with pytest.raises(ValueError, match="expected_trials"):
        lifecycle.wait_settle("reasoning", expected_trials=-1)


class _FailingFinalizerLifecycle:
    def __init__(self):
        self.calls = []

    def finalize(self, domain):
        self.calls.append(("finalize", domain))
        raise RuntimeError("finalize failed")

    def cleanup(self, domain, snapshot):
        self.calls.append(("cleanup", domain, snapshot))

    def checkpoint(self, domain):
        self.calls.append(("checkpoint", domain))

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
        ("checkpoint", "reasoning"),
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
        "checkpoint",
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
        ("checkpoint", "reasoning"),
        ("cleanup", "reasoning", None),
        ("restore_runtime_env",),
    ]
