from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.agentbench.plugin_lifecycle import hermes_mem0_ctl
from scripts.agentbench.plugin_lifecycle import mem0_openclaw_ctl


def _openclaw_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "plugins": {
                    "enabled": True,
                    "allow": ["openclaw-mem0"],
                    "slots": {"memory": "openclaw-mem0"},
                    "entries": {
                        "openclaw-mem0": {
                            "enabled": True,
                            "config": {
                                "mode": "open-source",
                                "userId": "user-owned",
                                "autoRecall": True,
                                "autoCapture": False,
                                "skills": {"triage": {"enabled": True}},
                                "oss": {
                                    "vectorStore": {
                                        "provider": "qdrant",
                                        "config": {
                                            "url": "http://127.0.0.1:6333",
                                            "collectionName": "mem0",
                                        },
                                    }
                                },
                            },
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_openclaw_prepare_scopes_user_and_test_disables_writes(tmp_path, monkeypatch):
    config = tmp_path / "openclaw.json"
    state = tmp_path / "mode.json"
    _openclaw_config(config)
    monkeypatch.setenv("OPENCLAW_CONFIG", str(config))
    monkeypatch.setenv("MEM0_MODE_STATE", str(state))
    monkeypatch.setenv("MEM0_EVAL_USER_ID", "omnimemeval:run-1")

    mem0_openclaw_ctl.prepare()
    prepared = mem0_openclaw_ctl.plugin_config()
    assert prepared["userId"] == "omnimemeval:run-1"

    mem0_openclaw_ctl.set_mode("train")
    train = mem0_openclaw_ctl.plugin_config()
    assert train["skills"]["triage"]["enabled"] is False
    assert train["autoRecall"] is False
    assert train["autoCapture"] is False

    mem0_openclaw_ctl.set_mode("test")
    test = mem0_openclaw_ctl.plugin_config()
    assert test["autoRecall"] is True
    assert test["autoCapture"] is False
    assert test["skills"]["triage"]["enabled"] is False


def test_openclaw_controller_refuses_non_run_scoped_user(monkeypatch):
    monkeypatch.setenv("MEM0_EVAL_USER_ID", "existing-user")
    with pytest.raises(RuntimeError, match="non-run-scoped"):
        mem0_openclaw_ctl.eval_user_id()


def test_openclaw_backup_rejects_empty_primary_collection(tmp_path, monkeypatch):
    config = tmp_path / "openclaw.json"
    _openclaw_config(config)
    monkeypatch.setenv("OPENCLAW_CONFIG", str(config))
    monkeypatch.setenv("MEM0_EVAL_USER_ID", "omnimemeval:run-1")
    monkeypatch.setattr(mem0_openclaw_ctl, "wait_stable", lambda *_: None)
    monkeypatch.setattr(mem0_openclaw_ctl, "managed_collections", lambda: ["mem0"])
    monkeypatch.setattr(
        mem0_openclaw_ctl,
        "scroll_all",
        lambda collection, *, vectors: [],
    )

    with pytest.raises(RuntimeError, match="empty OpenClaw Mem0"):
        mem0_openclaw_ctl.backup(tmp_path / "backup.tar.gz")


def test_openclaw_ingest_submits_verified_training_result(tmp_path, monkeypatch):
    config = tmp_path / "openclaw.json"
    _openclaw_config(config)
    node_path = tmp_path / "npm" / "node_modules"
    node_path.mkdir(parents=True)
    train_dir = tmp_path / "train"
    result_dir = train_dir / "omni_35__trial_1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "omni_35",
                "domain": "reasoning",
                "agent_result": {
                    "completion_status": "completed",
                    "response": "Yes",
                },
                "verifier_result": {
                    "reward": 1.0,
                    "expected": "verified construction",
                    "actual": "Yes",
                    "feedback": "Correct",
                },
                "feedback_result": {
                    "completion_status": "completed",
                    "response": "Use the concrete construction next time.",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_CONFIG", str(config))
    monkeypatch.setenv("MEM0_EVAL_USER_ID", "omnimemeval:run-1")
    monkeypatch.setenv("MEM0_NODE_PATH", str(node_path))
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"ok":true,"training_results":1,"memories":1}\n',
            stderr="",
        )

    monkeypatch.setattr(mem0_openclaw_ctl.subprocess, "run", fake_run)

    mem0_openclaw_ctl.ingest_training(train_dir)

    assert captured["command"][:2] == ["node", "-e"]
    assert "infer: false" in captured["command"][2]
    assert captured["command"][-1] == "omnimemeval:run-1"
    assert "verified construction" in captured["input"]
    assert captured["env"]["NODE_PATH"] == str(node_path)


def test_hermes_prepare_rewrites_only_run_owned_paths(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    home = run_dir / "runtime" / "hermes"
    home.mkdir(parents=True)
    (home / "mem0.json").write_text(
        json.dumps(
            {
                "mode": "oss",
                "user_id": "root",
                "agent_id": "hermes",
                "oss": {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {"path": "/root/.hermes/mem0_qdrant"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OMNIMEMEVAL_RUN_DIR", str(run_dir))
    monkeypatch.setenv("MEM0_DIR", str(run_dir / "runtime" / "mem0"))
    monkeypatch.setenv(
        "HERMES_MEM0_QDRANT_PATH",
        str(run_dir / "runtime" / "mem0-qdrant"),
    )
    monkeypatch.setenv("MEM0_EVAL_USER_ID", "omnimemeval:run-1")

    hermes_mem0_ctl.prepare()

    value = json.loads((home / "mem0.json").read_text(encoding="utf-8"))
    assert value["user_id"] == "omnimemeval:run-1"
    assert value["agent_id"] == "omnimemeval-hermes"
    assert value["oss"]["vector_store"]["config"]["path"] == str(
        run_dir / "runtime" / "mem0-qdrant"
    )


def test_hermes_controller_refuses_data_path_outside_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNIMEMEVAL_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("HERMES_MEM0_QDRANT_PATH", str(tmp_path / "user-data"))
    with pytest.raises(RuntimeError, match="outside run directory"):
        hermes_mem0_ctl.qdrant_path()


def test_hermes_ingest_uses_mem0_sdk_without_inference(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    home = run_dir / "runtime" / "hermes"
    home.mkdir(parents=True)
    (home / "mem0.json").write_text(
        json.dumps(
            {
                "mode": "oss",
                "user_id": "omnimemeval:run-1",
                "agent_id": "omnimemeval-hermes",
                "oss": {"vector_store": {"provider": "qdrant", "config": {}}},
            }
        ),
        encoding="utf-8",
    )
    result_dir = run_dir / "train" / "omni_35__trial_1"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "omni_35",
                "domain": "reasoning",
                "agent_result": {
                    "completion_status": "completed",
                    "response": "Yes",
                },
                "verifier_result": {
                    "reward": 1.0,
                    "expected": "verified construction",
                    "actual": "Yes",
                    "feedback": "Correct",
                },
                "feedback_result": {"response": "Use the construction."},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OMNIMEMEVAL_RUN_DIR", str(run_dir))
    monkeypatch.setenv("MEM0_DIR", str(run_dir / "runtime" / "mem0"))
    monkeypatch.setenv(
        "HERMES_MEM0_QDRANT_PATH",
        str(run_dir / "runtime" / "mem0-qdrant"),
    )
    monkeypatch.setenv("MEM0_EVAL_USER_ID", "omnimemeval:run-1")
    monkeypatch.setattr(hermes_mem0_ctl, "hermes_python", lambda: "/hermes/python")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"ok": true, "training_results": 1, "memories": 1}\n',
            stderr="",
        )

    monkeypatch.setattr(hermes_mem0_ctl.subprocess, "run", fake_run)

    hermes_mem0_ctl.ingest_training(run_dir / "train")

    assert captured["command"][0] == "/hermes/python"
    assert "infer=False" in captured["command"][2]
    assert "verified construction" in captured["input"]


def test_hermes_training_memories_excludes_retry_and_nested_results(tmp_path):
    train_dir = tmp_path / "train"

    def write_result(relative_dir: str, task_name: str) -> None:
        result_dir = train_dir / relative_dir
        result_dir.mkdir(parents=True)
        (result_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_name,
                    "domain": "reasoning",
                    "agent_result": {
                        "completion_status": "completed",
                        "response": "answer",
                    },
                    "verifier_result": {"reward": 1.0, "expected": "expected"},
                }
            ),
            encoding="utf-8",
        )

    write_result("omni_35__trial_1", "final")
    write_result("omni_35__trial_1_retry1", "retry")
    write_result("artifacts/nested", "nested")

    records = hermes_mem0_ctl.training_memories(train_dir)

    assert len(records) == 1
    assert "case final" in records[0]
    assert "case retry" not in records[0]
