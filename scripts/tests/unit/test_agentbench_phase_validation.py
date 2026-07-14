import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.runner import assert_phase_succeeded


def _write_phase(tmp_path: Path, result: dict) -> Path:
    (tmp_path / "phase_config.json").write_text(
        json.dumps({"tasks": 1, "trials": 1}),
        encoding="utf-8",
    )
    trial_dir = tmp_path / "task_a__trial_1"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return tmp_path


def _successful_result() -> dict:
    return {
        "agent_result": {"completion_status": "completed"},
        "verifier_result": {"reward": 0.0},
        "feedback_result": {"completion_status": "completed"},
        "plugin_feedback_result": {"status": "submitted"},
        "exception_info": None,
    }


def test_phase_validation_accepts_completed_wrong_answer(tmp_path):
    phase_dir = _write_phase(tmp_path, _successful_result())

    assert_phase_succeeded(
        phase_dir,
        require_feedback=True,
        require_plugin_feedback=True,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("agent_result", {"completion_status": "timeout"}, "agent completion_status"),
        ("feedback_result", {"completion_status": "error"}, "feedback completion_status"),
        ("plugin_feedback_result", {"status": "error"}, "plugin feedback status"),
        ("exception_info", {"type": "RuntimeError", "message": "setup failed"}, "setup failed"),
    ],
)
def test_phase_validation_raises_for_pipeline_failures(tmp_path, field, value, message):
    result = _successful_result()
    result[field] = value
    phase_dir = _write_phase(tmp_path, result)

    with pytest.raises(RuntimeError, match=message):
        assert_phase_succeeded(
            phase_dir,
            require_feedback=True,
            require_plugin_feedback=True,
        )


def test_phase_validation_raises_when_final_results_are_missing(tmp_path):
    (tmp_path / "phase_config.json").write_text(
        json.dumps({"tasks": 2, "trials": 1}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="expected 2 final result"):
        assert_phase_succeeded(tmp_path)


def test_phase_validation_raises_when_no_tasks_were_selected(tmp_path):
    (tmp_path / "phase_config.json").write_text(
        json.dumps({"tasks": 0, "trials": 1}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no task trials"):
        assert_phase_succeeded(tmp_path)
