import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.config import write_json
from agentbench.config import load_yaml
from agentbench.domains.reasoning.evaluate import verify_answer
from agentbench.domains.reasoning.omnimath import ReasoningDomain
from agentbench.runner import run_task_once, run_task_with_retry
from agentbench.session import SessionSpec
from agentbench.summary import build_summary, classify_failure, response_text_for_char_stats


def test_reasoning_exact_matches_evo_tuple_behavior():
    expected = "{(2, 1, 3), (1, 2, -3), (1, 0, 1), (0, 1, -1), (0, 0, 0)}"
    actual = r"\boxed{(0,0,0), (1,0,1), (0,1,-1), (2,1,3), (1,2,-3)}"

    result = verify_answer({"answer": expected}, actual, mode="exact")

    assert result["correct"] is False


def test_reasoning_task_filter_raises_on_no_match(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = {"_idx": 1, "problem": "1+1?", "answer": "2"}
    (data_dir / "test.jsonl").write_text(json.dumps(row) + "\n")

    domain = ReasoningDomain({"data_dir": str(data_dir)})

    with pytest.raises(ValueError, match="No reasoning tasks matched"):
        domain.load_tasks(Namespace(split="test", task="omni_999"))


def test_reasoning_llm_mode_requires_judge_config(tmp_path, monkeypatch):
    for key in ("JUDGE_API_KEY", "JUDGE_API_BASE", "EVALUATION_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    domain = ReasoningDomain({"verify_mode": "llm"})

    with pytest.raises(ValueError, match="verify_mode=llm requires"):
        domain.verify(
            {"task_id": "1", "problem": "1+1?", "answer": "2"},
            {},
            tmp_path,
            agent_result={"response": r"\boxed{2}"},
        )


def test_reasoning_llm_mode_treats_unresolved_env_placeholders_as_missing(tmp_path, monkeypatch):
    for key in ("JUDGE_API_KEY", "JUDGE_API_BASE", "EVALUATION_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    domain = ReasoningDomain({
        "verify_mode": "llm",
        "eval_api_key": "${JUDGE_API_KEY}",
        "eval_api_base": "${JUDGE_API_BASE}",
    })

    with pytest.raises(ValueError, match="verify_mode=llm requires"):
        domain.verify(
            {"task_id": "1", "problem": "1+1?", "answer": "2"},
            {},
            tmp_path,
            agent_result={"response": r"\boxed{2}"},
        )


def test_reasoning_default_config_uses_llm_judge():
    cfg = load_yaml(ROOT / "configs" / "agentbench" / "domains" / "reasoning.yaml")

    assert cfg["verify_mode"] == "llm"


def test_reasoning_passes_verifier_timeout_and_retries(tmp_path, monkeypatch):
    captured = {}

    def fake_verify_answer(task, agent_output, **kwargs):
        captured.update(kwargs)
        return {"reward": 1.0, "correct": True}

    monkeypatch.setattr(
        "agentbench.domains.reasoning.omnimath.verify_answer",
        fake_verify_answer,
    )
    domain = ReasoningDomain({
        "verify_mode": "llm",
        "eval_api_key": "key",
        "verify_timeout": 17,
        "verify_max_retries": 2,
    })
    domain.verify(
        {"task_id": "1", "problem": "1+1?", "answer": "2"},
        {},
        tmp_path,
        agent_result={"response": r"\boxed{2}"},
    )

    assert captured["timeout"] == 17.0
    assert captured["max_retries"] == 2


def test_summary_separates_pass_at_from_average_pass_rate(tmp_path):
    phase_dir = tmp_path / "phase"
    write_json(phase_dir / "task_a__trial_1" / "result.json", {
        "task_name": "task_a",
        "trial": 1,
        "agent_result": {"elapsed_sec": 1},
        "verifier_result": {"reward": 0.0},
        "token_usage": {"total": 10},
    })
    write_json(phase_dir / "task_a__trial_2" / "result.json", {
        "task_name": "task_a",
        "trial": 2,
        "agent_result": {"elapsed_sec": 1},
        "verifier_result": {"reward": 1.0},
        "token_usage": {"total": 10},
    })
    write_json(phase_dir / "task_b__trial_1" / "result.json", {
        "task_name": "task_b",
        "trial": 1,
        "agent_result": {"elapsed_sec": 1},
        "verifier_result": {"reward": 0.0},
        "token_usage": {"total": 10},
    })
    write_json(phase_dir / "task_b__trial_2" / "result.json", {
        "task_name": "task_b",
        "trial": 2,
        "agent_result": {"elapsed_sec": 1},
        "verifier_result": {"reward": 0.0},
        "token_usage": {"total": 10},
    })

    summary = build_summary(phase_dir, trials=2, pass_at=2)

    assert summary["pass@1"] == 0.0
    assert summary["pass@2"] == 0.5
    assert summary["avg_pass_rate"] == 0.25
    assert summary["per_task"]["task_a"]["avg_pass_rate"] == 0.5


def test_summary_classifies_model_placeholder_as_infra_error():
    result = {
        "agent_result": {
            "completion_status": "error",
            "error": "[Assistant reply unavailable due to model error.]",
        },
        "verifier_result": {"reward": 0.0},
    }

    assert classify_failure(result, 0.0) == "infra_error"


def test_summary_reports_infra_excluded_pass_rate_and_turns(tmp_path):
    phase_dir = tmp_path / "phase"
    write_json(phase_dir / "task_a__trial_1" / "result.json", {
        "task_name": "task_a",
        "trial": 1,
        "agent_result": {"completion_status": "error", "error": "HTTP 503", "elapsed_sec": 1},
        "verifier_result": {"reward": 0.0},
        "token_usage": {"turns": 2, "total": 10},
    })
    write_json(phase_dir / "task_b__trial_1" / "result.json", {
        "task_name": "task_b",
        "trial": 1,
        "agent_result": {"elapsed_sec": 1},
        "verifier_result": {"reward": 1.0},
        "token_usage": {"turns": 4, "total": 10},
    })

    summary = build_summary(phase_dir, trials=1)

    assert summary["failure_counts"] == {"infra_error": 1, "resolved": 1}
    assert summary["infra_excluded"]["tasks"] == 1
    assert summary["infra_excluded"]["excluded_tasks"] == 1
    assert summary["infra_excluded"]["pass@1"] == {"mean": 1.0, "stderr": 0.0}
    assert summary["avg_turns"] == 3.0


def test_response_chars_use_final_session_answer_for_generic_agents(tmp_path):
    trial_dir = tmp_path / "task__trial_1"
    trial_dir.mkdir()
    (trial_dir / "session.jsonl").write_text(
        "\n".join([
            '{"role":"user","content":"prompt"}',
            '{"role":"assistant","content":"working","reasoning_content":"hidden","tool_calls":[{"function":{"name":"exec"}}]}',
            '{"role":"tool","content":"tool output"}',
            '{"role":"assistant","content":"Final answer: \\\\boxed{12}","reasoning_content":"hidden"}',
        ])
        + "\n"
    )
    result = {"agent_result": {"response": "stdout log " * 2000}}

    assert response_text_for_char_stats(result, trial_dir) == "Final answer: \\boxed{12}"


def test_response_chars_use_all_openclaw_text_without_thinking(tmp_path):
    trial_dir = tmp_path / "task__trial_1"
    trial_dir.mkdir()
    (trial_dir / "session.jsonl").write_text(
        '{"type":"message","message":{"role":"assistant","stopReason":"stop",'
        '"content":[{"type":"thinking","thinking":"long hidden reasoning"},'
        '{"type":"text","text":"Work before tool"}]}}\n'
        '{"type":"message","message":{"role":"tool","content":[{"type":"text","text":"tool output"}]}}\n'
        '{"type":"message","message":{"role":"assistant","stopReason":"stop",'
        '"content":[{"type":"text","text":"Concise final answer"}]}}\n'
    )
    result = {"agent": "openclaw", "agent_result": {"response": "Saved truncated answer"}}

    assert response_text_for_char_stats(result, trial_dir) == "Work before tool\nConcise final answer"


def test_response_chars_use_all_hermes_assistant_text_with_reasoning(tmp_path):
    trial_dir = tmp_path / "task__trial_1"
    trial_dir.mkdir()
    (trial_dir / "session.jsonl").write_text(
        "\n".join([
            '{"role":"user","content":"prompt"}',
            '{"role":"assistant","content":"first visible","reasoning_content":"hidden"}',
            '{"role":"assistant","content":"","reasoning_content":"hidden only"}',
            '{"role":"assistant","content":"tool call text","tool_calls":[{"function":{"name":"exec"}}]}',
            '{"role":"tool","content":"tool output"}',
            '{"role":"assistant","content":"final visible","reasoning_content":"more hidden"}',
        ])
        + "\n"
    )
    result = {"agent": "hermes", "agent_result": {"response": "final visible\n"}}

    assert response_text_for_char_stats(result, trial_dir) == (
        "hidden\nfirst visible\nhidden only\ntool call text\nmore hidden\nfinal visible"
    )


def test_response_chars_skip_internal_context_compaction_records(tmp_path):
    trial_dir = tmp_path / "task__trial_1"
    trial_dir.mkdir()
    (trial_dir / "session.jsonl").write_text(
        "\n".join([
            '{"role":"assistant","content":"visible before compaction"}',
            '{"role":"assistant","content":"[CONTEXT COMPACTION - REFERENCE ONLY] stale internal summary"}',
            '{"role":"assistant","content":"final answer","reasoning_content":"hidden final"}',
        ])
        + "\n"
    )
    result = {"agent": "hermes", "agent_result": {"response": "final answer"}}

    assert response_text_for_char_stats(result, trial_dir) == (
        "visible before compaction\nhidden final\nfinal answer"
    )


def test_summary_response_chars_follow_evo_session_policy(tmp_path):
    phase_dir = tmp_path / "phase"
    trial_dir = phase_dir / "task_a__trial_1"
    trial_dir.mkdir(parents=True)
    (trial_dir / "session.jsonl").write_text(
        "\n".join([
            '{"role":"assistant","content":"visible","reasoning_content":"hidden"}',
            '{"role":"assistant","content":"final","reasoning_content":"more"}',
        ])
        + "\n"
    )
    write_json(trial_dir / "result.json", {
        "task_name": "task_a",
        "agent": "hermes",
        "trial": 1,
        "agent_result": {"response": "final", "response_chars": 5, "elapsed_sec": 1},
        "verifier_result": {"reward": 1.0},
        "token_usage": {"turns": 1, "total": 10},
    })

    summary = build_summary(phase_dir, trials=1)

    expected_chars = len("hidden\nvisible\nmore\nfinal")
    assert summary["avg_chars"] == float(expected_chars)
    assert summary["response_chars"] == {
        "avg": float(expected_chars),
        "median": float(expected_chars),
        "pass_avg": float(expected_chars),
        "fail_avg": 0.0,
        "empty": 0,
    }
    assert summary["per_task"]["task_a"]["trial_results"][0]["chars"] == expected_chars


class _FakeDomain:
    name = "fake"
    config = {}

    def setup(self, task, agent_name, trial):
        return {}

    def cleanup(self, task, env_info):
        pass

    def build_prompt(self, task, env_info, phase):
        return "prompt"

    def get_agent_timeout(self, task, env_info):
        return 1

    def verify(self, task, env_info, trial_dir, agent_result=None):
        return {"reward": 1.0, "correct": True}


class _FakeAgent:
    name = "fake"
    config = {}

    def __init__(self):
        self.calls = []
        self.prepared_env_info = []

    def build_session_spec(self, *, phase, domain, split, task, trial):
        return SessionSpec(
            cli_session_id=f"{task['name']}-t{trial}",
            semantic_session_id=f"{phase}:{domain}:{split}:{task['name']}:{trial}",
            source_ref="test",
        )

    def prepare_task(self, task, env_info, session):
        self.prepared_env_info.append(dict(env_info))

    def call(self, prompt, session, timeout=1):
        self.calls.append((prompt, session, timeout))
        return {"response": "x" * 10050, "completion_status": "completed"}

    def collect_session(self, session, trial_dir):
        return {"turns": 1, "input": 0, "output": 0, "total": 0, "last_stop_reason": None}

    def cleanup_task(self):
        pass


def test_runner_writes_full_response_file(tmp_path):
    agent = _FakeAgent()
    result = run_task_once(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent=agent,
        phase_dir=tmp_path,
        phase="test",
        split="test",
        trial=1,
        attempt=1,
        args=Namespace(),
    )

    trial_dir = tmp_path / "task_a__trial_1"
    saved = json.loads((trial_dir / "result.json").read_text())

    assert result["agent_result"]["response"] == "x" * 10050
    assert len(agent.calls) == 1
    assert (trial_dir / "response.txt").read_text() == "x" * 10050
    assert saved["agent_result"]["response_file"] == "response.txt"
    assert saved["agent_result"]["response_chars"] == 10050
    assert "truncated" in saved["agent_result"]["response"]


def test_runner_owns_a_clean_workspace_for_every_task_trial(tmp_path):
    class SharedWorkspaceDomain(_FakeDomain):
        def setup(self, task, agent_name, trial):
            # Domains cannot redirect the agent back to a shared directory.
            return {"workspace_dir": str(tmp_path / "shared")}

    stale_workspace = tmp_path / "task_a__trial_1" / "workspace"
    stale_workspace.mkdir(parents=True)
    (stale_workspace / "stale.txt").write_text("old attempt", encoding="utf-8")

    first_agent = _FakeAgent()
    run_task_once(
        task={"name": "task_a"},
        domain=SharedWorkspaceDomain(),
        agent=first_agent,
        phase_dir=tmp_path,
        phase="test",
        split="test",
        trial=1,
        attempt=1,
        args=Namespace(),
    )
    second_agent = _FakeAgent()
    run_task_once(
        task={"name": "task_b"},
        domain=SharedWorkspaceDomain(),
        agent=second_agent,
        phase_dir=tmp_path,
        phase="test",
        split="test",
        trial=1,
        attempt=1,
        args=Namespace(),
    )

    first_workspace = Path(first_agent.prepared_env_info[0]["workspace_dir"])
    second_workspace = Path(second_agent.prepared_env_info[0]["workspace_dir"])
    assert first_workspace == (tmp_path / "task_a__trial_1" / "workspace").resolve()
    assert second_workspace == (tmp_path / "task_b__trial_1" / "workspace").resolve()
    assert first_workspace != second_workspace
    assert first_workspace.is_dir()
    assert second_workspace.is_dir()
    assert not (first_workspace / "stale.txt").exists()


def test_train_feedback_reuses_same_session(tmp_path):
    agent = _FakeAgent()
    result = run_task_once(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent=agent,
        phase_dir=tmp_path,
        phase="train",
        split="train",
        trial=1,
        attempt=1,
        args=Namespace(train_feedback=True, feedback_timeout=7),
    )

    assert len(agent.calls) == 2
    first_prompt, first_session, first_timeout = agent.calls[0]
    feedback_prompt, feedback_session, feedback_timeout = agent.calls[1]

    assert first_prompt == "prompt"
    assert feedback_prompt.startswith("Verifier feedback for the previous attempt.")
    assert first_session is feedback_session
    assert first_session.cli_session_id == "task_a-t1"
    assert first_timeout == 1
    assert feedback_timeout == 7
    assert result["feedback_result"]["completion_status"] == "completed"

    saved = json.loads((tmp_path / "task_a__trial_1" / "result.json").read_text())
    assert saved["feedback_prompt"].startswith("Verifier feedback for the previous attempt.")
    assert saved["feedback_result"]["response_chars"] == 10050


def test_train_feedback_is_skipped_when_qa_times_out(tmp_path):
    class TimeoutAgent(_FakeAgent):
        def call(self, prompt, session, timeout=1):
            self.calls.append((prompt, session, timeout))
            return {
                "response": "",
                "completion_status": "timeout",
                "error": "subprocess timed out",
            }

    agent = TimeoutAgent()
    result = run_task_once(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent=agent,
        phase_dir=tmp_path,
        phase="train",
        split="train",
        trial=1,
        attempt=1,
        args=Namespace(
            train_feedback=True,
            feedback_timeout=7,
            plugin_structured_feedback=True,
            plugin_feedback_backend="memos",
        ),
    )

    assert len(agent.calls) == 1
    assert "feedback_prompt" not in result
    assert result["feedback_result"] == {
        "completion_status": "skipped",
        "reason": "qa_not_completed",
        "agent_completion_status": "timeout",
    }
    assert result["plugin_feedback_result"] == {
        "status": "skipped",
        "reason": "qa_not_completed",
        "backend": "memos",
    }
    assert result["memos_feedback_result"] == result["plugin_feedback_result"]

    saved = json.loads((tmp_path / "task_a__trial_1" / "result.json").read_text())
    assert saved["feedback_result"] == {
        **result["feedback_result"],
        "response_chars": 0,
    }
    assert saved["plugin_feedback_result"] == result["plugin_feedback_result"]


def test_train_retry_sends_feedback_only_after_completed_qa(tmp_path):
    agents = []

    class RetryAgent(_FakeAgent):
        def __init__(self, *, time_out):
            super().__init__()
            self.time_out = time_out

        def call(self, prompt, session, timeout=1):
            self.calls.append((prompt, session, timeout))
            if self.time_out:
                return {"response": "", "completion_status": "timeout"}
            return {"response": "answer", "completion_status": "completed"}

        def should_retry(self, result):
            if result["agent_result"]["completion_status"] == "timeout":
                return "timeout"
            return None

    def agent_factory():
        agent = RetryAgent(time_out=not agents)
        agents.append(agent)
        return agent

    result = run_task_with_retry(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent_factory=agent_factory,
        phase_dir=tmp_path,
        phase="train",
        split="train",
        trial=1,
        args=Namespace(train_feedback=True, feedback_timeout=7, max_retries=1),
    )

    assert len(agents) == 2
    assert len(agents[0].calls) == 1
    assert len(agents[1].calls) == 2
    assert result["attempt"] == 2
    assert result["feedback_result"]["completion_status"] == "completed"

    retry = json.loads(
        (tmp_path / "task_a__trial_1_retry1" / "result.json").read_text()
    )
    assert retry["feedback_result"]["completion_status"] == "skipped"
    assert retry["feedback_result"]["reason"] == "qa_not_completed"


def test_train_retry_exhaustion_marks_trial_skipped_and_does_not_send_feedback(tmp_path):
    agents = []

    class TimeoutAgent(_FakeAgent):
        def call(self, prompt, session, timeout=1):
            self.calls.append((prompt, session, timeout))
            return {"response": "", "completion_status": "timeout"}

        def should_retry(self, result):
            return "timeout"

    def agent_factory():
        agent = TimeoutAgent()
        agents.append(agent)
        return agent

    result = run_task_with_retry(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent_factory=agent_factory,
        phase_dir=tmp_path,
        phase="train",
        split="train",
        trial=1,
        args=Namespace(train_feedback=True, feedback_timeout=7, max_retries=1),
    )

    assert len(agents) == 2
    assert [len(agent.calls) for agent in agents] == [1, 1]
    assert result["trial_status"] == "skipped"
    assert result["skip_reason"] == "retries_exhausted:timeout"
    assert result["attempts_exhausted"] == 2
    assert result["feedback_result"]["reason"] == "qa_not_completed"

    saved = json.loads(
        (tmp_path / "task_a__trial_1" / "result.json").read_text()
    )
    assert saved["agent_result"]["completion_status"] == "timeout"
    assert saved["trial_status"] == "skipped"
    assert saved["skip_reason"] == "retries_exhausted:timeout"

    summary = build_summary(tmp_path, trials=1)
    assert summary["total_trials"] == 1
    assert summary["skipped_trials"] == 1
    assert summary["per_task"]["task_a"]["trial_results"][0]["trial_status"] == "skipped"


def test_train_plugin_feedback_writes_generic_and_memos_compat_results(tmp_path):
    agent = _FakeAgent()
    result = run_task_once(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent=agent,
        phase_dir=tmp_path,
        phase="train",
        split="train",
        trial=1,
        attempt=1,
        args=Namespace(
            train_feedback=True,
            feedback_timeout=7,
            plugin_structured_feedback=True,
            plugin_feedback_backend="memos",
            plugin_feedback_timeout=3,
        ),
    )

    assert result["plugin_feedback_result"] == {
        "status": "skipped",
        "reason": "missing_openclaw_gateway_session_id",
        "backend": "memos",
    }
    assert result["memos_feedback_result"] == result["plugin_feedback_result"]

    saved = json.loads((tmp_path / "task_a__trial_1" / "result.json").read_text())
    assert saved["plugin_feedback_result"] == result["plugin_feedback_result"]
    assert saved["memos_feedback_result"] == result["plugin_feedback_result"]


def test_train_plugin_feedback_unsupported_backend_does_not_fill_memos_alias(tmp_path):
    agent = _FakeAgent()
    result = run_task_once(
        task={"name": "task_a"},
        domain=_FakeDomain(),
        agent=agent,
        phase_dir=tmp_path,
        phase="train",
        split="train",
        trial=1,
        attempt=1,
        args=Namespace(
            train_feedback=True,
            feedback_timeout=7,
            plugin_structured_feedback=True,
            plugin_feedback_backend="custom",
            plugin_feedback_timeout=3,
        ),
    )

    assert result["plugin_feedback_result"] == {
        "status": "skipped",
        "reason": "unsupported_plugin_feedback_backend",
        "backend": "custom",
    }
    assert result["memos_feedback_result"] == {}
