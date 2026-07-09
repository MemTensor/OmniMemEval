from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from agentbench.session import SessionSpec


def normalize_plugin_feedback_backend(value: str | None) -> str:
    backend = str(value or "").strip().lower()
    if backend in {"", "none", "false", "off", "disabled"}:
        return "none"
    if backend in {"memos", "memos_structured", "memos-local-plugin"}:
        return "memos"
    return backend


def should_submit_plugin_feedback(args: Namespace, phase: str) -> bool:
    return phase == "train" and bool(
        getattr(args, "train_feedback", False)
        and getattr(args, "plugin_structured_feedback", False)
    )


def submit_plugin_structured_feedback(
    *,
    backend: str,
    session: SessionSpec,
    session_file: Path | None,
    feedback_prompt: str,
    feedback_result: dict[str, Any],
    verifier_result: dict[str, Any],
    domain_name: str,
    task: dict[str, Any],
    env_info: dict[str, Any],
    phase_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    backend = normalize_plugin_feedback_backend(backend)
    if backend == "none":
        return {"status": "skipped", "reason": "plugin_feedback_backend_disabled"}
    if backend == "memos":
        from agentbench.memos_feedback import submit_memos_structured_feedback

        result = submit_memos_structured_feedback(
            session=session,
            session_file=session_file,
            feedback_prompt=feedback_prompt,
            feedback_result=feedback_result,
            verifier_result=verifier_result,
            domain_name=domain_name,
            task=task,
            env_info=env_info,
            phase_dir=phase_dir,
            timeout=timeout,
        )
        result.setdefault("backend", "memos")
        return result
    return {
        "status": "skipped",
        "reason": "unsupported_plugin_feedback_backend",
        "backend": backend,
    }


def submit_plugin_feedback_artifact(
    *,
    backend: str,
    session: SessionSpec,
    session_file: Path | None,
    task_prompt: str,
    agent_result: dict[str, Any],
    feedback_prompt: str,
    feedback_result: dict[str, Any],
    verifier_result: dict[str, Any],
    domain_name: str,
    task: dict[str, Any],
    env_info: dict[str, Any],
    phase_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    backend = normalize_plugin_feedback_backend(backend)
    if backend == "none":
        return {"status": "skipped", "reason": "plugin_feedback_backend_disabled"}
    if backend == "memos":
        from agentbench.memos_feedback import submit_memos_feedback_artifact

        result = submit_memos_feedback_artifact(
            session=session,
            session_file=session_file,
            task_prompt=task_prompt,
            agent_result=agent_result,
            feedback_prompt=feedback_prompt,
            feedback_result=feedback_result,
            verifier_result=verifier_result,
            domain_name=domain_name,
            task=task,
            env_info=env_info,
            phase_dir=phase_dir,
            timeout=timeout,
        )
        result.setdefault("backend", "memos")
        return result
    return {
        "status": "skipped",
        "reason": "unsupported_plugin_feedback_backend",
        "backend": backend,
    }
