from __future__ import annotations

import json
import shutil
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from agentbench.config import write_json
from agentbench.feedback import build_feedback_prompt
from agentbench.plugin_feedback import normalize_plugin_feedback_backend
from agentbench.plugin_feedback import should_submit_plugin_feedback
from agentbench.plugin_feedback import submit_plugin_feedback_artifact
from agentbench.plugin_feedback import submit_plugin_structured_feedback
from agentbench.summary import build_summary


def _prompt_for_phase(domain, task: dict, env_info: dict, phase: str, agent_config: dict) -> str:
    prompt = domain.build_prompt(task, env_info, phase)
    prefixes = agent_config.get("prompt_prefix") or {}
    prefix = prefixes.get(phase) or prefixes.get("default")
    if prefix and not prompt.startswith(prefix):
        return f"{prefix}\n\n{prompt}"
    return prompt


def _stop_agent_gateway_if_available(agent) -> None:
    stop_gateway = getattr(agent, "stop_gateway", None)
    if callable(stop_gateway):
        stop_gateway()


def _should_send_train_feedback(args: Namespace, phase: str) -> bool:
    return phase == "train" and bool(getattr(args, "train_feedback", False))


def run_task_once(
    *,
    task: dict,
    domain,
    agent,
    phase_dir: Path,
    phase: str,
    split: str,
    trial: int,
    attempt: int,
    args: Namespace,
) -> dict:
    task_name = task["name"]
    session = agent.build_session_spec(
        phase=phase,
        domain=domain.name,
        split=split,
        task=task,
        trial=trial,
    )
    trial_dir = phase_dir / f"{task_name}__trial_{trial}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "task_name": task_name,
        "agent": agent.name,
        "phase": phase,
        "split": split,
        "trial": trial,
        "attempt": attempt,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "session": session.to_dict(),
        "agent_result": {},
        "verifier_result": {"reward": 0.0},
        "feedback_result": {},
        "plugin_feedback_result": {},
        "memos_feedback_result": {},
        "exception_info": None,
    }
    env_info = {}

    try:
        task["_phase_dir"] = str(phase_dir)
        env_info = domain.setup(task, agent.name, trial)
        agent.prepare_task(task, env_info, session)
        prompt = _prompt_for_phase(domain, task, env_info, phase, agent.config)
        timeout = domain.get_agent_timeout(task, env_info)
        agent_result = agent.call(prompt, session, timeout=timeout)
        result["agent_result"] = agent_result
        try:
            result["agent_result"]["_session_file"] = str(agent._session_file(session))
        except Exception:
            pass
        verifier_result = domain.verify(task, env_info, trial_dir, agent_result=agent_result)
        result["verifier_result"] = verifier_result
        if _should_send_train_feedback(args, phase):
            feedback_prompt = build_feedback_prompt(task_name, verifier_result)
            result["feedback_prompt"] = feedback_prompt
            feedback_timeout = int(getattr(args, "feedback_timeout", 300))
            capture_mode = str(
                getattr(args, "plugin_capture_mode", "automatic") or "automatic"
            ).strip().lower()
            if capture_mode == "manual_after_feedback":
                _stop_agent_gateway_if_available(agent)
            try:
                feedback_result = agent.call(
                    feedback_prompt,
                    session,
                    timeout=feedback_timeout,
                )
                result["feedback_result"] = feedback_result
            except Exception as exc:
                result["feedback_result"] = {
                    "completion_status": "error",
                    "error": str(exc),
                }
            finally:
                if capture_mode == "manual_after_feedback":
                    _stop_agent_gateway_if_available(agent)
            if should_submit_plugin_feedback(args, phase):
                try:
                    raw_session_file = result["agent_result"].get("_session_file")
                    session_file = Path(raw_session_file) if raw_session_file else None
                    if capture_mode == "manual_after_feedback":
                        result["plugin_feedback_result"] = submit_plugin_feedback_artifact(
                            backend=str(getattr(args, "plugin_feedback_backend", "none")),
                            session=session,
                            session_file=session_file,
                            task_prompt=prompt,
                            agent_result=result["agent_result"],
                            feedback_prompt=feedback_prompt,
                            feedback_result=result["feedback_result"],
                            verifier_result=verifier_result,
                            domain_name=domain.name,
                            task=task,
                            env_info=env_info,
                            phase_dir=phase_dir,
                            timeout=float(getattr(args, "plugin_feedback_timeout", 900)),
                        )
                    else:
                        result["plugin_feedback_result"] = submit_plugin_structured_feedback(
                            backend=str(getattr(args, "plugin_feedback_backend", "none")),
                            session=session,
                            session_file=session_file,
                            feedback_prompt=feedback_prompt,
                            feedback_result=result["feedback_result"],
                            verifier_result=verifier_result,
                            domain_name=domain.name,
                            task=task,
                            env_info=env_info,
                            phase_dir=phase_dir,
                            timeout=float(getattr(args, "plugin_feedback_timeout", 900)),
                        )
                except Exception as exc:
                    result["plugin_feedback_result"] = {
                        "status": "error",
                        "error": str(exc),
                    }
                if normalize_plugin_feedback_backend(
                    getattr(args, "plugin_feedback_backend", "")
                ) == "memos":
                    result["memos_feedback_result"] = dict(result["plugin_feedback_result"])
        try:
            domain.record_agent_outcome(task, env_info, trial_dir, agent_result, verifier_result)
        except Exception:
            pass
    except Exception as exc:
        result["exception_info"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        try:
            domain.cleanup(task, env_info)
        except Exception:
            pass
        result["ended_at"] = datetime.now(timezone.utc).isoformat()
        try:
            session_stats = agent.collect_session(session, trial_dir)
        except Exception:
            session_stats = {"turns": 0, "input": 0, "output": 0, "total": 0, "last_stop_reason": None}
        if not session_stats.get("total"):
            usage = (
                ((result.get("agent_result") or {}).get("response_json") or {})
                .get("meta", {})
                .get("agentMeta", {})
                .get("usage", {})
            )
            if usage:
                session_stats["input"] = int(usage.get("input") or usage.get("prompt_tokens") or 0)
                session_stats["output"] = int(usage.get("output") or usage.get("completion_tokens") or 0)
                session_stats["total"] = int(usage.get("total") or usage.get("total_tokens") or 0)
        result["last_stop_reason"] = session_stats.pop("last_stop_reason", None)
        result["token_usage"] = session_stats
        try:
            agent.cleanup_task()
        except Exception:
            pass

        save_result = dict(result)
        save_result["agent_result"] = dict(result.get("agent_result") or {})
        save_result["feedback_result"] = dict(result.get("feedback_result") or {})
        save_result["plugin_feedback_result"] = dict(result.get("plugin_feedback_result") or {})
        save_result["memos_feedback_result"] = dict(result.get("memos_feedback_result") or {})
        response = save_result["agent_result"].get("response", "")
        if isinstance(response, str):
            (trial_dir / "response.txt").write_text(response)
            save_result["agent_result"]["response_file"] = "response.txt"
            save_result["agent_result"]["response_chars"] = len(response)
        if isinstance(response, str) and len(response) > 10000:
            save_result["agent_result"]["response"] = response[:10000] + f"\n...(truncated, {len(response)} chars)"
        feedback_response = save_result["feedback_result"].get("response", "")
        if isinstance(feedback_response, str):
            save_result["feedback_result"]["response_chars"] = len(feedback_response)
        if isinstance(feedback_response, str) and len(feedback_response) > 10000:
            save_result["feedback_result"]["response"] = (
                feedback_response[:10000] + f"\n...(truncated, {len(feedback_response)} chars)"
            )
        feedback_prompt = save_result.get("feedback_prompt", "")
        if isinstance(feedback_prompt, str) and len(feedback_prompt) > 10000:
            save_result["feedback_prompt"] = (
                feedback_prompt[:10000] + f"\n...(truncated, {len(feedback_prompt)} chars)"
            )
        write_json(trial_dir / "result.json", save_result)
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "reward.txt").write_text(str(result["verifier_result"].get("reward", 0.0)))

    return result


def run_task_with_retry(task: dict, domain, agent_factory, phase_dir: Path, phase: str, split: str, trial: int, args: Namespace) -> dict:
    max_retries = getattr(args, "max_retries", 0)
    result = {}
    for attempt in range(1, max_retries + 2):
        agent = agent_factory()
        result = run_task_once(
            task=task,
            domain=domain,
            agent=agent,
            phase_dir=phase_dir,
            phase=phase,
            split=split,
            trial=trial,
            attempt=attempt,
            args=args,
        )
        retry_reason = agent.should_retry(result)
        if retry_reason and attempt <= max_retries:
            old_trial = phase_dir / f"{task['name']}__trial_{trial}"
            backup = phase_dir / f"{task['name']}__trial_{trial}_retry{attempt}"
            if old_trial.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                old_trial.rename(backup)
            continue
        break
    return result


def run_phase(
    *,
    phase: str,
    split: str,
    phase_dir: Path,
    domain,
    agent_factory,
    args: Namespace,
    task: str | None = None,
) -> dict:
    phase_dir.mkdir(parents=True, exist_ok=True)
    phase_args = Namespace(**vars(args))
    phase_args.split = split
    if task is not None:
        phase_args.task = task
    tasks = domain.load_tasks(phase_args)
    write_json(phase_dir / "phase_config.json", {
        "phase": phase,
        "split": split,
        "task": phase_args.task,
        "tasks": len(tasks),
        "trials": args.trials,
        "parallel": args.parallel,
    })

    print(f"\n=== AgentBench phase={phase} split={split} tasks={len(tasks)} trials={args.trials} ===")
    domain.initialize(phase_args)
    progress_lock = threading.Lock()
    completed = 0

    def run_one(task: dict) -> list[dict]:
        results = []
        try:
            domain.pre_task_trials(task)
            for trial in range(1, args.trials + 1):
                trial_dir = phase_dir / f"{task['name']}__trial_{trial}"
                if (trial_dir / "result.json").exists() and not args.force:
                    try:
                        results.append(json.loads((trial_dir / "result.json").read_text()))
                        continue
                    except (OSError, json.JSONDecodeError):
                        pass
                results.append(
                    run_task_with_retry(
                        task,
                        domain,
                        agent_factory,
                        phase_dir,
                        phase,
                        split,
                        trial,
                        args,
                    )
                )
        finally:
            domain.post_task_trials(task)
        nonlocal completed
        with progress_lock:
            completed += 1
            best = max((item.get("verifier_result", {}).get("reward", 0) for item in results), default=0)
            print(f"  [{completed}/{len(tasks)}] {task['name']} reward={best:.2f}", flush=True)
        return results

    try:
        if args.parallel <= 1:
            for task in tasks:
                run_one(task)
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(run_one, task) for task in tasks]
                for future in as_completed(futures):
                    future.result()
    finally:
        domain.finalize()

    summary = build_summary(
        phase_dir,
        trials=args.trials,
        pass_at=getattr(args, "pass_at", None) or args.trials,
        domain=domain,
    )
    pass_at = summary.get("pass_at", args.trials)
    if pass_at == 1:
        pass_text = f"pass@1={summary['pass@1']:.4f}"
    else:
        pass_text = (
            f"pass@1={summary['pass@1']:.4f} "
            f"pass@{pass_at}={summary.get(f'pass@{pass_at}', 0.0):.4f}"
        )
    print(f"Phase {phase} {pass_text} total_trials={summary['total_trials']}")
    return summary
