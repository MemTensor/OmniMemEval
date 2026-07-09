from __future__ import annotations

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.agents import create_agent
from agentbench.config import deep_merge, load_env_file, load_yaml, write_json
from agentbench.domains import create_domain
from agentbench.memory_lifecycle import CommandMemoryLifecycle
from agentbench.plugin_feedback import normalize_plugin_feedback_backend
from agentbench.runner import run_phase


def _default_domain_config(domain: str) -> Path:
    return ROOT / "configs" / "agentbench" / "domains" / f"{domain}.yaml"


def _default_agent_config(agent: str) -> Path:
    return ROOT / "configs" / "agentbench" / "agents" / f"{agent}.yaml"


def _default_memory_plugin_config(plugin: str) -> Path:
    return ROOT / "configs" / "agentbench" / "memory_plugins" / f"{plugin}.yaml"


def _default_env_file() -> Path:
    return ROOT / ".env.agent"


def _apply_env_aliases(loaded_env_keys: set[str]) -> None:
    if "LLM_BASE_URL" not in loaded_env_keys:
        if "DASHSCOPE_BASE_URL" in loaded_env_keys:
            os.environ["LLM_BASE_URL"] = os.environ["DASHSCOPE_BASE_URL"]
            loaded_env_keys.add("LLM_BASE_URL")
        elif "DASHSCOPE_API_BASE" in loaded_env_keys:
            os.environ["LLM_BASE_URL"] = os.environ["DASHSCOPE_API_BASE"]
            loaded_env_keys.add("LLM_BASE_URL")
        elif "LLM_API_KEY" in loaded_env_keys or "DASHSCOPE_API_KEY" in loaded_env_keys:
            os.environ["LLM_BASE_URL"] = (
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            loaded_env_keys.add("LLM_BASE_URL")

    if "LLM_API_KEY" not in loaded_env_keys and "DASHSCOPE_API_KEY" in loaded_env_keys:
        os.environ["LLM_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
        loaded_env_keys.add("LLM_API_KEY")

    if "DASHSCOPE_BASE_URL" not in loaded_env_keys:
        if "DASHSCOPE_API_BASE" in loaded_env_keys:
            os.environ["DASHSCOPE_BASE_URL"] = os.environ["DASHSCOPE_API_BASE"]
            loaded_env_keys.add("DASHSCOPE_BASE_URL")
        elif "DASHSCOPE_API_KEY" in loaded_env_keys:
            os.environ["DASHSCOPE_BASE_URL"] = (
                "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            loaded_env_keys.add("DASHSCOPE_BASE_URL")

    # Agent provider credentials should come from project .env.agent / --env, not from
    # an unrelated shell session. If absent, the adapter falls back to
    # ~/.openclaw/openclaw.json by dropping unresolved ${...} placeholders.
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_API_KEY"):
        if key not in loaded_env_keys:
            os.environ.pop(key, None)


def _agent_factory(agent_name: str, agent_config: dict):
    def factory():
        agent = create_agent(agent_name, agent_config)
        return agent
    return factory


def build_run_dir_name(*, profile_name: str, domain: str, version: str) -> str:
    return f"{profile_name}-{version}-{domain}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniMemEval AgentBench runner")
    parser.add_argument("--agent", default="openclaw", help="Agent runtime name")
    parser.add_argument("--agent-config", default=None, help="Agent YAML. Defaults to configs/agentbench/agents/<agent>.yaml")
    parser.add_argument("--domain", default="reasoning", help="Domain name")
    parser.add_argument("--domain-config", default=None, help="Domain YAML")
    parser.add_argument(
        "--protocol",
        choices=["test_only", "train_then_test", "memory_train_backup_test"],
        default="test_only",
    )
    parser.add_argument("--memory-plugin", default=None, help="Memory plugin config name under configs/agentbench/memory_plugins")
    parser.add_argument("--memory-plugin-config", default=None, help="Explicit memory lifecycle YAML")
    parser.add_argument("--version", default=None, help="Result version suffix")
    parser.add_argument("--env", default=None, help="Optional env file. Defaults also load project .env.agent when present.")
    parser.add_argument("--task", default=None, help="Task id(s), comma-separated")
    parser.add_argument("--train-task", default=None, help="Train phase task id(s), comma-separated. Overrides --task for train phases.")
    parser.add_argument("--test-task", default=None, help="Test phase task id(s), comma-separated. Overrides --task for test phases.")
    parser.add_argument("--trials", "--runs", dest="trials", type=int, default=1)
    parser.add_argument("--test-runs", type=int, default=1, help="For memory_train_backup_test: restore memory and run the test split this many times.")
    parser.add_argument("--train-feedback", dest="train_feedback", action="store_true", default=None, help="After train verification, send verifier feedback to the same agent session.")
    parser.add_argument("--no-train-feedback", dest="train_feedback", action="store_false", help="Disable train feedback turn.")
    parser.add_argument("--feedback-timeout", type=int, default=None, help="Timeout in seconds for the train feedback turn.")
    parser.add_argument("--plugin-structured-feedback", dest="plugin_structured_feedback", action="store_true", default=None, help="Submit plugin-side structured feedback after the train feedback turn.")
    parser.add_argument("--no-plugin-structured-feedback", dest="plugin_structured_feedback", action="store_false", help="Disable plugin-side structured feedback submit.")
    parser.add_argument("--plugin-feedback-backend", default=None, help="Structured feedback backend, such as memos. Defaults to memory plugin config.")
    parser.add_argument("--plugin-feedback-timeout", type=int, default=None, help="Timeout in seconds for plugin structured feedback submit.")
    parser.add_argument("--memos-structured-feedback", dest="memos_structured_feedback", action="store_true", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-memos-structured-feedback", dest="memos_structured_feedback", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--memos-feedback-timeout", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pass-at", type=int, default=None, help="Compute pass@n using the first n trials. Defaults to --trials.")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "agentbench"))
    parser.add_argument("--force", action="store_true", help="Re-run completed trials")
    return parser.parse_args()


def _load_memory_config(args: argparse.Namespace) -> tuple[Path | None, dict | None]:
    if args.protocol != "memory_train_backup_test":
        return None, None
    if args.memory_plugin_config:
        path = Path(args.memory_plugin_config)
    elif args.memory_plugin:
        path = _default_memory_plugin_config(args.memory_plugin)
    else:
        raise SystemExit("--protocol memory_train_backup_test requires --memory-plugin or --memory-plugin-config")
    return path, load_yaml(path)


def _agent_config_for_memory_protocol(agent_config: dict, memory_config: dict) -> dict:
    config = copy.deepcopy(agent_config)
    runtime = config.setdefault("runtime", {})
    # The default OpenClaw profile is also used for plain baseline runs, where
    # memory plugins are deliberately filtered out.  In memory plugin protocols,
    # the lifecycle config owns plugin enablement/mode, so this filter must not
    # remove the plugin that is under evaluation.
    runtime["disabled_plugin_names"] = []
    runtime["disabled_tool_prefixes"] = []
    runtime["disable_plugins"] = False
    runtime["home_mode"] = "isolated_copy"

    configured_links = list(runtime.get("home_links") or config.get("home_links") or [])
    memory_links = list(memory_config.get("home_links") or [])
    seen = set()
    runtime["home_links"] = [
        link for link in [*configured_links, *memory_links]
        if not (link in seen or seen.add(link))
    ]
    execution = memory_config.get("execution") or {}
    if isinstance(execution, dict):
        for key in (
            "transport",
            "gateway_mode",
            "gateway_start_timeout",
            "gateway_stop_timeout",
            "gateway_ws_log",
        ):
            if key in execution:
                runtime[key] = execution[key]
    agent_runtime = memory_config.get("agent_runtime") or {}
    patch = {}
    if isinstance(agent_runtime, dict):
        patch = agent_runtime.get("openclaw_config_patch") or {}
    patch = deep_merge(dict(patch), dict(memory_config.get("openclaw_config_patch") or {}))
    if patch:
        config["openclaw_config_patch"] = deep_merge(
            dict(config.get("openclaw_config_patch") or {}),
            patch,
        )
    return config


def _apply_memory_parallel_limit(args: argparse.Namespace, memory_config: dict | None) -> None:
    if not memory_config:
        return
    execution = memory_config.get("execution") or {}
    limit = execution.get("max_parallel") if isinstance(execution, dict) else None
    if limit is None:
        limit = memory_config.get("max_parallel")
    parallel_cfg = memory_config.get("parallel")
    if limit is None and isinstance(parallel_cfg, dict):
        limit = parallel_cfg.get("max")
    if limit is None:
        return
    try:
        max_parallel = int(limit)
    except (TypeError, ValueError):
        raise SystemExit(f"Invalid memory plugin max_parallel value: {limit!r}")
    if max_parallel < 1:
        raise SystemExit("memory plugin max_parallel must be >= 1")
    requested = int(args.parallel)
    args.requested_parallel = requested
    if requested > max_parallel:
        label = memory_config.get("plugin") or memory_config.get("name") or "memory"
        print(
            f"WARNING: memory plugin {label} caps --parallel at {max_parallel}; "
            f"requested {requested}. Using --parallel {max_parallel}.",
            flush=True,
        )
        args.parallel = max_parallel


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials/--runs must be >= 1")
    if args.test_runs < 1:
        raise SystemExit("--test-runs must be >= 1")
    if args.pass_at is not None and (args.pass_at < 1 or args.pass_at > args.trials):
        raise SystemExit("--pass-at must be >= 1 and <= --trials")
    loaded_env_keys = set()
    default_env = _default_env_file()
    if default_env.exists():
        loaded_env_keys.update(load_env_file(default_env, override=True))
    if args.env:
        loaded_env_keys.update(load_env_file(args.env, override=True))
    _apply_env_aliases(loaded_env_keys)
    agent_config_path = Path(args.agent_config) if args.agent_config else _default_agent_config(args.agent)
    raw_agent_config = load_yaml(agent_config_path)
    agent_config = raw_agent_config.get("agent", raw_agent_config)
    domain_config_path = Path(args.domain_config) if args.domain_config else _default_domain_config(args.domain)
    domain_config = load_yaml(domain_config_path)
    domain_config.setdefault("_config_path", str(domain_config_path.resolve()))
    memory_config_path, memory_config = _load_memory_config(args)
    if memory_config:
        agent_config = _agent_config_for_memory_protocol(agent_config, memory_config)
        _apply_memory_parallel_limit(args, memory_config)

    profile_name = agent_config.get("profile") or agent_config_path.stem
    memory_label = None
    if memory_config:
        memory_label = str(memory_config.get("plugin") or memory_config.get("name") or memory_config_path.stem)
        profile_name = f"{profile_name}-{memory_label}"
    feedback_config = (memory_config or {}).get("feedback") or {}
    execution_config = (memory_config or {}).get("execution") or {}
    if args.train_feedback is None:
        args.train_feedback = bool(feedback_config.get("enabled", args.protocol == "memory_train_backup_test"))
    if args.feedback_timeout is None:
        args.feedback_timeout = int(feedback_config.get("timeout", 300))
    if args.plugin_structured_feedback is None:
        if args.memos_structured_feedback is not None:
            args.plugin_structured_feedback = bool(args.memos_structured_feedback)
        else:
            args.plugin_structured_feedback = bool(
                feedback_config.get(
                    "structured_submit",
                    feedback_config.get(
                        "plugin_structured_submit",
                        feedback_config.get(
                            "memos_structured_submit",
                            feedback_config.get("memos_structured_feedback", memory_label == "memos"),
                        ),
                    ),
                )
            )
    if args.plugin_feedback_backend is None:
        configured_backend = feedback_config.get("backend") or feedback_config.get("plugin_feedback_backend")
        if configured_backend is None and (
            args.memos_structured_feedback
            or feedback_config.get("memos_structured_submit")
            or feedback_config.get("memos_structured_feedback")
            or memory_label == "memos"
        ):
            configured_backend = "memos"
        args.plugin_feedback_backend = normalize_plugin_feedback_backend(configured_backend)
    else:
        args.plugin_feedback_backend = normalize_plugin_feedback_backend(args.plugin_feedback_backend)
    if args.plugin_feedback_timeout is None:
        if args.memos_feedback_timeout is not None:
            args.plugin_feedback_timeout = args.memos_feedback_timeout
        else:
            args.plugin_feedback_timeout = int(
                feedback_config.get(
                    "submit_timeout",
                    feedback_config.get("plugin_submit_timeout", feedback_config.get("memos_submit_timeout", 900)),
                )
            )
    args.memos_structured_feedback = args.plugin_structured_feedback
    args.memos_feedback_timeout = args.plugin_feedback_timeout
    args.plugin_capture_mode = str(
        execution_config.get("capture_mode", "automatic")
        if isinstance(execution_config, dict)
        else "automatic"
    )
    if args.feedback_timeout < 1:
        raise SystemExit("--feedback-timeout must be >= 1")
    if args.plugin_feedback_timeout < 1:
        raise SystemExit("--plugin-feedback-timeout must be >= 1")
    args.memory_plugin_label = memory_label
    version = args.version or datetime.now().strftime("omnimemeval_%Y%m%d_%H%M%S")
    run_dir = Path(args.results_dir) / build_run_dir_name(
        profile_name=profile_name,
        domain=args.domain,
        version=version,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "experiment_config.json", {
        "agent": args.agent,
        "agent_config": str(agent_config_path.resolve()),
        "domain": args.domain,
        "domain_config": str(domain_config_path.resolve()),
        "protocol": args.protocol,
        "version": version,
        "trials": args.trials,
        "test_runs": args.test_runs,
        "train_feedback": args.train_feedback,
        "feedback_timeout": args.feedback_timeout,
        "plugin_structured_feedback": args.plugin_structured_feedback,
        "plugin_feedback_backend": args.plugin_feedback_backend,
        "plugin_feedback_timeout": args.plugin_feedback_timeout,
        "plugin_capture_mode": args.plugin_capture_mode,
        "memos_structured_feedback": args.memos_structured_feedback,
        "memos_feedback_timeout": args.memos_feedback_timeout,
        "parallel": args.parallel,
        "requested_parallel": getattr(args, "requested_parallel", args.parallel),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "task": args.task,
        "train_task": args.train_task,
        "test_task": args.test_task,
        "memory_plugin": memory_label,
        "memory_plugin_config": str(memory_config_path.resolve()) if memory_config_path else None,
    })

    def make_domain():
        return create_domain(args.domain, domain_config)

    agent_factory = _agent_factory(args.agent, agent_config)

    if args.protocol == "test_only":
        run_phase(
            phase="test",
            split=args.test_split,
            phase_dir=run_dir / "test",
            domain=make_domain(),
            agent_factory=agent_factory,
            args=args,
            task=args.test_task or args.task,
        )
    elif args.protocol == "train_then_test":
        run_phase(
            phase="train",
            split=args.train_split,
            phase_dir=run_dir / "train",
            domain=make_domain(),
            agent_factory=agent_factory,
            args=args,
            task=args.train_task or args.task,
        )
        run_phase(
            phase="test_after_train",
            split=args.test_split,
            phase_dir=run_dir / "test",
            domain=make_domain(),
            agent_factory=agent_factory,
            args=args,
            task=args.test_task or args.task,
        )
    else:
        assert memory_config is not None
        lifecycle = CommandMemoryLifecycle(
            config=memory_config,
            project_dir=ROOT,
            run_dir=run_dir,
            run_id=version,
            version=version,
        )
        lifecycle.validate(args.domain)
        lifecycle.set_mode("train", args.domain)
        lifecycle.clear(args.domain)
        run_phase(
            phase="train",
            split=args.train_split,
            phase_dir=run_dir / "train",
            domain=make_domain(),
            agent_factory=agent_factory,
            args=args,
            task=args.train_task or args.task,
        )
        lifecycle.wait_settle(args.domain)
        backup_file = lifecycle.backup(args.domain)

        for run_no in range(1, args.test_runs + 1):
            lifecycle.set_mode("test", args.domain)
            # Always restore before a test run.  This keeps test writes from
            # polluting later runs even when a plugin can disable writes.
            lifecycle.restore(args.domain, backup_file)
            phase_name = f"test_run_{run_no}"
            run_phase(
                phase=phase_name,
                split=args.test_split,
                phase_dir=run_dir / phase_name,
                domain=make_domain(),
                agent_factory=agent_factory,
                args=args,
                task=args.test_task or args.task,
            )
        lifecycle.finalize(args.domain)

    print(f"\nResults written to {run_dir}")


if __name__ == "__main__":
    main()
