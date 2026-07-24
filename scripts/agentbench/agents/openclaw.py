from __future__ import annotations

import json
import os
import secrets
import signal
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agentbench.agents.base import AgentAdapter
from agentbench.config import deep_merge
from agentbench.session import SessionSpec


def _is_unresolved_env_value(value: Any) -> bool:
    return isinstance(value, str) and ("${" in value or value.startswith("$"))


def _drop_unresolved_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if not _is_unresolved_env_value(cleaned := _drop_unresolved_env_values(item))
        }
    if isinstance(value, list):
        return [_drop_unresolved_env_values(item) for item in value]
    return value


class OpenClawAgentAdapter(AgentAdapter):
    name = "openclaw"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._temp_home: str | None = None
        self._workspace_dir: str | None = None
        self._task_env: dict[str, str] = {}
        self._task_env_info: dict[str, Any] = {}
        self._gateway_port: int | None = None
        self._gateway_token: str | None = None
        self._gateway_proc: subprocess.Popen | None = None
        self._gateway_log: Path | None = None

    def build_session_spec(self, **kwargs) -> SessionSpec:
        base = super().build_session_spec(**kwargs)
        session_key = f"agent:main:explicit:{base.cli_session_id}"
        gateway_id = f"openclaw::main::{session_key}"
        return SessionSpec(
            cli_session_id=base.cli_session_id,
            semantic_session_id=base.semantic_session_id,
            source_ref=base.source_ref,
            metadata=base.metadata,
            agent_session_ref=session_key,
            openclaw_session_key=session_key,
            openclaw_gateway_session_id=gateway_id,
        )

    def _command(self) -> str:
        return self.config.get("command") or self.config.get("agent", {}).get("command") or "openclaw"

    def _configured_thinking_level(self) -> str | None:
        value = self.config.get("thinking_level", self.config.get("thinking"))
        return str(value) if value is not None else None

    def _runtime(self) -> dict:
        return dict(self.config.get("runtime") or {})

    def _source_config_dir(self) -> Path:
        """Return the durable OpenClaw config used to seed isolated task homes.

        Memory evaluations can point this at a run-scoped clone so task homes
        never link back into the user's real ``~/.openclaw`` tree.
        """
        configured = os.environ.get("OMNIMEMEVAL_OPENCLAW_HOME_SOURCE")
        if configured:
            return Path(configured).expanduser().resolve()
        return Path.home() / ".openclaw"

    def _transport(self) -> str:
        runtime = self._runtime()
        return str(runtime.get("transport") or "local").strip().lower()

    def _configured_home_links(self) -> list[str]:
        runtime = self._runtime()
        links = list(runtime.get("home_links") or self.config.get("home_links") or [])
        seen = set()
        result = []
        for link in links:
            if not isinstance(link, str):
                raise RuntimeError(f"OpenClaw home link must be a string: {link!r}")
            if link not in seen:
                result.append(link)
                seen.add(link)
        return result

    def _link_global_openclaw_paths(self, config_dir: Path) -> None:
        source_home = self._source_config_dir()
        for link in self._configured_home_links():
            rel_path = Path(link)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise RuntimeError(f"OpenClaw home_links must be relative paths: {link}")
            source = source_home / rel_path
            if not source.exists():
                raise RuntimeError(f"OpenClaw home link source does not exist: {source}")
            target = config_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            target.symlink_to(source, target_is_directory=source.is_dir())

    def _build_openclaw_config(self) -> dict:
        global_config_path = self._source_config_dir() / "openclaw.json"
        if global_config_path.exists():
            config = json.loads(global_config_path.read_text())
        else:
            config = {}

        runtime = self._runtime()
        if runtime.get("strip_gateway", True):
            config.pop("gateway", None)
            config.pop("auth", None)
        if runtime.get("disable_plugins", False):
            config.pop("plugins", None)
            config.pop("mcp", None)
        if self._task_env_info.get("mcp_only"):
            config.pop("mcp", None)

        defaults = config.setdefault("agents", {}).setdefault("defaults", {})
        if self.config.get("model"):
            defaults["model"] = {"primary": self.config["model"]}
        thinking_level = self._configured_thinking_level()
        if thinking_level:
            defaults["thinkingDefault"] = thinking_level
        if self.config.get("max_concurrent"):
            defaults["maxConcurrent"] = self.config["max_concurrent"]
        if self._workspace_dir:
            defaults["workspace"] = self._workspace_dir
        if self.config.get("providers"):
            all_providers = config.setdefault("models", {}).setdefault("providers", {})
            providers = {}
            for name, provider_cfg in self.config["providers"].items():
                provider_cfg = _drop_unresolved_env_values(provider_cfg)
                merged_provider = deep_merge(
                    dict(all_providers.get(name) or {}),
                    dict(provider_cfg),
                )
                all_providers[name] = merged_provider
                providers[name] = merged_provider
            self._sync_provider_model_defaults(config, providers)
        if self.config.get("tools"):
            config.setdefault("tools", {}).update(self.config["tools"])
        if self._transport() == "gateway":
            port, token = self._ensure_gateway_identity()
            config["gateway"] = {
                "mode": "local",
                "bind": "loopback",
                "port": port,
                "auth": {
                    "mode": "token",
                    "token": token,
                },
                "remote": {
                    "url": f"ws://127.0.0.1:{port}",
                    "token": token,
                },
            }
            config.pop("auth", None)
        mcp_servers = self._task_env_info.get("mcp_servers") or {}
        mcp_section = {}
        if mcp_servers:
            for name, cfg in mcp_servers.items():
                mcp_section[name] = self._build_mcp_server_entry(cfg)
            if self._task_env_info.get("mcp_only"):
                config["mcp"] = {"servers": mcp_section}
            else:
                config.setdefault("mcp", {})["servers"] = mcp_section
        disabled_tools = self._task_env_info.get("disabled_tools") or []
        if disabled_tools:
            existing = config.setdefault("tools", {}).get("deny", [])
            if not isinstance(existing, list):
                existing = []
            seen = set()
            config["tools"]["deny"] = [
                tool for tool in [*existing, *disabled_tools]
                if not (tool in seen or seen.add(tool))
            ]

        patch = self.config.get("openclaw_config_patch") or {}
        if patch:
            config = deep_merge(config, patch)
        phase_patch = self._phase_config_patch()
        if phase_patch:
            config = deep_merge(config, phase_patch)
        self._drop_plugin_config_keys(config)
        if self._task_env_info.get("mcp_only"):
            self._restrict_mcp_only_config(config, mcp_section)
        self._filter_disabled_plugins(config)
        self._ensure_plugin_load_paths(config)
        self._enforce_workspace_config(config)
        return config

    def _enforce_workspace_config(self, config: dict) -> None:
        """Apply the trial workspace after all user/profile config patches."""

        if not self._workspace_dir:
            return
        agents = config.setdefault("agents", {})
        agents.setdefault("defaults", {})["workspace"] = self._workspace_dir
        entries = agents.get("list")
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    entry["workspace"] = self._workspace_dir

    def _phase_config_patch(self) -> dict:
        patches = self.config.get("openclaw_phase_config_patches") or {}
        phase = str(self._task_env_info.get("phase") or "").strip().lower()
        if phase == "train":
            key = "train"
        elif phase == "test" or phase.startswith("test_"):
            key = "test"
        else:
            return {}
        patch = patches.get(key) if isinstance(patches, dict) else None
        return dict(patch) if isinstance(patch, dict) else {}

    def _drop_plugin_config_keys(self, config: dict) -> None:
        """Remove obsolete plugin options from an isolated evaluation home.

        The source OpenClaw config may retain settings from an older plugin
        build. Profiles can list only the keys that are unsafe to inherit;
        unrelated plugin configuration is preserved.
        """

        configured = self._runtime().get("drop_plugin_config_keys") or {}
        if not configured:
            return
        if not isinstance(configured, dict):
            raise RuntimeError("runtime.drop_plugin_config_keys must be an object")

        entries = (config.get("plugins") or {}).get("entries")
        if not isinstance(entries, dict):
            return
        for plugin_name, keys in configured.items():
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                raise RuntimeError(
                    "runtime.drop_plugin_config_keys values must be lists of strings"
                )
            entry = entries.get(str(plugin_name))
            if not isinstance(entry, dict):
                continue
            plugin_config = entry.get("config")
            if not isinstance(plugin_config, dict):
                continue
            for key in keys:
                plugin_config.pop(key, None)
            if not plugin_config:
                entry.pop("config", None)

    def _mcp_only_preserved_plugin_names(self) -> set[str]:
        names = self._runtime().get("mcp_only_preserve_plugin_names") or []
        return {str(name) for name in names if str(name).strip()}

    def _restrict_mcp_only_config(self, config: dict, mcp_servers: dict) -> None:
        """Keep task MCP plus explicitly selected auto-injection plugins only."""

        config["mcp"] = {"servers": dict(mcp_servers)}
        preserved = self._mcp_only_preserved_plugin_names()
        plugins = config.get("plugins")
        if not preserved or not isinstance(plugins, dict):
            config.pop("plugins", None)
        else:
            allow = plugins.get("allow")
            if isinstance(allow, list):
                plugins["allow"] = [name for name in allow if str(name) in preserved]
            entries = plugins.get("entries")
            if isinstance(entries, dict):
                plugins["entries"] = {
                    name: value for name, value in entries.items() if str(name) in preserved
                }
            slots = plugins.get("slots")
            if isinstance(slots, dict):
                plugins["slots"] = {
                    slot: name for slot, name in slots.items() if str(name) in preserved
                }
            installs = plugins.get("installs")
            if isinstance(installs, dict):
                plugins["installs"] = {
                    name: value for name, value in installs.items() if str(name) in preserved
                }
            load = plugins.get("load")
            if isinstance(load, dict) and isinstance(load.get("paths"), list):
                load["paths"] = [
                    path for path in load["paths"]
                    if any(name in Path(str(path)).name for name in preserved)
                ]

        tools = config.setdefault("tools", {})
        # Global allow-lists must not leak plugin tools into an MCP-only domain.
        tools.pop("alsoAllow", None)
        denied = tools.get("deny")
        if not isinstance(denied, list):
            denied = []
        extra_denied = self._runtime().get("mcp_only_denied_tools") or []
        seen = set()
        tools["deny"] = [
            tool for tool in [*denied, *extra_denied]
            if not (tool in seen or seen.add(tool))
        ]

    def _disabled_plugin_names(self) -> set[str]:
        runtime = self._runtime()
        names = runtime.get("disabled_plugin_names") or self.config.get("disabled_plugin_names") or []
        return {str(name).lower() for name in names if str(name).strip()}

    def _disabled_tool_prefixes(self) -> tuple[str, ...]:
        runtime = self._runtime()
        prefixes = runtime.get("disabled_tool_prefixes") or self.config.get("disabled_tool_prefixes") or []
        return tuple(str(prefix) for prefix in prefixes if str(prefix))

    def _is_disabled_plugin_name(self, name: str) -> bool:
        lowered = str(name).lower()
        return any(disabled == lowered or disabled in lowered for disabled in self._disabled_plugin_names())

    def _filter_disabled_plugins(self, config: dict) -> None:
        disabled = self._disabled_plugin_names()
        if not disabled:
            return

        plugins = config.get("plugins")
        if isinstance(plugins, dict):
            allow = plugins.get("allow")
            if isinstance(allow, list):
                plugins["allow"] = [
                    item for item in allow
                    if not self._is_disabled_plugin_name(str(item))
                ]

            entries = plugins.get("entries")
            if isinstance(entries, dict):
                plugins["entries"] = {
                    key: value for key, value in entries.items()
                    if not self._is_disabled_plugin_name(str(key))
                }

            slots = plugins.get("slots")
            if isinstance(slots, dict):
                plugins["slots"] = {
                    key: value for key, value in slots.items()
                    if not self._is_disabled_plugin_name(str(value))
                }

            for key in ("installs",):
                section = plugins.get(key)
                if isinstance(section, dict):
                    plugins[key] = {
                        name: value for name, value in section.items()
                        if not self._is_disabled_plugin_name(str(name))
                    }

            load = plugins.get("load")
            if isinstance(load, dict) and isinstance(load.get("paths"), list):
                load["paths"] = [
                    path for path in load["paths"]
                    if not self._is_disabled_plugin_name(Path(str(path)).name)
                ]

        prefixes = self._disabled_tool_prefixes()
        tools = config.get("tools")
        if prefixes and isinstance(tools, dict) and isinstance(tools.get("alsoAllow"), list):
            tools["alsoAllow"] = [
                tool for tool in tools["alsoAllow"]
                if not str(tool).startswith(prefixes)
            ]

    def _proxy_script(self) -> Path:
        return Path(__file__).resolve().parents[1] / "utils" / "openclaw_mcp_stdio_proxy.js"

    def _openclaw_node_modules_dir(self) -> Path | None:
        override = os.environ.get("OPENCLAW_MCP_PROXY_NODE_MODULES")
        if override:
            path = Path(override).expanduser()
            return path if path.exists() else None

        command_path = shutil.which(self._command())
        if command_path is None:
            return None
        command_path = Path(command_path).resolve()
        for parent in [command_path.parent, *command_path.parents]:
            for candidate in (
                parent / "node_modules",
                parent / "lib" / "node_modules" / "openclaw" / "node_modules",
            ):
                if candidate.exists():
                    return candidate
        return None

    def _build_mcp_server_entry(self, cfg: dict) -> dict:
        if cfg.get("command"):
            entry = {"command": cfg["command"]}
            if cfg.get("args"):
                entry["args"] = list(cfg["args"])
            return entry

        url = cfg.get("url", "")
        if not url:
            raise RuntimeError("MCP config requires either command/args or url")

        node_modules = self._openclaw_node_modules_dir()
        if node_modules is None:
            raise RuntimeError(
                "Cannot locate OpenClaw node_modules for MCP stdio proxy. "
                "Set OPENCLAW_MCP_PROXY_NODE_MODULES or configure an MCP command."
            )

        proxy = self._proxy_script()
        if not proxy.exists():
            raise RuntimeError(f"MCP stdio proxy script missing: {proxy}")

        return {
            "command": shutil.which("node") or "node",
            "args": [
                str(proxy),
                "--url", url,
                "--transport", cfg.get("type", "auto"),
                "--node-modules", str(node_modules),
            ],
        }

    @staticmethod
    def _sync_provider_model_defaults(config: dict, providers: dict) -> None:
        """Expose provider model params where OpenClaw resolves runtime params.

        OpenClaw reads provider catalog metadata from ``models.providers`` but
        resolves request params such as ``params.extra_body`` from
        ``agents.defaults.models["provider/model"].params``.
        """

        defaults = config.setdefault("agents", {}).setdefault("defaults", {})
        default_models = defaults.setdefault("models", {})

        for provider_name, provider_cfg in providers.items():
            if not isinstance(provider_cfg, dict):
                continue
            for model_cfg in provider_cfg.get("models") or []:
                if not isinstance(model_cfg, dict):
                    continue
                model_id = model_cfg.get("id") or model_cfg.get("name")
                if not model_id:
                    continue

                key = f"{provider_name}/{model_id}"
                entry = dict(default_models.get(key) or {})
                for invalid_key in (
                    "maxTokens",
                    "contextWindow",
                    "contextWindowTokens",
                    "temperature",
                    "reasoningEffort",
                ):
                    entry.pop(invalid_key, None)

                params = model_cfg.get("params")
                if isinstance(params, dict):
                    entry["params"] = deep_merge(dict(entry.get("params") or {}), params)
                if entry:
                    default_models[key] = entry

    def _ensure_plugin_load_paths(self, config: dict) -> None:
        if self._runtime().get("disable_plugins", False):
            return
        plugin_paths = []
        source_home = self._source_config_dir()
        for link in self._configured_home_links():
            rel_path = Path(link)
            if rel_path.parts[:1] == ("extensions",):
                plugin_paths.append(str(source_home / rel_path))
        if not plugin_paths:
            return
        load = config.setdefault("plugins", {}).setdefault("load", {})
        existing = load.get("paths", [])
        if not isinstance(existing, list):
            existing = []
        seen = set()
        load["paths"] = [
            path for path in [*existing, *plugin_paths]
            if not (path in seen or seen.add(path))
        ]

    def _ensure_temp_config(self) -> None:
        if self._temp_home:
            return
        runtime = self._runtime()
        home_mode = runtime.get("home_mode", "isolated_copy")
        if home_mode == "global":
            self._temp_home = str(Path.home())
            return
        home_dir = Path(tempfile.mkdtemp(prefix="omnimemeval-openclaw-"))
        config_dir = home_dir / ".openclaw"
        config_dir.mkdir(parents=True)
        self._link_global_openclaw_paths(config_dir)
        config = self._build_openclaw_config()
        (config_dir / "openclaw.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n"
        )
        self._temp_home = str(home_dir)

    def prepare_task(self, task: dict, env_info: dict, session: SessionSpec) -> None:
        workspace_dir = env_info.get("workspace_dir")
        self._workspace_dir = str(Path(workspace_dir).resolve()) if workspace_dir else None
        if not self._workspace_dir:
            raise RuntimeError("OpenClaw task is missing its framework workspace_dir")
        Path(self._workspace_dir).mkdir(parents=True, exist_ok=True)
        if self._runtime().get("home_mode", "isolated_copy") == "global":
            raise RuntimeError(
                "OpenClaw runtime.home_mode=global is incompatible with per-trial workspace isolation"
            )
        self._task_env = {}
        self._task_env_info = dict(env_info or {})
        self._task_env_info["phase"] = session.metadata.get("phase")
        self._ensure_temp_config()

    def cleanup_task(self) -> None:
        self.stop_gateway()
        if self._temp_home and self._temp_home != str(Path.home()):
            shutil.rmtree(self._temp_home, ignore_errors=True)
        self._temp_home = None
        self._workspace_dir = None
        self._task_env = {}
        self._task_env_info = {}
        self._gateway_port = None
        self._gateway_token = None
        self._gateway_log = None

    def _session_dir(self) -> Path:
        if self._temp_home:
            return Path(self._temp_home) / ".openclaw" / "agents" / "main" / "sessions"
        return Path.home() / ".openclaw" / "agents" / "main" / "sessions"

    def _trajectory_file(self, session: SessionSpec) -> Path:
        return self._session_file(session).with_name(
            f"{session.cli_session_id}.trajectory.jsonl"
        )

    def _build_cli_cmd(self, prompt: str, session: SessionSpec, timeout: int) -> list[str]:
        self._ensure_temp_config()
        cmd = [
            self._command(),
            "agent",
            "--session-id",
            session.cli_session_id,
            "--message",
            prompt,
            "--timeout",
            str(timeout),
            "--json",
        ]
        if self._transport() != "gateway" and self._runtime().get("local", False):
            cmd.insert(2, "--local")
        return cmd

    def _get_subprocess_env(self, session: SessionSpec) -> dict[str, str]:
        self._ensure_temp_config()
        env = dict(os.environ)
        if self._transport() == "gateway":
            port, token = self._ensure_gateway_identity()
            env["OPENCLAW_GATEWAY_URL"] = f"ws://127.0.0.1:{port}"
            env["OPENCLAW_GATEWAY_TOKEN"] = token
        elif self._runtime().get("strip_gateway", True):
            for key in (
                "OPENCLAW_GATEWAY_URL",
                "OPENCLAW_GATEWAY_TOKEN",
                "OPENCLAW_GATEWAY_REMOTE_URL",
                "OPENCLAW_GATEWAY_REMOTE_TOKEN",
            ):
                env.pop(key, None)
        env.update({str(k): str(v) for k, v in (self.config.get("env") or {}).items()})
        if self._temp_home:
            env["OPENCLAW_HOME"] = self._temp_home
        context_env = (
            self.config.get("session", {}).get("expose_context_env")
            or "OMNIMEMEVAL_AGENT_CONTEXT"
        )
        env[context_env] = json.dumps(session.to_dict(), ensure_ascii=False)
        if any(env.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")):
            parts = env.get("NODE_OPTIONS", "").split()
            if "--use-env-proxy" not in parts:
                parts.append("--use-env-proxy")
                env["NODE_OPTIONS"] = " ".join(parts)
        return env

    def _ensure_gateway_identity(self) -> tuple[int, str]:
        if self._gateway_port is None:
            self._gateway_port = self._find_free_port()
        if self._gateway_token is None:
            self._gateway_token = secrets.token_hex(32)
        return self._gateway_port, self._gateway_token

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    @staticmethod
    def _tcp_ready(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            return False

    def start_gateway(self) -> None:
        if self._transport() != "gateway":
            return
        if self._gateway_proc and self._gateway_proc.poll() is None:
            return
        self._ensure_temp_config()
        port, token = self._ensure_gateway_identity()
        log_dir = Path(self._temp_home or str(Path.home())) / ".openclaw" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._gateway_log = log_dir / f"gateway-{port}.log"
        log_fh = self._gateway_log.open("a", encoding="utf-8", buffering=1)
        cmd = [
            self._command(),
            "gateway",
            "run",
            "--port",
            str(port),
            "--bind",
            "loopback",
            "--auth",
            "token",
            "--token",
            token,
            "--ws-log",
            str(self._runtime().get("gateway_ws_log", "compact")),
        ]
        self._gateway_proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            env=self._get_subprocess_env_for_gateway(),
            cwd=self._workspace_dir,
            start_new_session=True,
        )
        log_fh.close()
        timeout = self._runtime_float(self._runtime(), "gateway_start_timeout", 60.0)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._gateway_proc.poll() is not None:
                raise RuntimeError(
                    f"OpenClaw gateway exited early with code {self._gateway_proc.returncode}; "
                    f"log={self._gateway_log}"
                )
            if self._tcp_ready(port):
                return
            time.sleep(0.5)
        raise RuntimeError(f"OpenClaw gateway did not start on port {port}; log={self._gateway_log}")

    def _get_subprocess_env_for_gateway(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in (self.config.get("env") or {}).items()})
        if self._temp_home:
            env["OPENCLAW_HOME"] = self._temp_home
        if any(env.get(name) for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")):
            parts = env.get("NODE_OPTIONS", "").split()
            if "--use-env-proxy" not in parts:
                parts.append("--use-env-proxy")
                env["NODE_OPTIONS"] = " ".join(parts)
        return env

    def stop_gateway(self) -> None:
        proc = self._gateway_proc
        self._gateway_proc = None
        if proc is None or proc.poll() is not None:
            return
        grace = self._runtime_float(self._runtime(), "gateway_stop_timeout", 60.0)
        self._terminate_process_tree(proc, grace)

    def call(self, prompt: str, session: SessionSpec, timeout: int = 3600) -> dict:
        self.start_gateway()
        cmd = self._build_cli_cmd(prompt, session, timeout)
        env = self._get_subprocess_env(session)
        runtime = self._runtime()
        poll_interval = self._runtime_float(
            runtime, "session_completion_poll_interval", 0.5
        )
        exit_grace = self._runtime_float(
            runtime, "session_completion_exit_grace_seconds", 2.0
        )
        terminate_grace = self._runtime_float(
            runtime, "session_completion_terminate_grace_seconds", 5.0
        )
        timeout_grace = self._runtime_float(runtime, "cli_timeout_grace_seconds", 60.0)
        hard_timeout = timeout + timeout_grace
        recover_from_session = bool(runtime.get("recover_from_session", True))

        start = time.time()
        stdout_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
        stderr_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False)
        stdout_path = Path(stdout_file.name)
        stderr_path = Path(stderr_file.name)
        proc: subprocess.Popen | None = None
        recovered_response = ""
        response_seen_at: float | None = None
        recovered = False
        timed_out = False
        trajectory_terminal_state: dict[str, Any] | None = None
        completed_baseline = 0
        if recover_from_session:
            completed_baseline = self._terminal_assistant_response_state(
                session,
            )["count"]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
                cwd=self._workspace_dir,
                start_new_session=True,
            )

            while proc.poll() is None:
                now = time.time()
                if now - start >= hard_timeout:
                    timed_out = True
                    self._terminate_process_tree(proc, terminate_grace)
                    break

                if recover_from_session:
                    state = self._terminal_assistant_response_state(
                        session,
                    )
                    response = (
                        state["text"] if state["count"] > completed_baseline else ""
                    )
                    if response:
                        if response_seen_at is None or response != recovered_response:
                            response_seen_at = now
                            recovered_response = response
                        elif now - response_seen_at >= exit_grace:
                            recovered = True
                            self._terminate_process_tree(proc, terminate_grace)
                            break

                    trajectory_state = self._trajectory_terminal_state(session)
                    if trajectory_state and self._is_error_trajectory_state(trajectory_state):
                        trajectory_terminal_state = trajectory_state
                        self._terminate_process_tree(proc, terminate_grace)
                        break

                time.sleep(max(0.05, poll_interval))

            returncode = proc.poll() if proc else None
        except BaseException:
            if proc and proc.poll() is None:
                self._terminate_process_tree(proc, terminate_grace)
            stdout_path.unlink(missing_ok=True)
            stderr_path.unlink(missing_ok=True)
            raise
        finally:
            stdout_file.close()
            stderr_file.close()

        stdout = self._read_text(stdout_path)
        stderr = self._read_text(stderr_path)
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        elapsed = time.time() - start

        if recovered:
            return {
                "response": recovered_response,
                "completion_status": "completed",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli_session",
                "returncode": returncode,
                "stderr": stderr[:2000] or None,
                "cli_session_recovered": True,
                "cli_terminated_after_session_completion": True,
            }

        if trajectory_terminal_state:
            response = self._last_assistant_response_from_session(session)
            return {
                "response": response,
                "completion_status": (
                    "timeout"
                    if trajectory_terminal_state.get("timedOut")
                    or trajectory_terminal_state.get("idleTimedOut")
                    else "error"
                ),
                "elapsed_sec": round(elapsed, 1),
                "method": "cli_trajectory",
                "returncode": returncode,
                "stderr": stderr[:2000] or None,
                "error": trajectory_terminal_state.get("promptError")
                or trajectory_terminal_state.get("error")
                or "OpenClaw trajectory ended before the CLI exited",
                "openclaw_trajectory_state": trajectory_terminal_state,
                "cli_terminated_after_trajectory_end": True,
            }

        if timed_out:
            data = {
                "response": "",
                "completion_status": "timeout",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "returncode": returncode,
                "stderr": stderr[:2000] or None,
                "error": f"subprocess timed out after {hard_timeout:.0f}s",
            }
            data.update(self._parse_timeout_extra(session))
            return data

        completed = subprocess.CompletedProcess(
            args=cmd,
            returncode=int(returncode or 0),
            stdout=stdout,
            stderr=stderr,
        )
        data = {
            "response": completed.stdout,
            "completion_status": "completed" if completed.returncode == 0 else "error",
            "elapsed_sec": round(elapsed, 1),
            "method": "cli",
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "")[:2000] or None,
        }
        data.update(self._parse_extra(completed))
        return data

    @staticmethod
    def _runtime_float(runtime: dict, key: str, default: float) -> float:
        try:
            return float(runtime.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _terminate_process_tree(proc: subprocess.Popen, grace_seconds: float) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=max(0.1, grace_seconds))
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            proc.wait(timeout=max(0.1, grace_seconds))
        except subprocess.TimeoutExpired:
            pass

    def _parse_extra(self, result) -> dict:
        parsed = {}
        for source in (result.stdout, result.stderr):
            if not source:
                continue
            start = source.rfind("\n{")
            if start < 0:
                start = 0 if source.startswith("{") else -1
            else:
                start += 1
            if start < 0:
                continue
            try:
                data = json.loads(source[start:])
            except (json.JSONDecodeError, TypeError):
                continue
            payloads = data.get("payloads") or data.get("result", {}).get("payloads")
            if payloads:
                text = "\n".join(
                    item.get("text", "")
                    for item in payloads
                    if isinstance(item, dict) and item.get("text")
                )
                if text:
                    parsed.update({"response": text, "response_json": data})
                    break

        stderr = result.stderr or ""
        if "isError=true" in stderr:
            parsed["completion_status"] = "error"
            error = self._extract_embedded_error(stderr)
            if error:
                parsed["error"] = error
        return parsed

    @staticmethod
    def _extract_embedded_error(stderr: str) -> str | None:
        marker = " error="
        idx = stderr.rfind(marker)
        if idx < 0:
            return None
        text = stderr[idx + len(marker):].strip()
        for end_marker in (" rawError=", "\n"):
            end_idx = text.find(end_marker)
            if end_idx >= 0:
                text = text[:end_idx]
                break
        return text[:1000] if text else None

    def _parse_timeout_extra(self, session: SessionSpec) -> dict:
        response = self._last_terminal_assistant_response_from_session(session)
        if not response:
            return {}
        return {
            "response": response,
            "completion_status": "completed",
            "cli_timeout": True,
            "error": "subprocess timed out after response was written; recovered response from OpenClaw session",
        }

    def _last_assistant_response_from_session(self, session: SessionSpec) -> str:
        return self._last_assistant_response(session, require_stop_reason=False)

    def _last_completed_assistant_response_from_session(self, session: SessionSpec) -> str:
        return self._last_terminal_assistant_response_from_session(session)

    def _last_terminal_assistant_response_from_session(self, session: SessionSpec) -> str:
        return self._terminal_assistant_response_state(session)["text"]

    def _terminal_assistant_response_state(self, session: SessionSpec) -> dict:
        return self._assistant_response_state(
            session,
            require_stop_reason=True,
            terminal_only=True,
        )

    def _last_assistant_response(
        self,
        session: SessionSpec,
        *,
        require_stop_reason: bool,
    ) -> str:
        return self._assistant_response_state(
            session,
            require_stop_reason=require_stop_reason,
            terminal_only=False,
        )["text"]

    def _assistant_response_state(
        self,
        session: SessionSpec,
        *,
        require_stop_reason: bool,
        terminal_only: bool,
    ) -> dict:
        session_file = self._session_file(session)
        if not session_file.exists():
            return {"count": 0, "text": ""}
        last_text = ""
        count = 0
        try:
            for entry in self._iter_session_entries(session_file):
                msg = entry.get("message") or entry
                if msg.get("role") != "assistant":
                    continue
                stop_reason = msg.get("stopReason")
                if require_stop_reason and not stop_reason:
                    continue
                if terminal_only and stop_reason == "toolUse":
                    continue
                text = self._message_text(msg)
                if text:
                    count += 1
                    last_text = text
        except OSError:
            return {"count": 0, "text": ""}
        return {"count": count, "text": last_text}

    @staticmethod
    def _iter_session_entries(session_file: Path):
        buffer = ""
        with session_file.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                buffer += line
                try:
                    entry = json.loads(buffer)
                except json.JSONDecodeError:
                    continue
                if isinstance(entry, dict):
                    yield entry
                buffer = ""

    @staticmethod
    def _message_text(msg: dict) -> str:
        parts = msg.get("content") or []
        if isinstance(parts, str):
            return parts
        texts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(item for item in texts if item)

    def _trajectory_terminal_state(self, session: SessionSpec) -> dict[str, Any] | None:
        path = self._trajectory_file(session)
        if not path.exists():
            return None
        state = None
        try:
            with path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = entry.get("type")
                    data = entry.get("data") or {}
                    if event_type == "trace.artifacts":
                        final_status = data.get("finalStatus")
                        if final_status:
                            state = {
                                "event": event_type,
                                "status": final_status,
                                "timedOut": bool(data.get("timedOut")),
                                "idleTimedOut": bool(data.get("idleTimedOut")),
                                "aborted": bool(data.get("aborted")),
                                "promptError": data.get("promptError"),
                            }
                    elif event_type == "session.ended":
                        state = {
                            "event": event_type,
                            "status": data.get("status"),
                            "timedOut": bool(data.get("timedOut")),
                            "idleTimedOut": bool(data.get("idleTimedOut")),
                            "aborted": bool(data.get("aborted")),
                            "promptError": data.get("promptError"),
                        }
        except OSError:
            return None
        return state

    @staticmethod
    def _is_error_trajectory_state(state: dict[str, Any]) -> bool:
        status = str(state.get("status") or "").lower()
        return status in {"error", "failed", "timeout"} or bool(
            state.get("timedOut") or state.get("idleTimedOut")
        )

    def _parse_session_entry(self, entry: dict, stats: dict) -> None:
        if entry.get("role") == "assistant":
            stats["turns"] += 1
        if entry.get("type") == "message":
            msg = entry.get("message") or {}
            if msg.get("role") == "assistant":
                stats["turns"] += 1
                if msg.get("stopReason"):
                    stats["last_stop_reason"] = msg.get("stopReason")
            usage = msg.get("usage") or {}
        else:
            usage = entry.get("usage") or {}
        stats["input"] += int(usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input") or 0)
        stats["output"] += int(usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output") or 0)
        stats["total"] += int(usage.get("total_tokens") or usage.get("total") or 0)
