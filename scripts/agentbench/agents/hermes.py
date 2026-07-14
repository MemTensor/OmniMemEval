from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from agentbench.agents.base import AgentAdapter
from agentbench.config import deep_merge
from agentbench.session import SessionSpec


class HermesAgentAdapter(AgentAdapter):
    """AgentBench adapter for Hermes CLI.

    Hermes creates its own session ids in ``state.db``.  AgentBench keeps a
    framework-safe ``SessionSpec.cli_session_id`` and maps it to the real Hermes
    id after the first call, so train feedback can resume the same Hermes chat.
    """

    name = "hermes"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._temp_home: str | None = None
        self._workspace_dir: str | None = None
        self._task_env_info: dict[str, Any] = {}
        self._session_aliases: dict[str, str] = {}
        self._hermes_session_ids: dict[str, str] = {}
        self._mcp_server_names: list[str] = []

    def _command(self) -> str:
        return self.config.get("command") or self.config.get("agent", {}).get("command") or "hermes"

    def _runtime(self) -> dict[str, Any]:
        return dict(self.config.get("runtime") or {})

    def _phase(self) -> str:
        phase = str(self._task_env_info.get("phase") or "").strip().lower()
        if phase == "train":
            return "train"
        if phase == "test" or phase.startswith("test_"):
            return "test"
        return phase

    def _global_home(self) -> Path:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()

    def _global_config_path(self) -> Path:
        return self._global_home() / "config.yaml"

    def _session_dir(self) -> Path:
        if self._temp_home:
            return Path(self._temp_home) / "sessions"
        return self._global_home() / "sessions"

    @staticmethod
    def _safe_session_name(session_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in session_id)
        return safe[:96] or "session"

    def _session_file(self, session: SessionSpec) -> Path:
        alias = self._session_aliases.get(session.cli_session_id)
        if alias:
            return Path(alias)
        return self._session_dir() / f"{self._safe_session_name(session.cli_session_id)}.jsonl"

    def _configured_home_links(self) -> list[str]:
        runtime = self._runtime()
        links = list(runtime.get("home_links") or self.config.get("home_links") or [])
        env_links = os.environ.get("OMNIMEMEVAL_HERMES_HOME_LINKS") or os.environ.get(
            "EVOAGENTBENCH_HERMES_HOME_LINKS"
        )
        if env_links:
            try:
                parsed = json.loads(env_links)
            except json.JSONDecodeError:
                parsed = [item for item in env_links.split(os.pathsep) if item]
            if not isinstance(parsed, list):
                raise RuntimeError("OMNIMEMEVAL_HERMES_HOME_LINKS must be a JSON list or path list")
            links.extend(parsed)

        seen: set[str] = set()
        result: list[str] = []
        for link in links:
            if not isinstance(link, str):
                raise RuntimeError(f"Hermes home link must be a string: {link!r}")
            if link not in seen:
                result.append(link)
                seen.add(link)
        return result

    def _link_global_home_paths(self, home_dir: Path) -> None:
        global_home = self._global_home()
        for link in self._configured_home_links():
            rel_path = Path(link)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise RuntimeError(f"Hermes home_links must be relative paths: {link}")
            source = global_home / rel_path
            if not source.exists():
                raise RuntimeError(f"Hermes home link source does not exist: {source}")
            target = home_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                continue
            target.symlink_to(source, target_is_directory=source.is_dir())

    def _copy_optional_home_file(self, filename: str, home_dir: Path) -> None:
        source = self._global_home() / filename
        if source.exists() and source.is_file():
            shutil.copy2(source, home_dir / filename)

    def _base_config(self) -> dict[str, Any]:
        config_path = self._global_config_path()
        if config_path.exists():
            return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return {}

    def _apply_agent_config(self, config: dict[str, Any]) -> None:
        model = self.config.get("model")
        provider = self.config.get("provider")
        providers = self.config.get("providers") or {}

        model_cfg = config.setdefault("model", {})
        if model:
            model_cfg["default"] = model
        if provider:
            model_cfg["provider"] = provider

        provider_cfg = providers.get(provider) if provider else None
        if provider_cfg is None and len(providers) == 1:
            provider_cfg = next(iter(providers.values()))
        if isinstance(provider_cfg, dict):
            api_base = provider_cfg.get("apiBase") or provider_cfg.get("base_url") or provider_cfg.get("baseURL")
            api_key = provider_cfg.get("apiKey") or provider_cfg.get("api_key")
            if api_base:
                model_cfg["base_url"] = api_base
            if api_key:
                model_cfg["api_key"] = api_key
            if provider == "custom" and model and api_base:
                custom_providers = list(config.get("custom_providers") or [])
                custom_providers = [
                    entry
                    for entry in custom_providers
                    if not (isinstance(entry, dict) and entry.get("name") == model)
                ]
                custom_provider = {
                    "name": model,
                    "base_url": api_base,
                    "api_key": api_key or "EMPTY",
                    "model": model,
                }
                extra_body = provider_cfg.get("extra_body") or provider_cfg.get("extraBody")
                if isinstance(extra_body, dict):
                    custom_provider["extra_body"] = extra_body
                api_mode = provider_cfg.get("api_mode") or provider_cfg.get("apiMode")
                if api_mode:
                    custom_provider["api_mode"] = api_mode
                custom_providers.append(custom_provider)
                config["custom_providers"] = custom_providers

        agent_defaults = config.setdefault("agent", {})
        if self.config.get("maxTokens") is not None:
            agent_defaults["max_tokens"] = self.config["maxTokens"]
        if self.config.get("reasoningEffort") is not None:
            agent_defaults["reasoning_effort"] = self.config["reasoningEffort"]

        memory_overrides = self.config.get("memory") or {}
        if isinstance(memory_overrides, dict):
            memory_cfg = config.setdefault("memory", {})
            for key, value in memory_overrides.items():
                if value is not None:
                    memory_cfg[key] = value

        if self._workspace_dir:
            config.setdefault("terminal", {})["cwd"] = self._workspace_dir

        patch = self.config.get("hermes_config_patch") or {}
        if patch:
            config.update(deep_merge(dict(config), dict(patch)))

        phase_patches = self.config.get("hermes_phase_config_patches") or {}
        phase_patch = phase_patches.get(self._phase()) if isinstance(phase_patches, dict) else None
        if isinstance(phase_patch, dict) and phase_patch:
            config.update(deep_merge(dict(config), dict(phase_patch)))

        config.setdefault("sessions", {})["write_json_snapshots"] = True
        config.setdefault("display", {})["streaming"] = False
        config.setdefault("streaming", {})["enabled"] = False

    @staticmethod
    def _normalize_mcp_entry(cfg: dict[str, Any]) -> dict[str, Any]:
        entry: dict[str, Any] = {}
        if cfg.get("command"):
            entry["command"] = cfg["command"]
            if cfg.get("args"):
                entry["args"] = list(cfg["args"])
            if cfg.get("env"):
                entry["env"] = dict(cfg["env"])
        elif cfg.get("url"):
            entry["url"] = cfg["url"]
            transport = cfg.get("transport") or cfg.get("type")
            if transport == "sse":
                entry["transport"] = "sse"
            elif transport and transport not in {"auto", "streamable_http"}:
                entry["transport"] = transport
            if cfg.get("headers"):
                entry["headers"] = dict(cfg["headers"])
        else:
            raise RuntimeError("Hermes MCP config requires either command/args or url")

        for key in ("timeout", "connect_timeout", "keepalive_interval", "supports_parallel_tool_calls"):
            if key in cfg:
                entry[key] = cfg[key]
        entry["enabled"] = cfg.get("enabled", True)
        return entry

    @staticmethod
    def _web_tool_domains() -> set[str]:
        raw = os.environ.get("HERMES_WEB_TOOL_DOMAINS", "knowledge_work")
        return {item.strip() for item in raw.replace(",", " ").split() if item.strip()}

    def _apply_domain_tool_policy(self, config: dict[str, Any]) -> None:
        domain_name = str(self._task_env_info.get("domain_name") or "")
        if not domain_name or domain_name in self._web_tool_domains():
            return
        toolsets = config.setdefault("platform_toolsets", {})
        cli_toolsets = toolsets.get("cli")
        if isinstance(cli_toolsets, list):
            toolsets["cli"] = [name for name in cli_toolsets if name != "web"]

    def _write_config(self) -> None:
        if not self._temp_home:
            raise RuntimeError("Hermes temp home has not been created")

        config = self._base_config()
        self._apply_agent_config(config)
        self._apply_domain_tool_policy(config)

        mcp_servers = self._task_env_info.get("mcp_servers") or {}
        disabled_tools = self._task_env_info.get("disabled_tools") or []
        if mcp_servers:
            normalized = {
                name: self._normalize_mcp_entry(server_cfg)
                for name, server_cfg in mcp_servers.items()
            }
            config["mcp_servers"] = normalized
            self._mcp_server_names = list(normalized.keys())
            if disabled_tools:
                config.setdefault("platform_toolsets", {})["cli"] = self._mcp_server_names
        elif self._mcp_server_names:
            config.setdefault("platform_toolsets", {})["cli"] = self._mcp_server_names

        config_path = Path(self._temp_home) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def _ensure_temp_config(self) -> None:
        if self._temp_home:
            return
        if self._runtime().get("home_mode") == "global":
            self._temp_home = str(self._global_home())
            return

        home_dir = Path(tempfile.mkdtemp(prefix="omnimemeval-hermes-"))
        self._temp_home = str(home_dir)
        (home_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (home_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._link_global_home_paths(home_dir)
        self._install_phase_memory_provider(home_dir)
        self._copy_optional_home_file(".env", home_dir)
        self._copy_optional_home_file("auth.json", home_dir)
        self._write_config()

    def prepare_task(self, task: dict, env_info: dict, session: SessionSpec) -> None:
        workspace_dir = env_info.get("workspace_dir")
        self._workspace_dir = str(Path(workspace_dir).resolve()) if workspace_dir else None
        self._task_env_info = dict(env_info or {})
        self._task_env_info["domain_name"] = session.metadata.get("domain", "")
        self._task_env_info["phase"] = session.metadata.get("phase", "")
        self._ensure_temp_config()

    def _install_phase_memory_provider(self, home_dir: Path) -> None:
        if self._phase() != "test":
            return
        provider_name = str(self._runtime().get("test_memory_provider") or "").strip()
        if not provider_name:
            return
        source = Path(__file__).resolve().parents[1] / "integrations" / provider_name
        if not (source / "__init__.py").exists():
            raise RuntimeError(f"Hermes test memory provider is missing: {source}")
        target = home_dir / "plugins" / provider_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)

    def cleanup_task(self) -> None:
        if self._temp_home and self._temp_home != str(self._global_home()):
            shutil.rmtree(self._temp_home, ignore_errors=True)
        self._temp_home = None
        self._workspace_dir = None
        self._task_env_info = {}
        self._session_aliases = {}
        self._hermes_session_ids = {}
        self._mcp_server_names = []

    def _build_cli_cmd(self, prompt: str, session: SessionSpec, timeout: int) -> list[str]:
        self._ensure_temp_config()
        session_file = self._session_dir() / f"{self._safe_session_name(session.cli_session_id)}.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        session_file.touch(exist_ok=True)
        self._session_aliases[session.cli_session_id] = str(session_file)

        cmd = [self._command(), "chat", "-Q"]
        hermes_session_id = self._hermes_session_ids.get(session.cli_session_id)
        if hermes_session_id:
            cmd.extend(["--resume", hermes_session_id])
        cmd.extend(["--query", prompt])
        if self.config.get("model"):
            cmd.extend(["--model", self.config["model"]])
            if self.config.get("provider"):
                cmd.extend(["--provider", self.config["provider"]])
        if self._mcp_server_names:
            cmd.extend(["--toolsets", ",".join(self._mcp_server_names)])
        return cmd

    def _get_subprocess_env(self, session: SessionSpec) -> dict[str, str]:
        self._ensure_temp_config()
        env = dict(os.environ)
        if self._temp_home:
            env["HERMES_HOME"] = self._temp_home
        env["HERMES_YOLO_MODE"] = "1"
        env["HERMES_ACCEPT_HOOKS"] = "1"
        if self._workspace_dir:
            env["HERMES_CWD"] = self._workspace_dir
            env["TERMINAL_CWD"] = self._workspace_dir
        env.update({str(k): str(v) for k, v in (self.config.get("env") or {}).items()})
        context_env = (
            self.config.get("session", {}).get("expose_context_env")
            or "OMNIMEMEVAL_AGENT_CONTEXT"
        )
        env[context_env] = json.dumps(session.to_dict(), ensure_ascii=False)
        return env

    def call(self, prompt: str, session: SessionSpec, timeout: int = 3600) -> dict:
        started_at = time.time()
        cmd = self._build_cli_cmd(prompt, session, timeout)
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 60,
                env=self._get_subprocess_env(session),
            )
            elapsed = time.time() - start
            self._materialize_session_jsonl(session, started_at=started_at)
            data = {
                "response": result.stdout,
                "completion_status": "completed" if result.returncode == 0 else "error",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[:2000] or None,
            }
            data.update(self._parse_extra(result))
            if self._hermes_session_ids.get(session.cli_session_id):
                data["hermes_session_id"] = self._hermes_session_ids[session.cli_session_id]
            return data
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            self._materialize_session_jsonl(session, started_at=started_at)
            data = {
                "response": "",
                "completion_status": "timeout",
                "elapsed_sec": round(elapsed, 1),
                "method": "cli",
                "error": f"subprocess timed out after {timeout + 60}s",
            }
            if self._hermes_session_ids.get(session.cli_session_id):
                data["hermes_session_id"] = self._hermes_session_ids[session.cli_session_id]
            return data

    def _latest_snapshot(self, started_at: float | None = None) -> Path | None:
        candidates = sorted(self._session_dir().glob("session_*.json"), key=lambda p: p.stat().st_mtime)
        if started_at is not None:
            candidates = [p for p in candidates if p.stat().st_mtime >= started_at - 1]
        return candidates[-1] if candidates else None

    @staticmethod
    def _decode_json_field(value: Any) -> Any:
        if not value or not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    @staticmethod
    def _session_usage_record(session_row: sqlite3.Row) -> dict[str, int]:
        input_tokens = int(session_row["input_tokens"] or 0)
        output_tokens = int(session_row["output_tokens"] or 0)
        cache_read = int(session_row["cache_read_tokens"] or 0)
        cache_write = int(session_row["cache_write_tokens"] or 0)
        reasoning = int(session_row["reasoning_tokens"] or 0)
        return {
            "input": input_tokens,
            "output": output_tokens,
            "totalTokens": input_tokens + output_tokens + cache_read + cache_write + reasoning,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "reasoning": reasoning,
        }

    def _state_db_session(
        self,
        *,
        started_at: float | None = None,
        hermes_session_id: str | None = None,
    ) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
        db_path = Path(self._temp_home or "") / "state.db"
        if not db_path.exists():
            return None, []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            params: list[Any] = []
            where = ["source = 'cli'"]
            if hermes_session_id:
                where.append("id = ?")
                params.append(hermes_session_id)
            if started_at is not None:
                where.append("started_at >= ?")
                params.append(started_at - 1)
            session = conn.execute(
                f"""
                SELECT id, input_tokens, output_tokens, cache_read_tokens,
                       cache_write_tokens, reasoning_tokens
                FROM sessions
                WHERE {' AND '.join(where)}
                ORDER BY started_at DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
            if session is None:
                return None, []
            messages = conn.execute(
                """
                SELECT role, content, tool_call_id, tool_calls, tool_name,
                       finish_reason, reasoning, reasoning_content
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp, id
                """,
                (session["id"],),
            ).fetchall()
            return session, messages
        finally:
            conn.close()

    def _materialize_session_from_state_db(
        self,
        target: Path,
        *,
        started_at: float | None = None,
        hermes_session_id: str | None = None,
    ) -> str | None:
        try:
            session, messages = self._state_db_session(
                started_at=None if hermes_session_id else started_at,
                hermes_session_id=hermes_session_id,
            )
        except sqlite3.Error:
            return None
        if session is None or not messages:
            return None

        records = []
        usage = self._session_usage_record(session)
        for message in messages:
            rec = {
                "role": message["role"] or "",
                "content": message["content"],
            }
            if message["reasoning_content"] or message["reasoning"]:
                rec["reasoning_content"] = message["reasoning_content"] or message["reasoning"]
            tool_calls = self._decode_json_field(message["tool_calls"])
            if tool_calls:
                rec["tool_calls"] = tool_calls
            if message["tool_call_id"]:
                rec["tool_call_id"] = message["tool_call_id"]
            if message["tool_name"]:
                rec["tool_name"] = message["tool_name"]
            if message["finish_reason"]:
                rec["finish_reason"] = message["finish_reason"]
            records.append(rec)

        for rec in reversed(records):
            if rec.get("role") == "assistant":
                rec["usage"] = usage
                break

        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(session["id"])

    @staticmethod
    def _message_to_jsonl_record(message: dict[str, Any]) -> dict[str, Any]:
        rec = {"role": message.get("role", "")}
        for key in ("content", "reasoning_content", "tool_calls", "tool_call_id"):
            if message.get(key):
                rec[key] = message[key]
        return rec

    def _materialize_session_from_snapshot(self, target: Path, *, started_at: float | None = None) -> None:
        snapshot = self._latest_snapshot(started_at=started_at)
        if not snapshot or not snapshot.exists():
            return
        try:
            data = json.loads(snapshot.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for message in messages:
                if isinstance(message, dict):
                    f.write(json.dumps(self._message_to_jsonl_record(message), ensure_ascii=False) + "\n")

    def _materialize_session_jsonl(self, session: SessionSpec, *, started_at: float | None = None) -> None:
        target = self._session_file(session)
        hermes_session_id = self._hermes_session_ids.get(session.cli_session_id)
        found = self._materialize_session_from_state_db(
            target,
            hermes_session_id=hermes_session_id,
            started_at=started_at,
        )
        if found:
            self._hermes_session_ids[session.cli_session_id] = found
            return
        self._materialize_session_from_snapshot(target, started_at=started_at)

    @staticmethod
    def _request_dump_enabled() -> bool:
        return os.environ.get("HERMES_DUMP_REQUESTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _copy_request_dumps(self, trial_dir: Path) -> None:
        if not self._request_dump_enabled() or not self._temp_home:
            return
        dumps: list[Path] = []
        for dump_dir in (Path(self._temp_home) / "logs", Path(self._temp_home) / "sessions"):
            if dump_dir.exists():
                dumps.extend(sorted(dump_dir.glob("request_dump_*.json")))
        if not dumps:
            return
        target_dir = trial_dir / "hermes_request_dumps"
        target_dir.mkdir(parents=True, exist_ok=True)
        first_dump = sorted(dumps)[0]
        shutil.copy2(first_dump, target_dir / first_dump.name)

    def collect_session(self, session: SessionSpec, trial_dir: Path) -> dict:
        self._materialize_session_jsonl(session)
        stats = super().collect_session(session, trial_dir)
        try:
            self._copy_request_dumps(trial_dir)
        except OSError:
            pass
        return stats

    def _parse_session_entry(self, entry: dict, stats: dict) -> None:
        if entry.get("role") == "assistant":
            stats["turns"] += 1
        usage = entry.get("usage") or entry.get("token_usage")
        if isinstance(usage, dict):
            stats["input"] += int(usage.get("input") or usage.get("input_tokens") or 0)
            stats["output"] += int(usage.get("output") or usage.get("output_tokens") or 0)
            stats["total"] += int(
                usage.get("totalTokens")
                or usage.get("total_tokens")
                or usage.get("total")
                or 0
            )
        if entry.get("finish_reason"):
            stats["last_stop_reason"] = entry.get("finish_reason")

    def should_retry(self, result: dict) -> str | None:
        turns = result.get("token_usage", {}).get("turns", 0)
        completion = result.get("agent_result", {}).get("completion_status", "")
        if turns == 0 and completion == "completed":
            return "zero_turns"
        return super().should_retry(result)
