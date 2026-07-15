from __future__ import annotations

import os
from string import Template
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentbench.config import write_json


class CommandMemoryLifecycle:
    """Command-backed lifecycle hooks for agent memory plugins.

    The runner owns the benchmark protocol; plugin-specific behavior lives in
    YAML as shell commands.  Templates use ``@name@`` tokens so embedded shell,
    JSON, and Python snippets can still use braces freely.
    """

    REQUIRED_STAGES = frozenset({
        "prepare_global_snapshot",
        "clear",
        "backup",
        "restore",
        "cleanup",
    })

    def __init__(
        self,
        *,
        config: dict[str, Any],
        project_dir: Path,
        run_dir: Path,
        run_id: str,
        version: str,
    ) -> None:
        self.config = config
        self.project_dir = project_dir
        self.run_dir = run_dir
        self.run_id = run_id
        self.version = version
        self.run_date = datetime.now().strftime("%F")
        self.plugin = str(config.get("plugin") or config.get("name") or "memory")
        self.agent = str(config.get("agent") or "")
        self._saved_process_env: dict[str, str | None] | None = None
        self._original_home_env = self._capture_original_home_env()
        self.backup_dir = self._path(config.get("backup_dir") or "~/memory_backup")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = run_dir / "memory_lifecycle.log"
        self.manifest_file = run_dir / "memory_lifecycle.json"
        self._events: list[dict[str, Any]] = []
        self._validate_configuration()

    def activate_runtime_env(self) -> dict[str, str]:
        """Expose lifecycle runtime paths to adapters for this protocol run.

        Lifecycle YAML uses run-scoped homes.  Agent adapters and feedback code
        are created outside lifecycle subprocesses, so they must see the same
        values in ``os.environ``.  The original process environment is retained
        and can be restored exactly with :meth:`restore_runtime_env`.
        """

        if self._saved_process_env is not None:
            return self.runtime_env()

        values = self.runtime_env()
        original_key = self._original_home_key()
        if original_key:
            values[original_key] = self._original_home_env
        self._saved_process_env = {key: os.environ.get(key) for key in values}
        os.environ.update(values)
        self._record("activate_runtime_env", "", {"keys": sorted(values)})
        return dict(values)

    def restore_runtime_env(self) -> None:
        """Restore the process environment captured by activate_runtime_env."""

        saved = self._saved_process_env
        if saved is None:
            return
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._saved_process_env = None
        self._record("restore_runtime_env", "", {"keys": sorted(saved)})
        self._write_manifest()

    def runtime_env(self) -> dict[str, str]:
        """Return rendered lifecycle environment values without mutating it."""

        source = self.config.get("env") or {}
        if not isinstance(source, dict):
            raise TypeError("Lifecycle env must be a mapping")
        rendered: dict[str, str] = {}
        variables = os.environ.copy()
        variables[self._original_home_key() or "OMNIMEMEVAL_ORIGINAL_HOME"] = (
            self._original_home_env
        )
        for key, value in source.items():
            text = self._render(str(value), "", backup_file="")
            text = Template(text).safe_substitute(variables)
            rendered[str(key)] = text
            variables[str(key)] = text
        return rendered

    def validate(self, domain: str) -> None:
        self._run_stage("validate", domain)

    def set_mode(self, mode: str, domain: str) -> None:
        modes = self.config.get("modes") or {}
        if isinstance(modes, dict) and mode in modes:
            self._run_commands(f"set_mode:{mode}", modes[mode], domain)
            return
        self._run_stage(f"set_mode_{mode}", domain)

    def clear(self, domain: str) -> None:
        self._run_stage("clear", domain)

    def wait_settle(self, domain: str, *, expected_trials: int | None = None) -> None:
        extra_env = {}
        if expected_trials is not None:
            if expected_trials < 1:
                raise ValueError("expected_trials must be >= 1")
            extra_env["OMNIMEMEVAL_EXPECTED_TRIALS"] = str(expected_trials)
        if self._has_stage("wait_settle"):
            self._run_stage("wait_settle", domain, extra_env=extra_env)
            return
        seconds = int(self.config.get("settle_seconds", 0) or 0)
        if seconds > 0:
            self._record("wait_settle", domain, {"seconds": seconds, "method": "sleep"})
            time.sleep(seconds)

    def prepare_global_snapshot(self, domain: str) -> Path:
        """Back up the user's pre-evaluation memory before destructive stages."""

        snapshot_file = self.global_backup_file(domain)
        snapshot_file.parent.mkdir(parents=True, exist_ok=True)
        self._run_stage(
            "prepare_global_snapshot",
            domain,
            backup_file=snapshot_file,
        )
        if (
            not snapshot_file.exists()
            and self.config.get("require_global_backup_file", True)
        ):
            raise RuntimeError(
                "Global snapshot command did not create expected file: "
                f"{snapshot_file}"
            )
        self._record(
            "prepare_global_snapshot",
            domain,
            {"global_backup_file": str(snapshot_file)},
        )
        return snapshot_file

    def cleanup(
        self,
        domain: str,
        global_backup_file: str | os.PathLike[str] | None = None,
    ) -> None:
        """Remove run-scoped state without overwriting the user's global home.

        The global snapshot is disaster-recovery material, not an automatic
        rollback source: restoring it here could overwrite legitimate user
        writes made while a long evaluation was running.  Its path is exposed
        to the command for auditing, even if snapshot preparation failed.
        """

        snapshot = (
            Path(global_backup_file).expanduser()
            if global_backup_file is not None
            else self.global_backup_file(domain)
        )
        try:
            self._run_stage("cleanup", domain, backup_file=snapshot)
            self._record("cleanup", domain, {"global_backup_file": str(snapshot)})
        finally:
            self._write_manifest()

    def backup(self, domain: str) -> Path:
        backup_file = self.backup_file(domain)
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        self._run_stage("backup", domain, backup_file=backup_file)
        if not backup_file.exists() and self.config.get("require_backup_file", True):
            raise RuntimeError(f"Backup command did not create expected file: {backup_file}")
        self._record("backup", domain, {"backup_file": str(backup_file)})
        return backup_file

    def restore(self, domain: str, backup_file: str | os.PathLike[str]) -> None:
        backup = Path(backup_file).expanduser()
        if not backup.exists():
            raise FileNotFoundError(f"Backup file not found for {domain}: {backup}")
        self._run_stage("restore", domain, backup_file=backup)
        self._record("restore", domain, {"backup_file": str(backup)})

    def finalize(self, domain: str | None = None) -> None:
        try:
            self._run_stage("finalize", domain or "")
        finally:
            self._write_manifest()

    def backup_file(self, domain: str) -> Path:
        template = str(
            self.config.get("backup_file_template")
            or "@backup_dir@/@plugin@-@domain@-@run_date@-@run_id@.tar.gz"
        )
        rendered = self._render(template, domain, backup_file="")
        return self._path(rendered)

    def global_backup_file(self, domain: str) -> Path:
        template = str(
            self.config.get("global_backup_file_template")
            or "@backup_dir@/@plugin@-global-@domain@-@run_date@-@run_id@.tar.gz"
        )
        rendered = self._render(template, domain, backup_file="")
        return self._path(rendered)

    def _validate_configuration(self) -> None:
        commands = self.config.get("commands")
        if not isinstance(commands, dict):
            raise ValueError("Lifecycle config must define a commands mapping")
        missing = sorted(stage for stage in self.REQUIRED_STAGES if not self._has_stage(stage))
        if missing:
            raise ValueError(
                "Lifecycle config is missing required non-empty command stage(s): "
                + ", ".join(missing)
            )
        for stage, command in commands.items():
            if command in (None, "", []):
                continue
            if not isinstance(command, (str, list)):
                raise TypeError(
                    f"Lifecycle stage {stage!r} must be a string or list of strings"
                )

    def _has_stage(self, stage: str) -> bool:
        commands = self.config.get("commands") or {}
        if not isinstance(commands, dict) or stage not in commands:
            return False
        value = commands[stage]
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(str(item).strip() for item in value)
        return value is not None

    def _run_stage(
        self,
        stage: str,
        domain: str,
        *,
        backup_file: str | os.PathLike[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        commands = self.config.get("commands") or {}
        if not isinstance(commands, dict):
            return
        if stage not in commands:
            return
        self._run_commands(
            stage,
            commands[stage],
            domain,
            backup_file=backup_file,
            extra_env=extra_env,
        )

    def _run_commands(
        self,
        stage: str,
        commands: Any,
        domain: str,
        *,
        backup_file: str | os.PathLike[str] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        if commands in (None, "", []):
            return
        if isinstance(commands, str):
            command_list = [commands]
        elif isinstance(commands, list):
            command_list = [str(item) for item in commands if str(item).strip()]
        else:
            raise TypeError(f"Lifecycle stage {stage!r} must be a string or list of strings")

        backup = Path(backup_file).expanduser() if backup_file else self.backup_file(domain)
        for index, command in enumerate(command_list, start=1):
            rendered = self._render(command, domain, backup_file=str(backup))
            self._run_command(
                stage, domain, rendered, index=index, extra_env=extra_env
            )

    def _run_command(
        self,
        stage: str,
        domain: str,
        command: str,
        *,
        index: int,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        started = datetime.now(timezone.utc)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started.isoformat()}] stage={stage} domain={domain} command#{index}\n")
            log.write(command.rstrip() + "\n")
            log.flush()
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_dir,
                env=self._env(domain, extra_env=extra_env),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        ended = datetime.now(timezone.utc)
        event = {
            "stage": stage,
            "domain": domain,
            "command_index": index,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "returncode": result.returncode,
        }
        self._events.append(event)
        if result.returncode != 0:
            raise RuntimeError(
                f"Memory lifecycle stage={stage} domain={domain} command#{index} "
                f"failed with returncode={result.returncode}. See {self.log_file}"
            )

    def _env(
        self,
        domain: str,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update({
            "OMNIMEMEVAL_MEMORY_PLUGIN": self.plugin,
            "OMNIMEMEVAL_MEMORY_DOMAIN": domain,
            "OMNIMEMEVAL_RUN_ID": self.run_id,
            "OMNIMEMEVAL_VERSION": self.version,
            "OMNIMEMEVAL_BACKUP_DIR": str(self.backup_dir),
            "OMNIMEMEVAL_PROJECT_DIR": str(self.project_dir),
            "OMNIMEMEVAL_RUN_DIR": str(self.run_dir),
        })
        env.update(self.runtime_env())
        if extra_env:
            env.update({str(key): str(value) for key, value in extra_env.items()})
        return env

    def _render(
        self,
        value: str,
        domain: str,
        *,
        backup_file: str | os.PathLike[str],
    ) -> str:
        replacements = {
            "plugin": self.plugin,
            "domain": domain,
            "run_id": self.run_id,
            "run_date": self.run_date,
            "version": self.version,
            "backup_dir": str(getattr(self, "backup_dir", "")),
            "backup_file": str(backup_file),
            # Snapshot/cleanup stages use the same explicit path argument but
            # expose a distinct token so lifecycle YAML is self-documenting.
            "global_backup_file": str(backup_file),
            "project_dir": str(self.project_dir),
            "run_dir": str(self.run_dir),
            "home": str(Path.home()),
            "openclaw_home": os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw")),
            "original_home": self._original_home_env,
            "original_openclaw_home": os.environ.get(
                "OMNIMEMEVAL_ORIGINAL_OPENCLAW_HOME",
                self._original_home_env if self.agent == "openclaw" else str(Path.home() / ".openclaw"),
            ),
            "original_hermes_home": os.environ.get(
                "OMNIMEMEVAL_ORIGINAL_HERMES_HOME",
                self._original_home_env if self.agent == "hermes" else str(Path.home() / ".hermes"),
            ),
        }
        rendered = value
        for key, replacement in replacements.items():
            rendered = rendered.replace(f"@{key}@", replacement)
        return rendered

    def _path(self, value: str | os.PathLike[str]) -> Path:
        text = str(value)
        rendered = self._render(text, "", backup_file="")
        path = Path(rendered).expanduser()
        if not path.is_absolute():
            path = self.project_dir / path
        return path

    def _original_home_key(self) -> str | None:
        if self.agent == "openclaw":
            return "OMNIMEMEVAL_ORIGINAL_OPENCLAW_HOME"
        if self.agent == "hermes":
            return "OMNIMEMEVAL_ORIGINAL_HERMES_HOME"
        return None

    def _capture_original_home_env(self) -> str:
        if self.agent == "openclaw":
            return os.environ.get("OPENCLAW_HOME", str(Path.home() / ".openclaw"))
        if self.agent == "hermes":
            return os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
        return str(Path.home())

    def _record(self, stage: str, domain: str, data: dict[str, Any]) -> None:
        payload = {
            "stage": stage,
            "domain": domain,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(data)
        self._events.append(payload)

    def _write_manifest(self) -> None:
        write_json(self.manifest_file, {
            "plugin": self.plugin,
            "run_id": self.run_id,
            "version": self.version,
            "backup_dir": str(self.backup_dir),
            "events": self._events,
        })
