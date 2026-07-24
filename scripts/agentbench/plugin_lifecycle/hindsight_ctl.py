#!/usr/bin/env python3
"""Bank-scoped lifecycle operations shared by OpenClaw and Hermes Hindsight."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def config_path() -> Path:
    explicit = os.environ.get("HINDSIGHT_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    hermes = os.environ.get("HERMES_HOME")
    if hermes:
        return Path(hermes).expanduser() / "hindsight" / "config.json"
    openclaw = os.environ.get("OPENCLAW_HOME")
    if openclaw:
        return Path(openclaw).expanduser() / "openclaw.json"
    raise RuntimeError("HINDSIGHT_CONFIG, HERMES_HOME, or OPENCLAW_HOME is required")


def config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if path.name == "openclaw.json":
        plugin = os.environ.get("HINDSIGHT_PLUGIN_NAME", "hindsight-openclaw")
        value = (((value.get("plugins") or {}).get("entries") or {}).get(plugin) or {}).get("config") or {}
    return value if isinstance(value, dict) else {}


def api_url() -> str:
    explicit = os.environ.get("HINDSIGHT_API_URL")
    if explicit:
        return explicit.rstrip("/")
    value = config()
    return str(value.get("hindsightApiUrl") or value.get("api_url") or "http://127.0.0.1:9177").rstrip("/")


def api_key() -> str:
    explicit = os.environ.get("HINDSIGHT_API_KEY")
    if explicit:
        return explicit
    try:
        value = config()
    except FileNotFoundError:
        return ""
    return str(value.get("api_key") or "")


def bank_id() -> str:
    bank = os.environ.get("HINDSIGHT_EVAL_BANK_ID")
    if not bank:
        value = config()
        bank = value.get("dynamicBankId") or value.get("bank_id")
    if not bank:
        raise RuntimeError("Hindsight evaluation bank is not configured")
    bank = str(bank)
    if not bank.startswith("omnimemeval-"):
        raise RuntimeError(f"refusing to manage non-run-scoped Hindsight bank: {bank}")
    return bank


def request(method: str, path: str, payload: dict | None = None, *, allow_404: bool = False) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key():
        headers["Authorization"] = f"Bearer {api_key()}"
    req = urllib.request.Request(api_url() + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if allow_404 and exc.code == 404:
            return None
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail[:2000]}") from exc
    return json.loads(raw) if raw else None


def bank_path(suffix: str = "") -> str:
    return "/v1/default/banks/" + urllib.parse.quote(bank_id(), safe="") + suffix


def banks() -> set[str]:
    payload = request("GET", "/v1/default/banks") or {}
    return {
        str(item["bank_id"])
        for item in payload.get("banks") or []
        if isinstance(item, dict) and item.get("bank_id")
    }


def clear() -> None:
    request("DELETE", bank_path(), allow_404=True)
    request("PUT", bank_path(), {})
    print(f"cleared Hindsight bank {bank_id()}")


def delete() -> None:
    request("DELETE", bank_path(), allow_404=True)
    print(f"deleted Hindsight bank {bank_id()}")


def wait_idle(timeout: int, idle_seconds: int) -> None:
    deadline = time.monotonic() + timeout
    idle_since: float | None = None
    while time.monotonic() < deadline:
        payload = request("GET", bank_path("/operations?limit=100")) or {}
        operations = payload.get("operations") or []
        failed = [
            item for item in operations
            if str(item.get("status") or "").lower()
            in {"failed", "error", "cancelled", "canceled"}
        ]
        if failed:
            statuses = sorted(
                str(item.get("status") or "unknown").lower() for item in failed
            )
            raise RuntimeError(
                f"Hindsight bank operation failed for {bank_id()}: {', '.join(statuses)}"
            )
        active = [
            item for item in operations
            if str(item.get("status") or "").lower()
            in {"pending", "processing", "running", "queued", "in_progress"}
        ]
        if not active:
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= idle_seconds:
                print(f"Hindsight bank idle: {bank_id()}")
                return
        else:
            idle_since = None
        time.sleep(5)
    raise RuntimeError(f"Hindsight bank did not become idle within {timeout}s: {bank_id()}")


def _listening_pid() -> str | None:
    port = urllib.parse.urlparse(api_url()).port
    if not port:
        return None
    result = subprocess.run(
        ["ss", "-ltnp"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    for line in result.stdout.splitlines():
        if f":{port} " in line and (match := re.search(r"pid=(\d+)", line)):
            return match.group(1)
    return None


def _process_env(pid: str | None) -> dict[str, str]:
    if not pid:
        return {}
    try:
        raw = (Path("/proc") / pid / "environ").read_bytes()
    except OSError:
        return {}
    result = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    return result


def _direct_database_url(value: str, daemon_home: str | None) -> str:
    if value == "pg0":
        instance = "hindsight"
    elif value.startswith("pg0://"):
        instance = value[6:].split(":", 1)[0] or "hindsight"
    else:
        return value
    homes: list[Path] = []
    if daemon_home:
        homes.append(Path(daemon_home))
    homes.extend([
        Path("/home/hermes"),
        Path("/var/lib/hindsight-openclaw"),
        Path.home(),
    ])
    for home in homes:
        path = home / ".pg0" / "instances" / instance / "instance.json"
        if not path.is_file():
            continue
        info = json.loads(path.read_text(encoding="utf-8"))
        user = urllib.parse.quote(str(info.get("username") or "hindsight"), safe="")
        password = urllib.parse.quote(str(info.get("password") or ""), safe="")
        database = urllib.parse.quote(str(info.get("database") or "hindsight"), safe="")
        return f"postgresql://{user}:{password}@127.0.0.1:{int(info['port'])}/{database}"
    return value


def admin_env() -> dict[str, str]:
    result = dict(os.environ)
    daemon = _process_env(_listening_pid())
    for key, value in daemon.items():
        if key.startswith("HINDSIGHT_"):
            result.setdefault(key, value)
    database = result.get("HINDSIGHT_API_DATABASE_URL")
    if not database:
        value = config()
        database = value.get("database_url") or value.get("api_database_url")
    if not database and os.environ.get("HINDSIGHT_INSTANCE_JSON"):
        path = Path(os.environ["HINDSIGHT_INSTANCE_JSON"])
        if path.is_file():
            info = json.loads(path.read_text(encoding="utf-8"))
            user = urllib.parse.quote(str(info.get("username") or "hindsight"), safe="")
            password = urllib.parse.quote(str(info.get("password") or ""), safe="")
            name = urllib.parse.quote(str(info.get("database") or "hindsight"), safe="")
            database = (
                f"postgresql://{user}:{password}@127.0.0.1:"
                f"{int(info['port'])}/{name}"
            )
    if not database:
        raise RuntimeError("HINDSIGHT_API_DATABASE_URL is unavailable")
    result["HINDSIGHT_API_DATABASE_URL"] = _direct_database_url(str(database), daemon.get("HOME"))
    return result


def admin_binary() -> str:
    daemon_exe = None
    daemon_argv_sibling = None
    pid = _listening_pid()
    if pid:
        try:
            daemon_exe = str((Path("/proc") / pid / "exe").resolve().parent / "hindsight-admin")
        except OSError:
            daemon_exe = None
        try:
            argv0 = (Path("/proc") / pid / "cmdline").read_bytes().split(b"\0", 1)[0]
            daemon_argv_sibling = str(Path(os.fsdecode(argv0)).parent / "hindsight-admin")
        except OSError:
            daemon_argv_sibling = None
    daemon_path = _process_env(pid).get("PATH") if pid else None
    candidates = [
        os.environ.get("HINDSIGHT_ADMIN_BIN"),
        shutil.which("hindsight-admin"),
        shutil.which("hindsight-admin", path=daemon_path) if daemon_path else None,
        daemon_argv_sibling,
        daemon_exe,
        "/usr/local/lib/hermes-agent/venv/bin/hindsight-admin",
        "/opt/hindsight/venv/bin/hindsight-admin",
    ]
    for value in candidates:
        if value and Path(value).is_file():
            return str(value)
    raise RuntimeError("hindsight-admin was not found; set HINDSIGHT_ADMIN_BIN")


def backup(path: Path) -> None:
    wait_idle(int(os.environ.get("MEMORY_FINALIZE_TIMEOUT", "1800")), 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exists = bank_id() in banks()
        if exists:
            subprocess.run(
                [admin_binary(), "export-bank", "--bank", bank_id(), "--output", str(root / "bank.zip")],
                check=True,
                env=admin_env(),
            )
        (root / "manifest.json").write_text(
            json.dumps({"format": "omnimemeval-hindsight-bank-v1", "has_bank": exists}) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(path, "w:gz") as archive:
            for item in root.iterdir():
                archive.add(item, arcname=item.name)
    path.chmod(0o600)
    print(f"backed up Hindsight bank {bank_id()} to {path}")


def restore(path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(root, filter="data")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "omnimemeval-hindsight-bank-v1":
            raise RuntimeError(f"unsupported Hindsight backup: {manifest.get('format')!r}")
        request("DELETE", bank_path(), allow_404=True)
        if manifest.get("has_bank") and (root / "bank.zip").is_file():
            subprocess.run(
                [admin_binary(), "import-bank", "--archive", str(root / "bank.zip"), "--target-bank", bank_id()],
                check=True,
                env=admin_env(),
            )
        else:
            request("PUT", bank_path(), {})
    print(f"restored Hindsight bank {bank_id()} from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("clear")
    sub.add_parser("delete")
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout", type=int, default=1800)
    wait.add_argument("--idle-seconds", type=int, default=5)
    save = sub.add_parser("backup")
    save.add_argument("path", type=Path)
    load = sub.add_parser("restore")
    load.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "status":
        request("GET", "/health")
        print(json.dumps({"api_url": api_url(), "bank": bank_id(), "exists": bank_id() in banks()}))
    elif args.command == "clear":
        clear()
    elif args.command == "delete":
        delete()
    elif args.command == "wait":
        wait_idle(args.timeout, args.idle_seconds)
    elif args.command == "backup":
        backup(args.path)
    elif args.command == "restore":
        restore(args.path)


if __name__ == "__main__":
    main()
