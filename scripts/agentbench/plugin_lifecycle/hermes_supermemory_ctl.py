#!/usr/bin/env python3
"""Run-scoped lifecycle operations for Hermes' Supermemory provider."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PROCESSING = {"unknown", "queued", "extracting", "chunking", "embedding", "indexing", "processing"}
FAILED = {"failed", "error", "cancelled", "canceled"}


def home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def config() -> dict[str, Any]:
    path = home() / "supermemory.json"
    if not path.is_file():
        raise RuntimeError(f"Supermemory config is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Supermemory config must be an object: {path}")
    return value


def dotenv() -> dict[str, str]:
    result: dict[str, str] = {}
    path = home() / ".env"
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def base_url() -> str:
    return str(
        os.environ.get("SUPERMEMORY_BASE_URL")
        or config().get("base_url")
        or "http://127.0.0.1:6767"
    ).rstrip("/")


def api_key() -> str:
    value = os.environ.get("SUPERMEMORY_API_KEY") or dotenv().get("SUPERMEMORY_API_KEY")
    local = Path("/var/lib/supermemory/api-key")
    if not value and local.is_file():
        value = local.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("SUPERMEMORY_API_KEY is not configured")
    return value


def container() -> str:
    value = os.environ.get("SUPERMEMORY_CONTAINER_TAG") or config().get("container_tag")
    if not value:
        raise RuntimeError("Supermemory container_tag is not configured")
    value = str(value)
    if not value.startswith("omnimemeval_"):
        raise RuntimeError(f"refusing to manage non-run-scoped container: {value}")
    return value


def request(method: str, path: str, payload: Any | None = None, *, allow_404: bool = False) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url() + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if allow_404 and exc.code == 404:
            return {}
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail[:2000]}") from exc
    return json.loads(raw) if raw else {}


def paged(path: str, payload: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    page = 1
    while True:
        current = dict(payload, page=page, limit=100)
        data = request("POST", path, current)
        batch: list[dict[str, Any]] = []
        for key in keys:
            if isinstance(data.get(key), list):
                batch = data[key]
                break
        result.extend(batch)
        total = int((data.get("pagination") or {}).get("totalPages") or 0)
        if (total and page >= total) or (not total and len(batch) < 100):
            return result
        page += 1


def documents(*, content: bool) -> list[dict[str, Any]]:
    return paged(
        "/v3/documents/list",
        {"containerTags": [container()], "includeContent": content},
        ("memories", "documents"),
    )


def memories() -> list[dict[str, Any]]:
    return paged(
        "/v4/memories/list",
        {"containerTags": [container()]},
        ("memoryEntries", "memories"),
    )


def clear() -> None:
    request(
        "DELETE",
        "/v3/container-tags/" + urllib.parse.quote(container(), safe=""),
        allow_404=True,
    )
    print(f"cleared Supermemory container {container()}")


def wait_idle(timeout: int) -> None:
    deadline = time.monotonic() + timeout
    statuses: Counter[str] = Counter()
    while time.monotonic() < deadline:
        docs = documents(content=False)
        statuses = Counter(str(item.get("status") or "unknown").lower() for item in docs)
        failed = [item for item in docs if str(item.get("status") or "unknown").lower() in FAILED]
        if failed:
            raise RuntimeError(
                "Supermemory document ingestion failed: "
                + ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
            )
        active = [
            item
            for item in docs
            if str(item.get("status") or "unknown").lower() in PROCESSING
        ]
        if not active:
            print(f"Supermemory idle: documents={len(docs)} memories={len(memories())}")
            return
        time.sleep(5)
    detail = ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
    raise RuntimeError(
        f"Supermemory did not become idle within {timeout}s"
        + (f": {detail}" if detail else "")
    )


def backup(path: Path, *, wait: bool) -> None:
    if wait:
        wait_idle(int(os.environ.get("MEMORY_FINALIZE_TIMEOUT", "1800")))
    payload = {
        "format": "omnimemeval-supermemory-v1",
        "container": container(),
        "documents": documents(content=True),
        "memories": memories(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "backup.json"
        source.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source, arcname="backup.json")
    path.chmod(0o600)
    print(f"backed up Supermemory container {container()} to {path}")


def restore(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(tmp, filter="data")
        payload = json.loads((Path(tmp) / "backup.json").read_text(encoding="utf-8"))
    if payload.get("format") != "omnimemeval-supermemory-v1":
        raise RuntimeError(f"unsupported Supermemory backup: {payload.get('format')!r}")
    clear()
    memory_items = []
    for item in payload.get("memories") or []:
        content = str(item.get("memory") or item.get("content") or "").strip()
        if content:
            memory_items.append({"content": content[:10000], "isStatic": bool(item.get("isStatic", False))})
    for offset in range(0, len(memory_items), 100):
        request(
            "POST",
            "/v4/memories",
            {"containerTag": container(), "memories": memory_items[offset:offset + 100]},
        )
    if not memory_items:
        for item in payload.get("documents") or []:
            content = str(item.get("content") or "").strip()
            if content:
                request(
                    "POST",
                    "/v3/documents",
                    {"content": content, "containerTag": container(), "taskType": "superrag"},
                )
    wait_idle(int(os.environ.get("MEMORY_FINALIZE_TIMEOUT", "1800")))
    print(f"restored Supermemory container {container()} from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("clear")
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout", type=int, default=1800)
    save = sub.add_parser("backup")
    save.add_argument("path", type=Path)
    load = sub.add_parser("restore")
    load.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps({"container": container(), "documents": len(documents(content=False)), "memories": len(memories())}))
    elif args.command == "clear":
        clear()
    elif args.command == "wait":
        wait_idle(args.timeout)
    elif args.command == "backup":
        backup(args.path, wait=True)
    elif args.command == "restore":
        restore(args.path)


if __name__ == "__main__":
    main()
