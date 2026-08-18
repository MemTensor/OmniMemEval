#!/usr/bin/env python3
"""Run-scoped lifecycle control for Hermes' Mem0 OSS provider."""

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


def home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def config_path() -> Path:
    return home() / "mem0.json"


def config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Hermes mem0.json must be an object: {path}")
    return value


def run_dir() -> Path:
    value = os.environ.get("OMNIMEMEVAL_RUN_DIR")
    if not value:
        raise RuntimeError("OMNIMEMEVAL_RUN_DIR is required")
    return Path(value).expanduser().resolve()


def assert_run_owned(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    root = run_dir()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing Mem0 path outside run directory: {resolved}")
    return resolved


def qdrant_path() -> Path:
    value = os.environ.get("HERMES_MEM0_QDRANT_PATH")
    if not value:
        raise RuntimeError("HERMES_MEM0_QDRANT_PATH is required")
    return assert_run_owned(Path(value))


def mem0_dir() -> Path:
    value = os.environ.get("MEM0_DIR")
    if not value:
        raise RuntimeError("MEM0_DIR is required")
    return assert_run_owned(Path(value))


def eval_user_id() -> str:
    value = str(os.environ.get("MEM0_EVAL_USER_ID") or "").strip()
    if not value.startswith("omnimemeval:"):
        raise RuntimeError(f"refusing non-run-scoped Mem0 user: {value!r}")
    return value


def collection_name(value: dict[str, Any] | None = None) -> str:
    value = value or config()
    settings = (
        (((value.get("oss") or {}).get("vector_store") or {}).get("config") or {})
    )
    return str(settings.get("collection_name") or "mem0")


def qdrant_url(value: dict[str, Any] | None = None) -> str:
    value = value or config()
    settings = ((((value.get("oss") or {}).get("vector_store") or {}).get("config") or {}))
    return str(settings.get("url") or "").rstrip("/")


def remote_qdrant() -> bool:
    return bool(qdrant_url())


def request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    allow_404: bool = False,
) -> dict[str, Any]:
    base = qdrant_url()
    if not base:
        raise RuntimeError("remote Qdrant URL is not configured")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if allow_404 and exc.code == 404:
            return {}
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} failed: HTTP {exc.code}: {detail[:2000]}"
        ) from exc
    value = json.loads(raw) if raw else {}
    if isinstance(value, dict) and value.get("status") not in (None, "ok"):
        raise RuntimeError(f"Qdrant request failed: {value}")
    return value


def collections() -> set[str]:
    value = request("GET", "/collections")
    return {
        str(item["name"])
        for item in ((value.get("result") or {}).get("collections") or [])
        if isinstance(item, dict) and item.get("name")
    }


def managed_collections() -> list[str]:
    primary = collection_name()
    available = collections()
    return [name for name in (primary, f"{primary}_entities") if name in available]


def user_filter() -> dict[str, Any]:
    return {
        "must": [{"key": "user_id", "match": {"value": eval_user_id()}}]
    }


def scroll_all(collection: str, *, vectors: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    offset: Any | None = None
    while True:
        payload: dict[str, Any] = {
            "filter": user_filter(),
            "limit": 100,
            "with_payload": True,
            "with_vector": vectors,
        }
        if offset is not None:
            payload["offset"] = offset
        value = request(
            "POST",
            f"/collections/{urllib.parse.quote(collection, safe='')}/points/scroll",
            payload,
        )
        page = value.get("result") or {}
        points = page.get("points") or []
        result.extend(item for item in points if isinstance(item, dict))
        offset = page.get("next_page_offset")
        if offset is None or not points:
            return result


def remote_counts() -> dict[str, int]:
    return {name: len(scroll_all(name, vectors=False)) for name in managed_collections()}


def delete_user_points(collection: str) -> None:
    request(
        "POST",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/delete?wait=true",
        {"filter": user_filter()},
    )


def upsert(collection: str, points: list[dict[str, Any]]) -> None:
    for offset in range(0, len(points), 100):
        request(
            "PUT",
            f"/collections/{urllib.parse.quote(collection, safe='')}/points?wait=true",
            {"points": points[offset : offset + 100]},
        )


def hermes_python() -> str:
    value = os.environ.get("HERMES_PYTHON") or "/usr/local/lib/hermes-agent/venv/bin/python"
    if not Path(value).is_file():
        raise RuntimeError(f"Hermes Python does not exist: {value}")
    return value


def prepare() -> None:
    value = config()
    if str(value.get("mode") or "").lower() != "oss":
        raise RuntimeError("Hermes Mem0 lifecycle currently requires OSS mode")
    vector = ((value.get("oss") or {}).get("vector_store") or {})
    if str(vector.get("provider") or "").lower() != "qdrant":
        raise RuntimeError("Hermes Mem0 lifecycle currently requires Qdrant")
    settings = vector.setdefault("config", {})
    endpoint = str(os.environ.get("HERMES_MEM0_QDRANT_URL") or "").rstrip("/")
    value["user_id"] = eval_user_id()
    value["agent_id"] = f"omnimemeval-{os.environ.get('MEM0_EVAL_AGENT_ID', 'hermes')}"
    if endpoint:
        settings.pop("path", None)
        settings["url"] = endpoint
        with urllib.request.urlopen(endpoint + "/collections", timeout=10) as response:
            if response.status >= 400:
                raise RuntimeError(f"Qdrant health returned HTTP {response.status}")
    else:
        settings.pop("url", None)
        settings["path"] = str(qdrant_path())
    config_path().write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "prepared": "mem0",
                "user_id": value["user_id"],
                "agent_id": value["agent_id"],
                "qdrant": settings.get("url") or settings.get("path"),
                "mem0_dir": str(mem0_dir()),
            }
        )
    )


def _clip(value: Any, limit: int = 4000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def training_memories(train_dir: Path) -> list[str]:
    records: list[str] = []
    for result_path in sorted(train_dir.glob("*/result.json")):
        if re.fullmatch(r".+__trial_\d+", result_path.parent.name) is None:
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        agent_result = result.get("agent_result") or {}
        feedback_result = result.get("feedback_result") or {}
        verifier_result = result.get("verifier_result") or {}
        if agent_result.get("completion_status") != "completed":
            continue
        domain = result.get("domain") or result_path.parents[2].name
        case = result.get("task_name")
        records.append(
            f"OmniMemEval learned verified {domain} benchmark case {case}. "
            f"The verified reward is {verifier_result.get('reward')}. "
            "Treat the expected answer and verifier feedback as authoritative.\n"
            f"Expected answer: {_clip(verifier_result.get('expected'))}\n"
            "Attempted answer: "
            f"{_clip(verifier_result.get('actual') or agent_result.get('response'))}\n"
            f"Verifier feedback: {_clip(verifier_result.get('feedback'))}\n"
            f"Agent reflection: {_clip(feedback_result.get('response'))}\n"
            "Apply this verified experience when solving related future tasks."
        )
    return records


def ingest_training(train_dir: Path) -> None:
    records = training_memories(train_dir)
    if not records:
        raise RuntimeError(f"no completed Mem0 training results found under {train_dir}")
    script = r"""
import json, sys
from mem0 import Memory

config_path = sys.argv[1]
root = json.load(open(config_path, encoding="utf-8"))
memory = Memory.from_config(config_dict=root["oss"])
total = 0
for text in json.load(sys.stdin):
    result = memory.add(
        text,
        user_id=root["user_id"],
        agent_id=root["agent_id"],
        metadata={"source": "OmniMemEval", "kind": "verified_training_feedback"},
        infer=False,
    )
    if isinstance(result, dict):
        total += len(result.get("results") or [])
    elif isinstance(result, list):
        total += len(result)
print(json.dumps({"ok": True, "training_results": int(sys.argv[2]), "memories": total}))
"""
    child_env = {**os.environ, "MEM0_DIR": str(mem0_dir())}
    if not child_env.get("OPENAI_API_KEY") and child_env.get("LLM_API_KEY"):
        child_env["OPENAI_API_KEY"] = child_env["LLM_API_KEY"]
    completed = subprocess.run(
        [
            hermes_python(),
            "-c",
            script,
            str(config_path()),
            str(len(records)),
        ],
        check=False,
        text=True,
        input=json.dumps(records, ensure_ascii=False),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(os.environ.get("MEM0_INGEST_TIMEOUT", "900")),
        env=child_env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Hermes Mem0 explicit training ingest failed "
            f"(rc={completed.returncode}): {completed.stderr[-2000:]}"
        )
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    if value.get("ok") is not True or int(value.get("memories") or 0) == 0:
        raise RuntimeError(f"Hermes Mem0 explicit training ingest produced no memories: {value}")
    print(json.dumps({"user_id": eval_user_id(), **value}))


def count() -> int:
    if remote_qdrant():
        return remote_counts().get(collection_name(), 0)
    path = qdrant_path()
    if not path.is_dir():
        return 0
    script = """
import json, sys
from qdrant_client import QdrantClient
path, collection, user_id = sys.argv[1:]
client = QdrantClient(path=path)
try:
    if not client.collection_exists(collection):
        print(0)
    else:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        result = client.count(
            collection,
            count_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            exact=True,
        )
        print(int(result.count))
finally:
    client.close()
"""
    completed = subprocess.run(
        [
            hermes_python(),
            "-c",
            script,
            str(path),
            collection_name(),
            eval_user_id(),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "MEM0_DIR": str(mem0_dir())},
    )
    return int(completed.stdout.strip().splitlines()[-1])


def clear() -> None:
    before = count()
    if remote_qdrant():
        for name in managed_collections():
            delete_user_points(name)
        after = count()
        if after:
            raise RuntimeError(f"Mem0 clear did not remove run-scoped points: {after}")
    else:
        shutil.rmtree(qdrant_path(), ignore_errors=True)
    shutil.rmtree(mem0_dir(), ignore_errors=True)
    if not remote_qdrant():
        qdrant_path().parent.mkdir(parents=True, exist_ok=True)
    mem0_dir().mkdir(parents=True, mode=0o700, exist_ok=True)
    print(json.dumps({"before": before, "after": 0}))


def wait_stable(timeout: int, stable_seconds: int) -> None:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    previous: int | None = None
    while time.monotonic() < deadline:
        current = count()
        if current == previous:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= stable_seconds:
                print(json.dumps({"count": current, "stable": True}))
                return
        else:
            previous = current
            stable_since = None
        time.sleep(2)
    raise RuntimeError(f"Hermes Mem0 did not stabilize within {timeout}s: count={previous}")


def backup(path: Path) -> None:
    wait_stable(int(os.environ.get("MEM0_SETTLE_TIMEOUT", "300")), 6)
    memories = count()
    if memories == 0:
        raise RuntimeError("refusing empty Hermes Mem0 training backup")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        is_remote = remote_qdrant()
        manifest = {
            "format": "omnimemeval-hermes-mem0-qdrant-v1" if is_remote else "omnimemeval-hermes-mem0-local-v1",
            "user_id": eval_user_id(),
            "collection": collection_name(),
            "memory_count": memories,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        if is_remote:
            (root / "qdrant.json").write_text(
                json.dumps(
                    {name: scroll_all(name, vectors=True) for name in managed_collections()},
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
        elif qdrant_path().is_dir():
            shutil.copytree(qdrant_path(), root / "qdrant", symlinks=True)
        if mem0_dir().is_dir():
            shutil.copytree(mem0_dir(), root / "mem0", symlinks=True)
        with tarfile.open(path, "w:gz") as archive:
            for item in root.iterdir():
                archive.add(item, arcname=item.name)
    path.chmod(0o600)
    print(json.dumps({"backup": str(path), "memory_count": memories}))


def restore(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    clear()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(root, filter="data")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        supported = {
            "omnimemeval-hermes-mem0-local-v1",
            "omnimemeval-hermes-mem0-qdrant-v1",
        }
        if manifest.get("format") not in supported:
            raise RuntimeError(f"unsupported Mem0 backup: {manifest.get('format')!r}")
        if manifest.get("user_id") != eval_user_id():
            raise RuntimeError("Mem0 backup user does not match the current run")
        if manifest.get("format") == "omnimemeval-hermes-mem0-qdrant-v1":
            if not remote_qdrant():
                raise RuntimeError("remote Mem0 backup requires remote Qdrant")
            payload = json.loads((root / "qdrant.json").read_text(encoding="utf-8"))
            available = collections()
            for name, points in payload.items():
                if name not in available:
                    raise RuntimeError(f"Mem0 restore collection is missing: {name}")
                upsert(name, list(points or []))
        elif (root / "qdrant").is_dir():
            shutil.copytree(root / "qdrant", qdrant_path())
        if (root / "mem0").is_dir():
            shutil.rmtree(mem0_dir(), ignore_errors=True)
            shutil.copytree(root / "mem0", mem0_dir())
    restored = count()
    if restored != int(manifest.get("memory_count") or 0) or restored == 0:
        raise RuntimeError(
            f"Hermes Mem0 restore count mismatch: expected={manifest.get('memory_count')} "
            f"actual={restored}"
        )
    print(json.dumps({"restore": str(path), "memory_count": restored}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    sub.add_parser("clear")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("train_dir", type=Path)
    wait = sub.add_parser("wait")
    wait.add_argument("--timeout", type=int, default=300)
    wait.add_argument("--stable-seconds", type=int, default=6)
    save = sub.add_parser("backup")
    save.add_argument("path", type=Path)
    load = sub.add_parser("restore")
    load.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "status":
        print(json.dumps({"user_id": eval_user_id(), "memory_count": count()}))
    elif args.command == "clear":
        clear()
    elif args.command == "ingest":
        ingest_training(args.train_dir)
    elif args.command == "wait":
        wait_stable(args.timeout, args.stable_seconds)
    elif args.command == "backup":
        backup(args.path)
    elif args.command == "restore":
        restore(args.path)


if __name__ == "__main__":
    main()
