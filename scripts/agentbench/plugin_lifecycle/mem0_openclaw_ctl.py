#!/usr/bin/env python3
"""Run-scoped lifecycle control for the OpenClaw Mem0 OSS plugin."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PLUGIN_NAME = os.environ.get("MEM0_PLUGIN_NAME", "openclaw-mem0")


def config_path() -> Path:
    value = os.environ.get("OPENCLAW_CONFIG")
    if not value:
        raise RuntimeError("OPENCLAW_CONFIG is required")
    return Path(value).expanduser()


def config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"OpenClaw config must be an object: {path}")
    return value


def plugin_config(root: dict[str, Any] | None = None) -> dict[str, Any]:
    root = root or config()
    value = (
        (((root.get("plugins") or {}).get("entries") or {}).get(PLUGIN_NAME) or {})
        .get("config")
        or {}
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{PLUGIN_NAME} config must be an object")
    return value


def eval_user_id() -> str:
    value = str(os.environ.get("MEM0_EVAL_USER_ID") or "").strip()
    if not value.startswith("omnimemeval:"):
        raise RuntimeError(f"refusing non-run-scoped Mem0 user: {value!r}")
    return value


def mode_state_path() -> Path:
    value = os.environ.get("MEM0_MODE_STATE")
    if not value:
        raise RuntimeError("MEM0_MODE_STATE is required")
    return Path(value).expanduser()


def qdrant_settings(root: dict[str, Any] | None = None) -> tuple[str, str]:
    cfg = plugin_config(root)
    if str(cfg.get("mode") or "").lower() not in {"open-source", "oss"}:
        raise RuntimeError("OpenClaw Mem0 lifecycle currently requires OSS mode")
    vector = ((cfg.get("oss") or {}).get("vectorStore") or {})
    if str(vector.get("provider") or "").lower() != "qdrant":
        raise RuntimeError("OpenClaw Mem0 lifecycle currently requires Qdrant")
    settings = vector.get("config") or {}
    base = str(settings.get("url") or "").rstrip("/")
    collection = str(settings.get("collectionName") or "").strip()
    if not base or not collection:
        raise RuntimeError("Mem0 Qdrant url and collectionName are required")
    return base, collection


def request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    allow_404: bool = False,
) -> dict[str, Any]:
    base, _ = qdrant_settings()
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
    _, primary = qdrant_settings()
    available = collections()
    return [name for name in (primary, f"{primary}_entities") if name in available]


def user_filter() -> dict[str, Any]:
    return {
        "must": [
            {
                "key": "user_id",
                "match": {"value": eval_user_id()},
            }
        ]
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


def counts() -> dict[str, int]:
    return {name: len(scroll_all(name, vectors=False)) for name in managed_collections()}


def delete_user_points(collection: str) -> None:
    request(
        "POST",
        f"/collections/{urllib.parse.quote(collection, safe='')}/points/delete?wait=true",
        {"filter": user_filter()},
    )


def clear() -> None:
    before = counts()
    for name in before:
        delete_user_points(name)
    after = counts()
    if any(after.values()):
        raise RuntimeError(f"Mem0 clear did not remove all run-scoped points: {after}")
    print(json.dumps({"user_id": eval_user_id(), "before": before, "after": after}))


def prepare() -> None:
    root = config()
    plugins = root.get("plugins") or {}
    if plugins.get("enabled") is False:
        raise RuntimeError("OpenClaw plugins are disabled")
    if isinstance(plugins.get("allow"), list) and PLUGIN_NAME not in plugins["allow"]:
        raise RuntimeError(f"OpenClaw plugins.allow does not include {PLUGIN_NAME}")
    if (plugins.get("slots") or {}).get("memory") != PLUGIN_NAME:
        raise RuntimeError(f"OpenClaw memory slot is not {PLUGIN_NAME}")
    entry = (plugins.get("entries") or {}).get(PLUGIN_NAME)
    if not isinstance(entry, dict) or entry.get("enabled") is False:
        raise RuntimeError(f"{PLUGIN_NAME} is missing or disabled")
    cfg = plugin_config(root)
    qdrant_settings(root)
    triage = bool((((cfg.get("skills") or {}).get("triage") or {}).get("enabled")))
    auto_capture = bool(cfg.get("autoCapture"))
    state = {
        "triage": triage,
        "auto_capture": auto_capture,
    }
    state_path = mode_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    state_path.chmod(0o600)
    cfg["userId"] = eval_user_id()
    cfg["autoRecall"] = True
    config_path().write_text(
        json.dumps(root, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"prepared": PLUGIN_NAME, "user_id": eval_user_id(), **state}))


def set_mode(mode: str) -> None:
    if mode not in {"train", "test"}:
        raise ValueError("mode must be train or test")
    state = json.loads(mode_state_path().read_text(encoding="utf-8"))
    root = config()
    cfg = plugin_config(root)
    triage_cfg = (cfg.setdefault("skills", {}).setdefault("triage", {}))
    cfg["userId"] = eval_user_id()
    if mode == "train":
        # The plugin's automatic write is fire-and-forget and a one-shot CLI
        # can exit before it reaches Qdrant. The lifecycle performs an explicit
        # awaited ingest of verified feedback after the train phase instead.
        cfg["autoRecall"] = False
        cfg["autoCapture"] = False
        triage_cfg["enabled"] = False
    else:
        # Keep native auto-recall active, but disable every automatic write
        # path while evaluating the restored training snapshot.
        cfg["autoRecall"] = True
        cfg["autoCapture"] = False
        triage_cfg["enabled"] = False
    config_path().write_text(
        json.dumps(root, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": mode,
                "user_id": eval_user_id(),
                "autoCapture": cfg["autoCapture"],
                "triage": triage_cfg["enabled"],
            }
        )
    )


def _clip(value: Any, limit: int = 4000) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def training_memories(train_dir: Path) -> list[str]:
    records: list[str] = []
    for result_path in sorted(train_dir.rglob("result.json")):
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
    command = os.environ.get("MEM0_NODE_COMMAND", "node")
    timeout = int(os.environ.get("MEM0_INGEST_TIMEOUT", "900"))
    env = dict(os.environ)
    module_path = Path(
        os.environ.get("MEM0_NODE_PATH")
        or config_path().parent / "npm" / "node_modules"
    )
    if not module_path.is_dir():
        raise RuntimeError(f"OpenClaw Mem0 Node module path does not exist: {module_path}")
    env["NODE_PATH"] = str(module_path)
    script = r"""
const fs = require("fs");
const { Memory } = require("mem0ai/oss");

(async () => {
  const root = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
  const plugin = process.argv[2];
  const userId = process.argv[3];
  const records = JSON.parse(fs.readFileSync(0, "utf8"));
  const cfg = root.plugins.entries[plugin].config;
  const memory = new Memory(cfg.oss);
  let total = 0;
  for (const text of records) {
    const result = await memory.add(text, {
      userId,
      infer: false,
      metadata: { source: "OmniMemEval", kind: "verified_training_feedback" },
    });
    total += Array.isArray(result) ? result.length : (result.results || []).length;
  }
  process.stdout.write(JSON.stringify({
    ok: true,
    training_results: records.length,
    memories: total,
  }) + "\n");
})().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
"""
    completed = subprocess.run(
        [command, "-e", script, str(config_path()), PLUGIN_NAME, eval_user_id()],
        capture_output=True,
        text=True,
        input=json.dumps(records, ensure_ascii=False),
        timeout=timeout,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenClaw Mem0 explicit training ingest failed "
            f"(rc={completed.returncode}): {(completed.stderr or completed.stdout)[-2000:]}"
        )
    value = json.loads(completed.stdout.strip().splitlines()[-1])
    if value.get("ok") is not True or int(value.get("memories") or 0) == 0:
        raise RuntimeError(f"OpenClaw Mem0 explicit training ingest produced no memories: {value}")
    print(json.dumps({"user_id": eval_user_id(), **value}))


def wait_stable(timeout: int, stable_seconds: int) -> None:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    previous: dict[str, int] | None = None
    while time.monotonic() < deadline:
        current = counts()
        if current == previous:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= stable_seconds:
                print(json.dumps({"user_id": eval_user_id(), "counts": current, "stable": True}))
                return
        else:
            previous = current
            stable_since = None
        time.sleep(2)
    raise RuntimeError(
        f"Mem0 points did not stabilize within {timeout}s: {previous or {}}"
    )


def backup(path: Path) -> None:
    wait_stable(int(os.environ.get("MEM0_SETTLE_TIMEOUT", "300")), 6)
    _, primary = qdrant_settings()
    payload = {
        "format": "omnimemeval-openclaw-mem0-qdrant-v1",
        "user_id": eval_user_id(),
        "collections": {
            name: scroll_all(name, vectors=True) for name in managed_collections()
        },
    }
    primary_count = len(payload["collections"].get(primary) or [])
    if primary_count == 0:
        raise RuntimeError("refusing empty OpenClaw Mem0 training backup")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "backup.json"
        source.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        with tarfile.open(path, "w:gz") as archive:
            archive.add(source, arcname="backup.json")
    path.chmod(0o600)
    print(json.dumps({"backup": str(path), "primary_count": primary_count}))


def upsert(collection: str, points: list[dict[str, Any]]) -> None:
    for offset in range(0, len(points), 100):
        batch = points[offset : offset + 100]
        request(
            "PUT",
            f"/collections/{urllib.parse.quote(collection, safe='')}/points?wait=true",
            {"points": batch},
        )


def restore(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(tmp, filter="data")
        payload = json.loads((Path(tmp) / "backup.json").read_text(encoding="utf-8"))
    if payload.get("format") != "omnimemeval-openclaw-mem0-qdrant-v1":
        raise RuntimeError(f"unsupported Mem0 backup: {payload.get('format')!r}")
    if payload.get("user_id") != eval_user_id():
        raise RuntimeError("Mem0 backup user does not match the current run")
    available = collections()
    clear()
    for name, points in (payload.get("collections") or {}).items():
        if name not in available:
            raise RuntimeError(f"Mem0 restore collection is missing: {name}")
        upsert(name, list(points or []))
    restored = counts()
    _, primary = qdrant_settings()
    if restored.get(primary, 0) == 0:
        raise RuntimeError("Mem0 restore produced no primary memories")
    print(json.dumps({"restore": str(path), "counts": restored}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    mode = sub.add_parser("mode")
    mode.add_argument("value", choices=["train", "test"])
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
    elif args.command == "mode":
        set_mode(args.value)
    elif args.command == "status":
        print(json.dumps({"user_id": eval_user_id(), "counts": counts()}))
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
