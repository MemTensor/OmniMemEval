#!/usr/bin/env python3
"""Prepare a run-owned agent home from a user-maintained source home.

The lifecycle owns the durable run home. Agent adapters may then create
per-trial homes from it without linking mutable state back into the user's
global OpenClaw or Hermes directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


AGENT_FILES = {
    "openclaw": ("openclaw.json",),
    "hermes": ("config.yaml", ".env", "auth.json"),
}


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"home path must be a safe relative path: {value!r}")
    return path


def _assert_run_owned(runtime: Path, run_dir: Path) -> None:
    runtime = runtime.resolve(strict=False)
    run_dir = run_dir.resolve(strict=False)
    if runtime == run_dir or run_dir not in runtime.parents:
        raise ValueError(f"runtime home must be below run dir {run_dir}: {runtime}")


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target)


def prepare(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    runtime = Path(args.runtime).expanduser().resolve(strict=False)
    run_dir = Path(args.run_dir).expanduser().resolve()
    _assert_run_owned(runtime, run_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source home does not exist: {source}")

    shutil.rmtree(runtime, ignore_errors=True)
    runtime.mkdir(parents=True, mode=0o700)

    required = AGENT_FILES[args.agent][0]
    if not (source / required).is_file():
        raise FileNotFoundError(f"required source config does not exist: {source / required}")
    for name in AGENT_FILES[args.agent]:
        path = source / name
        if path.is_file():
            _copy(path, runtime / name)

    for raw in args.copy:
        relative = _relative(raw)
        path = source / relative
        if not path.exists():
            raise FileNotFoundError(f"required source path does not exist: {path}")
        _copy(path, runtime / relative)

    for raw in args.link:
        relative = _relative(raw)
        path = source / relative
        if not path.exists():
            raise FileNotFoundError(f"required source link does not exist: {path}")
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path, target_is_directory=path.is_dir())

    print(f"prepared {args.agent} runtime home: {runtime} (source={source})")


def cleanup(args: argparse.Namespace) -> None:
    runtime = Path(args.runtime).expanduser().resolve(strict=False)
    run_dir = Path(args.run_dir).expanduser().resolve()
    _assert_run_owned(runtime, run_dir)
    shutil.rmtree(runtime, ignore_errors=True)
    print(f"removed runtime home: {runtime}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--agent", choices=sorted(AGENT_FILES), required=True)
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--runtime", required=True)
    prepare_parser.add_argument("--run-dir", required=True)
    prepare_parser.add_argument("--copy", action="append", default=[])
    prepare_parser.add_argument("--link", action="append", default=[])
    prepare_parser.set_defaults(func=prepare)

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--runtime", required=True)
    cleanup_parser.add_argument("--run-dir", required=True)
    cleanup_parser.set_defaults(func=cleanup)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
