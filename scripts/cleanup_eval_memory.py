#!/usr/bin/env python3
"""Delete MemEval ingestion data for a LoCoMo or LongMemEval run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import suppress
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = SCRIPT_DIR
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

sys.path.insert(0, SCRIPTS_DIR)

from dotenv import load_dotenv

from client_factory import SUPPORTED_LIBS, create_client

LME_JSON = os.path.join(
    PROJECT_DIR,
    "data",
    "longmemeval",
    "longmemeval_s_cleaned.json",
)
LOCOMO_JSON = os.path.join(PROJECT_DIR, "data", "locomo", "locomo10.json")


def _delete_user_generic(
    client: Any,
    lib_name: str,
    user_id: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"  would delete user: {user_id}")
        return
    try:
        if getattr(client, "delete_all", None) and "mem0" in lib_name:
            client.delete_all(user_id)
            print(f"  deleted user: {user_id}")
            return
        if getattr(client, "delete", None):
            client.delete(user_id)
            print(f"  deleted user: {user_id}")
            return
        if getattr(client, "delete_user", None):
            client.delete_user(user_id)
            print(f"  deleted user: {user_id}")
            return
        print(f"  no delete method available for {user_id!r}")
    except Exception as exc:
        print(f"  delete failed for {user_id!r}: {exc}")


def _zep_use_group() -> bool:
    return os.environ.get("ZEP_USE_GROUP", "true").lower() in ("true", "1", "yes")


def _everos_use_group() -> bool:
    return os.environ.get("EVEROS_USE_GROUP", "true").lower() in ("true", "1", "yes")


def cleanup_locomo(client: Any, lib_name: str, version: str, dry_run: bool) -> None:
    with open(LOCOMO_JSON) as f:
        locomo_df = json.load(f)
    n = len(locomo_df)

    if lib_name == "zep" and _zep_use_group():
        for conv_idx in range(n):
            graph_id = f"locomo_exp_group_{conv_idx}_{version}"
            if dry_run:
                print(f"  would delete zep graph: {graph_id}")
            elif hasattr(client, "sdk_graph_delete"):
                with suppress(Exception):
                    client.sdk_graph_delete(graph_id)
                print(f"  zep graph delete: {graph_id}")
        return

    if lib_name == "everos" and _everos_use_group():
        for conv_idx in range(n):
            group_id = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
            if dry_run:
                print(f"  would delete everos group: {group_id}")
            elif getattr(client, "delete_group", None):
                with suppress(Exception):
                    client.delete_group(group_id)
                print(f"  everos delete_group: {group_id}")
        return

    for conv_idx in range(n):
        a = f"locomo_exp_user_{conv_idx}_speaker_a_{version}"
        b = f"locomo_exp_user_{conv_idx}_speaker_b_{version}"
        if lib_name == "supermemory":
            _delete_user_generic(client, lib_name, a, dry_run=dry_run)
            continue
        _delete_user_generic(client, lib_name, a, dry_run=dry_run)
        _delete_user_generic(client, lib_name, b, dry_run=dry_run)


def cleanup_lme(client: Any, lib_name: str, version: str, dry_run: bool) -> None:
    with open(LME_JSON) as f:
        lme_df = json.load(f)
    n = len(lme_df) if isinstance(lme_df, list) else len(lme_df)
    for conv_idx in range(n):
        uid = f"lme_exper_user_{version}_{conv_idx}"
        _delete_user_generic(client, lib_name, uid, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete LoCoMo or LongMemEval backend data for a given version.",
    )
    parser.add_argument("--lib", required=True, choices=SUPPORTED_LIBS)
    parser.add_argument(
        "--env",
        required=True,
        help="Dotenv path (shell sets via MEMEVAL_ENV_FILE)",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target ids without deleting backend data.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm destructive deletion. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated: locomo,lme or all",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        print("Refusing to delete without --yes. Use --dry-run to inspect targets.", file=sys.stderr)
        return 2

    if not os.path.isfile(args.env):
        print(f"Env file not found: {args.env}", file=sys.stderr)
        return 2
    load_dotenv(args.env, override=True)

    requested = {"locomo", "lme"} if args.datasets == "all" else {
        item.strip() for item in args.datasets.split(",") if item.strip()
    }
    valid = {"locomo", "lme"}
    unknown = requested - valid
    if unknown:
        print(f"Unknown datasets: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    client = create_client(args.lib)

    if "locomo" in requested:
        print("[locomo]")
        cleanup_locomo(client, args.lib, args.version, args.dry_run)
    if "lme" in requested:
        print("[lme]")
        cleanup_lme(client, args.lib, args.version, args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
