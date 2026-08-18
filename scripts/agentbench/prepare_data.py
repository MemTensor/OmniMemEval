#!/usr/bin/env python3
"""Download, normalize, migrate, and verify OmniMemEval AgentBench data.

The public ``EverMind-AI/EvoAgentBench`` repository contains task splits,
OmniMath rows, and GDPVal rubrics. The other benchmark payloads live in their
upstream repositories. This command turns those separate sources into the
layout consumed by ``configs/agentbench/domains/*.yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "agentbench"
ALL_DOMAINS = (
    "information_retrieval",
    "reasoning",
    "software_engineering",
    "code_implementation",
    "knowledge_work",
)

EVOAGENTBENCH_REPO = "EverMind-AI/EvoAgentBench"
BROWSECOMP_REPO = "Tevatron/browsecomp-plus"
BROWSECOMP_CORPUS_REPO = "Tevatron/browsecomp-plus-corpus"
SWEBENCH_REPO = "princeton-nlp/SWE-bench_Verified"
SWEBENCH_PARQUET = "data/test-00000-of-00001.parquet"
LIVECODE_REPO = "livecodebench/code_generation_lite"
GDPVAL_REPO = "openai/gdpval"
GDPVAL_PARQUET = "data/train-00000-of-00001.parquet"


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _parse_domains(raw: str | None) -> tuple[str, ...]:
    if not raw or raw.strip().lower() == "all":
        return ALL_DOMAINS
    requested = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = sorted(set(requested) - set(ALL_DOMAINS))
    if unknown:
        raise ValueError(
            f"Unknown domain(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(ALL_DOMAINS)}"
        )
    return requested


def _split_ids(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def add(values: Iterable[Any]) -> None:
        for value in values:
            item = str(value)
            if item not in seen:
                seen.add(item)
                ids.append(item)

    found_flat = False
    for part in ("train", "test"):
        values = payload.get(part)
        if isinstance(values, list):
            add(values)
            found_flat = True
    if found_flat:
        return ids

    clusters = payload.get("clusters")
    if isinstance(clusters, dict):
        for cluster in clusters.values():
            if not isinstance(cluster, dict):
                continue
            for part in ("train", "test"):
                values = cluster.get(part)
                if isinstance(values, list):
                    add(values)
    return ids


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _iter_json_array(
    path: Path, *, chunk_size: int = 16 * 1024 * 1024
) -> Iterable[Any]:
    """Read a large top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False

    with path.open(encoding="utf-8") as handle:
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = not buffer
            while position < len(buffer) and buffer[position].isspace():
                position += 1

            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array in {path}")
                position += 1
                started = True
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return

            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"Incomplete JSON array in {path}") from None
                chunk = handle.read(chunk_size)
                buffer = buffer[position:] + chunk
                position = 0
                if not chunk:
                    eof = True
                continue
            yield value
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def _write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
    temporary.replace(path)


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required source file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required source directory not found: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _ensure_local_data_dir(data_dir: Path, *, force: bool) -> None:
    """Make the destination local without deleting real downloaded directories."""
    if data_dir.is_symlink():
        if not force:
            raise RuntimeError(
                f"{data_dir} is a symlink. Re-run with --force to replace only the link."
            )
        data_dir.unlink()
    elif data_dir.exists() and not data_dir.is_dir():
        raise NotADirectoryError(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    foreign_links = [child for child in data_dir.iterdir() if child.is_symlink()]
    if foreign_links and not force:
        names = ", ".join(path.name for path in foreign_links[:8])
        raise RuntimeError(
            f"{data_dir} contains linked assets ({names}). Re-run with --force to "
            "unlink those entries and prepare a self-contained local dataset."
        )
    if force:
        for link in foreign_links:
            link.unlink()


def normalize_evoagentbench_layout(data_dir: Path) -> None:
    """Copy public split/rubric files to the runtime paths in domain YAMLs."""
    mappings = (
        (
            data_dir / "Information Retrieval" / "task_split.json",
            data_dir / "BrowseComp-Plus" / "task_split.json",
        ),
        (
            data_dir / "Code Implementation" / "task_split.json",
            data_dir / "livecode" / "task_split.json",
        ),
        (
            data_dir / "Software Engineering" / "task_split.json",
            data_dir / "swebench" / "task_split.json",
        ),
        (
            data_dir / "Knowledge Work" / "task_split.json",
            data_dir / "gdpval" / "clusters.json",
        ),
    )
    for source, destination in mappings:
        _copy_file(source, destination)
    _copy_tree_contents(
        data_dir / "Knowledge Work" / "meta_prompts",
        data_dir / "gdpval" / "meta_prompts",
    )


def download_evoagentbench(data_dir: Path, *, revision: str) -> None:
    print(f"[splits] Downloading {EVOAGENTBENCH_REPO}@{revision} ...")
    _snapshot_download(
        repo_id=EVOAGENTBENCH_REPO,
        repo_type="dataset",
        revision=revision,
        local_dir=str(data_dir),
    )
    normalize_evoagentbench_layout(data_dir)


def prepare_browsecomp(
    data_dir: Path,
    *,
    revision: str,
    build_index: bool,
    force: bool,
) -> None:
    output_dir = data_dir / "BrowseComp-Plus"
    setup_script = (
        ROOT
        / "scripts"
        / "agentbench"
        / "utils"
        / "browsecomp-plus-tools"
        / "setup_data.py"
    )
    print(f"[information_retrieval] Downloading and decrypting {BROWSECOMP_REPO} ...")
    subprocess.run(
        [
            sys.executable,
            str(setup_script),
            "--output-dir",
            str(output_dir),
            "--revision",
            revision,
            "--skip-index",
        ],
        cwd=ROOT,
        check=True,
    )

    if not build_index:
        print(
            "[information_retrieval] Query data is ready. Dense index was not built; "
            "pass --build-ir-index after configuring IR_EMBEDDING_* in .env.agent."
        )
        return

    builder = setup_script.with_name("build_dense_index.py")
    index_dir = output_dir / "indexes" / "openai-compatible"
    command = [
        sys.executable,
        str(builder),
        "--config",
        str(ROOT / "configs" / "agentbench" / "domains" / "information_retrieval.yaml"),
        "--output-dir",
        str(index_dir),
        "--corpus-revision",
        revision,
    ]
    if force:
        command.append("--overwrite")
    print(f"[information_retrieval] Building dense index from {BROWSECOMP_CORPUS_REPO} ...")
    subprocess.run(command, cwd=ROOT, check=True)


def prepare_swebench(data_dir: Path, *, revision: str) -> None:
    output_dir = data_dir / "swebench"
    source_dir = output_dir / "source"
    print(f"[software_engineering] Downloading {SWEBENCH_REPO}@{revision} ...")
    _snapshot_download(
        repo_id=SWEBENCH_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[SWEBENCH_PARQUET],
        local_dir=str(source_dir),
    )
    _copy_file(
        source_dir / SWEBENCH_PARQUET,
        output_dir / "test-00000-of-00001.parquet",
    )
    print(
        "[software_engineering] Parquet is ready. SWE task images are pulled from "
        "the Docker registry on demand when local tar files are absent."
    )


def _natural_test_file_key(path: Path) -> int:
    match = re.fullmatch(r"test(\d*)\.jsonl", path.name)
    if not match:
        return 10_000
    return int(match.group(1) or "1")


def build_livecode_cache(source_dir: Path, split_file: Path, output_file: Path) -> int:
    """Stream selected LiveCodeBench rows into the cache consumed by the adapter."""
    split_payload = _read_json(split_file)
    wanted_order = _split_ids(split_payload)
    wanted = set(wanted_order)
    if not wanted:
        raise ValueError(f"No train/test task IDs found in {split_file}")

    source_files = sorted(source_dir.glob("test*.jsonl"), key=_natural_test_file_key)
    if not source_files:
        raise FileNotFoundError(f"No LiveCodeBench test*.jsonl files under {source_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.tmp")
    found: set[str] = set()
    first = True
    with temporary.open("w", encoding="utf-8") as output:
        output.write("[")
        for source_file in source_files:
            with source_file.open(encoding="utf-8") as rows:
                for line in rows:
                    row = json.loads(line)
                    question_id = str(row["question_id"])
                    if question_id not in wanted or question_id in found:
                        continue
                    if not first:
                        output.write(",")
                    output.write(line.strip())
                    found.add(question_id)
                    first = False
        output.write("]\n")

    missing = [question_id for question_id in wanted_order if question_id not in found]
    if missing:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"LiveCodeBench source is missing {len(missing)} split task(s): {missing[:20]}"
        )
    temporary.replace(output_file)
    _write_json(output_file.with_suffix(".ids.json"), sorted(found))
    return len(found)


def prepare_livecode(data_dir: Path, *, revision: str) -> None:
    output_dir = data_dir / "livecode"
    source_dir = output_dir / "source"
    print(f"[code_implementation] Downloading {LIVECODE_REPO}@{revision} ...")
    _snapshot_download(
        repo_id=LIVECODE_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["test*.jsonl"],
        local_dir=str(source_dir),
    )
    count = build_livecode_cache(
        source_dir,
        output_dir / "task_split.json",
        output_dir / "release_v6.json",
    )
    print(f"[code_implementation] Cached {count} split tasks in release_v6.json")


def _gdpval_records_from_parquet(parquet_file: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "GDPVal preparation requires pyarrow; install requirements_agentbench.txt"
        ) from exc

    rows = parquet.read_table(parquet_file).to_pylist()
    records = []
    fields = (
        "task_id",
        "sector",
        "occupation",
        "prompt",
        "reference_files",
        "reference_file_urls",
        "deliverable_files",
        "deliverable_file_urls",
        "rubric_json",
        "rubric_pretty",
    )
    for row in rows:
        record = {field: row.get(field) for field in fields}
        for field in (
            "reference_files",
            "reference_file_urls",
            "deliverable_files",
            "deliverable_file_urls",
        ):
            record[field] = record.get(field) or []
        rubric_json = record.get("rubric_json")
        if rubric_json is not None and not isinstance(rubric_json, str):
            record["rubric_json"] = json.dumps(rubric_json, ensure_ascii=False)
        records.append(record)
    return records


def link_gdpval_references(gdpval_dir: Path, records: Sequence[dict[str, Any]]) -> int:
    """Expose HF hash-organized files under the task-id layout used by the adapter."""
    created = 0
    missing: list[Path] = []
    reference_root = gdpval_dir / "reference_files"
    for row in records:
        task_id = str(row["task_id"])
        references = row.get("reference_files") or []
        if not references:
            continue
        task_dir = reference_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        for relative in references:
            source = gdpval_dir / str(relative)
            destination = task_dir / Path(str(relative)).name
            if not source.is_file():
                missing.append(source)
                continue
            if destination.exists():
                continue
            if destination.is_symlink():
                destination.unlink()
            destination.symlink_to(os.path.relpath(source, task_dir))
            created += 1
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(
            f"GDPVal download is missing {len(missing)} reference file(s): {examples}"
        )
    return created


def prepare_gdpval(data_dir: Path, *, revision: str, download_references: bool) -> None:
    output_dir = data_dir / "gdpval"
    patterns = [GDPVAL_PARQUET]
    if download_references:
        patterns.append("reference_files/*")
    print(f"[knowledge_work] Downloading {GDPVAL_REPO}@{revision} ...")
    _snapshot_download(
        repo_id=GDPVAL_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=str(output_dir),
    )
    records = _gdpval_records_from_parquet(output_dir / GDPVAL_PARQUET)
    _write_json(output_dir / "dataset.json", records, indent=None)
    print(f"[knowledge_work] Wrote {len(records)} tasks to dataset.json")
    if download_references:
        created = link_gdpval_references(output_dir, records)
        print(f"[knowledge_work] Prepared {created} task-local reference links")
    else:
        print(
            "[knowledge_work] Reference files were skipped. The adapter will fetch them "
            "from reference_file_urls on first use."
        )


def _file_ready(path: Path, minimum_size: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= minimum_size
    except OSError:
        return False


def verify_data(
    data_dir: Path,
    domains: Sequence[str],
    *,
    require_ir_index: bool,
    require_gdpval_references: bool,
    deep: bool,
) -> list[Check]:
    checks: list[Check] = []

    def record(ok: bool, name: str, detail: str) -> None:
        checks.append(Check("PASS" if ok else "FAIL", name, detail))

    if "information_retrieval" in domains:
        base = data_dir / "BrowseComp-Plus"
        dataset = base / "browsecomp_plus_decrypted.jsonl"
        split = base / "task_split.json"
        record(_file_ready(dataset), "information_retrieval dataset", str(dataset))
        record(_file_ready(split), "information_retrieval split", str(split))
        if deep and _file_ready(dataset) and _file_ready(split):
            wanted = set(_split_ids(_read_json(split)))
            source_ids: set[str] = set()
            with dataset.open(encoding="utf-8") as rows:
                for line in rows:
                    source_ids.add(str(json.loads(line)["query_id"]))
            record(
                wanted <= source_ids,
                "information_retrieval split coverage",
                f"{len(wanted)} split IDs / {len(source_ids)} source rows",
            )
        index_dir = base / "indexes" / "openai-compatible"
        shards = list(index_dir.glob("corpus.*.pkl"))
        index_ok = bool(shards) and _file_ready(index_dir / "metadata.json")
        status = "PASS" if index_ok else ("FAIL" if require_ir_index else "WARN")
        checks.append(
            Check(
                status,
                "information_retrieval index",
                f"{len(shards)} shard(s) in {index_dir}",
            )
        )

    if "reasoning" in domains:
        base = data_dir / "Reasoning & Problem Decomposition" / "test_set_100"
        for split_name in ("train", "test"):
            path = base / f"{split_name}.jsonl"
            if deep and _file_ready(path):
                count = sum(1 for _ in path.open(encoding="utf-8"))
                record(count > 0, f"reasoning {split_name}", f"{count} rows in {path}")
            else:
                record(_file_ready(path), f"reasoning {split_name}", str(path))

    if "software_engineering" in domains:
        base = data_dir / "swebench"
        parquet_file = base / "test-00000-of-00001.parquet"
        split_file = base / "task_split.json"
        record(_file_ready(parquet_file), "software_engineering parquet", str(parquet_file))
        record(_file_ready(split_file), "software_engineering split", str(split_file))
        if deep and _file_ready(parquet_file) and _file_ready(split_file):
            import pyarrow.parquet as parquet

            table = parquet.read_table(parquet_file, columns=["instance_id"])
            source_ids = set(table["instance_id"].to_pylist())
            wanted = set(_split_ids(_read_json(split_file)))
            record(
                wanted <= source_ids,
                "software_engineering split coverage",
                f"{len(wanted)} split IDs / {len(source_ids)} source rows",
            )

    if "code_implementation" in domains:
        base = data_dir / "livecode"
        split_file = base / "task_split.json"
        cache_file = base / "release_v6.json"
        ids_file = base / "release_v6.ids.json"
        record(_file_ready(split_file), "code_implementation split", str(split_file))
        record(_file_ready(cache_file), "code_implementation cache", str(cache_file))
        if _file_ready(split_file) and (_file_ready(ids_file) or deep):
            wanted = set(_split_ids(_read_json(split_file)))
            if _file_ready(ids_file):
                cached = set(map(str, _read_json(ids_file)))
            elif _file_ready(cache_file):
                cached = {
                    str(row["question_id"])
                    for row in _iter_json_array(cache_file)
                }
            else:
                cached = set()
            record(
                wanted <= cached,
                "code_implementation split coverage",
                f"{len(wanted)} split IDs / {len(cached)} cached rows",
            )
        elif deep:
            record(False, "code_implementation split coverage", f"Missing {ids_file}")

    if "knowledge_work" in domains:
        base = data_dir / "gdpval"
        dataset_file = base / "dataset.json"
        split_file = base / "clusters.json"
        meta_dir = base / "meta_prompts"
        record(_file_ready(dataset_file), "knowledge_work dataset", str(dataset_file))
        record(_file_ready(split_file), "knowledge_work split", str(split_file))
        record(
            meta_dir.is_dir() and any(meta_dir.glob("*.json")),
            "knowledge_work meta prompts",
            str(meta_dir),
        )
        if require_gdpval_references and _file_ready(dataset_file) and _file_ready(split_file):
            records = {str(row["task_id"]): row for row in _read_json(dataset_file)}
            wanted = _split_ids(_read_json(split_file))
            missing: list[str] = []
            for task_id in wanted:
                row = records.get(task_id)
                if row is None:
                    missing.append(f"task:{task_id}")
                    continue
                for relative in row.get("reference_files") or []:
                    path = base / "reference_files" / task_id / Path(str(relative)).name
                    if not path.is_file():
                        missing.append(str(path))
            record(
                not missing,
                "knowledge_work reference coverage",
                "all selected references present" if not missing else f"{len(missing)} missing",
            )

    return checks


def print_checks(checks: Sequence[Check]) -> None:
    for check in checks:
        print(f"{check.status}\t{check.name}\t{check.detail}")


def migrate_existing(source_root: Path, data_dir: Path, *, force: bool) -> None:
    candidates = (source_root, source_root / "data")
    source_data = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_dir()
            and (candidate / "Reasoning & Problem Decomposition").exists()
        ),
        None,
    )
    if source_data is None:
        raise FileNotFoundError(f"No EvoAgentBench data directory found under {source_root}")

    data_dir.mkdir(parents=True, exist_ok=True)
    for source in source_data.iterdir():
        destination = data_dir / source.name
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        if destination.exists() or destination.is_symlink():
            if not force:
                raise FileExistsError(f"Refusing to replace existing asset: {destination}")
            if not destination.is_symlink():
                raise RuntimeError(
                    f"--force only replaces links during migration, not real data: {destination}"
                )
            destination.unlink()
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    print(f"Migrated links from {source_data} to {data_dir}")


def _write_manifest(
    data_dir: Path,
    domains: Sequence[str],
    *,
    revision: str,
    upstream_revision: str,
    gdpval_references: bool,
    ir_index: bool,
) -> None:
    _write_json(
        data_dir / "prepare_manifest.json",
        {
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "domains": list(domains),
            "sources": {
                EVOAGENTBENCH_REPO: revision,
                BROWSECOMP_REPO: upstream_revision if "information_retrieval" in domains else None,
                BROWSECOMP_CORPUS_REPO: upstream_revision if ir_index else None,
                SWEBENCH_REPO: upstream_revision if "software_engineering" in domains else None,
                LIVECODE_REPO: upstream_revision if "code_implementation" in domains else None,
                GDPVAL_REPO: upstream_revision if "knowledge_work" in domains else None,
            },
            "gdpval_references_downloaded": gdpval_references,
            "information_retrieval_index_built": ir_index,
        },
    )


def _add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Prepared data root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--domains",
        default="all",
        help="Comma-separated domains or 'all'.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download", help="Download and normalize required upstream data."
    )
    _add_common_data_args(download)
    download.add_argument("--revision", default="main", help="EvoAgentBench dataset revision.")
    download.add_argument(
        "--upstream-revision", default="main", help="Revision for upstream datasets."
    )
    download.add_argument(
        "--force",
        action="store_true",
        help="Replace foreign symlinks and overwrite rebuilt assets.",
    )
    download.add_argument(
        "--build-ir-index",
        action="store_true",
        help="Build the configured BrowseComp dense index.",
    )
    download.add_argument(
        "--skip-gdpval-references",
        action="store_true",
        help="Prepare GDPVal metadata only; fetch reference files lazily at runtime.",
    )
    download.add_argument(
        "--disable-xet",
        action="store_true",
        help="Use standard HF HTTP downloads instead of Xet.",
    )
    download.add_argument("--no-verify", action="store_true", help="Skip post-download checks.")

    verify = subparsers.add_parser(
        "verify", help="Verify prepared domain data without downloading."
    )
    _add_common_data_args(verify)
    verify.add_argument("--require-ir-index", action="store_true")
    verify.add_argument("--allow-missing-gdpval-references", action="store_true")
    verify.add_argument(
        "--deep",
        action="store_true",
        help="Scan source rows and verify split ID coverage.",
    )

    migrate = subparsers.add_parser(
        "migrate", help="Link an existing complete EvoAgentBench data tree."
    )
    migrate.add_argument(
        "--source", required=True, help="Existing EvoAgentBench checkout or data directory."
    )
    migrate.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    migrate.add_argument("--force", action="store_true", help="Replace conflicting symlinks only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()

    if args.command == "migrate":
        migrate_existing(Path(args.source).expanduser().resolve(), data_dir, force=args.force)
        return 0

    domains = _parse_domains(args.domains)
    if args.command == "verify":
        checks = verify_data(
            data_dir,
            domains,
            require_ir_index=args.require_ir_index,
            require_gdpval_references=not args.allow_missing_gdpval_references,
            deep=args.deep,
        )
        print_checks(checks)
        return 1 if any(check.status == "FAIL" for check in checks) else 0

    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    _ensure_local_data_dir(data_dir, force=args.force)
    download_evoagentbench(data_dir, revision=args.revision)

    if "information_retrieval" in domains:
        prepare_browsecomp(
            data_dir,
            revision=args.upstream_revision,
            build_index=args.build_ir_index,
            force=args.force,
        )
    if "software_engineering" in domains:
        prepare_swebench(data_dir, revision=args.upstream_revision)
    if "code_implementation" in domains:
        prepare_livecode(data_dir, revision=args.upstream_revision)
    if "knowledge_work" in domains:
        prepare_gdpval(
            data_dir,
            revision=args.upstream_revision,
            download_references=not args.skip_gdpval_references,
        )

    _write_manifest(
        data_dir,
        domains,
        revision=args.revision,
        upstream_revision=args.upstream_revision,
        gdpval_references=(
            "knowledge_work" in domains and not args.skip_gdpval_references
        ),
        ir_index=("information_retrieval" in domains and args.build_ir_index),
    )

    if args.no_verify:
        return 0
    checks = verify_data(
        data_dir,
        domains,
        require_ir_index=("information_retrieval" in domains and args.build_ir_index),
        require_gdpval_references=(
            "knowledge_work" in domains and not args.skip_gdpval_references
        ),
        deep=False,
    )
    print_checks(checks)
    return 1 if any(check.status == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
