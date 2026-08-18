from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.prepare_data import (  # noqa: E402
    _ensure_local_data_dir,
    _iter_json_array,
    _split_ids,
    build_livecode_cache,
    link_gdpval_references,
    normalize_evoagentbench_layout,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_normalize_evoagentbench_layout_populates_runtime_paths(tmp_path: Path):
    _write_json(tmp_path / "Information Retrieval" / "task_split.json", {"train": ["q1"]})
    _write_json(tmp_path / "Code Implementation" / "task_split.json", {"train": ["c1"]})
    _write_json(tmp_path / "Software Engineering" / "task_split.json", {"test": ["s1"]})
    _write_json(tmp_path / "Knowledge Work" / "task_split.json", {"test": ["k1"]})
    _write_json(
        tmp_path / "Knowledge Work" / "meta_prompts" / "Editors.json",
        {"occupation": "Editors"},
    )

    normalize_evoagentbench_layout(tmp_path)

    browsecomp_split = tmp_path / "BrowseComp-Plus" / "task_split.json"
    assert json.loads(browsecomp_split.read_text())["train"] == ["q1"]
    assert json.loads((tmp_path / "livecode" / "task_split.json").read_text())["train"] == ["c1"]
    assert json.loads((tmp_path / "swebench" / "task_split.json").read_text())["test"] == ["s1"]
    assert json.loads((tmp_path / "gdpval" / "clusters.json").read_text())["test"] == ["k1"]
    assert (tmp_path / "gdpval" / "meta_prompts" / "Editors.json").is_file()


def test_split_ids_supports_flat_and_cluster_payloads():
    assert _split_ids({"train": [1, 2], "test": [2, 3]}) == ["1", "2", "3"]
    assert _split_ids(
        {
            "clusters": {
                "a": {"train": ["x"], "test": ["y"]},
                "b": {"train": ["x"], "test": ["z"]},
            }
        }
    ) == ["x", "y", "z"]


def test_iter_json_array_streams_values_across_small_chunks(tmp_path: Path):
    path = tmp_path / "large.json"
    values = [{"question_id": f"q-{index}", "value": "x" * 20} for index in range(5)]
    path.write_text(json.dumps(values), encoding="utf-8")

    assert list(_iter_json_array(path, chunk_size=13)) == values


def test_build_livecode_cache_streams_only_split_tasks(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "test.jsonl").write_text(
        "\n".join(
            json.dumps({"question_id": qid, "payload": qid})
            for qid in ("keep-a", "drop")
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "test2.jsonl").write_text(
        json.dumps({"question_id": "keep-b", "payload": "keep-b"}) + "\n",
        encoding="utf-8",
    )
    split_file = tmp_path / "task_split.json"
    _write_json(split_file, {"train": ["keep-a"], "test": ["keep-b"]})
    output_file = tmp_path / "release_v6.json"

    count = build_livecode_cache(source_dir, split_file, output_file)

    assert count == 2
    assert {row["question_id"] for row in json.loads(output_file.read_text())} == {
        "keep-a",
        "keep-b",
    }
    assert json.loads((tmp_path / "release_v6.ids.json").read_text()) == [
        "keep-a",
        "keep-b",
    ]


def test_build_livecode_cache_fails_when_split_task_is_missing(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "test.jsonl").write_text(
        json.dumps({"question_id": "present"}) + "\n",
        encoding="utf-8",
    )
    split_file = tmp_path / "task_split.json"
    _write_json(split_file, {"train": ["present"], "test": ["missing"]})

    with pytest.raises(RuntimeError, match="missing 1 split task"):
        build_livecode_cache(source_dir, split_file, tmp_path / "release_v6.json")


def test_link_gdpval_references_creates_task_id_layout(tmp_path: Path):
    source = tmp_path / "reference_files" / "hash-id" / "input.xlsx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"xlsx")
    records = [
        {
            "task_id": "task-uuid",
            "reference_files": ["reference_files/hash-id/input.xlsx"],
        }
    ]

    created = link_gdpval_references(tmp_path, records)
    destination = tmp_path / "reference_files" / "task-uuid" / "input.xlsx"

    assert created == 1
    assert destination.is_symlink()
    assert destination.read_bytes() == b"xlsx"


def test_ensure_local_data_dir_only_replaces_links_with_force(tmp_path: Path):
    external = tmp_path / "external"
    external.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "linked").symlink_to(external, target_is_directory=True)
    real = data_dir / "real"
    real.mkdir()

    with pytest.raises(RuntimeError, match="contains linked assets"):
        _ensure_local_data_dir(data_dir, force=False)

    _ensure_local_data_dir(data_dir, force=True)

    assert not (data_dir / "linked").exists()
    assert real.is_dir()
