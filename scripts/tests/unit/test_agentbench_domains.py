from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.domains import DOMAIN_REGISTRY, create_domain
from agentbench.domains.code_implementation.livecode import _extract_code_from_text
from agentbench.domains.software_engineering import swebench
from agentbench.agents.openclaw import OpenClawAgentAdapter


def test_domain_registry_includes_migrated_domains():
    assert {
        "code_implementation",
        "information_retrieval",
        "knowledge_work",
        "reasoning",
        "software_engineering",
    } <= set(DOMAIN_REGISTRY)


def test_information_retrieval_loads_tasks_and_reports_missing_task(tmp_path: Path):
    dataset_file = tmp_path / "browsecomp.jsonl"
    dataset_file.write_text(
        json.dumps({"query_id": "q1", "query": "Who?", "answer": "Ada"}) + "\n"
    )
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"clusters": {"c": {"train": [], "test": ["q1"]}}}))

    domain = create_domain(
        "information_retrieval",
        {"dataset_file": str(dataset_file), "split_file": str(split_file)},
    )
    tasks = domain.load_tasks(Namespace(split="test", task=None))
    assert [task["name"] for task in tasks] == ["q1"]

    with pytest.raises(ValueError, match="No matching information_retrieval"):
        domain.load_tasks(Namespace(split="test", task="missing"))


def test_information_retrieval_setup_is_mcp_only(tmp_path: Path):
    dataset_file = tmp_path / "browsecomp.jsonl"
    dataset_file.write_text(
        json.dumps({"query_id": "q1", "query": "Who?", "answer": "Ada"}) + "\n"
    )
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"clusters": {"c": {"train": [], "test": ["q1"]}}}))
    domain = create_domain(
        "information_retrieval",
        {
            "dataset_file": str(dataset_file),
            "split_file": str(split_file),
            "disabled_tools": ["exec", "web_search"],
        },
    )

    env_info = domain.setup({"name": "q1"}, "openclaw", 1)

    assert env_info["mcp_only"] is True
    assert set(env_info["mcp_servers"]) == {"bcp-search"}
    assert env_info["disabled_tools"] == ["exec", "web_search"]


def test_knowledge_work_loads_tasks_and_reports_missing_task(tmp_path: Path):
    dataset_file = tmp_path / "dataset.json"
    task_id = "12345678-aaaa-bbbb-cccc-123456789abc"
    dataset_file.write_text(json.dumps([
        {
            "task_id": task_id,
            "sector": "sector",
            "occupation": "occupation",
            "prompt": "Create a report.",
            "rubric_json": [],
        }
    ]))
    split_file = tmp_path / "clusters.json"
    split_file.write_text(json.dumps({"clusters": {"c": {"train": [], "test": [task_id]}}}))

    domain = create_domain(
        "knowledge_work",
        {"dataset_file": str(dataset_file), "split_file": str(split_file)},
    )
    tasks = domain.load_tasks(Namespace(split="test", task=None))
    assert [task["name"] for task in tasks] == ["12345678"]

    with pytest.raises(ValueError, match="No matching knowledge_work"):
        domain.load_tasks(Namespace(split="test", task="missing"))


def test_knowledge_work_populates_framework_workspace(tmp_path: Path):
    workspace = tmp_path / "phase" / "task__trial_1" / "workspace"
    domain = create_domain("knowledge_work", {})

    env_info = domain.setup(
        {
            "name": "task",
            "task_id": "task-id",
            "reference_file_urls": [],
            "reference_files": [],
            "_workspace_dir": str(workspace),
        },
        "openclaw",
        1,
    )

    assert Path(env_info["workspace_dir"]) == workspace
    assert workspace.is_dir()


def test_swe_setup_disables_interactive_pagers(monkeypatch):
    commands = []
    monkeypatch.setattr(swebench, "_create_started_container", lambda *_: "container-id")
    monkeypatch.setattr(swebench, "setup_container_tmux", lambda *_: None)
    monkeypatch.setattr(swebench, "create_wrapper_script", lambda *_: "/tmp/wrapper")
    monkeypatch.setattr(
        swebench,
        "_docker_exec_in_tmux",
        lambda _container, command, timeout=30: commands.append(command),
    )

    adapter = swebench.SWEBenchAdapter({})
    monkeypatch.setattr(adapter, "_get_test_spec", lambda *_: object())
    env_info = adapter.setup({"name": "django__django-10880"}, "hermes", 1)

    assert env_info["container_name"] == "swebench-hermes-django__django-10880-t1"
    assert any("GIT_PAGER=cat" in command for command in commands)
    assert any("PAGER=cat" in command for command in commands)


def test_code_implementation_extracts_last_valid_python_block():
    text = """
First attempt:
```python
def broken(:
    pass
```

Final:
```python
def solve():
    print("ok")

if __name__ == "__main__":
    solve()
```
"""
    assert "def solve" in _extract_code_from_text(text)


def test_openclaw_parse_extra_marks_embedded_error():
    class Result:
        stdout = ""
        stderr = (
            '[agent/embedded] embedded run agent end: runId=x isError=true '
            'error=The model returned incomplete tool_call arguments. rawError=details'
        )

    parsed = OpenClawAgentAdapter({})._parse_extra(Result())
    assert parsed["completion_status"] == "error"
    assert "incomplete tool_call" in parsed["error"]
