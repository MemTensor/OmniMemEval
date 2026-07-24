from __future__ import annotations

from pathlib import Path

import pytest

from scripts.agentbench.plugin_lifecycle import hermes_supermemory_ctl
from scripts.agentbench.plugin_lifecycle import hindsight_ctl


def test_wait_idle_fails_closed_for_failed_documents(monkeypatch):
    monkeypatch.setattr(
        hermes_supermemory_ctl,
        "documents",
        lambda *, content: [
            {"status": "completed"},
            {"status": "failed"},
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"Supermemory document ingestion failed: completed=1, failed=1",
    ):
        hermes_supermemory_ctl.wait_idle(1)


def test_openclaw_supermemory_wait_fails_closed_for_failed_documents():
    source = (
        Path(__file__).parents[2]
        / "agentbench"
        / "plugin_lifecycle"
        / "supermemory_node.js"
    ).read_text(encoding="utf-8")

    assert "failedStatuses" in source
    assert "Supermemory document ingestion failed" in source
    assert "entry.config.containerTag = runContainerTag" in source
    assert "refusing non-run-scoped Supermemory container" in source
    assert "process.env.SUPERMEMORY_ENTITY_CONTEXT" in source


def test_hindsight_wait_fails_closed_for_failed_operations(monkeypatch):
    monkeypatch.setattr(
        hindsight_ctl,
        "request",
        lambda *args, **kwargs: {"operations": [{"status": "failed"}]},
    )
    monkeypatch.setattr(hindsight_ctl, "bank_id", lambda: "omnimemeval-test")

    with pytest.raises(RuntimeError, match=r"Hindsight bank operation failed"):
        hindsight_ctl.wait_idle(1, 0)
