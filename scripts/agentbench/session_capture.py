from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def now_ms() -> int:
    return int(time.time() * 1000)


def text_from_content(content: Any, field: str = "text") -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == field and item.get(field):
            parts.append(str(item[field]))
    return "\n".join(parts)


def iter_session_records(session_file: Path | None):
    if not session_file or not session_file.exists():
        return
    buffer = ""
    try:
        with session_file.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                buffer += line
                try:
                    record = json.loads(buffer)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
                buffer = ""
    except OSError:
        return


def extract_turns_from_session(session_file: Path | None) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    tool_calls_by_id: dict[str, dict[str, Any]] = {}

    for rec in iter_session_records(session_file):
        if rec.get("type") != "message":
            continue
        msg = rec.get("message") or {}
        role = msg.get("role")
        if role == "user":
            if current is not None:
                turns.append(current)
            current = {
                "userText": text_from_content(msg.get("content", [])),
                "agentText": "",
                "toolCalls": [],
            }
            tool_calls_by_id = {}
            continue
        if current is None:
            continue
        if role == "assistant":
            text = text_from_content(msg.get("content", []))
            if text:
                current["agentText"] = text
            for item in msg.get("content", []):
                if not isinstance(item, dict) or item.get("type") != "toolCall":
                    continue
                tool_call = {
                    "name": str(item.get("name") or ""),
                    "input": item.get("arguments") if item.get("arguments") is not None else {},
                    "output": "",
                    "errorCode": None,
                    "toolCallId": item.get("id"),
                    "startedAt": msg.get("timestamp") or now_ms(),
                    "endedAt": msg.get("timestamp") or now_ms(),
                }
                current["toolCalls"].append(tool_call)
                if item.get("id"):
                    tool_calls_by_id[str(item["id"])] = tool_call
        elif role == "toolResult":
            tool_call_id = msg.get("toolCallId")
            tool_call = tool_calls_by_id.get(str(tool_call_id)) if tool_call_id else None
            if tool_call is None:
                tool_call = {
                    "name": str(msg.get("toolName") or "tool"),
                    "input": {},
                    "output": "",
                    "errorCode": None,
                    "toolCallId": tool_call_id,
                    "startedAt": msg.get("timestamp") or now_ms(),
                    "endedAt": msg.get("timestamp") or now_ms(),
                }
                current["toolCalls"].append(tool_call)
            tool_call["output"] = (
                text_from_content(msg.get("content", []))
                or (msg.get("details") or {}).get("aggregated", "")
            )
            tool_call["endedAt"] = msg.get("timestamp") or now_ms()
            if msg.get("isError"):
                tool_call["errorCode"] = "tool_error"

    if current is not None:
        turns.append(current)
    return turns


def build_task_feedback_turns(
    *,
    session_file: Path | None,
    task_prompt: str,
    agent_result: dict[str, Any],
    feedback_prompt: str,
    feedback_result: dict[str, Any],
    feedback_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    turns = extract_turns_from_session(session_file)
    task_turn = turns[0] if turns else {"userText": task_prompt, "agentText": "", "toolCalls": []}
    feedback_turn = None
    for turn in turns[1:]:
        if str(turn.get("userText", "")).startswith(feedback_prefix):
            feedback_turn = turn
            break
    if feedback_turn is None:
        feedback_turn = turns[1] if len(turns) > 1 else {
            "userText": feedback_prompt,
            "agentText": "",
            "toolCalls": [],
        }

    task_turn = dict(task_turn)
    feedback_turn = dict(feedback_turn)
    task_turn["userText"] = task_turn.get("userText") or task_prompt
    feedback_turn["userText"] = feedback_turn.get("userText") or feedback_prompt
    task_turn["agentText"] = task_turn.get("agentText") or agent_result.get("response", "")
    feedback_turn["agentText"] = feedback_turn.get("agentText") or feedback_result.get("response", "")
    task_turn["toolCalls"] = task_turn.get("toolCalls") or []
    feedback_turn["toolCalls"] = feedback_turn.get("toolCalls") or []
    return task_turn, feedback_turn, len(turns)
