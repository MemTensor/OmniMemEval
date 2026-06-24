"""DingTalk webhook notification for MemEval.

Generic module — any benchmark script can call `send_eval_result()` to push
a formatted evaluation summary to a DingTalk group robot.

Env vars (set in .env):
    DINGTALK_ACCESS_TOKEN  — robot webhook access_token
    DINGTALK_SECRET        — HMAC-SHA256 signing secret
"""

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
import urllib.request
import json as _json

_WEBHOOK_BASE = "https://oapi.dingtalk.com/robot/send"


def _sign(secret: str) -> tuple[str, str]:
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode(), string_to_sign.encode(), digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign


def _send(access_token: str, secret: str, payload: dict) -> dict:
    timestamp, sign = _sign(secret)
    url = f"{_WEBHOOK_BASE}?access_token={access_token}&timestamp={timestamp}&sign={sign}"
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = _json.loads(resp.read())
    if result.get("errcode") != 0:
        raise RuntimeError(f"DingTalk API error: {result}")
    return result


def _build_eval_markdown(
    *,
    benchmark: str,
    framework: str,
    version: str,
    run_time: str,
    overall_score: float,
    overall_std: float,
    category_scores: list[dict],
    context_tokens: float | None = None,
    metric_name: str = "LLM-as-Judge",
) -> str:
    """Build a markdown string for DingTalk message.

    category_scores: list of {"name": str, "score": float, "count": int}
    """
    lines = [
        f"### MemEval · {benchmark}",
        "",
        f"> **{framework}** / {version}",
        "",
        f"- Run time: {run_time}",
        f"- **{metric_name}: {overall_score:.4f}** ± {overall_std:.4f}",
    ]
    if context_tokens is not None:
        lines.append(f"- **Context Tokens (avg): {context_tokens:.1f}**")
    lines.extend([
        "",
        "| Category | Score | Questions |",
        "|------|------|--------|",
    ])
    for cat in category_scores:
        lines.append(f"| {cat['name']} | {cat['score']:.4f} | {cat['count']} |")
    return "\n".join(lines)


def send_eval_result(
    *,
    benchmark: str,
    framework: str,
    version: str,
    run_time: str,
    overall_score: float,
    overall_std: float = 0.0,
    category_scores: list[dict] | None = None,
    context_tokens: float | None = None,
    metric_name: str = "LLM-as-Judge",
    access_token: str | None = None,
    secret: str | None = None,
) -> bool:
    """Send evaluation result to DingTalk.  Returns True on success.

    Parameters
    ----------
    benchmark, framework, version, run_time : str
        Experiment metadata shown in the header.
    overall_score, overall_std : float
        Overall metric value and its std across runs.
    category_scores : list[dict], optional
        Each dict: {"name": str, "score": float, "count": int}.
    context_tokens : float, optional
        Value shown in the optional DingTalk "Context Tokens (avg)" summary line.
    metric_name : str
        Display name for the primary metric (e.g. "LLM-as-Judge", "Accuracy").
    access_token, secret : str, optional
        Override env vars DINGTALK_ACCESS_TOKEN / DINGTALK_SECRET.
    """
    token = access_token or os.environ.get("DINGTALK_ACCESS_TOKEN", "")
    sec = secret or os.environ.get("DINGTALK_SECRET", "")
    if not token or not sec:
        print("  [DingTalk] skipped: DINGTALK_ACCESS_TOKEN / DINGTALK_SECRET not configured")
        return False

    md = _build_eval_markdown(
        benchmark=benchmark,
        framework=framework,
        version=version,
        run_time=run_time,
        overall_score=overall_score,
        overall_std=overall_std,
        category_scores=category_scores or [],
        context_tokens=context_tokens,
        metric_name=metric_name,
    )

    payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": f"MemEval · {benchmark} · {framework}",
            "text": md,
        },
    }

    try:
        _send(token, sec, payload)
        print("  [DingTalk] notification sent")
        return True
    except Exception as e:
        print(f"  [DingTalk] notification failed: {e}")
        return False
