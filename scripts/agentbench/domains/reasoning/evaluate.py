from __future__ import annotations

import logging
import re

log = logging.getLogger("omnimemeval.agentbench")


def _extract_braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        return "", start
    depth = 1
    i = start + 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start + 1:i - 1], i


def _normalize_answer(answer: str) -> str:
    s = str(answer or "").strip()
    marker = "\\boxed{"
    pos = s.find(marker)
    if pos != -1:
        content, _ = _extract_braced(s, pos + len(marker) - 1)
        s = content.strip()
    s = re.sub(r"^\$+|\$+$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac{", "\\frac{").replace("\\tfrac{", "\\frac{")
    while "\\frac{" in s:
        pos = s.find("\\frac{")
        numer, after_numer = _extract_braced(s, pos + 5)
        if after_numer < len(s) and s[after_numer] == "{":
            denom, after_denom = _extract_braced(s, after_numer)
            s = s[:pos] + f"{numer}/{denom}" + s[after_denom:]
        else:
            break
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").lower()
    return s


def _answers_match(expected: str, actual: str) -> bool:
    norm_expected = _normalize_answer(expected)
    norm_actual = _normalize_answer(actual)
    if norm_expected == norm_actual:
        return True
    try:
        return abs(float(norm_expected.replace(",", "")) - float(norm_actual.replace(",", ""))) < 1e-9
    except (ValueError, OverflowError):
        return False


def extract_answer(agent_output: str) -> str:
    if not agent_output:
        return ""

    results = []
    marker = "\\boxed{"
    i = 0
    while i < len(agent_output):
        pos = agent_output.find(marker, i)
        if pos == -1:
            break
        start = pos + len(marker)
        depth = 1
        j = start
        while j < len(agent_output) and depth > 0:
            if agent_output[j] == "{":
                depth += 1
            elif agent_output[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            results.append(agent_output[start:j - 1].strip())
        i = j
    if results:
        return f"\\boxed{{{results[-1]}}}"

    match = re.search(r"\*\*Answer:\*\*\s*(.+?)(?:\n|$)", agent_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", agent_output, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in agent_output.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def filter_verifier_input(agent_output: str) -> str:
    if not agent_output:
        return ""

    log_line = re.compile(
        r"^\s*(?:"
        r"\[[^\]]*(?:INFO|WARN|WARNING|ERROR|DEBUG|openclaw|memos|gateway)[^\]]*\]"
        r"|(?:INFO|WARN|WARNING|ERROR|DEBUG)\b"
        r"|(?:OpenClaw|Gateway|memos\.|embedded run |CLI:|subprocess )"
        r")",
        re.IGNORECASE,
    )
    lines = [
        line for line in agent_output.splitlines()
        if not log_line.search(line)
    ]
    return "\n".join(lines).strip()


def _llm_verify(
    expected: str,
    actual: str,
    problem: str,
    *,
    api_key: str,
    model: str,
    api_base: str,
    timeout: float,
    max_retries: int,
) -> tuple[bool, str]:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or "EMPTY",
        base_url=api_base or None,
        timeout=timeout,
        max_retries=max_retries,
    )
    prompt = (
        "You are a math judge. Compare the student's answer with the reference answer.\n\n"
        f"Problem: {problem}\n\n"
        f"Reference answer: {expected}\n\n"
        f"Student final answer: {actual}\n\n"
        "Respond with exactly TRUE or FALSE, then a short justification."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )
    text = (response.choices[0].message.content or "").strip()
    upper = text.upper()
    if "TRUE" in upper and "FALSE" not in upper:
        return True, text
    if "FALSE" in upper:
        return False, text
    return False, f"Judge response did not contain TRUE/FALSE: {text}"


def verify_answer(
    task: dict,
    agent_output: str,
    *,
    mode: str = "exact",
    api_key: str = "",
    model: str = "gpt-4o",
    api_base: str = "",
    timeout: float = 240.0,
    max_retries: int = 3,
) -> dict:
    expected = task.get("answer", "")
    agent_output = filter_verifier_input(agent_output)
    actual = extract_answer(agent_output)
    if not actual:
        return {
            "reward": 0.0,
            "correct": False,
            "expected": expected,
            "actual": "",
            "feedback": "No answer extracted from agent output.",
        }
    if mode == "llm" and (api_key or api_base):
        try:
            correct, feedback = _llm_verify(
                expected,
                actual,
                task.get("problem", ""),
                api_key=api_key,
                model=model,
                api_base=api_base,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception as exc:
            log.warning("LLM judge failed, falling back to exact match: %s", exc)
            correct = _answers_match(expected, actual)
            feedback = f"LLM judge failed; exact fallback. Error: {exc}"
    else:
        correct = _answers_match(expected, actual)
        feedback = "exact match" if correct else (
            f"Expected: {_normalize_answer(expected)}, got: {_normalize_answer(actual)}"
        )
    return {
        "reward": 1.0 if correct else 0.0,
        "correct": correct,
        "expected": expected,
        "actual": actual,
        "feedback": feedback,
    }
