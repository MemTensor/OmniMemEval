from __future__ import annotations

import json
import os
from pathlib import Path

from agentbench.domains.base import DomainAdapter
from agentbench.domains.reasoning.evaluate import verify_answer


_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.md").read_text()


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class ReasoningDomain(DomainAdapter):
    name = "reasoning"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._dataset_cache: dict[str, list[dict]] = {}

    def _data_dir(self) -> Path:
        data_dir = self.config.get("data_dir")
        if not data_dir:
            raise ValueError("reasoning domain requires data_dir in domain config")
        return Path(data_dir).expanduser()

    def _get_dataset(self, split: str) -> list[dict]:
        if split not in self._dataset_cache:
            path = self._data_dir() / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Reasoning data not found: {path}")
            self._dataset_cache[split] = _load_jsonl(path)
        return self._dataset_cache[split]

    def load_tasks(self, args) -> list[dict]:
        split = args.split or "test"
        if split.isdigit():
            dataset = self._get_dataset("test")[:int(split)]
        else:
            dataset = self._get_dataset(split)

        if args.task:
            ids = [item.strip() for item in args.task.split(",") if item.strip()]
            normalized = {item.removeprefix("omni_") for item in ids}
            dataset = [
                row for row in dataset
                if str(row.get("_idx")) in normalized or str(row.get("id")) in ids
            ]
            if not dataset:
                raise ValueError(
                    f"No reasoning tasks matched --task {args.task!r} in split {split!r}"
                )

        tasks = []
        for rec in dataset:
            idx = rec.get("_idx", rec.get("id", len(tasks)))
            domain = rec.get("domain", [])
            domain_str = " | ".join(domain) if isinstance(domain, list) else str(domain)
            tasks.append({
                "name": f"omni_{idx}",
                "task_id": str(idx),
                "problem": rec["problem"],
                "answer": rec.get("answer", ""),
                "solution": rec.get("solution", ""),
                "domain": domain_str,
                "difficulty": rec.get("difficulty", 0),
                "source": rec.get("source", ""),
                "problem_type": rec.get("problem_type", ""),
                "test_category": rec.get("test_category", ""),
            })
        return tasks

    def build_prompt(self, task: dict, env_info: dict, phase: str) -> str:
        prompt = _PROMPT_TEMPLATE.format(problem=task["problem"])
        prefixes = self.config.get("prompt_prefix") or {}
        prefix = prefixes.get(phase) or prefixes.get("default")
        if prefix and not prompt.startswith(prefix):
            return f"{prefix}\n\n{prompt}"
        return prompt

    @staticmethod
    def _resolved_config_value(value) -> str:
        value = str(value or "").strip()
        if value.startswith("${") or value.startswith("$"):
            return ""
        return value

    def verify(self, task: dict, env_info: dict, trial_dir: Path, agent_result: dict | None = None) -> dict:
        cfg = self.config
        api_base = (
            self._resolved_config_value(cfg.get("eval_api_base"))
            or os.environ.get("JUDGE_API_BASE", "")
        )
        api_key = (
            self._resolved_config_value(cfg.get("eval_api_key"))
            or os.environ.get("JUDGE_API_KEY", "")
            or os.environ.get("EVALUATION_API_KEY", "")
        )
        model = (
            self._resolved_config_value(cfg.get("eval_model_name"))
            or os.environ.get("JUDGE_MODEL", "gpt-4o")
        )
        verify_timeout = float(
            self._resolved_config_value(cfg.get("verify_timeout"))
            or os.environ.get("EVALUATION_TIMEOUT", "240")
        )
        verify_max_retries = int(
            self._resolved_config_value(cfg.get("verify_max_retries"))
            or os.environ.get("EVALUATION_MAX_RETRIES", "3")
        )
        mode = cfg.get("verify_mode", "exact")
        if mode == "llm" and not (api_key or api_base):
            raise ValueError(
                "reasoning verify_mode=llm requires eval_api_key/eval_api_base "
                "or JUDGE_API_KEY/JUDGE_API_BASE in the environment"
            )
        agent_output = (agent_result or {}).get("response", "")
        result = verify_answer(
            task,
            agent_output,
            mode=mode,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout=verify_timeout,
            max_retries=verify_max_retries,
        )
        result.update({
            "task_id": task["task_id"],
            "difficulty": task.get("difficulty", 0),
            "domain": task.get("domain", ""),
            "test_category": task.get("test_category", ""),
            "source": task.get("source", ""),
        })
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "eval_details.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        return result

    def aggregate_metrics(self, results: list[dict]) -> dict:
        total = len(results)
        if not total:
            return {}
        correct = sum(1 for item in results if item.get("correct"))
        return {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4),
        }
