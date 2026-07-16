"""GDPVal domain adapter.

No Docker needed. Agent creates deliverable files (Excel, PDF, Word, etc.)
in a workspace directory. Verification: LLM-based rubric scoring.

Data source: HuggingFace openai/gdpval dataset (loaded directly via datasets lib).
"""

import json
import logging
import os
import shutil
from pathlib import Path

from agentbench.domains.base import DomainAdapter
from agentbench.domains.knowledge_work.evaluate import evaluate_rubric

log = logging.getLogger("agentbench")

_PROMPT_TEMPLATE = (Path(__file__).parent / "prompt.md").read_text()
_ACTIVE_CONFIG = {}


def _cfg():
    return _ACTIVE_CONFIG


def _resolved_config_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and ("${" in value or value.startswith("$")):
        return default
    return value


def _load_dataset() -> list[dict]:
    """Load GDPVal dataset from local data/gdpval/dataset.json."""
    cfg = _cfg()
    candidates = []
    if cfg.get("dataset_file"):
        candidates.append(Path(cfg["dataset_file"]))
    if cfg.get("split_file"):
        candidates.append(Path(cfg["split_file"]).with_name("dataset.json"))

    for dataset_file in candidates:
        if not dataset_file.exists():
            continue
        with open(dataset_file) as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("rows") or rows.get("train") or []
        records = []
        for row in rows:
            rubric_json = row.get("rubric_json", "[]")
            if not isinstance(rubric_json, str):
                rubric_json = json.dumps(rubric_json, ensure_ascii=False)
            records.append({
                "task_id": row["task_id"],
                "sector": row["sector"],
                "occupation": row["occupation"],
                "prompt": row["prompt"],
                "reference_files": row.get("reference_files", []),
                "reference_file_urls": row.get("reference_file_urls", []),
                "deliverable_files": row.get("deliverable_files", []),
                "deliverable_file_urls": row.get("deliverable_file_urls", []),
                "rubric_json": rubric_json,
                "rubric_pretty": row.get("rubric_pretty", ""),
            })
        log.info("Loaded %d tasks from %s", len(records), dataset_file)
        return records

    raise FileNotFoundError(
        "GDPVal dataset.json not found. Expected dataset_file in config or "
        "dataset.json next to the split file."
    )


class GDPValAdapter(DomainAdapter):
    name = "knowledge_work"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        global _ACTIVE_CONFIG
        _ACTIVE_CONFIG = self.config
        self._dataset = None

    def _get_dataset(self) -> list[dict]:
        if self._dataset is None:
            self._dataset = _load_dataset()
        return self._dataset

    def pass_threshold(self) -> float:
        return float(_cfg().get("pass_threshold", 0.6))

    def _load_split_ids(self, split: str) -> list[str] | None:
        """Load task IDs from split file (clusters format)."""
        split_file = _cfg().get("split_file")
        if not split_file or not Path(split_file).exists():
            return None
        with open(split_file) as f:
            raw = json.load(f)
        if split == "all":
            seen = set()
            ids = []
            for cluster in raw.get("clusters", {}).values():
                for part in ("train", "test"):
                    for tid in cluster.get(part, []):
                        if tid not in seen:
                            seen.add(tid)
                            ids.append(tid)
            return ids
        if split in raw and isinstance(raw[split], list):
            return raw[split]
        if "clusters" in raw:
            seen = set()
            ids = []
            for cluster in raw["clusters"].values():
                for tid in cluster.get(split, []):
                    if tid not in seen:
                        seen.add(tid)
                        ids.append(tid)
            return ids if ids else None
        return None

    def load_tasks(self, args) -> list[dict]:
        dataset = self._get_dataset()

        # 1. Split narrows the pool
        if args.split:
            split_ids = self._load_split_ids(args.split)
            if split_ids is not None:
                id_set = set(split_ids)
                dataset = [r for r in dataset if r["task_id"] in id_set]
            elif args.split.isdigit():
                dataset = dataset[:int(args.split)]
            else:
                raise ValueError(
                    f"Unknown split: {args.split}. "
                    f"Available: train, test, all (or a number for first N)")

        # 2. Task filters within the pool
        if args.task:
            ids = [t.strip() for t in args.task.split(",")]
            dataset = [r for r in dataset if r["task_id"] in ids
                       or r["task_id"][:8] in ids]
            matched = {r["task_id"] for r in dataset} | {r["task_id"][:8] for r in dataset}
            missing = set(ids) - matched
            if missing:
                raise ValueError(f"No matching knowledge_work task(s): {sorted(missing)}")

        tasks = []
        for rec in dataset:
            tid = rec["task_id"]
            short_id = tid[:8]
            tasks.append({
                "name": short_id,
                "task_id": tid,
                "sector": rec["sector"],
                "occupation": rec["occupation"],
                "prompt": rec["prompt"],
                "reference_files": rec.get("reference_files", []),
                "reference_file_urls": rec.get("reference_file_urls", []),
                "deliverable_files": rec.get("deliverable_files", []),
                "deliverable_file_urls": rec.get("deliverable_file_urls", []),
                "rubric_json": rec.get("rubric_json", "[]"),
                "rubric_pretty": rec.get("rubric_pretty", ""),
            })
        return tasks

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        """Create workspace directory and prepare reference files."""
        cfg = _cfg()
        framework_workspace = task.get("_workspace_dir")
        phase_dir = task.get("_phase_dir")
        if framework_workspace:
            workspace = Path(framework_workspace)
        elif phase_dir:
            workspace_root = Path(phase_dir) / "workspaces"
            workspace = workspace_root / f"{task['name']}_t{trial}"
        elif task.get("_job_dir"):
            workspace_root = Path(task["_job_dir"]) / "workspaces"
            workspace = workspace_root / f"{task['name']}_t{trial}"
        else:
            workspace_root = Path(cfg.get("workspace_dir", "./jobs/workspaces"))
            workspace = workspace_root / f"{task['name']}_t{trial}"
        workspace.mkdir(parents=True, exist_ok=True)

        # Copy reference files to workspace if available locally
        ref_dir = Path(cfg.get("reference_dir", ""))
        ref_paths = []
        ref_section_parts = []

        if ref_dir.exists():
            task_ref_dir = ref_dir / task["task_id"]
            if task_ref_dir.exists():
                for f in task_ref_dir.iterdir():
                    dest = workspace / f.name
                    shutil.copy2(f, dest)
                    ref_paths.append(str(dest))
                    ref_section_parts.append(f"- `{dest}` ({f.name})")

        # If no local files, download from URLs and cache
        if not ref_paths and task.get("reference_file_urls"):
            from urllib.request import urlretrieve
            cache_dir = ref_dir / task["task_id"] if ref_dir else workspace / ".ref_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for url, rel_path in zip(task["reference_file_urls"], task["reference_files"]):
                fname = Path(rel_path).name
                cached = cache_dir / fname
                dest = workspace / fname
                if not cached.exists():
                    try:
                        urlretrieve(url, str(cached))
                    except Exception as e:
                        log.warning(f"  Download error for {fname}: {e}")
                        continue
                if not dest.exists():
                    shutil.copy2(str(cached), str(dest))
                ref_paths.append(str(dest))
                ref_section_parts.append(f"- `{dest}` ({fname})")

        return {
            "workspace_dir": str(workspace),
            "reference_paths": ref_paths,
            "reference_section": "\n".join(ref_section_parts) if ref_section_parts
                                 else "No reference files for this task.",
        }

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 1800))

    def build_prompt(self, task: dict, env_info: dict, phase: str = "test") -> str:
        return _PROMPT_TEMPLATE.format(
            prompt=task["prompt"],
            reference_section=env_info.get("reference_section", "None"),
            workspace_dir=env_info["workspace_dir"],
        )

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        """Evaluate deliverables in workspace against rubric (ClawWork LLMEvaluator).

        Note: openclaw system files and reference files are filtered in evaluate.py.
        """
        cfg = _cfg()
        cfg_api_base = str(_resolved_config_value(cfg.get("eval_api_base"), "") or "").strip()
        env_api_base = os.getenv("EVALUATION_API_BASE", "").strip()
        api_base = cfg_api_base or env_api_base
        api_key = (
            str(_resolved_config_value(cfg.get("eval_api_key"), "") or "").strip()
            or os.environ.get("EVALUATION_API_KEY", "")
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
        )
        if not api_key:
            log.error(
                "  No evaluation API key: set eval_api_key in knowledge_work.yaml "
                "or EVALUATION_API_KEY / OPENROUTER_API_KEY / DASHSCOPE_API_KEY"
            )
            return {"reward": 0.0, "error": "missing_api_key"}

        workspace = Path(env_info["workspace_dir"])
        model_owner = _resolved_config_value(cfg.get("eval_model_owner"), "openai")
        model_name = _resolved_config_value(cfg.get("eval_model_name"), "gpt-4o")
        # OpenRouter uses owner/name; other OpenAI-compatible endpoints use the provider's model id.
        if api_base:
            model = model_name
        else:
            model = f"{model_owner}/{model_name}"
        meta_prompts_dir = cfg.get("meta_prompts_dir", "")
        if cfg.get("reference_dir"):
            os.environ["OMNIMEMEVAL_GDPVAL_REFERENCE_DIR"] = str(cfg["reference_dir"])
        eval_timeout = float(
            _resolved_config_value(cfg.get("eval_timeout"))
            or os.environ.get("EVALUATION_TIMEOUT")
            or 240
        )
        eval_max_retries = int(
            _resolved_config_value(cfg.get("eval_max_retries"))
            or os.environ.get("EVALUATION_MAX_RETRIES")
            or 3
        )

        result = evaluate_rubric(
            task,
            workspace,
            api_key,
            meta_prompts_dir=meta_prompts_dir,
            model=model,
            api_base=api_base,
            timeout=eval_timeout,
            max_retries=eval_max_retries,
        )

        # Save evaluation details
        verifier_dir = trial_dir / "verifier"
        verifier_dir.mkdir(parents=True, exist_ok=True)
        with open(verifier_dir / "eval_details.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    def cleanup(self, task: dict, env_info: dict):
        """Optionally clean up workspace (keep for now for debugging)."""
        pass
