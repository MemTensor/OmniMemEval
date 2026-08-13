"""SWE-bench Verified domain adapter.

Docker-based: loads pre-built instance images from tar, runs agent via
tmux wrapper, verifies with official swebench eval harness.
"""

import json
import logging
import os
import shlex
import socket
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from typing import Any

from agentbench.domains.base import DomainAdapter
from agentbench.utils.docker import (
    ensure_image, setup_container_tmux, create_wrapper_script,
    remove_container, docker_exec,
)

try:
    import docker
except ImportError:
    docker = None
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
    from swebench.harness.constants import (
        DOCKER_PATCH, DOCKER_WORKDIR, DOCKER_USER,
        KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION,
    )
    from swebench.harness.docker_utils import copy_to_container, exec_run_with_timeout
    from swebench.harness.grading import get_eval_report
except ImportError:
    make_test_spec = None
    TestSpec = Any
    DOCKER_PATCH = "/tmp/patch.diff"
    DOCKER_WORKDIR = "/testbed"
    DOCKER_USER = "root"
    KEY_INSTANCE_ID = "instance_id"
    KEY_MODEL = "model_name_or_path"
    KEY_PREDICTION = "model_patch"
    copy_to_container = None
    exec_run_with_timeout = None
    get_eval_report = None

log = logging.getLogger("agentbench")
_ACTIVE_CONFIG = {}

# Git apply strategies (same as official swebench harness)
GIT_APPLY_CMDS = [
    "git apply -v",
    "git apply -v --3way",
    "patch --batch --fuzz=5 -p1 -i",
]

MEMOS_OUTCOME_ENV = "OMNIMEMEVAL_MEMOS_RECORD_OUTCOMES"
MEMOS_OUTCOME_DIR_ENV = "OMNIMEMEVAL_MEMOS_OUTCOME_DIR"
SWE_SETUP_PARALLEL_ENV = "SWE_SETUP_PARALLEL"
_setup_semaphore_lock = threading.Lock()
_setup_semaphore: threading.BoundedSemaphore | None = None
_setup_semaphore_limit: int | None = None
_network_timeout_lock = threading.Lock()
_network_timeout_installed = False


def _cfg():
    return _ACTIVE_CONFIG


def _network_timeout() -> float:
    raw = os.environ.get("SWE_NETWORK_TIMEOUT") or _cfg().get("network_timeout", 300)
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        log.warning("Invalid SWE network timeout %r; using 300s", raw)
        return 300.0


def _install_network_timeouts() -> None:
    """Bound network calls made inside the upstream SWE-bench harness."""
    global _network_timeout_installed

    with _network_timeout_lock:
        if _network_timeout_installed:
            return

        timeout = _network_timeout()
        socket.setdefaulttimeout(timeout)

        try:
            import requests
        except ImportError:
            requests = None

        if requests is not None:
            original_request = requests.sessions.Session.request

            def request_with_default_timeout(self, method, url, **kwargs):
                if kwargs.get("timeout") is None:
                    kwargs["timeout"] = timeout
                return original_request(self, method, url, **kwargs)

            requests.sessions.Session.request = request_with_default_timeout

        _network_timeout_installed = True
        log.info("SWE network default timeout set to %.0fs", timeout)


def _docker_bin() -> str:
    return os.environ.get("OMNIMEMEVAL_DOCKER_BIN", "docker")


def _require_swe_dependencies() -> None:
    missing = []
    if docker is None:
        missing.append("docker")
    if pd is None:
        missing.append("pandas")
    if make_test_spec is None or copy_to_container is None or get_eval_report is None:
        missing.append("swebench")
    if missing:
        raise RuntimeError(
            "software_engineering domain requires missing dependencies: "
            + ", ".join(sorted(set(missing)))
        )


def _setup_parallel_limit() -> int:
    raw = os.environ.get(SWE_SETUP_PARALLEL_ENV, "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        log.warning("Invalid %s=%r; SWE setup concurrency is unlimited", SWE_SETUP_PARALLEL_ENV, raw)
        return 0


def _get_setup_semaphore() -> tuple[threading.BoundedSemaphore | None, int]:
    """Return a process-local semaphore for Docker-heavy SWE setup work."""
    global _setup_semaphore, _setup_semaphore_limit

    limit = _setup_parallel_limit()
    if limit <= 0:
        return None, 0

    with _setup_semaphore_lock:
        if _setup_semaphore is None or _setup_semaphore_limit != limit:
            _setup_semaphore = threading.BoundedSemaphore(limit)
            _setup_semaphore_limit = limit
    return _setup_semaphore, limit


@contextmanager
def _swe_setup_slot(instance_id: str, phase: str):
    setup_semaphore, setup_limit = _get_setup_semaphore()
    if setup_semaphore is None:
        yield
        return

    log.info(
        f"[{instance_id}] Waiting for SWE setup slot "
        f"({SWE_SETUP_PARALLEL_ENV}={setup_limit}, phase={phase})"
    )
    setup_semaphore.acquire()
    log.info(f"[{instance_id}] Acquired SWE setup slot (phase={phase})")
    try:
        yield
    finally:
        setup_semaphore.release()
        log.info(f"[{instance_id}] Released SWE setup slot (phase={phase})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _instance_id_to_tar(instance_id: str, tar_dir: str) -> Path | None:
    """Map instance_id to tar file path.

    Naming: astropy__astropy-12907 → sweb.eval.x86_64.astropy_1776_astropy-12907.tar
    Convention: replace '__' with '_1776_' in instance_id.
    """
    id_docker = instance_id.replace("__", "_1776_").lower()
    tar_file = Path(tar_dir) / f"sweb.eval.x86_64.{id_docker}.tar"
    if tar_file.exists():
        return tar_file
    return None


def _get_docker_client():
    """Get a cached Docker client."""
    _require_swe_dependencies()
    if not hasattr(_get_docker_client, "_client"):
        _get_docker_client._client = docker.from_env(timeout=600)
    return _get_docker_client._client


def _docker_exec_in_tmux(container_name: str, cmd: str, timeout: int = 30):
    """Execute a command inside the task tmux session."""
    tmux_cmd = f"tmux send-keys -t main {shlex.quote(cmd)} Enter"
    result = docker_exec(container_name, tmux_cmd, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            "tmux command failed: "
            f"{result.stderr.strip() or result.stdout.strip() or result.returncode}"
        )
    time.sleep(3)


def _docker_cli(*args: str, timeout: int = 120, check: bool = True,
                retries: int = 0) -> subprocess.CompletedProcess:
    """Run Docker CLI with small retry support for transient daemon resets."""
    docker_bin = _docker_bin()
    last = None
    for attempt in range(retries + 1):
        last = subprocess.run(
            [docker_bin, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if last.returncode == 0 or not check:
            return last
        if attempt < retries:
            stderr = (last.stderr or last.stdout or "").strip()
            log.warning(
                "docker %s failed (attempt %d/%d): %s",
                " ".join(args[:2]),
                attempt + 1,
                retries + 1,
                stderr,
            )
            time.sleep(min(2 * (attempt + 1), 10))

    if check and last is not None:
        stderr = (last.stderr or last.stdout or "").strip()
        raise RuntimeError(f"docker {' '.join(args)} failed: {stderr}")
    return last


def _create_started_container(container_name: str, image: str) -> str:
    """Create and start a SWE-bench container without Docker SDK."""
    _docker_cli("rm", "-f", container_name, timeout=30, check=False)
    created = False
    try:
        _docker_cli(
            "create",
            "--name", container_name,
            "--user", DOCKER_USER,
            image,
            "tail", "-f", "/dev/null",
            timeout=120,
            retries=2,
        )
        created = True
        _docker_cli("start", container_name, timeout=120, retries=2)
        inspect = _docker_cli(
            "inspect",
            "--format", "{{.Id}}",
            container_name,
            timeout=30,
            retries=1,
        )
        return inspect.stdout.strip()[:12]
    except Exception:
        if created:
            remove_container(container_name)
        raise


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _outcome_dir() -> Path:
    override = os.environ.get(MEMOS_OUTCOME_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".openclaw" / "memos-plugin" / "agentbench-outcomes"


def _changed_files_from_patch(patch_text: str) -> list[str]:
    files = []
    seen = set()
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path not in seen:
            files.append(path)
            seen.add(path)
    return files


def _patch_stats(patch_text: str) -> dict:
    added = sum(1 for line in patch_text.splitlines()
                if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in patch_text.splitlines()
                  if line.startswith("-") and not line.startswith("---"))
    files = _changed_files_from_patch(patch_text)
    return {
        "files": files[:20],
        "file_count": len(files),
        "added_lines": added,
        "deleted_lines": deleted,
        "empty": not bool(patch_text.strip()),
    }


def _load_eval_report(verifier_dir: Path, instance_id: str) -> dict:
    report_path = verifier_dir / "eval_report.json"
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return {}
    return report.get(instance_id, {}) if isinstance(report, dict) else {}


def _flatten_failed_tests(report_entry: dict) -> list[str]:
    failed = []
    tests_status = report_entry.get("tests_status", {})
    if not isinstance(tests_status, dict):
        return failed
    for group, details in tests_status.items():
        if not isinstance(details, dict):
            continue
        for test in details.get("failure", []) or []:
            failed.append(f"{group}:{test}")
    return failed


def _outcome_status(verifier_result: dict) -> str:
    if verifier_result.get("reward", 0) > 0 or verifier_result.get("resolved") is True:
        return "resolved"
    error = verifier_result.get("error")
    if error == "No patch generated":
        return "no_patch"
    if error == "Patch apply failed":
        return "patch_apply_failed"
    if verifier_result.get("resolved") is False:
        return "resolved_false"
    if error:
        return "verifier_error"
    return "failed"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_verification(test_spec: TestSpec, container_name: str,
                      pred: dict, trial_dir: Path, timeout: int) -> dict:
    """Run official swebench eval: git diff → apply patch → run tests → grade."""
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    instance_id = test_spec.instance_id

    client = _get_docker_client()
    container = client.containers.get(container_name)

    try:
        # 1. Get agent's diff
        git_diff = container.exec_run(
            "git -c core.fileMode=false diff",
            workdir=DOCKER_WORKDIR, user=DOCKER_USER,
        )
        agent_patch = git_diff.output.decode("utf-8", errors="replace").strip()
        (verifier_dir / "agent_patch.diff").write_text(agent_patch)
        pred[KEY_PREDICTION] = agent_patch

        if not agent_patch:
            log.warning(f"  Agent produced no changes")
            return {"reward": 0.0, "error": "No patch generated"}

        # 2. Reset and re-apply patch
        container.exec_run("git checkout -- .", workdir=DOCKER_WORKDIR, user=DOCKER_USER)

        patch_file = verifier_dir / "patch.diff"
        patch_file.write_text(agent_patch)
        copy_to_container(container, patch_file, PurePosixPath(DOCKER_PATCH))

        applied = False
        for git_apply_cmd in GIT_APPLY_CMDS:
            val = container.exec_run(
                f"{git_apply_cmd} {DOCKER_PATCH}",
                workdir=DOCKER_WORKDIR, user=DOCKER_USER,
            )
            if val.exit_code == 0:
                applied = True
                log.info(f"  Patch applied with: {git_apply_cmd}")
                break
        if not applied:
            log.warning(f"  Failed to apply patch")
            return {"reward": 0.0, "error": "Patch apply failed"}

        # 3. Run eval script
        eval_file = verifier_dir / "eval.sh"
        eval_file.write_text(test_spec.eval_script)
        copy_to_container(container, eval_file, PurePosixPath("/eval.sh"))

        log.info(f"  Running eval script (timeout={timeout}s)...")
        test_output, timed_out, total_runtime = exec_run_with_timeout(
            container, "/bin/bash /eval.sh", timeout
        )
        log.info(f"  Test runtime: {total_runtime:.1f}s")

        test_output_path = verifier_dir / "test_output.txt"
        test_output_path.write_text(test_output)

        if timed_out:
            return {"reward": 0.0, "error": f"Tests timed out ({timeout}s)"}

        # 4. Grade with official get_eval_report()
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=str(test_output_path),
            include_tests_status=True,
        )
        resolved = report.get(instance_id, {}).get("resolved", False)

        (verifier_dir / "eval_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        reward = 1.0 if resolved else 0.0
        (verifier_dir / "reward.txt").write_text(str(reward))
        return {"reward": reward, "resolved": resolved}

    except Exception as e:
        log.error(f"  Verification error: {e}\n{traceback.format_exc()}")
        return {"reward": 0.0, "error": str(e)}


# ---------------------------------------------------------------------------
# SWEBench Adapter
# ---------------------------------------------------------------------------

class SWEBenchAdapter(DomainAdapter):
    name = "software_engineering"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        global _ACTIVE_CONFIG
        _ACTIVE_CONFIG = self.config
        self._dataset = None
        self._test_specs = {}

    def _load_dataset(self):
        _require_swe_dependencies()
        if self._dataset is None:
            self._dataset = pd.read_parquet(_cfg()["parquet_file"])
        return self._dataset

    def _get_test_spec(self, instance_id: str) -> TestSpec:
        if instance_id not in self._test_specs:
            _install_network_timeouts()
            df = self._load_dataset()
            matched = df[df["instance_id"] == instance_id]
            if matched.empty:
                raise ValueError(f"instance_id '{instance_id}' not found in dataset")
            row = matched.iloc[0].to_dict()
            # Parse JSON string fields for swebench compatibility
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                if isinstance(row.get(key), str):
                    row[key] = json.loads(row[key])
            self._test_specs[instance_id] = make_test_spec(row)
        return self._test_specs[instance_id]

    def _image_name(self, instance_id: str) -> str:
        """Get the Docker image name for loading from tar.

        Tar images use naming: swebench/sweb.eval.x86_64.{id_docker}:latest
        """
        id_docker = instance_id.replace("__", "_1776_").lower()
        return f"swebench/sweb.eval.x86_64.{id_docker}:latest"

    def _load_split_ids(self, split: str) -> list[str] | None:
        """Load instance IDs from split_file if configured."""
        split_file = _cfg().get("split_file")
        if not split_file:
            return None
        with open(split_file) as f:
            raw = json.load(f)
        if split == "all":
            train = raw.get("train", [])
            test = raw.get("test", [])
            if train or test:
                return train + test
            if "clusters" in raw:
                seen = set()
                ids = []
                for cluster in raw["clusters"].values():
                    for part in ("train", "test"):
                        for iid in cluster.get(part, []):
                            if iid not in seen:
                                seen.add(iid)
                                ids.append(iid)
                return ids
            return []
        if split in raw and isinstance(raw[split], list):
            return raw[split]
        if "clusters" in raw:
            seen = set()
            ids = []
            for cluster in raw["clusters"].values():
                for iid in cluster.get(split, []):
                    if iid not in seen:
                        seen.add(iid)
                        ids.append(iid)
            return ids if ids else None
        return None

    def load_tasks(self, args) -> list[dict]:
        df = self._load_dataset()
        # 1. Split narrows the pool
        if args.split:
            split_ids = self._load_split_ids(args.split)
            if split_ids is not None:
                missing = set(split_ids) - set(df["instance_id"])
                if missing:
                    log.warning(f"split_file has {len(missing)} IDs not in parquet: {list(missing)[:5]}")
                df = df[df["instance_id"].isin(split_ids)]
            elif args.split.isdigit():
                df = df.head(int(args.split))

        # 2. Task filters within the pool
        if args.task:
            task_ids = set(t.strip() for t in args.task.split(","))
            df = df[df["instance_id"].isin(task_ids)]
            missing = task_ids - set(df["instance_id"])
            if missing:
                raise ValueError(f"No matching software_engineering task(s): {sorted(missing)}")
        return [{"name": row["instance_id"],
                 "problem_statement": row["problem_statement"],
                 "repo": row["repo"],
                 "hints_text": row.get("hints_text", "")}
                for _, row in df.iterrows()]

    def pre_task_trials(self, task: dict):
        """Ensure instance image exists: local → tar → pull."""
        instance_id = task["name"]
        image = self._image_name(instance_id)
        tar_dir = _cfg().get("tar_dir")
        tar_file = _instance_id_to_tar(instance_id, tar_dir) if tar_dir else None
        with _swe_setup_slot(instance_id, "image"):
            ensure_image(image, tar_file=tar_file)

    def post_task_trials(self, task: dict):
        # Keep SWE-bench images cached locally. They are large, and repeated
        # experiments should not pay the registry pull cost after every task.
        pass

    def setup(self, task: dict, agent_name: str, trial: int) -> dict:
        instance_id = task["name"]
        image = self._image_name(instance_id)
        job_tag = task.get("_job_dir", "")
        if job_tag:
            job_tag = Path(job_tag).name
        container_name = f"swebench-{job_tag}-{instance_id}-t{trial}" if job_tag else f"swebench-{agent_name}-{instance_id}-t{trial}"

        # Start container via Docker SDK. This block is intentionally gated by
        # the setup semaphore because create/start/cp/exec bursts can reset the
        # Docker API connection under high SWE parallelism.
        container_created = False
        try:
            with _swe_setup_slot(instance_id, "container"):
                test_spec = self._get_test_spec(instance_id)
                short_id = _create_started_container(container_name, image)
                container_created = True
                log.info(f"[{instance_id}] Container started: {short_id}")

                # Setup tmux
                log.info(f"[{instance_id}] Setting up tmux...")
                setup_container_tmux(container_name)
                wrapper_path = create_wrapper_script(instance_id, container_name)

                # Keep commands non-interactive.  In a detached tmux pane Git
                # may still start `less`; the agent then sees a permanently
                # running command and can loop on q/C-c until the model API's
                # repeated-tool-call guard aborts the trial.
                _docker_exec_in_tmux(container_name,
                                     "export GIT_PAGER=cat PAGER=cat LESS=-FRX GIT_TERMINAL_PROMPT=0; "
                                     "source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed")
        except Exception:
            if container_created:
                log.warning(f"[{instance_id}] setup failed after container creation; removing {container_name}")
                remove_container(container_name)
            raise

        return {
            "container_name": container_name,
            "wrapper_path": wrapper_path,
            "test_spec": test_spec,
            "pred": {
                KEY_INSTANCE_ID: instance_id,
                KEY_MODEL: agent_name,
                KEY_PREDICTION: "",
            },
        }

    def get_agent_timeout(self, task: dict, env_info: dict) -> int:
        return int(_cfg().get("agent_timeout", 1800))

    _prompt_template = (Path(__file__).parent / "prompt.md").read_text()

    def build_prompt(self, task: dict, env_info: dict, phase: str = "test") -> str:
        prompt = self._prompt_template.format(
            wrapper_path=env_info["wrapper_path"],
            timeout_min=self.get_agent_timeout(task, env_info) // 60,
            problem=task["problem_statement"],
            repo=task["repo"],
        )
        hints = task.get("hints_text", "")
        if hints:
            prompt += f"\n## Hints\n\n{hints}\n"
        return prompt

    def verify(self, task: dict, env_info: dict, trial_dir: Path,
               agent_result: dict | None = None) -> dict:
        return _run_verification(
            env_info["test_spec"],
            env_info["container_name"],
            env_info["pred"],
            trial_dir,
            int(_cfg().get("verify_timeout", 1800)),
        )

    def record_agent_outcome(self, task: dict, env_info: dict, trial_dir: Path,
                             agent_result: dict, verifier_result: dict):
        """Persist reward-aware SWE outcome facts for memos retrieval.

        This is intentionally opt-in so benchmark test runs can freeze memory
        and avoid leaking verifier-only results across tasks.
        """
        if not _truthy_env(MEMOS_OUTCOME_ENV, default=False):
            return

        verifier_dir = trial_dir / "verifier"
        patch_path = verifier_dir / "agent_patch.diff"
        patch_text = patch_path.read_text(errors="replace") if patch_path.exists() else ""
        report_entry = _load_eval_report(verifier_dir, task["name"])
        status = _outcome_status(verifier_result)
        record = {
            "schema": "agentbench.swe_outcome.v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "domain": self.name,
            "task": task["name"],
            "repo": task.get("repo"),
            "status": status,
            "reward": verifier_result.get("reward", 0.0),
            "resolved": verifier_result.get("resolved"),
            "error": verifier_result.get("error"),
            "completion_status": agent_result.get("completion_status"),
            "turns": None,
            "elapsed_sec": agent_result.get("elapsed_sec"),
            "patch": _patch_stats(patch_text),
            "failed_tests": _flatten_failed_tests(report_entry)[:30],
            "response_summary": str(agent_result.get("response", ""))[:2000],
        }

        out_dir = _outcome_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "software_engineering.jsonl"
        with open(out_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def cleanup(self, task: dict, env_info: dict):
        wrapper_path = env_info.get("wrapper_path")
        if wrapper_path:
            try:
                os.unlink(wrapper_path)
            except OSError:
                pass
        container_name = env_info.get("container_name")
        if container_name:
            remove_container(container_name)
