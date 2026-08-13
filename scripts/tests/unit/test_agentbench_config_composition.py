import argparse
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.config import load_yaml
from agentbench.memory_lifecycle import CommandMemoryLifecycle
from agentbench.run_agent_eval import (
    _compose_agent_config,
    _default_memory_plugin_config,
    _has_existing_trial_results,
    _load_profile_config,
    _validate_config_identity,
)


def _args(agent, memory_plugin=None, profile=None, profile_config=None):
    return argparse.Namespace(
        agent=agent,
        memory_plugin=memory_plugin,
        profile=profile,
        profile_config=profile_config,
    )


def test_every_declared_memory_profile_and_lifecycle_compose(tmp_path):
    profiles_root = ROOT / "configs" / "agentbench" / "profiles"
    lifecycles_root = ROOT / "configs" / "agentbench" / "memory_plugins"

    declared_profiles = {}
    for path in sorted(profiles_root.glob("*/*.yaml")):
        profile = load_yaml(path)
        plugin = profile.get("memory_plugin")
        if not plugin or plugin == "none":
            continue
        key = (str(profile["agent"]), str(plugin))
        assert key not in declared_profiles, f"duplicate memory profile for {key}: {path}"
        declared_profiles[key] = path

    declared_lifecycles = {}
    for path in sorted(lifecycles_root.glob("*/lifecycle/*.yaml")):
        lifecycle = load_yaml(path)
        key = (str(lifecycle["agent"]), str(lifecycle["plugin"]))
        assert key not in declared_lifecycles, f"duplicate memory lifecycle for {key}: {path}"
        declared_lifecycles[key] = path
        CommandMemoryLifecycle(
            config=lifecycle,
            project_dir=ROOT,
            run_dir=tmp_path / f"{key[0]}-{key[1]}",
            run_id="configuration-matrix",
            version="unit-test",
        )

    assert declared_profiles.keys() == declared_lifecycles.keys()
    for (agent, plugin), lifecycle_path in declared_lifecycles.items():
        assert lifecycle_path == _default_memory_plugin_config(agent, plugin)
        profile_path, profile, profile_name = _load_profile_config(
            _args(agent, memory_plugin=plugin),
            load_yaml(lifecycle_path),
        )
        assert profile_path == declared_profiles[(agent, plugin)]
        assert profile_name == plugin
        assert profile["agent"] == agent
        assert profile["memory_plugin"] == plugin


def test_non_memos_profiles_leave_product_selection_to_user_configuration():
    for path in sorted((ROOT / "configs" / "agentbench" / "profiles").glob("*/*.yaml")):
        profile = load_yaml(path)
        if profile.get("memory_plugin") in {None, "none", "memos"}:
            continue
        patch = profile.get("agent_patch") or {}
        assert "memory" not in patch, path
        assert "openclaw_config_patch" not in patch, path
        assert "hermes_config_patch" not in patch, path


def test_product_lifecycles_fail_closed_around_settle_and_restore():
    everos = load_yaml(_default_memory_plugin_config("openclaw", "everos"))
    everos_backup = everos["commands"]["backup"]
    assert "memory/flush" in everos_backup
    assert "everos cascade" in everos_backup
    assert "everos cascade --root \"$EVEROS_ROOT\" sync || true" not in everos_backup

    supermemory = load_yaml(
        _default_memory_plugin_config("openclaw", "supermemory")
    )
    supermemory_restore = supermemory["commands"]["restore"]
    assert supermemory_restore.index(" import ") < supermemory_restore.index(
        " wait 1800"
    )
    assert "reusable expertise" in supermemory["env"]["SUPERMEMORY_ENTITY_CONTEXT"]

    openviking = load_yaml(
        _default_memory_plugin_config("openclaw", "openviking")
    )
    assert "did not settle within 1800s" in openviking["commands"]["wait_settle"]
    assert "OpenViking task failure" in openviking["commands"]["wait_settle"]
    for stage in ("clear", "restore", "cleanup"):
        assert "curl -fsS" in openviking["commands"][stage]

    hermes_openviking = load_yaml(
        _default_memory_plugin_config("hermes", "openviking")
    )
    assert "/api/v1/tasks?limit=200" in hermes_openviking["commands"]["wait_settle"]
    assert "OpenViking task failure" in hermes_openviking["commands"]["wait_settle"]
    assert 'server["auth_mode"] = "trusted"' in hermes_openviking["commands"][
        "prepare_global_snapshot"
    ]
    assert "OPENVIKING_API_KEY={root_api_key}" in hermes_openviking["commands"][
        "prepare_global_snapshot"
    ]
    assert 'headers["X-OpenViking-Account"]' in hermes_openviking["commands"][
        "wait_settle"
    ]
    backup = hermes_openviking["commands"]["backup"]
    assert backup.index("docker stop") < backup.index("tar ")

    hindsight = load_yaml(
        _default_memory_plugin_config("openclaw", "hindsight")
    )
    prepare = hindsight["commands"]["prepare_global_snapshot"]
    assert 'plugin_config["dynamicBankId"] = False' in prepare
    assert 'plugin_config["bankId"] = bank' in prepare

    openclaw_mem0 = load_yaml(
        _default_memory_plugin_config("openclaw", "mem0")
    )
    assert "mode test" in openclaw_mem0["commands"]["set_mode_test"]
    assert "mem0_openclaw_ctl.py" in openclaw_mem0["commands"]["backup"]

    hermes_mem0 = load_yaml(
        _default_memory_plugin_config("hermes", "mem0")
    )
    assert "HERMES_MEM0_QDRANT_PATH" in hermes_mem0["env"]
    assert "hermes_mem0_ctl.py" in hermes_mem0["commands"]["restore"]


@pytest.mark.parametrize("agent", ["openclaw", "hermes"])
def test_memos_resolves_runtime_specific_profile_and_lifecycle(agent):
    lifecycle_path = _default_memory_plugin_config(agent, "memos")
    lifecycle = load_yaml(lifecycle_path)
    profile_path, profile, profile_name = _load_profile_config(
        _args(agent, memory_plugin="memos"), lifecycle
    )

    assert lifecycle_path == ROOT / "configs" / "agentbench" / "memory_plugins" / "memos" / "lifecycle" / f"{agent}.yaml"
    assert lifecycle["plugin"] == "memos"
    assert lifecycle["agent"] == agent
    assert lifecycle["env"]["MEMOS_HOME"] == lifecycle["env"]["MEMOS_PLUGIN_HOME"]
    assert lifecycle["env"]["MEMOS_DB"].startswith(lifecycle["env"]["MEMOS_HOME"])
    if agent == "hermes":
        assert "MEMOS_HERMES_RPC_TIMEOUT_SECONDS" not in lifecycle["env"]
    assert profile_path == ROOT / "configs" / "agentbench" / "profiles" / agent / "memos.yaml"
    assert profile_name == "memos"
    assert profile["agent"] == agent


def test_explicit_lifecycle_infers_profile_from_plugin():
    lifecycle = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    path, _, name = _load_profile_config(_args("hermes"), lifecycle)

    assert name == "memos"
    assert path.name == "memos.yaml"
    assert path.parent.name == "hermes"


def test_rejects_runtime_mismatch_before_execution():
    lifecycle = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    with pytest.raises(SystemExit, match="targets agent 'hermes'"):
        _validate_config_identity(
            lifecycle,
            kind="memory_lifecycle",
            agent="openclaw",
            plugin="memos",
        )


def test_profiles_keep_runtime_specific_patches_separate():
    openclaw = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "openclaw" / "memos.yaml")
    hermes = load_yaml(ROOT / "configs" / "agentbench" / "profiles" / "hermes" / "memos.yaml")
    openclaw_config = _compose_agent_config({}, openclaw)
    hermes_config = _compose_agent_config({}, hermes)

    assert "openclaw_config_patch" in openclaw_config
    assert "hermes_config_patch" not in openclaw_config
    assert "hermes_config_patch" in hermes_config
    assert "openclaw_config_patch" not in hermes_config
    assert hermes_config["runtime"]["home_links"] == ["memos-plugin"]
    assert hermes_config["runtime"]["train_memory_provider"] == "omnimemeval_memos"
    assert hermes_config["runtime"]["verify_memos_capture"] is True
    assert hermes_config["memory"]["memory_enabled"] is False
    assert hermes_config["memory"]["user_profile_enabled"] is False
    compression = hermes_config["hermes_config_patch"]["compression"]
    assert compression == {
        "enabled": True,
        "threshold": 0.35,
        "target_ratio": 0.2,
        "protect_last_n": 8,
        "protect_first_n": 1,
    }
    assert hermes_config["hermes_config_patch"]["auxiliary"]["compression"][
        "timeout"
    ] == 300


def test_knowledge_work_uses_domain_specific_evaluator_timeout():
    config = load_yaml(
        ROOT / "configs" / "agentbench" / "domains" / "knowledge_work.yaml"
    )

    assert config["eval_timeout"] == 360


def test_memos_lifecycles_use_private_sqlite_only_backups():
    required = {
        "prepare_global_snapshot",
        "clear",
        "wait_settle",
        "backup",
        "restore",
        "cleanup",
    }
    for agent in ("openclaw", "hermes"):
        config = load_yaml(_default_memory_plugin_config(agent, "memos"))
        assert required <= config["commands"].keys()
        assert config["backup_file_template"].endswith(".sqlite3")
        assert config["global_backup_file_template"].endswith(".sqlite3")
        assert config["resume_file_template"].endswith("-resume.sqlite3")
        assert config["backup_dir"] == "@run_dir@/memory_backups"
        assert "checkpoint" in config["commands"]
        assert "tar " not in config["commands"]["backup"]
        assert ".auth.json" not in config["commands"]["backup"]
        assert "config.yaml" not in config["commands"]["backup"]
        assert "pgrep" not in "\n".join(config["commands"].values())


@pytest.mark.parametrize("agent", ["openclaw", "hermes"])
def test_memos_settle_gates_distinct_sessions_and_reconciles_rollover_episodes(agent):
    config = load_yaml(_default_memory_plugin_config(agent, "memos"))
    command = config["commands"]["wait_settle"]
    helper = ""
    if agent == "hermes":
        helper = (ROOT / "scripts" / "agentbench" / "hermes_memos_drain.sh").read_text(
            encoding="utf-8"
        )
    drain_command = command + helper

    # A long trial may roll over into multiple episodes.  The lifecycle must
    # count stable trial sessions, then require every resulting episode to be
    # closed after a private reconciliation bridge drains the pipeline.
    assert "count(DISTINCT session_id)" in command
    assert 'total" -ge "$expected' in command
    assert 'closed" -eq "$total' in command
    assert "--no-viewer" in drain_command
    assert "settle-reconcile.log" in drain_command
    reconcile_marker = (
        'node "$bridge"'
        if agent == "openclaw"
        else "hermes_memos_drain.sh"
    )
    reconcile = command.index(reconcile_marker)
    assert command.index("session_gate ||") < reconcile
    assert reconcile < command.index("settled_gate ||")
    assert "total\" -eq \"$expected" not in command


def test_openclaw_memos_lifecycle_tracks_shared_runtime_daemon_and_all_bridge_entries():
    config = load_yaml(_default_memory_plugin_config("openclaw", "memos"))
    for stage in ("clear", "wait_settle", "restore", "cleanup"):
        command = config["commands"][stage]
        assert "runtime-daemon.js" in command
        assert "runtime-daemon.ts" in command
        assert "runtime-stdio-proxy.js" in command
        assert "bridge.mjs" in command


def test_openclaw_memos_settle_rejects_duplicate_live_skills_per_policy():
    config = load_yaml(_default_memory_plugin_config("openclaw", "memos"))
    command = config["commands"]["wait_settle"]

    assert "json_each(s.source_policies_json)" in command
    assert "HAVING count(*) > 1" in command
    assert "duplicate non-archived skills" in command


def test_openclaw_memos_settle_waits_for_persistent_evolution_queue_before_shutdown():
    config = load_yaml(_default_memory_plugin_config("openclaw", "memos"))
    command = config["commands"]["wait_settle"]

    assert "evolution_idle" in command
    assert "status IN ('queued','leased','failed')" in command
    assert 'if [ -n "$(runtime_pids)" ]; then' in command
    assert '[ -n "$(runtime_pids)" ] || break' in command
    assert "background_idle" in command
    assert "status IN ('pending','in_progress')" in command
    wait_for_idle = command.index("background_idle && break")
    terminate_runtime = command.index("for pid in $(runtime_pids); do kill -TERM")
    assert wait_for_idle < terminate_runtime
    assert "evolution queue did not become idle" in command


def test_openclaw_memos_settle_rejects_terminal_background_failures():
    config = load_yaml(_default_memory_plugin_config("openclaw", "memos"))
    command = config["commands"]["wait_settle"]

    assert "status='dead_letter'" in command
    assert "status='failed'" in command
    assert "terminal background failures" in command


def test_hermes_memos_settle_drains_all_background_queues_before_backup():
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    command = config["commands"]["wait_settle"]
    helper = (ROOT / "scripts" / "agentbench" / "hermes_memos_drain.sh").read_text(
        encoding="utf-8"
    )
    implementation = command + helper

    assert "evolution_jobs" in implementation
    assert "status IN ('queued','leased','failed')" in implementation
    assert "status IN ('pending','in_progress')" in implementation
    assert "status='dead_letter'" in implementation
    assert "terminal background failures" in implementation
    assert "hermes_memos_drain.sh" in command
    assert 'grep -Fxq "MEMOS_PLUGIN_HOME=$plugin"' in helper
    assert '"${exe##*/}" = "node"' in helper
    assert helper.index("stop_existing_runtime") < helper.index(
        'node "$bridge" --agent=hermes --no-viewer'
    )
    wait_for_idle = command.index('if settled_gate && [ -z "$(runtime_pids)" ]; then')
    terminate_runtime = command.index("for pid in $(runtime_pids); do kill -TERM")
    assert wait_for_idle < terminate_runtime
    assert "Hermes MemOS background work did not settle before shutdown" in command


def test_hermes_memos_prepares_run_scoped_llm_failure_policy():
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    prepare = config["commands"]["prepare_global_snapshot"]

    assert config["env"]["MEMOS_EVAL_LLM_MAX_RETRIES"] == "0"
    assert 'slot["maxRetries"] = max_retries' in prepare
    assert 'slot["fallbackToHost"] = False' in prepare
    assert 'for name in ("llm", "skillEvolver", "l3Llm")' in prepare


def test_hermes_memos_finalizes_test_pipeline_before_checkpoint_and_cleanup():
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))

    assert "finalize" in config["commands"]
    assert "hermes_memos_drain.sh" in config["commands"]["finalize"]


def test_hermes_memos_resume_checkpoint_keeps_the_train_baseline():
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    checkpoint = config["commands"]["checkpoint"]

    assert config["backup_file_template"] in checkpoint
    assert 'source="$train_baseline"' in checkpoint


def test_hermes_memos_drain_never_matches_lifecycle_shell_argv(tmp_path):
    plugin = tmp_path / "memos-plugin"
    bridge = plugin / "dist" / "bridge.cjs"
    db = plugin / "data" / "memos.db"
    bridge.parent.mkdir(parents=True)
    db.parent.mkdir(parents=True)
    bridge.write_text(
        "process.stdin.resume();\n"
        "const timer = setInterval(() => {}, 1000);\n"
        "process.on('SIGTERM', () => { clearInterval(timer); process.exit(0); });\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE episodes ("
            "status TEXT, trace_ids_json TEXT DEFAULT '[]', r_task REAL, "
            "meta_json TEXT DEFAULT '{}');"
            "CREATE TABLE traces (id TEXT);"
            "CREATE TABLE evolution_jobs ("
            "id TEXT, job_type TEXT, status TEXT, attempts INTEGER, "
            "max_attempts INTEGER, last_error TEXT);"
            "CREATE TABLE embedding_retry_queue ("
            "id TEXT, target_kind TEXT, status TEXT, attempts INTEGER, "
            "max_attempts INTEGER, last_error TEXT);"
        )

    env = os.environ.copy()
    env.update(
        {
            "MEMOS_PLUGIN_HOME": str(plugin),
            "MEMOS_DB": str(db),
            "MEMOS_FINALIZE_TIMEOUT": "10",
            "MEMOS_DAEMON_TERM_TIMEOUT": "2",
            "MEMOS_RECONCILE_QUIET_POLLS": "1",
        }
    )
    existing = subprocess.Popen(
        ["node", str(bridge), "--agent=hermes", "--daemon"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    helper = ROOT / "scripts" / "agentbench" / "hermes_memos_drain.sh"
    # Both ancestor shell argv strings contain the bridge signature. The
    # helper must inspect /proc/<pid>/exe and terminate only the actual Node
    # runtime, never either lifecycle shell.
    inner = f"# bridge.cjs --agent=hermes\nbash {helper}\nprintf drain-ok"
    try:
        completed = subprocess.run(
            ["sh", "-c", f"bash -c '{inner}'"],
            env=env,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        if existing.poll() is None:
            existing.terminate()
            existing.wait(timeout=5)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "drain-ok"
    assert existing.poll() is not None


def test_hermes_memos_drain_accepts_paused_empty_topic(tmp_path):
    plugin = tmp_path / "memos-plugin"
    bridge = plugin / "dist" / "bridge.cjs"
    db = plugin / "data" / "memos.db"
    bridge.parent.mkdir(parents=True)
    db.parent.mkdir(parents=True)
    bridge.write_text(
        "process.stdin.resume();\n"
        "const timer = setInterval(() => {}, 1000);\n"
        "process.on('SIGTERM', () => { clearInterval(timer); process.exit(0); });\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE episodes ("
            "status TEXT, trace_ids_json TEXT, r_task REAL, meta_json TEXT);"
            "INSERT INTO episodes VALUES ("
            "'open', '[]', NULL, "
            "'{\"topicState\":\"paused\",\"pauseReason\":\"session_closed:client\"}');"
            "CREATE TABLE traces (id TEXT);"
            "CREATE TABLE evolution_jobs ("
            "id TEXT, job_type TEXT, status TEXT, attempts INTEGER, "
            "max_attempts INTEGER, last_error TEXT);"
            "CREATE TABLE embedding_retry_queue ("
            "id TEXT, target_kind TEXT, status TEXT, attempts INTEGER, "
            "max_attempts INTEGER, last_error TEXT);"
        )

    env = os.environ.copy()
    env.update(
        {
            "MEMOS_PLUGIN_HOME": str(plugin),
            "MEMOS_DB": str(db),
            "MEMOS_FINALIZE_TIMEOUT": "2",
            "MEMOS_DAEMON_TERM_TIMEOUT": "2",
            "MEMOS_RECONCILE_QUIET_POLLS": "1",
        }
    )
    helper = ROOT / "scripts" / "agentbench" / "hermes_memos_drain.sh"
    completed = subprocess.run(
        ["bash", str(helper)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


def test_existing_trial_results_detect_only_real_trial_outputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "experiment_config.json").write_text("{}", encoding="utf-8")
    assert not _has_existing_trial_results(run_dir)

    result = run_dir / "train" / "task-1__trial_1" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}", encoding="utf-8")
    assert _has_existing_trial_results(run_dir)


def test_hermes_memos_clear_does_not_terminate_its_lifecycle_shell(tmp_path, monkeypatch):
    config = load_yaml(_default_memory_plugin_config("hermes", "memos"))
    hermes_home = tmp_path / "user-hermes"
    plugin_home = hermes_home / "memos-plugin"
    (plugin_home / "dist").mkdir(parents=True)
    (plugin_home / "dist" / "bridge.cjs").write_text("", encoding="utf-8")
    (plugin_home / "config.yaml").write_text("version: 1\n", encoding="utf-8")
    (hermes_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    config["backup_dir"] = str(tmp_path / "backups")
    config["env"].update({
        "MEMOS_DAEMON_TERM_TIMEOUT": "1",
        "MEMOS_START_DAEMON": "0",
        "MEMOS_PREWARM": "0",
        "MEMOS_FINALIZE_TIMEOUT": "0",
    })
    run_dir = tmp_path / "run"
    lifecycle = CommandMemoryLifecycle(
        config=config,
        project_dir=ROOT,
        run_dir=run_dir,
        run_id="clear-regression",
        version="test",
    )

    lifecycle.activate_runtime_env()
    try:
        lifecycle.validate("reasoning")
        snapshot = lifecycle.prepare_global_snapshot("reasoning")
        lifecycle.clear("reasoning")

        assert snapshot.stat().st_mode & 0o777 == 0o600
        assert (run_dir / "runtime" / "hermes" / "memos-plugin" / "data").is_dir()

        with pytest.raises(RuntimeError, match="stage=wait_settle"):
            lifecycle.wait_settle("reasoning", expected_trials=1)
        assert "Hermes MemOS DB is missing or empty" in lifecycle.log_file.read_text()

        db = run_dir / "runtime" / "hermes" / "memos-plugin" / "data" / "memos.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(
                "CREATE TABLE episodes (id TEXT, session_id TEXT, status TEXT, trace_ids_json TEXT);"
                "CREATE TABLE traces (id TEXT);"
                "CREATE TABLE api_logs (id TEXT);"
                "CREATE TABLE embedding_retry_queue (id TEXT, status TEXT);"
            )
            conn.commit()
        finally:
            conn.close()
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="stage=wait_settle"):
            lifecycle.wait_settle("reasoning", expected_trials=1)
        assert time.monotonic() - started < 3
        assert "capture bridges have exited but session gate is unsatisfied" in lifecycle.log_file.read_text()

        old_path = os.environ["PATH"]
        no_sqlite_bin = tmp_path / "no-sqlite-bin"
        no_sqlite_bin.mkdir()
        (no_sqlite_bin / "bash").symlink_to("/usr/bin/bash")
        monkeypatch.setenv("PATH", str(no_sqlite_bin))
        with pytest.raises(RuntimeError, match="stage=wait_settle"):
            lifecycle.wait_settle("reasoning", expected_trials=1)
        monkeypatch.setenv("PATH", old_path)
        assert "sqlite3 is required to verify" in lifecycle.log_file.read_text()
    finally:
        lifecycle.cleanup("reasoning")
        lifecycle.restore_runtime_env()
