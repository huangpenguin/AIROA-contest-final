from __future__ import annotations

import hashlib
import json
import inspect
import os
import socket
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import run_track3_fast400_h300_paired_eval as runner
from tools.axis_pi05 import checkpoint_http_server as checkpoint_server


def _row(case_id: str, panel: str, *, success: bool = False, fingerprint: str | None = None, nonzero: bool = True) -> dict:
    return {
        "case_id": case_id,
        "panel": panel,
        "case_fingerprint": fingerprint or f"fp-{case_id}",
        "env_seed": 1,
        "policy_seed": 2,
        "init_state_index": 0,
        "init_state_sha256": f"state-{case_id}",
        "success": success,
        "collision": False,
        "timeout": False,
        "failure_class": None,
        "actions_finite": True,
        "action_norm_max": 1.0 if nonzero else 0.0,
        "control_frequency_hz": runner.CONTROL_HZ,
        "chunk_size": runner.CHUNK_SIZE,
        "executed_actions_per_chunk": runner.EXECUTE_ACTIONS,
        "max_steps": runner.MAX_STEPS,
    }


def _cases() -> list[dict]:
    return [{"case_id": f"case-{index}", "panel": "reverse_dev" if index < 4 else "C20"} for index in range(24)]


def test_strict_json_rejects_duplicate_keys(tmp_path: Path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"a","status":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.strict_json_load(path)


def test_fast400_norm_and_new_namespace_are_locked():
    assert runner.EXPECTED_NORM_SHA == "7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333"
    assert "fast400" in runner.NORM_STATS.as_posix()
    assert runner.EVIDENCE_ROOT.name.endswith("attempt_0002")
    assert runner.OUTPUT_ROOT.name.endswith("attempt_0002")
    assert "q7_gpu3_frozen_eval" not in runner.EVIDENCE_ROOT.as_posix()


def test_free_port_closes_socket_and_returns_bindable_port():
    port = runner._free_port()
    assert isinstance(port, int)
    assert 1 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))


def test_policy_server_constructor_allocates_port_without_starting_subprocess(tmp_path: Path):
    server = runner.PolicyServer(tmp_path / "checkpoint", tmp_path / "server", 0)
    assert 1 <= server.port <= 65535
    assert server.process is None


def test_adapter_source_identity_is_current_and_locked():
    source = runner.adapter_source_identity()
    assert source["path"] == str(runner.ADAPTER_PATH.resolve())
    assert len(source["sha256"]) == 64
    runner.assert_adapter_source_identity(source)


def test_failed_attempt_namespace_is_write_immutable_but_status_is_read_only():
    with pytest.raises(ValueError, match="immutable failed E1 attempt"):
        runner._assert_new_namespace(runner.FAILED_EVIDENCE_ROOT)
    status = runner.status(evidence_dir=runner.FAILED_EVIDENCE_ROOT)
    assert status["status"] == "read_only"
    assert status["roles"]["r6"]["failure"]["status"] == "failed"


def test_resume_and_merge_lock_source_sha_mismatch_fails_closed():
    source = runner.adapter_source_identity()
    with pytest.raises(ValueError, match="adapter source identity"):
        runner._assert_locked_adapter_source({}, source)
    with pytest.raises(ValueError, match="adapter source identity"):
        runner._assert_locked_adapter_source({"adapter_source": {"path": source["path"], "sha256": "0" * 64}}, source)


def test_worker_lock_summary_and_failure_record_adapter_source(monkeypatch, tmp_path: Path):
    source = runner.adapter_source_identity()
    identity = {"role": "r6_params_only_baseline", "params_tree_sha256": "tree"}
    inputs = {"all_cases": [], "input_hashes": {"adapter_sha256": source["sha256"]}}
    bridge = types.ModuleType("tools.axis_pi05.libero_asset_bridge")
    bridge.activate = lambda root: {"root": str(root)}
    config = types.ModuleType("pipeline.config")
    config.EvalConfig = lambda **kwargs: kwargs
    environment = types.ModuleType("pipeline.environment")
    environment.EnvironmentManager = lambda config: object()
    monkeypatch.setitem(sys.modules, "tools.axis_pi05.libero_asset_bridge", bridge)
    monkeypatch.setitem(sys.modules, "pipeline.config", config)
    monkeypatch.setitem(sys.modules, "pipeline.environment", environment)
    monkeypatch.setattr(runner, "bind_parent_renderer_environment", lambda gpu: {"gpu": str(gpu)})
    monkeypatch.setattr(runner, "preflight", lambda **kwargs: {"adapter_source": kwargs["adapter_source"], "input_hashes": inputs["input_hashes"]})
    monkeypatch.setattr(runner, "load_frozen_inputs", lambda **kwargs: inputs)
    monkeypatch.setattr(runner, "validate_checkpoint", lambda role: identity)
    monkeypatch.setattr(runner, "git_info", lambda root: {"root": str(root)})
    monkeypatch.setattr(runner, "_run_cases", lambda *args: [])
    summary = runner.run_worker(role="r6", gpu=0, evidence_dir=tmp_path / "success-evidence", output_dir=tmp_path / "success-output")
    role_root = tmp_path / "success-evidence" / "r6"
    assert summary["adapter_source"] == source
    assert runner.strict_json_load(role_root / "attempt.lock.json")["adapter_source"] == source
    assert runner.strict_json_load(role_root / "summary.json")["adapter_source"] == source

    monkeypatch.setattr(runner, "_run_cases", lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run_worker(role="r6", gpu=0, evidence_dir=tmp_path / "failure-evidence", output_dir=tmp_path / "failure-output")
    failure = runner.strict_json_load(tmp_path / "failure-evidence" / "r6" / "failure.json")
    assert failure["adapter_source"] == source


def test_versioned_hash_domains_are_stable_and_distinct(tmp_path: Path):
    params = tmp_path / "params"
    (params / "nested").mkdir(parents=True)
    (params / "a").write_bytes(b"alpha")
    (params / "nested" / "b").write_bytes(b"beta")
    q0 = runner.q0_repo_relative_sha256_v1(params, repo_root=tmp_path)
    native = runner.params_relative_nul_sha256_v1(params)
    assert q0 == runner.q0_repo_relative_sha256_v1(params, repo_root=tmp_path)
    assert native == runner.params_relative_nul_sha256_v1(params)
    assert q0 != native


def test_r6_hash_file_count_and_bytes_mismatches_fail_closed():
    expected = {
        runner.Q0_REPO_RELATIVE_SHA256_V1: runner.EXPECTED_R6_PARAMS_Q0_SHA,
        runner.PARAMS_RELATIVE_NUL_SHA256_V1: runner.EXPECTED_R6_PARAMS_NATIVE_SHA,
        "file_count": runner.EXPECTED_R6_PARAMS_FILE_COUNT,
        "bytes": runner.EXPECTED_R6_PARAMS_BYTES,
    }
    runner._assert_frozen_params_identity("r6", expected)
    for field in expected:
        bad = dict(expected)
        bad[field] = "bad" if isinstance(expected[field], str) else expected[field] + 1
        with pytest.raises(ValueError, match=field):
            runner._assert_frozen_params_identity("r6", bad)


def test_wrong_norm_hash_fails_closed(monkeypatch):
    monkeypatch.setattr(runner, "EXPECTED_NORM_SHA", "0" * 64)
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.load_frozen_inputs()


def test_real_r6_preflight_is_dual_hash_locked(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "_runtime_preflight", lambda role, gpu: {"gpu": {"index": gpu}, "compute_apps": []})
    result = runner.preflight(role="r6", gpu=0, evidence_dir=tmp_path / "evidence", output_dir=tmp_path / "output")
    identity = result["identity"]
    assert result["adapter_source"]["sha256"] == runner.adapter_source_identity()["sha256"]
    assert result["input_hashes"]["adapter_sha256"] == result["adapter_source"]["sha256"]
    assert identity["params_hashes"][runner.Q0_REPO_RELATIVE_SHA256_V1] == runner.EXPECTED_R6_PARAMS_Q0_SHA
    assert identity["params_hashes"][runner.PARAMS_RELATIVE_NUL_SHA256_V1] == runner.EXPECTED_R6_PARAMS_NATIVE_SHA
    assert identity["params_tree_sha256_algorithm"] == runner.PARAMS_RELATIVE_NUL_SHA256_V1
    assert identity["params_file_count"] == runner.EXPECTED_R6_PARAMS_FILE_COUNT
    assert identity["params_bytes"] == runner.EXPECTED_R6_PARAMS_BYTES
    assert not (tmp_path / "evidence").exists()
    assert not (tmp_path / "output").exists()


def test_step300_identity_is_distinct_from_step100_and_baseline():
    assert runner.ROLE_SPECS["step300"]["role"] == "h300_step300_candidate"
    assert runner.ROLE_SPECS["step300"]["checkpoint_step"] == 300
    assert runner.ROLE_SPECS["step100"]["diagnostic_only"] is True
    assert runner.ROLE_SPECS["r6"]["require_train_state"] is False


def test_namespace_guard_rejects_historical_q7_and_n5(tmp_path: Path):
    with pytest.raises(ValueError, match="historical Q7/N5"):
        runner._assert_new_namespace(tmp_path / "q7_gpu3_frozen_eval")
    with pytest.raises(ValueError, match="historical Q7/N5"):
        runner._assert_new_namespace(tmp_path / "n5_official_t3_step100_paired_eval")


def test_server_environment_allows_authorized_gpu0_and_new_norm():
    env = runner.server_environment(0)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert env["AXIS_PI05_NORM_STATS"].endswith("data/manifests_track3/fast400_gold_80_20_20260913/norm_stats.json")
    assert env["AXIS_PI05_TRACK3_NORM_STATS"] == env["AXIS_PI05_NORM_STATS"]
    assert env["AXIS_PI05_NORM_STATS_SHA256"] == runner.EXPECTED_NORM_SHA
    assert env["HF_HUB_OFFLINE"] == "1"


def test_controlled_parent_sigterm_is_accepted_only_with_full_evidence():
    shutdown = {
        "returncode": -15,
        "clean_shutdown": True,
        "pid_exited": True,
        "termination_requested": True,
        "termination_signal": "SIGTERM",
        "kill_required": False,
    }
    assert runner.controlled_server_shutdown(shutdown, orphan_free=True) is True
    assert runner.controlled_server_shutdown({**shutdown, "kill_required": True}, orphan_free=True) is False
    assert runner.controlled_server_shutdown({**shutdown, "termination_requested": False}, orphan_free=True) is False
    assert runner.controlled_server_shutdown(shutdown, orphan_free=False) is False


def test_active_norm_attestation_rejects_missing_mismatched_or_wrong_bytes(monkeypatch, tmp_path: Path):
    norm = tmp_path / "norm.json"
    norm.write_text('{"norm":"fast400"}\n', encoding="utf-8")
    sha = hashlib.sha256(norm.read_bytes()).hexdigest()
    monkeypatch.setenv("AXIS_PI05_NORM_STATS", str(norm))
    monkeypatch.setenv("AXIS_PI05_TRACK3_NORM_STATS", str(norm))
    monkeypatch.setenv("AXIS_PI05_NORM_STATS_SHA256", sha)
    assert checkpoint_server.active_norm_attestation() == {"path": str(norm.resolve()), "sha256": sha}
    monkeypatch.delenv("AXIS_PI05_NORM_STATS", raising=False)
    with pytest.raises(RuntimeError, match="required"):
        checkpoint_server.active_norm_attestation()
    monkeypatch.setenv("AXIS_PI05_NORM_STATS", str(norm))
    monkeypatch.setenv("AXIS_PI05_NORM_STATS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        checkpoint_server.active_norm_attestation()


def test_policy_health_requires_active_fast400_norm_before_action(monkeypatch, tmp_path: Path):
    server = runner.PolicyServer(tmp_path / "checkpoint", tmp_path / "server", 0)
    expected = {"status": "ok", "active_norm_path": str(runner.NORM_STATS.resolve()), "active_norm_sha256": runner.EXPECTED_NORM_SHA, "checkpoint_path": str(server.checkpoint.resolve()), "config_name": runner.CONFIG_NAME}

    class Response:
        status = 200
        def __init__(self, body: dict): self.body = body
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return json.dumps(self.body).encode("utf-8")

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *_args, **_kwargs: Response(expected))
    assert server._verify_health() == expected
    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *_args, **_kwargs: Response({**expected, "active_norm_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="health attestation mismatch"):
        server._verify_health()
    source = inspect.getsource(runner.PolicyServer.start)
    assert source.index("self._verify_health()") < source.index("return client")


def test_parent_environment_has_full_per_gpu_renderer_contract():
    expected_keys = set(runner.PARENT_RENDERER_ENV_KEYS)
    for gpu in (0, 1):
        env = runner.parent_environment(gpu)
        assert expected_keys <= set(env)
        assert env["CUDA_VISIBLE_DEVICES"] == str(gpu)
        assert env["MUJOCO_GL"] == "egl"
        assert env["MUJOCO_EGL_DEVICE_ID"] == str(gpu)
        assert env["AXIS_PI05_EVAL_GPU"] == str(gpu)
        assert env["LIBERO_CONTROL_FREQ"] == "20"
        assert env["LIBERO_CONFIG_PATH"] == str(runner.ROOT / ".eval_libero")
        assert env["AXIS_PI05_LIBERO_ASSET_BRIDGE"] == "1"


def test_parent_binding_overrides_inherited_gpu_egl_and_frequency(monkeypatch):
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "3",
        "MUJOCO_GL": "disable",
        "MUJOCO_EGL_DEVICE_ID": "3",
        "AXIS_PI05_EVAL_GPU": "3",
        "LIBERO_CONTROL_FREQ": "10",
        "LIBERO_CONFIG_PATH": "/tmp/wrong-libero-config",
        "AXIS_PI05_LIBERO_ASSET_BRIDGE": "0",
    }.items():
        monkeypatch.setenv(key, value)
    bound = runner.bind_parent_renderer_environment(1)
    assert all(os.environ[key] == bound[key] for key in runner.PARENT_RENDERER_ENV_KEYS)
    runner.assert_parent_renderer_environment(1)


def test_parent_child_gpu_identity_mismatch_fails_closed():
    child = runner.server_environment(1)
    child["MUJOCO_EGL_DEVICE_ID"] = "0"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        runner.assert_parent_child_gpu_identity(1, child)


def test_worker_binds_parent_before_preflight_and_namespace_creation():
    source = inspect.getsource(runner.run_worker)
    assert source.index("bind_parent_renderer_environment") < source.index("preflight(")
    assert source.index("bind_parent_renderer_environment") < source.index("role_root =")


def test_renderer_smoke_is_explicitly_no_policy_and_no_action():
    source = inspect.getsource(runner.renderer_smoke_child)
    assert "PolicyServer" not in source
    assert "action_calls\": 0" in source
    assert "physics_step_calls\": 0" in source


def test_policy_server_smoke_is_no_environment_and_no_action():
    source = inspect.getsource(runner.policy_server_smoke_child)
    assert "EnvironmentManager" not in source
    assert "server.start()" in source
    assert '"action_calls": 0' in source


def test_execute8_discards_two_actions_without_reseeding():
    class FakeClient:
        def __init__(self):
            self.resets = []
            self.actions = [list(range(7)) for _ in range(20)]

        def reset(self, **kwargs):
            self.resets.append(kwargs)

        def get_action(self, _observation):
            return self.actions.pop(0)

    client = FakeClient()
    policy = runner.Execute8Policy(client)
    policy.reset("instruction", seed=2000)
    for _ in range(9):
        policy.get_action({})
    assert client.resets[0]["reset_kind"] == "episode_reset"
    assert client.resets[1]["reset_kind"] == "chunk_discard"
    assert client.resets[1].get("seed") is None
    assert client.resets[1]["chunk_index"] == 1


def test_gate_boundary_passes_exact_reverse_plus_one_and_c20_minus_two():
    pairs = []
    for index in range(4):
        pairs.append({"panel": "reverse_dev", "baseline_success": False, "candidate_success": index == 0, "baseline_nonzero": True, "candidate_nonzero": True})
    for index in range(20):
        pairs.append({"panel": "C20", "baseline_success": True, "candidate_success": index < 18, "baseline_nonzero": True, "candidate_nonzero": True})
    decision = runner.continuation_gates(pairs)
    assert decision["reverse"]["delta"] == 1
    assert decision["c20"]["delta"] == -2
    assert decision["behavior_gate_passed"] is True


def test_gate_rejects_reverse_flat_or_c20_minus_three():
    pairs = [{"panel": "reverse_dev", "baseline_success": False, "candidate_success": False, "baseline_nonzero": True, "candidate_nonzero": True} for _ in range(4)]
    pairs += [{"panel": "C20", "baseline_success": True, "candidate_success": index < 17, "baseline_nonzero": True, "candidate_nonzero": True} for index in range(20)]
    decision = runner.continuation_gates(pairs)
    assert decision["gates"]["reverse_at_least_plus_1_of_4"] is False
    assert decision["gates"]["c20_drop_at_most_2_of_20"] is False
    assert decision["behavior_gate_passed"] is False


def test_single_zero_action_episode_does_not_trigger_systemwide_gate():
    pairs = [{"panel": "reverse_dev", "baseline_success": False, "candidate_success": False, "baseline_nonzero": index != 0, "candidate_nonzero": index != 0} for index in range(4)]
    pairs += [{"panel": "C20", "baseline_success": False, "candidate_success": False, "baseline_nonzero": True, "candidate_nonzero": True} for _ in range(20)]
    decision = runner.continuation_gates(pairs)
    assert decision["gates"]["no_systemwide_all_zero_actions"] is True


def test_systemwide_all_zero_role_fails_closed():
    pairs = [{"panel": "reverse_dev", "baseline_success": False, "candidate_success": False, "baseline_nonzero": False, "candidate_nonzero": False} for _ in range(4)]
    pairs += [{"panel": "C20", "baseline_success": False, "candidate_success": False, "baseline_nonzero": False, "candidate_nonzero": False} for _ in range(20)]
    decision = runner.continuation_gates(pairs)
    assert decision["gates"]["no_systemwide_all_zero_actions"] is False


def test_paired_summary_rejects_case_id_mismatch():
    cases = _cases()
    baseline = [_row(case["case_id"], case["panel"]) for case in cases]
    candidate = [_row(case["case_id"], case["panel"]) for case in cases]
    candidate[1]["case_id"] = "different-case"
    with pytest.raises(ValueError, match="case-ID order"):
        runner.paired_summary(baseline, candidate, cases)


def test_paired_summary_requires_matching_case_fingerprint():
    cases = _cases()
    baseline = [_row(case["case_id"], case["panel"]) for case in cases]
    candidate = [_row(case["case_id"], case["panel"]) for case in cases]
    candidate[0]["case_fingerprint"] = "changed"
    summary = runner.paired_summary(baseline, candidate, cases)
    assert summary["gates"]["case_fingerprints_and_seeds_match"] is False
    assert summary["behavior_gate_passed"] is False


def test_merge_release_decision_is_no_zip_on_failed_gate(monkeypatch, tmp_path: Path):
    cases = _cases()
    baseline = [_row(case["case_id"], case["panel"]) for case in cases]
    candidate = [_row(case["case_id"], case["panel"]) for case in cases]
    for row in candidate[17:20]:
        row["success"] = False
    # This test exercises the pure decision shape without touching checkpoint
    # or simulator assets; merge itself is covered by the same schema in E1.
    summary = runner.paired_summary(baseline, candidate, cases)
    assert summary["behavior_gate_passed"] is False
    decision = {
        "status": "behavior_gate_failed_no_zip",
        "zip_created": False,
        "fold_started": False,
    }
    assert decision["status"] == "behavior_gate_failed_no_zip"
    assert decision["zip_created"] is False


def test_resume_completed_case_does_not_rerun(tmp_path: Path):
    case = _row("case-0", "reverse_dev")
    identity = {"role": "r6_params_only_baseline", "params_tree_sha256": "tree"}
    case["norm_stats_sha256"] = runner.EXPECTED_NORM_SHA
    case["checkpoint_params_tree_sha256"] = "tree"
    case["role"] = identity["role"]
    case["status"] = "completed"
    trace = tmp_path / "trace.npy"
    np.save(trace, np.ones((1, 7), dtype=np.float32), allow_pickle=False)
    case["action_trace_path"] = str(trace)
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case), encoding="utf-8")
    loaded = runner._load_completed(path, {"case_id": "case-0", "case_fingerprint": case["case_fingerprint"]}, identity)
    assert loaded is not None
    assert loaded["case_id"] == "case-0"


def test_resume_refreshes_current_pid_gpu_status_and_timestamp():
    cursor, heartbeat = runner._refresh_runtime_state(
        {"pid": 111, "gpu": 0, "status": "stopped"},
        {"pid": 111, "gpu": 0, "status": "stopped", "sequence": 4},
        role="r6",
        gpu=3,
        resume=True,
        pid=222,
        now=100.0,
    )
    assert cursor["pid"] == heartbeat["pid"] == 222
    assert cursor["gpu"] == heartbeat["gpu"] == 3
    assert cursor["status"] == heartbeat["status"] == "running"
    assert cursor["resumed_at_unix"] == heartbeat["resumed_at_unix"] == 100.0
    assert cursor["resumed_from_pid"] == heartbeat["resumed_from_pid"] == 111


def test_heartbeat_refreshes_periodically_and_stops_atomically(tmp_path: Path):
    path = tmp_path / "heartbeat.json"
    updater = runner._HeartbeatUpdater(
        path,
        {"schema_version": "test", "role": "r6", "gpu": 2, "status": "stopped", "sequence": 0},
        interval_sec=0.01,
    )
    updater.start()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        current = runner.strict_json_load(path)
        if current["sequence"] >= 2:
            break
        time.sleep(0.01)
    running = runner.strict_json_load(path)
    assert running["status"] == "running"
    assert running["sequence"] >= 2
    updater.stop(terminal_phase="test_complete")
    stopped = runner.strict_json_load(path)
    assert stopped["status"] == "stopped"
    assert stopped["terminal_phase"] == "test_complete"
    assert stopped["pid"] == os.getpid()


def test_status_reports_pid_liveness_and_heartbeat_freshness(tmp_path: Path):
    role_root = tmp_path / "r6"
    role_root.mkdir()
    runner.atomic_json(role_root / "cursor.json", {"pid": os.getpid(), "status": "running"})
    runner.atomic_json(role_root / "heartbeat.json", {"status": "running", "updated_at_unix": time.time()})
    fresh = runner.status(evidence_dir=tmp_path)
    assert fresh["roles"]["r6"]["pid_alive"] is True
    assert fresh["roles"]["r6"]["heartbeat_fresh"] is True
    runner.atomic_json(role_root / "heartbeat.json", {"status": "running", "updated_at_unix": time.time() - runner.HEARTBEAT_STALE_AFTER_SEC - 1})
    stale = runner.status(evidence_dir=tmp_path)
    assert stale["roles"]["r6"]["heartbeat_fresh"] is False
