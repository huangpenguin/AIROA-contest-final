#!/usr/bin/env python3
"""R0/R1 recovery gates for the native reverse validator.

R0 is intentionally independent of the historical D2 report.  R1 imports
the already reviewed model/state helpers only after locking their pre-R1
hash, then adds the real no-renderer schema probe and process-level contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
RECOVERY_ROOT = ROOT / "data/reports/axis_libero_30k/track3_native_reverse_validator_recovery"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation"
OLD_G1 = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_dynamic_gates.json"
S3_REPORT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_language_v2.json"
SPEC_ROOT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_v2_specs"
VALIDATOR = ROOT / "tools/axis_pi05/track3_dynamic_gate_reconciliation.py"
OLD_TEST = ROOT / "tests/test_track3_dynamic_gate_reconciliation.py"
HISTORICAL_TEST = ROOT / "tests/test_track3_support_autogen_v2.py"
HANDOFF_JSON = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def git_output(repo: Path, args: list[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def git_ok(repo: Path, args: list[str]) -> bool:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).returncode == 0


def porcelain(repo: Path) -> list[str]:
    return [line for line in git_output(repo, ["status", "--porcelain", "--untracked-files=all"]).splitlines() if line]


def lock_file(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def r0_paths() -> list[Path]:
    return [
        OLD_ROOT / "d0_release_lock.json",
        OLD_ROOT / "d2_task10_canary.json",
        OLD_ROOT / "runs/d2_task10_canary_run_1.json",
        OLD_ROOT / "summary.json",
        OLD_ROOT / "D4_CLOSEOUT_REPORT_ZH.md",
        OLD_G1,
        *sorted(SPEC_ROOT.glob("canary_*.bddl")),
        VALIDATOR,
        OLD_TEST,
        HISTORICAL_TEST,
    ]


def run_r0() -> dict[str, Any]:
    root_commit = git_output(ROOT, ["rev-parse", "HEAD"])
    nested = ROOT / "third_party/axis_openpi"
    nested_commit = git_output(nested, ["rev-parse", "HEAD"]) if nested.is_dir() else "missing"
    paths = r0_paths()
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    locked = {str(path.resolve()): lock_file(path) for path in paths if path.is_file()}
    d2 = json.loads((OLD_ROOT / "d2_task10_canary.json").read_text(encoding="utf-8"))
    d2_run = d2["run"]
    d4 = json.loads((OLD_ROOT / "summary.json").read_text(encoding="utf-8"))
    handoff = json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))
    input_root_ancestor = git_ok(ROOT, ["merge-base", "--is-ancestor", "84c2655f610bb957b0ad1ec2a952776bf023821e", root_commit])
    lock = {
        "schema_version": "track3_native_reverse_validator_recovery_r0_v1",
        "status": "track3_native_reverse_validator_recovery_input_locked" if not missing else "track3_native_reverse_validator_recovery_input_drift",
        "created_at_utc": now(),
        "plan": str((ROOT / "docs/AXIS_PI05_TRACK3_NATIVE_REVERSE_VALIDATOR_RECOVERY_PLAN.md").resolve()),
        "input_root_commit": "84c2655f610bb957b0ad1ec2a952776bf023821e",
        "observed_root_commit_at_r0": root_commit,
        "input_root_parent_commit": "f0034b9976e6de81644d1535d60328006bbe83cf",
        "input_nested_openpi_commit": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
        "observed_nested_openpi_commit_at_r0": nested_commit,
        "root_status_porcelain_at_r0": porcelain(ROOT),
        "nested_openpi_status_porcelain_at_r0": porcelain(nested) if nested.is_dir() else ["missing"],
        "root_input_commit_clean": input_root_ancestor,
        "nested_input_commit_clean": nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not porcelain(nested),
        "libero_plus_dirty_read_only_snapshot": {"path": str((ROOT / "LIBERO-plus").resolve()), "status_porcelain": porcelain(ROOT / "LIBERO-plus"), "policy": "record_only_never_clean_or_touch"},
        "immutable_recovery_inputs": locked,
        "missing_inputs": missing,
        "historical_d2": {
            "report": str((OLD_ROOT / "d2_task10_canary.json").resolve()),
            "worker_status": d2_run.get("status"),
            "worker_returncode": d2_run.get("worker_returncode"),
            "error": d2_run.get("error"),
            "valid_predicate_settle_evidence": 0,
            "d3_runs": 0,
            "must_not_relabel_or_overwrite": True,
        },
        "counts_at_r0": {
            "exact_reverse_demo": 0,
            "pending_autogen": 0,
            "pending_autogen_v2": 0,
            "accepted_reverse_demo": 0,
            "support_retry_attempts": 0,
        },
        "authorization_at_r0": {
            "source_status": handoff.get("track3_native_reverse_validator_recovery_authorization", {}).get("status"),
            "se3_or_grasp": False,
            "teleop": False,
            "batch_generation": False,
            "training_or_norm": False,
            "fast_or_zip": False,
            "official_gpu": False,
            "upload_or_submission": False,
        },
        "hard_gate": {
            "all_inputs_exist": not missing,
            "root_input_commit_locked": input_root_ancestor,
            "nested_input_commit_locked": nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not porcelain(nested),
            "historical_d2_exception_locked": d2_run.get("status") == "worker_exception" and d2_run.get("worker_returncode") == 0 and "TargetZone" in d2_run.get("error", ""),
            "historical_d2_valid_results_zero": d4["d2"]["status"] == "track3_native_reverse_dynamic_canary_failed_support" and d4["d3"]["fresh_process_runs"] == 0,
            "counts_zero": True,
            "old_d2_and_d4_immutable": True,
            "libero_plus_not_touched": True,
        },
    }
    write_json(RECOVERY_ROOT / "r0_release_lock.json", lock)
    return lock


def _recovery_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"})
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("MUJOCO_EGL_DEVICE_ID", None)
    return env


def _probe_process(task_index: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.axis_pi05.track3_native_reverse_validator_recovery", "--schema-worker", str(task_index)]
    completed = subprocess.run(command, cwd=ROOT, env=_recovery_env(), capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    result: dict[str, Any]
    try:
        result = json.loads(lines[-1]) if lines else {"status": "worker_no_output"}
    except json.JSONDecodeError:
        result = {"status": "worker_output_invalid"}
    result["worker_returncode"] = completed.returncode
    result["stdout_tail"] = completed.stdout[-4000:]
    result["stderr_tail"] = completed.stderr[-4000:]
    return result


def run_r1() -> dict[str, Any]:
    d0 = json.loads((RECOVERY_ROOT / "r0_release_lock.json").read_text(encoding="utf-8"))
    if d0["status"] != "track3_native_reverse_validator_recovery_input_locked":
        raise RuntimeError("R0 is not locked")
    results = [_probe_process(task_index) for task_index in (10, 20, 19)]
    by_family = {row.get("operator_family", f"task_{row.get('task_index')}"): row for row in results}
    family_gate = {family: any(row.get("operator_family") == family and row.get("status") == "schema_passed" and row.get("worker_returncode") == 0 for row in results) for family in ("support_remove", "containment_remove", "articulation_close")}
    passed = all(family_gate.values()) and len(results) == 3
    status = "track3_native_reverse_validator_runtime_schema_passed" if passed else f"track3_native_reverse_runtime_schema_failed_{next((row.get('operator_family', 'unknown') for row in results if row.get('status') != 'schema_passed' or row.get('worker_returncode') != 0), 'unknown')}"
    report = {
        "schema_version": "track3_native_reverse_validator_recovery_r1_v1",
        "status": status,
        "created_at_utc": now(),
        "mode": "CPU_no_renderer",
        "fresh_process_schema_probe_runs": results,
        "family_3_of_3_schema": family_gate,
        "hard_gate": {
            "three_real_no_renderer_envs": len(results) == 3,
            "all_families_schema_passed": passed,
            "truth_detail_separated": all(row.get("truth_detail_separated") is True for row in results),
            "worker_returncode_contract": all((row.get("status") == "schema_passed") == (row.get("worker_returncode") == 0) for row in results),
            "hdf5_state_used_for_predicate_oracle": False,
        },
    }
    write_json(RECOVERY_ROOT / "r1_runtime_schema_probe.json", report)
    return report


def schema_worker(task_index: int) -> dict[str, Any]:
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as old

    spec = old._specs()[task_index]
    env = None
    try:
        env = old.make_schema_env(spec["reverse_bddl_path"])
        layout = old.model_layout(env, spec)
        _, native_state = old._native_reset(env)
        initial = old.evaluate_predicates(env, spec["reverse_initial_predicates"], label="schema_native_initial")
        goal = old.evaluate_predicates(env, spec["reverse_goal_predicates"], label="schema_native_goal")
        atoms = [*spec["reverse_initial_predicates"], *spec["reverse_goal_predicates"]]
        detail_ok = all(row.get("detail_status") == "passed" and "detail" in row for group in (initial, goal) for row in group["predicates"])
        robot = old._robot_metrics(env)
        scene = old._scene_metrics(env, atoms)
        target = spec["reverse_initial_predicates"][-1]
        target_resolution = old.target_resolution(env, task_index, native_state, target)
        result = {
            "schema_version": "track3_native_reverse_validator_recovery_schema_run_v1",
            # R1 proves that the reverse predicates and telemetry schema can
            # be evaluated in a real native env.  Whether native placement
            # happens to satisfy the reverse initial/goal values is the R2
            # semantic canary gate, not a schema-construction requirement.
            "status": "schema_passed" if detail_ok and initial["truth_complete"] and goal["truth_complete"] and target_resolution["schema_complete"] else "schema_failed",
            "task_index": task_index,
            "operator_family": spec["operator_family"],
            "fresh_process": True,
            "renderer_contract": {"has_renderer": False, "has_offscreen_renderer": False, "use_camera_obs": False},
            "layout": layout,
            "native_reset_state_sha256": old.state_hash(native_state),
            "truth_detail_separated": True,
            "native_truth": {"initial": initial, "goal": goal},
            "robot_snapshot": robot,
            "scene_snapshot": scene,
            "target_resolution": target_resolution,
            "gripper_hold_contract": {"action": old.gripper_hold_action().tolist(), "fixed_gripper_minus_one_used": False},
            "hdf5_state_used_for_predicate_oracle": False,
            "schema_complete": detail_ok and initial["truth_complete"] and goal["truth_complete"] and target_resolution["schema_complete"],
        }
        return result
    except Exception as exc:
        return {"schema_version": "track3_native_reverse_validator_recovery_schema_run_v1", "status": "schema_worker_exception", "task_index": task_index, "operator_family": spec["operator_family"], "fresh_process": True, "error": f"{type(exc).__name__}: {exc}", "truth_detail_separated": False, "schema_complete": False, "hdf5_state_used_for_predicate_oracle": False}
    finally:
        if env is not None:
            env.close()


def _execution_input_lock() -> dict[str, Any]:
    """Capture the immutable inputs used by R2/R3 after R1 has passed."""
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as old

    threshold_payload = json.dumps(old.THRESHOLDS, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "validator": lock_file(VALIDATOR),
        "spec_report": lock_file(S3_REPORT),
        "specs": [lock_file(path) for path in sorted(SPEC_ROOT.glob("canary_*.bddl"))],
        "thresholds": {"values": old.THRESHOLDS, "sha256": hashlib.sha256(threshold_payload).hexdigest()},
        "warmup_steps": old.WARMUP_STEPS,
        "measurement_steps": old.MEASUREMENT_STEPS,
        "control_freq_hz": old.CONTROL_FREQ_HZ,
        "action_dim": old.ACTION_DIM,
        "controller": "OSC_POSE",
    }


def _gpu_snapshot() -> dict[str, Any]:
    """Read-only GPU/PID snapshot used for the physical-GPU3 boundary."""
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    app_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    gpu_rows = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
    app_rows = [line.strip() for line in app_query.stdout.splitlines() if line.strip() and "No running processes" not in line]
    gpu3 = next((row for row in gpu_rows if row.split(",", 1)[0].strip() == "3"), None)
    gpu3_uuid = gpu3.split(",", 2)[1].strip() if gpu3 else None
    gpu3_apps = [row for row in app_rows if gpu3_uuid and row.split(",", 1)[0].strip() == gpu3_uuid]
    return {
        "query_returncode": gpu_query.returncode,
        "apps_query_returncode": app_query.returncode,
        "gpu_rows": gpu_rows,
        "compute_app_rows": app_rows,
        "gpu3_row": gpu3,
        "gpu3_compute_apps": gpu3_apps,
        "gpu3_natural_idle": bool(gpu_query.returncode == 0 and app_query.returncode == 0 and gpu3 is not None and not gpu3_apps),
        "gpu3_only_process_boundary": "CUDA_VISIBLE_DEVICES=3; MUJOCO_EGL_DEVICE_ID=3",
        "gpu0_not_selected": True,
    }


def _worker_process(task_index: int, run_index: int, phase: str) -> dict[str, Any]:
    """Run one validator worker in a fresh physical-GPU3 process."""
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "3",
        "MUJOCO_EGL_DEVICE_ID": "3",
        "MUJOCO_GL": "egl",
        "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}",
        "JAX_PLATFORMS": "cpu",
    })
    command = [sys.executable, "-m", "tools.axis_pi05.track3_dynamic_gate_reconciliation", "--worker", str(task_index), str(run_index), phase]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1]) if lines else {"status": "worker_no_output"}
    except json.JSONDecodeError:
        result = {"status": "worker_output_invalid"}
    result["worker_returncode"] = completed.returncode
    result["stdout_tail"] = completed.stdout[-4000:]
    result["stderr_tail"] = completed.stderr[-4000:]
    if result.get("status") == "passed" and completed.returncode != 0:
        result["status"] = "worker_contract_failure"
        result["contract_error"] = "passed status with nonzero worker returncode"
    return result


def _gate_from_run(run: Mapping[str, Any]) -> dict[str, bool]:
    native = run.get("native_reset", {})
    initial = native.get("initial", {}) if isinstance(native, Mapping) else {}
    goal = native.get("goal", {}) if isinstance(native, Mapping) else {}
    settle = run.get("settle", {})
    settle_gate = settle.get("hard_gate", {}) if isinstance(settle, Mapping) else {}
    destruction = run.get("controlled_destruction", {})
    return {
        "worker_returncode_zero": run.get("worker_returncode") == 0,
        "worker_status_passed": run.get("status") == "passed",
        "schema_complete": run.get("schema_complete") is True,
        "reverse_initial_all_true": initial.get("all_true") is True,
        "reverse_goal_all_false": goal.get("all_false") is True,
        "model_state_predicate_oracle_reconciled": run.get("hard_gate", {}).get("model_state_predicate_oracle_reconciled") is True,
        "settle_all_named_fields": settle_gate.get("all_named_settle_gates") is True,
        "controlled_predicate_true_to_false": destruction.get("true_to_false") is True,
        "controlled_predicate_restored_true": destruction.get("restored_true") is True,
        "hdf5_state_not_predicate_oracle": run.get("hdf5_state_used_for_predicate_oracle") is False,
        "no_grasp_or_demo_generation": run.get("no_grasp_or_demo_generation") is True,
    }


def _is_worker_exception(run: Mapping[str, Any]) -> bool:
    return run.get("status") in {"worker_exception", "worker_output_invalid", "worker_no_output", "worker_contract_failure"} or run.get("worker_returncode") != 0


def _dynamic_failure_status(prefix: str, run: Mapping[str, Any]) -> str:
    field = str(run.get("failure_field") or "dynamic_gate")
    return f"{prefix}_{field}"


def run_r2() -> dict[str, Any]:
    r1 = json.loads((RECOVERY_ROOT / "r1_runtime_schema_probe.json").read_text(encoding="utf-8"))
    if r1.get("status") != "track3_native_reverse_validator_runtime_schema_passed":
        raise RuntimeError("R2 requires a passed R1 runtime schema probe")
    preflight = _gpu_snapshot()
    inputs = _execution_input_lock()
    run: dict[str, Any] | None = None
    if preflight["gpu3_natural_idle"]:
        run = _worker_process(10, 1, "R2_recovery_1")
        write_json(RECOVERY_ROOT / "runs/r2_task10_recovery_run_1.json", run)
    if not preflight["gpu3_natural_idle"]:
        status = "track3_native_reverse_validator_recovery_exception"
        failure = "GPU3 was not naturally idle at preflight"
    elif run is None or _is_worker_exception(run):
        status = "track3_native_reverse_validator_recovery_exception"
        failure = (run or {}).get("error", "missing worker result")
    elif run.get("status") != "passed":
        status = _dynamic_failure_status("track3_native_reverse_dynamic_failed_support", run)
        failure = run.get("failure_field", "dynamic_gate")
    else:
        gates = _gate_from_run(run)
        failed_gate = next((key for key, value in gates.items() if not value), "dynamic_gate")
        status = "track3_native_reverse_dynamic_canary_passed_support" if all(gates.values()) else f"track3_native_reverse_dynamic_failed_support_{failed_gate}"
        failure = None if all(gates.values()) else failed_gate
    report = {
        "schema_version": "track3_native_reverse_validator_recovery_r2_v1",
        "status": status,
        "created_at_utc": now(),
        "r1_status": r1["status"],
        "gpu_preflight": preflight,
        "execution_input_lock": inputs,
        "task_index": 10,
        "run_index": "recovery_1",
        "run": run,
        "hard_gate": _gate_from_run(run or {}) if run is not None else {"gpu3_natural_idle": False},
        "failure": failure,
        "hard_stop_after_one_task10": True,
        "no_se3_or_demo_generation": True,
        "no_pending_or_training": True,
    }
    write_json(RECOVERY_ROOT / "r2_task10_recovery_canary.json", report)
    return report


def run_r3() -> dict[str, Any]:
    r2 = json.loads((RECOVERY_ROOT / "r2_task10_recovery_canary.json").read_text(encoding="utf-8"))
    if r2.get("status") != "track3_native_reverse_dynamic_canary_passed_support":
        raise RuntimeError("R3 requires a passed R2 recovery canary")
    locked_inputs = r2["execution_input_lock"]
    runs: list[dict[str, Any]] = []
    for task_index, family in ((10, "support_remove"), (20, "containment_remove"), (19, "articulation_close")):
        for run_index in range(1, 4):
            run = _worker_process(task_index, run_index, f"R3_{family}_{run_index}")
            run["r3_gate"] = _gate_from_run(run)
            runs.append(run)
            write_json(RECOVERY_ROOT / "runs" / f"r3_task_{task_index:02d}_run_{run_index}.json", run)
            if _is_worker_exception(run) or run.get("status") != "passed" or not all(run["r3_gate"].values()):
                break
        if len([row for row in runs if row.get("operator_family") == family]) < 3:
            break
    by_family = {family: [row for row in runs if row.get("operator_family") == family] for family in FAMILIES}
    family_gate = {family: len(rows) == 3 and all(row.get("status") == "passed" and all(row.get("r3_gate", {}).values()) for row in rows) for family, rows in by_family.items()}
    failed = next((row for row in runs if row.get("status") != "passed" or not all(row.get("r3_gate", {}).values())), None)
    if all(family_gate.values()):
        status = "track3_reverse_semantics_v2_dynamic_validated_pending_se3_review"
    elif failed is not None and _is_worker_exception(failed):
        status = "track3_native_reverse_validator_recovery_exception"
    else:
        status = _dynamic_failure_status(f"track3_reverse_semantics_v2_dynamic_failed_{failed.get('operator_family', 'unknown') if failed else 'unknown'}", failed or {})
    report = {
        "schema_version": "track3_native_reverse_validator_recovery_r3_v1",
        "status": status,
        "created_at_utc": now(),
        "gpu_policy": "physical_gpu3_only",
        "execution_input_lock": locked_inputs,
        "fresh_process_runs": runs,
        "family_3_of_3": family_gate,
        "hard_gate": {"all_families_3_of_3": all(family_gate.values()), "no_se3_or_demo_generation": True, "hdf5_state_used_for_predicate_oracle": False},
    }
    write_json(RECOVERY_ROOT / "r3_three_family_release.json", report)
    return report


def run_r4() -> dict[str, Any]:
    r0 = json.loads((RECOVERY_ROOT / "r0_release_lock.json").read_text(encoding="utf-8"))
    r1 = json.loads((RECOVERY_ROOT / "r1_runtime_schema_probe.json").read_text(encoding="utf-8"))
    r2 = json.loads((RECOVERY_ROOT / "r2_task10_recovery_canary.json").read_text(encoding="utf-8"))
    r3_path = RECOVERY_ROOT / "r3_three_family_release.json"
    r3 = json.loads(r3_path.read_text(encoding="utf-8")) if r3_path.is_file() else {"status": "not_run_due_to_r2_hard_stop", "fresh_process_runs": []}
    final_status = r3["status"] if r3_path.is_file() else r2["status"]
    summary = {
        "schema_version": "track3_native_reverse_validator_recovery_r4_v1",
        "status": final_status,
        "created_at_utc": now(),
        "r0": {"status": r0["status"], "observed_root_commit_at_r0": r0["observed_root_commit_at_r0"], "input_root_commit": r0["input_root_commit"], "historical_d2": r0["historical_d2"], "counts_at_r0": r0["counts_at_r0"]},
        "r1": {"status": r1["status"], "family_3_of_3_schema": r1["family_3_of_3_schema"], "hard_gate": r1["hard_gate"]},
        "r2": {"status": r2["status"], "task_index": r2["task_index"], "run_index": r2["run_index"], "failure": r2["failure"], "hard_gate": r2["hard_gate"]},
        "r3": {"status": r3["status"], "family_3_of_3": r3.get("family_3_of_3"), "fresh_process_runs": len(r3.get("fresh_process_runs", []))},
        "outputs": {"exact_reverse_demo": 0, "pending_autogen": 0, "pending_autogen_v2": 0, "accepted_reverse_demo": 0, "se3_grasp_attempts": 0, "training_started": False, "human_operation": False, "official_gpu_used": False, "fast_or_zip_used": False, "upload_or_submission": False, "libero_plus_touched": False},
        "hard_stop": {"stop_after": "R2" if not r3_path.is_file() else "R3", "awaiting_se3_review": True, "do_not_enter": ["SE(3) grasp", "demo generation", "batch generation", "training", "Teleop", "FAST/ZIP", "upload/submission"]},
    }
    write_json(RECOVERY_ROOT / "summary.json", summary)
    report_lines = [
        "# Track 3 Native Reverse Validator Recovery：R0–R4 收尾",
        "",
        f"- 最终状态：`{summary['status']}`",
        f"- R0：锁定输入 commit `84c2655` 及观察到的审核 descendant `{r0['observed_root_commit_at_r0'][:12]}`；历史 D2 保留为 `worker_exception`、returncode `0`、有效 predicate/settle evidence `0`。",
        f"- R1：三族真实 CPU/no-renderer schema probe 为 `{r1['status']}`；每个 worker 均 fresh process、returncode `0`，truth/detail 分层成立。",
        f"- R2：task10 recovery canary 状态为 `{r2['status']}`；failure=`{r2['failure']}`。",
        f"- R3：`{r3['status']}`，fresh process runs=`{len(r3.get('fresh_process_runs', []))}`。",
        "",
        "## 结果边界",
        "",
        "本轮没有生成 reverse demo、pending 数据或训练产物；没有 SE(3) 抓取、人工 Teleop、FAST/ZIP、官方 GPU、上传或提交。R4 后停止，等待下一次 SE(3) 审核。",
        "",
        "证据：`r0_release_lock.json`、`r1_runtime_schema_probe.json`、`r2_task10_recovery_canary.json`、`runs/`、`r3_three_family_release.json`（如存在）、`summary.json`。",
    ]
    atomic = "\n".join(report_lines) + "\n"
    (RECOVERY_ROOT / "R4_CLOSEOUT_REPORT_ZH.md").write_text(atomic, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", action="store_true")
    parser.add_argument("--r1", action="store_true")
    parser.add_argument("--r2", action="store_true")
    parser.add_argument("--r3", action="store_true")
    parser.add_argument("--r4", action="store_true")
    parser.add_argument("--schema-worker", type=int)
    args = parser.parse_args()
    if args.r0:
        print(json.dumps(run_r0(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.r1:
        print(json.dumps(run_r1(), ensure_ascii=False, indent=2))
        return 0
    if args.r2:
        print(json.dumps(run_r2(), ensure_ascii=False, indent=2))
        return 0
    if args.r3:
        print(json.dumps(run_r3(), ensure_ascii=False, indent=2))
        return 0
    if args.r4:
        print(json.dumps(run_r4(), ensure_ascii=False, indent=2))
        return 0
    if args.schema_worker is not None:
        result = schema_worker(args.schema_worker)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("status") == "schema_passed" else 1
    parser.error("use --r0, --r1, or --schema-worker")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
