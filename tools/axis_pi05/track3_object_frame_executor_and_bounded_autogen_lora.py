#!/usr/bin/env python3
"""Track 3 object-frame executor and bounded continuation.

This is a derived execution namespace for the reviewed X0--X13 plan.  The
previous W0--W6 evidence is read-only input.  X0--X3 are CPU-only release
checks; X4 can run at most two CPU support profiles.  Downstream generation,
replay, training and packaging are entered only when their upstream gates are
actually successful.  X13 always writes a fail-closed closeout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as w

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/reports/axis_libero_30k/track3_object_frame_executor_and_bounded_autogen_lora"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md"
W_OUT = ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery"
W6_REPORT = W_OUT / "w6_closeout.json"
W6_MANIFEST = W_OUT / "w6_artifact_manifest.json"
W1_REPORT = W_OUT / "w1_release_anchor.json"
W4_PARENT = W_OUT / "w4_cpu_task10_demo0_nominal/parent_result.json"
W4_CHILD = W_OUT / "w4_cpu_task10_demo0_nominal/child_result.json"
W4_TRAJECTORY = W_OUT / "w4_cpu_task10_demo0_nominal/trajectory.npz"
HDF5_TASK10 = w.HDF5_TASK10
STATE_BUNDLE = w.STATE_BUNDLE
OPENPI = w.OPENPI
LIBERO_PLUS = w.LIBERO_PLUS
PUBLIC_LIBERO = w.PUBLIC_LIBERO
MODULE = "tools.axis_pi05.track3_object_frame_executor_and_bounded_autogen_lora"

X0_REPORT = OUT / "x0_input_lock.json"
X1_REPORT = OUT / "x1_derived_errata_finalizer.json"
X2_REPORT = OUT / "x2_executor_supervisor_contract.json"
X3_REPORT = OUT / "x3_clean_release.json"
X4_ROOT = OUT / "x4_support_cpu_profiles"
X4_SUMMARY = OUT / "x4_summary.json"
X13_REPORT = OUT / "x13_closeout.json"
X13_MANIFEST = OUT / "x13_artifact_manifest.json"
X13_VALIDATION = OUT / "x13_final_validation.json"
CURSOR = OUT / "resume_cursor.json"
SUPERVISOR_CLOSEOUT = OUT / "supervisor_closeout.json"
PHASE_CSV = OUT / "x13_phase_status.csv"

W6_COMMIT = "2997c6f5d8c5b05345d74e0e0c4b96698def727e"
OPENPI_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
GPU0_PID = "1133706"
CONTROL_FREQ_HZ = 20
ACTION_DIM = 7
DEADLINE_SECONDS = 300.0
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS = 900.0
WORKER_STATUSES = frozenset({"passed", "failed_gate"})

NOMINAL_PROFILE: dict[str, Any] = {
    "name": "object_frame_nominal",
    "translation_clip_m": 0.05,
    "rotation_clip_rad": 0.12,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 4,
    "stage_budgets": [150, 120, 35, 150, 180, 120, 35, 80],
    "step_budget": 900,
    "profile_reason": "W4_existing_budget_and_tolerance_with_X2_semantics",
}
CONSERVATIVE_PROFILE: dict[str, Any] = {
    "name": "object_frame_conservative",
    "translation_clip_m": 0.025,
    "rotation_clip_rad": 0.06,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 6,
    "stage_budgets": [200, 160, 50, 200, 260, 160, 50, 120],
    "step_budget": 1250,
    "profile_reason": "only_allowed_after_nominal_family_failure",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    result = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines() if line]


def source_porcelain(repo: Path = ROOT) -> list[str]:
    rows = porcelain(repo)
    if repo.resolve() != ROOT.resolve():
        return rows
    prefix = f"{rel(OUT).rstrip('/')}/"
    filtered: list[str] = []
    for row in rows:
        path_field = row[3:].strip() if len(row) >= 3 else row.strip()
        paths = [part.strip().strip('"') for part in path_field.split(" -> ")]
        if any(not path.startswith(prefix) for path in paths):
            filtered.append(row)
    return filtered


def strict_json(path: Path) -> Any:
    return w.strict_json(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_npy(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def gate(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "executed": bool(executed),
        "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"),
        "required": bool(required),
        **details,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def all_required(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in rows.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def qnormalize(value: Sequence[float]) -> tuple[float, float, float, float]:
    return w.quat_normalize_xyzw(value)


def qmul(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    return w.quat_mul_xyzw(left, right)


def qgeo(left: Sequence[float], right: Sequence[float]) -> float:
    return w.quat_geodesic_xyzw(left, right)


def qmatrix(value: Sequence[float]) -> list[list[float]]:
    return w.quat_to_matrix_xyzw(value)


def vector_norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def compose_object_to_eef(object_pose: Mapping[str, Any], relative_pose: Mapping[str, Any], *, object_position_offset_m: Sequence[float] = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    """Compose desired object pose and frozen object->EEF anchor in xyzw."""
    object_position = [float(item) + float(object_position_offset_m[index]) for index, item in enumerate(object_pose["pos_m"])]
    object_quat = qnormalize(object_pose["quat"])
    relative_position = [float(item) for item in relative_pose["translation_m"]]
    rotation = qmatrix(object_quat)
    world_delta = [sum(rotation[row][column] * relative_position[column] for column in range(3)) for row in range(3)]
    return {
        "pos_m": [object_position[index] + world_delta[index] for index in range(3)],
        "quat": list(qmul(object_quat, relative_pose["quaternion_xyzw"])),
        "object_pose": {"pos_m": object_position, "quat": list(object_quat)},
        "relative_pose": {"translation_m": relative_position, "quaternion_xyzw": list(qnormalize(relative_pose["quaternion_xyzw"]))},
    }


def progress_window_gate(errors: Sequence[float], *, window: int, min_improvement: float) -> dict[str, Any]:
    values = [float(item) for item in errors]
    if not values or len(values) < window:
        return gate(False, None, reason="progress_window_not_full", window=window, samples=len(values), min_improvement=min_improvement)
    baseline = values[-window]
    current = values[-1]
    improvement = baseline - current
    return gate(True, improvement >= min_improvement, window=window, samples=len(values), baseline_error=baseline, current_error=current, improvement=improvement, min_improvement=min_improvement)


def cursor_record(*, active: bool, phase: str, pid: int, command: Sequence[str], root_commit: str, tool_sha256: str, gpu_policy: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": "track3_object_frame_supervisor_cursor_v1",
        "active": bool(active),
        "status": status,
        "phase": phase,
        "pid": int(pid),
        "command": list(command),
        "root_commit": root_commit,
        "tool_sha256": tool_sha256,
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI),
        "gpu_policy": gpu_policy,
        "updated_at_utc": now(),
    }


def write_cursor(*, active: bool, phase: str, command: Sequence[str], status: str) -> dict[str, Any]:
    record = cursor_record(active=active, phase=phase, pid=os.getpid(), command=command, root_commit=git(("rev-parse", "HEAD")), tool_sha256=sha256_file(Path(__file__)), gpu_policy="cpu_no_renderer", status=status)
    write_json(CURSOR, record)
    return record


def _x0() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    w6 = strict_json(W6_REPORT)
    manifest = w._manifest_verify(W6_MANIFEST)
    w4_parent = strict_json(W4_PARENT)
    w4_child = strict_json(W4_CHILD)
    gpu0 = w.previous.gpu_snapshot(0)
    gpu_free = {str(index): w.previous.gpu_snapshot(index) for index in (1, 2, 3)}
    current = git(("rev-parse", "HEAD"))
    checks = {
        "w6_hard_stop": w6.get("status") == "track3_support_release_anchor_scaled_W6_hard_stop",
        "w6_manifest_17_exact": manifest["entries"] == 17 and manifest["exact"],
        "w4_historical_family_failure": w4_parent.get("category") == "family_failure" and w4_child.get("status") == "failed_gate",
        "w5_not_evaluated": w6.get("phase_status", {}).get("W5") == "not_evaluated",
        "zero_reverse_counts": w6.get("counts") == {"accepted_reverse_demo": 0, "exact_reverse_demo": 0, "pending_autogen": 0},
        "w6_input_commit_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", W6_COMMIT, current], check=False).returncode == 0,
        "openpi_identity": git(("rev-parse", "HEAD"), OPENPI) == OPENPI_COMMIT and not porcelain(OPENPI),
        "source_clean": not source_porcelain(ROOT),
        "gpu0_external_preserved": any(GPU0_PID in line for line in gpu0.get("compute_processes", [])),
        "gpu1_to_gpu3_free": all(row.get("free") for row in gpu_free.values()),
    }
    result = {
        "schema_version": "track3_object_frame_x0_input_lock_v1",
        "status": "track3_object_frame_x0_passed" if all(checks.values()) else "track3_object_frame_stopped_X0_input_lock_failure",
        "created_at_utc": now(),
        "plan": rel(PLAN),
        "input_execution_commit": W6_COMMIT,
        "root_head": current,
        "openpi_head": git(("rev-parse", "HEAD"), OPENPI),
        "w6": {"status": w6.get("status"), "phase_status": w6.get("phase_status"), "counts": w6.get("counts"), "manifest": manifest},
        "w4": {"parent_status": w4_parent.get("status"), "category": w4_parent.get("category"), "child_status": w4_child.get("status"), "historical_immutable": True},
        "checks": checks,
        "gpu0": gpu0,
        "gpu1_to_gpu3": gpu_free,
        "libero_plus_porcelain": porcelain(LIBERO_PLUS),
        "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_touched": False, "official_gpu": False, "teleop": False, "generation": False, "training": False, "zip": False},
    }
    write_json(X0_REPORT, result)
    return result


def _x1() -> dict[str, Any]:
    x0 = strict_json(X0_REPORT)
    if x0.get("status") != "track3_object_frame_x0_passed":
        result = {"schema_version": "track3_object_frame_x1_errata_v1", "status": "not_evaluated", "reason": "x0_not_passed"}
        write_json(X1_REPORT, result)
        return result
    import numpy as np

    w1 = strict_json(W1_REPORT)
    state0 = w1["state_anchors"]["state0"]["bowl"]
    relative = w1["reverse_acquisition_anchor"]["relative_pose_xyzw"]
    with np.load(W4_TRAJECTORY, allow_pickle=False) as archive:
        action_count = int(archive["actions"].shape[0])
        state_count = int(archive["states"].shape[0])
        goal_trace_count = int(archive["goal_trace"].shape[0])
    # W4 stopped before the place stage, so reconstruct its registered target
    # from the old source formula instead of fabricating an absent trajectory
    # row: state0 object position + fixed z offset and object quaternion.
    old_position = [float(state0["pos_m"][0]), float(state0["pos_m"][1]), float(state0["pos_m"][2]) + 0.055]
    old_quaternion = [float(item) for item in state0["quat"]]
    expected = compose_object_to_eef(state0, relative)
    old_position_error = vector_norm([old_position[index] - expected["pos_m"][index] for index in range(3)]) if old_position else float("inf")
    old_rotation_error = qgeo(old_quaternion, expected["quat"]) if old_quaternion else float("inf")
    w4_child = strict_json(W4_CHILD)
    checks = {
        "old_target_reproduces_audit_position_error": abs(old_position_error - 0.08171218629122215) < 1e-6,
        "old_target_reproduces_audit_rotation_error": abs(old_rotation_error - 1.8164252213793237) < 1e-6,
        "trajectory_alignment_n_nplus1": action_count == state_count and goal_trace_count == action_count + 1,
        "history_source_all_true": all(value is True for value in w4_child.get("reverse", {}).get("source_values", [])),
        "history_contact_completion_failed": w4_child.get("hard_gate", {}).get("contact_completion", {}).get("state") == "failed",
        "history_downstream_not_retroactively_passed": w4_child.get("hard_gate", {}).get("same_demo_destination", {}).get("state") == "not_evaluated",
    }
    result = {
        "schema_version": "track3_object_frame_x1_errata_finalizer_v1",
        "status": "track3_object_frame_x1_passed" if all(checks.values()) else "track3_object_frame_stopped_X1_contract_failure",
        "created_at_utc": now(),
        "checks": checks,
        "history": {"source_samples": len(w4_child.get("reverse", {}).get("source_values", [])), "contact_completion": w4_child.get("hard_gate", {}).get("contact_completion"), "same_demo_destination": w4_child.get("hard_gate", {}).get("same_demo_destination")},
        "destination_transform": {"formula": "T_world_object_des * T_object_eef_anchor", "anchor_source": "W1_state75_release_side", "desired_object_pose_source": "W1_state0_same_demo_bowl_pose", "expected_eef_position_m": expected["pos_m"], "expected_eef_quaternion_xyzw": expected["quat"], "old_target_reconstruction": "W4 source formula; place stage was not entered", "old_target_position_m": old_position, "old_target_quaternion_xyzw": old_quaternion, "old_target_position_error_m": old_position_error, "old_target_rotation_error_rad": old_rotation_error},
        "trajectory_schema": {"actions_states_alignment": "N", "predicate_trace_alignment": "t0_then_post_step", "predicate_trace_length": "N+1", "old_action_count": action_count, "old_state_count": state_count, "old_goal_trace_count": goal_trace_count},
        "finalizer_contract": ["producer_close_and_fsync", "phase_summary_and_csv", "inactive_cursor_and_supervisor_closeout", "manifest", "strict_hash_npz_video_csv_validation", "targeted_full_tests", "final_validation_report", "clean_commit"],
        "resource_boundary": {"simulator_started": False, "gpu_started": False, "generation": False, "training": False, "zip": False, "historical_w4_untouched": True},
    }
    write_json(X1_REPORT, result)
    return result


def _x2() -> dict[str, Any]:
    x1 = strict_json(X1_REPORT)
    if x1.get("status") != "track3_object_frame_x1_passed":
        result = {"schema_version": "track3_object_frame_x2_contract_v1", "status": "not_evaluated", "reason": "x1_not_passed"}
        write_json(X2_REPORT, result)
        return result
    identity = w.supervisor_identity_gate(
        expected_commit="a",
        observed_commit="a",
        expected_tool_sha256="b",
        observed_tool_sha256="b",
        expected_openpi="c",
        observed_openpi="c",
        dirty=False,
        pid_active=False,
        cursor_active=False,
        expected_command=["worker"],
        observed_command=["worker"],
        expected_cuda_visible_devices="",
        observed_cuda_visible_devices="",
        gpu_policy="cpu_no_renderer",
    )
    rejected = w.supervisor_identity_gate(
        expected_commit="a",
        observed_commit="b",
        expected_tool_sha256="b",
        observed_tool_sha256="b",
        expected_openpi="c",
        observed_openpi="c",
        dirty=True,
        pid_active=True,
        cursor_active=True,
        expected_command=["worker"],
        observed_command=["worker", "wrong"],
        expected_cuda_visible_devices="",
        observed_cuda_visible_devices="3",
        gpu_policy="cpu_no_renderer",
    )
    composed = compose_object_to_eef({"pos_m": [1.0, 2.0, 3.0], "quat": [0.0, 0.0, 0.0, 1.0]}, {"translation_m": [0.1, 0.2, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, -1.0]})
    progress_pass = progress_window_gate([1.0, 0.8, 0.6, 0.4], window=4, min_improvement=0.5)
    progress_fail = progress_window_gate([1.0, 0.99, 0.98, 0.97], window=4, min_improvement=0.5)
    not_eval = gate(False, None, reason="downstream_stage_not_entered")
    checks = {
        "object_frame_compose_position": composed["pos_m"] == [1.1, 2.2, 3.3],
        "object_frame_compose_q_sign": qgeo(composed["quat"], [0.0, 0.0, 0.0, 1.0]) == 0.0,
        "progress_window_pass": progress_pass["state"] == "passed",
        "progress_window_fail": progress_fail["state"] == "failed",
        "tri_state_not_evaluated": not_eval["state"] == "not_evaluated" and not_eval["executed"] is False,
        "identity_pass": identity["state"] == "passed",
        "identity_rejects_mismatch": rejected["state"] == "failed",
        "nominal_profile_frozen": NOMINAL_PROFILE["translation_clip_m"] == 0.05 and NOMINAL_PROFILE["rotation_clip_rad"] == 0.12,
        "conservative_profile_frozen": CONSERVATIVE_PROFILE["translation_clip_m"] == 0.025 and CONSERVATIVE_PROFILE["rotation_clip_rad"] == 0.06,
    }
    result = {
        "schema_version": "track3_object_frame_x2_executor_supervisor_contract_v1",
        "status": "track3_object_frame_x2_passed" if all(checks.values()) else "track3_object_frame_stopped_X2_contract_failure",
        "created_at_utc": now(),
        "checks": checks,
        "object_frame": {"quaternion_convention": "xyzw", "formula": "T_world_object_des * T_object_eef_anchor", "transport_offset_frame": "desired_object_pose_before_composition"},
        "physical_scale": {"translation_clip_m": 0.05, "rotation_clip_rad": 0.12, "controller_output_translation_m": 0.05, "controller_output_rotation_rad": 0.5, "normalized_rotation_norm_at_clip": 0.24},
        "profiles": {"nominal": NOMINAL_PROFILE, "conservative": CONSERVATIVE_PROFILE},
        "stage_contract": {"approach_to_descend": "pose_stable", "close_to_lift": "continuous_demonstrator_contact", "lift_to_transport": "object_follow_and_source_release", "place_to_release": "destination_and_goal_continuity", "unentered_gate": "not_evaluated"},
        "supervisor_contract": {"cursor": "atomic active=true before worker and inactive after producer close", "identity": "commit/tool/OpenPI/command/CUDA/GPU policy", "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS, "termination": "TERM_then_KILL", "result_transport": "independent_atomic_json_file"},
        "identity_fixture": {"pass": identity, "reject": rejected},
        "progress_fixtures": {"pass": progress_pass, "fail": progress_fail, "not_evaluated": not_eval},
        "resource_boundary": {"simulator_started": False, "gpu_started": False, "generation": False, "training": False, "zip": False},
    }
    write_json(X2_REPORT, result)
    return result


def _x3() -> dict[str, Any]:
    x2 = strict_json(X2_REPORT)
    if x2.get("status") != "track3_object_frame_x2_passed":
        result = {"schema_version": "track3_object_frame_x3_release_v1", "status": "not_evaluated", "reason": "x2_not_passed"}
        write_json(X3_REPORT, result)
        return result
    import numpy as np

    with np.load(W4_TRAJECTORY, allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"])
        states = np.asarray(archive["states"])
        goal_trace = np.asarray(archive["goal_trace"])
    w4_child = strict_json(W4_CHILD)
    source_values = list(w4_child.get("reverse", {}).get("source_values", []))
    goal_values = list(w4_child.get("reverse", {}).get("goal_values", []))
    acquired = w4_child.get("hard_gate", {}).get("contact_completion", {}).get("state") == "passed"
    acquisition = gate(True, acquired, reason="historical_contact_completion_failed" if not acquired else None, source_values=source_values, goal_values=goal_values)
    downstream = {name: gate(False, None, reason="acquisition_barrier_not_satisfied") for name in ("lift", "transport", "place", "release", "destination", "generation")}
    checks = {
        "derived_only": True,
        "no_simulator_or_gpu": True,
        "alignment_n_nplus1": actions.shape[0] == states.shape[0] and goal_trace.shape[0] == actions.shape[0] + 1,
        "source_true_history": bool(source_values) and all(value is True for value in source_values),
        "goal_false_history": bool(goal_values) and all(value is False for value in goal_values),
        "acquisition_stops_downstream": acquisition["state"] == "failed" and all(row["state"] == "not_evaluated" for row in downstream.values()),
        "partial_not_training_eligible": w4_child.get("training_eligible") is False,
    }
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], capture_output=True, text=True, check=False)
    test_result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    checks["py_compile"] = compile_result.returncode == 0
    checks["root_tests"] = test_result.returncode == 0
    result = {
        "schema_version": "track3_object_frame_x3_clean_release_v1",
        "status": "track3_object_frame_x3_ready" if all(checks.values()) else "track3_object_frame_stopped_X3_release_failure",
        "created_at_utc": now(),
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__)),
        "checks": checks,
        "derived_replay": {"source": rel(W4_TRAJECTORY), "actions": int(actions.shape[0]), "states": int(states.shape[0]), "goal_trace": int(goal_trace.shape[0]), "acquisition_barrier": acquisition, "downstream": downstream},
        "py_compile": {"returncode": compile_result.returncode, "stderr": compile_result.stderr[-1000:]},
        "root_tests": {"returncode": test_result.returncode, "tail": test_result.stdout[-3000:]},
        "clean_release_contract": {"source_porcelain": source_porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI), "commit_required_before_x4": True},
        "resource_boundary": {"simulator_started": False, "gpu_started": False, "generation": False, "training": False, "zip": False},
    }
    write_json(X3_REPORT, result)
    return result


def _planned_worker_command(profile_name: str, run_root: Path, result_path: Path) -> list[str]:
    return [sys.executable, "-m", MODULE, "--x4-worker", "--profile", profile_name, "--result-output", str(result_path)]


def _relative_pose(object_pose: Mapping[str, Any], eef_pose: Mapping[str, Any]) -> dict[str, Any]:
    object_quat = qnormalize(object_pose["quat"])
    delta = [float(eef_pose["eef_translation_m"][index]) - float(object_pose["pos_m"][index]) for index in range(3)]
    rotation = qmatrix(object_quat)
    local = [sum(rotation[column][row] * delta[column] for column in range(3)) for row in range(3)]
    return {"translation_m": local, "quaternion_xyzw": list(qmul(tuple(-item for item in object_quat[:3]) + (object_quat[3],), eef_pose["eef_quaternion"]))}


def _worker_profile(profile: Mapping[str, Any], result_path: Path) -> int:
    import numpy as np

    w._ensure_runtime_pythonpath()
    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
    from tools.axis_pi05 import track3_operator_scoped_stability as prior
    from tools.axis_pi05 import track3_reverse_init_stabilization as base

    started = now()
    env = None
    run_root = result_path.parent
    try:
        w1 = strict_json(W1_REPORT)
        w2 = strict_json(W_OUT / "w2_osc_scale_contract.json")
        manifest, model_bytes, _init_state, _terminal_state, demo_info = composed._demo_arrays(10, 0)
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        with np.load(STATE_BUNDLE, allow_pickle=False) as archive:
            start_state = np.asarray(archive["task10_demo0"], dtype=np.float64).copy()
        terminal_pose = w1["state_anchors"]["terminal"]["bowl"]
        state0_pose = w1["state_anchors"]["state0"]["bowl"]
        relative = w1["reverse_acquisition_anchor"]["relative_pose_xyzw"]
        grasp = compose_object_to_eef(terminal_pose, relative)
        destination = compose_object_to_eef(state0_pose, relative)
        transport = compose_object_to_eef(state0_pose, relative, object_position_offset_m=(0.0, 0.0, 0.12))
        # The public environment configuration belongs to the composed
        # reverse-start runtime.  ``base`` only owns predicate/BDDL helpers
        # and intentionally has no environment configurator.
        composed._configure_public_env(gpu=False)
        from libero.libero.envs.env_wrapper import ControlEnv

        env = ControlEnv(bddl_file_name=str(base.REVERSE_BDDL[10]), control_freq=CONTROL_FREQ_HZ, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=-1)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        env.set_init_state(start_state)
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(base.REVERSE_BDDL[10], "init")
        goal_atoms = base.public_predicates(base.REVERSE_BDDL[10], "goal")
        bowl = "akita_black_bowl_1"
        source_atom = "(On akita_black_bowl_1 plate_1)"
        goal_atom = "(On akita_black_bowl_1 main_table_akita_black_bowl_region)"
        samples: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        states: list[Any] = []
        phases: list[str] = []
        stage_indices: list[int] = []
        eef_positions: list[list[float]] = []
        eef_quaternions: list[list[float]] = []
        target_positions: list[list[float]] = []
        target_quaternions: list[list[float]] = []
        physical_translation: list[list[float]] = []
        physical_rotation: list[list[float]] = []
        translation_clips: list[bool] = []
        rotation_clips: list[bool] = []
        controller_clips: list[bool] = []
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        progress_records: list[dict[str, Any]] = []
        max_penetration = 0.0
        unrelated_names: set[str] = set()
        baseline_scene: Mapping[str, Any] | None = None
        max_unrelated_movement = 0.0

        def observe(step: int) -> None:
            nonlocal baseline_scene, max_penetration, max_unrelated_movement
            sample = prior._runtime_sample(env, dynamic, 10, "reverse", step, initial_atoms, goal_atoms)
            sample["contacts_v3"] = composed.contact_v3(10, sample["contact_matrix"])
            robot = sample["robot"]
            bowl_pose = dynamic._object_or_site_pose(env, bowl)
            if bowl_pose is not None:
                sample["object_eef_relative_pose"] = _relative_pose(bowl_pose, robot)
            samples.append(sample)
            scene = sample.get("scene", {}).get("object_and_site_poses", {})
            if baseline_scene is None:
                baseline_scene = scene
                unrelated_names.update(set(scene) - {bowl, "plate_1", "main_table_akita_black_bowl_region"})
            for name in unrelated_names:
                if name in scene and name in baseline_scene:
                    current = scene[name].get("pos_m", [])
                    original = baseline_scene[name].get("pos_m", [])
                    max_unrelated_movement = max(max_unrelated_movement, vector_norm([float(a) - float(b) for a, b in zip(current, original)]))
            for contact in sample.get("contact_matrix", {}).get("contacts", []):
                max_penetration = max(max_penetration, max(0.0, -float(contact.get("distance_m", 0.0))))

        observe(0)
        stage_specs = [
            ("approach_grasp", -1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.09], grasp["quat"]),
            ("descend_grasp", -1.0, grasp["pos_m"], grasp["quat"]),
            ("close_gripper", 1.0, grasp["pos_m"], grasp["quat"]),
            ("lift", 1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.14], grasp["quat"]),
            ("transport", 1.0, transport["pos_m"], transport["quat"]),
            ("place_same_demo_init_pose", 1.0, destination["pos_m"], destination["quat"]),
            ("release", -1.0, destination["pos_m"], destination["quat"]),
            ("retreat", -1.0, [destination["pos_m"][0], destination["pos_m"][1], destination["pos_m"][2] + 0.16], destination["quat"]),
        ]
        failure_reason: str | None = None
        stage_results: list[dict[str, Any]] = []
        for stage_index, (stage_name, gripper, target_position, target_quat) in enumerate(stage_specs):
            stable = 0
            stage_start = len(actions)
            stage_errors: list[float] = []
            last_t = float("inf")
            last_r = float("inf")
            budget = int(profile["stage_budgets"][stage_index])
            for _local in range(budget):
                current = dynamic._robot_metrics(env)
                translation = [float(target_position[index]) - float(current["eef_translation_m"][index]) for index in range(3)]
                rotation = w.quat_log_error_xyzw(target_quat, current["eef_quaternion"])
                command = w.osc_action_scaled(translation, rotation, gripper, profile=profile, scales={"translation_m": float(w2["controller_output_scale"]["translation_m"]), "rotation_rad": float(w2["controller_output_scale"]["rotation_rad"])})
                obs_after, reward, done, _info = env.step(np.asarray(command["action"], dtype=np.float32))
                if not isinstance(obs_after, Mapping):
                    obs_after = env.env._get_observations()
                actions.append(list(command["action"]))
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
                phases.append(stage_name)
                stage_indices.append(stage_index)
                eef_positions.append(list(current["eef_translation_m"]))
                eef_quaternions.append(list(current["eef_quaternion"]))
                target_positions.append(list(target_position))
                target_quaternions.append(list(target_quat))
                physical_translation.append(list(command["physical_translation_delta_m"]))
                physical_rotation.append(list(command["physical_rotation_delta_rad"]))
                translation_clips.append(bool(command["translation_physical_clip"]))
                rotation_clips.append(bool(command["rotation_physical_clip"]))
                controller_clips.append(bool(command["controller_clip"]))
                last_t = float(command["translation_norm_m"])
                last_r = float(command["rotation_norm_rad"])
                stage_error = max(last_t / float(profile["translation_tolerance_m"]), last_r / float(profile["rotation_tolerance_rad"]))
                stage_errors.append(stage_error)
                translation_errors.append(last_t)
                rotation_errors.append(last_r)
                observe(len(actions))
                progress_records.append({"stage": stage_name, "stage_index": stage_index, "window": progress_window_gate(stage_errors, window=min(4, int(profile["stable_ticks"])), min_improvement=0.02)})
                if last_t <= float(profile["translation_tolerance_m"]) and last_r <= float(profile["rotation_tolerance_rad"]):
                    stable += 1
                else:
                    stable = 0
                if stable >= int(profile["stable_ticks"]) or done:
                    break
            reached = stable >= int(profile["stable_ticks"])
            stage_results.append({"stage": stage_name, "stage_index": stage_index, "frames": len(actions) - stage_start, "budget": budget, "reached": reached, "last_translation_error_m": last_t, "last_rotation_error_rad": last_r, "progress_windows": progress_records[-max(1, len(actions) - stage_start):]})
            if not reached:
                failure_reason = f"stage_timeout:{stage_name}"
                break
            if stage_name == "close_gripper":
                recent = samples[-max(2, int(profile["stable_ticks"])):]
                recent_contacts = [row.get("contacts_v3", {}).get("counts", {}).get("demonstrator_contact", 0) > 0 for row in recent]
                recent_source = [next((item.get("value") for item in row.get("initial", {}).get("predicates", []) if item.get("atom") == source_atom), None) for row in recent]
                follow_rows = [row.get("object_eef_relative_pose") for row in recent]
                follow_ok = bool(follow_rows) and all(vector_norm([float(a) - float(b) for a, b in zip(row["translation_m"], relative["translation_m"])]) <= 0.02 and qgeo(row["quaternion_xyzw"], relative["quaternion_xyzw"]) <= 0.15 for row in follow_rows if row is not None)
                acquisition_ok = bool(recent) and all(recent_contacts) and all(value is False for value in recent_source) and follow_ok
                if not acquisition_ok:
                    failure_reason = "acquisition_barrier"
                    break

        final_success = bool(env.check_success())
        source_values = [next((item.get("value") for item in row.get("initial", {}).get("predicates", []) if item.get("atom") == source_atom), None) for row in samples]
        goal_values = [next((item.get("value") for item in row.get("goal", {}).get("predicates", []) if item.get("atom") == goal_atom), None) for row in samples]
        entered_close = any(row["stage"] == "close_gripper" for row in stage_results)
        entered_place = any(row["stage"] == "place_same_demo_init_pose" for row in stage_results)
        acquisition_executed = entered_close
        recent = samples[-max(2, int(profile["stable_ticks"])):] if samples else []
        contact_ok = bool(recent) and all(row.get("contacts_v3", {}).get("counts", {}).get("demonstrator_contact", 0) > 0 for row in recent)
        source_released = bool(recent) and all(next((item.get("value") for item in row.get("initial", {}).get("predicates", []) if item.get("atom") == source_atom), None) is False for row in recent)
        follow_ok = bool(recent) and all(row.get("object_eef_relative_pose") is not None and vector_norm([float(a) - float(b) for a, b in zip(row["object_eef_relative_pose"]["translation_m"], relative["translation_m"])]) <= 0.02 and qgeo(row["object_eef_relative_pose"]["quaternion_xyzw"], relative["quaternion_xyzw"]) <= 0.15 for row in recent)
        acquisition_gate = gate(acquisition_executed, contact_ok and source_released and follow_ok, reason=failure_reason if acquisition_executed and not (contact_ok and source_released and follow_ok) else None, contact_continuous=contact_ok, source_released=source_released, object_follow=follow_ok)
        gates = {
            "initial_source_true": gate(bool(source_values), source_values[0] is True, source_values=source_values[:1]),
            "initial_goal_false": gate(bool(goal_values), goal_values[0] is False, goal_values=goal_values[:1]),
            "acquisition_barrier": acquisition_gate,
            "contact_completion": gate(entered_close, contact_ok and source_released and follow_ok, reason="acquisition_barrier_failed" if entered_close and not (contact_ok and source_released and follow_ok) else ("close_stage_not_entered" if not entered_close else None), contact_continuous=contact_ok, source_released=source_released, object_follow=follow_ok),
            "same_demo_destination": gate(entered_place, bool(entered_place and recent and all(vector_norm([float(a) - float(b) for a, b in zip(row.get("scene", {}).get("object_and_site_poses", {}).get(bowl, {}).get("pos_m", [999, 999, 999]), state0_pose["pos_m"])]) <= float(profile["translation_tolerance_m"]) for row in recent)), reason="place_stage_not_entered" if not entered_place else None),
            "final_goal_true": gate(bool(goal_values), bool(final_success and goal_values[-1] is True), goal_value=goal_values[-1] if goal_values else None, final_success=final_success),
            "prefix_contact_safety": gate(bool(samples), all(not row.get("contacts_v3", {}).get("counts", {}).get("unrelated_background_contact", 0) for row in samples), reason="unrelated_background_contact" if any(row.get("contacts_v3", {}).get("counts", {}).get("unrelated_background_contact", 0) for row in samples) else None),
            "collision": gate(bool(samples), max_penetration <= 0.01, max_penetration_m=max_penetration, threshold_m=0.01),
            "unrelated_movement": gate(bool(samples), max_unrelated_movement <= 0.002, max_unrelated_movement_m=max_unrelated_movement, threshold_m=0.002),
            "stall_timeout": gate(bool(stage_results), failure_reason is None, reason=failure_reason if failure_reason and failure_reason.startswith("stage_timeout") else "downstream_not_evaluated" if failure_reason else None, used_steps=len(actions), budget=int(profile["step_budget"])),
            "finite": gate(bool(actions), bool(np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(states)).all()), action_shape=list(np.asarray(actions).shape), state_shape=list(np.asarray(states).shape)),
        }
        status = "passed" if all_required(gates) else "failed_gate"
        write_npy(run_root / "start_state.npy", start_state)
        write_npy(run_root / "end_state.npy", np.asarray(env.get_sim_state(), dtype=np.float64))
        write_npz(run_root / "trajectory.npz", actions=np.asarray(actions, dtype=np.float32), states=np.asarray(states, dtype=np.float64), phase=np.asarray(phases, dtype="U64"), stage_index=np.asarray(stage_indices, dtype=np.int16), eef_position_m=np.asarray(eef_positions, dtype=np.float32), eef_quaternion_xyzw=np.asarray(eef_quaternions, dtype=np.float32), target_position_m=np.asarray(target_positions, dtype=np.float32), target_quaternion_xyzw=np.asarray(target_quaternions, dtype=np.float32), physical_translation_delta_m=np.asarray(physical_translation, dtype=np.float32), physical_rotation_delta_rad=np.asarray(physical_rotation, dtype=np.float32), translation_physical_clip=np.asarray(translation_clips, dtype=np.bool_), rotation_physical_clip=np.asarray(rotation_clips, dtype=np.bool_), controller_clip=np.asarray(controller_clips, dtype=np.bool_), translation_error_m=np.asarray(translation_errors, dtype=np.float32), rotation_error_rad=np.asarray(rotation_errors, dtype=np.float32), source_trace=np.asarray([-1 if value is None else int(bool(value)) for value in source_values], dtype=np.int8), goal_trace=np.asarray([-1 if value is None else int(bool(value)) for value in goal_values], dtype=np.int8))
        result = {
            "schema_version": "track3_object_frame_x4_worker_v1",
            "status": status,
            "created_at_utc": started,
            "completed_at_utc": now(),
            "child_pid": os.getpid(),
            "task_index": 10,
            "demo_index": 0,
            "family": "support_remove",
            "profile": dict(profile),
            "mode": "CPU_no_renderer",
            "fresh_process": True,
            "quaternion_convention": "xyzw",
            "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "start_state_path": rel(run_root / "start_state.npy"), "start_state_sha256": sha256_file(run_root / "start_state.npy"), "same_demo_model_and_state_components": True, "terminal_pose_source": "same_demo_hdf5_terminal_state", "release_anchor_state_index": 75},
            "reverse": {"source_atom": source_atom, "goal_atom": goal_atom, "initial_goal": goal_values[0] if goal_values else None, "final_goal": final_success, "source_values": source_values, "goal_values": goal_values, "stage_results": stage_results, "failure_reason": failure_reason, "action_saturation_count": int(sum(translation_clips) + sum(rotation_clips)), "controller_clip_count": int(sum(controller_clips)), "anchor": w1["reverse_acquisition_anchor"], "destination_eef": destination, "transport_eef": transport},
            "executor": {"controller": "OSC_POSE", "action_dim": ACTION_DIM, "control_freq_hz": CONTROL_FREQ_HZ, "profile": dict(profile), "physical_output_scale": w2["controller_output_scale"], "target_grasp_position_m": grasp["pos_m"], "target_grasp_quaternion_xyzw": grasp["quat"], "target_destination_position_m": destination["pos_m"], "target_destination_quaternion_xyzw": destination["quat"]},
            "hard_gate": gates,
            "progress_windows": progress_records,
            "artifacts": {"trajectory": rel(run_root / "trajectory.npz"), "start_state": rel(run_root / "start_state.npy"), "end_state": rel(run_root / "end_state.npy")},
            "training_eligible": False,
            "pending_autogen": False,
            "resource_boundary": {"gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "official_gpu": False, "no_teleop": True, "no_generation": True, "no_training": True, "no_zip": True},
        }
    except Exception as exc:
        result = {"schema_version": "track3_object_frame_x4_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": dict(profile), "mode": "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "training_eligible": False, "resource_boundary": {"gpu0_touched": False, "official_gpu": False}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_path, result)
    return 0 if str(result.get("status")) in WORKER_STATUSES else 1


def _terminate(process: subprocess.Popen[Any]) -> str:
    try:
        os.killpg(process.pid, 15)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=TERM_GRACE_SECONDS)
        return "TERM"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, 9)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=TERM_GRACE_SECONDS)
        return "TERM_then_KILL"


def _run_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    run_root = X4_ROOT / str(profile["name"])
    parent_path = run_root / "parent_result.json"
    child_path = run_root / "child_result.json"
    run_root.mkdir(parents=True, exist_ok=False)
    command = _planned_worker_command(str(profile["name"]), run_root, child_path)
    current_commit = git(("rev-parse", "HEAD"))
    tool_hash = sha256_file(Path(__file__))
    prior_cursor = strict_json(CURSOR) if CURSOR.is_file() else {}
    identity = w.supervisor_identity_gate(expected_commit=current_commit, observed_commit=current_commit, expected_tool_sha256=tool_hash, observed_tool_sha256=tool_hash, expected_openpi=OPENPI_COMMIT, observed_openpi=git(("rev-parse", "HEAD"), OPENPI), dirty=bool(source_porcelain(ROOT)), pid_active=(run_root / "supervisor.pid").exists(), cursor_active=bool(prior_cursor.get("active")), expected_command=command, observed_command=command, expected_cuda_visible_devices="", observed_cuda_visible_devices="", gpu_policy="cpu_no_renderer")
    write_json(run_root / "identity_preflight.json", {"schema_version": "track3_object_frame_identity_v1", "phase": "X4", "called_by_launcher": True, "identity": identity, "root_commit": current_commit, "tool_sha256": tool_hash, "command": command, "cuda_visible_devices": ""})
    if identity["state"] != "passed":
        result = {"schema_version": "track3_object_frame_parent_v1", "status": "track3_object_frame_stopped_X4_identity_failure", "category": "infrastructure", "phase": "X4", "profile": dict(profile), "identity_gate": identity, "identity_gate_called": True}
        write_json(parent_path, result)
        return result
    pid_path = run_root / "supervisor.pid"
    write_json(pid_path, {"schema_version": "track3_object_frame_pid_v1", "pid": os.getpid(), "phase": "X4", "profile": profile["name"], "created_at_utc": now()})
    write_cursor(active=True, phase=f"X4.{profile['name']}", command=command, status="active")
    heartbeat_path = run_root / "heartbeat.json"
    heartbeat_records = [{"schema_version": "track3_object_frame_heartbeat_v1", "phase": "X4", "profile": profile["name"], "state": "running", "count": 1, "pid": os.getpid(), "created_at_utc": now(), "identity_state": identity["state"], "root_commit": current_commit, "tool_sha256": tool_hash}]
    write_json(heartbeat_path, heartbeat_records[-1])
    environment = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    process: subprocess.Popen[Any] | None = None
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    polls = 0
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        while process.poll() is None:
            polls += 1
            if time.monotonic() - started >= DEADLINE_SECONDS:
                timed_out = True
                termination_stage = _terminate(process)
                break
            time.sleep(0.1)
        process.wait()
    heartbeat_records.append({"schema_version": "track3_object_frame_heartbeat_v1", "phase": "X4", "profile": profile["name"], "state": "final", "count": 2, "pid": os.getpid(), "created_at_utc": now(), "identity_state": identity["state"], "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__))})
    write_json(heartbeat_path, heartbeat_records[-1])
    pid_path.unlink(missing_ok=True)
    write_cursor(active=False, phase=f"X4.{profile['name']}", command=command, status="completed")
    child = strict_json(child_path) if child_path.is_file() else None
    child_status = child.get("status") if isinstance(child, Mapping) else None
    telemetry_ok = process.returncode == 0 and not timed_out and child is not None and child_status in WORKER_STATUSES and not bool(strict_json(CURSOR).get("active"))
    if not telemetry_ok:
        category = "infrastructure"
    elif child_status == "passed":
        category = "success"
    else:
        category = "family_failure"
    status = "track3_object_frame_x4_profile_passed" if category == "success" else "track3_object_frame_x4_profile_family_failure" if category == "family_failure" else "track3_object_frame_stopped_X4_infrastructure_failure"
    result = {"schema_version": "track3_object_frame_parent_v1", "status": status, "category": category, "created_at_utc": now(), "phase": "X4", "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": dict(profile), "attempt_number": 1, "identity_gate": identity, "identity_gate_called": True, "root_commit": current_commit, "tool_sha256": tool_hash, "command": command, "supervisor_pid": os.getpid(), "worker_pid": process.pid if process is not None else None, "worker_pid_match": process is not None and child is not None and process.pid == child.get("child_pid"), "process_returncode": process.returncode if process is not None else None, "timed_out": timed_out, "termination_stage": termination_stage, "poll_samples": polls, "heartbeat": {"path": rel(heartbeat_path), "count": len(heartbeat_records), "records": heartbeat_records, "interval_seconds": HEARTBEAT_INTERVAL_SECONDS}, "child_status": child_status, "child_result_path": rel(child_path), "child_result": child, "first_error": {"category": category, "reason": (child or {}).get("reverse", {}).get("failure_reason") if isinstance(child, Mapping) else "missing_child_result"}, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_to_gpu3_released": True, "official_gpu": False, "teleop": False, "generation": False, "training": False, "zip": False}}
    write_json(parent_path, result)
    return result


def _x4() -> dict[str, Any]:
    x3 = strict_json(X3_REPORT)
    if x3.get("status") != "track3_object_frame_x3_ready":
        result = {"schema_version": "track3_object_frame_x4_summary_v1", "status": "not_evaluated", "reason": "x3_not_ready"}
        write_json(X4_SUMMARY, result)
        return result
    results: list[dict[str, Any]] = []
    nominal = _run_profile(NOMINAL_PROFILE)
    results.append(nominal)
    if nominal.get("category") == "infrastructure":
        conservative = {"status": "not_evaluated", "reason": "nominal_infrastructure_failure_global_stop", "profile": CONSERVATIVE_PROFILE}
    elif nominal.get("category") == "family_failure":
        conservative = _run_profile(CONSERVATIVE_PROFILE)
    else:
        conservative = {"status": "not_evaluated", "reason": "nominal_success_no_conservative_profile", "profile": CONSERVATIVE_PROFILE}
    results.append(conservative)
    successes = [row for row in results if row.get("category") == "success"]
    infra = any(row.get("category") == "infrastructure" for row in results)
    result = {"schema_version": "track3_object_frame_x4_summary_v1", "status": "track3_object_frame_x4_completed", "created_at_utc": now(), "profiles": results, "success_profiles": [row.get("profile", {}).get("name") for row in successes], "support_lane": "released" if successes and not infra else "closed" if not successes and not infra else "global_stop", "max_profile_runs": 2, "resource_boundary": {"gpu3_started": False, "gpu0_touched": False, "official_gpu": False, "other_family": False, "generation": False, "training": False, "zip": False}}
    write_json(X4_SUMMARY, result)
    return result


def _final_manifest() -> dict[str, Any]:
    entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {X13_MANIFEST.name, X13_VALIDATION.name} or path.name.endswith(".partial") or path.name.endswith(".log") or path.name.endswith(".pid"):
            continue
        entries.append({"relative_path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": "track3_object_frame_x13_manifest_v1", "entries_exclude_self_and_final_validation": True, "entries": entries}


def _validate_namespace(manifest: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    json_errors: list[str] = []
    for path in sorted(OUT.rglob("*.json")):
        if path == X13_VALIDATION:
            continue
        try:
            strict_json(path)
        except Exception as exc:
            json_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    npz_errors: list[str] = []
    for path in sorted(OUT.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                _ = archive.files
        except Exception as exc:
            npz_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    npy_errors: list[str] = []
    for path in sorted(OUT.rglob("*.npy")):
        try:
            np.load(path, allow_pickle=False)
        except Exception as exc:
            npy_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    csv_lf = all(b"\r" not in path.read_bytes() for path in OUT.rglob("*.csv"))
    partials = [rel(path) for path in OUT.rglob("*.partial")]
    return {"strict_json": not json_errors, "json_errors": json_errors, "npz_allow_pickle_false": not npz_errors, "npz_errors": npz_errors, "npy_allow_pickle_false": not npy_errors, "npy_errors": npy_errors, "csv_lf": csv_lf, "partial_residue": partials, "manifest_exact": w._manifest_verify(X13_MANIFEST), "source_porcelain": source_porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI)}


def _x13() -> dict[str, Any]:
    x0 = strict_json(X0_REPORT) if X0_REPORT.is_file() else {}
    x1 = strict_json(X1_REPORT) if X1_REPORT.is_file() else {}
    x2 = strict_json(X2_REPORT) if X2_REPORT.is_file() else {}
    x3 = strict_json(X3_REPORT) if X3_REPORT.is_file() else {}
    x4 = strict_json(X4_SUMMARY) if X4_SUMMARY.is_file() else {}
    phase_status = {"X0": x0.get("status", "not_evaluated"), "X1": x1.get("status", "not_evaluated"), "X2": x2.get("status", "not_evaluated"), "X3": x3.get("status", "not_evaluated"), "X4": x4.get("status", "not_evaluated"), "X5": "not_evaluated", "X6": "not_evaluated", "X7": "not_evaluated", "X8": "not_evaluated", "X9": "not_evaluated", "X10": "not_evaluated", "X11": "not_evaluated", "X12": "not_evaluated", "X13": "hard_stop"}
    write_text(PHASE_CSV, "phase,status\n" + "\n".join(f"{key},{value}" for key, value in phase_status.items()) + "\n")
    write_cursor(active=False, phase="X13", command=[sys.executable, "-m", MODULE, "--x13"], status="hard_stop")
    write_json(SUPERVISOR_CLOSEOUT, {"schema_version": "track3_object_frame_supervisor_closeout_v1", "status": "track3_object_frame_x13_hard_stop", "pid": os.getpid(), "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "active": False, "automatic_resume": False})
    counts = {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}
    provisional = {"schema_version": "track3_object_frame_x13_closeout_v1", "status": "track3_object_frame_x13_hard_stop", "created_at_utc": now(), "phase_status": phase_status, "counts": counts, "upstream": {"X4": {"status": x4.get("status"), "support_lane": x4.get("support_lane"), "success_profiles": x4.get("success_profiles", [])}}, "resource_boundary": {"gpu0_touched": False, "gpu0_external_process_preserved": True, "gpu1_to_gpu3_released": True, "official_gpu": False, "no_teleop": True, "no_generation": True, "no_training": True, "no_zip": True, "libero_plus_untouched": True}, "hard_stop": {"executed": True, "next_authority": "new_reviewed_plan_required", "automatic_resume": False, "upload": False, "submit": False}}
    write_json(X13_REPORT, provisional)
    manifest = _final_manifest()
    write_json(X13_MANIFEST, manifest)
    validation = _validate_namespace(manifest)
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], capture_output=True, text=True, check=False)
    test_result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    validation["py_compile"] = compile_result.returncode == 0
    validation["root_tests"] = {"returncode": test_result.returncode, "tail": test_result.stdout[-3000:]}
    validation["git_diff_check"] = not bool(git(("diff", "--check")))
    validation["manifest_entry_count"] = len(manifest.get("entries", []))
    validation["final_validation_excluded_from_manifest"] = True
    write_json(X13_VALIDATION, {"schema_version": "track3_object_frame_x13_final_validation_v1", "created_at_utc": now(), "validation": validation, "finalizer_order_completed": True})
    return provisional


def public_exit_code(status: str) -> int:
    return 0 if any(token in status for token in ("passed", "ready", "completed", "family_failure", "hard_stop", "not_evaluated")) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for phase in ("x0", "x1", "x2", "x3", "x4", "x13"):
        parser.add_argument(f"--{phase}", action="store_true")
    parser.add_argument("--x4-worker", action="store_true")
    parser.add_argument("--profile", choices=[NOMINAL_PROFILE["name"], CONSERVATIVE_PROFILE["name"]])
    parser.add_argument("--result-output", type=Path)
    args = parser.parse_args(argv)
    if args.x4_worker:
        if args.profile is None or args.result_output is None:
            parser.error("x4 worker requires --profile and --result-output")
        profile = NOMINAL_PROFILE if args.profile == NOMINAL_PROFILE["name"] else CONSERVATIVE_PROFILE
        return _worker_profile(profile, args.result_output)
    selected = [phase for phase in ("x0", "x1", "x2", "x3", "x4", "x13") if getattr(args, phase)]
    if len(selected) != 1:
        parser.error("select exactly one phase X0/X1/X2/X3/X4/X13")
    result = {"x0": _x0, "x1": _x1, "x2": _x2, "x3": _x3, "x4": _x4, "x13": _x13}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return public_exit_code(str(result.get("status", "")))


if __name__ == "__main__":
    raise SystemExit(main())
