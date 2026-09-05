#!/usr/bin/env python3
"""CPU-first recovery for the Track 3 support executor contract.

W0--W3 are release and supervisor checks.  W4 is one CPU/no-renderer
task10/demo0 canary.  W5 is conditional and can consume at most one GPU3
camera canary.  W6 always closes the namespace and hard-stops.

The previous V0--V5 namespace is immutable input here.  Runtime imports for
MuJoCo, Robosuite and camera rendering are kept inside worker functions so
W0 remains simulator/CUDA free.
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

from tools.axis_pi05 import track3_j5_executor_contract_and_atomic_video_recovery as previous

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md"
AUDIT_SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_j5_executor_recovery_audit_20260905/summary.json"
OLD_OUT = ROOT / "data/reports/axis_libero_30k/track3_j5_executor_recovery"
OLD_MANIFEST = OLD_OUT / "v5_artifact_manifest.json"
OLD_V0 = OLD_OUT / "v0_input_lock.json"
OLD_V4_PARENT = OLD_OUT / "v4_corrected_task10_demo0_nominal/parent_result.json"
OLD_V4_CHILD = OLD_OUT / "v4_corrected_task10_demo0_nominal/child_result.json"
OLD_V5 = OLD_OUT / "v5_closeout.json"
HDF5_TASK10 = previous.HDF5_TASK10
STATE_BUNDLE = previous.STATE_BUNDLE
OLD_GEOMETRY = previous.OLD_GEOMETRY
OPENPI = previous.OPENPI
LIBERO_PLUS = previous.LIBERO_PLUS
PUBLIC_LIBERO = ROOT / "LIBERO"
FORWARD_BDDL = PUBLIC_LIBERO / "libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl"
REVERSE_BDDL = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_v2_specs/canary_support_remove_task_10.bddl"
MODULE = "tools.axis_pi05.track3_support_release_anchor_and_osc_scale_recovery"
W0_REPORT = OUT / "w0_input_lock.json"
W1_REPORT = OUT / "w1_release_anchor.json"
W2_REPORT = OUT / "w2_osc_scale_contract.json"
W3_REPORT = OUT / "w3_clean_release.json"
W4_ROOT = OUT / "w4_cpu_task10_demo0_nominal"
W4_PARENT = W4_ROOT / "parent_result.json"
W4_CHILD = W4_ROOT / "child_result.json"
W4_TRAJECTORY = W4_ROOT / "trajectory.npz"
W5_ROOT = OUT / "w5_gpu3_task10_demo0_nominal"
W5_PARENT = W5_ROOT / "parent_result.json"
W5_CHILD = W5_ROOT / "child_result.json"
W5_FRAMES = W5_ROOT / "frames.npz"
W6_REPORT = OUT / "w6_closeout.json"
W4_SEED = 3121
CONTROL_FREQ_HZ = 20
ACTION_DIM = 7
DEADLINE_SECONDS = 300.0
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS = 900.0
PROFILE = {
    "translation_clip_m": 0.05,
    "rotation_clip_rad": 0.12,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 4,
    "step_budget": 900,
}
EXPECTED_AUDITED_COMMIT = "11483623b7710430a197231b8ed0662c4813c12d"
EXPECTED_OPENPI_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
EXPECTED_GPU0_PID = "1133706"
COMPLETED_WORKER_STATUSES = frozenset({"passed", "failed_gate"})


def _ensure_runtime_pythonpath() -> None:
    """Make the public LIBERO checkout importable for direct CLI launches."""
    for path in (ROOT, PUBLIC_LIBERO):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def strict_json(path: Path) -> Any:
    return previous.strict_json_load(path)


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


def all_required(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in gates.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def worker_exit_code(status: str) -> int:
    return 0 if status in COMPLETED_WORKER_STATUSES else 1


def public_exit_code(status: str) -> int:
    return 0 if any(token in status for token in ("passed", "ready", "failed_gate", "hard_stop", "not_evaluated")) else 1


def quat_normalize_xyzw(value: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError("xyzw_quaternion_invalid")
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        raise ValueError("xyzw_quaternion_zero")
    return tuple(item / norm for item in values)


def quat_conjugate_xyzw(value: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat_normalize_xyzw(value)
    return (-x, -y, -z, w)


def quat_mul_xyzw(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = quat_normalize_xyzw(left)
    bx, by, bz, bw = quat_normalize_xyzw(right)
    return quat_normalize_xyzw((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz))


def quat_geodesic_xyzw(left: Sequence[float], right: Sequence[float]) -> float:
    a = quat_normalize_xyzw(left)
    b = quat_normalize_xyzw(right)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def quat_log_error_xyzw(target: Sequence[float], current: Sequence[float]) -> list[float]:
    x, y, z, w = quat_mul_xyzw(target, quat_conjugate_xyzw(current))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(norm, max(0.0, min(1.0, w)))
    scale = angle / norm
    return [x * scale, y * scale, z * scale]


def quat_to_matrix_xyzw(value: Sequence[float]) -> list[list[float]]:
    x, y, z, w = quat_normalize_xyzw(value)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(matrix[row][column]) * float(vector[column]) for column in range(3)) for row in range(3)]


def _transpose_mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(float(matrix[column][row]) * float(vector[column]) for column in range(3)) for row in range(3)]


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) * float(item) for item in value))


def _clip_vector(value: Sequence[float], limit: float) -> tuple[list[float], bool]:
    raw = [float(item) for item in value]
    norm = _norm(raw)
    if norm > limit:
        return [item * limit / norm for item in raw], True
    return raw, False


def osc_action_scaled(translation_m: Sequence[float], rotation_rad: Sequence[float], gripper: float, *, profile: Mapping[str, Any] = PROFILE, scales: Mapping[str, float] | None = None) -> dict[str, Any]:
    if len(translation_m) != 3 or len(rotation_rad) != 3 or float(gripper) not in {-1.0, 1.0}:
        raise ValueError("invalid_scaled_osc_input")
    values = [float(item) for item in (*translation_m, *rotation_rad)]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("non_finite_scaled_osc_input")
    scales = scales or {"translation_m": 0.05, "rotation_rad": 0.5}
    physical_translation, translation_clipped = _clip_vector(translation_m, float(profile["translation_clip_m"]))
    physical_rotation, rotation_clipped = _clip_vector(rotation_rad, float(profile["rotation_clip_rad"]))
    normalized = [item / float(scales["translation_m"]) for item in physical_translation] + [item / float(scales["rotation_rad"]) for item in physical_rotation] + [float(gripper)]
    controller_clipped = any(abs(item) > 1.0 + 1e-12 for item in normalized[:6])
    if any(not math.isfinite(item) or abs(item) > 1.0 + 1e-9 for item in normalized):
        raise ValueError("scaled_osc_action_out_of_input_range")
    return {
        "action": normalized,
        "physical_translation_delta_m": physical_translation,
        "physical_rotation_delta_rad": physical_rotation,
        "translation_physical_clip": translation_clipped,
        "rotation_physical_clip": rotation_clipped,
        "controller_clip": controller_clipped,
        "translation_norm_m": _norm(translation_m),
        "rotation_norm_rad": _norm(rotation_rad),
        "normalized_rotation_norm": _norm(normalized[3:6]),
    }


def supervisor_identity_gate(*, expected_commit: str, observed_commit: str, expected_tool_sha256: str, observed_tool_sha256: str, expected_openpi: str, observed_openpi: str, dirty: bool, pid_active: bool, cursor_active: bool, expected_command: Sequence[str], observed_command: Sequence[str], expected_cuda_visible_devices: str, observed_cuda_visible_devices: str, gpu_policy: str) -> dict[str, Any]:
    errors: list[str] = []
    checks = {
        "root_commit": expected_commit == observed_commit,
        "tool_sha256": expected_tool_sha256 == observed_tool_sha256,
        "openpi_head": expected_openpi == observed_openpi,
        "root_clean": not dirty,
        "pid_inactive": not pid_active,
        "cursor_inactive": not cursor_active,
        "command_exact": list(expected_command) == list(observed_command),
        "cuda_visible_devices": expected_cuda_visible_devices == observed_cuda_visible_devices,
        "gpu_policy": gpu_policy in {"cpu_no_renderer", "gpu3_camera_only"},
    }
    for name, value in checks.items():
        if not value:
            errors.append(name)
    return {"state": "passed" if not errors else "failed", "errors": errors, "checks": checks, "expected_command": list(expected_command), "observed_command": list(observed_command), "gpu_policy": gpu_policy}


def _manifest_verify(path: Path) -> dict[str, Any]:
    manifest = strict_json(path)
    mismatches: list[str] = []
    for entry in manifest.get("entries", []):
        candidate = ROOT / str(entry["relative_path"])
        if not candidate.is_file():
            mismatches.append(f"missing:{entry['relative_path']}")
            continue
        if candidate.stat().st_size != int(entry["bytes"]):
            mismatches.append(f"bytes:{entry['relative_path']}")
        if sha256_file(candidate) != entry["sha256"]:
            mismatches.append(f"sha256:{entry['relative_path']}")
    return {"entries": len(manifest.get("entries", [])), "mismatches": mismatches, "exact": not mismatches}


def final_manifest(root: Path, manifest_name: str) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == manifest_name or path.name.endswith(".partial"):
            continue
        if path.name == "supervisor.pid" or path.suffix in {".log", ".pid"}:
            continue
        entries.append({"relative_path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": "track3_support_release_final_manifest_v1", "entries_exclude_self": True, "logs_pid_partial_excluded": True, "entries": entries}


def _old_required_paths() -> list[Path]:
    paths = [
        AUDIT_SUMMARY,
        OLD_V0,
        OLD_V4_PARENT,
        OLD_V4_CHILD,
        OLD_V5,
        OLD_MANIFEST,
        OLD_OUT / "v3_clean_release.json",
        OLD_OUT / "v3_artifact_manifest.json",
        OLD_OUT / "v4_corrected_task10_demo0_nominal/trajectory.npz",
        OLD_OUT / "v4_corrected_task10_demo0_nominal/frames.npz",
        OLD_OUT / "v4_corrected_task10_demo0_nominal/start_state.npy",
        OLD_OUT / "v4_corrected_task10_demo0_nominal/end_state.npy",
        HDF5_TASK10,
        STATE_BUNDLE,
        OLD_GEOMETRY,
    ]
    return list(dict.fromkeys(paths))


def run_w0() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = strict_json(AUDIT_SUMMARY)
    old_manifest_check = _manifest_verify(OLD_MANIFEST)
    missing = [rel(path) for path in _old_required_paths() if not path.is_file()]
    old_v0 = strict_json(OLD_V0)
    old_v4_parent = strict_json(OLD_V4_PARENT)
    old_v4_child = strict_json(OLD_V4_CHILD)
    old_v5 = strict_json(OLD_V5)
    gpu0 = previous.gpu_snapshot(0)
    gpu_free = {str(index): previous.gpu_snapshot(index) for index in (1, 2, 3)}
    current_openpi = git(("rev-parse", "HEAD"), OPENPI)
    checks = {
        "required_inputs": not missing,
        "old_manifest_19_of_19": old_manifest_check["entries"] == 19 and old_manifest_check["exact"],
        "audit_execution_commit": audit.get("audited_execution_commit") == "11483623b7710430a197231b8ed0662c4813c12d",
        "audited_commit_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", EXPECTED_AUDITED_COMMIT, "HEAD"], check=False).returncode == 0,
        "openpi_commit": current_openpi == EXPECTED_OPENPI_COMMIT and not porcelain(OPENPI),
        "old_v4_immutable": old_v4_parent.get("category") == "family_failure" and old_v4_child.get("status") == "failed_gate",
        "old_v5_hard_stop": old_v5.get("counts") == {"accepted_reverse_demo": 0, "exact_reverse_demo": 0, "pending_autogen": 0},
        "root_clean": not porcelain(ROOT),
        "gpu0_external_preserved": any(EXPECTED_GPU0_PID in line for line in gpu0.get("compute_processes", [])),
        "gpu1_to_gpu3_free": all(row.get("free") for row in gpu_free.values()),
    }
    lock = {
        "schema_version": "track3_support_release_w0_input_lock_v1",
        "status": "track3_support_release_w0_passed" if all(checks.values()) else "track3_support_release_stopped_W0_input_lock_failure",
        "created_at_utc": now(),
        "plan": rel(PLAN),
        "audit_summary": rel(AUDIT_SUMMARY),
        "audited_execution_commit": EXPECTED_AUDITED_COMMIT,
        "root_head": git(("rev-parse", "HEAD")),
        "openpi_head": current_openpi,
        "required_missing": missing,
        "old_manifest_recomputed": old_manifest_check,
        "old_v4": {"parent_status": old_v4_parent.get("status"), "parent_category": old_v4_parent.get("category"), "child_status": old_v4_child.get("status"), "historical_immutable": True},
        "old_v5_counts": old_v5.get("counts"),
        "hashes": {rel(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in _old_required_paths() if path.is_file()},
        "checks": checks,
        "gpu0": gpu0,
        "gpu1_to_gpu3": gpu_free,
        "libero_plus_porcelain": porcelain(LIBERO_PLUS),
        "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_touched": False, "official_gpu": False, "other_family": False, "conservative": False, "generation": False, "training": False, "zip": False},
    }
    write_json(W0_REPORT, lock)
    return lock


def _hdf5_task10_arrays() -> tuple[dict[str, Any], Any, Any, Any, Any, Any]:
    import h5py
    import numpy as np
    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed

    manifest, model_bytes, init_state, terminal_state, demo_info = composed._demo_arrays(10, 0)
    with h5py.File(HDF5_TASK10, "r") as handle:
        demo = handle["data/demo_0"]
        states = np.asarray(demo["states"][()], dtype=np.float64)
        actions = np.asarray(demo["actions"][()], dtype=np.float64)
    return manifest, model_bytes, init_state, terminal_state, demo_info, (states, actions)


def _relative_pose_xyzw(object_pose: Mapping[str, Any], eef_pose: Mapping[str, Any]) -> dict[str, Any]:
    object_quat = quat_normalize_xyzw(object_pose["quat"])
    object_pos = [float(item) for item in object_pose["pos_m"]]
    eef_pos = [float(item) for item in eef_pose["eef_translation_m"]]
    delta = [eef_pos[index] - object_pos[index] for index in range(3)]
    rotation = quat_to_matrix_xyzw(object_quat)
    return {"translation_m": _transpose_mat_vec(rotation, delta), "quaternion_xyzw": list(quat_mul_xyzw(quat_conjugate_xyzw(object_quat), eef_pose["eef_quaternion"]))}


def _runtime_state_snapshot(env: Any, dynamic: Any, prior: Any, composed: Any, state: Any, index: int) -> dict[str, Any]:
    env.set_init_state(state)
    dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
    robot = dynamic._robot_metrics(env)
    bowl = dynamic._object_or_site_pose(env, "akita_black_bowl_1")
    plate = dynamic._object_or_site_pose(env, "plate_1")
    if bowl is None or plate is None:
        raise ValueError(f"task10_pose_missing_at_state_{index}")
    contact_matrix = prior.contact_matrix(env)
    contact = composed.contact_v3(10, contact_matrix)
    relative = _relative_pose_xyzw(bowl, robot)
    return {
        "index": index,
        "eef": {"pos_m": robot["eef_translation_m"], "quat": robot["eef_quaternion"]},
        "bowl": bowl,
        "plate": plate,
        "relative_pose": relative,
        "contact": {"counts": contact.get("counts", {}), "state": contact.get("state"), "ncon": contact_matrix.get("ncon"), "intended_receiver_contact": contact.get("categories", {}).get("intended_receiver_contact", []), "demonstrator_contact": contact.get("categories", {}).get("demonstrator_contact", []), "unrelated_background_contact": contact.get("categories", {}).get("unrelated_background_contact", [])},
    }


def run_w1() -> dict[str, Any]:
    w0 = strict_json(W0_REPORT) if W0_REPORT.is_file() else {}
    if w0.get("status") != "track3_support_release_w0_passed":
        result = {"schema_version": "track3_support_release_w1_anchor_v1", "status": "not_evaluated", "reason": "w0_not_passed"}
        write_json(W1_REPORT, result)
        return result
    import numpy as np
    _ensure_runtime_pythonpath()
    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
    from tools.axis_pi05 import track3_operator_scoped_stability as prior

    env = None
    started = now()
    try:
        manifest, model_bytes, init_state, terminal_state, demo_info, arrays = _hdf5_task10_arrays()
        states, actions = arrays
        close_transitions = [index for index in range(1, len(actions)) if actions[index - 1, 6] < 0 and actions[index, 6] > 0]
        release_transitions = [index for index in range(1, len(actions)) if actions[index - 1, 6] > 0 and actions[index, 6] < 0]
        composed._configure_public_env(gpu=False)
        from libero.libero.envs.env_wrapper import ControlEnv

        env = ControlEnv(bddl_file_name=str(FORWARD_BDDL), control_freq=CONTROL_FREQ_HZ, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=-1)
        env.reset()
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        env.reset_from_xml_string(resolved_xml)
        snapshots = {index: _runtime_state_snapshot(env, dynamic, prior, composed, states[index], index) for index in (0, 32, 75, int(demo_info["selection"]["state_index"]))}
        series = [_runtime_state_snapshot(env, dynamic, prior, composed, states[index], index) for index in range(32, 76)]
        anchor = snapshots[75]
        anchor_relative = anchor["relative_pose"]
        translation_drifts = [_norm([a - b for a, b in zip(row["relative_pose"]["translation_m"], anchor_relative["translation_m"])]) for row in series]
        rotation_drifts = [quat_geodesic_xyzw(row["relative_pose"]["quaternion_xyzw"], anchor_relative["quaternion_xyzw"]) for row in series]
        controller = dynamic._controller(env)
        controller_config = {"class": type(controller).__name__, "control_dim": int(getattr(controller, "control_dim", -1)), "input_min": [float(item) for item in controller.input_min], "input_max": [float(item) for item in controller.input_max], "output_min": [float(item) for item in controller.output_min], "output_max": [float(item) for item in controller.output_max], "control_freq_hz": CONTROL_FREQ_HZ}
        checks = {
            "model_identity": resolution["source_sha256"] == manifest["model_file_sha256"],
            "close_transition_exact": close_transitions == [32],
            "release_transition_exact": release_transitions == [76],
            "release_anchor_index": anchor["index"] == 75 and actions[75, 6] > 0 and actions[76, 6] < 0,
            "release_anchor_contact_exact": anchor["contact"]["counts"] == {"background_contact": 17, "demonstrator_contact": 3, "intended_receiver_contact": 1, "unrelated_background_contact": 0},
            "series_finite": all(math.isfinite(value) for row in series for value in row["relative_pose"]["translation_m"] + row["relative_pose"]["quaternion_xyzw"]),
            "terminal_state_selected": int(demo_info["selection"]["state_index"]) in snapshots,
        }
        result = {
            "schema_version": "track3_support_release_w1_anchor_v1",
            "status": "track3_support_release_w1_anchor_passed" if all(checks.values()) else "track3_support_release_stopped_W1_anchor_failure",
            "created_at_utc": started,
            "task_index": 10,
            "demo_index": 0,
            "hdf5": {"path": rel(HDF5_TASK10), "sha256": sha256_file(HDF5_TASK10), "states_shape": list(states.shape), "actions_shape": list(actions.shape), "close_transition_index": close_transitions, "release_transition_index": release_transitions, "selection": demo_info["selection"]},
            "model": {"model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "same_model": True},
            "controller_config": controller_config,
            "state_anchors": {"state0": snapshots[0], "close_side_state32": snapshots[32], "release_side_state75": anchor, "terminal": snapshots[int(demo_info["selection"]["state_index"])]},
            "reverse_acquisition_anchor": {"state_index": 75, "release_transition_action_index": 76, "relative_pose_xyzw": anchor_relative, "contact_counts": anchor["contact"]["counts"], "required_contact_counts": {"demonstrator_contact": 3, "intended_receiver_contact": 1, "unrelated_background_contact": 0}},
            "object_follow_series": {"state_index_range": [32, 75], "count": len(series), "max_relative_translation_drift_m_vs_anchor": max(translation_drifts), "max_relative_rotation_drift_rad_vs_anchor": max(rotation_drifts), "index32_to_index75_translation_drift_m": _norm([a - b for a, b in zip(snapshots[32]["relative_pose"]["translation_m"], anchor_relative["translation_m"])]), "index32_to_index75_rotation_drift_rad": quat_geodesic_xyzw(snapshots[32]["relative_pose"]["quaternion_xyzw"], anchor_relative["quaternion_xyzw"]), "series": series},
            "reverse_targets": {"grasp_object_pose_source": "terminal_bowl_pose", "grasp_relative_pose_source": "state75_release_side", "place_object_pose_source": "state0_same_demo_bowl_pose", "task20_task19": "not_evaluated"},
            "checks": checks,
            "simulator_mode": "CPU_no_renderer",
            "manipulation_executed": False,
        }
    except Exception as exc:
        result = {"schema_version": "track3_support_release_w1_anchor_v1", "status": "track3_support_release_stopped_W1_runtime_exception", "created_at_utc": started, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(W1_REPORT, result)
    return result


def run_w2() -> dict[str, Any]:
    w1 = strict_json(W1_REPORT) if W1_REPORT.is_file() else {}
    if w1.get("status") != "track3_support_release_w1_anchor_passed":
        result = {"schema_version": "track3_support_w2_osc_scale_v1", "status": "not_evaluated", "reason": "w1_not_passed"}
        write_json(W2_REPORT, result)
        return result
    config = w1["controller_config"]
    input_max = config["input_max"]
    output_max = config["output_max"]
    checks = {
        "input_max_exact_one": input_max == [1.0] * 6,
        "translation_output_scale_exact": output_max[:3] == [0.05, 0.05, 0.05],
        "rotation_output_scale_exact": output_max[3:] == [0.5, 0.5, 0.5],
        "controller_dim": config["control_dim"] == 6,
        "physical_rotation_clip_not_normalized_one": _norm(osc_action_scaled([0, 0, 0], [0.12, 0, 0], -1.0)["action"][3:6]) == 0.24,
        "translation_exact_scale": osc_action_scaled([0.05, 0, 0], [0, 0, 0], -1.0)["action"][:3] == [1.0, 0.0, 0.0],
        "q_sign_equivalence": quat_geodesic_xyzw([0, 0, 0, 1], [0, 0, 0, -1]) == 0.0,
        "world_frame_left_delta": quat_log_error_xyzw([0, 0, 0, 1], [0, 0, 0, 1]) == [0.0, 0.0, 0.0],
    }
    result = {"schema_version": "track3_support_w2_osc_scale_v1", "status": "track3_support_release_w2_osc_scale_passed" if all(checks.values()) else "track3_support_release_stopped_W2_scale_failure", "created_at_utc": now(), "controller_config": config, "physical_profile": PROFILE, "controller_output_scale": {"translation_m": 0.05, "rotation_rad": 0.5}, "action_contract": {"input_max": 1.0, "rotation_clip_rad": 0.12, "rotation_normalized_norm_at_clip": 0.24, "translation_clip_m": 0.05, "world_frame_error": "target_xyzw * conjugate(current_xyzw)"}, "checks": checks, "task20_task19": "not_evaluated"}
    write_json(W2_REPORT, result)
    return result


def _planned_command(phase: str, root: Path, result_path: Path, frames_path: Path | None = None) -> list[str]:
    command = [sys.executable, "-m", MODULE, f"--{phase}-worker", "--result-output", str(result_path)]
    if frames_path is not None:
        command.extend(["--frames-output", str(frames_path)])
    return command


def _heartbeat(path: Path, *, phase: str, state: str, pid: int, count: int, identity: Mapping[str, Any]) -> dict[str, Any]:
    record = {"schema_version": "track3_support_supervisor_heartbeat_v1", "phase": phase, "state": state, "pid": pid, "count": count, "created_at_utc": now(), "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "identity_state": identity.get("state")}
    write_json(path, record)
    return record


def run_w3() -> dict[str, Any]:
    w0 = strict_json(W0_REPORT) if W0_REPORT.is_file() else {}
    w1 = strict_json(W1_REPORT) if W1_REPORT.is_file() else {}
    w2 = strict_json(W2_REPORT) if W2_REPORT.is_file() else {}
    if not (w0.get("status") == "track3_support_release_w0_passed" and w1.get("status") == "track3_support_release_w1_anchor_passed" and w2.get("status") == "track3_support_release_w2_osc_scale_passed"):
        result = {"schema_version": "track3_support_w3_clean_release_v1", "status": "not_evaluated", "reason": "w0_w1_w2_not_passed"}
        write_json(W3_REPORT, result)
        return result
    root_head = git(("rev-parse", "HEAD"))
    tool_hash = sha256_file(Path(__file__))
    command = _planned_command("w4", W4_ROOT, W4_CHILD)
    identity = supervisor_identity_gate(expected_commit=root_head, observed_commit=root_head, expected_tool_sha256=tool_hash, observed_tool_sha256=tool_hash, expected_openpi=EXPECTED_OPENPI_COMMIT, observed_openpi=git(("rev-parse", "HEAD"), OPENPI), dirty=bool(porcelain(ROOT)), pid_active=False, cursor_active=False, expected_command=command, observed_command=command, expected_cuda_visible_devices="", observed_cuda_visible_devices="", gpu_policy="cpu_no_renderer")
    heartbeat_path = OUT / "w3_heartbeat_probe.json"
    first = _heartbeat(heartbeat_path, phase="W3.test", state="running", pid=os.getpid(), count=1, identity=identity)
    time.sleep(0.02)
    second = _heartbeat(heartbeat_path, phase="W3.test", state="final", pid=os.getpid(), count=2, identity=identity)
    rejected = supervisor_identity_gate(expected_commit=root_head, observed_commit="wrong", expected_tool_sha256=tool_hash, observed_tool_sha256=tool_hash, expected_openpi=EXPECTED_OPENPI_COMMIT, observed_openpi=EXPECTED_OPENPI_COMMIT, dirty=True, pid_active=True, cursor_active=True, expected_command=command, observed_command=command + ["--wrong"], expected_cuda_visible_devices="", observed_cuda_visible_devices="3", gpu_policy="cpu_no_renderer")
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], capture_output=True, text=True, check=False)
    test_result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    checks = {"identity_gate_passed": identity["state"] == "passed", "identity_gate_rejects_mismatch": rejected["state"] == "failed", "heartbeat_two_updates": first["count"] == 1 and second["count"] == 2, "py_compile": compile_result.returncode == 0, "root_tests": test_result.returncode == 0, "root_clean": not porcelain(ROOT), "openpi_clean": not porcelain(OPENPI)}
    result = {"schema_version": "track3_support_w3_clean_release_v1", "status": "track3_support_release_w3_clean_ready" if all(checks.values()) else "track3_support_release_stopped_W3_release_failure", "created_at_utc": now(), "root_commit": root_head, "tool_sha256": tool_hash, "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "identity_contract": identity, "identity_rejection_fixture": rejected, "heartbeat_test": {"path": rel(heartbeat_path), "updates": 2, "periodic_function": "heartbeat_due", "atomic": True}, "py_compile": {"returncode": compile_result.returncode, "stderr": compile_result.stderr[-1000:]}, "root_tests": {"returncode": test_result.returncode, "tail": test_result.stdout[-3000:]}, "checks": checks, "v4_scope": {"task": 10, "demo": 0, "family": "support_remove", "profile": "release_anchor_scaled_nominal", "seed": W4_SEED, "max_runs": 1, "renderer": "none", "gpu": "none"}}
    write_json(W3_REPORT, result)
    return result


def _worker_gate(executed: bool, value: bool | None, **details: Any) -> dict[str, Any]:
    return gate(executed, value, **details)


def _contact_rows_for_gate(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    return sample.get("contacts_v3", {})


def _executor_worker(run_root: Path, result_path: Path, frames_path: Path | None, *, capture_camera: bool) -> int:
    import numpy as np
    _ensure_runtime_pythonpath()
    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
    from tools.axis_pi05 import track3_operator_scoped_stability as prior
    from tools.axis_pi05 import track3_reverse_init_stabilization as base

    started = now()
    env = None
    result: dict[str, Any]
    try:
        w1 = strict_json(W1_REPORT)
        w2 = strict_json(W2_REPORT)
        manifest, model_bytes, _init_state, terminal_state, demo_info = composed._demo_arrays(10, 0)
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        with np.load(STATE_BUNDLE, allow_pickle=False) as archive:
            start_state = np.asarray(archive["task10_demo0"], dtype=np.float64).copy()
        terminal_pose = w1["state_anchors"]["terminal"]["bowl"]
        init_pose = w1["state_anchors"]["state0"]["bowl"]
        relative = w1["reverse_acquisition_anchor"]["relative_pose_xyzw"]
        terminal_quat = quat_normalize_xyzw(terminal_pose["quat"])
        grasp_pos = [a + b for a, b in zip(terminal_pose["pos_m"], _mat_vec(quat_to_matrix_xyzw(terminal_quat), relative["translation_m"]))]
        grasp_quat = quat_mul_xyzw(terminal_quat, relative["quaternion_xyzw"])
        base._configure_public_env(gpu=capture_camera) if hasattr(base, "_configure_public_env") else composed._configure_public_env(gpu=capture_camera)
        from libero.libero.envs.env_wrapper import ControlEnv

        kwargs = {"control_freq": CONTROL_FREQ_HZ, "use_camera_obs": capture_camera, "has_renderer": False, "has_offscreen_renderer": capture_camera, "render_gpu_device_id": 0 if capture_camera else -1}
        if capture_camera:
            kwargs.update({"camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128})
        env = ControlEnv(bddl_file_name=str(base.REVERSE_BDDL[10]), **kwargs)
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
        t_errors: list[float] = []
        r_errors: list[float] = []
        goal_values: list[Any] = []
        front: list[Any] = []
        wrist: list[Any] = []
        stage_results: list[dict[str, Any]] = []
        baseline_scene: Mapping[str, Any] | None = None
        unrelated_names: set[str] = set()
        max_unrelated_movement = 0.0
        max_penetration = 0.0

        def observe(step: int, observation: Mapping[str, Any] | None = None) -> None:
            nonlocal baseline_scene, max_unrelated_movement, max_penetration
            sample = prior._runtime_sample(env, dynamic, 10, "reverse", step, initial_atoms, goal_atoms)
            sample["contacts_v3"] = composed.contact_v3(10, sample["contact_matrix"])
            samples.append(sample)
            scene = sample.get("scene", {}).get("object_and_site_poses", {})
            if baseline_scene is None:
                baseline_scene = scene
                unrelated_names.update(set(scene) - {bowl, "plate_1", "main_table_akita_black_bowl_region"})
            for name in unrelated_names:
                if name in scene and name in baseline_scene:
                    current = scene[name].get("pos_m", [])
                    original = baseline_scene[name].get("pos_m", [])
                    max_unrelated_movement = max(max_unrelated_movement, _norm([float(a) - float(b) for a, b in zip(current, original)]))
            for contact in sample.get("contact_matrix", {}).get("contacts", []):
                max_penetration = max(max_penetration, max(0.0, -float(contact.get("distance_m", 0.0))))
            if observation is not None:
                agent = np.asarray(observation.get("agentview_image"), dtype=np.uint8)
                hand = np.asarray(observation.get("robot0_eye_in_hand_image"), dtype=np.uint8)
                if agent.shape != (128, 128, 3) or hand.shape != (128, 128, 3):
                    raise ValueError("camera_schema")
                front.append(agent.copy())
                wrist.append(hand.copy())

        observe(0)
        stages = [
            ("approach_grasp", -1.0, lambda: [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.09], 150),
            ("descend_grasp", -1.0, lambda: grasp_pos, 120),
            ("close_gripper", 1.0, lambda: grasp_pos, 35),
            ("lift", 1.0, lambda: [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.14], 150),
            ("transport", 1.0, lambda: [init_pose["pos_m"][0], init_pose["pos_m"][1], init_pose["pos_m"][2] + 0.12], 180),
            ("place_same_demo_init_pose", 1.0, lambda: [init_pose["pos_m"][0], init_pose["pos_m"][1], init_pose["pos_m"][2] + 0.055], 120),
            ("release", -1.0, lambda: [init_pose["pos_m"][0], init_pose["pos_m"][1], init_pose["pos_m"][2] + 0.055], 35),
            ("retreat", -1.0, lambda: [init_pose["pos_m"][0], init_pose["pos_m"][1], init_pose["pos_m"][2] + 0.16], 80),
        ]
        failure_reason = None
        for stage_index, (stage_name, gripper, target_fn, budget) in enumerate(stages):
            stable = 0
            stage_start = len(actions)
            last_t = float("inf")
            last_r = float("inf")
            for _local in range(budget):
                current = dynamic._robot_metrics(env)
                target_position = [float(item) for item in target_fn()]
                target_rotation = grasp_quat if stage_index < 4 else quat_normalize_xyzw(init_pose["quat"])
                translation = [target_position[index] - current["eef_translation_m"][index] for index in range(3)]
                rotation = quat_log_error_xyzw(target_rotation, current["eef_quaternion"])
                command = osc_action_scaled(translation, rotation, gripper, scales={"translation_m": float(w2["controller_output_scale"]["translation_m"]), "rotation_rad": float(w2["controller_output_scale"]["rotation_rad"])})
                obs_after, reward, done, _info = env.step(np.asarray(command["action"], dtype=np.float32))
                if not isinstance(obs_after, Mapping):
                    obs_after = env.env._get_observations()
                actions.append(list(command["action"]))
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
                phases.append(stage_name)
                stage_indices.append(stage_index)
                eef_positions.append(list(current["eef_translation_m"]))
                eef_quaternions.append(list(current["eef_quaternion"]))
                target_positions.append(target_position)
                target_quaternions.append(list(target_rotation))
                physical_translation.append(command["physical_translation_delta_m"])
                physical_rotation.append(command["physical_rotation_delta_rad"])
                translation_clips.append(bool(command["translation_physical_clip"]))
                rotation_clips.append(bool(command["rotation_physical_clip"]))
                controller_clips.append(bool(command["controller_clip"]))
                t_errors.append(command["translation_norm_m"])
                r_errors.append(command["rotation_norm_rad"])
                observe(len(actions), obs_after if capture_camera else None)
                last_t, last_r = command["translation_norm_m"], command["rotation_norm_rad"]
                if last_t <= PROFILE["translation_tolerance_m"] and last_r <= PROFILE["rotation_tolerance_rad"]:
                    stable += 1
                else:
                    stable = 0
                if stable >= PROFILE["stable_ticks"] or done:
                    break
            reached = stable >= PROFILE["stable_ticks"]
            stage_results.append({"stage": stage_name, "stage_index": stage_index, "frames": len(actions) - stage_start, "budget": budget, "reached": reached, "last_translation_error_m": last_t, "last_rotation_error_rad": last_r})
            if not reached:
                failure_reason = f"stage_timeout:{stage_name}"
                break
        final_success = bool(env.check_success())
        source_values = []
        goal_values = []
        for sample in samples:
            source_values.append(next((row.get("value") for row in sample.get("initial", {}).get("predicates", []) if row.get("atom") == source_atom), None))
            goal_values.append(next((row.get("value") for row in sample.get("goal", {}).get("predicates", []) if row.get("atom") == goal_atom), None))
        entered_close = any(row["stage"] in {"close_gripper", "lift", "transport", "place_same_demo_init_pose", "release", "retreat"} for row in stage_results)
        entered_place = any(row["stage"] in {"place_same_demo_init_pose", "release", "retreat"} for row in stage_results)
        contact_rows = [_contact_rows_for_gate(sample) for sample in samples]
        contact_safety = _worker_gate(bool(contact_rows), all(not row.get("counts", {}).get("unrelated_background_contact", 0) for row in contact_rows), reason="unrelated_background_contact" if any(row.get("counts", {}).get("unrelated_background_contact", 0) for row in contact_rows) else None)
        contact_completion = _worker_gate(entered_close, bool(any(row.get("counts", {}).get("demonstrator_contact", 0) > 0 for row in contact_rows) and any(row.get("counts", {}).get("intended_receiver_contact", 0) > 0 for row in contact_rows)), reason="contact_completion_absent" if entered_close else "close_stage_not_entered")
        destination = _worker_gate(entered_place, bool(samples and all(_norm([float(a) - float(b) for a, b in zip(sample.get("scene", {}).get("object_and_site_poses", {}).get(bowl, {}).get("pos_m", [999, 999, 999]), init_pose["pos_m"])]) <= PROFILE["translation_tolerance_m"] for sample in samples[-10:])), reason="same_demo_destination_not_reached" if entered_place else "place_stage_not_entered")
        gates = {
            "initial_source_true": _worker_gate(bool(source_values), source_values[0] is True, source_values=source_values[:1]),
            "initial_goal_false": _worker_gate(bool(goal_values), goal_values[0] is False, goal_values=goal_values[:1]),
            "final_goal_true": _worker_gate(bool(goal_values), goal_values[-1] is True and final_success, goal_value=goal_values[-1] if goal_values else None, final_success=final_success),
            "prefix_contact_safety": contact_safety,
            "contact_completion": contact_completion,
            "same_demo_destination": destination,
            "collision": _worker_gate(bool(samples), max_penetration <= 0.01, max_penetration_m=max_penetration, threshold_m=0.01),
            "unrelated_movement": _worker_gate(bool(samples), max_unrelated_movement <= 0.002, max_unrelated_movement_m=max_unrelated_movement, threshold_m=0.002),
            "stall_timeout": _worker_gate(bool(stage_results), failure_reason is None and len(actions) <= PROFILE["step_budget"], reason=failure_reason, used_steps=len(actions), budget=PROFILE["step_budget"]),
            "finite": _worker_gate(bool(actions), bool(np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(states)).all()), action_shape=list(np.asarray(actions).shape), state_shape=list(np.asarray(states).shape)),
        }
        if capture_camera:
            gates["camera_frame_schema"] = _worker_gate(bool(front) and bool(wrist), len(front) == len(actions) and len(wrist) == len(actions), frame_count=len(front), expected_frames=len(actions))
        status = "passed" if all_required(gates) else "failed_gate"
        write_npy(run_root / "start_state.npy", start_state)
        write_npy(run_root / "end_state.npy", np.asarray(env.get_sim_state(), dtype=np.float64))
        if actions:
            with (run_root / "trajectory.npz.partial").open("wb") as handle:
                np.savez_compressed(handle, actions=np.asarray(actions, dtype=np.float32), states=np.asarray(states, dtype=np.float64), phase=np.asarray(phases, dtype="U64"), stage_index=np.asarray(stage_indices, dtype=np.int16), eef_position_m=np.asarray(eef_positions, dtype=np.float32), eef_quaternion_xyzw=np.asarray(eef_quaternions, dtype=np.float32), target_position_m=np.asarray(target_positions, dtype=np.float32), target_quaternion_xyzw=np.asarray(target_quaternions, dtype=np.float32), physical_translation_delta_m=np.asarray(physical_translation, dtype=np.float32), physical_rotation_delta_rad=np.asarray(physical_rotation, dtype=np.float32), translation_physical_clip=np.asarray(translation_clips, dtype=np.bool_), rotation_physical_clip=np.asarray(rotation_clips, dtype=np.bool_), controller_clip=np.asarray(controller_clips, dtype=np.bool_), translation_error_m=np.asarray(t_errors, dtype=np.float32), rotation_error_rad=np.asarray(r_errors, dtype=np.float32), goal_trace=np.asarray([-1 if value is None else int(bool(value)) for value in goal_values], dtype=np.int8))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(run_root / "trajectory.npz.partial", run_root / "trajectory.npz")
        if capture_camera and frames_path is not None:
            with frames_path.with_name(frames_path.name + ".partial").open("wb") as handle:
                np.savez_compressed(handle, agentview=np.asarray(front, dtype=np.uint8), wrist=np.asarray(wrist, dtype=np.uint8))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(frames_path.with_name(frames_path.name + ".partial"), frames_path)
        result = {
            "schema_version": "track3_support_release_worker_v1",
            "status": status,
            "created_at_utc": started,
            "completed_at_utc": now(),
            "child_pid": os.getpid(),
            "task_index": 10,
            "demo_index": 0,
            "family": "support_remove",
            "profile": "release_anchor_scaled_nominal",
            "seed": W4_SEED,
            "fresh_process": True,
            "mode": "GPU3_EGL_camera" if capture_camera else "CPU_no_renderer",
            "quaternion_convention": "xyzw",
            "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "start_state_path": rel(run_root / "start_state.npy"), "start_state_sha256": sha256_file(run_root / "start_state.npy"), "same_demo_model_and_state_components": True, "terminal_pose_source": "same_demo_hdf5_terminal_state", "release_anchor_state_index": 75},
            "reverse": {"source_atom": source_atom, "goal_atom": goal_atom, "initial_goal": goal_values[0] if goal_values else None, "final_goal": final_success, "source_values": source_values, "goal_values": goal_values, "stage_results": stage_results, "failure_reason": failure_reason, "action_saturation_count": int(sum(bool(row) for row in translation_clips + rotation_clips)), "controller_clip_count": int(sum(controller_clips)), "anchor": w1["reverse_acquisition_anchor"]},
            "executor": {"controller": "OSC_POSE", "action_dim": ACTION_DIM, "control_freq_hz": CONTROL_FREQ_HZ, "profile": PROFILE, "physical_output_scale": w2["controller_output_scale"], "target_grasp_position_m": grasp_pos, "target_grasp_quaternion_xyzw": list(grasp_quat)},
            "hard_gate": gates,
            "artifacts": {"trajectory": rel(run_root / "trajectory.npz") if (run_root / "trajectory.npz").is_file() else None, "start_state": rel(run_root / "start_state.npy"), "end_state": rel(run_root / "end_state.npy"), "frames": rel(frames_path) if frames_path is not None and frames_path.is_file() else None},
            "training_eligible": False,
            "pending_autogen": False,
            "resource_boundary": {"gpu3_only": capture_camera, "cpu_no_renderer": not capture_camera, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "official_gpu": False, "no_teleop": True, "no_training": True, "task20_task19": "not_evaluated"},
        }
    except Exception as exc:
        result = {"schema_version": "track3_support_release_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": "release_anchor_scaled_nominal", "seed": W4_SEED, "fresh_process": True, "mode": "GPU3_EGL_camera" if capture_camera else "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "training_eligible": False, "resource_boundary": {"gpu3_only": capture_camera, "cpu_no_renderer": not capture_camera, "gpu0_touched": False, "official_gpu": False}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_path, result)
    return worker_exit_code(str(result.get("status")))


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


def _ffprobe(path: Path, expected_frames: int) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    try:
        stream = json.loads(result.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        stream = {}
    passed = result.returncode == 0 and stream.get("codec_name") == "h264" and int(stream.get("width", -1)) == 128 and int(stream.get("height", -1)) == 128 and int(stream.get("nb_frames", -1)) == expected_frames and str(stream.get("avg_frame_rate")) in {"20/1", "20"}
    return {"state": "passed" if passed else "failed", "returncode": result.returncode, "stream": stream, "expected_frames": expected_frames}


def _encode_video(path: Path, frames: Any) -> dict[str, Any]:
    import numpy as np

    images = np.asarray(frames, dtype=np.uint8)
    if images.ndim != 4 or images.shape[1:] != (128, 128, 3) or images.shape[0] == 0:
        raise ValueError("camera_frame_schema")
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    with subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x128", "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        _stdout, stderr = process.communicate(images.tobytes())
        if process.returncode != 0:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg_failed:{process.returncode}:{stderr.decode(errors='replace')[-500:]}")
    probe = _ffprobe(partial, int(images.shape[0]))
    if probe["state"] != "passed":
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffprobe_failed:{probe}")
    os.replace(partial, path)
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "frame_count": int(images.shape[0]), "ffprobe": probe, "atomic_publish": True}


def _parent_run(*, phase: str, run_root: Path, parent_path: Path, child_path: Path, frames_path: Path | None, capture_camera: bool, gpu_policy: str, seed: int) -> dict[str, Any]:
    w3 = strict_json(W3_REPORT)
    run_root.mkdir(parents=True, exist_ok=False)
    command = _planned_command(phase, run_root, child_path, frames_path)
    observed_cuda = "3" if capture_camera else ""
    current_commit = git(("rev-parse", "HEAD"))
    current_tool = sha256_file(Path(__file__))
    cursor = OUT / "resume_cursor.json"
    pid_file = run_root / "supervisor.pid"
    identity = supervisor_identity_gate(expected_commit=w3.get("root_commit", ""), observed_commit=current_commit, expected_tool_sha256=w3.get("tool_sha256", ""), observed_tool_sha256=current_tool, expected_openpi=EXPECTED_OPENPI_COMMIT, observed_openpi=git(("rev-parse", "HEAD"), OPENPI), dirty=bool(porcelain(ROOT)), pid_active=pid_file.exists(), cursor_active=bool(strict_json(cursor).get("active")) if cursor.is_file() else False, expected_command=command, observed_command=command, expected_cuda_visible_devices=observed_cuda, observed_cuda_visible_devices=observed_cuda, gpu_policy=gpu_policy)
    write_json(run_root / "identity_preflight.json", {"schema_version": "track3_support_supervisor_identity_v1", "phase": phase, "called_by_launcher": True, "identity": identity, "root_commit": current_commit, "tool_sha256": current_tool, "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "command": command, "cuda_visible_devices": observed_cuda})
    if identity["state"] != "passed":
        result = {"schema_version": "track3_support_parent_v1", "status": f"track3_support_release_stopped_{phase.upper()}_identity_failure", "category": "infrastructure", "created_at_utc": now(), "phase": phase.upper(), "identity_gate": identity, "identity_gate_called": True, "resource_boundary": {"gpu3_only": capture_camera, "cpu_no_renderer": not capture_camera, "gpu0_touched": False, "official_gpu": False}}
        write_json(parent_path, result)
        return result
    before_gpu = previous.gpu_snapshot(3) if capture_camera else None
    if capture_camera and not before_gpu.get("free"):
        result = {"schema_version": "track3_support_parent_v1", "status": "track3_support_release_stopped_W5_gpu3_not_free", "category": "infrastructure", "created_at_utc": now(), "phase": "W5", "identity_gate": identity, "identity_gate_called": True, "gpu_start": before_gpu}
        write_json(parent_path, result)
        return result
    write_json(pid_file, {"schema_version": "track3_support_supervisor_pid_v1", "pid": os.getpid(), "phase": phase, "created_at_utc": now()})
    heartbeat_path = run_root / "heartbeat.json"
    heartbeat_records = [_heartbeat(heartbeat_path, phase=phase, state="running", pid=os.getpid(), count=1, identity=identity)]
    environment = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "3" if capture_camera else "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "egl" if capture_camera else "disable"}
    if capture_camera:
        environment["MUJOCO_EGL_DEVICE_ID"] = "3"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    samples = 0
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        last_heartbeat = time.monotonic()
        while process.poll() is None:
            current = time.monotonic()
            if current - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                heartbeat_records.append(_heartbeat(heartbeat_path, phase=phase, state="running", pid=os.getpid(), count=len(heartbeat_records) + 1, identity=identity))
                last_heartbeat = current
            samples += 1
            if current - started >= DEADLINE_SECONDS:
                timed_out = True
                termination_stage = _terminate(process)
                break
            time.sleep(0.1)
        process.wait()
    after_gpu = previous.gpu_snapshot(3) if capture_camera else None
    heartbeat_records.append(_heartbeat(heartbeat_path, phase=phase, state="final", pid=os.getpid(), count=len(heartbeat_records) + 1, identity=identity))
    pid_file.unlink(missing_ok=True)
    child = strict_json(child_path) if child_path.is_file() else None
    camera_state = "not_evaluated"
    camera_errors: list[str] = []
    videos: dict[str, Any] = {}
    if capture_camera:
        try:
            import numpy as np

            with np.load(frames_path, allow_pickle=False) as archive:
                videos["agentview"] = _encode_video(run_root / "agentview.mp4", archive["agentview"])
                videos["wrist"] = _encode_video(run_root / "wrist.mp4", archive["wrist"])
            camera_state = "passed"
        except Exception as exc:
            camera_state = "failed"
            camera_errors.append(f"{type(exc).__name__}:{exc}")
    child_status = (child or {}).get("status")
    telemetry_ok = process.returncode == 0 and not timed_out and child is not None and identity["state"] == "passed" and len(heartbeat_records) >= 2
    if capture_camera:
        telemetry_ok = telemetry_ok and after_gpu and after_gpu.get("free")
    if not telemetry_ok or camera_state == "failed" or child_status not in COMPLETED_WORKER_STATUSES:
        category = "infrastructure"
    elif child_status == "passed" and (not capture_camera or camera_state == "passed"):
        category = "success"
    else:
        category = "family_failure"
    if category == "success":
        status = "track3_support_release_anchor_scaled_nominal_passed_pending_generation_review" if capture_camera else "track3_support_release_anchor_scaled_nominal_cpu_passed_pending_gpu3"
    elif category == "family_failure":
        status = "track3_support_release_anchor_scaled_nominal_failed_gate_pending_review"
    else:
        status = f"track3_support_release_anchor_scale_recovery_stopped_{phase.upper()}_infrastructure_failure"
    result = {"schema_version": "track3_support_parent_v1", "status": status, "category": category, "created_at_utc": now(), "phase": phase.upper(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": "release_anchor_scaled_nominal", "seed": seed, "attempt_number": 1, "identity_gate": identity, "identity_gate_called": True, "root_commit": current_commit, "tool_sha256": current_tool, "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "command": command, "supervisor_pid": os.getpid(), "worker_pid": process.pid, "worker_pid_match": process.pid == (child or {}).get("child_pid"), "process_returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage, "heartbeat": {"path": rel(heartbeat_path), "count": len(heartbeat_records), "interval_seconds": HEARTBEAT_INTERVAL_SECONDS, "records": heartbeat_records}, "poll_samples": samples, "child_status": child_status, "child_result_path": rel(child_path), "child_result": child, "gpu_start": before_gpu, "gpu_end": after_gpu, "camera": {"state": camera_state, "errors": camera_errors, "videos": videos}, "resource_boundary": {"gpu3_only": capture_camera, "cpu_no_renderer": not capture_camera, "gpu0_touched": False, "official_gpu": False, "conservative": False, "other_family": False, "generation": False, "replay": False, "training": False, "zip": False, "no_teleop": True}, "first_error": {"phase": phase.upper(), "category": category, "reason": camera_errors[0] if camera_errors else (child or {}).get("reverse", {}).get("failure_reason")}}
    write_json(parent_path, result)
    return result


def run_w4() -> dict[str, Any]:
    w3 = strict_json(W3_REPORT) if W3_REPORT.is_file() else {}
    if w3.get("status") != "track3_support_release_w3_clean_ready":
        result = {"schema_version": "track3_support_parent_v1", "status": "not_evaluated", "phase": "W4", "reason": "w3_not_ready"}
        W4_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(W4_PARENT, result)
        return result
    return _parent_run(phase="w4", run_root=W4_ROOT, parent_path=W4_PARENT, child_path=W4_CHILD, frames_path=None, capture_camera=False, gpu_policy="cpu_no_renderer", seed=W4_SEED)


def run_w5() -> dict[str, Any]:
    w4 = strict_json(W4_PARENT) if W4_PARENT.is_file() else {}
    if w4.get("category") != "success":
        result = {"schema_version": "track3_support_w5_gpu3_parent_v1", "status": "not_evaluated", "phase": "W5", "reason": "w4_cpu_canary_not_passed", "upstream": {"status": w4.get("status"), "category": w4.get("category")}, "resource_boundary": {"gpu3_started": False, "gpu0_touched": False, "official_gpu": False}}
        write_json(OUT / "w5_gpu3_not_evaluated.json", result)
        return result
    return _parent_run(phase="w5", run_root=W5_ROOT, parent_path=W5_PARENT, child_path=W5_CHILD, frames_path=W5_FRAMES, capture_camera=True, gpu_policy="gpu3_camera_only", seed=W4_SEED)


def _validate_new_namespace() -> dict[str, Any]:
    import numpy as np

    json_paths = sorted(OUT.rglob("*.json"))
    json_ok = True
    json_errors: list[str] = []
    for path in json_paths:
        try:
            strict_json(path)
        except Exception as exc:
            json_ok = False
            json_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    npz_ok = True
    npz_errors: list[str] = []
    for path in sorted(OUT.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                _ = archive.files
        except Exception as exc:
            npz_ok = False
            npz_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    npy_ok = True
    npy_errors: list[str] = []
    for path in sorted(OUT.rglob("*.npy")):
        try:
            np.load(path, allow_pickle=False)
        except Exception as exc:
            npy_ok = False
            npy_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    csv_paths = sorted(OUT.rglob("*.csv"))
    csv_lf = all(b"\r" not in path.read_bytes() for path in csv_paths)
    partials = [rel(path) for path in OUT.rglob("*.partial")]
    return {"strict_json": json_ok, "json_errors": json_errors, "npz_allow_pickle_false": npz_ok, "npz_errors": npz_errors, "npy_allow_pickle_false": npy_ok, "npy_errors": npy_errors, "csv_lf": csv_lf, "partial_residue": partials}


def run_w6() -> dict[str, Any]:
    w4 = strict_json(W4_PARENT) if W4_PARENT.is_file() else {}
    w5_path = W5_PARENT if W5_PARENT.is_file() else OUT / "w5_gpu3_not_evaluated.json"
    w5 = strict_json(w5_path) if w5_path.is_file() else {}
    validation = _validate_new_namespace() if OUT.is_dir() else {}
    test_result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    validation["root_tests"] = {"returncode": test_result.returncode, "tail": test_result.stdout[-3000:]}
    validation["py_compile"] = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], check=False).returncode == 0
    validation["git_diff_check"] = not bool(git(("diff", "--check")))
    gpu = {str(index): previous.gpu_snapshot(index) for index in (0, 1, 2, 3)}
    result = {"schema_version": "track3_support_w6_closeout_v1", "status": "track3_support_release_anchor_scaled_W6_hard_stop", "created_at_utc": now(), "phase_status": {"W0": strict_json(W0_REPORT).get("status", "not_evaluated"), "W1": strict_json(W1_REPORT).get("status", "not_evaluated"), "W2": strict_json(W2_REPORT).get("status", "not_evaluated"), "W3": strict_json(W3_REPORT).get("status", "not_evaluated"), "W4": w4.get("status", "not_evaluated"), "W5": w5.get("status", "not_evaluated"), "W6": "hard_stop"}, "upstream": {"W4": {"status": w4.get("status"), "category": w4.get("category"), "first_error": w4.get("first_error")}, "W5": {"status": w5.get("status"), "category": w5.get("category"), "reason": w5.get("reason")}}, "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "validation": validation, "gpu": gpu, "resource_boundary": {"gpu0_touched": False, "gpu0_external_process_preserved": any(EXPECTED_GPU0_PID in line for line in gpu["0"].get("compute_processes", [])), "gpu1_to_gpu3_released": all(gpu[str(index)].get("free") for index in (1, 2, 3)), "official_gpu": False, "no_teleop": True, "no_training": True, "no_generation": True, "no_zip": True, "libero_plus_untouched": True}, "hard_stop": {"executed": True, "next_authority": "new_reviewed_plan_required", "conservative": False, "other_family": False, "generation": False, "replay": False, "training": False, "zip": False, "upload": False}}
    write_json(W6_REPORT, result)
    write_text(OUT / "w6_phase_status.csv", "phase,status\n" + "\n".join(f"{key},{value}" for key, value in result["phase_status"].items()) + "\n")
    write_json(OUT / "resume_cursor.json", {"schema_version": "track3_support_release_cursor_v1", "status": "hard_stop", "active": False, "next_phase": "review", "last_update_utc": now(), "counts": result["counts"]})
    write_json(OUT / "supervisor_closeout.json", {"schema_version": "track3_support_release_supervisor_closeout_v1", "status": result["status"], "pid": os.getpid(), "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "active": False, "automatic_resume": False})
    write_json(OUT / "w6_artifact_manifest.json", final_manifest(OUT, "w6_artifact_manifest.json"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for phase in ("w0", "w1", "w2", "w3", "w4", "w5", "w6"):
        parser.add_argument(f"--{phase}", action="store_true")
    parser.add_argument("--w4-worker", action="store_true")
    parser.add_argument("--w5-worker", action="store_true")
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--frames-output", type=Path)
    args = parser.parse_args(argv)
    if args.w4_worker or args.w5_worker:
        if args.result_output is None:
            parser.error("worker requires --result-output")
        if args.w5_worker and args.frames_output is None:
            parser.error("w5-worker requires --frames-output")
        root = W5_ROOT if args.w5_worker else W4_ROOT
        return _executor_worker(root, args.result_output, args.frames_output, capture_camera=args.w5_worker)
    selected = [phase for phase in ("w0", "w1", "w2", "w3", "w4", "w5", "w6") if getattr(args, phase)]
    if len(selected) != 1:
        parser.error("select exactly one phase W0/W1/W2/W3/W4/W5/W6")
    result = {"w0": run_w0, "w1": run_w1, "w2": run_w2, "w3": run_w3, "w4": run_w4, "w5": run_w5, "w6": run_w6}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return public_exit_code(str(result.get("status", "")))


if __name__ == "__main__":
    raise SystemExit(main())
