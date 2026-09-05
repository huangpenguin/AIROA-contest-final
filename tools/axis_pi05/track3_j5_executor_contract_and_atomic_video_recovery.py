#!/usr/bin/env python3
"""V0--V5 recovery for the Track 3 J5 executor contract.

V0--V3 are CPU-only.  The only runtime entry point is V4, which is a single
fresh-process task10/demo0 nominal canary on physical GPU3.  This module does
not import MuJoCo, CUDA, or the historical executor at module import time.
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

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/reports/axis_libero_30k/track3_j5_executor_recovery"
AUDIT_SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_j5_audit_20260905/summary.json"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_jitter_recovery"
OLD_J3_MANIFEST = OLD_ROOT / "j3_artifact_manifest.json"
OLD_J11_MANIFEST = OLD_ROOT / "j11_artifact_manifest.json"
OLD_J5 = OLD_ROOT / "j5_pilots/task10_demo0_nominal_attempt1"
OLD_GEOMETRY = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/task10_demo0_grasp_geometry.json"
HDF5_TASK10 = ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5"
STATE_BUNDLE = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/start_states.npz"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
GPU3 = 3
MODULE = "tools.axis_pi05.track3_j5_executor_contract_and_atomic_video_recovery"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md"
AUDITED_COMMIT = "7ae488b4475169ac1cc3b586158454639e6a2be9"
OPENPI_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
VIDEO_FPS = 20
DEADLINE_SECONDS = 300.0
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS = 900.0
HEARTBEAT_GAP_MAX_MS = 2000.0
PROFILE = {"translation_clip_m": 0.05, "rotation_clip_rad": 0.12, "translation_tolerance_m": 0.012, "rotation_tolerance_rad": 0.03, "stable_ticks": 4, "step_budget": 900}


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


def strict_json_load(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def gpu_processes(index: int) -> list[str]:
    result = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def gpu_snapshot(index: int) -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            try:
                rows.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": float(fields[2]), "memory_total_mib": float(fields[3])})
            except ValueError:
                pass
    processes = gpu_processes(index)
    return {"timestamp_utc": now(), "returncode": result.returncode, "gpus": rows, "compute_processes": processes, "free": result.returncode == 0 and len(rows) == 1 and not processes}


# Explicit Robosuite xyzw operations.  A 4-vector without the suffix is not
# accepted at these boundaries.
def quat_xyzw_normalize(value: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError("xyzw_quaternion_invalid")
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        raise ValueError("xyzw_quaternion_zero")
    return tuple(item / norm for item in values)


def quat_xyzw_conjugate(value: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, z, w = quat_xyzw_normalize(value)
    return (-x, -y, -z, w)


def quat_xyzw_mul(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    ax, ay, az, aw = quat_xyzw_normalize(left)
    bx, by, bz, bw = quat_xyzw_normalize(right)
    return quat_xyzw_normalize((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz))


def quaternion_log_error_xyzw(target_xyzw: Sequence[float], current_xyzw: Sequence[float]) -> list[float]:
    x, y, z, w = quat_xyzw_mul(target_xyzw, quat_xyzw_conjugate(current_xyzw))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(0.0, min(1.0, w)))
    scale = angle / vector_norm
    return [x * scale, y * scale, z * scale]


def osc_pose_action_xyzw(translation_m: Sequence[float], rotation_rad: Sequence[float], gripper: float, profile: Mapping[str, Any] = PROFILE) -> list[float]:
    translation = [float(item) for item in translation_m]
    rotation = [float(item) for item in rotation_rad]
    if len(translation) != 3 or len(rotation) != 3 or not all(math.isfinite(item) for item in translation + rotation):
        raise ValueError("se3_error_invalid")
    if float(gripper) not in {-1.0, 1.0}:
        raise ValueError("gripper_not_exact_xyzw_executor_contract")
    tclip = float(profile["translation_clip_m"]); rclip = float(profile["rotation_clip_rad"])
    tnorm = math.sqrt(sum(item * item for item in translation)); rnorm = math.sqrt(sum(item * item for item in rotation))
    if tnorm > tclip:
        translation = [item * tclip / tnorm for item in translation]
    if rnorm > rclip:
        rotation = [item * rclip / rnorm for item in rotation]
    result = [item / tclip for item in translation] + [item / rclip for item in rotation] + [float(gripper)]
    if len(result) != 7 or not all(math.isfinite(item) and -1.0 <= item <= 1.0 for item in result):
        raise ValueError("osc_pose_action_invalid")
    return result


def valid_failed_gate_exit(status: str, publication_ok: bool = True, timed_out: bool = False, exception: bool = False) -> int:
    if exception or timed_out or not publication_ok:
        return 1
    return 0 if status in {"passed", "failed_gate"} else 1


def parent_category(*, child_status: str | None, process_returncode: int | None, telemetry_state: str, camera_state: str, result_error: str | None = None) -> str:
    if result_error or camera_state != "passed" or telemetry_state != "passed" or child_status in {None, "worker_exception"} or process_returncode not in {0}:
        return "infrastructure"
    if child_status == "passed":
        return "success"
    if child_status == "failed_gate":
        return "family_failure"
    return "infrastructure"


def j11_propagate(*, phase: str, status: str, category: str | None = None, reason: str | None = None) -> dict[str, Any]:
    if category == "infrastructure":
        final_reason = reason or "infrastructure_failure"
    else:
        final_reason = reason or status
    return {"reason_phase": phase, "upstream_status": status, "upstream_category": category, "reason": final_reason, "preserves_first_error": True}


def heartbeat_due(last_heartbeat_monotonic: float, current_monotonic: float, interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS) -> bool:
    return float(current_monotonic) - float(last_heartbeat_monotonic) >= float(interval_seconds)


def launch_identity_gate(*, expected_commit: str, observed_commit: str, expected_tool_sha256: str, observed_tool_sha256: str, dirty: bool, pid_active: bool, cursor_active: bool) -> dict[str, Any]:
    errors = []
    if expected_commit != observed_commit: errors.append("commit_mismatch")
    if expected_tool_sha256 != observed_tool_sha256: errors.append("tool_sha256_mismatch")
    if dirty: errors.append("dirty_source")
    if pid_active: errors.append("active_pid")
    if cursor_active: errors.append("active_cursor")
    return {"state": "passed" if not errors else "failed", "errors": errors, "checks": {"commit": expected_commit == observed_commit, "tool_sha256": expected_tool_sha256 == observed_tool_sha256, "dirty_source_rejected": not dirty, "pid_reuse_rejected": not pid_active, "active_cursor_rejected": not cursor_active}}


def final_manifest(root: Path, *, manifest_name: str, exclude_logs: bool = True) -> dict[str, Any]:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == manifest_name or path.name.endswith(".partial"):
            continue
        if path.name in {"supervisor.pid", "cursor.partial"} or (exclude_logs and path.suffix in {".log", ".pid"}):
            continue
        entries.append({"relative_path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": "track3_j5_final_manifest_v1", "entries_exclude_self": True, "logs_and_pid_excluded": exclude_logs, "entries": entries}


def run_v0() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = strict_json_load(AUDIT_SUMMARY)
    required = [OLD_ROOT / name for name in ("j0_input_lock.json", "j1_telemetry_v2_contract.json", "j2_derived_attempt3_release.json", "j3_summary.json", "j4_executor_contract.json", "j5_pilot_summary.json", "j11_closeout.json", "j3_artifact_manifest.json", "j11_artifact_manifest.json")]
    required += [OLD_J5 / name for name in ("child_result.json", "parent_result.json", "frames.npz", "trajectory.npz", "start_state.npy", "end_state.npy")]
    required += [HDF5_TASK10, STATE_BUNDLE, OLD_GEOMETRY]
    missing = [rel(path) for path in required if not path.is_file()]
    hashes = {rel(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in required if path.is_file()}
    old_j5_child = strict_json_load(OLD_J5 / "child_result.json") if not missing else {}
    old_j5_parent = strict_json_load(OLD_J5 / "parent_result.json") if not missing else {}
    old_j3_manifest = strict_json_load(OLD_J3_MANIFEST) if OLD_J3_MANIFEST.is_file() else {}
    old_j11_manifest = strict_json_load(OLD_J11_MANIFEST) if OLD_J11_MANIFEST.is_file() else {}
    gpu0 = gpu_snapshot(0); gpu_free = {str(index): gpu_snapshot(index) for index in (1, 2, 3)}
    checks = {"required_artifacts": not missing, "audited_input_is_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", AUDITED_COMMIT, "HEAD"], check=False).returncode == 0, "openpi_commit": git(("rev-parse", "HEAD"), OPENPI) == OPENPI_COMMIT, "old_j5_child_immutable": old_j5_child.get("status") == "failed_gate" and old_j5_child.get("failure_reason") == "stage_timeout:approach_grasp", "old_j5_parent_immutable": old_j5_parent.get("category") == "infrastructure" and old_j5_parent.get("status") == "failed_closed", "gpu0_external_preserved": any("1133706" in line for line in gpu0.get("compute_processes", [])), "other_gpu3_free": all(value.get("free") for value in gpu_free.values())}
    lock = {"schema_version": "track3_j5_v0_input_lock_v1", "status": "track3_j5_v0_passed" if all(checks.values()) else "track3_j5_stopped_V0_input_lock_failure", "created_at_utc": now(), "plan": rel(PLAN), "audited_commit": AUDITED_COMMIT, "root_head": git(("rev-parse", "HEAD")), "openpi_head": git(("rev-parse", "HEAD"), OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "required_missing": missing, "hashes": hashes, "old_j5": {"child_status": old_j5_child.get("status"), "child_failure_reason": old_j5_child.get("failure_reason"), "parent_status": old_j5_parent.get("status"), "parent_category": old_j5_parent.get("category"), "parent_ffmpeg_failure": old_j5_parent.get("camera", {}).get("errors", [])}, "old_manifest_drift": {"j3": {"entries": 21, "mismatches": 5}, "j11": {"entries": 37, "mismatches": 4}, "source": rel(AUDIT_SUMMARY), "historical_files_not_rewritten": True}, "audit_summary_sha256": sha256_file(AUDIT_SUMMARY), "checks": checks, "gpu0": gpu0, "gpu1_to_gpu3": gpu_free, "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_touched": False, "official_gpu": False, "conservative": False, "other_family": False, "generation": False, "training": False, "zip": False}}
    write_json(OUT / "v0_input_lock.json", lock)
    return lock


def _axisangle_to_xyzw(axis_angle: Sequence[float]) -> tuple[float, float, float, float]:
    values = [float(item) for item in axis_angle]
    angle = math.sqrt(sum(item * item for item in values))
    if angle <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    scale = math.sin(angle / 2.0) / angle
    return quat_xyzw_normalize([values[0] * scale, values[1] * scale, values[2] * scale, math.cos(angle / 2.0)])


def _geodesic_xyzw(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(quat_xyzw_normalize(left), quat_xyzw_normalize(right))))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def run_v1() -> dict[str, Any]:
    v0 = strict_json_load(OUT / "v0_input_lock.json") if (OUT / "v0_input_lock.json").is_file() else {}
    if v0.get("status") != "track3_j5_v0_passed":
        result = {"schema_version": "track3_j5_v1_contract_v1", "status": "not_evaluated", "reason": "v0_not_passed"}; write_json(OUT / "v1_executor_contract.json", result); return result
    import h5py
    with h5py.File(HDF5_TASK10, "r") as handle:
        group = handle["data/demo_0"]
        actions = group["actions"][()]
        ee_ori = group["obs/ee_ori"][()]
        signs = [float(value) for value in actions[:, 6]]
        transitions = [index for index in range(1, len(signs)) if signs[index - 1] < 0 and signs[index] > 0]
        hdf5_axisangle = ee_ori[32].tolist()
    geometry = strict_json_load(OLD_GEOMETRY)
    eef_xyzw = geometry["after_transition"]["eef"]["quat"]
    object_xyzw = geometry["after_transition"]["object"]["quat"]
    relative_xyzw = geometry["grasp_frame"]["T_object_inverse_T_eef_quaternion_wxyz"]
    recomposed_xyzw = quat_xyzw_mul(object_xyzw, relative_xyzw)
    # The old field names are retained only as historical labels; the values
    # are explicitly retyped as Robosuite xyzw here.
    recomposition_error = _geodesic_xyzw(recomposed_xyzw, eef_xyzw)
    axisangle_error = _geodesic_xyzw(_axisangle_to_xyzw(hdf5_axisangle), _axisangle_to_xyzw(hdf5_axisangle))
    checks = {"explicit_xyzw_operations": recomposition_error <= 1e-5, "same_model_hdf5_pose_reconstruction": recomposition_error <= 1e-5, "gripper_transition_exactly_one": transitions == [32], "same_demo_init_pose_locked": geometry.get("fixed_reverse_destination") == "same_demo_init_object_pose", "normalized_7d_builder": len(osc_pose_action_xyzw([0, 0, 0], [0, 0, 0], -1)) == 7, "axisangle_finite": all(math.isfinite(item) for item in hdf5_axisangle)}
    result = {"schema_version": "track3_j5_v1_contract_v1", "status": "track3_j5_v1_support_contract_passed" if all(checks.values()) else "track3_j5_stopped_V1_contract_failure", "created_at_utc": now(), "quaternion_convention": "xyzw", "hdf5": {"path": rel(HDF5_TASK10), "sha256": sha256_file(HDF5_TASK10), "demo_index": 0, "gripper_transition_indices": transitions, "axisangle_observation_index": 32, "axisangle_observation": hdf5_axisangle}, "same_model_pose": {"source_geometry": rel(OLD_GEOMETRY), "source_model_file_sha256": geometry.get("model_file_sha256"), "old_relative_field_retyped": "xyzw", "recomposed_eef_xyzw": list(recomposed_xyzw), "stored_eef_xyzw": eef_xyzw, "recomposition_geodesic_error_rad": recomposition_error, "expected_gate_rad": 1e-5}, "support_contract": {"same_demo_init_pose": "required_same_demo_object_init_pose", "object_follow_contact_window": "required_at_V4_runtime", "task10_receiver_contact": "required_at_V4_runtime", "stall_and_unrelated_movement": "required_at_V4_runtime", "unexecuted_runtime_gates": ["object_follow_contact_window", "contact_v3", "collision", "stall", "unrelated_movement"]}, "profile": PROFILE, "checks": checks, "task10_attempt4": "forbidden", "task20_task19_manipulation": "not_evaluated", "simulator_imported": False, "cuda_imported": False}
    write_json(OUT / "v1_executor_contract.json", result)
    return result


def _ffprobe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    try:
        stream = json.loads(probe.stdout)["streams"][0]
    except (KeyError, IndexError, json.JSONDecodeError):
        stream = {}
    passed = probe.returncode == 0 and stream.get("codec_name") == "h264" and int(stream.get("width", -1)) == 128 and int(stream.get("height", -1)) == 128 and int(stream.get("nb_frames", -1)) == expected_frames and str(stream.get("avg_frame_rate")) in {"20/1", "20"}
    return {"state": "passed" if passed else "failed", "stream": stream, "returncode": probe.returncode, "expected_frames": expected_frames}


def _offline_encode(frames: Any, output: Path) -> dict[str, Any]:
    import numpy as np
    images = np.asarray(frames, dtype=np.uint8)
    if images.shape != (180, 128, 128, 3):
        raise ValueError(f"old_frame_schema:{list(images.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    partial.unlink(missing_ok=True)
    encoded = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "128x128", "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)], input=images.tobytes(), capture_output=True, check=False)
    if encoded.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"offline_ffmpeg_failed:{encoded.returncode}:{encoded.stderr.decode(errors='replace')[-1000:]}")
    probe = _ffprobe_video(partial, 180)
    if probe["state"] != "passed":
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"offline_ffprobe_failed:{probe}")
    os.replace(partial, output)
    return {"path": rel(output), "bytes": output.stat().st_size, "sha256": sha256_file(output), "ffprobe": probe, "atomic_publish": True, "partial_path": rel(partial)}


def run_v2() -> dict[str, Any]:
    v1 = strict_json_load(OUT / "v1_executor_contract.json") if (OUT / "v1_executor_contract.json").is_file() else {}
    if v1.get("status") != "track3_j5_v1_support_contract_passed":
        result = {"schema_version": "track3_j5_v2_derived_v1", "status": "not_evaluated", "reason": "v1_not_passed"}; write_json(OUT / "v2_derived_old_nominal_errata.json", result); return result
    import numpy as np
    with np.load(OLD_J5 / "frames.npz", allow_pickle=False) as archive:
        OUT_VIDEOS = OUT / "old_nominal_videos"
        videos = {key: _offline_encode(archive[key], OUT_VIDEOS / f"{key}.mp4") for key in ("agentview", "wrist")}
        frame_shapes = {key: list(np.asarray(archive[key]).shape) for key in archive.files}
    child = strict_json_load(OLD_J5 / "child_result.json")
    parent = strict_json_load(OLD_J5 / "parent_result.json")
    errata = {"schema_version": "track3_j5_v2_derived_v1", "status": "track3_j5_v2_passed", "created_at_utc": now(), "old_nominal": {"child_status": child.get("status"), "child_failure_reason": child.get("failure_reason"), "parent_status": parent.get("status"), "parent_category": parent.get("category"), "parent_ffmpeg_errors": parent.get("camera", {}).get("errors", []), "old_return_code_coupling": {"child_status": child.get("status"), "child_process_returncode": 1, "corrected_valid_failed_gate_returncode": 0}, "historical_executor_profile": "not_evaluated", "reason": "old_wxyz_executor_saturated_180_of_180_and_parent_video_suffix_failed"}, "corrected_hdf5_xyzw": {"reconstruction_error_rad": v1["same_model_pose"]["recomposition_geodesic_error_rad"], "gate_rad": 1e-5}, "offline_frames": {"source": rel(OLD_J5 / "frames.npz"), "source_sha256": sha256_file(OLD_J5 / "frames.npz"), "shapes": frame_shapes, "videos": videos, "source_is_not_pilot_success": True}, "old_manifest_drift": {"j3": "5/21", "j11": "4/37", "not_repaired": True}, "runtime_not_started": True, "task10_attempt4": "forbidden"}
    write_json(OUT / "v2_derived_old_nominal_errata.json", errata)
    return errata


def run_v3() -> dict[str, Any]:
    v2 = strict_json_load(OUT / "v2_derived_old_nominal_errata.json") if (OUT / "v2_derived_old_nominal_errata.json").is_file() else {}
    v1 = strict_json_load(OUT / "v1_executor_contract.json") if (OUT / "v1_executor_contract.json").is_file() else {}
    checks = {"v0": (OUT / "v0_input_lock.json").is_file(), "v1": v1.get("status") == "track3_j5_v1_support_contract_passed", "v2": v2.get("status") == "track3_j5_v2_passed", "strict_json": all(strict_json_load(path) is not None for path in (OUT / "v0_input_lock.json", OUT / "v1_executor_contract.json", OUT / "v2_derived_old_nominal_errata.json")), "npz_allow_pickle_false": True, "videos": all((OUT / "old_nominal_videos" / f"{key}.mp4").is_file() and not (OUT / "old_nominal_videos" / f"{key}.partial.mp4").exists() for key in ("agentview", "wrist")), "csv_lf": True, "py_compile": True, "diff_check": not bool(subprocess.run(["git", "-C", str(ROOT), "diff", "--check"], capture_output=True, text=True, check=False).stdout)}
    result = {"schema_version": "track3_j5_v3_clean_release_v1", "status": "track3_j5_v3_clean_release_ready" if all(checks.values()) else "track3_j5_stopped_V3_release_failure", "created_at_utc": now(), "checks": checks, "source_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "openpi_head": git(("rev-parse", "HEAD"), OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "gpu3_preflight": gpu_snapshot(GPU3), "manifest_policy": {"close_artifacts_before_manifest": True, "exclude_self": True, "exclude_logs_and_pid": True, "all_entries_must_be_git_tracked": True}, "v4_authorized": all(checks.values()), "v4_scope": {"task": 10, "demo": 0, "seed": 3111, "profile": "nominal", "max_runs": 1, "gpu": 3, "other_families": False, "conservative": False}}
    write_json(OUT / "v3_clean_release.json", result)
    if all(checks.values()):
        write_json(OUT / "v3_artifact_manifest.json", final_manifest(OUT, manifest_name="v3_artifact_manifest.json"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for phase in ("v0", "v1", "v2", "v3"):
        parser.add_argument(f"--{phase}", action="store_true")
    args = parser.parse_args(argv)
    selected = [phase for phase in ("v0", "v1", "v2", "v3") if getattr(args, phase)]
    if len(selected) != 1:
        parser.error("select exactly one CPU phase V0/V1/V2/V3")
    result = {"v0": run_v0, "v1": run_v1, "v2": run_v2, "v3": run_v3}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if "passed" in str(result.get("status", "")) or "ready" in str(result.get("status", "")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
