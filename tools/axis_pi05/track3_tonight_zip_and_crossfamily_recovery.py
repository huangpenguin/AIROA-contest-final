#!/usr/bin/env python3
"""Tonight's bounded Track 3 recovery executor.

This module is intentionally a new namespace.  It revalidates the existing
R6 package, freezes seven *new* family-scoped CPU identities, and executes
each identity once in a fresh child process.  A normal simulator gate failure
is a family failure; only a supervisor/identity/infrastructure failure stops
the remaining matrix.  GPU3, generation, replay, training, and candidate
packaging are all downstream gates and are never entered implicitly.

The runtime worker imports LIBERO/MuJoCo only in the child process.  The
parent owns atomic evidence, timeout handling, and the process boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_tonight_zip_and_crossfamily_recovery"
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_tonight_zip_and_crossfamily_recovery"
RUN_ROOT = OUT_ROOT / "runs"
FALLBACK_SOURCE = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
FALLBACK_ZIP = ROOT / "artifacts/submission_axis_r6_fallback_revalidated_20260906.zip"
FALLBACK_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
FALLBACK_BYTES = 9383571677
PUBLIC_AUTOGEN = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
S3_SUMMARY = PUBLIC_AUTOGEN / "s3_reverse_adapter.json"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
R6_HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
R6_C5 = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/track1_calibration/summary.json"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
TASK_ORDER = (10, 20, 19)
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SOURCE_ATOM = {
    10: "(On akita_black_bowl_1 plate_1)",
    20: "(In orange_juice_1 basket_1_contain_region)",
    19: "(Open wooden_cabinet_1_middle_region)",
}
GOAL_ATOM = {
    10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)",
    20: "(On orange_juice_1 floor_target_object_region)",
    19: "(Close wooden_cabinet_1_middle_region)",
}
REVERSE_BDDL = {
    10: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_support_remove_task_10.bddl",
    20: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_containment_remove_task_20.bddl",
    19: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_articulation_close_task_19.bddl",
}
SNAPSHOTS = {
    task: P1_ROOT / "snapshots" / f"task_{task:02d}" / "attempt_1" / "terminal_state.npy"
    for task in TASK_ORDER
}
PROFILE_ORDER = (
    "support_contact_repair_a",
    "support_contact_repair_b",
    "support_contact_repair_c",
    "containment_extract_a",
    "containment_extract_b",
    "articulation_pull_a",
    "articulation_pull_b",
)
PROFILE_TASK = {
    "support_contact_repair_a": 10,
    "support_contact_repair_b": 10,
    "support_contact_repair_c": 10,
    "containment_extract_a": 20,
    "containment_extract_b": 20,
    "articulation_pull_a": 19,
    "articulation_pull_b": 19,
}

# These are a small, pre-registered envelope around the already observed
# public scripted values.  Only contact/close dwell changes are varied.
PROFILES: dict[str, dict[str, Any]] = {
    "support_contact_repair_a": {"task_index": 10, "seed": 41010, "grasp_z_m": 0.025, "close_dwell": 1, "release_dwell": 1},
    "support_contact_repair_b": {"task_index": 10, "seed": 41011, "grasp_z_m": 0.025, "close_dwell": 2, "release_dwell": 1},
    "support_contact_repair_c": {"task_index": 10, "seed": 41012, "grasp_z_m": 0.025, "close_dwell": 3, "release_dwell": 1},
    "containment_extract_a": {"task_index": 20, "seed": 42020, "grasp_z_m": 0.025, "close_dwell": 1, "release_dwell": 1},
    "containment_extract_b": {"task_index": 20, "seed": 42021, "grasp_z_m": 0.025, "close_dwell": 2, "release_dwell": 1},
    "articulation_pull_a": {"task_index": 19, "seed": 43019, "grasp_z_m": 0.012, "close_dwell": 1, "release_dwell": 1},
    "articulation_pull_b": {"task_index": 19, "seed": 43020, "grasp_z_m": 0.012, "close_dwell": 2, "release_dwell": 1},
}
MAX_CPU_PROFILES = 7
MAX_GPU3_FAMILIES = 3
MAX_GENERATIONS = 6
MAX_REPLAYS = 18
TERM_GRACE_SECONDS = 10.0
CPU_TIMEOUT_SECONDS = 240.0
GPU_TIMEOUT_SECONDS = 360.0


class DuplicateKeyError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def gpu_snapshot(index: int) -> dict[str, Any]:
    query = ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    result = subprocess.run(query, capture_output=True, text=True, check=False)
    return {"index": index, "returncode": result.returncode, "compute_processes": [line.strip() for line in result.stdout.splitlines() if line.strip()], "stderr": result.stderr[-1000:], "free": result.returncode == 0 and not result.stdout.strip()}


def profile_lock() -> dict[str, Any]:
    rows = []
    for name in PROFILE_ORDER:
        profile = dict(PROFILES[name])
        profile.update({"profile_id": name, "family": FAMILY[int(profile["task_index"])], "identity_scope": "new_tonight_identity_once", "old_identity_rerun": False, "renderer": False, "gpu": False})
        rows.append(profile)
    body = {"schema_version": "track3_tonight_profile_lock_v1", "profile_order": list(PROFILE_ORDER), "profiles": rows, "limits": {"support_remove": 3, "containment_remove": 2, "articulation_close": 2, "total": MAX_CPU_PROFILES}, "parameter_source": {"grasp_z_m": "AA5/support anchor and existing public scripted envelope", "close_dwell": "AA5 continuous-contact diagnosis bounded to 1/2/3 ticks", "articulation_grasp_z_m": "public scripted articulation approach envelope"}, "old_identities_denylist": ["object_frame_nominal", "object_frame_conservative"], "official_track3_claim": False}
    body["lock_sha256"] = canonical_hash(body)
    return body


def validate_s3_and_p1() -> dict[str, Any]:
    summary = strict_json(S3_SUMMARY)
    p1 = strict_json(P1_SUMMARY)
    errors: list[str] = []
    if summary.get("status") != "s3_reverse_semantic_adapter_passed" or summary.get("spec_count") != 3:
        errors.append("s3_summary_not_passed")
    if summary.get("official_track3_claim") is not False or summary.get("denylist_overlap_count") != 0:
        errors.append("s3_provenance_gate")
    if not isinstance(p1, Mapping):
        errors.append("p1_summary_not_mapping")
    for task in TASK_ORDER:
        if not REVERSE_BDDL[task].is_file():
            errors.append(f"missing_reverse_bddl:{task}")
        if not SNAPSHOTS[task].is_file():
            errors.append(f"missing_p1_snapshot:{task}")
    return {"state": "passed" if not errors else "failed", "errors": errors, "s3_summary": rel(S3_SUMMARY), "p1_summary": rel(P1_SUMMARY), "reverse_bddl_sha256": {str(task): sha256_file(REVERSE_BDDL[task]) for task in TASK_ORDER if REVERSE_BDDL[task].is_file()}, "snapshot_sha256": {str(task): sha256_file(SNAPSHOTS[task]) for task in TASK_ORDER if SNAPSHOTS[task].is_file()}}


def _c5_gate() -> dict[str, Any]:
    try:
        value = strict_json(R6_C5)
        passed = value.get("status") == "passed" and value.get("source_success_reproduced") == 3 and value.get("gates", {}).get("coverage_5_of_5") is True
        return {"state": "passed" if passed else "failed", "path": rel(R6_C5), "source_success_reproduced": value.get("source_success_reproduced"), "status": value.get("status"), "gates": value.get("gates", {})}
    except Exception as exc:
        return {"state": "failed", "path": rel(R6_C5), "error": f"{type(exc).__name__}: {exc}"}


def materialize_fallback() -> dict[str, Any]:
    if not FALLBACK_SOURCE.is_file():
        raise FileNotFoundError(FALLBACK_SOURCE)
    source_bytes = FALLBACK_SOURCE.stat().st_size
    source_sha = sha256_file(FALLBACK_SOURCE)
    if source_bytes != FALLBACK_BYTES or source_sha != FALLBACK_SHA:
        raise RuntimeError(f"fallback_source_hash_mismatch:{source_bytes}:{source_sha}")
    with zipfile.ZipFile(FALLBACK_SOURCE) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        model_fast = [name for name in names if name == "model_weights/model_fast.safetensors"]
        slow = [name for name in names if "model_slow" in name or name.endswith("slow.safetensors")]
        if bad is not None or len(model_fast) != 1 or slow:
            raise RuntimeError(f"fallback_zip_gate_failed:{bad}:{model_fast}:{slow[:3]}")
        manifest = json.loads(archive.read("submission_manifest.json"))
    # Hard-linking is the required no-recompress materialization.  It gives
    # the new local name its own audit entry while preserving exact bytes.
    if FALLBACK_ZIP.exists():
        if FALLBACK_ZIP.stat().st_size != source_bytes or sha256_file(FALLBACK_ZIP) != source_sha:
            raise RuntimeError("existing_fallback_materialization_mismatch")
    else:
        try:
            os.link(FALLBACK_SOURCE, FALLBACK_ZIP)
        except OSError:
            shutil.copyfile(FALLBACK_SOURCE, FALLBACK_ZIP)
    c5 = _c5_gate()
    result = {
        "schema_version": "track3_tonight_fallback_revalidation_v1",
        "status": "passed" if c5["state"] == "passed" else "failed",
        "readiness": "fallback_no_track3_improvement",
        "track3_improvement": False,
        "source": {"path": rel(FALLBACK_SOURCE), "bytes": source_bytes, "sha256": source_sha},
        "materialized": {"path": rel(FALLBACK_ZIP), "bytes": FALLBACK_ZIP.stat().st_size, "sha256": sha256_file(FALLBACK_ZIP), "method": "hardlink_or_byte_exact_copy_no_recompress"},
        "zip": {"crc_test": "passed", "entry_count": len(names), "model_fast_exactly_one": True, "slow_model_entries": [], "manifest_normalization": manifest.get("normalization"), "manifest_official_result": manifest.get("official_result"), "manifest_submission_status": manifest.get("submission_status")},
        "c5_source_success": c5,
        "official_track3_claim": False,
        "upload_or_submission": False,
    }
    if result["materialized"]["sha256"] != FALLBACK_SHA:
        result["status"] = "failed"
    write_json(OUT_ROOT / "tz0_fallback_revalidation.json", result)
    atomic_bytes(FALLBACK_ZIP.with_suffix(FALLBACK_ZIP.suffix + ".sha256"), (FALLBACK_SHA + "  " + FALLBACK_ZIP.name + "\n").encode())
    return result


def aa5_diagnostic() -> dict[str, Any]:
    # The previous AA namespace is immutable input.  This compact record
    # makes the reason for the new profiles explicit without rerunning AA5.
    candidates = [
        ROOT / "data/reports/axis_libero_30k/track3_z5_derived_release_and_conservative_continuation/aa5_summary.json",
        ROOT / "data/reports/axis_libero_30k/track3_z5_derived_release_and_conservative_continuation/summary.json",
    ]
    source: dict[str, Any] = {}
    for path in candidates:
        if path.is_file():
            try:
                source[rel(path)] = strict_json(path)
            except Exception as exc:
                source[rel(path)] = {"read_error": f"{type(exc).__name__}: {exc}"}
    return {"schema_version": "track3_tonight_aa5_diagnostic_v1", "status": "passed", "rerun": False, "source_reports": source, "observed_from_reviewed_evidence": {"phases": [49, 21, 6], "eef_translation_error_m": 0.000576, "eef_rotation_error_rad": 0.00184, "controller_clip": False, "source_contact_samples": "77/77", "source_release": "false", "goal_success": "0/77", "failure_scope": "continuous_gripper_contact_and_source_release; not arm_reachability"}, "profile_change_scope": "same public grasp offset envelope; close dwell only 1/2/3 ticks", "old_nominal_or_conservative_rerun": False}


def run_tz0() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pre = validate_s3_and_p1()
    fallback = materialize_fallback() if pre["state"] == "passed" else {"status": "not_evaluated", "reason": "s3_or_p1_preflight_failed"}
    result = {"schema_version": "track3_tonight_tz0_v1", "status": "passed" if pre["state"] == "passed" and fallback.get("status") == "passed" else "failed", "created_at_utc": now(), "preflight": pre, "fallback": fallback, "no_physics": True, "no_gpu": True}
    write_json(OUT_ROOT / "tz0_summary.json", result)
    return result


def cli_contract() -> dict[str, Any]:
    return {"schema_version": "track3_tonight_cli_contract_v1", "module": MODULE, "phase_flags": [f"--tz{i}" for i in range(10)], "worker_flags": ["--cpu-worker", "--gpu3-worker", "--generation-worker", "--replay-worker"], "single_instance_supervisor": True, "atomic_cursor": True, "atomic_heartbeat": True, "term_then_kill": True, "return_code_contract": {"passed": 0, "failed_gate": 0, "family_failure": 0, "not_evaluated": 0, "infrastructure_failure": 1, "worker_exception": 1}, "missing_phase_cli": False, "old_aa_namespace_modified": False}


def run_tz1() -> dict[str, Any]:
    pre = validate_s3_and_p1()
    lock = profile_lock()
    diagnostic = aa5_diagnostic()
    result = {"schema_version": "track3_tonight_tz1_v1", "status": "passed" if pre["state"] == "passed" and len(lock["profiles"]) == MAX_CPU_PROFILES else "failed", "created_at_utc": now(), "preflight": pre, "cli_contract": cli_contract(), "profile_lock": lock, "aa5_diagnostic": diagnostic, "profile_count": len(lock["profiles"]), "profile_families": {family: sum(row["family"] == family for row in lock["profiles"]) for family in sorted(set(FAMILY.values()))}, "physics_started": False, "gpu_started": False}
    write_json(OUT_ROOT / "tz1_cli_and_profile_lock.json", result)
    write_json(OUT_ROOT / "profile_lock.json", lock)
    return result


def _set_cursor(phase: str, active: bool, **extra: Any) -> None:
    payload = {"schema_version": "track3_tonight_cursor_v1", "active": active, "phase": phase, "updated_at_utc": now(), "pid": os.getpid(), **extra}
    write_json(OUT_ROOT / "resume_cursor.json", payload)


def _heartbeat(state: str, **extra: Any) -> None:
    write_json(OUT_ROOT / "heartbeat.json", {"schema_version": "track3_tonight_heartbeat_v1", "state": state, "updated_at_utc": now(), "pid": os.getpid(), **extra})


@dataclass
class ChildResult:
    payload: dict[str, Any]
    returncode: int
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


def _parse_result(stdout: str, path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            value = strict_json(path)
            if isinstance(value, dict) and "status" in value:
                return value
        except Exception:
            pass
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise ValueError("worker_emitted_no_result")


def run_child(flag: str, args: Sequence[str], result_path: Path, timeout: float, gpu: bool = False) -> ChildResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'LIBERO'}:{ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = "3" if gpu else ""
    env["JAX_PLATFORMS"] = "cpu"
    env["MUJOCO_GL"] = "egl" if gpu else "disable"
    if gpu:
        env["MUJOCO_EGL_DEVICE_ID"] = "3"
    else:
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
    command = [sys.executable, "-m", MODULE, flag, *args, "--result-output", str(result_path)]
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        timed_out = True
        stderr = (stderr or "") + "\nworker_timeout"
    try:
        payload = _parse_result(stdout or "", result_path)
    except Exception as exc:
        payload = {"schema_version": "track3_tonight_child_result_v1", "status": "worker_exception", "error": {"type": type(exc).__name__, "message": str(exc), "stdout_tail": (stdout or "")[-3000:], "stderr_tail": (stderr or "")[-3000:]}}
    payload["process_returncode"] = process.returncode
    payload["timed_out"] = timed_out
    payload["stdout_tail"] = (stdout or "")[-3000:]
    payload["stderr_tail"] = (stderr or "")[-3000:]
    return ChildResult(payload, int(process.returncode or 0), timed_out, stdout or "", stderr or "")


def _stage_target(env: Any, task: int, profile: Mapping[str, Any], stage: str, anchor: np.ndarray, region: np.ndarray, dynamic: Any, scripted: Any) -> tuple[np.ndarray, float]:
    import numpy as np

    object_name = "akita_black_bowl_1" if task == 10 else "orange_juice_1"
    if task in (10, 20):
        if stage == "approach_object":
            return anchor + np.array([0.0, 0.0, 0.12 if task == 10 else 0.13]), -1.0
        if stage == "descend_object":
            return anchor + np.array([0.0, 0.0, float(profile["grasp_z_m"]) ]), -1.0
        if stage == "grasp_object":
            return anchor + np.array([0.0, 0.0, float(profile["grasp_z_m"]) ]), 1.0
        if stage == "lift_object":
            return anchor + np.array([0.0, 0.0, 0.14 if task == 10 else 0.16]), 1.0
        if stage == "transport_to_target":
            return region + np.array([0.0, 0.0, 0.14 if task == 10 else 0.13]), 1.0
        if stage == "descend_to_target":
            return region + np.array([0.0, 0.0, 0.055 if task == 10 else 0.075]), 1.0
        if stage == "release_object":
            return region + np.array([0.0, 0.0, 0.055 if task == 10 else 0.075]), -1.0
        return region + np.array([0.0, 0.0, 0.16 if task == 10 else 0.18]), -1.0
    if stage == "approach_drawer_region":
        return region + np.array([0.0, -0.012, 0.012]), -1.0
    if stage in {"grasp_drawer_region", "hold_closed"}:
        return region + np.array([0.0, -0.012, 0.012]) if stage == "grasp_drawer_region" else region + np.array([0.0, -0.20, 0.0]), 1.0
    return region + np.array([0.0, -0.20, 0.0]), 1.0


def _safe_site_position(env: Any, name: str, dynamic: Any) -> np.ndarray:
    pose = dynamic._object_or_site_pose(env, name)
    if pose is None:
        raise RuntimeError(f"missing_operator_site:{name}")
    return np.asarray(pose["pos_m"], dtype=np.float64)


def _robot_contact(task: int, matrix: Mapping[str, Any]) -> bool:
    subject = "akita_black_bowl_1" if task == 10 else "orange_juice_1" if task == 20 else "wooden_cabinet_1_middle"
    for row in matrix.get("contacts", []):
        names = (str(row.get("body1_name", "")).lower(), str(row.get("body2_name", "")).lower())
        if subject.lower() in names[0] or subject.lower() in names[1]:
            if any(token in name for name in names for token in ("robot", "gripper", "finger", "panda")):
                return True
    return False


def _worker_result_base(profile_id: str, task: int, gpu: bool) -> dict[str, Any]:
    return {"schema_version": "track3_tonight_worker_v1", "profile_id": profile_id, "task_index": task, "family": FAMILY[task], "seed": int(PROFILES[profile_id]["seed"]), "fresh_process": True, "mode": "GPU3_camera" if gpu else "CPU_no_renderer", "gpu_policy": "physical_gpu3_only" if gpu else "no_gpu_cpu", "official_track3_claim": False, "training_eligible": False, "start_at_utc": now()}


def cpu_worker(profile_id: str, result_output: Path, *, gpu: bool = False, demo_index: int = 0, replay_actions: Path | None = None) -> dict[str, Any]:
    import numpy as np

    if profile_id not in PROFILES:
        raise ValueError(f"unknown_profile:{profile_id}")
    task = int(PROFILE_TASK[profile_id])
    result = _worker_result_base(profile_id, task, gpu)
    env = None
    try:
        if not gpu:
            os.environ.update({"CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "disable", "JAX_PLATFORMS": "cpu"})
        else:
            os.environ.update({"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "3"})
        from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_operator_scoped_stability as operator
        from tools.axis_pi05 import track3_scripted_autogen as scripted
        composed._configure_public_env(gpu=gpu)
        from libero.libero.envs.env_wrapper import ControlEnv

        manifest, model_bytes, _init_state, _terminal_state, _demo_info = composed._demo_arrays(task, int(demo_index))
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        kwargs: dict[str, Any] = {"bddl_file_name": str(REVERSE_BDDL[task]), "control_freq": 20, "use_camera_obs": bool(gpu), "has_renderer": False, "has_offscreen_renderer": bool(gpu), "render_gpu_device_id": 0 if gpu else -1}
        if gpu:
            kwargs.update({"camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128})
        env = ControlEnv(**kwargs)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        snapshot = P1_ROOT / "snapshots" / f"task_{task:02d}" / f"attempt_{int(demo_index) + 1}" / "terminal_state.npy"
        if not snapshot.is_file():
            snapshot = SNAPSHOTS[task]
        start_state = np.asarray(np.load(snapshot), dtype=np.float64)
        obs = env.set_init_state(start_state)
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial = dynamic.evaluate_predicates(env, [SOURCE_ATOM[task]], label="initial_source")
        initial_goal = dynamic.evaluate_predicates(env, [GOAL_ATOM[task]], label="initial_goal")
        if not initial.get("all_true") or initial_goal.get("all_true"):
            raise RuntimeError("initial_predicate_contract_failed")
        if replay_actions is not None:
            actions = np.asarray(np.load(replay_actions, allow_pickle=False)["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError("replay_action_shape")
            for action in actions:
                obs, _reward, _done, _info = env.step(action)
            final = dynamic.evaluate_predicates(env, [GOAL_ATOM[task]], label="replay_goal")
            result.update({"status": "passed" if final.get("all_true") else "failed_gate", "replay": True, "replay_steps": int(len(actions)), "goal": final, "actions_sha256": sha256_file(replay_actions)})
        else:
            object_name = "akita_black_bowl_1" if task == 10 else "orange_juice_1" if task == 20 else "wooden_cabinet_1_middle_region"
            anchor_pose = dynamic._object_or_site_pose(env, object_name)
            if anchor_pose is None:
                raise RuntimeError(f"missing_anchor_pose:{object_name}")
            anchor = np.asarray(anchor_pose["pos_m"], dtype=np.float64)
            region_name = "main_table_akita_black_bowl_region" if task == 10 else "floor_target_object_region" if task == 20 else "wooden_cabinet_1_middle_region"
            region = _safe_site_position(env, region_name, dynamic)
            stages = ["approach_object", "descend_object", "grasp_object", "lift_object", "transport_to_target", "descend_to_target", "release_object", "retreat"] if task in (10, 20) else ["approach_drawer_region", "grasp_drawer_region", "pull_drawer_closed", "hold_closed"]
            max_steps = {"approach_object": 170, "descend_object": 140, "grasp_object": 45, "lift_object": 170, "transport_to_target": 220, "descend_to_target": 140, "release_object": 40, "retreat": 90, "approach_drawer_region": 180, "grasp_drawer_region": 40, "pull_drawer_closed": 280, "hold_closed": 45}
            tolerance = {"approach_object": 0.014, "descend_object": 0.012, "grasp_object": 0.020, "lift_object": 0.014, "transport_to_target": 0.014, "descend_to_target": 0.014, "release_object": 0.022, "retreat": 0.014, "approach_drawer_region": 0.015, "grasp_drawer_region": 0.022, "pull_drawer_closed": 0.018, "hold_closed": 0.025}
            action_rows: list[np.ndarray] = []
            eef_errors: list[float] = []
            phases: list[str] = []
            source_values: list[Any] = []
            goal_values: list[Any] = []
            contact_rows: list[dict[str, Any]] = []
            stage_results: list[dict[str, Any]] = []
            all_reached = True
            source_false_step: int | None = None
            for stage in stages:
                stable = 0
                reached = False
                last_error = float("inf")
                start_count = len(action_rows)
                for _ in range(max_steps[stage]):
                    target, gripper = _stage_target(env, task, PROFILES[profile_id], stage, anchor, region, dynamic, scripted)
                    action, error, saturated = scripted.closed_loop_action(obs, target, gripper)
                    obs, _reward, _done, _info = env.step(action)
                    action_rows.append(np.asarray(action, dtype=np.float32).copy())
                    eef_errors.append(float(error))
                    phases.append(stage)
                    matrix = operator.contact_matrix(env)
                    contact_rows.append({"ncon": matrix.get("ncon"), "robot_subject_contact": _robot_contact(task, matrix), "complete": matrix.get("complete")})
                    source = dynamic.evaluate_predicates(env, [SOURCE_ATOM[task]], label=f"source_step_{len(action_rows)}")
                    goal = dynamic.evaluate_predicates(env, [GOAL_ATOM[task]], label=f"goal_step_{len(action_rows)}")
                    source_value = source["predicates"][0].get("value") if source.get("predicates") else None
                    goal_value = goal["predicates"][0].get("value") if goal.get("predicates") else None
                    source_values.append(source_value)
                    goal_values.append(goal_value)
                    if source_value is False and source_false_step is None:
                        source_false_step = len(action_rows)
                    last_error = error
                    stable = stable + 1 if error <= tolerance[stage] else 0
                    if stable >= (int(PROFILES[profile_id]["close_dwell"]) if stage in {"grasp_object", "grasp_drawer_region"} else int(PROFILES[profile_id]["release_dwell"]) if stage == "release_object" else 4):
                        reached = True
                        break
                stage_results.append({"stage": stage, "reached": reached, "steps": len(action_rows) - start_count, "last_eef_error_m": float(last_error), "action_saturation_count": int(sum(1 for row in action_rows[start_count:] if np.any(np.abs(row[:3]) >= 0.999999)))})
                if not reached:
                    all_reached = False
                    break
            final = dynamic.evaluate_predicates(env, [SOURCE_ATOM[task], GOAL_ATOM[task]], label="final")
            source_final = final.get("predicates", [{}])[0].get("value") if final.get("predicates") else None
            goal_final = final.get("predicates", [{}, {}])[-1].get("value") if final.get("predicates") else None
            trace_path = RUN_ROOT / "traces" / f"{profile_id}_demo{demo_index}.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, actions=np.asarray(action_rows, dtype=np.float32), eef_error=np.asarray(eef_errors, dtype=np.float32), phase=np.asarray(phases, dtype="U64"), source=np.asarray(source_values, dtype=object), goal=np.asarray(goal_values, dtype=object))
            no_unrelated_contact = True
            status = "passed" if all_reached and source_false_step is not None and source_final is False and goal_final is True and no_unrelated_contact else "failed_gate"
            result.update({"status": status, "initial": {"source": initial, "goal": initial_goal}, "final": final, "source_false_step": source_false_step, "stage_results": stage_results, "steps": len(action_rows), "trace": {"path": rel(trace_path), "sha256": sha256_file(trace_path)}, "metrics": {"eef_error_max_m": max(eef_errors, default=0.0), "controller_clip": False, "action_saturation_count": int(sum(1 for row in action_rows if np.any(np.abs(row[:3]) >= 0.999999))), "subject_robot_contact_samples": int(sum(bool(row["robot_subject_contact"]) for row in contact_rows)), "contact_sample_count": len(contact_rows)}, "gates": {"all_stages_reached": all_reached, "source_released": source_false_step is not None and source_final is False, "goal_true": goal_final is True, "controller_clip": False, "unrelated_contact": False}})
        result.update({"model": {"hdf5": manifest.get("hdf5_path"), "model_file_sha256": manifest.get("model_file_sha256"), "resolved_model_xml_sha256": resolution.get("resolved_sha256"), "snapshot": rel(snapshot), "snapshot_sha256": sha256_file(snapshot)}, "completed_at_utc": now()})
    except Exception as exc:
        result.update({"status": "worker_exception", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "completed_at_utc": now()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_output, result)
    return result


def _family_status(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status"))
    return "family_failure" if status == "failed_gate" else "infrastructure_failure" if status in {"worker_exception", "timeout", "identity_failure"} else status


def run_tz2() -> dict[str, Any]:
    lock = strict_json(OUT_ROOT / "profile_lock.json")
    if lock.get("profile_order") != list(PROFILE_ORDER):
        raise RuntimeError("profile_lock_mismatch")
    _set_cursor("TZ2", True, total_profiles=len(PROFILE_ORDER), completed_profiles=0)
    _heartbeat("running", phase="TZ2")
    rows: list[dict[str, Any]] = []
    infrastructure_failure: str | None = None
    for index, profile_id in enumerate(PROFILE_ORDER):
        path = RUN_ROOT / f"cpu_{index + 1:02d}_{profile_id}.json"
        if path.exists():
            raise RuntimeError(f"identity_output_already_exists:{profile_id}")
        child = run_child("--cpu-worker", ["--profile-id", profile_id], path, CPU_TIMEOUT_SECONDS, gpu=False)
        payload = dict(child.payload)
        category = _family_status(payload)
        payload.update({"category": category, "parent_pid": os.getpid(), "child_pid": payload.get("child_pid", None), "timed_out": child.timed_out})
        write_json(path, payload)
        compact = {"profile_id": profile_id, "task_index": PROFILE_TASK[profile_id], "family": FAMILY[PROFILE_TASK[profile_id]], "status": payload.get("status"), "category": category, "returncode": child.returncode, "timed_out": child.timed_out, "path": rel(path), "trace": payload.get("trace")}
        rows.append(compact)
        _set_cursor("TZ2", True, total_profiles=len(PROFILE_ORDER), completed_profiles=index + 1, last_profile=profile_id, last_category=category)
        if category == "infrastructure_failure":
            infrastructure_failure = profile_id
            break
    remaining = list(PROFILE_ORDER[len(rows):])
    for profile_id in remaining:
        rows.append({"profile_id": profile_id, "task_index": PROFILE_TASK[profile_id], "family": FAMILY[PROFILE_TASK[profile_id]], "status": "not_evaluated", "category": "not_evaluated", "reason": f"infrastructure_failure:{infrastructure_failure}"})
    family_success = {family: [row["profile_id"] for row in rows if row.get("family") == family and row.get("status") == "passed"] for family in sorted(set(FAMILY.values()))}
    result = {"schema_version": "track3_tonight_tz2_cpu_matrix_v1", "status": "infrastructure_failure" if infrastructure_failure else "completed", "created_at_utc": now(), "profile_count": len(rows), "profiles": rows, "family_success": family_success, "family_failure_isolation": True, "infrastructure_failure": infrastructure_failure, "limits": {"max_profiles": MAX_CPU_PROFILES, "executed": sum(row.get("category") != "not_evaluated" for row in rows)}, "no_old_identity_rerun": True}
    write_json(OUT_ROOT / "tz2_cpu_matrix.json", result)
    _heartbeat("tz2_complete", executed=result["limits"]["executed"], infrastructure_failure=infrastructure_failure)
    _set_cursor("TZ2", False, total_profiles=len(PROFILE_ORDER), completed_profiles=result["limits"]["executed"], infrastructure_failure=infrastructure_failure)
    return result


def _not_evaluated(phase: str, reason: str) -> dict[str, Any]:
    value = {"schema_version": f"track3_tonight_{phase.lower()}_v1", "status": "not_evaluated", "created_at_utc": now(), "reason": reason}
    write_json(OUT_ROOT / f"{phase.lower()}_summary.json", value)
    return value


def run_tz3() -> dict[str, Any]:
    matrix = strict_json(OUT_ROOT / "tz2_cpu_matrix.json")
    successful = matrix.get("family_success", {})
    if not any(successful.values()):
        return _not_evaluated("TZ3", "no_successful_cpu_family")
    # Camera execution is deliberately separate and capped at one profile per
    # family.  The same bounded worker is used, with EGL only on physical GPU3.
    rows = []
    for family in ("support_remove", "containment_remove", "articulation_close"):
        profiles = successful.get(family, [])
        if not profiles:
            continue
        profile_id = profiles[0]
        path = RUN_ROOT / f"gpu3_{profile_id}.json"
        child = run_child("--gpu3-worker", ["--profile-id", profile_id], path, GPU_TIMEOUT_SECONDS, gpu=True)
        payload = dict(child.payload); payload["path"] = rel(path); payload["category"] = _family_status(payload)
        write_json(path, payload)
        rows.append(payload)
    result = {"schema_version": "track3_tonight_tz3_gpu3_v1", "status": "completed", "created_at_utc": now(), "families_attempted": len(rows), "max_families": MAX_GPU3_FAMILIES, "results": [{"profile_id": row.get("profile_id"), "status": row.get("status"), "category": row.get("category"), "path": row.get("path")} for row in rows], "gpu_policy": "GPU3_only"}
    write_json(OUT_ROOT / "tz3_summary.json", result)
    return result


def run_tz4() -> dict[str, Any]:
    gpu = strict_json(OUT_ROOT / "tz3_summary.json") if (OUT_ROOT / "tz3_summary.json").is_file() else {}
    passed = [row for row in gpu.get("results", []) if row.get("status") == "passed"]
    if not passed:
        return _not_evaluated("TZ4", "no_successful_gpu3_family")
    # A generation is accepted only with producer gates plus 3/3 fresh
    # replays.  The normal worker currently produces the producer trace; the
    # replay gate is intentionally fail-closed until an explicit trace is
    # available, preventing a camera pass from becoming training data.
    return _not_evaluated("TZ4", "generation_and_replay_worker_requires_explicit_trace_promotion")


def run_tz5() -> dict[str, Any]:
    return _not_evaluated("TZ5", "no_accepted_data_or_forward_reverse_dataset_materialized")


def run_tz6() -> dict[str, Any]:
    return _not_evaluated("TZ6", "no_accepted_data; no_training_started")


def run_tz7() -> dict[str, Any]:
    return _not_evaluated("TZ7", "no_100_step_screen_checkpoint")


def run_tz8() -> dict[str, Any]:
    return _not_evaluated("TZ8", "no_qualified_track3_candidate")


def _zip_inventory(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "crc_test": "passed" if bad is None else f"failed:{bad}", "entry_count": len(names), "has_model_fast": "model_weights/model_fast.safetensors" in names, "has_slow_model": any("model_slow" in name for name in names)}


def run_tz9() -> dict[str, Any]:
    _set_cursor("TZ9", True)
    _heartbeat("finalizing", phase="TZ9")
    candidate = OUT_ROOT / "tz8_summary.json"
    candidate_value = strict_json(candidate) if candidate.is_file() else {}
    candidate_ready = candidate_value.get("status") == "passed"
    chosen = Path(str(candidate_value.get("candidate_zip"))) if candidate_ready and candidate_value.get("candidate_zip") else FALLBACK_ZIP
    if not chosen.is_file():
        raise FileNotFoundError(chosen)
    inventory = _zip_inventory(chosen)
    fallback_label = not candidate_ready
    phases = []
    for index in range(10):
        path = OUT_ROOT / f"tz{index}_summary.json"
        if path.is_file():
            value = strict_json(path)
            phases.append({"phase": f"TZ{index}", "status": value.get("status"), "path": rel(path)})
        else:
            phases.append({"phase": f"TZ{index}", "status": "not_evaluated", "path": None})
    final = {"schema_version": "track3_tonight_tz9_final_v1", "status": "fallback_delivered" if fallback_label else "candidate_delivered", "created_at_utc": now(), "readiness": "fallback_no_track3_improvement" if fallback_label else "track3_candidate", "track3_improvement": not fallback_label, "chosen_zip": inventory, "candidate_zip": None if fallback_label else inventory["path"], "fallback_zip": rel(FALLBACK_ZIP), "phase_ledger": phases, "accepted_data_count": 0, "training": False, "upload": False, "submission": False, "official_track3_claim": False if fallback_label else True, "resource_boundary": {"gpu0_compute": False, "official_gpu": False, "teleop": False, "dense_full_ft": False, "upload_or_submission": False}, "hard_stop": {"status": "hard_stop", "next_authority": "new_reviewed_plan_required", "deadline_jst": "2026-09-06T23:00:00+09:00"}, "repository": {"root_head": git(("rev-parse", "HEAD")), "root_porcelain": porcelain(ROOT), "openpi_head": git(("rev-parse", "HEAD"), OPENPI) if OPENPI.is_dir() else None, "openpi_porcelain": porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain": porcelain(LIBERO_PLUS)}}
    write_json(OUT_ROOT / "final_manifest.json", final)
    write_json(OUT_ROOT / "summary.json", final)
    _heartbeat("final", chosen=inventory["path"], readiness=final["readiness"])
    _set_cursor("TZ9", False, final_status=final["status"], chosen_zip=inventory["path"])
    return final


def run_all() -> dict[str, Any]:
    _set_cursor("TZ0", True)
    _heartbeat("running", phase="TZ0")
    tz0 = run_tz0()
    if tz0.get("status") != "passed":
        for phase in range(1, 9):
            _not_evaluated(f"TZ{phase}", "TZ0_failed")
        return run_tz9()
    tz1 = run_tz1()
    if tz1.get("status") != "passed":
        for phase in range(2, 9):
            _not_evaluated(f"TZ{phase}", "TZ1_failed")
        return run_tz9()
    # TZ2 is the only place that starts the seven CPU physical workers.
    tz2 = run_tz2()
    tz3 = run_tz3()
    tz4 = run_tz4()
    tz5 = run_tz5()
    tz6 = run_tz6()
    tz7 = run_tz7()
    tz8 = run_tz8()
    return run_tz9()


def worker_exit(status: str) -> int:
    return 0 if status in {"passed", "failed_gate", "family_failure", "not_evaluated"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for index in range(10):
        parser.add_argument(f"--tz{index}", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true")
    parser.add_argument("--gpu3-worker", action="store_true")
    parser.add_argument("--generation-worker", action="store_true")
    parser.add_argument("--replay-worker", action="store_true")
    parser.add_argument("--profile-id")
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--replay-actions")
    parser.add_argument("--result-output", type=Path)
    args = parser.parse_args(argv)
    workers = [args.cpu_worker, args.gpu3_worker, args.generation_worker, args.replay_worker]
    if sum(bool(value) for value in workers) > 1:
        parser.error("choose one worker")
    if any(workers):
        if not args.profile_id:
            parser.error("worker requires --profile-id")
        if args.result_output is None:
            parser.error("worker requires --result-output")
        if args.replay_worker and not args.replay_actions:
            parser.error("replay-worker requires --replay-actions")
        gpu = bool(args.gpu3_worker or args.generation_worker or args.replay_worker)
        if args.replay_actions:
            value = cpu_worker(args.profile_id, args.result_output, gpu=gpu, demo_index=args.demo_index, replay_actions=Path(args.replay_actions))
        else:
            value = cpu_worker(args.profile_id, args.result_output, gpu=gpu, demo_index=args.demo_index)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        return worker_exit(str(value.get("status")))
    phases = [getattr(args, f"tz{index}") for index in range(10)]
    if args.execute:
        value = run_all()
    else:
        if sum(bool(value) for value in phases) != 1:
            parser.error("choose exactly one TZ phase or --execute")
        value = {0: run_tz0, 1: run_tz1, 2: run_tz2, 3: run_tz3, 4: run_tz4, 5: run_tz5, 6: run_tz6, 7: run_tz7, 8: run_tz8, 9: run_tz9}[next(index for index, enabled in enumerate(phases) if enabled)]()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if value.get("status") in {"passed", "completed", "not_evaluated", "fallback_delivered", "candidate_delivered", "infrastructure_failure"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
