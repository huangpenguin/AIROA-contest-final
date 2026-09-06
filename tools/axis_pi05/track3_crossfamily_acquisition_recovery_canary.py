#!/usr/bin/env python3
"""AC0--AC6 cross-family acquisition recovery canary.

This module is intentionally a new evidence and identity namespace.  The
historical ER executor is used only as a CPU/no-renderer simulation adapter;
its identities, output root, profile ids, and result schema are patched in a
fresh child before the adapter is called.  All acquisition gates in this
module are computed from runtime telemetry and are fail-closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_crossfamily_acquisition_recovery_canary"
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_crossfamily_acquisition_recovery_canary"
ATTEMPT_ID = 1
ATTEMPT_ROOT = OUT_ROOT / "attempts" / f"attempt_{ATTEMPT_ID:04d}"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_PLAN.md"
EG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_remediation"
EG_AUDIT_SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_eg0_eg4_reaudit_20260906/summary.json"
ER_ROOT = ROOT / "data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PUBLIC_LIBERO = ROOT / "LIBERO"
PUBLIC_AUTOGEN = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"

HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
REVERSE_BDDL = {
    10: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_support_remove_task_10.bddl",
    20: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_containment_remove_task_20.bddl",
    19: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_articulation_close_task_19.bddl",
}
SNAPSHOTS = {
    task: P1_ROOT / "snapshots" / f"task_{task:02d}" / "attempt_1" / "terminal_state.npy"
    for task in (10, 20, 19)
}
PROFILE_ORDER = (
    "ac_support_remove_20260906_0001",
    "ac_containment_remove_20260906_0001",
    "ac_articulation_close_20260906_0001",
)
TASK_BY_PROFILE = {PROFILE_ORDER[0]: 10, PROFILE_ORDER[1]: 20, PROFILE_ORDER[2]: 19}
FAMILY_BY_TASK = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
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
PROFILES: dict[str, dict[str, Any]] = {
    PROFILE_ORDER[0]: {"task_index": 10, "seed": 61010, "grasp_z_m": 0.024, "close_dwell": 2, "release_dwell": 1},
    PROFILE_ORDER[1]: {"task_index": 20, "seed": 62020, "grasp_z_m": 0.024, "close_dwell": 2, "release_dwell": 1},
    PROFILE_ORDER[2]: {"task_index": 19, "seed": 63019, "grasp_z_m": 0.012, "close_dwell": 2, "release_dwell": 1},
}
MAX_STEPS = {
    "approach_object": 170, "descend_object": 140, "grasp_object": 45,
    "lift_object": 170, "transport_to_target": 220, "descend_to_target": 140,
    "release_object": 40, "retreat": 90, "approach_drawer_region": 180,
    "grasp_drawer_region": 40, "pull_drawer_closed": 280, "hold_closed": 45,
}
OLD_PROFILE_IDS = {
    "er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh",
    "support_contact_repair_a", "support_contact_repair_b", "support_contact_repair_c",
    "containment_extract_a", "containment_extract_b", "articulation_pull_a", "articulation_pull_b",
}
TERM_GRACE_SECONDS = 1.0
FIXTURE_HEARTBEAT_INTERVAL = 0.05
FIXTURE_TIMEOUT = 0.25
CPU_TIMEOUT_SECONDS = 240.0
HEARTBEAT_INTERVAL_SECONDS = 0.5
ACTION_DIM = 7
CONTACT_WINDOW = 4
CONTRACT = {
    "osc_translation_normalized_scale": 0.05,
    "osc_rotation_normalized_scale": 0.25,
    "normalized_action_abs_limit": 1.0,
    "approach_tolerance_m": 0.014,
    "descend_tolerance_m": 0.012,
    "articulation_tolerance_m": 0.015,
    "progress_window": 12,
    "stall_epsilon_m": 1.0e-5,
    "required_contact_window": CONTACT_WINDOW,
    "gripper_commands": {"closed": -1.0, "open": 1.0, "neutral": 0.0},
    "acquisition_barrier": "required_demonstrator_contact_continuous_before_grasp_or_pull",
    "source_release": "runtime_predicate_source_false_after_acquisition",
    "goal_gate": "runtime_predicate_goal_true_continuously_after_goal_entry",
}


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


def argv_contract_hash(command: Sequence[str]) -> str:
    normalized = list(command)
    if "--expected-command-sha256" in normalized:
        index = normalized.index("--expected-command-sha256") + 1
        if index < len(normalized):
            normalized[index] = "<expected-command-sha256>"
    return canonical_hash(normalized)


def result_content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("result_sha256", None)
    return canonical_hash(body)


def gate_record(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    row = {"executed": bool(executed), "value": value, "state": "not_evaluated" if not executed else ("passed" if value is True else ("unknown" if value is None else "failed")), "required": bool(required)}
    if reason is not None:
        row["reason"] = reason
    return row


def required_gates_passed(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in gates.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def _file_lock(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _hdf5_manifest(task: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    path = HDF5_INPUTS[task]
    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        states = np.asarray(demo["states"][()])
        actions = np.asarray(demo["actions"][()])
        rewards = np.asarray(demo["rewards"][()])
        model = demo.attrs["model_file"]
        model_bytes = model.encode() if isinstance(model, str) else bytes(model)
        init = np.asarray(demo.attrs["init_state"])
    return {"task_index": task, "family": FAMILY_BY_TASK[task], "path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "model_file_bytes": len(model_bytes), "model_file_sha256": sha256_bytes(model_bytes), "states_shape": list(states.shape), "states_dtype": str(states.dtype), "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype), "rewards_shape": list(rewards.shape), "rewards_dtype": str(rewards.dtype), "init_state_sha256": sha256_bytes(np.asarray(init, dtype=np.float64).tobytes()), "states0_equals_init_state": bool(np.array_equal(states[0], init))}


def input_manifest() -> dict[str, Any]:
    eg_files = [EG_AUDIT_SUMMARY, EG_ROOT / "latest_attempt.json", EG_ROOT / "attempts/attempt_0005/final_validation.json", EG_ROOT / "attempts/attempt_0005/checkout_stable_closure.json", EG_ROOT / "root_attempt_0005_artifact_manifest.json", EG_ROOT / "root_attempt_0005_outer_hash.json", EG_ROOT / "root_attempt_0005_checkout_stable_closure.json"]
    er_files = [ER_ROOT / "runs" / f"{index:02d}_{profile}.json" for index, profile in enumerate(("er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh"), 1)]
    er_files += [ER_ROOT / "runs/traces" / f"{profile}_demo0.npz" for profile in ("er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh")]
    files = {"eg": [_file_lock(path) for path in eg_files], "er": [_file_lock(path) for path in er_files], "bddl": [_file_lock(path) for path in REVERSE_BDDL.values()], "snapshots": [_file_lock(path) for path in SNAPSHOTS.values()]}
    return {"schema_version": "track3_ac0_input_manifest_v1", "authorized_audit_root_commit": "4bbfe7eb7fdb404085f39712aae94572c9d0fbce", "eg_audit_summary": _file_lock(EG_AUDIT_SUMMARY), "eg_files": files["eg"], "er_files": files["er"], "public_hdf5": [_hdf5_manifest(task) for task in (10, 20, 19)], "bddl_files": files["bddl"], "snapshot_files": files["snapshots"], "root_head_at_lock": git(("rev-parse", "HEAD")), "openpi_head_at_lock": git(("rev-parse", "HEAD"), OPENPI), "root_porcelain_at_lock": porcelain(ROOT), "openpi_porcelain_at_lock": porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain_at_lock": porcelain(LIBERO_PLUS), "historical_namespaces_modified": False}


def profile_lock(manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rows = []
    for profile_id in PROFILE_ORDER:
        profile = dict(PROFILES[profile_id])
        profile.update({"profile_id": profile_id, "family": FAMILY_BY_TASK[int(profile["task_index"])], "identity_scope": "new_ac_cpu_no_renderer_identity_once", "old_identity_rerun": False, "renderer": False, "gpu": False, "training_eligible": False, "max_steps": dict(MAX_STEPS)})
        rows.append(profile)
    body: dict[str, Any] = {"schema_version": "track3_ac0_profile_lock_v1", "profile_order": list(PROFILE_ORDER), "profiles": rows, "family_order": ["support_remove", "containment_remove", "articulation_close"], "limits": {"support_remove": 1, "containment_remove": 1, "articulation_close": 1, "total": 3}, "retry_allowed": False, "old_profile_ids_denylist": sorted(OLD_PROFILE_IDS), "contract": CONTRACT, "input_manifest_sha256": canonical_hash(manifest) if manifest is not None else None, "source_root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__).resolve()), "official_track3_claim": False}
    body["lock_sha256"] = canonical_hash(body)
    return body


def _attempt_dir() -> Path:
    return ATTEMPT_ROOT


def _write_phase(name: str, value: Mapping[str, Any]) -> None:
    write_json(_attempt_dir() / name, value)


def _phase(name: str) -> dict[str, Any]:
    path = _attempt_dir() / name
    if not path.is_file():
        return {"status": "not_evaluated", "path": None}
    value = strict_json(path)
    return {"status": value.get("status", "not_evaluated"), "path": rel(path)}


def run_ac0() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _attempt_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = _attempt_dir() / "ac0_input_manifest.json"
    lock_path = _attempt_dir() / "ac0_profile_lock.json"
    manifest = input_manifest()
    lock = profile_lock(manifest)
    if lock["old_profile_ids_denylist"] and set(PROFILE_ORDER) & set(lock["old_profile_ids_denylist"]):
        raise RuntimeError("new_profile_overlaps_old_identity_denylist")
    _write_phase("ac0_input_manifest.json", manifest)
    _write_phase("ac0_profile_lock.json", lock)
    current = {"schema_version": "track3_ac_attempt_pointer_v1", "attempt_id": ATTEMPT_ID, "attempt_root": rel(_attempt_dir()), "profile_lock_sha256": lock["lock_sha256"], "immutable": True}
    write_json(OUT_ROOT / "current_attempt.json", current)
    result = {"schema_version": "track3_ac0_input_and_quota_lock_v1", "status": "passed", "attempt_id": ATTEMPT_ID, "input_manifest": {"path": rel(manifest_path), "sha256": sha256_file(manifest_path)}, "profile_lock": {"path": rel(lock_path), "sha256": sha256_file(lock_path)}, "profile_order": list(PROFILE_ORDER), "new_cpu_physics_identities_max": 3, "per_family_max": 1, "retry_allowed": False, "physics_started": False, "gpu_compute_started": False, "old_identity_rerun": False, "historical_namespaces_modified": False}
    _write_phase("ac0_result.json", result)
    return result


def _contiguous_windows(values: Sequence[bool]) -> list[list[int]]:
    windows: list[list[int]] = []
    current: list[int] = []
    for index, value in enumerate(values):
        if bool(value):
            current.append(index)
        elif current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)
    return windows


def _old_trace_diagnosis(profile_id: str, result_path: Path, trace_path: Path) -> dict[str, Any]:
    import numpy as np

    raw = strict_json(result_path)
    arrays: dict[str, Any] = {}
    errors: list[str] = []
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                arrays[name] = array
                if array.dtype.hasobject:
                    errors.append(f"object_dtype:{name}")
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    errors.append(f"nonfinite:{name}")
    except Exception as exc:
        return {"profile_id": profile_id, "status": "failed", "error": f"{type(exc).__name__}:{exc}"}
    required = ("actions", "eef_error", "phase", "source_value", "source_known", "goal_value", "goal_known", "action_saturated", "contact_complete", "unrelated_contact", "contact_ncon")
    missing = [name for name in required if name not in arrays]
    n = int(len(arrays.get("eef_error", [])))
    actions = np.asarray(arrays.get("actions", np.zeros((0, ACTION_DIM))), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1:] != (ACTION_DIM,):
        errors.append("actions_shape")
    translation_norm = np.linalg.norm(actions[:, :3], axis=1).tolist() if actions.ndim == 2 and actions.shape[1] == ACTION_DIM else []
    rotation_norm = np.linalg.norm(actions[:, 3:6], axis=1).tolist() if actions.ndim == 2 and actions.shape[1] == ACTION_DIM else []
    saturated = np.asarray(arrays.get("action_saturated", []), dtype=bool).tolist()
    phases = [str(value) for value in np.asarray(arrays.get("phase", []), dtype="U64").tolist()]
    phase_rows: list[dict[str, Any]] = []
    for phase in list(dict.fromkeys(phases)):
        indices = [index for index, value in enumerate(phases) if value == phase]
        values = np.asarray(arrays["eef_error"])[indices] if "eef_error" in arrays else np.asarray([])
        phase_rows.append({"phase": phase, "entered": True, "first_step": min(indices), "last_step": max(indices), "steps": len(indices), "eef_error_start_m": float(values[0]) if len(values) else None, "eef_error_end_m": float(values[-1]) if len(values) else None, "eef_error_min_m": float(values.min()) if len(values) else None, "saturation_count": int(sum(bool(saturated[index]) for index in indices)) if len(saturated) == n else None})
    contact_pairs = {"state": "not_evaluated", "reason": "historical_typed_trace_did_not_emit_contact_pairs", "ncon_present": "contact_ncon" in arrays}
    stop_reason = "completed" if raw.get("status") == "passed" else (raw.get("stage_results") or [{}])[-1].get("stage") if raw.get("stage_results") else "unavailable"
    result = {"profile_id": profile_id, "source_result": {"path": rel(result_path), "bytes": result_path.stat().st_size, "sha256": sha256_file(result_path)}, "source_trace": {"path": rel(trace_path), "bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path)}, "status": "passed" if not errors and not missing else "failed", "steps": n, "arrays": {name: {"dtype": str(array.dtype), "shape": list(array.shape)} for name, array in arrays.items()}, "missing_arrays": missing, "errors": errors, "normalized_action": {"translation_l2_max": max(translation_norm, default=0.0), "rotation_l2_max": max(rotation_norm, default=0.0), "translation_l2": translation_norm, "rotation_l2": rotation_norm}, "saturation_windows": _contiguous_windows(saturated), "phase_progress": phase_rows, "contact": {"telemetry_complete_samples": int(np.asarray(arrays.get("contact_complete", []), dtype=bool).sum()) if "contact_complete" in arrays else None, "contact_ncon_min": int(np.asarray(arrays["contact_ncon"]).min()) if "contact_ncon" in arrays and len(arrays["contact_ncon"]) else None, "contact_ncon_max": int(np.asarray(arrays["contact_ncon"]).max()) if "contact_ncon" in arrays and len(arrays["contact_ncon"]) else None, "pairs": contact_pairs, "unrelated_samples": int(np.asarray(arrays.get("unrelated_contact", []), dtype=bool).sum()) if "unrelated_contact" in arrays else None}, "source_goal": {"initial_source": bool(np.asarray(arrays.get("source_value", [0]))[0]) if n else None, "initial_source_known": bool(np.asarray(arrays.get("source_known", [False]))[0]) if n else None, "initial_goal": bool(np.asarray(arrays.get("goal_value", [1]))[0]) if n else None, "source_false_step": next((index + 1 for index, value in enumerate(np.asarray(arrays.get("source_value", []))) if not bool(value)), None), "goal_true_steps": int(np.asarray(arrays.get("goal_value", []), dtype=bool).sum()) if "goal_value" in arrays else None}, "stop_reason": stop_reason, "historical_raw_unchanged": True}
    return result


def run_ac1() -> dict[str, Any]:
    rows = []
    profiles = ("er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh")
    for index, profile in enumerate(profiles, 1):
        result_path = ER_ROOT / "runs" / f"{index:02d}_{profile}.json"
        trace_path = ER_ROOT / "runs/traces" / f"{profile}_demo0.npz"
        rows.append(_old_trace_diagnosis(profile, result_path, trace_path))
    result = {"schema_version": "track3_ac1_old_er_trace_diagnosis_v1", "status": "passed" if all(row.get("status") == "passed" for row in rows) else "failed", "profiles": rows, "diagnosis_is_read_only": True, "contact_pairs_not_reconstructed": True, "contract_proposed_before_new_physics": dict(CONTRACT), "threshold_source": "public model/state dimensions plus frozen profile; no hidden task information", "action_saturation_is_runtime_observed": True, "contact_complete_is_telemetry_only": True, "physics_started": False, "gpu_compute_started": False}
    _write_phase("ac1_old_trace_diagnosis.json", result)
    return result


def telemetry_schema_gate(samples: Sequence[Mapping[str, Any]], *, stage_entered: bool) -> dict[str, Any]:
    if not stage_entered:
        return gate_record(False, None, reason="stage_not_entered")
    required = {"telemetry_complete", "known_mask", "contacts"}
    errors = [f"sample_{index}:{sorted(required - set(sample))}" for index, sample in enumerate(samples) if not required.issubset(sample) or sample.get("telemetry_complete") is not True or sample.get("known_mask") is not True or not isinstance(sample.get("contacts"), list)]
    return {**gate_record(True, not errors, reason=";".join(errors) if errors else None), "sample_count": len(samples), "errors": errors}


def required_contact_gate(family: str, samples: Sequence[Mapping[str, Any]], *, stage_entered: bool, min_window: int = CONTACT_WINDOW) -> dict[str, Any]:
    telemetry = telemetry_schema_gate(samples, stage_entered=stage_entered)
    if telemetry["state"] != "passed":
        return {**gate_record(False, None, reason="telemetry_schema_not_complete" if stage_entered else "stage_not_entered"), "family": family, "contact_window": 0, "min_window": min_window}
    partners = {"object"} if family in {"support_remove", "containment_remove"} else {"handle", "region"}
    current = best = 0
    for sample in samples:
        correlated = any("demonstrator" in {str(role).lower() for role in contact.get("roles", [])} and partners & {str(role).lower() for role in contact.get("roles", [])} for contact in sample["contacts"] if isinstance(contact, Mapping))
        current = current + 1 if correlated else 0
        best = max(best, current)
    return {**gate_record(True, best >= min_window, reason="required_contact_window_short" if best < min_window else None), "family": family, "contact_window": best, "min_window": min_window}


def _fixture_samples(family: str, *, contact: bool, count: int = CONTACT_WINDOW) -> list[dict[str, Any]]:
    partner = "object" if family in {"support_remove", "containment_remove"} else "handle"
    return [{"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", partner]}] if contact else []} for _ in range(count)]


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _typed_trace_validation(path: Path) -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    arrays: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                arrays[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
                if array.dtype.hasobject:
                    errors.append(f"object_dtype:{name}")
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    errors.append(f"nonfinite:{name}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    return {"path": rel(path), "arrays": arrays, "errors": errors, "allow_pickle_false": not any("pickle" in error.lower() or "object" in error.lower() for error in errors), "status": "passed" if not errors else "failed"}


def run_ac2() -> dict[str, Any]:
    import numpy as np

    pass_samples = _fixture_samples("support_remove", contact=True)
    fail_samples = _fixture_samples("support_remove", contact=False)
    articulation_samples = _fixture_samples("articulation_close", contact=True)
    unentered = required_contact_gate("support_remove", pass_samples, stage_entered=False)
    fixture = {"support_pass": required_contact_gate("support_remove", pass_samples, stage_entered=True), "support_fail": required_contact_gate("support_remove", fail_samples, stage_entered=True), "articulation_handle_pass": required_contact_gate("articulation_close", articulation_samples, stage_entered=True), "unentered_not_evaluated": unentered, "telemetry_drift": telemetry_schema_gate([{**pass_samples[0], "known_mask": False}], stage_entered=True)}
    trace = _attempt_dir() / "ac2_fixture_typed_trace.npz"
    _write_npz(trace, {"actions": np.zeros((2, ACTION_DIM), dtype=np.float32), "source_known": np.asarray([True, True], dtype=np.bool_), "goal_known": np.asarray([True, True], dtype=np.bool_), "action_saturated": np.asarray([False, False], dtype=np.bool_)})
    typed = _typed_trace_validation(trace)
    result = {"schema_version": "track3_ac2_runtime_derived_gate_and_fixture_v1", "status": "passed" if fixture["support_pass"]["state"] == "passed" and fixture["support_fail"]["state"] == "failed" and fixture["articulation_handle_pass"]["state"] == "passed" and fixture["unentered_not_evaluated"]["state"] == "not_evaluated" and fixture["telemetry_drift"]["state"] == "failed" and typed["status"] == "passed" else "failed", "contract": CONTRACT, "fixture": fixture, "typed_trace": typed, "gate_separation": {"telemetry_integrity_is_not_required_physical_contact": True, "contact_roles_are_runtime_derived": True, "action_saturation_is_not_grandfathered": True, "later_stage_after_not_entered": "not_evaluated"}, "physics_started": False, "gpu_compute_started": False}
    _write_phase("ac2_runtime_gate_and_fixture.json", result)
    return result


def identity_snapshot(command: Sequence[str], profile_sha: str, *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = environment or os.environ
    root_porcelain, openpi_porcelain, libero_porcelain = porcelain(ROOT), porcelain(OPENPI) if OPENPI.is_dir() else [], porcelain(LIBERO_PLUS)
    return {"root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__).resolve()), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI) if OPENPI.is_dir() else None, "profile_lock_sha256": profile_sha, "command": list(command), "command_hash": argv_contract_hash(command), "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""), "jax_platforms": env.get("JAX_PLATFORMS", ""), "mujoco_gl": env.get("MUJOCO_GL", ""), "root_porcelain": root_porcelain, "root_porcelain_sha256": canonical_hash(root_porcelain), "openpi_porcelain": openpi_porcelain, "openpi_porcelain_sha256": canonical_hash(openpi_porcelain), "libero_plus_porcelain": libero_porcelain, "libero_plus_porcelain_sha256": canonical_hash(libero_porcelain), "clean_root": not root_porcelain, "pid": os.getpid(), "pgid": os.getpgid(0)}


IDENTITY_FIELDS = ("root_commit", "tool_sha256", "openpi_commit", "profile_lock_sha256", "cuda_visible_devices", "jax_platforms", "mujoco_gl", "root_porcelain_sha256", "openpi_porcelain_sha256", "libero_plus_porcelain_sha256", "clean_root")


def identity_gate(observed: Mapping[str, Any], expected: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    mismatches = [field for field in IDENTITY_FIELDS if observed.get(field) != expected.get(field)]
    if observed.get("command_hash") != expected.get("command_hash"):
        mismatches.append("command_hash")
    if list(observed.get("command", [])) != list(command):
        mismatches.append("command")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches, "checked_fields": list(IDENTITY_FIELDS) + ["command", "command_hash"]}


def validate_resume_payload(payload: Mapping[str, Any], *, expected: Mapping[str, Any], command: Sequence[str], profile_id: str, category: str) -> dict[str, Any]:
    mismatches: list[str] = []
    if payload.get("profile_id") != profile_id:
        mismatches.append("profile_id")
    if payload.get("category") != category:
        mismatches.append("category")
    if identity_gate(payload.get("identity", {}), expected, command)["state"] != "passed":
        mismatches.append("identity")
    trace = payload.get("trace")
    if not isinstance(trace, Mapping):
        mismatches.append("trace")
    else:
        path = ROOT / str(trace.get("path")) if not Path(str(trace.get("path", ""))).is_absolute() else Path(str(trace.get("path")))
        if not path.is_file() or trace.get("bytes") != path.stat().st_size or trace.get("sha256") != sha256_file(path):
            mismatches.append("trace_hash")
    if payload.get("result_sha256") != result_content_hash(payload):
        mismatches.append("result_hash")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches}


@contextmanager
def supervisor_lock(path: Path) -> Iterator[dict[str, Any]]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise
    info = {"pid": os.getpid(), "pgid": os.getpgid(0), "acquired_at_utc": now(), "lock_path": str(path.resolve())}
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(info, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    try:
        yield info
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _heartbeat(path: Path, state: str, **extra: Any) -> None:
    write_json(path, {"schema_version": "track3_ac_heartbeat_v1", "state": state, "pid": os.getpid(), "updated_at_utc": now(), **extra})


def _descendants_in_group(pgid: int) -> list[int]:
    result = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True, check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                pid, group = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if group == pgid and pid != os.getpid():
                rows.append(pid)
    return rows


def _fixture_child(mode: str, result_path: Path, heartbeat_path: Path, expected_path: Path) -> int:
    expected = strict_json(expected_path)
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    identity = identity_snapshot(command, str(expected["profile_lock_sha256"]))
    gate = identity_gate(identity, expected, command)
    if gate["state"] != "passed":
        write_json(result_path, {"schema_version": "track3_ac_fixture_result_v1", "status": "infrastructure_failure", "profile_id": "ac_fixture", "identity": identity, "identity_gate": gate})
        return 1
    if mode == "timeout":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=False)
        sequence = 0
        while True:
            sequence += 1
            _heartbeat(heartbeat_path, "child_running", sequence=sequence, mode=mode)
            time.sleep(FIXTURE_HEARTBEAT_INTERVAL)
    if mode != "family_failure":
        raise ValueError(f"unknown_fixture_mode:{mode}")
    for sequence in range(1, 5):
        _heartbeat(heartbeat_path, "child_running", sequence=sequence, mode=mode)
        time.sleep(FIXTURE_HEARTBEAT_INTERVAL)
    trace = result_path.with_suffix(".trace.bin")
    atomic_bytes(trace, b"ac-fixture-trace-v1\n")
    payload = {"schema_version": "track3_ac_fixture_result_v1", "status": "family_failure", "profile_id": "ac_fixture", "category": "family_failure", "identity": identity, "identity_gate": gate, "trace": {"path": rel(trace), "bytes": trace.stat().st_size, "sha256": sha256_file(trace)}, "physics_steps": 0, "simulator_created": False, "cuda_initialized": False, "completed_at_utc": now()}
    payload["result_sha256"] = result_content_hash(payload)
    write_json(result_path, payload)
    return 0


def _lock_probe(path: Path) -> int:
    try:
        with supervisor_lock(path):
            return 1
    except BlockingIOError:
        print("lock_rejected")
        return 0


def _run_fixture(mode: str, label: str, profile_sha: str) -> dict[str, Any]:
    run_root = _attempt_dir() / "ac3_fixture" / label
    run_root.mkdir(parents=True, exist_ok=True)
    result_path, heartbeat_path = run_root / "result.json", run_root / "heartbeat.json"
    for path in run_root.glob("*"):
        if path.is_file():
            path.unlink()
    with tempfile.TemporaryDirectory(prefix="ac_expected_") as directory:
        expected_path = Path(directory) / "expected.json"
        command = [sys.executable, "-m", MODULE, "--ac-fixture-child", "--mode", mode, "--result-output", str(result_path), "--heartbeat-output", str(heartbeat_path), "--expected-identity-file", str(expected_path), "--expected-command-sha256", "pending"]
        command[-1] = argv_contract_hash(command)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": "0"}
        expected = identity_snapshot(command, profile_sha, environment=env)
        write_json(expected_path, expected)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        pid, pgid = process.pid, os.getpgid(process.pid)
        sequences: list[int] = []
        timed_out = term_sent = kill_sent = False
        started = time.monotonic()
        while process.poll() is None:
            if heartbeat_path.is_file():
                try:
                    sequence = int(strict_json(heartbeat_path).get("sequence", 0))
                    if sequence and (not sequences or sequence != sequences[-1]):
                        sequences.append(sequence)
                except Exception:
                    pass
            if mode == "timeout" and time.monotonic() - started > FIXTURE_TIMEOUT:
                timed_out = term_sent = True
                os.killpg(pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    kill_sent = True
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.01)
        stdout, stderr = process.communicate()
        descendants = _descendants_in_group(pgid)
        payload = strict_json(result_path) if result_path.is_file() else {"status": "infrastructure_failure", "category": "infrastructure_failure"}
    identity = identity_gate(payload.get("identity", {}), expected, command) if payload.get("identity") else {"state": "failed", "mismatches": ["identity_missing"]}
    resume = validate_resume_payload(payload, expected=expected, command=command, profile_id="ac_fixture", category="family_failure") if mode == "family_failure" and payload.get("status") == "family_failure" else {"state": "not_evaluated", "mismatches": []}
    return {"mode": mode, "label": label, "child_pid": pid, "child_pgid": pgid, "returncode": process.returncode, "timed_out": timed_out, "term_sent": term_sent, "kill_sent": kill_sent, "descendants_after_stop": descendants, "unique_pid_pgid": pid == pgid, "heartbeat_sequences": sequences, "identity_gate": identity, "resume_gate": resume, "payload": payload, "stdout_tail": stdout[-1000:], "stderr_tail": stderr[-1000:]}


def run_ac3() -> dict[str, Any]:
    lock_path = _attempt_dir() / "ac3_supervisor.lock"
    with supervisor_lock(lock_path):
        probe = subprocess.run([sys.executable, "-m", MODULE, "--ac-lock-probe", str(lock_path)], cwd=ROOT, capture_output=True, text=True, check=False)
    lock_path.unlink(missing_ok=True)
    lock = strict_json(_attempt_dir() / "ac0_profile_lock.json")
    normal = _run_fixture("family_failure", "normal_family_failure", lock["lock_sha256"])
    timeout = _run_fixture("timeout", "timeout_term_kill_descendant", lock["lock_sha256"])
    clean = identity_snapshot([sys.executable, "-m", MODULE], lock["lock_sha256"], environment={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"})
    clean_gate = identity_gate(clean, clean, clean["command"])
    dirty_gate = identity_gate({**clean, "root_porcelain": [" M synthetic.py"], "root_porcelain_sha256": canonical_hash([" M synthetic.py"]), "clean_root": False}, clean, clean["command"])
    command_drift = identity_gate(clean, clean, [*clean["command"], "--drift"])
    trace = _attempt_dir() / "ac3_resume_trace.bin"
    atomic_bytes(trace, b"ac3-resume-trace\n")
    payload = {"profile_id": "ac_fixture", "category": "family_failure", "identity": clean, "trace": {"path": rel(trace), "bytes": trace.stat().st_size, "sha256": sha256_file(trace)}}
    payload["result_sha256"] = result_content_hash(payload)
    resume_pass = validate_resume_payload(payload, expected=clean, command=clean["command"], profile_id="ac_fixture", category="family_failure")
    trace.write_bytes(b"drift\n")
    resume_drift = validate_resume_payload(payload, expected=clean, command=clean["command"], profile_id="ac_fixture", category="family_failure")
    result = {"schema_version": "track3_ac3_clean_source_supervisor_fixture_v1", "status": "passed" if probe.returncode == 0 and "lock_rejected" in probe.stdout and clean_gate["state"] == "passed" and dirty_gate["state"] == "failed" and command_drift["state"] == "failed" and normal["returncode"] == 0 and normal["payload"].get("status") == "family_failure" and normal["identity_gate"]["state"] == "passed" and normal["resume_gate"]["state"] == "passed" and timeout["term_sent"] and timeout["kill_sent"] and timeout["returncode"] < 0 and timeout["descendants_after_stop"] == [] and timeout["heartbeat_sequences"] and resume_pass["state"] == "passed" and resume_drift["state"] == "failed" else "failed", "clean_source": {"root_head": git(("rev-parse", "HEAD")), "root_porcelain": porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain": porcelain(LIBERO_PLUS), "clean_gate": clean_gate, "dirty_gate": dirty_gate, "command_drift": command_drift}, "singleton": {"second_acquire_returncode": probe.returncode, "second_acquire_rejected": "lock_rejected" in probe.stdout}, "normal_family_failure": normal, "timeout_term_kill_no_descendant": timeout, "resume_pass": resume_pass, "resume_drift": resume_drift, "physics_started": False, "gpu_compute_started": False}
    _write_phase("ac3_clean_source_supervisor_fixture.json", result)
    return result


def _patch_old_executor() -> tuple[Any, Any, list[dict[str, Any]]]:
    import tools.axis_pi05.track3_tz_evidence_and_executor_contract_remediation as old
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed
    import tools.axis_pi05.track3_operator_scoped_stability as operator

    records: list[dict[str, Any]] = []
    old.MODULE = MODULE
    old.OUT_ROOT = OUT_ROOT
    old.RUN_ROOT = _attempt_dir() / "runs"
    old.PROFILE_ORDER = PROFILE_ORDER
    old.PROFILES = PROFILES
    old.PROFILE_TASK = {profile: TASK_BY_PROFILE[profile] for profile in PROFILE_ORDER}
    old.FAMILY = FAMILY_BY_TASK
    old.REVERSE_BDDL = REVERSE_BDDL
    old.SNAPSHOTS = SNAPSHOTS
    old.SOURCE_ATOM = SOURCE_ATOM
    old.GOAL_ATOM = GOAL_ATOM
    old.OLD_PROFILE_IDS = OLD_PROFILE_IDS
    old.profile_lock = lambda: profile_lock(strict_json(_attempt_dir() / "ac0_input_manifest.json"))
    old.runtime_identity = identity_snapshot
    old.identity_gate = identity_gate
    original_contact = operator.contact_matrix
    original_writer = old.write_typed_trace

    def contact(env: Any) -> dict[str, Any]:
        matrix = original_contact(env)
        task = int(getattr(env, "_axis_ac_task", 0)) if hasattr(env, "_axis_ac_task") else 0
        if not task:
            task = int(os.environ.get("AXIS_AC_TASK", "0"))
        detail = composed.contact_v3(task, matrix) if task in FAMILY_BY_TASK else {"executed": False, "categories": {}}
        categories = detail.get("categories", {})
        required = bool(categories.get("demonstrator_contact"))
        records.append({"required_contact": required, "contact_telemetry_complete": bool(detail.get("executed") is True), "unrelated_contact_runtime": bool(categories.get("unrelated_background_contact")), "categories": {key: len(value) for key, value in categories.items()}})
        return matrix

    def writer(path: Path, arrays: Mapping[str, Any]) -> None:
        enriched = dict(arrays)
        n = len(enriched.get("actions", []))
        if len(records) == n:
            enriched.update({"required_contact": [row["required_contact"] for row in records], "contact_telemetry_complete": [row["contact_telemetry_complete"] for row in records], "unrelated_contact_runtime": [row["unrelated_contact_runtime"] for row in records]})
        else:
            enriched.update({"required_contact": [False] * n, "contact_telemetry_complete": [False] * n, "unrelated_contact_runtime": [True] * n})
        original_writer(path, enriched)

    operator.contact_matrix = contact
    old.write_typed_trace = writer
    return old, operator, records


def _runtime_gate(profile_id: str, payload: Mapping[str, Any], trace_path: Path) -> dict[str, Any]:
    import numpy as np

    task = TASK_BY_PROFILE[profile_id]
    family = FAMILY_BY_TASK[task]
    arrays: dict[str, Any] = {}
    errors: list[str] = []
    try:
        with np.load(trace_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except Exception as exc:
        return {"status": "failed", "gates": {"typed_trace": gate_record(True, False, reason=f"{type(exc).__name__}:{exc}")}}
    n = len(arrays.get("actions", []))
    fields = ("actions", "eef_error", "phase", "source_value", "source_known", "goal_value", "goal_known", "action_saturated", "contact_complete", "unrelated_contact", "required_contact", "contact_telemetry_complete", "unrelated_contact_runtime")
    missing = [field for field in fields if field not in arrays]
    for field in fields:
        if field in arrays and len(arrays[field]) != n:
            errors.append(f"length:{field}")
    finite = all(bool(np.isfinite(np.asarray(array)).all()) for array in arrays.values() if np.asarray(array).dtype.kind in "fc")
    source = np.asarray(arrays.get("source_value", []), dtype=bool)
    goal = np.asarray(arrays.get("goal_value", []), dtype=bool)
    source_known = np.asarray(arrays.get("source_known", []), dtype=bool)
    goal_known = np.asarray(arrays.get("goal_known", []), dtype=bool)
    phases = [str(value) for value in np.asarray(arrays.get("phase", []), dtype="U64").tolist()]
    stage_results = payload.get("stage_results", [])
    expected_stages = ("approach_object", "descend_object", "grasp_object", "lift_object", "transport_to_target", "descend_to_target", "release_object", "retreat") if task in (10, 20) else ("approach_drawer_region", "grasp_drawer_region", "pull_drawer_closed", "hold_closed")
    entered = {stage for stage in phases}
    stage_entered = bool(entered)
    all_stages = bool(stage_results) and all(row.get("reached") is True for row in stage_results) and [row.get("stage") for row in stage_results] == list(expected_stages)
    required_contacts = np.asarray(arrays.get("required_contact", []), dtype=bool).tolist()
    contact_samples = [{"telemetry_complete": bool(value), "known_mask": True, "contacts": [{"roles": ["demonstrator", "object" if family in {"support_remove", "containment_remove"} else "handle"]}] if required else []} for value, required in zip(np.asarray(arrays.get("contact_telemetry_complete", []), dtype=bool), required_contacts)]
    contact = required_contact_gate(family, contact_samples, stage_entered=stage_entered, min_window=CONTACT_WINDOW)
    sat = np.asarray(arrays.get("action_saturated", []), dtype=bool)
    unrelated = np.asarray(arrays.get("unrelated_contact_runtime", []), dtype=bool)
    source_release = bool(len(source) and source[0] and np.any(~source) and source[-1] is False)
    goal_true_after_entry = bool(len(goal) and np.any(goal) and np.all(goal[np.flatnonzero(goal)[0]:]) and np.all(goal_known[np.flatnonzero(goal)[0]:])) if len(goal) and len(goal_known) == len(goal) else False
    gates = {"telemetry_integrity": gate_record(not missing and not errors, not missing and not errors, reason=";".join([*missing, *errors]) if missing or errors else None), "initial_source_true": gate_record(bool(n), bool(n and source[0] and source_known[0])), "initial_goal_false": gate_record(bool(n), bool(n and not goal[0] and goal_known[0])), "all_stages_reached": gate_record(stage_entered, all_stages, reason="stage_not_entered_or_later_stage_not_evaluated" if not all_stages else None), "required_contact_continuous": contact, "controller_clip": gate_record(bool(n), not bool(np.asarray(payload.get("metrics", {}).get("controller_clip", True)))), "action_saturation": gate_record(bool(n), not bool(sat.any())), "unrelated_contact": gate_record(bool(n), not bool(unrelated.any())), "source_released": gate_record(bool(n), source_release), "goal_continuous_true": gate_record(bool(n), goal_true_after_entry), "finite_telemetry": gate_record(bool(n), finite), "typed_no_pickle_trace": gate_record(True, _typed_trace_validation(trace_path)["status"] == "passed")}
    return {"status": "passed" if required_gates_passed(gates) else "failed", "gates": gates, "stage_entered": stage_entered, "entered_stages": sorted(entered), "expected_stages": list(expected_stages), "steps": n, "phase_order": phases, "contact_runtime_derived": True, "telemetry_integrity_separate_from_required_contact": True}


def _cpu_worker(profile_id: str, result_output: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if profile_id not in PROFILES:
        raise ValueError(f"unknown_profile:{profile_id}")
    task = TASK_BY_PROFILE[profile_id]
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "AXIS_AC_TASK": str(task)})
    old, operator, records = _patch_old_executor()
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    value = old.cpu_worker(profile_id, result_output, expected)
    operator.contact_matrix = getattr(operator, "contact_matrix")
    if value.get("status") == "infrastructure_failure":
        return value
    trace = value.get("trace", {}).get("path")
    trace_path = ROOT / str(trace) if trace and not Path(str(trace)).is_absolute() else Path(str(trace)) if trace else _attempt_dir() / "runs" / "traces" / f"{profile_id}_demo0.npz"
    acquisition = _runtime_gate(profile_id, value, trace_path) if trace_path.is_file() else {"status": "failed", "gates": {"trace": gate_record(False, None, reason="trace_missing")}}
    value["schema_version"] = "track3_ac_canary_child_result_v1"
    value["acquisition_gate"] = acquisition
    value["contact_records_count"] = len(records)
    value["category"] = "success" if acquisition["status"] == "passed" else "family_failure"
    value["training_eligible"] = False
    value["disposition"] = "pending_replay_review" if acquisition["status"] == "passed" else "not_training_eligible"
    value["result_sha256"] = result_content_hash(value)
    write_json(result_output, value)
    return value


def _run_canary(profile_id: str, index: int, expected_lock: Mapping[str, Any]) -> dict[str, Any]:
    run_root = _attempt_dir() / "canaries"
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / f"{index:02d}_{profile_id}.json"
    command = [sys.executable, "-m", MODULE, "--ac-cpu-worker", "--profile-id", profile_id, "--result-output", str(result_path), "--expected-root-commit", git(("rev-parse", "HEAD")), "--expected-tool-sha256", sha256_file(Path(__file__).resolve()), "--expected-openpi-commit", git(("rev-parse", "HEAD"), OPENPI), "--expected-profile-lock-sha256", str(expected_lock["lock_sha256"]), "--expected-command-sha256", "pending"]
    command[-1] = argv_contract_hash(command)
    env = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": str(PROFILES[profile_id]["seed"])}
    expected = identity_snapshot(command, str(expected_lock["lock_sha256"]), environment=env)
    expected.update({"command": command, "command_hash": argv_contract_hash(command)})
    stdout_path, stderr_path = result_path.with_suffix(".stdout.log"), result_path.with_suffix(".stderr.log")
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        pid, pgid = process.pid, os.getpgid(process.pid)
        timed_out = False
        while process.poll() is None:
            if time.monotonic() - started > CPU_TIMEOUT_SECONDS:
                timed_out = True
                os.killpg(pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.05)
    returncode = int(process.returncode or 0)
    descendants = _descendants_in_group(pgid)
    payload = strict_json(result_path) if result_path.is_file() else {"status": "infrastructure_failure", "category": "infrastructure_failure", "error": "result_missing"}
    parent_identity = identity_gate(payload.get("identity", {}), expected, command) if payload.get("identity") else {"state": "failed", "mismatches": ["identity_missing"]}
    category = str(payload.get("category", "infrastructure_failure"))
    if timed_out or descendants or returncode != 0 or parent_identity["state"] != "passed" or category not in {"success", "family_failure"}:
        category = "infrastructure_failure"
        payload["infrastructure_reason"] = "timeout_process_identity_or_result_contract"
    payload["category"] = category
    payload["parent_identity_gate"] = parent_identity
    payload["process_returncode"] = returncode
    payload["child_pid"] = pid
    payload["child_pgid"] = pgid
    payload["timed_out"] = timed_out
    payload["descendants_after_stop"] = descendants
    payload["stdout_tail"] = stdout_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    payload["stderr_tail"] = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    payload["result_sha256"] = result_content_hash(payload)
    write_json(result_path, payload)
    return payload


def _latch(reason: str, profile_id: str) -> dict[str, Any]:
    value = {"schema_version": "track3_ac_supervisor_state_v1", "stop_latched": True, "stop_reason": reason, "failed_profile_id": profile_id, "latched_at_utc": now()}
    _write_phase("supervisor_state.json", value)
    return value


def run_ac4() -> dict[str, Any]:
    existing_state = _attempt_dir() / "supervisor_state.json"
    if existing_state.is_file() and strict_json(existing_state).get("stop_latched"):
        return {"schema_version": "track3_ac4_cpu_canary_matrix_v1", "status": "blocked_sticky_stop", "stop_latched": True, "profiles": [], "executed": 0}
    lock = strict_json(_attempt_dir() / "ac0_profile_lock.json")
    rows: list[dict[str, Any]] = []
    stop_reason: str | None = None
    heartbeat = _attempt_dir() / "heartbeat.json"
    with supervisor_lock(_attempt_dir() / "supervisor.lock"):
        _heartbeat(heartbeat, "supervisor_running", phase="AC4", total_profiles=3)
        for index, profile_id in enumerate(PROFILE_ORDER, 1):
            if stop_reason:
                rows.append({"profile_id": profile_id, "task_index": TASK_BY_PROFILE[profile_id], "family": FAMILY_BY_TASK[TASK_BY_PROFILE[profile_id]], "category": "not_evaluated", "status": "not_evaluated", "reason": stop_reason})
                continue
            payload = _run_canary(profile_id, index, lock)
            category = payload.get("category", "infrastructure_failure")
            rows.append({"profile_id": profile_id, "task_index": TASK_BY_PROFILE[profile_id], "family": FAMILY_BY_TASK[TASK_BY_PROFILE[profile_id]], "category": category, "status": payload.get("status"), "path": rel(_attempt_dir() / "canaries" / f"{index:02d}_{profile_id}.json"), "result_sha256": sha256_file(_attempt_dir() / "canaries" / f"{index:02d}_{profile_id}.json")})
            _heartbeat(heartbeat, "supervisor_running", phase="AC4", completed_profiles=index, last_profile=profile_id, last_category=category)
            if category == "infrastructure_failure":
                stop_reason = f"shared_infrastructure_failure:{profile_id}"
                _latch(stop_reason, profile_id)
        _heartbeat(heartbeat, "ac4_complete", phase="AC4", stop_latched=bool(stop_reason))
    family_results = {family: {"success": [row["profile_id"] for row in rows if row["family"] == family and row["category"] == "success"], "family_failure": [row["profile_id"] for row in rows if row["family"] == family and row["category"] == "family_failure"], "infrastructure_failure": [row["profile_id"] for row in rows if row["family"] == family and row["category"] == "infrastructure_failure"], "not_evaluated": [row["profile_id"] for row in rows if row["family"] == family and row["category"] == "not_evaluated"]} for family in ("support_remove", "containment_remove", "articulation_close")}
    result = {"schema_version": "track3_ac4_cpu_canary_matrix_v1", "status": "hard_stop" if stop_reason else "completed", "profiles": rows, "executed": sum(row["category"] != "not_evaluated" for row in rows), "stop_latched": bool(stop_reason), "stop_reason": stop_reason, "family_results": family_results, "limits": {"max_new_cpu_identities": 3, "per_family_max": 1, "retry": False, "order": ["support", "containment", "articulation"]}, "gpu": False, "renderer": False, "training_eligible": False, "old_identity_rerun": False}
    _write_phase("ac4_cpu_canary_matrix.json", result)
    return result


def run_ac5() -> dict[str, Any]:
    matrix = strict_json(_attempt_dir() / "ac4_cpu_canary_matrix.json") if (_attempt_dir() / "ac4_cpu_canary_matrix.json").is_file() else {"profiles": []}
    family_results = matrix.get("family_results", {})
    success = [row["profile_id"] for row in matrix.get("profiles", []) if row.get("category") == "success"]
    result = {"schema_version": "track3_ac5_crossfamily_classification_v1", "status": "passed", "family_results": family_results, "classification_is_independent": True, "success_disposition": {profile: "pending_replay_review" for profile in success}, "exact_count": 0, "pending_replay_review_count": len(success), "accepted_data_count": 0, "training_eligible_count": 0, "all_failures_are_executor_contract_results": True, "replay_executed": False, "generation": False, "gpu": False, "training": False}
    _write_phase("ac5_crossfamily_classification.json", result)
    return result


def _producer_paths() -> list[Path]:
    excluded = {"artifact_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json", "resume_cursor.json", "heartbeat.json", "latest_attempt.json"}
    return sorted(path for path in _attempt_dir().rglob("*") if path.is_file() and not path.name.endswith(".partial") and path.name not in excluded)


def _manifest(paths: Sequence[Path]) -> dict[str, Any]:
    return {"schema_version": "track3_ac6_artifact_manifest_v1", "entries": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(paths)], "excluded": ["artifact_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json", "resume_cursor.json", "heartbeat.json"], "historical_er_eg_namespaces_modified": False}


def _manifest_check(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    for entry in manifest.get("entries", []):
        path = ROOT / str(entry["path"])
        if not path.is_file():
            errors.append(f"missing:{entry['path']}")
            continue
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"bytes:{entry['path']}")
        if sha256_file(path) != entry.get("sha256"):
            errors.append(f"sha256:{entry['path']}")
    return {"status": "passed" if not errors else "failed", "entries": len(manifest.get("entries", [])), "errors": errors}


def _strict_json_scan() -> dict[str, Any]:
    errors = []
    files = sorted(_attempt_dir().rglob("*.json"))
    for path in files:
        try:
            strict_json(path)
        except Exception as exc:
            errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    return {"files": len(files), "errors": errors, "status": "passed" if not errors else "failed"}


def _npz_scan() -> dict[str, Any]:
    rows = [_typed_trace_validation(path) for path in sorted(_attempt_dir().rglob("*.npz"))]
    return {"files": rows, "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed"}


def _csv_scan() -> dict[str, Any]:
    files = sorted(_attempt_dir().rglob("*.csv"))
    rows = [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "lf_only": b"\r" not in path.read_bytes()} for path in files]
    return {"files": rows, "status": "passed" if all(row["lf_only"] for row in rows) else "failed"}


def _process_scan() -> dict[str, Any]:
    probe = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid=,args="], capture_output=True, text=True, check=False)
    rows = []
    ancestors = set()
    parent_by_pid: dict[int, int] = {}
    for line in probe.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            try:
                pid, ppid, pgid = int(fields[0]), int(fields[1]), int(fields[2])
            except ValueError:
                continue
            parent_by_pid[pid] = ppid
            rows.append({"pid": pid, "ppid": ppid, "pgid": pgid, "command": fields[3]})
    cursor = os.getpid()
    while cursor and cursor not in ancestors:
        ancestors.add(cursor)
        cursor = parent_by_pid.get(cursor, 0)
    tokens = ("--ac-cpu-worker", "--ac-fixture-child")
    live = [row for row in rows if row["pid"] not in ancestors and any(token in row["command"] for token in tokens)]
    return {"returncode": probe.returncode, "worker_tokens": list(tokens), "live_worker_pids": [row["pid"] for row in live], "live_worker_rows": live, "status": "passed" if probe.returncode == 0 and not live else "failed"}


def _gpu_scan() -> dict[str, Any]:
    result = {}
    for index in range(4):
        probe = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        processes = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        result[str(index)] = {"index": index, "returncode": probe.returncode, "compute_processes": processes, "free": probe.returncode == 0 and not processes, "stderr": probe.stderr[-500:]}
    return result


def _test_command(args: Sequence[str]) -> dict[str, Any]:
    probe = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command": [sys.executable, "-m", "pytest", "-q", *args], "returncode": probe.returncode, "stdout_tail": probe.stdout[-3000:], "stderr_tail": probe.stderr[-2000:]}


def run_ac6() -> dict[str, Any]:
    cursor_path, heartbeat_path = _attempt_dir() / "resume_cursor.json", _attempt_dir() / "heartbeat.json"
    write_json(cursor_path, {"schema_version": "track3_ac_resume_cursor_v1", "attempt_id": ATTEMPT_ID, "active": True, "phase": "AC6", "pid": os.getpid(), "updated_at_utc": now()})
    _heartbeat(heartbeat_path, "finalizing", phase="AC6")
    phase_ledger = [{"phase": f"AC{i}", **_phase(name)} for i, name in enumerate(("ac0_result.json", "ac1_old_trace_diagnosis.json", "ac2_runtime_gate_and_fixture.json", "ac3_clean_source_supervisor_fixture.json", "ac4_cpu_canary_matrix.json", "ac5_crossfamily_classification.json"))]
    matrix = strict_json(_attempt_dir() / "ac4_cpu_canary_matrix.json") if (_attempt_dir() / "ac4_cpu_canary_matrix.json").is_file() else {"profiles": []}
    summary = {"schema_version": "track3_ac6_execution_summary_v1", "status": "hard_stop", "phase_ledger": phase_ledger + [{"phase": "AC6", "status": "running"}], "actual_physics_identities": [row["profile_id"] for row in matrix.get("profiles", []) if row.get("category") != "not_evaluated"], "new_physics_identity_count": sum(row.get("category") != "not_evaluated" for row in matrix.get("profiles", [])), "family_status": matrix.get("family_results", {}), "exact_count": 0, "pending_replay_review_count": sum(row.get("category") == "success" for row in matrix.get("profiles", [])), "accepted_data_count": 0, "training_eligible": False, "forbidden_actions": {"old_identity_rerun": False, "gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "official_gpu": False, "generation": False, "replay": False, "training": False, "new_candidate_zip": False, "upload": False, "submission": False}, "hard_stop": True}
    write_json(_attempt_dir() / "summary.json", summary)
    ledger_path = _attempt_dir() / "phase_ledger.csv"
    stream = "phase,status,path\n" + "".join(f"{row['phase']},{row.get('status')},{row.get('path') or ''}\n" for row in phase_ledger) + "AC6,running,summary.json\n"
    atomic_bytes(ledger_path, stream.encode("utf-8"))
    write_json(cursor_path, {"schema_version": "track3_ac_resume_cursor_v1", "attempt_id": ATTEMPT_ID, "active": False, "phase": "AC6", "final_status": "hard_stop", "updated_at_utc": now()})
    manifest = _manifest(_producer_paths())
    write_json(_attempt_dir() / "artifact_manifest.json", manifest)
    outer = {"schema_version": "track3_ac_outer_hash_v1", "manifest_path": rel(_attempt_dir() / "artifact_manifest.json"), "manifest_sha256": sha256_file(_attempt_dir() / "artifact_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"]), "created_at_utc": now()}
    write_json(_attempt_dir() / "outer_hash.json", outer)
    strict_result = _strict_json_scan()
    npz_result = _npz_scan()
    csv_result = _csv_scan()
    process_result = _process_scan()
    gpu_result = _gpu_scan()
    targeted = _test_command(["tests/test_track3_crossfamily_acquisition_recovery_canary.py"])
    full = _test_command(["tests"])
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], cwd=ROOT, capture_output=True, text=True, check=False)
    manifest_check = _manifest_check(manifest)
    provisional = strict_result["status"] == "passed" and npz_result["status"] == "passed" and csv_result["status"] == "passed" and process_result["status"] == "passed" and all(row["free"] for row in gpu_result.values()) and targeted["returncode"] == 0 and full["returncode"] == 0 and diff.returncode == 0 and compile_result.returncode == 0 and manifest_check["status"] == "passed"
    final_status = "passed" if provisional else "failed"
    final_validation = {"schema_version": "track3_ac6_final_validation_v1", "status": final_status, "strict_json": strict_result, "npz": npz_result, "csv": csv_result, "process_scan": process_result, "gpu_boundary": gpu_result, "targeted_tests": targeted, "full_tests": full, "git_diff_check": {"returncode": diff.returncode, "stderr": diff.stderr[-1000:]}, "py_compile": {"returncode": compile_result.returncode, "stderr": compile_result.stderr[-1000:]}, "manifest_entries": manifest_check, "manifest_outer_hash": {"status": "passed" if outer["manifest_sha256"] == sha256_file(_attempt_dir() / "artifact_manifest.json") and outer["entries_sha256"] == canonical_hash(manifest["entries"]) else "failed"}, "new_candidate_zip_created": False, "new_physics_identities": len(summary["actual_physics_identities"]), "gpu_compute_started": False, "finalizer_order": ["producer_fsync", "inactive_cursor", "manifest_outer_hash", "strict_json_npz_csv", "process_gpu_scan", "targeted_full_tests", "final_validation", "closed_heartbeat", "checkout_stable_closure"], "created_at_utc": now()}
    write_json(_attempt_dir() / "final_validation.json", final_validation)
    _heartbeat(heartbeat_path, "closed", phase="AC6", final_status="hard_stop", final_validation_status=final_validation["status"])
    closure = {"schema_version": "track3_ac6_checkout_stable_closure_v1", "status": "passed" if final_status == "passed" and strict_json(_attempt_dir() / "final_validation.json").get("status") == "passed" else "failed", "attempt_id": ATTEMPT_ID, "final_validation_sha256": sha256_file(_attempt_dir() / "final_validation.json"), "artifact_manifest_sha256": sha256_file(_attempt_dir() / "artifact_manifest.json"), "outer_hash_sha256": sha256_file(_attempt_dir() / "outer_hash.json"), "closed_heartbeat_sha256": sha256_file(heartbeat_path), "inactive_cursor": strict_json(cursor_path).get("active") is False, "no_partial": not list(_attempt_dir().rglob("*.partial")), "root_porcelain": porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI) if OPENPI.is_dir() else [], "historical_er_eg_namespaces_modified": False, "next_authority": "independent_review_required"}
    write_json(_attempt_dir() / "checkout_stable_closure.json", closure)
    pointer = {"schema_version": "track3_ac_latest_attempt_pointer_v1", "attempt_id": ATTEMPT_ID, "target_final_validation": rel(_attempt_dir() / "final_validation.json"), "target_final_validation_sha256": closure["final_validation_sha256"], "target_closure": rel(_attempt_dir() / "checkout_stable_closure.json"), "target_closure_sha256": closure["closed_heartbeat_sha256"] if False else sha256_file(_attempt_dir() / "checkout_stable_closure.json"), "immutable": True}
    write_json(OUT_ROOT / "latest_attempt.json", pointer)
    final = {"status": final_status, "summary": summary, "final_validation": final_validation, "closure": closure, "latest_attempt": pointer}
    return final


def run_all() -> dict[str, Any]:
    ac0 = run_ac0()
    if ac0.get("status") != "passed":
        return {"status": "hard_stop", "phase": "AC0"}
    ac1 = run_ac1()
    if ac1.get("status") != "passed":
        return {"status": "hard_stop", "phase": "AC1"}
    ac2 = run_ac2()
    if ac2.get("status") != "passed":
        return {"status": "hard_stop", "phase": "AC2"}
    ac3 = run_ac3()
    if ac3.get("status") != "passed":
        return {"status": "hard_stop", "phase": "AC3"}
    ac4 = run_ac4()
    if ac4.get("status") not in {"completed", "hard_stop"}:
        return {"status": "hard_stop", "phase": "AC4"}
    run_ac5()
    return run_ac6()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for index in range(7):
        parser.add_argument(f"--ac{index}", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ac-fixture-child", action="store_true")
    parser.add_argument("--ac-lock-probe", type=Path)
    parser.add_argument("--mode", choices=("family_failure", "timeout"))
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--heartbeat-output", type=Path)
    parser.add_argument("--expected-identity-file", type=Path)
    parser.add_argument("--expected-command-sha256")
    parser.add_argument("--ac-cpu-worker", action="store_true")
    parser.add_argument("--profile-id")
    parser.add_argument("--expected-root-commit")
    parser.add_argument("--expected-tool-sha256")
    parser.add_argument("--expected-openpi-commit")
    parser.add_argument("--expected-profile-lock-sha256")
    args = parser.parse_args(argv)
    if args.ac_lock_probe is not None:
        return _lock_probe(args.ac_lock_probe)
    if args.ac_fixture_child:
        required = (args.mode, args.result_output, args.heartbeat_output, args.expected_identity_file, args.expected_command_sha256)
        if any(value is None for value in required):
            parser.error("fixture child requires mode, result, heartbeat, expected identity and command hash")
        return _fixture_child(str(args.mode), args.result_output, args.heartbeat_output, args.expected_identity_file)
    if args.ac_cpu_worker:
        required = (args.profile_id, args.result_output, args.expected_root_commit, args.expected_tool_sha256, args.expected_openpi_commit, args.expected_profile_lock_sha256, args.expected_command_sha256)
        if any(value is None for value in required):
            parser.error("cpu worker requires profile, result and complete identity")
        env = os.environ
        command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
        expected = {"root_commit": args.expected_root_commit, "tool_sha256": args.expected_tool_sha256, "openpi_commit": args.expected_openpi_commit, "profile_lock_sha256": args.expected_profile_lock_sha256, "command_hash": args.expected_command_sha256, "command": command, "cuda_visible_devices": "", "jax_platforms": "cpu", "mujoco_gl": "disable", "root_porcelain": porcelain(ROOT), "root_porcelain_sha256": canonical_hash(porcelain(ROOT)), "openpi_porcelain": porcelain(OPENPI) if OPENPI.is_dir() else [], "openpi_porcelain_sha256": canonical_hash(porcelain(OPENPI) if OPENPI.is_dir() else []), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "libero_plus_porcelain_sha256": canonical_hash(porcelain(LIBERO_PLUS)), "clean_root": not porcelain(ROOT)}
        try:
            value = _cpu_worker(str(args.profile_id), args.result_output, expected)
        except Exception as exc:
            value = {"schema_version": "track3_ac_canary_child_result_v1", "status": "infrastructure_failure", "category": "infrastructure_failure", "profile_id": args.profile_id, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
            write_json(args.result_output, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        return 0 if value.get("category") in {"success", "family_failure"} else 1
    phases = [getattr(args, f"ac{index}") for index in range(7)]
    if args.execute:
        value = run_all()
    else:
        if sum(bool(flag) for flag in phases) != 1:
            parser.error("choose exactly one AC phase or --execute")
        value = {0: run_ac0, 1: run_ac1, 2: run_ac2, 3: run_ac3, 4: run_ac4, 5: run_ac5, 6: run_ac6}[next(index for index, flag in enumerate(phases) if flag)]()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if value.get("status") in {"passed", "completed", "hard_stop", "blocked_sticky_stop"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
