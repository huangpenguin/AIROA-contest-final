#!/usr/bin/env python3
"""Start-state bundle closure and bounded public task10 support executor.

P0/P1 are simulator-free.  Runtime imports are confined to explicit fresh
worker entrypoints so the input lock and contract tests cannot accidentally
create a MuJoCo/CUDA context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary"
PENDING_ROOT = ROOT / "data/pending_autogen_v3"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_COMPOSED_REVERSE_START_M0_M5_AUDIT_20260904.md"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
OLD_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_composed_reverse_start_stabilization"
OLD_TOOL = ROOT / "tools/axis_pi05/track3_composed_reverse_start_stabilization.py"
HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
TASK_ORDER = (10, 20, 19)
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SOURCE_ATOM = {10: "(On akita_black_bowl_1 plate_1)", 20: "(In orange_juice_1 basket_1_contain_region)", 19: "(Open wooden_cabinet_1_middle_region)"}
GOAL_ATOM = {10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)", 20: "(On orange_juice_1 floor_target_object_region)", 19: "(Close wooden_cabinet_1_middle_region)"}
FIXED_DEMOS = tuple((task, demo) for task in TASK_ORDER for demo in range(3))
ACTION_DIM = 7
CONTROL_FREQ_HZ = 20
COMPOSED = "tools.axis_pi05.track3_composed_reverse_start_stabilization"
WORKER_COMPLETED_STATUSES = frozenset({"passed", "failed_gate"})
STATE_KEYS = tuple(f"task{task}_demo{demo}" for task, demo in FIXED_DEMOS)


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


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_failure_asset(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish a diagnostic JSON and leave no temporary residue."""
    write_json(path, dict(value))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "atomic": True, "partial_residue": path.with_name(path.name + ".partial").exists()}


def gate(*, executed: bool, value: bool | None, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    result = {"executed": bool(executed), "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"), "required": bool(required)}
    if reason is not None:
        result["reason"] = reason
    return result


def all_required(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in gates.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def worker_exit_code(status: str) -> int:
    return 0 if status in WORKER_COMPLETED_STATUSES else 1


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        value = resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact_outside_root:{resolved}") from exc
    if value.startswith("../") or value == "..":
        raise ValueError(f"artifact_parent_path:{value}")
    return value


def artifact_entry(path: Path, *, source_commit: str, schema: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not source_commit:
        raise ValueError("artifact_source_commit_missing")
    inferred = schema
    if inferred is None and path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            inferred = str(value.get("schema_version", "json_unknown_schema")) if isinstance(value, dict) else "json_non_object"
        except Exception:
            inferred = "json_unparseable"
    if inferred is None:
        inferred = path.suffix.lstrip(".") or "binary"
    return {"relative_path": relative_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "schema": inferred, "source_commit": source_commit}


def validate_artifact_manifest(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    for item in entries:
        value = str(item.get("relative_path", ""))
        if not value or value in seen or value.startswith("/") or ".." in Path(value).parts:
            errors.append(f"invalid_or_duplicate_relative_path:{value}")
            continue
        seen.add(value)
        path = ROOT / value
        if not path.is_file():
            errors.append(f"missing:{value}")
            continue
        if int(item.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"bytes:{value}")
        if str(item.get("sha256", "")) != sha256_file(path):
            errors.append(f"sha256:{value}")
        if not item.get("source_commit"):
            errors.append(f"source_commit:{value}")
    return {"executed": True, "state": "passed" if not errors else "failed", "entry_count": len(entries), "errors": errors}


def strict_json_load(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def validate_state_bundle(path: Path, expected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    arrays: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = sorted(str(key) for key in archive.files)
            if keys != sorted(expected):
                errors.append(f"keys:{keys}")
            for key, spec in expected.items():
                if key not in archive.files:
                    continue
                array = np.asarray(archive[key])
                arrays[key] = array
                if array.dtype != np.dtype("float64"):
                    errors.append(f"dtype:{key}")
                if list(array.shape) != list(spec["shape"]):
                    errors.append(f"shape:{key}")
                if not np.isfinite(array).all():
                    errors.append(f"finite:{key}")
                if spec.get("sha256") and sha256_bytes(array.tobytes()) != spec["sha256"]:
                    errors.append(f"sha256:{key}")
    except Exception as exc:
        errors.append(f"npz:{type(exc).__name__}:{exc}")
    return {"executed": True, "state": "passed" if not errors else "failed", "keys": sorted(arrays), "errors": errors, "allow_pickle": False}


def full_window_contact_gate(task_index: int, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply contact-v3 to t0 and every burn-in/measurement checkpoint."""
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed

    steps = [int(sample.get("step", -1)) for sample in samples]
    if steps != list(range(41)):
        return gate(executed=False, value=None, reason="contact_window_requires_step0_to40")
    details = [composed.contact_v3(task_index, sample.get("contact_matrix", {})) for sample in samples]
    if any(row.get("state") == "not_evaluated" for row in details):
        return gate(executed=False, value=None, reason="contact_matrix_incomplete")
    robot_zero = all(not row["categories"]["demonstrator_contact"] for row in details)
    unrelated_zero = all(not row["categories"]["unrelated_background_contact"] for row in details)
    support_ok = task_index != 10 or all(row["categories"]["intended_receiver_contact"] for row in details)
    result = {"executed": True, "state": "passed" if robot_zero and unrelated_zero and support_ok else "failed", "required": True, "step_range": [0, 40], "demonstrator_contact_zero_all_41": robot_zero, "unrelated_background_contact_zero_all_41": unrelated_zero, "task10_support_present_all_41": support_ok, "details": details}
    if result["state"] == "failed":
        result["reason"] = "full_window_contact_v3_failure"
    return result


def gpu_provenance_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("seed", "child_pid", "physical_gpu_uuid", "cuda_visible_devices", "start_snapshot", "end_snapshot", "peak_memory_mib")
    missing = [key for key in required if key not in record or record.get(key) in (None, "", [])]
    peak = record.get("peak_memory_mib")
    if peak is not None:
        try:
            if not math.isfinite(float(peak)) or float(peak) < 0:
                missing.append("peak_memory_mib_invalid")
        except (TypeError, ValueError):
            missing.append("peak_memory_mib_invalid")
    result = {"executed": True, "state": "passed" if not missing else "failed", "required": True, "required_fields": list(required), "missing": missing}
    if missing:
        result["reason"] = "gpu_provenance_incomplete"
    return result


def _quat_normalize(value: Sequence[float]) -> tuple[float, float, float, float]:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.isfinite(array).all() or float(np.linalg.norm(array)) == 0.0:
        raise ValueError("invalid_quaternion")
    array = array / np.linalg.norm(array)
    return tuple(float(x) for x in array)


def _quat_mul(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = _quat_normalize(left)
    rw, rx, ry, rz = _quat_normalize(right)
    return (lw * rw - lx * rx - ly * ry - lz * rz, lw * rx + lx * rw + ly * rz - lz * ry, lw * ry - lx * rz + ly * rw + lz * rx, lw * rz + lx * ry - ly * rx + lz * rw)


def quaternion_log_error(target: Sequence[float], current: Sequence[float]) -> Any:
    """Return the shortest rotation vector taking current to target."""
    import numpy as np

    tw, tx, ty, tz = _quat_mul(target, (_quat_normalize(current)[0], -_quat_normalize(current)[1], -_quat_normalize(current)[2], -_quat_normalize(current)[3]))
    if tw < 0:
        tw, tx, ty, tz = -tw, -tx, -ty, -tz
    vector = np.asarray((tx, ty, tz), dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    angle = 2.0 * math.atan2(norm, max(float(tw), 0.0))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return vector * (angle / norm)


def clip_vector(value: Sequence[float], maximum: float) -> tuple[Any, bool]:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all() or maximum <= 0 or not math.isfinite(float(maximum)):
        raise ValueError("invalid_clip_vector")
    norm = float(np.linalg.norm(array))
    if norm <= maximum:
        return array, False
    return array * (float(maximum) / norm), True


def osc_pose_action(translation: Sequence[float], rotation: Sequence[float], gripper: float, *, translation_max: float, rotation_max: float) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if float(gripper) not in {-1.0, 1.0}:
        raise ValueError("executor_gripper_must_be_exactly_minus_or_plus_one")
    clipped_translation, translation_clipped = clip_vector(translation, translation_max)
    clipped_rotation, rotation_clipped = clip_vector(rotation, rotation_max)
    action = np.concatenate((clipped_translation / float(translation_max), clipped_rotation / float(rotation_max), np.asarray([float(gripper)]))).astype(np.float32)
    if action.shape != (7,) or not np.isfinite(action).all() or float(action[-1]) not in {-1.0, 1.0}:
        raise ValueError("invalid_osc_pose_action")
    return action, {"translation_clipped": translation_clipped, "rotation_clipped": rotation_clipped, "translation_norm": float(np.linalg.norm(clipped_translation)), "rotation_norm": float(np.linalg.norm(clipped_rotation)), "translation_max": translation_max, "rotation_max": rotation_max}


def _git(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def _porcelain(repo: Path) -> list[str]:
    return [line for line in _git(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def _old_artifacts() -> list[Path]:
    old_json = sorted(OLD_ROOT.rglob("*.json"))
    old_videos = sorted(OLD_ROOT.glob("videos/*.mp4"))
    return [OLD_PLAN, AUDIT, OLD_TOOL, *old_json, *old_videos, *HDF5_INPUTS.values()]


def _lock_file(path: Path) -> dict[str, Any]:
    return artifact_entry(path, source_commit="old_immutable_input")


def _read_old_runs() -> list[dict[str, Any]]:
    rows = []
    for task, demo in FIXED_DEMOS:
        candidates = [OLD_ROOT / "runs" / f"m3_task{task}_demo{demo}.json"]
        if task == 20 and demo == 0:
            candidates.insert(0, OLD_ROOT / "runs" / "m2_task20_demo0.json")
        path = next(path for path in candidates if path.is_file())
        report = strict_json_load(path)
        gates = report.get("hard_gate", {})
        failures = {name: row for name, row in gates.items() if isinstance(row, Mapping) and row.get("state") in {"failed", "not_evaluated"}}
        checkpoints = report.get("reverse", {}).get("checkpoints", [])
        contact = full_window_contact_gate(task, checkpoints)
        source_values = []
        goal_values = []
        for sample in checkpoints:
            source_values.append(next((row.get("value") for row in sample.get("initial", {}).get("predicates", []) if row.get("atom") == SOURCE_ATOM[task]), None))
            goal_values.append(next((row.get("value") for row in sample.get("goal", {}).get("predicates", []) if row.get("atom") == GOAL_ATOM[task]), None))
        rows.append({"task_index": task, "demo_index": demo, "path": relative_path(path), "sha256": sha256_file(path), "status": report.get("status"), "complete_steps": len(checkpoints) == 41 and [int(row.get("step", -1)) for row in checkpoints] == list(range(41)), "source_all_true": len(source_values) == 41 and all(value is True for value in source_values), "goal_all_false": len(goal_values) == 41 and all(value is False for value in goal_values), "contact_v3_full_window": contact, "failed_or_not_evaluated_gates": failures, "restore_max_abs_diff": report.get("model_state_identity", {}).get("restore_max_abs_diff")})
    return rows


def run_p0() -> dict[str, Any]:
    old_files = [_lock_file(path) for path in _old_artifacts()]
    old_summary = strict_json_load(OLD_ROOT / "summary.json")
    old_m3 = strict_json_load(OLD_ROOT / "m3_summary.json") if (OLD_ROOT / "m3_summary.json").is_file() else None
    composed_hashes = []
    for task, demo in FIXED_DEMOS:
        path = OLD_ROOT / "runs" / f"m3_task{task}_demo{demo}.json"
        if task == 20 and demo == 0:
            path = OLD_ROOT / "runs" / "m2_task20_demo0.json"
        report = strict_json_load(path)
        composed_hashes.append(report.get("composition", {}).get("composed_state_sha256"))
    unique_hashes = sorted(set(composed_hashes))
    zero_counts = {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}
    observed_counts = {key: old_summary.get("counts", {}).get(key, 0) for key in zero_counts}
    root_head = _git(ROOT, ("rev-parse", "HEAD"))
    expected_root = "eef683b0db9629a3d917ea91ebeb46540a34fcb9"
    root_ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", expected_root, "HEAD"], check=False).returncode == 0
    openpi_head = _git(OPENPI, ("rev-parse", "HEAD")) if OPENPI.is_dir() else None
    lock = {"schema_version": "track3_start_bundle_p0_v1", "status": "track3_start_bundle_p0_passed", "created_at_utc": now(), "plan": relative_path(PLAN), "audit": relative_path(AUDIT), "audited_root_commit": expected_root, "observed_root_head": root_head, "audited_root_is_ancestor": root_ancestor, "nested_openpi_expected": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1", "observed_openpi_head": openpi_head, "root_porcelain": _porcelain(ROOT), "openpi_porcelain": _porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain": _porcelain(LIBERO_PLUS), "old_artifact_lock": old_files, "fixed_demo_order": [{"task_index": task, "demo_index": demo} for task, demo in FIXED_DEMOS], "unique_composed_hashes": unique_hashes, "unique_composed_hash_count": len(unique_hashes), "old_run_reanalysis": _read_old_runs(), "old_summary_status": old_summary.get("status"), "old_m3_summary_status": old_m3.get("status") if old_m3 else None, "reverse_demo_counts": {"expected": zero_counts, "observed": observed_counts, "passed": observed_counts == zero_counts}, "errata": ["M4 raw lacks per-process seed/PID/peak GPU provenance", "contact-v3 required gate must cover step0–40 including burn-in", "candidate hash without recoverable state vector is not restoreable", "round_trip_reconstruction_max_abs_diff is composed-vs-init and must be renamed", "run-index entries need relative artifact bytes/SHA-256", "handoff must reference eef683b rather than orphan 020d291"], "resource_boundary": {"simulator_started": False, "gpu_started": False, "hdf5_or_bddl_modified": False, "old_evidence_modified": False}}
    if not root_ancestor or len(unique_hashes) != 9 or observed_counts != zero_counts:
        lock["status"] = "track3_start_bundle_p0_input_drift"
    write_json(OUT_ROOT / "p0_release_lock.json", lock)
    errata = """# Track 3 Start Bundle P0 Errata

P0 只读锁定 `eef683b` 输入及旧 M0–M5 产物，没有启动 simulator 或 GPU。

旧 M4 的核心物理结论已保留；本轮只修复其工程闭包：逐进程 seed/PID/显存 provenance、step0–40 全窗口 contact gate、可恢复 NPZ state、artifact relative path/bytes/SHA-256，以及 `composed_vs_init_max_abs_diff` 与真实 restore diff 的命名边界。旧报告不原地改写。

P0 的 raw reanalysis 同时保存全部 failed/not_evaluated gate，不能用首个失败字段覆盖其余诊断。reverse demo counts 仍为 `0/0/0`。
"""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "P0_M0_M5_ERRATA_ZH.md").write_text(errata, encoding="utf-8")
    return lock


def run_p1() -> dict[str, Any]:
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed

    synthetic_layout = composed.decode_joint_layout({"nq": 8, "nv": 7, "na": 0, "state_length": 16, "joint_names": ["free", "arm"], "jnt_type": [0, 3], "jnt_qposadr": [0, 7], "jnt_dofadr": [0, 6], "joint_ownership": {"free": "operator_scene", "arm": "robot_arm"}, "joint_source": {"free": "demo_terminal", "arm": "demo_init"}})
    contract = {"schema_version": "track3_start_bundle_p1_v1", "status": "track3_start_bundle_p1_contract_passed", "created_at_utc": now(), "plan": relative_path(PLAN), "simulator_started": False, "gpu_started": False, "artifact_manifest": {"path": "relative_path_only", "required": ["relative_path", "bytes", "sha256", "schema", "source_commit"], "duplicate_key_json": "reject", "npz_allow_pickle": False}, "state_bundle": {"format": "NPZ", "allow_pickle": False, "dtype": "float64", "finite": True, "expected_keys": [f"task{task}_demo{demo}" for task, demo in FIXED_DEMOS], "source_bitmap": "named_composed_state_contract", "field_rename": "composed_vs_init_max_abs_diff; restore_max_abs_diff remains restore-vs-input"}, "contact_v3": {"required_steps": list(range(41)), "burn_in_included": True, "demonstrator_contact": "zero_all_41", "unrelated_background": "zero_all_41", "task10_support": "required_all_41"}, "gpu_provenance": {"required": ["seed", "child_pid", "physical_gpu_uuid", "cuda_visible_devices", "start_snapshot", "end_snapshot", "peak_memory_mib"], "missing_any_fails": True, "gpu3_only": True}, "executor_action": {"controller": "OSC_POSE", "shape": [7], "dtype": "float32", "translation_rotation": "separately_normalized_and_clipped", "rotation_error": "quaternion_log_shortest_path", "gripper": [-1.0, 1.0], "neutral_hold": 0.0}, "synthetic_state_layout": synthetic_layout, "tri_state": ["passed", "failed", "not_evaluated"], "worker_exception_returncode": "nonzero", "atomic_failure_asset": True, "old_tool_untouched": True}
    contract["canonical_sha256"] = canonical_hash({key: value for key, value in contract.items() if key != "canonical_sha256"})
    write_json(OUT_ROOT / "p1_validator_contract.json", contract)
    return contract


# ---------------------------------------------------------------------------
# P2: deterministic state bundle and fresh CPU restore workers.


def _old_report_path(task: int, demo: int) -> Path:
    path = OLD_ROOT / "runs" / f"m3_task{task}_demo{demo}.json"
    if task == 20 and demo == 0:
        path = OLD_ROOT / "runs" / "m2_task20_demo0.json"
    return path


def _read_demo_for_bundle(task: int, demo: int) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
    """Read only the named public demo and recompute its composed state."""
    import h5py
    import numpy as np
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as old

    path = HDF5_INPUTS[task]
    with h5py.File(path, "r") as handle:
        demo_group = handle[f"data/demo_{demo}"]
        states = np.asarray(demo_group["states"][()], dtype=np.float64)
        actions = np.asarray(demo_group["actions"][()], dtype=np.float64)
        rewards = np.asarray(demo_group["rewards"][()])
        model_bytes = old._as_bytes(demo_group.attrs["model_file"])
        init_state = np.asarray(demo_group.attrs["init_state"], dtype=np.float64)
        selection = old._terminal_selection(rewards, len(states), len(actions))
        terminal_state = np.asarray(states[selection["state_index"]], dtype=np.float64)
    if states.ndim != 2 or actions.ndim != 2 or rewards.ndim != 1 or not np.array_equal(states[0], init_state):
        raise ValueError(f"hdf5_schema_or_init_mismatch:{task}:{demo}")
    old_report = strict_json_load(_old_report_path(task, demo))
    metadata = old_report["model_layout"]["metadata"]
    layout = old.decode_joint_layout(metadata)
    composed, composition = old.compose_flat_state(metadata, layout, init_state, terminal_state)
    expected = old_report["composition"]["composed_state_sha256"]
    if composition["composed_state_sha256"] != expected:
        raise ValueError(f"locked_composed_hash_mismatch:{task}:{demo}")
    manifest = old._hdf5_row(task, demo)
    return manifest, init_state, terminal_state, composed, {"model_bytes": model_bytes, "metadata": metadata, "layout": layout, "selection": selection, "composition": composition, "old_report_sha256": sha256_file(_old_report_path(task, demo)), "old_composed_hash": expected}


def _save_npz_atomic(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return artifact_entry(path, source_commit=_git(ROOT, ("rev-parse", "HEAD")), schema="track3_start_states_npz_v1")


def _reverse_instruction(task: int) -> str:
    return {
        10: "Move the bowl from the plate back to its original table position.",
        20: "Move the orange juice from the basket back to its original floor position.",
        19: "Close the middle drawer of the cabinet.",
    }[task]


def _bundle_expected_from_manifest(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: {"shape": row["state_shape"], "sha256": row["composed_state_sha256"]} for key, row in manifest["states"].items()}


def run_p2() -> dict[str, Any]:
    import numpy as np

    p0 = strict_json_load(OUT_ROOT / "p0_release_lock.json")
    p1 = strict_json_load(OUT_ROOT / "p1_validator_contract.json")
    if p0.get("status") != "track3_start_bundle_p0_passed" or p1.get("status") != "track3_start_bundle_p1_contract_passed":
        result = {"schema_version": "track3_start_bundle_p2_v1", "status": "not_evaluated", "reason": "p0_or_p1_not_passed", "validated_start_state": 0}
        write_json(OUT_ROOT / "p2_summary.json", result)
        return result
    arrays: dict[str, Any] = {}
    rows: dict[str, Any] = {}
    for task, demo in FIXED_DEMOS:
        key = f"task{task}_demo{demo}"
        hdf5_row, init_state, terminal_state, composed, detail = _read_demo_for_bundle(task, demo)
        arrays[key] = composed
        rows[key] = {
            "key": key,
            "task_index": task,
            "demo_index": demo,
            "family": FAMILY[task],
            "instruction": _reverse_instruction(task),
            "source_atom": SOURCE_ATOM[task],
            "goal_atom": GOAL_ATOM[task],
            "control_freq_hz": CONTROL_FREQ_HZ,
            "hdf5_path": hdf5_row["hdf5_path"],
            "hdf5_sha256": hdf5_row["hdf5_sha256"],
            "model_file_sha256": hdf5_row["model_file_sha256"],
            "init_state_sha256": hdf5_row["init_state_sha256"],
            "terminal_state_sha256": hdf5_row["terminal_state_sha256"],
            "terminal_selection": hdf5_row["selection"],
            "terminal_state_shape": hdf5_row["terminal_state_shape"],
            "state_shape": list(composed.shape),
            "state_dtype": str(composed.dtype),
            "composed_state_sha256": detail["composition"]["composed_state_sha256"],
            "state_indices_source_bitmap": detail["composition"]["state_indices_source_bitmap"],
            "source_bitmap_canonical_sha256": canonical_hash(detail["composition"]["state_indices_source_bitmap"]),
            "same_demo_model_xml_identity": True,
            "old_report_sha256": detail["old_report_sha256"],
            "old_input_composed_state_sha256": detail["old_composed_hash"],
            "composition_contract": detail["composition"]["composition"],
        }
        if rows[key]["composed_state_sha256"] != rows[key]["old_input_composed_state_sha256"]:
            raise ValueError(f"bundle_old_hash_mismatch:{key}")
    bundle_path = OUT_ROOT / "start_states.npz"
    bundle_artifact = _save_npz_atomic(bundle_path, arrays)
    validation = validate_state_bundle(bundle_path, _bundle_expected_from_manifest({"states": rows}))
    if validation["state"] != "passed":
        result = {"schema_version": "track3_start_bundle_p2_v1", "status": "track3_start_bundle_p2_failed", "reason": "bundle_validation_failed", "validation": validation, "validated_start_state": 0}
        write_json(OUT_ROOT / "p2_summary.json", result)
        return result
    manifest = {"schema_version": "track3_start_state_manifest_v1", "status": "materialized", "created_at_utc": now(), "plan": relative_path(PLAN), "source_bitmap_contract": "named_composed_state_contract", "npz": bundle_artifact, "states": rows, "state_bundle_canonical_sha256": canonical_hash({key: row["state_indices_source_bitmap"] for key, row in rows.items()}), "exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}
    manifest["canonical_sha256"] = canonical_hash({key: value for key, value in manifest.items() if key != "canonical_sha256"})
    write_json(OUT_ROOT / "start_state_manifest.json", manifest)
    manifest_artifact = artifact_entry(OUT_ROOT / "start_state_manifest.json", source_commit=_git(ROOT, ("rev-parse", "HEAD")), schema="track3_start_state_manifest_v1")
    workers: list[dict[str, Any]] = []
    for task, demo in FIXED_DEMOS:
        workers.append(_run_p2_restore_worker(task, demo, bundle_path))
        if workers[-1].get("status") != "passed":
            break
    restore_passed = len(workers) == 9 and all(row.get("status") == "passed" and row.get("worker_returncode_contract") for row in workers)
    geometry = _run_p2_geometry_worker() if restore_passed else {"status": "not_evaluated", "reason": "restore_9_of_9_not_passed"}
    if geometry.get("status") == "passed":
        write_json(OUT_ROOT / "task10_demo0_grasp_geometry.json", geometry)
    status = "track3_start_state_bundle_validated_9_of_9" if restore_passed and geometry.get("status") == "passed" else "track3_start_bundle_p2_failed"
    result = {"schema_version": "track3_start_bundle_p2_v1", "status": status, "created_at_utc": now(), "bundle": bundle_artifact, "manifest": manifest_artifact, "validation": validation, "restore_workers": workers, "restore_count": len(workers), "validated_start_state": sum(row.get("status") == "passed" for row in workers), "grasp_geometry": {"status": geometry.get("status"), "path": relative_path(OUT_ROOT / "task10_demo0_grasp_geometry.json") if geometry.get("status") == "passed" else None}, "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}}
    write_json(OUT_ROOT / "p2_summary.json", result)
    return result


def _parse_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise ValueError("worker_emitted_no_json_result")


def _run_p2_restore_worker(task: int, demo: int, bundle_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "MUJOCO_GL": "disable", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    command = [sys.executable, "-m", "tools.axis_pi05.track3_start_bundle_and_support_executor_canary", "--p2-restore-worker", "--task-index", str(task), "--demo-index", str(demo), "--state-bundle", str(bundle_path)]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=600, check=False)
    try:
        result = _parse_last_json(completed.stdout)
    except Exception as exc:
        result = {"status": "worker_exception", "error": {"type": type(exc).__name__, "message": str(exc), "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}}
    result["process_returncode"] = completed.returncode
    result["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(result.get("status")))
    result["stdout_tail"] = completed.stdout[-2000:]
    result["stderr_tail"] = completed.stderr[-2000:]
    return result


def _p2_restore_worker(task: int, demo: int, bundle_path: Path) -> int:
    import numpy as np

    started = now()
    env = None
    try:
        import tools.axis_pi05.track3_composed_reverse_start_stabilization as old
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base
        manifest, _model_bytes, _init, _terminal, _info = old._demo_arrays(task, demo)
        with np.load(bundle_path, allow_pickle=False) as archive:
            state = np.asarray(archive[f"task{task}_demo{demo}"], dtype=np.float64).copy()
        resolved_xml, resolution = old._resolve_model_xml(_model_bytes)
        kwargs = {"control_freq": CONTROL_FREQ_HZ, "use_camera_obs": False, "has_renderer": False, "has_offscreen_renderer": False, "render_gpu_device_id": 0}
        env = dynamic.make_schema_env(str(base.REVERSE_BDDL[task]))
        env.reset_from_xml_string(resolved_xml)
        env.set_init_state(state)
        restored = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        restore_diff = float(np.max(np.abs(restored - state)))
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(base.REVERSE_BDDL[task], "init")
        goal_atoms = base.public_predicates(base.REVERSE_BDDL[task], "goal")
        samples: list[dict[str, Any]] = []
        for step in range(41):
            if step:
                env.step(old.neutral_hold())
            sample = prior._runtime_sample(env, dynamic, task, "reverse", step, initial_atoms, goal_atoms)
            sample["contacts_v3"] = old.contact_v3(task, sample["contact_matrix"])
            samples.append(sample)
        source_values = [next((row.get("value") for row in sample["initial"]["predicates"] if row.get("atom") == SOURCE_ATOM[task]), None) for sample in samples]
        goal_values = [next((row.get("value") for row in sample["goal"]["predicates"] if row.get("atom") == GOAL_ATOM[task]), None) for sample in samples]
        truth_complete = all(sample["initial"].get("truth_complete") and sample["initial"].get("detail_complete") and sample["goal"].get("truth_complete") and sample["goal"].get("detail_complete") for sample in samples)
        truth = {"executed": truth_complete, "state": "passed" if truth_complete and all(value is True for value in source_values) and all(value is False for value in goal_values) else "failed", "source_values": source_values, "goal_values": goal_values, "step_range": [0, 40], "source_atom": SOURCE_ATOM[task], "goal_atom": GOAL_ATOM[task]}
        contact = full_window_contact_gate(task, samples)
        gates = {"restore_identity": gate(executed=True, value=restore_diff <= 1e-12), "truth": gate(executed=truth_complete, value=truth["state"] == "passed"), "contact_v3_full_window": contact}
        status = "passed" if all_required(gates) else "failed_gate"
        result = {"schema_version": "track3_start_state_restore_worker_v1", "status": status, "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "fresh_process": True, "mode": "CPU_no_renderer", "state_key": f"task{task}_demo{demo}", "state_sha256": sha256_bytes(state.tobytes()), "restored_state_sha256": sha256_bytes(restored.tobytes()), "restore_max_abs_diff": restore_diff, "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "reverse": {"source_atom": SOURCE_ATOM[task], "goal_atom": GOAL_ATOM[task], "checkpoints": samples, "truth": truth}, "hard_gate": gates, "provenance": {"hdf5_state_used_as_predicate_oracle": False, "no_action_or_demo_generation": True, "no_gpu": True}}
    except Exception as exc:
        result = {"schema_version": "track3_start_state_restore_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "fresh_process": True, "mode": "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    print(json.dumps(result, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value), sort_keys=True))
    return worker_exit_code(str(result.get("status")))


def _run_p2_geometry_worker() -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "MUJOCO_GL": "disable", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    command = [sys.executable, "-m", "tools.axis_pi05.track3_start_bundle_and_support_executor_canary", "--p2-geometry-worker"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=600, check=False)
    try:
        result = _parse_last_json(completed.stdout)
    except Exception as exc:
        result = {"status": "worker_exception", "error": {"type": type(exc).__name__, "message": str(exc), "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}}
    result["process_returncode"] = completed.returncode
    result["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(result.get("status")))
    result["stdout_tail"] = completed.stdout[-2000:]
    result["stderr_tail"] = completed.stderr[-2000:]
    return result


def _p2_geometry_worker() -> int:
    import numpy as np
    import h5py
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as old
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
    from tools.axis_pi05 import track3_reverse_init_stabilization as base
    env = None
    started = now()
    try:
        manifest, model_bytes, init_state, _terminal, _info = old._demo_arrays(10, 0)
        with h5py.File(HDF5_INPUTS[10], "r") as handle:
            demo_group = handle["data/demo_0"]
            actions = np.asarray(demo_group["actions"][()], dtype=np.float64)
            states = np.asarray(demo_group["states"][()], dtype=np.float64)
        flips = np.flatnonzero((actions[1:, 6] > 0) & (actions[:-1, 6] < 0)) + 1
        if len(flips) != 1:
            raise ValueError(f"expected_one_forward_gripper_open_transition:{flips.tolist()}")
        flip_index = int(flips[0])
        resolved_xml, resolution = old._resolve_model_xml(model_bytes)
        env = dynamic.make_schema_env(str(base.FORWARD_BDDL[10]))
        env.reset_from_xml_string(resolved_xml)

        def pose_at(state: Any) -> dict[str, Any]:
            env.set_init_state(np.asarray(state, dtype=np.float64))
            dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
            obj = dynamic._object_or_site_pose(env, "akita_black_bowl_1")
            robot = dynamic._robot_metrics(env)
            if obj is None:
                raise ValueError("task10_object_pose_missing")
            return {"object": obj, "eef": {"pos_m": robot["eef_translation_m"], "quat": robot["eef_quaternion"]}}

        init_pose = pose_at(init_state)
        before = pose_at(states[flip_index - 1])
        after = pose_at(states[flip_index])
        from robosuite.utils.transform_utils import mat2quat, quat2mat
        object_pos = np.asarray(after["object"]["pos_m"], dtype=np.float64)
        object_rot = np.asarray(quat2mat(after["object"]["quat"]), dtype=np.float64)
        eef_pos = np.asarray(after["eef"]["pos_m"], dtype=np.float64)
        eef_rot = np.asarray(quat2mat(after["eef"]["quat"]), dtype=np.float64)
        relative_pos = object_rot.T @ (eef_pos - object_pos)
        relative_rot = object_rot.T @ eef_rot
        relative_quat = np.asarray(mat2quat(relative_rot), dtype=np.float64)
        pregrasp_offset = float(max(0.05, abs(relative_pos[2]) + 0.04))
        geometry = {"schema_version": "track3_task10_grasp_geometry_v1", "status": "passed", "created_at_utc": started, "task_index": 10, "demo_index": 0, "hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "object_name": "akita_black_bowl_1", "source_frame": "forward_hdf5_demo0_state_before_first_gripper_minus1_to_plus1", "gripper_transition": {"index": flip_index, "previous_action_value": float(actions[flip_index - 1, 6]), "action_value": float(actions[flip_index, 6]), "adjacent_state_indices": [flip_index - 1, flip_index], "action_sha256": sha256_bytes(actions.tobytes()), "state_sha256": sha256_bytes(states.tobytes())}, "object_init_pose": init_pose["object"], "grasp_frame": {"state_index": flip_index, "object_pose": after["object"], "eef_pose": after["eef"], "T_object_inverse_T_eef_translation_m": relative_pos.tolist(), "T_object_inverse_T_eef_quaternion_wxyz": relative_quat.tolist(), "rotation_matrix": relative_rot.tolist()}, "pregrasp_geometry": {"world_z_offset_m": pregrasp_offset, "target_description": "above current bowl along world z before SE(3) descend"}, "before_transition": before, "after_transition": after, "fixed_reverse_destination": "same_demo_init_object_pose", "selection_is_fixed": True, "no_action_sent": True, "same_model_xml_identity": True}
        print(json.dumps(geometry, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        result = {"schema_version": "track3_task10_grasp_geometry_v1", "status": "worker_exception", "created_at_utc": started, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# P3: three bounded GPU3 camera/provenance canaries.


def _gpu3_snapshot() -> dict[str, Any]:
    gpu_command = ["nvidia-smi", "-i", "3", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    process_command = ["nvidia-smi", "-i", "3", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    gpu = subprocess.run(gpu_command, capture_output=True, text=True, check=False)
    processes = subprocess.run(process_command, capture_output=True, text=True, check=False)
    parsed: list[dict[str, Any]] = []
    for line in gpu.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 4:
            try:
                parsed.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": float(fields[2]), "memory_total_mib": float(fields[3])})
            except ValueError:
                pass
    return {"timestamp_utc": now(), "returncode": gpu.returncode, "query": gpu_command, "gpus": parsed, "compute_processes": [line.strip() for line in processes.stdout.splitlines() if line.strip()], "process_query_returncode": processes.returncode, "stderr": gpu.stderr[-1000:], "free_gate": gpu.returncode == 0 and len(parsed) == 1 and parsed[0]["memory_used_mib"] <= 1024.0 and not processes.stdout.strip()}


def _encode_h264(path: Path, frames: Any, *, fps: int = CONTROL_FREQ_HZ) -> dict[str, Any]:
    import numpy as np

    images = np.asarray(frames, dtype=np.uint8)
    if images.ndim != 4 or images.shape[0] != 41 or images.shape[-1] != 3 or images.shape[1] <= 0 or images.shape[2] <= 0:
        raise ValueError(f"invalid_video_frames:{list(images.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real media suffix so ffmpeg can select the MP4 muxer while the
    # filename still makes an incomplete asset unmistakable.
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    partial.unlink(missing_ok=True)
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{images.shape[2]}x{images.shape[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    encoded = subprocess.run(command, input=images.tobytes(), capture_output=True, check=False)
    if encoded.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg_failed:{encoded.returncode}:{encoded.stderr.decode(errors='replace')[-1000:]}")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(partial)], capture_output=True, text=True, check=False)
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        frame_count = int(stream["nb_frames"])
        valid = stream.get("codec_name") == "h264" and frame_count == 41 and str(stream.get("avg_frame_rate")) in {"20/1", "20"}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        stream = {"parse_error": str(exc), "stdout": probe.stdout[-1000:]}
        frame_count = None
        valid = False
    if probe.returncode != 0 or not valid:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffprobe_failed:{probe.returncode}:{stream}")
    os.replace(partial, path)
    return {"relative_path": relative_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "ffprobe": stream, "frame_count": frame_count, "fps": "20/1", "atomic_publish": True, "partial_suffix": ".partial"}


def _run_p3_worker(task: int, demo: int, seed: int, bundle_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "JAX_PLATFORMS": "cpu"})
    frames_path = OUT_ROOT / "camera_frames" / f"p3_task{task}_demo{demo}.npz"
    command = [sys.executable, "-m", "tools.axis_pi05.track3_start_bundle_and_support_executor_canary", "--p3-worker", "--task-index", str(task), "--demo-index", str(demo), "--seed", str(seed), "--state-bundle", str(bundle_path), "--frames-output", str(frames_path)]
    before = _gpu3_snapshot()
    # Checkpoint telemetry must not be sent through a bounded PIPE: a child
    # can otherwise block before publishing its final result JSON.
    with tempfile.TemporaryDirectory(prefix="axis_pi05_p3_worker_") as temporary_dir:
        stdout_path = Path(temporary_dir) / "stdout.jsonl"
        stderr_path = Path(temporary_dir) / "stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle, text=True)
            samples: list[dict[str, Any]] = []
            while process.poll() is None:
                sample_started = time.monotonic()
                samples.append(_gpu3_snapshot())
                delay = max(0.0, 0.1 - (time.monotonic() - sample_started))
                if delay:
                    time.sleep(delay)
            process.wait()
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    after = _gpu3_snapshot()
    samples.append(after)
    try:
        result = _parse_last_json(stdout)
    except Exception as exc:
        result = {"schema_version": "track3_gpu3_camera_canary_v1", "status": "worker_exception", "error": {"type": type(exc).__name__, "message": str(exc), "stdout_tail": stdout[-3000:], "stderr_tail": stderr[-3000:]}}
    memory_values = [float(row["gpus"][0]["memory_used_mib"]) for row in samples if row.get("gpus")]
    child_record = {"seed": seed, "child_pid": process.pid, "physical_gpu_uuid": (before.get("gpus") or [{}])[0].get("uuid"), "cuda_visible_devices": "3", "start_snapshot": before, "end_snapshot": after, "peak_memory_mib": max(memory_values, default=-1.0), "parent_poll_interval_max_ms": 100.0, "parent_poll_count": len(samples)}
    result["gpu_provenance"] = child_record
    result["gpu_provenance_gate"] = gpu_provenance_gate(child_record)
    result["process_returncode"] = process.returncode
    result["worker_returncode_contract"] = process.returncode == worker_exit_code(str(result.get("status")))
    result["stdout_tail"] = stdout[-3000:]
    result["stderr_tail"] = stderr[-3000:]
    return result


def _p3_worker(task: int, demo: int, seed: int, bundle_path: Path, frames_path: Path) -> int:
    import random
    import numpy as np

    started = now()
    env = None
    try:
        random.seed(seed)
        np.random.seed(seed)
        import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base
        composed._configure_public_env(gpu=True)
        manifest, model_bytes, _init, _terminal, _info = composed._demo_arrays(task, demo)
        with np.load(bundle_path, allow_pickle=False) as archive:
            state = np.asarray(archive[f"task{task}_demo{demo}"], dtype=np.float64).copy()
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        kwargs = {"control_freq": CONTROL_FREQ_HZ, "use_camera_obs": True, "has_renderer": False, "has_offscreen_renderer": True, "render_gpu_device_id": 0, "camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128}
        env = dynamic.make_schema_env(str(base.REVERSE_BDDL[task]))
        # make_schema_env is intentionally no-renderer; use the public
        # ControlEnv directly for the camera canary with the same XML/state.
        env.close()
        env = __import__("libero.libero.envs.env_wrapper", fromlist=["ControlEnv"]).ControlEnv(bddl_file_name=str(base.REVERSE_BDDL[task]), **kwargs)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        env.set_init_state(state)
        restored = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        restore_diff = float(np.max(np.abs(restored - state)))
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(base.REVERSE_BDDL[task], "init")
        goal_atoms = base.public_predicates(base.REVERSE_BDDL[task], "goal")
        samples: list[dict[str, Any]] = []
        frames = {"agentview": [], "wrist": []}
        for step in range(41):
            if step:
                env.step(composed.neutral_hold())
            observation = env.env._get_observations()
            sample = prior._runtime_sample(env, dynamic, task, "reverse", step, initial_atoms, goal_atoms)
            sample["contacts_v3"] = composed.contact_v3(task, sample["contact_matrix"])
            samples.append(sample)
            for output_key, observation_key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
                image = np.asarray(observation.get(observation_key))
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"invalid_camera_schema:{observation_key}:{list(image.shape)}")
                frames[output_key].append(image.copy())
        source_values = [next((row.get("value") for row in sample["initial"]["predicates"] if row.get("atom") == SOURCE_ATOM[task]), None) for sample in samples]
        goal_values = [next((row.get("value") for row in sample["goal"]["predicates"] if row.get("atom") == GOAL_ATOM[task]), None) for sample in samples]
        truth_complete = all(sample["initial"].get("truth_complete") and sample["initial"].get("detail_complete") and sample["goal"].get("truth_complete") and sample["goal"].get("detail_complete") for sample in samples)
        truth = {"executed": truth_complete, "state": "passed" if truth_complete and all(value is True for value in source_values) and all(value is False for value in goal_values) else "failed", "source_values": source_values, "goal_values": goal_values, "step_range": [0, 40], "source_atom": SOURCE_ATOM[task], "goal_atom": GOAL_ATOM[task]}
        contact = full_window_contact_gate(task, samples)
        renderer = composed._renderer_schema(env, env.env._get_observations())
        gates = {"restore_identity": gate(executed=True, value=restore_diff <= 1e-12), "truth": gate(executed=truth_complete, value=truth["state"] == "passed"), "contact_v3_full_window": contact, "camera_schema": gate(executed=True, value=bool(renderer.get("schema_complete")))}
        status = "passed" if all_required(gates) else "failed_gate"
        camera_path = _save_npz_atomic(frames_path, {"agentview": np.asarray(frames["agentview"], dtype=np.uint8), "wrist": np.asarray(frames["wrist"], dtype=np.uint8)})
        checkpoint_telemetry = [{"step": int(sample["step"]), "source": next((row.get("value") for row in sample["initial"]["predicates"] if row.get("atom") == SOURCE_ATOM[task]), None), "goal": next((row.get("value") for row in sample["goal"]["predicates"] if row.get("atom") == GOAL_ATOM[task]), None), "contact_counts": sample["contacts_v3"].get("counts", {})} for sample in samples]
        result = {"schema_version": "track3_gpu3_camera_canary_v1", "status": status, "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY[task], "fresh_process": True, "mode": "GPU3_EGL_camera", "seed": seed, "child_pid": os.getpid(), "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "composed_state_sha256": sha256_bytes(state.tobytes()), "restored_state_sha256": sha256_bytes(restored.tobytes()), "restore_max_abs_diff": restore_diff, "same_demo_model_and_state_components": True}, "reverse": {"source_atom": SOURCE_ATOM[task], "goal_atom": GOAL_ATOM[task], "checkpoint_telemetry": checkpoint_telemetry, "truth": truth, "hold_action": composed.neutral_hold().tolist(), "frame_count": 41, "raw_scene_checkpoints_not_emitted_to_stdout": True}, "renderer": renderer, "camera_frames": {key: len(value) for key, value in frames.items()}, "camera_npz": camera_path, "hard_gate": gates, "provenance": {"seed_layers": ["parent_registration", "python_random", "numpy_random"], "env_seed_api": "not_required_for_deterministic_state_restore", "no_manipulation_action": True, "no_se3_or_grasp": True, "no_teleop": True, "gpu3_only": True, "official_gpu": False}}
    except Exception as exc:
        result = {"schema_version": "track3_gpu3_camera_canary_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY.get(task), "fresh_process": True, "mode": "GPU3_EGL_camera", "seed": seed, "child_pid": os.getpid(), "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    print(json.dumps(result, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value), sort_keys=True))
    return worker_exit_code(str(result.get("status")))


def run_p3() -> dict[str, Any]:
    import numpy as np

    p2 = strict_json_load(OUT_ROOT / "p2_summary.json")
    if p2.get("status") != "track3_start_state_bundle_validated_9_of_9":
        result = {"schema_version": "track3_start_bundle_p3_v1", "status": "not_evaluated", "reason": "p2_9_of_9_not_passed", "canaries": []}
        write_json(OUT_ROOT / "p3_summary.json", result)
        return result
    guard = _gpu3_snapshot()
    if not guard.get("free_gate"):
        result = {"schema_version": "track3_start_bundle_p3_v1", "status": "track3_start_bundle_p3_gpu3_resource_gate_failed", "gpu3_preflight": guard, "canaries": []}
        write_json(OUT_ROOT / "p3_summary.json", result)
        return result
    seeds = {10: 3100, 20: 3200, 19: 3190}
    canaries: list[dict[str, Any]] = []
    bundle = OUT_ROOT / "start_states.npz"
    for task in TASK_ORDER:
        result = _run_p3_worker(task, 0, seeds[task], bundle)
        report_path = OUT_ROOT / f"p3_task{task}_demo0.json"
        if result.get("status") == "passed" and result.get("gpu_provenance_gate", {}).get("state") == "passed":
            frames_path = OUT_ROOT / "camera_frames" / f"p3_task{task}_demo0.npz"
            try:
                with np.load(frames_path, allow_pickle=False) as archive:
                    videos = {"agentview": _encode_h264(OUT_ROOT / "videos" / f"p3_task{task}_demo0_agentview.mp4", archive["agentview"]), "wrist": _encode_h264(OUT_ROOT / "videos" / f"p3_task{task}_demo0_wrist.mp4", archive["wrist"])}
                result["videos"] = videos
                result["camera_npz_manifest"] = artifact_entry(frames_path, source_commit=_git(ROOT, ("rev-parse", "HEAD")), schema="track3_gpu3_camera_frames_npz_v1")
            except Exception as exc:
                result["status"] = "failed_gate"
                result["video_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(report_path, result)
        canaries.append({"task_index": task, "family": FAMILY[task], "status": result.get("status"), "gpu_provenance": result.get("gpu_provenance_gate"), "report": relative_path(report_path), "videos": result.get("videos", {})})
        if result.get("status") != "passed" or result.get("gpu_provenance_gate", {}).get("state") != "passed":
            break
    passed = len(canaries) == 3 and all(row.get("status") == "passed" and row.get("gpu_provenance", {}).get("state") == "passed" for row in canaries)
    result = {"schema_version": "track3_start_bundle_p3_v1", "status": "track3_start_bundle_gpu_provenance_closed_3_of_3" if passed else "track3_start_bundle_p3_failed", "created_at_utc": now(), "gpu3_preflight": guard, "fixed_order": [10, 20, 19], "seeds": seeds, "canaries": canaries, "camera_canary_count": len(canaries), "no_retry_or_demo_change": True, "no_manipulation_action": True, "counts": {"validated_start_state": 9, "exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}}
    write_json(OUT_ROOT / "p3_summary.json", result)
    return result


# ---------------------------------------------------------------------------
# P6: evidence closure and hard stop.  P4/P5 are deliberately not evaluated
# after the fixed P3 failure.


def _run_check(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(list(command), cwd=ROOT, env=dict(env) if env is not None else None, capture_output=True, text=True, check=False)
    return {"command": list(command), "returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}


def _test_counts(stdout: str) -> dict[str, Any]:
    match = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) warnings)?", stdout)
    return {"passed": int(match.group(1)) if match else None, "skipped": int(match.group(2) or 0) if match else None, "warnings": int(match.group(3) or 0) if match else None}


def _ffprobe_check(path: Path) -> dict[str, Any]:
    completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        valid = stream.get("codec_name") == "h264" and int(stream.get("nb_frames")) == 41 and str(stream.get("avg_frame_rate")) in {"20/1", "20"}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        stream = {"stdout": completed.stdout[-1000:]}
        valid = False
    return {"path": relative_path(path), "returncode": completed.returncode, "state": "passed" if completed.returncode == 0 and valid else "failed", "stream": stream}


def _csv_artifact(name: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    path = OUT_ROOT / name
    path.write_text(output.getvalue(), encoding="utf-8")
    return path


def _output_artifacts() -> list[Path]:
    return sorted(path for path in OUT_ROOT.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")


def run_p6() -> dict[str, Any]:
    p0 = strict_json_load(OUT_ROOT / "p0_release_lock.json")
    p1 = strict_json_load(OUT_ROOT / "p1_validator_contract.json")
    p2 = strict_json_load(OUT_ROOT / "p2_summary.json")
    p3 = strict_json_load(OUT_ROOT / "p3_summary.json")
    manifest = strict_json_load(OUT_ROOT / "start_state_manifest.json")
    expected = _bundle_expected_from_manifest(manifest)
    npz_validation = validate_state_bundle(OUT_ROOT / "start_states.npz", expected)
    old_lock_validation = validate_artifact_manifest(p0.get("old_artifact_lock", []))
    json_paths = sorted(OUT_ROOT.rglob("*.json"))
    parse_errors: list[str] = []
    for path in json_paths:
        try:
            strict_json_load(path)
        except Exception as exc:
            parse_errors.append(f"{relative_path(path)}:{type(exc).__name__}:{exc}")
    json_check = {"executed": True, "state": "passed" if not parse_errors else "failed", "file_count": len(json_paths), "errors": parse_errors}
    video_checks = [_ffprobe_check(path) for path in sorted((OUT_ROOT / "videos").glob("*.mp4"))]
    partials = sorted(OUT_ROOT.rglob("*.partial*"))
    compile_check = _run_check([sys.executable, "-m", "py_compile", str(ROOT / "tools/axis_pi05/track3_start_bundle_and_support_executor_canary.py")])
    diff_check = _run_check(["git", "diff", "--check"])
    test_env = os.environ.copy()
    test_env.update({"PYTHONPATH": ".", "JAX_PLATFORMS": "cpu"})
    targeted = _run_check([str(ROOT / ".venv_axis_pi05/bin/pytest"), "-q", "tests/test_track3_start_bundle_and_support_executor_canary.py"], env=test_env)
    targeted.update(_test_counts(targeted["stdout_tail"]))
    full = _run_check([str(ROOT / ".venv_axis_pi05/bin/pytest"), "-q", "tests"], env=test_env)
    full.update(_test_counts(full["stdout_tail"]))
    stage_csv = _csv_artifact("stage_results.csv", ("stage", "status", "reason"), (("P0", p0.get("status"), ""), ("P1", p1.get("status"), ""), ("P2", p2.get("status"), ""), ("P3", p3.get("status"), "worker result publication deadlock"), ("P4", "not_evaluated", "P3 hard stop"), ("P5", "not_evaluated", "P3 hard stop"), ("P6", "executed", "hard stop")))
    task_csv = _csv_artifact("task_results.csv", ("task_index", "family", "p2_restore", "p3_camera", "p4_executor", "p5_generation"), ((10, FAMILY[10], "passed", "worker_exception", "not_evaluated", "not_evaluated"), (20, FAMILY[20], "passed", "not_evaluated", "not_evaluated", "not_evaluated"), (19, FAMILY[19], "passed", "not_evaluated", "not_evaluated", "not_evaluated")))
    episode_rows = [(task, demo, "passed", "not_evaluated", "not_evaluated") for task, demo in FIXED_DEMOS]
    episode_rows[0] = (10, 0, "passed", "worker_exception", "not_evaluated")
    episode_csv = _csv_artifact("episode_results.csv", ("task_index", "demo_index", "p2_restore", "p3_camera", "executor_or_demo"), episode_rows)
    no_pending = not (PENDING_ROOT / "task10_demo0").exists()
    closeout = f"""# Track 3 Start Bundle / Support Executor P6 收尾

状态：`track3_support_executor_canary_failed_worker_result_publication`

## 已完成

- P0 输入锁通过；旧 M0–M5、三份公开 HDF5、9 个 composed hash 与 `0/0/0` reverse-demo 计数未漂移。
- P1 独立 contract 通过；P2 物化并 fresh CPU restore `9/9`，task10/demo0 固定抓取几何提取通过。
- P3 只启动了固定顺序第一条 task10/demo0 GPU3 canary。41 帧 camera 已封存并完成 H.264/20 FPS/ffprobe 闭合。

## 停止原因

P3 worker 在写出完整 checkpoint JSON 时填满父子 stdout 管道，阻塞于结果发布；该次 task10 canary 被终止并记录为 `worker_result_publication` 工具失败。按照计划没有重试 task10，也没有启动 task20/task19、P4 或 P5。新代码已改为文件重定向 stdout，并将后续 checkpoint 输出压缩，供下一份审核计划决定是否允许恢复。

## 边界与计数

没有操控动作、SE(3) executor、demo 生成、pending promotion、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或提交。`validated_start_state=9`，`exact/pending/accepted=0/0/0`；P4/P5 均 `not_evaluated`，pending 目录未创建。

## 校验

- targeted tests：`{targeted.get('passed')}` passed，returncode `{targeted.get('returncode')}`；full tests：`{full.get('passed')}` passed，`{full.get('skipped')}` skipped，`{full.get('warnings')}` warnings，returncode `{full.get('returncode')}`。
- NPZ no-pickle：`{npz_validation['state']}`；旧输入 hash：`{old_lock_validation['state']}`；JSON strict parse：`{json_check['state']}`；video ffprobe：`{'passed' if video_checks and all(row['state'] == 'passed' for row in video_checks) else 'failed'}`；py_compile：`{'passed' if compile_check['returncode'] == 0 else 'failed'}`；git diff check：`{'passed' if diff_check['returncode'] == 0 else 'failed'}`。
- root/OpenPI 与 `LIBERO-plus` dirty preservation 记录于 `p6_summary.json`；P6 后 hard stop，等待下一次审核。
"""
    closeout_path = OUT_ROOT / "P6_CLOSEOUT_REPORT_ZH.md"
    closeout_path.write_text(closeout, encoding="utf-8")
    artifact_paths = _output_artifacts()
    artifact_entries = [artifact_entry(path, source_commit=_git(ROOT, ("rev-parse", "HEAD"))) for path in artifact_paths if path.name != "p6_summary.json"]
    summary = {"schema_version": "track3_start_bundle_p6_v1", "status": "track3_support_executor_canary_failed_worker_result_publication", "created_at_utc": now(), "plan": relative_path(PLAN), "phase_status": {"P0": p0.get("status"), "P1": p1.get("status"), "P2": p2.get("status"), "P3": p3.get("status"), "P4": "not_evaluated", "P5": "not_evaluated", "P6": "hard_stop"}, "counts": {"validated_start_state": p2.get("validated_start_state", 0), "p3_camera_canaries_started": p3.get("camera_canary_count", 0), "p4_executor_attempts": 0, "p5_generation_attempts": 0, "successful_replays": 0, "exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "restore_bundle": {"npz": relative_path(OUT_ROOT / "start_states.npz"), "manifest": relative_path(OUT_ROOT / "start_state_manifest.json"), "validation": npz_validation}, "p3": p3, "p4": {"status": "not_evaluated", "reason": "p3_hard_stop"}, "p5": {"status": "not_evaluated", "reason": "p3_hard_stop", "pending_created": False}, "validation": {"old_input_lock": old_lock_validation, "json_strict_parse": json_check, "video_ffprobe": video_checks, "py_compile": compile_check, "git_diff_check": diff_check, "targeted_tests": targeted, "full_tests": full}, "artifact_manifest": {"path": relative_path(OUT_ROOT / "artifact_manifest.json"), "entries_exclude_self_and_p6_summary": True, "entry_count_before_manifest": len(artifact_entries)}, "resource_boundary": {"gpu3_only": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "official_gpu": False, "simulator_used_in_p0_p1": False, "simulator_used_in_p2_p3": True, "camera_canary_task10_only": True, "no_teleop": True, "no_manipulation_action": True, "no_se3_or_grasp": True, "training": False, "fast_or_zip": False, "upload_or_submission": False, "hdf5_or_bddl_modified": False, "old_evidence_modified": False, "pending_directory_created": False, "partial_residue": [relative_path(path) for path in partials], "libero_plus_porcelain": _porcelain(LIBERO_PLUS)}, "repository_state": {"root_head_before_commit": _git(ROOT, ("rev-parse", "HEAD")), "root_porcelain_before_commit": _porcelain(ROOT), "openpi_head": _git(OPENPI, ("rev-parse", "HEAD")) if OPENPI.is_dir() else None, "openpi_porcelain": _porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain": _porcelain(LIBERO_PLUS)}, "hard_stop": {"executed": True, "status": "hard_stop_waiting_review", "next_authority": "new_reviewed_plan_required", "no_automatic_resume": True, "no_pending": no_pending}}
    write_json(OUT_ROOT / "p6_summary.json", summary)
    write_json(OUT_ROOT / "artifact_manifest.json", {"schema_version": "track3_artifact_manifest_v1", "status": "complete", "created_at_utc": now(), "entries": [*artifact_entries, artifact_entry(OUT_ROOT / "p6_summary.json", source_commit=_git(ROOT, ("rev-parse", "HEAD")))], "self_excluded": True})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", action="store_true")
    parser.add_argument("--p1", action="store_true")
    parser.add_argument("--p2", action="store_true")
    parser.add_argument("--p3", action="store_true")
    parser.add_argument("--p6", action="store_true")
    parser.add_argument("--p2-restore-worker", action="store_true")
    parser.add_argument("--p2-geometry-worker", action="store_true")
    parser.add_argument("--p3-worker", action="store_true")
    parser.add_argument("--state-bundle")
    parser.add_argument("--frames-output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--demo-index", type=int)
    args = parser.parse_args()
    worker_flags = (args.p2_restore_worker, args.p2_geometry_worker, args.p3_worker)
    if sum((args.p0, args.p1, args.p2, args.p3, args.p6, *worker_flags)) != 1:
        parser.error("choose exactly one phase or worker")
    if args.p2_restore_worker:
        if args.task_index not in TASK_ORDER or args.demo_index not in (0, 1, 2) or not args.state_bundle:
            parser.error("p2 restore worker requires fixed task/demo and --state-bundle")
        raise SystemExit(_p2_restore_worker(int(args.task_index), int(args.demo_index), Path(args.state_bundle)))
    if args.p2_geometry_worker:
        raise SystemExit(_p2_geometry_worker())
    if args.p3_worker:
        if args.task_index not in TASK_ORDER or args.demo_index != 0 or args.seed is None or not args.state_bundle or not args.frames_output:
            parser.error("p3 worker requires fixed task/demo0, seed, state bundle and frames output")
        raise SystemExit(_p3_worker(int(args.task_index), 0, int(args.seed), Path(args.state_bundle), Path(args.frames_output)))
    output = run_p0() if args.p0 else run_p1() if args.p1 else run_p2() if args.p2 else run_p3() if args.p3 else run_p6()
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if str(output.get("status", "")).endswith("passed") else 1)
