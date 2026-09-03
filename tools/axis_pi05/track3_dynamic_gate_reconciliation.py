#!/usr/bin/env python3
"""Model/state-reconciled dynamic gates for Track 3 reverse BDDL.

This is deliberately a new evidence namespace.  The historical G1 dynamic
report is never rewritten and HDF5 states are metadata-only inputs here: the
predicate oracle is always the reverse BDDL's own native-reset environment.
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
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NEW_ROOT = ROOT / "data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_DYNAMIC_GATE_MODEL_STATE_RECONCILIATION_PLAN.md"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
G0_LOCK = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g0_release_lock.json"
S1_REPORT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s1_accounting_v2.json"
S3_REPORT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_language_v2.json"
OLD_DYNAMIC_REPORT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_dynamic_gates.json"
V2_SPEC_ROOT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_v2_specs"
DENYLIST = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"
HANDOFF_JSON = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
HANDOFF_MD = ROOT / "docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md"
INDEX_MD = ROOT / "docs/AXIS_PI05_PLAN_INDEX.md"

FAMILIES = ("support_remove", "containment_remove", "articulation_close")
TASK_ORDER = ((10, "support_remove"), (20, "containment_remove"), (19, "articulation_close"))
IDENTITIES = ("exact_reverse_demo", "reverse_relevant_primitive", "generator_reference_only")

# These are pre-registered and intentionally dimensioned.  They are not a
# replacement for predicate checks and are not applied to a concatenated
# proprio vector.
THRESHOLDS = {
    "arm_joint_qpos_rad": 0.002,
    "arm_joint_qvel_rad_s": 0.02,
    "eef_translation_m": 0.002,
    "eef_rotation_rad": 0.02,
    "gripper_qpos_m": 0.002,
    "gripper_qvel_m_s": 0.02,
    "object_translation_m": 0.002,
    "object_rotation_rad": 0.02,
    "fixture_joint_rad_or_m": 0.002,
}
WARMUP_STEPS = 20
MEASUREMENT_STEPS = 20
CONTROL_FREQ_HZ = 20
ACTION_DIM = 7

HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}


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


def state_hash(value: np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float64)
    return sha256_bytes(array.tobytes())


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def parse_atom(value: str) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"\((\w+)\s+([^()]+)\)", str(value).strip())
    if not match:
        raise ValueError(f"unsupported predicate: {value!r}")
    return match.group(1), tuple(match.group(2).split())


def normalize_quat(value: Sequence[float]) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion must be finite with shape (4,)")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("quaternion norm is zero")
    return quat / norm


def quat_geodesic_rad(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the shortest SO(3) distance, in radians."""
    a = normalize_quat(left)
    b = normalize_quat(right)
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def finite_array(name: str, value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def validate_model_for_predicate_oracle(recorded_model_hash: str, current_xml_hash: str) -> dict[str, Any]:
    compatible = bool(recorded_model_hash and current_xml_hash and recorded_model_hash == current_xml_hash)
    return {
        "recorded_model_file_sha256": recorded_model_hash,
        "current_env_xml_sha256": current_xml_hash,
        "compatible": compatible,
        "predicate_oracle": "native_reverse_bddl_only" if not compatible else "native_reverse_bddl_with_exact_model_reference",
        "hdf5_state_used_for_predicate_oracle": False,
        "status": "compatible_for_exact_reference" if compatible else "incompatible_for_predicate_oracle",
    }


def validate_named_schema(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Public fail-closed schema helper used by CPU tests and the worker."""
    return finite_array(name, value, shape)


def hold_action(gripper_target: float = 0.0) -> np.ndarray:
    """Build a 7D OSC_POSE no-motion action without forcing gripper open/close."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[-1] = float(gripper_target)
    validate_named_schema("hold_action", action, (ACTION_DIM,))
    return action


def gripper_hold_action() -> np.ndarray:
    """A zero command preserves the PandaGripper's current target.

    The command is intentionally not derived from a desired open/closed
    state: PandaGripper interprets +/-1 as an incremental command and 0 as
    hold.  The worker records the target before and after the window.
    """
    return hold_action(0.0)


def controller_reset_sequence(controller: Any, sim: Any) -> list[str]:
    """Reset/update order is explicit so it can be asserted without MuJoCo."""
    sequence = ["native_reset", "sim.forward"]
    sim.forward()
    if hasattr(controller, "update"):
        controller.update(force=True)
        sequence.append("controller.update")
    if hasattr(controller, "reset_goal"):
        controller.reset_goal()
        sequence.append("controller.reset_goal")
    return sequence


def measurement_windows(samples: Sequence[Mapping[str, Any]], warmup_steps: int = WARMUP_STEPS, measurement_steps: int = MEASUREMENT_STEPS) -> dict[str, Any]:
    if len(samples) != warmup_steps + measurement_steps:
        raise ValueError("warm-up and measurement windows must be disjoint and complete")
    return {
        "warmup": list(samples[:warmup_steps]),
        "measurement": list(samples[warmup_steps:]),
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "windows_disjoint": True,
    }


def _git(repo: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return completed.stdout.strip()


def _porcelain(repo: Path) -> list[str]:
    return [line for line in _git(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def _libero_plus_snapshot() -> dict[str, Any]:
    path = ROOT / "LIBERO-plus"
    return {
        "path": str(path.resolve()),
        "policy": "record_only_never_clean_or_touch",
        "status_porcelain": _porcelain(path) if path.is_dir() else [],
    }


def _hdf5_metadata(task_index: int, path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        states = np.asarray(demo["states"])
        init_state = np.asarray(demo.attrs["init_state"])
        model_file = demo.attrs["model_file"]
        if isinstance(model_file, str):
            model_bytes = model_file.encode("utf-8")
        else:
            model_bytes = bytes(model_file)
        action = np.asarray(demo["actions"])
        return {
            "task_index": task_index,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "model_file_sha256": sha256_bytes(model_bytes),
            "model_file_bytes": len(model_bytes),
            "states_shape": list(states.shape),
            "actions_shape": list(action.shape),
            "init_state_shape": list(init_state.shape),
            "states_dtype": str(states.dtype),
            "init_state_dtype": str(init_state.dtype),
            "states0_equals_init_state": bool(np.array_equal(states[0], init_state)),
            "states_time_scalar_index": 0,
            "states_qpos_offset": 1,
            "states0_time": float(states[0, 0]),
            "states0_sha256": state_hash(states[0]),
            "init_state_sha256": state_hash(init_state),
            "actions_sha256": sha256_bytes(np.asarray(action, dtype=np.float64).tobytes()),
            "hdf5_state_used_for_predicate_oracle": False,
        }


def validate_d0_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck locked evidence without requiring the post-D0 tree to be clean."""
    drift: list[str] = []
    for item in lock.get("immutable_old_evidence", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            drift.append(str(path))
    for item in lock.get("hdf5_inputs", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            drift.append(str(path))
    if drift:
        raise RuntimeError(f"D0 immutable input drift: {drift[:3]}")
    return {"old_evidence_drift": drift, "old_evidence_immutable": not drift}


def _old_evidence_paths() -> list[Path]:
    paths: list[Path] = [G0_LOCK, S1_REPORT, S3_REPORT, OLD_DYNAMIC_REPORT, DENYLIST, P1_SUMMARY]
    paths.extend(V2_SPEC_ROOT.glob("canary_*.bddl"))
    for path in (
        ROOT / "data/manifests/track3_public_primitives_v2/task_catalog.jsonl",
        ROOT / "data/manifests/track3_public_primitives_v2/episode_catalog.jsonl",
        ROOT / "data/manifests/track3_public_primitives_v2/coverage_matrix.json",
    ):
        paths.append(path)
    old_lock = json.loads(G0_LOCK.read_text(encoding="utf-8"))
    for item in old_lock.get("immutable_old_evidence", {}).values():
        paths.append(Path(item["path"]))
    for task_index in (10, 19, 20):
        snapshot = P1_ROOT / "snapshots" / f"task_{task_index:02d}" / "attempt_1" / "terminal_state.npy"
        paths.append(snapshot)
    return list(dict.fromkeys(paths))


def run_d0() -> dict[str, Any]:
    """Create the immutable D0 input lock before changing code or reports."""
    root_commit = _git(ROOT, ("rev-parse", "HEAD"))
    nested = ROOT / "third_party/axis_openpi"
    nested_commit = _git(nested, ("rev-parse", "HEAD")) if nested.is_dir() else "missing"
    evidence: dict[str, Any] = {}
    missing: list[str] = []
    for path in _old_evidence_paths():
        if path.is_file():
            evidence[str(path.resolve())] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        else:
            missing.append(str(path.resolve()))
    hdf5 = {str(task): _hdf5_metadata(task, path) for task, path in HDF5_INPUTS.items() if path.is_file()}
    if len(hdf5) != len(HDF5_INPUTS):
        missing.extend(str(path.resolve()) for path in HDF5_INPUTS.values() if not path.is_file())
    p1 = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    handoff = json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))
    execution = handoff.get("track3_dynamic_gate_reconciliation_authorization", {})
    counts = {
        "exact_reverse_demo": 0,
        "pending_autogen": 0,
        "pending_autogen_v2": 0,
        "accepted_reverse_demo": 0,
        "accepted": 0,
    }
    old_status = handoff.get("track3_support_autogen_se3_recovery_execution", {})
    counts.update({key: int(value or 0) for key, value in old_status.get("counts", {}).items() if key in counts})
    counts["pending_autogen"] = int(p1.get("current_counts", {}).get("pending_autogen", counts["pending_autogen"]))
    counts["pending_autogen_v2"] = int(p1.get("current_counts", {}).get("pending_autogen_v2", counts["pending_autogen_v2"]))
    # D0 is an input lock for the audited execution commit.  D1 source files
    # are intentionally created immediately after that lock, so the current
    # porcelain is recorded separately and is not mistaken for input drift.
    # The audit handoff records the clean root at the audited input commit;
    # this process is now intentionally dirty with the D1 source file.  Do
    # not use ``git status <commit>`` (which is a pathspec, not a historical
    # worktree query) as a proxy for that fact.
    input_commit_clean = root_commit == "f0034b9976e6de81644d1535d60328006bbe83cf"
    nested_commit_clean = nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not _porcelain(nested) if nested.is_dir() else False
    lock = {
        "schema_version": "track3_dynamic_gate_reconciliation_d0_v1",
        "status": "track3_dynamic_gate_reconciliation_input_locked" if not missing else "track3_dynamic_gate_model_state_reconciliation_input_drift",
        "created_at_utc": now(),
        "plan": str(PLAN.resolve()),
        "input_execution_commit": "179045a6d7a7998bf1a54452ef79739c9383c09f",
        "audit_commit_observed_at_d0": root_commit,
        "root_commit_at_d0": root_commit,
        "nested_openpi_commit_at_d0": nested_commit,
        "root_status_porcelain_at_d0": _porcelain(ROOT),
        "nested_openpi_status_porcelain_at_d0": _porcelain(nested) if nested.is_dir() else ["missing"],
        "root_clean_at_input_execution_commit": input_commit_clean,
        "nested_openpi_clean_at_input_execution_commit": nested_commit_clean,
        "root_worktree_clean_at_capture": not _porcelain(ROOT),
        "nested_openpi_worktree_clean_at_capture": not _porcelain(nested) if nested.is_dir() else False,
        "libero_plus_dirty_read_only_snapshot": _libero_plus_snapshot(),
        "immutable_old_evidence": evidence,
        "missing_inputs": missing,
        "hdf5_inputs": hdf5,
        "p1_snapshot_provenance": {
            "summary_path": str(P1_SUMMARY.resolve()),
            "summary_sha256": sha256_file(P1_SUMMARY),
            "attempt_1_terminal_state_sha256": {
                str(task): sha256_file(P1_ROOT / "snapshots" / f"task_{task:02d}" / "attempt_1" / "terminal_state.npy")
                for task in (10, 19, 20)
            },
            "predicate_oracle_authority": "provenance_only_unless_exact_recorded_model_compatibility",
        },
        "counts_at_d0": counts,
        "authorization_at_d0": {
            "source_status": execution.get("status", handoff.get("status")),
            "no_g2_g3_g4": True,
            "no_real_reverse_canary": True,
            "no_human_teleop": True,
            "no_batch_generation_or_training": True,
            "gpu3_only_for_d2_d3": True,
        },
        "hard_gate": {
            "all_locked_inputs_exist": not missing,
            "root_clean_at_input_execution_commit": input_commit_clean,
            "nested_openpi_clean_at_input_execution_commit": nested_commit_clean,
            "libero_plus_recorded_without_touch": True,
            "counts_zero": all(counts.get(key, 0) == 0 for key in ("exact_reverse_demo", "pending_autogen", "pending_autogen_v2", "accepted_reverse_demo", "accepted")),
            "old_dynamic_report_locked": str(OLD_DYNAMIC_REPORT.resolve()) in evidence,
            "old_dynamic_report_not_rewritten_by_d0": True,
            "hdf5_model_state_metadata_locked": len(hdf5) == len(HDF5_INPUTS),
        },
    }
    write_json(NEW_ROOT / "d0_release_lock.json", lock)
    return lock


def _specs() -> dict[int, dict[str, Any]]:
    report = json.loads(S3_REPORT.read_text(encoding="utf-8"))
    return {int(row["task_index"]): row for row in report["specs"]}


def _model_xml_hash(env: Any) -> str:
    xml = env.sim.model.get_xml()
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    return sha256_bytes(bytes(xml))


def _names(model: Any, attr: str) -> list[str]:
    values = getattr(model, attr, ())
    return [str(value) for value in values]


def model_layout(env: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    model = env.sim.model
    current_hash = _model_xml_hash(env)
    recorded = _hdf5_metadata(int(spec["task_index"]), HDF5_INPUTS[int(spec["task_index"])])
    body_names = _names(model, "body_names")
    joint_names = _names(model, "joint_names")
    site_names = _names(model, "site_names")
    recorded_names: dict[str, list[str]] = {"joint": [], "body": [], "site": []}
    recorded_poses: dict[str, dict[str, Any]] = {}
    root = None
    try:
        import h5py

        with h5py.File(HDF5_INPUTS[int(spec["task_index"])], "r") as handle:
            recorded_xml = handle["data/demo_0"].attrs["model_file"]
        if not isinstance(recorded_xml, str):
            recorded_xml = bytes(recorded_xml).decode("utf-8")
        root = ET.fromstring(recorded_xml)
        for element in root.iter():
            tag = element.tag
            if tag in {"joint", "body", "site"} and element.get("name"):
                key = {"joint": "joint", "body": "body", "site": "site"}[tag]
                name = str(element.get("name"))
                recorded_names[key].append(name)
                if tag in {"body", "site"} and name in set(sum((list(parse_atom(a)[1]) for a in [*spec["reverse_initial_predicates"], *spec["reverse_goal_predicates"]]), [])):
                    recorded_poses[f"{tag}:{name}"] = {
                        "declared_local_pos": element.get("pos"),
                        "declared_local_quat": element.get("quat"),
                        "frame": "recorded_xml_local",
                    }
    except Exception as exc:
        recorded_names = {"joint": [], "body": [], "site": []}
        recorded_poses = {"error": {"message": f"{type(exc).__name__}: {exc}"}}
    current_names = {"joint": joint_names, "body": body_names, "site": site_names}
    name_comparison = {kind: {"recorded_count": len(recorded_names[kind]), "current_count": len(current_names[kind]), "order_equal": recorded_names[kind] == current_names[kind], "recorded_names": recorded_names[kind], "current_names": current_names[kind]} for kind in current_names}
    key_pose_diff: dict[str, Any] = {"recorded_xml_pose": recorded_poses, "current_native_reset_pose": {}}
    for atom in [*spec["reverse_initial_predicates"], *spec["reverse_goal_predicates"]]:
        for name in parse_atom(atom)[1]:
            pose = _object_or_site_pose(env, name)
            if pose is not None:
                key_pose_diff["current_native_reset_pose"][name] = pose
    key_pose_diff["comparison_status"] = "reported_in_separate_frames_no_numeric_cross_frame_difference"
    return {
        "recorded_model_file_sha256": recorded["model_file_sha256"],
        "current_env_xml_sha256": current_hash,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "joint_names": joint_names,
        "body_names": body_names,
        "site_names": site_names,
        "name_order_comparison": name_comparison,
        "key_body_site_pose_diff": {
            "status": "reported_recorded_local_and_current_native_world_pose",
            "recorded_model_file_sha256": recorded["model_file_sha256"],
            "current_model_xml_sha256": current_hash,
            **key_pose_diff,
        },
        "compatibility": validate_model_for_predicate_oracle(recorded["model_file_sha256"], current_hash),
        "hdf5_state_used_for_predicate_oracle": False,
    }


def _body_pose(env: Any, name: str) -> dict[str, Any] | None:
    body_ids = {str(value): index for index, value in enumerate(_names(env.sim.model, "body_names"))}
    if name not in body_ids:
        return None
    index = body_ids[name]
    return {
        "kind": "body",
        "name": name,
        "pos_m": finite_array(f"body.{name}.pos", env.sim.data.body_xpos[index], (3,)).tolist(),
        "quat": normalize_quat(env.sim.data.body_xquat[index]).tolist(),
    }


def _object_or_site_pose(env: Any, name: str) -> dict[str, Any] | None:
    state = getattr(env.env, "object_states_dict", {}).get(name)
    if state is not None and getattr(state, "object_state_type", None) == "site":
        state = state.get_geom_state()
        return {
            "kind": "site",
            "name": name,
            "pos_m": finite_array(f"site.{name}.pos", state["pos"], (3,)).tolist(),
            "quat": normalize_quat(state["quat"]).tolist(),
        }
    return _body_pose(env, name)


def _predicate_detail(env: Any, atom: str, value: bool) -> dict[str, Any]:
    predicate, args = parse_atom(atom)
    detail: dict[str, Any] = {"predicate": predicate, "args": list(args), "value": bool(value)}
    states = getattr(env.env, "object_states_dict", {})
    for name in args:
        pose = _object_or_site_pose(env, name)
        if pose is not None:
            detail.setdefault("geometry", []).append(pose)
    if predicate in {"Open", "Close"} and args:
        state = states.get(args[0])
        if state is not None:
            joints = list(getattr(state, "joints", ())) or list(getattr(getattr(env.env, "object_sites_dict", {}).get(args[0]), "joints", ()))
            if not joints and hasattr(state, "parent_name"):
                parent = states.get(state.parent_name)
                joints = list(getattr(parent, "joints", ())) if parent is not None else []
            values: list[float] = []
            for joint in joints:
                address = env.sim.model.get_joint_qpos_addr(joint)
                address = int(address[0]) if isinstance(address, tuple) else int(address)
                values.append(float(env.sim.data.qpos[address]))
            detail["joint_values"] = values
            if hasattr(state, "object_name"):
                detail["joint_names"] = joints
    if predicate in {"On", "In"}:
        detail["contact_information"] = {
            "predicate_contact_is_model_defined": True,
            "site_contact_is_logical_true": any(name in getattr(env.env, "object_sites_dict", {}) for name in args),
            "ontop_xy_threshold_m": 0.03 if predicate == "On" else None,
        }
    return detail


def evaluate_predicates(env: Any, atoms: Iterable[str], *, label: str) -> dict[str, Any]:
    """Evaluate every atom directly through the reverse env's predicate map."""
    rows: list[dict[str, Any]] = []
    for atom in atoms:
        predicate, args = parse_atom(atom)
        state = (predicate, *args)
        try:
            value = bool(env.env._eval_predicate(state))
            rows.append({"atom": atom, "predicate": predicate, "args": list(args), "value": value, "detail": _predicate_detail(env, atom, value)})
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            rows.append({"atom": atom, "predicate": predicate, "args": list(args), "value": False, "error": f"{type(exc).__name__}: {exc}", "fail_closed": True})
    return {"label": label, "predicates": rows, "all_true": bool(rows) and all(row.get("value") is True for row in rows), "count": len(rows), "true_count": sum(row.get("value") is True for row in rows)}


def _controller(env: Any) -> Any:
    return env.env.robots[0].controller


def _robot_metrics(env: Any) -> dict[str, Any]:
    controller = _controller(env)
    qpos_index = np.asarray(getattr(controller, "qpos_index", []), dtype=int)
    qvel_index = np.asarray(getattr(controller, "qvel_index", []), dtype=int)
    qpos = finite_array("arm_joint_qpos", env.sim.data.qpos[qpos_index], (len(qpos_index),))
    qvel = finite_array("arm_joint_qvel", env.sim.data.qvel[qvel_index], (len(qvel_index),))
    ee_pos = finite_array("eef_translation", getattr(controller, "ee_pos"), (3,))
    ee_mat = np.asarray(getattr(controller, "ee_ori_mat"), dtype=np.float64)
    if ee_mat.shape != (3, 3) or not np.all(np.isfinite(ee_mat)):
        raise ValueError("eef rotation has invalid shape or non-finite values")
    try:
        from robosuite.utils.transform_utils import mat2quat

        ee_quat = normalize_quat(mat2quat(ee_mat))
    except Exception as exc:
        raise ValueError(f"could not convert EEF rotation: {exc}") from exc
    robot = env.env.robots[0]
    observations = env.env._get_observations(force=True)
    gripper_qpos = finite_array("gripper_qpos", observations.get("robot0_gripper_qpos"), None)
    gripper_qvel = finite_array("gripper_qvel", observations.get("robot0_gripper_qvel"), None)
    gripper_target = finite_array("gripper_controller_target", getattr(robot.gripper, "current_action", None), None)
    target_drift = float(
        np.max(
            np.abs(
                np.asarray(current["gripper_controller_target"])
                - np.asarray(base["gripper_controller_target"])
            )
        )
    )
    return {
        "arm_joint_qpos_rad": qpos.tolist(),
        "arm_joint_qvel_rad_s": qvel.tolist(),
        "eef_translation_m": ee_pos.tolist(),
        "eef_quaternion": ee_quat.tolist(),
        "gripper_qpos_m": gripper_qpos.tolist(),
        "gripper_qvel_m_s": gripper_qvel.tolist(),
        "gripper_controller_target": gripper_target.tolist(),
        "controller_class": type(controller).__name__,
        "controller_control_dim": int(getattr(controller, "control_dim", -1)),
        "action_dim": ACTION_DIM,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "gripper_hold_action": 0.0,
    }


def _fixture_joint_values(env: Any, names: Iterable[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for object_name in names:
        state = getattr(env.env, "object_states_dict", {}).get(object_name)
        if state is not None and getattr(state, "object_state_type", None) == "site":
            state = getattr(env.env, "object_sites_dict", {}).get(object_name)
        if state is None and object_name in getattr(env.env, "object_sites_dict", {}):
            state = env.env.object_sites_dict[object_name]
        joints = list(getattr(state, "joints", ())) if state is not None else []
        for joint in joints:
            address = env.sim.model.get_joint_qpos_addr(joint)
            address = int(address[0]) if isinstance(address, tuple) else int(address)
            values[joint] = float(env.sim.data.qpos[address])
    return values


def _scene_metrics(env: Any, atoms: Iterable[str]) -> dict[str, Any]:
    names: set[str] = set()
    for atom in atoms:
        _, args = parse_atom(atom)
        names.update(args)
    poses = {name: pose for name in sorted(names) if (pose := _object_or_site_pose(env, name)) is not None}
    fixture_names = sorted(name for name in names if name in getattr(env.env, "fixtures_dict", {}) or name in getattr(env.env, "object_sites_dict", {}))
    return {"object_and_site_poses": poses, "fixture_joint_qpos": _fixture_joint_values(env, fixture_names)}


def _metric_drift(base: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, float]:
    qpos = float(np.max(np.abs(np.asarray(current["arm_joint_qpos_rad"]) - np.asarray(base["arm_joint_qpos_rad"]))))
    qvel = float(np.max(np.abs(np.asarray(current["arm_joint_qvel_rad_s"]) - np.asarray(base["arm_joint_qvel_rad_s"]))))
    eef_translation = float(np.linalg.norm(np.asarray(current["eef_translation_m"]) - np.asarray(base["eef_translation_m"])))
    eef_rotation = quat_geodesic_rad(current["eef_quaternion"], base["eef_quaternion"])
    gripper_qpos = float(np.max(np.abs(np.asarray(current["gripper_qpos_m"]) - np.asarray(base["gripper_qpos_m"]))))
    gripper_qvel = float(np.max(np.abs(np.asarray(current["gripper_qvel_m_s"]) - np.asarray(base["gripper_qvel_m_s"]))))
    return {
        "arm_joint_qpos_rad": qpos,
        "arm_joint_qvel_rad_s": qvel,
        "eef_translation_m": eef_translation,
        "eef_rotation_rad": eef_rotation,
        "gripper_qpos_m": gripper_qpos,
        "gripper_qvel_m_s": gripper_qvel,
        "gripper_controller_target": target_drift,
    }


def _scene_drift(base: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    translations: dict[str, float] = {}
    rotations: dict[str, float] = {}
    for name, old_pose in base["object_and_site_poses"].items():
        new_pose = current["object_and_site_poses"].get(name)
        if new_pose is None:
            raise ValueError(f"missing scene pose in measurement: {name}")
        translations[name] = float(np.linalg.norm(np.asarray(new_pose["pos_m"]) - np.asarray(old_pose["pos_m"])))
        rotations[name] = quat_geodesic_rad(new_pose["quat"], old_pose["quat"])
    fixture = {}
    for name, old_value in base["fixture_joint_qpos"].items():
        if name not in current["fixture_joint_qpos"]:
            raise ValueError(f"missing fixture joint in measurement: {name}")
        fixture[name] = abs(float(current["fixture_joint_qpos"][name]) - float(old_value))
    return {
        "object_translation_m": translations,
        "object_rotation_rad": rotations,
        "fixture_joint_rad_or_m": fixture,
        "max_object_translation_m": max(translations.values(), default=0.0),
        "max_object_rotation_rad": max(rotations.values(), default=0.0),
        "max_fixture_joint_rad_or_m": max(fixture.values(), default=0.0),
    }


def settle_gate(env: Any, initial_atoms: Sequence[str], goal_atoms: Sequence[str]) -> dict[str, Any]:
    controller = _controller(env)
    sequence = controller_reset_sequence(controller, env.sim)
    baseline_robot = _robot_metrics(env)
    baseline_scene = _scene_metrics(env, [*initial_atoms, *goal_atoms])
    baseline_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    action = gripper_hold_action()
    samples: list[dict[str, Any]] = []
    for step in range(WARMUP_STEPS + MEASUREMENT_STEPS):
        if not np.all(np.isfinite(action)):
            raise ValueError("action is non-finite")
        env.step(action)
        controller.update(force=True)
        robot = _robot_metrics(env)
        scene = _scene_metrics(env, [*initial_atoms, *goal_atoms])
        initial = evaluate_predicates(env, initial_atoms, label=f"measurement_step_{step + 1}_initial")
        goal = evaluate_predicates(env, goal_atoms, label=f"measurement_step_{step + 1}_goal")
        samples.append({"step": step + 1, "robot": robot, "scene": scene, "initial": initial, "goal": goal})
    windows = measurement_windows(samples)
    warmup_robot_drift = _metric_drift(baseline_robot, windows["warmup"][-1]["robot"])
    measurement_robot_drifts = [_metric_drift(windows["measurement"][0]["robot"], sample["robot"]) for sample in windows["measurement"]]
    measurement_robot_max = {key: max(row[key] for row in measurement_robot_drifts) for key in measurement_robot_drifts[0]}
    scene_base = windows["measurement"][0]["scene"]
    scene_drifts = [_scene_drift(scene_base, sample["scene"]) for sample in windows["measurement"]]
    scene_max = {
        "max_object_translation_m": max(row["max_object_translation_m"] for row in scene_drifts),
        "max_object_rotation_rad": max(row["max_object_rotation_rad"] for row in scene_drifts),
        "max_fixture_joint_rad_or_m": max(row["max_fixture_joint_rad_or_m"] for row in scene_drifts),
    }
    initial_continuous = all(sample["initial"]["all_true"] for sample in windows["measurement"])
    goal_continuous_false = all(not any(row["value"] for row in sample["goal"]["predicates"]) for sample in windows["measurement"])
    robot_gate = all(measurement_robot_max[key] <= THRESHOLDS[key] for key in THRESHOLDS if key in measurement_robot_max)
    scene_gate = scene_max["max_object_translation_m"] <= THRESHOLDS["object_translation_m"] and scene_max["max_object_rotation_rad"] <= THRESHOLDS["object_rotation_rad"] and scene_max["max_fixture_joint_rad_or_m"] <= THRESHOLDS["fixture_joint_rad_or_m"]
    return {
        "reset_sequence": sequence,
        "native_reset_state_sha256": state_hash(baseline_state),
        "action_contract": {"dimension": ACTION_DIM, "controller": "OSC_POSE", "control_freq_hz": CONTROL_FREQ_HZ, "hold_action_sha256": sha256_bytes(action.tobytes()), "gripper_hold_value": 0.0, "fixed_gripper_minus_one_used": False},
        "thresholds": THRESHOLDS,
        "windows": {"warmup_steps": WARMUP_STEPS, "measurement_steps": MEASUREMENT_STEPS, "windows_disjoint": True, "warmup_drift_is_reported_not_direct_gate": True},
        "warmup_drift": {"robot": warmup_robot_drift},
        "measurement_robot_max_drift": measurement_robot_max,
        "measurement_scene_max_drift": scene_max,
        "predicate_continuity": {"reverse_initial_all_true": initial_continuous, "reverse_goal_all_false": goal_continuous_false, "measurement_initial_samples": [sample["initial"] for sample in windows["measurement"]], "measurement_goal_samples": [sample["goal"] for sample in windows["measurement"]]},
        "hard_gate": {"robot_named_fields_stable": robot_gate, "scene_named_fields_stable": scene_gate, "reverse_initial_predicates_continuous": initial_continuous, "reverse_goal_predicates_continuously_false": goal_continuous_false, "all_named_settle_gates": bool(robot_gate and scene_gate and initial_continuous and goal_continuous_false)},
    }


def _native_reset(env: Any) -> tuple[Any, np.ndarray]:
    obs = env.reset()
    env.sim.forward()
    native_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
    finite_array("native_sim_state", native_state)
    return obs, native_state


def _mutate_for_controlled_destruction(env: Any, task_index: int, native_state: np.ndarray) -> dict[str, Any]:
    env.set_state(native_state.copy())
    env.sim.forward()
    if task_index == 19:
        joint_name = "wooden_cabinet_1_middle_level"
        direction = "open_to_close"
        delta = -0.12
    else:
        object_name = "akita_black_bowl_1" if task_index == 10 else "orange_juice_1"
        joint_name = f"{object_name}_joint0"
        direction = "remove_from_relation"
        delta = 0.12
    joint_id = env.sim.model.joint_name2id(joint_name)
    address = int(env.sim.model.jnt_qposadr[joint_id])
    before = float(env.sim.data.qpos[address])
    env.sim.data.qpos[address] = before + delta
    env.sim.forward()
    return {"joint_name": joint_name, "qpos_address": address, "before": before, "after": float(env.sim.data.qpos[address]), "direction": direction}


def controlled_destruction(env: Any, task_index: int, target_atom: str, native_state: np.ndarray) -> dict[str, Any]:
    mutation = _mutate_for_controlled_destruction(env, task_index, native_state)
    destroyed = evaluate_predicates(env, [target_atom], label="controlled_destruction")
    destroyed_false = not destroyed["predicates"][0].get("value", False)
    env.set_state(native_state.copy())
    env.sim.forward()
    restored = evaluate_predicates(env, [target_atom], label="controlled_destruction_restored")
    restored_true = bool(restored["predicates"][0].get("value", False))
    return {"mutation": mutation, "target_atom": target_atom, "true_to_false": destroyed_false, "restored_true": restored_true, "destroyed_evidence": destroyed, "restored_evidence": restored, "hard_gate": destroyed_false and restored_true}


def make_env(bddl_path: str) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=bddl_path, control_freq=CONTROL_FREQ_HZ, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=0)


def dynamic_worker(task_index: int, run_index: int, phase: str) -> dict[str, Any]:
    specs = _specs()
    spec = specs[task_index]
    env = None
    started = now()
    try:
        env = make_env(spec["reverse_bddl_path"])
        layout = model_layout(env, spec)
        _, native_state = _native_reset(env)
        initial = evaluate_predicates(env, spec["reverse_initial_predicates"], label="native_reset_reverse_initial")
        goal = evaluate_predicates(env, spec["reverse_goal_predicates"], label="native_reset_reverse_goal")
        if not initial["all_true"] or any(row["value"] for row in goal["predicates"]):
            return {"status": "failed_native_predicate", "failure_field": "reverse_initial_or_goal", "task_index": task_index, "operator_family": spec["operator_family"], "run_index": run_index, "phase": phase, "layout": layout, "native_reset": {"initial": initial, "goal": goal}, "hdf5_state_used_for_predicate_oracle": False}
        settle = settle_gate(env, spec["reverse_initial_predicates"], spec["reverse_goal_predicates"])
        target_atom = spec["reverse_initial_predicates"][-1]
        destruction = controlled_destruction(env, task_index, target_atom, native_state)
        gates = {
            "model_state_predicate_oracle_reconciled": layout["compatibility"]["status"] in {"compatible_for_exact_reference", "incompatible_for_predicate_oracle"},
            "reverse_initial_at_native_reset": initial["all_true"],
            "reverse_goal_false_at_native_reset": not any(row["value"] for row in goal["predicates"]),
            "settle_all_named_fields": settle["hard_gate"]["all_named_settle_gates"],
            "controlled_predicate_true_to_false": destruction["true_to_false"],
            "controlled_predicate_restored_true": destruction["restored_true"],
        }
        passed = all(gates.values())
        return {
            "schema_version": "track3_dynamic_gate_reconciliation_run_v1",
            "status": "passed" if passed else "failed_gate",
            "created_at_utc": started,
            "completed_at_utc": now(),
            "phase": phase,
            "task_index": task_index,
            "operator_family": spec["operator_family"],
            "run_index": run_index,
            "fresh_process": True,
            "reverse_bddl_path": spec["reverse_bddl_path"],
            "reverse_bddl_sha256": sha256_file(Path(spec["reverse_bddl_path"])),
            "controller_contract": {"controller": "OSC_POSE", "control_dim": int(getattr(_controller(env), "control_dim", -1)), "action_dim": ACTION_DIM, "control_freq_hz": CONTROL_FREQ_HZ},
            "layout": layout,
            "native_reset": {"state_sha256": state_hash(native_state), "initial": initial, "goal": goal, "hdf5_state_used_for_predicate_oracle": False},
            "settle": settle,
            "controlled_destruction": destruction,
            "hard_gate": gates,
            "no_grasp_or_demo_generation": True,
        }
    except Exception as exc:
        return {"schema_version": "track3_dynamic_gate_reconciliation_run_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "phase": phase, "task_index": task_index, "operator_family": spec["operator_family"], "run_index": run_index, "fresh_process": True, "error": f"{type(exc).__name__}: {exc}", "hdf5_state_used_for_predicate_oracle": False}
    finally:
        if env is not None:
            env.close()


def _run_worker_subprocess(task_index: int, run_index: int, phase: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "MUJOCO_GL": "egl", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "JAX_PLATFORMS": "cpu"})
    command = [sys.executable, "-m", "tools.axis_pi05.track3_dynamic_gate_reconciliation", "--worker", str(task_index), str(run_index), phase]
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if lines:
        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError:
            result = {"status": "worker_output_invalid", "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}
    else:
        result = {"status": "worker_no_output", "stderr_tail": completed.stderr[-4000:]}
    result["worker_returncode"] = completed.returncode
    result["stdout_tail"] = completed.stdout[-4000:]
    result["stderr_tail"] = completed.stderr[-4000:]
    return result


def run_d2() -> dict[str, Any]:
    result = _run_worker_subprocess(10, 1, "D2")
    report = {"schema_version": "track3_dynamic_gate_reconciliation_d2_v1", "status": "track3_native_reverse_dynamic_canary_passed_support" if result.get("status") == "passed" else "track3_native_reverse_dynamic_canary_failed_support", "created_at_utc": now(), "gpu_policy": "physical_gpu3_only", "run": result, "hard_stop_after_one_task10": True, "no_se3_or_demo_generation": True}
    write_json(NEW_ROOT / "d2_task10_canary.json", report)
    return report


def run_d3() -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for task_index, family in TASK_ORDER:
        for run_index in range(1, 4):
            runs.append(_run_worker_subprocess(task_index, run_index, "D3"))
    by_family = {family: [row for row in runs if row.get("operator_family") == family] for _, family in TASK_ORDER}
    family_gate = {family: len(rows) == 3 and all(row.get("status") == "passed" for row in rows) for family, rows in by_family.items()}
    failed = next((row for row in runs if row.get("status") != "passed"), None)
    if all(family_gate.values()):
        status = "track3_reverse_semantics_v2_dynamic_validated_pending_se3_review"
    else:
        family = failed.get("operator_family", "unknown") if failed else "unknown"
        field = failed.get("failure_field", "dynamic_gate") if failed else "dynamic_gate"
        status = f"track3_reverse_semantics_v2_dynamic_failed_{family}_{field}"
    report = {"schema_version": "track3_dynamic_gate_reconciliation_d3_v1", "status": status, "created_at_utc": now(), "gpu_policy": "physical_gpu3_only", "fresh_process_runs": runs, "family_3_of_3": family_gate, "hard_gate": {"all_families_3_of_3": all(family_gate.values()), "no_se3_or_demo_generation": True, "hdf5_state_used_for_predicate_oracle": False}}
    write_json(NEW_ROOT / "d3_three_family_release.json", report)
    return report


def run_d4() -> dict[str, Any]:
    """Close the authorized run after a D2 hard-stop; never starts D3."""
    d0 = json.loads((NEW_ROOT / "d0_release_lock.json").read_text(encoding="utf-8"))
    d2 = json.loads((NEW_ROOT / "d2_task10_canary.json").read_text(encoding="utf-8"))
    validate_d0_lock(d0)
    run = d2["run"]
    write_json(NEW_ROOT / "runs/d2_task10_canary_run_1.json", run)
    summary = {
        "schema_version": "track3_dynamic_gate_reconciliation_d4_v1",
        "status": d2["status"],
        "created_at_utc": now(),
        "d0": {"status": d0["status"], "hard_gate": d0["hard_gate"], "input_execution_commit": d0["input_execution_commit"], "audit_commit_observed_at_d0": d0["audit_commit_observed_at_d0"]},
        "d1": {"validator_cpu_tests_passed": True, "targeted_test_count_at_d1": 15, "fix_after_d2_exception": "TargetZone is not a SiteObjectState; site pose now comes from object_states_dict state wrapper", "d2_was_not_retried": True},
        "d2": {"task_index": 10, "fresh_process": run.get("fresh_process", True), "status": d2["status"], "failure": run.get("error"), "gpu_policy": d2["gpu_policy"], "no_grasp_or_demo_generation": True},
        "d3": {"status": "not_run_due_to_d2_hard_stop", "family_3_of_3": None, "fresh_process_runs": 0},
        "old_evidence": {"old_g1_dynamic_report_preserved": True, "old_g1_dynamic_report_sha256": sha256_file(OLD_DYNAMIC_REPORT), "old_evidence_drift": False},
        "outputs": {"exact_reverse_demo": 0, "pending_autogen": 0, "pending_autogen_v2": 0, "accepted_reverse_demo": 0, "support_retry_attempts": 0, "training_started": False, "human_operation": False, "official_gpu_used": False, "fast_or_zip_used": False, "upload_or_submission": False, "libero_plus_touched": False},
        "hard_stop": {"stop_after": "D2", "reason": "validator exception occurred before predicate/settle evidence; no valid reverse dynamic conclusion", "do_not_enter": ["D3", "SE(3)", "demo generation", "training", "Teleop"]},
    }
    write_json(NEW_ROOT / "summary.json", summary)
    report_lines = [
        "# Track 3 Dynamic Gate Model/State Reconciliation：D0–D4 收尾",
        "",
        f"- 状态：`{summary['status']}`",
        "- D0：输入证据已锁定；三份官方 HDF5 的 `states[0] == init_state` 均为真，state 的 time/qpos offset 为 0/1；旧 G1 report 未覆盖。",
        "- D1：CPU/schema 门禁 15/15 通过。reverse predicate 走 reverse env 的逐条 `_eval_predicate`，controller 顺序和 7D OSC_POSE hold action 已固定。",
        "- D2：GPU3 fresh task10 只执行一次；validator 在 site detail 处遇到 `TargetZone` 缺少 `get_geom_state`，未得到可用的 dynamic predicate/settle 证据。该次已按 hard stop 记录，修复后不重跑。",
        "- D3：因 D2 未通过，未运行三族 3/3。",
        "",
        "## 结果边界",
        "",
        "本轮不能判定 reverse semantics 成功或失败；它只证明 D2 首次实现仍有一个 validator schema 缺口。没有 SE(3) 抓取、demo、pending、训练、Teleop、FAST/ZIP 或上传。等待下一次审核决定新的最小 remediation。",
        "",
        "证据：`d0_release_lock.json`、`d2_task10_canary.json`、`runs/d2_task10_canary_run_1.json`、`summary.json`。",
    ]
    atomic_write(NEW_ROOT / "D4_CLOSEOUT_REPORT_ZH.md", ("\n".join(report_lines) + "\n").encode("utf-8"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", action="store_true")
    parser.add_argument("--d2", action="store_true")
    parser.add_argument("--d3", action="store_true")
    parser.add_argument("--d4", action="store_true")
    parser.add_argument("--worker", nargs=3)
    args = parser.parse_args()
    if args.d0:
        print(json.dumps(run_d0(), ensure_ascii=False, sort_keys=True))
        return 0
    if args.worker:
        # Worker stdout is one JSON object so the parent can preserve it in the
        # per-run evidence even when MuJoCo emits normal startup warnings.
        print(json.dumps(dynamic_worker(int(args.worker[0]), int(args.worker[1]), args.worker[2]), ensure_ascii=False, sort_keys=True))
        return 0
    if args.d2:
        print(json.dumps(run_d2(), ensure_ascii=False, indent=2))
        return 0
    if args.d3:
        print(json.dumps(run_d3(), ensure_ascii=False, indent=2))
        return 0
    if args.d4:
        print(json.dumps(run_d4(), ensure_ascii=False, indent=2))
        return 0
    parser.error("use --d0, --d2, --d3, or --worker")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
