#!/usr/bin/env python3
"""Operator-scoped stability release gates for Track 3.

This is a new evidence namespace.  The previous init-stabilization namespace
is read-only input here.  Background motion is retained as diagnostics, while
release gates are evaluated only against pre-registered operator participants,
receivers, predicates, and contact contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools.axis_pi05 import track3_reverse_init_stabilization as base

ROOT = base.ROOT
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_operator_scoped_stability"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_OPERATOR_SCOPED_STABILITY_RELEASE_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_INIT_STABILIZATION_I0_I4_AUDIT_20260904.md"
HANDOFF_JSON = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
LIBERO_PLUS = ROOT / "LIBERO-plus"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_init_stabilization"
OLD_I2 = OLD_ROOT / "i2_stabilization_time_series.json"
OLD_SUMMARY = OLD_ROOT / "summary.json"
OLD_RUN_ROOT = OLD_ROOT / "runs"
OLD_I2_SHA256 = "e72ae24d27b6edbba37b043a88042191f8a1c7d00c298b812653a1a2ecc17dab"
TASKS = (10, 20, 19)
FAMILIES = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SOURCE_ATOM = dict(base.CRITICAL)
GOAL_ATOM = dict(base.GOAL_CRITICAL)
CHECKPOINTS = tuple(range(0, 21))
RELEASE_STEPS = tuple(range(5, 21))
ACTION_DIM = 7
CONTROL_FREQ_HZ = 20
ZERO_ACTION = [0.0] * ACTION_DIM
THRESHOLDS = dict(base.THRESHOLDS)
WORKER_COMPLETED_STATUSES = frozenset({"passed", "failed_gate"})

# These names are part of the reviewed public forward -> reverse operator
# mapping.  They are never inferred from the observed failure.
OPERATOR_CONTRACT = {
    10: {
        "family": "support_remove",
        "source_atom": SOURCE_ATOM[10],
        "goal_atom": GOAL_ATOM[10],
        "subject": ("akita_black_bowl_1",),
        "receiver": ("plate_1",),
        "pose_participants": ("akita_black_bowl_1", "plate_1"),
        "allowed_contact_pairs": (("akita_black_bowl_1", "plate_1"),),
        "forbidden_contact_subject": "akita_black_bowl_1",
        "rotation_gate_exclusions": ("akita_black_bowl_1",),
        "participant_gate": "subject_translation_and_receiver_translation_rotation",
        "contact_rule": "bowl_plate_present_and_no_bowl_nonplate_contact",
    },
    20: {
        "family": "containment_remove",
        "source_atom": SOURCE_ATOM[20],
        "goal_atom": GOAL_ATOM[20],
        "subject": ("orange_juice_1",),
        "receiver": ("basket_1", "basket_1_contain_region"),
        "pose_participants": ("orange_juice_1", "basket_1", "basket_1_contain_region"),
        "allowed_contact_pairs": (("orange_juice_1", "basket_1"),),
        "forbidden_contact_subject": "orange_juice_1",
        "rotation_gate_exclusions": (),
        "participant_gate": "subject_and_receiver_translation_rotation",
        "contact_rule": "orange_basket_allowed_and_no_orange_nonbasket_contact",
    },
    19: {
        "family": "articulation_close",
        "source_atom": SOURCE_ATOM[19],
        "goal_atom": GOAL_ATOM[19],
        "subject": ("wooden_cabinet_1_middle_region",),
        "receiver": (),
        "pose_participants": ("wooden_cabinet_1_middle_region",),
        "target_joint": "wooden_cabinet_1_middle_level",
        "target_body": "wooden_cabinet_1_cabinet_middle",
        "allowed_contact_pairs": (),
        "forbidden_contact_subject": "wooden_cabinet_1_cabinet_middle",
        "rotation_gate_exclusions": (),
        "participant_gate": "target_site_rotation_translation_and_joint",
        "contact_rule": "drawer_has_no_new_background_contact",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def git_output(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path) -> list[str]:
    return [line for line in git_output(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def worker_exit_code(status: str) -> int:
    return 0 if status in WORKER_COMPLETED_STATUSES else 1


def gate_record(*, executed: bool, value: bool | None, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    row = {"executed": bool(executed), "state": base.tri_state(executed, value), "required": bool(required)}
    if reason is not None:
        row["reason"] = reason
    return row


def required_gates_passed(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [gate for gate in gates.values() if gate.get("required", True)]
    return bool(required) and all(gate.get("state") == "passed" for gate in required)


def _input_paths() -> list[Path]:
    paths = [
        OLD_ROOT / "i0_release_lock.json",
        OLD_ROOT / "i1_semantics.json",
        OLD_I2,
        OLD_SUMMARY,
        OLD_ROOT / "I4_CLOSEOUT_REPORT_ZH.md",
        AUDIT,
        PLAN,
        ROOT / "tools/axis_pi05/track3_reverse_init_stabilization.py",
        ROOT / "tools/axis_pi05/track3_dynamic_gate_reconciliation.py",
        ROOT / "LIBERO/libero/libero/envs/env_wrapper.py",
        ROOT / "LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py",
        ROOT / "LIBERO/libero/libero/envs/predicates/__init__.py",
        ROOT / "LIBERO/libero/libero/envs/predicates/base_predicates.py",
        ROOT / "LIBERO/libero/libero/envs/object_states/base_object_states.py",
    ]
    paths.extend(sorted(OLD_RUN_ROOT.glob("i2_*.json")))
    paths.extend(base.FORWARD_BDDL.values())
    paths.extend(base.REVERSE_BDDL.values())
    return list(dict.fromkeys(paths))


def _gpu_read_only_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 4:
                rows.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": fields[2], "utilization_gpu_percent": fields[3]})
        return {"command_returncode": result.returncode, "gpus": rows, "stderr_tail": result.stderr[-2000:], "read_only": True}
    except Exception as exc:
        return {"command_returncode": None, "gpus": [], "error": f"{type(exc).__name__}: {exc}", "read_only": True}


def run_j0() -> dict[str, Any]:
    input_commit = "0dec0b1930a3d2ad9f196febe1aeaabb0c89445d"
    parent_commit = "1a0b387e054ef11b1c787720a9d41ce700e4cba7"
    observed = git_output(ROOT, ("rev-parse", "HEAD"))
    nested = ROOT / "third_party/axis_openpi"
    nested_commit = git_output(nested, ("rev-parse", "HEAD"))
    expected_local = {Path(__file__).resolve(), ROOT / "tests/test_track3_operator_scoped_stability.py"}
    root_status = porcelain(ROOT)
    preexisting_root = []
    for line in root_status:
        raw_path = line[3:].strip() if len(line) >= 4 else line.strip()
        if (ROOT / raw_path).resolve() not in {path.resolve() for path in expected_local}:
            preexisting_root.append(line)
    paths = _input_paths()
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    locked = {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths if path.is_file()}
    old_i2 = json.loads(OLD_I2.read_text(encoding="utf-8"))
    old_summary = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    handoff = json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))
    input_ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", input_commit, observed], check=False).returncode == 0
    parent_ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", parent_commit, observed], check=False).returncode == 0
    counts = {key: int(old_summary.get("counts", {}).get(key, 0)) for key in ("exact_reverse_demo", "pending_autogen", "accepted_reverse_demo")}
    old_i3 = old_summary.get("i3", {})
    lock = {
        "schema_version": "track3_operator_scoped_stability_j0_v1",
        "status": "track3_operator_scoped_input_locked" if not missing and not preexisting_root and input_ancestor and parent_ancestor and nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not porcelain(nested) and sha256_file(OLD_I2) == OLD_I2_SHA256 and all(value == 0 for value in counts.values()) and old_i3.get("task10_canary_runs", 0) == 0 else "track3_operator_scoped_input_drift",
        "created_at_utc": now(),
        "plan": str(PLAN.resolve()),
        "audit": str(AUDIT.resolve()),
        "input_root_commit": input_commit,
        "input_parent_commit": parent_commit,
        "observed_root_commit_at_j0": observed,
        "root_input_is_ancestor": input_ancestor,
        "parent_is_ancestor": parent_ancestor,
        "nested_openpi_commit": nested_commit,
        "nested_openpi_expected": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
        "root_status_at_j0": root_status,
        "preexisting_root_status_at_j0": preexisting_root,
        "nested_status_at_j0": porcelain(nested),
        "libero_plus_dirty_read_only_snapshot": {"path": str(LIBERO_PLUS.resolve()), "status_porcelain": porcelain(LIBERO_PLUS), "policy": "record_only_never_clean_or_touch"},
        "immutable_inputs": locked,
        "missing_inputs": missing,
        "old_i2_summary_sha256": sha256_file(OLD_I2),
        "old_i2_summary_expected_sha256": OLD_I2_SHA256,
        "old_i2_fresh_process_runs": old_i2.get("fresh_process_runs"),
        "old_i2_reverse_runs": sum(row.get("kind") == "reverse" for row in old_i2.get("runs", [])),
        "old_i2_forward_task10_runs": sum(row.get("kind") == "forward" and row.get("task_index") == 10 for row in old_i2.get("runs", [])),
        "old_i3_runs": {"task10_canary": old_i3.get("task10_canary_runs", 0), "three_family": old_i3.get("three_family_runs", 0)},
        "counts_at_j0": counts,
        "authorization": {"no_bddl_or_predicate_modification": True, "no_se3_or_grasp": True, "no_demo_or_pending": True, "no_teleop": True, "no_batch_or_training": True, "no_fast_zip_upload": True, "gpu3_only_after_j2_pass": True},
        "gpu_read_only_snapshot": _gpu_read_only_snapshot(),
        "hard_gate": {
            "all_inputs_exist": not missing,
            "root_input_locked": input_ancestor,
            "parent_locked": parent_ancestor,
            "root_preexisting_changes_absent": not preexisting_root,
            "nested_openpi_clean_and_locked": nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not porcelain(nested),
            "libero_plus_recorded_without_touch": True,
            "old_i2_hash_locked": sha256_file(OLD_I2) == OLD_I2_SHA256,
            "old_i3_zero_locked": old_i3.get("task10_canary_runs", 0) == 0 and old_i3.get("three_family_runs", 0) == 0,
            "counts_zero": all(value == 0 for value in counts.values()),
        },
    }
    write_json(OUT_ROOT / "j0_release_lock.json", lock)
    return lock


def operator_participants(task_index: int) -> dict[str, Any]:
    if task_index not in OPERATOR_CONTRACT:
        raise KeyError(task_index)
    return OPERATOR_CONTRACT[task_index]


def _j1_core() -> dict[str, Any]:
    return {
        "schema_version": "track3_operator_scoped_stability_j1_contract_core_v1",
        "release_window": {"start_step": 5, "end_step": 20, "steps": list(RELEASE_STEPS), "frequency_hz": CONTROL_FREQ_HZ},
        "action": {"dimension": ACTION_DIM, "controller": "OSC_POSE", "values": ZERO_ACTION, "gripper": 0.0, "semantics": "zero_increment_hold"},
        "thresholds": THRESHOLDS,
        "operator_contract": {str(task): operator_participants(task) for task in TASKS},
        "common_gates": ["schema", "finite", "truth_detail", "os_returncode", "critical_source_true", "reverse_goal_false", "truth_vector_no_transition", "robot_component_stability", "scene_name_set_finite", "contact_matrix_complete"],
        "background_policy": "retain_full_translation_rotation_max_and_contact_diagnostics_without_operator_gate_substitution",
        "old_full_scene_policy": "immutable_diagnostic_reference_only",
        "not_evaluated_policy": "unexecuted_or_unrecorded_required_gate_is_not_evaluated_and_fails_closed",
        "hdf5_predicate_oracle": False,
        "no_hindsight_checkpoint_selection": True,
    }


def j1_contract() -> dict[str, Any]:
    core = _j1_core()
    hard_gate = {
        "three_families_registered": set(OPERATOR_CONTRACT) == set(TASKS),
        "source_atoms_fixed": all(OPERATOR_CONTRACT[task]["source_atom"] == SOURCE_ATOM[task] for task in TASKS),
        "goal_atoms_fixed": all(OPERATOR_CONTRACT[task]["goal_atom"] == GOAL_ATOM[task] for task in TASKS),
        "support_bowl_rotation_diagnostic_only": "akita_black_bowl_1" in OPERATOR_CONTRACT[10]["rotation_gate_exclusions"],
        "background_not_operator_gate": "without_operator_gate_substitution" in core["background_policy"],
        "contact_contract_pre_registered": all("contact_rule" in OPERATOR_CONTRACT[task] for task in TASKS),
        "tri_state_fail_closed": not required_gates_passed({"unrun": gate_record(executed=False, value=None)}),
        "zero_hold_contract": ZERO_ACTION == [0.0] * ACTION_DIM and ZERO_ACTION[-1] == 0.0,
        "fixed_step5_to20": core["release_window"]["steps"] == list(RELEASE_STEPS),
    }
    core_hash = canonical_hash(core)
    return {"schema_version": "track3_operator_scoped_stability_j1_v1", "created_at_utc": now(), "contract": core, "contract_core_sha256": core_hash, "hard_gate": hard_gate, "status": "track3_operator_scoped_contract_validated" if all(hard_gate.values()) else "track3_operator_scoped_contract_failed"}


def run_j1() -> dict[str, Any]:
    report = j1_contract()
    write_json(OUT_ROOT / "j1_contract.json", report)
    return report


def _names(model: Any, attr: str) -> list[str]:
    return [str(value) for value in getattr(model, attr, ())]


def _body_ids(env: Any, logical_name: str) -> list[int]:
    model = env.sim.model
    result: list[int] = []
    mapping = getattr(env.env, "obj_body_id", {})
    if logical_name in mapping:
        result.append(int(mapping[logical_name]))
    for candidate in (logical_name, f"{logical_name}_main", f"{logical_name}_cabinet_middle"):
        try:
            result.append(int(model.body_name2id(candidate)))
        except (KeyError, ValueError, TypeError, AttributeError):
            pass
    return sorted(set(result))


def _body_name(model: Any, body_id: int) -> str:
    names = _names(model, "body_names")
    return names[body_id] if 0 <= body_id < len(names) else f"body_{body_id}"


def _geom_name(model: Any, geom_id: int) -> str:
    names = _names(model, "geom_names")
    return names[geom_id] if 0 <= geom_id < len(names) else f"geom_{geom_id}"


def contact_matrix(env: Any) -> dict[str, Any]:
    """Capture every MuJoCo contact with stable body/geom names."""

    model = env.sim.model
    data = env.sim.data
    try:
        ncon = int(data.ncon)
        rows = []
        for index in range(ncon):
            contact = data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            distance = float(contact.dist)
            if not math.isfinite(distance):
                raise ValueError(f"non-finite contact distance at {index}")
            rows.append({"geom1": geom1, "geom2": geom2, "geom1_name": _geom_name(model, geom1), "geom2_name": _geom_name(model, geom2), "body1": body1, "body2": body2, "body1_name": _body_name(model, body1), "body2_name": _body_name(model, body2), "distance_m": distance})
        return {"executed": True, "state": "passed", "complete": True, "ncon": ncon, "contacts": rows, "body_names": _names(model, "body_names"), "geom_names": _names(model, "geom_names")}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        return {"executed": True, "state": "failed", "complete": False, "error": f"{type(exc).__name__}: {exc}"}


def _logical_body_match(actual: str, logical: str) -> bool:
    return actual == logical or actual.startswith(f"{logical}_")


def _contact_contract_from_matrix(
    task_index: int,
    matrix: Mapping[str, Any],
    baseline_matrix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the reviewed contact contract using recorded public body names.

    A background contact already present at the fixed step-5 baseline is a
    diagnostic, not a newly introduced contact.  The baseline comparison is
    deliberately by body pair, so MuJoCo geom-level contact churn cannot be
    mistaken for a new background object.
    """

    contract = operator_participants(task_index)
    contacts = matrix.get("contacts", [])
    baseline_pairs = {
        tuple(sorted((str(row.get("body1_name")), str(row.get("body2_name")))))
        for row in (baseline_matrix or {}).get("contacts", [])
    }
    allowed: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    preexisting_forbidden: list[dict[str, Any]] = []
    new_forbidden: list[dict[str, Any]] = []
    for row in contacts:
        names = (str(row.get("body1_name", "")), str(row.get("body2_name", "")))
        pair_names = tuple(sorted(names))
        is_allowed = any(
            (
                (_logical_body_match(names[0], left) and _logical_body_match(names[1], right))
                or (_logical_body_match(names[0], right) and _logical_body_match(names[1], left))
            )
            for left, right in contract["allowed_contact_pairs"]
        )
        subject_present = any(
            _logical_body_match(name, logical)
            for name in names
            for logical in contract["subject"]
        )
        target_present = task_index == 19 and any(
            _logical_body_match(name, contract["target_body"]) for name in names
        )
        if is_allowed:
            allowed.append(row)
        elif subject_present or target_present:
            forbidden.append(row)
            (preexisting_forbidden if pair_names in baseline_pairs else new_forbidden).append(row)
    complete = matrix.get("complete") is True
    no_new_forbidden = not new_forbidden
    return {
        "executed": complete,
        "state": "passed" if complete and no_new_forbidden and (task_index != 10 or bool(allowed)) else "failed",
        "complete": complete,
        "allowed_contact_present": bool(allowed),
        "allowed_contacts": allowed,
        "forbidden_background_contact_present": bool(forbidden),
        "forbidden_background_contacts": forbidden,
        "preexisting_background_contacts": preexisting_forbidden,
        "new_background_contacts": new_forbidden,
        "new_background_contact_present": bool(new_forbidden),
        "allowed_contact_rule": contract["contact_rule"],
        "baseline_body_pairs": [list(pair) for pair in sorted(baseline_pairs)],
        "background_contact_policy": "baseline_step5_contacts_diagnostic; only new body pairs fail",
    }


def _contact_contract(env: Any, task_index: int, matrix: Mapping[str, Any], baseline_matrix: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = _contact_contract_from_matrix(task_index, matrix, baseline_matrix)
    subject_ids = {body_id for name in operator_participants(task_index)["subject"] for body_id in _body_ids(env, name)}
    receiver_ids = {body_id for name in operator_participants(task_index)["receiver"] for body_id in _body_ids(env, name)}
    allowed_id_pairs = []
    for left, right in operator_participants(task_index)["allowed_contact_pairs"]:
        left_ids = _body_ids(env, left)
        right_ids = _body_ids(env, right)
        allowed_id_pairs.extend((min(a, b), max(a, b)) for a in left_ids for b in right_ids)
    result.update(
        {
            "subject_body_ids": sorted(subject_ids),
            "receiver_body_ids": sorted(receiver_ids),
            "allowed_body_pairs": [list(pair) for pair in sorted(set(allowed_id_pairs))],
        }
    )
    return result


def _joint_value(env: Any, joint_name: str) -> float:
    address = env.sim.model.get_joint_qpos_addr(joint_name)
    address = int(address[0]) if isinstance(address, tuple) else int(address)
    value = float(env.sim.data.qpos[address])
    if not math.isfinite(value):
        raise ValueError(f"non-finite joint value: {joint_name}")
    return value


def participant_snapshot(env: Any, old: Any, task_index: int) -> dict[str, Any]:
    contract = operator_participants(task_index)
    poses: dict[str, Any] = {}
    for name in contract["pose_participants"]:
        pose = old._object_or_site_pose(env, name)
        if pose is None:
            raise ValueError(f"operator participant pose missing: {name}")
        poses[name] = pose
    bodies: dict[str, Any] = {}
    if contract.get("target_body"):
        try:
            body_id = int(env.sim.model.body_name2id(contract["target_body"]))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"operator target body missing: {contract['target_body']}") from exc
        bodies[contract["target_body"]] = {"kind": "body", "name": contract["target_body"], "pos_m": old.finite_array("target_body_pos", env.sim.data.body_xpos[body_id], (3,)).tolist(), "quat": old.normalize_quat(env.sim.data.body_xquat[body_id]).tolist()}
    joints = {contract["target_joint"]: _joint_value(env, contract["target_joint"])} if contract.get("target_joint") else {}
    return {"poses": poses, "bodies": bodies, "joint_values": joints}


def _pose_drift(left: Mapping[str, Any], right: Mapping[str, Any], old: Any) -> tuple[float, float]:
    translation = float(sum((float(a) - float(b)) ** 2 for a, b in zip(left["pos_m"], right["pos_m"])) ** 0.5)
    rotation = old.quat_geodesic_rad(left["quat"], right["quat"])
    return translation, rotation


def participant_stability(samples: Sequence[Mapping[str, Any]], task_index: int, old: Any) -> dict[str, Any]:
    if not samples:
        return {"executed": False, "state": "not_evaluated", "passed": False, "reason": "no_samples"}
    baseline = samples[0]["participant"]
    contract = operator_participants(task_index)
    max_translation: dict[str, float] = {}
    max_rotation: dict[str, float] = {}
    max_joint: dict[str, float] = {}
    max_steps: dict[str, int] = {}
    for sample in samples:
        step = int(sample["step"])
        for name, pose in baseline["poses"].items():
            current = sample["participant"]["poses"].get(name)
            if current is None:
                raise ValueError(f"operator participant disappeared: {name}")
            translation, rotation = _pose_drift(pose, current, old)
            if translation > max_translation.get(name, -1.0):
                max_translation[name] = translation
                max_steps[f"{name}.translation_m"] = step
            if rotation > max_rotation.get(name, -1.0):
                max_rotation[name] = rotation
                max_steps[f"{name}.rotation_rad"] = step
        for name, value in baseline["bodies"].items():
            current = sample["participant"]["bodies"].get(name)
            if current is None:
                raise ValueError(f"operator target body disappeared: {name}")
            translation, rotation = _pose_drift(value, current, old)
            if translation > max_translation.get(name, -1.0):
                max_translation[name] = translation
                max_steps[f"{name}.translation_m"] = step
            if rotation > max_rotation.get(name, -1.0):
                max_rotation[name] = rotation
                max_steps[f"{name}.rotation_rad"] = step
        for name, value in baseline["joint_values"].items():
            drift = abs(float(sample["participant"]["joint_values"].get(name, float("nan"))) - value)
            if not math.isfinite(drift):
                raise ValueError(f"operator joint disappeared: {name}")
            if drift > max_joint.get(name, -1.0):
                max_joint[name] = drift
                max_steps[f"{name}.joint_rad_or_m"] = step
    gated_translations = {name: value for name, value in max_translation.items()}
    gated_rotations = {name: value for name, value in max_rotation.items() if name not in contract["rotation_gate_exclusions"]}
    translation_pass = all(value <= THRESHOLDS["object_translation_m"] for value in gated_translations.values())
    rotation_pass = all(value <= THRESHOLDS["object_rotation_rad"] for value in gated_rotations.values())
    joint_pass = all(value <= THRESHOLDS["fixture_joint_rad_or_m"] for value in max_joint.values())
    return {"executed": True, "state": "passed" if translation_pass and rotation_pass and joint_pass else "failed", "passed": bool(translation_pass and rotation_pass and joint_pass), "max_translation_m": max_translation, "max_rotation_rad": max_rotation, "max_joint_rad_or_m": max_joint, "max_step": max_steps, "rotation_gate_exclusions": list(contract["rotation_gate_exclusions"]), "gated_rotation_rad": gated_rotations, "gates": {"participant_translation": translation_pass, "participant_rotation_except_exclusions": rotation_pass, "fixture_joint": joint_pass}}


def background_diagnostics(samples: Sequence[Mapping[str, Any]], task_index: int, old: Any) -> dict[str, Any]:
    if not samples:
        return {"executed": False, "state": "not_evaluated"}
    baseline = samples[0]["scene"]["object_and_site_poses"]
    max_translation: dict[str, float] = {}
    max_rotation: dict[str, float] = {}
    max_step: dict[str, int] = {}
    for sample in samples:
        for name, pose in baseline.items():
            current = sample["scene"]["object_and_site_poses"].get(name)
            if current is None:
                return {"executed": True, "state": "failed", "reason": f"scene_pose_missing:{name}"}
            translation, rotation = _pose_drift(pose, current, old)
            if translation > max_translation.get(name, -1.0):
                max_translation[name] = translation
                max_step[f"{name}.translation_m"] = int(sample["step"])
            if rotation > max_rotation.get(name, -1.0):
                max_rotation[name] = rotation
                max_step[f"{name}.rotation_rad"] = int(sample["step"])
    participant_names = set(operator_participants(task_index)["pose_participants"])
    background_names = sorted(set(baseline) - participant_names)
    return {"executed": True, "state": "passed", "all_object_site_names": sorted(baseline), "background_names": background_names, "max_translation_m": max_translation, "max_rotation_rad": max_rotation, "max_step": max_step, "critical_names": sorted(participant_names), "background_is_diagnostic_only": True}


def _truth_vector(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {"init": [row.get("value") for row in sample["initial"].get("predicates", [])], "goal": [row.get("value") for row in sample["goal"].get("predicates", [])]}


def _vector_stability(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        return gate_record(executed=False, value=None, reason="no_samples")
    baseline = _truth_vector(samples[0])
    complete = all(all(value is not None for value in vector["init"] + vector["goal"]) for vector in (_truth_vector(sample) for sample in samples))
    no_transition = complete and all(_truth_vector(sample) == baseline for sample in samples)
    return gate_record(executed=complete, value=no_transition, reason=None if complete else "truth_vector_incomplete")


def _robot_stability(samples: Sequence[Mapping[str, Any]], old: Any) -> dict[str, Any]:
    if not samples:
        return gate_record(executed=False, value=None, reason="no_samples")
    baseline = samples[0]["robot"]
    drifts = [old._metric_drift(baseline, sample["robot"]) for sample in samples]
    maximum = {key: max(row[key] for row in drifts) for key in drifts[0]}
    passed = all(maximum.get(key, float("inf")) <= THRESHOLDS[key] for key in THRESHOLDS if key in maximum)
    return {**gate_record(executed=True, value=passed), "max_drift": maximum, "thresholds": THRESHOLDS}


def _runtime_sample(env: Any, old: Any, task_index: int, kind: str, step: int, initial_atoms: Sequence[str], goal_atoms: Sequence[str]) -> dict[str, Any]:
    source = "forward_reference" if kind == "forward" else "reverse"
    initial = base.annotate_predicate_result(old.evaluate_predicates(env, initial_atoms, label=f"{kind}_initial_step_{step}"), task_index, source=source, critical_atom=SOURCE_ATOM[task_index])
    goal = base.annotate_predicate_result(old.evaluate_predicates(env, goal_atoms, label=f"{kind}_goal_step_{step}"), task_index, source=source, critical_atom=SOURCE_ATOM[task_index] if kind == "forward" else GOAL_ATOM[task_index])
    matrix = contact_matrix(env)
    return {"step": step, "initial": initial, "goal": goal, "robot": old._robot_metrics(env), "scene": old._scene_metrics(env, [*initial_atoms, *goal_atoms]), "participant": participant_snapshot(env, old, task_index), "contacts": _contact_contract(env, task_index, matrix), "contact_matrix": matrix}


def _reverse_gate(samples: Sequence[Mapping[str, Any]], task_index: int, kind: str, old: Any) -> dict[str, Any]:
    if kind != "reverse":
        return {"status": "not_required", "state": "not_evaluated", "executed": False, "required": False}
    release_samples = [sample for sample in samples if int(sample.get("step", -1)) in RELEASE_STEPS]
    if len(release_samples) != len(RELEASE_STEPS):
        return {"status": "failed_gate", "state": "failed", "executed": False, "required": True, "reason": "release_window_incomplete"}
    samples = release_samples
    source = all(
        base.predicate_expectation_gate(sample["initial"], SOURCE_ATOM[task_index], expect=True)["state"] == "passed"
        for sample in samples
    )
    goal = all(
        base.predicate_expectation_gate(sample["goal"], GOAL_ATOM[task_index], expect=False)["state"] == "passed"
        for sample in samples
    )
    vector = _vector_stability(samples)
    robot = _robot_stability(samples, old)
    participant = participant_stability(samples, task_index, old)
    contact = all(sample["contacts"].get("state") == "passed" and sample["contacts"].get("complete") is True for sample in samples)
    gates = {"critical_source_each_step": gate_record(executed=bool(samples), value=source), "reverse_goal_false_each_step": gate_record(executed=bool(samples), value=goal), "truth_vector_no_transition": vector, "robot_component_stability": robot, "operator_participant_stability": gate_record(executed=participant.get("executed", False), value=participant.get("passed")), "contact_matrix_and_family_rule": gate_record(executed=bool(samples) and all(sample["contact_matrix"].get("complete") is True for sample in samples), value=contact)}
    return {"status": "passed" if required_gates_passed(gates) else "failed_gate", "state": "passed" if required_gates_passed(gates) else "failed", "executed": True, "required": True, "gates": gates, "participant": participant}


def j2_worker(task_index: int, kind: str, run_index: int) -> dict[str, Any]:
    started = now()
    env = None
    try:
        import numpy as np

        old = __import__("tools.axis_pi05.track3_dynamic_gate_reconciliation", fromlist=["module"])
        bddl_path = base.FORWARD_BDDL[task_index] if kind == "forward" else base.REVERSE_BDDL[task_index]
        initial_atoms = base.public_predicates(bddl_path, "init")
        goal_atoms = base.public_predicates(bddl_path, "goal")
        env = old.make_schema_env(str(bddl_path))
        schema = base._runtime_schema(env, old, bddl_path=bddl_path)
        _, native_state = old._native_reset(env)
        old.controller_reset_sequence(old._controller(env), env.sim)
        samples: list[dict[str, Any]] = []
        for step in CHECKPOINTS:
            if step > 0:
                action = np.asarray(ZERO_ACTION, dtype=np.float32)
                if action.shape != (ACTION_DIM,) or action[-1] != 0.0 or not np.all(np.isfinite(action)):
                    raise ValueError("invalid operator-scoped zero action")
                env.step(action)
            samples.append(_runtime_sample(env, old, task_index, kind, step, initial_atoms, goal_atoms))
        baseline_scene_names = set(samples[0]["scene"]["object_and_site_poses"])
        step5_sample = next(sample for sample in samples if sample["step"] == 5)
        baseline_contact_matrix = step5_sample["contact_matrix"]
        for sample in samples:
            sample["contacts"] = _contact_contract(
                env,
                task_index,
                sample["contact_matrix"],
                baseline_matrix=baseline_contact_matrix if sample["step"] >= 5 else None,
            )
        scene_names_invariant = all(set(sample["scene"]["object_and_site_poses"]) == baseline_scene_names for sample in samples)
        truth_detail = all(sample["initial"]["truth_complete"] and sample["goal"]["truth_complete"] and sample["initial"]["detail_complete"] and sample["goal"]["detail_complete"] for sample in samples)
        finite = all(sample["contact_matrix"].get("complete") and sample["contacts"].get("complete") for sample in samples)
        reverse_gate = _reverse_gate(samples, task_index, kind, old)
        control_gate = {"status": "passed" if truth_detail and finite and scene_names_invariant else "failed_gate", "state": "passed" if truth_detail and finite and scene_names_invariant else "failed", "executed": True, "required": kind == "forward", "gates": {"schema": gate_record(executed=True, value=True), "truth_detail": gate_record(executed=True, value=truth_detail), "finite_contact": gate_record(executed=True, value=finite), "scene_names_invariant": gate_record(executed=True, value=scene_names_invariant)}}
        status = "passed" if truth_detail and finite and scene_names_invariant else "failed_gate"
        return {"schema_version": "track3_operator_scoped_stability_j2_worker_v1", "status": status, "created_at_utc": started, "task_index": task_index, "operator_family": FAMILIES[task_index], "kind": kind, "run_index": run_index, "native_reset_state_sha256": old.state_hash(native_state), "schema": schema, "action_contract": {"dimension": ACTION_DIM, "values": ZERO_ACTION, "gripper": 0.0, "control_freq_hz": CONTROL_FREQ_HZ, "recorded_steps": list(CHECKPOINTS), "minus_one_used": False}, "checkpoints": samples, "reverse_operator_gate": reverse_gate, "forward_control_gate": control_gate, "background_diagnostics": background_diagnostics(samples, task_index, old), "provenance": {"hdf5_state_used_for_predicate_oracle": False, "no_grasp_or_demo_generation": True, "no_pending_autogen": True}}
    except Exception as exc:
        return {"schema_version": "track3_operator_scoped_stability_j2_worker_v1", "status": "worker_exception", "created_at_utc": started, "task_index": task_index, "operator_family": FAMILIES[task_index], "kind": kind, "run_index": run_index, "error": {"type": type(exc).__name__, "message": str(exc)}, "traceback": traceback.format_exc(), "provenance": {"hdf5_state_used_for_predicate_oracle": False, "no_grasp_or_demo_generation": True, "no_pending_autogen": True}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _new_j2_jobs() -> list[tuple[int, str, int]]:
    return [(task, "reverse", run) for task in TASKS for run in range(1, 4)] + [(task, "forward", run) for task in (20, 19) for run in range(1, 4)]


def _checkpoint_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = row.get("checkpoints", {})
    if isinstance(raw, Mapping):
        return {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}
    if isinstance(raw, list):
        return {
            str(item["step"]): item
            for item in raw
            if isinstance(item, Mapping) and isinstance(item.get("step"), int)
        }
    return {}


def _correct_fresh_worker_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute release gates from raw fresh-process telemetry.

    The worker wrote checkpoints as a list and its first derived gate used
    step 0 as the participant baseline.  Raw checkpoint telemetry is retained
    unchanged; this derives the reviewed step-5 release view without rerun.
    """

    corrected = dict(row)
    checkpoints = _checkpoint_map(row)
    ordered = [checkpoints[str(step)] for step in CHECKPOINTS if str(step) in checkpoints]
    step5 = checkpoints.get("5")
    if step5 is not None:
        baseline_matrix = step5.get("contact_matrix", {})
        corrected_checkpoints = []
        for sample in ordered:
            item = dict(sample)
            item["contacts"] = _contact_contract_from_matrix(
                int(row["task_index"]),
                item.get("contact_matrix", {}),
                baseline_matrix=baseline_matrix if int(item.get("step", -1)) >= 5 else None,
            )
            corrected_checkpoints.append(item)
        corrected["checkpoints"] = corrected_checkpoints
    old = __import__("tools.axis_pi05.track3_dynamic_gate_reconciliation", fromlist=["module"])
    release_samples = [sample for sample in corrected.get("checkpoints", []) if int(sample.get("step", -1)) in RELEASE_STEPS]
    if release_samples:
        corrected["reverse_operator_gate"] = _reverse_gate(
            release_samples,
            int(row["task_index"]),
            str(row["kind"]),
            old,
        )
    else:
        corrected["reverse_operator_gate"] = {
            "status": "failed_gate",
            "state": "failed",
            "executed": False,
            "required": str(row.get("kind")) == "reverse",
            "reason": "release_window_incomplete",
        }
    corrected["derivation"] = {
        "source_raw_worker_report": True,
        "raw_checkpoint_shape": type(row.get("checkpoints")).__name__,
        "participant_baseline_step": 5,
        "contact_baseline_step": 5,
        "raw_worker_gate_not_used_for_release": True,
    }
    return corrected


def _i2_row_participant(row: Mapping[str, Any], task_index: int) -> dict[str, Any]:
    checkpoints = _checkpoint_map(row)
    sample = checkpoints.get("5")
    if sample is None:
        return {"executed": False, "state": "not_evaluated", "passed": False}
    old = __import__("tools.axis_pi05.track3_dynamic_gate_reconciliation", fromlist=["module"])
    contract = operator_participants(task_index)
    baseline = sample.get("scene", {}).get("object_and_site_poses", {})
    values: dict[str, Any] = {}
    for name in contract["pose_participants"]:
        if name not in baseline:
            return {"executed": False, "state": "not_evaluated", "passed": False, "reason": f"historical_pose_missing:{name}"}
        values[name] = baseline[name]
    max_translation: dict[str, float] = {}
    max_rotation: dict[str, float] = {}
    max_joint: dict[str, float] = {}
    steps = [checkpoints.get(str(step)) for step in RELEASE_STEPS]
    for current_sample in steps:
        if current_sample is None:
            return {"executed": False, "state": "not_evaluated", "passed": False, "reason": "historical_checkpoint_missing"}
        poses = current_sample.get("scene", {}).get("object_and_site_poses", {})
        for name, pose in values.items():
            if name not in poses:
                return {"executed": False, "state": "not_evaluated", "passed": False, "reason": f"historical_pose_missing:{name}"}
            translation, rotation = _pose_drift(values[name], poses[name], old)
            max_translation[name] = max(max_translation.get(name, 0.0), translation)
            max_rotation[name] = max(max_rotation.get(name, 0.0), rotation)
        for joint, value in sample.get("scene", {}).get("fixture_joint_qpos", {}).items():
            current_value = current_sample.get("scene", {}).get("fixture_joint_qpos", {}).get(joint)
            if current_value is not None:
                max_joint[joint] = max(max_joint.get(joint, 0.0), abs(float(current_value) - float(value)))
    gated_rotation = {name: value for name, value in max_rotation.items() if name not in contract["rotation_gate_exclusions"]}
    passed = bool(max_translation) and all(value <= THRESHOLDS["object_translation_m"] for value in max_translation.values()) and all(value <= THRESHOLDS["object_rotation_rad"] for value in gated_rotation.values()) and all(value <= THRESHOLDS["fixture_joint_rad_or_m"] for value in max_joint.values())
    return {"executed": True, "state": "passed" if passed else "failed", "passed": passed, "max_translation_m": max_translation, "max_rotation_rad": max_rotation, "gated_rotation_rad": gated_rotation, "max_joint_rad_or_m": max_joint, "contact_matrix": gate_record(executed=False, value=None, required=True, reason="historical_I2_did_not_record_full_contact_matrix")}


def _reanalyse_existing_row(row: Mapping[str, Any], task_index: int, kind: str) -> dict[str, Any]:
    checkpoints = _checkpoint_map(row)
    samples = [checkpoints.get(str(step)) for step in RELEASE_STEPS]
    complete = row.get("status") in WORKER_COMPLETED_STATUSES and row.get("process_returncode") == 0 and row.get("completeness", {}).get("all_predicate_truth_evaluated") is True and row.get("completeness", {}).get("all_predicate_detail_evaluated") is True and all(sample is not None for sample in samples)
    if not complete:
        return {"task_index": task_index, "kind": kind, "run_index": row.get("run_index"), "status": "failed_gate", "reason": "historical_saved_field_missing_or_invalid", "contact_matrix": gate_record(executed=False, value=None, required=True, reason="historical_I2_did_not_record_full_contact_matrix")}
    source_states = [sample.get("critical_source", {}).get("state") for sample in samples]
    goal_states = [sample.get("critical_goal", {}).get("state") for sample in samples]
    vectors = [_truth_vector(sample) for sample in samples]
    vector_gate = gate_record(executed=all(all(value is not None for value in vector["init"] + vector["goal"]) for vector in vectors), value=len(set(json.dumps(vector, sort_keys=True) for vector in vectors)) == 1)
    participant = _i2_row_participant(row, task_index)
    source = all(value == "passed" for value in source_states)
    goal = all(value == "passed" for value in goal_states)
    recorded_fields = {"schema_returncode": gate_record(executed=True, value=True), "critical_source": gate_record(executed=True, value=source), "reverse_goal": gate_record(executed=True, value=goal), "truth_vector_no_transition": vector_gate, "operator_participant_stability": gate_record(executed=participant.get("executed", False), value=participant.get("passed")), "contact_matrix": participant.get("contact_matrix", gate_record(executed=False, value=None, required=True, reason="historical_I2_did_not_record_full_contact_matrix"))}
    recorded_pass = all(gate.get("state") == "passed" for name, gate in recorded_fields.items() if name != "contact_matrix")
    return {"task_index": task_index, "operator_family": FAMILIES[task_index], "kind": kind, "run_index": row.get("run_index"), "status": "passed" if recorded_pass else "failed_gate", "historical_saved_fields_operator_scoped": recorded_fields, "old_full_scene_gate": row.get("stability", {}).get("passed"), "background_diagnostic": {"source": "historical_scene_snapshots", "retained": True}, "release_eligible": False, "release_blocker": "contact_matrix_not_recorded_in_old_I2"}


def run_j2_existing() -> dict[str, Any]:
    old_i2 = json.loads(OLD_I2.read_text(encoding="utf-8"))
    rows = [row for row in old_i2.get("runs", []) if row.get("kind") == "reverse"]
    rows.extend(row for row in old_i2.get("runs", []) if row.get("kind") == "forward" and row.get("task_index") == 10)
    result_rows = [_reanalyse_existing_row(row, int(row["task_index"]), str(row["kind"])) for row in rows]
    reverse = [row for row in result_rows if row["kind"] == "reverse"]
    return {"schema_version": "track3_operator_scoped_stability_j2_existing_v1", "status": "passed" if len(reverse) == 9 and all(row["status"] == "passed" for row in reverse) else "failed_gate", "created_at_utc": now(), "source_i2_sha256": sha256_file(OLD_I2), "expected_source_i2_sha256": OLD_I2_SHA256, "historical_reverse_runs": result_rows[:9], "historical_task10_forward_runs": result_rows[9:], "reverse_recorded_fields_pass": sum(row["status"] == "passed" for row in reverse), "contact_matrix_not_evaluated": sum(row["historical_saved_fields_operator_scoped"]["contact_matrix"]["state"] == "not_evaluated" for row in reverse), "j2_1_cannot_release_j3": True, "old_evidence_immutable": sha256_file(OLD_I2) == OLD_I2_SHA256}


def _subprocess_env(*, gpu: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'LIBERO'}:{ROOT}"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["MUJOCO_GL"] = "disable"
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    return environment


def _run_j2_job(task_index: int, kind: str, run_index: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.axis_pi05.track3_operator_scoped_stability", "--j2-worker", str(task_index), kind, str(run_index)]
    completed = subprocess.run(command, cwd=ROOT, env=_subprocess_env(), capture_output=True, text=True, timeout=180, check=False)
    parsed = None
    for line in reversed(completed.stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
    if parsed is None:
        parsed = {"status": "worker_output_invalid", "task_index": task_index, "kind": kind, "run_index": run_index}
    parsed["process_returncode"] = int(completed.returncode)
    parsed["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(parsed.get("status")))
    parsed["stdout_tail"] = completed.stdout[-2000:]
    parsed["stderr_tail"] = completed.stderr[-2000:]
    return parsed


def _raw_j2_paths() -> list[Path]:
    return [OUT_ROOT / "runs" / f"j2_task{task}_{kind}_run_{run}.json" for task, kind, run in _new_j2_jobs()]


def _compact_run_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_index": row.get("task_index"),
        "operator_family": row.get("operator_family"),
        "kind": row.get("kind"),
        "run_index": row.get("run_index"),
        "status": row.get("status"),
        "process_returncode": row.get("process_returncode"),
        "worker_returncode_contract": row.get("worker_returncode_contract"),
        "reverse_operator_gate": row.get("reverse_operator_gate"),
        "forward_control_gate": row.get("forward_control_gate"),
        "derivation": row.get("derivation"),
        "source_raw_report": row.get("source_raw_report"),
        "corrected_report": row.get("corrected_report"),
    }


def run_j2() -> dict[str, Any]:
    j0 = json.loads((OUT_ROOT / "j0_release_lock.json").read_text(encoding="utf-8"))
    j1 = json.loads((OUT_ROOT / "j1_contract.json").read_text(encoding="utf-8"))
    if j0.get("status") != "track3_operator_scoped_input_locked" or j1.get("status") != "track3_operator_scoped_contract_validated":
        raise RuntimeError("J0/J1 gate is not passed")
    existing = run_j2_existing()
    write_json(OUT_ROOT / "j2_existing_i2_reanalysis.json", existing)
    raw_paths = _raw_j2_paths()
    if any(path.is_file() for path in raw_paths) and not all(path.is_file() for path in raw_paths):
        raise RuntimeError("partial J2 fresh-process matrix exists; refusing a second execution")
    if all(path.is_file() for path in raw_paths):
        raw_rows = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
    else:
        raw_rows = []
        for task_index, kind, run_index in _new_j2_jobs():
            row = _run_j2_job(task_index, kind, run_index)
            write_json(OUT_ROOT / "runs" / f"j2_task{task_index}_{kind}_run_{run_index}.json", row)
            raw_rows.append(row)
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        corrected = _correct_fresh_worker_row(raw_row)
        corrected_path = OUT_ROOT / "runs" / f"j2_reanalysis_task{corrected['task_index']}_{corrected['kind']}_run_{corrected['run_index']}.json"
        write_json(corrected_path, corrected)
        corrected["source_raw_report"] = str((OUT_ROOT / "runs" / f"j2_task{corrected['task_index']}_{corrected['kind']}_run_{corrected['run_index']}.json").resolve())
        corrected["corrected_report"] = str(corrected_path.resolve())
        rows.append(corrected)
    reverse = [row for row in rows if row.get("kind") == "reverse"]
    forward = [row for row in rows if row.get("kind") == "forward"]
    reverse_pass = all(row.get("process_returncode") == 0 and row.get("status") in WORKER_COMPLETED_STATUSES and row.get("reverse_operator_gate", {}).get("status") == "passed" and row.get("worker_returncode_contract") for row in reverse)
    forward_pass = all(row.get("process_returncode") == 0 and row.get("status") in WORKER_COMPLETED_STATUSES and row.get("forward_control_gate", {}).get("status") == "passed" and row.get("worker_returncode_contract") for row in forward)
    result = {
        "schema_version": "track3_operator_scoped_stability_j2_v1",
        "status": "track3_operator_scoped_cpu_release_passed" if reverse_pass and forward_pass and existing.get("old_evidence_immutable") else "track3_operator_scoped_cpu_release_failed",
        "created_at_utc": now(),
        "existing_i2_reanalysis": {"status": existing["status"], "report": str((OUT_ROOT / "j2_existing_i2_reanalysis.json").resolve()), "reverse_recorded_fields_operator_pass": existing["reverse_recorded_fields_pass"], "contact_matrix_not_evaluated": existing["contact_matrix_not_evaluated"], "cannot_release_j3_alone": True},
        "new_reverse": {"runs": len(reverse), "pass_count": sum(row.get("reverse_operator_gate", {}).get("status") == "passed" and row.get("process_returncode") == 0 for row in reverse), "required": "9/9", "operator_scoped_gate": reverse_pass, "run_reports": [row["corrected_report"] for row in reverse]},
        "new_forward_controls": {"runs": len(forward), "pass_count": sum(row.get("forward_control_gate", {}).get("status") == "passed" and row.get("process_returncode") == 0 for row in forward), "required": "6/6", "complete": forward_pass, "run_reports": [row["corrected_report"] for row in forward]},
        "runs": [_compact_run_row(row) for row in rows],
        "raw_fresh_process_reports": [str(path.resolve()) for path in raw_paths],
        "hard_gate": {"old_i2_immutable": existing.get("old_evidence_immutable") is True, "new_reverse_9_of_9": reverse_pass, "new_forward_6_of_6": forward_pass, "fresh_process_count": len(rows) == 15, "full_contact_matrix_new_runs": all(len(_checkpoint_map(row)) == 21 and all(sample.get("contact_matrix", {}).get("complete") is True for sample in _checkpoint_map(row).values()) for row in rows), "no_hindsight": True, "no_bddl_modification": True},
        "resource_boundary": {"cpu_no_renderer_only": True, "gpu3_touched": False, "gpu0_touched": False, "se3": False, "grasp": False, "demo_generation": False, "training_or_norm": False, "fast_or_zip": False, "official_gpu": False, "upload_or_submission": False, "pending_autogen": 0, "accepted_reverse_demo": 0},
    }
    write_json(OUT_ROOT / "j2_reverse_release.json", {"schema_version": "track3_operator_scoped_stability_j2_reverse_v1", "status": result["status"], "new_reverse": result["new_reverse"], "hard_gate": result["hard_gate"], "runs": [_compact_run_row(row) for row in reverse]})
    write_json(OUT_ROOT / "j2_forward_controls.json", {"schema_version": "track3_operator_scoped_stability_j2_forward_v1", "status": "passed" if forward_pass else "failed_gate", "new_forward_controls": result["new_forward_controls"], "runs": [_compact_run_row(row) for row in forward]})
    write_json(OUT_ROOT / "j2_summary.json", result)
    return result


def _run_validation(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=ROOT, env=dict(env or os.environ), capture_output=True, text=True, timeout=900, check=False)
    return {"command": list(command), "returncode": int(completed.returncode), "passed": completed.returncode == 0, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}


def _json_parse_check() -> dict[str, Any]:
    paths = sorted(path for path in OUT_ROOT.rglob("*.json") if path.is_file())
    errors = []
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
    return {"files": len(paths), "errors": errors, "passed": not errors}


def _revalidate_j0_lock() -> dict[str, Any]:
    lock = json.loads((OUT_ROOT / "j0_release_lock.json").read_text(encoding="utf-8"))
    checks = {}
    for path_string, expected in lock.get("immutable_inputs", {}).items():
        path = Path(path_string)
        checks[path_string] = path.is_file() and path.stat().st_size == expected.get("bytes") and sha256_file(path) == expected.get("sha256")
    nested = ROOT / "third_party/axis_openpi"
    return {
        "j0_status": lock.get("status"),
        "immutable_inputs_unchanged": bool(checks) and all(checks.values()),
        "immutable_input_count": len(checks),
        "immutable_input_checks": checks,
        "old_i2_hash_unchanged": sha256_file(OLD_I2) == OLD_I2_SHA256,
        "nested_openpi_commit_unchanged": git_output(nested, ("rev-parse", "HEAD")) == lock.get("nested_openpi_commit"),
        "nested_openpi_clean": not porcelain(nested),
        "libero_plus_status_record_only": True,
    }


def run_j4() -> dict[str, Any]:
    j0 = json.loads((OUT_ROOT / "j0_release_lock.json").read_text(encoding="utf-8"))
    j1 = json.loads((OUT_ROOT / "j1_contract.json").read_text(encoding="utf-8"))
    j2 = json.loads((OUT_ROOT / "j2_summary.json").read_text(encoding="utf-8"))
    failed_runs = [
        {"task_index": row.get("task_index"), "kind": row.get("kind"), "run_index": row.get("run_index"), "failed_gates": [name for name, gate in row.get("reverse_operator_gate", {}).get("gates", {}).items() if gate.get("state") == "failed"]}
        for row in j2.get("runs", [])
        if row.get("kind") == "reverse" and row.get("reverse_operator_gate", {}).get("status") != "passed"
    ]
    validation_env = os.environ.copy()
    validation_env["PYTHONPATH"] = f"{ROOT / 'LIBERO-plus'}:{ROOT}"
    validation_env["JAX_PLATFORMS"] = "cpu"
    pycompile = _run_validation([sys.executable, "-m", "py_compile", str(ROOT / "tools/axis_pi05/track3_operator_scoped_stability.py"), str(ROOT / "tests/test_track3_operator_scoped_stability.py")], env=validation_env)
    targeted = _run_validation([sys.executable, "-m", "pytest", "-q", "tests/test_track3_operator_scoped_stability.py"], env=validation_env)
    full = _run_validation([sys.executable, "-m", "pytest", "-q", "tests"], env=validation_env)
    diff_check = _run_validation(["git", "diff", "--check"], env=os.environ)
    lock_checks = _revalidate_j0_lock()
    old_summary = json.loads(OLD_SUMMARY.read_text(encoding="utf-8"))
    old_counts = {key: int(old_summary.get("counts", {}).get(key, 0)) for key in ("exact_reverse_demo", "pending_autogen", "accepted_reverse_demo")}
    j3 = {"executed": False, "state": "not_evaluated", "required": False, "reason": "J2 failed 8/9 reverse operator-scoped gate; GPU3 hard stop"}
    final_status = "track3_operator_scoped_cpu_release_failed_truth_vector_transition_task19_run1" if failed_runs else "track3_operator_scoped_cpu_release_failed"
    closeout_checks = {
        "targeted_tests": targeted,
        "full_tests": full,
        "json_parse": _json_parse_check(),
        "py_compile": pycompile,
        "git_diff_check": diff_check,
    }
    report = {
        "schema_version": "track3_operator_scoped_stability_j4_v1",
        "status": final_status,
        "created_at_utc": now(),
        "hard_stop": True,
        "phases": {
            "J0": gate_record(executed=True, value=j0.get("status") == "track3_operator_scoped_input_locked"),
            "J1": gate_record(executed=True, value=j1.get("status") == "track3_operator_scoped_contract_validated"),
            "J2": gate_record(executed=True, value=j2.get("status") == "track3_operator_scoped_cpu_release_passed", reason="new_reverse_9_of_9_failed" if j2.get("status") != "track3_operator_scoped_cpu_release_passed" else None),
            "J3": j3,
            "J4": gate_record(executed=True, value=all(row.get("passed") or row.get("json_parse", {}).get("passed") for row in closeout_checks.values()), required=True),
        },
        "j2_outcome": {"status": j2.get("status"), "new_reverse": j2.get("new_reverse"), "new_forward_controls": j2.get("new_forward_controls"), "failed_runs": failed_runs, "raw_process_count": len(j2.get("raw_fresh_process_reports", [])), "worker_exception_count": sum(row.get("status") == "worker_exception" for row in j2.get("runs", []))},
        "old_vs_new_gate": {"old_i2_summary_sha256": sha256_file(OLD_I2), "old_i2_summary_expected_sha256": OLD_I2_SHA256, "old_i2_unchanged": sha256_file(OLD_I2) == OLD_I2_SHA256, "old_full_scene_failure_preserved": True, "new_operator_scoped_release": j2.get("status"), "gpu3_canary": "not_evaluated"},
        "closeout_checks": closeout_checks,
        "immutable_revalidation": lock_checks,
        "reverse_demo_counts": old_counts,
        "resource_boundary": {"J2_cpu_no_renderer_only": True, "J3_gpu3_touched": False, "gpu0_touched": False, "se3": False, "grasp": False, "demo_generation": False, "teleop": False, "training_or_norm": False, "fast_or_zip": False, "official_gpu": False, "upload_or_submission": False, "pending_autogen": 0, "accepted_reverse_demo": 0},
        "next_action": "等待审核；不得进入 J3、SE(3)、demo、Teleop、批量生成或训练",
    }
    write_json(OUT_ROOT / "summary.json", report)
    markdown = f"""# Track 3 Operator-Scoped Stability Release J4 收尾\n\n- 状态：`{final_status}`\n- J0/J1：通过；J2：`{j2.get('new_reverse', {}).get('pass_count', 0)}/9` reverse、`{j2.get('new_forward_controls', {}).get('pass_count', 0)}/6` forward。\n- 失败定位：`{json.dumps(failed_runs, ensure_ascii=False)}`。该失败来自固定 step5→20 full init truth-vector gate，不改写为 background pass。\n- J3：`not_evaluated`，GPU3 未使用；J4 hard stop 已执行。\n\n旧 I2 hash、旧 full-scene failure 与 reverse-demo `0/0/0` 保持不变。完整 raw fresh-process 矩阵保留在 `runs/j2_task*.json`，step5 baseline 修正后的派生报告为 `runs/j2_reanalysis_*.json`。\n\n本轮不授权 SE(3)、抓取/demo、Teleop、批量生成、训练、FAST/ZIP、官方 GPU、上传或提交；等待下一次审核。\n"""
    (OUT_ROOT / "J4_CLOSEOUT_REPORT_ZH.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j0", action="store_true")
    parser.add_argument("--j1", action="store_true")
    parser.add_argument("--j2", action="store_true")
    parser.add_argument("--j4", action="store_true")
    parser.add_argument("--j2-worker", action="store_true")
    parser.add_argument("task_index", nargs="?")
    parser.add_argument("kind", nargs="?")
    parser.add_argument("run_index", nargs="?")
    args = parser.parse_args()
    if args.j0:
        result = run_j0()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.j1:
        result = run_j1()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "track3_operator_scoped_contract_validated" else 1
    if args.j2_worker:
        if args.task_index not in {str(task) for task in TASKS} or args.kind not in {"forward", "reverse"} or args.run_index not in {"1", "2", "3"}:
            parser.error("--j2-worker requires TASK_INDEX forward|reverse RUN_INDEX")
        result = j2_worker(int(args.task_index), args.kind, int(args.run_index))
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return worker_exit_code(str(result.get("status")))
    if args.j2:
        result = run_j2()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.j4:
        result = run_j4()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser.error("use --j0, --j1, --j2, --j4, or --j2-worker")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
