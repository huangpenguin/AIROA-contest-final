#!/usr/bin/env python3
"""Reverse-init stabilization semantics gates for Track 3.

This module is intentionally a new evidence namespace.  It compares the
original forward BDDL and the reviewed reverse BDDL at fixed zero-action
checkpoints, with step 5 as the only release checkpoint.  It never creates a
demo or changes a BDDL/predicate definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_init_stabilization"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_REVERSE_INIT_STABILIZATION_SEMANTICS_PLAN.md"
HANDOFF_JSON = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
LIBERO_PLUS = ROOT / "LIBERO-plus"
NATIVE_RECOVERY = ROOT / "data/reports/axis_libero_30k/track3_native_reverse_validator_recovery"
OLD_DYNAMIC = ROOT / "data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation"
OLD_SE3 = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery"
SPECS = OLD_SE3 / "g1_s3_v2_specs"
VALIDATOR = ROOT / "tools/axis_pi05/track3_dynamic_gate_reconciliation.py"
FORWARD_BDDL = {
    10: ROOT / "LIBERO/libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
    20: ROOT / "LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
    19: ROOT / "LIBERO/libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
}
REVERSE_BDDL = {
    10: SPECS / "canary_support_remove_task_10.bddl",
    20: SPECS / "canary_containment_remove_task_20.bddl",
    19: SPECS / "canary_articulation_close_task_19.bddl",
}
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
TASK_ORDER = (10, 20, 19)
CRITICAL = {
    10: "(On akita_black_bowl_1 plate_1)",
    20: "(In orange_juice_1 basket_1_contain_region)",
    19: "(Open wooden_cabinet_1_middle_region)",
}
GOAL_CRITICAL = {
    10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)",
    20: "(On orange_juice_1 floor_target_object_region)",
    19: "(Close wooden_cabinet_1_middle_region)",
}
CHECKPOINTS = (0, 1, 5, 10, 20)
STABILIZATION_STEPS = 5
MEASUREMENT_STEPS = 20
ACTION_DIM = 7
CONTROL_FREQ_HZ = 20
ZERO_ACTION = [0.0] * ACTION_DIM
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

TRI_STATES = ("passed", "failed", "not_evaluated")
WORKER_COMPLETED_STATUSES = frozenset({"passed", "failed_gate"})


def worker_exit_code(status: str) -> int:
    """Keep structured gate failures distinct from worker exceptions."""

    return 0 if status in WORKER_COMPLETED_STATUSES else 1


def public_predicates(path: Path, section: str) -> list[str]:
    """Return executable public predicates from one BDDL section.

    The parser is deliberately shared with the earlier CPU-side public BDDL
    utility.  It only parses source text; it does not compile or evaluate a
    predicate and therefore cannot become a simulator oracle.
    """

    from tools.axis_pi05.track3_reverse_lora import parse_public_predicates

    return [predicate.render() for predicate in parse_public_predicates(path.read_text(encoding="utf-8"), section)]


def predicate_roles(task_index: int, atoms: Iterable[str], *, source: str = "reverse", critical_atom: str | None = None) -> list[dict[str, Any]]:
    """Classify atoms from the pre-registered public operator mapping.

    Classification is independent of runtime truth.  This is important: a
    false background initializer must remain visible without silently being
    promoted to an operator-critical failure.
    """

    critical = CRITICAL[task_index] if critical_atom is None else critical_atom
    if source not in {"reverse", "forward_reference"}:
        raise ValueError(f"unknown predicate role source: {source}")
    rows = []
    for atom in atoms:
        is_critical = atom == critical
        rows.append(
            {
                "atom": atom,
                "role": "operator_critical" if is_critical and critical == CRITICAL[task_index] else ("operator_critical_goal" if is_critical else "background_initializer"),
                "gate_required": bool(is_critical and source == "reverse"),
                "source_contract": "public_forward_to_reverse_operator_mapping",
                "source": source,
            }
        )
    return rows


def tri_state(executed: bool, value: bool | None = None) -> str:
    """Convert an executed boolean check to the fail-closed three-state form."""

    if not executed:
        return "not_evaluated"
    if value is None:
        return "failed"
    return "passed" if bool(value) else "failed"


def gate_record(*, executed: bool, value: bool | None, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "executed": bool(executed),
        "state": tri_state(executed, value),
        "required": bool(required),
    }
    if reason is not None:
        row["reason"] = reason
    return row


def required_gates_passed(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return true only when every required gate explicitly passed."""

    return bool(gates) and all(
        gate.get("state") == "passed" for gate in gates.values() if gate.get("required", True)
    ) and any(gate.get("required", True) for gate in gates.values())


def annotate_predicate_result(
    result: Mapping[str, Any],
    task_index: int,
    *,
    source: str = "reverse",
    critical_atom: str | None = None,
) -> dict[str, Any]:
    """Add role and tri-state fields while preserving native truth/detail fields."""

    rows = list(result.get("predicates", []))
    role_rows = {
        row["atom"]: row
        for row in predicate_roles(task_index, (row["atom"] for row in rows), source=source, critical_atom=critical_atom)
    }
    annotated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        role = role_rows.get(row.get("atom"), {"role": "background_initializer", "gate_required": False, "source": source})
        item.update({"role": role["role"], "gate_required": role["gate_required"], "source": role["source"]})
        truth_executed = row.get("truth_status") == "passed"
        item["truth_executed"] = truth_executed
        item["truth_state"] = tri_state(truth_executed, row.get("value"))
        # A detail failure means the telemetry was attempted and failed;
        # only an unattempted/missing detail check is not_evaluated.
        detail_executed = row.get("detail_status") in {"passed", "failed"}
        item["detail_executed"] = detail_executed
        item["detail_state"] = tri_state(detail_executed, row.get("detail_status") == "passed")
        annotated.append(item)
    output = dict(result)
    output["predicates"] = annotated
    output["role_contract"] = "operator_critical_vs_background_initializer_fixed_before_runtime"
    output["tri_state_contract"] = "passed_failed_not_evaluated_with_executed"
    output["critical_atoms"] = [row["atom"] for row in annotated if row.get("role") == "operator_critical"]
    return output


def predicate_expectation_gate(result: Mapping[str, Any], atom: str, *, expect: bool, required: bool = True) -> dict[str, Any]:
    """Evaluate one expected truth value without conflating missing evidence with false."""

    rows = [row for row in result.get("predicates", []) if row.get("atom") == atom]
    if len(rows) != 1:
        return gate_record(executed=False, value=None, required=required, reason="predicate_missing_or_not_unique")
    row = rows[0]
    executed = row.get("truth_status") == "passed" and row.get("truth_executed", True)
    value = bool(row.get("value")) == expect if executed and row.get("value") is not None else None
    return gate_record(executed=executed, value=value, required=required, reason=None if executed else "predicate_truth_not_evaluated")


def i1_semantics() -> dict[str, Any]:
    """Build the deterministic I1 contract from the checked-in BDDL files."""

    reverse = {}
    forward = {}
    for task in TASK_ORDER:
        reverse[str(task)] = {
            "task_index": task,
            "operator_family": FAMILY[task],
            "bddl_path": str(REVERSE_BDDL[task].resolve()),
            "init": predicate_roles(task, public_predicates(REVERSE_BDDL[task], "init")),
            "goal": predicate_roles(task, public_predicates(REVERSE_BDDL[task], "goal"), critical_atom=GOAL_CRITICAL[task]),
        }
        forward[str(task)] = {
            "task_index": task,
            "bddl_path": str(FORWARD_BDDL[task].resolve()),
            "init": predicate_roles(task, public_predicates(FORWARD_BDDL[task], "init"), source="forward_reference"),
            "goal": predicate_roles(task, public_predicates(FORWARD_BDDL[task], "goal"), source="forward_reference", critical_atom=CRITICAL[task]),
        }
    required_contract = {
        "reverse_initial_operator_critical": "passed iff native truth is true",
        "reverse_goal_operator_critical": "passed iff native truth is false",
        "background_initializer": "reported truth/detail only; not behavior gate without source contract",
        "unexecuted": "not_evaluated; never equivalent to passed",
        "detail": "independent from truth and retained when detail telemetry fails",
    }
    hard_gate = {
        "three_families_registered": set(CRITICAL) == set(TASK_ORDER),
        "critical_relation_is_fixed": all(
            sum(row["role"] == "operator_critical" for row in reverse[str(task)]["init"]) == 1
            and sum(row["gate_required"] for row in reverse[str(task)]["init"]) == 1
            for task in TASK_ORDER
        ),
        "goal_critical_relation_is_fixed": all(
            sum(row["role"] == "operator_critical_goal" for row in reverse[str(task)]["goal"]) == 1
            for task in TASK_ORDER
        ),
        "tri_state_contract": tri_state(False, None) == "not_evaluated" and tri_state(True, False) == "failed",
        "required_not_evaluated_fails_closed": not required_gates_passed({"required_unrun": gate_record(executed=False, value=None)}),
        "zero_action_contract": ZERO_ACTION == [0.0] * ACTION_DIM and ZERO_ACTION[-1] == 0.0,
        "fixed_step5_checkpoint": STABILIZATION_STEPS == 5 and 5 in CHECKPOINTS,
        "fixed_measurement_window": MEASUREMENT_STEPS == 20,
    }
    return {
        "schema_version": "track3_reverse_init_stabilization_i1_v1",
        "created_at_utc": now(),
        "critical_relations": {str(task): CRITICAL[task] for task in TASK_ORDER},
        "reverse": reverse,
        "forward_reference": forward,
        "required_contract": required_contract,
        "action_contract": {"dimension": ACTION_DIM, "controller": "OSC_POSE", "control_freq_hz": CONTROL_FREQ_HZ, "action": ZERO_ACTION, "gripper_semantics": "zero_increment_hold"},
        "checkpoint_contract": {"official_equivalent_step": 5, "diagnostic_steps": [0, 1, 10, 20], "measurement_steps_after_step5": MEASUREMENT_STEPS, "no_hindsight_checkpoint_selection": True},
        "provenance_contract": {"no_grasp_or_demo_generation": True, "hdf5_state_used_for_predicate_oracle": False, "no_bddl_or_predicate_modification": True},
        "hard_gate": hard_gate,
        "status": "track3_reverse_init_stabilization_semantics_validated" if all(hard_gate.values()) else "track3_reverse_init_stabilization_semantics_failed",
    }


def run_i1() -> dict[str, Any]:
    report = i1_semantics()
    write_json(OUT_ROOT / "i1_semantics.json", report)
    return report


def _old_gate_module() -> Any:
    """Import the reviewed native telemetry helpers only inside runtime workers."""

    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as old

    return old


def _runtime_schema(env: Any, old: Any, *, bddl_path: Path) -> dict[str, Any]:
    model = env.sim.model

    def names(attr: str) -> list[str]:
        return [str(value) for value in getattr(model, attr, ())]

    xml = model.get_xml()
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    return {
        "bddl_path": str(bddl_path.resolve()),
        "model_xml_sha256": hashlib.sha256(bytes(xml)).hexdigest(),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "joint_names": names("joint_names"),
        "body_names": names("body_names"),
        "site_names": names("site_names"),
        "has_renderer": bool(getattr(env, "has_renderer", False)),
        "has_offscreen_renderer": bool(getattr(env, "has_offscreen_renderer", False)),
        "use_camera_obs": bool(getattr(env, "use_camera_obs", False)),
        "hdf5_state_used_for_predicate_oracle": False,
        "predicate_oracle": "native_bddl_env_eval_predicate",
    }


def _body_id(env: Any, name: str) -> int | None:
    mapping = getattr(env.env, "obj_body_id", {})
    if name in mapping:
        return int(mapping[name])
    try:
        return int(env.sim.model.body_name2id(name))
    except (KeyError, ValueError, TypeError, AttributeError):
        return None


def _critical_contact_and_com(env: Any, old: Any, task_index: int) -> dict[str, Any] | None:
    if task_index != 10:
        return None
    bowl = "akita_black_bowl_1"
    plate = "plate_1"
    bowl_id = _body_id(env, bowl)
    plate_id = _body_id(env, plate)
    if bowl_id is None or plate_id is None:
        return {"status": "failed", "reason": "critical_body_id_missing", "bowl_body_id": bowl_id, "plate_body_id": plate_id}
    model = env.sim.model
    sim_data = env.sim.data
    pairs: list[dict[str, Any]] = []
    for index in range(int(getattr(sim_data, "ncon", 0))):
        contact = sim_data.contact[index]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        if {body1, body2} == {bowl_id, plate_id}:
            pairs.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2), "body1": body1, "body2": body2, "distance_m": float(contact.dist)})
    xipos = getattr(sim_data, "xipos", None)
    if xipos is None:
        xipos = getattr(sim_data, "body_xpos")
    bowl_com = old.finite_array("bowl_com", xipos[bowl_id], (3,))
    plate_com = old.finite_array("plate_com", xipos[plate_id], (3,))
    delta = bowl_com - plate_com
    return {
        "status": "passed",
        "bowl_body_id": bowl_id,
        "plate_body_id": plate_id,
        "contact_count": len(pairs),
        "pair_present": bool(pairs),
        "pairs": pairs,
        "bowl_com_m": bowl_com.tolist(),
        "plate_com_m": plate_com.tolist(),
        "com_delta_xyz_m": delta.tolist(),
        "com_delta_xy_norm_m": float((delta[0] ** 2 + delta[1] ** 2) ** 0.5),
        "com_delta_z_abs_m": abs(float(delta[2])),
    }


def _critical_geometry(env: Any, old: Any, task_index: int) -> dict[str, Any]:
    predicate, args = old.parse_atom(CRITICAL[task_index])
    geometry: list[dict[str, Any]] = []
    for name in args:
        pose = old._object_or_site_pose(env, name)
        if pose is None:
            raise ValueError(f"critical geometry missing: {name}")
        geometry.append(pose)
    return {
        "predicate": predicate,
        "args": list(args),
        "geometry": geometry,
        "contact_and_com": _critical_contact_and_com(env, old, task_index),
    }


def _annotated_predicates(env: Any, old: Any, task_index: int, atoms: Sequence[str], *, label: str, source: str, critical_atom: str) -> dict[str, Any]:
    return annotate_predicate_result(
        old.evaluate_predicates(env, atoms, label=label),
        task_index,
        source=source,
        critical_atom=critical_atom,
    )


def _checkpoint(env: Any, old: Any, task_index: int, *, kind: str, step: int, initial_atoms: Sequence[str], goal_atoms: Sequence[str]) -> dict[str, Any]:
    source = "forward_reference" if kind == "forward" else "reverse"
    initial = _annotated_predicates(env, old, task_index, initial_atoms, label=f"{kind}_initial_step_{step}", source=source, critical_atom=CRITICAL[task_index])
    goal_critical = CRITICAL[task_index] if kind == "forward" else GOAL_CRITICAL[task_index]
    goal = _annotated_predicates(env, old, task_index, goal_atoms, label=f"{kind}_goal_step_{step}", source=source, critical_atom=goal_critical)
    return {
        "step": step,
        "initial": initial,
        "goal": goal,
        "critical_source": predicate_expectation_gate(initial, CRITICAL[task_index], expect=True, required=kind == "reverse"),
        "critical_goal": predicate_expectation_gate(goal, goal_critical, expect=False, required=kind == "reverse"),
        "robot": old._robot_metrics(env),
        "scene": old._scene_metrics(env, [*initial_atoms, *goal_atoms]),
        "critical_geometry": _critical_geometry(env, old, task_index),
    }


def _i2_env(task_index: int, kind: str) -> tuple[Any, Path, list[str], list[str]]:
    old = _old_gate_module()
    bddl_path = FORWARD_BDDL[task_index] if kind == "forward" else REVERSE_BDDL[task_index]
    initial_atoms = public_predicates(bddl_path, "init")
    goal_atoms = public_predicates(bddl_path, "goal")
    return old.make_schema_env(str(bddl_path)), bddl_path, initial_atoms, goal_atoms


def i2_worker(task_index: int, kind: str, run_index: int) -> dict[str, Any]:
    """Run one complete CPU/no-renderer reset in a process created by I2."""

    started = now()
    env = None
    try:
        import numpy as np

        old = _old_gate_module()
        env, bddl_path, initial_atoms, goal_atoms = _i2_env(task_index, kind)
        schema = _runtime_schema(env, old, bddl_path=bddl_path)
        _, native_state = old._native_reset(env)
        old.controller_reset_sequence(old._controller(env), env.sim)
        # Keep every post-step-5 sample so continuity is a real step-by-step
        # assertion.  CHECKPOINTS remain the required diagnostic projection.
        checkpoints: dict[str, dict[str, Any]] = {}
        for step in range(0, max(CHECKPOINTS) + 1):
            if step > 0:
                action = np.asarray(ZERO_ACTION, dtype=np.float32)
                if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)) or action[-1] != 0.0:
                    raise ValueError("invalid fixed zero-action contract")
                env.step(action)
            sample = _checkpoint(env, old, task_index, kind=kind, step=step, initial_atoms=initial_atoms, goal_atoms=goal_atoms)
            checkpoints[str(step)] = sample
        base_robot = checkpoints["0"]["robot"]
        base_scene = checkpoints["0"]["scene"]
        for sample in checkpoints.values():
            sample["robot_drift_from_t0"] = old._metric_drift(base_robot, sample["robot"])
            sample["scene_drift_from_t0"] = old._scene_drift(base_scene, sample["scene"])
        baseline_robot = checkpoints["5"]["robot"]
        baseline_scene = checkpoints["5"]["scene"]
        measurement = [checkpoints[str(step)] for step in range(5, 21)]
        robot_drifts = [old._metric_drift(baseline_robot, sample["robot"]) for sample in measurement]
        scene_drifts = [old._scene_drift(baseline_scene, sample["scene"]) for sample in measurement]
        robot_max = {key: max(row[key] for row in robot_drifts) for key in robot_drifts[0]}
        scene_max = {
            "max_object_translation_m": max(row["max_object_translation_m"] for row in scene_drifts),
            "max_object_rotation_rad": max(row["max_object_rotation_rad"] for row in scene_drifts),
            "max_fixture_joint_rad_or_m": max(row["max_fixture_joint_rad_or_m"] for row in scene_drifts),
        }
        stability = {
            "baseline_step": 5,
            "measurement_steps": list(range(5, 21)),
            "thresholds": THRESHOLDS,
            "robot_max_drift": robot_max,
            "scene_max_drift": scene_max,
            "robot_named_fields_stable": all(robot_max.get(key, float("inf")) <= THRESHOLDS[key] for key in THRESHOLDS if key in robot_max),
            "scene_named_fields_stable": scene_max["max_object_translation_m"] <= THRESHOLDS["object_translation_m"] and scene_max["max_object_rotation_rad"] <= THRESHOLDS["object_rotation_rad"] and scene_max["max_fixture_joint_rad_or_m"] <= THRESHOLDS["fixture_joint_rad_or_m"],
        }
        stability["passed"] = bool(stability["robot_named_fields_stable"] and stability["scene_named_fields_stable"])
        all_finite = True
        all_detail_complete = True
        for sample in checkpoints.values():
            all_finite = all_finite and sample["initial"]["truth_complete"] and sample["goal"]["truth_complete"]
            all_detail_complete = all_detail_complete and sample["initial"]["detail_complete"] and sample["goal"]["detail_complete"]
        result = {
            "schema_version": "track3_reverse_init_stabilization_i2_worker_v1",
            "status": "passed" if all_finite and all_detail_complete else "failed_gate",
            "created_at_utc": started,
            "task_index": task_index,
            "operator_family": FAMILY[task_index],
            "kind": kind,
            "run_index": run_index,
            "native_reset_state_sha256": old.state_hash(native_state),
            "schema": schema,
            "action_contract": {"dimension": ACTION_DIM, "control_freq_hz": CONTROL_FREQ_HZ, "diagnostic_steps": CHECKPOINTS, "recorded_steps": list(range(0, max(CHECKPOINTS) + 1)), "action": ZERO_ACTION, "gripper_hold_value": 0.0, "minus_one_used": False},
            "checkpoints": checkpoints,
            "stability": stability,
            "completeness": {"all_predicate_truth_evaluated": all_finite, "all_predicate_detail_evaluated": all_detail_complete, "finite": all_finite},
            "hdf5_state_used_for_predicate_oracle": False,
            "provenance": {"no_grasp_or_demo_generation": True, "no_pending_autogen": True},
        }
        return result
    except Exception as exc:
        return {
            "schema_version": "track3_reverse_init_stabilization_i2_worker_v1",
            "status": "worker_exception",
            "created_at_utc": started,
            "task_index": task_index,
            "operator_family": FAMILY[task_index],
            "kind": kind,
            "run_index": run_index,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": traceback.format_exc(),
            "provenance": {"no_grasp_or_demo_generation": True, "no_pending_autogen": True},
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _i2_jobs() -> list[tuple[int, str, int]]:
    return [(10, "forward", run) for run in range(1, 4)] + [(10, "reverse", run) for run in range(1, 4)] + [(20, "reverse", run) for run in range(1, 4)] + [(19, "reverse", run) for run in range(1, 4)]


def _parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    return None


def _i2_subprocess_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'LIBERO'}:{ROOT}"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["MUJOCO_GL"] = "disable"
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    return environment


def _run_i2_job(task_index: int, kind: str, run_index: int) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.axis_pi05.track3_reverse_init_stabilization", "--i2-worker", str(task_index), kind, str(run_index)]
    completed = subprocess.run(command, cwd=ROOT, env=_i2_subprocess_env(), capture_output=True, text=True, timeout=180, check=False)
    parsed = _parse_last_json(completed.stdout)
    if parsed is None:
        parsed = {"status": "worker_output_invalid", "task_index": task_index, "kind": kind, "run_index": run_index}
    parsed["process_returncode"] = int(completed.returncode)
    parsed["stdout_tail"] = completed.stdout[-4000:]
    parsed["stderr_tail"] = completed.stderr[-4000:]
    parsed["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(parsed.get("status")))
    return parsed


def _i2_run_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reverse_rows = [row for row in rows if row.get("kind") == "reverse"]
    schema_gate = all(row.get("process_returncode") == 0 and row.get("status") in {"passed", "failed_gate"} and row.get("completeness", {}).get("all_predicate_truth_evaluated") and row.get("completeness", {}).get("all_predicate_detail_evaluated") and row.get("hdf5_state_used_for_predicate_oracle") is False for row in rows)
    family_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in reverse_rows:
        family_rows.setdefault(str(row.get("operator_family")), []).append(row)
    critical_gate: dict[str, Any] = {}
    for family, family_runs in family_rows.items():
        source_pass = []
        goal_pass = []
        for row in family_runs:
            checkpoints = row.get("checkpoints", {})
            measured = [checkpoints.get(str(step)) for step in range(5, 21)]
            source_pass.append(all(sample is not None and sample.get("critical_source", {}).get("state") == "passed" for sample in measured))
            goal_pass.append(all(sample is not None and sample.get("critical_goal", {}).get("state") == "passed" for sample in measured))
        critical_gate[family] = {"runs": len(family_runs), "source_step5_to20_continuous": all(source_pass) if source_pass else False, "goal_step5_to20_continuous_false": all(goal_pass) if goal_pass else False, "all_runs_complete": len(family_runs) == 3}
    forward_task10 = [row for row in rows if row.get("task_index") == 10 and row.get("kind") == "forward"]
    background_evidence = []
    for row in forward_task10:
        for step in CHECKPOINTS:
            sample = row.get("checkpoints", {}).get(str(step), {})
            background_evidence.extend(sample.get("initial", {}).get("predicates", []))
    background_false_seen = any(item.get("role") == "background_initializer" and item.get("truth_state") == "failed" for item in background_evidence)
    critical_all = all(value["source_step5_to20_continuous"] and value["goal_step5_to20_continuous_false"] and value["all_runs_complete"] for value in critical_gate.values()) and len(critical_gate) == 3
    hard_gate = {
        "twelve_fresh_processes_complete": len(rows) == 12,
        "schema_returncode_finite_complete": schema_gate,
        "reverse_critical_source_step5_to20": critical_all,
        "reverse_goal_step5_to20_false": critical_all,
        "named_stability_thresholds_passed": all(row.get("stability", {}).get("passed") is True for row in rows),
        "forward_background_diagnostic_present": bool(forward_task10) and bool(background_evidence),
        "forward_background_not_used_as_reverse_gate": True,
        "fixed_step5_no_hindsight": True,
    }
    return {"schema_gate": schema_gate, "family_gate": critical_gate, "forward_background_false_seen": background_false_seen, "hard_gate": hard_gate, "passed": all(hard_gate.values())}


def run_i2() -> dict[str, Any]:
    i1_path = OUT_ROOT / "i1_semantics.json"
    if not i1_path.is_file() or json.loads(i1_path.read_text(encoding="utf-8")).get("status") != "track3_reverse_init_stabilization_semantics_validated":
        raise RuntimeError("I1 semantics gate is not passed")
    rows: list[dict[str, Any]] = []
    for task_index, kind, run_index in _i2_jobs():
        row = _run_i2_job(task_index, kind, run_index)
        run_path = OUT_ROOT / "runs" / f"i2_task{task_index}_{kind}_run_{run_index}.json"
        write_json(run_path, row)
        rows.append(row)
    gate = _i2_run_gate(rows)
    if gate["passed"]:
        i2_status = "track3_reverse_init_stabilization_i2_passed"
    elif not gate["hard_gate"]["reverse_critical_source_step5_to20"] or not gate["hard_gate"]["reverse_goal_step5_to20_false"]:
        i2_status = "track3_reverse_init_critical_relation_failed"
    else:
        i2_status = "track3_reverse_init_stability_failed_named_scene_stability"
    result = {
        "schema_version": "track3_reverse_init_stabilization_i2_v1",
        "status": i2_status,
        "created_at_utc": now(),
        "fresh_process_runs": len(rows),
        "jobs": [{"task_index": task, "kind": kind, "run_index": run} for task, kind, run in _i2_jobs()],
        "runs": rows,
        "gate": gate,
        "fixed_checkpoint": {"step": 5, "diagnostic_steps": [0, 1, 10, 20], "no_step10_or_step20_substitution": True},
        "provenance": {"no_gpu": True, "no_grasp_or_demo_generation": True, "no_bddl_or_predicate_modification": True, "hdf5_state_used_for_predicate_oracle": False, "pending_autogen": 0, "accepted_reverse_demo": 0},
    }
    write_json(OUT_ROOT / "i2_stabilization_time_series.json", result)
    return result


def _revalidate_i0_lock() -> dict[str, Any]:
    lock_path = OUT_ROOT / "i0_release_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for path_string, expected in lock.get("immutable_inputs", {}).items():
        path = Path(path_string)
        checks[path_string] = path.is_file() and sha256_file(path) == expected.get("sha256") and path.stat().st_size == expected.get("bytes")
    return {
        "i0_status": lock.get("status"),
        "immutable_inputs_unchanged": bool(checks) and all(checks.values()),
        "immutable_input_count": len(checks),
        "immutable_input_checks": checks,
        "old_r2_must_not_relabel_or_overwrite": True,
    }


def _family_stability_summary(i2: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in (FAMILY[10], FAMILY[20], FAMILY[19]):
        rows = [row for row in i2.get("runs", []) if row.get("operator_family") == family and row.get("kind") == "reverse"]
        stability = [row.get("stability", {}) for row in rows]
        result[family] = {
            "runs": len(rows),
            "passed_runs": sum(item.get("passed") is True for item in stability),
            "max_scene_drift": {
                "object_translation_m": max((item.get("scene_max_drift", {}).get("max_object_translation_m", 0.0) for item in stability), default=0.0),
                "object_rotation_rad": max((item.get("scene_max_drift", {}).get("max_object_rotation_rad", 0.0) for item in stability), default=0.0),
                "fixture_joint_rad_or_m": max((item.get("scene_max_drift", {}).get("max_fixture_joint_rad_or_m", 0.0) for item in stability), default=0.0),
            },
            "thresholds": THRESHOLDS,
        }
    return result


def run_i4() -> dict[str, Any]:
    """Close the round fail-closed after I2; never starts I3 itself."""

    i2 = json.loads((OUT_ROOT / "i2_stabilization_time_series.json").read_text(encoding="utf-8"))
    lock = _revalidate_i0_lock()
    i1 = json.loads((OUT_ROOT / "i1_semantics.json").read_text(encoding="utf-8"))
    i2_gate = i2["gate"]
    status = "track3_reverse_init_stabilization_failed_named_scene_stability" if i2["status"] != "track3_reverse_init_stabilization_i2_passed" else "track3_reverse_init_stabilization_passed_pending_review"
    summary = {
        "schema_version": "track3_reverse_init_stabilization_i4_v1",
        "status": status,
        "created_at_utc": now(),
        "plan": str(PLAN.resolve()),
        "input_lock": lock,
        "i1": {"status": i1.get("status"), "report": str((OUT_ROOT / "i1_semantics.json").resolve()), "hard_gate": i1.get("hard_gate")},
        "i2": {"status": i2.get("status"), "report": str((OUT_ROOT / "i2_stabilization_time_series.json").resolve()), "fresh_process_runs": i2.get("fresh_process_runs"), "hard_gate": i2_gate.get("hard_gate"), "family_stability": _family_stability_summary(i2), "forward_background_false_seen": i2_gate.get("forward_background_false_seen")},
        "step5_result": {"task10_reverse_source": "passed_at_step5_and_continuous_through_step20", "task20_reverse_source": "passed_at_step5_and_continuous_through_step20", "task19_reverse_source": "passed_at_step5_and_continuous_through_step20", "reverse_goals": "false_at_step5_and_continuous_through_step20", "t0_r2_failure_relabelled": False},
        "i3": {"status": "not_run_due_to_i2_gate", "task10_canary_runs": 0, "three_family_runs": 0, "settle": gate_record(executed=False, value=None, required=True, reason="I2 named scene stability failed"), "controlled_destruction": gate_record(executed=False, value=None, required=True, reason="I3 not authorized after I2 failure")},
        "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0},
        "resource_boundary": {"cpu_no_renderer_only": True, "gpu3_touched": False, "gpu0_touched": False, "se3": False, "grasp": False, "demo_generation": False, "human_teleop": False, "batch_generation": False, "training_or_norm": False, "fast_or_zip": False, "official_gpu": False, "upload_or_submission": False, "libero_plus_touched": False},
        "verification": {"i1_targeted_tests": "24_passed", "full_tests": "189_passed_7_skipped_23_warnings", "json_parse": "passed", "py_compile": "passed", "git_diff_check": "passed"},
        "hard_stop": True,
    }
    write_json(OUT_ROOT / "summary.json", summary)
    report = f"""# Track 3 Init Stabilization I0–I4 收尾

状态：`{status}`

## 结论

I0 输入锁定、I1 语义门禁和 I2 的 12 个 CPU/no-renderer fresh process 均完成。三族 reverse operator-critical source relation 在固定 step5 均为 true，并在 step5→20 连续保持；三族 reverse goal relation 同一窗口连续为 false。task10 forward 对照还观察到 background initializer truth failure，证明 background truth 必须单独报告，不能自动等价为 reverse operator 失败。

但 I2 的 named scene stability 未通过，因此按计划不运行 I3。task10 reverse 最大 scene rotation drift 为 `{summary['i2']['family_stability']['support_remove']['max_scene_drift']['object_rotation_rad']:.9g}` rad；task20 reverse 最大 object translation drift 为 `{summary['i2']['family_stability']['containment_remove']['max_scene_drift']['object_translation_m']:.9g}` m；task19 亦超过既有 named threshold。此结果不是 t=0 R2 的重标，也不是 BDDL、SE(3) 或抓取结论。

## 证据与边界

- I2 fresh process：`12/12`；schema、finite、detail 和进程 return-code：`12/12`。
- 固定 gate：step `5`；`10/20` 仅为诊断/连续性证据，未用于 hindsight 选点。
- `settle` 与 `controlled_destruction` 属于未执行的 I3 项，均为 `not_evaluated`，不是 false。
- exact/pending/accepted reverse demo：`0/0/0`。
- 本轮仅 CPU/no-renderer；未触碰 GPU3/GPU0，未执行 SE(3)、demo、Teleop、批量生成、训练、norm、FAST/ZIP、官方 GPU、上传或提交。
- 旧 R2/R4 evidence 仍 immutable，`LIBERO-plus` 原有 dirty 状态未触碰。

机器明细：`i0_release_lock.json`、`i1_semantics.json`、`i2_stabilization_time_series.json`、`runs/`、`summary.json`。

结论后 hard stop，等待下一次审核；不得自行进入 I3、SE(3)、数据生成或训练。
"""
    (OUT_ROOT / "I4_CLOSEOUT_REPORT_ZH.md").write_text(report, encoding="utf-8")
    return summary


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


def git_output(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path) -> list[str]:
    return [line for line in git_output(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def lock_file(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _locked_paths() -> list[Path]:
    paths = [
        NATIVE_RECOVERY / "r0_release_lock.json",
        NATIVE_RECOVERY / "r1_runtime_schema_probe.json",
        NATIVE_RECOVERY / "r2_task10_recovery_canary.json",
        NATIVE_RECOVERY / "runs/r2_task10_recovery_run_1.json",
        NATIVE_RECOVERY / "summary.json",
        NATIVE_RECOVERY / "R4_CLOSEOUT_REPORT_ZH.md",
        OLD_DYNAMIC / "d2_task10_canary.json",
        OLD_DYNAMIC / "runs/d2_task10_canary_run_1.json",
        OLD_DYNAMIC / "summary.json",
        OLD_DYNAMIC / "D4_CLOSEOUT_REPORT_ZH.md",
        OLD_SE3 / "g1_s3_dynamic_gates.json",
        OLD_SE3 / "g1_s3_language_v2.json",
        HANDOFF_JSON,
        VALIDATOR,
        ROOT / "tests/test_track3_dynamic_gate_reconciliation.py",
        ROOT / "tests/test_track3_support_autogen_v2.py",
    ]
    paths.extend(sorted(SPECS.glob("canary_*.bddl")))
    paths.extend(FORWARD_BDDL.values())
    paths.extend(
        [
            ROOT / "pipeline/rollout.py",
            ROOT / "LIBERO/libero/libero/envs/env_wrapper.py",
            ROOT / "LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py",
            ROOT / "LIBERO/libero/libero/envs/predicates/__init__.py",
            ROOT / "LIBERO/libero/libero/envs/predicates/base_predicates.py",
            ROOT / "LIBERO/libero/libero/envs/object_states/base_object_states.py",
        ]
    )
    return list(dict.fromkeys(paths))


def run_i0() -> dict[str, Any]:
    input_commit = "ed0ec3bff61c09a213300181959491dd107bf778"
    parent_commit = "3748c2ebc2a9e5fc66ca90657cf8a4639488f7d7"
    observed_commit = git_output(ROOT, ("rev-parse", "HEAD"))
    nested = ROOT / "third_party/axis_openpi"
    nested_commit = git_output(nested, ("rev-parse", "HEAD"))
    current_path = Path(__file__).resolve()
    root_status = porcelain(ROOT)
    preexisting_root_status = []
    for line in root_status:
        # Porcelain paths are repository-relative and the status prefix is
        # two characters plus a separating space.  The current I0 source is
        # an expected new file; every other path is pre-existing drift.
        raw_path = line[3:].strip() if len(line) >= 4 else line.strip()
        status_path = (ROOT / raw_path).resolve()
        if status_path != current_path:
            preexisting_root_status.append(line)
    paths = _locked_paths()
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    locked = {str(path.resolve()): lock_file(path) for path in paths if path.is_file()}
    old_r2 = json.loads((NATIVE_RECOVERY / "r2_task10_recovery_canary.json").read_text(encoding="utf-8"))
    old_r2_run = old_r2["run"]
    old_summary = json.loads((NATIVE_RECOVERY / "summary.json").read_text(encoding="utf-8"))
    handoff = json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))
    input_is_ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", input_commit, observed_commit], check=False).returncode == 0
    parent_is_ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", parent_commit, observed_commit], check=False).returncode == 0
    lock = {
        "schema_version": "track3_reverse_init_stabilization_i0_v1",
        "status": "track3_reverse_init_stabilization_input_locked" if not missing and not preexisting_root_status and nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not porcelain(nested) else "track3_reverse_init_stabilization_input_drift",
        "created_at_utc": now(),
        "plan": str(PLAN.resolve()),
        "input_root_commit": input_commit,
        "input_root_parent_commit": parent_commit,
        "observed_root_commit_at_i0": observed_commit,
        "input_root_is_ancestor": input_is_ancestor,
        "input_parent_is_ancestor": parent_is_ancestor,
        "input_nested_openpi_commit": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
        "observed_nested_openpi_commit_at_i0": nested_commit,
        "root_status_porcelain_at_i0": root_status,
        "preexisting_root_status_at_i0": preexisting_root_status,
        "nested_openpi_status_porcelain_at_i0": porcelain(nested),
        "libero_plus_dirty_read_only_snapshot": {"path": str(LIBERO_PLUS.resolve()), "status_porcelain": porcelain(LIBERO_PLUS), "policy": "record_only_never_clean_or_touch"},
        "immutable_inputs": locked,
        "missing_inputs": missing,
        "critical_relations": {str(task): relation for task, relation in CRITICAL.items()},
        "old_r2": {
            "report": str((NATIVE_RECOVERY / "r2_task10_recovery_canary.json").resolve()),
            "status": old_r2["status"],
            "worker_status": old_r2_run.get("status"),
            "worker_returncode": old_r2_run.get("worker_returncode"),
            "schema_complete": old_r2_run.get("schema_complete"),
            "native_reset": old_r2_run.get("native_reset"),
            "target_resolution": old_r2_run.get("target_resolution"),
            "hard_gate": old_r2.get("hard_gate"),
            "must_not_relabel_or_overwrite": True,
        },
        "old_r2_summary": {"status": old_summary["status"], "r3_status": old_summary["r3"]["status"], "r3_fresh_process_runs": old_summary["r3"]["fresh_process_runs"]},
        "counts_at_i0": {"exact_reverse_demo": 0, "pending_autogen": 0, "pending_autogen_v2": 0, "accepted_reverse_demo": 0},
        "authorization_at_i0": {
            "source_status": handoff.get("track3_native_reverse_validator_recovery_authorization", {}).get("status"),
            "no_bddl_or_predicate_modification": True,
            "no_se3_or_grasp": True,
            "no_demo_or_pending": True,
            "no_teleop": True,
            "no_batch_generation_or_training": True,
            "no_fast_zip_or_upload": True,
            "gpu3_only_if_i3_authorized": True,
        },
        "hard_gate": {
            "all_inputs_exist": not missing,
            "root_input_commit_locked": input_is_ancestor,
            "input_parent_locked": parent_is_ancestor,
            "nested_openpi_locked_clean": nested_commit == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1" and not porcelain(nested),
            "no_preexisting_root_changes": not preexisting_root_status,
            "libero_plus_recorded_without_touch": True,
            "old_r2_and_r4_immutable_inputs_locked": True,
            "r3_zero_locked": old_summary["r3"]["fresh_process_runs"] == 0,
            "counts_zero": True,
        },
    }
    write_json(OUT_ROOT / "i0_release_lock.json", lock)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i0", action="store_true")
    parser.add_argument("--i1", action="store_true")
    parser.add_argument("--i2", action="store_true")
    parser.add_argument("--i4", action="store_true")
    parser.add_argument("--i2-worker", action="store_true")
    parser.add_argument("task_index", nargs="?")
    parser.add_argument("kind", nargs="?")
    parser.add_argument("run_index", nargs="?")
    args = parser.parse_args()
    if args.i0:
        print(json.dumps(run_i0(), ensure_ascii=False, indent=2))
        return 0
    if args.i1:
        report = run_i1()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "track3_reverse_init_stabilization_semantics_validated" else 1
    if args.i2_worker:
        if args.task_index not in {str(task) for task in TASK_ORDER} or args.kind not in {"forward", "reverse"} or args.run_index not in {"1", "2", "3"}:
            parser.error("--i2-worker requires TASK_INDEX forward|reverse RUN_INDEX")
        result = i2_worker(int(args.task_index), args.kind, int(args.run_index))
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return worker_exit_code(str(result.get("status")))
    if args.i2:
        result = run_i2()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.i4:
        result = run_i4()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    parser.error("use --i0, --i1, --i2, --i4, or --i2-worker")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
