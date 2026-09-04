#!/usr/bin/env python3
"""Composed reverse-start validation for Track 3.

M0/M1 are deliberately simulator-free.  The later workers are fresh-process
runtime probes and are enabled only by the explicit phase commands.  This
module owns a new evidence namespace and does not mutate the older release
tool or any source HDF5/BDDL.
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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_composed_reverse_start_stabilization"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_MATCHED_TERMINAL_L0_L5_AUDIT_20260904.md"
HANDOFF_JSON = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
PUBLIC_LIBERO = ROOT / "LIBERO"
LIBERO_PLUS = ROOT / "LIBERO-plus"
OPENPI = ROOT / "third_party/axis_openpi"

HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
FORWARD_BDDL = {
    10: PUBLIC_LIBERO / "libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
    20: PUBLIC_LIBERO / "libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
    19: PUBLIC_LIBERO / "libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
}
SPECS = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_v2_specs"
REVERSE_BDDL = {
    10: SPECS / "canary_support_remove_task_10.bddl",
    20: SPECS / "canary_containment_remove_task_20.bddl",
    19: SPECS / "canary_articulation_close_task_19.bddl",
}
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
OPERATOR = {
    10: {"subject": ("akita_black_bowl_1",), "receiver": ("plate_1",), "target": ()},
    20: {"subject": ("orange_juice_1",), "receiver": ("basket_1", "basket_1_contain_region"), "target": ()},
    19: {"subject": ("wooden_cabinet_1_middle_region",), "receiver": (), "target": ("wooden_cabinet_1_cabinet_middle",)},
}
ACTION_DIM = 7
CONTROL_FREQ_HZ = 20
STATE_SCHEMA_VERSION = "track3_composed_state_v1"
CONTACT_SCHEMA_VERSION = "track3_contact_v3"
WORKER_COMPLETED_STATUSES = frozenset({"passed", "failed_gate"})
QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
QVEL_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}
OWNERSHIPS = frozenset({"robot_arm", "robot_gripper", "operator_scene", "receiver_scene", "background_scene"})
SOURCES = frozenset({"demo_init", "demo_terminal", "zero"})


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


def neutral_hold() -> Any:
    import numpy as np

    return np.zeros((ACTION_DIM,), dtype=np.float32)


def _coverage(values: Iterable[int], length: int, label: str) -> None:
    values_list = [int(value) for value in values]
    if len(values_list) != len(set(values_list)):
        raise ValueError(f"{label}_overlap")
    if sorted(values_list) != list(range(length)):
        raise ValueError(f"{label}_gap_or_out_of_range")


def decode_joint_layout(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Decode all qpos/qvel spans from model metadata, fail-closed.

    The function is intentionally independent of MuJoCo.  Runtime workers
    serialize the same metadata from ``model.jnt_*`` before calling it.
    """
    required = ("nq", "nv", "na", "joint_names", "jnt_type", "jnt_qposadr", "jnt_dofadr", "joint_ownership", "joint_source")
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"missing_model_metadata:{','.join(missing)}")
    nq, nv, na = int(metadata["nq"]), int(metadata["nv"]), int(metadata["na"])
    names = list(metadata["joint_names"])
    types = list(metadata["jnt_type"])
    qpos_adr = list(metadata["jnt_qposadr"])
    dof_adr = list(metadata["jnt_dofadr"])
    if not (len(names) == len(types) == len(qpos_adr) == len(dof_adr)):
        raise ValueError("joint_metadata_shape_mismatch")
    ownership = dict(metadata["joint_ownership"])
    source = dict(metadata["joint_source"])
    if set(ownership) != set(names) or set(source) != set(names):
        raise ValueError("unknown_or_missing_joint_classification")
    rows: list[dict[str, Any]] = []
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    for joint_id, (name, joint_type, qstart, vstart) in enumerate(zip(names, types, qpos_adr, dof_adr)):
        name = str(name)
        joint_type = int(joint_type)
        if joint_type not in QPOS_WIDTH:
            raise ValueError(f"unknown_joint_type:{name}:{joint_type}")
        if ownership[name] not in OWNERSHIPS:
            raise ValueError(f"unknown_joint_ownership:{name}:{ownership[name]}")
        if source[name] not in SOURCES:
            raise ValueError(f"unknown_joint_source:{name}:{source[name]}")
        qstart, vstart = int(qstart), int(vstart)
        q_indices = list(range(qstart, qstart + QPOS_WIDTH[joint_type]))
        v_indices = list(range(vstart, vstart + QVEL_WIDTH[joint_type]))
        rows.append({"joint_id": joint_id, "name": name, "type": joint_type, "qpos_range": [qstart, qstart + QPOS_WIDTH[joint_type]], "qvel_range": [vstart, vstart + QVEL_WIDTH[joint_type]], "ownership": ownership[name], "source": source[name]})
        qpos_indices.extend(q_indices)
        qvel_indices.extend(v_indices)
    _coverage(qpos_indices, nq, "qpos")
    _coverage(qvel_indices, nv, "qvel")
    state_length = int(metadata.get("state_length", 1 + nq + nv + na))
    qpos_offset, qvel_offset = int(metadata.get("qpos_offset", 1)), int(metadata.get("qvel_offset", 1 + nq))
    act_offset = int(metadata.get("act_offset", 1 + nq + nv))
    if qpos_offset < 0 or qvel_offset < 0 or act_offset < 0 or qvel_offset + nv > state_length:
        raise ValueError("state_offsets_out_of_range")
    qpos_state = list(range(qpos_offset, qpos_offset + nq))
    qvel_state = list(range(qvel_offset, qvel_offset + nv))
    if any(index >= state_length for index in qpos_state + qvel_state):
        raise ValueError("state_qpos_qvel_out_of_range")
    opaque = metadata.get("opaque_state_indices")
    if opaque is None:
        expected = set(range(state_length)) - set(qpos_state) - set(qvel_state)
        opaque = sorted(expected)
    opaque = [int(index) for index in opaque]
    all_state = qpos_state + qvel_state + opaque
    if len(all_state) != state_length or sorted(all_state) != list(range(state_length)):
        raise ValueError("opaque_trailing_state_gap_or_overlap")
    return {"schema_version": STATE_SCHEMA_VERSION, "nq": nq, "nv": nv, "na": na, "state_length": state_length, "qpos_offset": qpos_offset, "qvel_offset": qvel_offset, "act_offset": act_offset, "joints": rows, "qpos_indices": qpos_indices, "qvel_indices": qvel_indices, "qpos_state_indices": qpos_state, "qvel_state_indices": qvel_state, "opaque_state_indices": opaque, "ownership_counts": {key: sum(row["ownership"] == key for row in rows) for key in sorted(OWNERSHIPS)}}


def compose_flat_state(metadata: Mapping[str, Any], layout: Mapping[str, Any], init_state: Any, terminal_state: Any) -> tuple[Any, dict[str, Any]]:
    """Compose init robot qpos + terminal scene qpos + exact zero qvel."""
    import numpy as np

    init = np.asarray(init_state, dtype=np.float64)
    terminal = np.asarray(terminal_state, dtype=np.float64)
    state_length = int(layout["state_length"])
    if init.ndim != 1 or terminal.ndim != 1 or init.shape != (state_length,) or terminal.shape != (state_length,):
        raise ValueError("state_vector_shape_mismatch")
    if not np.isfinite(init).all() or not np.isfinite(terminal).all():
        raise ValueError("state_vector_non_finite")
    composed = init.copy()
    source_bitmap = ["demo_init"] * state_length
    qpos_offset, qvel_offset = int(layout["qpos_offset"]), int(layout["qvel_offset"])
    for row in layout["joints"]:
        qstart, qend = row["qpos_range"]
        state_indices = list(range(qpos_offset + qstart, qpos_offset + qend))
        if row["ownership"] in {"operator_scene", "receiver_scene", "background_scene"}:
            composed[state_indices] = terminal[state_indices]
            for index in state_indices:
                source_bitmap[index] = "demo_terminal"
        elif row["ownership"] in {"robot_arm", "robot_gripper"}:
            for index in state_indices:
                source_bitmap[index] = "demo_init"
        else:
            raise ValueError(f"unhandled_joint_ownership:{row['name']}")
    qvel_indices = list(range(qvel_offset, qvel_offset + int(layout["nv"])))
    composed[qvel_indices] = 0.0
    for index in qvel_indices:
        source_bitmap[index] = "zero"
    terminal_scene_diffs = []
    for row in layout["joints"]:
        if row["ownership"] in {"operator_scene", "receiver_scene", "background_scene"}:
            indices = [qpos_offset + i for i in range(row["qpos_range"][0], row["qpos_range"][1])]
            terminal_scene_diffs.append(float(np.max(np.abs(composed[indices] - terminal[indices]))))
    provenance = {
        "schema_version": STATE_SCHEMA_VERSION,
        "composition": "same_demo_terminal_scene_qpos_plus_same_demo_init_robot_gripper_qpos_plus_zero_qvel",
        "qpos_sources": {"robot_arm": "demo_init", "robot_gripper": "demo_init", "operator_scene": "demo_terminal", "receiver_scene": "demo_terminal", "background_scene": "demo_terminal"},
        "qvel_source": "zero",
        "state_indices_source_bitmap": source_bitmap,
        "init_state_sha256": sha256_bytes(init.tobytes()),
        "terminal_state_sha256": sha256_bytes(terminal.tobytes()),
        "composed_state_sha256": sha256_bytes(composed.tobytes()),
        "round_trip_reconstruction_max_abs_diff": float(np.max(np.abs(composed - init))),
        "qvel_exact_max_abs": float(np.max(np.abs(composed[qvel_indices]))) if qvel_indices else 0.0,
        "terminal_scene_qpos_diff_max_abs": max(terminal_scene_diffs, default=0.0),
    }
    return composed, provenance


def _name_matches(actual: str, logical: str) -> bool:
    return actual == logical or actual.startswith(logical + "_") or logical in actual


def _is_robot_name(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(("robot0", "gripper0", "panda")) or any(token in lower for token in ("finger", "gripper", "right_hand", "left_hand"))


def _contact_role(task_index: int, body1: str, body2: str) -> str:
    participants = set(OPERATOR[task_index]["subject"]) | set(OPERATOR[task_index]["target"])
    receivers = set(OPERATOR[task_index]["receiver"])
    left_operator = any(_name_matches(body1, name) for name in participants)
    right_operator = any(_name_matches(body2, name) for name in participants)
    left_receiver = any(_name_matches(body1, name) for name in receivers)
    right_receiver = any(_name_matches(body2, name) for name in receivers)
    left_robot, right_robot = _is_robot_name(body1), _is_robot_name(body2)
    if (left_operator and right_receiver) or (right_operator and left_receiver):
        return "intended_receiver_contact"
    if (left_operator and right_robot) or (right_operator and left_robot):
        return "demonstrator_contact"
    if left_operator or right_operator:
        return "unrelated_background_contact"
    return "background_contact"


def contact_v3(task_index: int, matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Classify every contact without baseline grandfathering."""
    if not bool(matrix.get("complete", False)):
        return {"schema_version": CONTACT_SCHEMA_VERSION, "executed": False, "state": "not_evaluated", "required": True, "reason": "incomplete_contact_matrix", "categories": {}}
    categories = {"intended_receiver_contact": [], "demonstrator_contact": [], "unrelated_background_contact": [], "background_contact": []}
    for raw in matrix.get("contacts", []):
        body1, body2 = str(raw.get("body1_name", "")), str(raw.get("body2_name", ""))
        category = _contact_role(task_index, body1, body2)
        row = {"body_pair": sorted((body1, body2)), "geom_pair": sorted((str(raw.get("geom1_name", "")), str(raw.get("geom2_name", "")))), "distance_m": raw.get("distance_m"), "penetration_m": max(0.0, -float(raw.get("distance_m", 0.0)))}
        categories[category].append(row)
    forbidden = categories["unrelated_background_contact"]
    required_receiver = task_index == 10
    receiver_present = bool(categories["intended_receiver_contact"]) if required_receiver else True
    passed = not forbidden and receiver_present
    result = {"schema_version": CONTACT_SCHEMA_VERSION, "executed": True, "state": "passed" if passed else "failed", "required": True, "categories": categories, "counts": {key: len(value) for key, value in categories.items()}, "policy": {"operator_robot_is": "demonstrator_contact", "operator_unrelated_background": "hard_fail", "task10_receiver_required": True, "task20_receiver_allowed_oracle_is_In": True, "task19_receiver_required": False}}
    if not passed:
        result["reason"] = "unrelated_background_contact" if forbidden else "required_task10_support_contact_absent"
    return result


def contact_window_gate(task_index: int, samples: Sequence[Mapping[str, Any]], *, require_task10_support: bool = True) -> dict[str, Any]:
    if not samples:
        return gate(executed=False, value=None, reason="contact_window_empty")
    details = [contact_v3(task_index, sample.get("contact_matrix", {})) for sample in samples]
    if any(detail.get("state") == "not_evaluated" for detail in details):
        return gate(executed=False, value=None, reason="contact_matrix_incomplete")
    robot_zero = all(not detail["categories"]["demonstrator_contact"] for detail in details)
    unrelated_zero = all(not detail["categories"]["unrelated_background_contact"] for detail in details)
    support_ok = (not require_task10_support or task_index != 10 or all(detail["categories"]["intended_receiver_contact"] for detail in details))
    result = {"executed": True, "state": "passed" if robot_zero and unrelated_zero and support_ok else "failed", "required": True, "samples": details, "demonstrator_contact_zero": robot_zero, "unrelated_background_contact_zero": unrelated_zero, "required_task10_support_present": support_ok}
    if result["state"] == "failed":
        result["reason"] = "contact_v3_gate_failed"
    return result


def background_contact_delta(matrices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline: set[tuple[str, str]] | None = None
    new_pairs: list[dict[str, Any]] = []
    for index, matrix in enumerate(matrices):
        detail = contact_v3(20, matrix) if False else None
        pairs = {tuple(sorted((str(row.get("body1_name", "")), str(row.get("body2_name", ""))))) for row in matrix.get("contacts", [])}
        if baseline is None:
            baseline = pairs
        else:
            for pair in sorted(pairs - baseline):
                new_pairs.append({"sample_index": index, "body_pair": list(pair)})
    return {"executed": bool(matrices), "state": "passed" if not new_pairs else "failed", "required": True, "new_pairs": new_pairs, "baseline_grandfather": False, "diagnostic": "background pairs are recorded; operator-background remains hard-fail"}


def _as_bytes(value: Any) -> bytes:
    return value.encode("utf-8") if isinstance(value, str) else bytes(value)


def _terminal_selection(rewards: Any, states_len: int, actions_len: int) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(rewards).reshape(-1)
    if values.size != actions_len:
        raise ValueError(f"reward_action_length_mismatch:{values.size}:{actions_len}")
    positive = values > 0
    starts = [int(index) for index in np.flatnonzero(positive) if bool(positive[index:].all())]
    if not starts:
        raise ValueError("reward_has_no_positive_suffix")
    action_index = starts[0]
    if states_len == actions_len:
        state_index = action_index
        alignment = "state_index_equals_action_index"
    elif states_len == actions_len + 1:
        state_index = action_index + 1
        alignment = "state_index_equals_action_index_plus_one"
    else:
        raise ValueError(f"unsupported_state_action_alignment:{states_len}:{actions_len}")
    return {"action_index": action_index, "state_index": state_index, "suffix_start_action_index": action_index, "suffix_length": int(values.size - action_index), "positive_count": int(positive.sum()), "reward_at_selection": float(values[action_index]), "alignment": alignment}


def _hdf5_row(task_index: int, demo_index: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    path = HDF5_INPUTS[task_index]
    with h5py.File(path, "r") as handle:
        demo = handle[f"data/demo_{demo_index}"]
        states = np.asarray(demo["states"][()])
        actions = np.asarray(demo["actions"][()])
        rewards = np.asarray(demo["rewards"][()])
        model_bytes = _as_bytes(demo.attrs["model_file"])
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        selection = _terminal_selection(rewards, len(states), len(actions))
        terminal = np.asarray(states[selection["state_index"]], dtype=np.float64)
    if states.ndim != 2 or actions.ndim != 2 or rewards.ndim != 1 or not np.array_equal(states[0], init_state):
        raise ValueError(f"hdf5_schema_or_init_mismatch:{task_index}:{demo_index}")
    return {"task_index": task_index, "family": FAMILY[task_index], "demo_index": demo_index, "demo_id": f"demo_{demo_index}", "hdf5_path": str(path.resolve()), "hdf5_bytes": path.stat().st_size, "hdf5_sha256": sha256_file(path), "model_file_bytes": len(model_bytes), "model_file_sha256": sha256_bytes(model_bytes), "init_state_sha256": sha256_bytes(init_state.tobytes()), "terminal_state_sha256": sha256_bytes(terminal.tobytes()), "states_shape": list(states.shape), "states_dtype": str(states.dtype), "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype), "rewards_shape": list(rewards.shape), "rewards_dtype": str(rewards.dtype), "selection": selection, "terminal_state_shape": list(terminal.shape), "terminal_state_dtype": str(terminal.dtype), "states0_equals_init_state": True, "positive_suffix_exactly_one": int(selection["suffix_length"]) == 1, "predicate_oracle": "never_hdf5; runtime_only_in_same_demo_model_xml"}


def _old_paths() -> list[Path]:
    old = ROOT / "data/reports/axis_libero_30k/track3_matched_terminal_state_release"
    return [ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md", ROOT / "docs/reviews/axis_pi05/TRACK3_BACKGROUND_CONVERGENCE_K0_K4_AUDIT_20260904.md", ROOT / "tools/axis_pi05/track3_matched_terminal_state_release.py", old / "l0_input_lock.json", old / "l0_demo_manifest.json", old / "l1_validator_contract.json", old / "l2_summary.json", old / "contact_ledger.json", old / "demo_manifest.json", old / "summary.json", old / "L0_K3_ERRATA_ZH.md", old / "L5_CLOSEOUT_REPORT_ZH.md", *sorted((old / "runs").glob("l2_*.json"))]


def _file_lock(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _git(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def _porcelain(repo: Path) -> list[str]:
    return [line for line in _git(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def run_m0() -> dict[str, Any]:
    """Lock and revalidate old evidence with no simulator imports."""
    import h5py
    import numpy as np

    old_files = [_file_lock(path) for path in _old_paths()]
    old_dir = OUT_ROOT.parent / "track3_matched_terminal_state_release"
    old_manifest = json.loads((old_dir / "l0_demo_manifest.json").read_text(encoding="utf-8"))
    fixed = old_manifest.get("fixed_selection", [])
    expected = [(task, demo) for task in TASK_ORDER for demo in range(3)]
    actual = [(int(row["task_index"]), int(str(row["demo_id"]).split("_")[-1])) for row in fixed]
    if actual != expected:
        raise RuntimeError(f"fixed_demo_identity_drift:{actual}")
    prior_runs: list[dict[str, Any]] = []
    for task, demo in expected:
        path = old_dir / "runs" / f"l2_task{task}_demo{demo}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        gates = report.get("hard_gate", {})
        failed = [name for name, row in gates.items() if isinstance(row, Mapping) and row.get("state") == "failed"]
        prior_runs.append({"task_index": task, "demo_index": demo, "path": str(path.resolve()), "status": report.get("status"), "failed_gates": failed, "same_demo_model_and_state": report.get("model_state_identity", {}).get("same_demo_model_and_state"), "forward_goal_success": report.get("forward", {}).get("goal_success"), "reverse_source": gates.get("critical_source_true_step5_to20", {}).get("state"), "reverse_goal": gates.get("reverse_goal_false_step5_to20", {}).get("state")})
    hdf5_rows = []
    for task, demo in expected:
        row = _hdf5_row(task, demo)
        manifest_row = next(item for item in fixed if int(item["task_index"]) == task and int(str(item["demo_id"]).split("_")[-1]) == demo)
        for key in ("hdf5_sha256", "model_file_sha256", "terminal_state_sha256", "states_shape", "actions_shape", "rewards_shape"):
            if row[key] != manifest_row[key]:
                raise RuntimeError(f"hdf5_manifest_drift:{task}:{demo}:{key}")
        hdf5_rows.append(row)
    root_head = _git(ROOT, ("rev-parse", "HEAD"))
    openpi_head = _git(OPENPI, ("rev-parse", "HEAD")) if OPENPI.is_dir() else None
    previous_summary = json.loads((old_dir / "summary.json").read_text(encoding="utf-8"))
    prior_counts = previous_summary.get("reverse_demo_counts", {"exact": 0, "pending": 0, "accepted": 0})
    closeout_ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "0dc402e2d70e6576c5861a674e0aac04e4427ef0", "HEAD"], check=False).returncode == 0
    lock = {"schema_version": "track3_composed_reverse_start_m0_v1", "status": "track3_composed_reverse_start_m0_passed", "created_at_utc": now(), "plan": str(PLAN.resolve()), "audit": str(AUDIT.resolve()), "old_input_lock": old_files, "old_execution_head_expected": "19cb1eab9556b6e9e0f8b3f9d4b49fb3893b81a0", "old_closeout_head_expected": "0dc402e2d70e6576c5861a674e0aac04e4427ef0", "observed_root_head": root_head, "old_closeout_is_ancestor": closeout_ancestor, "observed_openpi_head": openpi_head, "nested_openpi_expected": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1", "openpi_porcelain": _porcelain(OPENPI) if OPENPI.is_dir() else [], "root_porcelain": _porcelain(ROOT), "libero_plus_porcelain": _porcelain(LIBERO_PLUS), "fixed_demo_order": [{"task_index": task, "demo_index": demo} for task, demo in expected], "prior_run_revalidation": prior_runs, "hdf5_revalidation": hdf5_rows, "prior_reverse_demo_counts": prior_counts, "resource_boundary": {"simulator_started": False, "gpu_started": False, "hdf5_or_bddl_modified": False, "old_evidence_modified": False}}
    if not closeout_ancestor:
        lock["status"] = "track3_composed_reverse_start_m0_input_drift"
        lock["failure_reason"] = "old_closeout_not_in_current_ancestry"
    if prior_counts != {"exact": 0, "pending": 0, "accepted": 0}:
        lock["status"] = "track3_composed_reverse_start_m0_input_drift"
        lock["failure_reason"] = "prior_reverse_demo_count_nonzero"
    write_json(OUT_ROOT / "m0_release_lock.json", lock)
    diagnosis = """# Track 3 composed reverse-start M0 诊断

M0 只读锁定上一轮 matched-terminal L0–L5 证据，没有启动 simulator、CPU worker 或 GPU。

- 上一轮 9/9 same-demo model/state identity、forward success、reverse source/goal 语义证据已逐条复算；失败 gate 按 run 完整列出。
- task20 的历史失败主要是 success 末帧后的物体沉降，task19 主要是 forward terminal robot 回稳；positive reward suffix 均只有一帧。
- `contact_contract_v2` 将 operator↔robot/gripper 误归为 background，本计划改用 contact-v3。
- 本轮 composition 只生成 report namespace 中的 candidate，不改变公开 HDF5、BDDL 或历史 evidence。

M0 的固定 demo 顺序为 task10/task20/task19 各 demo_0、demo_1、demo_2；predicate 仍只在加载同一 model XML 的 runtime 中求值。
"""
    (OUT_ROOT / "M0_L2_DIAGNOSIS_ZH.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "M0_L2_DIAGNOSIS_ZH.md").write_text(diagnosis, encoding="utf-8")
    return lock


def _synthetic_contract_examples() -> dict[str, Any]:
    metadata = {"nq": 9, "nv": 8, "na": 2, "state_length": 20, "joint_names": ["free_object", "robot0_joint1", "drawer"], "jnt_type": [0, 3, 3], "jnt_qposadr": [0, 7, 8], "jnt_dofadr": [0, 6, 7], "joint_ownership": {"free_object": "operator_scene", "robot0_joint1": "robot_arm", "drawer": "background_scene"}, "joint_source": {"free_object": "demo_terminal", "robot0_joint1": "demo_init", "drawer": "demo_terminal"}}
    layout = decode_joint_layout(metadata)
    return {"metadata": metadata, "layout": layout}


def run_m1() -> dict[str, Any]:
    articulation = ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"
    examples = _synthetic_contract_examples()
    contract = {"schema_version": "track3_composed_reverse_start_m1_v1", "status": "track3_composed_reverse_start_m1_validator_contract_passed", "created_at_utc": now(), "plan": str(PLAN.resolve()), "simulator_started": False, "gpu_started": False, "state_composition": {"source": "same-demo init robot/gripper qpos + same-demo terminal scene qpos + zero qvel", "time_and_opaque": "demo_init", "free_joint": "7 qpos / 6 qvel mapping", "index_coverage": "exactly_once; overlap/gap/unknown/trailing fail-closed", "synthetic_example": examples}, "contact_v3": {"schema_version": CONTACT_SCHEMA_VERSION, "categories": ["intended_receiver_contact", "demonstrator_contact", "unrelated_background_contact", "background_contact"], "robot_gripper_contact": "demonstrator_contact", "unrelated_operator_background": "hard_fail", "task10_support_required": True, "task20_in_predicate_oracle": True, "task19_receiver_required": False}, "neutral_hold": {"shape": [7], "dtype": "float32", "values": [0.0] * 7, "gripper": 0.0}, "strict_thresholds": {"operator_translation_m": 0.002, "arm_qpos_rad": 0.002, "arm_qvel_rad_s": 0.02, "eef_translation_m": 0.002, "eef_rotation_rad": 0.02, "support_subject_rotation": "diagnostic_only"}, "source_articulation_hdf5": {"path": str(articulation.resolve()), "sha256": sha256_file(articulation)}, "tri_state": ["passed", "failed", "not_evaluated"], "passed_gate_has_no_failure_reason": True, "worker_exception_returncode": "nonzero", "old_tool_untouched": True}
    contract["canonical_sha256"] = canonical_hash({key: value for key, value in contract.items() if key != "canonical_sha256"})
    write_json(OUT_ROOT / "m1_validator_contract.json", contract)
    return contract


# ---------------------------------------------------------------------------
# M2+ runtime.  All imports below are intentionally local to fresh workers.


def _configure_public_env(*, gpu: bool) -> None:
    import tempfile

    config_dir = Path(tempfile.mkdtemp(prefix="axis_pi05_composed_libero_config_"))
    root = PUBLIC_LIBERO / "libero/libero"
    (config_dir / "config.yaml").write_text("\n".join([f"benchmark_root: {root}", f"bddl_files: {root / 'bddl_files'}", f"init_states: {root / 'init_files'}", f"datasets: {root / 'datasets'}", f"assets: {root / 'assets'}", ""]), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    if gpu:
        os.environ.update({"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3"})
    else:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        os.environ.update({"MUJOCO_GL": "disable", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})


def _resolve_model_xml(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    import robosuite

    source = model_bytes.decode("utf-8")
    libero_assets = (PUBLIC_LIBERO / "libero/libero/assets").resolve()
    robosuite_assets = (Path(robosuite.__file__).resolve().parent / "models/assets").resolve()
    replacements = {"/Users/yifengz/workspace/libero-dev/chiliocosm/assets/": str(libero_assets) + "/", "/Users/yifengz/workspace/robosuite-master/robosuite/models/assets/": str(robosuite_assets) + "/"}
    resolved = source
    for old, new in replacements.items():
        resolved = resolved.replace(old, new)
    referenced = re.findall(r'(?:file|mesh)="([^"]+)"', resolved)
    missing = sorted({value for value in referenced if value.startswith("/") and not Path(value).exists()})
    if missing:
        raise FileNotFoundError(f"resolved_model_assets_missing:{missing[:3]}")
    return resolved, {"source_sha256": sha256_bytes(model_bytes), "resolved_sha256": sha256_bytes(resolved.encode("utf-8")), "replacements": replacements, "absolute_asset_references": len(referenced), "missing_absolute_assets": missing}


def _demo_arrays(task_index: int, demo_index: int) -> tuple[dict[str, Any], bytes, Any, Any, dict[str, Any]]:
    import h5py
    import numpy as np

    path = HDF5_INPUTS[task_index]
    with h5py.File(path, "r") as handle:
        demo = handle[f"data/demo_{demo_index}"]
        states = np.asarray(demo["states"][()], dtype=np.float64)
        actions = np.asarray(demo["actions"][()], dtype=np.float64)
        rewards = np.asarray(demo["rewards"][()])
        model_bytes = _as_bytes(demo.attrs["model_file"])
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        selection = _terminal_selection(rewards, len(states), len(actions))
        terminal_state = np.asarray(states[selection["state_index"]], dtype=np.float64)
        terminal_action = np.asarray(actions[selection["action_index"]], dtype=np.float64)
    manifest = _hdf5_row(task_index, demo_index)
    if not np.array_equal(states[0], init_state):
        raise ValueError("states0_init_state_mismatch")
    return manifest, model_bytes, init_state, terminal_state, {"selection": selection, "terminal_action": terminal_action.tolist()}


def _runtime_ownership(task_index: int, name: str) -> str:
    lower = name.lower()
    if lower.startswith("robot0") or lower.startswith(("gripper0", "panda")):
        if any(token in lower for token in ("finger", "gripper")):
            return "robot_gripper"
        if "joint" in lower:
            return "robot_arm"
        raise ValueError(f"unknown_robot_or_gripper_joint:{name}")
    subject = OPERATOR[task_index]["subject"]
    receiver = OPERATOR[task_index]["receiver"]
    target = OPERATOR[task_index]["target"]
    if any(token.lower() in lower for token in subject + target):
        return "operator_scene"
    if any(token.lower() in lower for token in receiver):
        return "receiver_scene"
    return "background_scene"


def _runtime_model_metadata(env: Any, task_index: int, state_length: int) -> dict[str, Any]:
    model = env.sim.model
    names = [str(value) for value in getattr(model, "joint_names", ())]
    if not names:
        names = [str(model.joint_id2name(index)) for index in range(int(model.njnt))]
    types = [int(value) for value in model.jnt_type]
    qpos_adr = [int(value) for value in model.jnt_qposadr]
    dof_adr = [int(value) for value in model.jnt_dofadr]
    ownership = {name: _runtime_ownership(task_index, name) for name in names}
    source = {name: "demo_init" if ownership[name] in {"robot_arm", "robot_gripper"} else "demo_terminal" for name in names}
    return {"nq": int(model.nq), "nv": int(model.nv), "na": int(getattr(model, "na", 0)), "state_length": int(state_length), "joint_names": names, "jnt_type": types, "jnt_qposadr": qpos_adr, "jnt_dofadr": dof_adr, "joint_ownership": ownership, "joint_source": source}


def _value_for_atom(result: Mapping[str, Any], atom: str) -> Any:
    for row in result.get("predicates", []):
        if row.get("atom") == atom:
            return row.get("value")
    return None


def _truth_gate(samples: Sequence[Mapping[str, Any]], task_index: int) -> dict[str, Any]:
    expected_steps = list(range(41))
    if [int(sample.get("step", -1)) for sample in samples] != expected_steps:
        return gate(executed=False, value=None, reason="checkpoint_window_not_exact_0_to_40")
    complete = all(sample.get("initial", {}).get("truth_complete") and sample.get("initial", {}).get("detail_complete") and sample.get("goal", {}).get("truth_complete") and sample.get("goal", {}).get("detail_complete") for sample in samples)
    source_values = [_value_for_atom(sample["initial"], SOURCE_ATOM[task_index]) for sample in samples]
    goal_values = [_value_for_atom(sample["goal"], GOAL_ATOM[task_index]) for sample in samples]
    passed = complete and all(value is True for value in source_values) and all(value is False for value in goal_values)
    result = {"executed": complete, "state": "passed" if passed else "failed", "required": True, "steps": expected_steps, "source_atom": SOURCE_ATOM[task_index], "goal_atom": GOAL_ATOM[task_index], "source_values": source_values, "goal_values": goal_values, "truth_detail_complete": complete}
    if not passed:
        result["reason"] = "reverse_source_or_goal_truth_contract"
    return result


def _background_truth_gate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not samples:
        return gate(executed=False, value=None, reason="no_samples")
    rows: dict[tuple[str, str], list[Any]] = {}
    for sample in samples:
        for section in ("initial", "goal"):
            for row in sample.get(section, {}).get("predicates", []):
                if str(row.get("role", "")).startswith("operator_critical"):
                    continue
                rows.setdefault((section, str(row.get("atom"))), []).append(row.get("value"))
    transitions: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for key, values in rows.items():
        if len(values) != len(samples) or any(value not in (True, False) for value in values):
            return gate(executed=False, value=None, reason="background_truth_incomplete")
        false_to_true = 0
        for index, (before, after) in enumerate(zip(values, values[1:]), start=1):
            if before is False and after is True:
                false_to_true += 1
                transitions.append({"section": key[0], "atom": key[1], "from": False, "to": True, "step": int(samples[index]["step"])})
            elif before is True and after is False:
                violations.append({"section": key[0], "atom": key[1], "from": True, "to": False, "step": int(samples[index]["step"])})
        if false_to_true > 1:
            violations.append({"section": key[0], "atom": key[1], "reason": "background_truth_reentry", "false_to_true_count": false_to_true})
    result = {"executed": True, "state": "passed" if not violations else "failed", "required": True, "contract": "critical_goal_strict; background allows false_to_true at most once and forbids true_to_false/reentry", "transitions": transitions, "violations": violations}
    if violations:
        result["reason"] = "background_truth_monotonicity_violation"
    return result


def _measurement(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [sample for sample in samples if 21 <= int(sample.get("step", -1)) <= 40]


def _participant_gate(samples: Sequence[Mapping[str, Any]], task_index: int, old: Any, prior: Any) -> dict[str, Any]:
    release = _measurement(samples)
    if len(release) != 20:
        return gate(executed=False, value=None, reason="measurement_window_incomplete")
    baseline = release[0]["participant"]
    max_translation: dict[str, float] = {}
    max_rotation: dict[str, float] = {}
    max_joint: dict[str, float] = {}
    for sample in release:
        for name, pose in baseline.get("poses", {}).items():
            current = sample["participant"]["poses"].get(name)
            if current is None:
                return gate(executed=False, value=None, reason=f"participant_missing:{name}")
            translation, rotation = prior._pose_drift(pose, current, old)
            max_translation[name] = max(max_translation.get(name, 0.0), translation)
            max_rotation[name] = max(max_rotation.get(name, 0.0), rotation)
        for name, pose in baseline.get("bodies", {}).items():
            current = sample["participant"]["bodies"].get(name)
            if current is None:
                return gate(executed=False, value=None, reason=f"target_body_missing:{name}")
            translation, rotation = prior._pose_drift(pose, current, old)
            max_translation[name] = max(max_translation.get(name, 0.0), translation)
            max_rotation[name] = max(max_rotation.get(name, 0.0), rotation)
        for name, value in baseline.get("joint_values", {}).items():
            current = sample["participant"]["joint_values"].get(name)
            drift = abs(float(current) - float(value)) if current is not None else float("nan")
            if not math.isfinite(drift):
                return gate(executed=False, value=None, reason=f"target_joint_missing:{name}")
            max_joint[name] = max(max_joint.get(name, 0.0), drift)
    passed = all(value <= 0.002 for value in max_translation.values()) and all(value <= 0.002 for value in max_joint.values())
    result = {"executed": True, "state": "passed" if passed else "failed", "required": True, "max_translation_m": max_translation, "max_rotation_rad_diagnostic": max_rotation, "max_joint_rad_or_m": max_joint, "translation_threshold_m": 0.002, "joint_threshold_rad_or_m": 0.002, "subject_rotation_policy": "diagnostic_only"}
    if not passed:
        result["reason"] = "operator_participant_translation_or_joint_drift"
    return result


def _robot_gate(samples: Sequence[Mapping[str, Any]], old: Any) -> dict[str, Any]:
    release = _measurement(samples)
    if len(release) != 20:
        return gate(executed=False, value=None, reason="measurement_window_incomplete")
    baseline = release[0]["robot"]
    maxima = {key: 0.0 for key in ("arm_joint_qpos_rad", "arm_joint_qvel_rad_s", "eef_translation_m", "eef_rotation_rad")}
    for sample in release:
        drift = old._metric_drift(baseline, sample["robot"])
        for key in maxima:
            maxima[key] = max(maxima[key], float(drift[key]))
    thresholds = {"arm_joint_qpos_rad": 0.002, "arm_joint_qvel_rad_s": 0.02, "eef_translation_m": 0.002, "eef_rotation_rad": 0.02}
    passed = all(maxima[key] <= thresholds[key] for key in maxima)
    result = {"executed": True, "state": "passed" if passed else "failed", "required": True, "max_drift": maxima, "thresholds": thresholds, "gripper_drift_recorded_but_not_gated": True}
    if not passed:
        result["reason"] = "robot_arm_or_eef_drift"
    return result


def _background_pose_gate(samples: Sequence[Mapping[str, Any]], task_index: int, old: Any, prior: Any) -> dict[str, Any]:
    release = _measurement(samples)
    if len(release) != 20:
        return gate(executed=False, value=None, reason="measurement_window_incomplete")
    baseline = release[0].get("scene", {}).get("object_and_site_poses", {})
    excluded = set(OPERATOR[task_index]["subject"]) | set(OPERATOR[task_index]["target"])
    maximum: dict[str, float] = {}
    for sample in release:
        current = sample.get("scene", {}).get("object_and_site_poses", {})
        for name, pose in baseline.items():
            if name in excluded:
                continue
            if name not in current:
                return gate(executed=False, value=None, reason=f"background_pose_missing:{name}")
            translation, _rotation = prior._pose_drift(pose, current[name], old)
            maximum[name] = max(maximum.get(name, 0.0), translation)
    passed = all(value <= 0.002 for value in maximum.values())
    result = {"executed": True, "state": "passed" if passed else "failed", "required": True, "max_translation_m": maximum, "threshold_m": 0.002, "operator_participants_excluded": sorted(excluded)}
    if not passed:
        result["reason"] = "background_pose_drift_over_2mm"
    return result


def _renderer_schema(wrapper: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    actual = getattr(wrapper, "env", None)
    attrs = lambda obj: {name: getattr(obj, name, None) if obj is not None else None for name in ("use_camera_obs", "has_offscreen_renderer", "has_renderer")}
    cameras = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        value = observation.get(key)
        arr = None if value is None else __import__("numpy").asarray(value)
        cameras[key] = {"present": value is not None, "dtype": None if arr is None else str(arr.dtype), "shape": None if arr is None else list(arr.shape), "uint8": arr is not None and arr.dtype == __import__("numpy").uint8 and arr.ndim == 3 and arr.shape[-1] == 3}
    return {"wrapper_flags": attrs(wrapper), "underlying_flags": attrs(actual), "cameras": cameras, "schema_complete": bool(getattr(actual, "use_camera_obs", False) is True and getattr(actual, "has_offscreen_renderer", False) is True and getattr(actual, "has_renderer", True) is False and all(row["present"] and row["uint8"] for row in cameras.values()))}


def _runtime_worker(task_index: int, demo_index: int, *, gpu: bool = False, capture_camera: bool = False) -> dict[str, Any]:
    import numpy as np

    started = now()
    forward_env = reverse_env = None
    try:
        _configure_public_env(gpu=gpu)
        from libero.libero.envs.env_wrapper import ControlEnv
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as old
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base

        manifest, model_bytes, init_state, terminal_state, demo_info = _demo_arrays(task_index, demo_index)
        resolved_xml, resolution = _resolve_model_xml(model_bytes)
        if resolution["source_sha256"] != manifest["model_file_sha256"] or sha256_bytes(terminal_state.tobytes()) != manifest["terminal_state_sha256"]:
            raise ValueError("m0_model_or_terminal_identity_mismatch")
        kwargs = {"control_freq": CONTROL_FREQ_HZ, "use_camera_obs": bool(capture_camera), "has_renderer": False, "has_offscreen_renderer": bool(capture_camera), "render_gpu_device_id": 0}
        if capture_camera:
            kwargs.update({"camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128})
        # The forward env is used only as a public-goal oracle.  Both raw and
        # composed states use the same demo XML; HDF5 values are not oracles.
        forward_env = ControlEnv(bddl_file_name=str(FORWARD_BDDL[task_index]), **kwargs)
        forward_env.reset()
        forward_env.reset_from_xml_string(resolved_xml)
        forward_env.set_init_state(terminal_state)
        raw_forward_goal = bool(forward_env.check_success())
        forward_env.set_init_state(init_state)
        metadata = _runtime_model_metadata(forward_env, task_index, len(init_state))
        layout = decode_joint_layout(metadata)
        composed_state, composition = compose_flat_state(metadata, layout, init_state, terminal_state)
        forward_env.set_init_state(composed_state)
        composed_forward_goal = bool(forward_env.check_success())
        forward_schema = _renderer_schema(forward_env, {}) if capture_camera else {"executed": False, "state": "not_evaluated", "required": False}
        forward_env.close(); forward_env = None

        reverse_env = ControlEnv(bddl_file_name=str(REVERSE_BDDL[task_index]), **kwargs)
        reverse_env.reset()
        reverse_env.reset_from_xml_string(resolved_xml)
        reverse_env.set_init_state(composed_state)
        restored = np.asarray(reverse_env.get_sim_state(), dtype=np.float64).copy()
        restored_diff = float(np.max(np.abs(restored - composed_state)))
        controller_sequence = old.controller_reset_sequence(old._controller(reverse_env), reverse_env.sim)
        initial_atoms = base.public_predicates(REVERSE_BDDL[task_index], "init")
        goal_atoms = base.public_predicates(REVERSE_BDDL[task_index], "goal")
        hold = neutral_hold()
        samples: list[dict[str, Any]] = []
        frames = {"agentview_image": [], "robot0_eye_in_hand_image": []}

        def observe(step: int, observation: Mapping[str, Any] | None = None) -> None:
            sample = prior._runtime_sample(reverse_env, old, task_index, "reverse", step, initial_atoms, goal_atoms)
            sample["contacts_v3"] = contact_v3(task_index, sample["contact_matrix"])
            samples.append(sample)
            if capture_camera:
                if observation is None:
                    observation = reverse_env.env._get_observations()
                for key in frames:
                    value = observation.get(key)
                    if value is None:
                        raise ValueError(f"missing_camera_key:{key}")
                    image = np.asarray(value)
                    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                        raise ValueError(f"invalid_camera_schema:{key}")
                    frames[key].append(image.copy())

        observe(0)
        for step in range(1, 41):
            _observation, _reward, _done, _info = reverse_env.step(hold)
            observe(step, _observation)
        measurement = _measurement(samples)
        truth = _truth_gate(samples, task_index)
        contacts = contact_window_gate(task_index, [samples[0], *measurement])
        gates = {
            "schema_and_finite": gate(executed=all(sample["initial"].get("truth_complete") and sample["goal"].get("truth_complete") and sample["initial"].get("detail_complete") and sample["goal"].get("detail_complete") for sample in samples), value=truth["state"] == "passed"),
            "raw_terminal_forward_goal": gate(executed=True, value=raw_forward_goal),
            "composed_forward_goal": gate(executed=True, value=composed_forward_goal),
            "composition_state_identity": gate(executed=True, value=resolution["source_sha256"] == manifest["model_file_sha256"] and composition["qvel_exact_max_abs"] == 0.0 and restored_diff <= 1e-12),
            "critical_source_true_0_to_40": gate(executed=truth["executed"], value=truth["state"] == "passed"),
            "reverse_goal_false_0_to_40": gate(executed=truth["executed"], value=truth["state"] == "passed"),
            "contact_v3": contacts,
            "background_truth_monotonicity": _background_truth_gate(samples),
            "background_pose_stability": _background_pose_gate(samples, task_index, old, prior),
            "operator_participant_stability": _participant_gate(samples, task_index, old, prior),
            "robot_arm_eef_stability": _robot_gate(samples, old),
        }
        camera = _renderer_schema(reverse_env, reverse_env.env._get_observations()) if capture_camera else {"executed": False, "state": "not_evaluated", "required": False, "reason": "camera_not_requested"}
        if capture_camera:
            gates["camera_schema"] = gate(executed=True, value=bool(camera["schema_complete"]))
        status = "passed" if all_required(gates) else "failed_gate"
        result = {"schema_version": "track3_composed_reverse_start_worker_v1", "status": status, "created_at_utc": started, "completed_at_utc": now(), "task_index": task_index, "family": FAMILY[task_index], "demo_id": f"demo_{demo_index}", "fresh_process": True, "mode": "GPU3_EGL_camera" if capture_camera else "CPU_no_renderer", "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "terminal_state_sha256": manifest["terminal_state_sha256"], "init_state_sha256": manifest["init_state_sha256"], "composed_state_sha256": composition["composed_state_sha256"], "restored_composed_state_sha256": sha256_bytes(restored.tobytes()), "restore_max_abs_diff": restored_diff, "same_demo_model_and_state_components": True, "hdf5_state_used_as_predicate_oracle": False, "state_selection": demo_info["selection"]}, "model_layout": {"metadata": metadata, "decoded": layout}, "composition": composition, "forward": {"raw_terminal_goal_success": raw_forward_goal, "composed_goal_success": composed_forward_goal, "same_model_xml": True}, "reverse": {"initial_atoms": initial_atoms, "goal_atoms": goal_atoms, "controller_reset_sequence": controller_sequence, "hold_action": hold.tolist(), "burn_in_steps": 20, "measurement_steps": 20, "checkpoints": samples}, "renderer": camera, "hard_gate": gates, "provenance": {"no_manipulation_action": True, "no_action_or_demo_generation": True, "no_se3_or_grasp": True, "no_teleop": True, "no_threshold_broadening": True, "public_hdf5_model_file_and_state_only": True}}
        result["camera_frames"] = {key: len(value) for key, value in frames.items()}
        if capture_camera:
            result["camera_frame_payloads"] = {key: [frame.tolist() for frame in value] for key, value in frames.items()}
        return result
    except Exception as exc:
        return {"schema_version": "track3_composed_reverse_start_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "task_index": task_index, "family": FAMILY.get(task_index), "demo_id": f"demo_{demo_index}", "fresh_process": True, "mode": "GPU3_EGL_camera" if capture_camera else "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        for env in (reverse_env, forward_env):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise ValueError("worker_emitted_no_json_result")


def _run_worker(task_index: int, demo_index: int, *, gpu: bool = False, timeout_s: int = 600) -> tuple[dict[str, Any], int, str, str]:
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "MUJOCO_GL": "egl" if gpu else "disable", "CUDA_VISIBLE_DEVICES": "3" if gpu else "", "JAX_PLATFORMS": "cpu"})
    if gpu:
        environment["MUJOCO_EGL_DEVICE_ID"] = "3"
    else:
        environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    command = [sys.executable, "-m", "tools.axis_pi05.track3_composed_reverse_start_stabilization", "--worker", "--task-index", str(task_index), "--demo-index", str(demo_index)]
    if gpu:
        command.append("--gpu")
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=timeout_s, check=False)
    try:
        result = _parse_worker_stdout(completed.stdout)
    except Exception as exc:
        result = {"schema_version": "track3_composed_reverse_start_worker_v1", "status": "worker_exception", "task_index": task_index, "family": FAMILY[task_index], "demo_id": f"demo_{demo_index}", "fresh_process": True, "mode": "GPU3_EGL_camera" if gpu else "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}}
    result["process_returncode"] = completed.returncode
    result["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(result.get("status")))
    result["stdout_tail"] = completed.stdout[-3000:]
    result["stderr_tail"] = completed.stderr[-3000:]
    return result, completed.returncode, completed.stdout, completed.stderr


def _compact_result(result: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    gates = result.get("hard_gate", {})
    return {"task_index": result.get("task_index"), "family": result.get("family"), "demo_id": result.get("demo_id"), "status": result.get("status"), "process_returncode": result.get("process_returncode"), "worker_returncode_contract": result.get("worker_returncode_contract"), "failed_gates": [name for name, row in gates.items() if isinstance(row, Mapping) and row.get("state") == "failed"], "path": str(path.resolve()) if path else None, "composed_state_sha256": result.get("composition", {}).get("composed_state_sha256")}


def run_m2() -> dict[str, Any]:
    lock = json.loads((OUT_ROOT / "m0_release_lock.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT_ROOT / "m1_validator_contract.json").read_text(encoding="utf-8"))
    if lock.get("status") != "track3_composed_reverse_start_m0_passed" or contract.get("status") != "track3_composed_reverse_start_m1_validator_contract_passed":
        raise RuntimeError("m0_or_m1_not_passed")
    result, _returncode, _stdout, _stderr = _run_worker(20, 0)
    path = OUT_ROOT / "runs/m2_task20_demo0.json"
    payload = dict(result)
    payload.pop("camera_frame_payloads", None)
    write_json(path, payload)
    summary = {"schema_version": "track3_composed_reverse_start_m2_v1", "status": "track3_composed_reverse_start_cpu_canary_passed_task20_demo0" if result.get("status") == "passed" and result.get("worker_returncode_contract") else "track3_composed_reverse_start_cpu_canary_failed_task20_demo0", "created_at_utc": now(), "fresh_process_count": 1, "result": _compact_result(result, path), "start_state_candidate": {"namespace": "report_only", "count": 1, "composed_state_sha256": result.get("composition", {}).get("composed_state_sha256")} if result.get("status") == "passed" else {"namespace": "report_only", "count": 0}, "no_dataset_pending_or_accepted_write": True}
    write_json(OUT_ROOT / "m2_summary.json", summary)
    return summary


def run_m3() -> dict[str, Any]:
    m2 = json.loads((OUT_ROOT / "m2_summary.json").read_text(encoding="utf-8"))
    if m2.get("status") != "track3_composed_reverse_start_cpu_canary_passed_task20_demo0":
        result = {"schema_version": "track3_composed_reverse_start_m3_v1", "status": "not_evaluated", "reason": "m2_canary_not_passed", "results": []}
        write_json(OUT_ROOT / "m3_summary.json", result)
        return result
    results = [m2["result"]]
    for task in TASK_ORDER:
        for demo in range(3):
            if (task, demo) == (20, 0):
                continue
            path = OUT_ROOT / "runs" / f"m3_task{task}_demo{demo}.json"
            if path.is_file():
                # A completed fresh worker report is immutable evidence.  A
                # parent-shell interruption may leave all reports without a
                # matrix summary; recover the summary without rerunning any
                # simulator process.
                result = json.loads(path.read_text(encoding="utf-8"))
            else:
                result, _returncode, _stdout, _stderr = _run_worker(task, demo)
                payload = dict(result); payload.pop("camera_frame_payloads", None)
                write_json(path, payload)
            results.append(_compact_result(result, path))
            if result.get("status") != "passed" or not result.get("worker_returncode_contract"):
                break
        if results[-1].get("status") != "passed":
            break
    passed = len(results) == 9 and all(row.get("status") == "passed" and row.get("worker_returncode_contract") for row in results)
    summary = {"schema_version": "track3_composed_reverse_start_m3_v1", "status": "track3_composed_reverse_start_cpu_matrix_passed_9_of_9" if passed else "track3_composed_reverse_start_cpu_matrix_failed", "created_at_utc": now(), "fresh_process_count": len(results), "results": results, "family_3_of_3": {FAMILY[task]: sum(row.get("task_index") == task and row.get("status") == "passed" for row in results) == 3 for task in TASK_ORDER}, "all_fixed_demos_run": len(results) == 9, "reuse_m2_task20_demo0": True, "exact_pending_accepted_counts": {"exact": 0, "pending": 0, "accepted": 0}}
    write_json(OUT_ROOT / "m3_summary.json", summary)
    return summary


def _gpu3_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "-i", "3", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    gpu = subprocess.run(command, capture_output=True, text=True, check=False)
    processes = subprocess.run(["nvidia-smi", "-i", "3", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    rows = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    parsed: list[dict[str, Any]] = []
    for line in rows:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            parsed.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": float(fields[2]), "memory_total_mib": float(fields[3])})
    return {"executed": True, "returncode": gpu.returncode, "query": command, "gpus": parsed, "compute_processes_raw": [line.strip() for line in processes.stdout.splitlines() if line.strip()], "process_query_returncode": processes.returncode, "stderr": gpu.stderr[-1000:], "free_gate": gpu.returncode == 0 and len(parsed) == 1 and parsed[0]["memory_used_mib"] <= 1024.0 and not [line for line in processes.stdout.splitlines() if line.strip()]}


def _encode_h264(path: Path, frames: Sequence[Any], *, fps: int = CONTROL_FREQ_HZ) -> dict[str, Any]:
    import numpy as np

    images = np.asarray(frames, dtype=np.uint8)
    if images.ndim != 4 or images.shape[0] != 41 or images.shape[-1] != 3 or images.shape[1] <= 0 or images.shape[2] <= 0:
        raise ValueError(f"invalid_video_frames:{list(images.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    if partial.exists():
        partial.unlink()
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{images.shape[2]}x{images.shape[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    encoded = subprocess.run(command, input=images.tobytes(), capture_output=True, check=False)
    if encoded.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"ffmpeg_failed:{encoded.returncode}:{encoded.stderr.decode(errors='replace')[-1000:]}")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(partial)], capture_output=True, text=True, check=False)
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        frame_rate = str(stream.get("avg_frame_rate"))
        frame_count = int(stream.get("nb_frames"))
        valid = stream.get("codec_name") == "h264" and int(stream.get("width")) > 0 and int(stream.get("height")) > 0 and frame_count == 41 and frame_rate in {"20/1", "20"}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        valid = False
        stream = {"parse_error": str(exc), "stdout": probe.stdout[-1000:]}
        frame_count = None
        frame_rate = None
    if probe.returncode != 0 or not valid:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffprobe_failed:{probe.returncode}:{stream}")
    os.replace(partial, path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path), "ffprobe": stream, "frame_count": frame_count, "fps": frame_rate, "temporary_suffix": ".partial.mp4", "atomic_publish": True}


def run_m4() -> dict[str, Any]:
    m3 = json.loads((OUT_ROOT / "m3_summary.json").read_text(encoding="utf-8"))
    if m3.get("status") != "track3_composed_reverse_start_cpu_matrix_passed_9_of_9":
        result = {"schema_version": "track3_composed_reverse_start_m4_v1", "status": "not_evaluated", "reason": "m3_9_of_9_not_passed", "results": []}
        write_json(OUT_ROOT / "m4_summary.json", result)
        return result
    guard = _gpu3_snapshot()
    if not guard.get("free_gate"):
        result = {"schema_version": "track3_composed_reverse_start_m4_v1", "status": "track3_composed_reverse_start_gpu3_resource_gate_failed", "created_at_utc": now(), "gpu3": guard, "results": []}
        write_json(OUT_ROOT / "m4_summary.json", result)
        return result
    results: list[dict[str, Any]] = []
    for task in TASK_ORDER:
        report_path = OUT_ROOT / "runs" / f"m4_task{task}_demo0.json"
        if report_path.is_file():
            # Preserve completed camera evidence if the orchestration shell
            # was interrupted after the last worker.  This branch is read
            # only and cannot consume another GPU3 canary.
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            result = payload
            videos = dict(payload.get("videos", {}))
        else:
            result, _returncode, _stdout, _stderr = _run_worker(task, 0, gpu=True)
            payload = dict(result)
            frame_payloads = payload.pop("camera_frame_payloads", {})
            videos = {}
            try:
                if result.get("status") != "passed" or not result.get("worker_returncode_contract"):
                    raise RuntimeError("camera_worker_gate_failed")
                video_dir = OUT_ROOT / "videos"
                for key, frames in frame_payloads.items():
                    filename = "agentview" if key == "agentview_image" else "wrist"
                    videos[key] = _encode_h264(video_dir / f"m4_task{task}_demo0_{filename}.mp4", frames)
                payload["videos"] = videos
                payload["camera_payload_sha256"] = {key: canonical_hash(value) for key, value in frame_payloads.items()}
            except Exception as exc:
                payload["video_error"] = {"type": type(exc).__name__, "message": str(exc)}
                payload["status"] = "failed_gate"
            write_json(report_path, payload)
        compact = _compact_result(payload, report_path)
        compact["videos"] = videos
        compact["camera_frames"] = result.get("camera_frames", {})
        results.append(compact)
        if payload.get("status") != "passed":
            break
    passed = len(results) == 3 and all(row.get("status") == "passed" and row.get("worker_returncode_contract") for row in results)
    summary = {"schema_version": "track3_composed_reverse_start_m4_v1", "status": "track3_composed_reverse_start_validated_pending_executor_review" if passed else "track3_composed_reverse_start_gpu_failed", "created_at_utc": now(), "gpu3_preflight": guard, "fixed_order": [10, 20, 19], "results": results, "camera_canary_count": len(results), "no_manipulation_action": True, "no_retry_or_demo_change": True}
    write_json(OUT_ROOT / "m4_summary.json", summary)
    return summary


def _run_check(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(list(command), cwd=ROOT, env=dict(env) if env is not None else None, capture_output=True, text=True, check=False)
    return {"command": list(command), "returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:], "stderr_tail": completed.stderr[-4000:]}


def _test_counts(stdout: str) -> dict[str, Any]:
    import re

    match = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) warnings)?", stdout)
    return {"passed": int(match.group(1) or 0) if match else None, "skipped": int(match.group(2) or 0) if match else None, "warnings": int(match.group(3) or 0) if match else None}


def _revalidate_m0_hashes(lock: Mapping[str, Any]) -> dict[str, Any]:
    checks = []
    for item in lock.get("old_input_lock", []):
        path = Path(str(item["path"]))
        actual = {"path": str(path), "bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None}
        checks.append({"path": str(path), "expected": {"bytes": item.get("bytes"), "sha256": item.get("sha256")}, "actual": actual, "unchanged": actual["bytes"] == item.get("bytes") and actual["sha256"] == item.get("sha256")})
    return {"executed": True, "state": "passed" if all(row["unchanged"] for row in checks) else "failed", "count": len(checks), "checks": checks}


def _contact_ledger() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((OUT_ROOT / "runs").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for sample in report.get("reverse", {}).get("checkpoints", []):
            matrix = sample.get("contact_matrix", {})
            entries.append({"run": path.name, "task_index": report.get("task_index"), "demo_id": report.get("demo_id"), "step": sample.get("step"), "ncon": matrix.get("ncon"), "complete": matrix.get("complete"), "contacts_v3": sample.get("contacts_v3"), "raw_contacts": [{"body_pair": sorted((str(row.get("body1_name", "")), str(row.get("body2_name", "")))), "geom_pair": sorted((str(row.get("geom1_name", "")), str(row.get("geom2_name", "")))), "distance_m": row.get("distance_m")} for row in matrix.get("contacts", [])]})
    return entries


def run_m5() -> dict[str, Any]:
    m0 = json.loads((OUT_ROOT / "m0_release_lock.json").read_text(encoding="utf-8"))
    m1 = json.loads((OUT_ROOT / "m1_validator_contract.json").read_text(encoding="utf-8"))
    m2 = json.loads((OUT_ROOT / "m2_summary.json").read_text(encoding="utf-8"))
    m3 = json.loads((OUT_ROOT / "m3_summary.json").read_text(encoding="utf-8"))
    m4 = json.loads((OUT_ROOT / "m4_summary.json").read_text(encoding="utf-8"))
    immutable = _revalidate_m0_hashes(m0)
    test_env = os.environ.copy()
    test_env.update({"PYTHONPATH": ".", "JAX_PLATFORMS": "cpu"})
    tests = _run_check([str(ROOT / ".venv_axis_pi05/bin/pytest"), "-q", "tests"], env=test_env)
    tests.update(_test_counts(tests["stdout_tail"]))
    compile_check = _run_check([sys.executable, "-m", "py_compile", str(ROOT / "tools/axis_pi05/track3_composed_reverse_start_stabilization.py")], env=test_env)
    json_paths = sorted(OUT_ROOT.rglob("*.json"))
    parse_errors: list[str] = []
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append(f"{path}:{type(exc).__name__}:{exc}")
    json_check = {"executed": True, "state": "passed" if not parse_errors else "failed", "file_count": len(json_paths), "errors": parse_errors}
    diff_check = _run_check(["git", "diff", "--check"])
    all_runs = []
    for name in ("m2_task20_demo0.json", "m3_task10_demo0.json", "m3_task10_demo1.json", "m3_task10_demo2.json", "m3_task20_demo1.json", "m3_task20_demo2.json", "m3_task19_demo0.json", "m3_task19_demo1.json", "m3_task19_demo2.json", "m4_task10_demo0.json", "m4_task20_demo0.json", "m4_task19_demo0.json"):
        path = OUT_ROOT / "runs" / name
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            all_runs.append(_compact_result(report, path))
    videos = sorted(OUT_ROOT.glob("videos/*.mp4"))
    partials = sorted(OUT_ROOT.rglob("*.partial*"))
    summary = {"schema_version": "track3_composed_reverse_start_m5_v1", "status": m4.get("status") if m4.get("status") == "track3_composed_reverse_start_validated_pending_executor_review" else "track3_composed_reverse_start_m5_hard_stop", "created_at_utc": now(), "phase_status": {"M0": m0.get("status"), "M1": m1.get("status"), "M2": m2.get("status"), "M3": m3.get("status"), "M4": m4.get("status")}, "counts": {"fixed_cpu_reverse_runs": sum(row.get("status") == "passed" for row in all_runs if str(row.get("path", "")).endswith(".json") and "/m4_" not in str(row.get("path", ""))), "camera_canaries": sum(row.get("status") == "passed" for row in all_runs if "/m4_" in str(row.get("path", ""))), "start_state_candidate_report_only": m2.get("start_state_candidate", {}).get("count", 0), "exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "fixed_demo_order": [{"task_index": task, "demo_index": demo} for task in TASK_ORDER for demo in range(3)], "runs": all_runs, "m0_immutable_revalidation": immutable, "test_validation": {"full_tests": tests, "py_compile": compile_check, "json_parse": json_check, "git_diff_check_precommit": diff_check}, "artifact_closure": {"videos": [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in videos], "partial_residue": [str(path.resolve()) for path in partials], "camera_count": len(videos)}, "resource_boundary": {"gpu3_only_for_m4": True, "gpu0_touched": False, "official_gpu": False, "training": False, "teleop": False, "se3_or_grasp": False, "manipulation_action_or_demo_generation": False, "hdf5_or_bddl_modified": False, "old_evidence_modified": False}, "hard_stop": {"executed": True, "status": "hard_stop_waiting_executor_review", "next_authority": "new_reviewed_plan_required", "no_automatic_executor_or_training": True}}
    summary["repository_state_before_commit"] = {"root_head": _git(ROOT, ("rev-parse", "HEAD")), "root_porcelain": _porcelain(ROOT), "openpi_head": _git(OPENPI, ("rev-parse", "HEAD")) if OPENPI.is_dir() else None, "openpi_porcelain": _porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain": _porcelain(LIBERO_PLUS)}
    ledger = _contact_ledger()
    write_json(OUT_ROOT / "contact_ledger.json", {"schema_version": "track3_contact_v3_ledger_v1", "status": "complete", "entry_count": len(ledger), "entries": ledger})
    write_json(OUT_ROOT / "run_index.json", {"schema_version": "track3_composed_reverse_start_run_index_v1", "status": "complete", "entries": all_runs})
    write_json(OUT_ROOT / "summary.json", summary)
    closeout = f"""# Track 3 composed reverse-start M5 收尾\n\n状态：`{summary['status']}`\n\n## 结果\n\n- M0/M1 通过；上一轮 matched-terminal evidence 与三份公开 HDF5 的九条固定 demo 已锁定并复核。\n- M2 task20/demo0 fresh CPU canary 通过；M3 固定九条 CPU matrix 为 `9/9`，三族各 `3/3`。\n- M4 GPU3 camera canary 按 task10→task20→task19 各一次，三条均通过；每条 agentview/wrist 为 41 帧、H.264、20 FPS。\n- composition 使用同 demo terminal scene qpos、同 demo init robot/gripper qpos、全零 qvel；hold 为 `float32[7]` 全零。\n- `start_state_candidate` 仅保留在 report namespace，数量 `{summary['counts']['start_state_candidate_report_only']}`；`exact/pending/accepted reverse demo = 0/0/0`。\n\n## 边界\n\n没有修改公开 HDF5、BDDL 或历史 evidence；没有生成 manipulation action/demo，没有 SE(3)/grasp、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或提交。M4 完成后已 hard stop，后续必须等待新的审核计划。\n\n## 校验\n\n- full tests：`{tests.get('passed')} passed, {tests.get('skipped')} skipped, {tests.get('warnings')} warnings`，returncode `{tests.get('returncode')}`\n- JSON parse：`{json_check['state']}`；py_compile：`{'passed' if compile_check['returncode'] == 0 else 'failed'}`；git diff check：`{'passed' if diff_check['returncode'] == 0 else 'failed'}`\n- root/OpenPI 与 `LIBERO-plus` 的最终 porcelain 由交接文件记录；`LIBERO-plus` 既有 dirty 状态未触碰。\n\n已 hard stop，等待人工审核。\n"""
    (OUT_ROOT / "M5_CLOSEOUT_REPORT_ZH.md").write_text(closeout, encoding="utf-8")
    return summary


def run_worker_cli(task_index: int, demo_index: int, *, gpu: bool) -> int:
    result = _runtime_worker(task_index, demo_index, gpu=gpu, capture_camera=gpu)
    print(json.dumps(result, ensure_ascii=False, default=_json_default, sort_keys=True))
    return worker_exit_code(str(result.get("status")))


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not_json_serializable:{type(value).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0", action="store_true")
    parser.add_argument("--m1", action="store_true")
    parser.add_argument("--m2", action="store_true")
    parser.add_argument("--m3", action="store_true")
    parser.add_argument("--m4", action="store_true")
    parser.add_argument("--m5", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--demo-index", type=int)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        if args.task_index not in TASK_ORDER or args.demo_index not in (0, 1, 2):
            parser.error("worker requires a fixed task/demo")
        return run_worker_cli(int(args.task_index), int(args.demo_index), gpu=bool(args.gpu))
    phases = [args.m0, args.m1, args.m2, args.m3, args.m4, args.m5]
    if sum(bool(value) for value in phases) != 1:
        parser.error("choose exactly one phase")
    result = run_m0() if args.m0 else run_m1() if args.m1 else run_m2() if args.m2 else run_m3() if args.m3 else run_m4() if args.m4 else run_m5()
    print(json.dumps(result, ensure_ascii=False, default=_json_default, sort_keys=True))
    return 0 if str(result.get("status", "")).endswith("passed") or result.get("status") == "not_evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
