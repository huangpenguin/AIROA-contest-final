#!/usr/bin/env python3
"""Derive the O1.0 red-mug keypose and retiming lock.

This command is intentionally read-only with respect to historical episodes.  It
loads the preferred 388-frame seed and the 468-frame reference with
``allow_pickle=False``, derives phase/kinematic landmarks, and writes only a
small, auditable report namespace.  It never replays actions or starts a
simulator, GPU job, or Teleop process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/pending"
NAMESPACE = REPO / "data/reports/axis_libero_30k/track3_operations_o1_0_red_mug_retiming_20260908"
PREFERRED = "official_t3_red_mug_1788838632"
REFERENCE = "official_t3_red_mug_1788838293"
PLAN = "docs/AXIS_PI05_TRACK3_OPERATIONS_O1_O2_RED_MUG_NATIVE_MUJOCO_PLAN.md"
ATTEMPT_ID = "canonical_red_mug_attempt_0001"


def strict_json(path: Path) -> dict[str, Any]:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def git_head(directory: Path = REPO) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite derived artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def first_gripper_close(gripper: np.ndarray) -> int | None:
    changes = np.flatnonzero((gripper[1:] != gripper[:-1]) & (gripper[1:] > gripper[:-1])) + 1
    return int(changes[0]) if len(changes) else None


def first_gripper_open_after(gripper: np.ndarray, start: int) -> int | None:
    changes = np.flatnonzero((gripper[1:] != gripper[:-1]) & (gripper[1:] < gripper[:-1])) + 1
    later = changes[changes >= start]
    return int(later[0]) if len(later) else None


def derive_episode(episode_id: str, role: str) -> dict[str, Any]:
    directory = RAW_ROOT / episode_id
    metadata_path = directory / "metadata.json"
    trajectory_path = directory / "trajectory.npz"
    metadata = strict_json(metadata_path)
    if metadata["frame_count"] != metadata["trajectory"]["frames"]:
        raise ValueError(f"frame count mismatch in {episode_id}")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        required = {"phase", "source_predicate", "goal_predicate", "gripper", "eef_pos", "contact_n"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"missing fields in {episode_id}: {sorted(missing)}")
        phase = archive["phase"].astype(str)
        source = archive["source_predicate"].astype(bool)
        goal = archive["goal_predicate"].astype(bool)
        gripper = archive["gripper"].astype(float)
        eef = archive["eef_pos"].astype(float)
        contact_n = archive["contact_n"].astype(int)
        frame_count = len(phase)
        if frame_count != metadata["frame_count"]:
            raise ValueError(f"NPZ frame count mismatch in {episode_id}")
        phase_runs = {name: runs(phase == name) for name in sorted(set(phase.tolist()))}
        grasp_runs = runs((phase == "grasp_dwell") & source & ~goal)
        if not grasp_runs:
            raise ValueError(f"no grasp dwell window in {episode_id}")
        grasp_start, grasp_end = grasp_runs[-1]
        approach_candidates = [end for start, end in phase_runs.get("precontact_xy", []) if end < grasp_start]
        if not approach_candidates:
            raise ValueError(f"no precontact approach before grasp in {episode_id}")
        approach = approach_candidates[-1]
        lift_candidates = np.flatnonzero(~source)
        if not len(lift_candidates):
            raise ValueError(f"no source release in {episode_id}")
        lift = int(lift_candidates[0])
        place_candidates = np.flatnonzero(goal)
        if not len(place_candidates):
            raise ValueError(f"no goal predicate in {episode_id}")
        place = int(place_candidates[0])
        release = first_gripper_open_after(gripper, place)
        if release is None:
            raise ValueError(f"no post-place gripper release in {episode_id}")
        transport_window = np.arange(lift, place, dtype=int)
        clearance_apex = int(transport_window[np.argmax(eef[transport_window, 2])])
        close = first_gripper_close(gripper)
        contact_capacity = archive["contact_body1"].shape[1] if "contact_body1" in archive else None
        truncated_frames = int(np.sum(contact_n > contact_capacity)) if contact_capacity is not None else None
        keyposes = {
            "approach": {
                "frame": approach,
                "evidence": ["phase=precontact_xy", "immediately_before_first_grasp_dwell"],
                "confidence": "derived_phase_boundary",
            },
            "grasp": {
                "frame": grasp_start,
                "window": [grasp_start, grasp_end],
                "evidence": ["phase=grasp_dwell", "source_predicate=true", "goal_predicate=false"],
                "gripper_close_frame": close,
                "confidence": "derived_only_contact_raw_truncated",
            },
            "lift": {
                "frame": lift,
                "evidence": ["first source_predicate=false"],
                "confidence": "derived_predicate_boundary",
            },
            "transport": {
                "frame": clearance_apex,
                "evidence": ["maximum_eef_z_between_source_release_and_goal"],
                "confidence": "derived_kinematic_clearance_apex",
            },
            "place": {
                "frame": place,
                "evidence": ["first goal_predicate=true", "phase=place"],
                "confidence": "derived_goal_boundary",
            },
            "release": {
                "frame": release,
                "evidence": ["post-place gripper transition to OPEN"],
                "confidence": "derived_gripper_boundary",
            },
            "settle_end": {
                "frame": frame_count - 1,
                "evidence": ["historical final frame only"],
                "confidence": "historical_reference_not_gold",
            },
        }
        artifact_paths = {}
        for name in ("metadata.json", "trajectory.npz", "start_state.npy", "end_state.npy", "front.mp4", "wrist.mp4"):
            path = directory / name
            artifact_paths[name] = {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        return {
            "episode_id": episode_id,
            "role": role,
            "frame_count": frame_count,
            "metadata_schema": metadata["schema_version"],
            "historical_status": metadata["status"],
            "candidate_gate_historical": metadata["gold_gates"]["candidate_gate"],
            "historical_metadata_unchanged": True,
            "replay": "not_evaluated",
            "gold_eligible": False,
            "contact_raw": {
                "contact_n_min": int(contact_n.min()),
                "contact_n_max": int(contact_n.max()),
                "capacity": contact_capacity,
                "truncated_frames": truncated_frames,
                "complete": False,
                "usable_for_keypose_only": True,
            },
            "phase_runs": {name: [[start, end] for start, end in values] for name, values in phase_runs.items()},
            "keyposes": keyposes,
            "artifacts": artifact_paths,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("O1.0 requires --run")
    if NAMESPACE.exists():
        raise SystemExit(f"refusing to reuse existing O1.0 namespace: {NAMESPACE}")
    preferred = derive_episode(PREFERRED, "preferred_seed")
    reference = derive_episode(REFERENCE, "reference_only")
    source_commit = git_head()
    nested_commit = git_head(REPO / "third_party/axis_openpi")
    inputs = {"preferred": preferred, "reference": reference}
    target_keyposes = {
        "approach": 80,
        "grasp": 91,
        "lift": 102,
        "transport": 160,
        "place": 220,
        "release": 230,
        "settle_end": 240,
    }
    preferred_keyposes = preferred["keyposes"]
    blueprint_keyposes = {}
    for name, target in target_keyposes.items():
        original = preferred_keyposes[name]["frame"]
        blueprint_keyposes[name] = {
            "source_frame": original,
            "target_frame": target,
            "target_frame_is_planning_only": True,
            "source_evidence_confidence": preferred_keyposes[name]["confidence"],
        }
    blueprint = {
        "schema_version": "track3_o1_0_red_mug_goal_first_retiming_blueprint_v1",
        "status": "derived_not_executed",
        "source_seed": PREFERRED,
        "reference_seed": REFERENCE,
        "method": "goal_first_phase_boundary_and_clearance_apex_mapping",
        "historical_raw_modified": False,
        "action_generation": "not_evaluated",
        "replay": "not_evaluated",
        "gpu_used": False,
        "mujoco_started": False,
        "teleop_started": False,
        "budgets": {
            "goal_first_target_max_frames": 240,
            "total_target_max_frames": 260,
            "hard_max_frames": 300,
            "planned_goal_predicate_frame": 220,
            "planned_release_frame": 230,
            "planned_settle_frame": 240,
            "post_goal_reserve_frames": 20,
        },
        "contact_contract": {
            "schema": "track3_reverse_pilot_v4",
            "storage": "typed_flat_table_with_offsets",
            "saved_contact_n_equals_contact_n": True,
            "acquisition": "first stable gripper_robot_subject contact during close/grasp_dwell before source release",
            "historical_contact_is_not_canonical_evidence": True,
        },
        "keyposes": blueprint_keyposes,
        "target_frame_order_valid": all(
            target_keyposes[a] < target_keyposes[b]
            for a, b in zip(target_keyposes, list(target_keyposes)[1:])
        ),
        "next_gate": "ready_for_native_vnc_no_save_canary",
    }
    lock = {
        "schema_version": "track3_o1_0_canonical_attempt_lock_v1",
        "lock_id": ATTEMPT_ID,
        "status": "ready_for_native_vnc_no_save_canary",
        "unique_canonical_attempt_count": 1,
        "preferred_seed": PREFERRED,
        "reference_seed": REFERENCE,
        "source_commit": source_commit,
        "nested_openpi_commit": nested_commit,
        "active_plan": PLAN,
        "recording_started": False,
        "canonical_recording": "not_started",
        "fresh_replay": "not_evaluated",
        "gpu_used": False,
        "mujoco_started": False,
        "teleop_started": False,
        "historical_raw_modified": False,
        "historical_metadata_modified": False,
        "infinity_or_scenic_used": False,
        "user_takeover_required_for_recording": True,
        "hard_stop_after_o1_0": True,
    }
    input_lock = {
        "schema_version": "track3_o1_0_input_lock_v1",
        "root_commit": source_commit,
        "nested_openpi_commit": nested_commit,
        "active_plan": PLAN,
        "preferred_seed": preferred,
        "reference_seed": reference,
        "raw_mutation": False,
        "historical_metadata_unchanged": True,
    }
    summary = {
        "schema_version": "track3_o1_0_summary_v1",
        "status": "ready_for_native_vnc_no_save_canary",
        "phase": "O1.0",
        "preferred_seed": PREFERRED,
        "reference_seed": REFERENCE,
        "source_commit": source_commit,
        "nested_openpi_commit": nested_commit,
        "historical_raw_modified": False,
        "replay": "not_evaluated",
        "gpu": "not_used",
        "mujoco": "not_started",
        "teleop": "not_started",
        "canonical_attempt_lock": "canonical_attempt_lock.json",
        "retiming_blueprint": "retiming_blueprint.json",
        "keypose_report": "keypose_report.json",
        "input_lock": "input_lock.json",
        "next_action": "user_online_native_vnc_no_save_canary",
        "hard_stop": True,
    }
    validation = {
        "schema_version": "track3_o1_0_validation_commands_v1",
        "analysis_argv": ["env", "PYTHONPATH=.", ".venv_axis_pi05/bin/python", __file__, "--run"],
        "load_mode": "numpy.load allow_pickle=False",
        "raw_write": False,
        "replay": "not_evaluated",
        "gpu": "not_used",
        "mujoco": "not_started",
        "teleop": "not_started",
        "strict_json": "required_after_generation",
        "py_compile": "required_after_generation",
        "git_diff_check": "required_after_generation",
    }
    keypose_report = {
        "schema_version": "track3_o1_0_keypose_report_v1",
        "status": "derived_not_executed",
        "preferred": preferred,
        "reference": reference,
        "interpretation": {
            "approach": "last precontact_xy frame before the selected stable grasp_dwell window",
            "grasp": "start of final stable source-side grasp_dwell window; contact raw is diagnostic only",
            "lift": "first source_predicate=false frame",
            "transport": "maximum EEF z between source release and first goal predicate",
            "place": "first goal_predicate=true frame",
            "release": "first post-place gripper transition to OPEN",
        },
    }
    write_json(NAMESPACE / "input_lock.json", input_lock)
    write_json(NAMESPACE / "keypose_report.json", keypose_report)
    write_json(NAMESPACE / "retiming_blueprint.json", blueprint)
    write_json(NAMESPACE / "canonical_attempt_lock.json", lock)
    write_json(NAMESPACE / "validation_commands.json", validation)
    write_json(NAMESPACE / "summary.json", summary)
    print(json.dumps({"namespace": rel(NAMESPACE), "status": summary["status"], "preferred": PREFERRED}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
