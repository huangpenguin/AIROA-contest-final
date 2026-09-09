#!/usr/bin/env python3
"""MG1 deterministic assisted red-mug re-execution.

The historical 314-frame success is used only as a read-only waypoint source.
This command creates a fresh MuJoCo trajectory from the frozen historical
start state. It never edits the historical episode and never deletes or
interpolates historical frames into a training sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.axis_pi05 import track3_public_teleop as teleop


ROOT = Path(__file__).resolve().parents[2]
SEED_ID = "official_t3_red_mug_1788921471"
SEED_DIR = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/pending" / SEED_ID
O10_DIR = ROOT / "data/reports/axis_libero_30k/track3_operations_o1_0_red_mug_retiming_20260908"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_MULTITASK_GOLD_AND_R6_SHORT_CONTINUATION_PLAN.md"
NAMESPACE = ROOT / "data/reports/axis_libero_30k/track3_operations_mg1_red_mug_assisted_20260909"
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"
HISTORICAL_HASHES = {
    "metadata.json": "7828256a84e286058d6d602196166732e39b535e6842f2944056a612b70d5a33",
    "trajectory.npz": "91ddeeddb98f424c88f9c87edaa190a0cdad2f7af466d390d7cd859e77f3e6d8",
    "start_state.npy": "5774476328e5d655b1b7b428a1bf0732ec964f72d9a7d4b105fefd781bc0b3a2",
    "end_state.npy": "3276587f38c5e94b9572c253948b45a326ed625c150e80bd8422457bd974c1e2",
    "front.mp4": "73e5838a3dc37f8887b0fe4a47a330c3dfd79d67b014be9cad82a4092c1ee9eb",
    "wrist.mp4": "f73ec9f99e6b23cb43bf1656aa1eff57f43d17540c982320e90545c223581357",
}
BLUEPRINT_SHA = "9e07cc83ea888039c85c5a94b9ddada769add7bd74189b04f70a83deefe53903"
BDDLSHA = "7b35f8fad5defe6d7952973e5916e102b6db56b6abf353e0d6e6d6579664f559"
RECORDER_SHA = "dee3cd22abae4adc3e118e7c472642f9e4730bb0ec2e1cc0aa01c62800c83d5a"
TARGET_STEPS = 260
HARD_STEPS = 300
ACTION_CAP = 0.30
CONTROL_GAIN_M = 0.0125
POSITION_TOLERANCE_M = 0.006
WAYPOINT_FRAMES = {
    "approach": 108,
    "grasp": 117,
    "lift": 163,
    "transport": 255,
    "place": 308,
}


def strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_seed() -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, str]]:
    if not SEED_DIR.is_dir():
        raise RuntimeError(f"missing frozen seed: {SEED_DIR}")
    observed = {name: sha256_file(SEED_DIR / name) for name in HISTORICAL_HASHES}
    if observed != HISTORICAL_HASHES:
        raise RuntimeError(f"historical seed hash drift: {observed}")
    metadata = strict_json(SEED_DIR / "metadata.json")
    if metadata.get("task_id") != "official_t3_red_mug" or metadata.get("frame_count") != 314:
        raise RuntimeError("MG1 requires the locked 314-frame official red-mug seed")
    with np.load(SEED_DIR / "trajectory.npz", allow_pickle=False) as archive:
        required = {"actions", "eef_pos", "source_predicate", "goal_predicate", "gripper", "phase"}
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"seed trajectory missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    if arrays["actions"].shape != (314, 7) or arrays["eef_pos"].shape != (314, 3):
        raise RuntimeError("unexpected locked seed array shape")
    if not np.isfinite(arrays["actions"]).all() or not np.isfinite(arrays["eef_pos"]).all():
        raise RuntimeError("non-finite locked seed data")
    return metadata, arrays, observed


def build_schedule_blueprint(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    targets = {
        name: np.asarray(arrays["eef_pos"][frame], dtype=np.float64).round(9).tolist()
        for name, frame in WAYPOINT_FRAMES.items()
    }
    return {
        "schema_version": "track3_mg1_assisted_waypoint_schedule_v1",
        "status": "derived_not_executed",
        "source_episode": SEED_ID,
        "source_frame_count": int(metadata["frame_count"]),
        "source_metadata_status": metadata.get("status"),
        "source_frames_are_read_only": True,
        "historical_frame_deletion": False,
        "action_interpolation": False,
        "execution": "fresh_continuous_mujoco_from_frozen_start_state",
        "controller": {
            "translation_action_cap": ACTION_CAP,
            "control_gain_m": CONTROL_GAIN_M,
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "sim_control_hz": 20.0,
            "simultaneous_xyz": True,
            "orientation_action": [0.0, 0.0, 0.0],
        },
        "budgets": {
            "goal_first_target_max_frames": 240,
            "total_target_max_frames": TARGET_STEPS,
            "hard_max_frames": HARD_STEPS,
            "post_goal_stable_frames": 4,
        },
        "waypoints": [
            {"stage": "approach", "source_frame": WAYPOINT_FRAMES["approach"], "target": targets["approach"], "gripper": -1.0, "stage_budget": 75},
            {"stage": "grasp_approach", "source_frame": WAYPOINT_FRAMES["grasp"], "target": targets["grasp"], "gripper": -1.0, "stage_budget": 18},
            {"stage": "grasp_close_dwell", "source_frame": WAYPOINT_FRAMES["grasp"], "target": targets["grasp"], "gripper": 1.0, "stage_budget": 8},
            {"stage": "lift", "source_frame": WAYPOINT_FRAMES["lift"], "target": targets["lift"], "gripper": 1.0, "stage_budget": 55},
            {"stage": "transport", "source_frame": WAYPOINT_FRAMES["transport"], "target": targets["transport"], "gripper": 1.0, "stage_budget": 65},
            {"stage": "place", "source_frame": WAYPOINT_FRAMES["place"], "target": targets["place"], "gripper": 1.0, "stage_budget": 55},
            {"stage": "release_settle", "source_frame": WAYPOINT_FRAMES["place"], "target": None, "gripper": -1.0, "stage_budget": 10},
        ],
        "source_keypose_mapping": WAYPOINT_FRAMES,
        "next_gate": "fresh_physics_attempt",
    }


def input_lock(metadata: dict[str, Any], observed: dict[str, str], blueprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "track3_mg1_assisted_input_lock_v1",
        "status": "locked",
        "plan": rel(PLAN),
        "source_episode": SEED_ID,
        "source_episode_artifacts": [
            {"path": rel(SEED_DIR / name), "sha256": value}
            for name, value in observed.items()
        ],
        "source_metadata_status": metadata.get("status"),
        "source_frame_count": metadata.get("frame_count"),
        "o1_0_blueprint": rel(O10_DIR / "retiming_blueprint.json"),
        "o1_0_blueprint_sha256": BLUEPRINT_SHA,
        "mg1_schedule_sha256": hashlib.sha256(json.dumps(blueprint, sort_keys=True).encode("utf-8")).hexdigest(),
        "official_bddl": rel(ROOT / "compe/t3/assets/bddl_files/libero_t3/take_the_red_mug_off_the_left_plate_and_put_it_on_the_table.bddl"),
        "official_bddl_sha256": BDDLSHA,
        "recorder_sha256": RECORDER_SHA,
        "historical_raw_modified": False,
        "historical_metadata_modified": False,
        "replay": "not_evaluated",
        "human_review": "not_evaluated",
    }


def eef_position(env: Any) -> np.ndarray:
    sim = env.sim
    site_id = int(sim.model.site_name2id("gripper0_grip_site"))
    return np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()


def action_for_target(env: Any, target: np.ndarray | None, gripper: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    if target is not None:
        delta = np.asarray(target, dtype=np.float64) - eef_position(env)
        motion = np.clip(delta / CONTROL_GAIN_M, -ACTION_CAP, ACTION_CAP)
        if float(np.max(np.abs(delta))) <= POSITION_TOLERANCE_M:
            motion[:] = 0.0
        action[:3] = motion.astype(np.float32)
    action[-1] = float(gripper)
    teleop.validate_action(action)
    return action


def take_step(
    env: Any,
    profile: dict[str, Any],
    frames: dict[str, list[Any]],
    samples: list[dict[str, Any]],
    action: np.ndarray,
    started: float,
) -> dict[str, Any]:
    obs, reward, done, _info = env.step(action)
    contact = teleop.raw_contact_matrix_from_sim(env)
    check_success = bool(env.check_success())
    source, goal = teleop._eval_profile_predicates(env, profile)
    eef = eef_position(env)
    sample = teleop.append_gold_frame(
        frames,
        action=action,
        state=np.asarray(env.get_sim_state(), dtype=np.float64),
        reward=float(reward),
        done=bool(done),
        success=check_success,
        front=teleop._image_from_obs(obs, "agentview"),
        wrist=teleop._image_from_obs(obs, "robot0_eye_in_hand"),
        timestamp=time.monotonic() - started,
        deadline_miss=False,
        contact_matrix=contact,
        source=source,
        goal=goal,
        eef_pos=eef,
        gripper=float(action[-1]),
        check_success=check_success,
        contact_roles=profile.get("roles"),
    )
    samples.append(sample)
    return sample


def run_waypoint_stage(
    env: Any,
    profile: dict[str, Any],
    frames: dict[str, list[Any]],
    samples: list[dict[str, Any]],
    stage: dict[str, Any],
    started: float,
) -> tuple[bool, str | None, int]:
    target = None if stage["target"] is None else np.asarray(stage["target"], dtype=np.float64)
    settled = 0
    acquired = 0
    for count in range(int(stage["stage_budget"])):
        action = action_for_target(env, target, float(stage["gripper"]))
        sample = take_step(env, profile, frames, samples, action, started)
        if not teleop.scene_bodies_ok(env, ["red_coffee_mug_1"]):
            return False, "family_failure_subject_unstable", count + 1
        if bool(sample.get("unrelated_contact")):
            return False, "family_failure_unrelated_contact", count + 1
        if stage["stage"] == "grasp_close_dwell":
            if bool(sample.get("demonstrator_contact")) and bool(sample.get("source_predicate")):
                acquired += 1
        if target is None:
            if bool(sample.get("goal_predicate")) and bool(sample.get("check_success")):
                settled += 1
                if settled >= 4:
                    return True, None, count + 1
            else:
                settled = 0
        else:
            if float(np.max(np.abs(eef_position(env) - target))) <= POSITION_TOLERANCE_M:
                settled += 1
                if stage["stage"] == "grasp_close_dwell" and acquired >= 4:
                    return True, None, count + 1
                if stage["stage"] != "grasp_close_dwell" and settled >= 3:
                    return True, None, count + 1
            else:
                settled = 0
    if stage["stage"] == "grasp_close_dwell" and acquired < 4:
        return False, "family_failure_acquisition_window_not_observed", int(stage["stage_budget"])
    return False, "family_failure_waypoint_timeout", int(stage["stage_budget"])


def run_attempt(profile: dict[str, Any], schedule: dict[str, Any], start_state: np.ndarray) -> dict[str, Any]:
    guard = teleop.gpu3_guard()
    if guard["status"] != "passed":
        return {"status": "infrastructure_failure_gpu3_guard", "gpu3_guard": guard, "physics_started": False}
    teleop.configure_public_libero()
    from compe.t3 import register_t3
    register_t3()
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(profile["bddl_path"]),
        control_freq=20,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=0,
        horizon=HARD_STEPS + 20,
        ignore_done=True,
    )
    frames = teleop.empty_gold_buffers()
    samples: list[dict[str, Any]] = []
    stage_results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        env.reset()
        env.set_init_state(np.asarray(start_state, dtype=np.float64))
        source0, goal0 = teleop._eval_profile_predicates(env, profile)
        if source0 is not True or goal0 is not False or bool(env.check_success()):
            raise RuntimeError(f"frozen start predicates invalid: source={source0} goal={goal0} success={env.check_success()}")
        for stage in schedule["waypoints"]:
            if len(frames["actions"]) >= HARD_STEPS:
                break
            ok, failure, used = run_waypoint_stage(env, profile, frames, samples, stage, started)
            stage_results.append({"stage": stage["stage"], "ok": ok, "failure": failure, "used_frames": used, "total_frames": len(frames["actions"])})
            if not ok:
                break
        gates = teleop.evaluate_gold_gates(samples) if samples else {"status": "failed", "candidate_gate": "failed", "save_allowed": False, "reason": "empty"}
        success = bool(samples and samples[-1].get("check_success"))
        frame_count = len(frames["actions"])
        failure = next((row["failure"] for row in reversed(stage_results) if row.get("failure")), None)
        if success and gates.get("save_allowed") and frame_count <= TARGET_STEPS:
            status = "pending_replay"
        elif success and frame_count > TARGET_STEPS:
            status = "rejected_over_target_budget"
            failure = "family_failure_over_target_budget"
        elif success:
            status = "rejected_predicate_fail"
            failure = "family_failure_gate"
        else:
            status = "rejected_assisted_failure"
            failure = failure or "family_failure_goal_not_reached"
        metadata = teleop._metadata(profile, "mg1-assisted", teleop.RecorderConfig(operator_hz=4.0, max_steps=HARD_STEPS, eval_step_budget=HARD_STEPS), {
            **dict(profile.get("lock_info") or {}),
            "assisted_source_episode": SEED_ID,
            "execution_mode": "deterministic_assisted_waypoint_reexecution",
        })
        metadata.update({
            "status": status,
            "gold_eligible": False,
            "execution_mode": "deterministic_assisted_waypoint_reexecution",
            "assisted_schedule": "mg1_schedule_blueprint_v1",
            "source_episode": SEED_ID,
            "source_start_state": "frozen_historical_start_state",
            "physics_started": True,
            "stage_results": stage_results,
            "failure": failure if status != "pending_replay" else None,
            "gold_gates": gates,
            "target_budget_frames": TARGET_STEPS,
            "hard_budget_frames": HARD_STEPS,
            "operator_tick_hz": None,
            "timing": {
                **dict(metadata.get("timing") or {}),
                "execution_mode": "continuous_sim_control",
                "sim_step_wall_duration_s": time.monotonic() - started,
            },
        })
        name = f"official_t3_red_mug_mg1_assisted_{int(time.time())}"
        bucket = "pending" if status == "pending_replay" else "rejected"
        target = teleop._atomic_episode(
            OUTPUT_ROOT / bucket,
            name,
            lambda directory: teleop._write_episode(
                directory,
                metadata,
                frames,
                np.asarray(start_state, dtype=np.float64),
                np.asarray(env.get_sim_state(), dtype=np.float64),
                teleop.RecorderConfig(operator_hz=4.0, max_steps=HARD_STEPS, eval_step_budget=HARD_STEPS),
            ),
        )
        return {
            "status": status,
            "episode_dir": str(target),
            "episode_id": target.name,
            "frame_count": frame_count,
            "success": success,
            "gates": gates,
            "stage_results": stage_results,
            "failure": failure if status != "pending_replay" else None,
            "gpu3_guard": guard,
            "physics_started": True,
            "historical_raw_modified": False,
        }
    finally:
        env.close()


def subprocess_head(directory: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def run() -> dict[str, Any]:
    if NAMESPACE.exists():
        raise RuntimeError(f"refusing to reuse MG1 namespace: {NAMESPACE}")
    metadata, arrays, observed = load_seed()
    if sha256_file(O10_DIR / "retiming_blueprint.json") != BLUEPRINT_SHA:
        raise RuntimeError("O1.0 blueprint hash drift")
    bddl = ROOT / "compe/t3/assets/bddl_files/libero_t3/take_the_red_mug_off_the_left_plate_and_put_it_on_the_table.bddl"
    if sha256_file(bddl) != BDDLSHA:
        raise RuntimeError("official red-mug BDDL hash drift")
    schedule = build_schedule_blueprint(metadata, arrays)
    NAMESPACE.mkdir(parents=True)
    teleop.write_json(NAMESPACE / "input_lock.json", input_lock(metadata, observed, schedule))
    teleop.write_json(NAMESPACE / "schedule_blueprint.json", schedule)
    preflight = {
        "schema_version": "track3_mg1_assisted_preflight_v1",
        "status": "passed",
        "root_head": subprocess_head(ROOT),
        "nested_openpi_head": subprocess_head(ROOT / "third_party/axis_openpi"),
        "gpu3_guard": teleop.gpu3_guard(),
        "historical_raw_modified": False,
        "historical_metadata_modified": False,
        "teleop_manual_started": False,
        "training_started": False,
        "replay": "not_evaluated",
    }
    if preflight["gpu3_guard"].get("status") != "passed":
        teleop.write_json(NAMESPACE / "preflight.json", {**preflight, "status": "blocked"})
        raise RuntimeError(f"GPU3 preflight blocked: {preflight['gpu3_guard']}")
    teleop.write_json(NAMESPACE / "preflight.json", preflight)
    profile = teleop.resolve_record_profile(task_id="official_t3_red_mug")
    start_state = np.load(SEED_DIR / "start_state.npy", allow_pickle=False)
    result = run_attempt(profile, schedule, np.asarray(start_state, dtype=np.float64))
    post_hashes = {name: sha256_file(SEED_DIR / name) for name in HISTORICAL_HASHES}
    result["historical_hashes_unchanged"] = post_hashes == observed
    if not result["historical_hashes_unchanged"]:
        result["status"] = "infrastructure_failure_historical_seed_drift"
    teleop.write_json(NAMESPACE / "attempt_result.json", result)
    replay_result: dict[str, Any] | None = None
    review_card: dict[str, Any] | None = None
    if result.get("status") == "pending_replay":
        from tools.axis_pi05 import track3_public_replay as replay
        replay_result = replay.run_replay(Path(result["episode_dir"]))
        if replay_result.get("passed_runs") == 3:
            episode_dir = Path(result["episode_dir"])
            review_card = {
                "schema_version": "track3_mg1_human_review_card_v1",
                "status": "pending_human_review",
                "episode_id": result["episode_id"],
                "episode_dir": str(episode_dir),
                "machine_status": replay_result.get("status"),
                "fresh_replay": "3/3_passed",
                "human_review_required": True,
                "required_user_reply": f"review {result['episode_id']}: accept or reject:<reason>",
                "accepted_promotion": False,
                "review_fields": ["instruction_match", "objects_correct", "no_teleport_or_failure_trim", "front_wrist_confirmed"],
            }
            teleop.write_json(NAMESPACE / "review_card.json", review_card)
    summary = {
        "schema_version": "track3_mg1_assisted_summary_v1",
        "status": "ready_for_human_review" if review_card else ("family_failure" if str(result.get("status", "")).startswith("rejected") else result.get("status")),
        "attempt": result,
        "fresh_replay": replay_result or "not_evaluated",
        "review_card": review_card or "not_evaluated",
        "source_episode": SEED_ID,
        "historical_raw_modified": result.get("historical_hashes_unchanged") is not True,
        "manual_fallback": "authorized_once_after_complete_telemetry_behavior_failure" if str(result.get("status", "")).startswith("rejected") else "not_evaluated",
        "training": "not_evaluated",
        "upload_or_submission": "not_evaluated",
    }
    teleop.write_json(NAMESPACE / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        metadata = {"frame_count": 314, "status": "pending_replay"}
        arrays = {
            "eef_pos": np.arange(314 * 3, dtype=np.float64).reshape(314, 3),
            "actions": np.zeros((314, 7), dtype=np.float32),
            "source_predicate": np.ones(314, dtype=bool),
            "goal_predicate": np.zeros(314, dtype=bool),
            "gripper": np.full(314, -1.0),
            "phase": np.full(314, "precontact_xy"),
        }
        schedule = build_schedule_blueprint(metadata, arrays)
        assert schedule["source_frames_are_read_only"] is True
        assert schedule["historical_frame_deletion"] is False
        assert schedule["action_interpolation"] is False
        assert schedule["budgets"]["hard_max_frames"] == 300
        assert [row["stage"] for row in schedule["waypoints"]] == ["approach", "grasp_approach", "grasp_close_dwell", "lift", "transport", "place", "release_settle"]
        print(json.dumps({"status": "passed", "physics_started": False, "gpu_touched": False}, ensure_ascii=False))
        return 0
    if not args.run:
        parser.error("choose --run or --self-test")
    result = run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {"ready_for_human_review", "family_failure"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
