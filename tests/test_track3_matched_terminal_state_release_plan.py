from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
CANARY = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence/k3_gpu3_release/task19_canary.json"


def test_only_the_start_bundle_and_support_executor_plan_is_active() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_start_bundle_and_support_executor_canary_planned"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md"
    assert handoff["next_phase_only"] == "P0_to_P6_then_hard_stop"
    assert (ROOT / handoff["next_plan"]).is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_BACKGROUND_PREDICATE_CONVERGENCE_RELEASE_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_BACKGROUND_PREDICATE_CONVERGENCE_RELEASE_PLAN.md").is_file()


def test_k3_errata_is_grounded_in_immutable_raw_canary() -> None:
    raw = json.loads(CANARY.read_text(encoding="utf-8"))
    sensitivity = raw["direct_state_sensitivity"]
    assert sensitivity["before_true"] is True
    assert sensitivity["destroyed_false"] is False
    assert sensitivity["restored_true"] is True
    assert sensitivity["mutation"]["before"] < -0.14
    assert sensitivity["mutation"]["after"] < sensitivity["mutation"]["before"]
    assert raw["renderer_schema"]["use_camera_obs"] is False
    assert raw["hard_gate"]["camera_frame_count_21"] is True
    assert raw["videos"]["agentview"]["reason"] == "ffmpeg_failed"
    assert ".mp4.partial" in raw["videos"]["agentview"]["stderr_tail"]

    step5 = next(row for row in raw["checkpoints"] if row["step"] == 5)
    contacts = [
        row
        for row in step5["contact_matrix"]["contacts"]
        if {row["body1_name"], row["body2_name"]}
        == {"akita_black_bowl_1_main", "wooden_cabinet_1_cabinet_middle"}
    ]
    assert len(contacts) == 39
    assert min(row["distance_m"] for row in contacts) == -0.006797624546751698


def test_composed_state_authorization_is_data_source_only_and_bounded() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    historical = handoff["track3_matched_terminal_state_release_authorization"]
    assert historical["status"] == "track3_matched_terminal_state_release_completed_hard_stop"
    assert historical["plan"].startswith("docs/archive/axis_pi05/")
    authorization = handoff["track3_composed_reverse_start_stabilization_authorization"]
    assert authorization["status"] == "track3_composed_reverse_start_stabilization_completed_and_audited"
    assert authorization["plan"].startswith("docs/archive/axis_pi05/")
    assert authorization["authorized_phases"] == ["M0", "M1", "M2", "M3", "M4", "M5_closeout"]
    scope = authorization["authorization"]
    assert scope["fixed_demo_ids"] == ["demo_0", "demo_1", "demo_2"]
    assert scope["state_composition"] == "terminal_scene_qpos_plus_init_robot_gripper_qpos_plus_zero_qvel"
    assert scope["neutral_hold_action"] == [0.0] * 7
    assert scope["burn_in_steps"] == 20
    assert scope["measurement_steps"] == 20
    assert scope["cpu_fresh_processes_max"] == 9
    assert scope["task20_demo0_canary_first"] is True
    assert scope["gpu3_camera_processes_max"] == 3
    assert scope["manipulation_action_or_reverse_demo"] is False
    assert scope["se3_or_grasp_executor"] is False
    assert scope["lora_or_dense_training"] is False
    assert scope["official_gpu"] is False

    audit = handoff["track3_composed_reverse_start_m0_m5_audit"]
    assert audit["status"] == "remediation_required"
    assert audit["core_result_approved"] is True
    assert audit["cpu_start_matrix"] == "9/9"
    assert audit["gpu3_camera_canaries"] == "3/3"
    assert audit["user_manual_action_required"] is False
    assert audit["human_teleop_required"] is False

    next_authorization = handoff["track3_start_bundle_and_support_executor_canary_authorization"]
    assert next_authorization["authorized_phases"] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6_closeout"]
    next_scope = next_authorization["authorization"]
    assert next_scope["materialize_restorable_start_bundles"] == 9
    assert next_scope["cpu_fresh_restore_matrix_required"] == "9/9"
    assert next_scope["support_executor_family"] == "task10_demo0_only"
    assert next_scope["support_generation_attempts_max"] == 1
    assert next_scope["successful_action_fresh_replays_required"] == "3/3"
    assert next_scope["human_teleop"] is False
    assert next_scope["training_or_norm"] is False
    assert next_scope["official_gpu"] is False
