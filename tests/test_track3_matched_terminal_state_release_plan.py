from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
CANARY = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence/k3_gpu3_release/task19_canary.json"


def test_only_the_composed_reverse_start_plan_is_active() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_composed_reverse_start_validated_pending_executor_review"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_COMPOSED_REVERSE_START_STABILIZATION_PLAN.md"
    assert handoff["next_phase_only"] == "new_reviewed_plan_required_after_M5_hard_stop"
    assert (ROOT / handoff["next_plan"]).is_file()
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
