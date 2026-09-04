from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
CANARY = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence/k3_gpu3_release/task19_canary.json"


def test_only_the_matched_terminal_state_plan_is_active() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_matched_terminal_state_cpu_failed_containment_remove"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md"
    assert handoff["next_phase_only"] == "await_next_audit_after_l5_hard_stop"
    assert (ROOT / handoff["next_plan"]).is_file()
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


def test_new_authorization_is_data_source_only_and_bounded() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    authorization = handoff["track3_matched_terminal_state_release_authorization"]
    assert authorization["authorized_phases"] == ["L0", "L1", "L2", "L3", "L4", "L5_closeout"]
    scope = authorization["authorization"]
    assert scope["fixed_demo_ids"] == ["demo_0", "demo_1", "demo_2"]
    assert scope["same_demo_model_xml_and_full_state_required"] is True
    assert scope["cpu_fresh_processes_max"] == 9
    assert scope["gpu3_task19_camera_canary_max"] == 1
    assert scope["gpu3_additional_family_camera_canaries_max"] == 2
    assert scope["action_or_reverse_demo"] is False
    assert scope["se3_or_grasp_executor"] is False
    assert scope["lora_or_dense_training"] is False
    assert scope["official_gpu"] is False
