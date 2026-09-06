from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
CANARY = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence/k3_gpu3_release/task19_canary.json"


def test_only_the_track3_tonight_zip_plan_is_active() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_clean_launch_and_diversified_canary_recovery_planned"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_PLAN.md"
    assert handoff["next_phase_only"] == "CR0_to_CR14_then_hard_stop"
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_SUPPORT_RELEASE_ANCHOR_AND_OSC_SCALE_RECOVERY_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_J5_EXECUTOR_CONTRACT_AND_ATOMIC_VIDEO_RECOVERY_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_OVERNIGHT_AUTOGEN_LORA_AND_ZIP_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_OVERNIGHT_AUTOGEN_LORA_AND_ZIP_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_P3_IO_RECOVERY_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_P3_IO_RECOVERY_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_START_BUNDLE_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md").is_file()
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

    bundle_authorization = handoff["track3_start_bundle_and_support_executor_canary_authorization"]
    assert bundle_authorization["status"] == "track3_start_bundle_and_support_executor_completed_and_audited"
    assert bundle_authorization["plan"].startswith("docs/archive/axis_pi05/")
    assert bundle_authorization["authorized_phases"] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6_closeout"]

    bundle_audit = handoff["track3_start_bundle_p0_p6_audit"]
    assert bundle_audit["status"] == "remediation_required"
    assert bundle_audit["approved"]["p2_fresh_cpu_restore"] == "9/9"
    assert bundle_audit["artifact_manifest_entries"] == 17
    assert bundle_audit["artifact_entries_not_git_tracked"] == 4
    assert bundle_audit["valid_p3_worker_results"] == 0
    assert bundle_audit["user_manual_action_required"] is False

    recovery = handoff["track3_p3_io_recovery_and_support_executor_canary_authorization"]
    assert recovery["status"] == "track3_p3_io_recovery_completed_and_audited"
    assert recovery["plan"].startswith("docs/archive/axis_pi05/")
    assert recovery["authorized_phases"] == ["R0", "R1", "R2", "R3", "R4", "R5", "R6_closeout"]
    next_scope = recovery["authorization"]
    assert next_scope["rerun_p2"] is False
    assert next_scope["git_track_existing_hash_exact_artifacts"] == 4
    assert next_scope["atomic_result_file_required"] is True
    assert next_scope["stdout_stress_bytes_min"] == 16 * 1024 * 1024
    assert next_scope["worker_deadline_seconds"] == 300
    assert next_scope["p3_task10_recovery_attempts_max"] == 1
    assert next_scope["p3_remaining_camera_processes_max"] == 2
    assert next_scope["human_teleop"] is False
    assert next_scope["training_or_norm"] is False
    assert next_scope["official_gpu"] is False

    p3_audit = handoff["track3_p3_io_recovery_r0_r6_audit"]
    assert p3_audit["status"] == "remediation_required"
    assert p3_audit["approved"]["new_artifacts_tracked_hash_exact"] == "33/33"
    assert p3_audit["approved"]["task10_child_physics_camera"] is True
    assert p3_audit["gpu_source_has_gpu_provenance_gate_definition"] is False
    assert p3_audit["final_commit_has_gpu_provenance_gate_definition"] is True
    assert p3_audit["user_manual_action_required"] is False

    overnight = handoff["track3_overnight_autogen_lora_zip_authorization"]
    assert overnight["status"] == "track3_overnight_autogen_lora_zip_completed_and_audited"
    assert overnight["plan"].startswith("docs/archive/axis_pi05/")
    assert overnight["authorized_phases"] == ["O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10_closeout"]
    scope = overnight["authorization"]
    assert scope["unattended_runtime_hours_max"] == 12
    assert scope["task10_p3_attempt_number"] == 3
    assert scope["profiles_per_family_max"] == 2
    assert scope["total_train_generations_max"] == 12
    assert scope["total_replays_max"] == 36
    assert scope["forward_reverse_sampling"] == [0.8, 0.2]
    assert scope["initialization"] == "r6_dual_l2_fresh500_r1_step500_params"
    assert scope["training_gpus"] == [1, 2]
    assert scope["winner_max_steps"] == 1000
    assert scope["gpu0"] is False
    assert scope["official_gpu"] is False
    assert scope["automatic_upload_or_submission"] is False

    execution = overnight["execution"]
    assert execution["task10_attempt3_child_status"] == "passed"
    assert execution["task10_attempt3_parent_status"] == "failed_closed"
    assert execution["task10_attempt3_parent_poll_count"] == 113
    assert execution["o3_o9"] == "not_evaluated"

    audit = handoff["track3_overnight_o0_o2_audit"]
    assert audit["status"] == "remediation_required"
    assert audit["approved"]["task10_child_physics_camera"] is True
    assert audit["approved"]["process_returncode"] == 0
    assert audit["approved"]["timed_out"] is False
    assert audit["old_o10_manifest_entries"] == 10
    assert audit["old_o10_manifest_hash_matches"] == 9
    assert audit["task10_attempt4_authorized"] is False
    assert audit["user_manual_action_required"] is False

    continuation = handoff["track3_provenance_jitter_recovery_and_overnight_continuation_authorization"]
    assert continuation["status"] == "track3_provenance_jitter_recovery_and_overnight_continuation_completed_and_audited"
    assert continuation["plan"].startswith("docs/archive/axis_pi05/")
    assert continuation["authorized_phases"] == ["J0", "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10", "J11_closeout"]
    next_scope = continuation["authorization"]
    assert next_scope["persistent_supervisor"] == "setsid_single_instance_pidfile_atomic_cursor"
    assert next_scope["heartbeat_minutes"] == 15
    assert next_scope["task10_attempt3_derived_reanalysis"] is True
    assert next_scope["task10_attempt4"] is False
    assert next_scope["gpu_query_latency_hard_gate"] is False
    assert next_scope["heartbeat_gap_max_ms"] == 2000
    assert next_scope["checkout_stable_manifest_required"] is True
    assert next_scope["total_train_generations_max"] == 12
    assert next_scope["total_replays_max"] == 36
    assert next_scope["training_gpus"] == [1, 2]
    assert next_scope["winner_max_steps"] == 1000
    assert next_scope["gpu0"] is False
    assert next_scope["official_gpu"] is False
    assert next_scope["automatic_upload_or_submission"] is False

    execution = continuation["execution"]
    assert execution["status"] == "track3_jitter_stopped_J5_infrastructure_failure"
    assert execution["j5_unique_pilots"] == 1
    assert execution["j5_task10_nominal_frames"] == 180
    assert execution["exact_pending_accepted"] == [0, 0, 0]

    j5_audit = handoff["track3_jitter_j0_j5_audit"]
    assert j5_audit["status"] == "remediation_required"
    assert (ROOT / j5_audit["audit_report"]).is_file()
    assert (ROOT / j5_audit["audit_summary"]).is_file()
    assert j5_audit["approved"]["j3_task20_task19_p3"] == "2/2"
    assert j5_audit["approved"]["j6_to_j10_not_evaluated"] is True
    assert j5_audit["findings"]["robosuite_quaternion_convention"] == "xyzw"
    assert j5_audit["findings"]["j4_executor_release_approved"] is False
    assert j5_audit["findings"]["corrected_executor_profile_result"] == "not_evaluated"
    assert j5_audit["findings"]["j11_infrastructure_mapped_to_no_family_release"] is True
    assert j5_audit["findings"]["j3_manifest_hash_or_bytes_mismatches"] == "5/21"
    assert j5_audit["findings"]["j11_manifest_hash_or_bytes_mismatches"] == "4/37"

    recovery = handoff["track3_j5_executor_contract_and_atomic_video_recovery_authorization"]
    assert recovery["status"] == "track3_j5_executor_contract_and_atomic_video_recovery_completed_and_audited"
    assert recovery["plan"].startswith("docs/archive/axis_pi05/")
    assert recovery["authorized_phases"] == ["V0", "V1", "V2", "V3", "V4", "V5_closeout"]
    scope = recovery["authorization"]
    assert scope["quaternion_convention"] == "explicit_xyzw"
    assert scope["corrected_task10_demo0_nominal_gpu3_canaries_max"] == 1
    assert scope["support_conservative_profile"] is False
    assert scope["task20_or_task19_manipulation"] is False
    assert scope["j6_generation_or_replay"] is False
    assert scope["training_or_norm"] is False
    assert scope["gpu0"] is False
    assert scope["automatic_upload_or_submission"] is False
    execution = recovery["execution"]
    assert execution["final_commit"] == "11483623b7710430a197231b8ed0662c4813c12d"
    assert execution["v4_unique_workers"] == 1
    assert execution["v4_parent_category"] == "family_failure"
    assert execution["manifest_hash_exact"] == "19/19"
    assert execution["exact_pending_accepted"] == [0, 0, 0]

    audit = handoff["track3_j5_executor_recovery_v0_v5_audit"]
    assert audit["status"] == "remediation_required"
    assert (ROOT / audit["audit_report"]).is_file()
    assert (ROOT / audit["audit_summary"]).is_file()
    assert audit["approved"]["manifest_git_tracked_hash_exact"] == "19/19"
    assert audit["findings"]["explicit_xyzw_fixed"] is True
    assert audit["findings"]["rotation_gain_vs_declared_profile"] == 0.5 / 0.12
    assert audit["findings"]["hdf5_release_transition_index"] == 76
    assert audit["findings"]["contact_completion_corrected_state"] == "not_evaluated"
    assert audit["findings"]["intended_corrected_nominal_profile_result"] == "not_evaluated"

    next_recovery = handoff["track3_support_release_anchor_and_osc_scale_recovery_authorization"]
    assert next_recovery["status"] == "track3_support_release_anchor_and_osc_scale_recovery_completed_and_audited"
    assert next_recovery["plan"].startswith("docs/archive/axis_pi05/")
    assert next_recovery["authorized_phases"] == ["W0", "W1", "W2", "W3", "W4", "W5", "W6_closeout"]
    next_scope = next_recovery["authorization"]
    assert next_scope["reverse_acquisition_anchor_index"] == 75
    assert next_scope["controller_rotation_output_max_rad"] == 0.5
    assert next_scope["physical_rotation_clip_rad"] == 0.12
    assert next_scope["cpu_no_renderer_full_canaries_max"] == 1
    assert next_scope["gpu3_camera_canaries_max_after_cpu_pass"] == 1
    assert next_scope["support_conservative_profile"] is False
    assert next_scope["task20_or_task19_manipulation"] is False
    assert next_scope["training_or_norm"] is False
    assert next_scope["gpu0"] is False
    assert next_recovery["execution"]["final_commit"] == "2997c6f5d8c5b05345d74e0e0c4b96698def727e"
    assert next_recovery["execution"]["w4_parent_category"] == "family_failure"
    assert next_recovery["execution"]["w4_transport_frames"] == 180
    assert next_recovery["execution"]["w5"] == "not_evaluated"
    assert next_recovery["execution"]["manifest_git_tracked_hash_exact"] == "17/17"

    audit = handoff["track3_support_release_anchor_w0_w6_audit"]
    assert audit["status"] == "remediation_required"
    assert (ROOT / audit["audit_report"]).is_file()
    assert (ROOT / audit["audit_summary"]).is_file()
    assert audit["approved"]["w0_immutable_hashes"] == "15/15"
    assert audit["approved"]["manifest_git_tracked_hash_exact"] == "17/17"
    assert audit["findings"]["destination_object_to_eef_transform_applied"] is False
    assert audit["findings"]["destination_eef_position_error_m"] == 0.08171218629122215
    assert audit["findings"]["acquisition_barrier_before_transport"] is False
    assert audit["findings"]["intended_object_frame_nominal_result"] == "not_evaluated"

    authorization = handoff["track3_object_frame_executor_and_bounded_autogen_lora_authorization"]
    assert authorization["status"] == "track3_object_frame_executor_and_bounded_autogen_lora_completed_and_audited"
    assert authorization["plan"].startswith("docs/archive/axis_pi05/")
    assert authorization["authorized_phases"] == ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13_closeout"]
    scope = authorization["authorization"]
    assert scope["object_to_eef_destination_transform"] is True
    assert scope["phase_scoped_contact_and_acquisition_barrier"] is True
    assert scope["active_cursor_single_instance_supervisor"] is True
    assert scope["support_cpu_profiles_max"] == 2
    assert scope["support_gpu3_camera_canaries_max"] == 2
    assert scope["task20_task19_cpu_profiles_max_total"] == 4
    assert scope["task20_task19_gpu3_camera_canaries_max_total"] == 4
    assert scope["total_train_generations_max"] == 12
    assert scope["total_fresh_replays_max"] == 36
    assert scope["forward_reverse_sampling"] == [0.8, 0.2]
    assert scope["training_gpus"] == [1, 2]
    assert scope["winner_max_steps"] == 1000
    assert scope["gpu0"] is False
    assert scope["official_gpu"] is False
    assert scope["automatic_upload_or_submission"] is False
