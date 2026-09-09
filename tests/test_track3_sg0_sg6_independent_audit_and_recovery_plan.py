from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_sg0_sg6_independent_audit_20260907/summary.json"
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
SG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_contact_truth_and_support_gold_demo"
PLAN = "docs/archive/axis_pi05/AXIS_PI05_TRACK3_OPERATIONS_O1_O2_RED_MUG_NATIVE_MUJOCO_PLAN.md"
ARCHIVED_SG_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CONTACT_TRUTH_AND_SUPPORT_GOLD_DEMO_PLAN.md"


def strict_json(path: Path) -> dict:
    def reject(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def test_audit_has_one_remediation_verdict_and_exact_commit() -> None:
    audit = strict_json(AUDIT)
    assert audit["status"] == audit["classification"] == "remediation_required"
    assert audit["audit_input_root_commit"] == "55acd39c17d82c89b4f8e809b8f84d9623c3eb81"
    assert audit["execution_source_commit"] == "d5c319b055431d2ec46681775438c91ad64a2be0"
    assert audit["post_failure_signature_fix_commit"] == "3e156dd6177e435316b721a79ff7a0da6d0359f8"


def test_sg6_failure_is_closed_and_has_no_physics_episode() -> None:
    audit = strict_json(AUDIT)
    child = strict_json(SG_ROOT / "attempts/attempt_0004/canary/child_result.json")
    current = strict_json(SG_ROOT / "current_attempt.json")
    assert child["category"] == "infrastructure_failure"
    assert child["error"]["type"] == "TypeError"
    assert audit["verified"]["environment_created_count"] == 0
    assert audit["verified"]["physics_identity_count"] == 0
    assert audit["verified"]["sg6_trajectory_count"] == 0
    assert current["active"] is False and current["status"] == "hard_stop"
    assert not (SG_ROOT / "attempts/attempt_0004/canary/trajectory.npz").exists()


def test_attempt_0004_manifest_and_outer_hash_are_exact() -> None:
    attempt = SG_ROOT / "attempts/attempt_0004"
    manifest = strict_json(attempt / "final_manifest.json")
    assert len(manifest["entries"]) == 8
    for row in manifest["entries"]:
        path = attempt / row["relative_path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    outer = strict_json(attempt / "outer_hash.json")
    assert hashlib.sha256((attempt / "final_manifest.json").read_bytes()).hexdigest() == outer["final_manifest_sha256"]
    assert strict_json(attempt / "closure.json")["complete_set"]["exact"] is True


def test_audit_records_all_success_path_blockers() -> None:
    findings = strict_json(AUDIT)["findings"]
    assert findings["acquisition_gate_requires_entire_precontact_prefix"] is True
    assert findings["source_release_required_before_lift"] is True
    assert findings["goal_receiver_main_table_misclassified_unrelated"] is True
    assert findings["goal_continuity_is_only_final_frame"] is True
    assert findings["signature_fix_is_valid_but_not_sufficient_for_physics_release"] is True


def test_no_human_episode_is_waiting_for_review() -> None:
    verified = strict_json(AUDIT)["verified"]
    assert verified["human_episode_review_queue_count"] == 0
    assert verified["replay_video_count"] == 0
    assert verified["teleop_recorded_candidate_count"] == 0
    assert verified["pending_gold_review_count"] == 0
    assert verified["accepted_count"] == verified["training_eligible_count"] == 0


def test_lifecycle_has_exactly_one_recovery_plan() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_multitask_gold_and_r6_short_continuation_reviewed_ready_for_mg0"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_MULTITASK_GOLD_AND_R6_SHORT_CONTINUATION_PLAN.md"
    assert handoff["next_phase_only"] == "MG0_evidence_lock_and_recorder_budget_remediation"
    assert (ROOT / PLAN).is_file()
    assert ARCHIVED_SG_PLAN.is_file()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_CONTACT_TRUTH_AND_SUPPORT_GOLD_DEMO_PLAN.md").exists()


def test_stop_policy_is_scoped_but_never_auto_retries() -> None:
    policy = strict_json(AUDIT)["stop_policy"]
    assert policy["immediate_child_stop_preserved"] is True
    assert policy["identity_telemetry_gpu_partial_or_supervisor_unknown"] == "sticky_global_hard_stop"
    assert "no_auto_retry" in policy["deterministic_pre_env_code_exception"]
    auth = strict_json(HANDOFF)["track3_support_runtime_gate_remediation_and_gold_continuation_authorization"]
    assert auth["same_strategy_retry_allowed"] is False
    assert auth["corrected_primary_physics_identities_max"] == 1
    assert auth["conditional_alternative_physics_identities_max"] == 1


def test_recovery_authority_keeps_training_zip_and_submission_closed() -> None:
    auth = strict_json(HANDOFF)["track3_support_runtime_gate_remediation_and_gold_continuation_authorization"]
    assert auth["human_episode_review_queue_count_at_authorization"] == 0
    assert auth["human_teleop_requires_user_online_control"] is True
    assert auth["gpu3_exact_action_replays_per_success_max"] == 3
    permissions = auth["authorization"]
    for key in (
        "containment_or_articulation_physics",
        "old_identity_rerun",
        "gpu0",
        "gpu1",
        "gpu2",
        "official_gpu",
        "unattended_teleop",
        "accepted_promotion",
        "scale_generation",
        "norm",
        "training",
        "fast_export",
        "new_candidate_zip",
        "upload",
        "submission",
    ):
        assert permissions[key] is False
