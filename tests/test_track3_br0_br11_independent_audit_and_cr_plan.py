from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_br0_br11_independent_audit_20260907/summary.json"
BR_ROOT = ROOT / "data/reports/axis_libero_30k/track3_acquisition_gate_attempt_lifecycle_remediation"
PLAN = "docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_INDEPENDENT_AUDIT_PLAN.md"
ARCHIVED_AUDIT_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ACQUISITION_GATE_ATTEMPT_LIFECYCLE_REMEDIATION_INDEPENDENT_AUDIT_PLAN.md"
ARCHIVED_CR_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_PLAN.md"
CR_SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_cr0_cr14_execution_20260907/summary.json"


def strict_json(path: Path) -> dict:
    def reject(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_has_exactly_one_remediation_verdict() -> None:
    audit = strict_json(AUDIT)
    assert audit["status"] == audit["classification"] == "remediation_required"
    assert audit["audit_input_root_commit"] == "474d36d7e36aa3f428c8fff3fbf38b8d5db03f05"


def test_actual_partial_is_not_replaced_by_synthetic_observation() -> None:
    provenance = strict_json(AUDIT)["partial_provenance"]
    actual = ROOT / provenance["actual_raw_partial_path"]
    synthetic = ROOT / provenance["synthetic_observation_path"]
    assert actual.stat().st_size == provenance["actual_raw_partial_bytes"] == 43
    assert sha256(actual) == provenance["actual_raw_partial_sha256"]
    assert synthetic.stat().st_size == provenance["synthetic_observation_bytes"] == 39
    assert sha256(synthetic) == provenance["synthetic_observation_sha256"]
    assert actual.read_bytes() != synthetic.read_bytes()
    assert provenance["synthetic_observation_is_actual_partial"] is False


def test_br_traces_are_exact_duplicates_of_ac_traces() -> None:
    br = BR_ROOT / "attempts/attempt_0004/runs/traces"
    ac = ROOT / "data/reports/axis_libero_30k/track3_crossfamily_acquisition_recovery_canary/attempts/attempt_0001/runs/traces"
    families = ("support", "containment", "articulation")
    for family in families:
        br_trace = next(br.glob(f"br_{family}*_demo0.npz"))
        ac_trace = next(ac.glob(f"ac_{family}*_demo0.npz"))
        assert sha256(br_trace) == sha256(ac_trace)


def test_findings_cannot_be_silently_cleared() -> None:
    findings = strict_json(AUDIT)["findings"]
    for key in (
        "dirty_expected_identity_can_pass_parent_identity_gate",
        "manifest_current_complete_set_claim_incorrect",
        "actual_partial_excluded_from_manifest",
        "synthetic_constant_mislabeled_as_raw_partial_observation",
        "unknown_telemetry_can_be_misclassified_family_failure",
        "normalized_actions_double_normalized",
        "three_br_traces_add_no_behavioral_diversity",
    ):
        assert findings[key] is True


def test_lifecycle_releases_exactly_one_cr_plan() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_clean_launch_and_diversified_canary_recovery_completed_hard_stop_pending_independent_audit"
    assert handoff["next_plan"] == PLAN
    assert handoff["next_phase_only"] == "independent_audit_only_then_hard_stop"
    assert (ROOT / PLAN).is_file()
    assert ARCHIVED_CR_PLAN.is_file()
    assert ARCHIVED_AUDIT_PLAN.is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_PLAN.md").exists()


def test_cr_authority_is_code_first_and_bounded() -> None:
    auth = strict_json(HANDOFF)["track3_clean_launch_and_diversified_canary_recovery_authorization"]
    assert auth["status"] == "completed_hard_stop"
    assert auth["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_PLAN.md"
    assert auth["authorized_phases"] == [f"CR{i}" for i in range(15)]
    assert auth["physics_requires_cr0_through_cr8_passed"] is True
    assert auth["new_cpu_physics_identities_max"] == 6
    assert auth["new_cpu_physics_identities_per_family_max"] == 2
    assert auth["same_strategy_retry_allowed"] is False
    assert auth["gpu3_replay_requires_cpu_success"] is True


def test_cr_failure_policies_are_fail_closed() -> None:
    auth = strict_json(HANDOFF)["track3_clean_launch_and_diversified_canary_recovery_authorization"]
    assert auth["family_failure_policy"] == "one_frozen_alternative_then_stop_family"
    assert auth["shared_infrastructure_failure_policy"] == "sticky_global_hard_stop"
    assert auth["byte_identical_trace_policy"] == "no_effect_variant_family_failure"
    assert auth["success_disposition"] == "pending_replay_or_human_review_only"


def test_cr_authority_does_not_release_training_zip_or_submission() -> None:
    permissions = strict_json(HANDOFF)["track3_clean_launch_and_diversified_canary_recovery_authorization"]["authorization"]
    assert permissions["new_cpu_no_renderer_physics"] is True
    assert permissions["gpu3_exact_action_camera_replay"] is True
    for key in ("old_identity_rerun", "gpu0", "gpu1", "gpu2", "official_gpu", "unattended_teleop", "generation", "accepted_data", "training", "norm", "fast_export", "new_candidate_zip", "upload", "submission"):
        assert permissions[key] is False


def test_cr_execution_is_closed_with_zero_success_and_no_replay() -> None:
    summary = strict_json(CR_SUMMARY)
    assert summary["status"] == "completed_hard_stop_pending_independent_audit"
    assert summary["cr9_to_cr10"]["total_physics_identities"] == 6
    assert summary["cr9_to_cr10"]["family_failure_count"] == 6
    assert summary["cr9_to_cr10"]["retry_count"] == 0
    assert summary["counts"] == {"success": 0, "pending_replay_review": 0, "exact": 0, "accepted": 0, "training_eligible": 0, "new_candidate_zip": 0}
    assert summary["cr11_to_cr14"]["cr11_camera_replay"] == "not_evaluated"
    assert summary["cr11_to_cr14"]["physics_rerun_for_correction"] is False
