from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_ac0_ac6_independent_audit_20260907/summary.json"
AC_ROOT = ROOT / "data/reports/axis_libero_30k/track3_crossfamily_acquisition_recovery_canary"
PLAN = "docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md"
ARCHIVED_BR_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md"
ARCHIVED_AUDIT_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_INDEPENDENT_AUDIT_PLAN.md"


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
    assert audit["audit_input_root_commit"] == "720733047a687de5cb456830467151c0102e4171"
    assert audit["execution_source_commit"] == "afabffdb2124a870dd16bd9e2fcf3ba9df246327"


def test_ac_pointer_and_checkout_closure_remain_exact() -> None:
    pointer = strict_json(AC_ROOT / "latest_attempt.json")
    closure = strict_json(AC_ROOT / "attempts/attempt_0001/checkout_stable_closure.json")
    assert pointer["attempt_id"] == closure["attempt_id"] == 1
    assert sha256(ROOT / pointer["target_final_validation"]) == pointer["target_final_validation_sha256"]
    assert sha256(ROOT / pointer["target_closure"]) == pointer["target_closure_sha256"]
    assert closure["status"] == "passed"


def test_valid_diagnostic_results_and_partial_provenance_are_preserved() -> None:
    audit = strict_json(AUDIT)
    assert audit["verified"]["family_categories"] == ["family_failure"] * 3
    assert audit["verified"]["trace_steps"] == [155, 149, 180]
    assert audit["partial_provenance"]["recorded_bytes"] == 190
    assert audit["partial_provenance"]["raw_partial_currently_available"] is False
    assert audit["partial_provenance"]["reconstructed"] is False


def test_contract_findings_cannot_be_silently_cleared() -> None:
    findings = strict_json(AUDIT)["findings"]
    for key in (
        "ac3_partial_should_have_triggered_shared_infrastructure_stop",
        "ac4_started_after_unresolved_ac3_partial_gate",
        "stage_dependent_gates_misclassified_as_failed_instead_of_not_evaluated",
        "source_release_numpy_identity_bug",
        "required_contact_not_scoped_before_acquisition_barrier",
        "attempt_id_constant_and_overwrite_possible",
        "nested_fixture_heartbeats_missing_from_manifest",
        "ac6_phase_ledger_left_running",
    ):
        assert findings[key] is True


def test_lifecycle_releases_exactly_one_br_plan() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_task10_user_teleop_ready_for_user_calibration"
    assert handoff["next_plan"] == PLAN
    assert handoff["next_phase_only"] == "UT4_explicit_user_takeover_then_UT5_to_UT9"
    assert (ROOT / PLAN).is_file()
    assert ARCHIVED_BR_PLAN.is_file()
    assert ARCHIVED_AUDIT_PLAN.is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_INDEPENDENT_AUDIT_PLAN.md").exists()


def test_br_authority_is_code_first_then_at_most_three_cpu_identities() -> None:
    auth = strict_json(HANDOFF)["track3_acquisition_gate_and_attempt_lifecycle_remediation_authorization"]
    assert auth["status"] == "completed_hard_stop_audited_remediation_required"
    assert auth["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md"
    assert auth["authorized_phases"] == [f"BR{i}" for i in range(11)] + ["BR11_closeout"]
    assert auth["physics_requires_br0_through_br8_passed"] is True
    assert auth["new_cpu_physics_identities_max"] == 3
    assert auth["new_cpu_physics_identities_per_family_max"] == 1
    assert auth["retry_allowed"] is False


def test_br_execution_preserves_dirty_source_correction_and_hard_stop() -> None:
    execution = strict_json(HANDOFF)["track3_acquisition_gate_attempt_lifecycle_remediation_execution"]
    assert execution["status"] == "passed_hard_stop_with_remediation_required"
    assert execution["br0_br7"]["attempt_id"] == 5
    assert execution["br8"]["attempt_id"] == 6
    assert execution["br8"]["clean_source_gate"] is True
    assert execution["historical_br9_br10"]["attempt_id"] == 4
    assert execution["historical_br9_br10"]["clean_root_observed"] is False
    assert execution["historical_br9_br10"]["three_identities_consumed_once"] is True
    assert execution["historical_br9_br10"]["retry"] is False
    assert execution["br11"]["status"] == "passed_hard_stop"
    assert execution["counts"] == {"success": 0, "pending_replay_review": 0, "exact": 0, "accepted": 0, "training_eligible": 0}


def test_br_authority_does_not_release_gpu_data_or_training() -> None:
    auth = strict_json(HANDOFF)["track3_acquisition_gate_and_attempt_lifecycle_remediation_authorization"]["authorization"]
    for key in (
        "old_identity_rerun",
        "gpu0",
        "gpu1",
        "gpu2",
        "gpu3",
        "official_gpu",
        "teleop",
        "generation",
        "replay",
        "accepted_data",
        "training",
        "norm",
        "fast_export",
        "new_candidate_zip",
        "upload",
        "submission",
    ):
        assert auth[key] is False
