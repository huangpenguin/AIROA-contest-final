from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_tonight_tz0_tz9_audit_20260906/summary.json"


def test_tonight_audit_archives_old_plan_and_releases_one_remediation_plan() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_ef_failure_provenance_and_lifecycle_remediation_completed_hard_stop"
    assert handoff["next_plan"] is None
    assert handoff["next_phase_only"] == "new_reviewed_plan_required"
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_TONIGHT_ZIP_AND_CROSSFAMILY_RECOVERY_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_TONIGHT_ZIP_AND_CROSSFAMILY_RECOVERY_PLAN.md").is_file()


def test_tonight_audit_preserves_fallback_and_failure_semantics() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["classification"] == audit["status"] == "remediation_required"
    assert audit["fallback"]["readiness"] == "fallback_no_track3_improvement"
    assert audit["fallback"]["track3_improvement"] is False
    assert audit["verified"]["cpu_attempts"] == 7
    assert audit["verified"]["first_attempt_category"] == "infrastructure_failure"
    assert audit["verified"]["diagnostic_family_failures"] == {
        "support_remove": 2,
        "containment_remove": 2,
        "articulation_close": 2,
    }
    assert audit["verified"]["tz3_to_tz8"] == "not_evaluated"
    assert audit["findings"]["tz1_pass_claim_valid"] is False
    assert audit["findings"]["tz2_compliant_completion_claim_valid"] is False


def test_remediation_authorization_keeps_all_expansive_actions_off() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    authorization = handoff["track3_tz_evidence_and_executor_contract_remediation_authorization"]
    assert authorization["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md"
    assert authorization["next_phase_only"] == "superseded_by_track3_er_evidence_anchor_and_finalizer_verification_plan"
    scope = authorization["authorization"]
    assert scope["new_cpu_identities_max"] == 3
    assert scope["old_tz_identity_rerun"] is False
    for key in ("gpu0", "gpu1", "gpu2", "gpu3", "official_gpu", "teleop", "generation", "replay", "accepted_data", "training", "new_candidate_zip", "upload", "submission"):
        assert scope[key] is False
