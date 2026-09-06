from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_ef0_ef4_independent_audit_20260906/summary.json"
EF_ROOT = ROOT / "data/reports/axis_libero_30k/track3_er_evidence_anchor_and_finalizer_verification"
PLAN = "docs/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md"
ARCHIVED_EG_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_audit_has_one_remediation_verdict() -> None:
    audit = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert audit["status"] == audit["classification"] == "remediation_required"
    assert audit["findings"]["ef0_to_ef3_technical_contracts_valid"] is True
    assert audit["findings"]["corrected_ef4_checkout_closure_valid"] is True


def test_first_failed_final_validation_is_not_misrepresented_as_raw() -> None:
    audit = json.loads(SUMMARY.read_text(encoding="utf-8"))
    first = json.loads((EF_ROOT / "ef4_first_attempt_failure.json").read_text(encoding="utf-8"))
    assert first["status"] == "failed"
    assert first["historical_report_replaced_by_corrected_attempt"] is True
    assert audit["findings"]["first_failed_final_validation_raw_preserved"] is False
    assert audit["findings"]["lost_raw_recovered_from_repository_tmp_or_unreachable_git_blobs"] is False


def test_existing_corrected_checkout_closure_remains_exact() -> None:
    closure = json.loads((EF_ROOT / "checkout_stable_closure.json").read_text(encoding="utf-8"))
    assert sha256(EF_ROOT / "final_validation.json") == closure["final_validation_sha256"]
    assert sha256(EF_ROOT / "artifact_manifest.json") == closure["artifact_manifest_sha256"]
    assert sha256(EF_ROOT / "outer_hash.json") == closure["outer_hash_sha256"]
    assert sha256(EF_ROOT / "heartbeat.json") == closure["closed_heartbeat_sha256"]


def test_lifecycle_closes_eg_after_independent_audit() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_acquisition_gate_attempt_lifecycle_remediation_planned"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md"
    assert handoff["next_phase_only"] == "BR0_to_BR11_then_hard_stop"
    assert not (ROOT / PLAN).exists()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").is_file()
    assert ARCHIVED_EG_PLAN.is_file()


def test_eg_authorization_remains_code_fixture_only() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    auth = handoff["track3_ef_failure_provenance_and_lifecycle_remediation_authorization"]
    assert ROOT / auth["plan"] == ARCHIVED_EG_PLAN
    assert auth["status"] == "completed_hard_stop"
    assert auth["authorized_phases"] == ["EG0", "EG1", "EG2", "EG3", "EG4_closeout"]
    for key in ("new_physics", "old_identity_rerun", "gpu0", "gpu1", "gpu2", "gpu3", "official_gpu", "teleop", "generation", "replay", "accepted_data", "training", "new_candidate_zip", "upload", "submission"):
        assert auth["authorization"][key] is False
