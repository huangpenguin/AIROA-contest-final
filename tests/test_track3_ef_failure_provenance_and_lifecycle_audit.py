from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_audit_20260906/summary.json"
EG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_remediation"


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


def test_independent_eg_audit_is_approved_and_separate() -> None:
    audit = strict_json(SUMMARY)
    handoff = strict_json(HANDOFF)
    assert audit["status"] == audit["classification"] == "approved"
    assert audit["audit_input_execution_source_commit"] == "0799142b92ad3e799dd2d1b78ca964f07dd87587"
    assert audit["audit_input_evidence_commit"] == "5d20e25d3890e937c0278ea2656236449b82898f"
    assert audit["audit_report_commit"] == "ad4f34e81ccaa39397f5ac550dd41f21734f47cb"
    assert all(audit["findings"].values())
    current = handoff["track3_ef_failure_provenance_and_lifecycle_remediation_audit"]
    assert current["status"] == current["classification"] == "approved"
    assert current["audit_report_commit"] == audit["audit_report_commit"]
    assert current["next_plan"] is None
    assert current["next_phase_only"] == "new_reviewed_plan_required"
    assert handoff["status"] == "track3_contact_truth_and_support_gold_demo_planned"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_CONTACT_TRUTH_AND_SUPPORT_GOLD_DEMO_PLAN.md"
    assert handoff["next_phase_only"] == "SG0_to_SG12_then_hard_stop"


def test_eg_audit_preserves_unavailable_gap_and_final_closure() -> None:
    audit = strict_json(SUMMARY)
    gap = audit["provenance_gap"]
    assert gap["first_failed_final_validation_raw"] == "unavailable"
    assert gap["first_failed_test_stdout_stderr_raw"] == "unavailable"
    assert gap["first_failed_partial_inventory_raw"] == "unavailable"
    assert gap["raw_recovery_candidate_found"] is False
    closure = strict_json(EG_ROOT / "root_attempt_0005_checkout_stable_closure.json")
    assert closure["status"] == "passed"
    validation = strict_json(EG_ROOT / "attempts/attempt_0005/final_validation.json")
    assert validation["status"] == "passed"
    assert sha256(EG_ROOT / "attempts/attempt_0005/final_validation.json") == closure["attempt_final_validation_sha256"]
    assert strict_json(EG_ROOT / "attempts/attempt_0003/final_validation.json")["status"] == "failed"
    assert strict_json(EG_ROOT / "attempts/attempt_0004/final_validation.json")["status"] == "passed"
