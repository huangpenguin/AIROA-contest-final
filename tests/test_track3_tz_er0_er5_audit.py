from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_tz_er0_er5_audit_20260906/summary.json"
PLAN = "docs/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md"


def test_er_audit_has_one_remediation_verdict_and_one_active_plan() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["classification"] == audit["status"] == "remediation_required"
    assert handoff["status"] == "track3_er_evidence_anchor_and_finalizer_verification_planned"
    assert handoff["next_plan"] == audit["next_plan"] == PLAN
    assert handoff["next_phase_only"] == audit["next_phase_only"] == "EF0_to_EF4_then_hard_stop"
    assert (ROOT / PLAN).is_file()


def test_er_plan_is_archived_without_reauthorizing_consumed_physics() -> None:
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md").is_file()
    plan = (ROOT / PLAN).read_text(encoding="utf-8")
    assert "任何新physics identity或旧identity重跑" in plan
    assert "GPU0–3" in plan


def test_er_audit_preserves_valid_failures_and_reports_untracked_evidence() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["findings"]["three_family_failure_classifications_valid"] is True
    assert audit["findings"]["er_namespace_git_tracked"] is False
    assert audit["verified"]["new_cpu_attempts"] == 3
    assert audit["verified"]["successful_families"] == 0
    assert audit["verified"]["trace_no_pickle_finite"] == "3/3"
    assert audit["verified"]["controller_saturation_counts"] == {
        "support_remove": 10,
        "containment_remove": 5,
        "articulation_close": 10,
    }


def test_ef_authorization_keeps_expansive_actions_off() -> None:
    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    authorization = handoff["track3_er_evidence_anchor_and_finalizer_verification_authorization"]
    assert authorization["plan"] == PLAN
    assert authorization["new_physics_identities"] == 0
    for key in ("gpu0", "gpu1", "gpu2", "gpu3", "official_gpu", "teleop", "generation", "replay", "accepted_data", "training", "new_candidate_zip", "upload", "submission"):
        assert authorization["authorization"][key] is False
