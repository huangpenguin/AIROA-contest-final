from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05 import track3_object_frame_executor_and_bounded_autogen_lora as tool


ROOT = Path(__file__).resolve().parents[1]


def test_object_frame_composition_uses_object_to_eef_anchor() -> None:
    result = tool.compose_object_to_eef(
        {"pos_m": [1.0, 2.0, 3.0], "quat": [0.0, 0.0, 0.0, 1.0]},
        {"translation_m": [0.1, 0.2, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, -1.0]},
    )
    assert result["pos_m"] == [1.1, 2.2, 3.3]
    assert tool.qgeo(result["quat"], [0.0, 0.0, 0.0, 1.0]) == 0.0


def test_progress_window_has_explicit_pass_fail_and_not_evaluated_states() -> None:
    assert tool.progress_window_gate([1.0, 0.8, 0.6, 0.4], window=4, min_improvement=0.5)["state"] == "passed"
    assert tool.progress_window_gate([1.0, 0.99, 0.98, 0.97], window=4, min_improvement=0.5)["state"] == "failed"
    assert tool.progress_window_gate([1.0, 0.9], window=4, min_improvement=0.1)["state"] == "not_evaluated"


def test_unentered_gate_is_not_evaluated() -> None:
    result = tool.gate(False, None, reason="stage_not_entered")
    assert result["executed"] is False
    assert result["state"] == "not_evaluated"


def test_x0_x13_audit_archives_x_and_releases_exactly_one_y_plan() -> None:
    handoff = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
        ).read_text(encoding="utf-8")
    )
    assert handoff["status"] == "track3_tz_evidence_and_executor_contract_remediation_completed_hard_stop_pending_review"
    assert handoff["next_plan"] == (
        "docs/AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md"
    )
    assert handoff["next_phase_only"] == "new_reviewed_plan_required"
    assert (ROOT / handoff["next_plan"]).is_file()
    assert not (
        ROOT
        / "docs/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md"
    ).exists()
    assert (
        ROOT
        / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_OBJECT_FRAME_EXECUTOR_AND_BOUNDED_AUTOGEN_LORA_PLAN.md"
    ).is_file()

    old = handoff["track3_object_frame_executor_and_bounded_autogen_lora_authorization"]
    assert old["status"] == "track3_object_frame_executor_and_bounded_autogen_lora_completed_and_audited"
    assert old["execution"]["x4_parent_category"] == "infrastructure"
    assert old["execution"]["x4_physics_steps"] == 0
    assert old["execution"]["x5_to_x12"] == "not_evaluated"

    audit = handoff["track3_object_frame_x0_x13_audit"]
    assert audit["status"] == "remediation_required"
    assert (ROOT / audit["audit_report"]).is_file()
    assert (ROOT / audit["audit_summary"]).is_file()
    assert audit["verified"]["manifest_git_tracked_hash_exact"] == "13/13"
    assert audit["findings"]["runtime_dependency_preflight_complete"] is False
    assert audit["findings"]["progress_window_enforced"] is False
    assert audit["findings"]["source_release_gate_phase"] == "incorrectly_at_close_before_lift"

    next_authorization = handoff["track3_runtime_preflight_and_bounded_continuation_authorization"]
    assert next_authorization["status"] == "track3_runtime_preflight_and_bounded_continuation_completed_and_audited"
    assert next_authorization["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md"
    assert next_authorization["authorized_phases"] == [
        "Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "Y10", "Y11", "Y12", "Y13_closeout"
    ]
    scope = next_authorization["authorization"]
    assert scope["runtime_symbol_and_startup_smoke"] is True
    assert scope["phase_correct_contact_and_source_release"] is True
    assert scope["progress_window_stall_enforced"] is True
    assert scope["total_train_generations_max"] == 12
    assert scope["total_fresh_replays_max"] == 36
    assert scope["gpu0"] is False
    assert scope["official_gpu"] is False
    assert scope["automatic_upload_or_submission"] is False

    y_audit = handoff["track3_runtime_preflight_y0_y2_audit"]
    assert y_audit["status"] == "remediation_required"
    assert y_audit["findings"]["child_exact_argv_dispatches"] is False
    z_authorization = handoff["track3_y3_launch_phase_ledger_remediation_authorization"]
    assert z_authorization["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md"
    assert z_authorization["next_phase_only"] == "superseded_by_track3_z5_derived_release_and_conservative_continuation_plan"
