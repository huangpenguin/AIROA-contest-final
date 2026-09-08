from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_cr0_cr14_independent_audit_20260907/summary.json"
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
MASTER = ROOT / "PARC_TRACK3_MASTER_PLAN_20260907.md"
PLAN = "docs/AXIS_PI05_TRACK3_OPERATIONS_O1_O2_RED_MUG_NATIVE_MUJOCO_PLAN.md"
ARCHIVED_EXECUTION_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md"
ARCHIVED_AUDIT_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_INDEPENDENT_AUDIT_PLAN.md"
CR_TRACES = ROOT / "data/reports/axis_libero_30k/track3_clean_launch_and_diversified_canary_recovery/attempts/attempt_0004/runs/traces"


def strict_json(path: Path) -> dict:
    def reject(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def test_audit_has_one_remediation_verdict_and_exact_input() -> None:
    audit = strict_json(AUDIT)
    assert audit["status"] == audit["classification"] == "remediation_required"
    assert audit["audit_input_root_commit"] == "0ce605cf33d33a0d670d190ecf7013d122d8a1a0"
    assert audit["audited_execution_evidence_commit"] == "760dc65adc74193112692ab003e19a74d588ea9f"


def test_master_plan_is_an_exact_advisory_input() -> None:
    audit = strict_json(AUDIT)["master_plan_reference"]
    assert MASTER.stat().st_size == audit["bytes"] == 24677
    assert hashlib.sha256(MASTER.read_bytes()).hexdigest() == audit["sha256"]
    assert audit["role"] == "advisory_strategy_input"
    assert audit["external_claims_independently_verified"] is False


def test_cr_contact_arrays_preserve_the_invalid_runtime_pattern() -> None:
    total = 0
    for path in sorted(CR_TRACES.glob("*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            required = np.asarray(arrays["required_contact"], dtype=bool)
            complete = np.asarray(arrays["contact_telemetry_complete"], dtype=bool)
            unrelated = np.asarray(arrays["unrelated_contact_runtime"], dtype=bool)
            assert required.all()
            assert np.array_equal(required, complete)
            assert not unrelated.any()
            total += len(required)
    assert total == 966


def test_audit_preserves_six_family_failures_but_rejects_contact_promotion() -> None:
    audit = strict_json(AUDIT)
    assert len(audit["failure_matrix"]) == 6
    assert {row["category"] for row in audit["failure_matrix"]} == {"family_failure"}
    assert audit["findings"]["required_contact_conflated_with_telemetry_completeness"] is True
    assert audit["findings"]["unrelated_contact_hardcoded_false"] is True
    assert audit["verified"]["accepted_data_count"] == 0
    assert audit["verified"]["training_started"] is False


def test_support_is_selected_from_measured_failure_progress() -> None:
    audit = strict_json(AUDIT)
    assert audit["priority_family"] == "support_remove"
    support_alt = next(row for row in audit["failure_matrix"] if row["profile_id"] == "cr_support_alternative_20260907_0001")
    assert support_alt["last_stage"] == "descend_object"
    assert np.isclose(support_alt["last_eef_error_m"], 0.013362138900100747)
    assert support_alt["source_false_steps"] == 3


def test_lifecycle_has_exactly_one_new_active_plan() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_operations_o1_0_ready_for_native_vnc_no_save_canary"
    assert handoff["next_plan"] == PLAN
    assert handoff["next_phase_only"] == "O1_1_native_mujoco_turbovnc_no_save_canary_requires_user_online"
    assert (ROOT / PLAN).is_file()
    assert ARCHIVED_AUDIT_PLAN.is_file()
    assert ARCHIVED_EXECUTION_PLAN.is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_INDEPENDENT_AUDIT_PLAN.md").exists()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md").exists()


def test_support_gold_authority_is_bounded_and_human_gated() -> None:
    auth = strict_json(HANDOFF)["track3_contact_truth_and_support_gold_demo_authorization"]
    assert auth["priority_family"] == "support_remove"
    assert auth["fresh_cpu_support_identities_max"] == 2
    assert auth["human_teleop_recorded_candidates_max"] == 1
    assert auth["human_teleop_requires_user_online_control"] is True
    assert auth["gpu3_exact_action_replays_per_success_max"] == 3
    assert auth["success_disposition"] == "pending_gold_review_only"


def test_support_gold_authority_does_not_release_training_or_submission() -> None:
    permissions = strict_json(HANDOFF)["track3_contact_truth_and_support_gold_demo_authorization"]["authorization"]
    assert permissions["contact_truth_and_code_fixtures"] is True
    assert permissions["new_support_cpu_no_renderer_physics"] is True
    assert permissions["user_controlled_support_teleop"] is True
    assert permissions["gpu3_exact_action_replay_after_success"] is True
    for key in ("containment_or_articulation_physics", "old_identity_rerun", "gpu0", "gpu1", "gpu2", "official_gpu", "unattended_teleop", "accepted_promotion", "scale_generation", "norm", "training", "fast_export", "new_candidate_zip", "upload", "submission"):
        assert permissions[key] is False
