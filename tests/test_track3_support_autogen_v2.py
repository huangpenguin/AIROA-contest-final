from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.axis_pi05.track3_support_autogen_v2 import (
    IDENTITIES,
    language_has_internal_tokens,
    parse_atom,
    render_typed_language,
    validate_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v2_accounting_keeps_task_and_episode_totals_separate() -> None:
    matrix = json.loads(
        (ROOT / "data/manifests/track3_public_primitives_v2/coverage_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert matrix["task"]["row_count"] == 195
    assert matrix["task"]["identity_sum"] == 195
    assert matrix["episode"]["row_count"] == 21961
    assert matrix["episode"]["identity_sum"] == 21961
    assert matrix["task"]["identity_counts"]["exact_reverse_demo"] == 0
    assert matrix["episode"]["identity_counts"]["exact_reverse_demo"] == 0


def test_accounting_rejects_multiple_identity_values() -> None:
    with pytest.raises(RuntimeError, match="invalid identity"):
        validate_rows(
            [{"identity": list(IDENTITIES[:2]), "operator": "support_remove"}],
            1,
            "synthetic",
        )


def test_typed_language_has_no_internal_identifiers() -> None:
    for family in ("support_remove", "containment_remove", "articulation_close"):
        language = render_typed_language(family)
        assert not language_has_internal_tokens(language)
    assert language_has_internal_tokens("put object_1 in target_region")


def test_dynamic_gate_report_is_fresh_process_fail_closed() -> None:
    report = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/"
            "g1_s3_dynamic_gates.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "track3_reverse_semantic_adapter_v2_failed_support_remove"
    assert len(report["fresh_process_runs"]) == 9
    assert report["failed_families"] == [
        "support_remove",
        "containment_remove",
        "articulation_close",
    ]
    assert report["state_layout"]["hdf5_states_qpos_offset"] == 1
    assert report["hard_gate"]["hashes_recorded_each_run"] is True
    assert report["hard_gate"]["all_families_3_of_3"] is False


def test_historical_dynamic_failure_is_preserved_but_new_plan_repairs_validator() -> None:
    handoff = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
        ).read_text(encoding="utf-8")
    )
    assert handoff["status"] == "track3_support_runtime_gate_remediation_and_gold_continuation_planned"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md"
    audit = handoff["track3_support_se3_g0_g1_audit"]
    assert audit["status"] == "remediation_required"
    assert (
        audit["invalidating_findings"]["historical_dynamic_gate_interpretation"]
        == "inconclusive_not_reverse_semantics_failure"
    )
    authorization = handoff["track3_dynamic_gate_reconciliation_authorization"]
    assert authorization["authorization"]["reverse_native_reset_predicate_evaluation"] is True
    assert authorization["authorization"]["old_g2_hdf5_se3_mining"] is False
    assert authorization["authorization"]["real_reverse_canary"] is False
    assert authorization["authorization"]["lora_or_dense_training"] is False

    d0_d4_audit = handoff["track3_dynamic_gate_d0_d4_audit"]
    assert d0_d4_audit["status"] == "remediation_required"
    assert d0_d4_audit["d2_interpretation"]["valid_predicate_settle_results"] == 0
    assert d0_d4_audit["d2_interpretation"]["reverse_semantics_conclusion"] == "inconclusive"
    assert d0_d4_audit["source_review"]["post_d2_fix_direction_source_aligned"] is True
    recovery = handoff["track3_native_reverse_validator_recovery_authorization"]
    assert recovery["authorization"]["gpu3_task10_recovery_canary_max"] == 1
    assert recovery["authorization"]["se3_or_grasp_executor"] is False
    assert recovery["authorization"]["lora_or_dense_training"] is False

    r0_r4_audit = handoff["track3_native_validator_r0_r4_audit"]
    assert r0_r4_audit["status"] == "remediation_required"
    assert r0_r4_audit["r2_predicate_evidence"]["initial_true_count"] == 4
    assert r0_r4_audit["r2_predicate_evidence"]["bowl_plate_xy_exact"] is True
    assert r0_r4_audit["r2_predicate_evidence"]["settle_executed"] is False
    assert r0_r4_audit["report_findings"]["tri_state_gate_required"] is True
    stabilization = handoff["track3_reverse_init_stabilization_authorization"]
    assert stabilization["authorization"]["release_stabilization_step"] == 5
    assert stabilization["authorization"]["bddl_region_or_predicate_modification"] is False
    assert stabilization["authorization"]["se3_or_grasp_executor"] is False
    assert stabilization["authorization"]["lora_or_dense_training"] is False
    execution = handoff["track3_reverse_init_stabilization_execution"]
    assert execution["status"] == "track3_reverse_init_stabilization_failed_named_scene_stability"
    assert execution["i2"]["fresh_process_runs"] == 12
    assert execution["i3"]["status"] == "not_run_due_to_i2_gate"
    assert execution["resource_boundary"]["gpu3_touched"] is False
    latest_audit = handoff["track3_init_stabilization_i0_i4_audit"]
    assert latest_audit["status"] == "remediation_required"
    assert latest_audit["execution_approved"] is True
    assert latest_audit["findings"]["init_truth_vector_transitions_step5_to20"] == 0
    assert latest_audit["findings"]["goal_truth_vector_transitions_step5_to20"] == 0
    assert latest_audit["findings"]["containment"]["full_scene_failure_source"].startswith(
        "background_"
    )
    operator_scope = handoff["track3_operator_scoped_stability_authorization"]
    assert operator_scope["authorization"]["new_reverse_cpu_processes"] == 9
    assert operator_scope["authorization"]["new_forward_cpu_processes"] == 6
    assert operator_scope["authorization"]["gpu3_task10_canary_max"] == 1
    assert operator_scope["authorization"]["gpu3_three_family_release_runs_max"] == 9
    assert operator_scope["authorization"]["se3_or_grasp_executor"] is False
    assert operator_scope["authorization"]["lora_or_dense_training"] is False
    j_execution = handoff["track3_operator_scoped_stability_release_execution"]
    assert j_execution["status"] == (
        "track3_operator_scoped_cpu_release_failed_truth_vector_transition_task19_run1"
    )
    assert j_execution["j2"]["reverse_pass"] == "8/9"
    assert j_execution["j3"]["status"] == "not_evaluated"
    convergence_audit = handoff["track3_operator_scoped_j0_j4_audit"]
    assert convergence_audit["status"] == "remediation_required"
    assert convergence_audit["transition"]["role"] == "background_initializer"
    assert convergence_audit["transition"]["from"] is False
    assert convergence_audit["transition"]["to"] is True
    assert convergence_audit["transition"]["critical_or_goal_transition"] is False
    assert convergence_audit["evidence_gap"]["raw_worker_reports_expected"] == 15
    assert convergence_audit["evidence_gap"]["raw_worker_reports_git_tracked"] == 0
    convergence = handoff["track3_background_predicate_convergence_authorization"]
    assert convergence["authorization"]["raw_worker_reports_to_track"] == 15
    assert convergence["authorization"]["new_cpu_simulator_runs"] == 0
    assert convergence["authorization"]["gpu3_task19_canary_max"] == 1
    assert convergence["authorization"]["se3_or_grasp_executor"] is False
    assert convergence["authorization"]["lora_or_dense_training"] is False
    assert convergence["monotonic_contract"]["true_to_false_allowed"] is False


def test_predicate_parser_remains_strict() -> None:
    assert parse_atom("(Close wooden_cabinet_1_middle_region)") == (
        "Close",
        ("wooden_cabinet_1_middle_region",),
    )
