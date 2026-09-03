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
    assert handoff["status"] == "track3_native_reverse_validator_recovery_planned"
    assert handoff["next_plan"].endswith(
        "AXIS_PI05_TRACK3_NATIVE_REVERSE_VALIDATOR_RECOVERY_PLAN.md"
    )
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


def test_predicate_parser_remains_strict() -> None:
    assert parse_atom("(Close wooden_cabinet_1_middle_region)") == (
        "Close",
        ("wooden_cabinet_1_middle_region",),
    )
