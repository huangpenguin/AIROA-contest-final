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


def test_predicate_parser_remains_strict() -> None:
    assert parse_atom("(Close wooden_cabinet_1_middle_region)") == (
        "Close",
        ("wooden_cabinet_1_middle_region",),
    )
