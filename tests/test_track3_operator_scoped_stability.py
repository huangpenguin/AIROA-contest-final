from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05 import track3_operator_scoped_stability as release


ROOT = Path(__file__).resolve().parents[1]


def test_operator_contract_is_fixed_for_all_three_families() -> None:
    assert release.OPERATOR_CONTRACT[10]["source_atom"] == "(On akita_black_bowl_1 plate_1)"
    assert release.OPERATOR_CONTRACT[20]["source_atom"] == "(In orange_juice_1 basket_1_contain_region)"
    assert release.OPERATOR_CONTRACT[19]["source_atom"] == "(Open wooden_cabinet_1_middle_region)"
    assert release.OPERATOR_CONTRACT[10]["rotation_gate_exclusions"] == ("akita_black_bowl_1",)
    assert all("contact_rule" in release.OPERATOR_CONTRACT[task] for task in release.TASKS)


def test_required_gate_is_fail_closed_for_unexecuted_and_false() -> None:
    assert not release.required_gates_passed(
        {"unrun": release.gate_record(executed=False, value=None)}
    )
    assert not release.required_gates_passed(
        {"failed": release.gate_record(executed=True, value=False)}
    )
    assert release.required_gates_passed(
        {"passed": release.gate_record(executed=True, value=True)}
    )


def test_j0_and_j1_reports_are_locked_and_validated() -> None:
    j0 = json.loads(
        (ROOT / "data/reports/axis_libero_30k/track3_operator_scoped_stability/j0_release_lock.json").read_text(
            encoding="utf-8"
        )
    )
    j1 = json.loads(
        (ROOT / "data/reports/axis_libero_30k/track3_operator_scoped_stability/j1_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert j0["status"] == "track3_operator_scoped_input_locked"
    assert j0["old_i2_summary_sha256"] == release.OLD_I2_SHA256
    assert j0["hard_gate"]["old_i3_zero_locked"] is True
    assert j1["status"] == "track3_operator_scoped_contract_validated"
    assert all(j1["hard_gate"].values())


def test_old_i2_reanalysis_explicitly_blocks_release_without_contact_matrix() -> None:
    report = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/track3_operator_scoped_stability/j2_existing_i2_reanalysis.json"
        ).read_text(encoding="utf-8")
    )
    assert report["old_evidence_immutable"] is True
    assert report["reverse_recorded_fields_pass"] == 9
    assert report["contact_matrix_not_evaluated"] == 9
    assert report["j2_1_cannot_release_j3"] is True


def test_j2_fresh_matrix_has_exact_cpu_job_shape() -> None:
    report = json.loads(
        (ROOT / "data/reports/axis_libero_30k/track3_operator_scoped_stability/j2_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(report["runs"]) == 15
    assert report["new_reverse"]["runs"] == 9
    assert report["new_forward_controls"]["runs"] == 6
    assert report["resource_boundary"]["gpu3_touched"] is False
    assert report["resource_boundary"]["gpu0_touched"] is False
    assert report["resource_boundary"]["se3"] is False
    assert report["resource_boundary"]["pending_autogen"] == 0
