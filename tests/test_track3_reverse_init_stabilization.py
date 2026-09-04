from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05 import track3_reverse_init_stabilization as stabilization


ROOT = Path(__file__).resolve().parents[1]


def test_operator_critical_mapping_is_fixed_and_goal_is_separate() -> None:
    report = stabilization.i1_semantics()
    assert report["status"] == "track3_reverse_init_stabilization_semantics_validated"
    assert report["critical_relations"] == {
        "10": "(On akita_black_bowl_1 plate_1)",
        "20": "(In orange_juice_1 basket_1_contain_region)",
        "19": "(Open wooden_cabinet_1_middle_region)",
    }
    for task in stabilization.TASK_ORDER:
        init_required = [row for row in report["reverse"][str(task)]["init"] if row["gate_required"]]
        goal_required = [row for row in report["reverse"][str(task)]["goal"] if row["gate_required"]]
        assert len(init_required) == 1
        assert init_required[0]["role"] == "operator_critical"
        assert len(goal_required) == 1
        assert goal_required[0]["role"] == "operator_critical_goal"


def test_tri_state_and_required_gate_fail_closed() -> None:
    assert stabilization.tri_state(False, False) == "not_evaluated"
    assert stabilization.tri_state(True, True) == "passed"
    assert stabilization.tri_state(True, False) == "failed"
    assert not stabilization.required_gates_passed(
        {"unrun": stabilization.gate_record(executed=False, value=None)}
    )


def test_worker_exception_has_nonzero_exit_code() -> None:
    assert stabilization.worker_exit_code("passed") == 0
    assert stabilization.worker_exit_code("failed_gate") == 0
    assert stabilization.worker_exit_code("worker_exception") != 0
    assert stabilization.worker_exit_code("worker_output_invalid") != 0


def test_truth_and_detail_have_independent_states() -> None:
    result = stabilization.annotate_predicate_result(
        {
            "label": "test",
            "predicates": [
                {
                    "atom": "(On akita_black_bowl_1 plate_1)",
                    "truth_status": "passed",
                    "value": True,
                    "detail_status": "failed",
                }
            ],
        },
        10,
    )
    row = result["predicates"][0]
    assert row["truth_state"] == "passed"
    assert row["detail_state"] == "failed"
    assert row["truth_executed"] is True
    assert row["detail_executed"] is True


def test_expectation_gate_distinguishes_false_from_not_evaluated() -> None:
    evaluated = stabilization.annotate_predicate_result(
        {
            "predicates": [
                {
                    "atom": "(On akita_black_bowl_1 plate_1)",
                    "truth_status": "passed",
                    "value": False,
                    "detail_status": "not_attempted",
                }
            ]
        },
        10,
    )
    assert stabilization.predicate_expectation_gate(
        evaluated, "(On akita_black_bowl_1 plate_1)", expect=False
    )["state"] == "passed"
    assert stabilization.predicate_expectation_gate(
        {"predicates": []}, "(On akita_black_bowl_1 plate_1)", expect=False
    )["state"] == "not_evaluated"


def test_i1_report_is_json_and_keeps_zero_hold_contract() -> None:
    path = ROOT / "data/reports/axis_libero_30k/track3_reverse_init_stabilization/i1_semantics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "track3_reverse_init_stabilization_semantics_validated"
    assert report["action_contract"]["action"] == [0.0] * 7
    assert report["checkpoint_contract"]["official_equivalent_step"] == 5
    assert report["provenance_contract"]["hdf5_state_used_for_predicate_oracle"] is False


def test_i2_and_i4_are_fail_closed_without_gpu3_release() -> None:
    summary = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/track3_reverse_init_stabilization/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["status"] == "track3_reverse_init_stabilization_failed_named_scene_stability"
    assert summary["i2"]["fresh_process_runs"] == 12
    assert summary["i2"]["hard_gate"]["reverse_critical_source_step5_to20"] is True
    assert summary["i2"]["hard_gate"]["reverse_goal_step5_to20_false"] is True
    assert summary["i2"]["hard_gate"]["named_stability_thresholds_passed"] is False
    assert summary["i3"]["status"] == "not_run_due_to_i2_gate"
    assert summary["i3"]["task10_canary_runs"] == 0
    assert summary["resource_boundary"]["gpu3_touched"] is False
    assert summary["i3"]["settle"]["state"] == "not_evaluated"
    assert summary["i3"]["controlled_destruction"]["state"] == "not_evaluated"
