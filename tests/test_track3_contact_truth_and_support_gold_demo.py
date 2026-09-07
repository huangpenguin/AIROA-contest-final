from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_contact_truth_and_support_gold_demo as sg


def test_contact_truth_does_not_infer_contact_from_telemetry() -> None:
    result = sg.contact_truth_v3(10, {"complete": True, "contacts": [], "required_contact": True, "unrelated_contact_runtime": False})
    assert result["contact_telemetry_complete"] is True
    assert result["required_contact"] is False
    assert result["unrelated_contact_runtime"] is False
    assert result["state"] == "failed"


def test_contact_truth_uses_actual_pair_roles() -> None:
    result = sg.contact_truth_v3(10, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "robot0_right_hand"}]})
    assert result["required_contact"] is True
    assert result["counts"]["demonstrator_contact"] == 1
    assert result["counts"]["intended_receiver_contact"] == 0


def test_unrelated_contact_and_unknown_schema_are_distinct() -> None:
    unrelated = sg.contact_truth_v3(10, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "main_table"}]})
    unknown = sg.contact_truth_v3(10, {"complete": False, "contacts": []})
    assert unrelated["unrelated_contact_runtime"] is True
    assert unrelated["state"] == "failed"
    assert unknown["state"] == "infrastructure_failure"


def test_acquisition_barrier_excludes_late_contact() -> None:
    samples = [{"contact_matrix": {"complete": True, "contacts": []}}, {"contact_matrix": {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "robot0_right_hand"}]}}]
    result = sg.required_contact_gate(samples, 0)
    assert result["state"] == "failed"
    assert result["late_contact_excluded"] is True
    assert result["required_contact_values"] == [False]


def test_numpy_boolean_and_normalized_action_contract() -> None:
    assert sg.predicate_truth(np.bool_(True)) is True
    assert sg.predicate_truth(np.bool_(False)) is False
    result = sg.normalized_action_contract([0.5, 0.0, 0.0, 0.2, 0.0, 0.0, -1.0])
    assert result["state"] == "passed"
    assert result["normalized_action_abs_max"] == 1.0
    assert result["double_normalization"] is False


def test_waypoint_contract_has_real_axis_split_and_orientation_stage() -> None:
    contract = sg.waypoint_contract()
    assert contract["stage_names"] == list(sg.STAGE_NAMES)
    assert contract["stages"][0]["translation_axes"] == [0, 1]
    assert contract["stages"][1]["translation_axes"] == [2]
    assert contract["stages"][2]["rotation_enabled"] is True
    assert contract["keypose_is_not_trajectory"] is True


def test_manifest_detects_post_seal_new_nested_file(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "heartbeat.json").write_text("{}\n", encoding="utf-8")
    manifest = sg.make_manifest(tmp_path)
    (tmp_path / "nested" / "late.log").write_text("late\n", encoding="utf-8")
    check = sg.verify_complete_set(tmp_path, manifest)
    assert check["exact"] is False
    assert "nested/late.log" in check["extra"]


def test_finalizer_seals_complete_set(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "heartbeat.json").write_text("{}\n", encoding="utf-8")
    result = sg.finalize_attempt(tmp_path)
    assert result["status"] == "passed"
    assert sg.strict_json(tmp_path / "closure.json")["complete_set"]["exact"] is True


def test_identity_gate_rejects_dirty_checkout_even_when_values_match() -> None:
    result = sg.identity_gate(expected_commit="a", expected_tool_sha256="b", expected_openpi="c", expected_command=["worker"], observed_commit="a", observed_tool_sha256="b", observed_openpi="c", observed_command=["worker"], dirty=True)
    assert result["state"] == "failed"
    assert "root_clean_now" in result["errors"]


def test_recorder_command_uses_task_index_and_record() -> None:
    command = sg.recorder_command(10)
    assert "--record" in command
    assert "--task-index" in command
    assert "--task" not in command
