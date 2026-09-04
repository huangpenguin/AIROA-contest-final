from __future__ import annotations

import numpy as np
import pytest

from tools.axis_pi05 import track3_composed_reverse_start_stabilization as module


def _metadata() -> dict:
    return {
        "nq": 9,
        "nv": 8,
        "na": 2,
        "state_length": 20,
        "joint_names": ["free_object", "robot0_joint1", "drawer"],
        "jnt_type": [0, 3, 3],
        "jnt_qposadr": [0, 7, 8],
        "jnt_dofadr": [0, 6, 7],
        "joint_ownership": {"free_object": "operator_scene", "robot0_joint1": "robot_arm", "drawer": "background_scene"},
        "joint_source": {"free_object": "demo_terminal", "robot0_joint1": "demo_init", "drawer": "demo_terminal"},
    }


def test_free_joint_and_exact_joint_coverage() -> None:
    layout = module.decode_joint_layout(_metadata())
    assert layout["joints"][0]["qpos_range"] == [0, 7]
    assert layout["joints"][0]["qvel_range"] == [0, 6]
    assert sorted(layout["qpos_indices"]) == list(range(9))
    assert sorted(layout["qvel_indices"]) == list(range(8))


@pytest.mark.parametrize(
    "field,value",
    [("jnt_qposadr", [0, 7, 7]), ("jnt_dofadr", [0, 6, 6]), ("joint_ownership", {"free_object": "unknown", "robot0_joint1": "robot_arm", "drawer": "background_scene"})],
)
def test_layout_fails_closed_on_overlap_or_unknown(field: str, value: object) -> None:
    metadata = _metadata()
    metadata[field] = value
    with pytest.raises(ValueError):
        module.decode_joint_layout(metadata)


def test_composition_has_index_provenance_and_zero_qvel() -> None:
    metadata = _metadata()
    layout = module.decode_joint_layout(metadata)
    init = np.arange(20, dtype=np.float64)
    terminal = 1000.0 + np.arange(20, dtype=np.float64)
    composed, provenance = module.compose_flat_state(metadata, layout, init, terminal)
    # State layout is: time, qpos[9], qvel[8], actuator[2].
    assert composed[0] == init[0]
    assert np.array_equal(composed[1:8], terminal[1:8])
    assert composed[8] == init[8]  # robot0_joint1 qpos comes from demo init
    assert composed[9] == terminal[9]  # drawer qpos comes from demo terminal
    assert np.array_equal(composed[10:18], np.zeros(8))
    assert np.array_equal(composed[18:], init[18:])
    assert provenance["qvel_exact_max_abs"] == 0.0
    assert provenance["state_indices_source_bitmap"][10] == "zero"
    assert provenance["state_indices_source_bitmap"][8] == "demo_init"
    assert provenance["state_indices_source_bitmap"][9] == "demo_terminal"


def test_neutral_hold_is_float32_zero_seven_d() -> None:
    action = module.neutral_hold()
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert np.array_equal(action, np.zeros(7, dtype=np.float32))


def _matrix(*pairs: tuple[str, str]) -> dict:
    return {"complete": True, "contacts": [{"body1_name": left, "body2_name": right, "geom1_name": left + "_g", "geom2_name": right + "_g", "distance_m": -0.001} for left, right in pairs]}


def test_contact_v3_robot_contact_is_demonstrator_not_background() -> None:
    detail = module.contact_v3(20, _matrix(("orange_juice_1", "robot0_right_finger")))
    assert detail["state"] == "passed"
    assert detail["counts"]["demonstrator_contact"] == 1
    assert detail["counts"]["unrelated_background_contact"] == 0


def test_contact_v3_intended_receiver_and_unrelated_background() -> None:
    allowed = module.contact_v3(10, _matrix(("akita_black_bowl_1", "plate_1")))
    assert allowed["state"] == "passed"
    assert allowed["counts"]["intended_receiver_contact"] == 1
    forbidden = module.contact_v3(10, _matrix(("akita_black_bowl_1", "main_table")))
    assert forbidden["state"] == "failed"
    assert forbidden["reason"] == "unrelated_background_contact"


def test_contact_window_v3_requires_support_only_for_task10() -> None:
    samples = [{"contact_matrix": _matrix(("akita_black_bowl_1", "plate_1"))}, {"contact_matrix": _matrix(("akita_black_bowl_1", "plate_1"))}]
    result = module.contact_window_gate(10, samples)
    assert result["state"] == "passed"
    assert module.contact_window_gate(10, [{"contact_matrix": _matrix()}])["state"] == "failed"


def test_background_contact_delta_records_new_pairs() -> None:
    result = module.background_contact_delta([_matrix(("table", "cup")), _matrix(("table", "cup"), ("table", "bowl"))])
    assert result["state"] == "failed"
    assert result["new_pairs"][0]["body_pair"] == ["bowl", "table"]


def test_gate_pass_has_no_failure_reason_and_worker_contract() -> None:
    passed = module.gate(executed=True, value=True)
    assert passed == {"executed": True, "state": "passed", "required": True}
    failed = module.gate(executed=True, value=False, reason="x")
    assert failed["reason"] == "x"
    assert module.worker_exit_code("passed") == 0
    assert module.worker_exit_code("failed_gate") == 0
    assert module.worker_exit_code("worker_exception") != 0


def test_m0_lock_and_m1_contract_are_closed() -> None:
    assert (module.OUT_ROOT / "m0_release_lock.json").is_file()
    assert (module.OUT_ROOT / "m1_validator_contract.json").is_file()
    assert module.OUT_ROOT.joinpath("m0_release_lock.json").read_text(encoding="utf-8")
