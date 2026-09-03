from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.axis_pi05 import track3_dynamic_gate_reconciliation as gate


ROOT = Path(__file__).resolve().parents[1]


def test_d0_lock_rechecks_old_evidence_and_hdf5_metadata() -> None:
    lock = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation/d0_release_lock.json"
        ).read_text(encoding="utf-8")
    )
    assert lock["status"] == "track3_dynamic_gate_reconciliation_input_locked"
    assert gate.validate_d0_lock(lock)["old_evidence_immutable"] is True
    assert lock["hard_gate"]["counts_zero"] is True
    assert len(lock["hdf5_inputs"]) == 3
    assert all(row["states0_equals_init_state"] for row in lock["hdf5_inputs"].values())
    assert all(row["states_qpos_offset"] == 1 for row in lock["hdf5_inputs"].values())


def test_model_mismatch_rejects_hdf5_predicate_oracle() -> None:
    result = gate.validate_model_for_predicate_oracle("recorded", "fresh_randomized")
    assert result["status"] == "incompatible_for_predicate_oracle"
    assert result["hdf5_state_used_for_predicate_oracle"] is False
    assert result["predicate_oracle"] == "native_reverse_bddl_only"


def test_reverse_initial_is_evaluated_atom_by_atom_without_forward_proxy() -> None:
    class FakeModel:
        body_names = ()

    class FakeSim:
        model = FakeModel()

    class FakeEnv:
        sim = FakeSim()
        object_states_dict = {}
        object_sites_dict = {}

        @staticmethod
        def _eval_predicate(state: tuple[str, ...]) -> bool:
            return state[0] == "on"

    result = gate.evaluate_predicates(
        SimpleNamespace(env=FakeEnv(), sim=FakeSim()),
        ["(On object_a region_a)", "(Close drawer_region)"],
        label="reverse_initial",
    )
    assert result["count"] == 2
    assert [row["value"] for row in result["predicates"]] == [True, False]
    assert "check_success" not in inspect.getsource(gate.evaluate_predicates)


def test_truth_is_preserved_when_detail_telemetry_fails() -> None:
    class FakeModel:
        body_names = ()

    class FakeSim:
        model = FakeModel()

    class FakeEnv:
        sim = FakeSim()
        object_states_dict = {}
        object_sites_dict = {}

        @staticmethod
        def _eval_predicate(state: tuple[str, ...]) -> bool:
            return True

    result = gate.evaluate_predicates(
        SimpleNamespace(env=FakeEnv(), sim=FakeSim()),
        ["(On object_a region_a)"],
        label="truth_detail_split",
    )
    row = result["predicates"][0]
    assert row["truth_status"] == "passed"
    assert row["value"] is True
    assert row["detail_status"] == "failed"
    assert result["all_true"] is True
    assert result["detail_complete"] is False


def test_worker_exception_has_nonzero_process_exit_contract() -> None:
    assert gate.worker_exit_code("passed") == 0
    assert gate.worker_exit_code("failed_gate") == 0
    assert gate.worker_exit_code("worker_exception") != 0
    assert gate.worker_exit_code("worker_output_invalid") != 0


def test_site_predicate_details_use_site_state_wrapper_not_raw_target_zone() -> None:
    class FakeModel:
        body_names = ("object_a",)

    class FakeSim:
        model = FakeModel()
        data = SimpleNamespace(
            body_xpos=np.zeros((1, 3)),
            body_xquat=np.array([[1.0, 0.0, 0.0, 0.0]]),
        )

    class SiteState:
        object_state_type = "site"

        @staticmethod
        def get_geom_state() -> dict[str, np.ndarray]:
            return {"pos": np.zeros(3), "quat": np.array([1.0, 0.0, 0.0, 0.0])}

    class FakeEnv:
        sim = FakeSim()
        object_states_dict = {"region_a": SiteState()}
        object_sites_dict = {"region_a": object()}

        @staticmethod
        def _eval_predicate(state: tuple[str, ...]) -> bool:
            return True

    result = gate.evaluate_predicates(
        SimpleNamespace(env=FakeEnv(), sim=FakeSim()),
        ["(On object_a region_a)"],
        label="site_detail",
    )
    assert result["all_true"] is True
    geometry = result["predicates"][0]["detail"]["geometry"]
    assert any(item["kind"] == "site" and item["name"] == "region_a" for item in geometry)


def test_controller_reset_order_is_native_forward_update_then_goal() -> None:
    events: list[str] = []

    class Sim:
        def forward(self) -> None:
            events.append("sim.forward")

    class Controller:
        def update(self, force: bool = False) -> None:
            assert force is True
            events.append("controller.update")

        def reset_goal(self) -> None:
            events.append("controller.reset_goal")

    assert gate.controller_reset_sequence(Controller(), Sim()) == [
        "native_reset",
        "sim.forward",
        "controller.update",
        "controller.reset_goal",
    ]
    assert events == ["sim.forward", "controller.update", "controller.reset_goal"]


def test_gripper_hold_is_zero_command_not_forced_open_command() -> None:
    action = gate.gripper_hold_action()
    assert action.shape == (7,)
    assert action[-1] == 0.0
    assert gate.hold_action(-1.0)[-1] == -1.0
    assert gate.hold_action(-1.0)[-1] != action[-1]


def test_named_schema_quaternion_and_thresholds_fail_closed() -> None:
    assert gate.validate_named_schema("eef", np.zeros(3), (3,)).shape == (3,)
    assert gate.quat_geodesic_rad([1, 0, 0, 0], [-1, 0, 0, 0]) == pytest.approx(0.0)
    assert gate.THRESHOLDS["eef_translation_m"] != gate.THRESHOLDS["eef_rotation_rad"]
    with pytest.raises(ValueError):
        gate.validate_named_schema("bad_shape", [1, 2], (3,))
    with pytest.raises(ValueError):
        gate.validate_named_schema("bad_nan", [0, float("nan"), 0], (3,))
    with pytest.raises(ValueError):
        gate.normalize_quat([0, 0, 0, 0])


def test_warmup_and_measurement_windows_are_disjoint() -> None:
    windows = gate.measurement_windows(list(range(40)))
    assert windows["warmup"] == list(range(20))
    assert windows["measurement"] == list(range(20, 40))
    assert windows["windows_disjoint"] is True
    with pytest.raises(ValueError):
        gate.measurement_windows(list(range(39)))


def test_old_dynamic_report_is_still_immutable_and_not_used_as_new_gate() -> None:
    old = (
        ROOT
        / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_dynamic_gates.json"
    )
    lock = json.loads(
        (
            ROOT
            / "data/reports/axis_libero_30k/track3_dynamic_gate_reconciliation/d0_release_lock.json"
        ).read_text(encoding="utf-8")
    )
    assert old.exists()
    assert lock["immutable_old_evidence"][str(old.resolve())]["sha256"] == gate.sha256_file(old)
    assert "g1_s3_dynamic_gates.json" not in inspect.getsource(gate.dynamic_worker)
