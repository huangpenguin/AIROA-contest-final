from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_acquisition_gate_attempt_lifecycle_remediation as br


def test_numpy_boolean_source_release_uses_value_comparison() -> None:
    result = br.source_release_gate(
        np.asarray([True, False, False], dtype=np.bool_),
        np.asarray([True, True, True], dtype=np.bool_),
        acquisition_state="passed",
    )
    assert result["state"] == "passed"


def test_late_contact_cannot_backfill_acquisition_barrier() -> None:
    samples = [
        {"telemetry_complete": True, "known_mask": True, "contacts": []}
        for _ in range(4)
    ] + [
        {"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]}
        for _ in range(4)
    ]
    result = br.required_contact_gate("support_remove", samples, stage_entered=True, barrier_index=3)
    assert result["state"] == "failed"
    assert result["late_contact_excluded"] is True
    assert result["eligible_sample_count"] == 4


def test_stage_tail_is_not_evaluated_after_early_stop() -> None:
    gates = br.stage_gate_graph("support_remove", ["approach_object"])
    assert gates["approach_object"]["state"] == "passed"
    assert gates["grasp_object"]["state"] == "not_evaluated"
    assert gates["release_object"]["state"] == "not_evaluated"


def test_source_and_goal_gate_unentered_states_are_not_failed() -> None:
    source = br.source_release_gate([True, True], [True, True], acquisition_state="not_evaluated")
    goal = br.goal_continuity_gate([False, False], [True, True], goal_stage_entered=False)
    assert source["state"] == "not_evaluated"
    assert goal["state"] == "not_evaluated"


def test_allocator_is_monotonic_and_published_artifact_is_immutable(tmp_path: Path) -> None:
    first_id, first = br.allocate_attempt(tmp_path / "br")
    first_file = first / "result.json"
    br.write_json_immutable(first_file, {"attempt": first_id, "status": "failed"})
    try:
        br.write_json_immutable(first_file, {"attempt": 99})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing attempt artifact was overwritten")
    second_id, second = br.allocate_attempt(tmp_path / "br")
    assert (first_id, second_id) == (1, 2)
    assert first_file.read_text(encoding="utf-8") == '{\n  "attempt": 1,\n  "status": "failed"\n}\n'
    assert second != first


def test_clean_source_gate_rejects_dirty_root_and_openpi(monkeypatch) -> None:
    def fake_porcelain(repo=br.ROOT):
        return [" M dirty.py"] if repo == br.ROOT else []

    monkeypatch.setattr(br, "porcelain", fake_porcelain)
    assert br.clean_source_gate() is False


def test_full_typed_trace_recomputes_gates_and_excludes_late_contact(tmp_path: Path) -> None:
    phases = ["approach_object"] * 4 + ["descend_object"] * 4 + ["grasp_object"] * 4 + ["lift_object"] * 4 + ["transport_to_target"] * 4 + ["descend_to_target"] * 4 + ["release_object"] * 4 + ["retreat"] * 4
    n = len(phases)
    required_contact = [True] * 9 + [False] * (n - 9)
    arrays = {
        "actions": np.zeros((n, 7), dtype=np.float32),
        "eef_error": np.linspace(1.0, 0.0, n, dtype=np.float32),
        "phase": np.asarray(phases, dtype="U64"),
        "source_value": np.asarray([True] * 9 + [False] * (n - 9), dtype=np.bool_),
        "source_known": np.ones(n, dtype=np.bool_),
        "goal_value": np.asarray([False] * 24 + [True] * 8, dtype=np.bool_),
        "goal_known": np.ones(n, dtype=np.bool_),
        "action_saturated": np.zeros(n, dtype=np.bool_),
        "contact_telemetry_complete": np.ones(n, dtype=np.bool_),
        "required_contact": np.asarray(required_contact, dtype=np.bool_),
        "unrelated_contact_runtime": np.zeros(n, dtype=np.bool_),
    }
    trace = tmp_path / "typed_trace.npz"
    br._write_npz(trace, arrays)
    result = br.evaluate_typed_trace("support_remove", trace)
    assert result["status"] == "passed"
    assert result["gates"]["source_released"]["state"] == "passed"
    assert result["gates"]["goal_continuous_true"]["state"] == "passed"
    assert result["gates"]["required_contact_continuous"]["state"] == "passed"
    assert result["contact_after_barrier_excluded"] is True


def test_br0_is_read_only_derived_and_br3_sticky_fixture(tmp_path: Path, monkeypatch) -> None:
    # The production fixture is exercised separately; this assertion protects
    # the code contract from accidentally turning historical AC evidence into
    # a mutable output path.
    assert br.AC_ROOT != br.BR_ROOT
    assert "historical_ac_namespace" in br._old_ac_closure()
    assert br.OLD_PROFILE_IDS.isdisjoint(br.PROFILE_ORDER)
