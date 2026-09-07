from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_public_teleop as teleop
from tools.axis_pi05 import track3_sr_closure_and_task10_user_teleop_gold_canary as ut
from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md"
UT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_sr_closure_and_task10_user_teleop_gold_canary"


def strict_json(path: Path) -> dict:
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(key)
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def test_ut0_ut3_ready_for_user_calibration_without_simulator() -> None:
    summary = strict_json(UT_ROOT / "summary.json")
    assert summary["status"] == "ready_for_user_calibration"
    attempt = ROOT / summary["attempt_root"]
    ut0 = strict_json(attempt / "ut0_input_lock.json")
    ut1 = strict_json(attempt / "ut1_summary.json")
    ut2 = strict_json(attempt / "ut2_finalizer_regression.json")
    ut3 = strict_json(attempt / "ut3_teleop_gold_readiness.json")
    assert ut0["status"] == "passed"
    assert ut0["complete_set"]["old_manifest_count"] == 15
    assert ut0["complete_set"]["observed_count"] == 18
    assert ut1["not_evaluated"] == ["SR9", "SR10", "SR11"]
    assert ut2["status"] == "passed"
    assert ut2["cases"]["late_producer_rejected"] is True
    assert ut2["cases"]["repeat_finalization_fail_close"] is True
    assert ut3["status"] == "ready_for_user_calibration"
    assert ut3["self_test"]["simulator_started"] is False
    assert ut3["start_gripper"]["label"] == "OPEN"
    assert (attempt / "closure.json").is_file()
    assert strict_json(attempt / "closure.json")["status"] == "passed"


def test_teleop_gold_npz_and_explicit_gripper() -> None:
    result = teleop.self_test()
    assert result["default_gripper_open"] is True
    assert result["gold_gates_fixture_passed"] is True
    assert teleop.RecorderConfig().default_gripper == -1.0
    machine = teleop.ActionStateMachine()
    assert machine.snapshot()["gripper_label"] == "OPEN"
    machine.event("keydown", "close")
    assert machine.snapshot()["gripper_label"] == "CLOSED"
    machine.event("disconnect")
    assert machine.snapshot()["paused"] is True


def test_handoff_waits_for_explicit_takeover() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_task10_user_teleop_ready_for_user_calibration"
    assert handoff["next_phase_only"] == "UT4_explicit_user_takeover_then_UT5_to_UT9"
    assert PLAN.is_file()
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md"
    auth = handoff["track3_sr_closure_and_task10_user_teleop_gold_canary_authorization"]
    assert auth["human_teleop_requires_explicit_user_online_takeover"] is True
    assert auth["authorization"]["unattended_teleop"] is False
    assert auth["authorization"]["training"] is False
    assert np.isfinite(1.0)
    assert ut.LOCKED["audit_input"].startswith("64f278c")
    assert sr.derive_not_evaluated(sr.SR_ROOT / "attempts/attempt_0004") == ["SR9", "SR10", "SR11"]
