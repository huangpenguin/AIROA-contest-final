from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05 import track3_public_teleop as teleop


ROOT = Path(__file__).resolve().parents[1]


def test_mg0_budget_and_frequency_contract() -> None:
    config = teleop.RecorderConfig()
    assert config.recording_step_limit() == 300
    assert config.budget_status(220)["stage"] == "early_warning"
    assert config.budget_status(260)["stage"] == "target_window"
    assert config.budget_status(290)["stage"] == "hard_warning"
    assert config.budget_status(300)["stage"] == "hard_stop"

    fast = teleop.RecorderConfig(operator_hz=4.0)
    assert fast.tick_ms() == 250
    assert fast.recording_step_limit() == 300
    assert fast.operator_tick_wait_seconds(10.25, now=10.0) == 0.25
    assert fast.operator_tick_wait_seconds(10.0, now=10.25) == 0.0


def test_mg0_keyup_and_pause_are_fail_safe() -> None:
    machine = teleop.ActionStateMachine()
    machine.event("connect")
    machine.event("keydown", "w")
    assert machine.next_action()[0] < 0
    machine.event("keyup", "w")
    assert machine.next_action()[0] == 0

    machine.event("keydown", "s")
    machine.event("keydown", "p")
    assert machine.snapshot()["paused"] is True
    assert machine.next_action()[0] == 0
    machine.event("keydown", "p")
    assert machine.snapshot()["paused"] is False


def test_mg0_metadata_separates_sim_operator_and_video_rates() -> None:
    config = teleop.RecorderConfig(operator_hz=4.0)
    profile = teleop.resolve_record_profile(task_id="official_t3_red_mug")
    metadata = teleop._metadata(profile, "test", config, {})
    assert metadata["sim_control_hz"] == 20.0
    assert metadata["operator_tick_hz"] == 4.0
    assert metadata["operator_tick_period_s"] == 0.25
    assert metadata["video_fps"] == 20.0
    assert metadata["timing"]["deadline_basis"] == "operator_tick_start_vs_scheduled_tick"
    assert metadata["budget"]["hard_stop_at_step"] == 300


def strict_json(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def test_mg0_execution_closeout_is_no_save_and_ready_for_user_calibration() -> None:
    namespace = ROOT / "data/reports/axis_libero_30k/track3_operations_mg0_recorder_remediation_20260909"
    summary = strict_json(namespace / "summary.json")
    assert summary["status"] == "passed_hard_stop_ready_for_user_calibration"
    assert summary["formal_recording_started"] is False
    assert summary["candidate_created"] is False
    assert summary["pending_created"] is False
    assert summary["replay"] == "not_evaluated"
    assert summary["historical_raw_modified"] is False
    for attempt in summary["canary_attempts"]:
        result = strict_json(namespace / attempt["result"])
        assert result["no_formal_episode_written"] is True
        assert result["save_event_seen"] is False
        assert result["candidate_created"] is False
        assert result["pending_created"] is False
        assert result["replay"] == "not_evaluated"
        assert result["deadline_miss_count"] == 0
