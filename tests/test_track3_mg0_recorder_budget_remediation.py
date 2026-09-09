from __future__ import annotations

from tools.axis_pi05 import track3_public_teleop as teleop


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
