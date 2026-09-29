from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lxml import etree

from tools.axis_pi05 import track3_public_teleop as teleop


ROOT = Path(__file__).resolve().parents[1]


def test_mg0_budget_and_frequency_contract() -> None:
    config = teleop.RecorderConfig()
    assert config.recording_step_limit() == 400
    assert config.operator_hz == 8.0
    assert config.budget_status(300)["stage"] == "early_warning"
    assert config.budget_status(350)["stage"] == "target_window"
    assert config.budget_status(390)["stage"] == "hard_warning"
    assert config.budget_status(400)["stage"] == "gold_limit"
    assert config.budget_status(401)["stage"] == "postprocess_required"

    fast = teleop.RecorderConfig(operator_hz=4.0)
    assert fast.tick_ms() == 250
    assert fast.recording_step_limit() == 400
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
    assert metadata["budget"]["direct_gold_limit"] == 400
    assert metadata["budget"]["hard_stop_at_step"] is None


def strict_json(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def test_reset_resyncs_osc_goal_to_restored_eef() -> None:
    events: list[object] = []

    class DummyController:
        def update(self, force: bool = False) -> None:
            events.append(("update", force))

        def reset_goal(self) -> None:
            events.append("reset_goal")

    class DummySim:
        def forward(self) -> None:
            events.append("forward")

    class DummyInner:
        robots = [type("R", (), {"controller": DummyController()})()]

    class DummyEnv:
        sim = DummySim()
        env = DummyInner()

        def set_init_state(self, state):
            events.append(("set_init_state", tuple(state)))
            return "obs"

    assert teleop.restore_frozen_start(DummyEnv(), [1.0, 2.0]) == "obs"
    assert events == [("set_init_state", (1.0, 2.0)), "forward", ("update", True), "reset_goal"]
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


def test_table_goal_marker_is_visual_only_and_skips_camera_overlay() -> None:
    xml = """
    <mujoco>
      <worldbody>
        <body name="study_table">
          <site name="study_table_yellow_book_left_init_region" pos="-0.05 -0.25 0.90" size="0.075 0.075 0.001" type="box"/>
        </body>
      </worldbody>
    </mujoco>
    """
    injected = teleop.inject_table_goal_marker_xml(xml, ["study_table_yellow_book_left_init_region"])
    assert injected is not None
    root = etree.fromstring(injected.encode("utf-8"))
    site = next(el for el in root.iter("site") if el.get("name") == "study_table_yellow_book_left_init_region")
    plate = next(el for el in root.iter("geom") if el.get("name") == teleop.GOAL_MARKER_GEOM)
    pin = next(el for el in root.iter("geom") if el.get("name") == teleop.GOAL_MARKER_PIN)
    assert site.get("size") == "0.075 0.075 0.001"
    assert plate.get("contype") == "0" and plate.get("conaffinity") == "0" and plate.get("mass") == "0"
    assert plate.get("group") == "1" and pin.get("group") == "1"
    assert plate.get("density") == "0" and pin.get("density") == "0"
    assert pin.get("contype") == "0" and pin.get("type") == "cylinder"
    sx, sy, _sz = (float(value) for value in plate.get("size").split())
    assert abs(sx - 0.075) < 1e-9 and abs(sy - 0.075) < 1e-9
    assert teleop.inject_table_goal_marker_xml(xml, ["missing_site"]) is None

    class Model:
        geom_rgba = np.array([[1.0, 0.85, 0.0, 1.0], [1.0, 0.85, 0.0, 1.0]])

        def geom_name2id(self, name: str) -> int:
            if name == teleop.GOAL_MARKER_GEOM:
                return 0
            if name == teleop.GOAL_MARKER_PIN:
                return 1
            raise ValueError(name)

    class Env:
        sim = type("Sim", (), {"model": Model()})()

    env = Env()
    assert teleop.goal_marker_installed(env)
    teleop.set_goal_marker_visible(env, False)
    assert float(env.sim.model.geom_rgba[0][3]) == 0.0
    assert float(env.sim.model.geom_rgba[1][3]) == 0.0
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    preview = teleop.make_front_preview(blank, env=env, profile=teleop.resolve_record_profile(task_id="official_t3_yellow_book"))
    assert int(preview.sum()) == 0
