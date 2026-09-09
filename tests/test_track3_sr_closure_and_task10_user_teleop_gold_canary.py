from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import track3_public_teleop as teleop
from tools.axis_pi05 import track3_sr_closure_and_task10_user_teleop_gold_canary as ut
from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_OPERATIONS_O1_O2_RED_MUG_NATIVE_MUJOCO_PLAN.md"
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
    assert teleop.RecorderConfig().hz == 20.0
    assert teleop.RecorderConfig().operator_hz == 2.0
    assert teleop.RecorderConfig().tick_ms() == 500
    assert teleop.RecorderConfig().preview_refresh_ms <= 50
    assert teleop.RecorderConfig().translation_step >= 0.30
    assert teleop.RecorderConfig().eval_step_budget == 300
    machine = teleop.ActionStateMachine()
    assert machine.snapshot()["gripper_label"] == "OPEN"
    machine.event("keydown", "close")
    assert machine.snapshot()["gripper_label"] == "CLOSED"
    machine.event("disconnect")
    assert machine.snapshot()["paused"] is True
    red = teleop.resolve_record_profile(task_id="official_t3_red_mug")
    assert red["instruction"].startswith("take the red mug")
    assert not red["instruction"].lower().startswith("reverse")
    assert red["language_gold_ok"] is True
    two = teleop.resolve_record_profile(task_id="official_t3_two_mugs")
    assert two["recordable"] is True
    assert two["training_holdout"] is False
    assert two["instruction"].startswith("take the white mug")
    assert teleop.resolve_record_profile(task_id="official_t3_yellow_book")["instruction"].startswith("take the yellow book")
    book = teleop.resolve_record_profile(task_id="official_t3_yellow_book")
    assert book["init_height_sites"][0] == ("yellow_book_2", "wooden_two_layer_shelf_1_top_side")
    assert teleop.resolve_record_profile(task_id="official_t3_alphabet_soup")["instruction"].startswith("take the alphabet soup")
    assert all(bool(row["recordable"]) and not bool(row["training_holdout"]) for row in teleop.OFFICIAL_T3_TASKS.values())
    preview = teleop.flip_front_preview(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    assert int(preview[0, 0, 0]) == 9
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    marked = teleop.overlay_goal_box(blank, np.array([[4, 4], [28, 4], [28, 28], [4, 28]], dtype=np.int32))
    assert marked[16, 16].sum() > 0
    assert "red_coffee_mug_init_region" in teleop.goal_site_name_candidates(red)[-1]


def test_handoff_waits_for_explicit_takeover() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_operations_o1_2_user_recorded_over_budget_hard_stop"
    assert handoff["next_phase_only"] == "new_reviewed_plan_required"
    assert PLAN.is_file()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md").is_file()
    assert handoff["next_plan"] is None
    o0r = handoff["track3_operations_o0r_remediation_20260908"]
    assert o0r["audit_class"] == "approved"
    assert o0r["historical_metadata_mutated"] is False
    auth = handoff["track3_operations_o1_o2_authorization"]
    assert auth["status"] == "consumed_hard_stop"
    assert auth["preferred_seed"].endswith("1788838632")
    assert auth["o1"]["fresh_replay_success_required"] == "3/3"
    assert auth["o1"]["user_online_takeover_required"] is True
    assert auth["o2"]["requires_o1_accepted"] is True
    assert auth["authorization"]["unattended_teleop"] is False
    assert auth["authorization"]["gpu0"] is False
    assert np.isfinite(1.0)
    assert ut.LOCKED["audit_input"].startswith("64f278c")
    assert sr.derive_not_evaluated(sr.SR_ROOT / "attempts/attempt_0004") == ["SR9", "SR10", "SR11"]

def test_o1_0_red_mug_retiming_is_derived_and_hard_stopped() -> None:
    handoff = strict_json(HANDOFF)
    execution = handoff["track3_operations_o1_0_execution"]
    assert execution["status"] == "ready_for_native_vnc_no_save_canary"
    assert execution["preferred_seed"].endswith("1788838632")
    assert execution["reference_seed"].endswith("1788838293")
    assert execution["keyposes"] == {
        "approach": 145,
        "grasp": [146, 156],
        "lift": 157,
        "transport": 278,
        "place": 365,
        "release": 373,
    }
    assert execution["historical_raw_modified"] is False
    assert execution["replay"] == "not_evaluated"
    lock = strict_json(ROOT / execution["canonical_attempt_lock"])
    blueprint = strict_json(ROOT / execution["retiming_blueprint"])
    assert lock["lock_id"] == "canonical_red_mug_attempt_0001"
    assert lock["unique_canonical_attempt_count"] == 1
    assert lock["hard_stop_after_o1_0"] is True
    assert blueprint["budgets"]["planned_goal_predicate_frame"] == 220
    assert blueprint["budgets"]["total_target_max_frames"] == 260
    assert blueprint["budgets"]["hard_max_frames"] == 300


def test_o1_1_native_vnc_no_save_canary_passed() -> None:
    handoff = strict_json(HANDOFF)
    execution = handoff["track3_operations_o1_1_execution"]
    assert execution["status"] == "passed_native_vnc_no_save_canary"
    assert execution["formal_recording"] == "not_started"
    assert execution["replay"] == "not_evaluated"
    assert execution["candidate_created"] is False
    correction = strict_json(ROOT / execution["derived_correction_summary"])
    assert correction["reset"]["source_true"] is True
    assert correction["reset"]["goal_false"] is True
    assert correction["save"]["save_event_seen"] is False
    assert correction["writer"]["contact_offsets_equal_contact_n"] is True


def test_o1_2_user_recording_over_budget_is_not_gold() -> None:
    handoff = strict_json(HANDOFF)
    execution = handoff["track3_operations_o1_2_execution"]
    assert execution["status"] == "pending_over_budget_hard_stop"
    assert execution["frame_count"] == 314
    assert execution["step_budget"] == 300
    assert execution["gold_eligible"] is False
    assert execution["plan_gate"] == "failed_over_budget"
    assert execution["replay"] == "not_evaluated"
    assert execution["additional_formal_recording_authorized"] is False
    derived = strict_json(ROOT / execution["derived_eligibility"])
    assert derived["physics"]["end_success"] is True
    assert derived["over_budget_by"] == 14
    assert derived["historical_metadata_unchanged"] is True


def test_dynamic_contact_table_preserves_every_contact_and_offsets() -> None:
    contacts = [
        {
            "body1_name": "red_coffee_mug_1",
            "body2_name": f"contact_body_{index}",
            "geom1_name": "mug_geom",
            "geom2_name": f"contact_geom_{index}",
            "distance_m": -0.001,
        }
        for index in range(54)
    ]
    frames = teleop.empty_gold_buffers()
    action = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    teleop.append_gold_frame(
        frames,
        action=action,
        state=np.zeros(84, dtype=np.float64),
        reward=0,
        done=False,
        success=False,
        front=np.zeros((2, 2, 3), dtype=np.uint8),
        wrist=np.zeros((2, 2, 3), dtype=np.uint8),
        timestamp=0,
        deadline_miss=False,
        contact_matrix={"complete": True, "contacts": contacts},
        source=True,
        goal=False,
        eef_pos=[0, 0, 0],
        gripper=1,
        check_success=False,
        contact_roles=teleop.OFFICIAL_T3_TASKS["official_t3_red_mug"]["roles"],
    )
    arrays = teleop.typed_npz_arrays(frames)
    assert arrays["contact_n"].tolist() == [54]
    assert arrays["contact_offsets"].tolist() == [0, 54]
    assert arrays["contact_body1"].shape == (54,)
    assert arrays["contact_body2"].shape == (54,)
    assert arrays["contact_role"].shape == (54,)


def test_acquisition_gate_uses_first_dwell_and_rejects_late_goal_contact() -> None:
    samples = []
    for index in range(4):
        samples.append({
            "phase": "precontact_xy",
            "source_predicate": True,
            "demonstrator_contact": False,
            "contact_telemetry_complete": True,
            "contact_raw_complete": True,
        })
    for index in range(4):
        samples.append({
            "phase": "grasp_dwell",
            "source_predicate": True,
            "demonstrator_contact": True,
            "contact_telemetry_complete": True,
            "contact_raw_complete": True,
        })
    for index in range(4):
        samples.append({
            "phase": "place",
            "source_predicate": False,
            "demonstrator_contact": False,
            "contact_telemetry_complete": True,
            "contact_raw_complete": True,
            "required_contact": True,
        })
    result = teleop.acquisition_contact_gate_v4(samples, stable_ticks=4)
    assert result["state"] == "passed"
    assert result["window_start"] == 4
    assert result["barrier_index"] == 7

    goal_only = [
        {
            "phase": "place",
            "source_predicate": True,
            "demonstrator_contact": False,
            "contact_telemetry_complete": True,
            "contact_raw_complete": True,
            "required_contact": True,
        }
        for _ in range(4)
    ]
    rejected = teleop.acquisition_contact_gate_v4(goal_only, stable_ticks=4)
    assert rejected["state"] == "failed"
    assert rejected["barrier_index"] is None


def test_atomic_episode_move_and_clone_keep_relative_final_paths(tmp_path, monkeypatch) -> None:
    frames = teleop.empty_gold_buffers()
    teleop.append_gold_frame(
        frames,
        action=np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
        state=np.zeros(84, dtype=np.float64),
        reward=0,
        done=True,
        success=False,
        front=np.zeros((2, 2, 3), dtype=np.uint8),
        wrist=np.zeros((2, 2, 3), dtype=np.uint8),
        timestamp=0,
        deadline_miss=False,
        contact_matrix={"complete": True, "contacts": []},
        source=True,
        goal=False,
        eef_pos=[0, 0, 0],
        gripper=-1,
        check_success=False,
        contact_roles=teleop.OFFICIAL_T3_TASKS["official_t3_red_mug"]["roles"],
    )

    def no_video(path, frames, fps):
        return {"status": "no_frames", "path": teleop.validate_episode_relative_path(path.name)}

    monkeypatch.setattr(teleop, "_save_video", no_video)
    metadata = {"status": "recording"}
    target = teleop._atomic_episode(
        tmp_path / "pending",
        "episode",
        lambda directory: teleop._write_episode(
            directory,
            metadata,
            frames,
            np.zeros(84, dtype=np.float64),
            np.zeros(84, dtype=np.float64),
            teleop.RecorderConfig(),
        ),
    )
    saved = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == "track3_reverse_pilot_v4"
    assert saved["trajectory"]["path"] == "trajectory.npz"
    assert saved["videos"]["front"]["path"] == "front.mp4"
    assert saved["videos"]["wrist"]["path"] == "wrist.mp4"
    assert not list(tmp_path.rglob("*.partial"))

    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    shutil.copytree(target, clone_root / "episode")
    cloned = json.loads((clone_root / "episode" / "metadata.json").read_text(encoding="utf-8"))
    for value in (
        cloned["trajectory"]["path"],
        cloned["videos"]["front"]["path"],
        cloned["videos"]["wrist"]["path"],
    ):
        assert teleop.validate_episode_relative_path(value) == value
    assert (clone_root / "episode" / cloned["trajectory"]["path"]).is_file()


def test_episode_metadata_rejects_absolute_parent_and_partial_paths() -> None:
    for value in ("/tmp/front.mp4", "../front.mp4", "front.mp4.partial"):
        with pytest.raises(ValueError):
            teleop.validate_episode_relative_path(value)
