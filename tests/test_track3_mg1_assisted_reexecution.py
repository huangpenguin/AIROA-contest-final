from __future__ import annotations

import json

from tools.axis_pi05 import track3_operations_mg1_red_mug_assisted as assisted
from tools.axis_pi05 import track3_public_replay as replay


def test_mg1_schedule_is_fresh_and_read_only() -> None:
    metadata, arrays, observed = assisted.load_seed()
    blueprint = assisted.build_schedule_blueprint(metadata, arrays)
    assert metadata["frame_count"] == 314
    assert len(observed) == 6
    assert blueprint["execution"] == "fresh_continuous_mujoco_from_frozen_start_state"
    assert blueprint["source_frames_are_read_only"] is True
    assert blueprint["historical_frame_deletion"] is False
    assert blueprint["action_interpolation"] is False
    assert blueprint["budgets"]["total_target_max_frames"] == 260
    assert blueprint["budgets"]["hard_max_frames"] == 300
    assert [row["stage"] for row in blueprint["waypoints"]] == [
        "approach",
        "grasp_approach",
        "grasp_close_dwell",
        "lift",
        "transport",
        "place",
        "release_settle",
    ]
    assert blueprint["waypoints"][2]["gripper"] == 1.0
    assert blueprint["waypoints"][-1]["gripper"] == -1.0


def test_replay_resolves_official_task_id_without_task_index() -> None:
    task_index, bddl = replay.load_replay_profile({"task_id": "official_t3_red_mug"})
    assert task_index is None
    assert bddl.name == "take_the_red_mug_off_the_left_plate_and_put_it_on_the_table.bddl"


def test_mg1_closeout_preserves_failures_and_keeps_training_closed() -> None:
    summary = assisted.strict_json(assisted.NAMESPACE / "summary.json")
    assert summary["status"] == "mg1_red_mug_attempts_exhausted_family_failure"
    assert summary["attempt"]["status"] == "rejected_assisted_failure"
    assert summary["attempt"]["frame_count"] == 77
    assert summary["manual_fallback"]["status"] == "family_failure_rejected_over_budget"
    assert summary["manual_fallback"]["frame_count"] == 300
    assert summary["manual_fallback"]["telemetry"]["contact_offsets_equal_contact_n"] is True
    assert summary["fresh_replay"] == "not_evaluated"
    assert summary["training"] == "not_evaluated"
    closure = assisted.strict_json(assisted.NAMESPACE / "closure.json")
    assert closure["status"] == "passed_family_failure_closed"
    assert closure["manual_quota_consumed"] is True
