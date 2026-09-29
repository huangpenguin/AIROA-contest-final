from __future__ import annotations

import json

import numpy as np
import pytest

from tools.axis_pi05 import run_track3_q5_step10_frozen_eval as runner


def test_strict_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"a","status":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.strict_json_load(path)


def test_step_validation_rejects_non_frozen_step():
    with pytest.raises(ValueError, match="only accepts step 0 or 10"):
        runner._validate_checkpoint(1)


def test_server_environment_is_track3_specific():
    env = runner.server_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert env["LIBERO_CONTROL_FREQ"] == "20"
    assert env["AXIS_PI05_LIBERO_ASSET_BRIDGE"] == "1"
    assert env["AXIS_PI05_NORM_STATS"].endswith("data/manifests_track3/four_gold_80_20_v1/norm_stats.json")


class FakeClient:
    def __init__(self):
        self.resets = []
        self.actions = [np.arange(7, dtype=np.float32) for _ in range(20)]

    def reset(self, **kwargs):
        self.resets.append(kwargs)

    def get_action(self, _observation):
        return self.actions.pop(0)


def test_execute8_discards_two_actions_without_reseeding():
    client = FakeClient()
    policy = runner.Execute8Policy(client)
    policy.reset("instruction", seed=2000)
    for _ in range(9):
        policy.get_action({})
    assert client.resets[0]["reset_kind"] == "episode_reset"
    assert client.resets[1]["reset_kind"] == "chunk_discard"
    assert client.resets[1].get("seed") is None
    assert client.resets[1]["chunk_index"] == 1


def test_paired_summary_rejects_changed_case_order():
    base = {
        "case_id": "a",
        "panel": "C20",
        "success": False,
        "collision": False,
        "timeout": True,
        "action_norm_mean": 1.0,
        "action_norm_max": 1.0,
        "gripper_unique_signs": [-1, 1],
        "failure_class": "episode_failure",
    }
    inputs = {"all_cases": [{"case_id": "a"}, {"case_id": "b"}], "input_hashes": {}}
    with pytest.raises(ValueError, match="case order"):
        runner._paired_summary([base, {**base, "case_id": "b"}], [{**base, "case_id": "b"}, {**base, "case_id": "a"}], inputs, [], [], [], [])
