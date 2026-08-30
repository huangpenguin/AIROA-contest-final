from __future__ import annotations

import json

import numpy as np

from tools.axis_pi05.r6_s1_egl import (
    CAMERAS,
    TRAJECTORY_MODES,
    _corrected_reference_label,
    _dedupe_semantic_pairs,
    _trajectory_candidates_from_comparison,
)


def _candidate(*, left_path: str, right_path: str, left_replica: str, right_replica: str) -> dict:
    return {
        "case_id": "case0",
        "phase_or_trajectory_frame": "E3-7",
        "camera": "agentview_image",
        "policy_state_hash": "state",
        "prompt_hash": "prompt",
        "left_image_data_sha256": "left-image",
        "right_image_data_sha256": "right-image",
        "source_comparison": "comparison.json",
        "left_replica": left_replica,
        "right_replica": right_replica,
        "left_image_path": left_path,
        "right_image_path": right_path,
        "frame_index": None,
    }


def _descriptor(path) -> dict:
    return {"path": str(path)}


def _trajectory_case(tmp_path, *, changed: bool, dynamics_mismatch: bool = False) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = {"case_id": "case0", "manifest": {"language": "put object"}, "trajectories": {}}
    for mode in TRAJECTORY_MODES:
        image = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        if changed:
            image[1, 0, 0, 0] = 1
        wrist = np.zeros_like(image)
        pos = np.zeros((2, 3), dtype=np.float32)
        quat = np.zeros((2, 4), dtype=np.float32)
        quat[:, 3] = 1.0
        gripper = np.zeros((2, 2), dtype=np.float32)
        zeros = np.zeros((2, 4), dtype=np.float32)
        qpos = np.zeros((2, 4), dtype=np.float64)
        if dynamics_mismatch:
            qpos[1, 0] = 1.0
        values = {
            "obs:agentview_image": image,
            "obs:robot0_eye_in_hand_image": wrist,
            "obs:robot0_eef_pos": pos,
            "obs:robot0_eef_quat": quat,
            "obs:robot0_gripper_qpos": gripper,
            "sim_state_flattened": zeros,
            "qpos": qpos,
            "qvel": zeros,
            "ctrl": zeros,
        }
        arrays = {}
        for key, value in values.items():
            path = tmp_path / f"{mode}_{key.replace(':', '_')}.npy"
            np.save(path, value)
            arrays[key] = _descriptor(path)
        result["trajectories"][mode] = {"arrays": arrays, "length": 2}
    return result


def _trajectory_comparison() -> dict:
    return {
        "cases": [
            {
                "case_id": "case0",
                "trajectory_renderer_exact_by_camera_and_stage": {
                    "zero64:agentview": {"exact": False},
                    "zero64:robot0_eye_in_hand": {"exact": True},
                    "fixed64:agentview": {"exact": False},
                    "fixed64:robot0_eye_in_hand": {"exact": True},
                },
            }
        ]
    }


def test_reference_label_cannot_identify_r6_folded_jax_as_axis():
    assert _corrected_reference_label("axis_reference_jax_external_full_chunk") == "r6_folded_jax_reference_external_full_chunk"
    assert _corrected_reference_label("deployment_pytorch_external_full_chunk") == "deployment_pytorch_external_full_chunk"


def test_semantic_dedupe_ignores_replica_paths_for_same_content():
    first = _candidate(left_path="/replica0/a.npy", right_path="/replica1/b.npy", left_replica="replica0", right_replica="replica1")
    second = _candidate(left_path="/replica1/a.npy", right_path="/replica2/b.npy", left_replica="replica1", right_replica="replica2")
    unique, duplicates = _dedupe_semantic_pairs([first, second])
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert len(unique[0]["source_occurrences"]) == 2


def test_zero64_and_fixed64_affected_frames_are_covered(tmp_path):
    left = _trajectory_case(tmp_path / "left", changed=False)
    right = _trajectory_case(tmp_path / "right", changed=True)
    result = _trajectory_candidates_from_comparison(
        comparison=_trajectory_comparison(),
        left_cases={"case0": left},
        right_cases={"case0": right},
        left_replica="replica0",
        right_replica="replica1",
        source_comparison="comparison.json",
    )
    assert {item["phase_or_trajectory_frame"] for item in result} == {"zero64/frame-0001", "fixed64/frame-0001"}
    assert {item["camera"] for item in result} == {"agentview_image"}


def test_dynamics_mismatch_is_not_renderer_only(tmp_path):
    left = _trajectory_case(tmp_path / "left", changed=False)
    right = _trajectory_case(tmp_path / "right", changed=True, dynamics_mismatch=True)
    result = _trajectory_candidates_from_comparison(
        comparison=_trajectory_comparison(),
        left_cases={"case0": left},
        right_cases={"case0": right},
        left_replica="replica0",
        right_replica="replica1",
        source_comparison="comparison.json",
    )
    assert result
    assert all(item["dynamics_exact"] is False for item in result)
    assert all(item["renderer_only_evaluable"] is False for item in result)
