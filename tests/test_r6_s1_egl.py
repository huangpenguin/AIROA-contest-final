from __future__ import annotations

import json

import numpy as np
import pytest

from tools.axis_pi05.r6_s1_egl import (
    CAMERAS,
    TRAJECTORY_MODES,
    _capture_locked_old_s1_3,
    _corrected_reference_label,
    _dedupe_semantic_pairs,
    _exact_array_pair,
    _coverage_set_audit,
    build_e0_reanalysis,
    _trajectory_candidates_from_comparison,
    _validate_snapshot_coverage,
    _validate_trajectory_coverage,
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


def _snapshot_case(tmp_path, *, right_shape: tuple[int, ...] = (2, 2, 3)) -> tuple[dict, dict]:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    left_image = np.zeros((2, 2, 3), dtype=np.uint8)
    right_image = np.zeros(right_shape, dtype=np.uint8)
    right_image.flat[0] = 1
    arrays = {}
    for side, directory in (("left", left_dir), ("right", right_dir)):
        values = {
            "image": left_image if side == "left" else right_image,
            "state": np.zeros((8,), dtype=np.float32),
            "flattened": np.zeros((4,), dtype=np.float64),
            "qpos": np.zeros((4,), dtype=np.float64),
            "qvel": np.zeros((4,), dtype=np.float64),
        }
        arrays[side] = {}
        for name, value in values.items():
            path = directory / f"{name}.npy"
            np.save(path, value)
            arrays[side][name] = {"path": str(path)}

    def make_case(side: str) -> dict:
        snapshot_list = []
        for phase in ("E1", "E2", *(f"E3-{step}" for step in range(1, 11))):
            if phase.startswith("E3-"):
                stage = "E3"
                step = int(phase.split("-", 1)[1])
                snapshot = {"stage": stage, "warmup_step": step}
            else:
                snapshot = {"stage": phase}
            image = arrays[side]["image"]
            snapshot_list.append(
                {
                    **snapshot,
                    "observation": {
                        "agentview_image": image,
                        "robot0_eye_in_hand_image": image,
                    },
                    "policy_state": arrays[side]["state"],
                    "simulator": {
                        "flattened": arrays[side]["flattened"],
                        "qpos": arrays[side]["qpos"],
                        "qvel": arrays[side]["qvel"],
                    },
                }
            )
        return {"case_id": "case0", "manifest": {"language": "put object"}, "snapshots": snapshot_list}

    # Replace the arrays with descriptors after constructing the same snapshot shape.
    left_case = make_case("left")
    right_case = make_case("right")
    for case, side in ((left_case, "left"), (right_case, "right")):
        for snapshot in case["snapshots"]:
            snapshot["observation"]["agentview_image"] = arrays[side]["image"]
            snapshot["observation"]["robot0_eye_in_hand_image"] = arrays[side]["image"]
            snapshot["policy_state"] = arrays[side]["state"]
            snapshot["simulator"] = {
                "flattened": arrays[side]["flattened"],
                "qpos": arrays[side]["qpos"],
                "qvel": arrays[side]["qvel"],
            }
    return left_case, right_case


def _snapshot_comparison(*, non_exact: bool = True) -> dict:
    renderer = {}
    for phase in ("E1", "E2", *(f"E3-{step}" for step in range(1, 11))):
        for camera in CAMERAS:
            renderer[f"{camera}:{phase}"] = {"exact": non_exact if phase == "E1" and camera == "agentview_image" else True, "affected_frame_count": 1 if non_exact and phase == "E1" and camera == "agentview_image" else 0}
    return {"cases": [{"case_id": "case0", "renderer_exact_by_camera_and_stage": renderer, "trajectory_renderer_exact_by_camera_and_stage": {}}]}


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


@pytest.mark.parametrize(
    ("left", "right", "reason"),
    [(None, None, "missing_both"), (None, np.zeros(1), "missing_left"), (np.zeros(1), None, "missing_right")],
)
def test_missing_array_pair_is_not_exact_or_evaluable(left, right, reason):
    result = _exact_array_pair(left, right)
    assert result["present_left"] is (left is not None)
    assert result["present_right"] is (right is not None)
    assert result["exact"] is False
    assert result["evaluable"] is False
    assert result["reason"] == reason


def test_shape_mismatch_is_not_evaluable():
    result = _exact_array_pair(np.zeros((1, 2), dtype=np.uint8), np.zeros((1, 3), dtype=np.uint8))
    assert result["exact"] is False
    assert result["evaluable"] is False
    assert result["reason"] == "shape_mismatch"


def test_coverage_set_audit_is_fail_closed_for_missing_and_extra_groups():
    issues = []
    result = _coverage_set_audit({"expected-a", "expected-b"}, {"expected-a", "extra"}, issues)
    assert result["complete"] is False
    assert result["missing"] == ["expected-b"]
    assert result["extra"] == ["extra"]


def test_snapshot_missing_phase_and_shape_mismatch_form_coverage_issues(tmp_path):
    left, right = _snapshot_case(tmp_path)
    right["snapshots"].pop()
    issues = []
    _validate_snapshot_coverage(
        comparison=_snapshot_comparison(),
        left_cases={"case0": left},
        right_cases={"case0": right},
        source_comparison="comparison.json",
        expected=set(),
        observed=set(),
        issues=issues,
    )
    assert any(issue["code"] == "expected_snapshot_phase_missing" for issue in issues)

    left, right = _snapshot_case(tmp_path / "shape", right_shape=(2, 3, 3))
    issues = []
    _validate_snapshot_coverage(
        comparison=_snapshot_comparison(),
        left_cases={"case0": left},
        right_cases={"case0": right},
        source_comparison="comparison.json",
        expected=set(),
        observed=set(),
        issues=issues,
    )
    assert any(issue["code"] == "snapshot_image_pair_not_evaluable" for issue in issues)


def test_source_non_exact_without_changed_frame_forms_coverage_issue(tmp_path):
    left = _trajectory_case(tmp_path / "left", changed=False)
    right = _trajectory_case(tmp_path / "right", changed=False)
    comparison = _trajectory_comparison()
    comparison["cases"][0]["trajectory_renderer_exact_by_camera_and_stage"]["zero64:agentview"]["affected_frame_count"] = 0
    issues = []
    _validate_trajectory_coverage(
        comparison=comparison,
        left_cases={"case0": left},
        right_cases={"case0": right},
        source_comparison="comparison.json",
        expected=set(),
        observed=set(),
        issues=issues,
    )
    assert any(issue["code"] == "source_non_exact_without_changed_frame" for issue in issues)


def test_trajectory_missing_dynamics_evidence_forms_coverage_issue(tmp_path):
    left = _trajectory_case(tmp_path / "left", changed=False)
    right = _trajectory_case(tmp_path / "right", changed=True)
    del right["trajectories"]["zero64"]["arrays"]["ctrl"]
    issues = []
    _validate_trajectory_coverage(
        comparison=_trajectory_comparison(),
        left_cases={"case0": left},
        right_cases={"case0": right},
        source_comparison="comparison.json",
        expected=set(),
        observed=set(),
        issues=issues,
    )
    assert any(issue["code"] == "missing_image_or_evidence_descriptor" for issue in issues)


def test_e0_closure_artifact_is_fail_closed_and_preserves_old_s1_3(tmp_path):
    before = _capture_locked_old_s1_3()
    result = build_e0_reanalysis(tmp_path)
    after = _capture_locked_old_s1_3()
    assert after["status"] == before["status"] == "s1_real_egl_pairs_failed"
    assert after["summary_sha256"] == before["summary_sha256"]
    assert after["manifest_sha256"] == before["manifest_sha256"]
    assert result["status"] == "e0_closure_complete_pending_review"
    assert result["coverage"]["coverage_complete"] is True
    assert result["coverage"]["trajectory_affected_frame_occurrence_count"] == 302
    assert result["coverage"]["trajectory_content_unique_pair_count"] == 157
    assert result["coverage"]["trajectory_affected_frame_occurrence_count"] != result["coverage"]["trajectory_content_unique_pair_count"]
    assert result["legacy_selection_audit"]["old_content_unique_pair_count"] == 3
    manifest = json.loads((tmp_path / "content_unique_pair_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 170
    assert all(item["source_occurrence_hashes"] for item in manifest["pairs"])
    assert all(item["frame_index"] is not None for item in manifest["pairs"] if item["source_kind"] == "trajectory")
    assert "axis_reference_jax_" not in json.dumps(result, ensure_ascii=False)
