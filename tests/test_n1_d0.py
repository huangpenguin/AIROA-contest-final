import json
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05.evaluator_reproducibility import _trace_diagnostics, _validate_gpu_evidence
from tools.axis_pi05.n1_environment_probe import _image_diff_metrics, compare_probe_summaries_semantic
from tools.axis_pi05.n1_snapshot import _policy_state, gpu_boundary_evidence, save_array, validate_gpu_namespace


def test_first_nonzero_and_case_max_are_separate():
    left = np.zeros((2, 7), dtype=np.float32)
    right = left.copy()
    right[0, 0] = np.float32(1e-7)
    right[1, 3] = np.float32(0.5)
    result = _trace_diagnostics(left, right)
    assert result["first_nonzero_index"] == [0, 0]
    assert result["abs_diff_at_first_nonzero"] == pytest.approx(1e-7, abs=1e-12)
    assert result["case_max_abs_diff"] == pytest.approx(0.5)
    assert result["case_max_abs_diff_index"] == [1, 3]
    assert result["first_step_exceeding"]["0.1"] == 1


def test_trace_shape_mismatch_uses_common_prefix():
    left = np.zeros((3, 7), dtype=np.float32)
    right = np.zeros((1, 7), dtype=np.float32)
    right[0, 2] = np.float32(0.25)
    result = _trace_diagnostics(left, right)
    assert result["common_prefix_len"] == 1
    assert result["case_max_abs_diff"] == pytest.approx(0.25)
    assert result["first_nonzero_index"] == [0, 2]


def test_gpu0_false_requires_monitor_evidence():
    with pytest.raises(ValueError, match="monitor samples"):
        gpu_boundary_evidence([], target_pids={123}, allowed_gpu_ids={1})


def test_comparator_cannot_mark_missing_gpu_evidence_false():
    with pytest.raises(ValueError, match="gpu_monitor_evidence"):
        _validate_gpu_evidence({"gpu0_touched": False}, allow_legacy=False)


def test_outer_child_gpu_namespace_must_match():
    outer = {"CUDA_VISIBLE_DEVICES": "1", "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": "0", "JAX_PLATFORMS": "cuda"}
    child = {**outer, "CUDA_VISIBLE_DEVICES": "0,1"}
    with pytest.raises(ValueError, match="namespace mismatch"):
        validate_gpu_namespace(outer, child)


def test_snapshot_array_round_trip(tmp_path: Path):
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    descriptor = save_array(tmp_path, "fixture/value.npy", source)
    restored = np.load(tmp_path / "fixture/value.npy", allow_pickle=False)
    np.testing.assert_array_equal(restored, source)
    assert descriptor["dtype"] == "float32"
    assert descriptor["shape"] == [2, 3]
    assert descriptor["data_sha256"]
    assert descriptor["file_sha256"]


def test_snapshot_policy_state_is_explicit_float32():
    state = _policy_state({
        "robot0_eef_pos": np.array([1, 2, 3], dtype=np.float64),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float64),
        "robot0_gripper_qpos": np.array([4, 5], dtype=np.float64),
    })
    assert state is not None
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state, np.array([1, 2, 3, 0, 0, 0, 4, 5], dtype=np.float32))


def test_gpu_boundary_detects_target_gpu0():
    samples = [
        {"gpus": [{"index": 0, "pids": [{"pid": 123}]}, {"index": 1, "pids": []}]},
        {"gpus": [{"index": 0, "pids": [{"pid": 123}]}, {"index": 1, "pids": []}]},
    ]
    result = gpu_boundary_evidence(samples, target_pids={123}, allowed_gpu_ids={1})
    assert result["gpu0_touched"] is True
    assert result["forbidden_target_gpu_ids"] == [0]


def test_image_metrics_separate_rgb_elements_from_pixels():
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    right = left.copy()
    right[0, 1, 0] = 1
    right[0, 1, 1] = 2
    result = _image_diff_metrics(left, right)
    assert result["changed_element_count"] == 2
    assert result["changed_pixel_count"] == 1
    assert result["max_abs_diff"] == 2
    assert result["affected_frame_count"] == 1
    assert result["first_affected_frame"] == 0


def _semantic_summary(root: Path, name: str, *, e1_image_delta: bool) -> Path:
    def array(label: str, value: np.ndarray) -> dict:
        return save_array(root, f"{name}/{label}.npy", value)

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    e1_other = image.copy()
    if e1_image_delta:
        e1_other[0, 0, 0] = 1
    scalar = np.zeros((2,), dtype=np.float32)
    sim = {key: array(f"sim_{key}", scalar) for key in ("flattened", "qpos", "qvel")}
    snapshots = []
    for stage, warmup, stage_image in (
        ("E0", None, None),
        ("E1", None, e1_other),
        ("E2", None, image),
        ("E3", 10, image),
    ):
        observations = {}
        if stage_image is not None:
            observations = {
                "agentview_image": array(f"{stage}_agentview", stage_image),
                "robot0_eye_in_hand_image": array(f"{stage}_wrist", stage_image),
                "robot0_eef_pos": array(f"{stage}_pos", np.zeros((3,), dtype=np.float32)),
            }
        snapshots.append({
            "stage": stage,
            "warmup_step": warmup,
            "model": {"nq": 1},
            "controller": {"class": "controller"},
            "simulator": sim,
            "observation": observations,
            "policy_state": array(f"{stage}_policy", np.zeros((8,), dtype=np.float32)) if stage_image is not None else None,
        })
    trajectory_arrays = {
        key: array(f"zero_{key}", np.zeros((2, 2), dtype=np.float32))
        for key in ("actions", "sim_state_flattened", "qpos", "qvel", "ctrl")
    }
    trajectory_arrays.update({
        "obs:agentview_image": array("zero_agentview_trace", np.zeros((2, 2, 2, 3), dtype=np.uint8)),
        "obs:robot0_eye_in_hand_image": array("zero_wrist_trace", np.zeros((2, 2, 2, 3), dtype=np.uint8)),
    })
    record = {
        "case_id": "C20-test",
        "model": {"nq": 1},
        "controller": {"class": "controller"},
        "snapshots": snapshots,
        "trajectories": {"zero64": {"arrays": trajectory_arrays}, "fixed64": {"arrays": trajectory_arrays}},
    }
    summary = {
        "probe": "environment_only",
        "manifest_sha256": "manifest",
        "case_ids": ["C20-test"],
        "gpu0_touched": False,
        "records": [{"replica": 0, "cases": [record]}],
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_semantic_comparator_keeps_intermediate_renderer_jitter_out_of_dynamics(tmp_path: Path):
    first = _semantic_summary(tmp_path, "first", e1_image_delta=False)
    second = _semantic_summary(tmp_path, "second", e1_image_delta=True)
    result = compare_probe_summaries_semantic(first, second, tmp_path / "result.json")
    case = result["cases"][0]
    assert result["status"] == "passed"
    assert case["dynamics_exact"] is True
    assert case["policy_boundary_exact"] is True
    assert case["renderer_exact_by_camera_and_stage"]["agentview_image:E1"]["changed_element_count"] == 1
    assert result["technical_state"].endswith("renderer_jitter_pending_full_c20")


def test_image_metrics_reports_trajectory_first_frame_and_max():
    left = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    right = left.copy()
    right[1, 0, 0, 0] = 1
    right[2, 1, 1, 0] = 7
    result = _image_diff_metrics(left, right)
    assert result["changed_element_count"] == 2
    assert result["changed_pixel_count"] == 2
    assert result["affected_frame_count"] == 2
    assert result["total_frame_count"] == 3
    assert result["first_affected_frame"] == 1
    assert result["max_abs_diff"] == 7


def test_image_metrics_does_not_truncate_normalized_float_difference():
    left = np.zeros((1, 1, 1), dtype=np.float32)
    right = np.array([[[2.0 / 255.0]]], dtype=np.float32)
    result = _image_diff_metrics(left, right)
    assert result["changed_element_count"] == 1
    assert result["max_abs_diff"] == pytest.approx(2.0 / 255.0, rel=1e-6)
