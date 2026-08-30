import json
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05.evaluator_reproducibility import _trace_diagnostics, _validate_gpu_evidence
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
