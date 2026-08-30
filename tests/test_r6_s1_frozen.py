from __future__ import annotations

import json

import numpy as np

from tools.axis_pi05.r6_s1_frozen import _sign_flip, compare_frozen_grid


def _write_replica(root, *, external_delta: float = 0.0) -> None:
    root.mkdir()
    internal = np.zeros((32, 10, 32), dtype=np.float32)
    external = np.zeros((32, 10, 7), dtype=np.float32)
    if external_delta:
        external[0, 0, 0] = external_delta
    np.savez_compressed(
        root / "e2e_action_parity.npz",
        folded_jax_actions=internal,
        folded_pytorch_actions=internal,
        folded_pytorch_actions_replay=internal,
        lerobot_postprocessed_actions=external,
        folded_jax_external_actions=external,
    )
    report = {
        "status": "passed",
        "backend": "lerobot_disk_processors_and_model_safetensors_with_openpi_pytorch_core",
        "batch_size": 32,
        "normalization": "QUANTILES",
        "processor_token_mask_mismatch_count": 0,
        "processor_token_max_abs_diff": 0,
        "processor_state_max_abs_diff": 0.0,
        "same_process_replay_required": True,
        "same_process_replay_within_tolerance": True,
        "model_sha256": "a" * 64,
        "norm_stats_sha256": "b" * 64,
        "source_config_sha256": "c" * 64,
        "processor_hashes": {},
        "same_process_replay": {"internal": {"exact": True}, "external": {"exact": True}},
    }
    (root / "e2e_action_parity_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_sign_flip_uses_external_gripper_dimension():
    left = np.zeros((2, 10, 7), dtype=np.float32)
    right = left.copy()
    right[1, 3, 6] = 1.0
    assert _sign_flip(left, right) == 1


def test_compare_frozen_grid_rejects_fresh_external_drift(tmp_path):
    first = tmp_path / "replica0"
    second = tmp_path / "replica1"
    _write_replica(first)
    _write_replica(second, external_delta=2e-6)
    result = compare_frozen_grid(first, second, tmp_path / "summary")
    assert result["status"] == "s1_frozen_grid_failed"
    assert result["first_failure"]["gate"] == "fresh_process_external_max_abs_le_1e-6"
    assert result["not_started_phases"] == ["S1.3", "S2", "S3", "S4", "S5", "S6", "S7"]
