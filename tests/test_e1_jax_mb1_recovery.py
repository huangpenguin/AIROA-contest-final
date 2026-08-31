from __future__ import annotations

import json

import numpy as np
import pytest

from tools.axis_pi05.e1_jax_mb1_recovery import (
    _shard_paths,
    aggregate_shards,
    validate_contiguous_shards,
    validate_shard,
    write_atomic_shard,
)


PROVENANCE = {
    "lock_sha256": "lock",
    "manifest_sha256": "manifest",
    "inputs_sha256": "inputs",
    "model_sha256": "model",
    "norm_stats_sha256": "norm",
    "raw_params_tree_sha256": "raw",
    "microbatch_size": 1,
    "backend": "r6_folded_jax_source",
    "gpu_id": "1",
}


def _actions(count: int, marker: float) -> tuple[np.ndarray, np.ndarray]:
    return np.full((count, 10, 32), marker, dtype=np.float32), np.full((count, 10, 7), marker, dtype=np.float32)


def _write(path, start: int, stop: int, marker: float = 1.0):
    return write_atomic_shard(path, start, stop, *_actions(stop - start, marker), PROVENANCE, gpu_monitor={"runtime": {"pid": 1}})


def test_atomic_partial_shard_is_rejected(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    npz_path, _ = _shard_paths(shard_dir, 0, 16)
    np.savez_compressed(npz_path, internal_actions=np.zeros((16, 10, 32), dtype=np.float32), external_actions=np.zeros((16, 10, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="partial shard"):
        validate_contiguous_shards(shard_dir, 32, PROVENANCE)


def test_resume_rejects_gap_overlap_and_unexpected_temp_file(tmp_path):
    shard_dir = tmp_path / "shards"
    _write(shard_dir, 0, 16)
    _write(shard_dir, 32, 48)
    with pytest.raises(ValueError, match="isolated shard"):
        validate_contiguous_shards(shard_dir, 48, PROVENANCE)

    overlap = tmp_path / "overlap"
    _write(overlap, 0, 16)
    (overlap / ".0016_0032.npz.tmp").write_bytes(b"partial")
    with pytest.raises(ValueError, match="unexpected shard file"):
        validate_contiguous_shards(overlap, 32, PROVENANCE)


def test_corruption_and_provenance_drift_are_fail_closed(tmp_path):
    shard_dir = tmp_path / "shards"
    _write(shard_dir, 0, 16)
    npz_path, json_path = _shard_paths(shard_dir, 0, 16)
    with np.load(npz_path, allow_pickle=False) as file:
        internal, external = np.asarray(file["internal_actions"]), np.asarray(file["external_actions"])
    internal[0, 0, 0] = 99.0
    np.savez_compressed(npz_path, internal_actions=internal, external_actions=external)
    with pytest.raises(ValueError, match="file hash mismatch"):
        validate_shard(shard_dir, 0, 16, PROVENANCE)

    _write(tmp_path / "drift", 0, 16)
    drift_dir = tmp_path / "drift"
    metadata_path = _shard_paths(drift_dir, 0, 16)[1]
    metadata = json.loads(metadata_path.read_text())
    metadata["provenance"]["inputs_sha256"] = "changed"
    metadata["metadata_sha256"] = __import__("hashlib").sha256((json.dumps({key: value for key, value in metadata.items() if key != "metadata_sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="provenance drift"):
        validate_shard(drift_dir, 0, 16, PROVENANCE)


def test_aggregate_uses_explicit_observation_order_and_exact_340_coverage(tmp_path):
    shard_dir = tmp_path / "shards"
    # Write out of order; validation must follow 0,16,...,336,340.
    _write(shard_dir, 16, 32, marker=2.0)
    _write(shard_dir, 0, 16, marker=1.0)
    for start in range(32, 340, 16):
        _write(shard_dir, start, min(start + 16, 340), marker=float(start))
    internal, external, shards = aggregate_shards(shard_dir, 340, PROVENANCE)
    assert internal.shape == (340, 10, 32)
    assert external.shape == (340, 10, 7)
    assert len(shards) == 22
    assert float(internal[0, 0, 0]) == 1.0
    assert float(internal[16, 0, 0]) == 2.0
    assert float(internal[32, 0, 0]) == 32.0
    assert float(internal[339, 0, 0]) == 336.0

