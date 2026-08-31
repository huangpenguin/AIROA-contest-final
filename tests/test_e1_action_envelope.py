from __future__ import annotations

import json

import numpy as np
import pytest

from tools.axis_pi05.e1_action_envelope import (
    _noise_for_pair,
    _source_occurrence_hash,
    validate_locked_inputs,
)
from tools.axis_pi05.common import sha256_file, write_json


def test_pair_local_noise_is_deterministic_and_shared_by_both_sides():
    first_seed, first = _noise_for_pair("pair-content-hash", 20260831)
    second_seed, second = _noise_for_pair("pair-content-hash", 20260831)
    other_seed, other = _noise_for_pair("other-content-hash", 20260831)
    assert first_seed == second_seed
    assert np.array_equal(first, second)
    assert first.shape == (10, 32)
    assert first.dtype == np.dtype("float32")
    assert not np.array_equal(first, other)


def test_source_occurrence_hash_is_immutable_to_path_record_mutation():
    occurrence = {
        "source_comparison": "comparison.json",
        "left_replica": "replica0",
        "right_replica": "replica1",
        "left_image_path": "/data/left.npy",
        "right_image_path": "/data/right.npy",
        "left_image_data_sha256": "left",
        "right_image_data_sha256": "right",
        "policy_state_hash": "state",
        "prompt_hash": "prompt",
        "frame_index": 3,
    }
    occurrence["source_occurrence_hash"] = _source_occurrence_hash(occurrence)
    assert _source_occurrence_hash(occurrence) == occurrence["source_occurrence_hash"]
    occurrence["frame_index"] = 4
    assert _source_occurrence_hash(occurrence) != occurrence["source_occurrence_hash"]


def test_validate_locked_inputs_rejects_missing_artifact_before_any_load(tmp_path):
    lock = tmp_path / "lock.json"
    manifest = tmp_path / "manifest.json"
    inputs = tmp_path / "inputs.npz"
    write_json(lock, {"actions_observed": False})
    write_json(manifest, {})
    with pytest.raises(ValueError, match="artifact is missing"):
        validate_locked_inputs(lock, manifest, inputs)


def test_validate_locked_inputs_rejects_manifest_mutation_closed_form(tmp_path):
    lock = tmp_path / "lock.json"
    manifest = tmp_path / "manifest.json"
    inputs = tmp_path / "inputs.npz"
    manifest.write_text('{"pair_count": 170}\n', encoding="utf-8")
    np.savez_compressed(inputs, sentinel=np.zeros(1, dtype=np.float32))
    write_json(
        lock,
        {
            "actions_observed": False,
            "e1_frozen_pair_manifest_sha256": sha256_file(manifest),
            "e1_frozen_inputs_and_noise_sha256": sha256_file(inputs),
        },
    )
    manifest.write_text('{"pair_count": 169}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen manifest"):
        validate_locked_inputs(lock, manifest, inputs)

