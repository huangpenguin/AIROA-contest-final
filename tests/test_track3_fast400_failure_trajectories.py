from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import analyze_track3_fast400_failure_trajectories as d0


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        d0.strict_json(path)


def test_aligned_action_metrics_are_explicit_and_deterministic() -> None:
    first = np.asarray([[0.0] * 7, [1.0] * 7, [2.0] * 7])
    second = np.asarray([[0.0] * 7, [1.0] * 7])
    metrics = d0._aligned_metrics(first, second)
    assert metrics["aligned_frames"] == 2
    assert metrics["first_frames"] == 3
    assert metrics["second_frames"] == 2
    assert metrics["length_delta"] == 1
    assert metrics["action_rmse"] == 0.0
    assert metrics["mean_l2"] == 0.0


def test_lineage_audit_does_not_infer_independence_from_episode_names() -> None:
    metadata = {
        "task_id": "task_a",
        "created_at_utc": "2026-09-13T00:00:00+00:00",
        "operator": "operator-local",
    }
    assert d0._recursive_named_fields(metadata) == {}


def test_exposure_counts_use_probability_and_batch() -> None:
    values = d0._exposure_counts(0.1, 32)
    assert values == {"steps_25": 80.0, "steps_100": 320.0, "steps_300": 960.0}


def test_duplicate_groups_only_report_repeated_hashes() -> None:
    records = [
        {"episode_id": "a", "raw_trajectory_sha256": "x"},
        {"episode_id": "b", "raw_trajectory_sha256": "x"},
        {"episode_id": "c", "raw_trajectory_sha256": "y"},
    ]
    assert d0._duplicate_groups(records, "raw_trajectory_sha256") == {"x": ["a", "b"]}


def test_output_root_refuses_existing_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        d0.analyze(output)
