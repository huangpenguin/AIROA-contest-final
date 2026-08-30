import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05.phase1_paired_report import (
    _mcnemar_exact,
    build_report,
)
from tools.axis_pi05.phase1_rollout import ChunkPolicy


def _write_rollout_bundle(root: Path, *, name: str, success_count: int, collision_count: int, timeout_count: int) -> Path:
    run = root / name
    videos = run / "videos"
    videos.mkdir(parents=True)
    manifest = root / "B80.jsonl"
    rows = []
    manifest_rows = []
    for index in range(80):
        case_id = f"B80-task{index // 2:02d}-init{index % 2}"
        video = videos / f"{case_id}.mp4"
        video.write_bytes(b"video")
        row = {
            "case_id": case_id,
            "official_task_index": index // 2,
            "task_name": f"task-{index // 2}",
            "language": f"do task {index // 2}",
            "bddl_sha256": "bddl",
            "init_state_index": index % 2,
            "init_state_sha256": f"init-{index}",
            "env_seed": 53000 + index,
            "policy_seed": 54000 + index,
            "max_steps": 300,
            "success": index < success_count,
            "collision": index < collision_count,
            "timeout": collision_count <= index < collision_count + timeout_count,
            "action_norm_mean": 1.0,
            "video_path": str(video),
        }
        rows.append(row)
        manifest_rows.append({"case_id": case_id})
    manifest.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")
    fields = sorted(rows[0])
    with (run / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary = {
        "panel": "B80",
        "execution_protocol": "execute8",
        "panel_manifest_sha256": manifest_sha,
        "case_count": 80,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "stall_count": 0,
        "video_count": 80,
        "checkpoint_identity": name,
        "checkpoint_path": f"/{name}",
        "checkpoint_step": 500,
        "checkpoint_kind": "raw_lora_orbax",
        "lora_arm": "L2_action_lora_heads",
        "norm_stats_path": "/norm.json",
        "norm_stats_sha256": "norm",
        "normalization_mode": "QUANTILES",
        "policy_seed_source": "frozen manifest policy_seed -> reset(seed) per episode",
        "artifacts": {
            "episodes_csv": str(run / "episodes.csv"),
            "videos_dir": str(videos),
        },
    }
    summary_path = run / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_paired_report_rejects_per_case_seed_mismatch(tmp_path):
    candidate = _write_rollout_bundle(tmp_path, name="candidate", success_count=49, collision_count=20, timeout_count=15)
    baseline = _write_rollout_bundle(tmp_path, name="baseline", success_count=48, collision_count=23, timeout_count=19)
    baseline_csv = Path(json.loads(baseline.read_text())["artifacts"]["episodes_csv"])
    rows = list(csv.DictReader(baseline_csv.open(newline="", encoding="utf-8")))
    rows[0]["policy_seed"] = "99999"
    with baseline_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="per-case pairing invariant"):
        build_report(candidate, baseline, seed=1002, draws=100)


def test_paired_report_rejects_summary_csv_count_mismatch(tmp_path):
    candidate = _write_rollout_bundle(tmp_path, name="candidate", success_count=49, collision_count=20, timeout_count=15)
    baseline = _write_rollout_bundle(tmp_path, name="baseline", success_count=48, collision_count=23, timeout_count=19)
    candidate_data = json.loads(candidate.read_text())
    candidate_data["success_count"] = 48
    candidate.write_text(json.dumps(candidate_data), encoding="utf-8")
    with pytest.raises(ValueError, match="summary/CSV count mismatch"):
        build_report(candidate, baseline, seed=1002, draws=100)


def test_chunk_discard_does_not_reseed_and_failure_delta_direction_is_explicit(tmp_path):
    class FakeClient:
        def __init__(self):
            self.resets = []

        def reset(self, **kwargs):
            self.resets.append(kwargs)

        def get_action(self, _obs):
            return np.zeros(7, dtype=np.float32)

    client = FakeClient()
    policy = ChunkPolicy(client, mode="execute8")
    policy.reset(instruction="task", seed=54000)
    for _ in range(9):
        policy.get_action({})
    assert client.resets == [
        {"instruction": "task", "seed": 54000, "reset_kind": "episode_reset", "chunk_index": 0},
        {"instruction": "task", "reset_kind": "chunk_discard", "chunk_index": 1},
    ]
    assert _mcnemar_exact(9, 8)["candidate_wins"] == 9
    assert _mcnemar_exact(9, 8)["candidate_losses"] == 8

    candidate = _write_rollout_bundle(tmp_path, name="candidate", success_count=49, collision_count=20, timeout_count=15)
    baseline = _write_rollout_bundle(tmp_path, name="baseline", success_count=48, collision_count=23, timeout_count=19)
    report = build_report(candidate, baseline, seed=1002, draws=100)
    assert report["failure_count_delta"] == {"collision_count": -3, "timeout_count": -4, "stall_count": 0}

    reverse = build_report(baseline, candidate, seed=1002, draws=100)
    assert reverse["failure_count_delta"] == {"collision_count": 3, "timeout_count": 4, "stall_count": 0}
