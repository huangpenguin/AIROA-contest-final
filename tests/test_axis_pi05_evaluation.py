from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.axis_pi05.eval_manifest import case_id, validate_case, validate_manifest_rows  # noqa: E402
from tools.axis_pi05.eval_result import coverage_report, summarize_episodes, write_result_bundle  # noqa: E402
from tools.axis_pi05.sampler_audit import audit_manifest  # noqa: E402
from tools.axis_pi05.select_winner import select  # noqa: E402


def _case(index: int, *, task: str = "task") -> dict:
    row = {
        "schema_version": 1,
        "case_id": "",
        "panel": "select_core",
        "selection_eligible": True,
        "benchmark": "libero_t1",
        "suite": "libero_t1",
        "task_id": index,
        "task_name": f"{task}-{index}",
        "language": f"do task {index}",
        "bddl_path": f"bddl/{index}.bddl",
        "bddl_sha256": "a" * 64,
        "init_state_index": 0,
        "init_state_sha256": "b" * 64,
        "env_seed": 1000,
        "policy_seed": 2000,
        "max_steps": 300,
        "perturbation_id": "clean",
        "perturbation_family": "clean",
        "perturbation_params": {},
        "repeat_id": 0,
    }
    row["case_id"] = case_id(row)
    return row


def _episode(case: dict, success: bool) -> dict:
    return {
        **case,
        "success": success,
        "collision": False,
        "goal_reached": success,
        "goal_reached_but_collision": False,
        "timeout": not success,
        "steps": 10 if success else 300,
        "elapsed_sec": 0.1,
        "failure_reason": None if success else "timeout",
    }


def test_case_id_and_duplicate_gate_are_deterministic():
    row = _case(0)
    validate_case(row)
    changed = dict(row, max_steps=299)
    changed["case_id"] = case_id(changed)
    assert changed["case_id"] != row["case_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_manifest_rows([row, row])


def test_sampler_audit_uses_frame_expansion(tmp_path):
    parquet = tmp_path / "sampling.parquet"
    import tools.axis_pi05.common as common

    common.write_table(parquet, [
        {"source_repo": "org/a", "source_episode_index": 0, "episode_sampling_probability": 0.5, "observed_frame_count": 1, "frame_weight": 0.5},
        {"source_repo": "org/b", "source_episode_index": 1, "episode_sampling_probability": 0.5, "observed_frame_count": 2, "frame_weight": 0.25},
    ])
    report = audit_manifest(parquet, seed=7, draws=10_000)
    assert report["sampling_unit"] == "frame"
    assert report["frame_count"] == 3
    assert report["empirical_gate_pass"]


def test_result_coverage_and_selection(tmp_path):
    cases = [_case(0), _case(1)]
    assert coverage_report([case["case_id"] for case in cases], [_episode(cases[0], True)]) ["complete"] is False
    result_dirs = []
    for candidate, successes in (("f0_m0_step1000", [True, False]), ("f0_m1_step1000", [True, True]), ("f0_m2_step1000", [False, False])):
        out = tmp_path / candidate
        write_result_bundle(
            out,
            [_episode(case, ok) for case, ok in zip(cases, successes)],
            expected_case_ids=[case["case_id"] for case in cases],
            summary_kwargs={
                "candidate": candidate,
                "panel": "select_core",
                "checkpoint": "/checkpoint/1000",
                "checkpoint_sha256": "c" * 64,
                "manifest_sha256": "d" * 64,
                "protocol_sha256": "e" * 64,
                "bootstrap_draws": 100,
            },
        )
        result_dirs.append((candidate, out))
    decision = select(result_dirs, out=tmp_path / "winner_decision.json", bootstrap_draws=100)
    assert decision["decision"] == "automatic"
    assert decision["winner"] == "f0_m1_step1000"
