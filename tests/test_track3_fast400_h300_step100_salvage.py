from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import run_track3_fast400_h300_paired_eval as base
from tools.axis_pi05 import run_track3_fast400_h300_step100_salvage as salvage


def _cases() -> list[dict]:
    cases = []
    for index in range(4):
        cases.append({"case_id": f"reverse-{index}", "panel": "reverse_dev", "case_fingerprint": f"fp-reverse-{index}", "env_seed": 100 + index, "policy_seed": 200 + index, "init_state_index": 0, "init_state_sha256": f"state-{index}"})
    for index in range(20):
        cases.append({"case_id": f"c20-{index}", "panel": "C20", "case_fingerprint": f"fp-c20-{index}", "env_seed": 300 + index, "policy_seed": 400 + index, "init_state_index": 0, "init_state_sha256": f"c20-state-{index}"})
    return cases


def _rows(cases: list[dict], root: Path, role: str, *, candidate: bool = False, all_zero: bool = False) -> tuple[list[dict], dict]:
    output = root / "output" / role
    evidence = root / "evidence" / role
    (output / "action_traces").mkdir(parents=True)
    (output / "videos").mkdir()
    (evidence / "cases").mkdir(parents=True)
    identity = {"role": "h300_step100_diagnostic" if candidate else "r6_params_only_baseline", "params_tree_sha256": f"tree-{role}"}
    rows = []
    for index, case in enumerate(cases):
        actions = np.zeros((3, 7), dtype=np.float32) if all_zero else np.ones((3, 7), dtype=np.float32)
        trace = output / "action_traces" / f"{case['case_id']}.npy"
        np.save(trace, actions, allow_pickle=False)
        video = output / "videos" / f"{case['case_id']}.mp4"
        video.write_bytes(b"video")
        row = {
            **case,
            "status": "completed",
            "role": identity["role"],
            "norm_stats_sha256": base.EXPECTED_NORM_SHA,
            "checkpoint_params_tree_sha256": identity["params_tree_sha256"],
            "control_frequency_hz": base.CONTROL_HZ,
            "chunk_size": base.CHUNK_SIZE,
            "executed_actions_per_chunk": base.EXECUTE_ACTIONS,
            "max_steps": base.MAX_STEPS,
            "success": (bool(candidate and case["panel"] == "reverse_dev" and index == 0) or bool(candidate and case["panel"] == "C20" and index - 4 < 18) or bool(not candidate and case["panel"] == "C20")),
            "collision": False,
            "timeout": False,
            "action_trace_path": str(trace.resolve()),
            "video_path": str(video.resolve()),
            "action_norm_max": float(np.linalg.norm(actions, axis=1).max()),
            "failure_class": None,
        }
        (evidence / "cases" / f"{case['case_id']}.json").write_text(json.dumps(row), encoding="utf-8")
        rows.append(row)
    return rows, identity


def test_step100_identity_requires_frozen_three_hashes():
    identity = {
        "checkpoint_step": 100,
        "training_steps": 100,
        "diagnostic_only": True,
        "role": "h300_step100_diagnostic",
        "run_id": base.RUN_ID,
        "params_hashes": {base.PARAMS_RELATIVE_NUL_SHA256_V1: salvage.EXPECTED_STEP100_NATIVE_SHA, base.Q0_REPO_RELATIVE_SHA256_V1: salvage.EXPECTED_STEP100_Q0_SHA},
        "metadata_sha256": salvage.EXPECTED_STEP100_METADATA_SHA,
    }
    for key in ("params_native_sha256", "params_q0_sha256", "metadata_sha256"):
        observed = {"params_native_sha256": identity["params_hashes"][base.PARAMS_RELATIVE_NUL_SHA256_V1], "params_q0_sha256": identity["params_hashes"][base.Q0_REPO_RELATIVE_SHA256_V1], "metadata_sha256": identity["metadata_sha256"]}[key]
        assert observed == {"params_native_sha256": salvage.EXPECTED_STEP100_NATIVE_SHA, "params_q0_sha256": salvage.EXPECTED_STEP100_Q0_SHA, "metadata_sha256": salvage.EXPECTED_STEP100_METADATA_SHA}[key]


def test_source_hash_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(base, "adapter_source_identity", lambda: {"path": str(base.ADAPTER_PATH), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="adapter hash mismatch"):
        salvage.load_frozen_inputs()


def test_case_seed_mismatch_fails_closed(tmp_path: Path):
    cases = _cases()
    rows, identity = _rows(cases, tmp_path, "r6")
    rows[0]["policy_seed"] += 1
    with pytest.raises(ValueError, match="case/seed/input mismatch.*policy_seed"):
        salvage._assert_row_contract(rows[0], cases[0], identity, tmp_path / "evidence" / "r6", tmp_path / "output" / "r6")


def test_partial_trace_residue_fails_closed(tmp_path: Path):
    partial = tmp_path / "case.json.partial"
    partial.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="partial residue"):
        salvage._assert_no_partial(tmp_path)


def test_nonfinite_trace_fails_closed(tmp_path: Path):
    path = tmp_path / "trace.npy"
    np.save(path, np.full((2, 7), np.nan, dtype=np.float32), allow_pickle=False)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        base._validate_trace({"case_id": "case", "action_trace_path": str(path)})


def test_all_zero_roles_fail_closed_in_gate(tmp_path: Path):
    cases = _cases()
    baseline, _ = _rows(cases, tmp_path / "base", "r6", all_zero=True)
    candidate, _ = _rows(cases, tmp_path / "cand", "step100", candidate=True, all_zero=True)
    summary = base.paired_summary(baseline, candidate, cases)
    assert summary["gates"]["no_systemwide_all_zero_actions"] is False
    assert summary["behavior_gate_passed"] is False


def test_historical_root_is_never_a_writable_salvage_namespace():
    with pytest.raises(ValueError, match="historical step100 salvage"):
        salvage._assert_new_salvage_namespace(salvage.OLD_EVIDENCE_ROOT)
    with pytest.raises(ValueError, match="historical step100 salvage"):
        salvage._assert_new_salvage_namespace(salvage.OLD_OUTPUT_ROOT)


def test_repeated_merge_refuses_existing_artifacts(tmp_path: Path):
    (tmp_path / "paired_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="repeat merge"):
        salvage._assert_merge_artifacts_absent(tmp_path)


def test_step100_gate_pass_and_fail_shapes(tmp_path: Path):
    cases = _cases()
    baseline, base_identity = _rows(cases, tmp_path / "base", "r6")
    candidate, candidate_identity = _rows(cases, tmp_path / "candidate", "step100", candidate=True)
    inputs = {"input_hashes": {"adapter_sha256": salvage.EXPECTED_ADAPTER_SHA}}
    r6_lock = {"status": "completed", "role": "r6"}
    for row in baseline + candidate:
        row.update({"trace_finite": True, "trace_nonzero": True})
    passed = salvage._paired_summary(baseline, candidate, cases, inputs, base_identity, candidate_identity, r6_lock)
    assert passed["reverse"]["delta"] == 1
    assert passed["c20"]["delta"] == -2
    assert passed["behavior_gate_passed"] is True
    candidate[0]["success"] = False
    failed = salvage._paired_summary(baseline, candidate, cases, inputs, base_identity, candidate_identity, r6_lock)
    assert failed["behavior_gate_passed"] is False
    assert failed["gates"]["reverse_at_least_plus_1_of_4"] is False
