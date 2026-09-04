from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_p3_io_recovery_and_support_executor_canary as tool


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    try:
        tool.strict_json_load(path)
    except ValueError as exc:
        assert "duplicate_json_key" in str(exc)
    else:
        raise AssertionError("duplicate JSON key was accepted")


def test_npz_validation_is_no_pickle_and_hash_exact(tmp_path: Path) -> None:
    path = tmp_path / "states.npz"
    array = np.arange(4, dtype=np.float64)
    with path.open("wb") as handle:
        np.savez_compressed(handle, task10_demo0=array)
    expected = {"task10_demo0": {"shape": [4], "sha256": tool.sha256_bytes(array.tobytes())}}
    assert tool.validate_npz(path, expected)["state"] == "passed"
    with path.open("wb") as handle:
        np.savez_compressed(handle, task10_demo0=array + 1.0)
    assert tool.validate_npz(path, expected)["state"] == "failed"


def test_manifest_rejects_duplicates_hash_mismatch_and_untracked(tmp_path: Path) -> None:
    path = tool.OUT_ROOT / "r1_io_recovery_contract.json"
    entry = tool.artifact_entry(path, source_commit="test")
    assert tool.validate_artifact_manifest([entry])["state"] == "passed"
    old_manifest = tool.strict_json_load(tool.OLD_MANIFEST)
    assert tool.validate_artifact_manifest(old_manifest["entries"], require_git_tracking=True)["state"] == "passed"
    duplicate = tool.validate_artifact_manifest([entry, entry])
    assert duplicate["state"] == "failed"
    bad = dict(entry)
    bad["sha256"] = "0" * 64
    assert tool.validate_artifact_manifest([bad])["state"] == "failed"


def test_result_protocol_handles_large_stdout_without_pipe_deadlock(tmp_path: Path) -> None:
    result = tool._run_result_protocol("large", deadline_seconds=10.0, term_grace_seconds=1.0, output_dir=tmp_path / "large")
    assert result["status"] == "passed"
    assert result["stdout_bytes"] >= 16 * 1024 * 1024
    assert result["worker_pid_match"] is True
    assert result["atomic_partial_residue"] == []


def test_result_protocol_terminates_hang_fail_closed(tmp_path: Path) -> None:
    result = tool._run_result_protocol("hang", deadline_seconds=0.2, term_grace_seconds=0.1, output_dir=tmp_path / "hang")
    assert result["status"] == "timeout_terminated"
    assert result["timed_out"] is True
    assert result["atomic_partial_residue"] == []


def test_result_protocol_closes_malformed_nonzero_and_exception(tmp_path: Path) -> None:
    for mode in ("malformed", "nonzero", "exception"):
        result = tool._run_result_protocol(mode, deadline_seconds=10.0, term_grace_seconds=1.0, output_dir=tmp_path / mode)
        assert result["status"] == "failed_closed"
        assert result["atomic_partial_residue"] == []


def test_r0_artifact_closure_report_is_passed() -> None:
    report = json.loads((tool.OUT_ROOT / "r0_artifact_closure.json").read_text(encoding="utf-8"))
    assert report["status"] == "track3_p3_recovery_r0_artifact_closure_passed"
    assert report["git_tracking"]["tracked_entry_count"] == 17
    assert (tool.OUT_ROOT / "R0_ERRATA_ZH.md").is_file()


def test_r1_result_contract_is_passed() -> None:
    report = json.loads((tool.OUT_ROOT / "r1_io_recovery_contract.json").read_text(encoding="utf-8"))
    assert report["status"] == "track3_p3_recovery_r1_result_protocol_passed"
    assert report["protocol"]["result_transport"] == "independent_atomic_json_file"
    assert report["tests"]["large_stdout_no_deadlock"] is True
    assert report["tests"]["hang_term_then_kill_fail_closed"] is True
