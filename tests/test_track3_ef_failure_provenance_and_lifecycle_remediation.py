from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tools.axis_pi05 import track3_ef_failure_provenance_and_lifecycle_remediation as tool


def test_duplicate_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(tool.DuplicateKeyError):
        tool.strict_json(path)


def test_identity_gate_rejects_root_and_command_drift() -> None:
    command = [sys.executable, "-m", tool.MODULE, "--eg-fixture-child"]
    expected = tool.identity_snapshot(command)
    assert tool.identity_gate(expected, expected, command)["state"] == "passed"
    dirty = dict(expected)
    dirty["root_porcelain"] = [" M synthetic"]
    dirty["root_porcelain_sha256"] = tool.porcelain_digest(dirty["root_porcelain"])
    assert tool.identity_gate(dirty, expected, command)["state"] == "failed"
    drift = dict(expected)
    drift["command"] = ["different"]
    drift["command_hash"] = tool.command_hash(drift["command"])
    assert tool.identity_gate(drift, expected, command)["state"] == "failed"


def test_eg0_gap_is_unavailable_not_reconstructed(tmp_path: Path) -> None:
    result = tool.run_eg0(tmp_path)
    assert result["status"] == "passed"
    assert result["first_failed_final_validation_raw"]["status"] == "unavailable"
    assert result["first_failed_test_stdout_stderr_raw"]["status"] == "unavailable"
    assert result["first_failed_partial_inventory_raw"]["status"] == "unavailable"
    assert result["reconstruction_forbidden"] is True


def test_eg1_fail_retry_preserves_first_attempt(tmp_path: Path) -> None:
    result = tool.run_eg1(tmp_path)
    assert result["status"] == "passed"
    assert [row["status"] for row in result["attempts"]] == ["failed", "passed"]
    assert result["first_attempt_bytes_unchanged_after_retry"] is True
    assert result["non_reusable_monotonic_ids"] is True
    assert result["aggregate_does_not_replace_attempt"] is True


def test_eg2_clean_source_fixture_and_drift_fail_closed(tmp_path: Path) -> None:
    result = tool.run_eg2(tmp_path, require_clean_source=False)
    assert result["status"] == "passed"
    assert result["clean_source"]["child_identity_gate"]["state"] == "passed"
    assert result["drift_fail_closed"]["root_dirty"]["state"] == "failed"
    assert result["drift_fail_closed"]["temporary_git_dirty_detected"] is True


def test_attempt_manifest_is_checkout_stable(tmp_path: Path) -> None:
    result = tool.run_eg1(tmp_path)
    attempt = tmp_path / "attempts" / "attempt_0001"
    closure = json.loads((attempt / "checkout_stable_closure.json").read_text(encoding="utf-8"))
    assert closure["status"] == "passed"
    assert closure["inactive_cursor"] is True
    assert tool._scan_json(attempt)["status"] == "passed"


def test_no_physics_profile_and_trace_are_typed() -> None:
    assert tool.profile_sha()
    assert tool.EF_AUDIT_INPUT_COMMIT == "b1195dd59830f7c44d831bd0007e61229446d9b5"
