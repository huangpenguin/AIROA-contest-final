from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.axis_pi05 import track3_y3_launch_phase_ledger_remediation as tool


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_json_is_rejected() -> None:
    path = ROOT / "data/reports/axis_libero_30k" / "_z_test_duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    try:
        try:
            tool.strict_json(path)
        except tool.DuplicateKeyError:
            pass
        else:
            raise AssertionError("duplicate key accepted")
    finally:
        path.unlink(missing_ok=True)


def test_exact_worker_dispatch_accepts_result_output_without_simulator(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    command = [sys.executable, "-m", tool.MODULE, "--z1-worker", "--result-output", str(result_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["physics_steps"] == 0
    assert payload["result_path"] == str(result_path)


def test_worker_dispatch_rejects_unknown_duplicate_and_missing_args(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    base = [sys.executable, "-m", tool.MODULE, "--z1-worker"]
    assert subprocess.run(base, cwd=ROOT, capture_output=True, check=False).returncode == 2
    assert subprocess.run(base + ["--result-output", str(result_path), "--unknown"], cwd=ROOT, capture_output=True, check=False).returncode == 2
    assert subprocess.run(base + ["--result-output", str(result_path), "--result-output", str(result_path)], cwd=ROOT, capture_output=True, check=False).returncode == 2


def test_failed_gate_and_worker_exception_are_distinct() -> None:
    assert tool.classify_worker_result(0, {"status": "failed_gate"})["category"] == "family_failure"
    assert tool.classify_worker_result(1, {"status": "worker_exception"})["category"] == "infrastructure"
    assert tool.classify_worker_result(2, None)["category"] == "infrastructure"


def test_source_identity_is_explicit_and_dirty_fail_closed() -> None:
    identity = {"root_commit": "c", "tool_sha256": "t", "openpi_commit": "o", "command": ["worker"], "cuda_visible_devices": "", "gpu_policy": "cpu_no_renderer", "source_porcelain": [], "libero_plus_porcelain": []}
    assert tool.identity_gate(identity, identity, expected_command=["worker"])["state"] == "passed"
    dirty = dict(identity, source_porcelain=[" M tests/test.py"])
    assert tool.identity_gate(dirty, identity, expected_command=["worker"])["state"] == "failed"
    drift = dict(identity, root_commit="other")
    assert tool.identity_gate(drift, identity, expected_command=["worker"])["state"] == "failed"


def test_stage_ledger_freezes_close_and_uses_lift_for_source_release() -> None:
    ledger = tool._complete_ledger_fixture(release_ok=False)
    result = tool.final_hard_gate(ledger)
    assert result["gates"]["contact_completion"]["state"] == "passed"
    assert result["gates"]["source_release"]["owner_stage"] == "lift"
    assert result["gates"]["goal_continuity"]["state"] == "failed"


def test_destination_gate_is_numeric_and_unentered_is_not_evaluated() -> None:
    assert tool.destination_numeric_gate(stage_entered=False, object_position_error_m=None, object_rotation_error_rad=None, eef_position_error_m=None, eef_rotation_error_rad=None)["state"] == "not_evaluated"
    failed = tool._complete_ledger_fixture(destination_ok=False)
    assert tool.final_hard_gate(failed)["gates"]["destination_numeric"]["state"] == "failed"


def test_progress_stall_stops_at_first_failed_window() -> None:
    result = tool.run_progress_stage([1.0, 0.8, 0.6, 0.79], window=3, grace=1, min_improvement=0.05)
    assert result["stopped"] is True
    assert result["stop_index"] == 3
    assert result["failure_reason"] == "stage_stall"


def test_z0_z4_execution_evidence_is_closed_before_z5() -> None:
    handoff = json.loads((ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json").read_text(encoding="utf-8"))
    execution = handoff["track3_y3_launch_phase_ledger_remediation_execution"]
    assert execution["phase_status"]["Z0"] == "passed"
    assert execution["phase_status"]["Z4"] == "passed_clean_release"
    assert all(execution["phase_status"][f"Z{i}"] == "not_evaluated" for i in range(5, 15))
    assert execution["counts"] == {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}
    assert all((ROOT / value).is_file() for value in execution["reports"].values())
    closeout = json.loads((ROOT / execution["reports"]["closeout"]).read_text(encoding="utf-8"))
    assert closeout["status"] == "track3_z4_passed_pending_clean_commit"
    assert closeout["hard_stop"]["automatic_resume"] is False
