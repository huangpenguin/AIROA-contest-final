from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import track3_tz_evidence_and_executor_contract_remediation as tool


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(tool.DuplicateKeyError):
        tool.strict_json(path)


def test_profile_lock_is_three_new_identities_only() -> None:
    lock = tool.profile_lock()
    assert lock["profile_order"] == list(tool.PROFILE_ORDER)
    assert len(lock["profiles"]) == 3
    assert {row["family"] for row in lock["profiles"]} == {"support_remove", "containment_remove", "articulation_close"}
    assert all(row["old_identity_rerun"] is False for row in lock["profiles"])
    assert all(row["training_eligible"] is False for row in lock["profiles"])
    assert not set(lock["profile_order"]) & tool.OLD_PROFILE_IDS


def test_cli_exposes_only_er_phases_and_cpu_worker() -> None:
    result = subprocess.run([sys.executable, "-m", tool.MODULE, "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    for flag in [*(f"--er{i}" for i in range(6)), "--execute", "--cpu-worker"]:
        assert flag in result.stdout
    for forbidden in ("--gpu3-worker", "--generation-worker", "--replay-worker", "--candidate-zip"):
        assert forbidden not in result.stdout


def test_seed_contract_is_applied_not_only_recorded() -> None:
    first = tool.apply_effective_seed(51010, np)
    first_values = (np.random.random(4),)
    second = tool.apply_effective_seed(51010, np)
    second_values = (np.random.random(4),)
    assert first["applied"] is True
    assert first["methods"] == second["methods"]
    np.testing.assert_array_equal(first_values[0], second_values[0])


def test_typed_trace_round_trip_is_no_pickle_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "trace.npz"
    tool.write_typed_trace(path, {"phase": np.asarray(["approach", "close"], dtype="U64"), "known": np.asarray([True, False], dtype=np.bool_), "value": np.asarray([0, 1], dtype=np.int8)})
    result = tool.validate_typed_trace(path)
    assert result["status"] == "passed"
    assert result["errors"] == []
    with np.load(path, allow_pickle=False) as archive:
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_object_trace_is_rejected_even_when_npz_can_be_opened_with_pickle(tmp_path: Path) -> None:
    path = tmp_path / "object.npz"
    np.savez_compressed(path, values=np.asarray([{"value": True}], dtype=object))
    result = tool.validate_typed_trace(path)
    assert result["status"] == "failed"
    assert any("allow_pickle" in error or "object_dtype" in error for error in result["errors"])


def test_gate_state_is_computed_and_tri_state() -> None:
    assert tool.gate_record(True, True)["state"] == "passed"
    assert tool.gate_record(True, False)["state"] == "failed"
    assert tool.gate_record(False, None)["state"] == "not_evaluated"
    assert tool.gate_record(True, None)["state"] == "unknown"
    assert tool.required_gates_passed({"a": tool.gate_record(True, True), "b": tool.gate_record(True, False)}) is False


def test_single_instance_lock_rejects_second_holder(tmp_path: Path) -> None:
    path = tmp_path / "supervisor.lock"
    with tool.supervisor_lock(path):
        with pytest.raises(BlockingIOError):
            with tool.supervisor_lock(path):
                pass


def test_identity_gate_detects_runtime_drift() -> None:
    expected = {"root_commit": "a", "tool_sha256": "b", "openpi_commit": "c", "profile_lock_sha256": "d", "cuda_visible_devices": "", "jax_platforms": "cpu", "mujoco_gl": "disable", "clean_root": True, "openpi_porcelain": [], "libero_plus_porcelain": []}
    observed = {**expected, "command": ["worker"]}
    assert tool.identity_gate({**observed, "command": ["worker"]}, expected, ["worker"])["state"] == "passed"
    assert tool.identity_gate({**observed, "root_commit": "drift"}, expected, ["worker"])["state"] == "failed"


def test_manifest_excludes_its_own_hash_outputs(tmp_path: Path) -> None:
    first = tmp_path / "one.json"
    second = tmp_path / "two.csv"
    first.write_text('{"ok":true}\n', encoding="utf-8")
    second.write_text("phase,status\nER0,passed\n", encoding="utf-8")
    manifest = tool._manifest_for([first, second])
    paths = {entry["path"] for entry in manifest["entries"]}
    assert "artifact_manifest.json" not in paths
    assert "outer_hash.json" not in paths
    assert manifest["historical_namespace_modified"] is False


def test_worker_exit_contract_keeps_family_failure_normal() -> None:
    assert tool.worker_exit("failed_gate") == 0
    assert tool.worker_exit("passed") == 0
    assert tool.worker_exit("infrastructure_failure") == 1


def test_er_execution_handoff_is_hard_stopped_and_fallback_only() -> None:
    handoff = tool.strict_json(ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json")
    assert handoff["status"] == "track3_multitask_gold_and_r6_short_continuation_reviewed_ready_for_mg0"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_MULTITASK_GOLD_AND_R6_SHORT_CONTINUATION_PLAN.md"
    assert handoff["next_phase_only"] == "MG0_evidence_lock_and_recorder_budget_remediation"
    execution = handoff["track3_tz_evidence_and_executor_contract_remediation_execution"]
    assert execution["status"] == "track3_tz_evidence_and_executor_contract_remediation_completed_hard_stop_pending_review"
    assert execution["phase_status"] == {"ER0": "passed", "ER1": "passed", "ER2": "passed", "ER3": "passed", "ER4": "completed_three_family_failures", "ER5": "hard_stop"}
    assert execution["new_cpu_canaries"]["executed"] == 3
    assert execution["new_cpu_canaries"]["successful_families"] == 0
    assert execution["fallback"]["recommended_for_track3_submission"] is False
    assert execution["resource_boundary"]["gpu3"] is False
