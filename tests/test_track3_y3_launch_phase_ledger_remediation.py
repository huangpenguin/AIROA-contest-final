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


def test_z5_dispatch_contract_and_profile_are_cpu_first(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    base = [sys.executable, "-m", tool.MODULE, "--z5-worker"]
    assert subprocess.run(base, cwd=ROOT, capture_output=True, check=False).returncode == 2
    assert subprocess.run(base + ["--result-output", str(result_path), "--unknown"], cwd=ROOT, capture_output=True, check=False).returncode == 2
    assert subprocess.run(base + ["--result-output", str(result_path), "--result-output", str(result_path)], cwd=ROOT, capture_output=True, check=False).returncode == 2
    assert tool.Z5_PROFILE["rotation_output_scale_rad"] == 0.5
    assert tool.Z5_PROFILE["rotation_clip_rad"] == 0.12
    assert tool.Z5_PROFILE["name"] == "object_frame_nominal"


def test_failed_gate_and_worker_exception_are_distinct() -> None:
    assert tool.classify_worker_result(0, {"status": "failed_gate"})["category"] == "family_failure"
    assert tool.classify_worker_result(1, {"status": "worker_exception"})["category"] == "infrastructure"
    assert tool.classify_worker_result(2, None)["category"] == "infrastructure"
    assert "failed_gate" in tool.WORKER_COMPLETED_STATUSES


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
    assert execution["status"] == "track3_z14_hard_stop_pending_review"
    assert execution["phase_status"]["Z0"] == "passed"
    assert execution["phase_status"]["Z4"] == "passed_clean_release"
    assert execution["phase_status"]["Z5"] == "track3_runtime_stopped_Z5_infrastructure_failure"
    assert all(execution["phase_status"][f"Z{i}"] == "not_evaluated" for i in range(6, 14))
    assert execution["phase_status"]["Z14"] == "hard_stop"
    assert execution["counts"] == {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}
    assert all((ROOT / value).is_file() for value in execution["reports"].values())
    closeout = json.loads((ROOT / execution["reports"]["z14_closeout"]).read_text(encoding="utf-8"))
    assert closeout["status"] == "track3_z14_hard_stop"
    assert closeout["hard_stop"]["automatic_resume"] is False


def test_z0_z14_audit_archives_z_and_releases_exactly_one_aa_plan() -> None:
    handoff_path = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
    handoff = tool.strict_json(handoff_path)
    assert handoff["status"] == "track3_ef_failure_provenance_and_lifecycle_remediation_completed_hard_stop_pending_independent_audit"
    assert handoff["next_plan"] is None
    assert handoff["next_phase_only"] == "independent_audit_required"
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md").is_file()

    audit = handoff["track3_y3_launch_z0_z14_audit"]
    assert audit["status"] == "remediation_required"
    assert (ROOT / audit["audit_report"]).is_file()
    assert (ROOT / audit["audit_summary"]).is_file()
    assert audit["verified"]["input_inventory_bytes_sha_exact"] == "45/45"
    assert audit["verified"]["z14_manifest_entries_bytes_sha_exact"] == "28/28"
    assert audit["verified"]["z14_manifest_tracked_namespace_coverage"] == "28/36"
    assert audit["derived_physical_result"]["historical_parent_category"] == "infrastructure"
    assert audit["derived_physical_result"]["derived_category"] == "family_failure"
    assert audit["derived_physical_result"]["nominal_rerun_authorized"] is False
    assert audit["findings"]["complete_artifact_and_self_hash_closure"] is False
    assert audit["findings"]["postmortem_gate_scope_exact"] is False

    authorization = handoff["track3_z5_derived_release_and_conservative_continuation_authorization"]
    assert authorization["status"] == "track3_z5_derived_release_and_conservative_continuation_completed_and_audited"
    assert authorization["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md"
    assert authorization["next_phase_only"] == "superseded_by_track3_tonight_zip_and_crossfamily_recovery_plan"
    scope = authorization["authorization"]
    assert scope["old_z_namespace_tracked_artifact_closure"] == "36/36"
    assert scope["nominal_rerun"] is False
    assert scope["support_conservative_cpu_canaries_max"] == 1
    assert scope["z5_failed_trajectory_training_eligible"] is False
    assert scope["gpu0"] is False
    assert scope["automatic_upload_or_submission"] is False

    execution = handoff["track3_z5_derived_release_and_conservative_continuation_execution"]
    assert execution["status"] == "track3_aa14_hard_stop_pending_review"
    assert execution["phase_status"]["AA5"] == "track3_aa5_support_conservative_cpu_family_failure"
    assert all(execution["phase_status"][f"AA{i}"] == "not_evaluated" for i in range(6, 14))
    assert execution["phase_status"]["AA14"] == "hard_stop"
    assert execution["checks"]["old_z_artifacts_exact"] == "36/36"
    assert execution["checks"]["aa5_category"] == "family_failure"
    assert execution["counts"] == {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}
