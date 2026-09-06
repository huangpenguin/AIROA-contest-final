from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05 import track3_runtime_preflight_and_bounded_continuation as tool


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    try:
        tool.strict_json(path)
    except tool.DuplicateKeyError:
        pass
    else:
        raise AssertionError("duplicate JSON key was accepted")


def test_progress_window_stops_at_first_failed_window() -> None:
    result = tool.run_progress_stage([1.0, 0.8, 0.6, 0.79], window=3, grace=1, min_improvement=0.05)
    assert result["stopped"] is True
    assert result["failure_reason"] == "stage_stall"
    assert result["stop_index"] == 3


def test_progress_window_keeps_unfilled_window_not_evaluated() -> None:
    result = tool.progress_window_update([1.0, 0.9], window=4, grace=1, min_improvement=0.1)
    assert result["state"] == "not_evaluated"
    assert result["executed"] is False


def test_stage_contract_places_release_after_lift() -> None:
    contract = tool.stage_contract()
    assert contract["contact_phase"] == "close_to_lift"
    assert contract["source_release_phase"] == "after_lift_before_transport"
    assert contract["order"].index("lift") < contract["order"].index("transport")


def test_unentered_stage_is_not_evaluated() -> None:
    result = tool.phase_gate("place_same_demo_init_pose", [{"stage": "approach_grasp", "reached": True}])
    assert result["state"] == "not_evaluated"
    assert result["executed"] is False


def test_failed_gate_is_family_failure_but_worker_exception_is_infrastructure() -> None:
    assert tool.classify_worker_result(0, {"status": "failed_gate"})["category"] == "family_failure"
    assert tool.classify_worker_result(1, {"status": "worker_exception"})["category"] == "infrastructure"
    assert tool.classify_worker_result(0, {"status": "passed"})["category"] == "success"


def test_periodic_heartbeat_is_injected_and_atomic(tmp_path: Path) -> None:
    clock_value = [0.0]
    heartbeat = tool.PeriodicHeartbeat(tmp_path / "heartbeat.json", interval=10.0, clock=lambda: clock_value[0])
    assert heartbeat.maybe_write(phase="Y1", state="running", force=True)["count"] == 1
    assert heartbeat.maybe_write(phase="Y1", state="running") is None
    clock_value[0] = 10.0
    assert heartbeat.maybe_write(phase="Y1", state="running")["count"] == 2
    assert json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))["count"] == 2
    assert not list(tmp_path.glob("*.partial"))


def test_identity_gate_rejects_dirty_or_command_drift() -> None:
    identity = {
        "root_commit": "abc",
        "tool_sha256": "tool",
        "openpi_commit": "openpi",
        "command": ["worker"],
        "cuda_visible_devices": "",
        "gpu_policy": "cpu_no_renderer",
        "root_dirty": False,
        "cursor_active_before_launch": False,
    }
    assert tool.identity_gate(identity, expected_commit="abc", expected_tool_sha256="tool", expected_openpi="openpi", expected_command=["worker"])["state"] == "passed"
    identity["root_dirty"] = True
    identity["command"] = ["wrong"]
    assert tool.identity_gate(identity, expected_commit="abc", expected_tool_sha256="tool", expected_openpi="openpi", expected_command=["worker"])["state"] == "failed"


def test_y_audit_archives_y_and_releases_exactly_one_z_plan() -> None:
    handoff = json.loads((ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json").read_text(encoding="utf-8"))
    assert handoff["status"] == "track3_ef_failure_provenance_and_lifecycle_remediation_completed_hard_stop"
    assert handoff["next_plan"] is None
    assert handoff["next_phase_only"] == "new_reviewed_plan_required"
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md").is_file()

    audit = handoff["track3_runtime_preflight_y0_y2_audit"]
    assert audit["status"] == "remediation_required"
    assert (ROOT / audit["audit_report"]).is_file()
    assert (ROOT / audit["audit_summary"]).is_file()
    assert audit["verified"]["input_inventory_bytes_sha_exact"] == "45/45"
    assert audit["verified"]["manifest_git_tracked_bytes_sha_exact"] == "6/6"
    assert audit["findings"]["child_exact_argv_dispatches"] is False
    assert audit["findings"]["hard_gates_use_phase_scoped_windows"] is False
    assert audit["findings"]["frozen_clean_source_identity_enforced"] is False

    authorization = handoff["track3_y3_launch_phase_ledger_remediation_authorization"]
    assert authorization["status"] == "track3_y3_launch_phase_ledger_remediation_completed_and_audited"
    assert authorization["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md"
    assert authorization["authorization"]["support_nominal_cpu_physics_canaries_max"] == 1
    assert authorization["authorization"]["total_train_generations_max"] == 12
    assert authorization["authorization"]["gpu0"] is False
    assert authorization["authorization"]["automatic_upload_or_submission"] is False


def test_y0_y2_execution_evidence_is_closed_before_y3() -> None:
    handoff = json.loads((ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json").read_text(encoding="utf-8"))
    execution = handoff["track3_runtime_preflight_and_bounded_continuation_execution"]
    assert execution["status"] == "track3_runtime_y2_clean_release_ready"
    assert execution["y0"]["status"] == "passed"
    assert execution["y1"]["status"] == "passed"
    assert execution["y2"]["status"] == "passed"
    assert execution["exact_pending_accepted"] == [0, 0, 0]
    assert execution["y3_to_y13"] == "not_evaluated"
    assert (ROOT / execution["y0"]["report"]).is_file()
    assert (ROOT / execution["y1"]["report"]).is_file()
    assert (ROOT / execution["y2"]["report"]).is_file()
    assert (ROOT / execution["y2"]["manifest"]).is_file()
    assert (ROOT / execution["y2"]["final_validation"]).is_file()
