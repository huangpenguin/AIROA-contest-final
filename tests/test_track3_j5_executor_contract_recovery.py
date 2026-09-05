from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.axis_pi05 import track3_j5_executor_contract_and_atomic_video_recovery as tool


def test_xyzw_quaternion_sign_and_shortest_log_contract() -> None:
    assert tool.quaternion_log_error_xyzw([0, 0, 0, 1], [0, 0, 0, -1]) == [0.0, 0.0, 0.0]
    action = tool.osc_pose_action_xyzw([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.0)
    assert action == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
    with pytest.raises(ValueError):
        tool.osc_pose_action_xyzw([0, 0, 0], [0, 0, 0], 0.0)


def test_failed_gate_publication_and_parent_mapping_are_distinct() -> None:
    assert tool.valid_failed_gate_exit("failed_gate", publication_ok=True) == 0
    assert tool.valid_failed_gate_exit("worker_exception", publication_ok=False, exception=True) == 1
    assert tool.parent_category(child_status="failed_gate", process_returncode=0, telemetry_state="passed", camera_state="passed") == "family_failure"
    assert tool.parent_category(child_status="worker_exception", process_returncode=1, telemetry_state="passed", camera_state="passed") == "infrastructure"
    assert tool.parent_category(child_status="failed_gate", process_returncode=0, telemetry_state="passed", camera_state="failed") == "infrastructure"


def test_j11_propagates_first_infrastructure_error_and_heartbeat_is_periodic() -> None:
    result = tool.j11_propagate(phase="V4", status="failed_closed", category="infrastructure", reason="camera_publish_failure")
    assert result["reason"] == "camera_publish_failure"
    assert result["preserves_first_error"] is True
    assert tool.heartbeat_due(0.0, 899.9) is False
    assert tool.heartbeat_due(0.0, 900.0) is True


def test_launch_identity_and_duplicate_json_fail_closed(tmp_path: Path) -> None:
    assert tool.launch_identity_gate(expected_commit="a", observed_commit="a", expected_tool_sha256="b", observed_tool_sha256="b", dirty=False, pid_active=False, cursor_active=False)["state"] == "passed"
    assert tool.launch_identity_gate(expected_commit="a", observed_commit="b", expected_tool_sha256="c", observed_tool_sha256="d", dirty=True, pid_active=True, cursor_active=True)["state"] == "failed"
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": 1, "x": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        tool.strict_json_load(duplicate)


def test_v0_v1_v2_reports_and_offline_video_release() -> None:
    for name, status in (("v0_input_lock.json", "track3_j5_v0_passed"), ("v1_executor_contract.json", "track3_j5_v1_support_contract_passed"), ("v2_derived_old_nominal_errata.json", "track3_j5_v2_passed")):
        report = json.loads((tool.OUT / name).read_text(encoding="utf-8"))
        assert report["status"] == status
    for name in ("agentview.mp4", "wrist.mp4"):
        assert (tool.OUT / "old_nominal_videos" / name).is_file()
        assert not (tool.OUT / "old_nominal_videos" / f"{name}.partial").exists()


def test_v4_v5_artifacts_are_recoverable_and_closeout_propagates_reason() -> None:
    parent = json.loads((tool.V4_ROOT / "parent_result.json").read_text(encoding="utf-8"))
    child = json.loads((tool.V4_ROOT / "child_result.json").read_text(encoding="utf-8"))
    assert parent["category"] == "family_failure"
    assert parent["process_returncode"] == 0
    assert parent["worker_pid_match"] is True
    assert child["status"] == "failed_gate"
    assert tool._v4_first_error(parent)["reason"] == "failed_gates:contact_v3,same_demo_init_pose,source_goal,stall"
    with np.load(tool.V4_ROOT / "trajectory.npz", allow_pickle=False) as archive:
        assert archive["goal_trace"].dtype == np.int8
        assert set(np.unique(archive["goal_trace"]).tolist()) <= {-1, 0, 1}
    closeout = json.loads((tool.OUT / "v5_closeout.json").read_text(encoding="utf-8"))
    manifest = json.loads((tool.OUT / "v5_artifact_manifest.json").read_text(encoding="utf-8"))
    assert closeout["status"] == "track3_j5_executor_recovery_stopped_V5_hard_stop"
    assert closeout["upstream"]["first_error"]["reason"] == "failed_gates:contact_v3,same_demo_init_pose,source_goal,stall"
    paths = {entry["relative_path"] for entry in manifest["entries"]}
    assert "data/reports/axis_libero_30k/track3_j5_executor_recovery/resume_cursor.json" in paths
    assert "data/reports/axis_libero_30k/track3_j5_executor_recovery/supervisor_closeout.json" in paths
    assert not any(path.endswith(".partial") or path.endswith(".log") for path in paths)
