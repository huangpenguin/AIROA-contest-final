from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05 import track3_provenance_jitter_recovery_and_overnight_continuation as tool


def test_latency_is_diagnostic_and_heartbeat_is_gate() -> None:
    for latency in (50.0, 148.46, 600.0):
        result = tool.telemetry_v2_gate(tool._valid_telemetry_fixture(latency, 1000.0))
        assert result["state"] == "passed"
    assert "gpu_query_p99_over_500ms" in tool.telemetry_v2_gate(tool._valid_telemetry_fixture(600.0, 1000.0))["warnings"]
    assert tool.telemetry_v2_gate(tool._valid_telemetry_fixture(50.0, 1999.0))["state"] == "passed"
    assert tool.telemetry_v2_gate(tool._valid_telemetry_fixture(50.0, 2001.0))["state"] == "failed"


def test_telemetry_v2_fails_closed_for_pid_query_timeout_and_sparse_coverage() -> None:
    base = tool._valid_telemetry_fixture(50.0, 1000.0)
    assert tool.telemetry_v2_gate({**base, "child_result_pid": 3})["state"] == "failed"
    assert tool.telemetry_v2_gate({**base, "start_snapshot": {"returncode": 1, "process_query_returncode": 0}})["state"] == "failed"
    assert tool.telemetry_v2_gate({**base, "timed_out": True})["state"] == "failed"
    assert tool.telemetry_v2_gate({**base, "sampling_coverage_count": 2})["state"] == "failed"


def test_j0_j1_j2_persisted_reports_are_passed_and_task10_retry_forbidden() -> None:
    root = tool.OUT_ROOT
    j0 = json.loads((root / "j0_input_lock.json").read_text(encoding="utf-8"))
    j1 = json.loads((root / "j1_telemetry_v2_contract.json").read_text(encoding="utf-8"))
    j2 = json.loads((root / "j2_derived_attempt3_release.json").read_text(encoding="utf-8"))
    assert j0["status"] == "track3_jitter_j0_passed"
    assert j1["status"] == "track3_jitter_j1_passed"
    assert j2["status"] == "track3_jitter_j2_support_remove_released"
    assert j2["release"]["task10_attempt4"] == "forbidden"
    assert j2["source_attempt"]["historical_failure_unchanged"] is True


def test_csv_probe_is_lf_only_and_no_partial() -> None:
    path = tool.OUT_ROOT / "j1_lf_manifest_probe.csv"
    assert b"\r\n" not in path.read_bytes()
    assert not Path(str(path) + ".partial").exists()

