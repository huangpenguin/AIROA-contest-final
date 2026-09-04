from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_overnight_autogen_lora_and_zip as tool


def _valid_record() -> dict[str, object]:
    return {
        "seed": 3100,
        "parent_pid": 101,
        "child_pid": 202,
        "child_result_pid": 202,
        "physical_gpu_uuid": "GPU-test",
        "cuda_visible_devices": "3",
        "start_snapshot": {},
        "end_snapshot": {},
        "peak_memory_mib": 1.0,
        "parent_poll_count": 1,
        "parent_poll_interval_max_ms": 1.0,
        "deadline_seconds": 300.0,
        "term_grace_seconds": 10.0,
        "process_returncode": 0,
        "timed_out": False,
    }


def test_provenance_gate_requires_every_parent_observation() -> None:
    valid = tool.gpu_provenance_gate(_valid_record())
    assert valid["state"] == "passed"
    for field in tool.PROVENANCE_REQUIRED_FIELDS:
        record = _valid_record()
        record.pop(field)
        assert tool.gpu_provenance_gate(record)["state"] == "failed"


def test_parent_assembly_is_fail_closed_for_name_error_pid_and_worker_exception() -> None:
    child = {"status": "passed", "child_pid": 202, "seed": 3100}
    assert tool.assemble_parent_result(child, _valid_record())["status"] == "passed"
    assert tool.assemble_parent_result(child, {**_valid_record(), "child_result_pid": 203})["status"] == "failed_closed"
    assert tool.assemble_parent_result(child, _valid_record(), gate_fn=tool.gate_name_error)["status"] == "failed_closed"
    assert tool.assemble_parent_result({"status": "worker_exception", "child_pid": 202}, _valid_record())["status"] == "failed_closed"
    assert tool.assemble_parent_result("malformed", _valid_record())["status"] == "failed_closed"


def test_o0_lock_records_historical_missing_definition_and_passes() -> None:
    report = json.loads((tool.OUT_ROOT / "o0_release_lock.json").read_text(encoding="utf-8"))
    assert report["status"] == "track3_overnight_o0_passed"
    assert report["parent_symbol_fix"]["historical_missing_definition"] is True
    assert report["parent_symbol_fix"]["fixed_definition_present"] is True
    assert report["old_artifacts"]["entry_count"] == 17
    assert report["new_artifacts"]["entry_count"] == 33
    assert report["counts"] == {"accepted": 0, "exact": 0, "pending": 0}


def test_o1_cpu_contract_report_passes_without_simulator_or_gpu() -> None:
    report = json.loads((tool.OUT_ROOT / "o1_provenance_contract.json").read_text(encoding="utf-8"))
    assert report["status"] == "track3_overnight_o1_provenance_watchdog_passed"
    assert report["simulator_started"] is False
    assert report["gpu_started"] is False
    assert all(report["tests"].values())


def test_o2_camera_schema_is_strict_and_no_pickle(tmp_path: Path) -> None:
    path = tmp_path / "camera.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, agentview=np.zeros((41, 128, 128, 3), dtype=np.uint8), wrist=np.ones((41, 128, 128, 3), dtype=np.uint8))
    result = tool._validate_camera_npz(path)
    assert result["state"] == "passed"
    assert result["allow_pickle"] is False
    bad = tmp_path / "bad.npz"
    with bad.open("wb") as handle:
        np.savez_compressed(handle, agentview=np.zeros((40, 128, 128, 3), dtype=np.uint8), wrist=np.zeros((41, 128, 128, 3), dtype=np.uint8))
    assert tool._validate_camera_npz(bad)["state"] == "failed"
