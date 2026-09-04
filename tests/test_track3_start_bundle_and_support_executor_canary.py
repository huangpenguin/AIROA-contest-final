from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_start_bundle_and_support_executor_canary as tool


def test_strict_json_rejects_duplicate_key(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    try:
        tool.strict_json_load(path)
    except ValueError as exc:
        assert "duplicate_json_key" in str(exc)
    else:
        raise AssertionError("duplicate JSON key was accepted")


def test_state_bundle_is_no_pickle_and_detects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "states.npz"
    array = np.arange(4, dtype=np.float64)
    with path.open("wb") as handle:
        np.savez_compressed(handle, task10_demo0=array)
    expected = {"task10_demo0": {"shape": [4], "sha256": tool.sha256_bytes(array.tobytes())}}
    assert tool.validate_state_bundle(path, expected)["state"] == "passed"
    tampered = tmp_path / "tampered.npz"
    with tampered.open("wb") as handle:
        np.savez_compressed(handle, task10_demo0=array + 1.0)
    assert tool.validate_state_bundle(tampered, expected)["state"] == "failed"


def test_artifact_manifest_rejects_absolute_and_hash_mismatch() -> None:
    path = tool.OUT_ROOT / "p1_validator_contract.json"
    valid = tool.artifact_entry(path, source_commit="test", schema="json")
    assert tool.validate_artifact_manifest([valid])["state"] == "passed"
    bad = dict(valid)
    bad["sha256"] = "0" * 64
    assert tool.validate_artifact_manifest([bad])["state"] == "failed"
    absolute = dict(valid)
    absolute["relative_path"] = str(path.resolve())
    assert tool.validate_artifact_manifest([absolute])["state"] == "failed"


def test_full_window_contact_gate_includes_burn_in() -> None:
    samples = [
        {
            "step": step,
            "contact_matrix": {
                "complete": True,
                "contacts": [
                    {
                        "body1_name": "akita_black_bowl_1_main",
                        "body2_name": "plate_1_main",
                        "geom1_name": "bowl_geom",
                        "geom2_name": "plate_geom",
                        "distance_m": 0.0,
                    }
                ],
            },
        }
        for step in range(41)
    ]
    assert tool.full_window_contact_gate(10, samples)["state"] == "passed"
    samples[3]["contact_matrix"]["contacts"].append(
        {
            "body1_name": "robot0_robot",
            "body2_name": "akita_black_bowl_1_main",
            "geom1_name": "finger",
            "geom2_name": "bowl_geom",
            "distance_m": 0.0,
        }
    )
    failed = tool.full_window_contact_gate(10, samples)
    assert failed["state"] == "failed"
    assert failed["details"][3]["counts"]["demonstrator_contact"] == 1


def test_gpu_provenance_and_worker_exception_fail_closed() -> None:
    assert tool.gpu_provenance_gate({"seed": 1})["state"] == "failed"
    complete = {
        "seed": 1,
        "child_pid": 123,
        "physical_gpu_uuid": "GPU-test",
        "cuda_visible_devices": "3",
        "start_snapshot": {},
        "end_snapshot": {},
        "peak_memory_mib": 0.0,
    }
    assert tool.gpu_provenance_gate(complete)["state"] == "passed"
    assert tool.worker_exit_code("worker_exception") != 0
    assert tool.worker_exit_code("failed_gate") == 0


def test_quaternion_log_and_se3_action_clip() -> None:
    error = tool.quaternion_log_error((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    assert np.allclose(error, np.zeros(3))
    action, detail = tool.osc_pose_action((2.0, 0.0, 0.0), (0.0, 0.0, 2.0), 1.0, translation_max=0.05, rotation_max=0.5)
    assert action.shape == (7,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert action[-1] == 1.0
    assert detail["translation_clipped"] is True
    assert detail["rotation_clipped"] is True


def test_atomic_failure_asset_has_no_partial(tmp_path: Path) -> None:
    target = tmp_path / "diagnostics" / "failure.json"
    artifact = tool.atomic_failure_asset(target, {"status": "failed_gate"})
    assert artifact["atomic"] is True
    assert artifact["partial_residue"] is False
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "failed_gate"


def test_p0_p1_p2_p3_evidence_is_structured() -> None:
    root = tool.OUT_ROOT
    assert json.loads((root / "p0_release_lock.json").read_text(encoding="utf-8"))["status"] == "track3_start_bundle_p0_passed"
    assert json.loads((root / "p1_validator_contract.json").read_text(encoding="utf-8"))["status"] == "track3_start_bundle_p1_contract_passed"
    p2 = json.loads((root / "p2_summary.json").read_text(encoding="utf-8"))
    assert p2["status"] == "track3_start_state_bundle_validated_9_of_9"
    p3 = json.loads((root / "p3_summary.json").read_text(encoding="utf-8"))
    assert p3["status"] == "track3_start_bundle_p3_failed"
    assert p3["camera_canary_count"] == 1
    assert not (tool.PENDING_ROOT / "task10_demo0").exists()
