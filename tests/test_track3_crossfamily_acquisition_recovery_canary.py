from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_crossfamily_acquisition_recovery_canary as ac


def test_profile_lock_is_new_once_only_and_contract_is_frozen() -> None:
    lock = ac.profile_lock({"input": "fixture"})
    assert lock["profile_order"] == list(ac.PROFILE_ORDER)
    assert len(set(lock["profile_order"])) == 3
    assert not set(lock["profile_order"]) & ac.OLD_PROFILE_IDS
    assert lock["limits"]["total"] == 3
    assert lock["retry_allowed"] is False
    assert lock["contract"] == ac.CONTRACT


def test_required_contact_separates_telemetry_physical_contact_and_unentered() -> None:
    passed = ac.required_contact_gate("support_remove", ac._fixture_samples("support_remove", contact=True), stage_entered=True)
    failed = ac.required_contact_gate("support_remove", ac._fixture_samples("support_remove", contact=False), stage_entered=True)
    unentered = ac.required_contact_gate("support_remove", ac._fixture_samples("support_remove", contact=True), stage_entered=False)
    drift = ac.telemetry_schema_gate([{**ac._fixture_samples("support_remove", contact=True)[0], "known_mask": False}], stage_entered=True)
    assert passed["state"] == "passed"
    assert failed["state"] == "failed"
    assert unentered["state"] == "not_evaluated"
    assert drift["state"] == "failed"


def test_identity_and_resume_fail_closed_on_drift(tmp_path: Path) -> None:
    command = ["python", "-m", ac.MODULE, "--fixture"]
    expected = ac.identity_snapshot(command, "lock", environment={**ac.os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"})
    assert ac.identity_gate(expected, expected, command)["state"] == "passed"
    assert ac.identity_gate({**expected, "root_commit": "drift"}, expected, command)["state"] == "failed"
    trace = tmp_path / "trace.bin"
    trace.write_bytes(b"trace\n")
    payload = {"profile_id": "fixture", "category": "family_failure", "identity": expected, "trace": {"path": str(trace), "bytes": trace.stat().st_size, "sha256": ac.sha256_file(trace)}}
    payload["result_sha256"] = ac.result_content_hash(payload)
    assert ac.validate_resume_payload(payload, expected=expected, command=command, profile_id="fixture", category="family_failure")["state"] == "passed"
    trace.write_bytes(b"drift\n")
    assert ac.validate_resume_payload(payload, expected=expected, command=command, profile_id="fixture", category="family_failure")["state"] == "failed"


def test_old_er_diagnosis_is_read_only_and_recomputes_actual_lengths() -> None:
    rows = []
    for index, profile in enumerate(("er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh"), 1):
        rows.append(ac._old_trace_diagnosis(profile, ac.ER_ROOT / "runs" / f"{index:02d}_{profile}.json", ac.ER_ROOT / "runs/traces" / f"{profile}_demo0.npz"))
    assert [row["steps"] for row in rows] == [155, 149, 180]
    assert all(row["status"] == "passed" for row in rows)
    assert all(row["historical_raw_unchanged"] for row in rows)
    assert all(row["contact"]["pairs"]["state"] == "not_evaluated" for row in rows)
    assert rows[0]["saturation_windows"]


def test_typed_trace_rejects_object_and_nonfinite_values(tmp_path: Path) -> None:
    good = tmp_path / "good.npz"
    np.savez(good, values=np.asarray([1.0, 2.0], dtype=np.float32))
    assert ac._typed_trace_validation(good)["status"] == "passed"
    bad = tmp_path / "bad.npz"
    np.savez(bad, values=np.asarray([np.nan], dtype=np.float32))
    assert ac._typed_trace_validation(bad)["status"] == "failed"


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    try:
        ac.strict_json(path)
    except ac.DuplicateKeyError:
        pass
    else:
        raise AssertionError("duplicate key was accepted")


def test_current_plan_is_the_only_active_plan() -> None:
    plan = Path(ac.PLAN)
    assert not plan.is_file()
    active = Path(ac.ROOT) / "docs/AXIS_PI05_TRACK3_OPERATIONS_O1_O2_RED_MUG_NATIVE_MUJOCO_PLAN.md"
    archived_br = Path(ac.ROOT) / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md"
    archived = Path(ac.ROOT) / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_PLAN.md"
    archived_audit = Path(ac.ROOT) / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_INDEPENDENT_AUDIT_PLAN.md"
    assert active.is_file()
    assert archived_br.is_file()
    assert archived.is_file()
    assert archived_audit.is_file()
    text = active.read_text(encoding="utf-8")
    assert "O1/O2" in text
    assert "388-step red-mug" in text
    assert "native MuJoCo" in text
