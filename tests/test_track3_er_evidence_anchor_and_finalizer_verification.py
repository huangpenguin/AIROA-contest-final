from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.axis_pi05 import track3_er_evidence_anchor_and_finalizer_verification as tool


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(tool.DuplicateKeyError):
        tool.strict_json(path)


def test_identity_gate_checks_all_parent_snapshot_fields() -> None:
    command = [sys.executable, "-m", tool.MODULE, "--ef-fixture-child", "--mode", "family_failure"]
    expected = tool.identity_snapshot(command, profile_sha=tool.profile_lock_sha())
    assert tool.identity_gate(dict(expected), expected, command)["state"] == "passed"
    for field, value in (("root_porcelain", [" M x"]), ("openpi_porcelain", [" M y"]), ("libero_plus_porcelain", ["?? z"]), ("command", ["different"])):
        observed = dict(expected)
        observed[field] = value
        if field != "command":
            observed[f"{field}_sha256"] = tool.porcelain_digest(value)
        assert tool.identity_gate(observed, expected, command)["state"] == "failed"


def test_resume_requires_identity_terminal_and_trace(tmp_path: Path) -> None:
    command = [sys.executable, "-m", tool.MODULE, "--fixture"]
    expected = tool.identity_snapshot(command, profile_sha=tool.profile_lock_sha())
    trace = tmp_path / "trace.bin"
    trace.write_bytes(b"trace\n")
    payload = {"profile_id": "ef_fixture", "terminal_category": "family_failure", "identity": expected, "trace": {"path": str(trace), "bytes": trace.stat().st_size, "sha256": tool.sha256_file(trace)}}
    payload["result_sha256"] = tool.result_content_hash(payload)
    assert tool.validate_resume_payload(payload, expected_identity=expected, command=command, expected_profile_id="ef_fixture", expected_terminal_category="family_failure")["state"] == "passed"
    trace.write_bytes(b"drift\n")
    assert tool.validate_resume_payload(payload, expected_identity=expected, command=command, expected_profile_id="ef_fixture", expected_terminal_category="family_failure")["state"] == "failed"


def test_temp_git_repo_dirty_baseline_is_observable(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "ef@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "EF"], cwd=tmp_path, check=True)
    (tmp_path / "base").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "base"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    clean = tool.porcelain(tmp_path)
    (tmp_path / "drift").write_text("drift\n", encoding="utf-8")
    dirty = tool.porcelain(tmp_path)
    assert clean == []
    assert dirty != clean
    assert tool.porcelain_digest(dirty) != tool.porcelain_digest(clean)


def test_fixture_child_real_path_and_sticky_stop(tmp_path: Path) -> None:
    original = tool.OUT_ROOT
    try:
        tool.OUT_ROOT = tmp_path / "ef"
        result = tool.run_ef2()
        assert result["status"] == "passed"
        assert result["normal_family_failure"]["payload"]["status"] == "family_failure"
        assert result["timeout_term_grace_kill"]["kill_sent"] is True
        assert result["sticky_stop"]["child_started"] is False
    finally:
        tool.OUT_ROOT = original


def test_contact_gate_separates_telemetry_and_physical_contact() -> None:
    samples = [{"step": i, "known_mask": True, "telemetry_complete": True, "contacts": [{"roles": ["demonstrator", "object"]}]} for i in range(2)]
    no_contact = [{"step": i, "known_mask": True, "telemetry_complete": True, "contacts": []} for i in range(2)]
    assert tool.telemetry_schema_gate(samples, stage_entered=True)["state"] == "passed"
    assert tool.required_contact_gate("support_remove", samples, stage_entered=True)["state"] == "passed"
    assert tool.telemetry_schema_gate(no_contact, stage_entered=True)["state"] == "passed"
    assert tool.required_contact_gate("support_remove", no_contact, stage_entered=True)["state"] == "failed"
    assert tool.required_contact_gate("support_remove", samples, stage_entered=False)["state"] == "not_evaluated"


def test_finalizer_status_contract_and_no_physics_authorization() -> None:
    plan = (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").read_text(encoding="utf-8")
    assert "任何新physics identity或旧identity重跑" in plan
    assert "GPU0–3" in plan
    assert tool.FALLBACK_BYTES == 9_383_571_677
    assert tool.FALLBACK_SHA.startswith("95415a")
