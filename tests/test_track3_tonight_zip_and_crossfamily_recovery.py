from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.axis_pi05 import track3_tonight_zip_and_crossfamily_recovery as tool


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    try:
        tool.strict_json(path)
    except tool.DuplicateKeyError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")


def test_profile_lock_has_exact_new_family_budget() -> None:
    lock = tool.profile_lock()
    assert lock["profile_order"] == list(tool.PROFILE_ORDER)
    assert len(lock["profiles"]) == 7
    assert [row["family"] for row in lock["profiles"]] == [
        "support_remove",
        "support_remove",
        "support_remove",
        "containment_remove",
        "containment_remove",
        "articulation_close",
        "articulation_close",
    ]
    assert all(row["old_identity_rerun"] is False for row in lock["profiles"])
    assert lock["limits"] == {"support_remove": 3, "containment_remove": 2, "articulation_close": 2, "total": 7}


def test_cli_exposes_all_tonight_phases_and_workers(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, "-m", tool.MODULE, "--help"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    for flag in [*(f"--tz{i}" for i in range(10)), "--execute", "--cpu-worker", "--gpu3-worker", "--generation-worker", "--replay-worker"]:
        assert flag in result.stdout


def test_cli_contract_is_fail_closed() -> None:
    contract = tool.cli_contract()
    assert contract["missing_phase_cli"] is False
    assert contract["single_instance_supervisor"] is True
    assert contract["term_then_kill"] is True
    assert contract["return_code_contract"]["family_failure"] == 0
    assert contract["return_code_contract"]["infrastructure_failure"] == 1


def test_status_classification_is_family_scoped() -> None:
    assert tool._family_status({"status": "failed_gate"}) == "family_failure"
    assert tool._family_status({"status": "worker_exception"}) == "infrastructure_failure"
    assert tool._family_status({"status": "passed"}) == "passed"
