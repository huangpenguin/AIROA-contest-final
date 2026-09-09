from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.axis_pi05 import track3_z5_derived_release_and_conservative_continuation as tool


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


def test_conservative_profile_is_exact_and_no_nominal_cli(tmp_path: Path) -> None:
    assert tool.CONSERVATIVE_PROFILE["name"] == "object_frame_conservative"
    assert tool.CONSERVATIVE_PROFILE["translation_clip_m"] == 0.025
    assert tool.CONSERVATIVE_PROFILE["rotation_clip_rad"] == 0.06
    assert tool.CONSERVATIVE_PROFILE["stage_budgets"] == [200, 160, 50, 200, 260, 160, 50, 120]
    result = subprocess.run([sys.executable, "-m", tool.MODULE, "--aa5-worker", "--profile", "object_frame_nominal", "--result-output", str(tmp_path / "result.json")], cwd=ROOT, capture_output=True, check=False)
    assert result.returncode == 2


def test_aa3_fixture_return_code_classification(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    command = [sys.executable, "-m", tool.MODULE, "--aa3-worker", "--status", "failed_gate", "--result-output", str(result_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_gate"
    assert payload["physics_steps"] == 0


def test_aa3_worker_exception_is_nonzero(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result = subprocess.run([sys.executable, "-m", tool.MODULE, "--aa3-worker", "--status", "worker_exception", "--result-output", str(result_path)], cwd=ROOT, capture_output=True, check=False)
    assert result.returncode == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "worker_exception"


def test_identity_gate_rejects_drift() -> None:
    frozen = {"root_commit": "a", "tool_sha256": "b", "openpi_commit": "c", "libero_plus_porcelain": []}
    observed = {**frozen, "command": ["worker"], "cuda_visible_devices": "", "gpu_policy": "cpu_no_renderer", "source_porcelain": [], "libero_plus_porcelain": []}
    assert tool.identity_gate(observed, frozen, command=["worker"])["state"] == "passed"
    assert tool.identity_gate({**observed, "root_commit": "drift"}, frozen, command=["worker"])["state"] == "failed"


def test_aa_audit_archives_plan_and_releases_tonight_zip_plan() -> None:
    handoff = tool.strict_json(
        ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
    )
    assert handoff["status"] == "track3_operations_o1_2_user_recorded_over_budget_hard_stop"
    assert handoff["next_plan"] is None
    assert handoff["next_phase_only"] == "new_reviewed_plan_required"
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md").is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md").is_file()

    audit = handoff["track3_aa0_aa14_tonight_zip_audit"]
    assert audit["status"] == "remediation_required"
    assert (ROOT / audit["audit_report"]).is_file()
    assert (ROOT / audit["audit_summary"]).is_file()
    assert audit["verified"]["old_z_artifacts_bytes_sha_exact"] == "36/36"
    assert audit["verified"]["aa_namespace_file_closure"] == "31+4_excluded=35/35"
    assert audit["verified"]["aa5_category"] == "family_failure"
    assert audit["verified"]["exact_pending_accepted"] == [0, 0, 0]
    assert audit["findings"]["aa6_to_aa13_cli_implemented"] is False
    assert audit["findings"]["family_isolation_execution_implemented"] is False
    assert audit["findings"]["new_track3_zip_created"] is False

    old = handoff["track3_z5_derived_release_and_conservative_continuation_authorization"]
    assert old["status"] == "track3_z5_derived_release_and_conservative_continuation_completed_and_audited"
    assert old["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md"

    authorization = handoff["track3_tonight_zip_and_crossfamily_recovery_authorization"]
    assert authorization["status"] == "track3_tonight_zip_and_crossfamily_recovery_completed_and_audited_remediation_required"
    assert authorization["plan"] == "docs/archive/axis_pi05/AXIS_PI05_TRACK3_TONIGHT_ZIP_AND_CROSSFAMILY_RECOVERY_PLAN.md"
    assert authorization["next_phase_only"] == "superseded_by_track3_tz_evidence_and_executor_contract_remediation_plan"
    scope = authorization["authorization"]
    assert scope["materialize_revalidated_r6_fallback_zip"] is True
    assert scope["family_failure_isolation"] is True
    assert scope["support_contact_repair_cpu_identities_max"] == 3
    assert scope["total_train_generations_max"] == 6
    assert scope["total_fresh_replays_max"] == 18
    assert scope["fallback_zip_required_by_deadline"] is True
    assert scope["gpu0"] is False
    assert scope["automatic_upload_or_submission"] is False

    execution = handoff["track3_tonight_zip_and_crossfamily_recovery_execution"]
    assert execution["phase_status"]["TZ2"] == "7_attempts_consumed_not_compliant_completion"
    assert execution["phase_status"]["TZ3"] == "not_evaluated"
    assert execution["cpu_matrix"]["executed"] == 7
    assert execution["accepted_reverse_demos"] == [0, 0, 0]
    assert execution["fallback"]["readiness"] == "fallback_no_track3_improvement"
