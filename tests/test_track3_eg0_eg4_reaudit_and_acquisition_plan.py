from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_eg0_eg4_reaudit_20260906/summary.json"
EG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_remediation"
PLAN = "docs/AXIS_PI05_TRACK3_CLEAN_LAUNCH_AND_DIVERSIFIED_CANARY_RECOVERY_INDEPENDENT_AUDIT_PLAN.md"
ARCHIVED_AC_PLAN = "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_PLAN.md"
ARCHIVED_AC_AUDIT_PLAN = "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_INDEPENDENT_AUDIT_PLAN.md"
ARCHIVED_BR_PLAN = "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ACQUISITION_GATE_AND_ATTEMPT_LIFECYCLE_REMEDIATION_PLAN.md"


def strict_json(path: Path) -> dict:
    def reject(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_eg_reaudit_has_one_approved_verdict_and_exact_current_pointer() -> None:
    audit = strict_json(SUMMARY)
    pointer = strict_json(EG_ROOT / "latest_attempt.json")
    assert audit["status"] == audit["classification"] == "approved"
    assert pointer["attempt_id"] == 5
    assert sha256(EG_ROOT / "attempts/attempt_0005/final_validation.json") == pointer["target_final_validation_sha256"]
    assert sha256(EG_ROOT / "attempts/attempt_0005/checkout_stable_closure.json") == pointer["target_closure_sha256"]


def test_shared_root_attempt3_is_historical_and_root_attempt5_is_current() -> None:
    audit = strict_json(SUMMARY)
    shared = strict_json(EG_ROOT / "resume_cursor.json")
    current = strict_json(EG_ROOT / "root_attempt_0005_resume_cursor.json")
    closure = strict_json(EG_ROOT / "root_attempt_0005_checkout_stable_closure.json")
    assert shared["attempt_id"] == audit["cursor_resolution"]["shared_root_cursor_is_historical_attempt"] == 3
    assert current["attempt_id"] == 5 and current["active"] is False
    assert closure["status"] == "passed"


def test_lifecycle_releases_exactly_one_acquisition_plan() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_clean_launch_and_diversified_canary_recovery_completed_hard_stop_pending_independent_audit"
    assert handoff["next_plan"] == PLAN
    assert handoff["next_phase_only"] == "independent_audit_only_then_hard_stop"
    assert (ROOT / PLAN).is_file()
    assert (ROOT / ARCHIVED_AC_PLAN).is_file()
    assert (ROOT / ARCHIVED_AC_AUDIT_PLAN).is_file()
    assert (ROOT / ARCHIVED_BR_PLAN).is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md").exists()
    assert (ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md").is_file()


def test_machine_latest_commits_name_actual_eg_boundaries() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["latest_audited_execution_commit"] == "474d36d7e36aa3f428c8fff3fbf38b8d5db03f05"
    assert handoff["latest_audit_input_commit"] == "474d36d7e36aa3f428c8fff3fbf38b8d5db03f05"
    assert handoff["latest_execution_root_commit"] == "ec1f3f4f99aa8d74ddc45ac454914a54b99f1c4a"


def test_acquisition_authority_is_three_cpu_identities_without_promotion() -> None:
    handoff = strict_json(HANDOFF)
    auth = handoff["track3_crossfamily_acquisition_recovery_canary_authorization"]
    assert auth["plan"] == ARCHIVED_AC_PLAN
    assert auth["status"] == "completed_hard_stop_audited_remediation_required"
    assert auth["next_plan"] == ARCHIVED_BR_PLAN
    assert auth["new_cpu_physics_identities_max"] == 3
    assert auth["new_cpu_physics_identities_per_family_max"] == 1
    assert auth["family_failure_policy"] == "continue_other_frozen_families"
    assert auth["shared_infrastructure_failure_policy"] == "sticky_global_hard_stop"
    for key in ("old_identity_rerun", "gpu0", "gpu1", "gpu2", "gpu3", "official_gpu", "teleop", "generation", "replay", "accepted_data", "training", "norm", "fast_export", "new_candidate_zip", "upload", "submission"):
        assert auth["authorization"][key] is False
