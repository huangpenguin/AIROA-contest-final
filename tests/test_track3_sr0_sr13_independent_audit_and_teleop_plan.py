from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / "data/reports/axis_libero_30k/track3_support_runtime_gate_remediation"
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_sr0_sr13_independent_audit_20260907/summary.json"
ANCHOR = ROOT / "data/reports/axis_libero_30k/track3_sr0_sr13_independent_audit_20260907/raw_evidence_anchor.json"
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
PLAN = "docs/AXIS_PI05_TRACK3_SR_CLOSURE_AND_TASK10_USER_TELEOP_GOLD_CANARY_PLAN.md"
ARCHIVED_AUDIT_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_INDEPENDENT_AUDIT_PLAN.md"


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


def test_audit_has_one_remediation_verdict_and_frozen_commits() -> None:
    audit = strict_json(AUDIT)
    assert audit["status"] == audit["classification"] == "remediation_required"
    assert audit["audit_input_commit"] == "64f278c7f59dc70f17770e1980522d0313ccda27"
    assert audit["clean_physics_source_commit"] == "5c5c74641ba39e41709e2c51ce576e7c065060fc"
    assert audit["nested_openpi_commit"] == "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"


def test_two_raw_traces_are_typed_family_failures_at_dwell() -> None:
    for name, steps, saturation in (
        ("support_se3_microstep_primary_v1", 96, 66),
        ("support_se3_microstep_lateral_first_v2", 112, 83),
    ):
        directory = SR / "attempts/attempt_0004/canary" / name
        child = strict_json(directory / "child_result.json")
        assert child["category"] == child["status"] == "family_failure"
        assert child["physics_steps"] == steps
        with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
            assert archive["actions"].shape == (steps, 7)
            assert archive["states"].shape == (steps, 79)
            assert np.isfinite(archive["actions"]).all()
            assert np.isfinite(archive["states"]).all()
            dwell = np.flatnonzero(archive["phase"] == "grasp_dwell")
            assert dwell.size == 4
            assert archive["contact_telemetry_complete"][dwell].all()
            assert not archive["required_contact"][dwell].any()
            assert not archive["demonstrator_contact"][dwell].any()
            assert int(archive["action_saturated"].sum()) == saturation


def test_original_finalizer_is_honestly_recorded_as_stale() -> None:
    attempt = SR / "attempts/attempt_0004"
    manifest = strict_json(attempt / "final_manifest.json")
    excluded = {"producer_manifest.json", "final_manifest.json", "outer_hash.json", "closure.json"}
    actual = {
        path.relative_to(attempt).as_posix()
        for path in attempt.rglob("*")
        if path.is_file() and path.name not in excluded and not path.name.endswith(".partial")
    }
    expected = {row["relative_path"] for row in manifest["entries"]}
    assert len(expected) == 15 and len(actual) == 18
    assert actual - expected == {
        "sr8_route.json",
        "sr12_readiness/sr12_code_only_readiness.json",
        "sr13_closeout.json",
    }
    assert strict_json(attempt / "sr13_closeout.json")["not_evaluated"] == []
    assert strict_json(SR / "summary.json")["not_evaluated"] == ["SR9", "SR10", "SR11"]


def test_claimed_alternative_has_the_same_actual_stage_topology() -> None:
    canary = SR / "attempts/attempt_0004/canary"
    phase_orders = []
    for directory in sorted(canary.iterdir()):
        child = strict_json(directory / "child_result.json")
        phase_orders.append([row["stage"] for row in child["stage_results"]])
    assert phase_orders[0] == phase_orders[1]
    assert phase_orders[0] == ["precontact_xy", "precontact_z", "orientation_align", "contact_descent", "grasp_dwell"]


def test_old_teleop_recorder_is_not_released_as_gold_ready() -> None:
    source = (ROOT / "tools/axis_pi05/track3_public_teleop.py").read_text(encoding="utf-8")
    findings = strict_json(AUDIT)["findings"]
    assert "default_gripper: float = 1.0" in source
    assert "contact_matrix" not in source
    assert findings["teleop_default_gripper_and_explicit_state_not_ready"] is True
    assert findings["teleop_lacks_per_step_gold_gate_telemetry"] is True


def test_raw_anchor_is_exact_and_git_persisted_without_rewriting() -> None:
    anchor = strict_json(ANCHOR)
    assert anchor["entry_count"] == len(anchor["entries"]) == 11
    assert anchor["git_tracked_at_audit_input"] == 0
    assert anchor["original_bytes_modified"] is False
    for row in anchor["entries"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert sha256(path) == row["sha256"]
        assert subprocess.run(["git", "ls-files", "--error-unmatch", row["path"]], cwd=ROOT, capture_output=True).returncode == 0


def test_lifecycle_has_one_new_active_plan_and_archived_old_audit_plan() -> None:
    handoff = strict_json(HANDOFF)
    assert handoff["status"] == "track3_sr_closure_and_task10_user_teleop_gold_canary_planned"
    assert handoff["next_plan"] == PLAN
    assert handoff["next_phase_only"] == "UT0_to_UT9_then_hard_stop"
    assert (ROOT / PLAN).is_file()
    assert ARCHIVED_AUDIT_PLAN.is_file()
    assert not (ROOT / "docs/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_INDEPENDENT_AUDIT_PLAN.md").exists()


def test_new_authority_requires_human_and_keeps_training_submission_closed() -> None:
    auth = strict_json(HANDOFF)["track3_sr_closure_and_task10_user_teleop_gold_canary_authorization"]
    assert auth["new_automatic_physics_identities_max"] == 0
    assert auth["human_teleop_recorded_candidates_max"] == 1
    assert auth["human_teleop_requires_explicit_user_online_takeover"] is True
    assert auth["gpu3_exact_action_replays_per_machine_success_max"] == 3
    assert auth["success_disposition"] == "pending_gold_review_only"
    permissions = auth["authorization"]
    for key in (
        "new_automatic_physics",
        "containment_or_articulation_physics",
        "old_identity_rerun",
        "gpu0",
        "gpu1",
        "gpu2",
        "official_gpu",
        "unattended_teleop",
        "accepted_promotion",
        "scale_generation",
        "norm",
        "training",
        "fast_export",
        "new_candidate_zip",
        "upload",
        "submission",
    ):
        assert permissions[key] is False
