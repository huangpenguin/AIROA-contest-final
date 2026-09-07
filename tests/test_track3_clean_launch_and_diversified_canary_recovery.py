from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_clean_launch_and_diversified_canary_recovery as cr


def test_identity_gate_rejects_dirty_expected_and_observed() -> None:
    dirty = {"clean_root": False, "root_porcelain": [" M dirty.py"]}
    result = cr.identity_gate(dirty, dirty, ["fixture"])
    assert result["state"] == "infrastructure_failure"
    assert "clean_root" in result["mismatches"]


def test_partial_provenance_constants_are_distinct_and_exact() -> None:
    assert len(cr.PARTIAL_BYTES) == 43
    assert len(cr.SYNTHETIC_BYTES) == 39
    assert cr.PARTIAL_BYTES != cr.SYNTHETIC_BYTES
    assert hashlib.sha256(cr.PARTIAL_BYTES).hexdigest() != hashlib.sha256(cr.SYNTHETIC_BYTES).hexdigest()


def test_telemetry_unknown_is_infrastructure_failure() -> None:
    result = cr.telemetry_schema_gate([{"telemetry_complete": False, "known_mask": False}], stage_entered=True)
    assert result["state"] == "infrastructure_failure"


def test_unentered_stages_are_not_evaluated_and_late_contact_is_excluded() -> None:
    stages = cr.stage_gate_graph("support_remove", ["approach_object", "grasp_object"])
    assert stages["descend_object"]["state"] == "not_evaluated"
    samples = [
        {"telemetry_complete": True, "known_mask": True, "contacts": []},
        {"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]},
        {"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]},
        {"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]},
        {"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]},
    ]
    result = cr.required_contact_gate("support_remove", samples, stage_entered=True, barrier_index=0)
    assert result["state"] == "failed"
    assert result["contact_window"] == 0
    assert result["late_contact_excluded"] is True


def test_normalized_actions_are_not_double_normalized() -> None:
    result = cr.normalized_action_metrics(
        np.asarray([[-1.0, 0.5, 0.0, 0.25, 0.0, -0.1, 1.0]], dtype=np.float32),
        physical_translation_m=np.asarray([[0.02, 0.0, 0.0]], dtype=np.float32),
    )
    assert result["normalized_action_abs_max"] == 1.0
    assert np.isclose(result["physical_translation_m_l2_max"], 0.02)


def test_profile_lock_has_six_distinct_new_identities_and_strategies() -> None:
    lock = cr.profile_lock()
    assert len(lock["profiles"]) == 6
    assert len({row["command_hash"] for row in lock["profiles"]}) == 6
    assert not set(lock["profile_order"]) & cr.OLD_IDS
    assert {row["strategy"] for row in lock["profiles"]} == {
        "object_frame_contact_aligned_v1",
        "waypoint_guarded_microstep_v1",
    }


def test_action_hash_deduplication_is_checked_separately_from_trace_bytes() -> None:
    historical = cr._historical_action_hashes()
    current = set()
    for path in (cr.CR_ROOT / "attempts/attempt_0004/runs/traces").glob("*.npz"):
        current.add(cr.sha256_bytes(cr._load_trace(path)["actions"].tobytes()))
    assert current
    assert not current & historical


def test_manifest_is_sealed_after_nested_producers_exist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cr, "ROOT", tmp_path)
    attempt = tmp_path / "attempts" / "attempt_0001"
    nested = attempt / "canaries" / "child.stderr.log"
    nested.parent.mkdir(parents=True)
    nested.write_text("producer", encoding="utf-8")
    manifest = cr._make_manifest(attempt)
    paths = {entry["path"] for entry in manifest["entries"]}
    assert "attempts/attempt_0001/canaries/child.stderr.log" in paths
    assert manifest["sealed_after_all_producers"] is True
    assert all(not path.endswith(".partial") for path in paths)


def test_finalizer_excludes_only_its_own_late_artifacts() -> None:
    excluded = set(cr._make_manifest(Path(cr.ROOT / "attempts" / "attempt_9999"))["excluded_exact_relative_paths"])
    assert excluded == {
        "attempts/attempt_9999/producer_manifest.json",
        "attempts/attempt_9999/outer_hash.json",
        "attempts/attempt_9999/final_validation.json",
        "attempts/attempt_9999/checkout_stable_closure.json",
    }
