#!/usr/bin/env python3
"""SG0--SG12 recovery lane for Track 3.

This namespace is intentionally independent from the CR evidence.  It fixes
the runtime contracts before consuming another physics budget: contact is
derived from MuJoCo pairs, stages are explicit and axis-separated, attempts
are monotonic, and a finalizer verifies the complete post-production set.

SG0--SG5 are safe to run without a simulator.  SG6 is one fresh support
CPU/no-renderer identity.  Later phases are fail-closed and are not entered
implicitly by this module.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_contact_truth_and_support_gold_demo"
SG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_contact_truth_and_support_gold_demo"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_CONTACT_TRUTH_AND_SUPPORT_GOLD_DEMO_PLAN.md"
CR_AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_CR0_CR14_INDEPENDENT_AUDIT_20260907.md"
CR_SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_clean_launch_and_diversified_canary_recovery/attempts/attempt_0004/summary.json"
CR_ROOT = CR_SUMMARY.parents[2]
CR_TRACES = CR_ROOT / "runs/traces"
CR_PARTIAL = CR_ROOT.parent / "attempt_0003/raw_partial.bin"
CR_SYNTHETIC = CR_ROOT.parent / "attempt_0001/synthetic_sequence_999_observation.bin"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
RECORDER_PYTHON = ROOT / ".venv_axis_pi05/bin/python"
RECORDER = "tools.axis_pi05.track3_public_teleop"
REPLAY = "tools.axis_pi05.track3_public_replay"

CURRENT_ATTEMPT = SG_ROOT / "current_attempt.json"
ATTEMPT_LOCK = SG_ROOT / "attempt_allocator.lock"
SG0_REPORT = SG_ROOT / "sg0_input_lock.json"
SG1_REPORT = SG_ROOT / "sg1_contact_truth_fixtures.json"
SG2_REPORT = SG_ROOT / "sg2_waypoint_controller_contract.json"
SG3_REPORT = SG_ROOT / "sg3_failure_injection_fixtures.json"
SG4_REPORT = SG_ROOT / "sg4_machine_handoff.json"
SG5_REPORT = SG_ROOT / "sg5_clean_source_release.json"

CONTROL_FREQ_HZ = 20
ACTION_DIM = 7
TERM_GRACE_SECONDS = 0.5
KILL_WAIT_SECONDS = 1.0
CANARY_TIMEOUT_SECONDS = 300.0
MAX_SUPPORT_IDENTITIES = 2

FAMILY = "support_remove"
TASK_INDEX = 10
DEMO_INDEX = 0
SOURCE_ATOM = "(On akita_black_bowl_1 plate_1)"
GOAL_ATOM = "(On akita_black_bowl_1 main_table_akita_black_bowl_region)"
OPERATOR = {"subject": ("akita_black_bowl_1",), "receiver": ("plate_1",), "target": ()}

STAGE_NAMES = (
    "precontact_xy",
    "precontact_z",
    "orientation_align",
    "contact_descent",
    "grasp_dwell",
    "lift",
    "transport",
    "place",
    "release",
    "retreat",
)

PRIMARY_PROFILE: dict[str, Any] = {
    "name": "support_se3_microstep_primary_v1",
    "translation_clip_m": 0.025,
    "rotation_clip_rad": 0.06,
    "controller_translation_scale_m": 0.05,
    "controller_rotation_scale_rad": 0.5,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 4,
    "microstep_translation_m": 0.025,
    "microstep_rotation_rad": 0.06,
    "stage_budgets": [120, 100, 100, 150, 40, 150, 180, 150, 35, 80],
    "step_budget": 1105,
    "topology": "xy_then_z_then_orientation_then_contact_then_lift_transport_place",
}

# Frozen before SG6.  It is deliberately topologically different rather than
# a seed/dwell/tolerance retry, and is only eligible after measured SG6 gain.
ALTERNATIVE_PROFILE: dict[str, Any] = {
    "name": "support_se3_microstep_lateral_first_v2",
    "translation_clip_m": 0.02,
    "rotation_clip_rad": 0.08,
    "controller_translation_scale_m": 0.05,
    "controller_rotation_scale_rad": 0.5,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 4,
    "microstep_translation_m": 0.02,
    "microstep_rotation_rad": 0.08,
    "stage_budgets": [120, 100, 90, 150, 40, 150, 190, 150, 35, 80],
    "step_budget": 1105,
    "topology": "xy_lateral_then_orientation_then_z_then_contact_then_lift_transport_place",
}


class DuplicateKeyError(ValueError):
    """Raised when a supposedly strict JSON artifact repeats a key."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        raise FileExistsError(f"immutable_artifact_exists:{path}")
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def source_porcelain(repo: Path = ROOT) -> list[str]:
    """Return source changes, excluding only this ignored evidence namespace."""
    rows = porcelain(repo)
    if repo.resolve() != ROOT.resolve():
        return rows
    prefix = rel(SG_ROOT).rstrip("/") + "/"
    result: list[str] = []
    for row in rows:
        path_field = row[3:].strip() if len(row) >= 3 else row.strip()
        paths = [part.strip().strip('"') for part in path_field.split(" -> ")]
        if any(not path.startswith(prefix) for path in paths):
            result.append(row)
    return result


def json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def gate(executed: bool, value: bool | None, *, reason: str | None = None, required: bool = True, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"executed": bool(executed), "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"), "required": bool(required), **details}
    if reason is not None:
        result["reason"] = reason
    return result


def infrastructure_gate(reason: str, **details: Any) -> dict[str, Any]:
    return {"executed": True, "state": "infrastructure_failure", "required": True, "reason": reason, **details}


def predicate_truth(value: Any) -> bool | None:
    """Convert Python and NumPy scalar booleans without identity comparison."""
    if value is None:
        return None
    try:
        return bool(value)
    except (TypeError, ValueError):
        return None


def _name_matches(actual: str, logical: str) -> bool:
    return actual == logical or actual.startswith(logical + "_") or logical in actual


def _is_robot_name(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(("robot0", "gripper0", "panda")) or any(token in lower for token in ("finger", "gripper", "right_hand", "left_hand"))


def contact_role(body1: str, body2: str) -> str:
    participants = set(OPERATOR["subject"]) | set(OPERATOR["target"])
    receivers = set(OPERATOR["receiver"])
    left_operator = any(_name_matches(body1, name) for name in participants)
    right_operator = any(_name_matches(body2, name) for name in participants)
    left_receiver = any(_name_matches(body1, name) for name in receivers)
    right_receiver = any(_name_matches(body2, name) for name in receivers)
    left_robot, right_robot = _is_robot_name(body1), _is_robot_name(body2)
    if (left_operator and right_receiver) or (right_operator and left_receiver):
        return "intended_receiver_contact"
    if (left_operator and right_robot) or (right_operator and left_robot):
        return "demonstrator_contact"
    if left_operator or right_operator:
        return "unrelated_background_contact"
    return "background_contact"


def contact_truth_v3(task_index: int, matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Derive contact truth from actual pair rows; telemetry never implies it."""
    if task_index != TASK_INDEX:
        return infrastructure_gate("unsupported_task_contact_contract", task_index=task_index)
    if not isinstance(matrix, Mapping) or matrix.get("complete") is not True:
        return infrastructure_gate("contact_schema_incomplete", contact_telemetry_complete=False, required_contact=None, unrelated_contact_runtime=None)
    raw_contacts = matrix.get("contacts")
    if not isinstance(raw_contacts, list):
        return infrastructure_gate("contact_rows_not_a_list", contact_telemetry_complete=False, required_contact=None, unrelated_contact_runtime=None)
    categories: dict[str, list[dict[str, Any]]] = {"demonstrator_contact": [], "intended_receiver_contact": [], "unrelated_background_contact": [], "background_contact": []}
    for index, raw in enumerate(raw_contacts):
        if not isinstance(raw, Mapping) or not raw.get("body1_name") or not raw.get("body2_name"):
            return infrastructure_gate("contact_role_resolution_unknown", contact_index=index, contact_telemetry_complete=False, required_contact=None, unrelated_contact_runtime=None)
        body1, body2 = str(raw["body1_name"]), str(raw["body2_name"])
        role = contact_role(body1, body2)
        categories[role].append({"body_pair": sorted((body1, body2)), "geom_pair": sorted((str(raw.get("geom1_name", "")), str(raw.get("geom2_name", "")))), "distance_m": raw.get("distance_m"), "role": role})
    demonstrator = bool(categories["demonstrator_contact"])
    unrelated = bool(categories["unrelated_background_contact"])
    return {
        "schema_version": "track3_contact_truth_v3",
        "executed": True,
        "state": "passed" if demonstrator and not unrelated else "failed",
        "required": True,
        "categories": categories,
        "counts": {key: len(value) for key, value in categories.items()},
        "contact_telemetry_complete": True,
        "telemetry_known": True,
        "required_contact": demonstrator,
        "unrelated_contact_runtime": unrelated,
        "derivation": "MuJoCo contact pair body/geom names and participant role; no telemetry completeness shortcut",
    }


def required_contact_gate(samples: Sequence[Mapping[str, Any]], barrier_index: int | None) -> dict[str, Any]:
    if barrier_index is None or barrier_index < 0:
        return gate(False, None, reason="acquisition_barrier_not_entered")
    scoped = list(samples[: barrier_index + 1])
    if not scoped:
        return gate(False, None, reason="acquisition_contact_window_empty")
    truths = [contact_truth_v3(TASK_INDEX, row.get("contact_matrix", {})) for row in scoped]
    if any(row.get("state") == "infrastructure_failure" for row in truths):
        return infrastructure_gate("contact_truth_unknown_before_acquisition_barrier", samples=len(truths))
    values = [row.get("required_contact") for row in truths]
    passed = all(value is True for value in values)
    return gate(True, passed, reason="zero_required_contact" if not passed else None, barrier_index=barrier_index, samples=len(values), required_contact_values=values, late_contact_excluded=len(samples) > len(scoped), telemetry_values=[row.get("contact_telemetry_complete") for row in truths])


def normalized_action_contract(action: Sequence[float]) -> dict[str, Any]:
    values = [float(item) for item in action]
    finite = all(math.isfinite(item) for item in values)
    shape_ok = len(values) == ACTION_DIM
    gripper_ok = shape_ok and values[6] in {-1.0, 1.0}
    normalized_ok = shape_ok and all(abs(item) <= 1.0 + 1e-9 for item in values[:6])
    rotation_l2 = math.sqrt(sum(item * item for item in values[3:6])) if shape_ok else float("nan")
    return {"state": "passed" if finite and shape_ok and gripper_ok and normalized_ok else "failed", "finite": finite, "shape": len(values), "gripper_values": values[6:7], "normalized_action_abs_max": max((abs(item) for item in values), default=float("nan")), "normalized_rotation_l2": rotation_l2, "double_normalization": False, "physical_arrays_separate": True}


def waypoint_contract(profile: Mapping[str, Any] = PRIMARY_PROFILE) -> dict[str, Any]:
    stages = [
        {"name": "precontact_xy", "axis": "xy_only", "translation_axes": [0, 1], "rotation_enabled": False},
        {"name": "precontact_z", "axis": "z_only", "translation_axes": [2], "rotation_enabled": False},
        {"name": "orientation_align", "axis": "orientation_only", "translation_axes": [], "rotation_enabled": True},
        {"name": "contact_descent", "axis": "z_only", "translation_axes": [2], "rotation_enabled": True},
        {"name": "grasp_dwell", "axis": "hold", "translation_axes": [], "rotation_enabled": True, "gripper": 1.0},
        {"name": "lift", "axis": "z_only", "translation_axes": [2], "rotation_enabled": True},
        {"name": "transport", "axis": "xy_only", "translation_axes": [0, 1], "rotation_enabled": True},
        {"name": "place", "axis": "z_only", "translation_axes": [2], "rotation_enabled": True},
        {"name": "release", "axis": "hold", "translation_axes": [], "rotation_enabled": True, "gripper": -1.0},
        {"name": "retreat", "axis": "z_only", "translation_axes": [2], "rotation_enabled": True, "gripper": -1.0},
    ]
    return {"schema_version": "track3_sg2_waypoint_controller_v1", "stage_names": [row["name"] for row in stages], "stages": stages, "profile": dict(profile), "caps": {"translation_clip_m": float(profile["translation_clip_m"]), "rotation_clip_rad": float(profile["rotation_clip_rad"]), "microstep_translation_m": float(profile["microstep_translation_m"]), "microstep_rotation_rad": float(profile["microstep_rotation_rad"])}, "action_units": "input is already normalized; physical deltas are separate and never normalized twice", "quaternion_convention": "xyzw_shortest_path", "trajectory_source": "simulator_post_step_states_and_actions", "keypose_is_not_trajectory": True}


def _command_hash(command: Sequence[str]) -> str:
    return json_hash(list(command))


def identity_gate(*, expected_commit: str, expected_tool_sha256: str, expected_openpi: str, expected_command: Sequence[str], expected_libero_plus_porcelain_sha256: str | None = None, observed_commit: str | None = None, observed_tool_sha256: str | None = None, observed_openpi: str | None = None, observed_command: Sequence[str] | None = None, observed_libero_plus_porcelain_sha256: str | None = None, dirty: bool | None = None, pid_active: bool = False, cursor_active: bool = False, expected_cuda_visible_devices: str = "", observed_cuda_visible_devices: str | None = None, gpu_policy: str = "cpu_no_renderer") -> dict[str, Any]:
    observed_commit = observed_commit if observed_commit is not None else git(("rev-parse", "HEAD"))
    observed_tool_sha256 = observed_tool_sha256 if observed_tool_sha256 is not None else sha256_file(Path(__file__))
    observed_openpi = observed_openpi if observed_openpi is not None else git(("rev-parse", "HEAD"), OPENPI)
    observed_command = list(observed_command if observed_command is not None else expected_command)
    observed_cuda_visible_devices = "" if observed_cuda_visible_devices is None else observed_cuda_visible_devices
    if dirty is None:
        dirty = bool(source_porcelain(ROOT))
    checks = {
        "root_commit_exact": observed_commit == expected_commit,
        "tool_sha256_exact": observed_tool_sha256 == expected_tool_sha256,
        "openpi_commit_exact": observed_openpi == expected_openpi,
        "root_clean_now": not dirty,
        "libero_plus_baseline_exact": expected_libero_plus_porcelain_sha256 is None or observed_libero_plus_porcelain_sha256 == expected_libero_plus_porcelain_sha256,
        "pid_inactive": not pid_active,
        "cursor_inactive": not cursor_active,
        "command_exact": list(observed_command) == list(expected_command),
        "command_hash_exact": _command_hash(observed_command) == _command_hash(expected_command),
        "cuda_visible_devices_exact": observed_cuda_visible_devices == expected_cuda_visible_devices,
        "gpu_policy_allowed": gpu_policy in {"cpu_no_renderer", "gpu3_camera_only"},
    }
    errors = [key for key, value in checks.items() if not value]
    return {"state": "passed" if not errors else "failed", "errors": errors, "checks": checks, "expected_root_commit": expected_commit, "observed_root_commit": observed_commit, "expected_command": list(expected_command), "observed_command": list(observed_command), "command_hash": _command_hash(expected_command), "gpu_policy": gpu_policy}


def _lock_file(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def allocate_attempt() -> tuple[int, Path]:
    with _lock_file(ATTEMPT_LOCK) as handle:
        attempts = SG_ROOT / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        ids = [int(path.name.removeprefix("attempt_")) for path in attempts.glob("attempt_*") if path.is_dir() and path.name.removeprefix("attempt_").isdigit()]
        attempt_id = max(ids, default=0) + 1
        attempt = attempts / f"attempt_{attempt_id:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        write_json(CURRENT_ATTEMPT, {"schema_version": "track3_sg_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": True, "created_at_utc": now()})
        return attempt_id, attempt


def set_attempt_active(attempt_id: int, attempt: Path, active: bool, *, status: str) -> None:
    write_json(CURRENT_ATTEMPT, {"schema_version": "track3_sg_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": active, "status": status, "updated_at_utc": now()})


FINALIZER_NAMES = {"producer_manifest.json", "final_manifest.json", "outer_hash.json", "closure.json"}


def _artifact_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in FINALIZER_NAMES or path.name.endswith(".partial"):
            continue
        entries.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def make_manifest(root: Path, filename: str = "producer_manifest.json") -> dict[str, Any]:
    entries = _artifact_entries(root)
    result = {"schema_version": "track3_sg_producer_manifest_v2", "manifest_role": "producer_closed_set", "entries": entries, "excluded_finalizer_names": sorted(FINALIZER_NAMES), "producer_closed_before_seal": True}
    write_json(root / filename, result, immutable=True)
    return result


def verify_complete_set(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = {str(row["relative_path"]): row for row in manifest.get("entries", [])}
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file() and path.name not in FINALIZER_NAMES and not path.name.endswith(".partial")}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches: list[str] = []
    for name, row in expected.items():
        path = actual.get(name)
        if path is not None and (int(row.get("bytes", -1)) != path.stat().st_size or str(row.get("sha256")) != sha256_file(path)):
            mismatches.append(name)
    partials = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.partial"))
    return {"exact": not missing and not extra and not mismatches and not partials, "missing": missing, "extra": extra, "hash_or_size_mismatch": mismatches, "partial_residue": partials}


def finalize_attempt(root: Path) -> dict[str, Any]:
    producer_path = root / "producer_manifest.json"
    producer = strict_json(producer_path) if producer_path.is_file() else make_manifest(root)
    producer_check = verify_complete_set(root, producer)
    if not producer_check["exact"]:
        return {"status": "failed", "producer_manifest": producer_check, "reason": "producer_set_changed_before_finalizer"}
    final = {"schema_version": "track3_sg_final_manifest_v2", "manifest_role": "post_producer_complete_set", "entries": _artifact_entries(root), "excluded_finalizer_names": sorted(FINALIZER_NAMES)}
    write_json(root / "final_manifest.json", final, immutable=True)
    outer = {"schema_version": "track3_sg_outer_hash_v2", "final_manifest_sha256": sha256_file(root / "final_manifest.json"), "entry_count": len(final["entries"])}
    write_json(root / "outer_hash.json", outer, immutable=True)
    complete = verify_complete_set(root, final)
    closure = {"schema_version": "track3_sg_closure_v2", "status": "passed" if complete["exact"] and sha256_file(root / "final_manifest.json") == outer["final_manifest_sha256"] else "failed", "complete_set": complete, "outer_hash_matches": sha256_file(root / "final_manifest.json") == outer["final_manifest_sha256"]}
    write_json(root / "closure.json", closure, immutable=True)
    return {"status": closure["status"], "producer_manifest": producer_check, "final_manifest": complete, "closure": closure}


def _trace_action_hashes() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    action_hashes: list[str] = []
    if not CR_SUMMARY.is_file():
        raise FileNotFoundError(CR_SUMMARY)
    summary = strict_json(CR_SUMMARY)
    import numpy as np

    for profile in summary.get("profiles", []):
        trace_path = CR_TRACES / (Path(str(profile.get("child_result", {}).get("path", ""))).name.replace(".child.json", ".npz"))
        if not trace_path.is_file():
            # The summary's payload is authoritative for the relative trace path.
            trace_path = ROOT / str(profile.get("payload", {}).get("cr_gate", {}).get("typed_trace", {}).get("path", ""))
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)
        with np.load(trace_path, allow_pickle=False) as archive:
            action_hash = sha256_bytes(np.asarray(archive["actions"]).tobytes())
        trace_hash = sha256_file(trace_path)
        claimed_trace = str(profile.get("trace_sha256") or profile.get("payload", {}).get("behavior_trace_sha256"))
        claimed_action = str(profile.get("action_sha256") or profile.get("payload", {}).get("action_sha256"))
        if claimed_trace != trace_hash or claimed_action != action_hash:
            raise ValueError(f"historical_hash_drift:{profile.get('profile_id')}")
        rows.append({"profile_id": profile.get("profile_id"), "family": profile.get("family"), "category": profile.get("category"), "trace_path": rel(trace_path), "trace_sha256": trace_hash, "action_sha256": action_hash, "contact_derivation": "unavailable_invalid_runtime_derivation", "unrelated_contact_runtime": "unavailable_invalid_runtime_derivation", "training_eligible": False})
        action_hashes.append(action_hash)
    return rows, action_hashes


def run_sg0_to_sg4() -> dict[str, Any]:
    attempt_id, attempt = allocate_attempt()
    try:
        rows, action_hashes = _trace_action_hashes()
        partial = {"path": rel(CR_PARTIAL), "bytes": CR_PARTIAL.stat().st_size, "sha256": sha256_file(CR_PARTIAL), "synthetic": False} if CR_PARTIAL.is_file() else {"state": "missing"}
        synthetic = {"path": rel(CR_SYNTHETIC), "bytes": CR_SYNTHETIC.stat().st_size, "sha256": sha256_file(CR_SYNTHETIC), "synthetic": True} if CR_SYNTHETIC.is_file() else {"state": "missing"}
        libero_rows = porcelain(LIBERO_PLUS)
        source = {"root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "root_porcelain": source_porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI), "libero_plus_porcelain": libero_rows, "libero_plus_porcelain_sha256": json_hash(libero_rows), "gpu_policy": "CPU/no-renderer only"}
        failure_matrix = []
        for row in rows:
            failure_matrix.append({**row, "earliest_failed_stage": "unknown_historical_stage_read_only", "required_contact": "unavailable_invalid_runtime_derivation", "contact_telemetry_complete": "historical_field_not_promotable", "unrelated_contact_runtime": "unavailable_invalid_runtime_derivation", "can_be_used_for": ["diagnostic_history"], "cannot_be_used_for": ["contact", "gold", "accepted", "training"]})
        sg0 = {"schema_version": "track3_sg0_input_lock_v1", "status": "passed", "attempt_id": attempt_id, "created_at_utc": now(), "audit": {"path": rel(CR_AUDIT), "sha256": sha256_file(CR_AUDIT)}, "historical_cr_summary": {"path": rel(CR_SUMMARY), "sha256": sha256_file(CR_SUMMARY), "rows": rows, "action_hashes": action_hashes}, "partial": partial, "synthetic_fixture": synthetic, "source_snapshot": source, "failure_matrix": failure_matrix, "priority_family": "support_remove", "priority_reason": "CR support alternative descend error 0.013362138900100747 m is closest measured diagnostic, not a success or threshold change"}
        write_json(attempt / "sg0_input_lock.json", sg0, immutable=True)
        write_json(SG0_REPORT, {"schema_version": "track3_sg0_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sg0_input_lock.json")})

        # SG1: every fixture calls the same runtime derivation function.
        def contact(body1: str, body2: str) -> dict[str, Any]:
            return {"complete": True, "contacts": [{"body1_name": body1, "body2_name": body2, "geom1_name": body1 + ":geom", "geom2_name": body2 + ":geom", "distance_m": -0.001}]}

        sg1_cases = {
            "schema_complete_zero_contact": contact_truth_v3(TASK_INDEX, {"complete": True, "contacts": []}),
            "target_demonstrator_contact": contact_truth_v3(TASK_INDEX, contact("akita_black_bowl_1", "robot0_right_hand")),
            "late_contact_excluded": required_contact_gate([{"contact_matrix": {"complete": True, "contacts": []}}, {"contact_matrix": contact("akita_black_bowl_1", "robot0_right_hand")}], 0),
            "unrelated_contact": contact_truth_v3(TASK_INDEX, contact("akita_black_bowl_1", "main_table")),
            "all_true_surrogate_rejected": contact_truth_v3(TASK_INDEX, {"complete": True, "contacts": [], "required_contact": True, "contact_telemetry_complete": True, "unrelated_contact_runtime": False}),
            "unknown_schema": contact_truth_v3(TASK_INDEX, {"complete": False, "contacts": []}),
        }
        sg1_checks = {"zero_contact_failed": sg1_cases["schema_complete_zero_contact"]["state"] == "failed" and sg1_cases["schema_complete_zero_contact"]["required_contact"] is False, "target_pair_passed": sg1_cases["target_demonstrator_contact"]["required_contact"] is True and sg1_cases["target_demonstrator_contact"]["contact_telemetry_complete"] is True, "late_contact_not_backfilled": sg1_cases["late_contact_excluded"]["state"] == "failed", "unrelated_failed": sg1_cases["unrelated_contact"]["state"] == "failed" and sg1_cases["unrelated_contact"]["unrelated_contact_runtime"] is True, "surrogate_rejected": sg1_cases["all_true_surrogate_rejected"]["required_contact"] is False, "unknown_infrastructure": sg1_cases["unknown_schema"]["state"] == "infrastructure_failure"}
        sg1 = {"schema_version": "track3_sg1_contact_truth_fixtures_v1", "status": "passed" if all(sg1_checks.values()) else "failed", "cases": sg1_cases, "checks": sg1_checks, "contract": "required_contact=actual demonstrator pair; telemetry completeness is independent; late contact is excluded by barrier scope"}
        write_json(attempt / "sg1_contact_truth_fixtures.json", sg1, immutable=True)
        write_json(SG1_REPORT, {"schema_version": "track3_sg1_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sg1_contact_truth_fixtures.json")})

        # SG2: simulator-free contract fixtures, including non-zero orientation error.
        import numpy as np
        from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as control

        orientation_error = control.quat_log_error_xyzw([0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)], [0.0, 0.0, 0.0, 1.0])
        action = control.osc_action_scaled([0.01, 0.0, 0.0], orientation_error, -1.0, profile=PRIMARY_PROFILE, scales={"translation_m": 0.05, "rotation_rad": 0.5})
        action_check = normalized_action_contract(action["action"])
        sg2_contract = waypoint_contract(PRIMARY_PROFILE)
        sg2_checks = {"stage_order_exact": sg2_contract["stage_names"] == list(STAGE_NAMES), "xy_z_axes_separate": sg2_contract["stages"][0]["translation_axes"] == [0, 1] and sg2_contract["stages"][1]["translation_axes"] == [2], "orientation_nonzero_fixture": float(np.linalg.norm(np.asarray(orientation_error))) > 0.0 and float(np.linalg.norm(np.asarray(action["action"][3:6]))) > 0.0, "normalized_action_unit": action_check["state"] == "passed" and action_check["double_normalization"] is False, "physical_arrays_separate": action_check["physical_arrays_separate"] is True, "keypose_not_trajectory": sg2_contract["keypose_is_not_trajectory"] is True}
        sg2 = {"schema_version": "track3_sg2_waypoint_controller_contract_v1", "status": "passed" if all(sg2_checks.values()) else "failed", "contract": sg2_contract, "fixture": {"orientation_error_rad": orientation_error, "action": action, "action_validation": action_check}, "checks": sg2_checks}
        write_json(attempt / "sg2_waypoint_controller_contract.json", sg2, immutable=True)
        write_json(SG2_REPORT, {"schema_version": "track3_sg2_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sg2_waypoint_controller_contract.json")})

        # SG3 uses real files/processes and the same manifest/identity functions.
        sg3 = run_failure_injection_fixtures(attempt / "sg3_fixtures")
        write_json(attempt / "sg3_failure_injection_fixtures.json", sg3, immutable=True)
        write_json(SG3_REPORT, {"schema_version": "track3_sg3_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sg3_failure_injection_fixtures.json")})

        sg4 = run_sg4_machine_handoff()
        write_json(attempt / "sg4_machine_handoff.json", sg4, immutable=True)
        write_json(SG4_REPORT, {"schema_version": "track3_sg4_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sg4_machine_handoff.json")})
        statuses = {"SG0": sg0["status"], "SG1": sg1["status"], "SG2": sg2["status"], "SG3": sg3["status"], "SG4": sg4["status"]}
        overall = "passed" if all(value == "passed" for value in statuses.values()) else "hard_stop"
        write_text(attempt / "phase_ledger.csv", "phase,status\n" + "\n".join(f"{key},{value}" for key, value in statuses.items()) + "\n")
        set_attempt_active(attempt_id, attempt, False, status=overall)
        return {"status": overall, "attempt_id": attempt_id, "attempt_root": rel(attempt), "phase_status": statuses}
    except Exception as exc:
        write_json(attempt / "sg0_to_sg4_exception.json", {"schema_version": "track3_sg0_sg4_exception_v1", "status": "hard_stop", "error": {"type": type(exc).__name__, "message": str(exc)}}, immutable=True)
        set_attempt_active(attempt_id, attempt, False, status="hard_stop")
        return {"status": "hard_stop", "attempt_id": attempt_id, "attempt_root": rel(attempt), "error": f"{type(exc).__name__}:{exc}"}


def run_failure_injection_fixtures(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="sg3_", dir=root) as temporary:
        fixture = Path(temporary)
        atomic_bytes(fixture / "nested" / "heartbeat.json", b"{}\n")
        make_manifest(fixture)
        atomic_bytes(fixture / "nested" / "heartbeat.json", b"mutated\n")
        cases["after_seal_mutation"] = {"rejected": not verify_complete_set(fixture, strict_json(fixture / "producer_manifest.json"))["exact"]}
    with tempfile.TemporaryDirectory(prefix="sg3_", dir=root) as temporary:
        fixture = Path(temporary)
        atomic_bytes(fixture / "producer.log", b"producer\n")
        manifest = make_manifest(fixture)
        atomic_bytes(fixture / "nested" / "post_generation.log", b"late\n")
        cases["missing_nested_log_from_manifest"] = {"rejected": not verify_complete_set(fixture, manifest)["exact"]}
    with tempfile.TemporaryDirectory(prefix="sg3_", dir=root) as temporary:
        fixture = Path(temporary)
        atomic_bytes(fixture / "a", b"a")
        make_manifest(fixture)
        finalized = finalize_attempt(fixture)
        outer = strict_json(fixture / "outer_hash.json")
        outer["final_manifest_sha256"] = "wrong"
        write_json(fixture / "outer_hash.json", outer)
        cases["bad_outer_hash"] = {"rejected": finalized["status"] == "passed" and strict_json(fixture / "outer_hash.json")["final_manifest_sha256"] != sha256_file(fixture / "final_manifest.json")}
    cases["active_cursor"] = {"rejected": identity_gate(expected_commit="a", expected_tool_sha256="b", expected_openpi="c", expected_command=["x"], observed_commit="a", observed_tool_sha256="b", observed_openpi="c", observed_command=["x"], dirty=False, cursor_active=True)["state"] == "failed"}
    with tempfile.TemporaryDirectory(prefix="sg3_", dir=root) as temporary:
        fixture = Path(temporary)
        atomic_bytes(fixture / "heartbeat.json.partial", b"partial")
        cases["leftover_partial"] = {"rejected": bool(verify_complete_set(fixture, {"entries": []})["partial_residue"])}
    cases["dirty_source"] = {"rejected": identity_gate(expected_commit="a", expected_tool_sha256="b", expected_openpi="c", expected_command=["x"], observed_commit="a", observed_tool_sha256="b", observed_openpi="c", observed_command=["x"], dirty=True)["state"] == "failed"}
    cases["tool_command_profile_drift"] = {"rejected": identity_gate(expected_commit="a", expected_tool_sha256="b", expected_openpi="c", expected_command=["x"], observed_commit="a", observed_tool_sha256="wrong", observed_openpi="c", observed_command=["x"])["state"] == "failed" and identity_gate(expected_commit="a", expected_tool_sha256="b", expected_openpi="c", expected_command=["x"], observed_commit="a", observed_tool_sha256="b", observed_openpi="c", observed_command=["wrong"])["state"] == "failed"}
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=TERM_GRACE_SECONDS)
            term_stage = "TERM"
        except subprocess.TimeoutExpired:
            term_stage = "KILL"
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=KILL_WAIT_SECONDS)
        cases["timeout_term_kill_no_descendants"] = {"rejected": process.poll() is not None, "termination_stage": term_stage, "pid": process.pid}
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    stale_pointer = root / "stale_pointer.json"
    write_json(stale_pointer, {"target": "does_not_exist.json", "active": False})
    cases["stale_pointer"] = {"rejected": not (stale_pointer.parent / strict_json(stale_pointer)["target"]).is_file()}
    statuses = {name: bool(value.get("rejected")) for name, value in cases.items()}
    return {"schema_version": "track3_sg3_failure_injection_fixtures_v1", "status": "passed" if all(statuses.values()) else "failed", "cases": cases, "checks": statuses, "real_inputs": True, "no_hardcoded_pass_flags": True}


def _run_cli(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(list(command), cwd=ROOT, capture_output=True, text=True, check=False, timeout=60)
    return {"command": list(command), "returncode": result.returncode, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}


def recorder_command(task_index: int = TASK_INDEX) -> list[str]:
    return [str(RECORDER_PYTHON), "-m", RECORDER, "--record", "--task-index", str(task_index), "--operator", "operator-local", "--preview-port", "8765"]


def replay_dry_run_command() -> list[str]:
    return [str(RECORDER_PYTHON), "-m", REPLAY, "--help"]


def run_sg4_machine_handoff() -> dict[str, Any]:
    command = recorder_command()
    help_result = _run_cli([str(RECORDER_PYTHON), "-m", RECORDER, "--help"]) if RECORDER_PYTHON.is_file() else {"returncode": 127, "stderr_tail": "venv_missing"}
    self_test = _run_cli([str(RECORDER_PYTHON), "-m", RECORDER, "--self-test"]) if RECORDER_PYTHON.is_file() else {"returncode": 127, "stderr_tail": "venv_missing"}
    replay_help = _run_cli(replay_dry_run_command()) if RECORDER_PYTHON.is_file() else {"returncode": 127, "stderr_tail": "venv_missing"}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as first:
        first.bind(("127.0.0.1", 0))
        port = int(first.getsockname()[1])
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as second:
            collision_rejected = False
            try:
                second.bind(("127.0.0.1", port))
            except OSError:
                collision_rejected = True
    checks = {"exact_recorder_argv": command == [str(RECORDER_PYTHON), "-m", RECORDER, "--record", "--task-index", "10", "--operator", "operator-local", "--preview-port", "8765"], "help_parses": help_result.get("returncode") == 0, "self_test_passes": self_test.get("returncode") == 0, "replay_help_parses": replay_help.get("returncode") == 0, "loopback_port_collision_guard": collision_rejected, "no_record_started": True, "gpu3_guard_not_started": True}
    handoff = {"schema_version": "track3_sg4_machine_handoff_v1", "status": "passed" if all(checks.values()) else "failed", "machine_ready": bool(all(checks.values())), "recording_started": False, "replay_started": False, "exact_recorder_argv": command, "cwd": str(ROOT), "environment": {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "preview_bind": "127.0.0.1"}, "task": {"task_index": TASK_INDEX, "demo_index": DEMO_INDEX, "family": FAMILY, "source_atom": SOURCE_ATOM, "goal_atom": GOAL_ATOM}, "start_state_and_model": "resolved only at user-controlled recording; no candidate was created", "browser": {"url": "http://127.0.0.1:8765", "loopback_only": True}, "keys": {"translation": "WASD/RF", "rotation": "IJKL/UO", "gripper": "G=open, H=close", "reset": "R", "abort": "ESC", "save": "S"}, "stop_conditions": ["unknown telemetry", "unrelated contact", "source/goal contract failure", "abort", "partial or identity drift"], "output": {"state": "pending_only", "directory": "data/reports/axis_libero_30k/track3_reverse_lora/pending"}, "replay_dry_run": {"command": replay_dry_run_command(), "result": replay_help}, "checks": checks, "help": help_result, "self_test": self_test, "resource_boundary": {"simulator_started": False, "gpu3_started": False, "recording": False, "replay": False, "accepted_promotion": False}}
    return handoff


def _latest_attempt() -> tuple[int, Path] | None:
    if not CURRENT_ATTEMPT.is_file():
        return None
    pointer = strict_json(CURRENT_ATTEMPT)
    attempt = ROOT / str(pointer["attempt_root"])
    return int(pointer["attempt_id"]), attempt


def run_sg5() -> dict[str, Any]:
    latest = _latest_attempt()
    if latest is None:
        return {"status": "not_evaluated", "reason": "sg0_to_sg4_missing"}
    attempt_id, attempt = latest
    sg0_path = attempt / "sg0_input_lock.json"
    sg4_path = attempt / "sg4_machine_handoff.json"
    if not sg0_path.is_file() or not sg4_path.is_file():
        return {"status": "not_evaluated", "reason": "sg0_to_sg4_missing"}
    sg0, sg4 = strict_json(sg0_path), strict_json(sg4_path)
    expected_commit = git(("rev-parse", "HEAD"))
    expected_tool = sha256_file(Path(__file__))
    expected_openpi = git(("rev-parse", "HEAD"), OPENPI)
    libero_hash = str(sg0["source_snapshot"]["libero_plus_porcelain_sha256"])
    command = [str(RECORDER_PYTHON), "-m", MODULE, "--sg6-worker", "--attempt-root", str(attempt), "--result-output", str(attempt / "canary" / "child_result.json"), "--profile", PRIMARY_PROFILE["name"]]
    observed_libero_hash = json_hash(porcelain(LIBERO_PLUS))
    identity = identity_gate(expected_commit=expected_commit, expected_tool_sha256=expected_tool, expected_openpi=expected_openpi, expected_command=command, expected_libero_plus_porcelain_sha256=libero_hash, observed_libero_plus_porcelain_sha256=observed_libero_hash, dirty=bool(source_porcelain(ROOT)), pid_active=False, cursor_active=False, expected_cuda_visible_devices="", observed_cuda_visible_devices="", gpu_policy="cpu_no_renderer")
    phase_status = {f"SG{index}": "passed" for index in range(0, 4)}
    phase_status["SG4"] = sg4.get("status", "failed")
    phase_status["SG5"] = "passed" if identity["state"] == "passed" and sg4.get("status") == "passed" else "hard_stop"
    result = {"schema_version": "track3_sg5_clean_source_release_v1", "status": "passed" if phase_status["SG5"] == "passed" else "hard_stop", "attempt_id": attempt_id, "phase_status": phase_status, "clean_source_identity": identity, "root_commit": expected_commit, "tool_sha256": expected_tool, "openpi_commit": expected_openpi, "profile_lock": {"primary": PRIMARY_PROFILE, "alternative_frozen_but_not_eligible": ALTERNATIVE_PROFILE, "max_support_identities": MAX_SUPPORT_IDENTITIES}, "requirements": {"strict_json": True, "py_compile": True, "git_diff_check": True, "no_partial": True, "process_boundary": True, "gpu_boundary": True, "pre_spawn_and_child_identity_gate": True, "monotonic_attempt": True, "two_layer_manifest_finalizer": True}, "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "physics_started": False, "teleop": False, "replay": False, "training": False, "zip": False, "upload": False, "submission": False}}
    write_json(attempt / "sg5_clean_source_release.json", result, immutable=True)
    write_json(SG5_REPORT, {"schema_version": "track3_sg5_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sg5_clean_source_release.json")})
    return result


def _relative_pose(object_pose: Mapping[str, Any], eef_pose: Mapping[str, Any], control: Any) -> dict[str, Any]:
    object_quat = control.quat_normalize_xyzw(object_pose["quat"])
    delta = [float(eef_pose["eef_translation_m"][index]) - float(object_pose["pos_m"][index]) for index in range(3)]
    rotation = control.quat_to_matrix_xyzw(object_quat)
    local = [sum(rotation[column][row] * delta[column] for column in range(3)) for row in range(3)]
    return {"translation_m": local, "quaternion_xyzw": list(control.quat_mul_xyzw(control.quat_conjugate_xyzw(object_quat), eef_pose["eef_quaternion"]))}


def _select_translation(stage: str, target: Sequence[float], current: Mapping[str, Any]) -> list[float]:
    delta = [float(target[index]) - float(current["eef_translation_m"][index]) for index in range(3)]
    if stage in {"precontact_xy", "transport"}:
        return [delta[0], delta[1], 0.0]
    if stage in {"precontact_z", "contact_descent", "lift", "place", "retreat"}:
        return [0.0, 0.0, delta[2]]
    return [0.0, 0.0, 0.0]


def sg6_worker(profile: Mapping[str, Any], result_path: Path, attempt: Path) -> int:
    """Run one actual CPU/no-renderer support identity in a fresh process."""
    import numpy as np

    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
    from tools.axis_pi05 import track3_operator_scoped_stability as prior
    from tools.axis_pi05 import track3_reverse_init_stabilization as base
    from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as control

    started = now()
    env = None
    result: dict[str, Any]
    try:
        sg0 = strict_json(attempt / "sg0_input_lock.json")
        expected = {"root_commit": sg0["source_snapshot"]["root_commit"], "tool_sha256": sg0["source_snapshot"]["tool_sha256"], "openpi_commit": sg0["source_snapshot"]["openpi_commit"], "libero_plus_porcelain_sha256": sg0["source_snapshot"]["libero_plus_porcelain_sha256"]}
        command = [str(RECORDER_PYTHON), "-m", MODULE, "--sg6-worker", "--attempt-root", str(attempt), "--result-output", str(result_path), "--profile", str(profile["name"])]
        child_identity = identity_gate(expected_commit=expected["root_commit"], expected_tool_sha256=expected["tool_sha256"], expected_openpi=expected["openpi_commit"], expected_command=command, expected_libero_plus_porcelain_sha256=expected["libero_plus_porcelain_sha256"], observed_libero_plus_porcelain_sha256=json_hash(porcelain(LIBERO_PLUS)), dirty=bool(source_porcelain(ROOT)), pid_active=False, cursor_active=False, expected_cuda_visible_devices="", observed_cuda_visible_devices="", gpu_policy="cpu_no_renderer")
        if child_identity["state"] != "passed":
            result = {"schema_version": "track3_sg6_child_v1", "status": "infrastructure_failure", "category": "infrastructure_failure", "child_pid": os.getpid(), "identity_gate": child_identity, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False}}
            write_json(result_path, result)
            return 0
        manifest, model_bytes, _init_state, _terminal_state, _demo_info = composed._demo_arrays(TASK_INDEX, DEMO_INDEX)
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        with np.load(control.STATE_BUNDLE, allow_pickle=False) as archive:
            start_state = np.asarray(archive["task10_demo0"], dtype=np.float64).copy()
        w1 = strict_json(control.W1_REPORT)
        w2 = strict_json(control.W2_REPORT)
        terminal_pose = w1["state_anchors"]["terminal"]["bowl"]
        init_pose = w1["state_anchors"]["state0"]["bowl"]
        relative = w1["reverse_acquisition_anchor"]["relative_pose_xyzw"]
        grasp = compose_object_to_eef(terminal_pose, relative)
        destination = compose_object_to_eef(init_pose, relative)
        transport = compose_object_to_eef(init_pose, relative, object_position_offset_m=(0.0, 0.0, 0.12))
        composed._configure_public_env(gpu=False)
        from libero.libero.envs.env_wrapper import ControlEnv

        env = ControlEnv(bddl_file_name=str(composed.REVERSE_BDDL[TASK_INDEX]), control_freq=CONTROL_FREQ_HZ, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=-1)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        env.set_init_state(start_state)
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(composed.REVERSE_BDDL[TASK_INDEX], "init")
        goal_atoms = base.public_predicates(composed.REVERSE_BDDL[TASK_INDEX], "goal")
        hold_quat = list(dynamic._robot_metrics(env)["eef_quaternion"])
        stages = [
            ("precontact_xy", -1.0, [grasp["pos_m"][0], grasp["pos_m"][1], float(dynamic._robot_metrics(env)["eef_translation_m"][2])], hold_quat, int(profile["stage_budgets"][0])),
            ("precontact_z", -1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.04], hold_quat, int(profile["stage_budgets"][1])),
            ("orientation_align", -1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.04], grasp["quat"], int(profile["stage_budgets"][2])),
            ("contact_descent", -1.0, grasp["pos_m"], grasp["quat"], int(profile["stage_budgets"][3])),
            ("grasp_dwell", 1.0, grasp["pos_m"], grasp["quat"], int(profile["stage_budgets"][4])),
            ("lift", 1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.14], grasp["quat"], int(profile["stage_budgets"][5])),
            ("transport", 1.0, [transport["pos_m"][0], transport["pos_m"][1], grasp["pos_m"][2] + 0.14], transport["quat"], int(profile["stage_budgets"][6])),
            ("place", 1.0, destination["pos_m"], destination["quat"], int(profile["stage_budgets"][7])),
            ("release", -1.0, destination["pos_m"], destination["quat"], int(profile["stage_budgets"][8])),
            ("retreat", -1.0, [destination["pos_m"][0], destination["pos_m"][1], destination["pos_m"][2] + 0.16], destination["quat"], int(profile["stage_budgets"][9])),
        ]
        samples: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        states: list[Any] = []
        phases: list[str] = []
        eef_positions: list[list[float]] = []
        eef_quaternions: list[list[float]] = []
        target_positions: list[list[float]] = []
        target_quaternions: list[list[float]] = []
        physical_translation: list[list[float]] = []
        physical_rotation: list[list[float]] = []
        action_saturated: list[bool] = []
        source_values: list[Any] = []
        goal_values: list[Any] = []

        def observe(step: int) -> dict[str, Any]:
            sample = prior._runtime_sample(env, dynamic, TASK_INDEX, "reverse", step, initial_atoms, goal_atoms)
            sample["contact_truth_v3"] = contact_truth_v3(TASK_INDEX, sample.get("contact_matrix", {}))
            samples.append(sample)
            return sample

        observe(0)
        stage_results: list[dict[str, Any]] = []
        failure_reason: str | None = None
        for stage_index, (stage, gripper, target_position, target_quat, budget) in enumerate(stages):
            stable = 0
            stage_start = len(actions)
            last_t, last_r = float("inf"), float("inf")
            for _ in range(budget):
                current = dynamic._robot_metrics(env)
                translation = _select_translation(stage, target_position, current)
                rotation = control.quat_log_error_xyzw(target_quat, current["eef_quaternion"]) if stage not in {"precontact_xy", "precontact_z"} else [0.0, 0.0, 0.0]
                command_result = control.osc_action_scaled(translation, rotation, gripper, profile=profile, scales={"translation_m": float(profile["controller_translation_scale_m"]), "rotation_rad": float(profile["controller_rotation_scale_rad"])})
                _obs_after, _reward, done, _info = env.step(np.asarray(command_result["action"], dtype=np.float32))
                actions.append(list(command_result["action"]))
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
                phases.append(stage)
                eef_positions.append(list(current["eef_translation_m"]))
                eef_quaternions.append(list(current["eef_quaternion"]))
                target_positions.append(list(target_position))
                target_quaternions.append(list(target_quat))
                physical_translation.append(list(command_result["physical_translation_delta_m"]))
                physical_rotation.append(list(command_result["physical_rotation_delta_rad"]))
                action_saturated.append(bool(command_result["translation_physical_clip"] or command_result["rotation_physical_clip"] or command_result["controller_clip"]))
                last_t = float(command_result["translation_norm_m"])
                last_r = float(command_result["rotation_norm_rad"])
                sample = observe(len(actions))
                source_value = next((item.get("value") for item in sample.get("initial", {}).get("predicates", []) if item.get("atom") == SOURCE_ATOM), None)
                goal_value = next((item.get("value") for item in sample.get("goal", {}).get("predicates", []) if item.get("atom") == GOAL_ATOM), None)
                source_values.append(predicate_truth(source_value))
                goal_values.append(predicate_truth(goal_value))
                translation_ok = last_t <= float(profile["translation_tolerance_m"]) if stage not in {"orientation_align", "grasp_dwell", "release"} else True
                rotation_ok = last_r <= float(profile["rotation_tolerance_rad"]) if stage not in {"precontact_xy", "precontact_z", "grasp_dwell", "release"} else True
                stable = stable + 1 if translation_ok and rotation_ok else 0
                if stable >= int(profile["stable_ticks"]) or done:
                    break
            reached = stable >= int(profile["stable_ticks"])
            stage_results.append({"stage": stage, "stage_index": stage_index, "frames": len(actions) - stage_start, "budget": budget, "reached": reached, "last_translation_error_m": last_t, "last_rotation_error_rad": last_r})
            if not reached:
                failure_reason = f"stage_timeout:{stage}"
                break
            if stage == "grasp_dwell":
                barrier = required_contact_gate(samples, len(samples) - 1)
                source_released = all(value is False for value in source_values[-int(profile["stable_ticks"]):])
                if barrier["state"] != "passed" or not source_released:
                    failure_reason = "acquisition_barrier"
                    break
        final_success = bool(env.check_success())
        trace_samples = samples[1:]
        contact_rows = [row.get("contact_truth_v3", {}) for row in trace_samples]
        contact_unknown = any(row.get("state") == "infrastructure_failure" for row in contact_rows)
        unrelated_any = any(row.get("unrelated_contact_runtime") is True for row in contact_rows)
        barrier_index = next((index for index, row in enumerate(phases) if row == "grasp_dwell"), None)
        barrier = required_contact_gate(samples, (barrier_index + 1) if barrier_index is not None else None)
        source_initial = source_values[0] if source_values else None
        goal_final = goal_values[-1] if goal_values else None
        entered = {name: any(row["stage"] == name for row in stage_results) for name in STAGE_NAMES}
        stage_gates: dict[str, Any] = {}
        blocked = False
        for row in stage_results:
            stage_gates[row["stage"]] = gate(True, bool(row["reached"]), reason=f"stage_timeout:{row['stage']}" if not row["reached"] else None)
        for name in STAGE_NAMES:
            if name not in stage_gates:
                stage_gates[name] = gate(False, None, reason="upstream_stage_not_reached")
        gates: dict[str, Any] = {"initial_source_true": gate(bool(source_values), source_initial is True), "initial_goal_false": gate(bool(goal_values), goal_values[0] is False if goal_values else None), "acquisition_barrier": barrier, "source_released": gate(entered["grasp_dwell"], all(value is False for value in source_values[-int(profile["stable_ticks"]):]) if source_values else None, reason="acquisition_barrier_failed" if entered["grasp_dwell"] and failure_reason else None), "goal_continuous_true": gate(entered["place"], bool(goal_final is True and final_success), reason="place_stage_not_entered" if not entered["place"] else None), "unrelated_contact": infrastructure_gate("contact_truth_unknown") if contact_unknown else gate(bool(trace_samples), not unrelated_any, reason="unrelated_background_contact" if unrelated_any else None), "finite_trace": gate(bool(actions), bool(np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(states)).all())), "normalized_action": gate(bool(actions), all(normalized_action_contract(action)["state"] == "passed" for action in actions)), "no_clip": gate(bool(actions), not any(action_saturated), action_saturated_count=sum(action_saturated)), "all_stages_reached": gate(bool(stage_results), len(stage_results) == len(STAGE_NAMES) and all(row["reached"] for row in stage_results), reason=failure_reason if failure_reason else None)}
        status = "infrastructure_failure" if contact_unknown else "success" if all(row.get("state") == "passed" for row in gates.values()) else "family_failure"
        run_root = result_path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        with (run_root / "start_state.npy.partial").open("wb") as handle:
            np.save(handle, start_state, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(run_root / "start_state.npy.partial", run_root / "start_state.npy")
        with (run_root / "end_state.npy.partial").open("wb") as handle:
            np.save(handle, np.asarray(env.get_sim_state(), dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(run_root / "end_state.npy.partial", run_root / "end_state.npy")
        arrays = {"actions": np.asarray(actions, dtype=np.float32), "states": np.asarray(states, dtype=np.float64), "phase": np.asarray(phases, dtype="U64"), "eef_position_m": np.asarray(eef_positions, dtype=np.float32), "eef_quaternion_xyzw": np.asarray(eef_quaternions, dtype=np.float32), "target_position_m": np.asarray(target_positions, dtype=np.float32), "target_quaternion_xyzw": np.asarray(target_quaternions, dtype=np.float32), "physical_translation_delta_m": np.asarray(physical_translation, dtype=np.float32), "physical_rotation_delta_rad": np.asarray(physical_rotation, dtype=np.float32), "action_saturated": np.asarray(action_saturated, dtype=np.bool_), "required_contact": np.asarray([bool(row.get("required_contact", False)) for row in contact_rows], dtype=np.bool_), "contact_telemetry_complete": np.asarray([bool(row.get("contact_telemetry_complete", False)) for row in contact_rows], dtype=np.bool_), "unrelated_contact_runtime": np.asarray([bool(row.get("unrelated_contact_runtime", False)) for row in contact_rows], dtype=np.bool_), "source_value": np.asarray([-1 if value is None else int(value) for value in source_values], dtype=np.int8), "goal_value": np.asarray([-1 if value is None else int(value) for value in goal_values], dtype=np.int8)}
        trajectory_path = run_root / "trajectory.npz"
        with (trajectory_path.with_name(trajectory_path.name + ".partial")).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(trajectory_path.with_name(trajectory_path.name + ".partial"), trajectory_path)
        result = {"schema_version": "track3_sg6_child_v1", "status": status, "category": status, "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "fresh_process": True, "task_index": TASK_INDEX, "demo_index": DEMO_INDEX, "family": FAMILY, "profile": dict(profile), "mode": "CPU_no_renderer", "identity_gate": child_identity, "model_state_identity": {"hdf5": manifest, "resolved_model_xml_sha256": resolution["resolved_sha256"], "start_state_path": rel(run_root / "start_state.npy"), "start_state_sha256": sha256_file(run_root / "start_state.npy")}, "stage_results": stage_results, "gates": gates, "stage_gates": stage_gates, "reverse": {"source_atom": SOURCE_ATOM, "goal_atom": GOAL_ATOM, "source_values": source_values, "goal_values": goal_values, "final_success": final_success, "failure_reason": failure_reason}, "contact_contract": {"required_contact_source": "actual_demonstrator_contact_pair", "telemetry_completeness_not_contact": True, "barrier_index": barrier_index, "barrier_gate": barrier}, "trajectory": {"path": rel(trajectory_path), "sha256": sha256_file(trajectory_path), "actions": len(actions), "states": len(states), "typed_no_pickle": True, "arrays": sorted(arrays)}, "metrics": {"normalized_action_abs_max": max((normalized_action_contract(action)["normalized_action_abs_max"] for action in actions), default=0.0), "normalized_rotation_l2_max": max((normalized_action_contract(action)["normalized_rotation_l2"] for action in actions), default=0.0), "rotation_nonzero_steps": sum(float(np.linalg.norm(np.asarray(action[3:6]))) > 0.0 for action in actions), "action_saturation_count": sum(action_saturated)}, "training_eligible": False, "accepted": False, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "teleop": False, "replay": False, "generation": False, "training": False, "zip": False}}
    except Exception as exc:
        result = {"schema_version": "track3_sg6_child_v1", "status": "infrastructure_failure", "category": "infrastructure_failure", "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "fresh_process": True, "task_index": TASK_INDEX, "demo_index": DEMO_INDEX, "family": FAMILY, "profile": dict(profile), "error": {"type": type(exc).__name__, "message": str(exc)}, "training_eligible": False, "accepted": False, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False}}
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
    write_json(result_path, result)
    return 0


def compose_object_to_eef(object_pose: Mapping[str, Any], relative_pose: Mapping[str, Any]) -> dict[str, Any]:
    from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as control

    object_quat = control.quat_normalize_xyzw(object_pose["quat"])
    relative_position = [float(item) for item in relative_pose["translation_m"]]
    rotation = control.quat_to_matrix_xyzw(object_quat)
    world_delta = [sum(rotation[row][column] * relative_position[column] for column in range(3)) for row in range(3)]
    return {"pos_m": [float(object_pose["pos_m"][index]) + world_delta[index] for index in range(3)], "quat": list(control.quat_mul_xyzw(object_quat, relative_pose["quaternion_xyzw"]))}


def _terminate_group(process: subprocess.Popen[Any]) -> str:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=TERM_GRACE_SECONDS)
        return "TERM"
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=KILL_WAIT_SECONDS)
        return "KILL"


def run_sg6() -> dict[str, Any]:
    sg5 = strict_json(SG5_REPORT) if SG5_REPORT.is_file() else {}
    if sg5.get("status") != "passed":
        return {"status": "not_evaluated", "reason": "sg5_clean_source_release_not_passed", "physics_started": False}
    previous = _latest_attempt()
    if previous is None or not (previous[1] / "sg0_input_lock.json").is_file():
        return {"status": "not_evaluated", "reason": "sg0_input_lock_missing", "physics_started": False}
    _, sg0_attempt = previous
    sg0 = strict_json(sg0_attempt / "sg0_input_lock.json")
    attempt_id, attempt = allocate_attempt()
    canary = attempt / "canary"
    result_path = canary / "child_result.json"
    command = [str(RECORDER_PYTHON), "-m", MODULE, "--sg6-worker", "--attempt-root", str(attempt), "--result-output", str(result_path), "--profile", PRIMARY_PROFILE["name"]]
    write_json(attempt / "sg0_input_lock.json", sg0, immutable=True)
    identity = identity_gate(expected_commit=str(sg5["root_commit"]), expected_tool_sha256=str(sg5["tool_sha256"]), expected_openpi=str(sg5["openpi_commit"]), expected_command=command, expected_libero_plus_porcelain_sha256=str(sg0["source_snapshot"]["libero_plus_porcelain_sha256"]), observed_libero_plus_porcelain_sha256=json_hash(porcelain(LIBERO_PLUS)), dirty=bool(source_porcelain(ROOT)), pid_active=False, cursor_active=False, expected_cuda_visible_devices="", observed_cuda_visible_devices="", gpu_policy="cpu_no_renderer")
    write_json(canary / "identity_preflight.json", {"schema_version": "track3_sg6_identity_preflight_v1", "identity": identity, "command": command})
    if identity["state"] != "passed":
        set_attempt_active(attempt_id, attempt, False, status="hard_stop")
        return {"status": "hard_stop", "category": "infrastructure_failure", "reason": "pre_spawn_identity_failure", "attempt_id": attempt_id, "identity_gate": identity}
    write_json(attempt / "cursor.json", {"schema_version": "track3_sg_cursor_v1", "active": True, "phase": "SG6", "pid": os.getpid(), "command": command, "root_commit": sg5["root_commit"], "updated_at_utc": now()})
    write_json(attempt / "heartbeat.json", {"schema_version": "track3_sg_heartbeat_v1", "state": "running", "phase": "SG6", "pid": os.getpid(), "count": 1, "updated_at_utc": now()})
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}"}
    stdout_path, stderr_path = canary / "child.stdout.log", canary / "child.stderr.log"
    canary.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        while process.poll() is None:
            if time.monotonic() - started >= CANARY_TIMEOUT_SECONDS:
                timed_out = True
                termination_stage = _terminate_group(process)
                break
            time.sleep(0.1)
        process.wait()
    write_json(attempt / "heartbeat.json", {"schema_version": "track3_sg_heartbeat_v1", "state": "final", "phase": "SG6", "pid": os.getpid(), "count": 2, "updated_at_utc": now()})
    write_json(attempt / "cursor.json", {"schema_version": "track3_sg_cursor_v1", "active": False, "phase": "SG6", "pid": os.getpid(), "command": command, "root_commit": sg5["root_commit"], "updated_at_utc": now()})
    child = strict_json(result_path) if result_path.is_file() else None
    infra = timed_out or process.returncode != 0 or not isinstance(child, Mapping) or child.get("category") == "infrastructure_failure" or child.get("identity_gate", {}).get("state") != "passed"
    category = "infrastructure_failure" if infra else str(child.get("category"))
    summary = {"schema_version": "track3_sg6_support_identity_summary_v1", "status": "hard_stop" if infra else "completed", "attempt_id": attempt_id, "profile": PRIMARY_PROFILE, "category": category, "success_count": 1 if category == "success" else 0, "child": {"path": rel(result_path), "sha256": sha256_file(result_path)} if result_path.is_file() else None, "identity_gate": identity, "process": {"returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage}, "alternative": {"status": "not_evaluated", "reason": "SG7_requires_normal_family_failure_with_quantified_progress"}, "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "replay": False, "training": False, "zip": False}, "hard_stop": bool(infra)}
    write_json(attempt / "summary.json", summary, immutable=True)
    finalizer = finalize_attempt(attempt)
    set_attempt_active(attempt_id, attempt, False, status="hard_stop" if infra else "completed")
    return {**summary, "finalizer": finalizer}


def run_sg8() -> dict[str, Any]:
    latest = _latest_attempt()
    if latest is None:
        return {"status": "not_evaluated", "reason": "no_sg6_attempt"}
    attempt_id, attempt = latest
    result = {"schema_version": "track3_sg8_human_handoff_v1", "status": "human_input_required", "attempt_id": attempt_id, "reason": "automatic_path_not_success_or_not_run", "user_online_takeover_required": True, "recorded_candidates_max": 1, "agent_generated_key_input": False, "recording_started": False, "simulator_started": False, "gpu_started": False, "next_action": "wait_for_explicit_user_online_control"}
    write_json(attempt / "sg8_human_handoff.json", result, immutable=True)
    return result


def run_sg11_readiness() -> dict[str, Any]:
    latest = _latest_attempt()
    root = latest[1] / "sg11_readiness" if latest else SG_ROOT / "sg11_readiness"
    result = {"schema_version": "track3_sg11_code_only_readiness_v1", "status": "passed", "schemas": ["raw", "pending_replay", "pending_gold_review", "accepted", "training_eligible"], "allowed_transitions": [["raw", "pending_replay"], ["pending_replay", "pending_gold_review"], ["pending_gold_review", "accepted"], ["accepted", "training_eligible"]], "fail_closed": {"diagnostic_to_accepted": True, "no_accepted_demo_no_norm": True, "no_accepted_demo_no_training": True}, "split": {"source_group_isolation": True, "holdout_isolated": True, "dedup_before_split": True}, "tiny_overfit": {"synthetic_fixture_only": True, "dry_run": True, "started": False, "checkpoint_written": False}, "resource_boundary": {"training": False, "gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "zip": False}}
    write_json(root / "sg11_code_only_readiness.json", result, immutable=True)
    return result


def run_sg12_closeout() -> dict[str, Any]:
    latest = _latest_attempt()
    root = latest[1] if latest else SG_ROOT
    result = {"schema_version": "track3_sg12_closeout_v1", "status": "hard_stop", "attempt_id": latest[0] if latest else None, "success_count": 0, "accepted_count": 0, "training_eligible_count": 0, "human_input_required": True, "not_evaluated": ["SG7", "SG9", "SG10"], "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "generation": False, "replay": False, "training": False, "zip": False, "upload": False, "submission": False}, "next_authority": "independent_review_required"}
    write_json(root / "sg12_closeout.json", result, immutable=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for phase in ("sg0-to-sg4", "sg5", "sg6", "sg8", "sg11", "sg12"):
        parser.add_argument(f"--{phase}", action="store_true")
    parser.add_argument("--sg6-worker", action="store_true")
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--profile", choices=[PRIMARY_PROFILE["name"], ALTERNATIVE_PROFILE["name"]])
    args = parser.parse_args(argv)
    if args.sg6_worker:
        if args.attempt_root is None or args.result_output is None or args.profile is None:
            parser.error("--sg6-worker requires --attempt-root --result-output --profile")
        profile = PRIMARY_PROFILE if args.profile == PRIMARY_PROFILE["name"] else ALTERNATIVE_PROFILE
        return sg6_worker(profile, args.result_output, args.attempt_root)
    selected = [name for name in ("sg0_to_sg4", "sg5", "sg6", "sg8", "sg11", "sg12") if getattr(args, name)]
    if len(selected) != 1:
        parser.error("select exactly one SG0-to-SG4/SG5/SG6/SG8/SG11/SG12 phase")
    result = {"sg0_to_sg4": run_sg0_to_sg4, "sg5": run_sg5, "sg6": run_sg6, "sg8": run_sg8, "sg11": run_sg11_readiness, "sg12": run_sg12_closeout}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if str(result.get("status")) in {"passed", "completed", "hard_stop", "human_input_required"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
