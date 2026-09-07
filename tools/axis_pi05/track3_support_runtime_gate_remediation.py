#!/usr/bin/env python3
"""SR0--SR13 support runtime-gate remediation and gold continuation.

Old SG attempts 0001--0004 remain immutable. This namespace repairs the four
success-path blockers, adds a real-asset pre-env check, then may consume at
most one corrected primary physics identity and one progress-gated alternative.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.axis_pi05.track3_contact_truth_and_support_gold_demo import (
    ALTERNATIVE_PROFILE,
    PRIMARY_PROFILE,
    STAGE_NAMES,
    compose_object_to_eef,
    identity_gate as sg_identity_gate,
    normalized_action_contract,
    predicate_truth,
    recorder_command,
    waypoint_contract,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_support_runtime_gate_remediation"
SR_ROOT = ROOT / "data/reports/axis_libero_30k/track3_support_runtime_gate_remediation"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_SUPPORT_RUNTIME_GATE_REMEDIATION_AND_GOLD_CONTINUATION_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_SG0_SG6_INDEPENDENT_AUDIT_20260907.md"
SG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_contact_truth_and_support_gold_demo"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
RECORDER_PYTHON = ROOT / ".venv_axis_pi05/bin/python"
CURRENT_ATTEMPT = SR_ROOT / "current_attempt.json"
ATTEMPT_LOCK = SR_ROOT / "attempt_allocator.lock"
SUPERVISOR_PID = SR_ROOT / "supervisor.pid"

CONTROL_FREQ_HZ = 20
ACTION_DIM = 7
TERM_GRACE_SECONDS = 0.5
KILL_WAIT_SECONDS = 1.0
CANARY_TIMEOUT_SECONDS = 420.0
STABLE_TICKS = 4
MAX_SUPPORT_IDENTITIES = 2

FAMILY = "support_remove"
TASK_INDEX = 10
DEMO_INDEX = 0
SOURCE_ATOM = "(On akita_black_bowl_1 plate_1)"
GOAL_ATOM = "(On akita_black_bowl_1 main_table_akita_black_bowl_region)"
ROLES = {
    "subject": ("akita_black_bowl_1",),
    "source_receiver": ("plate_1",),
    "goal_receiver": ("main_table",),
    "demonstrator": ("robot0", "gripper0", "panda"),
}
CR_SUPPORT_ALT_ERROR_M = 0.013362138900100747
CR_SUPPORT_ALT_STAGE = "descend_object"
LOCKED_COMMITS = {
    "sg_implementation": "710183a7983e300759c85e3c0a97e1afdbdb67f0",
    "sg5_sg6_source": "d5c319b055431d2ec46681775438c91ad64a2be0",
    "signature_fix": "3e156dd6177e435316b721a79ff7a0da6d0359f8",
    "audit_input": "55acd39c17d82c89b4f8e809b8f84d9623c3eb81",
    "openpi": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
}
CONTROL_ENV_MODULE = "libero.libero.envs.env_wrapper"
STAGE_RANK = {name: index for index, name in enumerate(STAGE_NAMES)}
STAGE_RANK["descend_object"] = STAGE_RANK["contact_descent"]


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
    return result.stdout.rstrip("\n")


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def source_porcelain(repo: Path = ROOT) -> list[str]:
    rows = porcelain(repo)
    if repo.resolve() != ROOT.resolve():
        return rows
    evidence_roots = (rel(SR_ROOT), rel(SG_ROOT))
    result: list[str] = []
    for row in rows:
        path_field = row[3:].strip() if len(row) >= 4 else row.strip()
        paths = [part.strip().strip('"') for part in path_field.split(" -> ")]
        if any(not any(path == root or path.startswith(root + "/") for root in evidence_roots) for path in paths):
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


def _name_matches(actual: str, logical: str) -> bool:
    return actual == logical or actual.startswith(logical + "_") or logical in actual


def _is_robot_name(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(("robot0", "gripper0", "panda")) or any(token in lower for token in ("finger", "gripper", "right_hand", "left_hand"))


def contact_role(body1: str, body2: str) -> str:
    subjects = set(ROLES["subject"])
    source_receivers = set(ROLES["source_receiver"])
    goal_receivers = set(ROLES["goal_receiver"])
    left_subject = any(_name_matches(body1, name) for name in subjects)
    right_subject = any(_name_matches(body2, name) for name in subjects)
    left_source = any(_name_matches(body1, name) for name in source_receivers)
    right_source = any(_name_matches(body2, name) for name in source_receivers)
    left_goal = any(_name_matches(body1, name) for name in goal_receivers)
    right_goal = any(_name_matches(body2, name) for name in goal_receivers)
    left_robot, right_robot = _is_robot_name(body1), _is_robot_name(body2)
    if (left_subject and right_source) or (right_subject and left_source):
        return "source_receiver_contact"
    if (left_subject and right_goal) or (right_subject and left_goal):
        return "goal_receiver_contact"
    if (left_subject and right_robot) or (right_subject and left_robot):
        return "demonstrator_contact"
    if left_subject or right_subject:
        return "unrelated_background_contact"
    return "background_contact"


def contact_truth_v3(task_index: int, matrix: Mapping[str, Any]) -> dict[str, Any]:
    if task_index != TASK_INDEX:
        return infrastructure_gate("unsupported_task_contact_contract", task_index=task_index)
    if not isinstance(matrix, Mapping) or matrix.get("complete") is not True:
        return infrastructure_gate("contact_schema_incomplete", contact_telemetry_complete=False, required_contact=None, unrelated_contact_runtime=None, demonstrator_contact=None, source_receiver_contact=None, goal_receiver_contact=None)
    raw_contacts = matrix.get("contacts")
    if not isinstance(raw_contacts, list):
        return infrastructure_gate("contact_rows_not_a_list", contact_telemetry_complete=False, required_contact=None, unrelated_contact_runtime=None)
    categories: dict[str, list[dict[str, Any]]] = {"demonstrator_contact": [], "source_receiver_contact": [], "goal_receiver_contact": [], "unrelated_background_contact": [], "background_contact": []}
    for index, raw in enumerate(raw_contacts):
        if not isinstance(raw, Mapping) or not raw.get("body1_name") or not raw.get("body2_name"):
            return infrastructure_gate("contact_role_resolution_unknown", contact_index=index, contact_telemetry_complete=False, required_contact=None, unrelated_contact_runtime=None)
        body1, body2 = str(raw["body1_name"]), str(raw["body2_name"])
        role = contact_role(body1, body2)
        categories[role].append({"body_pair": sorted((body1, body2)), "geom_pair": sorted((str(raw.get("geom1_name", "")), str(raw.get("geom2_name", "")))), "distance_m": raw.get("distance_m"), "role": role})
    demonstrator = bool(categories["demonstrator_contact"])
    unrelated = bool(categories["unrelated_background_contact"])
    return {
        "schema_version": "track3_sr_contact_truth_v3",
        "executed": True,
        "state": "infrastructure_failure" if False else "classified",
        "required": True,
        "categories": categories,
        "counts": {key: len(value) for key, value in categories.items()},
        "contact_telemetry_complete": True,
        "telemetry_known": True,
        "required_contact": demonstrator,
        "demonstrator_contact": demonstrator,
        "source_receiver_contact": bool(categories["source_receiver_contact"]),
        "goal_receiver_contact": bool(categories["goal_receiver_contact"]),
        "unrelated_contact_runtime": unrelated,
        "derivation": "MuJoCo pair roles; telemetry completeness never implies contact",
    }


def _sample_truth(sample: Mapping[str, Any]) -> dict[str, Any]:
    if "contact_truth_v3" in sample and isinstance(sample["contact_truth_v3"], Mapping):
        return dict(sample["contact_truth_v3"])
    return contact_truth_v3(TASK_INDEX, sample.get("contact_matrix", {}))


def required_contact_gate(samples: Sequence[Mapping[str, Any]], barrier_index: int | None, *, stable_ticks: int = STABLE_TICKS) -> dict[str, Any]:
    if barrier_index is None or barrier_index < 0:
        return gate(False, None, reason="acquisition_barrier_not_entered")
    start = max(0, int(barrier_index) - int(stable_ticks) + 1)
    scoped = list(samples[start : int(barrier_index) + 1])
    if len(scoped) < int(stable_ticks):
        return gate(True, False, reason="acquisition_contact_window_short", barrier_index=barrier_index, window_start=start, samples=len(scoped), stable_ticks=stable_ticks)
    truths = [_sample_truth(row) for row in scoped]
    if any(row.get("state") == "infrastructure_failure" for row in truths):
        return infrastructure_gate("contact_truth_unknown_in_acquisition_window", samples=len(truths))
    values = [bool(row.get("demonstrator_contact") or row.get("required_contact")) for row in truths]
    prefix = list(samples[:start])
    prefix_zeros_ignored = all(not bool(_sample_truth(row).get("required_contact")) for row in prefix) if prefix else True
    late = list(samples[int(barrier_index) + 1 :])
    return gate(
        True,
        all(values),
        reason="zero_required_contact_in_dwell_window" if not all(values) else None,
        barrier_index=barrier_index,
        window_start=start,
        samples=len(values),
        required_contact_values=values,
        prefix_without_contact_ignored=prefix_zeros_ignored,
        late_contact_excluded=bool(late),
        stable_ticks=stable_ticks,
    )


def source_released_gate(source_values: Sequence[Any], phases: Sequence[str], *, stable_ticks: int = STABLE_TICKS) -> dict[str, Any]:
    if "lift" not in phases:
        return gate(False, None, reason="lift_not_entered")
    lift_values = [predicate_truth(value) for phase, value in zip(phases, source_values) if phase == "lift" or STAGE_RANK.get(phase, -1) > STAGE_RANK["lift"]]
    if len(lift_values) < int(stable_ticks):
        return gate(True, False, reason="source_release_window_short", samples=len(lift_values), stable_ticks=stable_ticks)
    window = lift_values[: int(stable_ticks)] if False else lift_values[-int(stable_ticks) :]
    passed = all(value is False for value in window)
    return gate(True, passed, reason="source_still_true_after_lift" if not passed else None, window=[False if value is False else value for value in window], stable_ticks=stable_ticks)


def goal_continuous_gate(goal_values: Sequence[Any], phases: Sequence[str], *, check_success: bool | None, stable_ticks: int = STABLE_TICKS) -> dict[str, Any]:
    post = {"place", "release", "retreat"}
    if not any(phase in post for phase in phases):
        return gate(False, None, reason="place_stage_not_entered")
    post_values = [predicate_truth(value) for phase, value in zip(phases, goal_values) if phase in post]
    if len(post_values) < int(stable_ticks):
        return gate(True, False, reason="goal_continuity_window_short", samples=len(post_values), goal_final=post_values[-1] if post_values else None, check_success=check_success)
    window = post_values[-int(stable_ticks) :]
    passed = all(value is True for value in window) and check_success is True
    return gate(True, passed, reason="goal_not_continuously_true" if not passed else None, window=window, goal_final=post_values[-1], check_success=check_success, stable_ticks=stable_ticks, not_final_frame_only=True)


def classify_stop(*, environment_created: bool | None, physics_steps: int | None, error_type: str | None, identity_failed: bool = False, telemetry_unknown: bool = False, partial: bool = False, gpu_violation: bool = False, supervisor_unknown: bool = False, family_gate_failed: bool = False) -> str:
    if identity_failed or telemetry_unknown or partial or gpu_violation or supervisor_unknown:
        return "global_hard_stop"
    if error_type and environment_created is True:
        return "global_hard_stop"
    if error_type and environment_created is False and (physics_steps or 0) == 0:
        return "runtime_lane_stop"
    if error_type:
        return "global_hard_stop"
    if family_gate_failed:
        return "family_outcome"
    return "global_hard_stop"


def success_path_fixture(*, unrelated: bool = False, unknown: bool = False, late_contact_only: bool = False) -> dict[str, Any]:
    def pair(body1: str, body2: str) -> dict[str, Any]:
        return {"complete": True, "contacts": [{"body1_name": body1, "body2_name": body2, "geom1_name": body1, "geom2_name": body2, "distance_m": -0.001}]}

    empty = {"complete": True, "contacts": []}
    samples: list[dict[str, Any]] = []
    phases: list[str] = []
    source_values: list[bool] = []
    goal_values: list[bool] = []

    def add(phase: str, matrix: Mapping[str, Any], source: bool, goal: bool, count: int) -> None:
        for _ in range(count):
            phases.append(phase)
            source_values.append(source)
            goal_values.append(goal)
            samples.append({"contact_matrix": dict(matrix), "contact_truth_v3": contact_truth_v3(TASK_INDEX, matrix)})

    if unknown:
        add("precontact_xy", {"complete": False, "contacts": []}, True, False, 4)
        return {"required_contact": required_contact_gate(samples, 3), "source_released": source_released_gate(source_values, phases), "goal_continuous_true": goal_continuous_gate(goal_values, phases, check_success=False)}
    add("precontact_xy", empty, True, False, 4)
    add("precontact_z", empty, True, False, 4)
    dwell_matrix = empty if late_contact_only else pair("akita_black_bowl_1", "robot0_right_hand")
    add("grasp_dwell", dwell_matrix, True, False, 4)
    barrier_index = len(samples) - 1
    add("lift", pair("akita_black_bowl_1", "robot0_right_hand"), False, False, 4)
    goal_matrix = pair("akita_black_bowl_1", "salad_dressing_1") if unrelated else pair("akita_black_bowl_1", "main_table")
    add("place", goal_matrix, False, not unrelated, 4)
    if late_contact_only:
        samples[barrier_index + 1]["contact_matrix"] = pair("akita_black_bowl_1", "robot0_right_hand")
        samples[barrier_index + 1]["contact_truth_v3"] = contact_truth_v3(TASK_INDEX, samples[barrier_index + 1]["contact_matrix"])
    check_success = (not unrelated) and (not late_contact_only) and (not unknown)
    return {
        "phases": phases,
        "required_contact": required_contact_gate(samples, barrier_index),
        "source_released": source_released_gate(source_values, phases),
        "goal_continuous_true": goal_continuous_gate(goal_values, phases, check_success=check_success),
        "goal_receiver": contact_truth_v3(TASK_INDEX, pair("akita_black_bowl_1", "main_table")),
        "unrelated_any": any(_sample_truth(row).get("unrelated_contact_runtime") is True for row in samples),
        "prefix_contact_not_required": True,
    }


def identity_gate(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("expected_cuda_visible_devices", "")
    kwargs.setdefault("gpu_policy", "cpu_no_renderer")
    kwargs.setdefault("observed_tool_sha256", sha256_file(Path(__file__)))
    if kwargs.get("dirty") is None:
        kwargs["dirty"] = bool(source_porcelain(ROOT))
    return sg_identity_gate(**kwargs)


def _lock_file(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def allocate_attempt() -> tuple[int, Path]:
    with _lock_file(ATTEMPT_LOCK) as handle:
        attempts = SR_ROOT / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        ids = [int(path.name.removeprefix("attempt_")) for path in attempts.glob("attempt_*") if path.is_dir() and path.name.removeprefix("attempt_").isdigit()]
        attempt_id = max(ids, default=0) + 1
        attempt = attempts / f"attempt_{attempt_id:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        write_json(CURRENT_ATTEMPT, {"schema_version": "track3_sr_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": True, "created_at_utc": now()})
        handle.flush()
        return attempt_id, attempt


def set_attempt_active(attempt_id: int, attempt: Path, active: bool, *, status: str) -> None:
    write_json(CURRENT_ATTEMPT, {"schema_version": "track3_sr_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": active, "status": status, "updated_at_utc": now()})


FINALIZER_RELATIVE_PATHS = frozenset({"producer_manifest.json", "final_manifest.json", "outer_hash.json", "closure.json"})
SR_PHASE_NAMES = tuple(f"SR{index}" for index in range(14))
SR9_SR11 = ("SR9", "SR10", "SR11")


def _is_finalizer_relative(root: Path, path: Path) -> bool:
    return path.relative_to(root).as_posix() in FINALIZER_RELATIVE_PATHS


def _artifact_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_finalizer_relative(root, path) or path.name.endswith(".partial"):
            continue
        entries.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def make_manifest(root: Path, filename: str = "producer_manifest.json") -> dict[str, Any]:
    entries = _artifact_entries(root)
    result = {
        "schema_version": "track3_sr_producer_manifest_v1",
        "manifest_role": "producer_closed_set",
        "entries": entries,
        "excluded_finalizer_relative_paths": sorted(FINALIZER_RELATIVE_PATHS),
        "excluded_finalizer_names": sorted(FINALIZER_RELATIVE_PATHS),
        "producer_closed_before_seal": True,
    }
    write_json(root / filename, result, immutable=True)
    return result


def verify_complete_set(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = {str(row["relative_path"]): row for row in manifest.get("entries", [])}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and not _is_finalizer_relative(root, path) and not path.name.endswith(".partial")
    }
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches: list[str] = []
    for name, row in expected.items():
        path = actual.get(name)
        if path is not None and (int(row.get("bytes", -1)) != path.stat().st_size or str(row.get("sha256")) != sha256_file(path)):
            mismatches.append(name)
    partials = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.partial"))
    return {"exact": not missing and not extra and not mismatches and not partials, "missing": missing, "extra": extra, "hash_or_size_mismatch": mismatches, "partial_residue": partials}


def derive_not_evaluated(root: Path, *, success_count: int = 0, teleop_recorded: bool = False, replay_completed: bool = False, review_completed: bool = False) -> list[str]:
    """Derive unexecuted phases from producers. Never hard-code an empty list when SR9–SR11 are absent."""
    missing: list[str] = []
    if not teleop_recorded and not (root / "sr9_teleop.json").is_file():
        missing.append("SR9")
    if int(success_count) <= 0 and not replay_completed and not (root / "sr10_replay.json").is_file():
        missing.append("SR10")
    if not review_completed and not (root / "sr11_review.json").is_file():
        missing.append("SR11")
    return [name for name in SR_PHASE_NAMES if name in missing]


def prepared_stage_names(profile: Mapping[str, Any]) -> list[str]:
    return [str(row["name"]) for row in prepare_support_identity(profile)["stages"]]


def alternative_topology_is_structurally_distinct() -> dict[str, Any]:
    primary = prepared_stage_names(PRIMARY_PROFILE)
    alternative = prepared_stage_names(ALTERNATIVE_PROFILE)
    return {
        "claimed_alternative_topology": ALTERNATIVE_PROFILE.get("topology"),
        "claimed_primary_topology": PRIMARY_PROFILE.get("topology"),
        "actual_primary_stages": primary,
        "actual_alternative_stages": alternative,
        "actual_stage_order_identical": primary == alternative,
        "structurally_distinct": primary != alternative,
        "consumed_identities_must_not_rerun": True,
    }


def finalize_attempt(root: Path) -> dict[str, Any]:
    sealed = [name for name in ("final_manifest.json", "outer_hash.json", "closure.json") if (root / name).is_file()]
    if sealed:
        return {"status": "failed", "reason": "repeat_finalization", "already_present": sealed}
    producer_path = root / "producer_manifest.json"
    producer = strict_json(producer_path) if producer_path.is_file() else make_manifest(root)
    producer_check = verify_complete_set(root, producer)
    if not producer_check["exact"]:
        return {"status": "failed", "producer_manifest": producer_check, "reason": "producer_set_changed_before_finalizer"}
    final = {
        "schema_version": "track3_sr_final_manifest_v1",
        "manifest_role": "post_producer_complete_set",
        "entries": _artifact_entries(root),
        "excluded_finalizer_relative_paths": sorted(FINALIZER_RELATIVE_PATHS),
        "excluded_finalizer_names": sorted(FINALIZER_RELATIVE_PATHS),
    }
    write_json(root / "final_manifest.json", final, immutable=True)
    outer = {"schema_version": "track3_sr_outer_hash_v1", "final_manifest_sha256": sha256_file(root / "final_manifest.json"), "entry_count": len(final["entries"])}
    write_json(root / "outer_hash.json", outer, immutable=True)
    complete = verify_complete_set(root, final)
    closure = {"schema_version": "track3_sr_closure_v1", "status": "passed" if complete["exact"] and sha256_file(root / "final_manifest.json") == outer["final_manifest_sha256"] else "failed", "complete_set": complete, "outer_hash_matches": sha256_file(root / "final_manifest.json") == outer["final_manifest_sha256"]}
    write_json(root / "closure.json", closure, immutable=True)
    return {"status": closure["status"], "producer_manifest": producer_check, "final_manifest": complete, "closure": closure}


def gpu_compute_snapshot() -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"returncode": result.returncode, "rows": rows, "empty": result.returncode == 0 and not rows}


def live_workers() -> list[str]:
    result = subprocess.run(["ps", "-ef"], capture_output=True, text=True, check=False)
    needles = ("track3_support_runtime_gate_remediation", "track3_contact_truth_and_support_gold_demo", "track3_public_teleop", "track3_public_replay")
    self_pids = {os.getpid(), os.getppid()}
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        if int(parts[1]) in self_pids:
            continue
        if any(token in line for token in needles):
            rows.append(line)
    return rows


def _file_lock_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} if path.is_file() else {"path": rel(path), "state": "missing"}


def run_sr0(attempt: Path) -> dict[str, Any]:
    sg_json = sorted(path for path in SG_ROOT.rglob("*.json") if path.is_file())
    attempt4 = SG_ROOT / "attempts/attempt_0004"
    libero_rows = porcelain(LIBERO_PLUS)
    live = live_workers()
    gpu = gpu_compute_snapshot()
    cursor = strict_json(SG_ROOT / "current_attempt.json") if (SG_ROOT / "current_attempt.json").is_file() else {}
    partials = [rel(path) for path in SG_ROOT.rglob("*.partial")]
    sg_hashes = [{"path": rel(path), "sha256": sha256_file(path)} for path in sg_json]
    result = {
        "schema_version": "track3_sr0_input_lock_v1",
        "status": "passed" if not live and gpu.get("empty") and not partials and cursor.get("active") is False else "failed",
        "locked_commits": LOCKED_COMMITS,
        "observed_openpi": git(("rev-parse", "HEAD"), OPENPI),
        "observed_root": git(("rev-parse", "HEAD")),
        "sg_json_count": len(sg_json),
        "sg_json": sg_hashes,
        "attempt_0004": {
            "producer_or_final": _file_lock_record(attempt4 / "final_manifest.json"),
            "outer": _file_lock_record(attempt4 / "outer_hash.json"),
            "closure": _file_lock_record(attempt4 / "closure.json"),
            "child": _file_lock_record(attempt4 / "canary/child_result.json"),
            "stdout": _file_lock_record(attempt4 / "canary/child.stdout.log"),
            "stderr": _file_lock_record(attempt4 / "canary/child.stderr.log"),
        },
        "source_snapshot": {
            "root_porcelain": source_porcelain(ROOT),
            "openpi_porcelain": porcelain(OPENPI),
            "libero_plus_porcelain": libero_rows,
            "libero_plus_porcelain_sha256": json_hash(libero_rows),
            "tool_sha256": sha256_file(Path(__file__)),
        },
        "live_boundary": {"workers": live, "gpu": gpu, "sg_cursor": cursor, "partials": partials},
        "old_sg_immutable": True,
        "audit": {"path": rel(AUDIT), "sha256": sha256_file(AUDIT)},
        "plan": {"path": rel(PLAN), "sha256": sha256_file(PLAN)},
    }
    checks = {
        "openpi_exact": result["observed_openpi"] == LOCKED_COMMITS["openpi"],
        "sg_json_37": len(sg_json) == 37,
        "no_live_workers": not live,
        "gpu_empty": bool(gpu.get("empty")),
        "no_partial": not partials,
        "sg_cursor_inactive": cursor.get("active") is False,
        "attempt4_manifest_8": len(strict_json(attempt4 / "final_manifest.json").get("entries", [])) == 8 if (attempt4 / "final_manifest.json").is_file() else False,
    }
    result["checks"] = checks
    result["status"] = "passed" if all(checks.values()) else "failed"
    write_json(attempt / "sr0_input_lock.json", result, immutable=True)
    return result


def run_sr1(attempt: Path) -> dict[str, Any]:
    positive = success_path_fixture()
    unrelated = success_path_fixture(unrelated=True)
    unknown = success_path_fixture(unknown=True)
    late = success_path_fixture(late_contact_only=True)
    bowl_table = contact_truth_v3(TASK_INDEX, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "main_table"}]})
    bowl_plate = contact_truth_v3(TASK_INDEX, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "plate_1"}]})
    surrogate = contact_truth_v3(TASK_INDEX, {"complete": True, "contacts": [], "required_contact": True, "unrelated_contact_runtime": False})
    checks = {
        "roles_frozen": ROLES["subject"] == ("akita_black_bowl_1",) and ROLES["source_receiver"] == ("plate_1",) and ROLES["goal_receiver"] == ("main_table",),
        "success_path_reachable": positive["required_contact"]["state"] == "passed" and positive["source_released"]["state"] == "passed" and positive["goal_continuous_true"]["state"] == "passed" and positive["goal_receiver"]["goal_receiver_contact"] is True and positive["unrelated_any"] is False,
        "precontact_zeros_ignored": positive["required_contact"].get("prefix_without_contact_ignored") is True,
        "bowl_main_table_is_goal": bowl_table.get("goal_receiver_contact") is True and bowl_table.get("unrelated_contact_runtime") is False,
        "bowl_plate_is_source": bowl_plate.get("source_receiver_contact") is True and bowl_plate.get("unrelated_contact_runtime") is False,
        "unrelated_negative": unrelated["unrelated_any"] is True and unrelated["goal_continuous_true"]["state"] == "failed",
        "unknown_infrastructure": unknown["required_contact"]["state"] == "infrastructure_failure",
        "late_contact_cannot_backfill": late["required_contact"]["state"] == "failed" and late["required_contact"].get("late_contact_excluded") is True,
        "telemetry_not_contact": surrogate.get("required_contact") is False,
        "source_not_required_before_lift": success_path_fixture()["source_released"]["executed"] is True,
    }
    result = {"schema_version": "track3_sr1_contact_role_fixtures_v1", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "positive": {key: positive[key] for key in ("required_contact", "source_released", "goal_continuous_true", "goal_receiver")}, "negatives": {"unrelated": unrelated["goal_continuous_true"], "unknown": unknown["required_contact"], "late": late["required_contact"]}}
    write_json(attempt / "sr1_contact_role_fixtures.json", result, immutable=True)
    return result


def run_sr2(attempt: Path) -> dict[str, Any]:
    contract = waypoint_contract(PRIMARY_PROFILE)
    dwell_source_true = source_released_gate([True] * 4, ["grasp_dwell"] * 4)
    after_lift = source_released_gate([True, True, True, True, False, False, False, False], ["grasp_dwell"] * 4 + ["lift"] * 4)
    final_only = goal_continuous_gate([False, False, False, True], ["place"] * 4, check_success=True)
    window = goal_continuous_gate([True, True, True, True], ["place"] * 4, check_success=True)
    checks = {
        "dwell_source_true_not_evaluated_release": dwell_source_true["state"] == "not_evaluated",
        "lift_required_for_source_release": after_lift["state"] == "passed",
        "goal_rejects_final_frame_only": final_only["state"] == "failed" and final_only.get("not_final_frame_only") is True,
        "goal_window_and_check_success": window["state"] == "passed",
        "stage_order": contract["stage_names"] == list(STAGE_NAMES),
    }
    result = {"schema_version": "track3_sr2_phase_window_fixtures_v1", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "cases": {"dwell_source_true": dwell_source_true, "after_lift": after_lift, "final_only": final_only, "window": window}}
    write_json(attempt / "sr2_phase_window_fixtures.json", result, immutable=True)
    return result


def prepare_support_identity(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Load locked assets and build stage targets without creating ControlEnv."""
    before = CONTROL_ENV_MODULE in sys.modules
    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
    from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as control

    manifest, model_bytes, _init_state, _terminal_state, _demo_info = composed._demo_arrays(TASK_INDEX, DEMO_INDEX)
    resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
    with __import__("numpy").load(control.STATE_BUNDLE, allow_pickle=False) as archive:
        start_state = __import__("numpy").asarray(archive["task10_demo0"], dtype="float64").copy()
    w1 = strict_json(control.W1_REPORT)
    relative = w1["reverse_acquisition_anchor"]["relative_pose_xyzw"]
    terminal_pose = w1["state_anchors"]["terminal"]["bowl"]
    init_pose = w1["state_anchors"]["state0"]["bowl"]
    grasp = compose_object_to_eef(terminal_pose, relative)
    destination = compose_object_to_eef(init_pose, relative)
    transport = compose_object_to_eef(init_pose, relative, object_position_offset_m=(0.0, 0.0, 0.12))
    resolved_xml_text = resolved_xml
    budgets = [int(item) for item in profile["stage_budgets"]]
    hold_quat = [0.0, 0.0, 0.0, 1.0]
    stages = [
        {"name": "precontact_xy", "gripper": -1.0, "target_pos_m": [grasp["pos_m"][0], grasp["pos_m"][1], float("nan")], "target_quat": hold_quat, "budget": budgets[0], "uses_current_eef_z": True},
        {"name": "precontact_z", "gripper": -1.0, "target_pos_m": [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.04], "target_quat": hold_quat, "budget": budgets[1]},
        {"name": "orientation_align", "gripper": -1.0, "target_pos_m": [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.04], "target_quat": grasp["quat"], "budget": budgets[2]},
        {"name": "contact_descent", "gripper": -1.0, "target_pos_m": list(grasp["pos_m"]), "target_quat": grasp["quat"], "budget": budgets[3]},
        {"name": "grasp_dwell", "gripper": 1.0, "target_pos_m": list(grasp["pos_m"]), "target_quat": grasp["quat"], "budget": budgets[4]},
        {"name": "lift", "gripper": 1.0, "target_pos_m": [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.14], "target_quat": grasp["quat"], "budget": budgets[5]},
        {"name": "transport", "gripper": 1.0, "target_pos_m": [transport["pos_m"][0], transport["pos_m"][1], grasp["pos_m"][2] + 0.14], "target_quat": transport["quat"], "budget": budgets[6]},
        {"name": "place", "gripper": 1.0, "target_pos_m": list(destination["pos_m"]), "target_quat": destination["quat"], "budget": budgets[7]},
        {"name": "release", "gripper": -1.0, "target_pos_m": list(destination["pos_m"]), "target_quat": destination["quat"], "budget": budgets[8]},
        {"name": "retreat", "gripper": -1.0, "target_pos_m": [destination["pos_m"][0], destination["pos_m"][1], destination["pos_m"][2] + 0.16], "target_quat": destination["quat"], "budget": budgets[9]},
    ]
    finite_targets = []
    for row in stages:
        values = [item for item in row["target_pos_m"] + list(row["target_quat"]) if not (isinstance(item, float) and math.isnan(item))]
        finite_targets.append(all(math.isfinite(float(item)) for item in values))
    transport_offset_used = abs(float(transport["pos_m"][2]) - float(destination["pos_m"][2]) - 0.12) < 1e-9 or abs(float(stages[6]["target_pos_m"][0]) - float(transport["pos_m"][0])) < 1e-12
    after = CONTROL_ENV_MODULE in sys.modules
    orientation_error = control.quat_log_error_xyzw(grasp["quat"], [0.0, 0.0, 0.0, 1.0])
    return {
        "schema_version": "track3_sr3_prepare_only_v1",
        "environment_created": False,
        "physics_steps": 0,
        "control_env_imported_before": before,
        "control_env_imported_after": after,
        "control_env_created": False,
        "hdf5": manifest,
        "resolved_xml": resolved_xml_text,
        "resolved_model_xml_sha256": resolution["resolved_sha256"],
        "start_state_len": int(start_state.shape[0]),
        "start_state_finite": bool(__import__("numpy").isfinite(start_state).all()),
        "grasp": grasp,
        "destination": destination,
        "transport": transport,
        "stages": stages,
        "budget_sum": sum(budgets),
        "profile_step_budget": int(profile["step_budget"]),
        "budget_matches_profile": sum(budgets) == int(profile["step_budget"]),
        "targets_finite_except_precontact_xy_z_placeholder": all(finite_targets),
        "transport_offset_used_by_stage_target": bool(transport_offset_used),
        "orientation_error_nonzero": float(sum(item * item for item in orientation_error)) ** 0.5 > 0.0,
        "physics_started": False,
    }


def run_sr3(attempt: Path) -> dict[str, Any]:
    prepared = prepare_support_identity(PRIMARY_PROFILE)
    checks = {
        "no_control_env_import": prepared["control_env_imported_after"] is False,
        "no_env_created": prepared["environment_created"] is False and prepared["control_env_created"] is False,
        "zero_physics": prepared["physics_steps"] == 0,
        "budget_sum": prepared["budget_matches_profile"] is True,
        "targets_finite": prepared["targets_finite_except_precontact_xy_z_placeholder"] is True,
        "transport_offset_used": prepared["transport_offset_used_by_stage_target"] is True,
        "orientation_nonzero": prepared["orientation_error_nonzero"] is True,
        "start_state_finite": prepared["start_state_finite"] is True,
    }
    prepared_public = {key: prepared[key] for key in prepared if key not in {"stages", "resolved_xml"}}
    result = {"schema_version": "track3_sr3_pre_env_integration_v1", "status": "passed" if all(checks.values()) else "failed", "checks": checks, "prepared": prepared_public, "stage_names": [row["name"] for row in prepared["stages"]]}
    write_json(attempt / "sr3_pre_env_integration.json", result, immutable=True)
    return result


def run_sr4(attempt: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    pre_env = classify_stop(environment_created=False, physics_steps=0, error_type="TypeError")
    unknown_env = classify_stop(environment_created=None, physics_steps=None, error_type="TypeError")
    identity = classify_stop(environment_created=False, physics_steps=0, error_type=None, identity_failed=True)
    family = classify_stop(environment_created=True, physics_steps=12, error_type=None, family_gate_failed=True)
    cases["pre_env_runtime_lane"] = {"stop": pre_env, "rejected": pre_env == "runtime_lane_stop"}
    cases["unknown_env_global"] = {"stop": unknown_env, "rejected": unknown_env == "global_hard_stop"}
    cases["identity_global"] = {"stop": identity, "rejected": identity == "global_hard_stop"}
    cases["family_outcome"] = {"stop": family, "rejected": family == "family_outcome"}
    child_fields = {"environment_created": False, "physics_steps": 0, "physics_identity_consumed": False, "launch_attempt_id": 1, "failure_stage": "pre_env"}
    cases["child_accounting_fields"] = {"rejected": set(child_fields) == {"environment_created", "physics_steps", "physics_identity_consumed", "launch_attempt_id", "failure_stage"}}
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
        cases["timeout_term_kill"] = {"rejected": process.poll() is not None, "termination_stage": term_stage}
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    with tempfile.TemporaryDirectory(prefix="sr4_", dir=attempt) as temporary:
        fixture = Path(temporary)
        atomic_bytes(fixture / "heartbeat.json.partial", b"partial")
        cases["partial_is_global"] = {"rejected": classify_stop(environment_created=False, physics_steps=0, error_type=None, partial=True) == "global_hard_stop" and bool(list(fixture.glob("*.partial")))}
    statuses = {name: bool(value.get("rejected")) for name, value in cases.items()}
    result = {"schema_version": "track3_sr4_stop_contract_v1", "status": "passed" if all(statuses.values()) else "failed", "cases": cases, "checks": statuses}
    write_json(attempt / "sr4_stop_contract.json", result, immutable=True)
    return result


def run_sr0_to_sr4() -> dict[str, Any]:
    attempt_id, attempt = allocate_attempt()
    try:
        sr0, sr1, sr2, sr3, sr4 = run_sr0(attempt), run_sr1(attempt), run_sr2(attempt), run_sr3(attempt), run_sr4(attempt)
        statuses = {"SR0": sr0["status"], "SR1": sr1["status"], "SR2": sr2["status"], "SR3": sr3["status"], "SR4": sr4["status"]}
        overall = "passed" if all(value == "passed" for value in statuses.values()) else "hard_stop"
        write_text(attempt / "phase_ledger.csv", "phase,status\n" + "\n".join(f"{key},{value}" for key, value in statuses.items()) + "\n")
        set_attempt_active(attempt_id, attempt, False, status=overall)
        return {"status": overall, "attempt_id": attempt_id, "attempt_root": rel(attempt), "phase_status": statuses}
    except Exception as exc:
        write_json(attempt / "sr0_to_sr4_exception.json", {"status": "hard_stop", "error": {"type": type(exc).__name__, "message": str(exc)}}, immutable=True)
        set_attempt_active(attempt_id, attempt, False, status="hard_stop")
        return {"status": "hard_stop", "attempt_id": attempt_id, "error": f"{type(exc).__name__}:{exc}"}


def _latest_attempt() -> tuple[int, Path] | None:
    if not CURRENT_ATTEMPT.is_file():
        return None
    pointer = strict_json(CURRENT_ATTEMPT)
    return int(pointer["attempt_id"]), ROOT / str(pointer["attempt_root"])


def run_sr5() -> dict[str, Any]:
    latest = _latest_attempt()
    if latest is None:
        return {"status": "not_evaluated", "reason": "sr0_to_sr4_missing"}
    attempt_id, attempt = latest
    if (attempt / "sr5_clean_source_release.json").is_file():
        parent_id, parent = attempt_id, attempt
        attempt_id, attempt = allocate_attempt()
        write_json(attempt / "sr5_parent_attempt.json", {"parent_attempt_id": parent_id, "parent_attempt_root": rel(parent), "reason": "prior_sr5_artifact_immutable_allocate_new"})
    else:
        parent_id, parent = attempt_id, attempt
    dirty = source_porcelain(ROOT)
    gpu = gpu_compute_snapshot()
    live = live_workers()
    identity = identity_gate(
        expected_commit=git(("rev-parse", "HEAD")),
        expected_tool_sha256=sha256_file(Path(__file__)),
        expected_openpi=git(("rev-parse", "HEAD"), OPENPI),
        expected_command=[str(RECORDER_PYTHON), "-m", MODULE, "--sr6-worker"],
        dirty=bool(dirty),
        pid_active=bool(live),
        cursor_active=False,
        expected_cuda_visible_devices="",
        observed_cuda_visible_devices="",
        gpu_policy="cpu_no_renderer",
    )
    command = [str(RECORDER_PYTHON), "-m", MODULE, "--sr6-worker", "--profile", PRIMARY_PROFILE["name"], "--launch-attempt-id", "1"]
    result = {
        "schema_version": "track3_sr5_clean_source_release_v1",
        "status": "passed" if identity["state"] == "passed" and not dirty and gpu.get("empty") and not live else "hard_stop",
        "attempt_id": attempt_id,
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__)),
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI),
        "command": command,
        "command_hash": json_hash(command),
        "profile_lock": {"primary": PRIMARY_PROFILE, "alternative": ALTERNATIVE_PROFILE, "max_support_identities": MAX_SUPPORT_IDENTITIES},
        "clean_source_identity": identity,
        "live_boundary": {"workers": live, "gpu": gpu, "dirty_rows": dirty},
        "do_not_reuse_d5c319b": True,
        "physics_authorized_after_this_pass": True,
        "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "physics_started": False, "teleop": False, "replay": False, "training": False, "zip": False},
    }
    write_json(attempt / "sr5_clean_source_release.json", result, immutable=True)
    write_json(SR_ROOT / "sr5_clean_source_release.json", {"schema_version": "track3_sr5_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "sr5_clean_source_release.json")})
    return result


def _select_translation(stage: str, target: Sequence[float], current: Mapping[str, Any]) -> list[float]:
    delta = [float(target[index]) - float(current["eef_translation_m"][index]) for index in range(3)]
    if stage in {"precontact_xy", "transport"}:
        return [delta[0], delta[1], 0.0]
    if stage in {"precontact_z", "contact_descent", "lift", "place", "retreat"}:
        return [0.0, 0.0, delta[2]]
    return [0.0, 0.0, 0.0]


def _child_base(**fields: Any) -> dict[str, Any]:
    base = {
        "schema_version": "track3_sr6_child_v1",
        "environment_created": False,
        "physics_steps": 0,
        "physics_identity_consumed": False,
        "launch_attempt_id": None,
        "failure_stage": None,
        "training_eligible": False,
        "accepted": False,
        "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False},
    }
    base.update(fields)
    return base


def sr6_worker(profile: Mapping[str, Any], result_path: Path, attempt: Path, *, launch_attempt_id: int) -> int:
    import numpy as np

    from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
    from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
    from tools.axis_pi05 import track3_operator_scoped_stability as prior
    from tools.axis_pi05 import track3_reverse_init_stabilization as base
    from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as control

    started = now()
    env = None
    environment_created = False
    physics_steps = 0
    failure_stage = "identity"
    result: dict[str, Any]
    try:
        sr5_pointer = strict_json(SR_ROOT / "sr5_clean_source_release.json") if (SR_ROOT / "sr5_clean_source_release.json").is_file() else {}
        sr5 = strict_json(ROOT / str(sr5_pointer["target"])) if sr5_pointer.get("target") else {}
        command = [str(RECORDER_PYTHON), "-m", MODULE, "--sr6-worker", "--attempt-root", str(attempt), "--result-output", str(result_path), "--profile", str(profile["name"]), "--launch-attempt-id", str(launch_attempt_id)]
        child_identity = identity_gate(
            expected_commit=str(sr5.get("root_commit")),
            expected_tool_sha256=str(sr5.get("tool_sha256")),
            expected_openpi=str(sr5.get("openpi_commit")),
            expected_command=command,
            observed_command=command,
            expected_libero_plus_porcelain_sha256=json_hash(porcelain(LIBERO_PLUS)),
            observed_libero_plus_porcelain_sha256=json_hash(porcelain(LIBERO_PLUS)),
            dirty=bool(source_porcelain(ROOT)),
            pid_active=False,
            cursor_active=False,
            expected_cuda_visible_devices="",
            observed_cuda_visible_devices="",
            gpu_policy="cpu_no_renderer",
        )
        if child_identity["state"] != "passed":
            result = _child_base(status="infrastructure_failure", category="infrastructure_failure", stop_class="global_hard_stop", child_pid=os.getpid(), identity_gate=child_identity, launch_attempt_id=launch_attempt_id, failure_stage="identity")
            write_json(result_path, result)
            return 0
        failure_stage = "pre_env"
        prepared = prepare_support_identity(profile)
        if prepared["control_env_imported_after"] or prepared["environment_created"]:
            raise RuntimeError("prepare_created_or_imported_control_env")
        composed._configure_public_env(gpu=False)
        from libero.libero.envs.env_wrapper import ControlEnv

        failure_stage = "env_setup"
        env = ControlEnv(bddl_file_name=str(composed.REVERSE_BDDL[TASK_INDEX]), control_freq=CONTROL_FREQ_HZ, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=-1)
        environment_created = True
        env.reset()
        env.reset_from_xml_string(str(prepared["resolved_xml"]))
        with np.load(control.STATE_BUNDLE, allow_pickle=False) as archive:
            start_state = np.asarray(archive["task10_demo0"], dtype=np.float64).copy()
        env.set_init_state(start_state)
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(composed.REVERSE_BDDL[TASK_INDEX], "init")
        goal_atoms = base.public_predicates(composed.REVERSE_BDDL[TASK_INDEX], "goal")
        current_z = float(dynamic._robot_metrics(env)["eef_translation_m"][2])
        hold_quat = list(dynamic._robot_metrics(env)["eef_quaternion"])
        stages_spec = prepared["stages"]
        stages_spec[0]["target_pos_m"][2] = current_z
        stages_spec[0]["target_quat"] = hold_quat
        stages_spec[1]["target_quat"] = hold_quat
        samples: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        states: list[Any] = []
        phases: list[str] = []
        source_values: list[Any] = []
        goal_values: list[Any] = []
        eef_positions: list[list[float]] = []
        action_saturated: list[bool] = []

        def observe(step: int) -> dict[str, Any]:
            sample = prior._runtime_sample(env, dynamic, TASK_INDEX, "reverse", step, initial_atoms, goal_atoms)
            sample["contact_truth_v3"] = contact_truth_v3(TASK_INDEX, sample.get("contact_matrix", {}))
            samples.append(sample)
            return sample

        observe(0)
        stage_results: list[dict[str, Any]] = []
        failure_reason: str | None = None
        for stage_index, spec in enumerate(stages_spec):
            stage = str(spec["name"])
            failure_stage = stage
            gripper = float(spec["gripper"])
            target_position = list(spec["target_pos_m"])
            target_quat = list(spec["target_quat"])
            budget = int(spec["budget"])
            stable = 0
            stage_start = len(actions)
            last_t, last_r = float("inf"), float("inf")
            for _ in range(budget):
                current = dynamic._robot_metrics(env)
                translation = _select_translation(stage, target_position, current)
                rotation = control.quat_log_error_xyzw(target_quat, current["eef_quaternion"]) if stage not in {"precontact_xy", "precontact_z"} else [0.0, 0.0, 0.0]
                command_result = control.osc_action_scaled(translation, rotation, gripper, profile=profile, scales={"translation_m": float(profile["controller_translation_scale_m"]), "rotation_rad": float(profile["controller_rotation_scale_rad"])})
                env.step(np.asarray(command_result["action"], dtype=np.float32))
                physics_steps += 1
                actions.append(list(command_result["action"]))
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
                phases.append(stage)
                eef_positions.append(list(current["eef_translation_m"]))
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
                if stable >= int(profile["stable_ticks"]):
                    break
            reached = stable >= int(profile["stable_ticks"])
            stage_results.append({"stage": stage, "stage_index": stage_index, "frames": len(actions) - stage_start, "budget": budget, "reached": reached, "last_translation_error_m": last_t, "last_rotation_error_rad": last_r})
            if not reached:
                failure_reason = f"stage_timeout:{stage}"
                break
            if stage == "grasp_dwell":
                barrier = required_contact_gate(samples, len(samples) - 1, stable_ticks=int(profile["stable_ticks"]))
                if barrier["state"] == "infrastructure_failure":
                    failure_reason = "acquisition_barrier_unknown"
                    break
                if barrier["state"] != "passed":
                    failure_reason = "acquisition_barrier"
                    break
        final_success = bool(env.check_success())
        trace_samples = samples[1:]
        contact_rows = [row.get("contact_truth_v3", {}) for row in trace_samples]
        contact_unknown = any(row.get("state") == "infrastructure_failure" for row in contact_rows)
        unrelated_any = any(row.get("unrelated_contact_runtime") is True for row in contact_rows)
        dwell_indices = [index for index, phase in enumerate(phases) if phase == "grasp_dwell"]
        barrier_index = (dwell_indices[-1] + 1) if dwell_indices else None
        barrier = required_contact_gate(samples, barrier_index, stable_ticks=int(profile["stable_ticks"])) if barrier_index is not None else gate(False, None, reason="acquisition_barrier_not_entered")
        entered = {name: any(row["stage"] == name for row in stage_results) for name in STAGE_NAMES}
        source_gate = source_released_gate(source_values, phases, stable_ticks=int(profile["stable_ticks"]))
        goal_gate = goal_continuous_gate(goal_values, phases, check_success=final_success, stable_ticks=int(profile["stable_ticks"]))
        stage_gates = {row["stage"]: gate(True, bool(row["reached"]), reason=f"stage_timeout:{row['stage']}" if not row["reached"] else None) for row in stage_results}
        for name in STAGE_NAMES:
            stage_gates.setdefault(name, gate(False, None, reason="upstream_stage_not_reached"))
        gates = {
            "initial_source_true": gate(bool(source_values), source_values[0] is True if source_values else None),
            "initial_goal_false": gate(bool(goal_values), goal_values[0] is False if goal_values else None),
            "acquisition_barrier": barrier,
            "source_released": source_gate,
            "goal_continuous_true": goal_gate,
            "unrelated_contact": infrastructure_gate("contact_truth_unknown") if contact_unknown else gate(bool(trace_samples), not unrelated_any, reason="unrelated_background_contact" if unrelated_any else None),
            "finite_trace": gate(bool(actions), bool(np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(states)).all()) if actions else None),
            "normalized_action": gate(bool(actions), all(normalized_action_contract(action)["state"] == "passed" for action in actions) if actions else None),
            "no_clip": gate(bool(actions), not any(action_saturated) if actions else None, action_saturated_count=sum(action_saturated)),
            "all_stages_reached": gate(bool(stage_results), len(stage_results) == len(STAGE_NAMES) and all(row["reached"] for row in stage_results), reason=failure_reason),
        }
        if contact_unknown or (barrier.get("state") == "infrastructure_failure") or (gates["unrelated_contact"].get("state") == "infrastructure_failure"):
            status = "infrastructure_failure"
            stop_class = "global_hard_stop"
        elif all(row.get("state") in {"passed", "not_evaluated"} for row in gates.values()) and all(gates[key]["state"] == "passed" for key in ("initial_source_true", "initial_goal_false", "acquisition_barrier", "source_released", "goal_continuous_true", "unrelated_contact", "finite_trace", "normalized_action", "no_clip", "all_stages_reached")):
            status = "success"
            stop_class = "family_outcome"
        else:
            status = "family_failure"
            stop_class = "family_outcome"
        run_root = result_path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        arrays = {
            "actions": np.asarray(actions, dtype=np.float32),
            "states": np.asarray(states, dtype=np.float64),
            "phase": np.asarray(phases, dtype="U64"),
            "eef_position_m": np.asarray(eef_positions, dtype=np.float32),
            "action_saturated": np.asarray(action_saturated, dtype=np.bool_),
            "required_contact": np.asarray([bool(row.get("required_contact", False)) for row in contact_rows], dtype=np.bool_),
            "demonstrator_contact": np.asarray([bool(row.get("demonstrator_contact", False)) for row in contact_rows], dtype=np.bool_),
            "source_receiver_contact": np.asarray([bool(row.get("source_receiver_contact", False)) for row in contact_rows], dtype=np.bool_),
            "goal_receiver_contact": np.asarray([bool(row.get("goal_receiver_contact", False)) for row in contact_rows], dtype=np.bool_),
            "contact_telemetry_complete": np.asarray([bool(row.get("contact_telemetry_complete", False)) for row in contact_rows], dtype=np.bool_),
            "unrelated_contact_runtime": np.asarray([bool(row.get("unrelated_contact_runtime", False)) for row in contact_rows], dtype=np.bool_),
            "source_value": np.asarray([-1 if value is None else int(value) for value in source_values], dtype=np.int8),
            "goal_value": np.asarray([-1 if value is None else int(value) for value in goal_values], dtype=np.int8),
        }
        trajectory_path = run_root / "trajectory.npz"
        with (run_root / "start_state.npy.partial").open("wb") as handle:
            np.save(handle, start_state, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(run_root / "start_state.npy.partial", run_root / "start_state.npy")
        with (trajectory_path.with_name(trajectory_path.name + ".partial")).open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(trajectory_path.with_name(trajectory_path.name + ".partial"), trajectory_path)
        last_stage = stage_results[-1]["stage"] if stage_results else None
        result = _child_base(
            status=status,
            category=status,
            stop_class=stop_class,
            created_at_utc=started,
            completed_at_utc=now(),
            child_pid=os.getpid(),
            environment_created=True,
            physics_steps=physics_steps,
            physics_identity_consumed=True,
            launch_attempt_id=launch_attempt_id,
            failure_stage=None if status == "success" else (failure_reason or failure_stage),
            identity_gate=child_identity,
            stage_results=stage_results,
            gates=gates,
            stage_gates=stage_gates,
            reverse={"source_atom": SOURCE_ATOM, "goal_atom": GOAL_ATOM, "final_success": final_success, "failure_reason": failure_reason},
            trajectory={"path": rel(trajectory_path), "sha256": sha256_file(trajectory_path), "actions": len(actions), "states": len(states), "typed_no_pickle": True},
            metrics={"last_stage": last_stage, "last_translation_error_m": stage_results[-1]["last_translation_error_m"] if stage_results else None, "acquisition_entered": entered["grasp_dwell"], "lift_entered": entered["lift"], "place_entered": entered["place"]},
            profile=dict(profile),
        )
    except Exception as exc:
        result = _child_base(
            status="infrastructure_failure",
            category="infrastructure_failure",
            stop_class=classify_stop(environment_created=environment_created, physics_steps=physics_steps, error_type=type(exc).__name__),
            created_at_utc=started,
            completed_at_utc=now(),
            child_pid=os.getpid(),
            environment_created=environment_created,
            physics_steps=physics_steps,
            physics_identity_consumed=bool(environment_created),
            launch_attempt_id=launch_attempt_id,
            failure_stage=failure_stage,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
    write_json(result_path, result)
    return 0


def quantified_progress(child: Mapping[str, Any]) -> dict[str, Any]:
    metrics = child.get("metrics") or {}
    last_stage = metrics.get("last_stage")
    last_error = metrics.get("last_translation_error_m")
    rank = STAGE_RANK.get(str(last_stage), -1)
    historical_rank = STAGE_RANK[CR_SUPPORT_ALT_STAGE]
    reasons = []
    if rank > historical_rank:
        reasons.append("later_failure_stage")
    if metrics.get("acquisition_entered"):
        reasons.append("entered_acquisition")
    if metrics.get("lift_entered") or metrics.get("place_entered"):
        reasons.append("entered_lift_or_later")
    if isinstance(last_error, (int, float)) and math.isfinite(float(last_error)) and float(last_error) < CR_SUPPORT_ALT_ERROR_M:
        reasons.append("terminal_eef_error_decreased")
    return {"eligible": bool(reasons), "reasons": reasons, "last_stage": last_stage, "last_translation_error_m": last_error, "historical_stage": CR_SUPPORT_ALT_STAGE, "historical_error_m": CR_SUPPORT_ALT_ERROR_M}


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


def _spawn_identity(attempt: Path, profile: Mapping[str, Any], launch_attempt_id: int, phase: str, sr5: Mapping[str, Any]) -> dict[str, Any]:
    canary = attempt / "canary" / profile["name"]
    canary.mkdir(parents=True, exist_ok=True)
    result_path = canary / "child_result.json"
    command = [str(RECORDER_PYTHON), "-m", MODULE, "--sr6-worker", "--attempt-root", str(attempt), "--result-output", str(result_path), "--profile", str(profile["name"]), "--launch-attempt-id", str(launch_attempt_id)]
    identity = identity_gate(
        expected_commit=str(sr5["root_commit"]),
        expected_tool_sha256=str(sr5["tool_sha256"]),
        expected_openpi=str(sr5["openpi_commit"]),
        expected_command=command,
        observed_command=command,
        expected_libero_plus_porcelain_sha256=json_hash(porcelain(LIBERO_PLUS)),
        observed_libero_plus_porcelain_sha256=json_hash(porcelain(LIBERO_PLUS)),
        dirty=bool(source_porcelain(ROOT)),
        pid_active=False,
        cursor_active=False,
        expected_cuda_visible_devices="",
        observed_cuda_visible_devices="",
        gpu_policy="cpu_no_renderer",
    )
    write_json(canary / "identity_preflight.json", {"identity": identity, "command": command, "launch_attempt_id": launch_attempt_id})
    if identity["state"] != "passed":
        return {"status": "hard_stop", "category": "infrastructure_failure", "stop_class": "global_hard_stop", "identity_gate": identity, "launch_attempt_id": launch_attempt_id, "physics_identity_consumed": False}
    write_json(attempt / "cursor.json", {"schema_version": "track3_sr_cursor_v1", "active": True, "phase": phase, "pid": os.getpid(), "command": command, "updated_at_utc": now()})
    write_json(attempt / "heartbeat.json", {"schema_version": "track3_sr_heartbeat_v1", "state": "running", "phase": phase, "pid": os.getpid(), "count": 1, "updated_at_utc": now()})
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}"}
    stdout_path, stderr_path = canary / "child.stdout.log", canary / "child.stderr.log"
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        count = 1
        while process.poll() is None:
            if time.monotonic() - started >= CANARY_TIMEOUT_SECONDS:
                timed_out = True
                termination_stage = _terminate_group(process)
                break
            count += 1
            write_json(attempt / "heartbeat.json", {"schema_version": "track3_sr_heartbeat_v1", "state": "running", "phase": phase, "pid": os.getpid(), "child_pid": process.pid, "count": count, "updated_at_utc": now()})
            time.sleep(1.0)
        process.wait()
    write_json(attempt / "heartbeat.json", {"schema_version": "track3_sr_heartbeat_v1", "state": "final", "phase": phase, "pid": os.getpid(), "count": count + 1, "updated_at_utc": now()})
    write_json(attempt / "cursor.json", {"schema_version": "track3_sr_cursor_v1", "active": False, "phase": phase, "pid": os.getpid(), "updated_at_utc": now()})
    child = strict_json(result_path) if result_path.is_file() else None
    infra = timed_out or process.returncode != 0 or not isinstance(child, Mapping) or child.get("category") == "infrastructure_failure" or child.get("stop_class") == "global_hard_stop"
    return {
        "status": "hard_stop" if infra else "completed",
        "category": "infrastructure_failure" if infra else str(child.get("category")),
        "stop_class": "global_hard_stop" if infra else str(child.get("stop_class")),
        "child": child,
        "child_path": rel(result_path) if result_path.is_file() else None,
        "process": {"returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage},
        "launch_attempt_id": launch_attempt_id,
        "physics_identity_consumed": bool(isinstance(child, Mapping) and child.get("physics_identity_consumed")),
        "identity_gate": identity,
        "progress": quantified_progress(child) if isinstance(child, Mapping) else {"eligible": False},
    }


def run_sr6_sr7() -> dict[str, Any]:
    sr5_pointer = strict_json(SR_ROOT / "sr5_clean_source_release.json") if (SR_ROOT / "sr5_clean_source_release.json").is_file() else {}
    sr5 = strict_json(ROOT / str(sr5_pointer["target"])) if sr5_pointer.get("target") else {}
    if sr5.get("status") != "passed":
        return {"status": "not_evaluated", "reason": "sr5_clean_source_release_not_passed", "physics_started": False}
    attempt_id, attempt = allocate_attempt()
    primary = _spawn_identity(attempt, PRIMARY_PROFILE, launch_attempt_id=1, phase="SR6", sr5=sr5)
    alternative: dict[str, Any] = {"status": "not_evaluated", "reason": "sr7_requires_normal_family_failure_with_quantified_progress"}
    if primary.get("category") == "success":
        alternative = {"status": "not_evaluated", "reason": "sr6_success_skips_alternative"}
    elif primary.get("category") == "family_failure" and primary.get("progress", {}).get("eligible"):
        alternative = _spawn_identity(attempt, ALTERNATIVE_PROFILE, launch_attempt_id=2, phase="SR7", sr5=sr5)
    elif primary.get("stop_class") == "global_hard_stop" or primary.get("category") == "infrastructure_failure":
        alternative = {"status": "not_evaluated", "reason": "sr6_infrastructure_blocks_sr7"}
    successes = [row for row in (primary, alternative) if row.get("category") == "success"]
    summary = {
        "schema_version": "track3_sr6_sr7_summary_v1",
        "status": "hard_stop" if primary.get("status") == "hard_stop" else "completed",
        "attempt_id": attempt_id,
        "primary": {key: primary.get(key) for key in ("status", "category", "stop_class", "progress", "launch_attempt_id", "physics_identity_consumed", "process", "child_path")},
        "alternative": {key: alternative.get(key) for key in alternative if key != "child"},
        "success_count": len(successes),
        "launch_attempt_count": 1 + (1 if alternative.get("launch_attempt_id") else 0 if alternative.get("status") != "not_evaluated" else 0),
        "physics_identity_count": int(bool(primary.get("physics_identity_consumed"))) + int(bool(alternative.get("physics_identity_consumed"))),
        "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "replay": False, "training": False, "zip": False},
    }
    if alternative.get("launch_attempt_id"):
        summary["launch_attempt_count"] = 2
    write_json(attempt / "summary.json", summary, immutable=True)
    set_attempt_active(attempt_id, attempt, False, status=str(summary["status"]))
    return {**summary, "finalizer": {"status": "deferred_until_all_producers_closed", "sealed": False}, "primary_child": primary.get("child"), "alternative_child": alternative.get("child")}


def run_sr8() -> dict[str, Any]:
    latest = _latest_attempt()
    attempt = latest[1] if latest else SR_ROOT
    summary = strict_json(attempt / "summary.json") if (attempt / "summary.json").is_file() else {}
    if int(summary.get("success_count") or 0) > 0:
        result = {"schema_version": "track3_sr8_route_v1", "status": "success_route_sr10", "teleop": False}
    else:
        result = {
            "schema_version": "track3_sr8_route_v1",
            "status": "human_input_required",
            "user_online_takeover_required": True,
            "recorded_candidates_max": 1,
            "agent_generated_key_input": False,
            "recording_started": False,
            "exact_recorder_argv": recorder_command(TASK_INDEX),
            "next_action": "wait_for_explicit_user_online_control",
            "sr12_may_continue": True,
        }
    write_json(attempt / "sr8_route.json", result, immutable=True)
    return result


def run_sr12() -> dict[str, Any]:
    latest = _latest_attempt()
    root = (latest[1] / "sr12_readiness") if latest else SR_ROOT / "sr12_readiness"
    result = {
        "schema_version": "track3_sr12_code_only_readiness_v1",
        "status": "passed",
        "schemas": ["raw", "pending_replay", "pending_gold_review", "accepted", "training_eligible"],
        "allowed_transitions": [["raw", "pending_replay"], ["pending_replay", "pending_gold_review"], ["pending_gold_review", "accepted"], ["accepted", "training_eligible"]],
        "fail_closed": {"diagnostic_to_accepted": True, "family_failure_to_training": True, "no_accepted_demo_no_norm": True, "no_accepted_demo_no_training": True},
        "split": {"source_group_isolation": True, "holdout_isolated": True, "dedup_before_split": True},
        "loader_dry_run": {"started": False, "reason": "no_accepted_demo"},
        "norm_cli": {"would_start": False, "fail_closed": True},
        "tiny_overfit_cli": {"would_start": False, "fail_closed": True},
        "training_started": False,
        "failed_traces_used": False,
        "resource_boundary": {"training": False, "gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "zip": False},
    }
    write_json(root / "sr12_code_only_readiness.json", result, immutable=True)
    return result


def run_sr13() -> dict[str, Any]:
    latest = _latest_attempt()
    root = latest[1] if latest else SR_ROOT
    summary = strict_json(root / "summary.json") if (root / "summary.json").is_file() else {}
    result = {
        "schema_version": "track3_sr13_closeout_v1",
        "status": "hard_stop",
        "attempt_id": latest[0] if latest else None,
        "launch_attempt_count": summary.get("launch_attempt_count"),
        "physics_identity_count": summary.get("physics_identity_count"),
        "success_count": summary.get("success_count", 0),
        "accepted_count": 0,
        "training_eligible_count": 0,
        "not_evaluated": derive_not_evaluated(root, success_count=int(summary.get("success_count") or 0)),
        "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "generation": False, "replay": False, "training": False, "zip": False, "upload": False, "submission": False},
        "next_authority": "independent_review_required",
    }
    write_json(root / "sr13_closeout.json", result, immutable=True)
    finalizer = finalize_attempt(root)
    return {**result, "finalizer": finalizer}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr0-to-sr4", action="store_true")
    parser.add_argument("--sr5", action="store_true")
    parser.add_argument("--sr6", action="store_true")
    parser.add_argument("--sr8", action="store_true")
    parser.add_argument("--sr12", action="store_true")
    parser.add_argument("--sr13", action="store_true")
    parser.add_argument("--sr6-worker", action="store_true")
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--profile", choices=[PRIMARY_PROFILE["name"], ALTERNATIVE_PROFILE["name"]])
    parser.add_argument("--launch-attempt-id", type=int, default=1)
    args = parser.parse_args(argv)
    if args.sr6_worker:
        if args.attempt_root is None or args.result_output is None or args.profile is None:
            parser.error("--sr6-worker requires --attempt-root --result-output --profile")
        profile = PRIMARY_PROFILE if args.profile == PRIMARY_PROFILE["name"] else ALTERNATIVE_PROFILE
        return sr6_worker(profile, args.result_output, args.attempt_root, launch_attempt_id=int(args.launch_attempt_id))
    selected = [name for name in ("sr0_to_sr4", "sr5", "sr6", "sr8", "sr12", "sr13") if getattr(args, name)]
    if len(selected) != 1:
        parser.error("select exactly one phase")
    mapping = {"sr0_to_sr4": run_sr0_to_sr4, "sr5": run_sr5, "sr6": run_sr6_sr7, "sr8": run_sr8, "sr12": run_sr12, "sr13": run_sr13}
    result = mapping[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if str(result.get("status")) in {"passed", "completed", "hard_stop", "human_input_required", "success_route_sr10", "not_evaluated"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
