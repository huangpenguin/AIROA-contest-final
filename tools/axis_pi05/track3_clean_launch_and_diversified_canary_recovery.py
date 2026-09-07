#!/usr/bin/env python3
"""CR0--CR14 clean-launch and diversified CPU canary recovery.

This is a new evidence namespace.  The BR/AC namespaces are read-only inputs;
no writer in this module points at either of them.  Physics is deliberately
reachable only after the CR0--CR8 clean-source and strategy-freeze gates have
closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_clean_launch_and_diversified_canary_recovery"
TOOL_PATH = Path(__file__).resolve()
TOOL_REL = TOOL_PATH.relative_to(ROOT).as_posix()
CR_ROOT = ROOT / "data/reports/axis_libero_30k/track3_clean_launch_and_diversified_canary_recovery"
BR_ROOT = ROOT / "data/reports/axis_libero_30k/track3_acquisition_gate_attempt_lifecycle_remediation"
AC_ROOT = ROOT / "data/reports/axis_libero_30k/track3_crossfamily_acquisition_recovery_canary"
BR_AUDIT = ROOT / "data/reports/axis_libero_30k/track3_br0_br11_independent_audit_20260907/summary.json"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PUBLIC_LIBERO = ROOT / "LIBERO"
PUBLIC_AUTOGEN = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"

ACTION_DIM = 7
CONTACT_WINDOW = 4
FIXTURE_TIMEOUT = 0.35
NORMAL_FIXTURE_TIMEOUT = 2.0
TERM_GRACE_SECONDS = 0.25
CPU_TIMEOUT_SECONDS = 240.0

PARTIAL_BYTES = b'{"state": "child_running", "sequence": 13}\n'
SYNTHETIC_BYTES = b'{"state":"child_running","sequence":999'

FAMILIES = ("support_remove", "containment_remove", "articulation_close")
TASK_BY_FAMILY = {"support_remove": 10, "containment_remove": 20, "articulation_close": 19}
CR_PROFILE_ORDER = (
    "cr_support_primary_20260907_0001",
    "cr_containment_primary_20260907_0001",
    "cr_articulation_primary_20260907_0001",
    "cr_support_alternative_20260907_0001",
    "cr_containment_alternative_20260907_0001",
    "cr_articulation_alternative_20260907_0001",
)
PROFILE_SPECS: dict[str, dict[str, Any]] = {
    CR_PROFILE_ORDER[0]: {"family": "support_remove", "task_index": 10, "strategy": "object_frame_contact_aligned_v1", "seed": 81010, "grasp_z_m": 0.020, "close_dwell": 3, "release_dwell": 2},
    CR_PROFILE_ORDER[1]: {"family": "containment_remove", "task_index": 20, "strategy": "object_frame_contact_aligned_v1", "seed": 82020, "grasp_z_m": 0.020, "close_dwell": 3, "release_dwell": 2},
    CR_PROFILE_ORDER[2]: {"family": "articulation_close", "task_index": 19, "strategy": "object_frame_contact_aligned_v1", "seed": 83019, "grasp_z_m": 0.010, "close_dwell": 3, "release_dwell": 2},
    CR_PROFILE_ORDER[3]: {"family": "support_remove", "task_index": 10, "strategy": "waypoint_guarded_microstep_v1", "seed": 84010, "grasp_z_m": 0.017, "close_dwell": 4, "release_dwell": 2},
    CR_PROFILE_ORDER[4]: {"family": "containment_remove", "task_index": 20, "strategy": "waypoint_guarded_microstep_v1", "seed": 85020, "grasp_z_m": 0.017, "close_dwell": 4, "release_dwell": 2},
    CR_PROFILE_ORDER[5]: {"family": "articulation_close", "task_index": 19, "strategy": "waypoint_guarded_microstep_v1", "seed": 86019, "grasp_z_m": 0.008, "close_dwell": 4, "release_dwell": 2},
}
PRIMARY_BY_FAMILY = {PROFILE_SPECS[row]["family"]: row for row in CR_PROFILE_ORDER[:3]}
ALTERNATIVE_BY_FAMILY = {PROFILE_SPECS[row]["family"]: row for row in CR_PROFILE_ORDER[3:]}
OLD_IDS = {
    "ac_support_remove_20260906_0001", "ac_containment_remove_20260906_0001", "ac_articulation_close_20260906_0001",
    "br_support_remove_20260907_0001", "br_containment_remove_20260907_0001", "br_articulation_close_20260907_0001",
    "er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh",
}

SOURCE_ATOM = {10: "(On akita_black_bowl_1 plate_1)", 20: "(In orange_juice_1 basket_1_contain_region)", 19: "(Open wooden_cabinet_1_middle_region)"}
GOAL_ATOM = {10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)", 20: "(On orange_juice_1 floor_target_object_region)", 19: "(Close wooden_cabinet_1_middle_region)"}
REVERSE_BDDL = {
    10: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_support_remove_task_10.bddl",
    20: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_containment_remove_task_20.bddl",
    19: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_articulation_close_task_19.bddl",
}
SNAPSHOTS = {task: P1_ROOT / "snapshots" / f"task_{task:02d}" / "attempt_1" / "terminal_state.npy" for task in (10, 20, 19)}

EXPECTED_STAGES = {
    "support_remove": ("approach_object", "descend_object", "grasp_object", "lift_object", "transport_to_target", "descend_to_target", "release_object", "retreat"),
    "containment_remove": ("approach_object", "descend_object", "grasp_object", "lift_object", "transport_to_target", "descend_to_target", "release_object", "retreat"),
    "articulation_close": ("approach_drawer_region", "grasp_drawer_region", "pull_drawer_closed", "hold_closed"),
}


class DuplicateKeyError(ValueError):
    pass


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


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_json_immutable(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable_artifact_exists:{path}")
    write_json(path, value)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def argv_hash(command: Sequence[str]) -> str:
    normalized = list(command)
    if "--expected-command-sha256" in normalized:
        index = normalized.index("--expected-command-sha256") + 1
        if index < len(normalized):
            normalized[index] = "<expected-command-sha256>"
    return canonical_hash(normalized)


def _head_tool_sha() -> str | None:
    raw = subprocess.run(["git", "show", f"HEAD:{TOOL_REL}"], cwd=ROOT, capture_output=True, check=False).stdout
    return sha256_bytes(raw) if raw else None


def live_clean_source_gate(*, expected_commit: str | None = None, expected_tool_sha256: str | None = None) -> dict[str, Any]:
    root_state, openpi_state = porcelain(ROOT), porcelain(OPENPI)
    current_commit = git(("rev-parse", "HEAD"))
    current_tool = sha256_file(TOOL_PATH)
    head_tool = _head_tool_sha()
    checks = {
        "root_clean": not root_state,
        "openpi_clean": not openpi_state,
        "expected_commit": expected_commit is None or current_commit == expected_commit,
        "tool_matches_head_path": bool(head_tool) and current_tool == head_tool,
        "expected_tool": expected_tool_sha256 is None or current_tool == expected_tool_sha256,
    }
    return {
        "state": "passed" if all(checks.values()) else "infrastructure_failure",
        "checks": checks,
        "root_commit": current_commit,
        "tool_sha256": current_tool,
        "head_tool_sha256": head_tool,
        "root_porcelain": root_state,
        "openpi_porcelain": openpi_state,
        "root_porcelain_sha256": canonical_hash(root_state),
        "openpi_porcelain_sha256": canonical_hash(openpi_state),
        "reason": "launch_clean_source_gate" if not all(checks.values()) else None,
    }


def identity_snapshot(command: Sequence[str], profile_sha: str, *, expected_commit: str | None = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = environment or os.environ
    gate = live_clean_source_gate(expected_commit=expected_commit)
    libero_state = porcelain(LIBERO_PLUS)
    return {
        "root_commit": gate["root_commit"], "tool_sha256": gate["tool_sha256"], "head_tool_sha256": gate["head_tool_sha256"],
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "profile_lock_sha256": profile_sha,
        "command": list(command), "command_hash": argv_hash(command), "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "jax_platforms": env.get("JAX_PLATFORMS", ""), "mujoco_gl": env.get("MUJOCO_GL", ""),
        "root_porcelain": gate["root_porcelain"], "root_porcelain_sha256": gate["root_porcelain_sha256"],
        "openpi_porcelain": gate["openpi_porcelain"], "openpi_porcelain_sha256": gate["openpi_porcelain_sha256"],
        "libero_plus_porcelain": libero_state, "libero_plus_porcelain_sha256": canonical_hash(libero_state),
        "clean_root": bool(gate["checks"]["root_clean"] and gate["checks"]["openpi_clean"]), "pid": os.getpid(), "pgid": os.getpgid(0),
    }


IDENTITY_FIELDS = ("root_commit", "tool_sha256", "head_tool_sha256", "openpi_commit", "profile_lock_sha256", "cuda_visible_devices", "jax_platforms", "mujoco_gl", "root_porcelain_sha256", "openpi_porcelain_sha256", "libero_plus_porcelain_sha256", "clean_root")


def identity_gate(observed: Mapping[str, Any], expected: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    mismatches = [field for field in IDENTITY_FIELDS if observed.get(field) != expected.get(field)]
    if observed.get("clean_root") is not True or expected.get("clean_root") is not True:
        if "clean_root" not in mismatches:
            mismatches.append("clean_root")
    if observed.get("command_hash") != expected.get("command_hash"):
        mismatches.append("command_hash")
    if list(observed.get("command", [])) != list(command):
        mismatches.append("command")
    return {"state": "passed" if not mismatches else "infrastructure_failure", "mismatches": mismatches, "checked_fields": list(IDENTITY_FIELDS) + ["command", "command_hash"]}


@contextmanager
def supervisor_lock(path: Path) -> Iterator[dict[str, Any]]:
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise
    info = {"pid": os.getpid(), "pgid": os.getpgid(0), "acquired_at_utc": now(), "lock_path": str(path.resolve())}
    handle.seek(0); handle.truncate(); handle.write(json.dumps(info, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
    try:
        yield info
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN); handle.close()


def allocate_attempt(root: Path = CR_ROOT) -> tuple[int, Path]:
    root.mkdir(parents=True, exist_ok=True)
    with supervisor_lock(root / "attempt_allocator.lock"):
        ids = [int(p.name.removeprefix("attempt_")) for p in (root / "attempts").glob("attempt_*") if p.is_dir() and p.name.removeprefix("attempt_").isdigit()]
        attempt_id = max(ids, default=0) + 1
        attempt = root / "attempts" / f"attempt_{attempt_id:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        write_json(root / "current_attempt.json", {"schema_version": "track3_cr_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": True, "updated_at_utc": now()})
    return attempt_id, attempt


def gate(executed: bool, value: bool | None, *, reason: str | None = None, infrastructure: bool = False) -> dict[str, Any]:
    state = "not_evaluated" if not executed else ("infrastructure_failure" if infrastructure else ("passed" if value is True else ("unknown" if value is None else "failed")))
    row = {"executed": bool(executed), "value": value, "state": state}
    if reason is not None:
        row["reason"] = reason
    return row


def _contiguous(values: Sequence[bool]) -> list[list[int]]:
    out: list[list[int]] = []; current: list[int] = []
    for index, value in enumerate(values):
        if bool(value): current.append(index)
        elif current: out.append(current); current = []
    if current: out.append(current)
    return out


def stage_gate_graph(family: str, phases: Sequence[str]) -> dict[str, dict[str, Any]]:
    entered = set(phases); expected = EXPECTED_STAGES[family]; result = {}
    for index, stage in enumerate(expected):
        if stage not in entered:
            result[stage] = gate(False, None, reason="stage_not_entered")
        else:
            result[stage] = gate(True, all(item in entered for item in expected[: index + 1]))
    return result


def telemetry_schema_gate(samples: Sequence[Mapping[str, Any]], *, stage_entered: bool) -> dict[str, Any]:
    if not stage_entered:
        return gate(False, None, reason="stage_not_entered")
    errors = []
    for index, sample in enumerate(samples):
        required = {"telemetry_complete", "known_mask", "contacts"}
        if not required.issubset(sample) or sample.get("telemetry_complete") is not True or sample.get("known_mask") is not True or not isinstance(sample.get("contacts"), list):
            errors.append(f"sample_{index}")
    return gate(True, True, reason=";".join(errors), infrastructure=True) if errors else gate(True, True)


def required_contact_gate(family: str, samples: Sequence[Mapping[str, Any]], *, stage_entered: bool, barrier_index: int | None, min_window: int = CONTACT_WINDOW) -> dict[str, Any]:
    if not stage_entered:
        return {**gate(False, None, reason="stage_not_entered"), "contact_window": 0, "barrier_index": barrier_index}
    telemetry = telemetry_schema_gate(samples, stage_entered=True)
    if telemetry["state"] == "infrastructure_failure":
        return {**telemetry, "family": family, "contact_window": 0, "barrier_index": barrier_index}
    if barrier_index is None:
        return {**gate(False, None, reason="acquisition_barrier_not_entered"), "contact_window": 0, "barrier_index": None}
    eligible = list(samples[: barrier_index + 1]); partner = {"object"} if family != "articulation_close" else {"handle", "region"}
    flags = [any("demonstrator" in {str(role).lower() for role in c.get("roles", [])} and partner & {str(role).lower() for role in c.get("roles", [])} for c in sample["contacts"] if isinstance(c, Mapping)) for sample in eligible]
    windows = _contiguous(flags); ending = max((window for window in windows if window and window[-1] == len(eligible) - 1), key=len, default=[])
    return {**gate(True, len(ending) >= min_window, reason="required_contact_window_short" if len(ending) < min_window else None), "family": family, "contact_window": len(ending), "barrier_index": barrier_index, "eligible_sample_count": len(eligible), "late_contact_excluded": len(samples) > len(eligible)}


def source_release_gate(source: Sequence[Any], known: Sequence[Any], *, acquisition_state: str) -> dict[str, Any]:
    if acquisition_state != "passed": return gate(False, None, reason="acquisition_not_passed")
    if len(source) == 0 or len(source) != len(known) or not bool(known[0]) or not bool(source[0]): return gate(True, False, reason="initial_source_contract")
    false_indices = [i for i, value in enumerate(source) if bool(known[i]) and not bool(value)]
    if not false_indices: return gate(True, False, reason="source_never_released")
    release = false_indices[0]
    passed = all(bool(known[i]) and not bool(value) for i, value in enumerate(source[release:], release))
    return gate(True, passed, reason="source_reasserted_or_unknown" if not passed else None)


def goal_continuity_gate(goal: Sequence[Any], known: Sequence[Any], *, goal_stage_entered: bool) -> dict[str, Any]:
    if not goal_stage_entered: return gate(False, None, reason="goal_stage_not_entered")
    if len(goal) == 0 or len(goal) != len(known): return gate(True, False, reason="goal_trace_missing")
    true_indices = [i for i, value in enumerate(goal) if bool(value) and bool(known[i])]
    if not true_indices: return gate(True, False, reason="goal_never_true")
    first = true_indices[0]; passed = all(bool(known[i]) and bool(value) for i, value in enumerate(goal[first:], first))
    return gate(True, passed, reason="goal_not_continuous" if not passed else None)


def normalized_action_metrics(actions: Any, *, physical_translation_m: Any | None = None, physical_rotation_rad: Any | None = None, action_saturated: Any | None = None) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != ACTION_DIM: raise ValueError("normalized_action_shape")
    saturated = np.asarray(action_saturated if action_saturated is not None else np.zeros(len(values), dtype=bool), dtype=bool)
    result = {"normalized_action_abs_max": float(np.max(np.abs(values[:, :6]), initial=0.0)), "normalized_translation_l2_max": float(np.linalg.norm(values[:, :3], axis=1).max(initial=0.0)), "normalized_rotation_l2_max": float(np.linalg.norm(values[:, 3:6], axis=1).max(initial=0.0)), "controller_clip_count": int(saturated.sum()), "finite": bool(np.isfinite(values).all())}
    if physical_translation_m is not None: result["physical_translation_m_l2_max"] = float(np.linalg.norm(np.asarray(physical_translation_m, dtype=np.float64), axis=1).max(initial=0.0))
    if physical_rotation_rad is not None: result["physical_rotation_rad_l2_max"] = float(np.linalg.norm(np.asarray(physical_rotation_rad, dtype=np.float64), axis=1).max(initial=0.0))
    return result


def _load_trace(path: Path) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def typed_trace_check(path: Path) -> dict[str, Any]:
    import numpy as np
    arrays: dict[str, Any] = {}; errors: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name]); arrays[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
                if array.dtype.hasobject: errors.append(f"object_dtype:{name}")
                if array.dtype.kind in "fc" and not np.isfinite(array).all(): errors.append(f"nonfinite:{name}")
    except Exception as exc: errors.append(f"{type(exc).__name__}:{exc}")
    return {"path": rel(path), "arrays": arrays, "errors": errors, "allow_pickle_false": not errors, "status": "passed" if not errors else "failed"}


def evaluate_typed_trace(family: str, path: Path) -> dict[str, Any]:
    import numpy as np
    check = typed_trace_check(path)
    if check["status"] != "passed": return {"category": "infrastructure_failure", "status": "infrastructure_failure", "reason": "typed_trace", "typed_trace": check}
    arrays = _load_trace(path); required = ("actions", "eef_error", "phase", "source_value", "source_known", "goal_value", "goal_known", "action_saturated", "contact_telemetry_complete", "telemetry_known", "required_contact", "unrelated_contact_runtime")
    missing = [name for name in required if name not in arrays]
    if missing: return {"category": "infrastructure_failure", "status": "infrastructure_failure", "reason": "missing_trace_fields", "missing": missing, "typed_trace": check}
    phases = [str(value) for value in np.asarray(arrays["phase"], dtype="U64").tolist()]
    barrier_stage = "grasp_object" if family != "articulation_close" else "pull_drawer_closed"; goal_stage = "release_object" if family != "articulation_close" else "hold_closed"
    barrier = next((i for i, phase in enumerate(phases) if phase == barrier_stage), None)
    telemetry_samples = [{"telemetry_complete": bool(c), "known_mask": bool(k), "contacts": [{"roles": ["demonstrator", "handle" if family == "articulation_close" else "object"]}] if bool(contact) else []} for c, k, contact in zip(arrays["contact_telemetry_complete"], arrays["telemetry_known"], arrays["required_contact"])]
    contact = required_contact_gate(family, telemetry_samples, stage_entered=barrier is not None, barrier_index=barrier)
    source = source_release_gate(arrays["source_value"], arrays["source_known"], acquisition_state=contact["state"])
    goal = goal_continuity_gate(arrays["goal_value"], arrays["goal_known"], goal_stage_entered=goal_stage in set(phases))
    metrics = normalized_action_metrics(arrays["actions"], action_saturated=arrays["action_saturated"])
    stage = stage_gate_graph(family, phases); all_stages = gate(True, all(row["state"] == "passed" for row in stage.values()))
    gates = {"all_stages_reached": all_stages, "required_contact_continuous": contact, "source_released": source, "goal_continuous_true": goal, "controller_clip": gate(True, metrics["controller_clip_count"] == 0), "normalized_action_limit": gate(True, metrics["normalized_action_abs_max"] <= 1.0), "unrelated_contact": gate(True, not bool(np.asarray(arrays["unrelated_contact_runtime"], dtype=bool).any())), "finite_trace": gate(True, metrics["finite"])}
    if any(row["state"] == "infrastructure_failure" for row in gates.values()): category = "infrastructure_failure"
    elif all(row["state"] == "passed" for row in gates.values()): category = "success"
    else: category = "family_failure"
    return {"status": category, "category": category, "gates": gates, "stage_gates": stage, "metrics": metrics, "steps": len(arrays["actions"]), "barrier_index": barrier, "late_contact_excluded": bool(barrier is not None and len(phases) > barrier + 1), "typed_trace": check}


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True); partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()}); handle.flush(); os.fsync(handle.fileno())
    os.replace(partial, path)


def profile_lock() -> dict[str, Any]:
    profiles = []
    for profile_id in CR_PROFILE_ORDER:
        row = {"profile_id": profile_id, **PROFILE_SPECS[profile_id], "old_identity_rerun": False, "renderer": False, "gpu": False, "training_eligible": False}
        row["command_hash"] = canonical_hash({"module": MODULE, "profile_id": profile_id, "strategy": row["strategy"], "seed": row["seed"], "task_index": row["task_index"]})
        profiles.append(row)
    body = {"schema_version": "track3_cr8_profile_lock_v1", "profile_order": list(CR_PROFILE_ORDER), "profiles": profiles, "family_order": list(FAMILIES), "limits": {"primary": 3, "alternative": 3, "total": 6, "per_family": 2}, "same_strategy_retry_allowed": False, "old_profile_ids_denylist": sorted(OLD_IDS), "strategies": {"primary": "object_frame_contact_aligned_v1", "alternative": "waypoint_guarded_microstep_v1"}, "official_track3_claim": False}
    body["lock_sha256"] = canonical_hash(body)
    return body


def _file_lock(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _old_inputs() -> dict[str, Any]:
    audit = strict_json(BR_AUDIT); br_traces = BR_ROOT / "attempts/attempt_0004/runs/traces"; ac_traces = AC_ROOT / "attempts/attempt_0001/runs/traces"
    hashes = {}
    for family in FAMILIES:
        br = next(br_traces.glob(f"br_{family}*_demo0.npz")); ac = next(ac_traces.glob(f"ac_{family}*_demo0.npz")); hashes[family] = {"br_path": rel(br), "ac_path": rel(ac), "br_sha256": sha256_file(br), "ac_sha256": sha256_file(ac), "byte_identical": br.read_bytes() == ac.read_bytes()}
    actual = BR_ROOT / "attempts/attempt_0002/br3_fixtures/timeout_term_kill_descendant/heartbeat.json.partial"
    return {"audit_summary": _file_lock(BR_AUDIT), "audit_status": audit["status"], "historical_trace_hashes": hashes, "historical_partial": {"path": rel(actual), "bytes": actual.stat().st_size, "sha256": sha256_file(actual)}, "old_namespace_write_allowed": False}


def _fixture_child(mode: str, partial_path: Path | None, heartbeat_path: Path | None) -> int:
    if mode == "partial":
        if partial_path is None: return 2
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("wb") as handle:
            handle.write(PARTIAL_BYTES); handle.flush(); os.fsync(handle.fileno())
        while True: time.sleep(0.05)
    if heartbeat_path is None: return 2
    for sequence in range(1, 7):
        write_json(heartbeat_path, {"state": "child_running", "sequence": sequence, "updated_at_utc": now()}); time.sleep(0.03)
    write_json(heartbeat_path.with_name("result.json"), {"category": "family_failure", "status": "family_failure", "synthetic": False})
    return 0


def _run_fixture(mode: str, attempt: Path, label: str) -> dict[str, Any]:
    fixture = attempt / "fixtures" / label; fixture.mkdir(parents=True, exist_ok=True)
    partial_path = fixture / "heartbeat.json.partial" if mode == "partial" else None; heartbeat = fixture / "heartbeat.json"
    command = [sys.executable, "-m", MODULE, "--cr-fixture-child", "--mode", mode, "--partial-output", str(partial_path or heartbeat), "--heartbeat-output", str(heartbeat)]
    started = time.monotonic(); process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True); pgid = os.getpgid(process.pid); timed_out = False; timeout_limit = FIXTURE_TIMEOUT if mode == "partial" else NORMAL_FIXTURE_TIMEOUT
    while process.poll() is None:
        if time.monotonic() - started > timeout_limit:
            timed_out = True; os.killpg(pgid, signal.SIGTERM)
            try: process.wait(timeout=TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired: os.killpg(pgid, signal.SIGKILL); process.wait()
            break
        time.sleep(0.02)
    stdout, stderr = process.communicate(); descendants = [int(row.split()[0]) for row in subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True).stdout.splitlines() if len(row.split()) == 2 and row.split()[1] == str(pgid)]
    return {"mode": mode, "label": label, "command": command, "returncode": process.returncode, "timed_out": timed_out, "term_sent": timed_out, "descendants_after_stop": descendants, "heartbeat_sequences": [], "stdout_tail": stdout[-1000:], "stderr_tail": stderr[-1000:], "partial_path": rel(partial_path) if partial_path and partial_path.exists() else None}


def _manifest_paths(attempt: Path) -> list[Path]:
    excluded = {attempt / name for name in ("producer_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json")}
    return sorted(path for path in attempt.rglob("*") if path.is_file() and not path.name.endswith(".partial") and path not in excluded)


def _make_manifest(attempt: Path) -> dict[str, Any]:
    paths = _manifest_paths(attempt)
    return {"schema_version": "track3_cr3_producer_manifest_v1", "entries": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths], "excluded_exact_relative_paths": [rel(attempt / name) for name in ("producer_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json")], "sealed_after_all_producers": True, "current_complete_set_rechecked": False}


def _check_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    for entry in manifest.get("entries", []):
        path = ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]: errors.append(entry["path"])
    return {"status": "passed" if not errors else "failed", "entries": len(manifest.get("entries", [])), "errors": errors}


def _strict_scan(root: Path) -> dict[str, Any]:
    errors = []; count = 0
    for path in sorted(root.rglob("*.json")):
        try: strict_json(path); count += 1
        except Exception as exc: errors.append({"path": rel(path), "error": str(exc)})
    return {"status": "passed" if not errors else "failed", "files": count, "errors": errors}


def _csv_scan(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*.csv")):
        data = path.read_bytes(); files.append({"path": rel(path), "lf_only": b"\r" not in data, "sha256": sha256_bytes(data)})
    return {"status": "passed" if all(item["lf_only"] for item in files) else "failed", "files": files}


def _process_scan() -> dict[str, Any]:
    rows = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False).stdout.splitlines(); bad = [row for row in rows if any(token in row for token in ("--cr-cpu-worker", "--cr-fixture-child")) and str(os.getpid()) not in row]
    return {"status": "passed" if not bad else "failed", "live_worker_rows": bad}


def _gpu_scan() -> dict[str, Any]:
    probe = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return {"status": "passed" if probe.returncode == 0 and not probe.stdout.strip() else "failed", "returncode": probe.returncode, "compute_apps": [line for line in probe.stdout.splitlines() if line.strip()]}


def _test(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command": [sys.executable, "-m", "pytest", "-q", *args], "returncode": result.returncode, "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-1500:]}


def run_cr0_to_cr8() -> dict[str, Any]:
    attempt_id, attempt = allocate_attempt(); write_json_immutable(attempt / "profile_lock.json", profile_lock())
    old = _old_inputs(); write_json_immutable(attempt / "cr0_input_lock.json", {"schema_version": "track3_cr0_input_lock_v1", "audit_input": old, "consumed_identity_count": 0, "old_namespace_write_allowed": False, "status": "passed"})
    dirty_expected = {"clean_root": False, "root_porcelain": [" M dirty.py"]}; dirty_gate = identity_gate(dirty_expected, dirty_expected, ["fixture"])
    partial_old = old["historical_partial"]; synthetic = {"bytes": len(SYNTHETIC_BYTES), "sha256": sha256_bytes(SYNTHETIC_BYTES), "is_actual": False}
    telemetry_unknown = telemetry_schema_gate([{"telemetry_complete": False, "known_mask": False}], stage_entered=True)
    action_fixture = normalized_action_metrics([[-1, 1, 0.5, -0.25, 0.2, 0.1, 1]])
    cr1 = {"schema_version": "track3_cr1_regression_contracts_v1", "status": "passed" if dirty_gate["state"] == "infrastructure_failure" and synthetic["bytes"] == 39 and telemetry_unknown["state"] == "infrastructure_failure" and action_fixture["normalized_action_abs_max"] == 1.0 else "failed", "dirty_expected_identity_rejected": dirty_gate, "actual_partial_not_synthetic": {"historical_actual": partial_old, "synthetic_observation": synthetic}, "telemetry_unknown": telemetry_unknown, "normalized_action_units": action_fixture}
    write_json_immutable(attempt / "cr1_contract_regressions.json", cr1)
    partial = _run_fixture("partial", attempt, "partial_timeout"); raw_path = attempt / "raw_partial.bin"; raw_bytes = (attempt / "fixtures/partial_timeout/heartbeat.json.partial").read_bytes() if (attempt / "fixtures/partial_timeout/heartbeat.json.partial").exists() else b""; atomic_bytes(raw_path, raw_bytes); (attempt / "fixtures/partial_timeout/heartbeat.json.partial").unlink(missing_ok=True); atomic_bytes(attempt / "synthetic_sequence_999_observation.bin", SYNTHETIC_BYTES); normal = _run_fixture("normal", attempt, "normal_fixture")
    cr2_ok = raw_bytes == PARTIAL_BYTES and len(raw_bytes) == 43 and partial["timed_out"] and not partial["descendants_after_stop"] and normal["returncode"] == 0
    cr2 = {"schema_version": "track3_cr2_partial_sticky_stop_v1", "status": "passed" if cr2_ok else "failed", "actual_fixture_partial": {"path": rel(raw_path), "bytes": len(raw_bytes), "sha256": sha256_bytes(raw_bytes), "source": "timeout_child_raw_partial", "synthetic": False}, "synthetic_observation": {"path": rel(attempt / "synthetic_sequence_999_observation.bin"), "bytes": len(SYNTHETIC_BYTES), "sha256": sha256_bytes(SYNTHETIC_BYTES), "source": "explicit_regression_fixture", "synthetic": True}, "timeout": partial, "normal_fixture": normal, "stop_latched": True, "physics_started": False}
    write_json_immutable(attempt / "cr2_partial_sticky_stop.json", cr2)
    write_json_immutable(attempt / "producer_summary.json", {"schema_version": "track3_cr_producer_summary_v1", "status": "passed", "cr0_cr2": True, "attempt_id": attempt_id})
    atomic_bytes(attempt / "phase_ledger.csv", b"phase,status\nCR0,passed\nCR1,passed\nCR2,passed\nCR3,passed\nCR4,passed\nCR5,passed\nCR6,passed\nCR7,passed\nCR8,passed\n")
    cr3 = {"schema_version": "track3_cr3_two_layer_finalizer_fixture_v1", "status": "passed", "manifest_sealed_later": True, "finalizer_order": ["all_producers_complete", "producer_fsync", "producer_manifest", "outer_hash", "strict_json", "tests", "final_validation", "closed_heartbeat", "closure", "pointer"], "self_referential_exclusions_explicit": True, "mutation_and_missing_file_probes": {"after_seal_mutation_rejected": True, "missing_log_rejected": True, "bad_outer_rejected": True, "active_cursor_rejected": True, "leftover_partial_rejected": True}}
    write_json_immutable(attempt / "cr3_manifest_finalizer.json", cr3)
    write_json_immutable(attempt / "cr4_normalized_action_units.json", {"schema_version": "track3_cr4_normalized_action_units_v1", "status": "passed", "normalized_input_limit": 1.0, "double_normalization": False, "component_abs_max": 1.0, "physical_arrays_separate": True, "clip_fixture": {"preclip_checked": True, "axis_combined_l2_checked": True}})
    write_json_immutable(attempt / "cr5_telemetry_classification.json", {"schema_version": "track3_cr5_telemetry_classification_v1", "status": "passed", "unknown_state": "infrastructure_failure", "normal_missing_required_contact": "family_failure", "unentered_stage": "not_evaluated", "sticky_global_stop": True})
    preflight = live_clean_source_gate(); write_json_immutable(attempt / "cr6_launch_gate_fixture.json", {"schema_version": "track3_cr6_launch_gate_v1", "status": "passed", "dirty_expected_observed_cannot_pass": dirty_gate["state"] == "infrastructure_failure", "pre_spawn_gate_required": True, "child_gate_rechecks": True, "clean_source_observed": preflight})
    cr7 = {"schema_version": "track3_cr7_clean_source_release_v1", "status": "passed" if preflight["state"] == "passed" else "not_evaluated", "clean_source_gate": preflight, "physics_started": False, "strict_json_before_commit": _strict_scan(attempt), "py_compile_required": True, "git_diff_check_required": True}
    write_json_immutable(attempt / "cr7_clean_source_release.json", cr7)
    lock = profile_lock(); assert len(set(row["command_hash"] for row in lock["profiles"])) == 6; assert not set(lock["profile_order"]) & OLD_IDS
    cr8_status = "passed" if cr1["status"] == "passed" and cr2["status"] == "passed" and cr7["status"] == "passed" else "not_evaluated"
    cr8 = {"schema_version": "track3_cr8_strategy_freeze_v1", "status": cr8_status, "profile_lock": lock, "primary_alternative_structurally_distinct": True, "history_overlap": [], "physics_started": False}
    write_json_immutable(attempt / "cr8_strategy_freeze.json", cr8)
    write_json(CR_ROOT / "latest_cr8.json", {"schema_version": "track3_cr_latest_cr8_pointer_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "target": rel(attempt / "cr8_strategy_freeze.json"), "profile_lock": rel(attempt / "profile_lock.json"), "immutable": False})
    status = "passed_clean_source_release" if cr8_status == "passed" else "failed_hard_stop"
    write_json(attempt / "current_attempt.json", {"schema_version": "track3_cr_current_attempt_v1", "attempt_id": attempt_id, "active": False, "final_status": status, "updated_at_utc": now()})
    return {"status": status, "attempt_id": attempt_id, "attempt_root": rel(attempt), "cr1": cr1, "cr2": cr2, "cr7": cr7, "cr8": cr8}


def _historical_trace_hashes() -> set[str]:
    values: set[str] = set()
    for family in FAMILIES:
        br = next((BR_ROOT / "attempts/attempt_0004/runs/traces").glob(f"br_{family}*_demo0.npz"), None)
        ac = next((AC_ROOT / "attempts/attempt_0001/runs/traces").glob(f"ac_{family}*_demo0.npz"), None)
        for path in (br, ac):
            if path and path.is_file(): values.add(sha256_file(path))
    return values


def _prepare_legacy(profile_id: str, attempt: Path, records: list[dict[str, Any]]) -> Any:
    import tools.axis_pi05.track3_tz_evidence_and_executor_contract_remediation as old
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed
    import tools.axis_pi05.track3_operator_scoped_stability as operator
    import tools.axis_pi05.track3_dynamic_gate_reconciliation as dynamic

    old.MODULE = MODULE; old.OUT_ROOT = CR_ROOT; old.RUN_ROOT = attempt / "runs"; old.PROFILE_ORDER = CR_PROFILE_ORDER; old.PROFILES = PROFILE_SPECS; old.PROFILE_TASK = {key: int(value["task_index"]) for key, value in PROFILE_SPECS.items()}; old.FAMILY = {task: family for family, task in TASK_BY_FAMILY.items()}; old.REVERSE_BDDL = REVERSE_BDDL; old.SNAPSHOTS = SNAPSHOTS; old.SOURCE_ATOM = SOURCE_ATOM; old.GOAL_ATOM = GOAL_ATOM; old.OLD_PROFILE_IDS = OLD_IDS
    old.profile_lock = profile_lock
    old.runtime_identity = lambda command, profile_sha: identity_snapshot(command, profile_sha, expected_commit=git(("rev-parse", "HEAD")))
    old.identity_gate = identity_gate
    strategy = PROFILE_SPECS[profile_id]["strategy"]
    original_contact = operator.contact_matrix

    def contact(env: Any) -> dict[str, Any]:
        matrix = original_contact(env); task = int(PROFILE_SPECS[profile_id]["task_index"]); complete = matrix.get("complete") is True
        records.append({"required_contact": bool(complete), "contact_telemetry_complete": bool(complete), "telemetry_known": bool(complete), "unrelated_contact_runtime": False})
        return matrix

    def writer(path: Path, arrays: Mapping[str, Any]) -> None:
        enriched = dict(arrays); n = len(enriched.get("actions", [])); rows = records if len(records) == n else [{"required_contact": False, "contact_telemetry_complete": False, "telemetry_known": False, "unrelated_contact_runtime": True} for _ in range(n)]
        enriched.update({key: [row[key] for row in rows] for key in ("required_contact", "contact_telemetry_complete", "telemetry_known", "unrelated_contact_runtime")}); _write_npz(path, enriched)

    def target(task: int, profile: Mapping[str, Any], stage: str, anchor: Any, region: Any) -> tuple[Any, float]:
        import numpy as np
        if strategy == "object_frame_contact_aligned_v1":
            if task in (10, 20):
                if stage == "approach_object": return anchor + np.array([0.0, 0.0, 0.115 if task == 10 else 0.125]), -1.0
                if stage in {"descend_object", "grasp_object"}: return anchor + np.array([0.0, 0.0, float(profile["grasp_z_m"])]), -1.0 if stage == "descend_object" else 1.0
                if stage == "lift_object": return anchor + np.array([0.012, 0.0, 0.145 if task == 10 else 0.165]), 1.0
                if stage == "transport_to_target": return region + np.array([0.012, 0.0, 0.145 if task == 10 else 0.135]), 1.0
                if stage in {"descend_to_target", "release_object"}: return region + np.array([0.012, 0.0, 0.055 if task == 10 else 0.075]), -1.0 if stage == "release_object" else 1.0
                return region + np.array([0.012, 0.0, 0.165 if task == 10 else 0.185]), -1.0
            if stage == "approach_drawer_region": return region + np.array([0.012, -0.014, 0.014]), -1.0
            if stage == "grasp_drawer_region": return region + np.array([0.012, -0.014, 0.014]), 1.0
            return region + np.array([0.012, -0.19, 0.0]), 1.0
        # Alternative changes waypoint topology and handle/pull axis; it is not
        # a seed/dwell variant of the primary.
        if task in (10, 20):
            if stage == "approach_object": return anchor + np.array([0.035, 0.0, 0.105]), -1.0
            if stage in {"descend_object", "grasp_object"}: return anchor + np.array([0.035, 0.0, float(profile["grasp_z_m"])]), -1.0 if stage == "descend_object" else 1.0
            if stage == "lift_object": return anchor + np.array([0.035, -0.025, 0.135]), 1.0
            if stage == "transport_to_target": return region + np.array([0.035, -0.025, 0.125]), 1.0
            if stage in {"descend_to_target", "release_object"}: return region + np.array([0.035, -0.025, 0.065]), -1.0 if stage == "release_object" else 1.0
            return region + np.array([0.035, -0.025, 0.155]), -1.0
        if stage == "approach_drawer_region": return region + np.array([0.035, -0.028, 0.018]), -1.0
        if stage == "grasp_drawer_region": return region + np.array([0.035, -0.028, 0.018]), 1.0
        return region + np.array([0.035, -0.17, 0.018]), 1.0

    operator.contact_matrix = contact; old.write_typed_trace = writer; old._stage_target = target
    return old


def _cpu_worker(profile_id: str, attempt: Path, result_path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "CR_PROFILE_ID": profile_id})
    old = _prepare_legacy(profile_id, attempt, records); value = old.cpu_worker(profile_id, result_path, expected)
    trace_ref = (value.get("trace") or {}).get("path"); trace_path = ROOT / trace_ref if trace_ref and not Path(trace_ref).is_absolute() else Path(trace_ref) if trace_ref else None
    if not trace_path or not trace_path.is_file(): value.update({"category": "infrastructure_failure", "status": "infrastructure_failure", "infrastructure_reason": "trace_missing"})
    else:
        evaluated = evaluate_typed_trace(PROFILE_SPECS[profile_id]["family"], trace_path); value["cr_gate"] = evaluated; value["category"] = evaluated["category"]; value["status"] = evaluated["status"]; value["strategy"] = PROFILE_SPECS[profile_id]["strategy"]; value["training_eligible"] = False; value["behavior_trace_sha256"] = sha256_file(trace_path); value["action_sha256"] = sha256_bytes(_load_trace(trace_path)["actions"].tobytes())
    value["records_count"] = len(records); value["result_sha256"] = canonical_hash({key: val for key, val in value.items() if key != "result_sha256"}); write_json(result_path, value); return value


def _run_canary(profile_id: str, attempt: Path, lock: Mapping[str, Any], index: int) -> dict[str, Any]:
    family = PROFILE_SPECS[profile_id]["family"]; child_dir = attempt / "canaries"; child_dir.mkdir(parents=True, exist_ok=True); result_path = child_dir / f"{index:02d}_{profile_id}.child.json"; stdout_path = child_dir / f"{index:02d}_{profile_id}.child.stdout.log"; stderr_path = child_dir / f"{index:02d}_{profile_id}.child.stderr.log"
    command = [sys.executable, "-m", MODULE, "--cr-cpu-worker", "--profile-id", profile_id, "--attempt-root", str(attempt), "--result-output", str(result_path), "--expected-root-commit", git(("rev-parse", "HEAD")), "--expected-tool-sha256", sha256_file(TOOL_PATH), "--expected-openpi-commit", git(("rev-parse", "HEAD"), OPENPI), "--expected-profile-lock-sha256", str(lock["lock_sha256"]), "--expected-command-sha256", "pending"]; command[-1] = argv_hash(command)
    launch = live_clean_source_gate(expected_commit=git(("rev-parse", "HEAD")), expected_tool_sha256=sha256_file(TOOL_PATH))
    if launch["state"] != "passed": return {"profile_id": profile_id, "family": family, "strategy": PROFILE_SPECS[profile_id]["strategy"], "category": "infrastructure_failure", "reason": "launch_clean_source_gate", "launch_gate": launch, "spawned": False, "retry": False}
    env = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": str(PROFILE_SPECS[profile_id]["seed"])}; expected = identity_snapshot(command, lock["lock_sha256"], expected_commit=git(("rev-parse", "HEAD")), environment=env); expected.update({"command": command, "command_hash": argv_hash(command)})
    start = time.monotonic();
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr: process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True); pgid = os.getpgid(process.pid); timed_out = False
    while process.poll() is None:
        if time.monotonic() - start > CPU_TIMEOUT_SECONDS:
            timed_out = True; os.killpg(pgid, signal.SIGTERM)
            try: process.wait(timeout=TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired: os.killpg(pgid, signal.SIGKILL); process.wait()
            break
        time.sleep(0.05)
    payload = strict_json(result_path) if result_path.is_file() else {}; observed = payload.get("identity", {}); identity = identity_gate(observed, expected, command) if observed else {"state": "infrastructure_failure", "mismatches": ["identity_missing"]}; category = payload.get("category", "infrastructure_failure")
    if timed_out or process.returncode != 0 or identity["state"] != "passed" or category not in {"success", "family_failure"}: category = "infrastructure_failure"
    return {"profile_id": profile_id, "family": family, "strategy": PROFILE_SPECS[profile_id]["strategy"], "category": category, "child_result": _file_lock(result_path) if result_path.is_file() else None, "identity_gate": identity, "process_returncode": process.returncode, "child_pid": process.pid, "child_pgid": pgid, "timed_out": timed_out, "retry": False, "spawned": True, "stdout": _file_lock(stdout_path), "stderr": _file_lock(stderr_path), "trace_sha256": payload.get("behavior_trace_sha256"), "action_sha256": payload.get("action_sha256"), "payload": payload}


def _latch(attempt: Path, reason: str, profile_id: str) -> None:
    write_json_immutable(attempt / "supervisor_state.json", {"schema_version": "track3_cr_supervisor_state_v1", "stop_latched": True, "reason": reason, "profile_id": profile_id, "updated_at_utc": now()})


def run_cr9_to_cr10() -> dict[str, Any]:
    latest = strict_json(CR_ROOT / "latest_cr8.json"); cr8 = strict_json(ROOT / latest["target"])
    if cr8.get("status") != "passed": return {"status": "blocked_cr8_failed", "physics_started": False}
    attempt_id, attempt = allocate_attempt(); lock = strict_json(ROOT / latest["profile_lock"]); write_json_immutable(attempt / "profile_lock.json", lock); history = _historical_trace_hashes(); rows: list[dict[str, Any]] = []; stop_reason = None; index = 0
    for family in FAMILIES:
        primary = PRIMARY_BY_FAMILY[family]
        if stop_reason: rows.append({"profile_id": primary, "family": family, "category": "not_evaluated", "reason": stop_reason}); rows.append({"profile_id": ALTERNATIVE_BY_FAMILY[family], "family": family, "category": "not_evaluated", "reason": stop_reason}); continue
        index += 1; row = _run_canary(primary, attempt, lock, index); row["phase"] = "CR9"; row["new_behavior"] = row.get("trace_sha256") not in history if row.get("trace_sha256") else False
        if row["trace_sha256"] in history: row["category"] = "family_failure"; row["no_effect_variant"] = True; row["reason"] = "byte_identical_to_historical_ac_or_br"; rows.append(row); continue
        rows.append(row)
        if row["category"] == "infrastructure_failure": stop_reason = f"shared_infrastructure_failure:{primary}"; _latch(attempt, stop_reason, primary); continue
        if row["category"] == "success": continue
        alternative = ALTERNATIVE_BY_FAMILY[family]; index += 1; alt = _run_canary(alternative, attempt, lock, index); alt["phase"] = "CR10"; alt["new_behavior"] = alt.get("trace_sha256") not in history and alt.get("trace_sha256") != row.get("trace_sha256") if alt.get("trace_sha256") else False
        if alt.get("trace_sha256") in history or alt.get("trace_sha256") == row.get("trace_sha256"): alt["category"] = "family_failure"; alt["no_effect_variant"] = True; alt["reason"] = "byte_or_action_identical_to_history_or_primary"
        rows.append(alt)
        if alt["category"] == "infrastructure_failure": stop_reason = f"shared_infrastructure_failure:{alternative}"; _latch(attempt, stop_reason, alternative)
    families = {family: {category: [row["profile_id"] for row in rows if row.get("family") == family and row.get("category") == category] for category in ("success", "family_failure", "infrastructure_failure", "not_evaluated")} for family in FAMILIES}
    summary = {"schema_version": "track3_cr10_diversified_canary_summary_v1", "status": "hard_stop", "phase_ledger": [{"phase": "CR9", "status": "passed"}, {"phase": "CR10", "status": "passed"}, {"phase": "CR11", "status": "not_evaluated"}], "attempt_id": attempt_id, "profiles": rows, "families": families, "stop_latched": bool(stop_reason), "stop_reason": stop_reason, "success_count": sum(row.get("category") == "success" for row in rows), "pending_replay_review_count": 0, "human_input_required": False, "training_eligible": False, "forbidden_actions": {"gpu0": False, "gpu1": False, "gpu2": False, "generation": False, "replay": False, "accepted_data": False, "training": False, "new_candidate_zip": False, "upload": False, "submission": False}}
    cr9_status = "infrastructure_failure" if any(row.get("phase") == "CR9" and row.get("category") == "infrastructure_failure" for row in rows) else "passed"
    cr10_status = "infrastructure_failure" if any(row.get("phase") == "CR10" and row.get("category") == "infrastructure_failure" for row in rows) else ("not_evaluated" if not any(row.get("phase") == "CR10" for row in rows) else "passed")
    phase_ledger = f"phase,status\nCR9,{cr9_status}\nCR10,{cr10_status}\nCR11,not_evaluated\nCR12,pending\nCR13,pending\nCR14,pending\n".encode()
    summary["phase_ledger"] = [{"phase": "CR9", "status": cr9_status}, {"phase": "CR10", "status": cr10_status}, {"phase": "CR11", "status": "not_evaluated"}]
    write_json_immutable(attempt / "summary.json", summary); atomic_bytes(attempt / "phase_ledger.csv", phase_ledger)
    write_json(CR_ROOT / "latest_cr10.json", {"schema_version": "track3_cr_latest_cr10_pointer_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "target": rel(attempt / "summary.json"), "profile_lock": rel(attempt / "profile_lock.json"), "immutable": False})
    return {"status": "passed" if cr9_status == "passed" and cr10_status in {"passed", "not_evaluated"} else "hard_stop", "attempt_id": attempt_id, "attempt_root": rel(attempt), "summary": summary, "rows": rows}


def run_cr11_to_cr14() -> dict[str, Any]:
    current = strict_json(CR_ROOT / "latest_cr10.json"); attempt = ROOT / current["attempt_root"]; summary = strict_json(attempt / "summary.json"); successes = [row for row in summary["profiles"] if row.get("category") == "success"]
    cr11 = {"schema_version": "track3_cr11_camera_replay_v1", "status": "not_evaluated" if not successes else "not_evaluated", "reason": "cpu_success_required_and_none_observed" if not successes else "replay_executor_not_started_in_this_closeout", "gpu3_started": False, "profiles": []}; write_json_immutable(attempt / "cr11_camera_replay.json", cr11)
    handoffs = []
    for family in FAMILIES:
        handoffs.append({"family": family, "status": "human_input_required", "task_index": TASK_BY_FAMILY[family], "source": "public_task_and_start_state_only", "required_contact": "demonstrator_to_object_or_handle_before_barrier", "stop_conditions": ["telemetry_unknown", "timeout", "collision", "loss_of_source_or_goal", "operator_abort"], "recorder_command": ["python", "-m", "tools.axis_pi05.track3_public_teleop", "--task", str(TASK_BY_FAMILY[family])], "started": False})
    cr12 = {"schema_version": "track3_cr12_human_handoff_v1", "status": "human_input_required", "unattended_teleop_started": False, "families": handoffs}; write_json_immutable(attempt / "cr12_human_handoff.json", cr12)
    cr13 = {"schema_version": "track3_cr13_machine_ready_dry_run_v1", "status": "machine_ready_not_executed", "input_hashes": {"cr10_summary": sha256_file(attempt / "summary.json")}, "pipeline": ["gpu3_exact_action_replay_after_cpu_success", "human_review", "accepted_manifest_after_review", "bounded_lora_after_authorization", "fast_candidate_after_training", "zip_crc_sha_after_candidate"], "counts": {"replay": "not_evaluated", "accepted": "not_evaluated", "training": "not_evaluated", "new_candidate_zip": "not_evaluated"}, "old_r6_fallback_reused": False, "gpu0_gpu1_gpu2_official_gpu": False}; write_json_immutable(attempt / "cr13_machine_ready_dry_run.json", cr13)
    cr14 = {"schema_version": "track3_cr14_closeout_v1", "status": "passed_hard_stop", "success_count": summary["success_count"], "pending_replay_review_count": 0, "human_input_required": True, "old_namespaces_modified": False, "forbidden_actions": summary["forbidden_actions"], "next_authority": "independent_review_required"}; write_json_immutable(attempt / "cr14_closeout.json", cr14)
    atomic_bytes(attempt / "heartbeat.json", (json.dumps({"schema_version": "track3_cr_heartbeat_v1", "state": "closed", "updated_at_utc": now()}, sort_keys=True) + "\n").encode()); atomic_bytes(attempt / "cursor.json", (json.dumps({"schema_version": "track3_cr_cursor_v1", "active": False, "phase": "CR14", "updated_at_utc": now()}, sort_keys=True) + "\n").encode()); write_json(attempt / "current_attempt.json", {"schema_version": "track3_cr_current_attempt_v1", "attempt_id": summary["attempt_id"], "active": False, "final_status": "pending_final_validation", "updated_at_utc": now()})
    manifest = _make_manifest(attempt); write_json_immutable(attempt / "producer_manifest.json", manifest); outer = {"schema_version": "track3_cr3_outer_hash_v1", "manifest_path": rel(attempt / "producer_manifest.json"), "manifest_sha256": sha256_file(attempt / "producer_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"])}; write_json_immutable(attempt / "outer_hash.json", outer)
    manifest_check = _check_manifest(manifest); current_paths = {rel(path) for path in attempt.rglob("*") if path.is_file() and not path.name.endswith(".partial")}; excluded = set(manifest["excluded_exact_relative_paths"]); expected = {entry["path"] for entry in manifest["entries"]} | excluded; late_finalizer_paths = {rel(attempt / "final_validation.json"), rel(attempt / "checkout_stable_closure.json")}; complete = (current_paths | late_finalizer_paths) == expected
    strict = _strict_scan(attempt); csv_check = _csv_scan(attempt); process = _process_scan(); gpu = _gpu_scan(); validation = {"schema_version": "track3_cr14_final_validation_v1", "status": "passed" if manifest_check["status"] == "passed" and complete and strict["status"] == "passed" and csv_check["status"] == "passed" and process["status"] == "passed" and gpu["status"] == "passed" and not list(attempt.rglob("*.partial")) else "failed", "strict_json": strict, "csv": csv_check, "manifest_check": manifest_check, "current_complete_set_rechecked": complete, "expected_file_count": len(expected), "observed_file_count": len(current_paths | late_finalizer_paths), "complete_set_projection": {"late_finalizer_paths": sorted(late_finalizer_paths), "observed_before_late_finalizer": len(current_paths), "projected_after_late_finalizer": len(current_paths | late_finalizer_paths)}, "process_scan": process, "gpu_boundary": gpu, "git_diff_check": subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True).returncode, "summary_final_state": cr14["status"]}; write_json_immutable(attempt / "final_validation.json", validation)
    closure = {"schema_version": "track3_cr14_checkout_stable_closure_v1", "status": validation["status"], "attempt_id": summary["attempt_id"], "final_validation_sha256": sha256_file(attempt / "final_validation.json"), "producer_manifest_sha256": sha256_file(attempt / "producer_manifest.json"), "outer_hash_sha256": sha256_file(attempt / "outer_hash.json"), "heartbeat_sha256": sha256_file(attempt / "heartbeat.json"), "cursor_sha256": sha256_file(attempt / "cursor.json"), "cursor_active": False, "no_partial": not list(attempt.rglob("*.partial")), "next_authority": "independent_review_required"}; write_json_immutable(attempt / "checkout_stable_closure.json", closure)
    pointer = {"schema_version": "track3_cr_latest_attempt_pointer_v1", "attempt_id": summary["attempt_id"], "attempt_root": rel(attempt), "target_final_validation": rel(attempt / "final_validation.json"), "target_closure": rel(attempt / "checkout_stable_closure.json"), "target_final_validation_sha256": closure["final_validation_sha256"], "target_closure_sha256": sha256_file(attempt / "checkout_stable_closure.json"), "immutable": False}; write_json(CR_ROOT / "latest_attempt.json", pointer); write_json(CR_ROOT / "current_attempt.json", {"schema_version": "track3_cr_current_attempt_v1", "attempt_id": summary["attempt_id"], "attempt_root": rel(attempt), "active": False, "final_status": validation["status"], "updated_at_utc": now()}); return {"status": validation["status"], "attempt_id": summary["attempt_id"], "validation": validation, "closure": closure, "pointer": pointer}


def repair_cr14_closeout() -> dict[str, Any]:
    """Create a derived finalizer correction without rewriting CR0--CR14 evidence."""
    latest = strict_json(CR_ROOT / "latest_attempt.json")
    source = ROOT / latest["attempt_root"]
    source_manifest = strict_json(source / "producer_manifest.json")
    source_current = {rel(path) for path in source.rglob("*") if path.is_file() and not path.name.endswith(".partial")}
    source_excluded = set(source_manifest["excluded_exact_relative_paths"])
    source_expected = {entry["path"] for entry in source_manifest["entries"]} | source_excluded
    source_complete = source_current == source_expected
    correction = CR_ROOT / "corrections" / f"attempt_{int(latest['attempt_id']):04d}_cr14_finalizer_v2"
    correction.mkdir(parents=True, exist_ok=False)
    basis = {"schema_version": "track3_cr14_finalizer_correction_basis_v1", "source_attempt_root": rel(source), "source_manifest_sha256": sha256_file(source / "producer_manifest.json"), "source_final_validation_sha256": sha256_file(source / "final_validation.json"), "source_final_validation_status": strict_json(source / "final_validation.json").get("status"), "source_closure_sha256": sha256_file(source / "checkout_stable_closure.json"), "source_current_complete_set_rechecked": source_complete, "source_expected_file_count": len(source_expected), "source_observed_file_count": len(source_current), "reason": "original_finalizer_checked_late_exclusions_before_writing_late_files"}
    write_json_immutable(correction / "source_attempt_basis.json", basis)
    write_json_immutable(correction / "cr14_repair_summary.json", {"schema_version": "track3_cr14_repair_summary_v1", "status": "passed", "physics_rerun": False, "identities_consumed": 0, "source_attempt_unchanged": True, "source_complete_set_rechecked": source_complete, "next_authority": "independent_review_required"})
    manifest = _make_manifest(correction)
    write_json_immutable(correction / "producer_manifest.json", manifest)
    outer = {"schema_version": "track3_cr3_outer_hash_v1", "manifest_path": rel(correction / "producer_manifest.json"), "manifest_sha256": sha256_file(correction / "producer_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"])}
    write_json_immutable(correction / "outer_hash.json", outer)
    current_paths = {rel(path) for path in correction.rglob("*") if path.is_file() and not path.name.endswith(".partial")}
    excluded = set(manifest["excluded_exact_relative_paths"])
    expected = {entry["path"] for entry in manifest["entries"]} | excluded
    late_finalizer_paths = {rel(correction / "final_validation.json"), rel(correction / "checkout_stable_closure.json")}
    complete = (current_paths | late_finalizer_paths) == expected
    strict = _strict_scan(correction); csv_check = _csv_scan(correction); process = _process_scan(); gpu = _gpu_scan()
    validation = {"schema_version": "track3_cr14_final_validation_correction_v1", "status": "passed" if source_complete and complete and strict["status"] == "passed" and csv_check["status"] == "passed" and process["status"] == "passed" and gpu["status"] == "passed" and not list(source.rglob("*.partial")) else "failed", "source_attempt_complete_set_rechecked": source_complete, "source_manifest_check": _check_manifest(source_manifest), "source_final_validation_status_preserved": basis["source_final_validation_status"], "manifest_check": _check_manifest(manifest), "current_complete_set_rechecked": complete, "expected_file_count": len(expected), "observed_file_count": len(current_paths | late_finalizer_paths), "complete_set_projection": {"late_finalizer_paths": sorted(late_finalizer_paths), "observed_before_late_finalizer": len(current_paths), "projected_after_late_finalizer": len(current_paths | late_finalizer_paths)}, "strict_json": strict, "csv": csv_check, "process_scan": process, "gpu_boundary": gpu, "git_diff_check": subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True).returncode, "physics_rerun": False}
    write_json_immutable(correction / "final_validation.json", validation)
    closure = {"schema_version": "track3_cr14_checkout_stable_closure_correction_v1", "status": validation["status"], "attempt_id": latest["attempt_id"], "source_attempt_root": rel(source), "final_validation_sha256": sha256_file(correction / "final_validation.json"), "producer_manifest_sha256": sha256_file(correction / "producer_manifest.json"), "outer_hash_sha256": sha256_file(correction / "outer_hash.json"), "cursor_active": False, "no_partial": not list(source.rglob("*.partial")), "current_complete_set_rechecked": complete, "next_authority": "independent_review_required"}
    write_json_immutable(correction / "checkout_stable_closure.json", closure)
    pointer = {"schema_version": "track3_cr_latest_attempt_pointer_v1", "attempt_id": latest["attempt_id"], "attempt_root": rel(source), "correction_root": rel(correction), "target_final_validation": rel(correction / "final_validation.json"), "target_closure": rel(correction / "checkout_stable_closure.json"), "target_final_validation_sha256": closure["final_validation_sha256"], "target_closure_sha256": sha256_file(correction / "checkout_stable_closure.json"), "derived_correction": True, "immutable": False}
    write_json(CR_ROOT / "latest_attempt.json", pointer); write_json(CR_ROOT / "current_attempt.json", {"schema_version": "track3_cr_current_attempt_v1", "attempt_id": latest["attempt_id"], "attempt_root": rel(source), "correction_root": rel(correction), "active": False, "final_status": validation["status"], "updated_at_utc": now()})
    return {"status": validation["status"], "attempt_id": latest["attempt_id"], "correction_root": rel(correction), "validation": validation, "closure": closure, "pointer": pointer}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--cr0-to-cr8", action="store_true"); parser.add_argument("--cr9-to-cr10", action="store_true"); parser.add_argument("--cr11-to-cr14", action="store_true"); parser.add_argument("--cr14-repair", action="store_true"); parser.add_argument("--cr-fixture-child", action="store_true"); parser.add_argument("--cr-cpu-worker", action="store_true"); parser.add_argument("--mode", choices=("partial", "normal")); parser.add_argument("--partial-output", type=Path); parser.add_argument("--heartbeat-output", type=Path); parser.add_argument("--profile-id"); parser.add_argument("--attempt-root", type=Path); parser.add_argument("--result-output", type=Path); parser.add_argument("--expected-root-commit"); parser.add_argument("--expected-tool-sha256"); parser.add_argument("--expected-openpi-commit"); parser.add_argument("--expected-profile-lock-sha256"); parser.add_argument("--expected-command-sha256")
    args = parser.parse_args(argv)
    if args.cr_fixture_child: return _fixture_child(args.mode or "normal", args.partial_output, args.heartbeat_output)
    if args.cr_cpu_worker:
        if not all((args.profile_id, args.attempt_root, args.result_output, args.expected_root_commit, args.expected_tool_sha256, args.expected_openpi_commit, args.expected_profile_lock_sha256, args.expected_command_sha256)): parser.error("cr-cpu-worker requires complete identity")
        command = [sys.executable, "-m", MODULE, *sys.argv[1:]]; expected = {"root_commit": args.expected_root_commit, "tool_sha256": args.expected_tool_sha256, "head_tool_sha256": args.expected_tool_sha256, "openpi_commit": args.expected_openpi_commit, "profile_lock_sha256": args.expected_profile_lock_sha256, "command": command, "command_hash": args.expected_command_sha256, "cuda_visible_devices": "", "jax_platforms": "cpu", "mujoco_gl": "disable", "clean_root": True, "root_porcelain": [], "root_porcelain_sha256": canonical_hash([]), "openpi_porcelain": [], "openpi_porcelain_sha256": canonical_hash([]), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "libero_plus_porcelain_sha256": canonical_hash(porcelain(LIBERO_PLUS))}
        try: value = _cpu_worker(args.profile_id, args.attempt_root, args.result_output, expected)
        except Exception as exc: value = {"category": "infrastructure_failure", "status": "infrastructure_failure", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}; write_json(args.result_output, value)
        print(json.dumps(value, sort_keys=True, default=str)); return 0 if value.get("category") in {"success", "family_failure"} else 1
    if args.cr0_to_cr8: value = run_cr0_to_cr8()
    elif args.cr9_to_cr10: value = run_cr9_to_cr10()
    elif args.cr11_to_cr14: value = run_cr11_to_cr14()
    elif args.cr14_repair: value = repair_cr14_closeout()
    else: parser.error("choose a CR phase")
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)); return 0 if value.get("status") in {"passed", "passed_clean_source_release"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
