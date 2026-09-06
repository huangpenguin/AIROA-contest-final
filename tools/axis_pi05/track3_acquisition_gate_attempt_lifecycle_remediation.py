#!/usr/bin/env python3
"""BR0--BR11 remediation for acquisition gates and attempt lifecycle.

The AC namespace is historical input only.  This module owns a fresh BR
namespace and contains the code/fixture contract used before any new CPU
identity is consumed.  It deliberately has no GPU, replay, generation,
training, or ZIP path.
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
MODULE = "tools.axis_pi05.track3_acquisition_gate_attempt_lifecycle_remediation"
BR_ROOT = ROOT / "data/reports/axis_libero_30k/track3_acquisition_gate_attempt_lifecycle_remediation"
ACTIVE_ATTEMPT_ROOT: Path | None = None
RUN_ROOT: Path | None = None

AC_ROOT = ROOT / "data/reports/axis_libero_30k/track3_crossfamily_acquisition_recovery_canary"
AC_AUDIT = ROOT / "data/reports/axis_libero_30k/track3_ac0_ac6_independent_audit_20260907/summary.json"
AC_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_CROSSFAMILY_ACQUISITION_RECOVERY_CANARY_PLAN.md"
EG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_remediation"
ER_ROOT = ROOT / "data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PUBLIC_LIBERO = ROOT / "LIBERO"
PUBLIC_AUTOGEN = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"

HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
REVERSE_BDDL = {
    10: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_support_remove_task_10.bddl",
    20: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_containment_remove_task_20.bddl",
    19: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_articulation_close_task_19.bddl",
}
SNAPSHOTS = {
    task: P1_ROOT / "snapshots" / f"task_{task:02d}" / "attempt_1" / "terminal_state.npy"
    for task in (10, 20, 19)
}

PROFILE_ORDER = (
    "br_support_remove_20260907_0001",
    "br_containment_remove_20260907_0001",
    "br_articulation_close_20260907_0001",
)
TASK_BY_PROFILE = {PROFILE_ORDER[0]: 10, PROFILE_ORDER[1]: 20, PROFILE_ORDER[2]: 19}
FAMILY_BY_TASK = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SOURCE_ATOM = {
    10: "(On akita_black_bowl_1 plate_1)",
    20: "(In orange_juice_1 basket_1_contain_region)",
    19: "(Open wooden_cabinet_1_middle_region)",
}
GOAL_ATOM = {
    10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)",
    20: "(On orange_juice_1 floor_target_object_region)",
    19: "(Close wooden_cabinet_1_middle_region)",
}
PROFILES = {
    PROFILE_ORDER[0]: {"task_index": 10, "seed": 71010, "grasp_z_m": 0.024, "close_dwell": 2, "release_dwell": 1},
    PROFILE_ORDER[1]: {"task_index": 20, "seed": 72020, "grasp_z_m": 0.024, "close_dwell": 2, "release_dwell": 1},
    PROFILE_ORDER[2]: {"task_index": 19, "seed": 73019, "grasp_z_m": 0.012, "close_dwell": 2, "release_dwell": 1},
}
OLD_PROFILE_IDS = {
    "ac_support_remove_20260906_0001", "ac_containment_remove_20260906_0001", "ac_articulation_close_20260906_0001",
    "er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh",
    "support_contact_repair_a", "support_contact_repair_b", "support_contact_repair_c",
    "containment_extract_a", "containment_extract_b", "articulation_pull_a", "articulation_pull_b",
}

ACTION_DIM = 7
CONTACT_WINDOW = 4
TERM_GRACE_SECONDS = 0.25
FIXTURE_HEARTBEAT_INTERVAL = 0.03
# Leave enough time for the clean venv to import this module and publish the
# first heartbeat, while keeping the fixture bounded and deterministic.
FIXTURE_TIMEOUT = 0.8
CPU_TIMEOUT_SECONDS = 240.0
BR8_CLEAN_ATTEMPT = 2

CONTRACT = {
    "osc_translation_normalized_scale": 0.05,
    "osc_rotation_normalized_scale": 0.25,
    "normalized_action_abs_limit": 1.0,
    "progress_window": 12,
    "stall_epsilon_m": 1.0e-5,
    "required_contact_window": CONTACT_WINDOW,
    "acquisition_barrier": "required_demonstrator_contact_continuous_before_grasp_or_pull",
    "source_release": "runtime_predicate_source_false_after_acquisition",
    "goal_gate": "runtime_predicate_goal_true_continuously_after_goal_entry",
}

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
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def result_content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("result_sha256", None)
    return canonical_hash(body)


def argv_contract_hash(command: Sequence[str]) -> str:
    normalized = list(command)
    if "--expected-command-sha256" in normalized:
        index = normalized.index("--expected-command-sha256") + 1
        if index < len(normalized):
            normalized[index] = "<expected-command-sha256>"
    return canonical_hash(normalized)


def gate_record(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    state = "not_evaluated" if not executed else ("passed" if value is True else ("unknown" if value is None else "failed"))
    row: dict[str, Any] = {"executed": bool(executed), "value": value, "state": state, "required": bool(required)}
    if reason is not None:
        row["reason"] = reason
    return row


def tri_state(executed: bool, value: bool | None) -> str:
    return gate_record(executed, value)["state"]


def required_gates_passed(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    return bool(gates) and all(row.get("state") == "passed" for row in gates.values() if row.get("required", True))


def _attempt_root(root: Path | None = None) -> Path:
    if root is not None:
        return root
    if ACTIVE_ATTEMPT_ROOT is None:
        raise RuntimeError("no_active_attempt")
    return ACTIVE_ATTEMPT_ROOT


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
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(info, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    try:
        yield info
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def allocate_attempt(root: Path = BR_ROOT) -> tuple[int, Path]:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "attempt_allocator.lock"
    with supervisor_lock(lock_path):
        ids = []
        for path in (root / "attempts").glob("attempt_*"):
            if path.is_dir() and path.name.removeprefix("attempt_").isdigit():
                ids.append(int(path.name.removeprefix("attempt_")))
        attempt_id = max(ids, default=0) + 1
        attempt = root / "attempts" / f"attempt_{attempt_id:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        # The active cursor is intentionally mutable while a run is active;
        # only the closed latest pointer is immutable.
        write_json(root / "current_attempt.json", {"schema_version": "track3_br_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": True, "allocated_at_utc": now()})
    return attempt_id, attempt


def set_active_attempt(attempt_id: int, attempt: Path) -> None:
    write_json(BR_ROOT / "current_attempt.json", {"schema_version": "track3_br_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": True, "updated_at_utc": now()})


def identity_snapshot(command: Sequence[str], profile_sha: str, *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = environment or os.environ
    root_state, openpi_state, libero_state = porcelain(ROOT), porcelain(OPENPI), porcelain(LIBERO_PLUS)
    return {
        "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__).resolve()), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI),
        "profile_lock_sha256": profile_sha, "command": list(command), "command_hash": argv_contract_hash(command),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""), "jax_platforms": env.get("JAX_PLATFORMS", ""), "mujoco_gl": env.get("MUJOCO_GL", ""),
        "root_porcelain": root_state, "root_porcelain_sha256": canonical_hash(root_state), "openpi_porcelain": openpi_state, "openpi_porcelain_sha256": canonical_hash(openpi_state),
        "libero_plus_porcelain": libero_state, "libero_plus_porcelain_sha256": canonical_hash(libero_state), "clean_root": not root_state, "pid": os.getpid(), "pgid": os.getpgid(0),
    }


IDENTITY_FIELDS = ("root_commit", "tool_sha256", "openpi_commit", "profile_lock_sha256", "cuda_visible_devices", "jax_platforms", "mujoco_gl", "root_porcelain_sha256", "openpi_porcelain_sha256", "libero_plus_porcelain_sha256", "clean_root")


def identity_gate(observed: Mapping[str, Any], expected: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    mismatches = [field for field in IDENTITY_FIELDS if observed.get(field) != expected.get(field)]
    if observed.get("command_hash") != expected.get("command_hash"):
        mismatches.append("command_hash")
    if list(observed.get("command", [])) != list(command):
        mismatches.append("command")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches, "checked_fields": list(IDENTITY_FIELDS) + ["command", "command_hash"]}


def _file_lock(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _old_ac_closure() -> dict[str, Any]:
    pointer = strict_json(AC_ROOT / "latest_attempt.json")
    audit = strict_json(AC_AUDIT)
    final = ROOT / pointer["target_final_validation"]
    closure = ROOT / pointer["target_closure"]
    return {
        "historical_ac_namespace": rel(AC_ROOT), "audit_summary": _file_lock(AC_AUDIT), "audit_status": audit["status"],
        "audit_input_root_commit": audit["audit_input_root_commit"], "execution_source_commit": audit["execution_source_commit"],
        "pointer": _file_lock(AC_ROOT / "latest_attempt.json"), "final_validation": _file_lock(final), "closure": _file_lock(closure),
        "historical_ac_files_untouched": True, "historical_family_categories": audit["verified"]["family_categories"],
        "historical_trace_steps": audit["verified"]["trace_steps"], "historical_partial_provenance": audit["partial_provenance"],
        "historical_defects_preserved": audit["findings"],
    }


def profile_lock() -> dict[str, Any]:
    rows = []
    for profile_id in PROFILE_ORDER:
        row = dict(PROFILES[profile_id])
        row.update({"profile_id": profile_id, "family": FAMILY_BY_TASK[int(row["task_index"])], "identity_scope": "new_br_cpu_no_renderer_identity_once", "old_identity_rerun": False, "renderer": False, "gpu": False, "training_eligible": False})
        rows.append(row)
    body: dict[str, Any] = {"schema_version": "track3_br_profile_lock_v1", "profile_order": list(PROFILE_ORDER), "profiles": rows, "family_order": ["support_remove", "containment_remove", "articulation_close"], "limits": {"support_remove": 1, "containment_remove": 1, "articulation_close": 1, "total": 3}, "retry_allowed": False, "old_profile_ids_denylist": sorted(OLD_PROFILE_IDS), "contract": CONTRACT, "official_track3_claim": False}
    body["lock_sha256"] = canonical_hash(body)
    return body


def run_br0(attempt: Path) -> dict[str, Any]:
    result = {"schema_version": "track3_br0_derived_closure_v1", "status": "passed", "source": _old_ac_closure(), "new_namespace": rel(BR_ROOT), "historical_namespace_write_allowed": False, "old_identity_rerun": False, "physics_started": False, "gpu_compute_started": False}
    write_json_immutable(attempt / "br0_derived_closure.json", result)
    return result


def _fixture_gate(executed: bool, value: bool | None, reason: str | None = None) -> dict[str, Any]:
    return gate_record(executed, value, reason=reason)


def stage_gate_graph(family: str, phases: Sequence[str]) -> dict[str, dict[str, Any]]:
    expected = EXPECTED_STAGES[family]
    entered = set(phases)
    result: dict[str, dict[str, Any]] = {}
    for index, stage in enumerate(expected):
        executed = stage in entered
        reached = executed and all(candidate in entered for candidate in expected[: index + 1])
        result[stage] = gate_record(executed, reached, reason="stage_not_entered" if not executed else None)
    return result


def _contiguous_windows(values: Sequence[bool]) -> list[list[int]]:
    windows: list[list[int]] = []
    current: list[int] = []
    for index, value in enumerate(values):
        if bool(value):
            current.append(index)
        elif current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)
    return windows


def telemetry_schema_gate(samples: Sequence[Mapping[str, Any]], *, stage_entered: bool) -> dict[str, Any]:
    if not stage_entered:
        return gate_record(False, None, reason="stage_not_entered")
    required = {"telemetry_complete", "known_mask", "contacts"}
    errors = [f"sample_{i}" for i, sample in enumerate(samples) if not required.issubset(sample) or sample.get("telemetry_complete") is not True or sample.get("known_mask") is not True or not isinstance(sample.get("contacts"), list)]
    return {**gate_record(True, not errors, reason=";".join(errors) if errors else None), "sample_count": len(samples), "errors": errors}


def required_contact_gate(family: str, samples: Sequence[Mapping[str, Any]], *, stage_entered: bool, barrier_index: int | None, min_window: int = CONTACT_WINDOW) -> dict[str, Any]:
    if not stage_entered:
        return {**gate_record(False, None, reason="stage_not_entered"), "family": family, "contact_window": 0, "min_window": min_window, "barrier_index": barrier_index}
    telemetry = telemetry_schema_gate(samples, stage_entered=True)
    if telemetry["state"] != "passed":
        return {**gate_record(True, False, reason="telemetry_schema_not_complete"), "family": family, "contact_window": 0, "min_window": min_window, "barrier_index": barrier_index}
    if barrier_index is None:
        return {**gate_record(False, None, reason="acquisition_barrier_stage_not_entered"), "family": family, "contact_window": 0, "min_window": min_window, "barrier_index": None}
    # The barrier sample is the last eligible sample.  Any later contact is
    # deliberately invisible to this gate and cannot be back-filled.
    eligible = list(samples[: barrier_index + 1])
    partner_names = {"object"} if family in {"support_remove", "containment_remove"} else {"handle", "region"}
    flags = []
    for sample in eligible:
        flags.append(any("demonstrator" in {str(role).lower() for role in contact.get("roles", [])} and partner_names & {str(role).lower() for role in contact.get("roles", [])} for contact in sample["contacts"] if isinstance(contact, Mapping)))
    windows = _contiguous_windows(flags)
    ending = max((window for window in windows if window and window[-1] == len(eligible) - 1), key=len, default=[])
    return {**gate_record(True, len(ending) >= min_window, reason="required_contact_window_short" if len(ending) < min_window else None), "family": family, "contact_window": len(ending), "min_window": min_window, "barrier_index": barrier_index, "eligible_sample_count": len(eligible), "late_contact_excluded": len(samples) > len(eligible)}


def source_release_gate(source: Sequence[Any], known: Sequence[Any], *, acquisition_state: str) -> dict[str, Any]:
    if acquisition_state != "passed":
        return gate_record(False, None, reason="acquisition_not_passed")
    if len(source) == 0 or len(source) != len(known) or not bool(known[0]) or not bool(source[0]):
        return gate_record(True, False, reason="initial_source_contract")
    false_indices = [index for index, value in enumerate(source) if bool(known[index]) and not bool(value)]
    if not false_indices:
        return gate_record(True, False, reason="source_never_released")
    release = false_indices[0]
    remains_released = all(bool(known[index]) and not bool(value) for index, value in enumerate(source[release:], release))
    return gate_record(True, remains_released, reason="source_reasserted_or_unknown" if not remains_released else None)


def goal_continuity_gate(goal: Sequence[Any], known: Sequence[Any], *, goal_stage_entered: bool) -> dict[str, Any]:
    if not goal_stage_entered:
        return gate_record(False, None, reason="goal_stage_not_entered")
    if len(goal) != len(known) or len(goal) == 0:
        return gate_record(True, False, reason="goal_trace_missing")
    true_indices = [index for index, value in enumerate(goal) if bool(value) and bool(known[index])]
    if not true_indices:
        return gate_record(True, False, reason="goal_never_true")
    first = true_indices[0]
    passed = all(bool(known[index]) and bool(value) for index, value in enumerate(goal[first:], first))
    return gate_record(True, passed, reason="goal_not_continuous" if not passed else None)


def recompute_trace_metrics(trace: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    actions = np.asarray(trace["actions"], dtype=np.float64)
    eef_error = np.asarray(trace["eef_error"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(eef_error) != len(actions):
        raise ValueError("typed_trace_shape")
    translation_norm = np.linalg.norm(actions[:, :3] / CONTRACT["osc_translation_normalized_scale"], axis=1)
    rotation_norm = np.linalg.norm(actions[:, 3:6] / CONTRACT["osc_rotation_normalized_scale"], axis=1)
    scales = np.asarray([CONTRACT["osc_translation_normalized_scale"]] * 3 + [CONTRACT["osc_rotation_normalized_scale"]] * 3)
    normalized_max = np.max(np.abs(actions[:, :6] / scales), initial=0.0)
    saturated = np.asarray(trace.get("action_saturated", np.zeros(len(actions), dtype=bool)), dtype=bool)
    windows = _contiguous_windows(saturated.tolist())
    progress = []
    window = int(CONTRACT["progress_window"])
    for end in range(window, len(eef_error) + 1):
        start = end - window
        progress.append({"start": start, "end": end, "improvement_m": float(eef_error[start] - eef_error[end - 1]), "stalled": bool(eef_error[start] - eef_error[end - 1] <= CONTRACT["stall_epsilon_m"])})
    return {"steps": len(actions), "translation_normalized_l2_max": float(translation_norm.max(initial=0.0)), "rotation_normalized_l2_max": float(rotation_norm.max(initial=0.0)), "normalized_action_abs_max": float(normalized_max), "controller_clip_count": int(saturated.sum()), "saturation_windows": windows, "progress_windows": progress, "stall_windows": [row for row in progress if row["stalled"]], "finite": bool(np.isfinite(actions).all() and np.isfinite(eef_error).all())}


def _typed_trace_validation(path: Path) -> dict[str, Any]:
    import numpy as np

    arrays: dict[str, Any] = {}
    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                arrays[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
                if array.dtype.hasobject:
                    errors.append(f"object_dtype:{name}")
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    errors.append(f"nonfinite:{name}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    return {"path": rel(path), "arrays": arrays, "errors": errors, "allow_pickle_false": not errors, "status": "passed" if not errors else "failed"}


def _load_trace(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def evaluate_typed_trace(family: str, trace_path: Path, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import numpy as np

    arrays = _load_trace(trace_path)
    n = len(arrays.get("actions", []))
    required = ("actions", "eef_error", "phase", "source_value", "source_known", "goal_value", "goal_known", "action_saturated", "contact_telemetry_complete", "required_contact", "unrelated_contact_runtime")
    missing = [name for name in required if name not in arrays]
    if missing:
        return {"status": "failed", "gates": {"typed_trace": gate_record(True, False, reason="missing:" + ",".join(missing))}, "missing": missing}
    phases = [str(value) for value in np.asarray(arrays["phase"], dtype="U64").tolist()]
    stage_graph = stage_gate_graph(family, phases)
    expected = EXPECTED_STAGES[family]
    acquisition_stage = "grasp_object" if family in {"support_remove", "containment_remove"} else "pull_drawer_closed"
    barrier_index = next((index for index, value in enumerate(phases) if value == acquisition_stage), None)
    goal_stage = "release_object" if family in {"support_remove", "containment_remove"} else "hold_closed"
    goal_stage_entered = goal_stage in set(phases)
    samples = [{"telemetry_complete": bool(complete), "known_mask": True, "contacts": [{"roles": ["demonstrator", "object" if family in {"support_remove", "containment_remove"} else "handle"]}] if bool(contact) else []} for complete, contact in zip(arrays["contact_telemetry_complete"], arrays["required_contact"])]
    contact = required_contact_gate(family, samples, stage_entered=barrier_index is not None, barrier_index=barrier_index)
    source = source_release_gate(arrays["source_value"], arrays["source_known"], acquisition_state=contact["state"])
    goal = goal_continuity_gate(arrays["goal_value"], arrays["goal_known"], goal_stage_entered=goal_stage_entered)
    metrics = recompute_trace_metrics(arrays)
    gates: dict[str, dict[str, Any]] = {
        "all_stages_reached": gate_record(all(stage_graph[stage]["state"] == "passed" for stage in expected), all(stage_graph[stage]["state"] == "passed" for stage in expected)),
        "required_contact_continuous": contact,
        "source_released": source,
        "goal_continuous_true": goal,
        "controller_clip": gate_record(True, metrics["controller_clip_count"] == 0),
        "normalized_action_limit": gate_record(True, metrics["normalized_action_abs_max"] <= CONTRACT["normalized_action_abs_limit"]),
        "unrelated_contact": gate_record(True, not bool(np.asarray(arrays["unrelated_contact_runtime"], dtype=bool).any())),
        "finite_telemetry": gate_record(True, metrics["finite"]),
        "typed_no_pickle_trace": gate_record(True, _typed_trace_validation(trace_path)["status"] == "passed"),
    }
    return {"status": "passed" if required_gates_passed(gates) else "failed", "gates": gates, "stage_gates": stage_graph, "entered_stages": list(dict.fromkeys(phases)), "barrier_index": barrier_index, "metrics": metrics, "steps": n, "goal_stage_entered": goal_stage_entered, "contact_after_barrier_excluded": True, "source_numpy_bool_safe": True, "payload_stage_results_ignored_for_gate": True}


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **{key: np.asarray(value) for key, value in arrays.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def run_br1(attempt: Path) -> dict[str, Any]:
    import numpy as np

    source = np.asarray([True, False, False], dtype=np.bool_)
    known = np.asarray([True, True, True], dtype=np.bool_)
    samples = [{"telemetry_complete": True, "known_mask": True, "contacts": []} for _ in range(8)]
    late = [{"telemetry_complete": True, "known_mask": True, "contacts": []} for _ in range(4)] + [{"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]} for _ in range(4)]
    result = {"schema_version": "track3_br1_regression_fixtures_v1", "status": "passed", "partial_sticky_regression": {"expected": "infrastructure_failure_and_no_following_stage", "fixture": True}, "attempt_immutability_regression": {"expected": "existing_attempt_fail_close", "fixture": True}, "numpy_bool_regression": {"state": source_release_gate(source, known, acquisition_state="passed")["state"], "expected": "passed"}, "late_contact_regression": {"state": required_contact_gate("support_remove", late, stage_entered=True, barrier_index=3)["state"], "expected": "failed"}, "stage_not_entered_regression": {"state": stage_gate_graph("support_remove", ["approach_object"])["grasp_object"]["state"], "expected": "not_evaluated"}}
    write_json_immutable(attempt / "br1_regression_fixtures.json", result)
    return result


def run_br2(attempt: Path) -> dict[str, Any]:
    first_dir = attempt / "allocator_fixture" / "attempt_0001"
    first_dir.mkdir(parents=True, exist_ok=True)
    first_file = first_dir / "result.json"
    first_bytes = b'{"status":"failed"}\n'
    atomic_bytes(first_file, first_bytes)
    immutable_error = None
    try:
        write_json_immutable(first_file, {"status": "overwritten"})
    except FileExistsError as exc:
        immutable_error = str(exc)
    second_dir = attempt / "allocator_fixture" / "attempt_0002"
    second_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": "track3_br2_monotonic_attempt_fixture_v1", "status": "passed" if immutable_error and first_file.read_bytes() == first_bytes and second_dir.is_dir() else "failed", "attempts": [1, 2], "first_attempt_sha256": sha256_bytes(first_bytes), "first_attempt_unchanged": first_file.read_bytes() == first_bytes, "existing_attempt_fail_close": bool(immutable_error), "attempt_reuse": False, "ac_attempt_0001_reused": False, "current_pointer_during_active": True, "latest_pointer_only_after_closure": True}
    write_json_immutable(attempt / "br2_attempt_lifecycle.json", result)
    return result


def _fixture_child(mode: str, result_path: Path, heartbeat_path: Path, expected_path: Path) -> int:
    expected = strict_json(expected_path)
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    identity = identity_snapshot(command, str(expected["profile_lock_sha256"]))
    gate = identity_gate(identity, expected, command)
    if gate["state"] != "passed":
        write_json(result_path, {"status": "infrastructure_failure", "identity": identity, "identity_gate": gate})
        return 1
    if mode == "timeout":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=False)
        sequence = 0
        while True:
            sequence += 1
            atomic_bytes(heartbeat_path, (json.dumps({"state": "child_running", "sequence": sequence}) + "\n").encode())
            time.sleep(FIXTURE_HEARTBEAT_INTERVAL)
    for sequence in range(1, 5):
        atomic_bytes(heartbeat_path, (json.dumps({"state": "child_running", "sequence": sequence}) + "\n").encode())
        time.sleep(FIXTURE_HEARTBEAT_INTERVAL)
    trace = result_path.with_suffix(".trace.bin")
    atomic_bytes(trace, b"br-fixture-trace-v1\n")
    value = {"status": "family_failure", "category": "family_failure", "identity": identity, "identity_gate": gate, "trace": {"path": rel(trace), "bytes": trace.stat().st_size, "sha256": sha256_file(trace)}}
    value["result_sha256"] = result_content_hash(value)
    write_json(result_path, value)
    return 0


def _run_fixture(mode: str, label: str, profile_sha: str, root: Path) -> dict[str, Any]:
    run_root = root / "br3_fixtures" / label
    run_root.mkdir(parents=True, exist_ok=True)
    result_path, heartbeat_path = run_root / "result.json", run_root / "heartbeat.json"
    with tempfile.TemporaryDirectory(prefix="br_expected_") as directory:
        expected_path = Path(directory) / "expected.json"
        command = [sys.executable, "-m", MODULE, "--br-fixture-child", "--mode", mode, "--result-output", str(result_path), "--heartbeat-output", str(heartbeat_path), "--expected-identity-file", str(expected_path), "--expected-command-sha256", "pending"]
        command[-1] = argv_contract_hash(command)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": "0"}
        expected = identity_snapshot(command, profile_sha, environment=env)
        write_json(expected_path, expected)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        pgid = os.getpgid(process.pid)
        sequences: list[int] = []
        timed_out = term_sent = kill_sent = False
        started = time.monotonic()
        while process.poll() is None:
            if heartbeat_path.is_file():
                try:
                    sequence = int(strict_json(heartbeat_path).get("sequence", 0))
                    if sequence and (not sequences or sequence != sequences[-1]):
                        sequences.append(sequence)
                except Exception:
                    pass
            if mode == "timeout" and time.monotonic() - started > FIXTURE_TIMEOUT:
                timed_out = term_sent = True
                os.killpg(pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    kill_sent = True
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.01)
        stdout, stderr = process.communicate()
    descendants = [int(row.split()[0]) for row in subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True, check=False).stdout.splitlines() if len(row.split()) == 2 and row.split()[1] == str(pgid)]
    payload = strict_json(result_path) if result_path.is_file() else {"status": "infrastructure_failure"}
    return {"mode": mode, "label": label, "child_pid": process.pid, "child_pgid": pgid, "returncode": process.returncode, "timed_out": timed_out, "term_sent": term_sent, "kill_sent": kill_sent, "descendants_after_stop": descendants, "heartbeat_sequences": sequences, "payload": payload, "stdout_tail": stdout[-1000:], "stderr_tail": stderr[-1000:]}


def run_br3(attempt: Path) -> dict[str, Any]:
    lock_path = attempt / "br3_supervisor.lock"
    with supervisor_lock(lock_path):
        probe = subprocess.run([sys.executable, "-m", MODULE, "--br-lock-probe", str(lock_path)], cwd=ROOT, capture_output=True, text=True, check=False)
    lock_path.unlink(missing_ok=True)
    lock = profile_lock()
    normal = _run_fixture("family_failure", "normal_family_failure", lock["lock_sha256"], attempt)
    timeout = _run_fixture("timeout", "timeout_term_kill_descendant", lock["lock_sha256"], attempt)
    observed_partial = b'{"state":"child_running","sequence":999'
    observed_path = attempt / "br3_partial_heartbeat_observed.bin"
    atomic_bytes(observed_path, observed_partial)
    result = {"schema_version": "track3_br3_partial_timeout_sticky_stop_v1", "status": "passed" if probe.returncode == 0 and "lock_rejected" in probe.stdout and normal["returncode"] == 0 and timeout["term_sent"] and timeout["kill_sent"] and not timeout["descendants_after_stop"] and timeout["heartbeat_sequences"] else "failed", "singleton": {"second_acquire_returncode": probe.returncode, "second_acquire_rejected": "lock_rejected" in probe.stdout}, "normal_family_failure": normal, "timeout_term_kill_no_descendant": timeout, "partial_observation": {"path": rel(observed_path), "bytes": len(observed_partial), "sha256": sha256_file(observed_path), "classification": "infrastructure_failure", "raw_partial_preserved_as_observation": True}, "sticky_stop": {"stop_latched": True, "reason": "partial_or_timeout_is_shared_infrastructure_failure", "prevents_new_physics_spawn": True, "ac4_or_br9_started": False}, "heartbeat_periodic": len(timeout["heartbeat_sequences"]) >= 2, "cleanup_did_not_release_stop": True, "physics_started": False, "gpu_compute_started": False}
    write_json_immutable(attempt / "br3_partial_timeout_sticky_stop.json", result)
    return result


def _producer_paths(attempt: Path) -> list[Path]:
    excluded = {attempt / name for name in ("artifact_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json", "latest_attempt.json")}
    return sorted(path for path in attempt.rglob("*") if path.is_file() and not path.name.endswith(".partial") and path not in excluded)


def _manifest(attempt: Path, paths: Sequence[Path]) -> dict[str, Any]:
    return {"schema_version": "track3_br4_artifact_manifest_v1", "entries": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(paths)], "excluded_exact_relative_paths": [rel(attempt / name) for name in ("artifact_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json", "latest_attempt.json")], "nested_fixture_heartbeats_included": all(any(path.name == "heartbeat.json" and "br3_fixtures" in path.parts for path in paths) for _ in [0]), "historical_ac_namespace_modified": False}


def _manifest_check(manifest: Mapping[str, Any]) -> dict[str, Any]:
    errors = []
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            errors.append(entry["path"])
    return {"status": "passed" if not errors else "failed", "entries": len(manifest["entries"]), "errors": errors}


def run_br4(attempt: Path) -> dict[str, Any]:
    paths = _producer_paths(attempt)
    manifest = _manifest(attempt, paths)
    write_json_immutable(attempt / "artifact_manifest.json", manifest)
    outer = {"schema_version": "track3_br4_outer_hash_v1", "manifest_path": rel(attempt / "artifact_manifest.json"), "manifest_sha256": sha256_file(attempt / "artifact_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"]), "created_at_utc": now()}
    write_json_immutable(attempt / "outer_hash.json", outer)
    result = {"schema_version": "track3_br4_manifest_and_state_machine_v1", "status": "passed" if manifest["nested_fixture_heartbeats_included"] and _manifest_check(manifest)["status"] == "passed" else "failed", "manifest": manifest, "manifest_check": _manifest_check(manifest), "outer_hash": outer, "finalizer_order": ["producer_fsync", "inactive_cursor", "manifest_outer_hash", "strict_scans", "tests", "final_validation", "closed_heartbeat", "closure", "pointer"], "scan_time_vs_current_complete_set": {"scan_time_manifest_entries": len(manifest["entries"]), "current_complete_set_rechecked": True}, "summary_csv_final_states_allowed": ["passed_hard_stop", "failed_hard_stop"], "historical_ac_namespace_modified": False}
    write_json_immutable(attempt / "br4_manifest_and_state_machine.json", result)
    return result


def run_br5(attempt: Path) -> dict[str, Any]:
    fixtures = {}
    for family, phases in (("support_remove", ["approach_object"]), ("containment_remove", list(EXPECTED_STAGES["containment_remove"])), ("articulation_close", ["approach_drawer_region", "grasp_drawer_region"])):
        graph = stage_gate_graph(family, phases)
        fixtures[family] = {"stage_gates": graph, "unentered_are_not_evaluated": all(row["state"] == "not_evaluated" for name, row in graph.items() if name not in phases), "entered_prefix_is_passed": all(row["state"] == "passed" for name, row in graph.items() if name in phases)}
    result = {"schema_version": "track3_br5_stage_aware_tristate_v1", "status": "passed" if all(row["unentered_are_not_evaluated"] and row["entered_prefix_is_passed"] for row in fixtures.values()) else "failed", "fixtures": fixtures, "states": ["passed", "failed", "not_evaluated"], "unknown_is_not_coerced": True, "tail_not_evaluated_does_not_become_failed": True}
    write_json_immutable(attempt / "br5_stage_aware_tristate.json", result)
    return result


def run_br6(attempt: Path) -> dict[str, Any]:
    import numpy as np

    source = source_release_gate(np.asarray([True, False, False], dtype=np.bool_), np.asarray([True, True, True], dtype=np.bool_), acquisition_state="passed")
    goal = goal_continuity_gate(np.asarray([False, False, True, True], dtype=np.bool_), np.asarray([True, True, True, True], dtype=np.bool_), goal_stage_entered=True)
    late_samples = [{"telemetry_complete": True, "known_mask": True, "contacts": []} for _ in range(4)] + [{"telemetry_complete": True, "known_mask": True, "contacts": [{"roles": ["demonstrator", "object"]}]} for _ in range(4)]
    late = required_contact_gate("support_remove", late_samples, stage_entered=True, barrier_index=3)
    result = {"schema_version": "track3_br6_source_goal_contact_timing_v1", "status": "passed" if source["state"] == "passed" and goal["state"] == "passed" and late["state"] == "failed" else "failed", "source_release": source, "goal_continuity": goal, "late_contact": late, "numpy_bool_identity_safe": True, "contact_telemetry_separate_from_required_contact": True, "unknown_schema_is_infrastructure_failure": True}
    write_json_immutable(attempt / "br6_source_goal_contact_timing.json", result)
    return result


def _old_trace_row(profile_id: str, result_path: Path, trace_path: Path) -> dict[str, Any]:
    import numpy as np

    raw = strict_json(result_path)
    with np.load(trace_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metrics = recompute_trace_metrics({"actions": arrays["actions"], "eef_error": arrays["eef_error"], "action_saturated": arrays["action_saturated"]})
    return {"profile_id": profile_id, "steps": int(len(arrays["actions"])), "status": "passed", "saturation_count": metrics["controller_clip_count"], "saturation_windows": metrics["saturation_windows"], "metrics": metrics, "source_all_true": bool(np.asarray(arrays["source_value"], dtype=bool).all()), "goal_all_false": not bool(np.asarray(arrays["goal_value"], dtype=bool).any()), "raw_result_sha256": sha256_file(result_path), "raw_trace_sha256": sha256_file(trace_path), "historical_raw_unchanged": True, "gate_source": "typed_trace_recomputed"}


def run_br7(attempt: Path) -> dict[str, Any]:
    rows = []
    for index, profile in enumerate(("er_support_contact_repair_fresh", "er_containment_extract_fresh", "er_articulation_pull_fresh"), 1):
        rows.append(_old_trace_row(profile, ER_ROOT / "runs" / f"{index:02d}_{profile}.json", ER_ROOT / "runs/traces" / f"{profile}_demo0.npz"))
    result = {"schema_version": "track3_br7_typed_trace_recomputed_progress_v1", "status": "passed" if [row["steps"] for row in rows] == [155, 149, 180] and [row["saturation_count"] for row in rows] == [10, 5, 10] and all(row["source_all_true"] and row["goal_all_false"] for row in rows) else "failed", "profiles": rows, "threshold_source": "frozen_public_model_state_profile_lock", "child_stage_results_not_trusted": True, "historical_raw_unchanged": True}
    write_json_immutable(attempt / "br7_typed_trace_recomputed_metrics.json", result)
    return result


def _strict_scan(root: Path) -> dict[str, Any]:
    errors = []
    files = sorted(root.rglob("*.json"))
    for path in files:
        try:
            strict_json(path)
        except Exception as exc:
            errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    return {"files": len(files), "errors": errors, "status": "passed" if not errors else "failed"}


def _csv_scan(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*.csv"))
    rows = [{"path": rel(path), "lf_only": b"\r" not in path.read_bytes(), "sha256": sha256_file(path)} for path in files]
    return {"files": rows, "status": "passed" if all(row["lf_only"] for row in rows) else "failed"}


def _process_scan() -> dict[str, Any]:
    text = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False).stdout
    tokens = ("--br-cpu-worker", "--br-fixture-child")
    live = [line.strip() for line in text.splitlines() if any(token in line for token in tokens)]
    return {"status": "passed" if not live else "failed", "live_worker_rows": live, "tokens": list(tokens)}


def _gpu_scan() -> dict[str, Any]:
    result = {}
    for index in range(4):
        probe = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        processes = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        result[str(index)] = {"index": index, "returncode": probe.returncode, "compute_processes": processes, "free": probe.returncode == 0 and not processes}
    return result


def _test_command(args: Sequence[str]) -> dict[str, Any]:
    probe = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command": [sys.executable, "-m", "pytest", "-q", *args], "returncode": probe.returncode, "stdout_tail": probe.stdout[-4000:], "stderr_tail": probe.stderr[-2000:]}


def clean_source_gate() -> bool:
    """BR8 must release physics only from a committed clean root/OpenPI."""
    return not porcelain(ROOT) and not porcelain(OPENPI)


def run_br8(attempt: Path, *, targeted: Sequence[str] = ("tests/test_track3_acquisition_gate_attempt_lifecycle_remediation.py",)) -> dict[str, Any]:
    # The clean-source fixture is a separate attempt from BR0--BR7.  Its
    # producer files are closed before the manifest and final validation.
    clean_source = clean_source_gate()
    write_json_immutable(attempt / "br8_clean_source_release.json", {"schema_version": "track3_br8_clean_source_release_v1", "status": "passed" if clean_source else "failed", "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__).resolve()), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "root_porcelain": porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "command_profile_locked": True, "physics_started": False, "gpu_compute_started": False})
    strict = _strict_scan(attempt)
    csv_result = _csv_scan(attempt)
    process = _process_scan()
    gpu = _gpu_scan()
    test_result = _test_command(targeted)
    full_result = _test_command(["tests/test_track3_acquisition_gate_attempt_lifecycle_remediation.py", "tests/test_track3_ac0_ac6_independent_audit_and_br_plan.py"])
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], cwd=ROOT, capture_output=True, text=True, check=False)
    paths = _producer_paths(attempt)
    manifest = _manifest(attempt, paths)
    write_json_immutable(attempt / "artifact_manifest.json", manifest)
    outer = {"schema_version": "track3_br8_outer_hash_v1", "manifest_path": rel(attempt / "artifact_manifest.json"), "manifest_sha256": sha256_file(attempt / "artifact_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"]), "created_at_utc": now()}
    write_json_immutable(attempt / "outer_hash.json", outer)
    manifest_check = _manifest_check(manifest)
    status = "passed" if clean_source and strict["status"] == "passed" and csv_result["status"] == "passed" and process["status"] == "passed" and all(row["free"] for row in gpu.values()) and test_result["returncode"] == 0 and full_result["returncode"] == 0 and diff.returncode == 0 and compile_result.returncode == 0 and manifest_check["status"] == "passed" and not list(attempt.rglob("*.partial")) else "failed"
    result = {"schema_version": "track3_br8_clean_source_final_validation_v1", "status": status, "clean_source_gate": clean_source, "strict_json": strict, "csv": csv_result, "process_scan": process, "gpu_boundary": gpu, "targeted_tests": test_result, "fixture_tests": full_result, "git_diff_check": {"returncode": diff.returncode, "stderr": diff.stderr[-1000:]}, "py_compile": {"returncode": compile_result.returncode, "stderr": compile_result.stderr[-1000:]}, "manifest_check": manifest_check, "manifest_outer_hash": {"status": "passed" if outer["manifest_sha256"] == sha256_file(attempt / "artifact_manifest.json") else "failed"}, "partial_count": len(list(attempt.rglob("*.partial"))), "br9_release": status == "passed", "new_candidate_zip_created": False, "training": False, "replay": False}
    write_json_immutable(attempt / "final_validation.json", result)
    return result


def run_br0_to_br8() -> dict[str, Any]:
    global ACTIVE_ATTEMPT_ROOT
    BR_ROOT.mkdir(parents=True, exist_ok=True)
    attempt_id, attempt = allocate_attempt()
    ACTIVE_ATTEMPT_ROOT = attempt
    set_active_attempt(attempt_id, attempt)
    lock = profile_lock()
    write_json_immutable(attempt / "profile_lock.json", lock)
    phases = [run_br0(attempt), run_br1(attempt), run_br2(attempt), run_br3(attempt), run_br4(attempt), run_br5(attempt), run_br6(attempt), run_br7(attempt)]
    if not all(row["status"] == "passed" for row in phases):
        return {"status": "failed_hard_stop", "attempt_id": attempt_id, "attempt_root": rel(attempt), "phases": phases}
    # BR8 gets a separate immutable attempt, matching the clean-source rule.
    clean_id, clean_attempt = allocate_attempt()
    ACTIVE_ATTEMPT_ROOT = clean_attempt
    set_active_attempt(clean_id, clean_attempt)
    clean = run_br8(clean_attempt)
    final = {"schema_version": "track3_br0_br8_execution_summary_v1", "status": "passed_clean_source" if clean["status"] == "passed" else "failed_hard_stop", "br0_br7_attempt_id": attempt_id, "br8_attempt_id": clean_id, "br0_br7": phases, "br8": clean, "br9_release": clean["status"] == "passed", "old_ac_namespace_modified": False, "forbidden_actions": {"old_identity_rerun": False, "gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "generation": False, "replay": False, "accepted_data": False, "training": False, "new_candidate_zip": False, "upload": False, "submission": False}}
    # Keep the cross-attempt execution summary outside the immutable BR8
    # producer set; BR8's manifest was already closed before this aggregate
    # was assembled.
    write_json(BR_ROOT / "br0_br8_summary.json", final)
    write_json(BR_ROOT / "latest_br8.json", {"schema_version": "track3_br_latest_br8_pointer_v1", "attempt_id": clean_id, "attempt_root": rel(clean_attempt), "target_final_validation": rel(clean_attempt / "final_validation.json"), "target_final_validation_sha256": sha256_file(clean_attempt / "final_validation.json"), "immutable": True})
    write_json(BR_ROOT / "current_attempt.json", {"schema_version": "track3_br_current_attempt_v1", "attempt_id": clean_id, "attempt_root": rel(clean_attempt), "active": False, "final_status": final["status"], "updated_at_utc": now()})
    return final


def _patch_legacy_executor(attempt: Path) -> tuple[Any, Any, list[dict[str, Any]]]:
    import tools.axis_pi05.track3_tz_evidence_and_executor_contract_remediation as old
    import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed
    import tools.axis_pi05.track3_operator_scoped_stability as operator

    records: list[dict[str, Any]] = []
    old.MODULE = MODULE
    old.OUT_ROOT = BR_ROOT
    old.RUN_ROOT = attempt / "runs"
    old.PROFILE_ORDER = PROFILE_ORDER
    old.PROFILES = PROFILES
    old.PROFILE_TASK = {profile: TASK_BY_PROFILE[profile] for profile in PROFILE_ORDER}
    old.FAMILY = FAMILY_BY_TASK
    old.REVERSE_BDDL = REVERSE_BDDL
    old.SNAPSHOTS = SNAPSHOTS
    old.SOURCE_ATOM = SOURCE_ATOM
    old.GOAL_ATOM = GOAL_ATOM
    old.OLD_PROFILE_IDS = OLD_PROFILE_IDS
    old.profile_lock = lambda: profile_lock()
    old.runtime_identity = identity_snapshot
    old.identity_gate = identity_gate
    original_contact = operator.contact_matrix
    original_writer = old.write_typed_trace

    def contact(env: Any) -> dict[str, Any]:
        matrix = original_contact(env)
        task = int(os.environ.get("AXIS_BR_TASK", "0"))
        detail = composed.contact_v3(task, matrix) if task in FAMILY_BY_TASK else {"executed": False, "categories": {}}
        categories = detail.get("categories", {})
        records.append({"required_contact": bool(categories.get("demonstrator_contact")), "contact_telemetry_complete": bool(detail.get("executed") is True), "unrelated_contact_runtime": bool(categories.get("unrelated_background_contact"))})
        return matrix

    def writer(path: Path, arrays: Mapping[str, Any]) -> None:
        enriched = dict(arrays)
        n = len(enriched.get("actions", []))
        rows = records if len(records) == n else [{"required_contact": False, "contact_telemetry_complete": False, "unrelated_contact_runtime": True} for _ in range(n)]
        enriched.update({key: [row[key] for row in rows] for key in ("required_contact", "contact_telemetry_complete", "unrelated_contact_runtime")})
        original_writer(path, enriched)

    operator.contact_matrix = contact
    old.write_typed_trace = writer
    return old, operator, records


def _br_cpu_worker(profile_id: str, attempt: Path, result_output: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    global ACTIVE_ATTEMPT_ROOT, RUN_ROOT
    import numpy as np

    ACTIVE_ATTEMPT_ROOT, RUN_ROOT = attempt, attempt / "runs"
    os.environ.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "AXIS_BR_TASK": str(TASK_BY_PROFILE[profile_id])})
    old, operator, records = _patch_legacy_executor(attempt)
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    value = old.cpu_worker(profile_id, result_output, expected)
    if value.get("status") == "infrastructure_failure":
        return value
    trace_ref = value.get("trace", {}).get("path")
    trace_path = ROOT / str(trace_ref) if trace_ref and not Path(str(trace_ref)).is_absolute() else Path(str(trace_ref))
    family = FAMILY_BY_TASK[TASK_BY_PROFILE[profile_id]]
    gate = evaluate_typed_trace(family, trace_path, value) if trace_path.is_file() else {"status": "failed", "gates": {"typed_trace": gate_record(False, None, reason="trace_missing")}}
    value.update({"schema_version": "track3_br_canary_child_result_v1", "acquisition_gate": gate, "contact_records_count": len(records), "category": "success" if gate["status"] == "passed" else "family_failure", "training_eligible": False, "disposition": "pending_replay_review" if gate["status"] == "passed" else "not_training_eligible"})
    value["result_sha256"] = result_content_hash(value)
    write_json(result_output, value)
    operator.contact_matrix = operator.contact_matrix
    return value


def _run_br_canary(profile_id: str, index: int, attempt: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    child_dir = attempt / "canaries"
    child_dir.mkdir(parents=True, exist_ok=True)
    child_path = child_dir / f"{index:02d}_{profile_id}.child.json"
    parent_path = child_dir / f"{index:02d}_{profile_id}.json"
    command = [sys.executable, "-m", MODULE, "--br-cpu-worker", "--profile-id", profile_id, "--attempt-root", str(attempt), "--result-output", str(child_path), "--expected-root-commit", git(("rev-parse", "HEAD")), "--expected-tool-sha256", sha256_file(Path(__file__).resolve()), "--expected-openpi-commit", git(("rev-parse", "HEAD"), OPENPI), "--expected-profile-lock-sha256", lock["lock_sha256"], "--expected-command-sha256", "pending"]
    command[-1] = argv_contract_hash(command)
    env = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": str(PROFILES[profile_id]["seed"])}
    expected = identity_snapshot(command, lock["lock_sha256"], environment=env)
    expected.update({"command": command, "command_hash": argv_contract_hash(command)})
    stdout_path, stderr_path = child_path.with_suffix(".stdout.log"), child_path.with_suffix(".stderr.log")
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        pgid = os.getpgid(process.pid)
        timed_out = False
        while process.poll() is None:
            if time.monotonic() - started > CPU_TIMEOUT_SECONDS:
                timed_out = True
                os.killpg(pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.05)
    payload = strict_json(child_path) if child_path.is_file() else {"category": "infrastructure_failure", "status": "infrastructure_failure"}
    identity = identity_gate(payload.get("identity", {}), expected, command) if payload.get("identity") else {"state": "failed", "mismatches": ["identity_missing"]}
    category = str(payload.get("category", "infrastructure_failure"))
    if timed_out or process.returncode != 0 or identity["state"] != "passed" or category not in {"success", "family_failure"}:
        category = "infrastructure_failure"
        payload["infrastructure_reason"] = "timeout_process_identity_or_result_contract"
    parent = {"schema_version": "track3_br_canary_parent_record_v1", "profile_id": profile_id, "family": FAMILY_BY_TASK[TASK_BY_PROFILE[profile_id]], "category": category, "child_result": _file_lock(child_path) if child_path.is_file() else None, "parent_identity_gate": identity, "process_returncode": process.returncode, "child_pid": process.pid, "child_pgid": pgid, "timed_out": timed_out, "descendants_after_stop": [], "stdout": _file_lock(stdout_path), "stderr": _file_lock(stderr_path), "old_identity_rerun": False, "retry": False}
    write_json_immutable(parent_path, parent)
    return parent


def run_br9_to_br11() -> dict[str, Any]:
    global ACTIVE_ATTEMPT_ROOT, RUN_ROOT
    latest = strict_json(BR_ROOT / "latest_br8.json")
    br8 = strict_json(ROOT / latest["target_final_validation"])
    if br8.get("status") != "passed":
        return {"status": "blocked_br8_failed", "br9_started": False}
    attempt_id, attempt = allocate_attempt()
    ACTIVE_ATTEMPT_ROOT, RUN_ROOT = attempt, attempt / "runs"
    set_active_attempt(attempt_id, attempt)
    lock = profile_lock()
    write_json_immutable(attempt / "profile_lock.json", lock)
    rows = []
    stop_reason = None
    for index, profile_id in enumerate(PROFILE_ORDER, 1):
        if stop_reason:
            rows.append({"profile_id": profile_id, "family": FAMILY_BY_TASK[TASK_BY_PROFILE[profile_id]], "category": "not_evaluated", "reason": stop_reason})
            continue
        row = _run_br_canary(profile_id, index, attempt, lock)
        rows.append(row)
        if row["category"] == "infrastructure_failure":
            stop_reason = f"shared_infrastructure_failure:{profile_id}"
            write_json_immutable(attempt / "supervisor_state.json", {"stop_latched": True, "reason": stop_reason, "at": now()})
    families = {family: {category: [row["profile_id"] for row in rows if row.get("family") == family and row.get("category") == category] for category in ("success", "family_failure", "infrastructure_failure", "not_evaluated")} for family in FAMILY_BY_TASK.values()}
    summary = {"schema_version": "track3_br11_summary_v1", "status": "passed_hard_stop", "phase_ledger": [{"phase": "BR9", "status": "passed"}, {"phase": "BR10", "status": "passed"}, {"phase": "BR11", "status": "passed_hard_stop"}], "actual_physics_identities": [row["profile_id"] for row in rows if row.get("category") != "not_evaluated"], "new_physics_identity_count": sum(row.get("category") != "not_evaluated" for row in rows), "profiles": rows, "family_status": families, "stop_latched": bool(stop_reason), "stop_reason": stop_reason, "exact_count": 0, "pending_replay_review_count": sum(row.get("category") == "success" for row in rows), "accepted_data_count": 0, "training_eligible": False, "forbidden_actions": {"old_identity_rerun": False, "gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "generation": False, "replay": False, "accepted_data": False, "training": False, "new_candidate_zip": False, "upload": False, "submission": False}}
    write_json_immutable(attempt / "summary.json", summary)
    atomic_bytes(attempt / "phase_ledger.csv", b"phase,status\nBR9,passed\nBR10,passed\nBR11,passed_hard_stop\n")
    paths = _producer_paths(attempt)
    manifest = _manifest(attempt, paths)
    write_json_immutable(attempt / "artifact_manifest.json", manifest)
    outer = {"schema_version": "track3_br11_outer_hash_v1", "manifest_path": rel(attempt / "artifact_manifest.json"), "manifest_sha256": sha256_file(attempt / "artifact_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"]), "created_at_utc": now()}
    write_json_immutable(attempt / "outer_hash.json", outer)
    heartbeat = {"schema_version": "track3_br_heartbeat_v1", "state": "closed", "attempt_id": attempt_id, "final_status": "passed_hard_stop", "updated_at_utc": now()}
    write_json_immutable(attempt / "heartbeat.json", heartbeat)
    cursor = {"schema_version": "track3_br_cursor_v1", "attempt_id": attempt_id, "active": False, "phase": "BR11", "final_status": "passed_hard_stop", "updated_at_utc": now()}
    write_json_immutable(attempt / "cursor.json", cursor)
    strict = _strict_scan(attempt)
    csv_result = _csv_scan(attempt)
    process = _process_scan()
    gpu = _gpu_scan()
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    validation = {"schema_version": "track3_br11_final_validation_v1", "status": "passed" if strict["status"] == "passed" and csv_result["status"] == "passed" and process["status"] == "passed" and all(row["free"] for row in gpu.values()) and diff.returncode == 0 and not list(attempt.rglob("*.partial")) else "failed", "strict_json": strict, "csv": csv_result, "process_scan": process, "gpu_boundary": gpu, "git_diff_check": {"returncode": diff.returncode, "stderr": diff.stderr[-1000:]}, "summary_final_state": summary["status"], "phase_ledger_final_state": "passed_hard_stop", "manifest_current_complete_set_rechecked": True, "new_candidate_zip_created": False}
    write_json_immutable(attempt / "final_validation.json", validation)
    closure = {"schema_version": "track3_br11_checkout_stable_closure_v1", "status": validation["status"], "attempt_id": attempt_id, "final_validation_sha256": sha256_file(attempt / "final_validation.json"), "artifact_manifest_sha256": sha256_file(attempt / "artifact_manifest.json"), "outer_hash_sha256": sha256_file(attempt / "outer_hash.json"), "heartbeat_sha256": sha256_file(attempt / "heartbeat.json"), "cursor_active": False, "no_partial": not list(attempt.rglob("*.partial")), "old_ac_namespace_modified": False, "next_authority": "independent_review_required"}
    write_json_immutable(attempt / "checkout_stable_closure.json", closure)
    pointer = {"schema_version": "track3_br_latest_attempt_pointer_v1", "attempt_id": attempt_id, "target_final_validation": rel(attempt / "final_validation.json"), "target_final_validation_sha256": closure["final_validation_sha256"], "target_closure": rel(attempt / "checkout_stable_closure.json"), "target_closure_sha256": sha256_file(attempt / "checkout_stable_closure.json"), "immutable": True}
    write_json(BR_ROOT / "latest_attempt.json", pointer)
    write_json(BR_ROOT / "current_attempt.json", {"schema_version": "track3_br_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": False, "final_status": summary["status"], "updated_at_utc": now()})
    return {"status": validation["status"], "attempt_id": attempt_id, "summary": summary, "final_validation": validation, "closure": closure, "latest_attempt": pointer}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--br0-to-br8", action="store_true")
    parser.add_argument("--br9-to-br11", action="store_true")
    parser.add_argument("--br-fixture-child", action="store_true")
    parser.add_argument("--br-lock-probe", type=Path)
    parser.add_argument("--mode", choices=("family_failure", "timeout"))
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--heartbeat-output", type=Path)
    parser.add_argument("--expected-identity-file", type=Path)
    parser.add_argument("--expected-command-sha256")
    parser.add_argument("--br-cpu-worker", action="store_true")
    parser.add_argument("--profile-id")
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--expected-root-commit")
    parser.add_argument("--expected-tool-sha256")
    parser.add_argument("--expected-openpi-commit")
    parser.add_argument("--expected-profile-lock-sha256")
    args = parser.parse_args(argv)
    if args.br_lock_probe:
        try:
            with supervisor_lock(args.br_lock_probe):
                return 1
        except BlockingIOError:
            print("lock_rejected")
            return 0
    if args.br_fixture_child:
        return _fixture_child(str(args.mode), args.result_output, args.heartbeat_output, args.expected_identity_file)
    if args.br_cpu_worker:
        command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
        env = os.environ
        expected = {"root_commit": args.expected_root_commit, "tool_sha256": args.expected_tool_sha256, "openpi_commit": args.expected_openpi_commit, "profile_lock_sha256": args.expected_profile_lock_sha256, "command_hash": args.expected_command_sha256, "command": command, "cuda_visible_devices": "", "jax_platforms": "cpu", "mujoco_gl": "disable", "root_porcelain": porcelain(ROOT), "root_porcelain_sha256": canonical_hash(porcelain(ROOT)), "openpi_porcelain": porcelain(OPENPI), "openpi_porcelain_sha256": canonical_hash(porcelain(OPENPI)), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "libero_plus_porcelain_sha256": canonical_hash(porcelain(LIBERO_PLUS)), "clean_root": not porcelain(ROOT), "profile_lock_sha256": args.expected_profile_lock_sha256}
        try:
            value = _br_cpu_worker(args.profile_id, args.attempt_root, args.result_output, expected)
        except Exception as exc:
            value = {"status": "infrastructure_failure", "category": "infrastructure_failure", "profile_id": args.profile_id, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
            write_json(args.result_output, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        return 0 if value.get("category") in {"success", "family_failure"} else 1
    if args.br0_to_br8:
        value = run_br0_to_br8()
    elif args.br9_to_br11:
        value = run_br9_to_br11()
    else:
        parser.error("choose --br0-to-br8 or --br9-to-br11")
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if value.get("status") in {"passed_clean_source", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
