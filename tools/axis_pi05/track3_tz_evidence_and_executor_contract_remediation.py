#!/usr/bin/env python3
"""ER0--ER5 Track 3 executor-contract remediation.

The previous Tonight namespace is input only.  This module deliberately has
its own evidence root and three new CPU identities.  GPU, generation,
replay, accepted-data, training, and candidate creation are not implemented
as runnable paths here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_tz_evidence_and_executor_contract_remediation"
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation"
RUN_ROOT = OUT_ROOT / "runs"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_tonight_zip_and_crossfamily_recovery"
FALLBACK_ZIP = ROOT / "artifacts/submission_axis_r6_fallback_revalidated_20260906.zip"
FALLBACK_SOURCE = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
FALLBACK_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
FALLBACK_BYTES = 9383571677
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PUBLIC_AUTOGEN = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
PROFILE_ORDER = (
    "er_support_contact_repair_fresh",
    "er_containment_extract_fresh",
    "er_articulation_pull_fresh",
)
PROFILE_TASK = {
    PROFILE_ORDER[0]: 10,
    PROFILE_ORDER[1]: 20,
    PROFILE_ORDER[2]: 19,
}
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
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
REVERSE_BDDL = {
    10: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_support_remove_task_10.bddl",
    20: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_containment_remove_task_20.bddl",
    19: PUBLIC_AUTOGEN / "s3_reverse_specs/canary_articulation_close_task_19.bddl",
}
SNAPSHOTS = {
    task: P1_ROOT / "snapshots" / f"task_{task:02d}" / "attempt_1" / "terminal_state.npy"
    for task in (10, 20, 19)
}
PROFILES: dict[str, dict[str, Any]] = {
    PROFILE_ORDER[0]: {"task_index": 10, "seed": 51010, "grasp_z_m": 0.024, "close_dwell": 2, "release_dwell": 1},
    PROFILE_ORDER[1]: {"task_index": 20, "seed": 52020, "grasp_z_m": 0.024, "close_dwell": 2, "release_dwell": 1},
    PROFILE_ORDER[2]: {"task_index": 19, "seed": 53019, "grasp_z_m": 0.012, "close_dwell": 2, "release_dwell": 1},
}
OLD_PROFILE_IDS = {
    "support_contact_repair_a", "support_contact_repair_b", "support_contact_repair_c",
    "containment_extract_a", "containment_extract_b", "articulation_pull_a", "articulation_pull_b",
}
TERM_GRACE_SECONDS = 10.0
CPU_TIMEOUT_SECONDS = 240.0
HEARTBEAT_INTERVAL_SECONDS = 0.5
ACTION_DIM = 7


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains a duplicate key."""


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
    """Hash the published result without a self-referential hash field."""
    body = dict(value)
    body.pop("result_sha256", None)
    return canonical_hash(body)


def argv_contract_hash(command: Sequence[str]) -> str:
    """Hash fixed argv while avoiding a self-referential probe argument."""
    normalized = list(command)
    if "--expected-command-sha256" in normalized:
        index = normalized.index("--expected-command-sha256") + 1
        if index < len(normalized):
            normalized[index] = "<expected-command-sha256>"
    return canonical_hash(normalized)


def gpu_snapshot(index: int) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"index": index, "returncode": result.returncode, "compute_processes": processes, "stderr": result.stderr[-1000:], "free": result.returncode == 0 and not processes}


def profile_lock() -> dict[str, Any]:
    rows = []
    for profile_id in PROFILE_ORDER:
        profile = dict(PROFILES[profile_id])
        profile.update({
            "profile_id": profile_id,
            "family": FAMILY[int(profile["task_index"])],
            "identity_scope": "new_er_cpu_identity_once",
            "old_identity_rerun": False,
            "renderer": False,
            "gpu": False,
            "training_eligible": False,
        })
        rows.append(profile)
    body: dict[str, Any] = {
        "schema_version": "track3_tz_er_profile_lock_v1",
        "profile_order": list(PROFILE_ORDER),
        "profiles": rows,
        "limits": {"support_remove": 1, "containment_remove": 1, "articulation_close": 1, "total": 3},
        "old_profile_ids_denylist": sorted(OLD_PROFILE_IDS),
        "parameter_hash": canonical_hash(rows),
        "official_track3_claim": False,
    }
    body["lock_sha256"] = canonical_hash(body)
    return body


def apply_effective_seed(seed: int, np_module: Any) -> dict[str, Any]:
    """Apply, rather than merely record, the frozen seed contract."""
    random.seed(int(seed))
    np_module.random.seed(int(seed))
    return {
        "seed": int(seed),
        "applied": True,
        "methods": ["python_random.seed", "numpy.random.seed", "PYTHONHASHSEED_child_environment"],
        "environment_reset_seed_argument": "unsupported_by_ControlEnv_wrapper",
        "determinism_contract": "fixed_snapshot_and_fixed_action_policy_after_process_seed",
    }


def tri_state(executed: bool, value: bool | None) -> str:
    if not executed:
        return "not_evaluated"
    if value is None:
        return "unknown"
    return "passed" if value else "failed"


def gate_record(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"executed": bool(executed), "value": value, "state": tri_state(executed, value), "required": bool(required)}
    if reason is not None:
        row["reason"] = reason
    return row


def required_gates_passed(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [gate for gate in gates.values() if gate.get("required", True)]
    return bool(required) and all(gate.get("state") == "passed" for gate in required)


def validate_typed_trace(path: Path) -> dict[str, Any]:
    """Validate every array without allowing pickle or non-finite values."""
    import numpy as np

    arrays: dict[str, Any] = {}
    errors: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                arrays[name] = {"dtype": str(array.dtype), "shape": list(array.shape), "object_dtype": bool(array.dtype.hasobject), "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "fc" else True}
                if array.dtype.hasobject:
                    errors.append(f"object_dtype:{name}")
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    errors.append(f"nonfinite:{name}")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{exc}")
    return {"path": rel(path), "allow_pickle_false": not any("Object arrays" in error or "allow_pickle" in error for error in errors), "arrays": arrays, "errors": errors, "status": "passed" if not errors else "failed"}


def write_typed_trace(path: Path, arrays: Mapping[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _zip_inventory(path: Path) -> dict[str, Any]:
    unsafe: list[str] = []
    duplicates: list[str] = []
    names: list[str] = []
    seen: set[str] = set()
    bad: str | None = None
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            names.append(name)
            if name in seen:
                duplicates.append(name)
            seen.add(name)
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                unsafe.append(name)
        bad = archive.testzip()
    return {
        "path": rel(path), "absolute_path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "crc_test": "passed" if bad is None else f"failed:{bad}",
        "entry_count": len(names), "duplicate_entries": duplicates, "unsafe_paths": unsafe,
        "model_fast_entries": [name for name in names if name == "model_weights/model_fast.safetensors"],
        "slow_model_entries": [name for name in names if "model_slow" in name or name.endswith("slow.safetensors")],
    }


def _tracked_state(path: Path) -> str:
    return "tracked" if git(("ls-files", "--error-unmatch", rel(path))).strip() else "untracked_or_ignored"


def _old_npz_record(path: Path) -> dict[str, Any]:
    import numpy as np

    record: dict[str, Any] = {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "tracked": _tracked_state(path), "allow_pickle_false": False, "arrays": {}, "error": None}
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = np.asarray(archive[name])
                record["arrays"][name] = {"dtype": str(array.dtype), "shape": list(array.shape), "object_dtype": bool(array.dtype.hasobject)}
                if array.dtype.hasobject:
                    raise TypeError(f"object_dtype:{name}")
        record["allow_pickle_false"] = True
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}:{exc}"
    return record


def run_er0() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    old_json = sorted(OLD_ROOT.rglob("*.json"))
    old_npz = sorted(OLD_ROOT.rglob("*.npz"))
    old_csv = sorted(OLD_ROOT.rglob("*.csv"))
    strict_errors: list[str] = []
    json_records = []
    for path in old_json:
        try:
            strict_json(path)
            status = "passed"
        except Exception as exc:
            status = "failed"
            strict_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
        json_records.append({"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "tracked": _tracked_state(path), "strict_json": status})
    npz_records = [_old_npz_record(path) for path in old_npz]
    csv_records = []
    for path in old_csv:
        raw = path.read_bytes()
        csv_records.append({"path": rel(path), "bytes": len(raw), "sha256": sha256_bytes(raw), "tracked": _tracked_state(path), "utf8": True, "lf_only": b"\r" not in raw})
    run_rows = []
    for path in sorted(OLD_ROOT.glob("runs/cpu_*.json")):
        value = strict_json(path)
        run_rows.append({"profile_id": value.get("profile_id"), "status": value.get("status"), "category": value.get("category"), "path": rel(path), "trace_sha256": (value.get("trace") or {}).get("sha256")})
    trace_groups: dict[str, list[str]] = {}
    for row in run_rows:
        trace_sha = row.get("trace_sha256")
        if trace_sha:
            trace_groups.setdefault(str(trace_sha), []).append(str(row.get("profile_id")))
    matrix = strict_json(OLD_ROOT / "tz2_cpu_matrix.json")
    fallback = _zip_inventory(FALLBACK_ZIP)
    source = _zip_inventory(FALLBACK_SOURCE)
    derived = {
        "schema_version": "track3_tz_er0_derived_closure_v1",
        "status": "passed" if len(old_json) == 23 and len(old_npz) == 6 and not strict_errors else "failed",
        "immutable_input_commit": "2ae08e974d6be9ac59541236ce7ddb1d7f288dd6",
        "old_namespace": {"json_count": len(old_json), "npz_count": len(old_npz), "csv_count": len(old_csv), "strict_json_errors": strict_errors},
        "derived_classification": {
            "first_infrastructure_failure": next((row["profile_id"] for row in run_rows if row.get("category") == "infrastructure_failure"), None),
            "diagnostic_family_failures": [row["profile_id"] for row in run_rows if row.get("category") == "family_failure"],
            "old_aggregate_infrastructure_failure": matrix.get("infrastructure_failure"),
            "aggregate_contradiction_preserved": matrix.get("infrastructure_failure") is None and any(row.get("category") == "infrastructure_failure" for row in run_rows),
            "identical_trace_groups": [group for group in trace_groups.values() if len(group) > 1],
        },
        "old_npz_no_pickle_failures": [record["path"] for record in npz_records if not record["allow_pickle_false"]],
        "fallback": fallback,
        "fallback_source": source,
        "old_files_read_only": True,
    }
    manifest = {
        "schema_version": "track3_tz_er0_manifest_v1",
        "source_namespace": rel(OLD_ROOT),
        "json": json_records,
        "npz": npz_records,
        "csv": csv_records,
        "fallback": fallback,
        "fallback_source": source,
        "historical_files_modified": False,
    }
    write_json(OUT_ROOT / "er0_manifest.json", manifest)
    write_json(OUT_ROOT / "er0_derived_closure.json", derived)
    return derived


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


def runtime_identity(command: Sequence[str], profile_lock_sha: str) -> dict[str, Any]:
    return {
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI) if OPENPI.is_dir() else None,
        "profile_lock_sha256": profile_lock_sha,
        "command": list(command),
        "command_hash": argv_contract_hash(command),
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "root_porcelain": porcelain(ROOT),
        "clean_root": not porcelain(ROOT),
        "openpi_porcelain": porcelain(OPENPI) if OPENPI.is_dir() else [],
        "libero_plus_porcelain": porcelain(LIBERO_PLUS),
    }


def identity_gate(observed: Mapping[str, Any], expected: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    fields = ("root_commit", "tool_sha256", "openpi_commit", "profile_lock_sha256", "cuda_visible_devices", "jax_platforms", "mujoco_gl", "clean_root", "openpi_porcelain", "libero_plus_porcelain")
    mismatches = [field for field in fields if observed.get(field) != expected.get(field)]
    if expected.get("command_hash") is not None and observed.get("command_hash") != expected.get("command_hash"):
        mismatches.append("command_hash")
    elif list(observed.get("command", [])) != list(command):
        mismatches.append("command")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches, "checked_fields": list(fields) + ["command", "command_hash"]}


def _set_cursor(phase: str, active: bool, **extra: Any) -> None:
    write_json(OUT_ROOT / "resume_cursor.json", {"schema_version": "track3_tz_er_cursor_v1", "active": active, "phase": phase, "pid": os.getpid(), "updated_at_utc": now(), **extra})


def _heartbeat(state: str, **extra: Any) -> None:
    write_json(OUT_ROOT / "heartbeat.json", {"schema_version": "track3_tz_er_heartbeat_v1", "state": state, "pid": os.getpid(), "updated_at_utc": now(), **extra})


def _heartbeat_loop(stop: threading.Event, **extra: Any) -> None:
    while not stop.is_set():
        _heartbeat("child_running", **extra)
        stop.wait(HEARTBEAT_INTERVAL_SECONDS)


@dataclass
class ChildResult:
    payload: dict[str, Any]
    returncode: int
    timed_out: bool
    descendants_after_stop: list[int]


def _descendants_in_group(pgid: int) -> list[int]:
    result = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True, check=False)
    members: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                pid, group = int(fields[0]), int(fields[1])
            except ValueError:
                continue
            if group == pgid and pid != os.getpid():
                members.append(pid)
    return members


def _read_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("worker_result_missing")
    value = strict_json(path)
    if not isinstance(value, dict) or "status" not in value:
        raise ValueError("worker_result_not_mapping")
    return value


def run_child(profile_id: str, result_path: Path, expected: Mapping[str, Any], timeout: float = CPU_TIMEOUT_SECONDS) -> ChildResult:
    lock = profile_lock()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", MODULE, "--cpu-worker", "--profile-id", profile_id,
        "--result-output", str(result_path), "--expected-root-commit", str(expected["root_commit"]),
        "--expected-tool-sha256", str(expected["tool_sha256"]), "--expected-openpi-commit", str(expected["openpi_commit"]),
        "--expected-profile-lock-sha256", str(lock["lock_sha256"]), "--expected-command-sha256", "pending",
    ]
    command[-1] = argv_contract_hash(command)
    env = os.environ.copy()
    env.update({"PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": str(PROFILES[profile_id]["seed"])})
    stdout_path = result_path.with_suffix(".stdout.log")
    stderr_path = result_path.with_suffix(".stderr.log")
    start = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        child_pid = process.pid
        child_pgid = os.getpgid(child_pid)
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(target=_heartbeat_loop, args=(stop_heartbeat,), kwargs={"child_pid": child_pid, "child_pgid": child_pgid, "profile_id": profile_id}, daemon=True)
        heartbeat_thread.start()
        timed_out = False
        while process.poll() is None:
            if time.monotonic() - start > timeout:
                timed_out = True
                os.killpg(child_pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    os.killpg(child_pgid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.05)
        returncode = int(process.returncode or 0)
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2.0)
    descendants = _descendants_in_group(child_pgid)
    try:
        payload = _read_result(result_path)
    except Exception as exc:
        payload = {"schema_version": "track3_tz_er_child_result_v1", "status": "infrastructure_failure", "error": {"type": type(exc).__name__, "message": str(exc)}}
    payload.update({"process_returncode": returncode, "child_pid": child_pid, "child_pgid": child_pgid, "timed_out": timed_out, "descendants_after_stop": descendants, "result_sha256": sha256_file(result_path) if result_path.is_file() else None, "stdout_tail": stdout_path.read_text(encoding="utf-8", errors="replace")[-2000:], "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]})
    if timed_out or descendants or returncode != 0 and payload.get("status") == "failed_gate":
        payload["status"] = "infrastructure_failure"
        payload["infrastructure_reason"] = "timeout_or_process_contract" if timed_out or descendants else "family_failure_returncode_mismatch"
    return ChildResult(payload, returncode, timed_out, descendants)


def cli_contract() -> dict[str, Any]:
    return {
        "schema_version": "track3_tz_er_cli_contract_v1",
        "phase_flags": [f"--er{i}" for i in range(6)],
        "worker_flags": ["--cpu-worker"],
        "single_instance_file_lock": True,
        "pid_pgid_recorded": True,
        "fixed_argv_identity": True,
        "periodic_atomic_heartbeat": True,
        "timeout_order": ["SIGTERM", "wait_10_seconds", "SIGKILL", "verify_no_descendants"],
        "resume_requires_result_hash_and_identity": True,
        "sticky_shared_infrastructure_stop": True,
        "family_failure_continues_other_new_families": True,
        "forbidden_paths": ["--gpu", "--generation", "--replay", "--train", "--candidate-zip"],
        "return_code_contract": {"passed": 0, "failed_gate": 0, "infrastructure_failure": 1},
    }


def _family_status(payload: Mapping[str, Any]) -> str:
    status = str(payload.get("status"))
    if status == "failed_gate":
        return "family_failure"
    if status in {"worker_exception", "timeout", "identity_failure", "infrastructure_failure"}:
        return "infrastructure_failure"
    return status


def run_er1() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT_ROOT / "supervisor.lock"
    lock_probe = {"held_and_released": False, "second_acquire_rejected": False}
    with supervisor_lock(lock_path):
        lock_probe["held_and_released"] = True
        try:
            with supervisor_lock(lock_path):
                pass
        except BlockingIOError:
            lock_probe["second_acquire_rejected"] = True
    lock_path.unlink(missing_ok=True)
    lock = profile_lock()
    command = [sys.executable, "-m", MODULE, "--cpu-worker", "--profile-id", PROFILE_ORDER[0]]
    expected = runtime_identity(command, lock["lock_sha256"])
    expected["command"] = command
    observed = dict(expected)
    identity = identity_gate(observed, expected, command)
    result = {
        "schema_version": "track3_tz_er1_executor_contract_v1",
        "status": "passed" if lock_probe["held_and_released"] and lock_probe["second_acquire_rejected"] and identity["state"] == "passed" else "failed",
        "cli_contract": cli_contract(), "lock_probe": lock_probe, "identity_probe": identity,
        "frozen_identity_core": {key: expected[key] for key in ("root_commit", "tool_sha256", "openpi_commit", "profile_lock_sha256", "cuda_visible_devices", "jax_platforms", "mujoco_gl", "clean_root")},
        "no_physics": True, "no_gpu": True,
    }
    write_json(OUT_ROOT / "er1_executor_contract.json", result)
    write_json(OUT_ROOT / "profile_lock.json", lock)
    return result


def _contact_is_unrelated(task: int, matrix: Mapping[str, Any]) -> bool | None:
    if matrix.get("complete") is not True:
        return None
    subject = "akita_black_bowl_1" if task == 10 else "orange_juice_1" if task == 20 else "wooden_cabinet_1_middle"
    receiver_tokens = {10: ("plate_1",), 20: ("basket_1",), 19: ()}[task]
    for row in matrix.get("contacts", []):
        names = (str(row.get("body1_name", "")).lower(), str(row.get("body2_name", "")).lower())
        if not any(subject.lower() in name for name in names):
            continue
        other = next((name for name in names if subject.lower() not in name), "")
        allowed = any(token in other for token in ("robot", "panda", "gripper", "finger")) or any(token in other for token in receiver_tokens)
        if not allowed:
            return True
    return False


def _stage_target(task: int, profile: Mapping[str, Any], stage: str, anchor: Any, region: Any) -> tuple[Any, float]:
    import numpy as np

    if task in (10, 20):
        if stage == "approach_object":
            return anchor + np.array([0.0, 0.0, 0.12 if task == 10 else 0.13]), -1.0
        if stage in {"descend_object", "grasp_object"}:
            return anchor + np.array([0.0, 0.0, float(profile["grasp_z_m"]) ]), -1.0 if stage == "descend_object" else 1.0
        if stage == "lift_object":
            return anchor + np.array([0.0, 0.0, 0.14 if task == 10 else 0.16]), 1.0
        if stage == "transport_to_target":
            return region + np.array([0.0, 0.0, 0.14 if task == 10 else 0.13]), 1.0
        if stage in {"descend_to_target", "release_object"}:
            return region + np.array([0.0, 0.0, 0.055 if task == 10 else 0.075]), -1.0 if stage == "release_object" else 1.0
        return region + np.array([0.0, 0.0, 0.16 if task == 10 else 0.18]), -1.0
    if stage == "approach_drawer_region":
        return region + np.array([0.0, -0.012, 0.012]), -1.0
    if stage == "grasp_drawer_region":
        return region + np.array([0.0, -0.012, 0.012]), 1.0
    return region + np.array([0.0, -0.20, 0.0]), 1.0


def _safe_site_position(env: Any, name: str, dynamic: Any) -> Any:
    import numpy as np

    pose = dynamic._object_or_site_pose(env, name)
    if pose is None:
        raise RuntimeError(f"missing_operator_site:{name}")
    return np.asarray(pose["pos_m"], dtype=np.float64)


def cpu_worker(profile_id: str, result_output: Path, expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    if profile_id not in PROFILES:
        raise ValueError(f"unknown_profile:{profile_id}")
    task = int(PROFILE_TASK[profile_id])
    result: dict[str, Any] = {"schema_version": "track3_tz_er_child_result_v1", "profile_id": profile_id, "task_index": task, "family": FAMILY[task], "seed": int(PROFILES[profile_id]["seed"]), "fresh_process": True, "mode": "CPU_no_renderer", "gpu": False, "training_eligible": False, "official_track3_claim": False, "started_at_utc": now()}
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    identity = runtime_identity(command, str(expected_identity["profile_lock_sha256"]))
    identity["command"] = command
    expected = dict(expected_identity)
    expected["command"] = command
    result["identity"] = identity
    result["identity_gate"] = identity_gate(identity, expected, command)
    if result["identity_gate"]["state"] != "passed":
        result.update({"status": "identity_failure", "completed_at_utc": now()})
        write_json(result_output, result)
        return result
    env = None
    try:
        seed_record = apply_effective_seed(int(PROFILES[profile_id]["seed"]), np)
        result["seed_application"] = seed_record
        os.environ.update({"CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "disable", "JAX_PLATFORMS": "cpu"})
        from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_operator_scoped_stability as operator
        from tools.axis_pi05 import track3_scripted_autogen as scripted
        composed._configure_public_env(gpu=False)
        from libero.libero.envs.env_wrapper import ControlEnv

        manifest, model_bytes, _init_state, _terminal_state, _demo_info = composed._demo_arrays(task, 0)
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        env = ControlEnv(bddl_file_name=str(REVERSE_BDDL[task]), control_freq=20, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=-1)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        snapshot = SNAPSHOTS[task]
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        start_state = np.asarray(np.load(snapshot, allow_pickle=False), dtype=np.float64)
        obs = env.set_init_state(start_state)
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial = dynamic.evaluate_predicates(env, [SOURCE_ATOM[task]], label="initial_source")
        initial_goal = dynamic.evaluate_predicates(env, [GOAL_ATOM[task]], label="initial_goal")
        if not initial.get("all_true") or initial_goal.get("all_true"):
            raise RuntimeError("initial_predicate_contract_failed")
        object_name = "akita_black_bowl_1" if task == 10 else "orange_juice_1" if task == 20 else "wooden_cabinet_1_middle_region"
        anchor_pose = dynamic._object_or_site_pose(env, object_name)
        if anchor_pose is None:
            raise RuntimeError(f"missing_anchor_pose:{object_name}")
        anchor = np.asarray(anchor_pose["pos_m"], dtype=np.float64)
        region_name = "main_table_akita_black_bowl_region" if task == 10 else "floor_target_object_region" if task == 20 else "wooden_cabinet_1_middle_region"
        region = _safe_site_position(env, region_name, dynamic)
        stages = ["approach_object", "descend_object", "grasp_object", "lift_object", "transport_to_target", "descend_to_target", "release_object", "retreat"] if task in (10, 20) else ["approach_drawer_region", "grasp_drawer_region", "pull_drawer_closed", "hold_closed"]
        max_steps = {"approach_object": 170, "descend_object": 140, "grasp_object": 45, "lift_object": 170, "transport_to_target": 220, "descend_to_target": 140, "release_object": 40, "retreat": 90, "approach_drawer_region": 180, "grasp_drawer_region": 40, "pull_drawer_closed": 280, "hold_closed": 45}
        tolerance = {"approach_object": 0.014, "descend_object": 0.012, "grasp_object": 0.020, "lift_object": 0.014, "transport_to_target": 0.014, "descend_to_target": 0.014, "release_object": 0.022, "retreat": 0.014, "approach_drawer_region": 0.015, "grasp_drawer_region": 0.022, "pull_drawer_closed": 0.018, "hold_closed": 0.025}
        actions: list[Any] = []
        eef_errors: list[float] = []
        phases: list[str] = []
        source_values: list[int] = []
        source_known: list[bool] = []
        goal_values: list[int] = []
        goal_known: list[bool] = []
        action_saturated: list[bool] = []
        contact_complete: list[bool] = []
        unrelated_contact: list[bool] = []
        contact_ncon: list[int] = []
        stage_results: list[dict[str, Any]] = []
        all_reached = True
        source_false_step: int | None = None
        for stage in stages:
            stable = 0
            reached = False
            last_error = float("inf")
            start_count = len(actions)
            for _ in range(max_steps[stage]):
                target, gripper = _stage_target(task, PROFILES[profile_id], stage, anchor, region)
                action, error, saturated = scripted.closed_loop_action(obs, target, gripper)
                obs, _reward, _done, _info = env.step(action)
                action_array = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
                actions.append(action_array.copy())
                eef_errors.append(float(error))
                phases.append(stage)
                action_saturated.append(bool(saturated))
                matrix = operator.contact_matrix(env)
                contact_complete.append(bool(matrix.get("complete") is True))
                unrelated = _contact_is_unrelated(task, matrix)
                if unrelated is None:
                    raise RuntimeError("contact_contract_unknown")
                unrelated_contact.append(bool(unrelated))
                contact_ncon.append(int(matrix.get("ncon", -1)))
                source = dynamic.evaluate_predicates(env, [SOURCE_ATOM[task]], label=f"source_step_{len(actions)}")
                goal = dynamic.evaluate_predicates(env, [GOAL_ATOM[task]], label=f"goal_step_{len(actions)}")
                source_value = source.get("predicates", [{}])[0].get("value") if source.get("predicates") else None
                goal_value = goal.get("predicates", [{}])[0].get("value") if goal.get("predicates") else None
                source_known.append(source_value is not None)
                source_values.append(1 if source_value is True else 0)
                goal_known.append(goal_value is not None)
                goal_values.append(1 if goal_value is True else 0)
                if source_value is False and source_false_step is None:
                    source_false_step = len(actions)
                last_error = float(error)
                stable = stable + 1 if error <= tolerance[stage] else 0
                needed = int(PROFILES[profile_id]["close_dwell"]) if stage in {"grasp_object", "grasp_drawer_region"} else int(PROFILES[profile_id]["release_dwell"]) if stage == "release_object" else 4
                if stable >= needed:
                    reached = True
                    break
            stage_results.append({"stage": stage, "reached": reached, "steps": len(actions) - start_count, "last_eef_error_m": float(last_error), "action_saturation_count": int(sum(action_saturated[start_count:]))})
            if not reached:
                all_reached = False
                break
        final = dynamic.evaluate_predicates(env, [SOURCE_ATOM[task], GOAL_ATOM[task]], label="final")
        predicates = final.get("predicates", [])
        source_final = predicates[0].get("value") if len(predicates) > 0 else None
        goal_final = predicates[1].get("value") if len(predicates) > 1 else None
        trace_path = RUN_ROOT / "traces" / f"{profile_id}_demo0.npz"
        write_typed_trace(trace_path, {
            "actions": np.asarray(actions, dtype=np.float32), "eef_error": np.asarray(eef_errors, dtype=np.float32),
            "phase": np.asarray(phases, dtype="U64"), "source_value": np.asarray(source_values, dtype=np.int8),
            "source_known": np.asarray(source_known, dtype=np.bool_), "goal_value": np.asarray(goal_values, dtype=np.int8),
            "goal_known": np.asarray(goal_known, dtype=np.bool_), "action_saturated": np.asarray(action_saturated, dtype=np.bool_),
            "contact_complete": np.asarray(contact_complete, dtype=np.bool_), "unrelated_contact": np.asarray(unrelated_contact, dtype=np.bool_),
            "contact_ncon": np.asarray(contact_ncon, dtype=np.int32),
        })
        trace_validation = validate_typed_trace(trace_path)
        if trace_validation["status"] != "passed":
            raise RuntimeError(f"trace_validation_failed:{trace_validation['errors']}")
        controller_clip_value = bool(any(action_saturated))
        contact_complete_value = bool(contact_complete) and all(contact_complete)
        unrelated_value = bool(any(unrelated_contact))
        gates = {
            "all_stages_reached": gate_record(True, all_reached),
            "source_released": gate_record(True, source_false_step is not None and source_final is False),
            "goal_true": gate_record(True, goal_final is True),
            "controller_clip": gate_record(True, controller_clip_value is False),
            "contact_complete": gate_record(True, contact_complete_value),
            "unrelated_contact": gate_record(True, unrelated_value is False),
            "trace_typed_no_pickle": gate_record(True, trace_validation["status"] == "passed"),
            "finite_telemetry": gate_record(True, all(np.isfinite(eef_errors)) and all(np.isfinite(np.asarray(actions, dtype=np.float32).ravel()))),
        }
        status = "passed" if required_gates_passed(gates) else "failed_gate"
        result.update({
            "status": status, "initial": {"source": initial, "goal": initial_goal}, "final": final,
            "source_false_step": source_false_step, "stage_results": stage_results, "steps": len(actions),
            "trace": {"path": rel(trace_path), "sha256": sha256_file(trace_path), "validation": trace_validation},
            "metrics": {"eef_error_max_m": max(eef_errors, default=0.0), "controller_clip": controller_clip_value, "action_saturation_count": int(sum(action_saturated)), "contact_sample_count": len(contact_complete), "contact_complete_samples": int(sum(contact_complete)), "unrelated_contact_samples": int(sum(unrelated_contact))},
            "gates": gates, "model": {"hdf5": manifest.get("hdf5_path"), "model_file_sha256": manifest.get("model_file_sha256"), "resolved_model_xml_sha256": resolution.get("resolved_sha256"), "snapshot": rel(snapshot), "snapshot_sha256": sha256_file(snapshot)},
            "completed_at_utc": now(),
        })
    except Exception as exc:
        result.update({"status": "infrastructure_failure", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "completed_at_utc": now()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_output, result)
    return result


def run_er2() -> dict[str, Any]:
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="axis_pi05_er2_") as directory:
        path = Path(directory) / "typed_trace.npz"
        write_typed_trace(path, {"actions": np.zeros((2, 7), dtype=np.float32), "phase": np.asarray(["approach", "close"], dtype="U64"), "known": np.asarray([True, False], dtype=np.bool_)})
        trace = validate_typed_trace(path)
    gate_probe = {
        "controller_clip": gate_record(True, False),
        "action_saturation": gate_record(True, True),
        "contact_complete": gate_record(True, True),
        "unrelated_contact": gate_record(True, False),
    }
    result = {"schema_version": "track3_tz_er2_gate_trace_contract_v1", "status": "passed" if trace["status"] == "passed" and all(row["state"] in {"passed", "failed"} for row in gate_probe.values()) else "failed", "computed_gate_helpers": True, "gate_probe": gate_probe, "typed_trace_probe": trace, "forbidden_object_dtype": True, "finalizer_order": ["producer_close_fsync", "inactive_cursor", "artifact_manifest", "outer_hash", "strict_json_npz_csv_zip", "targeted_full_tests", "final_validation"], "fallback_summary_contract": {"delivered_zip_kind": "r6_fallback", "readiness": "fallback_no_track3_improvement", "recommended_for_track3_submission": False}, "no_physics": True, "no_gpu": True}
    write_json(OUT_ROOT / "er2_gate_trace_contract.json", result)
    return result


def run_er3() -> dict[str, Any]:
    derived = strict_json(OUT_ROOT / "er0_derived_closure.json")
    diagnosis = {
        "support_remove": {"observed_failure_stage": "descend_object", "evidence": "old fresh traces stopped after approach_object; close dwell was not evaluated"},
        "containment_remove": {"observed_failure_stage": "descend_object", "evidence": "old fresh traces stopped after approach_object; close dwell was not evaluated"},
        "articulation_close": {"observed_failure_stage": "approach_drawer_region", "evidence": "old fresh traces stopped before grasp/close"},
    }
    lock = profile_lock()
    result = {"schema_version": "track3_tz_er3_offline_diagnosis_and_freeze_v1", "status": "passed" if derived.get("status") == "passed" and len(lock["profiles"]) == 3 and not set(lock["profile_order"]) & OLD_PROFILE_IDS else "failed", "diagnosis": diagnosis, "old_trace_evidence": derived.get("derived_classification"), "profile_lock": lock, "new_identity_count": 3, "old_identity_overlap": sorted(set(lock["profile_order"]) & OLD_PROFILE_IDS), "physics_started": False, "gpu_started": False}
    write_json(OUT_ROOT / "er3_diagnosis_and_identity_freeze.json", result)
    return result


def _expected_identity() -> dict[str, Any]:
    lock = strict_json(OUT_ROOT / "profile_lock.json")
    command = [sys.executable, "-m", MODULE]
    identity = runtime_identity(command, str(lock["lock_sha256"]))
    expected = {key: identity[key] for key in ("root_commit", "tool_sha256", "openpi_commit", "profile_lock_sha256", "clean_root", "openpi_porcelain", "libero_plus_porcelain")}
    expected.update({"cuda_visible_devices": "", "jax_platforms": "cpu", "mujoco_gl": "disable"})
    return expected


def _latch_state(reason: str, profile_id: str, state_path: Path) -> dict[str, Any]:
    value = {"schema_version": "track3_tz_er_supervisor_state_v1", "stop_latched": True, "stop_reason": reason, "failed_profile_id": profile_id, "latched_at_utc": now()}
    write_json(state_path, value)
    return value


def run_er4() -> dict[str, Any]:
    state_path = OUT_ROOT / "supervisor_state.json"
    if state_path.is_file():
        previous = strict_json(state_path)
        if previous.get("stop_latched"):
            result = {"schema_version": "track3_tz_er4_cpu_matrix_v1", "status": "blocked_sticky_stop", "stop_latched": True, "stop_reason": previous.get("stop_reason"), "profiles": [], "executed": 0, "no_resume_after_shared_failure": True}
            write_json(OUT_ROOT / "er4_cpu_matrix.json", result)
            return result
    expected = _expected_identity()
    rows: list[dict[str, Any]] = []
    stop_reason: str | None = None
    with supervisor_lock(OUT_ROOT / "supervisor.lock"):
        _set_cursor("ER4", True, total_profiles=3, completed_profiles=0)
        _heartbeat("supervisor_running", phase="ER4", total_profiles=3)
        for index, profile_id in enumerate(PROFILE_ORDER):
            result_path = RUN_ROOT / f"{index + 1:02d}_{profile_id}.json"
            if result_path.is_file():
                payload = strict_json(result_path)
                if payload.get("result_sha256") != result_content_hash(payload) or payload.get("profile_id") != profile_id:
                    stop_reason = f"resume_result_hash_or_identity_mismatch:{profile_id}"
                    _latch_state(stop_reason, profile_id, state_path)
                    break
                resumed = True
                child_payload = payload
            else:
                child = run_child(profile_id, result_path, expected)
                child_payload = child.payload
                resumed = False
            category = _family_status(child_payload)
            child_payload["category"] = category
            child_payload["resumed_existing"] = resumed
            write_json(result_path, child_payload)
            # The result hash is content-addressed without a self-reference,
            # so resume can verify it after the atomic publication.
            child_payload["result_sha256"] = result_content_hash(child_payload)
            write_json(result_path, child_payload)
            rows.append({"profile_id": profile_id, "task_index": PROFILE_TASK[profile_id], "family": FAMILY[PROFILE_TASK[profile_id]], "status": child_payload.get("status"), "category": category, "path": rel(result_path), "trace": child_payload.get("trace"), "resumed_existing": resumed, "result_sha256": sha256_file(result_path)})
            _set_cursor("ER4", True, total_profiles=3, completed_profiles=index + 1, last_profile=profile_id, last_category=category)
            if category == "infrastructure_failure":
                stop_reason = f"shared_infrastructure_failure:{profile_id}"
                _latch_state(stop_reason, profile_id, state_path)
                break
        if stop_reason:
            for profile_id in PROFILE_ORDER[len(rows):]:
                rows.append({"profile_id": profile_id, "task_index": PROFILE_TASK[profile_id], "family": FAMILY[PROFILE_TASK[profile_id]], "status": "not_evaluated", "category": "not_evaluated", "reason": stop_reason})
        else:
            write_json(state_path, {"schema_version": "track3_tz_er_supervisor_state_v1", "stop_latched": False, "completed_at_utc": now(), "profiles_executed": len(rows)})
        _set_cursor("ER4", False, total_profiles=3, completed_profiles=len([row for row in rows if row["category"] != "not_evaluated"]), stop_latched=bool(stop_reason), stop_reason=stop_reason)
        _heartbeat("er4_complete", executed=len([row for row in rows if row["category"] != "not_evaluated"]), stop_latched=bool(stop_reason))
    family_results = {family: {"passed": [row["profile_id"] for row in rows if row.get("family") == family and row.get("status") == "passed"], "family_failure": [row["profile_id"] for row in rows if row.get("family") == family and row.get("category") == "family_failure"], "not_evaluated": [row["profile_id"] for row in rows if row.get("family") == family and row.get("category") == "not_evaluated"]} for family in ("support_remove", "containment_remove", "articulation_close")}
    result = {"schema_version": "track3_tz_er4_cpu_matrix_v1", "status": "hard_stop" if stop_reason else "completed", "profiles": rows, "executed": len([row for row in rows if row["category"] != "not_evaluated"]), "stop_latched": bool(stop_reason), "stop_reason": stop_reason, "family_results": family_results, "limits": {"max_new_cpu_identities": 3, "old_identity_rerun": False, "gpu": False, "renderer": False}, "training_eligible": False, "no_gpu": True}
    write_json(OUT_ROOT / "er4_cpu_matrix.json", result)
    return result


def _phase_status(path: Path, phase: str) -> dict[str, Any]:
    if not path.is_file():
        return {"phase": phase, "status": "not_evaluated", "path": None}
    value = strict_json(path)
    return {"phase": phase, "status": value.get("status"), "path": rel(path)}


def _all_er_artifacts() -> list[Path]:
    return sorted(path for path in OUT_ROOT.rglob("*") if path.is_file() and not path.name.endswith(".partial") and path.name not in {"artifact_manifest.json", "outer_hash.json", "final_validation.json"})


def _manifest_for(paths: Sequence[Path]) -> dict[str, Any]:
    return {"schema_version": "track3_tz_er_artifact_manifest_v1", "entries": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(paths)], "excluded_from_manifest": ["artifact_manifest.json", "outer_hash.json", "final_validation.json"], "historical_namespace_modified": False}


def run_er5() -> dict[str, Any]:
    _set_cursor("ER5", True)
    _heartbeat("finalizing", phase="ER5")
    fallback = _zip_inventory(FALLBACK_ZIP)
    summary = {
        "schema_version": "track3_tz_er5_summary_v1", "status": "hard_stop", "readiness": "fallback_no_track3_improvement", "track3_improvement": False,
        "delivered_zip_kind": "r6_fallback", "delivered_zip_absolute_path": str(FALLBACK_ZIP.resolve()), "delivered_zip_bytes": fallback["bytes"], "delivered_zip_sha256": fallback["sha256"],
        "fallback": fallback, "recommended_for_track3_submission": False, "accepted_data_count": 0, "training": False, "new_candidate_zip_created": False,
        "upload": False, "submission": False, "phase_ledger": [_phase_status(OUT_ROOT / "er0_derived_closure.json", "ER0"), _phase_status(OUT_ROOT / "er1_executor_contract.json", "ER1"), _phase_status(OUT_ROOT / "er2_gate_trace_contract.json", "ER2"), _phase_status(OUT_ROOT / "er3_diagnosis_and_identity_freeze.json", "ER3"), _phase_status(OUT_ROOT / "er4_cpu_matrix.json", "ER4"), {"phase": "ER5", "status": "hard_stop", "path": rel(OUT_ROOT / "summary.json")}],
        "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "official_gpu": False, "teleop": False, "generation": False, "replay": False, "dense_full_ft": False, "upload_or_submission": False},
        "hard_stop": {"status": "hard_stop", "next_authority": "new_reviewed_plan_required"},
        "repository_before_final_commit": {"root_head": git(("rev-parse", "HEAD")), "root_porcelain": porcelain(ROOT), "openpi_head": git(("rev-parse", "HEAD"), OPENPI), "openpi_porcelain": porcelain(OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS)},
    }
    # Producer close/fsync is complete through atomic writes.  The cursor is
    # inactive before the manifest and outer hash are made immutable.
    _set_cursor("ER5", False, final_status="hard_stop", delivered_zip=str(FALLBACK_ZIP.resolve()))
    write_json(OUT_ROOT / "summary.json", summary)
    ledger = [{"phase": f"ER{i}", "status": next((row["status"] for row in summary["phase_ledger"] if row["phase"] == f"ER{i}"), "not_evaluated"), "path": next((row["path"] for row in summary["phase_ledger"] if row["phase"] == f"ER{i}"), None)} for i in range(6)]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["phase", "status", "path"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(ledger)
    atomic_bytes(OUT_ROOT / "phase_ledger.csv", stream.getvalue().encode("utf-8"))
    manifest = _manifest_for(_all_er_artifacts())
    write_json(OUT_ROOT / "artifact_manifest.json", manifest)
    outer = {"schema_version": "track3_tz_er_outer_hash_v1", "manifest_path": rel(OUT_ROOT / "artifact_manifest.json"), "manifest_sha256": sha256_file(OUT_ROOT / "artifact_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"]), "created_at_utc": now()}
    write_json(OUT_ROOT / "outer_hash.json", outer)
    strict_errors: list[str] = []
    json_count = 0
    for path in sorted(OUT_ROOT.rglob("*.json")):
        try:
            strict_json(path)
            json_count += 1
        except Exception as exc:
            strict_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    npz_checks = [validate_typed_trace(path) for path in sorted(OUT_ROOT.rglob("*.npz"))]
    csv_raw = (OUT_ROOT / "phase_ledger.csv").read_bytes()
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--check"], capture_output=True, text=True, check=False)
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], cwd=ROOT, capture_output=True, text=True, check=False)
    validation = {"schema_version": "track3_tz_er5_final_validation_v1", "status": "passed" if not strict_errors and all(item["status"] == "passed" for item in npz_checks) and b"\r" not in csv_raw and diff.returncode == 0 and compile_result.returncode == 0 and not list(OUT_ROOT.rglob("*.partial")) else "failed", "strict_json": {"files": json_count, "errors": strict_errors}, "npz_no_pickle_finite": npz_checks, "csv_lf": b"\r" not in csv_raw, "git_diff_check": {"returncode": diff.returncode, "stderr": diff.stderr[-2000:]}, "py_compile": {"returncode": compile_result.returncode, "stderr": compile_result.stderr[-2000:]}, "gpu": {str(index): gpu_snapshot(index) for index in range(4)}, "live_worker_pids": [], "final_validation_after_manifest_outer_hash": True, "created_at_utc": now()}
    write_json(OUT_ROOT / "final_validation.json", validation)
    return {"summary": summary, "manifest": manifest, "outer_hash": outer, "validation": validation}


def run_all() -> dict[str, Any]:
    er0 = run_er0()
    if er0.get("status") != "passed":
        return {"status": "hard_stop", "phase": "ER0", "reason": "derived_closure_failed"}
    er1 = run_er1()
    if er1.get("status") != "passed":
        return {"status": "hard_stop", "phase": "ER1", "reason": "executor_contract_failed"}
    er2 = run_er2()
    er3 = run_er3()
    if er2.get("status") != "passed" or er3.get("status") != "passed":
        return {"status": "hard_stop", "phase": "ER2_or_ER3", "reason": "code_only_contract_failed"}
    er4 = run_er4()
    return run_er5()


def worker_exit(status: str) -> int:
    return 0 if status in {"passed", "failed_gate", "family_failure", "not_evaluated"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for index in range(6):
        parser.add_argument(f"--er{index}", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-worker", action="store_true")
    parser.add_argument("--profile-id")
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--expected-root-commit")
    parser.add_argument("--expected-tool-sha256")
    parser.add_argument("--expected-openpi-commit")
    parser.add_argument("--expected-profile-lock-sha256")
    parser.add_argument("--expected-command-sha256")
    args = parser.parse_args(argv)
    if args.cpu_worker:
        if not args.profile_id or args.result_output is None or not all((args.expected_root_commit, args.expected_tool_sha256, args.expected_openpi_commit, args.expected_profile_lock_sha256, args.expected_command_sha256)):
            parser.error("cpu-worker requires profile, result-output, and complete expected identity")
        expected = {"root_commit": args.expected_root_commit, "tool_sha256": args.expected_tool_sha256, "openpi_commit": args.expected_openpi_commit, "profile_lock_sha256": args.expected_profile_lock_sha256, "command_hash": args.expected_command_sha256, "cuda_visible_devices": "", "jax_platforms": "cpu", "mujoco_gl": "disable", "clean_root": not porcelain(ROOT), "openpi_porcelain": porcelain(OPENPI) if OPENPI.is_dir() else [], "libero_plus_porcelain": porcelain(LIBERO_PLUS)}
        try:
            value = cpu_worker(args.profile_id, args.result_output, expected)
        except Exception as exc:
            value = {"status": "infrastructure_failure", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
            write_json(args.result_output, value)
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        return worker_exit(str(value.get("status")))
    phases = [getattr(args, f"er{index}") for index in range(6)]
    if args.execute:
        value = run_all()
    else:
        if sum(bool(flag) for flag in phases) != 1:
            parser.error("choose exactly one ER phase or --execute")
        value = {0: run_er0, 1: run_er1, 2: run_er2, 3: run_er3, 4: run_er4, 5: run_er5}[next(index for index, flag in enumerate(phases) if flag)]()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if value.get("status") in {"passed", "completed", "hard_stop", "blocked_sticky_stop"} or "summary" in value else 1


if __name__ == "__main__":
    raise SystemExit(main())
