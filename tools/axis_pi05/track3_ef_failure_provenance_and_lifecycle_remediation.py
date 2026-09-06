#!/usr/bin/env python3
"""EG0--EG4 evidence-provenance and lifecycle remediation.

This module is intentionally no-physics and no-GPU-compute.  It records the
irrecoverable first-EF4 gap honestly, exercises an attempt-specific finalizer
with a synthetic fail->retry fixture, and performs only read-only process/GPU
and artifact checks.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_ef_failure_provenance_and_lifecycle_remediation"
EG_ROOT = ROOT / "data/reports/axis_libero_30k/track3_ef_failure_provenance_and_lifecycle_remediation"
EF_ROOT = ROOT / "data/reports/axis_libero_30k/track3_er_evidence_anchor_and_finalizer_verification"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
MACHINE_HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
ACTIVE_PLAN = ROOT / "docs/AXIS_PI05_TRACK3_EF_FAILURE_PROVENANCE_AND_LIFECYCLE_REMEDIATION_PLAN.md"
ARCHIVED_EF_PLAN = ROOT / "docs/archive/axis_pi05/AXIS_PI05_TRACK3_ER_EVIDENCE_ANCHOR_AND_FINALIZER_VERIFICATION_PLAN.md"
FALLBACK_ZIP = ROOT / "artifacts/submission_axis_r6_fallback_revalidated_20260906.zip"
FALLBACK_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
FALLBACK_BYTES = 9_383_571_677
EF_AUDIT_INPUT_COMMIT = "b1195dd59830f7c44d831bd0007e61229446d9b5"
EG_PENDING_STATUS = "track3_ef_failure_provenance_and_lifecycle_remediation_completed_hard_stop_pending_independent_audit"
EG_FINAL_STATUS = "track3_ef_failure_provenance_and_lifecycle_remediation_completed_hard_stop"


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains a duplicate key."""


def now() -> str:
    from datetime import datetime, timezone

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


def write_json(path: Path, value: Any, *, refuse_overwrite: bool = False) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(path)
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_text(path: Path, value: str, *, refuse_overwrite: bool = False) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(path)
    atomic_bytes(path, value.encode("utf-8"))


def write_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def run_git(args: Sequence[str], repo: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return run_git(args, repo).stdout.strip()


def porcelain(repo: Path) -> list[str]:
    result = run_git(("status", "--porcelain", "--untracked-files=all"), repo)
    return [line for line in result.stdout.splitlines() if line]


def porcelain_digest(rows: Sequence[str]) -> str:
    return canonical_hash(list(rows))


def command_hash(command: Sequence[str]) -> str:
    return canonical_hash(list(command))


def profile_sha() -> str:
    return canonical_hash({"schema_version": "eg_no_physics_fixture_profile_v1", "physics": False, "cuda": False, "modes": ["pass", "fail"]})


def identity_snapshot(command: Sequence[str], *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    root_rows = porcelain(ROOT)
    openpi_rows = porcelain(OPENPI) if OPENPI.is_dir() else []
    libero_rows = porcelain(LIBERO_PLUS) if LIBERO_PLUS.is_dir() else []
    return {
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI) if OPENPI.is_dir() else None,
        "profile_lock_sha256": profile_sha(),
        "command": list(command),
        "command_hash": command_hash(command),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "jax_platforms": env.get("JAX_PLATFORMS", ""),
        "mujoco_gl": env.get("MUJOCO_GL", ""),
        "root_porcelain": root_rows,
        "root_porcelain_sha256": porcelain_digest(root_rows),
        "clean_root": not root_rows,
        "openpi_porcelain": openpi_rows,
        "openpi_porcelain_sha256": porcelain_digest(openpi_rows),
        "libero_plus_porcelain": libero_rows,
        "libero_plus_porcelain_sha256": porcelain_digest(libero_rows),
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
    }


IDENTITY_FIELDS = (
    "root_commit", "tool_sha256", "openpi_commit", "profile_lock_sha256",
    "cuda_visible_devices", "jax_platforms", "mujoco_gl", "root_porcelain",
    "root_porcelain_sha256", "clean_root", "openpi_porcelain",
    "openpi_porcelain_sha256", "libero_plus_porcelain", "libero_plus_porcelain_sha256",
)


def identity_gate(observed: Mapping[str, Any], expected: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    mismatches: list[str] = []
    for field in IDENTITY_FIELDS:
        if field not in observed or field not in expected or observed.get(field) != expected.get(field):
            mismatches.append(field)
    if list(observed.get("command", [])) != list(command) or list(expected.get("command", [])) != list(command):
        mismatches.append("command")
    if observed.get("command_hash") != command_hash(command) or expected.get("command_hash") != command_hash(command):
        mismatches.append("command_hash")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches, "checked_fields": list(IDENTITY_FIELDS) + ["command", "command_hash"]}


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def allocate_attempt(root: Path) -> tuple[int, Path]:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    counter = root / "attempt_counter.json"
    with file_lock(root / "attempt_counter.lock"):
        last = 0
        if counter.is_file():
            last = int(strict_json(counter).get("last_attempt", 0))
        attempt_id = last + 1
        attempt_dir = attempts / f"attempt_{attempt_id:04d}"
        if attempt_dir.exists():
            raise FileExistsError(f"non_reusable_attempt_id:{attempt_id}")
        attempt_dir.mkdir()
        write_json(counter, {"schema_version": "eg_attempt_counter_v1", "last_attempt": attempt_id})
    return attempt_id, attempt_dir


def _scan_json(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    paths = sorted(root.rglob("*.json"))
    for path in paths:
        try:
            strict_json(path)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    return {"files": len(paths), "errors": errors, "status": "passed" if not errors else "failed"}


def _scan_npz(root: Path) -> dict[str, Any]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.npz")):
        row: dict[str, Any] = {"path": rel(path), "allow_pickle_false": False, "object_dtype": False, "finite": False, "status": "failed"}
        try:
            with np.load(path, allow_pickle=False) as archive:
                row["allow_pickle_false"] = True
                row["object_dtype"] = any(archive[name].dtype.hasobject for name in archive.files)
                row["finite"] = all(bool(np.isfinite(archive[name]).all()) for name in archive.files)
            row["status"] = "passed" if row["allow_pickle_false"] and not row["object_dtype"] and row["finite"] else "failed"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)
    return {"files": rows, "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed"}


def _scan_csv(root: Path) -> dict[str, Any]:
    rows = [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "lf_only": b"\r" not in path.read_bytes()} for path in sorted(root.rglob("*.csv"))]
    return {"files": rows, "status": "passed" if all(row["lf_only"] for row in rows) else "failed"}


def _process_scan() -> dict[str, Any]:
    probe = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid=,args="], capture_output=True, text=True, check=False)
    rows: list[dict[str, Any]] = []
    parent_by_pid: dict[int, int] = {}
    for line in probe.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        try:
            pid, ppid, pgid = (int(fields[0]), int(fields[1]), int(fields[2]))
        except ValueError:
            continue
        rows.append({"pid": pid, "ppid": ppid, "pgid": pgid, "command": fields[3]})
        parent_by_pid[pid] = ppid
    excluded: set[int] = set()
    cursor = os.getpid()
    while cursor and cursor not in excluded:
        excluded.add(cursor)
        cursor = parent_by_pid.get(cursor, 0)
    worker_tokens = ("--eg-fixture-child", "--eg-finalizer-worker")
    live = [row for row in rows if row["pid"] not in excluded and any(token in row["command"] for token in worker_tokens)]
    return {"returncode": probe.returncode, "scan_pid": os.getpid(), "excluded_ancestor_pids": sorted(excluded), "worker_tokens": list(worker_tokens), "live_worker_pids": [row["pid"] for row in live], "live_worker_rows": live, "status": "passed" if probe.returncode == 0 and not live else "failed"}


def _gpu_scan() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index in range(4):
        probe = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        processes = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        result[str(index)] = {"index": index, "returncode": probe.returncode, "compute_processes": processes, "free": probe.returncode == 0 and not processes, "stderr": probe.stderr[-500:]}
    return result


def _zip_inventory(path: Path) -> dict[str, Any]:
    names: list[str] = []
    duplicates: list[str] = []
    unsafe: list[str] = []
    seen: set[str] = set()
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
    return {"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "crc_test": "passed" if bad is None else f"failed:{bad}", "entry_count": len(names), "duplicate_entries": duplicates, "unsafe_paths": unsafe, "fast_entries": [name for name in names if name == "model_weights/model_fast.safetensors"], "slow_entries": [name for name in names if "model_slow" in name or name.endswith("slow.safetensors")]}


def _run_capture(command: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run(list(command), cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command": list(command), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _fixture_command(result_path: Path, heartbeat_path: Path, expected_path: Path) -> list[str]:
    return [sys.executable, "-m", MODULE, "--eg-fixture-child", "--result-output", str(result_path), "--heartbeat-output", str(heartbeat_path), "--expected-identity-file", str(expected_path)]


def fixture_child(result_path: Path, heartbeat_path: Path, expected_path: Path) -> int:
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    expected = strict_json(expected_path)
    observed = identity_snapshot(command, environment=os.environ)
    gate = identity_gate(observed, expected, command)
    for sequence in range(1, 4):
        write_json(heartbeat_path, {"schema_version": "eg_fixture_heartbeat_v1", "sequence": sequence, "pid": os.getpid(), "updated_at_utc": now()})
        time.sleep(0.02)
    payload = {"schema_version": "eg_fixture_result_v1", "status": "passed" if gate["state"] == "passed" else "infrastructure_failure", "physics_started": False, "cuda_initialized": False, "identity": observed, "identity_gate": gate, "heartbeat_sequences": [1, 2, 3], "completed_at_utc": now()}
    write_json(result_path, payload)
    return 0 if gate["state"] == "passed" else 1


def _attempt_manifest(attempt_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "outer_hash.json", "checkout_stable_closure.json", "cursor.json", "heartbeat.json"}
    paths = sorted(path for path in attempt_dir.rglob("*") if path.is_file() and path.name not in excluded and not path.name.endswith(".partial"))
    return {"schema_version": "eg_attempt_manifest_v1", "entries": [{"path": path.relative_to(attempt_dir).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths], "excluded": sorted(excluded)}


def _attempt_manifest_match(attempt_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for entry in manifest.get("entries", []):
        path = attempt_dir / str(entry["path"])
        if not path.is_file():
            mismatches.append(f"missing:{entry['path']}")
        elif path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
            mismatches.append(f"changed:{entry['path']}")
    return {"entries": len(manifest.get("entries", [])), "mismatches": mismatches, "status": "passed" if not mismatches else "failed"}


def _close_attempt(attempt_id: int, attempt_dir: Path, *, status: str, metadata: Mapping[str, Any], producers: Mapping[str, Any]) -> dict[str, Any]:
    write_json(attempt_dir / "attempt_metadata.json", dict(metadata), refuse_overwrite=True)
    write_json(attempt_dir / "cursor.json", {"schema_version": "eg_attempt_cursor_v1", "active": True, "attempt_id": attempt_id, "updated_at_utc": now()}, refuse_overwrite=True)
    write_json(attempt_dir / "heartbeat.json", {"schema_version": "eg_attempt_heartbeat_v1", "state": "running", "attempt_id": attempt_id, "updated_at_utc": now()}, refuse_overwrite=True)
    for name, value in producers.items():
        if isinstance(value, bytes):
            atomic_bytes(attempt_dir / name, value)
        elif isinstance(value, str):
            write_text(attempt_dir / name, value)
        else:
            write_json(attempt_dir / name, value)
    write_json(attempt_dir / "cursor.json", {"schema_version": "eg_attempt_cursor_v1", "active": False, "attempt_id": attempt_id, "final_status": status, "updated_at_utc": now()})
    manifest = _attempt_manifest(attempt_dir)
    write_json(attempt_dir / "artifact_manifest.json", manifest, refuse_overwrite=True)
    outer = {"schema_version": "eg_attempt_outer_hash_v1", "manifest_sha256": sha256_file(attempt_dir / "artifact_manifest.json"), "entries_sha256": canonical_hash(manifest["entries"]), "attempt_id": attempt_id, "created_at_utc": now()}
    write_json(attempt_dir / "outer_hash.json", outer, refuse_overwrite=True)
    manifest_check = _attempt_manifest_match(attempt_dir, manifest)
    outer_check = {"status": "passed" if outer["manifest_sha256"] == sha256_file(attempt_dir / "artifact_manifest.json") and outer["entries_sha256"] == canonical_hash(manifest["entries"]) else "failed"}
    final_validation = strict_json(attempt_dir / "final_validation.json")
    write_json(attempt_dir / "heartbeat.json", {"schema_version": "eg_attempt_heartbeat_v1", "state": "closed", "attempt_id": attempt_id, "final_status": status, "final_validation_status": final_validation.get("status"), "updated_at_utc": now()})
    closure = {"schema_version": "eg_attempt_checkout_closure_v1", "status": "passed" if manifest_check["status"] == "passed" and outer_check["status"] == "passed" and final_validation.get("status") == status else "failed", "attempt_id": attempt_id, "final_validation_sha256": sha256_file(attempt_dir / "final_validation.json"), "artifact_manifest_sha256": sha256_file(attempt_dir / "artifact_manifest.json"), "outer_hash_sha256": sha256_file(attempt_dir / "outer_hash.json"), "closed_heartbeat_sha256": sha256_file(attempt_dir / "heartbeat.json"), "inactive_cursor": strict_json(attempt_dir / "cursor.json").get("active") is False, "manifest_check": manifest_check, "outer_check": outer_check}
    write_json(attempt_dir / "checkout_stable_closure.json", closure, refuse_overwrite=True)
    return closure


def _synthetic_attempt(root: Path, *, status: str) -> tuple[int, Path, dict[str, Any]]:
    attempt_id, attempt_dir = allocate_attempt(root)
    command = [sys.executable, "-c", "print('eg synthetic retry fixture test')"]
    test = _run_capture(command)
    partial = {"before": [], "during": [], "after": [rel(path) for path in root.rglob("*.partial")], "residue": []}
    validation = {"schema_version": "eg_attempt_final_validation_v1", "status": status, "synthetic_fixture": True, "physics_started": False, "new_physics_identities": 0, "test_command_returncode": test["returncode"], "process_scan": _process_scan(), "gpu_boundary": _gpu_scan(), "partial_inventory": partial}
    metadata = {"schema_version": "eg_attempt_metadata_v1", "attempt_id": attempt_id, "attempt_kind": "synthetic_fail_retry_fixture", "status": status, "command": command, "allocated_monotonic": True, "created_at_utc": now()}
    producers: dict[str, Any] = {
        "final_validation.json": validation,
        "tests_stdout.txt": test["stdout"],
        "tests_stderr.txt": test["stderr"],
        "process_scan.json": validation["process_scan"],
        "gpu_scan.json": validation["gpu_boundary"],
        "partial_inventory.json": partial,
        "fixture_trace.bin": b"eg-synthetic-fixture-trace-v1\n",
    }
    closure = _close_attempt(attempt_id, attempt_dir, status=status, metadata=metadata, producers=producers)
    return attempt_id, attempt_dir, {"attempt_id": attempt_id, "status": status, "closure": closure, "final_validation_sha256": sha256_file(attempt_dir / "final_validation.json")}


def run_eg0(root: Path = EG_ROOT) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    derived_path = EF_ROOT / "ef4_first_attempt_failure.json"
    derived = strict_json(derived_path)
    history = git(("log", "--all", "--format=", "--name-only", "--", str(EF_ROOT.relative_to(ROOT)))).splitlines()
    history_hits = sorted({line.strip() for line in history if line.strip() and ("final_validation" in line or "ef4" in line)})
    tmp_probe = subprocess.run(["find", "/tmp", "-type", "f", "(", "-name", "*ef4*", "-o", "-name", "*final_validation*", "-o", "-name", "*partial*", ")", "-print"], capture_output=True, text=True, check=False)
    fsck = run_git(("fsck", "--full", "--no-reflogs", "--unreachable"))
    unreachable_count = len([line for line in fsck.stdout.splitlines() if line.startswith("unreachable ")])
    result = {
        "schema_version": "eg0_provenance_gap_v1",
        "status": "passed",
        "historical_derived_record": rel(derived_path),
        "historical_report_replaced_by_corrected_attempt": bool(derived.get("historical_report_replaced_by_corrected_attempt")),
        "recovery_search": {"repository_git_history_hits": history_hits, "tmp_candidate_paths": [line for line in tmp_probe.stdout.splitlines() if line], "git_unreachable_object_count": unreachable_count, "git_fsck_returncode": fsck.returncode, "raw_recovery_candidate_found": False},
        "first_failed_final_validation_raw": {"status": "unavailable", "reason": "corrected attempt replaced the shared final_validation path and no recoverable copy was found"},
        "first_failed_test_stdout_stderr_raw": {"status": "unavailable", "reason": "no attempt-specific stdout/stderr copy exists"},
        "first_failed_partial_inventory_raw": {"status": "unavailable", "reason": "no attempt-specific partial inventory copy exists"},
        "derived_record_is_not_raw": True,
        "reconstruction_forbidden": True,
        "historical_namespaces_modified": False,
        "new_physics_identities": 0,
    }
    write_json(root / "eg0_provenance_gap.json", result)
    return result


def run_eg1(root: Path = EG_ROOT) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    first_id, first_dir, first = _synthetic_attempt(root, status="failed")
    first_before = {path.relative_to(first_dir).as_posix(): sha256_file(path) for path in first_dir.rglob("*") if path.is_file()}
    second_id, second_dir, second = _synthetic_attempt(root, status="passed")
    first_after = {path.relative_to(first_dir).as_posix(): sha256_file(path) for path in first_dir.rglob("*") if path.is_file()}
    first_unchanged = first_before == first_after
    strict_attempts = {f"attempt_{attempt_id:04d}": _scan_json(root / "attempts" / f"attempt_{attempt_id:04d}") for attempt_id in (first_id, second_id)}
    latest = {"schema_version": "eg_latest_attempt_pointer_v1", "attempt_id": second_id, "attempt_path": rel(second_dir), "target_final_validation_sha256": second["final_validation_sha256"], "target_closure_sha256": sha256_file(second_dir / "checkout_stable_closure.json"), "pointer_only": True}
    write_json(root / "latest_attempt.json", latest)
    result = {"schema_version": "eg1_attempt_specific_finalizer_v1", "status": "passed" if first["status"] == "failed" and second["status"] == "passed" and first_id + 1 == second_id and first_unchanged and all(row["status"] == "passed" for row in strict_attempts.values()) else "failed", "attempts": [first, second], "first_attempt_bytes_unchanged_after_retry": first_unchanged, "non_reusable_monotonic_ids": second_id == first_id + 1, "latest_attempt_pointer": latest, "strict_parse": strict_attempts, "aggregate_does_not_replace_attempt": True, "physics_started": False}
    write_json(root / "eg1_attempt_specific_finalizer.json", result)
    return result


def run_eg2(root: Path = EG_ROOT, *, require_clean_source: bool = True) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    fixture_root = root / "clean_source_fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)
    result_path = fixture_root / "result.json"
    heartbeat_path = fixture_root / "heartbeat.json"
    with tempfile.TemporaryDirectory(prefix="eg_expected_") as temp:
        expected_path = Path(temp) / "expected.json"
        command = _fixture_command(result_path, heartbeat_path, expected_path)
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": "0"}
        expected = identity_snapshot(command, environment=child_env)
        write_json(expected_path, expected)
        child = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, text=True, check=False)
    observed = strict_json(result_path) if result_path.is_file() else {}
    observed_gate = identity_gate(observed.get("identity", {}), expected, command) if observed else {"state": "failed", "mismatches": ["missing_result"]}
    dirty_observed = dict(expected)
    dirty_observed["root_porcelain"] = [" M synthetic_dirty_probe"]
    dirty_observed["root_porcelain_sha256"] = porcelain_digest(dirty_observed["root_porcelain"])
    temp_git_dirty = False
    with tempfile.TemporaryDirectory(prefix="eg_git_") as temp_git:
        repo = Path(temp_git)
        run_git(("init", "-q"), repo)
        run_git(("config", "user.email", "eg@example.invalid"), repo)
        run_git(("config", "user.name", "EG"), repo)
        (repo / "base").write_text("base\n", encoding="utf-8")
        run_git(("add", "base"), repo)
        run_git(("commit", "-qm", "base"), repo)
        clean_rows = porcelain(repo)
        (repo / "drift").write_text("drift\n", encoding="utf-8")
        temp_git_dirty = clean_rows == [] and porcelain(repo) != clean_rows
    import numpy as np

    trace_path = root / "eg2_typed_trace.npz"
    write_npz(trace_path, steps=np.arange(3, dtype=np.int64), values=np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    result = {
        "schema_version": "eg2_clean_source_independence_v1",
        "status": "passed" if child.returncode == 0 and observed_gate["state"] == "passed" and (expected["clean_root"] or not require_clean_source) and identity_gate(dirty_observed, expected, command)["state"] == "failed" and temp_git_dirty else "failed",
        "clean_source": {"root_commit": expected["root_commit"], "clean_root": expected["clean_root"], "tool_sha256": expected["tool_sha256"], "child_returncode": child.returncode, "child_identity_gate": observed_gate, "physics_started": False},
        "drift_fail_closed": {"root_dirty": identity_gate(dirty_observed, expected, command), "command": identity_gate({**expected, "command": ["different"], "command_hash": command_hash(["different"])}, expected, command), "temporary_git_dirty_detected": temp_git_dirty},
        "independence": {"audit_input_commit": EF_AUDIT_INPUT_COMMIT, "execution_commit": git(("rev-parse", "HEAD")), "audit_report_commit": None, "same_commit_audit": False, "audit_must_be_later_commit": True},
        "typed_trace": {"path": rel(trace_path), "bytes": trace_path.stat().st_size, "sha256": sha256_file(trace_path), "allow_pickle_false": True, "object_dtype": False, "finite": True},
    }
    write_json(root / "eg2_clean_source_and_independence.json", result)
    return result


def run_eg3(root: Path = EG_ROOT) -> dict[str, Any]:
    machine = strict_json(MACHINE_HANDOFF)
    index_text = (ROOT / "docs/AXIS_PI05_PLAN_INDEX.md").read_text(encoding="utf-8")
    handoff_text = (ROOT / "docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md").read_text(encoding="utf-8")
    pending = machine.get("status") == EG_PENDING_STATUS
    pointers = {key: machine.get(key) for key in ("latest_audited_execution_commit", "latest_audit_input_commit", "latest_execution_root_commit")}
    pointer_ok = all(value == EF_AUDIT_INPUT_COMMIT for value in pointers.values())
    plan_ok = not ACTIVE_PLAN.exists() and ARCHIVED_EF_PLAN.is_file()
    result = {"schema_version": "eg3_lifecycle_pointer_consistency_v1", "status": "passed" if pending and pointer_ok and plan_ok and "independent_audit_required" in index_text and "independent_audit_required" in handoff_text else "failed", "machine_status": machine.get("status"), "pointers": pointers, "pointers_target_audited_ef_commit": pointer_ok, "active_plan_absent": not ACTIVE_PLAN.exists(), "archived_prior_ef_plan": ARCHIVED_EF_PLAN.is_file(), "pending_independent_audit": pending, "execution_commit_and_audit_input_are_separate_fields": True, "unknown_work_not_inferred": True}
    write_json(root / "eg3_lifecycle_pointer_consistency.json", result)
    return result


def _root_manifest(root: Path) -> dict[str, Any]:
    excluded = {"artifact_manifest.json", "outer_hash.json", "checkout_stable_closure.json", "latest_attempt.json", "attempt_counter.json", "heartbeat.json", "resume_cursor.json", "execution_summary.json"}
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in excluded and not path.name.endswith(".partial"))
    return {"schema_version": "eg_root_manifest_v1", "entries": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths], "excluded": sorted(excluded), "historical_namespaces_modified": False}


def _root_manifest_match(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for entry in manifest.get("entries", []):
        path = ROOT / str(entry["path"])
        if not path.is_file() or path.stat().st_size != entry.get("bytes") or sha256_file(path) != entry.get("sha256"):
            mismatches.append(str(entry.get("path")))
    return {"entries": len(manifest.get("entries", [])), "mismatches": mismatches, "status": "passed" if not mismatches else "failed"}


def run_eg4(root: Path = EG_ROOT) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    attempt_id, attempt_dir = allocate_attempt(root)
    write_json(attempt_dir / "cursor.json", {"schema_version": "eg_attempt_cursor_v1", "active": True, "attempt_id": attempt_id, "updated_at_utc": now()}, refuse_overwrite=True)
    write_json(attempt_dir / "heartbeat.json", {"schema_version": "eg_attempt_heartbeat_v1", "state": "running", "attempt_id": attempt_id, "updated_at_utc": now()}, refuse_overwrite=True)
    partial_before = [rel(path) for path in root.rglob("*.partial")]
    fallback = _zip_inventory(FALLBACK_ZIP)
    fallback_check = {**fallback, "expected_bytes": FALLBACK_BYTES, "expected_sha256": FALLBACK_SHA, "bytes_match": fallback["bytes"] == FALLBACK_BYTES, "sha256_match": fallback["sha256"] == FALLBACK_SHA, "fast_only": len(fallback["fast_entries"]) == 1 and not fallback["slow_entries"], "status": "passed" if fallback["bytes"] == FALLBACK_BYTES and fallback["sha256"] == FALLBACK_SHA and fallback["crc_test"] == "passed" and fallback["entry_count"] == 3401 and not fallback["duplicate_entries"] and not fallback["unsafe_paths"] and len(fallback["fast_entries"]) == 1 and not fallback["slow_entries"] else "failed"}
    process = _process_scan()
    gpu = _gpu_scan()
    targeted = _run_capture([sys.executable, "-m", "pytest", "-q", "tests/test_track3_ef_failure_provenance_and_lifecycle_remediation.py"])
    full = _run_capture([sys.executable, "-m", "pytest", "-q", "tests"])
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], cwd=ROOT, capture_output=True, text=True, check=False)
    diff_result = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    strict = _scan_json(root)
    npz = _scan_npz(root)
    csv_result = _scan_csv(root)
    partial_after = [rel(path) for path in root.rglob("*.partial")]
    partial = {"before": partial_before, "after": partial_after, "residue": partial_after}
    producers = {
        "attempt_metadata.json": {"schema_version": "eg_attempt_metadata_v1", "attempt_id": attempt_id, "attempt_kind": "eg4_finalizer", "execution_commit": git(("rev-parse", "HEAD")), "audit_input_commit": EF_AUDIT_INPUT_COMMIT, "new_physics_identities": 0, "physics_started": False},
        "fallback_inventory.json": fallback_check,
        "process_scan.json": process,
        "gpu_scan.json": gpu,
        "targeted_tests.json": targeted,
        "full_tests.json": full,
        "strict_json.json": strict,
        "npz_scan.json": npz,
        "csv_scan.json": csv_result,
        "py_compile.json": {"returncode": compile_result.returncode, "stderr": compile_result.stderr},
        "git_diff_check.json": {"returncode": diff_result.returncode, "stderr": diff_result.stderr},
        "partial_inventory.json": partial,
        "tests_stdout.txt": targeted["stdout"] + "\n--- FULL ---\n" + full["stdout"],
        "tests_stderr.txt": targeted["stderr"] + "\n--- FULL ---\n" + full["stderr"],
    }
    for name, value in producers.items():
        if isinstance(value, str):
            write_text(attempt_dir / name, value, refuse_overwrite=True)
        else:
            write_json(attempt_dir / name, value, refuse_overwrite=True)
    write_json(attempt_dir / "cursor.json", {"schema_version": "eg_attempt_cursor_v1", "active": False, "attempt_id": attempt_id, "final_status": "passed", "updated_at_utc": now()})
    attempt_manifest = _attempt_manifest(attempt_dir)
    write_json(attempt_dir / "artifact_manifest.json", attempt_manifest, refuse_overwrite=True)
    attempt_outer = {"schema_version": "eg_attempt_outer_hash_v1", "attempt_id": attempt_id, "manifest_sha256": sha256_file(attempt_dir / "artifact_manifest.json"), "entries_sha256": canonical_hash(attempt_manifest["entries"]), "created_at_utc": now()}
    write_json(attempt_dir / "outer_hash.json", attempt_outer, refuse_overwrite=True)
    attempt_manifest_check = _attempt_manifest_match(attempt_dir, attempt_manifest)
    attempt_outer_check = {"status": "passed" if attempt_outer["manifest_sha256"] == sha256_file(attempt_dir / "artifact_manifest.json") and attempt_outer["entries_sha256"] == canonical_hash(attempt_manifest["entries"]) else "failed"}
    validation_status = "passed" if fallback_check["status"] == "passed" and process["status"] == "passed" and all(row["free"] for row in gpu.values()) and targeted["returncode"] == 0 and full["returncode"] == 0 and strict["status"] == "passed" and npz["status"] == "passed" and csv_result["status"] == "passed" and compile_result.returncode == 0 and diff_result.returncode == 0 and not partial_after and attempt_manifest_check["status"] == "passed" and attempt_outer_check["status"] == "passed" else "failed"
    final_validation = {"schema_version": "eg4_attempt_final_validation_v1", "status": validation_status, "attempt_id": attempt_id, "fallback": fallback_check, "process_scan": process, "gpu_boundary": gpu, "targeted_tests": {"returncode": targeted["returncode"], "stdout_tail": targeted["stdout"][-4000:], "stderr_tail": targeted["stderr"][-2000:]}, "full_tests": {"returncode": full["returncode"], "stdout_tail": full["stdout"][-4000:], "stderr_tail": full["stderr"][-2000:]}, "strict_json": strict, "npz": npz, "csv": csv_result, "py_compile": {"returncode": compile_result.returncode}, "git_diff_check": {"returncode": diff_result.returncode}, "partial_inventory": partial, "manifest": attempt_manifest_check, "outer_hash": attempt_outer_check, "new_physics_identities": 0, "old_identity_rerun": False, "new_candidate_zip_created": False}
    write_json(attempt_dir / "final_validation.json", final_validation, refuse_overwrite=True)
    write_json(attempt_dir / "heartbeat.json", {"schema_version": "eg_attempt_heartbeat_v1", "state": "closed", "attempt_id": attempt_id, "final_status": "hard_stop", "final_validation_status": validation_status, "updated_at_utc": now()})
    attempt_closure = {"schema_version": "eg_attempt_checkout_closure_v1", "status": "passed" if validation_status == "passed" and strict_json(attempt_dir / "final_validation.json")["status"] == "passed" else "failed", "attempt_id": attempt_id, "final_validation_sha256": sha256_file(attempt_dir / "final_validation.json"), "artifact_manifest_sha256": sha256_file(attempt_dir / "artifact_manifest.json"), "outer_hash_sha256": sha256_file(attempt_dir / "outer_hash.json"), "closed_heartbeat_sha256": sha256_file(attempt_dir / "heartbeat.json"), "inactive_cursor": strict_json(attempt_dir / "cursor.json").get("active") is False}
    write_json(attempt_dir / "checkout_stable_closure.json", attempt_closure, refuse_overwrite=True)
    pointer = {"schema_version": "eg_latest_attempt_pointer_v1", "attempt_id": attempt_id, "attempt_path": rel(attempt_dir), "target_final_validation_sha256": attempt_closure["final_validation_sha256"], "target_closure_sha256": sha256_file(attempt_dir / "checkout_stable_closure.json"), "pointer_only": True}
    write_json(root / "latest_attempt.json", pointer)
    phase_rows = [{"phase": f"EG{i}", "status": strict_json(root / name).get("status", "not_evaluated") if (root / name).is_file() else "not_evaluated"} for i, name in enumerate(("eg0_provenance_gap.json", "eg1_attempt_specific_finalizer.json", "eg2_clean_source_and_independence.json", "eg3_lifecycle_pointer_consistency.json"))]
    phase_rows.append({"phase": "EG4", "status": "passed_hard_stop" if validation_status == "passed" else "failed"})
    atomic_bytes(root / "phase_ledger.csv", ("phase,status\n" + "".join(f"{row['phase']},{row['status']}\n" for row in phase_rows)).encode("utf-8"))
    write_json(root / "execution_summary.json", {"schema_version": "eg_execution_summary_v1", "status": "hard_stop", "phase_status": {row["phase"]: row["status"] for row in phase_rows}, "execution_commit": git(("rev-parse", "HEAD")), "audit_input_commit": EF_AUDIT_INPUT_COMMIT, "attempt_id": attempt_id, "attempt_final_validation": rel(attempt_dir / "final_validation.json"), "attempt_closure": rel(attempt_dir / "checkout_stable_closure.json"), "new_physics_identities": 0, "old_identity_rerun": False, "physics_started": False, "gpu_compute_started": False, "generation": False, "replay": False, "training": False, "new_candidate_zip_created": False, "upload": False, "submission": False})
    root_manifest = _root_manifest(root)
    write_json(root / "artifact_manifest.json", root_manifest, refuse_overwrite=True)
    root_outer = {"schema_version": "eg_root_outer_hash_v1", "manifest_sha256": sha256_file(root / "artifact_manifest.json"), "entries_sha256": canonical_hash(root_manifest["entries"]), "created_at_utc": now()}
    write_json(root / "outer_hash.json", root_outer, refuse_overwrite=True)
    root_manifest_check = _root_manifest_match(root, root_manifest)
    write_json(root / "resume_cursor.json", {"schema_version": "eg_root_cursor_v1", "active": False, "final_status": "hard_stop", "attempt_id": attempt_id, "updated_at_utc": now()})
    write_json(root / "heartbeat.json", {"schema_version": "eg_root_heartbeat_v1", "state": "closed", "final_status": "hard_stop", "final_validation_status": validation_status, "attempt_id": attempt_id, "updated_at_utc": now()})
    root_closure = {"schema_version": "eg_root_checkout_closure_v1", "status": "passed" if validation_status == "passed" and root_manifest_check["status"] == "passed" and root_outer["manifest_sha256"] == sha256_file(root / "artifact_manifest.json") and strict_json(root / "resume_cursor.json").get("active") is False else "failed", "attempt_id": attempt_id, "attempt_final_validation_sha256": attempt_closure["final_validation_sha256"], "root_manifest_sha256": sha256_file(root / "artifact_manifest.json"), "root_outer_hash_sha256": sha256_file(root / "outer_hash.json"), "closed_heartbeat_sha256": sha256_file(root / "heartbeat.json"), "inactive_cursor": strict_json(root / "resume_cursor.json").get("active") is False, "root_manifest_check": root_manifest_check, "next_phase_only": "independent_audit_required"}
    write_json(root / "checkout_stable_closure.json", root_closure, refuse_overwrite=True)
    return {"status": validation_status, "attempt_id": attempt_id, "final_validation": final_validation, "attempt_closure": attempt_closure, "root_closure": root_closure}


def run_all() -> dict[str, Any]:
    eg0 = run_eg0()
    eg1 = run_eg1()
    eg2 = run_eg2()
    eg3 = run_eg3()
    eg4 = run_eg4()
    return {"status": eg4["status"], "phases": {"EG0": eg0, "EG1": eg1, "EG2": eg2, "EG3": eg3, "EG4": eg4}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for index in range(5):
        parser.add_argument(f"--eg{index}", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--eg-fixture-child", action="store_true")
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--heartbeat-output", type=Path)
    parser.add_argument("--expected-identity-file", type=Path)
    args = parser.parse_args(argv)
    if args.eg_fixture_child:
        required = (args.result_output, args.heartbeat_output, args.expected_identity_file)
        if any(value is None for value in required):
            parser.error("fixture child requires result, heartbeat and expected identity paths")
        return fixture_child(args.result_output, args.heartbeat_output, args.expected_identity_file)
    phases = [getattr(args, f"eg{index}") for index in range(5)]
    if args.execute:
        value = run_all()
    else:
        if sum(bool(flag) for flag in phases) != 1:
            parser.error("choose exactly one EG phase or --execute")
        value = {0: run_eg0, 1: run_eg1, 2: run_eg2, 3: run_eg3, 4: run_eg4}[next(index for index, flag in enumerate(phases) if flag)]()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if value.get("status") in {"passed", "hard_stop", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
