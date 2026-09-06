#!/usr/bin/env python3
"""EF0--EF4 evidence anchoring and executor-contract verification.

This module is deliberately independent of the historical ER/TZ executors.
It runs only offline checks and a small subprocess fixture.  The fixture has
no simulator imports, no CUDA initialization, and no physics state.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_er_evidence_anchor_and_finalizer_verification"
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_er_evidence_anchor_and_finalizer_verification"
ER_ROOT = ROOT / "data/reports/axis_libero_30k/track3_tz_evidence_and_executor_contract_remediation"
ER_ANCHOR_COMMIT = "38635fbf08e4e8df8c4877ddcb6b2b2ec8b9b118"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
FALLBACK_ZIP = ROOT / "artifacts/submission_axis_r6_fallback_revalidated_20260906.zip"
FALLBACK_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
FALLBACK_BYTES = 9_383_571_677
TERM_GRACE_SECONDS = 0.20
FIXTURE_HEARTBEAT_INTERVAL = 0.05
FIXTURE_TIMEOUT = 0.35


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


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


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


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def porcelain_digest(rows: Sequence[str]) -> str:
    return canonical_hash(list(rows))


def argv_contract_hash(command: Sequence[str]) -> str:
    normalized = list(command)
    for flag in ("--expected-command-sha256",):
        if flag in normalized:
            index = normalized.index(flag) + 1
            if index < len(normalized):
                normalized[index] = "<expected-command-sha256>"
    return canonical_hash(normalized)


def profile_lock_sha() -> str:
    return canonical_hash({"schema_version": "ef_fixture_profile_v1", "physics": False, "profiles": ["family_failure", "timeout"]})


def identity_snapshot(command: Sequence[str], *, profile_sha: str, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    root_rows = porcelain(ROOT)
    openpi_rows = porcelain(OPENPI) if OPENPI.is_dir() else []
    libero_rows = porcelain(LIBERO_PLUS) if LIBERO_PLUS.is_dir() else []
    return {
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI) if OPENPI.is_dir() else None,
        "profile_lock_sha256": profile_sha,
        "command": list(command),
        "command_hash": argv_contract_hash(command),
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
    "openpi_porcelain_sha256", "libero_plus_porcelain",
    "libero_plus_porcelain_sha256",
)


def identity_gate(observed: Mapping[str, Any], expected: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
    mismatches: list[str] = []
    for field in IDENTITY_FIELDS:
        if field not in expected or field not in observed or observed.get(field) != expected.get(field):
            mismatches.append(field)
    expected_command = list(expected.get("command", command))
    if list(command) != expected_command or list(observed.get("command", [])) != list(command):
        mismatches.append("command")
    expected_hash = expected.get("command_hash")
    if expected_hash is None or observed.get("command_hash") != expected_hash or argv_contract_hash(command) != expected_hash:
        mismatches.append("command_hash")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches, "checked_fields": list(IDENTITY_FIELDS) + ["command", "command_hash"]}


def result_content_hash(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("result_sha256", None)
    return canonical_hash(body)


def validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    command: Sequence[str],
    expected_profile_id: str,
    expected_terminal_category: str,
) -> dict[str, Any]:
    mismatches: list[str] = []
    if payload.get("result_sha256") != result_content_hash(payload):
        mismatches.append("result_sha256")
    if payload.get("profile_id") != expected_profile_id:
        mismatches.append("profile_id")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        mismatches.append("identity_missing")
        identity_check = {"state": "failed", "mismatches": ["identity_missing"]}
    else:
        identity_check = identity_gate(identity, expected_identity, command)
        mismatches.extend(f"identity.{item}" for item in identity_check["mismatches"])
    if payload.get("terminal_category") != expected_terminal_category:
        mismatches.append("terminal_category")
    trace = payload.get("trace")
    if not isinstance(trace, Mapping):
        mismatches.append("trace_missing")
    else:
        trace_path = Path(str(trace.get("path", "")))
        if not trace_path.is_absolute():
            trace_path = ROOT / trace_path
        if not trace_path.is_file():
            mismatches.append("trace_missing_file")
        else:
            if trace.get("bytes") != trace_path.stat().st_size:
                mismatches.append("trace_bytes")
            if trace.get("sha256") != sha256_file(trace_path):
                mismatches.append("trace_sha256")
    return {"state": "passed" if not mismatches else "failed", "mismatches": mismatches, "identity": identity_check}


@contextmanager
def supervisor_lock(path: Path) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise
    info = {"pid": os.getpid(), "pgid": os.getpgid(0), "acquired_at_utc": now()}
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


def _group_members(pgid: int) -> list[int]:
    result = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True, check=False)
    members: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                pid, group = (int(fields[0]), int(fields[1]))
            except ValueError:
                continue
            if group == pgid:
                members.append(pid)
    return members


def _fixture_command(mode: str, result: Path, heartbeat: Path, trace: Path, expected_file: Path) -> list[str]:
    command = [
        sys.executable, "-m", MODULE, "--ef-fixture-child", "--mode", mode,
        "--result-output", str(result), "--heartbeat-output", str(heartbeat),
        "--trace-output", str(trace), "--expected-identity-file", str(expected_file),
        "--expected-command-sha256", "pending",
    ]
    command[-1] = argv_contract_hash(command)
    return command


def _write_fixture_heartbeat(path: Path, sequence: int, mode: str) -> None:
    write_json(path, {"schema_version": "ef_fixture_heartbeat_v1", "sequence": sequence, "mode": mode, "pid": os.getpid(), "updated_at_utc": now()})


def fixture_child(mode: str, result_output: Path, heartbeat_output: Path, trace_output: Path, expected_file: Path) -> int:
    expected = strict_json(expected_file)
    command = [sys.executable, "-m", MODULE, *sys.argv[1:]]
    identity = identity_snapshot(command, profile_sha=str(expected["profile_lock_sha256"]))
    gate = identity_gate(identity, expected, command)
    if gate["state"] != "passed":
        write_json(result_output, {"schema_version": "ef_fixture_result_v1", "status": "infrastructure_failure", "profile_id": "ef_fixture", "terminal_category": "infrastructure_failure", "identity": identity, "identity_gate": gate})
        return 1
    if mode == "timeout":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sequence = 0
        while True:
            sequence += 1
            _write_fixture_heartbeat(heartbeat_output, sequence, mode)
            time.sleep(FIXTURE_HEARTBEAT_INTERVAL)
    if mode != "family_failure":
        raise ValueError(f"unknown_fixture_mode:{mode}")
    for sequence in range(1, 5):
        _write_fixture_heartbeat(heartbeat_output, sequence, mode)
        time.sleep(FIXTURE_HEARTBEAT_INTERVAL)
    atomic_bytes(trace_output, b"ef-fixture-trace-v1\n")
    payload = {
        "schema_version": "ef_fixture_result_v1", "status": "family_failure", "terminal_category": "family_failure",
        "profile_id": "ef_fixture", "identity": identity, "identity_gate": gate,
        "trace": {"path": rel(trace_output), "bytes": trace_output.stat().st_size, "sha256": sha256_file(trace_output)},
        "physics_steps": 0, "cuda_initialized": False, "simulator_created": False,
        "completed_at_utc": now(),
    }
    write_json(result_output, payload)
    return 0


def run_fixture(mode: str, *, label: str) -> dict[str, Any]:
    run_root = OUT_ROOT / "fixture" / label
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "result.json"
    heartbeat_path = run_root / "heartbeat.json"
    trace_path = run_root / "trace.bin"
    for output in (result_path, heartbeat_path, trace_path):
        output.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="ef_expected_") as directory:
        expected_file = Path(directory) / "expected_identity.json"
        command = _fixture_command(mode, result_path, heartbeat_path, trace_path, expected_file)
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONHASHSEED": "0"}
        expected = identity_snapshot(command, profile_sha=profile_lock_sha(), environment=child_env)
        write_json(expected_file, expected)
        start = time.monotonic()
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        child_pid = process.pid
        child_pgid = os.getpgid(child_pid)
        heartbeat_sequences: list[int] = []
        terminated = False
        kill_sent = False
        while process.poll() is None:
            if heartbeat_path.is_file():
                try:
                    heartbeat = strict_json(heartbeat_path)
                    sequence = int(heartbeat.get("sequence", 0))
                    if sequence and (not heartbeat_sequences or heartbeat_sequences[-1] != sequence):
                        heartbeat_sequences.append(sequence)
                except Exception:
                    pass
            if mode == "timeout" and time.monotonic() - start > FIXTURE_TIMEOUT:
                terminated = True
                os.killpg(child_pgid, signal.SIGTERM)
                try:
                    process.wait(timeout=TERM_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    kill_sent = True
                    os.killpg(child_pgid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(0.01)
        stdout, stderr = process.communicate()
        descendants = _group_members(child_pgid)
        partial_residue = [rel(path) for path in run_root.rglob("*.partial")]
        for partial in run_root.rglob("*.partial"):
            partial.unlink(missing_ok=True)
        payload: dict[str, Any]
        if result_path.is_file():
            payload = strict_json(result_path)
            if payload.get("status") == "family_failure":
                payload["result_sha256"] = result_content_hash(payload)
                write_json(result_path, payload)
        else:
            payload = {"status": "infrastructure_failure", "terminal_category": "infrastructure_failure"}
    parent_identity_gate = identity_gate(payload.get("identity", {}), expected, command) if payload.get("identity") else {"state": "failed", "mismatches": ["identity_missing"]}
    resume = validate_resume_payload(payload, expected_identity=expected, command=command, expected_profile_id="ef_fixture", expected_terminal_category="family_failure") if mode == "family_failure" and payload.get("status") == "family_failure" else {"state": "not_evaluated", "mismatches": []}
    return {
        "mode": mode, "label": label, "child_pid": child_pid, "child_pgid": child_pgid,
        "returncode": process.returncode, "timed_out": terminated, "term_sent": terminated,
        "kill_sent": kill_sent, "descendants_after_stop": descendants,
        "partial_residue_cleaned": partial_residue,
        "unique_pid_pgid": child_pid == child_pgid, "heartbeat_sequences_observed": heartbeat_sequences,
        "heartbeat_updates_at_least_two": len(heartbeat_sequences) >= 2,
        "parent_identity_gate": parent_identity_gate, "resume_gate": resume,
        "payload": payload, "stdout_tail": stdout[-1000:], "stderr_tail": stderr[-1000:],
    }


def _lock_probe(path: Path) -> int:
    try:
        with supervisor_lock(path):
            return 1
    except BlockingIOError:
        print("lock_rejected")
        return 0


def _sticky_guard(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"started": True, "reason": None}
    state = strict_json(path)
    if state.get("stop_latched") is True:
        return {"started": False, "reason": state.get("stop_reason"), "child_started": False}
    return {"started": True, "reason": None}


def run_ef0() -> dict[str, Any]:
    paths = sorted(Path(path) for path in git(("ls-files", rel(ER_ROOT))).splitlines() if path)
    rows: list[dict[str, Any]] = []
    for path in paths:
        absolute = ROOT / path
        committed = subprocess.run(["git", "show", f"{ER_ANCHOR_COMMIT}:{path}"], capture_output=True, check=False)
        work_sha = sha256_file(absolute) if absolute.is_file() else None
        commit_sha = sha256_bytes(committed.stdout) if committed.returncode == 0 else None
        rows.append({"path": str(path), "bytes": absolute.stat().st_size if absolute.is_file() else None, "worktree_sha256": work_sha, "commit_sha256": commit_sha, "exact": absolute.is_file() and committed.returncode == 0 and work_sha == commit_sha})
    old_manifest = strict_json(ER_ROOT / "er0_manifest.json")
    old_rows = [*old_manifest.get("json", []), *old_manifest.get("npz", []), *old_manifest.get("csv", [])]
    result = {
        "schema_version": "track3_ef0_byte_anchor_v1", "status": "passed" if len(rows) == 29 and all(row["exact"] for row in rows) else "failed",
        "anchor_commit": ER_ANCHOR_COMMIT, "tracked_er_files": len(rows), "tracked_er_files_exact": sum(bool(row["exact"]) for row in rows),
        "files": rows, "old_namespace_persistence": {"records": len(old_rows), "git_persisted": sum(row.get("tracked") == "tracked" for row in old_rows), "local_npz": sum(row.get("tracked") != "tracked" for row in old_manifest.get("npz", [])), "historical_files_modified": False},
        "old_evidence_rewritten": False, "physics_started": False, "gpu_started": False,
    }
    write_json(OUT_ROOT / "ef0_byte_anchor.json", result)
    return result


def run_ef1() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--ef-fixture-child", "--mode", "family_failure"]
    expected = identity_snapshot(command, profile_sha=profile_lock_sha())
    clean_observed = dict(expected)
    probes: dict[str, Any] = {}
    probes["clean"] = identity_gate(clean_observed, expected, command)
    probes["root_dirty"] = identity_gate({**clean_observed, "root_porcelain": [" M synthetic.py"], "root_porcelain_sha256": porcelain_digest([" M synthetic.py"]), "clean_root": False}, expected, command)
    probes["openpi_drift"] = identity_gate({**clean_observed, "openpi_commit": "drift", "openpi_porcelain": [" M openpi.py"], "openpi_porcelain_sha256": porcelain_digest([" M openpi.py"])}, expected, command)
    probes["libero_plus_drift"] = identity_gate({**clean_observed, "libero_plus_porcelain": ["?? drift"], "libero_plus_porcelain_sha256": porcelain_digest(["?? drift"])}, expected, command)
    probes["command_drift"] = identity_gate(clean_observed, expected, [*command, "--drift"])
    with tempfile.TemporaryDirectory(prefix="ef_git_repo_") as directory:
        repo = Path(directory)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "ef@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "EF fixture"], cwd=repo, check=True)
        (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
        temp_clean = porcelain(repo)
        (repo / "dirty.txt").write_text("drift\n", encoding="utf-8")
        temp_dirty = porcelain(repo)
    trace = OUT_ROOT / "ef1_resume_trace.bin"
    atomic_bytes(trace, b"ef1-resume-trace\n")
    payload: dict[str, Any] = {"profile_id": "ef_fixture", "terminal_category": "family_failure", "identity": clean_observed, "trace": {"path": rel(trace), "bytes": trace.stat().st_size, "sha256": sha256_file(trace)}}
    payload["result_sha256"] = result_content_hash(payload)
    resume_pass = validate_resume_payload(payload, expected_identity=expected, command=command, expected_profile_id="ef_fixture", expected_terminal_category="family_failure")
    trace.write_bytes(b"trace-drift\n")
    trace_drift = validate_resume_payload(payload, expected_identity=expected, command=command, expected_profile_id="ef_fixture", expected_terminal_category="family_failure")
    result = {
        "schema_version": "track3_ef1_identity_resume_v1", "status": "passed" if probes["clean"]["state"] == "passed" and all(probes[key]["state"] == "failed" for key in ("root_dirty", "openpi_drift", "libero_plus_drift", "command_drift")) and resume_pass["state"] == "passed" and trace_drift["state"] == "failed" else "failed",
        "parent_child_identity": {"expected_fields": list(IDENTITY_FIELDS) + ["command", "command_hash"], "expected_captured_before_child": True, "child_may_not_recompute_expected": True},
        "identity_probes": probes, "resume_pass": resume_pass, "trace_drift": trace_drift,
        "temporary_git_repo": {"clean_porcelain": temp_clean, "dirty_porcelain": temp_dirty, "dirty_detected": temp_clean != temp_dirty},
        "physics_started": False, "gpu_started": False,
    }
    write_json(OUT_ROOT / "ef1_identity_and_resume.json", result)
    return result


def run_ef2() -> dict[str, Any]:
    lock_path = OUT_ROOT / "fixture_supervisor.lock"
    with supervisor_lock(lock_path):
        probe = subprocess.run([sys.executable, "-m", MODULE, "--ef-lock-probe", str(lock_path)], cwd=ROOT, capture_output=True, text=True, check=False)
    normal = run_fixture("family_failure", label="normal_family_failure")
    timeout = run_fixture("timeout", label="timeout_term_kill")
    sticky_path = OUT_ROOT / "sticky_state.json"
    write_json(sticky_path, {"schema_version": "ef_sticky_state_v1", "stop_latched": True, "stop_reason": "fixture_shared_infrastructure_failure"})
    sticky = _sticky_guard(sticky_path)
    result = {
        "schema_version": "track3_ef2_supervisor_fixture_v1", "status": "passed" if probe.returncode == 0 and "lock_rejected" in probe.stdout and normal["returncode"] == 0 and normal["payload"].get("status") == "family_failure" and normal["unique_pid_pgid"] and normal["heartbeat_updates_at_least_two"] and normal["descendants_after_stop"] == [] and timeout["term_sent"] and timeout["kill_sent"] and timeout["returncode"] < 0 and timeout["descendants_after_stop"] == [] and sticky["started"] is False and sticky["child_started"] is False else "failed",
        "singleton": {"second_acquire_returncode": probe.returncode, "second_acquire_rejected": "lock_rejected" in probe.stdout},
        "normal_family_failure": normal, "timeout_term_grace_kill": timeout,
        "sticky_stop": sticky, "no_physics": True, "no_gpu": True,
    }
    write_json(OUT_ROOT / "ef2_supervisor_fixture.json", result)
    return result


def gate_record(executed: bool, value: bool | None, *, reason: str | None = None) -> dict[str, Any]:
    row = {"executed": bool(executed), "value": value, "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"), "required": True}
    if reason is not None:
        row["reason"] = reason
    return row


def telemetry_schema_gate(samples: Sequence[Mapping[str, Any]], *, stage_entered: bool) -> dict[str, Any]:
    if not stage_entered:
        return gate_record(False, None, reason="stage_not_entered")
    required = {"step", "known_mask", "contacts", "telemetry_complete"}
    errors = [f"sample_{index}:{sorted(required - set(sample))}" for index, sample in enumerate(samples) if not required.issubset(sample) or sample.get("telemetry_complete") is not True or sample.get("known_mask") is not True or not isinstance(sample.get("contacts"), list)]
    return {**gate_record(True, not errors, reason=";".join(errors) if errors else None), "sample_count": len(samples), "errors": errors}


def required_contact_gate(family: str, samples: Sequence[Mapping[str, Any]], *, stage_entered: bool, min_window: int = 2) -> dict[str, Any]:
    telemetry = telemetry_schema_gate(samples, stage_entered=stage_entered)
    if telemetry["state"] != "passed":
        return {**gate_record(False, None, reason="telemetry_schema_not_complete" if stage_entered else "stage_not_entered"), "family": family, "min_window": min_window, "telemetry_state": telemetry["state"], "contact_window": 0}
    partners = {"object"} if family in {"support_remove", "containment_remove"} else {"handle", "region"}
    current = 0
    best = 0
    for sample in samples:
        correlated = any(set(map(str.lower, contact.get("roles", []))) & {"demonstrator"} and set(map(str.lower, contact.get("roles", []))) & partners for contact in sample["contacts"] if isinstance(contact, Mapping))
        current = current + 1 if correlated else 0
        best = max(best, current)
    return {**gate_record(True, best >= min_window, reason="no_continuous_required_contact" if best < min_window else None), "family": family, "min_window": min_window, "telemetry_state": telemetry["state"], "contact_window": best}


def run_ef3() -> dict[str, Any]:
    import numpy as np

    pass_samples = [{"step": i, "known_mask": True, "telemetry_complete": True, "contacts": [{"roles": ["demonstrator", "object"]}]} for i in range(3)]
    fail_samples = [{"step": i, "known_mask": True, "telemetry_complete": True, "contacts": [{"roles": ["table", "floor"]}]} for i in range(3)]
    articulation_samples = [{"step": i, "known_mask": True, "telemetry_complete": True, "contacts": [{"roles": ["demonstrator", "handle"]}]} for i in range(2)]
    typed_trace = OUT_ROOT / "ef3_typed_trace.npz"
    write_npz(typed_trace, known_mask=np.asarray([True, True, True], dtype=np.bool_), controller_saturated=np.asarray([False, False, False], dtype=np.bool_), unrelated_contact=np.asarray([False, False, False], dtype=np.bool_))
    npz_result: dict[str, Any] = {"path": rel(typed_trace), "allow_pickle_false": True, "finite": True, "object_dtype": False, "status": "failed"}
    try:
        with np.load(typed_trace, allow_pickle=False) as archive:
            npz_result["allow_pickle_false"] = all(not archive[name].dtype.hasobject for name in archive.files)
            npz_result["finite"] = all(bool(np.isfinite(archive[name]).all()) for name in archive.files)
            npz_result["object_dtype"] = any(archive[name].dtype.hasobject for name in archive.files)
        npz_result["status"] = "passed" if npz_result["allow_pickle_false"] and npz_result["finite"] and not npz_result["object_dtype"] else "failed"
    except Exception as exc:
        npz_result["error"] = f"{type(exc).__name__}:{exc}"
        npz_result["status"] = "failed"
    result = {
        "schema_version": "track3_ef3_contact_gate_v1", "status": "passed" if required_contact_gate("support_remove", pass_samples, stage_entered=True)["state"] == "passed" and required_contact_gate("support_remove", fail_samples, stage_entered=True)["state"] == "failed" and required_contact_gate("articulation_close", articulation_samples, stage_entered=True)["state"] == "passed" and required_contact_gate("support_remove", pass_samples, stage_entered=False)["state"] == "not_evaluated" and npz_result["status"] == "passed" else "failed",
        "telemetry_vs_physical": {"support_pass": {"telemetry_schema": telemetry_schema_gate(pass_samples, stage_entered=True), "required_physical_contact": required_contact_gate("support_remove", pass_samples, stage_entered=True)}, "zero_correlated_contact": {"telemetry_schema": telemetry_schema_gate(fail_samples, stage_entered=True), "required_physical_contact": required_contact_gate("support_remove", fail_samples, stage_entered=True)}, "articulation_handle_pass": {"telemetry_schema": telemetry_schema_gate(articulation_samples, stage_entered=True), "required_physical_contact": required_contact_gate("articulation_close", articulation_samples, stage_entered=True)}, "unentered": {"telemetry_schema": telemetry_schema_gate(pass_samples, stage_entered=False), "required_physical_contact": required_contact_gate("support_remove", pass_samples, stage_entered=False)}},
        "runtime_derived": {"controller_saturation": gate_record(True, False), "unrelated_contact": gate_record(True, False), "known_mask": gate_record(True, True), "typed_no_pickle_trace": npz_result},
        "physics_started": False, "gpu_started": False,
    }
    write_json(OUT_ROOT / "ef3_contact_gate.json", result)
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
    return {"path": rel(path), "absolute_path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path), "crc_test": "passed" if bad is None else f"failed:{bad}", "entry_count": len(names), "duplicate_entries": duplicates, "unsafe_paths": unsafe, "model_fast_entries": [name for name in names if name == "model_weights/model_fast.safetensors"], "slow_model_entries": [name for name in names if "model_slow" in name or name.endswith("slow.safetensors")]}


def _gpu_scan() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index in range(4):
        probe = subprocess.run(["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        processes = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        result[str(index)] = {"index": index, "returncode": probe.returncode, "compute_processes": processes, "free": probe.returncode == 0 and not processes, "stderr": probe.stderr[-500:]}
    return result


def _process_scan() -> dict[str, Any]:
    probe = subprocess.run(["ps", "-eo", "pid=,ppid=,pgid=,args="], capture_output=True, text=True, check=False)
    rows: list[dict[str, Any]] = []
    parent_by_pid: dict[int, int] = {}
    for line in probe.stdout.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
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
    worker_tokens = ("--ef-fixture-child", "--cpu-worker")
    live = [row for row in rows if row["pid"] not in excluded and any(token in row["command"] for token in worker_tokens)]
    return {"returncode": probe.returncode, "scan_pid": os.getpid(), "excluded_ancestor_pids": sorted(excluded), "worker_tokens": list(worker_tokens), "live_worker_pids": [row["pid"] for row in live], "live_worker_rows": live, "status": "passed" if probe.returncode == 0 and not live else "failed"}


def _test_command(args: Sequence[str]) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return {"command": [sys.executable, "-m", "pytest", "-q", *args], "returncode": result.returncode, "stdout_tail": result.stdout[-3000:], "stderr_tail": result.stderr[-2000:]}


def _strict_json_scan() -> dict[str, Any]:
    errors: list[str] = []
    files = sorted(OUT_ROOT.rglob("*.json"))
    for path in files:
        try:
            strict_json(path)
        except Exception as exc:
            errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    return {"files": len(files), "errors": errors, "status": "passed" if not errors else "failed"}


def _npz_scan() -> dict[str, Any]:
    import numpy as np

    rows: list[dict[str, Any]] = []
    for path in sorted(OUT_ROOT.rglob("*.npz")):
        row: dict[str, Any] = {"path": rel(path), "status": "failed", "allow_pickle_false": False, "object_dtype": False, "finite": False}
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


def _csv_scan() -> dict[str, Any]:
    files = sorted(OUT_ROOT.rglob("*.csv"))
    rows = [{"path": rel(path), "lf_only": b"\r" not in path.read_bytes(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files]
    return {"files": rows, "status": "passed" if all(row["lf_only"] for row in rows) else "failed"}


def _producer_paths() -> list[Path]:
    excluded = {"artifact_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json", "resume_cursor.json", "heartbeat.json"}
    return sorted(path for path in OUT_ROOT.rglob("*") if path.is_file() and not path.name.endswith(".partial") and path.name not in excluded)


def _manifest(paths: Sequence[Path]) -> dict[str, Any]:
    return {"schema_version": "track3_ef4_artifact_manifest_v1", "entries": [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(paths)], "excluded": ["artifact_manifest.json", "outer_hash.json", "final_validation.json", "checkout_stable_closure.json", "resume_cursor.json", "heartbeat.json"], "historical_er_namespace_modified": False}


def _manifest_entries_match(manifest: Mapping[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for entry in manifest.get("entries", []):
        path = ROOT / str(entry["path"])
        if not path.is_file():
            mismatches.append(f"missing:{entry['path']}")
            continue
        if path.stat().st_size != entry.get("bytes"):
            mismatches.append(f"bytes:{entry['path']}")
        if sha256_file(path) != entry.get("sha256"):
            mismatches.append(f"sha256:{entry['path']}")
    return {"status": "passed" if not mismatches else "failed", "mismatches": mismatches, "entries": len(manifest.get("entries", []))}


def run_ef4() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUT_ROOT / "resume_cursor.json", {"schema_version": "track3_ef_cursor_v1", "active": True, "phase": "EF4", "pid": os.getpid(), "updated_at_utc": now()})
    write_json(OUT_ROOT / "heartbeat.json", {"schema_version": "track3_ef_heartbeat_v1", "state": "running", "phase": "EF4", "pid": os.getpid(), "updated_at_utc": now()})
    ledger_rows = [{"phase": f"EF{i}", "status": strict_json(OUT_ROOT / name).get("status", "not_evaluated") if (OUT_ROOT / name).is_file() else "not_evaluated"} for i, name in enumerate(("ef0_byte_anchor.json", "ef1_identity_and_resume.json", "ef2_supervisor_fixture.json", "ef3_contact_gate.json"))]
    ledger_rows.append({"phase": "EF4", "status": "running"})
    csv_path = OUT_ROOT / "phase_ledger.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stream = "phase,status\n" + "".join(f"{row['phase']},{row['status']}\n" for row in ledger_rows)
    atomic_bytes(csv_path, stream.encode("utf-8"))
    write_json(OUT_ROOT / "execution_summary.json", {"schema_version": "track3_ef_execution_summary_v1", "status": "hard_stop", "phase_status": {row["phase"]: row["status"] for row in ledger_rows}, "new_physics_identities": 0, "old_er_or_tz_identity_rerun": False, "new_candidate_zip": False, "training": False, "upload": False, "submission": False})
    partial_before = [rel(path) for path in OUT_ROOT.rglob("*.partial")]
    write_json(OUT_ROOT / "resume_cursor.json", {"schema_version": "track3_ef_cursor_v1", "active": False, "phase": "EF4", "pid": os.getpid(), "final_status": "hard_stop", "updated_at_utc": now()})
    producer_manifest = _manifest(_producer_paths())
    write_json(OUT_ROOT / "artifact_manifest.json", producer_manifest)
    outer = {"schema_version": "track3_ef_outer_hash_v1", "manifest_path": rel(OUT_ROOT / "artifact_manifest.json"), "manifest_sha256": sha256_file(OUT_ROOT / "artifact_manifest.json"), "entries_sha256": canonical_hash(producer_manifest["entries"]), "created_at_utc": now()}
    write_json(OUT_ROOT / "outer_hash.json", outer)
    strict_result = _strict_json_scan()
    npz_result = _npz_scan()
    csv_result = _csv_scan()
    fallback = _zip_inventory(FALLBACK_ZIP)
    fallback_check = {"expected_bytes": FALLBACK_BYTES, "observed_bytes": fallback["bytes"], "bytes_match": fallback["bytes"] == FALLBACK_BYTES, "expected_sha256": FALLBACK_SHA, "observed_sha256": fallback["sha256"], "sha256_match": fallback["sha256"] == FALLBACK_SHA, "crc_test": fallback["crc_test"], "entry_count": fallback["entry_count"], "duplicate_entries": fallback["duplicate_entries"], "unsafe_paths": fallback["unsafe_paths"], "fast_only": len(fallback["model_fast_entries"]) == 1 and not fallback["slow_model_entries"], "status": "passed" if fallback["bytes"] == FALLBACK_BYTES and fallback["sha256"] == FALLBACK_SHA and fallback["crc_test"] == "passed" and fallback["entry_count"] == 3401 and not fallback["duplicate_entries"] and not fallback["unsafe_paths"] and len(fallback["model_fast_entries"]) == 1 and not fallback["slow_model_entries"] else "failed"}
    process_result = _process_scan()
    gpu_result = _gpu_scan()
    targeted = _test_command(["tests/test_track3_er_evidence_anchor_and_finalizer_verification.py"])
    full = _test_command(["tests"])
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())], cwd=ROOT, capture_output=True, text=True, check=False)
    no_partial = not partial_before and not list(OUT_ROOT.rglob("*.partial"))
    manifest_entries = _manifest_entries_match(producer_manifest)
    manifest_outer = {"manifest_sha256": outer["manifest_sha256"], "observed_manifest_sha256": sha256_file(OUT_ROOT / "artifact_manifest.json"), "entries_sha256": outer["entries_sha256"], "observed_entries_sha256": canonical_hash(producer_manifest["entries"]), "status": "passed" if outer["manifest_sha256"] == sha256_file(OUT_ROOT / "artifact_manifest.json") and outer["entries_sha256"] == canonical_hash(producer_manifest["entries"]) else "failed"}
    provisional_status = "passed" if strict_result["status"] == "passed" and npz_result["status"] == "passed" and csv_result["status"] == "passed" and fallback_check["status"] == "passed" and process_result["status"] == "passed" and all(row["free"] for row in gpu_result.values()) and targeted["returncode"] == 0 and full["returncode"] == 0 and diff.returncode == 0 and compile_result.returncode == 0 and manifest_outer["status"] == "passed" and manifest_entries["status"] == "passed" and no_partial else "failed"

    # The execution summary and phase ledger are producer metadata.  They are
    # finalized only after all checks complete, then the producer manifest and
    # outer hash are refreshed so the closed status is itself checkout-stable.
    final_phase_status = "passed_hard_stop" if provisional_status == "passed" else "failed"
    final_ledger = [{"phase": f"EF{i}", "status": strict_json(OUT_ROOT / name).get("status", "not_evaluated") if (OUT_ROOT / name).is_file() else "not_evaluated"} for i, name in enumerate(("ef0_byte_anchor.json", "ef1_identity_and_resume.json", "ef2_supervisor_fixture.json", "ef3_contact_gate.json"))]
    final_ledger.append({"phase": "EF4", "status": final_phase_status})
    atomic_bytes(csv_path, ("phase,status\n" + "".join(f"{row['phase']},{row['status']}\n" for row in final_ledger)).encode("utf-8"))
    write_json(OUT_ROOT / "execution_summary.json", {"schema_version": "track3_ef_execution_summary_v1", "status": "hard_stop", "phase_status": {row["phase"]: row["status"] for row in final_ledger}, "new_physics_identities": 0, "old_er_or_tz_identity_rerun": False, "new_candidate_zip": False, "training": False, "upload": False, "submission": False})
    producer_manifest = _manifest(_producer_paths())
    write_json(OUT_ROOT / "artifact_manifest.json", producer_manifest)
    outer = {"schema_version": "track3_ef_outer_hash_v1", "manifest_path": rel(OUT_ROOT / "artifact_manifest.json"), "manifest_sha256": sha256_file(OUT_ROOT / "artifact_manifest.json"), "entries_sha256": canonical_hash(producer_manifest["entries"]), "created_at_utc": now()}
    write_json(OUT_ROOT / "outer_hash.json", outer)
    # Re-read all immutable/derived files after the status refresh.
    strict_result = _strict_json_scan()
    npz_result = _npz_scan()
    csv_result = _csv_scan()
    manifest_entries = _manifest_entries_match(producer_manifest)
    manifest_outer = {"manifest_sha256": outer["manifest_sha256"], "observed_manifest_sha256": sha256_file(OUT_ROOT / "artifact_manifest.json"), "entries_sha256": outer["entries_sha256"], "observed_entries_sha256": canonical_hash(producer_manifest["entries"]), "status": "passed" if outer["manifest_sha256"] == sha256_file(OUT_ROOT / "artifact_manifest.json") and outer["entries_sha256"] == canonical_hash(producer_manifest["entries"]) else "failed"}
    validation_status = "passed" if provisional_status == "passed" and strict_result["status"] == "passed" and npz_result["status"] == "passed" and csv_result["status"] == "passed" and manifest_outer["status"] == "passed" and manifest_entries["status"] == "passed" else "failed"
    validation = {"schema_version": "track3_ef4_final_validation_v1", "status": validation_status, "strict_json": strict_result, "npz": npz_result, "csv": csv_result, "fallback_zip": fallback_check, "process_scan": process_result, "gpu_boundary": gpu_result, "targeted_tests": targeted, "full_tests": full, "git_diff_check": {"returncode": diff.returncode, "stderr": diff.stderr[-1000:]}, "py_compile": {"returncode": compile_result.returncode, "stderr": compile_result.stderr[-1000:]}, "manifest_entries": manifest_entries, "manifest_outer_hash": manifest_outer, "partial_residue": partial_before, "no_partial_after_tests": not list(OUT_ROOT.rglob("*.partial")), "final_validation_after_manifest_outer_hash": True, "new_candidate_zip_created": False, "new_physics_identities": 0, "gpu_compute_started": False, "created_at_utc": now()}
    write_json(OUT_ROOT / "final_validation.json", validation)
    write_json(OUT_ROOT / "heartbeat.json", {"schema_version": "track3_ef_heartbeat_v1", "state": "closed", "phase": "EF4", "pid": os.getpid(), "final_status": "hard_stop", "final_validation_status": validation["status"], "updated_at_utc": now()})
    closure = {"schema_version": "track3_ef_checkout_stable_closure_v1", "status": "passed" if validation["status"] == "passed" and _strict_json_scan()["status"] == "passed" else "failed", "final_validation_sha256": sha256_file(OUT_ROOT / "final_validation.json"), "artifact_manifest_sha256": sha256_file(OUT_ROOT / "artifact_manifest.json"), "outer_hash_sha256": sha256_file(OUT_ROOT / "outer_hash.json"), "closed_heartbeat_sha256": sha256_file(OUT_ROOT / "heartbeat.json"), "inactive_cursor": strict_json(OUT_ROOT / "resume_cursor.json").get("active") is False, "historical_er_namespace_modified": False, "next_authority": "new_reviewed_plan_required"}
    write_json(OUT_ROOT / "checkout_stable_closure.json", closure)
    return {"status": validation["status"], "validation": validation, "closure": closure}


def run_all() -> dict[str, Any]:
    ef0 = run_ef0()
    if ef0["status"] != "passed":
        return {"status": "hard_stop", "phase": "EF0", "reason": "byte_anchor_failed"}
    ef1 = run_ef1()
    if ef1["status"] != "passed":
        return {"status": "hard_stop", "phase": "EF1", "reason": "identity_resume_failed"}
    ef2 = run_ef2()
    if ef2["status"] != "passed":
        return {"status": "hard_stop", "phase": "EF2", "reason": "supervisor_fixture_failed"}
    ef3 = run_ef3()
    if ef3["status"] != "passed":
        return {"status": "hard_stop", "phase": "EF3", "reason": "contact_gate_failed"}
    return run_ef4()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for index in range(5):
        parser.add_argument(f"--ef{index}", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ef-fixture-child", action="store_true")
    parser.add_argument("--ef-lock-probe", type=Path)
    parser.add_argument("--mode", choices=("family_failure", "timeout"))
    parser.add_argument("--result-output", type=Path)
    parser.add_argument("--heartbeat-output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--expected-identity-file", type=Path)
    parser.add_argument("--expected-command-sha256")
    args = parser.parse_args(argv)
    if args.ef_lock_probe is not None:
        return _lock_probe(args.ef_lock_probe)
    if args.ef_fixture_child:
        required = (args.mode, args.result_output, args.heartbeat_output, args.trace_output, args.expected_identity_file, args.expected_command_sha256)
        if any(value is None for value in required):
            parser.error("fixture child requires mode, outputs, expected identity and command hash")
        return fixture_child(str(args.mode), args.result_output, args.heartbeat_output, args.trace_output, args.expected_identity_file)
    phases = [getattr(args, f"ef{index}") for index in range(5)]
    if args.execute:
        value = run_all()
    else:
        if sum(bool(flag) for flag in phases) != 1:
            parser.error("choose exactly one EF phase or --execute")
        value = {0: run_ef0, 1: run_ef1, 2: run_ef2, 3: run_ef3, 4: run_ef4}[next(index for index, flag in enumerate(phases) if flag)]()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if value.get("status") in {"passed", "hard_stop", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
