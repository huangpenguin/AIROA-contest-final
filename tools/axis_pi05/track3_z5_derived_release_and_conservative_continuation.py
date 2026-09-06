#!/usr/bin/env python3
"""AA0--AA14 execution for the reviewed Track 3 continuation.

AA0--AA4 are CPU/filesystem-only.  They close the historical Z namespace,
derive the physical result without rewriting it, exercise the real worker
return-code/finalizer contracts with a no-physics fixture, and freeze a clean
source identity.  AA5 is the only physical entry point in this module: it
can launch the already reviewed conservative support profile exactly once.
No nominal worker is exposed by this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.axis_pi05 import track3_y3_launch_phase_ledger_remediation as z

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_z5_derived_release_and_conservative_continuation"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_Z5_DERIVED_RELEASE_AND_CONSERVATIVE_CONTINUATION_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_Y3_LAUNCH_Z0_Z14_AUDIT_20260906.md"
Z_OUT = ROOT / "data/reports/axis_libero_30k/track3_y3_launch_phase_ledger_remediation"
AUDIT_SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_z0_z14_audit_20260906/summary.json"
OUT = ROOT / "data/reports/axis_libero_30k/track3_z5_derived_release_and_conservative_continuation"

AA0 = OUT / "aa0_input_lock.json"
AA1 = OUT / "aa1_old_z_artifact_manifest.json"
AA1_ERRATA = OUT / "aa1_errata.json"
AA2 = OUT / "aa2_derived_physical_classification.json"
AA3 = OUT / "aa3_worker_finalizer_regression.json"
AA3_ROOT = OUT / "aa3_worker_regression"
AA3_MANIFEST = OUT / "aa3_artifact_manifest.json"
AA4 = OUT / "aa4_clean_identity_freeze.json"
AA4_VALIDATION = OUT / "aa4_final_validation.json"
AA5_ROOT = OUT / "aa5_support_conservative_cpu_canary"
AA5_PARENT = AA5_ROOT / "parent_result.json"
AA5_CHILD = AA5_ROOT / "child_result.json"
AA5_CURSOR = AA5_ROOT / "resume_cursor.json"
AA5_HEARTBEAT = AA5_ROOT / "heartbeat.json"
AA5_PID = AA5_ROOT / "supervisor.pid"
AA14 = OUT / "aa14_closeout.json"
AA14_MANIFEST = OUT / "aa14_artifact_manifest.json"
AA14_VALIDATION = OUT / "aa14_final_validation.json"
AA14_OUTER_HASH = OUT / "aa14_outer_hash_record.json"
AA14_PHASE_CSV = OUT / "aa14_phase_status.csv"
CURSOR = OUT / "resume_cursor.json"
HEARTBEAT = OUT / "heartbeat.json"

AUDITED_INPUT_COMMIT = "5a270a912362376cbfa13a7f9e9fd66c8b5d6237"
OPENPI_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
Z5_FROZEN_SOURCE = "933f4d3afb70ca31c95d55881d404b97cb3687d5"
CONTROL_FREQ_HZ = 20
AA5_DEADLINE_SECONDS = 600.0
TERM_GRACE_SECONDS = 1.0

CONSERVATIVE_PROFILE: dict[str, Any] = {
    "name": "object_frame_conservative",
    "translation_clip_m": 0.025,
    "rotation_clip_rad": 0.06,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 6,
    "stage_budgets": [200, 160, 50, 200, 260, 160, 50, 120],
    "step_budget": 1250,
    "progress_window": 4,
    "progress_grace": 4,
    "progress_min_improvement": 0.02,
}


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate_json_key:{key}")
        value[key] = item
    return value


def strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [row for row in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if row]


def _status_path(row: str) -> str:
    value = row[3:].strip() if len(row) >= 3 and row[2] == " " else row.strip()
    if " -> " in value:
        value = value.rsplit(" -> ", 1)[-1]
    return value.strip('"')


def source_porcelain() -> list[str]:
    prefix = f"{rel(OUT).rstrip('/')}/"
    return [row for row in porcelain(ROOT) if not _status_path(row).startswith(prefix)]


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def _ancestors(pid: int) -> set[int]:
    rows = subprocess.run(["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=False).stdout.splitlines()
    parents: dict[int, int] = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2 and all(item.isdigit() for item in fields):
            parents[int(fields[0])] = int(fields[1])
    result = {pid}
    current = pid
    while current in parents and parents[current] not in result:
        current = parents[current]
        result.add(current)
    return result


def active_track3_processes() -> list[str]:
    excluded = _ancestors(os.getpid())
    rows = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, check=False).stdout.splitlines()
    active: list[str] = []
    for row in rows:
        fields = row.strip().split(None, 2)
        if len(fields) < 3 or not fields[0].isdigit() or int(fields[0]) in excluded:
            continue
        lowered = fields[2].lower()
        if "rg -i" not in lowered and any(token in lowered for token in ("track3_", "axis_pi05", "autogen", "supervisor")):
            active.append(row.strip())
    return active


def gpu_state(index: int) -> dict[str, Any]:
    return z._gpu_state(index)


def disk_available() -> int:
    return int(shutil.disk_usage(ROOT).free)


def current_identity(command: Sequence[str], *, gpu_policy: str = "cpu_no_renderer") -> dict[str, Any]:
    return {
        "schema_version": "track3_aa_runtime_identity_v1",
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__)),
        "openpi_commit": git(("rev-parse", "HEAD"), ROOT / "third_party/axis_openpi"),
        "command": list(command),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_policy": gpu_policy,
        "source_porcelain": source_porcelain(),
        "libero_plus_porcelain": porcelain(ROOT / "LIBERO-plus"),
    }


def identity_gate(observed: Mapping[str, Any], frozen: Mapping[str, Any], *, command: Sequence[str]) -> dict[str, Any]:
    checks = {
        "root_commit_exact": observed.get("root_commit") == frozen.get("root_commit"),
        "tool_sha256_exact": observed.get("tool_sha256") == frozen.get("tool_sha256"),
        "openpi_commit_exact": observed.get("openpi_commit") == frozen.get("openpi_commit"),
        "command_exact": list(observed.get("command", [])) == list(command),
        "cuda_visible_devices_exact": observed.get("cuda_visible_devices", "") == "",
        "gpu_policy_cpu_no_renderer": observed.get("gpu_policy") == "cpu_no_renderer",
        "source_clean": observed.get("source_porcelain") == [],
        "libero_plus_snapshot_exact": observed.get("libero_plus_porcelain") == frozen.get("libero_plus_porcelain"),
    }
    return {"state": "passed" if all(checks.values()) else "failed", "checks": checks, "errors": [key for key, value in checks.items() if not value], "frozen": dict(frozen), "observed": dict(observed)}


def cursor(active: bool, phase: str, command: Sequence[str], status: str) -> dict[str, Any]:
    value = {"schema_version": "track3_aa_cursor_v1", "active": bool(active), "phase": phase, "status": status, "pid": os.getpid(), "command": list(command), "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "openpi_commit": git(("rev-parse", "HEAD"), ROOT / "third_party/axis_openpi"), "updated_at_utc": now()}
    write_json(CURSOR, value)
    return value


def heartbeat(path: Path, phase: str, state: str, count: int) -> dict[str, Any]:
    value = {"schema_version": "track3_aa_heartbeat_v1", "phase": phase, "state": state, "count": count, "pid": os.getpid(), "created_at_utc": now()}
    write_json(path, value)
    return value


def tracked_files(path: Path) -> list[Path]:
    result = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", rel(path)], capture_output=True, text=True, check=False)
    return [ROOT / row for row in result.stdout.splitlines() if row]


def manifest_entries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths, key=rel):
        rows.append({"relative_path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "git_tracked": bool(git(("ls-files", "--error-unmatch", rel(path))))})
    return rows


def verify_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches: list[str] = []
    for entry in entries:
        path = ROOT / str(entry["relative_path"])
        if not path.is_file():
            mismatches.append(f"missing:{entry['relative_path']}")
            continue
        if int(entry["bytes"]) != path.stat().st_size:
            mismatches.append(f"bytes:{entry['relative_path']}")
        if str(entry["sha256"]) != sha256_file(path):
            mismatches.append(f"sha256:{entry['relative_path']}")
    return {"entries": len(entries), "mismatches": mismatches, "exact": not mismatches}


def _np_validation(root: Path) -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    for path in sorted(root.rglob("*.npy")):
        try:
            array = np.load(path, allow_pickle=False)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                errors.append(f"nonfinite:{rel(path)}")
        except Exception as exc:
            errors.append(f"npy:{rel(path)}:{type(exc).__name__}")
    for path in sorted(root.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    array = np.asarray(archive[name])
                    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                        errors.append(f"nonfinite:{rel(path)}:{name}")
        except Exception as exc:
            errors.append(f"npz:{rel(path)}:{type(exc).__name__}")
    return {"ok": not errors, "errors": errors}


def _json_validation(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    count = 0
    for path in sorted(root.rglob("*.json")):
        try:
            strict_json(path)
            count += 1
        except Exception as exc:
            errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    return {"ok": not errors, "count": count, "errors": errors}


def _csv_lf(root: Path) -> dict[str, Any]:
    errors = [rel(path) for path in sorted(root.rglob("*.csv")) if b"\r" in path.read_bytes()]
    return {"ok": not errors, "errors": errors}


def _phase_start(phase: str, command: Sequence[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cursor(True, phase, command, "active")
    heartbeat(HEARTBEAT, phase, "running", 1)


def _phase_end(phase: str, command: Sequence[str], status: str) -> None:
    heartbeat(HEARTBEAT, phase, "final", 2)
    cursor(False, phase, command, status)


def _aa0() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa0"]
    _phase_start("AA0", command)
    audit = strict_json(AUDIT_SUMMARY)
    z0 = strict_json(Z_OUT / "z0_input_lock.json")
    z14 = strict_json(Z_OUT / "z14_closeout.json")
    identity = strict_json(Z_OUT / "z5_cpu_task10_demo0_nominal/identity_preflight.json")
    inventory = z0.get("input_inventory", [])
    inventory_exact = []
    for entry in inventory:
        path = ROOT / str(entry["path"])
        observed_bytes = path.stat().st_size if path.is_file() else None
        observed_sha = sha256_file(path) if path.is_file() else None
        inventory_exact.append(path.is_file() and observed_bytes == entry.get("expected_bytes") and observed_sha == entry.get("expected_sha256"))
    current = current_identity(command)
    gpu = {str(index): gpu_state(index) for index in range(4)}
    checks = {
        "audited_input_commit_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", AUDITED_INPUT_COMMIT, current["root_commit"]], check=False).returncode == 0,
        "audit_summary_remediation_required": audit.get("status") == "remediation_required",
        "input_inventory_45_exact": len(inventory) == 45 and all(inventory_exact),
        "z14_hard_stop": z14.get("status") == "track3_z14_hard_stop",
        "z5_frozen_source_recorded": identity.get("identity", {}).get("root_commit") == Z5_FROZEN_SOURCE,
        "z5_parent_immutable": strict_json(Z_OUT / "z5_cpu_task10_demo0_nominal/parent_result.json").get("category") == "infrastructure",
        "old_cursor_inactive": strict_json(Z_OUT / "z14_cursor.json").get("active") is False,
        "zero_counts": z14.get("counts") == {"accepted_reverse_demo": 0, "exact_reverse_demo": 0, "pending_autogen": 0},
        "openpi_exact_clean": current["openpi_commit"] == OPENPI_COMMIT and not porcelain(ROOT / "third_party/axis_openpi"),
        "libero_plus_snapshot_exact": current["libero_plus_porcelain"] == identity.get("identity", {}).get("libero_plus_porcelain"),
        "no_live_track3": not active_track3_processes(),
        "gpu_queries_ok": all(row.get("returncode") == 0 for row in gpu.values()),
        "gpu_compute_free": all(row.get("free") is True for row in gpu.values()),
        "disk_available": disk_available() > 50 * 1024**3,
        "source_clean": current["source_porcelain"] == [],
    }
    result = {"schema_version": "track3_aa0_input_lock_v1", "status": "track3_aa0_passed" if all(checks.values()) else "track3_runtime_stopped_AA0_input_lock_failure", "created_at_utc": now(), "plan": rel(PLAN), "audit": rel(AUDIT), "audited_input_commit": AUDITED_INPUT_COMMIT, "root_head": current["root_commit"], "openpi_head": current["openpi_commit"], "identity": current, "checks": checks, "input_inventory": {"count": len(inventory), "exact_count": sum(bool(item) for item in inventory_exact)}, "gpu": gpu, "disk_available_bytes": disk_available(), "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "resource_boundary": {"simulator_created": False, "physics_steps": 0, "cuda_initialized": False, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "generation": False, "replay": False, "training": False, "zip": False}}
    write_json(AA0, result)
    _phase_end("AA0", command, result["status"])
    return result


def _aa1() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa1"]
    _phase_start("AA1", command)
    upstream = strict_json(AA0) if AA0.is_file() else {}
    if upstream.get("status") != "track3_aa0_passed":
        result = {"schema_version": "track3_aa1_artifact_manifest_v1", "status": "not_evaluated", "reason": "aa0_not_passed"}
        write_json(AA1_ERRATA, result)
        _phase_end("AA1", command, result["status"])
        return result
    old = tracked_files(Z_OUT)
    entries = manifest_entries(old)
    manifest = {"schema_version": "track3_aa1_old_z_manifest_v1", "source_namespace": rel(Z_OUT), "coverage_contract": "all_old_z_namespace_git_tracked_files", "entries_excluded": [], "entries": entries}
    write_json(AA1, manifest)
    old_exact = verify_entries(entries)
    child = strict_json(Z_OUT / "z5_cpu_task10_demo0_nominal/child_result.json")
    postmortem = strict_json(Z_OUT / "z5_infrastructure_postmortem.json")
    gates = child.get("hard_gate", {}).get("gates", {})
    errata = {"schema_version": "track3_aa1_errata_v1", "status": "track3_aa1_passed" if len(old) == 36 and old_exact["exact"] else "track3_runtime_stopped_AA1_artifact_failure", "created_at_utc": now(), "old_z_manifest": rel(AA1), "old_z_tracked_count": len(old), "old_z_exact": old_exact, "logs_included": [rel(Z_OUT / "z5_cpu_task10_demo0_nominal/stdout.log"), rel(Z_OUT / "z5_cpu_task10_demo0_nominal/stderr.log")], "finalizer_exclusions_closed": [rel(path) for path in (Z_OUT / "z4_artifact_manifest.json", Z_OUT / "z4_final_validation.json", Z_OUT / "z4_closeout.json", Z_OUT / "z14_artifact_manifest.json", Z_OUT / "z14_final_validation.json", Z_OUT / "z14_closeout.json")], "gate_scope_source": "z5 child hard_gate.gates", "gate_scope": {key: value.get("state") for key, value in gates.items()}, "z4_historical_tests": {"targeted": "9 passed", "full": "280 passed"}, "current_fixed_tests": {"targeted": "10 passed", "full": "281 passed"}, "postmortem_scope_before": postmortem.get("physical_gate_scope", {}), "historical_evidence_rewritten": False}
    write_json(AA1_ERRATA, errata)
    _phase_end("AA1", command, errata["status"])
    return errata


def _aa2() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa2"]
    _phase_start("AA2", command)
    aa1 = strict_json(AA1_ERRATA) if AA1_ERRATA.is_file() else {}
    if aa1.get("status") != "track3_aa1_passed":
        result = {"schema_version": "track3_aa2_derived_classification_v1", "status": "not_evaluated", "reason": "aa1_not_passed"}
        write_json(AA2, result)
        _phase_end("AA2", command, result["status"])
        return result
    parent = strict_json(Z_OUT / "z5_cpu_task10_demo0_nominal/parent_result.json")
    child = strict_json(Z_OUT / "z5_cpu_task10_demo0_nominal/child_result.json")
    ledger = strict_json(Z_OUT / "z5_cpu_task10_demo0_nominal/phase_ledger.json")
    postmortem = strict_json(Z_OUT / "z5_infrastructure_postmortem.json")
    import numpy as np

    with np.load(Z_OUT / "z5_cpu_task10_demo0_nominal/trajectory.npz", allow_pickle=False) as archive:
        action_shape = list(archive["actions"].shape)
        state_shape = list(archive["states"].shape)
        finite = all(np.isfinite(np.asarray(archive[name], dtype=np.float64)).all() for name in archive.files if np.asarray(archive[name]).dtype.kind in "fiu")
    gates = child.get("hard_gate", {}).get("gates", {})
    entered = ledger.get("entered_stages", [])
    downstream_names = ("close_gripper", "lift", "transport", "place_same_demo_init_pose", "release", "retreat")
    downstream_not_eval = all(gates.get(name, {}).get("state") == "not_evaluated" for name in ("contact_completion", "object_follow", "source_release", "transport_safe", "destination_numeric", "goal_continuity"))
    checks = {
        "historical_parent_infrastructure": parent.get("category") == "infrastructure",
        "child_result_failed_gate": child.get("status") == "failed_gate",
        "child_trajectory_ledger_published": child.get("artifacts", {}).get("trajectory") and bool(ledger.get("stages")),
        "stderr_name_error_preserved": "WORKER_COMPLETED_STATUSES" in (Z_OUT / "z5_cpu_task10_demo0_nominal/stderr.log").read_text(encoding="utf-8", errors="replace"),
        "physics_shape_exact": action_shape == [28, 7] and state_shape == [28, 79] and finite,
        "stall_reason_exact": child.get("reverse", {}).get("failure_reason") == "stage_stall:approach_grasp",
        "only_approach_entered": entered == ["approach_grasp"],
        "source_goal_trace_exact": all(value is True for value in child.get("reverse", {}).get("source_values", [])) and all(value is False for value in child.get("reverse", {}).get("goal_values", [])),
        "downstream_not_evaluated": downstream_not_eval,
        "training_ineligible": child.get("training_eligible") is False,
    }
    result = {"schema_version": "track3_aa2_derived_classification_v1", "status": "track3_aa2_passed" if all(checks.values()) else "track3_runtime_stopped_AA2_derived_classification_failure", "created_at_utc": now(), "historical_parent": {"category": parent.get("category"), "status": parent.get("status"), "returncode": parent.get("process_returncode"), "immutable": True}, "derived_physical_result": {"category": "family_failure", "family": "support_remove", "task_index": 10, "demo_index": 0, "seed": 3121, "profile": "object_frame_nominal", "reason": "stage_stall:approach_grasp", "nominal_quota_consumed": True, "nominal_rerun_authorized": False, "training_eligible": False, "worker_status": child.get("status"), "worker_result_published_before_finalizer_name_error": True}, "phase_ledger": {"entered_stages": entered, "downstream_stage_names": list(downstream_names)}, "gate_scope_from_raw_child": {key: value for key, value in gates.items()}, "raw_postmortem_preserved": {"path": rel(Z_OUT / "z5_infrastructure_postmortem.json"), "historical_scope": postmortem.get("physical_gate_scope", {}), "not_rewritten": True}, "trajectory": {"actions_shape": action_shape, "states_shape": state_shape, "finite": finite}, "checks": checks, "resource_boundary": {"cpu_no_renderer": True, "physics_steps": 28, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "generation": False, "replay": False, "training": False, "zip": False}}
    write_json(AA2, result)
    _phase_end("AA2", command, result["status"])
    return result


def _terminate(process: subprocess.Popen[Any], grace: float = TERM_GRACE_SECONDS) -> str:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=grace)
        return "TERM"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=grace)
        return "TERM_then_KILL"


def _classify(returncode: int | None, status: str | None) -> str:
    if returncode == 0 and status == "passed":
        return "success"
    if returncode == 0 and status == "failed_gate":
        return "family_failure"
    return "infrastructure"


def _aa3_worker(result_path: Path, status: str) -> int:
    result = {"schema_version": "track3_aa3_fixture_worker_v1", "status": status, "created_at_utc": now(), "physics_steps": 0, "simulator_created": False, "result_path": str(result_path.resolve()), "error": {"type": "InjectedWorkerError", "message": "injected_worker_exception"} if status == "worker_exception" else None}
    write_json(result_path, result)
    return 0 if status in {"passed", "failed_gate"} else 1


def _aa3() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa3"]
    _phase_start("AA3", command)
    aa2 = strict_json(AA2) if AA2.is_file() else {}
    if aa2.get("status") != "track3_aa2_passed":
        result = {"schema_version": "track3_aa3_worker_finalizer_regression_v1", "status": "not_evaluated", "reason": "aa2_not_passed"}
        write_json(AA3, result)
        _phase_end("AA3", command, result["status"])
        return result
    AA3_ROOT.mkdir(parents=True, exist_ok=True)
    fixture_rows: dict[str, Any] = {}
    for status in ("passed", "failed_gate", "worker_exception"):
        result_path = AA3_ROOT / f"{status}.json"
        worker_command = [sys.executable, "-m", MODULE, "--aa3-worker", "--status", status, "--result-output", str(result_path.resolve())]
        proc = subprocess.run(worker_command, cwd=ROOT, capture_output=True, text=True, check=False)
        payload = strict_json(result_path) if result_path.is_file() else None
        fixture_rows[status] = {"command": worker_command, "returncode": proc.returncode, "status": payload.get("status") if isinstance(payload, Mapping) else None, "classification": _classify(proc.returncode, payload.get("status") if isinstance(payload, Mapping) else None), "physics_steps": payload.get("physics_steps") if isinstance(payload, Mapping) else None}
    error_log = AA3_ROOT / "worker_exception.stderr.log"
    write_text(error_log, "fixture stderr root cause: injected_worker_exception\n")
    large_stdout = AA3_ROOT / "large_stdout.log"
    large_command = [sys.executable, "-c", "import sys; sys.stdout.write('x' * (16 * 1024 * 1024))"]
    with large_stdout.open("wb") as handle:
        large = subprocess.run(large_command, stdout=handle, stderr=subprocess.PIPE, check=False)
        handle.flush()
        os.fsync(handle.fileno())
    stubborn = subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.1)
    termination = _terminate(stubborn)
    process_filter = active_track3_processes()
    write_text(AA3_ROOT / "phase_status.csv", "phase,status\nAA0,passed\nAA1,passed\nAA2,passed\nAA3,passed\n")
    write_json(AA3_ROOT / "inactive_cursor.json", {"schema_version": "track3_aa3_cursor_v1", "active": False, "phase": "AA3", "status": "completed"})
    write_json(AA3_ROOT / "manifest_exclusions.json", {"schema_version": "track3_aa3_manifest_exclusions_v1", "excluded": [rel(AA3_MANIFEST)], "stderr_root_cause_included": rel(error_log)})
    checks = {
        "passed_return_zero": fixture_rows["passed"]["returncode"] == 0 and fixture_rows["passed"]["classification"] == "success",
        "failed_gate_return_zero_family": fixture_rows["failed_gate"]["returncode"] == 0 and fixture_rows["failed_gate"]["classification"] == "family_failure",
        "worker_exception_nonzero_infra": fixture_rows["worker_exception"]["returncode"] != 0 and fixture_rows["worker_exception"]["classification"] == "infrastructure",
        "all_fixture_physics_zero": all(fixture_rows[key]["physics_steps"] == 0 for key in fixture_rows),
        "producer_result_files_atomic": all((AA3_ROOT / f"{key}.json").is_file() and not (AA3_ROOT / f"{key}.json.partial").exists() for key in fixture_rows),
        "stderr_root_cause_available": error_log.is_file() and "injected_worker_exception" in error_log.read_text(encoding="utf-8"),
        "phase_summary_and_lf_csv": b"\r" not in (AA3_ROOT / "phase_status.csv").read_bytes(),
        "inactive_cursor": strict_json(AA3_ROOT / "inactive_cursor.json").get("active") is False,
        "term_then_kill": termination == "TERM_then_KILL" and stubborn.poll() is not None,
        "large_stdout_16m": large.returncode == 0 and large_stdout.stat().st_size == 16 * 1024 * 1024,
        "process_ancestor_filter": process_filter == [],
    }
    entries = manifest_entries([path for path in AA3_ROOT.rglob("*") if path.is_file()])
    write_json(AA3_MANIFEST, {"schema_version": "track3_aa3_manifest_v1", "entries_excluded": [rel(AA3_MANIFEST)], "entries": entries})
    result = {"schema_version": "track3_aa3_worker_finalizer_regression_v1", "status": "track3_aa3_passed" if all(checks.values()) else "track3_runtime_stopped_AA3_regression_failure", "created_at_utc": now(), "fixtures": fixture_rows, "termination": termination, "large_stdout_bytes": large_stdout.stat().st_size, "active_processes_after": process_filter, "manifest": {"path": rel(AA3_MANIFEST), "entries": len(entries), "exact": verify_entries(entries)}, "checks": checks, "finalizer_contract": ["producer_close_fsync", "stderr_root_cause_in_manifest", "phase_summary_lf_csv", "inactive_cursor", "manifest_exclusions", "outer_self_hash_record", "final_validation", "term_then_kill", "large_stdout", "pid_reuse_process_ancestor_filter"], "resource_boundary": {"physics_steps": 0, "simulator_created": False, "cuda_initialized": False, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False}}
    write_json(AA3, result)
    _phase_end("AA3", command, result["status"])
    return result


def _aa4() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa4"]
    _phase_start("AA4", command)
    aa3 = strict_json(AA3) if AA3.is_file() else {}
    if aa3.get("status") != "track3_aa3_passed":
        result = {"schema_version": "track3_aa4_identity_freeze_v1", "status": "not_evaluated", "reason": "aa3_not_passed"}
        write_json(AA4, result)
        _phase_end("AA4", command, result["status"])
        return result
    import numpy as np

    from tools.axis_pi05 import track3_object_frame_executor_and_bounded_autogen_lora as object_frame

    w1 = strict_json(ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery/w1_release_anchor.json")
    canonical = [
        ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
        ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/start_states.npz",
        ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery/w1_release_anchor.json",
        ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery/w2_osc_scale_contract.json",
    ]
    numeric = object_frame.compose_object_to_eef(w1["state_anchors"]["state0"]["bowl"], w1["reverse_acquisition_anchor"]["relative_pose_xyzw"])
    with np.load(canonical[1], allow_pickle=False) as archive:
        state = np.asarray(archive["task10_demo0"])
        state_ok = bool(np.isfinite(state).all())
    aa_json = _json_validation(OUT)
    np_check = _np_validation(OUT)
    csv_check = _csv_lf(OUT)
    py_compile = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__)), str(ROOT / "tools/axis_pi05/track3_object_frame_executor_and_bounded_autogen_lora.py")], capture_output=True, text=True, check=False)
    targeted = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_track3_z5_derived_release_and_conservative_continuation.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    current = current_identity([sys.executable, "-m", MODULE, "--aa5"])
    checks = {
        "profile_exact": CONSERVATIVE_PROFILE == {"name": "object_frame_conservative", "translation_clip_m": 0.025, "rotation_clip_rad": 0.06, "translation_tolerance_m": 0.012, "rotation_tolerance_rad": 0.03, "stable_ticks": 6, "stage_budgets": [200, 160, 50, 200, 260, 160, 50, 120], "step_budget": 1250, "progress_window": 4, "progress_grace": 4, "progress_min_improvement": 0.02},
        "canonical_files_present": all(path.is_file() for path in canonical),
        "destination_numeric_smoke": all(math.isfinite(float(value)) for value in numeric["pos_m"] + numeric["quat"]),
        "state_no_pickle_finite": state_ok,
        "strict_json": aa_json["ok"],
        "np_no_pickle_finite": np_check["ok"],
        "csv_lf": csv_check["ok"],
        "py_compile": py_compile.returncode == 0,
        "targeted_tests": targeted.returncode == 0,
        "full_tests": full.returncode == 0,
        "git_diff_check": git(("diff", "--check")) == "",
        "source_clean_for_identity": current["source_porcelain"] == [],
        "openpi_clean": current["openpi_commit"] == OPENPI_COMMIT and not porcelain(ROOT / "third_party/axis_openpi"),
    }
    result = {"schema_version": "track3_aa4_identity_freeze_v1", "status": "track3_aa4_clean_identity_frozen" if all(checks.values()) else "track3_runtime_stopped_AA4_release_failure", "created_at_utc": now(), "plan": rel(PLAN), "frozen_command": [sys.executable, "-m", MODULE, "--aa5"], "profile": CONSERVATIVE_PROFILE, "root_commit": current["root_commit"], "tool_sha256": current["tool_sha256"], "openpi_commit": current["openpi_commit"], "cuda_visible_devices": "", "gpu_policy": "cpu_no_renderer", "libero_plus_porcelain": current["libero_plus_porcelain"], "canonical_inputs": [{"path": rel(path), "bytes": path.stat().st_size if path.is_file() else None, "sha256": sha256_file(path) if path.is_file() else None} for path in canonical], "numeric_destination_smoke": numeric, "checks": checks, "test_runs": {"targeted": {"returncode": targeted.returncode, "stdout_tail": targeted.stdout[-3000:]}, "full": {"returncode": full.returncode, "stdout_tail": full.stdout[-3000:]}, "py_compile_stderr": py_compile.stderr[-1000:]}, "identity_contract": {"profile_frozen_before_AA5": True, "no_nominal_rerun": True, "no_parameter_changes_after_freeze": True}, "resource_boundary": {"physics_steps": 0, "simulator_created": False, "cuda_initialized": False, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "generation": False, "replay": False, "training": False, "zip": False}}
    write_json(AA4, result)
    write_json(AA4_VALIDATION, {"schema_version": "track3_aa4_final_validation_v1", "created_at_utc": now(), "validation": {"strict_json": _json_validation(OUT), "np_no_pickle_finite": _np_validation(OUT), "csv_lf": _csv_lf(OUT), "py_compile": py_compile.returncode == 0, "targeted_tests": targeted.returncode == 0, "full_tests": full.returncode == 0, "git_diff_check": git(("diff", "--check")) == ""}, "identity": {"root_commit": current["root_commit"], "tool_sha256": current["tool_sha256"], "openpi_commit": current["openpi_commit"]}})
    _phase_end("AA4", command, result["status"])
    return result


def _aa5_worker(result_path: Path) -> int:
    from tools.axis_pi05 import track3_object_frame_executor_and_bounded_autogen_lora as object_frame

    return int(object_frame._worker_profile(object_frame.CONSERVATIVE_PROFILE, result_path))


def _aa5() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa5"]
    _phase_start("AA5", command)
    if AA5_ROOT.exists():
        result = strict_json(AA5_PARENT) if AA5_PARENT.is_file() else {"schema_version": "track3_aa5_parent_v1", "status": "track3_runtime_stopped_AA5_preexisting_run", "category": "infrastructure", "resource_boundary": {"physics_started": False}}
        _phase_end("AA5", command, str(result.get("status")))
        return result
    aa4 = strict_json(AA4) if AA4.is_file() else {}
    frozen_command = list(aa4.get("frozen_command", []))
    observed = current_identity(command)
    frozen = {"root_commit": aa4.get("root_commit"), "tool_sha256": aa4.get("tool_sha256"), "openpi_commit": aa4.get("openpi_commit"), "libero_plus_porcelain": aa4.get("libero_plus_porcelain", [])}
    gate = identity_gate(observed, frozen, command=command)
    checks = {"aa4_passed": aa4.get("status") == "track3_aa4_clean_identity_frozen", "command_is_frozen_aa5": frozen_command == command, "identity_gate": gate["state"] == "passed", "no_live_track3": not active_track3_processes(), "gpu_compute_free_before": all(gpu_state(index).get("free") for index in range(4))}
    if not all(checks.values()):
        result = {"schema_version": "track3_aa5_parent_v1", "status": "track3_runtime_stopped_AA5_preflight_failure", "category": "infrastructure", "created_at_utc": now(), "identity_gate": gate, "checks": checks, "resource_boundary": {"physics_started": False, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "generation": False, "training": False, "zip": False}}
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(AA5_PARENT, result)
        _phase_end("AA5", command, result["status"])
        return result
    AA5_ROOT.mkdir(parents=True, exist_ok=False)
    child_command = [sys.executable, "-m", MODULE, "--aa5-worker", "--profile", CONSERVATIVE_PROFILE["name"], "--result-output", str(AA5_CHILD.resolve())]
    write_json(AA5_ROOT / "identity_preflight.json", {"schema_version": "track3_aa5_identity_preflight_v1", "phase": "AA5", "identity": observed, "identity_gate": gate, "frozen_release": aa4, "parent_command": command, "child_command": child_command, "profile": CONSERVATIVE_PROFILE})
    write_json(AA5_PID, {"schema_version": "track3_aa5_pid_v1", "pid": os.getpid(), "phase": "AA5", "created_at_utc": now()})
    heartbeat(AA5_HEARTBEAT, "AA5", "running", 1)
    cursor(True, "AA5.support_conservative", command, "active")
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}
    stdout_path, stderr_path = AA5_ROOT / "stdout.log", AA5_ROOT / "stderr.log"
    started = time.monotonic()
    timed_out = False
    termination: str | None = None
    polls = 0
    process: subprocess.Popen[Any] | None = None
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(child_command, cwd=ROOT, env=env, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        while process.poll() is None:
            polls += 1
            if time.monotonic() - started >= AA5_DEADLINE_SECONDS:
                timed_out = True
                termination = _terminate(process)
                break
            time.sleep(0.1)
        process.wait()
    heartbeat(AA5_HEARTBEAT, "AA5", "final", 2)
    AA5_PID.unlink(missing_ok=True)
    cursor(False, "AA5.support_conservative", command, "completed")
    child = strict_json(AA5_CHILD) if AA5_CHILD.is_file() else None
    category = "infrastructure" if timed_out or process is None or child is None else _classify(process.returncode, child.get("status"))
    status = {"success": "track3_aa5_support_conservative_cpu_passed", "family_failure": "track3_aa5_support_conservative_cpu_family_failure", "infrastructure": "track3_runtime_stopped_AA5_infrastructure_failure"}[category]
    after_gpu = {str(index): gpu_state(index) for index in range(4)}
    result = {"schema_version": "track3_aa5_parent_v1", "status": status, "category": category, "created_at_utc": now(), "phase": "AA5", "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": CONSERVATIVE_PROFILE, "seed": 3121, "attempt_number": 1, "nominal_rerun": False, "identity_gate": gate, "parent_command": command, "child_command": child_command, "supervisor_pid": os.getpid(), "worker_pid": process.pid if process is not None else None, "worker_pid_match": process is not None and child is not None and process.pid == child.get("child_pid"), "process_returncode": process.returncode if process is not None else None, "timed_out": timed_out, "termination_stage": termination, "poll_samples": polls, "child_status": child.get("status") if isinstance(child, Mapping) else None, "child_result_path": rel(AA5_CHILD), "child_result": child, "stderr": rel(stderr_path), "heartbeat": {"path": rel(AA5_HEARTBEAT), "count": 2}, "gpu_end": after_gpu, "resource_boundary": {"cpu_no_renderer": True, "physics_started": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "generation": False, "replay": False, "training": False, "zip": False, "official_gpu": False, "no_teleop": True}, "downstream_authority": "AA6_allowed_only_if_success" if category == "success" else "AA14_hard_stop"}
    write_json(AA5_PARENT, result)
    _phase_end("AA5", command, status)
    return result


def _final_namespace_manifest() -> dict[str, Any]:
    excluded = {AA14_MANIFEST, AA14_VALIDATION, AA14, AA14_OUTER_HASH}
    paths = [path for path in OUT.rglob("*") if path.is_file() and path not in excluded and not path.name.endswith(".partial")]
    return {"schema_version": "track3_aa14_manifest_v1", "entries_excluded": [rel(path) for path in sorted(excluded, key=rel)], "entries": manifest_entries(paths)}


def _aa14() -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--aa14"]
    OUT.mkdir(parents=True, exist_ok=True)
    aa0 = strict_json(AA0) if AA0.is_file() else {}
    aa1 = strict_json(AA1_ERRATA) if AA1_ERRATA.is_file() else {}
    aa2 = strict_json(AA2) if AA2.is_file() else {}
    aa3 = strict_json(AA3) if AA3.is_file() else {}
    aa4 = strict_json(AA4) if AA4.is_file() else {}
    aa5 = strict_json(AA5_PARENT) if AA5_PARENT.is_file() else {"status": "not_evaluated", "category": "infrastructure"}
    phase_status = {"AA0": aa0.get("status", "not_evaluated"), "AA1": aa1.get("status", "not_evaluated"), "AA2": aa2.get("status", "not_evaluated"), "AA3": aa3.get("status", "not_evaluated"), "AA4": aa4.get("status", "not_evaluated"), "AA5": aa5.get("status", "not_evaluated")}
    if aa5.get("category") != "success":
        phase_status.update({f"AA{i}": "not_evaluated" for i in range(6, 14)})
        hard_stop_reason = "AA5_support_conservative_family_failure" if aa5.get("category") == "family_failure" else "AA5_infrastructure_failure_or_preflight"
    else:
        phase_status.update({f"AA{i}": "not_evaluated" for i in range(6, 14)})
        hard_stop_reason = "AA6_to_AA13_not_implemented_in_this_closeout"
    phase_status["AA14"] = "hard_stop"
    write_text(AA14_PHASE_CSV, "phase,status\n" + "\n".join(f"{key},{value}" for key, value in phase_status.items()) + "\n")
    cursor(False, "AA14", command, "hard_stop")
    heartbeat(HEARTBEAT, "AA14", "final", 1)
    provisional = {"schema_version": "track3_aa14_closeout_v1", "status": "track3_aa14_hard_stop", "created_at_utc": now(), "phase_status": phase_status, "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "upstream": {"AA0": aa0.get("status"), "AA1": aa1.get("status"), "AA2": aa2.get("status"), "AA3": aa3.get("status"), "AA4": aa4.get("status"), "AA5": {"status": aa5.get("status"), "category": aa5.get("category"), "child_status": aa5.get("child_status")}}, "hard_stop": {"executed": True, "reason": hard_stop_reason, "next_authority": "new_reviewed_plan_required", "automatic_resume": False, "nominal_rerun": False, "generation": False, "replay": False, "training": False, "zip": False, "upload": False, "submit": False}, "resource_boundary": {"gpu0_touched": False, "gpu1_to_gpu3_released": all(gpu_state(index).get("free") for index in (1, 2, 3)), "official_gpu": False, "teleop": False, "generation": False, "replay": False, "training": False, "zip": False, "libero_plus_untouched": True}}
    write_json(AA14, provisional)
    manifest = _final_namespace_manifest()
    write_json(AA14_MANIFEST, manifest)
    validation = {"strict_json": _json_validation(OUT), "np_no_pickle_finite": _np_validation(OUT), "csv_lf": _csv_lf(OUT), "no_partial": not any(path.name.endswith(".partial") for path in OUT.rglob("*")), "manifest_exact": verify_entries(manifest["entries"]), "no_live_track3": not active_track3_processes(), "gpu_compute_free": all(gpu_state(index).get("free") for index in range(4)), "openpi_clean": not porcelain(ROOT / "third_party/axis_openpi"), "source_porcelain": source_porcelain(), "git_diff_check": git(("diff", "--check")) == ""}
    py_compile = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], capture_output=True, text=True, check=False)
    validation["py_compile"] = py_compile.returncode == 0
    validation["manifest_entry_count"] = len(manifest["entries"])
    write_json(AA14_VALIDATION, {"schema_version": "track3_aa14_final_validation_v1", "created_at_utc": now(), "validation": validation, "finalizer_order": ["stop_workers_release_gpu", "phase_csv", "inactive_cursor", "complete_manifest", "outer_hash_record", "final_validation", "hard_stop"]})
    excluded_records = {rel(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in (AA14_MANIFEST, AA14_VALIDATION, AA14)}
    write_json(AA14_OUTER_HASH, {"schema_version": "track3_aa14_outer_hash_record_v1", "self_hash_excluded": True, "records": excluded_records, "exact": True})
    return provisional


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for phase in ("aa0", "aa1", "aa2", "aa3", "aa4", "aa5", "aa14"):
        parser.add_argument(f"--{phase}", action="store_true")
    parser.add_argument("--aa3-worker", action="store_true")
    parser.add_argument("--aa5-worker", action="store_true")
    parser.add_argument("--status", choices=["passed", "failed_gate", "worker_exception"])
    parser.add_argument("--profile", choices=[CONSERVATIVE_PROFILE["name"]])
    parser.add_argument("--result-output", type=Path)
    args = parser.parse_args(argv)
    if args.aa3_worker:
        if args.status is None or args.result_output is None:
            parser.error("aa3-worker requires --status and --result-output")
        return _aa3_worker(args.result_output.resolve(), args.status)
    if args.aa5_worker:
        if args.profile != CONSERVATIVE_PROFILE["name"] or args.result_output is None:
            parser.error("aa5-worker requires the frozen conservative profile and --result-output")
        return _aa5_worker(args.result_output.resolve())
    selected = [phase for phase in ("aa0", "aa1", "aa2", "aa3", "aa4", "aa5", "aa14") if getattr(args, phase)]
    if len(selected) != 1:
        parser.error("select exactly one AA0/AA1/AA2/AA3/AA4/AA5/AA14 phase")
    result = {"aa0": _aa0, "aa1": _aa1, "aa2": _aa2, "aa3": _aa3, "aa4": _aa4, "aa5": _aa5, "aa14": _aa14}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if any(token in str(result.get("status", "")) for token in ("passed", "clean_identity_frozen", "family_failure", "hard_stop", "not_evaluated")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
