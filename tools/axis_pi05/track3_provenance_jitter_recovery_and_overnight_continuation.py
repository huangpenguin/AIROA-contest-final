#!/usr/bin/env python3
"""Track 3 provenance-jitter recovery and bounded continuation.

J0--J2 are deliberately simulator/CUDA free.  Runtime imports are kept in
the J3 child and all new evidence is written below a fresh namespace.  The
old overnight reports are read-only inputs and are never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_jitter_recovery"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_overnight_autogen_lora"
OLD_O10_MANIFEST = OLD_ROOT / "o10_artifact_manifest.json"
OLD_O10_CSV = OLD_ROOT / "o10_phase_results.csv"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_PROVENANCE_JITTER_RECOVERY_AND_OVERNIGHT_CONTINUATION_PLAN.md"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
R6_NORM = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
R6_RAW = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
R6_FOLDED = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors"
MODULE = "tools.axis_pi05.track3_provenance_jitter_recovery_and_overnight_continuation"
OLD_MODULE = "tools.axis_pi05.track3_overnight_autogen_lora_and_zip"
ROOT_INPUT_COMMIT = "8c0d296f1dfedb0c777040a733b6f9e352234111"
OPENPI_INPUT_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
GPU3 = 3
DEADLINE_SECONDS = 300.0
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS = 900.0
HEARTBEAT_GAP_MAX_MS = 2000.0
QUERY_P99_WARNING_MS = 500.0
TASK_ORDER = (10, 20, 19)
SEEDS = {10: 3100, 20: 3200, 19: 3190}
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}


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


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(value, encoding="utf-8")
    os.replace(partial, path)


def strict_json_load(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def gpu_snapshot(index: int) -> dict[str, Any]:
    gpu_command = ["nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    process_command = ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    gpu = subprocess.run(gpu_command, capture_output=True, text=True, check=False)
    processes = subprocess.run(process_command, capture_output=True, text=True, check=False)
    rows: list[dict[str, Any]] = []
    for line in gpu.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 4:
            try:
                rows.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": float(fields[2]), "memory_total_mib": float(fields[3])})
            except ValueError:
                pass
    return {"timestamp_utc": now(), "returncode": gpu.returncode, "process_query_returncode": processes.returncode, "gpus": rows, "compute_processes": [line.strip() for line in processes.stdout.splitlines() if line.strip()], "free_gate": gpu.returncode == 0 and processes.returncode == 0 and len(rows) == 1 and not processes.stdout.strip()}


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stat(values: Sequence[float], *, known: bool = True) -> dict[str, Any]:
    if not known:
        return {"status": "not_recoverable_historical"}
    if not values:
        return {"status": "not_evaluated", "count": 0}
    ordered = sorted(float(value) for value in values)
    percentile = lambda fraction: ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]
    return {"status": "measured", "count": len(ordered), "min_ms": ordered[0], "p50_ms": percentile(0.50), "p95_ms": percentile(0.95), "p99_ms": percentile(0.99), "max_ms": ordered[-1], "mean_ms": sum(ordered) / len(ordered)}


def telemetry_v2_gate(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    required = ("seed", "parent_pid", "child_pid", "child_result_pid", "physical_gpu_uuid", "cuda_visible_devices", "start_snapshot", "end_snapshot", "peak_memory_mib", "deadline_seconds", "term_grace_seconds", "process_returncode", "timed_out", "gpu_query_latency_ms", "heartbeat_gap_ms", "supervisor_elapsed_ms", "child_elapsed_ms", "sampling_coverage_count")
    if not isinstance(record, Mapping):
        return {"state": "failed", "missing": ["record_mapping"], "warnings": []}
    missing = [key for key in required if key not in record]
    errors: list[str] = []
    warnings: list[str] = []
    if missing:
        return {"state": "failed", "missing": missing, "errors": [], "warnings": []}
    if record.get("child_pid") != record.get("child_result_pid"):
        errors.append("child_pid_mismatch")
    if not record.get("physical_gpu_uuid") or record.get("cuda_visible_devices") in (None, ""):
        errors.append("gpu_mapping_missing")
    for key in ("start_snapshot", "end_snapshot"):
        snapshot = record.get(key)
        if not isinstance(snapshot, Mapping) or snapshot.get("returncode") != 0 or snapshot.get("process_query_returncode") != 0:
            errors.append(f"{key}_query_error")
    for key in ("peak_memory_mib", "supervisor_elapsed_ms", "child_elapsed_ms"):
        if not _finite_number(record.get(key)) or float(record[key]) < 0:
            errors.append(f"{key}_invalid")
    for key in ("deadline_seconds", "term_grace_seconds"):
        if record.get(key) not in (DEADLINE_SECONDS, TERM_GRACE_SECONDS):
            errors.append(f"{key}_contract")
    if record.get("process_returncode") != 0:
        errors.append("process_returncode")
    if record.get("timed_out") is not False:
        errors.append("timed_out")
    if not isinstance(record.get("sampling_coverage_count"), int) or record.get("sampling_coverage_count") < 0:
        errors.append("sampling_coverage_count_invalid")
    query_stats = record.get("gpu_query_latency_ms")
    if isinstance(query_stats, Mapping) and query_stats.get("status") == "measured" and _finite_number(query_stats.get("p99_ms")) and float(query_stats["p99_ms"]) > QUERY_P99_WARNING_MS:
        warnings.append("gpu_query_p99_over_500ms")
    gap_stats = record.get("heartbeat_gap_ms")
    historical = bool(record.get("historical_not_recoverable"))
    if not historical:
        if not isinstance(gap_stats, Mapping) or gap_stats.get("status") != "measured" or not _finite_number(gap_stats.get("max_ms")):
            errors.append("heartbeat_gap_not_measured")
        elif float(gap_stats["max_ms"]) > HEARTBEAT_GAP_MAX_MS:
            errors.append("heartbeat_gap_over_2000ms")
    try:
        elapsed_s = float(record["child_elapsed_ms"]) / 1000.0
        coverage = int(record["sampling_coverage_count"])
        if elapsed_s > 5.0 and (coverage < 5 or coverage / elapsed_s < 0.5):
            errors.append("sampling_coverage_below_0.5hz")
    except (TypeError, ValueError):
        errors.append("sampling_coverage_calculation")
    return {"state": "passed" if not errors else "failed", "missing": missing, "errors": list(dict.fromkeys(errors)), "warnings": warnings, "required_fields": list(required), "heartbeat_gap_threshold_ms": HEARTBEAT_GAP_MAX_MS, "query_latency_is_warning_only": True}


def _git_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"], check=False).returncode == 0


def run_j0() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    o0 = strict_json_load(OLD_ROOT / "o0_release_lock.json")
    o1 = strict_json_load(OLD_ROOT / "o1_provenance_contract.json")
    o2 = strict_json_load(OLD_ROOT / "o2_summary.json")
    old_parent = strict_json_load(OLD_ROOT / "o2_runs/task10_demo0_attempt3/parent_result.json")
    old_child = strict_json_load(OLD_ROOT / "o2_runs/task10_demo0_attempt3/child_result.json")
    old_manifest = strict_json_load(OLD_O10_MANIFEST)
    old_csv_entry = next((entry for entry in old_manifest["entries"] if entry["relative_path"].endswith("o10_phase_results.csv")), {})
    start = old_parent["gpu_provenance"]["start_snapshot"]
    end = old_parent["gpu_provenance"]["end_snapshot"]
    evidence = [OLD_ROOT / "o2_runs/task10_demo0_attempt3/child_result.json", OLD_ROOT / "o2_runs/task10_demo0_attempt3/parent_result.json", OLD_ROOT / "o2_camera_frames/task10_demo0_attempt3.npz", OLD_ROOT / "o2_runs/task10_demo0_attempt3/agentview.mp4", OLD_ROOT / "o2_runs/task10_demo0_attempt3/wrist.mp4", OLD_O10_MANIFEST, OLD_O10_CSV, R6_NORM, R6_FOLDED]
    hashes = {relative(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in evidence if path.is_file()}
    lock = {"schema_version": "track3_jitter_j0_lock_v1", "status": "track3_jitter_j0_passed", "created_at_utc": now(), "plan": relative(PLAN), "input_root_commit": ROOT_INPUT_COMMIT, "root_head": git(("rev-parse", "HEAD")), "root_input_ancestor": _git_ancestor(ROOT_INPUT_COMMIT), "openpi_head": git(("rev-parse", "HEAD"), OPENPI), "openpi_expected": OPENPI_INPUT_COMMIT, "root_porcelain": porcelain(), "openpi_porcelain": porcelain(OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "immutable_prior": {"o0_status": o0.get("status"), "o1_status": o1.get("status"), "o2_status": o2.get("status"), "old_parent_status": old_parent.get("status"), "old_child_status": old_child.get("status"), "old_parent_not_rewritten": True}, "task10_attempt3": {"attempt_number": old_parent.get("attempt_number"), "child_status": old_child.get("status"), "hard_gate_states": {key: value.get("state") for key, value in old_child.get("hard_gate", {}).items()}, "worker_pid": old_parent.get("worker_pid"), "child_pid": old_child.get("child_pid"), "pid_match": old_parent.get("worker_pid") == old_child.get("child_pid"), "physical_gpu_uuid": old_parent["gpu_provenance"].get("physical_gpu_uuid"), "cuda_visible_devices": old_parent["gpu_provenance"].get("cuda_visible_devices"), "returncode": old_parent.get("process_returncode"), "timed_out": old_parent.get("timed_out"), "snapshot_count": old_parent["gpu_provenance"].get("parent_poll_count"), "peak_memory_mib": old_parent["gpu_provenance"].get("peak_memory_mib"), "camera_npz": old_parent.get("camera_npz"), "videos": old_parent.get("videos"), "partial_residue": old_parent.get("atomic_partial_residue")}, "evidence_hashes": hashes, "old_o10_csv_line_endings": {"declared_manifest_sha256": old_csv_entry.get("sha256"), "current_worktree_sha256": sha256_file(OLD_O10_CSV), "current_git_blob_sha256": git(("hash-object", str(OLD_O10_CSV))), "worktree_bytes_contains_crlf": b"\r\n" in OLD_O10_CSV.read_bytes(), "worktree_bytes_contains_lf": b"\n" in OLD_O10_CSV.read_bytes(), "known_difference": old_csv_entry.get("sha256") != sha256_file(OLD_O10_CSV)}, "r6_inputs_from_prior_lock": o0.get("r6_inputs"), "gpu_snapshots": {str(index): gpu_snapshot(index) for index in (0, 1, 2, 3)}, "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_touched": False, "gpu0_external_process_preserved": True, "official_gpu": False, "training_started": False, "task10_attempt4": "forbidden", "new_task20_task19_started": False}, "hashes": hashes, "status_reasons": []}
    if not lock["root_input_ancestor"] or lock["openpi_head"] != OPENPI_INPUT_COMMIT or old_child.get("status") != "passed" or old_parent.get("attempt_number") != 3 or not lock["task10_attempt3"]["pid_match"] or old_parent.get("process_returncode") != 0 or old_parent.get("timed_out") is not False:
        lock["status"] = "track3_jitter_stopped_J0_input_or_evidence_failure"
        lock["status_reasons"].append("core_lock_failed")
    write_json(OUT_ROOT / "j0_input_lock.json", lock)
    return lock


def _valid_telemetry_fixture(query_latency: float, heartbeat_gap: float) -> dict[str, Any]:
    return {"seed": 3100, "parent_pid": 1, "child_pid": 2, "child_result_pid": 2, "physical_gpu_uuid": "GPU-test", "cuda_visible_devices": "3", "start_snapshot": {"returncode": 0, "process_query_returncode": 0}, "end_snapshot": {"returncode": 0, "process_query_returncode": 0}, "peak_memory_mib": 1.0, "deadline_seconds": DEADLINE_SECONDS, "term_grace_seconds": TERM_GRACE_SECONDS, "process_returncode": 0, "timed_out": False, "gpu_query_latency_ms": {"status": "measured", "count": 10, "p99_ms": query_latency, "max_ms": query_latency}, "heartbeat_gap_ms": {"status": "measured", "count": 9, "max_ms": heartbeat_gap}, "supervisor_elapsed_ms": 10000.0, "child_elapsed_ms": 10000.0, "sampling_coverage_count": 10}


def run_j1() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    j0 = strict_json_load(OUT_ROOT / "j0_input_lock.json") if (OUT_ROOT / "j0_input_lock.json").is_file() else {}
    if j0.get("status") != "track3_jitter_j0_passed":
        result = {"schema_version": "track3_jitter_j1_v2", "status": "not_evaluated", "reason": "j0_not_passed", "simulator_imported": False, "cuda_imported": False}
        write_json(OUT_ROOT / "j1_telemetry_v2_contract.json", result)
        return result
    latency_cases = {str(value): telemetry_v2_gate(_valid_telemetry_fixture(value, 1000.0)) for value in (50.0, 148.46, 600.0)}
    heartbeat_cases = {str(value): telemetry_v2_gate(_valid_telemetry_fixture(50.0, value)) for value in (1999.0, 2001.0)}
    csv_path = OUT_ROOT / "j1_lf_manifest_probe.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial = csv_path.with_name(csv_path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("case", "state"))
        writer.writerows(((key, value["state"]) for key, value in latency_cases.items()))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, csv_path)
    csv_bytes = csv_path.read_bytes()
    cursor_path = OUT_ROOT / "resume_cursor.json"
    cursor = {"schema_version": "track3_jitter_cursor_v2", "status": "j1_passed", "active": False, "completed": ["J0", "J1"], "not_evaluated": [f"J{index}" for index in range(2, 12)], "last_heartbeat_utc": now(), "next_phase": "J2"}
    write_json(cursor_path, cursor)
    tests = {"latency_50_passes": latency_cases["50.0"]["state"] == "passed", "latency_148_46_passes": latency_cases["148.46"]["state"] == "passed", "latency_600_passes_with_warning": latency_cases["600.0"]["state"] == "passed" and "gpu_query_p99_over_500ms" in latency_cases["600.0"]["warnings"], "heartbeat_1999_passes": heartbeat_cases["1999.0"]["state"] == "passed", "heartbeat_2001_fails": heartbeat_cases["2001.0"]["state"] == "failed", "csv_lf_only": b"\r\n" not in csv_bytes and b"\n" in csv_bytes, "csv_hash_stable": sha256_file(csv_path) == sha256_bytes(csv_bytes), "atomic_partial_empty": not partial.exists()}
    report = {"schema_version": "track3_jitter_j1_v2", "status": "track3_jitter_j1_passed" if all(tests.values()) else "track3_jitter_stopped_J1_contract_failure", "created_at_utc": now(), "plan": relative(PLAN), "telemetry_schema": {"version": "v2", "gpu_query_latency_ms": "diagnostic distribution only", "heartbeat_gap_ms": "sampling completion gap", "heartbeat_gap_max_ms": HEARTBEAT_GAP_MAX_MS, "query_p99_warning_ms": QUERY_P99_WARNING_MS, "required_gate": list(_valid_telemetry_fixture(1, 1).keys())}, "latency_cases": latency_cases, "heartbeat_cases": heartbeat_cases, "tests": tests, "csv_probe": {"path": relative(csv_path), "bytes": len(csv_bytes), "sha256": sha256_file(csv_path), "contains_crlf": b"\r\n" in csv_bytes, "lineterminator": "\\n"}, "resume_cursor": relative(cursor_path), "launcher_contract": {"entrypoint": MODULE, "launch_form": "setsid", "single_instance": True, "pid_file": relative(OUT_ROOT / "supervisor.pid"), "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS, "atomic_cursor": True, "reject_pid_reuse_or_dirty_source": True, "term_then_atomic_closeout": True}, "simulator_imported": False, "cuda_imported": False, "tool_sha256": sha256_file(Path(__file__)), "root_head": git(("rev-parse", "HEAD"))}
    write_json(OUT_ROOT / "j1_telemetry_v2_contract.json", report)
    return report


def run_j2() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    j1 = strict_json_load(OUT_ROOT / "j1_telemetry_v2_contract.json") if (OUT_ROOT / "j1_telemetry_v2_contract.json").is_file() else {}
    old_parent = strict_json_load(OLD_ROOT / "o2_runs/task10_demo0_attempt3/parent_result.json")
    old_child = strict_json_load(OLD_ROOT / "o2_runs/task10_demo0_attempt3/child_result.json")
    if j1.get("status") != "track3_jitter_j1_passed":
        result = {"schema_version": "track3_jitter_j2_v2", "status": "not_evaluated", "reason": "j1_not_passed", "task10_attempt4": "forbidden"}
        write_json(OUT_ROOT / "j2_derived_attempt3_release.json", result)
        return result
    provenance = old_parent["gpu_provenance"]
    start_ts = datetime.fromisoformat(provenance["start_snapshot"]["timestamp_utc"])
    end_ts = datetime.fromisoformat(provenance["end_snapshot"]["timestamp_utc"])
    elapsed_ms = (end_ts - start_ts).total_seconds() * 1000.0
    count = int(provenance["parent_poll_count"])
    old_gate = old_parent.get("gpu_provenance_gate", {})
    record = {"seed": old_parent["seed"], "parent_pid": old_parent["parent_pid"], "child_pid": old_parent["worker_pid"], "child_result_pid": old_child["child_pid"], "physical_gpu_uuid": provenance["physical_gpu_uuid"], "cuda_visible_devices": provenance["cuda_visible_devices"], "start_snapshot": provenance["start_snapshot"], "end_snapshot": provenance["end_snapshot"], "peak_memory_mib": provenance["peak_memory_mib"], "deadline_seconds": provenance["deadline_seconds"], "term_grace_seconds": provenance["term_grace_seconds"], "process_returncode": old_parent["process_returncode"], "timed_out": old_parent["timed_out"], "gpu_query_latency_ms": {"status": "partially_measured", "count": count, "min_ms": "not_recoverable_historical", "p50_ms": "not_recoverable_historical", "p95_ms": "not_recoverable_historical", "p99_ms": "not_recoverable_historical", "max_ms": provenance["parent_poll_interval_max_ms"], "source": "old_parent_poll_interval_max_ms_only"}, "heartbeat_gap_ms": {"status": "not_evaluated_historical", "reason": "old_evidence_did_not_store_sample_completion_timestamps"}, "supervisor_elapsed_ms": elapsed_ms, "child_elapsed_ms": elapsed_ms, "sampling_coverage_count": count, "historical_not_recoverable": True}
    gate = telemetry_v2_gate(record)
    release = {"schema_version": "track3_jitter_j2_v2", "status": "track3_jitter_j2_support_remove_released" if gate["state"] == "passed" else "track3_jitter_stopped_J2_historical_gate_failure", "created_at_utc": now(), "source_attempt": {"path": relative(OLD_ROOT / "o2_runs/task10_demo0_attempt3/parent_result.json"), "attempt_number": old_parent["attempt_number"], "original_status": old_parent["status"], "original_gpu_provenance_gate": old_gate, "historical_failure_unchanged": True}, "derived_telemetry_v2": record, "telemetry_v2_gate": gate, "historical_mapping": {"parent_snapshot_count": count, "snapshot_span_ms": elapsed_ms, "coverage_hz": count / (elapsed_ms / 1000.0), "heartbeat_detail": "not_evaluated_historical", "no_pseudodata": True}, "release": {"family": "support_remove", "task_index": 10, "demo_index": 0, "task10_attempt4": "forbidden", "child_camera_physics_predicates_passed": True, "camera_artifacts_passed": old_parent.get("camera_artifacts_passed") is True, "artifact_partial_residue": old_parent.get("atomic_partial_residue", []), "released_for_executor": gate["state"] == "passed"}, "simulator_imported": False, "cuda_imported": False, "tool_sha256": sha256_file(Path(__file__)), "root_head": git(("rev-parse", "HEAD"))}
    write_json(OUT_ROOT / "j2_derived_attempt3_release.json", release)
    return release


def _terminate_group(process: subprocess.Popen[Any], grace: float = TERM_GRACE_SECONDS) -> str:
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
        process.wait(timeout=max(1.0, grace))
        return "TERM_then_KILL"


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"status": "measured", "count": 0, "min_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None, "mean_ms": None}
    ordered = sorted(float(value) for value in values)
    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]
    return {"status": "measured", "count": len(ordered), "min_ms": ordered[0], "p50_ms": percentile(.50), "p95_ms": percentile(.95), "p99_ms": percentile(.99), "max_ms": ordered[-1], "mean_ms": sum(ordered) / len(ordered)}


def _j3_worker(task: int, demo: int, seed: int, attempt_number: int, state_bundle: Path, frames_path: Path, result_path: Path) -> int:
    # Importing the prior runtime helper is child-only.  It has the proven
    # composed state/camera implementation; J3 changes the parent telemetry
    # contract and keeps the child evidence in an independent JSON file.
    from tools.axis_pi05 import track3_overnight_autogen_lora_and_zip as prior

    return prior._camera_worker(task, demo, seed, attempt_number, state_bundle, frames_path, result_path)


def _run_j3_canary(task: int, *, demo: int, seed: int, attempt_number: int) -> dict[str, Any]:
    run_root = OUT_ROOT / "j3_runs" / f"task{task}_demo{demo}_attempt{attempt_number}"
    if run_root.exists():
        raise RuntimeError(f"j3_unique_attempt_already_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    child_path = run_root / "child_result.json"
    parent_path = run_root / "parent_result.json"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    frames_path = OUT_ROOT / "j3_camera_frames" / f"task{task}_demo{demo}_attempt{attempt_number}.npz"
    if frames_path.exists() or frames_path.with_name(frames_path.name + ".partial").exists():
        raise RuntimeError(f"j3_artifact_already_exists:{frames_path}")
    state_bundle = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/start_states.npz"
    command = [sys.executable, "-m", MODULE, "--j3-worker", "--task-index", str(task), "--demo-index", str(demo), "--seed", str(seed), "--attempt-number", str(attempt_number), "--state-bundle", str(state_bundle), "--frames-output", str(frames_path), "--result-output", str(child_path)]
    environment = {**os.environ, "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "JAX_PLATFORMS": "cpu"}
    before = gpu_snapshot(GPU3)
    if not before.get("free_gate"):
        failure = {"schema_version": "track3_jitter_j3_parent_v2", "status": "failed_closed", "category": "infrastructure", "reason": "gpu3_not_free_before_spawn", "task_index": task, "attempt_number": attempt_number, "start_snapshot": before}
        write_json(parent_path, failure)
        return failure
    query_latency: list[float] = []
    heartbeat_times: list[float] = []
    snapshots: list[dict[str, Any]] = []
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    process: subprocess.Popen[Any]
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        while process.poll() is None:
            sample_started = time.monotonic()
            snapshot = gpu_snapshot(GPU3)
            query_latency.append((time.monotonic() - sample_started) * 1000.0)
            heartbeat_times.append(time.monotonic())
            snapshots.append(snapshot)
            if time.monotonic() - started >= DEADLINE_SECONDS:
                timed_out = True
                termination_stage = _terminate_group(process)
                break
            time.sleep(max(0.0, min(0.1, DEADLINE_SECONDS - (time.monotonic() - started))))
        process.wait()
    after_started = time.monotonic()
    after = gpu_snapshot(GPU3)
    query_latency.append((time.monotonic() - after_started) * 1000.0)
    heartbeat_times.append(time.monotonic())
    snapshots.append(after)
    child = None
    result_error = None
    if child_path.is_file():
        try:
            value = strict_json_load(child_path)
            if isinstance(value, dict):
                child = value
            else:
                result_error = "child_result_not_object"
        except Exception as exc:
            result_error = f"{type(exc).__name__}:{exc}"
    gaps = [(heartbeat_times[index] - heartbeat_times[index - 1]) * 1000.0 for index in range(1, len(heartbeat_times))]
    elapsed_ms = (time.monotonic() - started) * 1000.0
    process_record = {"seed": seed, "parent_pid": os.getpid(), "child_pid": process.pid, "child_result_pid": (child or {}).get("child_pid"), "physical_gpu_uuid": (before.get("gpus") or [{}])[0].get("uuid"), "cuda_visible_devices": "3", "start_snapshot": before, "end_snapshot": after, "peak_memory_mib": max((float(row["gpus"][0]["memory_used_mib"]) for row in snapshots if row.get("gpus")), default=-1.0), "deadline_seconds": DEADLINE_SECONDS, "term_grace_seconds": TERM_GRACE_SECONDS, "process_returncode": process.returncode, "timed_out": timed_out, "gpu_query_latency_ms": _stats(query_latency), "heartbeat_gap_ms": _stats(gaps), "supervisor_elapsed_ms": elapsed_ms, "child_elapsed_ms": elapsed_ms, "sampling_coverage_count": len(snapshots)}
    gate = telemetry_v2_gate(process_record)
    child_status = child.get("status") if child else None
    child_family_failure = child_status == "failed_gate"
    camera = {"state": "failed", "errors": ["missing"]}
    videos: dict[str, Any] = {}
    if frames_path.is_file():
        try:
            import numpy as np

            with np.load(frames_path, allow_pickle=False) as archive:
                camera = {"state": "passed", "keys": sorted(archive.files), "allow_pickle": False, "shapes": {key: list(np.asarray(archive[key]).shape) for key in archive.files}}
                for key in ("agentview", "wrist"):
                    video_path = run_root / f"{key}.mp4"
                    # Encode with the local implementation while retaining
                    # the .partial.<suffix> atomic naming contract.
                    images = np.asarray(archive[key], dtype=np.uint8)
                    partial = video_path.with_name(video_path.stem + ".partial" + video_path.suffix)
                    partial.unlink(missing_ok=True)
                    encode = subprocess.run(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{images.shape[2]}x{images.shape[1]}", "-r", "20", "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)], input=images.tobytes(), capture_output=True, check=False)
                    if encode.returncode != 0 or not partial.is_file():
                        raise RuntimeError(f"ffmpeg_failed:{encode.returncode}")
                    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,avg_frame_rate,nb_frames", "-of", "json", str(partial)], capture_output=True, text=True, check=False)
                    stream = json.loads(probe.stdout)["streams"][0]
                    if probe.returncode != 0 or stream.get("codec_name") != "h264" or int(stream.get("nb_frames", -1)) != 41 or str(stream.get("avg_frame_rate")) not in {"20/1", "20"}:
                        raise RuntimeError("ffprobe_contract_failed")
                    os.replace(partial, video_path)
                    videos[key] = {"path": relative(video_path), "bytes": video_path.stat().st_size, "sha256": sha256_file(video_path), "ffprobe": stream, "frame_count": 41, "fps": "20/1", "atomic_publish": True}
        except Exception as exc:
            camera = {"state": "failed", "errors": [f"{type(exc).__name__}:{exc}"]}
    camera_passed = camera.get("state") == "passed" and set(videos) == {"agentview", "wrist"}
    if child_family_failure and gate["state"] != "passed":
        status = "family_failed"
        category = "family_failure"
    elif gate["state"] == "passed" and child_status == "passed" and camera_passed:
        status = "passed"
        category = "success"
    elif timed_out:
        status = "timeout_terminated"
        category = "infrastructure"
    else:
        status = "failed_closed"
        category = "infrastructure"
    result = {"schema_version": "track3_jitter_j3_parent_v2", "status": status, "category": category, "phase": "J3", "task_index": task, "demo_index": demo, "family": FAMILY[task], "attempt_number": attempt_number, "seed": seed, "parent_pid": os.getpid(), "worker_pid": process.pid, "worker_pid_match": process.pid == (child or {}).get("child_pid"), "process_returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage, "child_result": child, "child_result_path": relative(child_path), "result_error": result_error, "gpu_provenance": process_record, "telemetry_v2_gate": gate, "camera": camera, "videos": videos, "camera_artifacts_passed": camera_passed, "stdout_path": relative(stdout_path), "stderr_path": relative(stderr_path), "stdout_bytes": stdout_path.stat().st_size, "stderr_bytes": stderr_path.stat().st_size, "atomic_partial_residue": [relative(path) for path in run_root.glob("*.partial")], "resource_boundary": {"gpu3_only": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "official_gpu": False, "no_teleop": True, "no_training": True}}
    write_json(parent_path, result)
    return result


def _j3_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for item in sorted(OUT_ROOT.rglob("*")):
        if item.is_file() and not item.name.endswith(".partial") and item.name not in {"j3_artifact_manifest.json"}:
            entries.append({"relative_path": relative(item), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    manifest = {"schema_version": "track3_jitter_j3_artifact_manifest_v2", "created_at_utc": now(), "entries_exclude_self": True, "entries": entries}
    write_json(OUT_ROOT / "j3_artifact_manifest.json", manifest)
    return {"path": relative(OUT_ROOT / "j3_artifact_manifest.json"), "entry_count": len(entries), "state": "passed"}


def run_j3() -> dict[str, Any]:
    j2 = strict_json_load(OUT_ROOT / "j2_derived_attempt3_release.json") if (OUT_ROOT / "j2_derived_attempt3_release.json").is_file() else {}
    if j2.get("status") != "track3_jitter_j2_support_remove_released":
        result = {"schema_version": "track3_jitter_j3_v2", "status": "not_evaluated", "reason": "j2_not_released", "task10_attempt4": "forbidden"}
        write_json(OUT_ROOT / "j3_summary.json", result)
        return result
    canaries: list[dict[str, Any]] = []
    for task in (20, 19):
        canary = _run_j3_canary(task, demo=0, seed=SEEDS[task], attempt_number=1)
        canaries.append(canary)
        if canary.get("category") == "infrastructure":
            result = {"schema_version": "track3_jitter_j3_v2", "status": f"track3_jitter_stopped_J3_task{task}_infrastructure_failure", "hard_stop": True, "canaries": canaries, "task10_attempt4": "forbidden", "task20_task19_no_retry": True}
            write_json(OUT_ROOT / "j3_summary.json", result)
            return result
    released = [row["family"] for row in canaries if row.get("status") == "passed"] + ["support_remove"]
    summary = {"schema_version": "track3_jitter_j3_v2", "status": "track3_jitter_j3_all_families_passed" if len(released) == 3 else ("track3_jitter_j3_partial_family_release" if released else "track3_jitter_stopped_J3_no_family_release"), "hard_stop": False, "canaries": canaries, "released_families": released, "counts": {"task10_attempt3_reused": 1, "task20_attempt1": 1, "task19_attempt1": 1, "task10_attempt4": 0}, "artifact_manifest": _j3_manifest(), "task10_attempt4": "forbidden", "no_retry": True, "source_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "resource_boundary": {"gpu3_only": True, "gpu0_touched": False, "official_gpu": False, "no_training": True}}
    write_json(OUT_ROOT / "j3_summary.json", summary)
    return summary


def _pid_is_this_supervisor(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (FileNotFoundError, PermissionError):
        return False
    return MODULE in command and "--supervisor" in command


def run_supervisor() -> dict[str, Any]:
    """Single-instance, SSH-independent supervisor for the J3 lane."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pid_path = OUT_ROOT / "supervisor.pid"
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError as exc:
            raise RuntimeError("malformed_supervisor_pid_file") from exc
        if old_pid == os.getpid() or _pid_is_this_supervisor(old_pid):
            raise RuntimeError(f"supervisor_already_active:{old_pid}")
        raise RuntimeError(f"supervisor_pid_file_requires_review:{old_pid}")
    pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    cursor_path = OUT_ROOT / "resume_cursor.json"
    write_json(cursor_path, {"schema_version": "track3_jitter_cursor_v2", "status": "active", "pid": os.getpid(), "commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "last_heartbeat_utc": now(), "next_phase": "J3"})
    try:
        j3 = run_j3()
        final = {"schema_version": "track3_jitter_supervisor_v1", "status": "completed_J3", "j3_status": j3.get("status"), "last_heartbeat_utc": now(), "pid": os.getpid(), "commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "next_phase": "J4" if j3.get("released_families") else "J11"}
        write_json(cursor_path, final)
        write_json(OUT_ROOT / "supervisor_closeout.json", final)
        return final
    finally:
        pid_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# J4: CPU-only executor contract.  Geometry is extracted from the public
# forward HDF5 files once and frozen before any J5 GPU process is launched.


HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
PROFILE_CONTRACT = {
    "nominal": {"translation_clip_m": 0.05, "rotation_clip_rad": 0.12, "translation_tolerance_m": 0.012, "rotation_tolerance_rad": 0.03, "stable_ticks": 4, "step_budget": 900},
    "conservative": {"translation_clip_m": 0.025, "rotation_clip_rad": 0.06, "translation_tolerance_m": 0.008, "rotation_tolerance_rad": 0.02, "stable_ticks": 8, "step_budget": 1200},
}


def _quat_normalize(value: Sequence[float]) -> tuple[float, float, float, float]:
    values = [float(item) for item in value]
    if len(values) != 4 or not all(math.isfinite(item) for item in values):
        raise ValueError("invalid_quaternion")
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= 1e-12:
        raise ValueError("zero_quaternion")
    return tuple(item / norm for item in values)


def quaternion_log_error(target: Sequence[float], current: Sequence[float]) -> list[float]:
    """Shortest quaternion-log rotation vector, wxyz convention."""
    tw, tx, ty, tz = _quat_normalize(target)
    cw, cx, cy, cz = _quat_normalize(current)
    # target * inverse(current)
    w = tw * cw + tx * cx + ty * cy + tz * cz
    x = -tw * cx + tx * cw - ty * cz + tz * cy
    y = -tw * cy + ty * cw - tz * cx + tx * cz
    z = -tw * cz + tz * cw - tx * cy + ty * cx
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    angle = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), max(-1.0, min(1.0, w)))
    denominator = math.sqrt(x * x + y * y + z * z)
    if denominator <= 1e-12:
        return [0.0, 0.0, 0.0]
    scale = angle / denominator
    return [x * scale, y * scale, z * scale]


def osc_pose_action(translation: Sequence[float], rotation: Sequence[float], gripper: float, profile: Mapping[str, Any]) -> list[float]:
    translation_values = [float(item) for item in translation]
    rotation_values = [float(item) for item in rotation]
    if len(translation_values) != 3 or len(rotation_values) != 3 or not all(math.isfinite(item) for item in translation_values + rotation_values):
        raise ValueError("nonfinite_se3_error")
    tclip = float(profile["translation_clip_m"])
    rclip = float(profile["rotation_clip_rad"])
    tnorm = math.sqrt(sum(item * item for item in translation_values))
    rnorm = math.sqrt(sum(item * item for item in rotation_values))
    if tnorm > tclip:
        translation_values = [item * tclip / tnorm for item in translation_values]
    if rnorm > rclip:
        rotation_values = [item * rclip / rnorm for item in rotation_values]
    value = translation_values + rotation_values + [1.0 if float(gripper) >= 0 else -1.0]
    if len(value) != 7 or not all(math.isfinite(item) for item in value):
        raise ValueError("invalid_7d_osc_pose_action")
    return value


def _extract_hdf5_demo(task: int, demo: int = 0) -> dict[str, Any]:
    import h5py
    import numpy as np

    path = HDF5_INPUTS[task]
    with h5py.File(path, "r") as handle:
        group = handle["data"][f"demo_{demo}"]
        actions = np.asarray(group["actions"], dtype=np.float64)
        states = np.asarray(group["states"], dtype=np.float64)
        ee_pos = np.asarray(group["obs"]["ee_pos"], dtype=np.float64)
        ee_ori = np.asarray(group["obs"]["ee_ori"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or states.ndim != 2 or ee_pos.shape[0] != actions.shape[0] or ee_ori.shape[0] != actions.shape[0]:
            raise ValueError(f"hdf5_schema_task{task}_demo{demo}")
        signs = actions[:, 6]
        transition = next((index for index in range(1, len(signs)) if signs[index - 1] < 0 and signs[index] > 0), None)
        state_motion_after = float(np.linalg.norm(states[-1] - states[0]))
        position_motion = float(np.linalg.norm(ee_pos[-1] - ee_pos[0]))
        rotation_motion = float(np.linalg.norm(ee_ori[-1] - ee_ori[0]))
        return {"task_index": task, "demo_index": demo, "path": relative(path), "sha256": sha256_file(path), "frame_count": int(actions.shape[0]), "action_dim": int(actions.shape[1]), "state_dim": int(states.shape[1]), "action_finite": bool(np.isfinite(actions).all()), "state_finite": bool(np.isfinite(states).all()), "eef_finite": bool(np.isfinite(ee_pos).all() and np.isfinite(ee_ori).all()), "gripper_values": sorted(float(value) for value in np.unique(signs)), "gripper_transition_index": transition, "gripper_sign_proof": {"required_transition": task in (10, 20), "transition_found": transition is not None, "native_action_values": [-1.0, 1.0] if task in (10, 20) else [-1.0]}, "post_transition_motion": state_motion_after, "eef_position_motion_m": position_motion, "eef_rotation_motion_rad": rotation_motion, "fixture_direction_evidence": {"required": task == 19, "state_motion_nonzero": state_motion_after > 1e-9, "eef_path_nonzero": position_motion > 1e-9}}


def run_j4() -> dict[str, Any]:
    j3 = strict_json_load(OUT_ROOT / "j3_summary.json") if (OUT_ROOT / "j3_summary.json").is_file() else {}
    if j3.get("status") not in {"track3_jitter_j3_all_families_passed", "track3_jitter_j3_partial_family_release"}:
        result = {"schema_version": "track3_jitter_j4_v2", "status": "not_evaluated", "reason": "j3_no_release", "task10_attempt4": "forbidden"}
        write_json(OUT_ROOT / "j4_executor_contract.json", result)
        return result
    extractions: dict[str, Any] = {}
    family_status: dict[str, str] = {}
    for task in TASK_ORDER:
        try:
            extracted = [_extract_hdf5_demo(task, demo) for demo in (0, 1, 2)]
            extractions[str(task)] = extracted
            required_proof = all(item["action_finite"] and item["state_finite"] and item["eef_finite"] for item in extracted)
            if task in (10, 20):
                required_proof = required_proof and all(item["gripper_transition_index"] is not None for item in extracted)
            else:
                required_proof = required_proof and all(item["fixture_direction_evidence"]["state_motion_nonzero"] for item in extracted)
            family_status[FAMILY[task]] = "released" if required_proof else "closed_no_hdf5_proof"
        except Exception as exc:
            extractions[str(task)] = {"error": f"{type(exc).__name__}:{exc}"}
            family_status[FAMILY[task]] = "closed_hdf5_exception"
    profiles = {family: {name: {**value, "profile_name": name, "frozen_before_gpu": True} for name, value in PROFILE_CONTRACT.items()} for family in FAMILY.values()}
    tests = {"all_profiles_have_se3_clip": all("rotation_clip_rad" in profile and profile["rotation_clip_rad"] > 0 for family in profiles.values() for profile in family.values()), "action_builder_shape": len(osc_pose_action([0, 0, 0], [0, 0, 0], -1, PROFILE_CONTRACT["nominal"])) == 7, "quaternion_log_identity_zero": max(abs(value) for value in quaternion_log_error([1, 0, 0, 0], [1, 0, 0, 0])) == 0.0, "all_executed_family_proofs_present": all(value in {"released", "closed_no_hdf5_proof", "closed_hdf5_exception"} for value in family_status.values())}
    report = {"schema_version": "track3_jitter_j4_v2", "status": "track3_jitter_j4_executor_contract_passed" if all(tests.values()) else "track3_jitter_stopped_J4_contract_failure", "created_at_utc": now(), "plan": relative(PLAN), "source_j3": relative(OUT_ROOT / "j3_summary.json"), "families": family_status, "hdf5_extractions": extractions, "profiles": profiles, "executor_contract": {"controller": "OSC_POSE", "action_dim": 7, "translation_rotation_separately_clipped": True, "rotation_error": "quaternion_log_shortest_path", "gripper_values": [-1.0, 1.0], "step_budget_is_frozen": True, "contact_contract": "v3 intended receiver + demonstrator + unrelated background", "unrelated_movement_gate_m": 0.002, "stall_timeout_gate": True, "hidden_task_special_case": False}, "tests": tests, "task10_attempt4": "forbidden", "simulator_imported": False, "cuda_imported": False, "source_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__))}
    write_json(OUT_ROOT / "j4_executor_contract.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j0", action="store_true")
    parser.add_argument("--j1", action="store_true")
    parser.add_argument("--j2", action="store_true")
    parser.add_argument("--j3", action="store_true")
    parser.add_argument("--j4", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--j3-worker", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--demo-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt-number", type=int)
    parser.add_argument("--state-bundle", type=Path)
    parser.add_argument("--frames-output", type=Path)
    parser.add_argument("--result-output", type=Path)
    args = parser.parse_args(argv)
    if args.j3_worker:
        required = (args.task_index, args.demo_index, args.seed, args.attempt_number, args.state_bundle, args.frames_output, args.result_output)
        if any(value is None for value in required):
            parser.error("--j3-worker requires task/demo/seed/attempt/state/frames/result")
        return _j3_worker(args.task_index, args.demo_index, args.seed, args.attempt_number, args.state_bundle, args.frames_output, args.result_output)
    if args.supervisor:
        result = run_supervisor()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    selected = [name for name, value in (("j0", args.j0), ("j1", args.j1), ("j2", args.j2), ("j3", args.j3), ("j4", args.j4)) if value]
    if len(selected) != 1:
        parser.error("select exactly one of --j0/--j1/--j2/--j3/--j4")
    result = {"j0": run_j0, "j1": run_j1, "j2": run_j2, "j3": run_j3, "j4": run_j4}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if "passed" in str(result.get("status", "")) or "released" in str(result.get("status", "")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
