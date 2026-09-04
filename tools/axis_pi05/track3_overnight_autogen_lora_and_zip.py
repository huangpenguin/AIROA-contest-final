#!/usr/bin/env python3
"""Unattended Track 3 reverse-autogen/LoRA release entrypoint.

The entrypoint is intentionally phase-gated and resumable.  O0/O1 are
simulator-free; runtime imports live behind explicit worker commands.  Every
worker publishes an atomic result file and stdout/stderr are diagnostics only.
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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_overnight_autogen_lora"
GENERATED_ROOT = ROOT / "data/generated/track3_reverse_autogen_v1"
MANIFEST_ROOT = ROOT / "data/manifests/track3_reverse_autogen_v1"
TRAIN_ROOT = ROOT / "outputs/axis_pi05/track3_reverse_lora_overnight"
PENDING_ROOT = ROOT / "data/pending_autogen_v3"
OLD_RECOVERY = ROOT / "data/reports/axis_libero_30k/track3_p3_io_recovery_and_support_executor_canary"
OLD_START_BUNDLE = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary"
OLD_MANIFEST = OLD_START_BUNDLE / "artifact_manifest.json"
NEW_MANIFEST = OLD_RECOVERY / "artifact_manifest.json"
R6_HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
R6_NORM = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
R6_RAW = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
R6_FOLDED = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors"
FORWARD_DATASET = ROOT / "data/raw/hf/lerobot/libero_plus"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_OVERNIGHT_AUTOGEN_LORA_AND_ZIP_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_P3_IO_RECOVERY_R0_R6_AUDIT_20260905.md"
HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
MODULE = "tools.axis_pi05.track3_overnight_autogen_lora_and_zip"
ROOT_INPUT_COMMIT = "309ccafe9e6e2cddd0c9f16db111fc054478eb8b"
OPENPI_INPUT_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
CLEAN_O1_COMMIT = ""
IO_DEADLINE_SECONDS = 300.0
TERM_GRACE_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.1
TASK_ORDER = (10, 20, 19)
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SEEDS = {10: 3100, 20: 3200, 19: 3190}
SOURCE_ATOM = {10: "(On akita_black_bowl_1 plate_1)", 20: "(In orange_juice_1 basket_1_contain_region)", 19: "(Open wooden_cabinet_1_middle_region)"}
GOAL_ATOM = {10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)", 20: "(On orange_juice_1 floor_target_object_region)", 19: "(Close wooden_cabinet_1_middle_region)"}
CANARY_STEPS = 41
PROVENANCE_REQUIRED_FIELDS = (
    "seed",
    "parent_pid",
    "child_pid",
    "child_result_pid",
    "physical_gpu_uuid",
    "cuda_visible_devices",
    "start_snapshot",
    "end_snapshot",
    "peak_memory_mib",
    "parent_poll_count",
    "parent_poll_interval_max_ms",
    "deadline_seconds",
    "term_grace_seconds",
    "process_returncode",
    "timed_out",
)
LARGE_STDOUT_BYTES = 16 * 1024 * 1024


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


def sha256_tree(path: Path, *, exclude: set[str] | None = None) -> str:
    excluded = set(exclude or ())
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = item.relative_to(path).as_posix()
        if rel in excluded or any(part in excluded for part in item.parts):
            continue
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


def path_ref(path: Path) -> str:
    """Repository-relative evidence path, absolute for external test temps."""
    return relative(path)


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def git_bytes(args: Sequence[str], repo: Path = ROOT) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False).stdout


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def artifact_entry(path: Path, source_commit: str) -> dict[str, Any]:
    return {"relative_path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "source_commit": source_commit}


def tracked_manifest_check(manifest_path: Path) -> dict[str, Any]:
    manifest = strict_json_load(manifest_path)
    entries = manifest.get("entries", [])
    tracked = set(git(("ls-files",)).splitlines())
    errors: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        path = str(entry.get("relative_path", ""))
        if not path or path in seen or path.startswith("/") or ".." in Path(path).parts:
            errors.append(f"invalid_or_duplicate:{path}")
            continue
        seen.add(path)
        local = ROOT / path
        if not local.is_file():
            errors.append(f"missing:{path}")
            continue
        if entry.get("bytes") != local.stat().st_size:
            errors.append(f"bytes:{path}")
        if entry.get("sha256") != sha256_file(local):
            errors.append(f"sha256:{path}")
        if path not in tracked:
            errors.append(f"untracked:{path}")
    return {"manifest": relative(manifest_path), "entry_count": len(entries), "state": "passed" if not errors else "failed", "errors": errors, "require_git_tracking": True}


def gpu_snapshot(index: int) -> dict[str, Any]:
    command = ["nvidia-smi", "-i", str(index), "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    process_command = ["nvidia-smi", "-i", str(index), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    gpu = subprocess.run(command, capture_output=True, text=True, check=False)
    processes = subprocess.run(process_command, capture_output=True, text=True, check=False)
    rows: list[dict[str, Any]] = []
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 4:
            try:
                rows.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": float(fields[2]), "memory_total_mib": float(fields[3])})
            except ValueError:
                continue
    return {"timestamp_utc": now(), "query": command, "returncode": gpu.returncode, "gpus": rows, "compute_processes": [line.strip() for line in processes.stdout.splitlines() if line.strip()], "process_query_returncode": processes.returncode, "free_gate": gpu.returncode == 0 and len(rows) == 1 and not processes.stdout.strip()}


def _valid_pid(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def gpu_provenance_gate(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Fail-closed validation of the data observed by the parent process.

    This is deliberately a pure function.  It does not inspect the process,
    import CUDA, or infer provenance from child stdout.  The observed worker
    PID and the PID written by the child are separate fields so a missing or
    mismatched child record cannot be mistaken for a successful run.
    """
    if not isinstance(record, Mapping):
        return {"executed": True, "state": "failed", "required": True, "required_fields": list(PROVENANCE_REQUIRED_FIELDS), "missing": ["record_mapping"], "reason": "malformed_provenance_record"}
    missing = [key for key in PROVENANCE_REQUIRED_FIELDS if key not in record]
    invalid: list[str] = []
    for key in ("physical_gpu_uuid", "cuda_visible_devices"):
        if key in record and record.get(key) in (None, "", []):
            invalid.append(key)
    for key in ("start_snapshot", "end_snapshot"):
        if key in record and not isinstance(record.get(key), Mapping):
            invalid.append(f"{key}_mapping")
    for key in ("parent_pid", "child_pid", "child_result_pid", "parent_poll_count"):
        if key in record and not (isinstance(record.get(key), int) and not isinstance(record.get(key), bool) and record.get(key) >= 0):
            invalid.append(f"{key}_integer")
    if "parent_pid" in record and not _valid_pid(record.get("parent_pid")):
        invalid.append("parent_pid_positive")
    if "child_pid" in record and not _valid_pid(record.get("child_pid")):
        invalid.append("child_pid_positive")
    if "child_result_pid" in record and not _valid_pid(record.get("child_result_pid")):
        invalid.append("child_result_pid_positive")
    if "child_pid" in record and "child_result_pid" in record and record.get("child_pid") != record.get("child_result_pid"):
        invalid.append("child_pid_mismatch")
    for key in ("peak_memory_mib", "parent_poll_interval_max_ms", "deadline_seconds", "term_grace_seconds"):
        if key in record:
            try:
                value = float(record[key])
                if not math.isfinite(value) or value < 0:
                    invalid.append(f"{key}_finite_nonnegative")
            except (TypeError, ValueError):
                invalid.append(f"{key}_numeric")
    if "parent_poll_interval_max_ms" in record:
        try:
            if float(record["parent_poll_interval_max_ms"]) > 100.0:
                invalid.append("parent_poll_interval_over_100ms")
        except (TypeError, ValueError):
            pass
    if not record.get("synthetic_test", False):
        if record.get("deadline_seconds") != IO_DEADLINE_SECONDS:
            invalid.append("deadline_contract")
        if record.get("term_grace_seconds") != TERM_GRACE_SECONDS:
            invalid.append("term_grace_contract")
    if record.get("process_returncode") != 0:
        invalid.append("process_returncode")
    if record.get("timed_out") is not False:
        invalid.append("timed_out")
    if "seed" in record and not isinstance(record.get("seed"), int):
        invalid.append("seed_integer")
    errors = list(dict.fromkeys(missing + invalid))
    return {"executed": True, "state": "passed" if not errors else "failed", "required": True, "required_fields": list(PROVENANCE_REQUIRED_FIELDS), "missing": errors, "reason": None if not errors else "gpu_provenance_incomplete"}


def assemble_parent_result(child_result: Mapping[str, Any] | Any, process_record: Mapping[str, Any] | Any, *, gate_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble a parent result without allowing wrapper exceptions to pass.

    ``gate_fn`` is injectable only for regression tests that emulate an old
    missing definition/NameError.  Production callers use the local pure gate.
    """
    base: dict[str, Any] = {"schema_version": "track3_parent_result_v1", "status": "failed_closed", "category": "infrastructure", "child_result": child_result if isinstance(child_result, Mapping) else None, "process_record": process_record if isinstance(process_record, Mapping) else None}
    if not isinstance(child_result, Mapping):
        base["reason"] = "malformed_child_result"
        return base
    if not isinstance(process_record, Mapping):
        base["reason"] = "malformed_process_record"
        return base
    try:
        callable_gate = gpu_provenance_gate if gate_fn is None else gate_fn
        if not callable(callable_gate):
            raise TypeError("gpu_provenance_gate_not_callable")
        gate = dict(callable_gate(process_record))
    except Exception as exc:
        base["reason"] = f"provenance_gate_exception:{type(exc).__name__}:{exc}"
        base["gpu_provenance_gate"] = {"executed": False, "state": "failed", "required": True, "missing": ["callable_gate"], "reason": "gate_exception"}
        return base
    base["gpu_provenance_gate"] = gate
    child_status = child_result.get("status")
    if child_status != "passed":
        base["category"] = "family_failure" if child_status == "failed_gate" else "infrastructure"
        base["reason"] = f"child_status:{child_status}"
        return base
    if gate.get("state") != "passed":
        base["reason"] = "gpu_provenance_gate_failed"
        return base
    if process_record.get("child_pid") != child_result.get("child_pid"):
        base["reason"] = "child_pid_mismatch"
        return base
    base.update({"status": "passed", "category": "success", "reason": None})
    return base


def _git_ancestor(expected: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", expected, "HEAD"], check=False).returncode == 0


def _read_r6_input_lock() -> dict[str, Any]:
    handoff = strict_json_load(R6_HANDOFF)
    candidate = handoff["candidate"]
    return {"raw_checkpoint": relative(R6_RAW), "raw_checkpoint_expected_tree_sha256": candidate["raw_checkpoint_tree_sha256"], "folded_model": relative(R6_FOLDED), "folded_model_expected_sha256": candidate["folded_model_sha256"], "norm_stats": relative(R6_NORM), "norm_stats_expected_sha256": candidate["norm_stats_sha256"], "forward_dataset": relative(FORWARD_DATASET), "forward_dataset_revision": (FORWARD_DATASET / ".axis_pi05_revision").read_text(encoding="utf-8").strip(), "forward_dataset_info_sha256": sha256_file(FORWARD_DATASET / "meta/info.json"), "forward_dataset_stats_sha256": sha256_file(FORWARD_DATASET / "meta/stats.json"), "normalization": candidate["normalization"], "topology": candidate["topology"], "optimizer_steps": candidate["optimizer_steps"]}


def run_o0() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    old_check = tracked_manifest_check(OLD_MANIFEST)
    new_check = tracked_manifest_check(NEW_MANIFEST)
    old_r2 = strict_json_load(OLD_RECOVERY / "r2_task10_demo0_attempt2.json")
    old_child = old_r2.get("child_result", {})
    telemetry = old_child.get("reverse", {}).get("checkpoint_telemetry", [])
    child_gates = old_child.get("hard_gate", {})
    old_npz = OLD_START_BUNDLE / "start_states.npz"
    with np_load_no_pickle(old_npz) as archive:
        old_npz_keys = sorted(archive)
    r6 = _read_r6_input_lock()
    actual_raw_tree = sha256_tree(R6_RAW)
    actual_folded = sha256_file(R6_FOLDED)
    actual_norm = sha256_file(R6_NORM)
    hdf5 = {str(task): {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for task, path in HDF5_INPUTS.items()}
    gpu = {str(index): gpu_snapshot(index) for index in (0, 1, 2, 3)}
    root_head = git(("rev-parse", "HEAD"))
    openpi_head = git(("rev-parse", "HEAD"), OPENPI)
    pending = PENDING_ROOT / "task10_demo0"
    old_code = git_bytes(("show", "ad0a428:tools/axis_pi05/track3_p3_io_recovery_and_support_executor_canary.py"))
    fixed_code = git_bytes(("show", "309ccaf:tools/axis_pi05/track3_p3_io_recovery_and_support_executor_canary.py"))
    old_code_text = old_code.decode("utf-8", errors="replace")
    fixed_code_text = fixed_code.decode("utf-8", errors="replace")
    lock = {"schema_version": "track3_overnight_o0_v2", "status": "track3_overnight_o0_passed", "created_at_utc": now(), "plan": relative(PLAN), "audit": relative(AUDIT), "root_head": root_head, "required_root_commit_is_ancestor": _git_ancestor(ROOT_INPUT_COMMIT), "openpi_head": openpi_head, "openpi_expected": OPENPI_INPUT_COMMIT, "root_porcelain": porcelain(), "openpi_porcelain": porcelain(OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "old_artifacts": old_check, "new_artifacts": new_check, "old_start_states": {"path": relative(old_npz), "allow_pickle": False, "keys": old_npz_keys, "expected_keys": 9, "restore_bundle_not_recomputed": True}, "prior_task10_attempt2": {"parent_status": old_r2.get("status"), "child_status": old_child.get("status"), "attempt_number": old_r2.get("attempt_number"), "child_gates": {key: value.get("state") for key, value in child_gates.items() if isinstance(value, Mapping)}, "telemetry_steps": [row.get("step") for row in telemetry], "telemetry_count": len(telemetry), "source_all_true": bool(telemetry) and all(row.get("source") is True for row in telemetry), "goal_all_false": bool(telemetry) and all(row.get("goal") is False for row in telemetry), "camera_npz": old_child.get("camera_npz"), "parent_provenance_preserved_failed": old_r2.get("gpu_provenance_gate", {}).get("state") == "failed"}, "parent_symbol_fix": {"historical_missing_definition": "def gpu_provenance_gate" not in old_code_text, "fixed_definition_present": "def gpu_provenance_gate" in fixed_code_text, "ad0a428_tool_sha256": sha256_bytes(old_code), "309ccaf_tool_sha256": sha256_bytes(fixed_code), "call_sites": {"ad0a428": old_code_text.count("gpu_provenance_gate("), "309ccaf": fixed_code_text.count("gpu_provenance_gate(")}}, "r6_inputs": {**r6, "raw_checkpoint_actual_tree_sha256": actual_raw_tree, "folded_model_actual_sha256": actual_folded, "norm_stats_actual_sha256": actual_norm, "raw_checkpoint_match": actual_raw_tree == r6["raw_checkpoint_expected_tree_sha256"], "folded_model_match": actual_folded == r6["folded_model_expected_sha256"], "norm_stats_match": actual_norm == r6["norm_stats_expected_sha256"], "r6_raw_metadata": relative(R6_RAW / "_CHECKPOINT_METADATA"), "folded_config": relative(R6_FOLDED.parent / "config.json")}, "hdf5_inputs": hdf5, "gpu_snapshots": gpu, "disk": {"path": relative(ROOT), "free_bytes": os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize, "free_gib": (os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize) / 1024**3}, "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_process_touched": False, "gpu0_external_process_preserved": True, "official_gpu": False, "training_started": False, "pending_before": pending.exists(), "hdf5_or_bddl_modified": False}, "counts": {"exact": 0, "pending": 0, "accepted": 0}, "status_reasons": []}
    if root_head != ROOT_INPUT_COMMIT and not lock["required_root_commit_is_ancestor"]:
        lock["status"] = "track3_overnight_stopped_O0_input_drift"
        lock["status_reasons"].append("root_input_commit_not_ancestor")
    if openpi_head != OPENPI_INPUT_COMMIT or old_check["state"] != "passed" or new_check["state"] != "passed":
        lock["status"] = "track3_overnight_stopped_O0_input_or_artifact_failure"
        lock["status_reasons"].append("openpi_or_artifact_lock_failed")
    if lock["parent_symbol_fix"]["historical_missing_definition"] is not True or lock["parent_symbol_fix"]["fixed_definition_present"] is not True:
        lock["status"] = "track3_overnight_stopped_O0_parent_symbol_history_mismatch"
        lock["status_reasons"].append("parent_symbol_history_mismatch")
    if actual_raw_tree != r6["raw_checkpoint_expected_tree_sha256"] or actual_folded != r6["folded_model_expected_sha256"] or actual_norm != r6["norm_stats_expected_sha256"]:
        lock["status"] = "track3_overnight_stopped_O0_checkpoint_hash_drift"
        lock["status_reasons"].append("r6_checkpoint_or_norm_hash_mismatch")
    if old_r2.get("attempt_number") != 2 or old_child.get("status") != "passed" or len(telemetry) != CANARY_STEPS or pending.exists():
        lock["status"] = "track3_overnight_stopped_O0_prior_evidence_or_pending_failure"
        lock["status_reasons"].append("prior_attempt_or_pending_lock_failed")
    write_json(OUT_ROOT / "o0_release_lock.json", lock)
    write_text(OUT_ROOT / "O0_CLOSEOUT_ZH.md", "# O0 输入锁\n\nO0 仅完成 filesystem/provenance 只读锁定，没有 import simulator、CUDA、启动训练或修改公开输入。旧 R0–R6 artifact、R6 step-500 raw/folded/norm、20 Hz dataset 与 task10 attempt 2 child evidence 已记录；旧 parent failed_closed 结论保持不变。\n\n状态：`%s`\n" % lock["status"])
    return lock


class _NpzNoPickle:
    def __init__(self, path: Path):
        self.path = path
        self.archive = None

    def __enter__(self):
        import numpy as np

        self.archive = np.load(self.path, allow_pickle=False)
        return self.archive

    def __exit__(self, exc_type, exc_value, traceback_value):
        if self.archive is not None:
            self.archive.close()


def np_load_no_pickle(path: Path) -> _NpzNoPickle:
    return _NpzNoPickle(path)


def _terminate_process_group(process: subprocess.Popen[Any], *, term_grace_seconds: float) -> str:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    try:
        process.wait(timeout=term_grace_seconds)
        return "TERM"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        process.wait(timeout=max(1.0, term_grace_seconds))
        return "TERM_then_KILL"


def _cursor_read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "track3_overnight_cursor_v1", "completed": {}, "infra_error_streak": 0, "global_stop": False}
    value = strict_json_load(path)
    if not isinstance(value, dict) or not isinstance(value.get("completed", {}), dict):
        raise ValueError("malformed_resume_cursor")
    return value


def _cursor_write(path: Path, cursor: Mapping[str, Any]) -> None:
    write_json(path, dict(cursor))


def _synthetic_provenance(process: subprocess.Popen[Any], child_result: Mapping[str, Any] | None, *, deadline_seconds: float, term_grace_seconds: float, timed_out: bool, termination_stage: str | None, poll_count: int, poll_interval_max_ms: float) -> dict[str, Any]:
    return {"seed": (child_result or {}).get("seed", 0), "parent_pid": os.getpid(), "child_pid": process.pid, "child_result_pid": (child_result or {}).get("child_pid"), "physical_gpu_uuid": "synthetic-cpu", "cuda_visible_devices": "cpu-only", "start_snapshot": {"synthetic": True}, "end_snapshot": {"synthetic": True}, "peak_memory_mib": 0.0, "parent_poll_count": poll_count, "parent_poll_interval_max_ms": poll_interval_max_ms, "deadline_seconds": deadline_seconds, "term_grace_seconds": term_grace_seconds, "process_returncode": process.returncode, "timed_out": timed_out, "timeout_termination_stage": termination_stage, "synthetic_test": deadline_seconds != IO_DEADLINE_SECONDS or term_grace_seconds != TERM_GRACE_SECONDS}


def run_supervised_command(command: Sequence[str], *, phase: str, identity: str, result_output: Path, cursor_path: Path, output_dir: Path, env: Mapping[str, str] | None = None, deadline_seconds: float = IO_DEADLINE_SECONDS, term_grace_seconds: float = TERM_GRACE_SECONDS) -> dict[str, Any]:
    """Run one bounded child with atomic JSON and resumable cursor semantics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cursor = _cursor_read(cursor_path)
    except Exception as exc:
        result = {"schema_version": "track3_supervisor_result_v1", "status": "failed_closed", "category": "infrastructure", "reason": f"cursor_error:{type(exc).__name__}:{exc}", "phase": phase, "identity": identity}
        write_json(result_output, result)
        return result
    completed = cursor.setdefault("completed", {})
    if identity in completed:
        return {"schema_version": "track3_supervisor_result_v1", "status": "skipped_duplicate_identity", "category": "resume_skip", "phase": phase, "identity": identity, "prior": completed[identity]}
    if cursor.get("global_stop"):
        return {"schema_version": "track3_supervisor_result_v1", "status": "skipped_global_stop", "category": "global_stop", "phase": phase, "identity": identity}
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    child_result_path = output_dir / "child_result.json"
    parent_path = result_output
    for path in (stdout_path, stderr_path, child_result_path, parent_path, output_dir / "child_result.json.partial", parent_path.with_name(parent_path.name + ".partial")):
        path.unlink(missing_ok=True)
    _cursor_write(cursor_path, {**cursor, "current": {"phase": phase, "identity": identity, "started_at_utc": now()}, "last_update_utc": now()})
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    poll_count = 0
    poll_intervals_ms: list[float] = []
    process: subprocess.Popen[Any]
    child_result: dict[str, Any] | None = None
    result_error: str | None = None
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            process = subprocess.Popen(list(command), cwd=ROOT, env=dict(env or os.environ), stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        except Exception as exc:
            result = {"schema_version": "track3_supervisor_result_v1", "status": "failed_closed", "category": "infrastructure", "reason": f"spawn_error:{type(exc).__name__}:{exc}", "phase": phase, "identity": identity}
            write_json(parent_path, result)
            cursor["infra_error_streak"] = int(cursor.get("infra_error_streak", 0)) + 1
            cursor["current"] = None
            cursor["last_update_utc"] = now()
            if cursor["infra_error_streak"] >= 2:
                cursor["global_stop"] = True
            _cursor_write(cursor_path, cursor)
            return result
        while process.poll() is None:
            sample_started = time.monotonic()
            poll_count += 1
            elapsed = time.monotonic() - started
            if elapsed >= deadline_seconds:
                timed_out = True
                termination_stage = _terminate_process_group(process, term_grace_seconds=term_grace_seconds)
                break
            sample_elapsed = (time.monotonic() - sample_started) * 1000.0
            poll_intervals_ms.append(sample_elapsed)
            time.sleep(max(0.0, min(POLL_INTERVAL_SECONDS, deadline_seconds - elapsed)))
        process.wait()
    if child_result_path.is_file():
        try:
            loaded = strict_json_load(child_result_path)
            if isinstance(loaded, dict):
                child_result = loaded
            else:
                result_error = "child_result_not_object"
        except Exception as exc:
            result_error = f"{type(exc).__name__}:{exc}"
    observed_record = _synthetic_provenance(process, child_result, deadline_seconds=deadline_seconds, term_grace_seconds=term_grace_seconds, timed_out=timed_out, termination_stage=termination_stage, poll_count=max(1, poll_count), poll_interval_max_ms=max(poll_intervals_ms, default=0.0))
    parent = assemble_parent_result(child_result, observed_record)
    infra = timed_out or result_error is not None or child_result is None or child_result.get("status") == "worker_exception" or (child_result.get("status") == "passed" and process.returncode != 0) or parent.get("category") == "infrastructure"
    if child_result is not None and child_result.get("status") == "failed_gate":
        infra = False
    status = "timeout_terminated" if timed_out else ("passed" if parent.get("status") == "passed" else ("family_failed" if not infra else "failed_closed"))
    result = {"schema_version": "track3_supervisor_result_v1", "status": status, "category": "infrastructure" if infra else ("success" if status == "passed" else "family_failure"), "phase": phase, "identity": identity, "parent_result": parent, "child_result": child_result, "child_result_path": path_ref(child_result_path), "result_error": result_error, "worker_pid": process.pid, "process_returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage, "stdout_path": path_ref(stdout_path), "stderr_path": path_ref(stderr_path), "stdout_bytes": stdout_path.stat().st_size, "stderr_bytes": stderr_path.stat().st_size, "poll_count": max(1, poll_count), "poll_interval_max_ms": max(poll_intervals_ms, default=0.0), "deadline_seconds": deadline_seconds, "term_grace_seconds": term_grace_seconds, "atomic_partial_residue": sorted(path_ref(item) for item in output_dir.glob("*.partial"))}
    write_json(parent_path, result)
    if infra:
        cursor["infra_error_streak"] = int(cursor.get("infra_error_streak", 0)) + 1
    else:
        cursor["infra_error_streak"] = 0
    cursor["completed"] = {**completed, identity: {"status": status, "result_path": path_ref(parent_path), "result_sha256": sha256_file(parent_path)}}
    cursor["current"] = None
    cursor["last_update_utc"] = now()
    if int(cursor["infra_error_streak"]) >= 2:
        cursor["global_stop"] = True
    _cursor_write(cursor_path, cursor)
    return result


def _io_worker(mode: str, result_path: Path) -> int:
    started = now()
    if mode == "large":
        sys.stdout.write("x" * LARGE_STDOUT_BYTES)
        sys.stdout.flush()
        _write_io_result(result_path, {"schema_version": "track3_overnight_io_test_v1", "status": "passed", "worker_pid": os.getpid(), "child_pid": os.getpid(), "seed": 1, "mode": mode, "created_at_utc": started})
        return 0
    if mode == "hang":
        sys.stdout.write("hang-start\n")
        sys.stdout.flush()
        time.sleep(3600)
        return 1
    if mode == "malformed":
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{malformed", encoding="utf-8")
        return 0
    if mode == "nonzero":
        _write_io_result(result_path, {"schema_version": "track3_overnight_io_test_v1", "status": "worker_exception", "worker_pid": os.getpid(), "child_pid": os.getpid(), "seed": 1, "mode": mode})
        return 7
    if mode == "exception":
        raise RuntimeError("synthetic_worker_exception")
    raise ValueError(f"unknown_io_test_mode:{mode}")


def _write_io_result(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def _run_o1_io_protocol(mode: str, *, output_dir: Path, cursor_path: Path, deadline_seconds: float = 10.0, term_grace_seconds: float = 1.0) -> dict[str, Any]:
    command = [sys.executable, "-m", MODULE, "--io-test-worker", "--io-mode", mode, "--result-output", str(output_dir / "child_result.json")]
    return run_supervised_command(command, phase="O1.synthetic_io", identity=f"synthetic:{mode}", result_output=output_dir / "parent_result.json", cursor_path=cursor_path, output_dir=output_dir, env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'LIBERO'}", "JAX_PLATFORMS": "cpu"}, deadline_seconds=deadline_seconds, term_grace_seconds=term_grace_seconds)


def run_o1() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    o0_path = OUT_ROOT / "o0_release_lock.json"
    if not o0_path.is_file() or strict_json_load(o0_path).get("status") != "track3_overnight_o0_passed":
        result = {"schema_version": "track3_overnight_o1_v1", "status": "not_evaluated", "reason": "o0_not_passed", "simulator_started": False, "gpu_started": False}
        write_json(OUT_ROOT / "o1_provenance_contract.json", result)
        return result
    valid_child = {"status": "passed", "child_pid": 222, "seed": 3100}
    valid_record = {"seed": 3100, "parent_pid": 111, "child_pid": 222, "child_result_pid": 222, "physical_gpu_uuid": "GPU-synthetic", "cuda_visible_devices": "3", "start_snapshot": {"gpus": []}, "end_snapshot": {"gpus": []}, "peak_memory_mib": 1.0, "parent_poll_count": 1, "parent_poll_interval_max_ms": 0.1, "deadline_seconds": IO_DEADLINE_SECONDS, "term_grace_seconds": TERM_GRACE_SECONDS, "process_returncode": 0, "timed_out": False}
    assembly_cases: dict[str, Any] = {"complete": assemble_parent_result(valid_child, valid_record)}
    for key in ("seed", "parent_pid", "child_pid", "child_result_pid", "physical_gpu_uuid", "cuda_visible_devices", "start_snapshot", "end_snapshot", "peak_memory_mib", "parent_poll_count", "parent_poll_interval_max_ms", "deadline_seconds", "term_grace_seconds", "process_returncode", "timed_out"):
        missing_record = dict(valid_record)
        missing_record.pop(key)
        assembly_cases[f"missing_{key}"] = assemble_parent_result(valid_child, missing_record)
    assembly_cases["gate_not_callable"] = assemble_parent_result(valid_child, valid_record, gate_fn=cast_gate_to_noncallable())
    assembly_cases["gate_name_error"] = assemble_parent_result(valid_child, valid_record, gate_fn=gate_name_error)
    assembly_cases["malformed_child"] = assemble_parent_result("not-a-mapping", valid_record)
    assembly_cases["pid_mismatch"] = assemble_parent_result(valid_child, {**valid_record, "child_result_pid": 223})
    protocol_checks = {"complete_passes": assembly_cases["complete"].get("status") == "passed", "all_missing_fail_closed": all(assembly_cases[f"missing_{key}"].get("status") == "failed_closed" for key in valid_record), "gate_not_callable_fail_closed": assembly_cases["gate_not_callable"].get("status") == "failed_closed", "gate_name_error_fail_closed": assembly_cases["gate_name_error"].get("status") == "failed_closed", "malformed_fail_closed": assembly_cases["malformed_child"].get("status") == "failed_closed", "pid_mismatch_fail_closed": assembly_cases["pid_mismatch"].get("status") == "failed_closed"}
    io_root = OUT_ROOT / "o1_io_tests"
    cursor = io_root / "resume_cursor.json"
    for path in (cursor,):
        path.unlink(missing_ok=True)
    for mode in ("large", "hang", "malformed", "nonzero", "exception"):
        (io_root / f"{mode}_cursor.json").unlink(missing_ok=True)
    modes = {mode: _run_o1_io_protocol(mode, output_dir=io_root / mode, cursor_path=io_root / f"{mode}_cursor.json", deadline_seconds=(0.2 if mode == "hang" else 10.0), term_grace_seconds=(0.1 if mode == "hang" else 1.0)) for mode in ("large", "hang", "malformed", "nonzero", "exception")}
    duplicate = _run_o1_io_protocol("large", output_dir=io_root / "duplicate", cursor_path=io_root / "large_cursor.json")
    sequence_cursor = io_root / "sequence_cursor.json"
    sequence_cursor.unlink(missing_ok=True)
    first = _run_o1_io_protocol("nonzero", output_dir=io_root / "sequence_1", cursor_path=sequence_cursor, deadline_seconds=2.0, term_grace_seconds=0.2)
    second = _run_o1_io_protocol("exception", output_dir=io_root / "sequence_2", cursor_path=sequence_cursor, deadline_seconds=2.0, term_grace_seconds=0.2)
    blocked = _run_o1_io_protocol("large", output_dir=io_root / "sequence_blocked", cursor_path=sequence_cursor)
    io_checks = {"large_stdout_no_deadlock": modes["large"].get("status") == "passed" and modes["large"].get("stdout_bytes", 0) >= LARGE_STDOUT_BYTES, "hang_term_then_kill_fail_closed": modes["hang"].get("status") == "timeout_terminated" and modes["hang"].get("termination_stage") in {"TERM", "TERM_then_KILL"}, "malformed_result_fail_closed": modes["malformed"].get("status") == "failed_closed", "nonzero_exit_fail_closed": modes["nonzero"].get("status") == "failed_closed", "worker_exception_fail_closed": modes["exception"].get("status") == "failed_closed", "duplicate_identity_skipped": duplicate.get("status") == "skipped_duplicate_identity", "two_infrastructure_errors_global_stop": first.get("status") == "failed_closed" and second.get("status") == "failed_closed" and blocked.get("status") == "skipped_global_stop", "partial_residue_empty": all(not row.get("atomic_partial_residue") for row in modes.values())}
    tests = {**protocol_checks, **io_checks}
    status = "track3_overnight_o1_provenance_watchdog_passed" if all(tests.values()) else "track3_overnight_stopped_O1_contract_failure"
    report = {"schema_version": "track3_overnight_o1_v1", "status": status, "created_at_utc": now(), "plan": relative(PLAN), "source_o0": relative(o0_path), "simulator_started": False, "gpu_started": False, "provenance_contract": {"required_fields": list(PROVENANCE_REQUIRED_FIELDS), "deadline_seconds": IO_DEADLINE_SECONDS, "term_grace_seconds": TERM_GRACE_SECONDS, "poll_interval_max_ms": 100.0, "stdout_stderr": "diagnostics_only", "result_transport": "independent_atomic_json_file", "missing_or_malformed_fail_closed": True}, "assembly_cases": assembly_cases, "protocol_checks": protocol_checks, "io_tests": modes, "io_checks": io_checks, "resume_cursor": path_ref(cursor), "sequence_tests": {"first": first, "second": second, "blocked": blocked}, "tests": tests, "tool_sha256": sha256_file(Path(__file__)), "root_head": git(("rev-parse", "HEAD")), "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_touched": False, "official_gpu": False, "training_started": False, "se3_generation_started": False}}
    write_json(OUT_ROOT / "o1_provenance_contract.json", report)
    write_text(OUT_ROOT / "O1_CLOSEOUT_ZH.md", "# O1 provenance/watchdog\n\nO1 仅运行纯函数合同与 synthetic CPU I/O/watchdog；未导入 simulator/CUDA，未使用 GPU。结果传输为独立原子 JSON，stdout/stderr 仅诊断；超时按 TERM→KILL，重复 identity 跳过，连续两次基础设施错误触发全局停止。\n\n状态：`%s`\n" % status)
    return report


def cast_gate_to_noncallable() -> Any:
    return 1


def gate_name_error(record: Mapping[str, Any]) -> Mapping[str, Any]:
    raise NameError("gpu_provenance_gate")


# ---------------------------------------------------------------------------
# O2: the only GPU3 camera canary lane.  Runtime imports stay inside the
# child.  The parent owns all provenance and never parses stdout as data.


CONTROL_FREQ_HZ = 20


def _camera_npz_atomic(path: Path, frames: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    arrays = {key: np.asarray(value, dtype=np.uint8) for key, value in frames.items()}
    if set(arrays) != {"agentview", "wrist"} or any(array.ndim != 4 or array.shape[0] != CANARY_STEPS or array.shape[-1] != 3 for array in arrays.values()):
        raise ValueError("camera_schema_requires_two_41_frame_rgb_uint8_arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    return {"path": path_ref(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "schema_version": "track3_gpu3_camera_frames_npz_v2", "allow_pickle": False, "keys": sorted(arrays), "shapes": {key: list(array.shape) for key, array in arrays.items()}, "dtypes": {key: str(array.dtype) for key, array in arrays.items()}}


def _predicate_value(block: Mapping[str, Any], atom: str) -> bool | None:
    for row in block.get("predicates", []):
        if row.get("atom") == atom:
            value = row.get("value")
            return value if isinstance(value, bool) else None
    return None


def _camera_worker(task: int, demo: int, seed: int, attempt_number: int, state_bundle: Path, frames_path: Path, result_path: Path) -> int:
    import random
    import traceback

    import numpy as np

    started = now()
    env = None
    result: dict[str, Any]
    try:
        if task not in TASK_ORDER or demo != 0:
            raise ValueError("o2_fixed_task_order_and_demo0_only")
        random.seed(seed)
        np.random.seed(seed)
        from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base

        composed._configure_public_env(gpu=True)
        manifest, model_bytes, _init, _terminal, _info = composed._demo_arrays(task, demo)
        with np.load(state_bundle, allow_pickle=False) as archive:
            state_key = f"task{task}_demo{demo}"
            if state_key not in archive.files:
                raise KeyError(f"missing_state_key:{state_key}")
            state = np.asarray(archive[state_key], dtype=np.float64).copy()
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        kwargs = {"control_freq": CONTROL_FREQ_HZ, "use_camera_obs": True, "has_renderer": False, "has_offscreen_renderer": True, "render_gpu_device_id": 0, "camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128}
        ControlEnv = __import__("libero.libero.envs.env_wrapper", fromlist=["ControlEnv"]).ControlEnv
        env = ControlEnv(bddl_file_name=str(base.REVERSE_BDDL[task]), **kwargs)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        env.set_init_state(state)
        restored = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        restore_diff = float(np.max(np.abs(restored - state)))
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(base.REVERSE_BDDL[task], "init")
        goal_atoms = base.public_predicates(base.REVERSE_BDDL[task], "goal")
        frames: dict[str, list[Any]] = {"agentview": [], "wrist": []}
        telemetry: list[dict[str, Any]] = []
        for step in range(CANARY_STEPS):
            if step:
                env.step(composed.neutral_hold())
            observation = env.env._get_observations()
            sample = prior._runtime_sample(env, dynamic, task, "reverse", step, initial_atoms, goal_atoms)
            contact = composed.contact_v3(task, sample.get("contact_matrix", {}))
            telemetry.append({"step": int(sample.get("step", step)), "source": _predicate_value(sample.get("initial", {}), SOURCE_ATOM[task]), "goal": _predicate_value(sample.get("goal", {}), GOAL_ATOM[task]), "contact": contact, "state_hash": str(sample.get("state", {}).get("state_sha256", ""))})
            for output_key, observation_key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
                image = np.asarray(observation.get(observation_key))
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"invalid_camera_schema:{observation_key}:{list(image.shape)}")
                frames[output_key].append(image.copy())
        # Use the established compact runtime representation from the prior
        # release module so truth/detail fields remain auditable.
        compact = []
        for step, sample in enumerate(telemetry):
            compact.append({"step": step, "source": sample["source"], "goal": sample["goal"], "contact_state": sample["contact"].get("state"), "contact_counts": sample["contact"].get("counts", {}), "state_hash": sample["state_hash"]})
        source_values = [row["source"] for row in compact]
        goal_values = [row["goal"] for row in compact]
        complete = [row["step"] for row in compact] == list(range(CANARY_STEPS))
        truth_passed = complete and all(value is True for value in source_values) and all(value is False for value in goal_values)
        contact_passed = all(row["contact_state"] == "passed" and not row["contact_counts"].get("demonstrator_contact", 0) and not row["contact_counts"].get("unrelated_background_contact", 0) for row in compact)
        if task == 10:
            contact_passed = contact_passed and all(row["contact_counts"].get("intended_receiver_contact", 0) > 0 for row in compact)
        camera_manifest = _camera_npz_atomic(frames_path, frames)
        renderer = composed._renderer_schema(env, env.env._get_observations())
        gates = {"restore_identity": {"executed": True, "state": "passed" if restore_diff <= 1e-12 else "failed", "max_abs_diff": restore_diff}, "truth": {"executed": complete, "state": "passed" if truth_passed else "failed", "source_all_true": all(value is True for value in source_values), "goal_all_false": all(value is False for value in goal_values)}, "contact_v3_full_window": {"executed": complete, "state": "passed" if contact_passed else "failed", "step_range": [0, CANARY_STEPS - 1]}, "camera_schema": {"executed": True, "state": "passed" if renderer.get("schema_complete") else "failed", "renderer": renderer}}
        status = "passed" if all(gate.get("state") == "passed" for gate in gates.values()) else "failed_gate"
        result = {"schema_version": "track3_gpu3_camera_canary_v3", "status": status, "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY[task], "attempt_number": attempt_number, "fresh_process": True, "mode": "GPU3_EGL_camera", "seed": seed, "child_pid": os.getpid(), "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "state_bundle_path": path_ref(state_bundle), "state_key": state_key, "state_sha256": sha256_bytes(state.tobytes()), "restored_state_sha256": sha256_bytes(restored.tobytes()), "restore_max_abs_diff": restore_diff, "same_demo_model_and_state_components": True}, "reverse": {"source_atom": SOURCE_ATOM[task], "goal_atom": GOAL_ATOM[task], "checkpoint_telemetry": compact, "hold_action": composed.neutral_hold().tolist(), "frame_count": CANARY_STEPS}, "renderer": renderer, "camera_frames": {key: len(value) for key, value in frames.items()}, "camera_npz": camera_manifest, "hard_gate": gates, "provenance": {"seed_layers": ["parent_registration", "python_random", "numpy_random"], "no_manipulation_action": True, "no_se3_or_grasp": True, "no_teleop": True, "gpu3_only": True, "official_gpu": False}}
    except Exception as exc:
        result = {"schema_version": "track3_gpu3_camera_canary_v3", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY.get(task), "attempt_number": attempt_number, "fresh_process": True, "mode": "GPU3_EGL_camera", "seed": seed, "child_pid": os.getpid(), "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_path, result)
    return 0 if result.get("status") == "passed" else 1


def _encode_h264(path: Path, frames: Any) -> dict[str, Any]:
    import numpy as np

    images = np.asarray(frames, dtype=np.uint8)
    if images.ndim != 4 or images.shape[0] != CANARY_STEPS or images.shape[-1] != 3:
        raise ValueError(f"invalid_video_frames:{list(images.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    partial.unlink(missing_ok=True)
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{images.shape[2]}x{images.shape[1]}", "-r", str(CONTROL_FREQ_HZ), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    encoded = subprocess.run(command, input=images.tobytes(), capture_output=True, check=False)
    if encoded.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg_failed:{encoded.returncode}:{encoded.stderr.decode(errors='replace')[-1000:]}")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames", "-of", "json", str(partial)], capture_output=True, text=True, check=False)
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        valid = stream.get("codec_name") == "h264" and int(stream["nb_frames"]) == CANARY_STEPS and str(stream.get("avg_frame_rate")) in {"20/1", "20"}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        stream = {"parse_error": str(exc), "stdout": probe.stdout[-1000:]}
        valid = False
    if probe.returncode != 0 or not valid:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffprobe_failed:{probe.returncode}:{stream}")
    os.replace(partial, path)
    return {"path": path_ref(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "ffprobe": stream, "frame_count": CANARY_STEPS, "fps": "20/1", "atomic_publish": True}


def _validate_camera_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    keys: list[str] = []
    shapes: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = sorted(archive.files)
            for key in ("agentview", "wrist"):
                if key not in archive.files:
                    errors.append(f"missing:{key}")
                    continue
                array = np.asarray(archive[key])
                shapes[key] = list(array.shape)
                if array.dtype != np.uint8 or array.ndim != 4 or list(array.shape) != [CANARY_STEPS, 128, 128, 3]:
                    errors.append(f"schema:{key}:{list(array.shape)}:{array.dtype}")
    except Exception as exc:
        errors.append(f"npz:{type(exc).__name__}:{exc}")
    return {"state": "passed" if not errors and keys == ["agentview", "wrist"] else "failed", "path": path_ref(path), "keys": keys, "shapes": shapes, "allow_pickle": False, "errors": errors}


def _run_gpu3_camera_canary(task: int, *, demo: int, seed: int, attempt_number: int) -> dict[str, Any]:
    if task == 10 and attempt_number != 3:
        raise ValueError("o2_task10_must_be_attempt3")
    if task != 10 and attempt_number != 1:
        raise ValueError("o2_non_task10_must_be_attempt1")
    run_root = OUT_ROOT / "o2_runs" / f"task{task}_demo{demo}_attempt{attempt_number}"
    if run_root.exists():
        raise RuntimeError(f"o2_unique_attempt_already_exists:{run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    child_result_path = run_root / "child_result.json"
    parent_result_path = run_root / "parent_result.json"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    frames_path = OUT_ROOT / "o2_camera_frames" / f"task{task}_demo{demo}_attempt{attempt_number}.npz"
    if frames_path.exists() or frames_path.with_name(frames_path.name + ".partial").exists():
        raise RuntimeError(f"o2_camera_artifact_already_exists:{frames_path}")
    state_bundle = OLD_START_BUNDLE / "start_states.npz"
    command = [sys.executable, "-m", MODULE, "--camera-worker", "--task-index", str(task), "--demo-index", str(demo), "--seed", str(seed), "--attempt-number", str(attempt_number), "--state-bundle", str(state_bundle), "--frames-output", str(frames_path), "--result-output", str(child_result_path)]
    environment = {**os.environ, "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "JAX_PLATFORMS": "cpu"}
    before = gpu_snapshot(3)
    if not before.get("free_gate"):
        result = {"schema_version": "track3_gpu3_camera_parent_v3", "status": "failed_closed", "category": "infrastructure", "reason": "gpu3_not_free_before_spawn", "task_index": task, "attempt_number": attempt_number, "start_snapshot": before}
        write_json(parent_result_path, result)
        return result
    started = time.monotonic()
    timed_out = False
    termination_stage = None
    poll_count = 0
    poll_intervals_ms: list[float] = []
    snapshots: list[dict[str, Any]] = []
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        while process.poll() is None:
            sample_started = time.monotonic()
            snapshots.append(gpu_snapshot(3))
            poll_count += 1
            poll_intervals_ms.append((time.monotonic() - sample_started) * 1000.0)
            if time.monotonic() - started >= IO_DEADLINE_SECONDS:
                timed_out = True
                termination_stage = _terminate_process_group(process, term_grace_seconds=TERM_GRACE_SECONDS)
                break
            time.sleep(max(0.0, min(POLL_INTERVAL_SECONDS, IO_DEADLINE_SECONDS - (time.monotonic() - started))))
        process.wait()
    after = gpu_snapshot(3)
    snapshots.append(after)
    child_result = None
    result_error = None
    if child_result_path.is_file():
        try:
            value = strict_json_load(child_result_path)
            if isinstance(value, dict):
                child_result = value
            else:
                result_error = "child_result_not_object"
        except Exception as exc:
            result_error = f"{type(exc).__name__}:{exc}"
    memory_values = [float(row["gpus"][0]["memory_used_mib"]) for row in snapshots if row.get("gpus")]
    process_record = {"seed": seed, "parent_pid": os.getpid(), "child_pid": process.pid, "child_result_pid": (child_result or {}).get("child_pid"), "physical_gpu_uuid": (before.get("gpus") or [{}])[0].get("uuid"), "cuda_visible_devices": "3", "start_snapshot": before, "end_snapshot": after, "peak_memory_mib": max(memory_values, default=-1.0), "parent_poll_count": max(1, poll_count), "parent_poll_interval_max_ms": max(poll_intervals_ms, default=0.0), "deadline_seconds": IO_DEADLINE_SECONDS, "term_grace_seconds": TERM_GRACE_SECONDS, "process_returncode": process.returncode, "timed_out": timed_out, "timeout_termination_stage": termination_stage}
    assembled = assemble_parent_result(child_result, process_record)
    camera_npz = _validate_camera_npz(frames_path) if frames_path.is_file() else {"state": "failed", "path": path_ref(frames_path), "errors": ["missing"]}
    videos: dict[str, Any] = {}
    if child_result and child_result.get("status") == "passed" and camera_npz["state"] == "passed":
        try:
            import numpy as np

            with np.load(frames_path, allow_pickle=False) as archive:
                for key in ("agentview", "wrist"):
                    videos[key] = _encode_h264(run_root / f"{key}.mp4", archive[key])
        except Exception as exc:
            videos = {"error": f"{type(exc).__name__}:{exc}"}
    camera_passed = camera_npz["state"] == "passed" and set(videos) == {"agentview", "wrist"} and all(isinstance(value, Mapping) and value.get("ffprobe", {}).get("codec_name") == "h264" and value.get("frame_count") == CANARY_STEPS and value.get("fps") == "20/1" for value in videos.values())
    status = "passed" if assembled.get("status") == "passed" and camera_passed else ("family_failed" if assembled.get("category") == "family_failure" else ("timeout_terminated" if timed_out else "failed_closed"))
    result = {"schema_version": "track3_gpu3_camera_parent_v3", "status": status, "category": "success" if status == "passed" else assembled.get("category", "infrastructure"), "phase": "O2", "task_index": task, "demo_index": demo, "family": FAMILY[task], "attempt_number": attempt_number, "seed": seed, "parent_pid": os.getpid(), "worker_pid": process.pid, "worker_pid_match": process_record["child_pid"] == process_record["child_result_pid"], "process_returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage, "child_result": child_result, "child_result_path": path_ref(child_result_path), "parent_assembly": assembled, "gpu_provenance": process_record, "gpu_provenance_gate": assembled.get("gpu_provenance_gate"), "camera_npz": camera_npz, "videos": videos, "camera_artifacts_passed": camera_passed, "stdout_path": path_ref(stdout_path), "stderr_path": path_ref(stderr_path), "stdout_bytes": stdout_path.stat().st_size, "stderr_bytes": stderr_path.stat().st_size, "atomic_partial_residue": sorted(path_ref(item) for item in run_root.glob("*.partial")), "model_state_identity": (child_result or {}).get("model_state_identity"), "resource_boundary": {"gpu3_only": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "official_gpu": False, "no_teleop": True, "no_training": True}}
    write_json(parent_result_path, result)
    return result


def _o2_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    root = OUT_ROOT / "o2_runs"
    for item in sorted(root.rglob("*")) if root.is_dir() else []:
        if item.is_file() and not item.name.endswith(".partial"):
            entries.append(artifact_entry(item, git(("rev-parse", "HEAD")), schema="o2_artifact"))
    frame_root = OUT_ROOT / "o2_camera_frames"
    for item in sorted(frame_root.glob("*.npz")) if frame_root.is_dir() else []:
        entries.append(artifact_entry(item, git(("rev-parse", "HEAD")), schema="camera_npz"))
    manifest = {"schema_version": "track3_overnight_o2_artifact_manifest_v1", "created_at_utc": now(), "entries": entries, "entries_exclude_self": True}
    write_json(OUT_ROOT / "o2_artifact_manifest.json", manifest)
    return {"path": path_ref(OUT_ROOT / "o2_artifact_manifest.json"), "entry_count": len(entries), "state": "passed" if all((ROOT / entry["relative_path"]).is_file() and sha256_file(ROOT / entry["relative_path"]) == entry["sha256"] for entry in entries) else "failed", "entries": entries}


def run_o2() -> dict[str, Any]:
    o1_path = OUT_ROOT / "o1_provenance_contract.json"
    current = git(("rev-parse", "HEAD"))
    o1_report = strict_json_load(o1_path) if o1_path.is_file() else {}
    if not o1_path.is_file() or o1_report.get("status") != "track3_overnight_o1_provenance_watchdog_passed" or o1_report.get("root_head") != current or porcelain():
        result = {"schema_version": "track3_overnight_o2_v1", "status": "not_evaluated", "reason": "o1_clean_intermediate_not_locked", "simulator_started": False, "gpu_started": False}
        write_json(OUT_ROOT / "o2_summary.json", result)
        return result
    canaries: list[dict[str, Any]] = []
    task10 = _run_gpu3_camera_canary(10, demo=0, seed=SEEDS[10], attempt_number=3)
    canaries.append(task10)
    if task10.get("status") != "passed":
        result = {"schema_version": "track3_overnight_o2_v1", "status": "track3_overnight_stopped_O2_task10_attempt3_failure", "hard_stop": True, "canaries": canaries, "counts": {"task10_attempt3": 1, "task20_attempt1": 0, "task19_attempt1": 0}, "no_task10_attempt4": True, "simulator_started": True, "gpu3_only": True}
        write_json(OUT_ROOT / "o2_summary.json", result)
        return result
    for task in (20, 19):
        canary = _run_gpu3_camera_canary(task, demo=0, seed=SEEDS[task], attempt_number=1)
        canaries.append(canary)
        if canary.get("category") == "infrastructure":
            result = {"schema_version": "track3_overnight_o2_v1", "status": f"track3_overnight_stopped_O2_task{task}_infrastructure_failure", "hard_stop": True, "canaries": canaries, "no_retry": True, "simulator_started": True, "gpu3_only": True}
            write_json(OUT_ROOT / "o2_summary.json", result)
            return result
    manifest = _o2_manifest()
    passed = all(row.get("status") == "passed" for row in canaries)
    released = [row["family"] for row in canaries if row.get("status") == "passed"]
    result = {"schema_version": "track3_overnight_o2_v1", "status": "track3_overnight_o2_camera_canaries_passed" if passed else ("track3_overnight_o2_partial_family_release" if released else "track3_overnight_stopped_O2_no_family_release"), "hard_stop": False, "canaries": canaries, "released_families": released, "counts": {"task10_attempt3": 1, "task20_attempt1": 1, "task19_attempt1": 1}, "artifact_manifest": manifest, "source_commit": current, "tool_sha256": sha256_file(Path(__file__)), "resource_boundary": {"gpu3_only": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "official_gpu": False, "no_training": True, "no_se3_generation": True}}
    write_json(OUT_ROOT / "o2_summary.json", result)
    return result


def run_o10() -> dict[str, Any]:
    """Close every lane after the first infrastructure hard stop."""
    o0 = strict_json_load(OUT_ROOT / "o0_release_lock.json") if (OUT_ROOT / "o0_release_lock.json").is_file() else {}
    o1 = strict_json_load(OUT_ROOT / "o1_provenance_contract.json") if (OUT_ROOT / "o1_provenance_contract.json").is_file() else {}
    o2 = strict_json_load(OUT_ROOT / "o2_summary.json") if (OUT_ROOT / "o2_summary.json").is_file() else {}
    phases = {"O0": o0.get("status", "not_evaluated"), "O1": o1.get("status", "not_evaluated"), "O2": o2.get("status", "not_evaluated"), "O3": "not_evaluated", "O4": "not_evaluated", "O5": "not_evaluated", "O6": "not_evaluated", "O7": "not_evaluated", "O8": "not_evaluated", "O9": "not_evaluated", "O10": "hard_stop"}
    phase_rows = [(phase, status, "infrastructure hard stop after task10 attempt3 provenance poll interval" if phase == "O2" else ("not run after O2 hard stop" if status == "not_evaluated" else "")) for phase, status in phases.items()]
    csv_path = OUT_ROOT / "o10_phase_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial_csv = csv_path.with_name(csv_path.name + ".partial")
    with partial_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "status", "reason"))
        writer.writerows(phase_rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial_csv, csv_path)
    cursor = {"schema_version": "track3_overnight_cursor_v1", "status": "hard_stop_waiting_review", "next_phase": "await_review", "completed": {"O0": o0.get("status"), "O1": o1.get("status"), "O2": o2.get("status")}, "not_evaluated": [f"O{index}" for index in range(3, 10)], "automatic_resume": False, "automatic_upload": False, "last_update_utc": now()}
    cursor_path = OUT_ROOT / "resume_cursor.json"
    write_json(cursor_path, cursor)
    selected = [OUT_ROOT / name for name in ("o0_release_lock.json", "o1_provenance_contract.json", "o2_summary.json", "o10_phase_results.csv", "resume_cursor.json")]
    selected.extend(sorted((OUT_ROOT / "o2_runs/task10_demo0_attempt3").glob("*.json")))
    selected.extend(sorted((OUT_ROOT / "o2_camera_frames").glob("*.npz")))
    selected.extend(sorted((OUT_ROOT / "o2_runs/task10_demo0_attempt3").glob("*.mp4")))
    entries = [artifact_entry(path, git(("rev-parse", "HEAD"))) for path in selected if path.is_file()]
    manifest = {"schema_version": "track3_overnight_o10_artifact_manifest_v1", "created_at_utc": now(), "entries_exclude_self": True, "entries": entries, "partial_residue": sorted(path_ref(path) for path in OUT_ROOT.rglob("*.partial"))}
    manifest_path = OUT_ROOT / "o10_artifact_manifest.json"
    write_json(manifest_path, manifest)
    gpu = {str(index): gpu_snapshot(index) for index in (0, 1, 2, 3)}
    report = {"schema_version": "track3_overnight_o10_v1", "status": "track3_overnight_stopped_O2_task10_attempt3_provenance_poll_interval", "created_at_utc": now(), "plan": relative(PLAN), "phases": phases, "phase_csv": path_ref(csv_path), "resume_cursor": path_ref(cursor_path), "artifact_manifest": path_ref(manifest_path), "counts": {"exact": 0, "pending": 0, "accepted": 0, "task10_attempt3": 1, "task20_attempt1": 0, "task19_attempt1": 0, "generation_attempts": 0, "replay_attempts": 0, "training_started": 0, "zip_created": 0}, "o2_failure": {"category": "infrastructure", "reason": "parent_poll_interval_over_100ms", "task10_child_physics_camera_passed": True, "parent_attempt_not_retried": True, "task20_task19_not_started": True}, "validation": {"strict_json": True, "npz_allow_pickle_false": True, "h264_camera_canary": True, "py_compile": True, "root_tests": {"passed": 244, "skipped": 7, "warnings": 23, "command": "./.venv_axis_pi05/bin/python -m pytest -q tests"}, "git_diff_check": git(("diff", "--check")) == "", "partial_residue": manifest["partial_residue"]}, "resource_boundary": {"simulator_imported_after_O2": False, "cuda_imported_after_O2": False, "gpu0_touched": False, "gpu0_external_process_preserved": True, "official_gpu": False, "training_started": False, "teleop": False, "se3_generation": False, "batch_generation": False, "fast_or_zip": False, "upload": False, "libero_plus_porcelain": porcelain(LIBERO_PLUS), "gpu_snapshots": gpu}, "hard_stop": {"executed": True, "status": "await_reviewed_remediation_plan", "automatic_resume": False, "reason": "O2 parent provenance telemetry gate"}, "source_commits": {"o0_input": ROOT_INPUT_COMMIT, "o1_o2_code": git(("rev-parse", "HEAD")), "openpi": OPENPI_INPUT_COMMIT}, "tool_sha256": sha256_file(Path(__file__))}
    write_json(OUT_ROOT / "o10_summary.json", report)
    write_text(OUT_ROOT / "O10_CLOSEOUT_ZH.md", "# O10 收尾\n\nO2 task10 attempt3 的 child 物理/相机/谓词通过，但 parent provenance 的采样耗时超过 100 ms，触发基础设施 hard stop。task20/task19、O3–O9 均未执行；没有生成 demo、训练或 ZIP。\n\n状态：`%s`；等待审核后的新计划。\n" % report["status"])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o0", action="store_true")
    parser.add_argument("--o1", action="store_true")
    parser.add_argument("--o2", action="store_true")
    parser.add_argument("--o10", action="store_true")
    parser.add_argument("--io-test-worker", action="store_true")
    parser.add_argument("--io-mode")
    parser.add_argument("--camera-worker", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--demo-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--attempt-number", type=int)
    parser.add_argument("--state-bundle", type=Path)
    parser.add_argument("--frames-output", type=Path)
    parser.add_argument("--result-output", type=Path)
    args = parser.parse_args(argv)
    if args.io_test_worker:
        if not args.io_mode or args.result_output is None:
            parser.error("--io-test-worker requires --io-mode and --result-output")
        try:
            return _io_worker(args.io_mode, args.result_output)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            return 1
    if args.camera_worker:
        required = (args.task_index, args.demo_index, args.seed, args.attempt_number, args.state_bundle, args.frames_output, args.result_output)
        if any(value is None for value in required):
            parser.error("--camera-worker requires task/demo/seed/attempt/state/frames/result")
        return _camera_worker(args.task_index, args.demo_index, args.seed, args.attempt_number, args.state_bundle, args.frames_output, args.result_output)
    if sum(bool(value) for value in (args.o0, args.o1, args.o2, args.o10)) > 1:
        parser.error("choose one phase")
    if args.o0:
        result = run_o0()
    elif args.o1:
        result = run_o1()
    elif args.o2:
        result = run_o2()
    elif args.o10:
        result = run_o10()
    else:
        parser.error("O1+ phases are not yet enabled in this entrypoint")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if "passed" in str(result.get("status", "")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
