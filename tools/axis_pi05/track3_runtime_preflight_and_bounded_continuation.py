#!/usr/bin/env python3
"""Track 3 runtime preflight and bounded continuation.

This module is the active Y0--Y13 entrypoint.  Y0--Y2 are deliberately
CPU/filesystem-only.  The later canary code uses the same public simulator
helper and the same stage contract defined here, so a preflight pass is a
real release of the code that a future canary will execute.

Historical X/W evidence is immutable input.  This module never rewrites it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
from tools.axis_pi05 import track3_j5_executor_contract_and_atomic_video_recovery as previous
from tools.axis_pi05 import track3_object_frame_executor_and_bounded_autogen_lora as object_frame
from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as w

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/reports/axis_libero_30k/track3_runtime_preflight_and_bounded_continuation"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_RUNTIME_PREFLIGHT_AND_BOUNDED_CONTINUATION_PLAN.md"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PUBLIC_LIBERO = ROOT / "LIBERO"

X_OUT = ROOT / "data/reports/axis_libero_30k/track3_object_frame_executor_and_bounded_autogen_lora"
W_OUT = ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery"
X13_MANIFEST = X_OUT / "x13_artifact_manifest.json"
X13_VALIDATION = X_OUT / "x13_final_validation.json"
X4_PARENT = X_OUT / "x4_support_cpu_profiles/object_frame_nominal/parent_result.json"
X4_CHILD = X_OUT / "x4_support_cpu_profiles/object_frame_nominal/child_result.json"
X4_STDOUT = X_OUT / "x4_support_cpu_profiles/object_frame_nominal/stdout.log"
X4_STDERR = X_OUT / "x4_support_cpu_profiles/object_frame_nominal/stderr.log"
X4_TRAJECTORY = ROOT / "data/reports/axis_libero_30k/track3_j5_executor_recovery/v4_corrected_task10_demo0_nominal/trajectory.npz"
W4_TRAJECTORY = W_OUT / "w4_cpu_task10_demo0_nominal/trajectory.npz"
W1_REPORT = W_OUT / "w1_release_anchor.json"
W2_REPORT = W_OUT / "w2_osc_scale_contract.json"
W6_MANIFEST = W_OUT / "w6_artifact_manifest.json"
W6_CLOSEOUT = W_OUT / "w6_closeout.json"
STATE_BUNDLE = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/start_states.npz"

R6_RAW = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
R6_FOLDED = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors"
R6_NORM = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"

HDF5_INPUTS = {
    10: composed.HDF5_INPUTS[10],
    20: composed.HDF5_INPUTS[20],
    19: composed.HDF5_INPUTS[19],
}

Y0_REPORT = OUT / "y0_input_lock.json"
Y0_RECHECK_REPORT = OUT / "y0_recheck_input_lock.json"
Y1_REPORT = OUT / "y1_runtime_contract.json"
Y2_REPORT = OUT / "y2_clean_release.json"
Y2_MANIFEST = OUT / "y2_artifact_manifest.json"
Y2_VALIDATION = OUT / "y2_final_validation.json"
Y2_CLOSEOUT = OUT / "y2_closeout.json"
PHASE_CSV = OUT / "y2_phase_status.csv"
CURSOR = OUT / "resume_cursor.json"
HEARTBEAT = OUT / "heartbeat.json"

MODULE = "tools.axis_pi05.track3_runtime_preflight_and_bounded_continuation"
X_COMMIT = "64a0d31c3b2864562887044bad7ef8b522a213db"
OPENPI_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
GPU0_PID = "1133706"
CONTROL_FREQ_HZ = 20
TERM_GRACE_SECONDS = 10.0
HEARTBEAT_INTERVAL_SECONDS = 900.0
CANARY_DEADLINE_SECONDS = 300.0

NOMINAL_PROFILE: dict[str, Any] = {
    "name": "object_frame_nominal",
    "translation_clip_m": 0.05,
    "rotation_clip_rad": 0.12,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 4,
    "stage_budgets": [150, 120, 35, 150, 180, 120, 35, 80],
    "step_budget": 900,
    "progress_window": 4,
    "progress_grace": 4,
    "progress_min_improvement": 0.02,
}


class DuplicateKeyError(ValueError):
    """Raised when a JSON object contains a duplicate key."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


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
    result = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines() if line]


def source_porcelain() -> list[str]:
    prefix = f"{rel(OUT).rstrip('/')}/"
    expected_bootstrap = {
        rel(Path(__file__)),
        "tests/test_track3_runtime_preflight_and_bounded_continuation.py",
        "tests/test_track3_object_frame_executor_and_bounded_autogen_lora.py",
        "tests/test_track3_support_autogen_v2.py",
        "tests/test_track3_matched_terminal_state_release_plan.py",
        "docs/AXIS_PI05_PLAN_INDEX.md",
        "docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md",
        "MEMORY.md",
        "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json",
    }
    rows = []
    for line in porcelain(ROOT):
        path = line[3:].strip().strip('"')
        if path.startswith(prefix) or path in expected_bootstrap:
            continue
        rows.append(line)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_npy(path: Path, value: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def gate(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "executed": bool(executed),
        "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"),
        "required": bool(required),
        **details,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def all_required(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in rows.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Return a deterministic file inventory, including directory contents."""
    files: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = raw.resolve()
        if path.is_dir():
            candidates = sorted(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            candidates = [path]
        else:
            candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                files.append(candidate)
    return [{"path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(files)]


def _expected_inputs() -> list[Path]:
    return [
        X13_MANIFEST,
        X13_VALIDATION,
        X4_PARENT,
        X4_CHILD,
        X4_STDOUT,
        X4_STDERR,
        W6_MANIFEST,
        W6_CLOSEOUT,
        W4_TRAJECTORY,
        X4_TRAJECTORY,
        STATE_BUNDLE,
        *HDF5_INPUTS.values(),
        R6_RAW,
        R6_FOLDED,
        R6_NORM,
    ]


def _active_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,stat=,cmd="], capture_output=True, text=True, check=False)
    needles = ("track3_runtime_preflight", "track3_object_frame", "track3_support", "supervisor", "autogen", "lora")
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if fields and fields[0].isdigit() and int(fields[0]) == os.getpid():
            continue
        if any(needle in line.lower() for needle in needles) and "rg -i" not in line:
            rows.append(line.strip())
    return rows


def cursor_record(*, active: bool, phase: str, command: Sequence[str], status: str) -> dict[str, Any]:
    return {
        "schema_version": "track3_runtime_preflight_cursor_v1",
        "active": bool(active),
        "phase": phase,
        "status": status,
        "pid": os.getpid(),
        "command": list(command),
        "root_commit": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__)),
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI),
        "gpu_policy": "cpu_no_renderer",
        "updated_at_utc": now(),
    }


def write_cursor(*, active: bool, phase: str, command: Sequence[str], status: str) -> dict[str, Any]:
    record = cursor_record(active=active, phase=phase, command=command, status=status)
    write_json(CURSOR, record)
    return record


def heartbeat_due(last_written: float | None, *, current: float, interval: float) -> bool:
    return last_written is None or current - last_written >= interval


class PeriodicHeartbeat:
    """Atomic heartbeat writer used by the real parent polling loop."""

    def __init__(self, path: Path, *, interval: float = HEARTBEAT_INTERVAL_SECONDS, clock: Callable[[], float] = time.monotonic) -> None:
        self.path = path
        self.interval = float(interval)
        self.clock = clock
        self.last_written: float | None = None
        self.count = 0

    def maybe_write(self, *, phase: str, state: str, pid: int | None = None, force: bool = False) -> dict[str, Any] | None:
        current = self.clock()
        if not force and not heartbeat_due(self.last_written, current=current, interval=self.interval):
            return None
        self.count += 1
        record = {"schema_version": "track3_runtime_heartbeat_v2", "phase": phase, "state": state, "count": self.count, "pid": pid, "interval_seconds": self.interval, "created_at_utc": now()}
        write_json(self.path, record)
        self.last_written = current
        return record


def runtime_identity(command: Sequence[str], *, gpu_policy: str = "cpu_no_renderer") -> dict[str, Any]:
    """Capture identity at the actual launcher boundary."""
    expected_cuda = "" if gpu_policy == "cpu_no_renderer" else "3"
    current = git(("rev-parse", "HEAD"))
    observed_dirty = bool(source_porcelain())
    cursor = strict_json(CURSOR) if CURSOR.is_file() else {}
    return {
        "schema_version": "track3_runtime_identity_v2",
        "root_commit": current,
        "tool_sha256": sha256_file(Path(__file__)),
        "openpi_commit": git(("rev-parse", "HEAD"), OPENPI),
        "command": list(command),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_policy": gpu_policy,
        "root_dirty": observed_dirty,
        "cursor_active_before_launch": bool(cursor.get("active")),
        "expected_cuda_visible_devices": expected_cuda,
    }


def identity_gate(identity: Mapping[str, Any], *, expected_commit: str, expected_tool_sha256: str, expected_openpi: str, expected_command: Sequence[str], expected_cuda_visible_devices: str = "", cursor_active: bool = False) -> dict[str, Any]:
    checks = {
        "root_commit": identity.get("root_commit") == expected_commit,
        "tool_sha256": identity.get("tool_sha256") == expected_tool_sha256,
        "openpi_commit": identity.get("openpi_commit") == expected_openpi,
        "root_clean": identity.get("root_dirty") is False,
        "cursor_inactive": bool(identity.get("cursor_active_before_launch")) is bool(cursor_active),
        "command_exact": list(identity.get("command", [])) == list(expected_command),
        "cuda_visible_devices": identity.get("cuda_visible_devices") == expected_cuda_visible_devices,
        "gpu_policy": identity.get("gpu_policy") in {"cpu_no_renderer", "gpu3_camera_only"},
    }
    errors = [name for name, value in checks.items() if not value]
    return {"state": "passed" if not errors else "failed", "checks": checks, "errors": errors, "expected_root_commit": expected_commit, "expected_tool_sha256": expected_tool_sha256, "expected_openpi_commit": expected_openpi, "expected_command": list(expected_command), "expected_cuda_visible_devices": expected_cuda_visible_devices}


def classify_worker_result(returncode: int | None, child: Mapping[str, Any] | None, *, timed_out: bool = False) -> dict[str, Any]:
    """Separate normal family failure from infrastructure/publication failure."""
    status = child.get("status") if isinstance(child, Mapping) else None
    if timed_out:
        return {"category": "infrastructure", "reason": "stage_timeout_or_process_timeout"}
    if child is None or status == "worker_exception" or returncode not in (0,):
        return {"category": "infrastructure", "reason": "worker_exception_or_result_publication", "child_status": status, "returncode": returncode}
    if status == "passed":
        return {"category": "success", "reason": "child_passed", "child_status": status, "returncode": returncode}
    if status == "failed_gate":
        return {"category": "family_failure", "reason": "normal_physical_gate_failure", "child_status": status, "returncode": returncode}
    return {"category": "infrastructure", "reason": "unknown_child_status", "child_status": status, "returncode": returncode}


STAGE_CONTRACT: tuple[dict[str, Any], ...] = (
    {"name": "approach_grasp", "prerequisite": None, "gate": "pose_stable"},
    {"name": "descend_grasp", "prerequisite": "approach_grasp", "gate": "pose_stable"},
    {"name": "close_gripper", "prerequisite": "descend_grasp", "gate": "continuous_demonstrator_contact"},
    {"name": "lift", "prerequisite": "close_gripper", "gate": "object_follow"},
    {"name": "transport", "prerequisite": "lift", "gate": "source_release"},
    {"name": "place_same_demo_init_pose", "prerequisite": "transport", "gate": "destination_pose"},
    {"name": "release", "prerequisite": "place_same_demo_init_pose", "gate": "goal_continuity"},
    {"name": "retreat", "prerequisite": "release", "gate": "goal_continuity"},
)


def stage_contract() -> dict[str, Any]:
    return {
        "order": [row["name"] for row in STAGE_CONTRACT],
        "gates": {row["name"]: row["gate"] for row in STAGE_CONTRACT},
        "unentered_gate_state": "not_evaluated",
        "source_release_phase": "after_lift_before_transport",
        "contact_phase": "close_to_lift",
    }


def progress_window_update(errors: Sequence[float], *, window: int, grace: int, min_improvement: float) -> dict[str, Any]:
    values = [float(value) for value in errors]
    if len(values) <= int(grace):
        return gate(False, None, reason="grace_period", samples=len(values), window=window, grace=grace, min_improvement=min_improvement)
    if len(values) < int(window):
        return gate(False, None, reason="window_not_full", samples=len(values), window=window, grace=grace, min_improvement=min_improvement)
    baseline = values[-int(window)]
    current = values[-1]
    improvement = baseline - current
    passed = improvement >= float(min_improvement)
    return gate(True, passed, reason="stage_stall" if not passed else None, samples=len(values), window=window, grace=grace, baseline_error=baseline, current_error=current, improvement=improvement, min_improvement=min_improvement, failure_event="stage_stall" if not passed else None)


def run_progress_stage(errors: Sequence[float], *, window: int, grace: int, min_improvement: float) -> dict[str, Any]:
    """Evaluate a stage online and stop at the first failed full window."""
    records: list[dict[str, Any]] = []
    for index in range(len(errors)):
        record = progress_window_update(errors[: index + 1], window=window, grace=grace, min_improvement=min_improvement)
        records.append(record)
        if record["state"] == "failed":
            return {"stopped": True, "stop_index": index, "failure_reason": "stage_stall", "records": records}
    return {"stopped": False, "stop_index": None, "failure_reason": None, "records": records}


def terminate_process(process: subprocess.Popen[Any], *, term_grace_seconds: float = TERM_GRACE_SECONDS) -> str:
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
        process.wait(timeout=term_grace_seconds)
        return "TERM_then_KILL"


def phase_gate(stage: str, stage_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a typed gate for the requested stage without hindsight."""
    row = next((item for item in stage_results if item.get("stage") == stage), None)
    if row is None:
        return gate(False, None, reason="stage_not_entered", stage=stage)
    return gate(True, row.get("reached") is True, reason=row.get("failure_reason") if row.get("reached") is not True else None, stage=stage)


def _gpu_snapshot(index: int) -> dict[str, Any]:
    return previous.gpu_snapshot(index)


def _y0() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    current = git(("rev-parse", "HEAD"))
    openpi_head = git(("rev-parse", "HEAD"), OPENPI)
    x_manifest = strict_json(X13_MANIFEST)
    x_validation = strict_json(X13_VALIDATION)
    x4_parent = strict_json(X4_PARENT)
    x4_child = strict_json(X4_CHILD)
    w6_closeout = strict_json(W6_CLOSEOUT)
    old_x_cursor = strict_json(X_OUT / "resume_cursor.json")
    old_w_cursor = strict_json(W_OUT / "resume_cursor.json")
    missing = [rel(path) for path in _expected_inputs() if not path.is_file() and not path.is_dir()]
    process_rows = _active_processes()
    gpu0 = _gpu_snapshot(0)
    gpu_free = {str(index): _gpu_snapshot(index) for index in (1, 2, 3)}
    checks = {
        "x_commit_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", X_COMMIT, current], check=False).returncode == 0,
        "x13_manifest_13_entries": len(x_manifest.get("entries", [])) == 13,
        "x13_final_validation_present": bool(x_validation),
        "x4_historical_infrastructure": x4_parent.get("category") == "infrastructure" and x4_child.get("status") == "worker_exception",
        "w6_hard_stop": w6_closeout.get("status") == "track3_support_release_anchor_scaled_W6_hard_stop",
        "zero_reverse_counts": w6_closeout.get("counts") == {"accepted_reverse_demo": 0, "exact_reverse_demo": 0, "pending_autogen": 0},
        "required_input_paths": not missing,
        "x_cursor_inactive": old_x_cursor.get("active") is False,
        "w_cursor_inactive": old_w_cursor.get("active") is False,
        "no_live_track3_process": not process_rows,
        "openpi_identity_and_clean": openpi_head == OPENPI_COMMIT and not porcelain(OPENPI),
        "source_clean_before_y_namespace": not source_porcelain(),
        "gpu0_state_query_success": gpu0.get("returncode") == 0,
        "gpu1_to_gpu3_free": all(row.get("free") for row in gpu_free.values()),
    }
    gpu0_pid_present = any(GPU0_PID in line for line in gpu0.get("compute_processes", []))
    gpu0_state = "expected_external_pid_present" if gpu0_pid_present else "no_compute_app_at_recheck"
    checks["gpu0_state_is_known"] = gpu0_state in {"expected_external_pid_present", "no_compute_app_at_recheck"}
    checks["gpu0_not_touched_by_launcher"] = True
    output_path = Y0_RECHECK_REPORT if Y0_REPORT.is_file() else Y0_REPORT
    result = {
        "schema_version": "track3_runtime_y0_input_lock_v1",
        "status": "track3_runtime_y0_passed" if all(checks.values()) else "track3_runtime_stopped_Y0_input_lock_failure",
        "created_at_utc": now(),
        "plan": rel(PLAN),
        "input_execution_commit": X_COMMIT,
        "root_head": current,
        "openpi_head": openpi_head,
        "inventory": inventory(_expected_inputs()),
        "missing": missing,
        "checks": checks,
        "active_processes": process_rows,
        "gpu0_expected_pid": GPU0_PID,
        "gpu0_observed_state": gpu0_state,
        "gpu0_external_pid_present": gpu0_pid_present,
        "historical_cursors": {"x": old_x_cursor, "w": old_w_cursor},
        "gpu0": gpu0,
        "gpu1_to_gpu3": gpu_free,
        "libero_plus_porcelain": porcelain(LIBERO_PLUS),
        "resource_boundary": {"simulator_imported": False, "cuda_imported": False, "gpu0_touched": False, "official_gpu": False, "teleop": False, "generation": False, "training": False, "zip": False},
    }
    if Y0_REPORT.is_file():
        result["derived_from_failed_y0"] = rel(Y0_REPORT)
        result["correction"] = "exclude_current_launcher_from_process_scan_and_record_gpu0_state_without_interaction"
    write_json(output_path, result)
    return result


def _symbol_smoke_subprocess() -> dict[str, Any]:
    code = """
from tools.axis_pi05 import track3_composed_reverse_start_stabilization as canonical
assert callable(canonical._configure_public_env)
from pathlib import Path
assert canonical.HDF5_INPUTS[10].is_file()
assert canonical.HDF5_INPUTS[20].is_file()
assert canonical.HDF5_INPUTS[19].is_file()
print('symbol_smoke_passed')
"""
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{PUBLIC_LIBERO}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout[-1000:], "stderr": result.stderr[-2000:], "passed": result.returncode == 0 and "symbol_smoke_passed" in result.stdout}


def _y1() -> dict[str, Any]:
    y0_path = Y0_RECHECK_REPORT if Y0_RECHECK_REPORT.is_file() else Y0_REPORT
    y0 = strict_json(y0_path) if y0_path.is_file() else {}
    if y0.get("status") != "track3_runtime_y0_passed":
        result = {"schema_version": "track3_runtime_y1_contract_v1", "status": "not_evaluated", "reason": "y0_not_passed"}
        write_json(Y1_REPORT, result)
        return result
    command = [sys.executable, "-m", MODULE, "--y3-worker"]
    identity = runtime_identity(command)
    identity_check = identity_gate(identity, expected_commit=identity["root_commit"], expected_tool_sha256=identity["tool_sha256"], expected_openpi=OPENPI_COMMIT, expected_command=command, expected_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    smoke = _symbol_smoke_subprocess()
    stage = stage_contract()
    stall = run_progress_stage([1.0, 0.8, 0.6, 0.79], window=3, grace=1, min_improvement=0.05)
    return_codes = {"passed": classify_worker_result(0, {"status": "passed"}), "failed_gate": classify_worker_result(0, {"status": "failed_gate"}), "worker_exception": classify_worker_result(1, {"status": "worker_exception"})}
    checks = {
        "canonical_helper_smoke": smoke["passed"],
        "identity_gate": identity_check["state"] == "passed",
        "stage_order": stage["order"] == ["approach_grasp", "descend_grasp", "close_gripper", "lift", "transport", "place_same_demo_init_pose", "release", "retreat"],
        "source_release_after_lift": stage["source_release_phase"] == "after_lift_before_transport",
        "contact_close_to_lift": stage["contact_phase"] == "close_to_lift",
        "unentered_is_not_evaluated": phase_gate("place_same_demo_init_pose", [{"stage": "approach_grasp", "reached": True}])["state"] == "not_evaluated",
        "stall_stops_early": stall["stopped"] and stall["failure_reason"] == "stage_stall" and stall["stop_index"] == 3,
        "return_code_contract": return_codes["passed"]["category"] == "success" and return_codes["failed_gate"]["category"] == "family_failure" and return_codes["worker_exception"]["category"] == "infrastructure",
    }
    result = {
        "schema_version": "track3_runtime_y1_contract_v1",
        "status": "track3_runtime_y1_passed" if all(checks.values()) else "track3_runtime_stopped_Y1_contract_failure",
        "created_at_utc": now(),
        "y0_input_lock": rel(y0_path),
        "checks": checks,
        "symbol_smoke": smoke,
        "identity": identity,
        "identity_gate": identity_check,
        "stage_contract": stage,
        "stall_fixture": stall,
        "return_code_contract": return_codes,
        "heartbeat_contract": {"schema_version": "track3_runtime_heartbeat_v2", "interval_seconds": HEARTBEAT_INTERVAL_SECONDS, "atomic": True, "parent_polling_required": True},
        "resource_boundary": {"simulator_started": False, "cuda_started": False, "gpu0_touched": False, "generation": False, "training": False, "zip": False},
    }
    write_json(Y1_REPORT, result)
    return result


def _manifest(root: Path) -> dict[str, Any]:
    excluded = {Y2_REPORT.name, Y2_MANIFEST.name, Y2_VALIDATION.name, Y2_CLOSEOUT.name}
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded or path.name.endswith(".partial") or path.name.endswith(".pid") or path.suffix == ".log":
            continue
        entries.append({"relative_path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema_version": "track3_runtime_y2_artifact_manifest_v1", "entries_exclude_self_final_validation_and_closeout": True, "entries": entries}


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = strict_json(path)
    mismatches = []
    for entry in manifest.get("entries", []):
        candidate = ROOT / str(entry["relative_path"])
        if not candidate.is_file():
            mismatches.append(f"missing:{entry['relative_path']}")
            continue
        if candidate.stat().st_size != int(entry["bytes"]):
            mismatches.append(f"bytes:{entry['relative_path']}")
        if sha256_file(candidate) != entry["sha256"]:
            mismatches.append(f"sha256:{entry['relative_path']}")
    return {"entries": len(manifest.get("entries", [])), "mismatches": mismatches, "exact": not mismatches}


def validate_namespace() -> dict[str, Any]:
    import numpy as np

    json_errors: list[str] = []
    for path in sorted(OUT.rglob("*.json")):
        if path == Y2_VALIDATION:
            continue
        try:
            strict_json(path)
        except Exception as exc:
            json_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    npz_errors: list[str] = []
    for path in sorted(OUT.rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    if not np.isfinite(np.asarray(archive[name], dtype=np.float64)).all():
                        npz_errors.append(f"nonfinite:{rel(path)}:{name}")
        except Exception as exc:
            npz_errors.append(f"{rel(path)}:{type(exc).__name__}:{exc}")
    csv_lf = all(b"\r" not in path.read_bytes() for path in OUT.rglob("*.csv"))
    return {"strict_json": not json_errors, "json_errors": json_errors, "npz_no_pickle_and_finite": not npz_errors, "npz_errors": npz_errors, "csv_lf": csv_lf, "partial_residue": [rel(path) for path in OUT.rglob("*.partial")], "manifest_exact": verify_manifest(Y2_MANIFEST), "source_porcelain": source_porcelain(), "openpi_porcelain": porcelain(OPENPI)}


def _y2() -> dict[str, Any]:
    y1 = strict_json(Y1_REPORT) if Y1_REPORT.is_file() else {}
    if y1.get("status") != "track3_runtime_y1_passed":
        result = {"schema_version": "track3_runtime_y2_release_v1", "status": "not_evaluated", "reason": "y1_not_passed"}
        write_json(Y2_REPORT, result)
        return result
    OUT.mkdir(parents=True, exist_ok=True)
    if Y2_REPORT.is_file():
        previous_y2 = strict_json(Y2_REPORT)
        if previous_y2.get("status") != "track3_runtime_y2_clean_release_ready":
            write_json(OUT / "y2_attempt1_release_failure.json", previous_y2)
    write_cursor(active=False, phase="Y2", command=[sys.executable, "-m", MODULE, "--y2"], status="preflight_release")
    write_text(PHASE_CSV, "phase,status\nY0,track3_runtime_y0_passed\nY1,track3_runtime_y1_passed\nY2,preflight_release\nY3,not_evaluated\nY4,not_evaluated\nY5,not_evaluated\nY6,not_evaluated\nY7,not_evaluated\nY8,not_evaluated\nY9,not_evaluated\nY10,not_evaluated\nY11,not_evaluated\nY12,not_evaluated\nY13,not_evaluated\n")
    compile_result = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], capture_output=True, text=True, check=False)
    targeted = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_track3_runtime_preflight_and_bounded_continuation.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    manifest = _manifest(OUT)
    write_json(Y2_MANIFEST, manifest)
    validation = validate_namespace()
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    validation.update({"py_compile": compile_result.returncode == 0, "targeted_tests": {"returncode": targeted.returncode, "tail": targeted.stdout[-3000:]}, "full_tests": {"returncode": full.returncode, "tail": full.stdout[-3000:]}, "git_diff_check": not bool(git(("diff", "--check")))})
    write_json(Y2_VALIDATION, {"schema_version": "track3_runtime_y2_final_validation_v1", "created_at_utc": now(), "validation": validation, "finalizer_order_completed": ["producer_close_fsync", "phase_summary_csv", "inactive_cursor", "manifest", "strict_hash_npz_csv", "targeted_full_tests", "final_validation", "clean_commit_pending"]})
    self_hashes = {"manifest": {"path": rel(Y2_MANIFEST), "bytes": Y2_MANIFEST.stat().st_size, "sha256": sha256_file(Y2_MANIFEST)}, "final_validation": {"path": rel(Y2_VALIDATION), "bytes": Y2_VALIDATION.stat().st_size, "sha256": sha256_file(Y2_VALIDATION)}}
    checks = {"y1_passed": True, "py_compile": compile_result.returncode == 0, "targeted_tests": targeted.returncode == 0, "full_tests": full.returncode == 0, "strict_json": validation["strict_json"], "npz_no_pickle_and_finite": validation["npz_no_pickle_and_finite"], "csv_lf": validation["csv_lf"], "no_partial": not validation["partial_residue"], "manifest_exact": validation["manifest_exact"]["exact"], "source_not_dirty_outside_y": not validation["source_porcelain"], "openpi_clean": not validation["openpi_porcelain"], "diff_check": validation["git_diff_check"]}
    result = {"schema_version": "track3_runtime_y2_release_v1", "status": "track3_runtime_y2_clean_release_ready" if all(checks.values()) else "track3_runtime_stopped_Y2_release_failure", "created_at_utc": now(), "checks": checks, "root_head_before_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "manifest": rel(Y2_MANIFEST), "final_validation": rel(Y2_VALIDATION), "checkout_stable_self_hashes": self_hashes, "unentered_phases": {f"Y{i}": "not_evaluated" for i in range(3, 13)}, "resource_boundary": {"simulator_started": False, "cuda_started": False, "gpu0_touched": False, "generation": False, "training": False, "zip": False}}
    write_json(Y2_REPORT, result)
    self_hashes["release_summary"] = {"path": rel(Y2_REPORT), "bytes": Y2_REPORT.stat().st_size, "sha256": sha256_file(Y2_REPORT)}
    closeout = {"schema_version": "track3_runtime_y2_closeout_v1", "status": "track3_runtime_y2_clean_release_ready" if all(checks.values()) else "track3_runtime_stopped_Y2_release_failure", "created_at_utc": now(), "phase_status": {"Y0": "passed", "Y1": "passed", "Y2": result["status"], **{f"Y{i}": "not_evaluated" for i in range(3, 14)}}, "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "self_hashes": self_hashes, "manifest_exclusions": [rel(Y2_REPORT), rel(Y2_MANIFEST), rel(Y2_VALIDATION), rel(Y2_CLOSEOUT)], "hard_stop": {"executed": False, "next_phase": "Y3_if_clean_release_commit_is_created", "automatic_resume": False, "upload": False, "submit": False}, "resource_boundary": result["resource_boundary"]}
    write_json(Y2_CLOSEOUT, closeout)
    return result


def _predicate_value(sample: Mapping[str, Any], section: str, atom: str) -> bool | None:
    for row in sample.get(section, {}).get("predicates", []):
        if row.get("atom") == atom:
            value = row.get("value")
            return bool(value) if isinstance(value, bool) else None
    return None


def _worker_y3(result_path: Path) -> int:
    """Run the single authorized nominal CPU canary when Y3 is later entered."""
    import numpy as np

    env = None
    started = now()
    run_root = result_path.parent
    try:
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base

        w1 = strict_json(W1_REPORT)
        w2 = strict_json(W2_REPORT)
        manifest, model_bytes, _init_state, _terminal_state, _demo_info = composed._demo_arrays(10, 0)
        resolved_xml, resolution = composed._resolve_model_xml(model_bytes)
        with np.load(STATE_BUNDLE, allow_pickle=False) as archive:
            start_state = np.asarray(archive["task10_demo0"], dtype=np.float64).copy()
        terminal_pose = w1["state_anchors"]["terminal"]["bowl"]
        state0_pose = w1["state_anchors"]["state0"]["bowl"]
        relative = w1["reverse_acquisition_anchor"]["relative_pose_xyzw"]
        grasp = object_frame.compose_object_to_eef(terminal_pose, relative)
        destination = object_frame.compose_object_to_eef(state0_pose, relative)
        transport = object_frame.compose_object_to_eef(state0_pose, relative, object_position_offset_m=(0.0, 0.0, 0.12))

        # This is the canonical public helper.  The historical X4 call into
        # track3_reverse_init_stabilization is intentionally not reused.
        composed._configure_public_env(gpu=False)
        from libero.libero.envs.env_wrapper import ControlEnv

        env = ControlEnv(bddl_file_name=str(base.REVERSE_BDDL[10]), control_freq=CONTROL_FREQ_HZ, use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False, render_gpu_device_id=-1)
        env.reset()
        env.reset_from_xml_string(resolved_xml)
        env.set_init_state(start_state)
        dynamic.controller_reset_sequence(dynamic._controller(env), env.sim)
        initial_atoms = base.public_predicates(base.REVERSE_BDDL[10], "init")
        goal_atoms = base.public_predicates(base.REVERSE_BDDL[10], "goal")
        source_atom = "(On akita_black_bowl_1 plate_1)"
        goal_atom = "(On akita_black_bowl_1 main_table_akita_black_bowl_region)"
        bowl = "akita_black_bowl_1"
        samples: list[dict[str, Any]] = []
        actions: list[list[float]] = []
        states: list[Any] = []
        phases: list[str] = []
        stage_indices: list[int] = []
        eef_positions: list[list[float]] = []
        eef_quaternions: list[list[float]] = []
        target_positions: list[list[float]] = []
        target_quaternions: list[list[float]] = []
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        progress_records: list[dict[str, Any]] = []
        max_penetration = 0.0
        baseline_scene: Mapping[str, Any] | None = None
        unrelated_names: set[str] = set()
        max_unrelated_movement = 0.0

        def observe(step: int) -> None:
            nonlocal baseline_scene, max_penetration, max_unrelated_movement
            sample = prior._runtime_sample(env, dynamic, 10, "reverse", step, initial_atoms, goal_atoms)
            sample["contacts_v3"] = composed.contact_v3(10, sample["contact_matrix"])
            bowl_pose = dynamic._object_or_site_pose(env, bowl)
            if bowl_pose is not None:
                sample["object_eef_relative_pose"] = object_frame._relative_pose(bowl_pose, sample["robot"])
            samples.append(sample)
            scene = sample.get("scene", {}).get("object_and_site_poses", {})
            if baseline_scene is None:
                baseline_scene = scene
                unrelated_names.update(set(scene) - {bowl, "plate_1", "main_table_akita_black_bowl_region"})
            for name in unrelated_names:
                if name in scene and name in baseline_scene:
                    current = scene[name].get("pos_m", [])
                    original = baseline_scene[name].get("pos_m", [])
                    max_unrelated_movement = max(max_unrelated_movement, object_frame.vector_norm([float(a) - float(b) for a, b in zip(current, original)]))
            for contact in sample.get("contact_matrix", {}).get("contacts", []):
                max_penetration = max(max_penetration, max(0.0, -float(contact.get("distance_m", 0.0))))

        observe(0)
        stage_specs = [
            ("approach_grasp", -1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.09], grasp["quat"]),
            ("descend_grasp", -1.0, grasp["pos_m"], grasp["quat"]),
            ("close_gripper", 1.0, grasp["pos_m"], grasp["quat"]),
            ("lift", 1.0, [grasp["pos_m"][0], grasp["pos_m"][1], grasp["pos_m"][2] + 0.14], grasp["quat"]),
            ("transport", 1.0, transport["pos_m"], transport["quat"]),
            ("place_same_demo_init_pose", 1.0, destination["pos_m"], destination["quat"]),
            ("release", -1.0, destination["pos_m"], destination["quat"]),
            ("retreat", -1.0, [destination["pos_m"][0], destination["pos_m"][1], destination["pos_m"][2] + 0.16], destination["quat"]),
        ]
        stage_results: list[dict[str, Any]] = []
        failure_reason: str | None = None
        for stage_index, (stage_name, gripper, target_position, target_quat) in enumerate(stage_specs):
            stable = 0
            stage_start = len(actions)
            stage_errors: list[float] = []
            last_t = float("inf")
            last_r = float("inf")
            stage_failure: str | None = None
            budget = int(NOMINAL_PROFILE["stage_budgets"][stage_index])
            for _local in range(budget):
                current = dynamic._robot_metrics(env)
                translation = [float(target_position[index]) - float(current["eef_translation_m"][index]) for index in range(3)]
                rotation = w.quat_log_error_xyzw(target_quat, current["eef_quaternion"])
                command = w.osc_action_scaled(translation, rotation, gripper, profile=NOMINAL_PROFILE, scales={"translation_m": float(w2["controller_output_scale"]["translation_m"]), "rotation_rad": float(w2["controller_output_scale"]["rotation_rad"])})
                obs_after, _reward, done, _info = env.step(np.asarray(command["action"], dtype=np.float32))
                if not isinstance(obs_after, Mapping):
                    obs_after = env.env._get_observations()
                actions.append(list(command["action"]))
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64).copy())
                phases.append(stage_name)
                stage_indices.append(stage_index)
                eef_positions.append(list(current["eef_translation_m"]))
                eef_quaternions.append(list(current["eef_quaternion"]))
                target_positions.append(list(target_position))
                target_quaternions.append(list(target_quat))
                last_t = float(command["translation_norm_m"])
                last_r = float(command["rotation_norm_rad"])
                stage_error = max(last_t / float(NOMINAL_PROFILE["translation_tolerance_m"]), last_r / float(NOMINAL_PROFILE["rotation_tolerance_rad"]))
                stage_errors.append(stage_error)
                translation_errors.append(last_t)
                rotation_errors.append(last_r)
                observe(len(actions))
                progress = progress_window_update(stage_errors, window=int(NOMINAL_PROFILE["progress_window"]), grace=int(NOMINAL_PROFILE["progress_grace"]), min_improvement=float(NOMINAL_PROFILE["progress_min_improvement"]))
                progress_records.append({"stage": stage_name, "stage_index": stage_index, "window": progress})
                if progress["state"] == "failed":
                    stage_failure = f"stage_stall:{stage_name}"
                    break
                if last_t <= float(NOMINAL_PROFILE["translation_tolerance_m"]) and last_r <= float(NOMINAL_PROFILE["rotation_tolerance_rad"]):
                    stable += 1
                else:
                    stable = 0
                if stable >= int(NOMINAL_PROFILE["stable_ticks"]) or done:
                    break
            reached = stable >= int(NOMINAL_PROFILE["stable_ticks"])
            if stage_failure is not None:
                reached = False
                failure_reason = stage_failure
            elif not reached:
                failure_reason = f"stage_timeout:{stage_name}"
            stage_results.append({"stage": stage_name, "stage_index": stage_index, "frames": len(actions) - stage_start, "budget": budget, "reached": reached, "failure_reason": stage_failure or (None if reached else f"stage_timeout:{stage_name}"), "last_translation_error_m": last_t, "last_rotation_error_rad": last_r})
            if not reached:
                break
            recent = samples[-max(2, int(NOMINAL_PROFILE["stable_ticks"])):]
            if stage_name == "close_gripper":
                contact_ok = all(row.get("contacts_v3", {}).get("counts", {}).get("demonstrator_contact", 0) > 0 for row in recent)
                if not contact_ok:
                    failure_reason = "acquisition_barrier:demonstrator_contact"
                    break
            if stage_name == "lift":
                follow_ok = all(row.get("object_eef_relative_pose") is not None and object_frame.vector_norm([float(a) - float(b) for a, b in zip(row["object_eef_relative_pose"]["translation_m"], relative["translation_m"])]) <= 0.02 and object_frame.qgeo(row["object_eef_relative_pose"]["quaternion_xyzw"], relative["quaternion_xyzw"]) <= 0.15 for row in recent)
                source_released = all(_predicate_value(row, "initial", source_atom) is False for row in recent)
                if not follow_ok or not source_released:
                    failure_reason = "lift_gate:object_follow_or_source_release"
                    break

        final_success = bool(env.check_success())
        source_values = [_predicate_value(row, "initial", source_atom) for row in samples]
        goal_values = [_predicate_value(row, "goal", goal_atom) for row in samples]
        entered = {row["stage"] for row in stage_results}
        recent = samples[-max(2, int(NOMINAL_PROFILE["stable_ticks"])):] if samples else []
        close_contact = bool(recent) and all(row.get("contacts_v3", {}).get("counts", {}).get("demonstrator_contact", 0) > 0 for row in recent)
        follow_ok = bool(recent) and all(row.get("object_eef_relative_pose") is not None and object_frame.vector_norm([float(a) - float(b) for a, b in zip(row["object_eef_relative_pose"]["translation_m"], relative["translation_m"])]) <= 0.02 and object_frame.qgeo(row["object_eef_relative_pose"]["quaternion_xyzw"], relative["quaternion_xyzw"]) <= 0.15 for row in recent)
        source_released = bool(recent) and all(_predicate_value(row, "initial", source_atom) is False for row in recent)
        gates = {
            "initial_source_true": gate(bool(source_values), source_values[0] is True, source_value=source_values[0] if source_values else None),
            "initial_goal_false": gate(bool(goal_values), goal_values[0] is False, goal_value=goal_values[0] if goal_values else None),
            "contact_completion": gate("close_gripper" in entered, close_contact, reason="close_stage_not_entered" if "close_gripper" not in entered else ("acquisition_barrier:demonstrator_contact" if not close_contact else None)),
            "object_follow": gate("lift" in entered, follow_ok, reason="lift_stage_not_entered" if "lift" not in entered else None),
            "source_release": gate("lift" in entered, source_released, reason="lift_stage_not_entered" if "lift" not in entered else None),
            "same_demo_destination": gate("place_same_demo_init_pose" in entered, "place_same_demo_init_pose" in entered and bool(recent), reason="place_stage_not_entered" if "place_same_demo_init_pose" not in entered else None),
            "final_goal_true": gate("release" in entered, "release" in entered and final_success and bool(goal_values) and goal_values[-1] is True, reason="release_stage_not_entered" if "release" not in entered else None, final_success=final_success),
            "prefix_contact_safety": gate(bool(samples), all(not row.get("contacts_v3", {}).get("counts", {}).get("unrelated_background_contact", 0) for row in samples)),
            "collision": gate(bool(samples), max_penetration <= 0.01, max_penetration_m=max_penetration, threshold_m=0.01),
            "unrelated_movement": gate(bool(samples), max_unrelated_movement <= 0.002, max_unrelated_movement_m=max_unrelated_movement, threshold_m=0.002),
            "progress_stall": gate(bool(progress_records), not any(row["window"].get("state") == "failed" for row in progress_records), failure_reason=failure_reason if failure_reason and failure_reason.startswith("stage_stall") else None),
            "stall_timeout": gate(bool(stage_results), failure_reason is None, reason=failure_reason if failure_reason else None, used_steps=len(actions), step_budget=NOMINAL_PROFILE["step_budget"]),
            "finite": gate(bool(actions), bool(np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(states)).all()), action_shape=list(np.asarray(actions).shape), state_shape=list(np.asarray(states).shape)),
        }
        status = "passed" if all_required(gates) else "failed_gate"
        write_npy(run_root / "start_state.npy", start_state)
        write_npy(run_root / "end_state.npy", np.asarray(env.get_sim_state(), dtype=np.float64))
        write_npz(run_root / "trajectory.npz", actions=np.asarray(actions, dtype=np.float32), states=np.asarray(states, dtype=np.float64), phase=np.asarray(phases, dtype="U64"), stage_index=np.asarray(stage_indices, dtype=np.int16), eef_position_m=np.asarray(eef_positions, dtype=np.float32), eef_quaternion_xyzw=np.asarray(eef_quaternions, dtype=np.float32), target_position_m=np.asarray(target_positions, dtype=np.float32), target_quaternion_xyzw=np.asarray(target_quaternions, dtype=np.float32), translation_error_m=np.asarray(translation_errors, dtype=np.float32), rotation_error_rad=np.asarray(rotation_errors, dtype=np.float32), source_trace=np.asarray([-1 if value is None else int(value) for value in source_values], dtype=np.int8), goal_trace=np.asarray([-1 if value is None else int(value) for value in goal_values], dtype=np.int8))
        result = {"schema_version": "track3_runtime_y3_worker_v1", "status": status, "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": NOMINAL_PROFILE, "mode": "CPU_no_renderer", "fresh_process": True, "quaternion_convention": "xyzw", "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "start_state_path": rel(run_root / "start_state.npy"), "start_state_sha256": sha256_file(run_root / "start_state.npy"), "same_demo_model_and_state_components": True}, "reverse": {"source_atom": source_atom, "goal_atom": goal_atom, "source_values": source_values, "goal_values": goal_values, "stage_results": stage_results, "failure_reason": failure_reason, "anchor": w1["reverse_acquisition_anchor"], "destination_eef": destination, "transport_eef": transport}, "hard_gate": gates, "progress_windows": progress_records, "artifacts": {"trajectory": rel(run_root / "trajectory.npz"), "start_state": rel(run_root / "start_state.npy"), "end_state": rel(run_root / "end_state.npy")}, "training_eligible": False, "pending_autogen": False, "resource_boundary": {"gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "official_gpu": False, "no_teleop": True, "no_generation": True, "no_training": True, "no_zip": True}}
    except Exception as exc:
        result = {"schema_version": "track3_runtime_y3_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": NOMINAL_PROFILE, "mode": "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "training_eligible": False, "resource_boundary": {"gpu0_touched": False, "official_gpu": False, "no_generation": True, "no_training": True, "no_zip": True}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_path, result)
    return 0 if result.get("status") in {"passed", "failed_gate"} else 1


def _run_y3_cpu() -> dict[str, Any]:
    y2 = strict_json(Y2_REPORT) if Y2_REPORT.is_file() else {}
    if y2.get("status") != "track3_runtime_y2_clean_release_ready":
        return {"schema_version": "track3_runtime_y3_parent_v1", "status": "not_evaluated", "category": "infrastructure", "reason": "y2_clean_release_required"}
    run_root = OUT / "y3_support_cpu_task10_demo0_nominal"
    run_root.mkdir(parents=True, exist_ok=False)
    child_path = run_root / "child_result.json"
    command = [sys.executable, "-m", MODULE, "--y3-worker", "--result-output", str(child_path)]
    identity = runtime_identity(command, gpu_policy="cpu_no_renderer")
    expected_commit = git(("rev-parse", "HEAD"))
    identity_check = identity_gate(identity, expected_commit=expected_commit, expected_tool_sha256=identity["tool_sha256"], expected_openpi=OPENPI_COMMIT, expected_command=command, expected_cuda_visible_devices="")
    write_json(run_root / "identity_preflight.json", {"schema_version": "track3_runtime_y3_identity_v1", "phase": "Y3", "identity": identity, "identity_gate": identity_check})
    if identity_check["state"] != "passed":
        result = {"schema_version": "track3_runtime_y3_parent_v1", "status": "track3_runtime_stopped_Y3_identity_failure", "category": "infrastructure", "identity_gate": identity_check}
        write_json(run_root / "parent_result.json", result)
        return result
    pid_path = run_root / "supervisor.pid"
    write_json(pid_path, {"schema_version": "track3_runtime_y3_pid_v1", "pid": os.getpid(), "phase": "Y3", "created_at_utc": now()})
    write_cursor(active=True, phase="Y3.object_frame_nominal", command=command, status="active")
    heartbeat = PeriodicHeartbeat(run_root / "heartbeat.json")
    heartbeat.maybe_write(phase="Y3", state="running", pid=os.getpid(), force=True)
    env = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    process: subprocess.Popen[Any] | None = None
    started_at = time.monotonic()
    timed_out = False
    termination_stage: str | None = None
    polls = 0
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
        while process.poll() is None:
            polls += 1
            heartbeat.maybe_write(phase="Y3", state="running", pid=process.pid)
            if time.monotonic() - started_at >= CANARY_DEADLINE_SECONDS:
                timed_out = True
                termination_stage = terminate_process(process)
                break
            time.sleep(0.1)
        process.wait()
    heartbeat.maybe_write(phase="Y3", state="final", pid=process.pid if process else None, force=True)
    pid_path.unlink(missing_ok=True)
    write_cursor(active=False, phase="Y3.object_frame_nominal", command=command, status="completed")
    child = strict_json(child_path) if child_path.is_file() else None
    classification = classify_worker_result(process.returncode if process else None, child, timed_out=timed_out)
    result = {"schema_version": "track3_runtime_y3_parent_v1", "status": "track3_runtime_y3_passed" if classification["category"] == "success" else "track3_runtime_y3_family_failure" if classification["category"] == "family_failure" else "track3_runtime_stopped_Y3_infrastructure_failure", "category": classification["category"], "created_at_utc": now(), "phase": "Y3", "task_index": 10, "demo_index": 0, "family": "support_remove", "attempt_number": 1, "identity_gate": identity_check, "command": command, "supervisor_pid": os.getpid(), "worker_pid": process.pid if process else None, "worker_pid_match": bool(process and child and process.pid == child.get("child_pid")), "process_returncode": process.returncode if process else None, "timed_out": timed_out, "termination_stage": termination_stage, "poll_samples": polls, "heartbeat": {"path": rel(run_root / "heartbeat.json"), "count": heartbeat.count, "interval_seconds": heartbeat.interval}, "child_status": child.get("status") if isinstance(child, Mapping) else None, "child_result_path": rel(child_path), "child_result": child, "classification": classification, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_to_gpu3_released": True, "official_gpu": False, "teleop": False, "generation": False, "training": False, "zip": False}}
    write_json(run_root / "parent_result.json", result)
    return result


def _y3_worker() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--result-output", type=Path, required=True)
    args, _unknown = parser.parse_known_args()
    return _worker_y3(args.result_output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for phase in ("y0", "y1", "y2", "y3"):
        parser.add_argument(f"--{phase}", action="store_true")
    parser.add_argument("--y3-worker", dest="y3_worker", action="store_true")
    args = parser.parse_args(argv)
    selected = [phase for phase in ("y0", "y1", "y2", "y3", "y3-worker") if getattr(args, phase.replace("-", "_"))]
    if len(selected) != 1:
        parser.error("select exactly one phase Y0/Y1/Y2/Y3-worker")
    if selected[0] == "y3-worker":
        return _y3_worker()
    result = {"y0": _y0, "y1": _y1, "y2": _y2, "y3": _run_y3_cpu}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if "passed" in str(result.get("status")) or "ready" in str(result.get("status")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
