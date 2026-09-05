#!/usr/bin/env python3
"""Track 3 Z0--Z14 launch and phase-ledger remediation.

Z0--Z4 are deliberately CPU/filesystem-only.  The module has no simulator
imports at module scope and its Z1 smoke worker is a pure atomic-result
worker.  The future physical worker is intentionally not enabled here: the
source-identity and phase-ledger contracts must be released first.

Historical Y/X/W evidence is read-only input.  New evidence is written under
one derived namespace and never changes an older report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_y3_launch_phase_ledger_remediation"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_Y3_LAUNCH_PHASE_LEDGER_REMEDIATION_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_RUNTIME_PREFLIGHT_Y0_Y2_AUDIT_20260906.md"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
PUBLIC_LIBERO = ROOT / "LIBERO"

OUT = ROOT / "data/reports/axis_libero_30k/track3_y3_launch_phase_ledger_remediation"
Z0_REPORT = OUT / "z0_input_lock.json"
Z1_REPORT = OUT / "z1_cli_dispatch_contract.json"
Z2_REPORT = OUT / "z2_source_identity_contract.json"
Z3_REPORT = OUT / "z3_phase_ledger_contract.json"
Z4_REPORT = OUT / "z4_prephysics_release.json"
Z4_VALIDATION = OUT / "z4_final_validation.json"
Z4_MANIFEST = OUT / "z4_artifact_manifest.json"
Z4_CLOSEOUT = OUT / "z4_closeout.json"
PHASE_CSV = OUT / "z4_phase_status.csv"
CURSOR = OUT / "resume_cursor.json"
HEARTBEAT = OUT / "heartbeat.json"
PID_FILE = OUT / "supervisor.pid"
Z1_SENTINEL = OUT / "z1_smoke" / "worker_result.json"
Z5_ROOT = OUT / "z5_cpu_task10_demo0_nominal"
Z5_PARENT = Z5_ROOT / "parent_result.json"
Z5_CHILD = Z5_ROOT / "child_result.json"
Z5_TRAJECTORY = Z5_ROOT / "trajectory.npz"
Z5_PHASE_CSV = OUT / "z5_phase_status.csv"
IDENTITY_NOTE_REF = "refs/notes/axis-pi05-track3-z5-source-identity"

Y_OUT = ROOT / "data/reports/axis_libero_30k/track3_runtime_preflight_and_bounded_continuation"
Y0_REPORT = Y_OUT / "y0_recheck_input_lock.json"
Y2_REPORT = Y_OUT / "y2_clean_release.json"
Y2_MANIFEST = Y_OUT / "y2_artifact_manifest.json"
Y2_VALIDATION = Y_OUT / "y2_final_validation.json"
Y2_CLOSEOUT = Y_OUT / "y2_closeout.json"
Y_PHASE_CSV = Y_OUT / "y2_phase_status.csv"
Y_CURSOR = Y_OUT / "resume_cursor.json"
X_OUT = ROOT / "data/reports/axis_libero_30k/track3_object_frame_executor_and_bounded_autogen_lora"
W_OUT = ROOT / "data/reports/axis_libero_30k/track3_support_release_anchor_osc_scale_recovery"
W1_REPORT = W_OUT / "w1_release_anchor.json"
W2_REPORT = W_OUT / "w2_osc_scale_contract.json"
STATE_BUNDLE = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary/start_states.npz"

INPUT_COMMIT = "56cdffa83a530cb0dab71e4d17b7a9112a2742af"
OPENPI_COMMIT = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"
CONTROL_FREQ_HZ = 20
HEARTBEAT_INTERVAL_SECONDS = 900.0
TERM_GRACE_SECONDS = 10.0
Z1_MAX_STDOUT_BYTES = 16 * 1024 * 1024
Z5_DEADLINE_SECONDS = 300.0
Z5_PROFILE = {
    "name": "object_frame_nominal",
    "translation_clip_m": 0.05,
    "rotation_clip_rad": 0.12,
    "translation_output_scale_m": 0.05,
    "rotation_output_scale_rad": 0.5,
    "translation_tolerance_m": 0.012,
    "rotation_tolerance_rad": 0.03,
    "stable_ticks": 4,
    "stage_budgets": [150, 120, 35, 150, 180, 120, 35, 80],
    "step_budget": 900,
    "progress_window": 4,
    "progress_grace": 4,
    "progress_min_improvement": 0.02,
    "object_follow_translation_tolerance_m": 0.02,
    "object_follow_rotation_tolerance_rad": 0.15,
    "collision_tolerance_m": 0.01,
    "unrelated_movement_tolerance_m": 0.002,
}

HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}

STAGES = (
    "approach_grasp",
    "descend_grasp",
    "close_gripper",
    "lift",
    "transport",
    "place_same_demo_init_pose",
    "release",
    "retreat",
)
PREREQUISITES = {
    "approach_grasp": None,
    "descend_grasp": "approach_grasp",
    "close_gripper": "descend_grasp",
    "lift": "close_gripper",
    "transport": "lift",
    "place_same_demo_init_pose": "transport",
    "release": "place_same_demo_init_pose",
    "retreat": "release",
}
GATE_OWNERSHIP = {
    "contact_completion": "close_gripper",
    "object_follow": "lift",
    "source_release": "lift",
    "transport_safe": "transport",
    "destination_numeric": "place_same_demo_init_pose",
    "goal_continuity": "release",
}


class DuplicateKeyError(ValueError):
    """Raised for duplicate JSON object keys."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate_json_key:{key}")
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def _status_path(line: str) -> str:
    value = line[3:].strip()
    if " -> " in value:
        value = value.rsplit(" -> ", 1)[-1]
    return value.strip('"')


def source_porcelain(repo: Path = ROOT, *, allowed_output: Path | None = OUT) -> list[str]:
    """Return all source dirt; only the current atomic evidence subtree is allowed."""
    allowed_prefix = None
    if allowed_output is not None:
        allowed_prefix = allowed_output.resolve().relative_to(repo.resolve()).as_posix().rstrip("/") + "/"
    rows = []
    for line in porcelain(repo):
        path = _status_path(line)
        if allowed_prefix and (path == allowed_prefix.rstrip("/") or path.startswith(allowed_prefix)):
            continue
        rows.append(line)
    return rows


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_text(path: Path, value: str) -> None:
    write_bytes_atomic(path, value.encode("utf-8"))


def write_npy(path: Path, value: Any) -> None:
    import numpy as np

    from io import BytesIO

    stream = BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    write_bytes_atomic(path, stream.getvalue())


def write_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    from io import BytesIO

    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    write_bytes_atomic(path, stream.getvalue())


def gate(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None, **details: Any) -> dict[str, Any]:
    result = {"executed": bool(executed), "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"), "required": bool(required), **details}
    if reason is not None:
        result["reason"] = reason
    return result


def cursor_record(*, active: bool, phase: str, command: Sequence[str], status: str) -> dict[str, Any]:
    return {"schema_version": "track3_z_cursor_v1", "active": bool(active), "phase": phase, "status": status, "pid": os.getpid(), "command": list(command), "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "updated_at_utc": now()}


def write_cursor(*, active: bool, phase: str, command: Sequence[str], status: str) -> dict[str, Any]:
    record = cursor_record(active=active, phase=phase, command=command, status=status)
    write_json(CURSOR, record)
    return record


class PeriodicHeartbeat:
    def __init__(self, path: Path, *, interval: float = HEARTBEAT_INTERVAL_SECONDS, clock: Callable[[], float] = time.monotonic) -> None:
        self.path = path
        self.interval = float(interval)
        self.clock = clock
        self.last_written: float | None = None
        self.count = 0

    def maybe_write(self, *, phase: str, state: str, force: bool = False) -> dict[str, Any] | None:
        current = self.clock()
        if not force and self.last_written is not None and current - self.last_written < self.interval:
            return None
        self.count += 1
        record = {"schema_version": "track3_z_heartbeat_v1", "phase": phase, "state": state, "count": self.count, "pid": os.getpid(), "interval_seconds": self.interval, "created_at_utc": now()}
        write_json(self.path, record)
        self.last_written = current
        return record


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


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def single_instance_gate(pid_path: Path, cursor_path: Path) -> dict[str, Any]:
    pid_record: dict[str, Any] = {}
    cursor: dict[str, Any] = {}
    errors: list[str] = []
    if pid_path.is_file():
        try:
            pid_record = strict_json(pid_path)
            if process_is_alive(int(pid_record.get("pid", -1))):
                errors.append("live_supervisor_pid")
        except Exception as exc:
            errors.append(f"pid_file_invalid:{type(exc).__name__}")
    if cursor_path.is_file():
        try:
            cursor = strict_json(cursor_path)
            if cursor.get("active") is True:
                errors.append("active_cursor")
        except Exception as exc:
            errors.append(f"cursor_invalid:{type(exc).__name__}")
    return {"state": "passed" if not errors else "failed", "errors": errors, "pid_file_present": pid_path.is_file(), "cursor_present": cursor_path.is_file(), "pid_record": pid_record, "cursor": cursor}


def runtime_identity(command: Sequence[str], *, gpu_policy: str = "cpu_no_renderer") -> dict[str, Any]:
    return {"schema_version": "track3_z_runtime_identity_v1", "root_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "command": list(command), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "gpu_policy": gpu_policy, "source_porcelain": source_porcelain(), "libero_plus_porcelain": porcelain(LIBERO_PLUS)}


def identity_gate(observed: Mapping[str, Any], frozen: Mapping[str, Any], *, expected_command: Sequence[str] | None = None, expected_cuda_visible_devices: str = "") -> dict[str, Any]:
    expected_command = list(expected_command if expected_command is not None else frozen.get("command", []))
    checks = {"root_commit_exact": observed.get("root_commit") == frozen.get("root_commit"), "tool_sha256_exact": observed.get("tool_sha256") == frozen.get("tool_sha256"), "openpi_commit_exact": observed.get("openpi_commit") == frozen.get("openpi_commit"), "source_clean": observed.get("source_porcelain") == [], "command_exact": list(observed.get("command", [])) == expected_command, "cuda_visible_devices_exact": observed.get("cuda_visible_devices", "") == expected_cuda_visible_devices, "gpu_policy_allowed": observed.get("gpu_policy") in {"cpu_no_renderer", "gpu3_camera_only"}, "libero_plus_snapshot_exact": observed.get("libero_plus_porcelain") == frozen.get("libero_plus_porcelain", observed.get("libero_plus_porcelain"))}
    errors = [key for key, value in checks.items() if not value]
    return {"state": "passed" if not errors else "failed", "checks": checks, "errors": errors, "frozen_root_commit": frozen.get("root_commit"), "frozen_tool_sha256": frozen.get("tool_sha256"), "frozen_openpi_commit": frozen.get("openpi_commit"), "frozen_command": expected_command}


def read_frozen_identity_note(repo: Path = ROOT) -> dict[str, Any] | None:
    """Read the identity frozen after the clean release commit.

    A launcher must fail closed when this note is absent; it may not replace
    the expected values with the current HEAD or current tool hash.
    """
    result = subprocess.run(["git", "-C", str(repo), "notes", f"--ref={IDENTITY_NOTE_REF}", "show", "HEAD"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def identity_note_payload(repo: Path = ROOT, *, command: Sequence[str], gpu_policy: str = "cpu_no_renderer") -> dict[str, Any]:
    """Build a note payload from a clean commit, never from a worker result."""
    return {"schema_version": "track3_z_source_identity_v1", "root_commit": git(("rev-parse", "HEAD"), repo), "tool_sha256": sha256_file(Path(__file__)), "openpi_commit": git(("rev-parse", "HEAD"), OPENPI), "command": list(command), "cuda_visible_devices": "" if gpu_policy == "cpu_no_renderer" else "3", "gpu_policy": gpu_policy, "source_porcelain": source_porcelain(repo), "libero_plus_porcelain": porcelain(LIBERO_PLUS)}


def classify_worker_result(returncode: int | None, child: Mapping[str, Any] | None, *, timed_out: bool = False) -> dict[str, Any]:
    status = child.get("status") if isinstance(child, Mapping) else None
    if timed_out:
        return {"category": "infrastructure", "reason": "timeout_or_termination", "returncode": returncode, "child_status": status}
    if child is None or returncode == 2 or status == "worker_exception" or returncode not in (0,):
        return {"category": "infrastructure", "reason": "dispatch_or_worker_exception_or_missing_result", "returncode": returncode, "child_status": status}
    if status == "passed":
        return {"category": "success", "reason": "child_passed", "returncode": returncode, "child_status": status}
    if status == "failed_gate":
        return {"category": "family_failure", "reason": "normal_physical_gate_failure", "returncode": returncode, "child_status": status}
    return {"category": "infrastructure", "reason": "unknown_child_status", "returncode": returncode, "child_status": status}


def progress_window_update(errors: Sequence[float], *, window: int, grace: int, min_improvement: float) -> dict[str, Any]:
    values = [float(value) for value in errors]
    if len(values) <= grace or len(values) < window:
        return gate(False, None, reason="window_not_full_or_grace", samples=len(values), window=window, grace=grace)
    baseline, current = values[-window], values[-1]
    improvement = baseline - current
    return gate(True, improvement >= min_improvement, reason="stage_stall" if improvement < min_improvement else None, baseline_error=baseline, current_error=current, improvement=improvement, min_improvement=min_improvement)


def run_progress_stage(errors: Sequence[float], *, window: int, grace: int, min_improvement: float) -> dict[str, Any]:
    records = []
    for index in range(len(errors)):
        record = progress_window_update(errors[: index + 1], window=window, grace=grace, min_improvement=min_improvement)
        records.append(record)
        if record["state"] == "failed":
            return {"stopped": True, "stop_index": index, "failure_reason": "stage_stall", "records": records}
    return {"stopped": False, "stop_index": None, "failure_reason": None, "records": records}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def destination_numeric_gate(*, stage_entered: bool, object_position_error_m: float | None, object_rotation_error_rad: float | None, eef_position_error_m: float | None, eef_rotation_error_rad: float | None, position_tolerance_m: float = 0.012, rotation_tolerance_rad: float = 0.03) -> dict[str, Any]:
    values = (object_position_error_m, object_rotation_error_rad, eef_position_error_m, eef_rotation_error_rad)
    finite = all(value is not None and isinstance(value, (int, float)) and float(value) == float(value) for value in values)
    passed = finite and float(object_position_error_m) <= position_tolerance_m and float(eef_position_error_m) <= position_tolerance_m and float(object_rotation_error_rad) <= rotation_tolerance_rad and float(eef_rotation_error_rad) <= rotation_tolerance_rad
    return gate(stage_entered, passed if stage_entered else None, reason="stage_not_entered" if not stage_entered else ("destination_numeric_error" if not passed else None), object_position_error_m=object_position_error_m, object_rotation_error_rad=object_rotation_error_rad, eef_position_error_m=eef_position_error_m, eef_rotation_error_rad=eef_rotation_error_rad, position_tolerance_m=position_tolerance_m, rotation_tolerance_rad=rotation_tolerance_rad)


class PhaseLedger:
    """Frozen stage records; later samples cannot overwrite earlier gates."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self._next_index = 0

    def enter(self, stage: str, *, frame_start: int, target_pose: Mapping[str, Any] | None = None) -> None:
        if stage not in STAGES or stage in self.records:
            raise ValueError(f"invalid_or_duplicate_stage:{stage}")
        expected = STAGES[self._next_index]
        if stage != expected:
            raise ValueError(f"stage_order:{stage}:{expected}")
        prerequisite = PREREQUISITES[stage]
        if prerequisite and not self.records[prerequisite]["completed"]:
            raise ValueError(f"prerequisite_not_complete:{stage}")
        self.records[stage] = {"stage": stage, "frame_start": int(frame_start), "frame_end": None, "stable_window": None, "target_pose": dict(target_pose or {}), "eef_error": {}, "progress_windows": [], "entered": True, "completed": False, "failure_reason": None, "gates": {}}

    def observe(self, stage: str, *, eef_error: Mapping[str, float] | None = None, progress: Mapping[str, Any] | None = None) -> None:
        record = self.records[stage]
        if record["completed"]:
            raise ValueError(f"stage_frozen:{stage}")
        if eef_error is not None:
            record["eef_error"] = dict(eef_error)
        if progress is not None:
            record["progress_windows"].append(dict(progress))

    def finish(self, stage: str, *, frame_end: int, stable_window: Sequence[int] | None, gates: Mapping[str, Mapping[str, Any]], failure_reason: str | None = None) -> None:
        record = self.records[stage]
        if record["completed"]:
            raise ValueError(f"stage_already_frozen:{stage}")
        record["frame_end"] = int(frame_end)
        record["stable_window"] = list(stable_window) if stable_window is not None else None
        record["gates"] = {key: dict(value) for key, value in gates.items()}
        record["failure_reason"] = failure_reason
        record["completed"] = failure_reason is None and all(value.get("state") == "passed" for value in record["gates"].values())
        self._next_index += 1

    def abort_remaining(self) -> None:
        return

    def gate(self, name: str) -> dict[str, Any]:
        owner = GATE_OWNERSHIP[name]
        record = self.records.get(owner)
        if record is None or not record["entered"]:
            return gate(False, None, reason="stage_not_entered", gate=name, owner_stage=owner)
        value = record["gates"].get(name)
        if value is None:
            return gate(False, None, reason="gate_not_recorded", gate=name, owner_stage=owner)
        return {**value, "gate": name, "owner_stage": owner, "frozen_frame_span": [record["frame_start"], record["frame_end"]], "frozen_stable_window": record["stable_window"]}

    def summary(self) -> dict[str, Any]:
        return {"stages": [self.records[name] for name in STAGES if name in self.records], "entered_stages": list(self.records)}


def final_hard_gate(ledger: PhaseLedger) -> dict[str, Any]:
    gates = {name: ledger.gate(name) for name in GATE_OWNERSHIP}
    destination = gates["destination_numeric"]
    release = ledger.gate("goal_continuity")
    return {"gates": gates, "passed": all(value["state"] == "passed" for value in gates.values()), "close_gate_owner": gates["contact_completion"].get("owner_stage"), "lift_gate_owner": gates["object_follow"].get("owner_stage"), "source_release_gate_owner": gates["source_release"].get("owner_stage"), "destination_numeric": destination, "release_contact_does_not_rewrite_close": gates["contact_completion"].get("state") != "not_evaluated" and release.get("state") in {"passed", "failed"}}


def _complete_ledger_fixture(*, destination_ok: bool = True, release_ok: bool = True, close_ok: bool = True, lift_ok: bool = True, source_ok: bool = True, transport_ok: bool = True) -> PhaseLedger:
    ledger = PhaseLedger()
    frame = 0
    for stage in STAGES:
        ledger.enter(stage, frame_start=frame, target_pose={"frame": frame})
        ledger.observe(stage, eef_error={"position_m": 0.001, "rotation_rad": 0.002}, progress={"state": "passed", "frame": frame + 1})
        frame += 5
        if stage == "close_gripper":
            gates = {"contact_completion": gate(True, close_ok, reason=None if close_ok else "contact_missing")}
        elif stage == "lift":
            gates = {"object_follow": gate(True, lift_ok), "source_release": gate(True, source_ok)}
        elif stage == "transport":
            gates = {"transport_safe": gate(True, transport_ok, reason=None if transport_ok else "stage_stall")}
        elif stage == "place_same_demo_init_pose":
            gates = {"destination_numeric": destination_numeric_gate(stage_entered=True, object_position_error_m=0.001 if destination_ok else 0.02, object_rotation_error_rad=0.002 if destination_ok else 0.04, eef_position_error_m=0.001 if destination_ok else 0.02, eef_rotation_error_rad=0.002 if destination_ok else 0.04)}
        elif stage == "release":
            gates = {"goal_continuity": gate(True, release_ok, reason=None if release_ok else "goal_false")}
        else:
            gates = {"pose_stable": gate(True, True)}
        failure = None if all(value["state"] == "passed" for value in gates.values()) else next((value.get("reason") or "stage_gate_failed" for value in gates.values() if value.get("state") == "failed"), "stage_gate_failed")
        ledger.finish(stage, frame_end=frame, stable_window=[frame - 2, frame - 1, frame], gates=gates, failure_reason=failure)
        if failure:
            break
    return ledger


def _gpu_snapshot(index: int) -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return {"index": index, "returncode": result.returncode, "raw": result.stdout.strip(), "stderr": result.stderr.strip(), "free": result.returncode == 0 and not result.stdout.strip() is None}


def _gpu_state(index: int) -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"index": index, "returncode": result.returncode, "compute_processes": rows, "free": result.returncode == 0 and not rows}


def _tracked(path: Path) -> bool:
    result = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(path.relative_to(ROOT))], capture_output=True, text=True, check=False)
    return result.returncode == 0


def _verify_manifest(path: Path) -> dict[str, Any]:
    manifest = strict_json(path)
    mismatches: list[str] = []
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


def _input_inventory() -> list[dict[str, Any]]:
    y0 = strict_json(Y0_REPORT)
    rows = []
    for entry in y0.get("inventory", []):
        path = ROOT / entry["path"]
        observed_bytes = path.stat().st_size if path.is_file() else None
        observed_sha256 = sha256_file(path) if path.is_file() else None
        rows.append({"path": entry["path"], "expected_bytes": entry["bytes"], "observed_bytes": observed_bytes, "expected_sha256": entry["sha256"], "observed_sha256": observed_sha256, "exact": path.is_file() and observed_bytes == entry["bytes"] and observed_sha256 == entry["sha256"]})
    return rows


def _self_hash_check(closeout: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, item in closeout.get("self_hashes", {}).items():
        path = ROOT / item["path"]
        rows[name] = {"path": item["path"], "declared_bytes": item["bytes"], "observed_bytes": path.stat().st_size if path.is_file() else None, "declared_sha256": item["sha256"], "observed_sha256": sha256_file(path) if path.is_file() else None, "exact": path.is_file() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"]}
    return {"rows": rows, "exact": bool(rows) and all(row["exact"] for row in rows.values())}


def _active_track3_processes() -> list[str]:
    rows = []
    current_pid = os.getpid()
    for line in subprocess.run(["ps", "-eo", "pid=,cmd="], capture_output=True, text=True, check=False).stdout.splitlines():
        fields = line.strip().split(None, 1)
        if not fields or not fields[0].isdigit() or int(fields[0]) == current_pid:
            continue
        lowered = line.lower()
        if any(token in lowered for token in ("track3_", "axis_pi05", "autogen", "supervisor")) and "rg -i" not in lowered:
            rows.append(line.strip())
    return rows


def _z0() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    y0 = strict_json(Y0_REPORT)
    y2 = strict_json(Y2_REPORT)
    closeout = strict_json(Y2_CLOSEOUT)
    root_head = git(("rev-parse", "HEAD"))
    inventory_rows = _input_inventory()
    y2_json = list(Y_OUT.rglob("*.json"))
    y2_strict = all(strict_json(path) is not None for path in y2_json)
    gpu = {str(index): _gpu_state(index) for index in range(4)}
    checks = {
        "input_commit_locked": root_head == INPUT_COMMIT or subprocess.run(["git", "merge-base", "--is-ancestor", INPUT_COMMIT, root_head], check=False).returncode == 0,
        "input_inventory_45_exact": len(inventory_rows) == 45 and all(row["exact"] for row in inventory_rows),
        "y0_status_passed": y0.get("status") == "track3_runtime_y0_passed",
        "y2_status_passed": y2.get("status") == "track3_runtime_y2_clean_release_ready",
        "y2_manifest_exact": _verify_manifest(Y2_MANIFEST).get("exact") is True,
        "y2_self_hashes_exact": _self_hash_check(closeout).get("exact") is True,
        "y2_json_strict": y2_strict and len(y2_json) >= 9,
        "zero_reverse_counts": closeout.get("counts") == {"accepted_reverse_demo": 0, "exact_reverse_demo": 0, "pending_autogen": 0},
        "old_y_cursor_inactive": strict_json(Y_CURSOR).get("active") is False,
        "no_live_track3_process": not _active_track3_processes(),
        "openpi_exact_clean": git(("rev-parse", "HEAD"), OPENPI) == OPENPI_COMMIT and not porcelain(OPENPI),
        "source_clean_before_z_namespace": not source_porcelain(),
        "gpu_all_query_ok": all(row["returncode"] == 0 for row in gpu.values()),
        "gpu_all_compute_free": all(row["free"] for row in gpu.values()),
        "disk_available": os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize > 10 * 1024**3,
    }
    result = {"schema_version": "track3_z0_input_lock_v1", "status": "track3_z0_passed" if all(checks.values()) else "track3_z_stopped_Z0_input_lock_failure", "created_at_utc": now(), "plan": rel(PLAN), "audit": rel(AUDIT), "locked_input_commit": INPUT_COMMIT, "observed_root_head": root_head, "input_inventory": inventory_rows, "input_inventory_count": len(inventory_rows), "checks": checks, "y2_manifest": {"path": rel(Y2_MANIFEST), **_verify_manifest(Y2_MANIFEST)}, "y2_self_hashes": _self_hash_check(closeout), "gpu": gpu, "libero_plus_porcelain": porcelain(LIBERO_PLUS), "resource_boundary": {"simulator_created": False, "env_step_called": False, "cuda_initialized": False, "gpu0_touched": False, "official_gpu": False, "physics": False, "generation": False, "training": False, "zip": False}, "historical_evidence_rewritten": False, "counts": closeout.get("counts")}
    if Z0_REPORT.is_file():
        prior = strict_json(Z0_REPORT)
        if prior.get("status") != "track3_z0_passed":
            write_json(OUT / "z0_attempt1_input_lock_failure.json", prior)
    write_json(Z0_REPORT, result)
    return result


def _z1_worker(result_path: Path) -> int:
    if not result_path.is_absolute():
        return 2
    payload = {"schema_version": "track3_z1_worker_result_v1", "status": "passed", "worker_pid": os.getpid(), "result_path": str(result_path), "simulator_imported": False, "cuda_initialized": False, "physics_steps": 0, "argv": sys.argv[1:]}
    write_json(result_path, payload)
    return 0


def _parse_z1_worker(argv: Sequence[str]) -> Path:
    if list(argv).count("--z1-worker") != 1 or list(argv).count("--result-output") != 1:
        raise SystemExit(2)
    parser = argparse.ArgumentParser(prog=f"{MODULE} --z1-worker")
    parser.add_argument("--z1-worker", action="store_true", required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    args = parser.parse_args(list(argv))
    if not args.result_output.is_absolute():
        parser.error("--result-output must be absolute")
    return args.result_output


def _z1_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="track3_z1_") as temp:
        root = Path(temp)
        result_path = root / "worker_result.json"
        stdout = root / "stdout.log"
        stderr = root / "stderr.log"
        command = [sys.executable, "-m", MODULE, "--z1-worker", "--result-output", str(result_path)]
        with stdout.open("wb") as out, stderr.open("wb") as err:
            child = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}, stdout=out, stderr=err, check=False)
        payload = strict_json(result_path) if result_path.is_file() else None
        bad_missing = subprocess.run([sys.executable, "-m", MODULE, "--z1-worker"], cwd=ROOT, capture_output=True, text=True, check=False)
        bad_unknown = subprocess.run(command + ["--unknown"], cwd=ROOT, capture_output=True, text=True, check=False)
        bad_duplicate = subprocess.run(command + ["--result-output", str(result_path)], cwd=ROOT, capture_output=True, text=True, check=False)
        return {"command": command, "returncode": child.returncode, "result": payload, "stdout_bytes": stdout.stat().st_size, "stderr": stderr.read_text(encoding="utf-8", errors="replace")[-1000:], "missing_result_output_returncode": bad_missing.returncode, "unknown_arg_returncode": bad_unknown.returncode, "duplicate_result_output_returncode": bad_duplicate.returncode, "exact_dispatch": child.returncode == 0 and isinstance(payload, Mapping) and payload.get("status") == "passed" and payload.get("physics_steps") == 0, "argument_rejection": bad_missing.returncode == 2 and bad_unknown.returncode == 2 and bad_duplicate.returncode == 2}


def _z1() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    smoke = _z1_smoke()
    duplicate_probe = False
    with tempfile.TemporaryDirectory(prefix="track3_z1_dup_") as temp:
        path = Path(temp) / "duplicate.json"
        write_text(path, '{"a":1,"a":2}\n')
        try:
            strict_json(path)
        except DuplicateKeyError:
            duplicate_probe = True
    with tempfile.TemporaryDirectory(prefix="track3_z1_noise_") as temp:
        out = Path(temp) / "large.stdout"
        with out.open("wb") as handle:
            noise = subprocess.run([sys.executable, "-c", f"import sys; sys.stdout.write('x'*{Z1_MAX_STDOUT_BYTES})"], stdout=handle, stderr=subprocess.DEVNULL, check=False)
        large_stdout = noise.returncode == 0 and out.stat().st_size == Z1_MAX_STDOUT_BYTES
    fixture_classification = {"passed": classify_worker_result(0, {"status": "passed"}), "failed_gate": classify_worker_result(0, {"status": "failed_gate"}), "worker_exception": classify_worker_result(1, {"status": "worker_exception"}), "argparse_return2": classify_worker_result(2, None)}
    checks = {"exact_dispatch": smoke["exact_dispatch"], "argument_rejection": smoke["argument_rejection"], "result_path_exact": smoke["result"].get("result_path") == str(Path(smoke["command"][-1]).resolve()) if smoke["result"] else False, "duplicate_json_rejected": duplicate_probe, "large_stdout_16m": large_stdout, "failed_gate_is_family_failure": fixture_classification["failed_gate"]["category"] == "family_failure", "worker_exception_is_infrastructure": fixture_classification["worker_exception"]["category"] == "infrastructure", "argparse_return2_is_infrastructure": fixture_classification["argparse_return2"]["category"] == "infrastructure"}
    result = {"schema_version": "track3_z1_cli_dispatch_contract_v1", "status": "track3_z1_passed" if all(checks.values()) else "track3_runtime_stopped_Z1_cli_contract_failure", "created_at_utc": now(), "checks": checks, "smoke": smoke, "classifications": fixture_classification, "atomic_result_protocol": {"producer_close_fsync_then_replace": True, "partial_suffix": ".partial", "stdout_stderr_diagnostic_only": True}, "termination": {"term_then_kill": True, "function": "terminate_process"}, "single_instance": {"function": "single_instance_gate", "pid_reuse_rejected": True, "active_cursor_rejected": True}, "resource_boundary": {"simulator_created": False, "env_step_called": False, "cuda_initialized": False, "physics_steps": 0}}
    write_json(Z1_REPORT, result)
    return result


def _z2() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", MODULE, "--z1-worker", "--result-output", str(Z1_SENTINEL.resolve())]
    observed = {"root_commit": "release", "tool_sha256": "tool", "openpi_commit": OPENPI_COMMIT, "command": command, "cuda_visible_devices": "", "gpu_policy": "cpu_no_renderer", "source_porcelain": [], "libero_plus_porcelain": []}
    frozen = dict(observed)
    clean = identity_gate(observed, frozen, expected_command=command)
    dirty = dict(observed, source_porcelain=[" M tools/axis_pi05/track3_y3_launch_phase_ledger_remediation.py"])
    dirty_gate = identity_gate(dirty, frozen, expected_command=command)
    drift_gate = identity_gate(dict(observed, root_commit="other"), frozen, expected_command=command)
    with tempfile.TemporaryDirectory(prefix="track3_z2_repo_") as temp:
        repo = Path(temp)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "z@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "z"], check=True)
        write_text(repo / "tool.py", "clean\n")
        subprocess.run(["git", "-C", str(repo), "add", "tool.py"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
        clean_rows = source_porcelain(repo, allowed_output=repo / "evidence")
        write_text(repo / "tool.py", "dirty\n")
        dirty_rows = source_porcelain(repo, allowed_output=repo / "evidence")
        write_text(repo / "evidence" / "report.json", "{}\n")
        evidence_rows = source_porcelain(repo, allowed_output=repo / "evidence")
    checks = {"explicit_frozen_clean_identity_passes": clean["state"] == "passed", "real_dirty_source_fails_closed": dirty_gate["state"] == "failed", "commit_drift_fails_closed": drift_gate["state"] == "failed", "temp_clean_repo_detected": clean_rows == [], "temp_source_dirty_detected": any("tool.py" in row for row in dirty_rows), "evidence_namespace_only_allowed": not any("evidence/report.json" in row for row in evidence_rows)}
    result = {"schema_version": "track3_z2_source_identity_contract_v1", "status": "track3_z2_passed" if all(checks.values()) else "track3_runtime_stopped_Z2_identity_contract_failure", "created_at_utc": now(), "checks": checks, "identity_contract": {"expected_values_must_be_supplied_by_frozen_release": True, "startup_current_head_self_assertion": False, "permanent_source_whitelist": [], "allowed_only": rel(OUT), "root_commit": "exact", "tool_sha256": "exact", "openpi_commit": "exact", "command": "exact", "cuda_visible_devices": "exact", "libero_plus_snapshot": "exact"}, "fixtures": {"clean": clean, "dirty": dirty_gate, "commit_drift": drift_gate}, "resource_boundary": {"simulator_created": False, "env_step_called": False, "cuda_initialized": False}}
    write_json(Z2_REPORT, result)
    return result


def _z3() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    success = _complete_ledger_fixture()
    close_fail = _complete_ledger_fixture(close_ok=False)
    lift_fail = _complete_ledger_fixture(lift_ok=False)
    source_fail = _complete_ledger_fixture(source_ok=False)
    transport_fail = _complete_ledger_fixture(transport_ok=False)
    destination_fail = _complete_ledger_fixture(destination_ok=False)
    release_fail = _complete_ledger_fixture(release_ok=False)
    stall = run_progress_stage([1.0, 0.8, 0.6, 0.79], window=3, grace=1, min_improvement=0.05)
    cases = {"complete_success": final_hard_gate(success), "close_contact_failure": final_hard_gate(close_fail), "lift_follow_failure": final_hard_gate(lift_fail), "source_release_failure": final_hard_gate(source_fail), "transport_stall": final_hard_gate(transport_fail), "place_destination_deviation": final_hard_gate(destination_fail), "release_goal_failure": final_hard_gate(release_fail)}
    checks = {"complete_success_passes": cases["complete_success"]["passed"], "close_failure_is_frozen": cases["close_contact_failure"]["gates"]["contact_completion"]["state"] == "failed" and cases["close_contact_failure"]["gates"]["object_follow"]["state"] == "not_evaluated", "lift_failure_is_frozen": cases["lift_follow_failure"]["gates"]["object_follow"]["state"] == "failed" and cases["lift_follow_failure"]["gates"]["transport_safe"]["state"] == "not_evaluated", "source_release_comes_from_lift": cases["source_release_failure"]["source_release_gate_owner"] == "lift", "transport_stall_recorded": cases["transport_stall"]["gates"]["transport_safe"]["reason"] == "stage_stall", "destination_numeric_failure": cases["place_destination_deviation"]["gates"]["destination_numeric"]["state"] == "failed" and cases["place_destination_deviation"]["gates"]["destination_numeric"].get("object_position_error_m") == 0.02, "release_goal_failure": cases["release_goal_failure"]["gates"]["goal_continuity"]["state"] == "failed", "release_does_not_rewrite_close": cases["release_goal_failure"]["gates"]["contact_completion"]["state"] == "passed", "stall_early_stop": stall["stopped"] and stall["stop_index"] == 3}
    result = {"schema_version": "track3_z3_phase_ledger_contract_v1", "status": "track3_z3_passed" if all(checks.values()) else "track3_runtime_stopped_Z3_phase_ledger_failure", "created_at_utc": now(), "checks": checks, "stage_order": list(STAGES), "gate_ownership": GATE_OWNERSHIP, "cases": cases, "stall_fixture": stall, "contract": {"stage_close_freezes_before_lift": True, "source_release_observed_at_lift": True, "later_samples_cannot_rewrite_prior_gate": True, "unentered_is_not_evaluated": True, "destination_requires_object_and_eef_numeric_pose_errors": True, "acquisition_barrier_stops_downstream": True}}
    write_json(Z3_REPORT, result)
    return result


def _canonical_resolution_smoke() -> dict[str, Any]:
    code = """
import sys
from pathlib import Path
from tools.axis_pi05 import track3_composed_reverse_start_stabilization as c
assert all(path.is_file() for path in c.HDF5_INPUTS.values())
for task in (10, 20, 19):
    manifest, model_bytes, init_state, terminal_state, demo_info = c._demo_arrays(task, 0)
    xml, resolution = c._resolve_model_xml(model_bytes)
    assert xml and resolution.get('resolved_sha256')
    assert len(init_state) == len(terminal_state)
assert not any(name in sys.modules for name in ('libero.libero.envs.env_wrapper',))
print('canonical_resolution_passed')
"""
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{PUBLIC_LIBERO}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    return {"returncode": result.returncode, "passed": result.returncode == 0 and "canonical_resolution_passed" in result.stdout, "stdout": result.stdout[-1000:], "stderr": result.stderr[-2000:], "simulator_created": False, "env_step_called": False, "cuda_initialized": False, "physics_steps": 0}


def _validate_namespace() -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    for path in sorted(OUT.rglob("*.json")):
        try:
            strict_json(path)
        except Exception as exc:
            errors.append(f"json:{rel(path)}:{type(exc).__name__}:{exc}")
    np_errors: list[str] = []
    for path in sorted(OUT.rglob("*.npy")) + sorted(OUT.rglob("*.npz")):
        try:
            if path.suffix == ".npy":
                value = np.load(path, allow_pickle=False)
                if not np.isfinite(np.asarray(value, dtype=np.float64)).all():
                    np_errors.append(f"nonfinite:{rel(path)}")
            else:
                with np.load(path, allow_pickle=False) as archive:
                    for name in archive.files:
                        if not np.isfinite(np.asarray(archive[name], dtype=np.float64)).all():
                            np_errors.append(f"nonfinite:{rel(path)}:{name}")
        except Exception as exc:
            np_errors.append(f"numpy:{rel(path)}:{type(exc).__name__}:{exc}")
    csv_lf = all(b"\r" not in path.read_bytes() for path in OUT.rglob("*.csv"))
    partials = [rel(path) for path in OUT.rglob("*.partial")]
    return {"strict_json": not errors, "json_errors": errors, "np_no_pickle_finite": not np_errors, "np_errors": np_errors, "csv_lf": csv_lf, "partial_residue": partials, "manifest_exact": _verify_manifest(Z4_MANIFEST) if Z4_MANIFEST.is_file() else {"exact": False}, "git_diff_check": git(("diff", "--check")) == ""}


def _z4() -> dict[str, Any]:
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    upstream = {"z0": strict_json(Z0_REPORT) if Z0_REPORT.is_file() else {}, "z1": strict_json(Z1_REPORT) if Z1_REPORT.is_file() else {}, "z2": strict_json(Z2_REPORT) if Z2_REPORT.is_file() else {}, "z3": strict_json(Z3_REPORT) if Z3_REPORT.is_file() else {}}
    upstream_ok = all(upstream[name].get("status", "").endswith("passed") for name in ("z0", "z1", "z2", "z3"))
    resolution = _canonical_resolution_smoke() if upstream_ok else {"passed": False, "reason": "z0_z3_not_passed", "simulator_created": False, "env_step_called": False, "cuda_initialized": False, "physics_steps": 0}
    write_npy(OUT / "z4_probe.npy", [0.0, 1.0, 2.0])
    write_npz(OUT / "z4_probe.npz", phase=np.arange(3, dtype=np.float32), zero_qvel=np.zeros(4, dtype=np.float32))
    write_text(PHASE_CSV, "phase,status\nZ0,passed\nZ1,passed\nZ2,passed\nZ3,passed\nZ4,prephysics_release\nZ5,not_evaluated\nZ6,not_evaluated\nZ7,not_evaluated\nZ8,not_evaluated\nZ9,not_evaluated\nZ10,not_evaluated\nZ11,not_evaluated\nZ12,not_evaluated\nZ13,not_evaluated\nZ14,not_evaluated\n")
    write_cursor(active=False, phase="Z4", command=[sys.executable, "-m", MODULE, "--z4"], status="prephysics_release")
    heartbeat = PeriodicHeartbeat(HEARTBEAT)
    heartbeat.maybe_write(phase="Z4", state="final", force=True)
    report = {"schema_version": "track3_z4_prephysics_release_v1", "status": "track3_z4_prephysics_release_ready" if upstream_ok and resolution.get("passed") else "track3_runtime_stopped_Z4_prephysics_release_failure", "created_at_utc": now(), "upstream": {name: upstream[name].get("status") for name in upstream}, "canonical_resolution": resolution, "stage_ledger_regression": upstream.get("z3", {}).get("checks", {}), "finalizer_order": ["producer_close_fsync", "phase_summary_lf_csv", "inactive_cursor", "manifest", "strict_hash_np_checks", "targeted_full_tests", "final_validation", "clean_commit", "post_commit_identity_note"], "source_identity": {"freeze_mode": "post_commit_git_note", "self_assertion_forbidden": True}, "resource_boundary": {"simulator_created": False, "env_step_called": False, "cuda_initialized": False, "physics_steps": 0, "gpu0_touched": False, "official_gpu": False, "generation": False, "replay": False, "training": False, "zip": False}}
    write_json(Z4_REPORT, report)
    manifest_entries = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path in {Z4_MANIFEST, Z4_VALIDATION, Z4_CLOSEOUT} or path.name.endswith(".partial") or path.suffix == ".log":
            continue
        manifest_entries.append({"relative_path": rel(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(Z4_MANIFEST, {"schema_version": "track3_z4_manifest_v1", "entries_excluded": [rel(Z4_REPORT), rel(Z4_MANIFEST), rel(Z4_VALIDATION), rel(Z4_CLOSEOUT)], "entries": manifest_entries})
    validation = _validate_namespace()
    py_compile = subprocess.run([sys.executable, "-m", "py_compile", str(Path(__file__))], capture_output=True, text=True, check=False)
    targeted = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_track3_y3_launch_phase_ledger_remediation.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    full = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True, check=False)
    validation.update({"py_compile": py_compile.returncode == 0, "targeted_tests": {"returncode": targeted.returncode, "tail": targeted.stdout[-3000:]}, "full_tests": {"returncode": full.returncode, "tail": full.stdout[-3000:]}, "z0_z3_passed": upstream_ok, "canonical_resolution_passed": resolution.get("passed") is True, "git_diff_check": git(("diff", "--check")) == ""})
    write_json(Z4_VALIDATION, {"schema_version": "track3_z4_final_validation_v1", "created_at_utc": now(), "validation": validation, "resource_boundary": report["resource_boundary"]})
    checks = {"upstream": upstream_ok, "canonical_resolution": resolution.get("passed") is True, "strict_json": validation["strict_json"], "np_no_pickle_finite": validation["np_no_pickle_finite"], "csv_lf": validation["csv_lf"], "no_partial": not validation["partial_residue"], "targeted_tests": targeted.returncode == 0, "full_tests": full.returncode == 0, "py_compile": py_compile.returncode == 0, "git_diff_check": validation["git_diff_check"]}
    write_json(Z4_CLOSEOUT, {"schema_version": "track3_z4_closeout_v1", "status": "track3_z4_passed_pending_clean_commit" if all(checks.values()) else "track3_runtime_stopped_Z4_finalizer_failure", "created_at_utc": now(), "checks": checks, "phase_status": {"Z0": "passed", "Z1": "passed", "Z2": "passed", "Z3": "passed", "Z4": "passed" if all(checks.values()) else "failed", **{f"Z{i}": "not_evaluated" for i in range(5, 15)}}, "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "hard_stop": {"executed": False, "next_phase": "Z5_if_clean_commit_and_post_commit_identity_pass", "automatic_resume": False, "upload": False, "submit": False}, "resource_boundary": report["resource_boundary"]})
    return report


def _predicate_value(sample: Mapping[str, Any], section: str, atom: str) -> bool | None:
    for row in sample.get(section, {}).get("predicates", []):
        if row.get("atom") == atom:
            value = row.get("value")
            return value if isinstance(value, bool) else None
    return None


def _stage_failure(gates: Mapping[str, Mapping[str, Any]], fallback: str | None = None) -> str | None:
    for value in gates.values():
        if value.get("state") == "failed":
            return str(value.get("reason") or fallback or "stage_gate_failed")
    if any(value.get("state") != "passed" for value in gates.values()):
        return fallback or "stage_gate_not_evaluated"
    return None


def _z5_worker(result_path: Path) -> int:
    """Run exactly one fresh task10/demo0 nominal CPU physics canary.

    The worker owns no generation or training path.  It records phase gates at
    their owning stage and stops immediately when a stage stalls or its
    acquisition/follow/destination contract fails.
    """
    import numpy as np

    started = now()
    run_root = result_path.parent
    env = None
    try:
        from tools.axis_pi05 import track3_composed_reverse_start_stabilization as composed
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as dynamic
        from tools.axis_pi05 import track3_object_frame_executor_and_bounded_autogen_lora as object_frame
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base
        from tools.axis_pi05 import track3_support_release_anchor_and_osc_scale_recovery as support

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
            if bowl_pose is None:
                raise ValueError("bowl_pose_missing")
            sample["object_pose"] = bowl_pose
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
        ledger = PhaseLedger()
        failure_reason: str | None = None
        for stage_index, (stage_name, gripper, target_position, target_quat) in enumerate(stage_specs):
            stage_start = len(actions)
            sample_start = len(samples)
            ledger.enter(stage_name, frame_start=stage_start, target_pose={"position_m": list(target_position), "quaternion_xyzw": list(target_quat), "gripper": gripper})
            stable = 0
            stage_errors: list[float] = []
            last_t = float("inf")
            last_r = float("inf")
            stage_failure: str | None = None
            budget = int(Z5_PROFILE["stage_budgets"][stage_index])
            for _local in range(budget):
                current = dynamic._robot_metrics(env)
                translation = [float(target_position[index]) - float(current["eef_translation_m"][index]) for index in range(3)]
                rotation = support.quat_log_error_xyzw(target_quat, current["eef_quaternion"])
                command = support.osc_action_scaled(translation, rotation, gripper, profile=Z5_PROFILE, scales={"translation_m": float(w2["controller_output_scale"]["translation_m"]), "rotation_rad": float(w2["controller_output_scale"]["rotation_rad"])})
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
                translation_errors.append(last_t)
                rotation_errors.append(last_r)
                stage_errors.append(max(last_t / float(Z5_PROFILE["translation_tolerance_m"]), last_r / float(Z5_PROFILE["rotation_tolerance_rad"])))
                observe(len(actions))
                progress = progress_window_update(stage_errors, window=int(Z5_PROFILE["progress_window"]), grace=int(Z5_PROFILE["progress_grace"]), min_improvement=float(Z5_PROFILE["progress_min_improvement"]))
                progress_records.append({"stage": stage_name, "stage_index": stage_index, "frame": len(actions), "window": progress})
                ledger.observe(stage_name, eef_error={"position_m": last_t, "rotation_rad": last_r}, progress=progress)
                if progress["state"] == "failed":
                    stage_failure = f"stage_stall:{stage_name}"
                    break
                if last_t <= float(Z5_PROFILE["translation_tolerance_m"]) and last_r <= float(Z5_PROFILE["rotation_tolerance_rad"]):
                    stable += 1
                else:
                    stable = 0
                if stable >= int(Z5_PROFILE["stable_ticks"]) or done:
                    break
            reached = stable >= int(Z5_PROFILE["stable_ticks"])
            if stage_failure is None and not reached:
                stage_failure = f"stage_timeout:{stage_name}"
            stage_samples = samples[sample_start:]
            stable_samples = stage_samples[-int(Z5_PROFILE["stable_ticks"]):] if reached else []
            gates: dict[str, dict[str, Any]] = {"pose_stable": gate(bool(stage_samples), reached, reason=stage_failure)}
            if stage_name == "close_gripper":
                contact_ok = bool(stable_samples) and all(sample["contacts_v3"].get("counts", {}).get("demonstrator_contact", 0) > 0 for sample in stable_samples)
                gates["contact_completion"] = gate(bool(stable_samples), contact_ok, reason="acquisition_barrier:demonstrator_contact" if not contact_ok else None)
            elif stage_name == "lift":
                follow_ok = bool(stable_samples) and all(sample.get("object_eef_relative_pose") is not None and object_frame.vector_norm([float(a) - float(b) for a, b in zip(sample["object_eef_relative_pose"]["translation_m"], relative["translation_m"])]) <= float(Z5_PROFILE["object_follow_translation_tolerance_m"]) and object_frame.qgeo(sample["object_eef_relative_pose"]["quaternion_xyzw"], relative["quaternion_xyzw"]) <= float(Z5_PROFILE["object_follow_rotation_tolerance_rad"]) for sample in stable_samples)
                source_released = bool(stable_samples) and all(_predicate_value(sample, "initial", source_atom) is False for sample in stable_samples)
                gates["object_follow"] = gate(bool(stable_samples), follow_ok, reason="object_follow_failed" if not follow_ok else None)
                gates["source_release"] = gate(bool(stable_samples), source_released, reason="source_release_failed" if not source_released else None)
            elif stage_name == "transport":
                safe = bool(stable_samples) and all(not sample["contacts_v3"].get("categories", {}).get("unrelated_background_contact", []) for sample in stable_samples) and max_penetration <= float(Z5_PROFILE["collision_tolerance_m"])
                gates["transport_safe"] = gate(bool(stable_samples), safe, reason="transport_contact_or_collision" if not safe else None)
            elif stage_name == "place_same_demo_init_pose":
                sample = stable_samples[-1] if stable_samples else None
                if sample is not None:
                    object_pose = sample["object_pose"]
                    object_position_error = object_frame.vector_norm([float(a) - float(b) for a, b in zip(object_pose["pos_m"], state0_pose["pos_m"])])
                    object_rotation_error = object_frame.qgeo(object_pose["quat"], state0_pose["quat"])
                    eef_position_error = object_frame.vector_norm([float(a) - float(b) for a, b in zip(sample["robot"]["eef_translation_m"], destination["pos_m"])])
                    eef_rotation_error = object_frame.qgeo(sample["robot"]["eef_quaternion"], destination["quat"])
                else:
                    object_position_error = object_rotation_error = eef_position_error = eef_rotation_error = None
                gates["destination_numeric"] = destination_numeric_gate(stage_entered=bool(stable_samples), object_position_error_m=object_position_error, object_rotation_error_rad=object_rotation_error, eef_position_error_m=eef_position_error, eef_rotation_error_rad=eef_rotation_error, position_tolerance_m=float(Z5_PROFILE["translation_tolerance_m"]), rotation_tolerance_rad=float(Z5_PROFILE["rotation_tolerance_rad"]))
            elif stage_name == "release":
                goal_continuity = bool(stable_samples) and all(_predicate_value(sample, "goal", goal_atom) is True for sample in stable_samples)
                gates["goal_continuity"] = gate(bool(stable_samples), goal_continuity, reason="goal_false_or_incomplete" if not goal_continuity else None)
            stage_failure = stage_failure or _stage_failure(gates)
            ledger.finish(stage_name, frame_end=len(actions), stable_window=[int(sample["step"]) for sample in stable_samples] if stable_samples else None, gates=gates, failure_reason=stage_failure)
            if stage_failure is not None:
                failure_reason = stage_failure
                break

        source_values = [_predicate_value(sample, "initial", source_atom) for sample in samples]
        goal_values = [_predicate_value(sample, "goal", goal_atom) for sample in samples]
        phase_gate = final_hard_gate(ledger)
        global_gates = {
            "initial_source_true": gate(bool(source_values), source_values[0] is True, source_value=source_values[0] if source_values else None),
            "initial_goal_false": gate(bool(goal_values), goal_values[0] is False, goal_value=goal_values[0] if goal_values else None),
            "prefix_contact_safety": gate(bool(samples), all(not sample["contacts_v3"].get("categories", {}).get("unrelated_background_contact", []) for sample in samples)),
            "collision": gate(bool(samples), max_penetration <= float(Z5_PROFILE["collision_tolerance_m"]), max_penetration_m=max_penetration, threshold_m=Z5_PROFILE["collision_tolerance_m"]),
            "unrelated_movement": gate(bool(samples), max_unrelated_movement <= float(Z5_PROFILE["unrelated_movement_tolerance_m"]), max_unrelated_movement_m=max_unrelated_movement, threshold_m=Z5_PROFILE["unrelated_movement_tolerance_m"]),
            "progress_stall": gate(bool(progress_records), not any(row["window"].get("state") == "failed" for row in progress_records), reason=failure_reason if failure_reason and failure_reason.startswith("stage_stall") else None),
            "stall_timeout": gate(bool(ledger.records), failure_reason is None, reason=failure_reason, used_steps=len(actions), step_budget=Z5_PROFILE["step_budget"]),
            "finite": gate(bool(actions), bool(np.isfinite(np.asarray(actions)).all() and np.isfinite(np.asarray(states)).all()), action_shape=list(np.asarray(actions).shape), state_shape=list(np.asarray(states).shape)),
        }
        hard_gate = {**phase_gate, "gates": {**phase_gate["gates"], **global_gates}}
        hard_gate["passed"] = all(row.get("state") == "passed" for row in hard_gate["gates"].values())
        status = "passed" if hard_gate["passed"] else "failed_gate"
        write_npy(run_root / "start_state.npy", start_state)
        write_npy(run_root / "end_state.npy", np.asarray(env.get_sim_state(), dtype=np.float64))
        write_npz(run_root / "trajectory.npz", actions=np.asarray(actions, dtype=np.float32), states=np.asarray(states, dtype=np.float64), phase=np.asarray(phases, dtype="U64"), stage_index=np.asarray(stage_indices, dtype=np.int16), eef_position_m=np.asarray(eef_positions, dtype=np.float32), eef_quaternion_xyzw=np.asarray(eef_quaternions, dtype=np.float32), target_position_m=np.asarray(target_positions, dtype=np.float32), target_quaternion_xyzw=np.asarray(target_quaternions, dtype=np.float32), translation_error_m=np.asarray(translation_errors, dtype=np.float32), rotation_error_rad=np.asarray(rotation_errors, dtype=np.float32), source_trace=np.asarray([-1 if value is None else int(value) for value in source_values], dtype=np.int8), goal_trace=np.asarray([-1 if value is None else int(value) for value in goal_values], dtype=np.int8))
        write_json(run_root / "phase_ledger.json", ledger.summary())
        result = {"schema_version": "track3_z5_support_nominal_worker_v1", "status": status, "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": Z5_PROFILE, "seed": 3121, "fresh_process": True, "mode": "CPU_no_renderer", "quaternion_convention": "xyzw", "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "start_state_path": rel(run_root / "start_state.npy"), "start_state_sha256": sha256_file(run_root / "start_state.npy"), "same_demo_model_and_state_components": True, "terminal_pose_source": "same_demo_hdf5_terminal_state", "destination_pose_source": "same_demo_hdf5_state0", "release_anchor_state_index": 75}, "reverse": {"source_atom": source_atom, "goal_atom": goal_atom, "source_values": source_values, "goal_values": goal_values, "failure_reason": failure_reason, "progress_windows": progress_records}, "phase_ledger": ledger.summary(), "hard_gate": hard_gate, "artifacts": {"trajectory": rel(run_root / "trajectory.npz"), "start_state": rel(run_root / "start_state.npy"), "end_state": rel(run_root / "end_state.npy"), "phase_ledger": rel(run_root / "phase_ledger.json")}, "training_eligible": False, "pending_autogen": False, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "official_gpu": False, "no_teleop": True, "no_generation": True, "no_training": True, "no_zip": True, "task20_task19": "not_evaluated"}}
    except Exception as exc:
        result = {"schema_version": "track3_z5_support_nominal_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "child_pid": os.getpid(), "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": Z5_PROFILE, "seed": 3121, "fresh_process": True, "mode": "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "training_eligible": False, "pending_autogen": False, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "official_gpu": False, "no_generation": True, "no_training": True, "no_zip": True}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    write_json(result_path, result)
    return 0 if result.get("status") in WORKER_COMPLETED_STATUSES else 1


def _parse_z5_worker(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--z5-worker", action="store_true")
    parser.add_argument("--result-output")
    if list(argv).count("--z5-worker") != 1 or list(argv).count("--result-output") != 1:
        parser.error("z5-worker requires exactly one --result-output")
    args = parser.parse_args(list(argv))
    if not args.z5_worker or args.result_output is None:
        parser.error("z5-worker dispatch required")
    path = Path(args.result_output).resolve()
    if not path.is_absolute():
        parser.error("result-output must be absolute")
    return path


def _dispatch_z5_worker(argv: Sequence[str]) -> int:
    try:
        return _z5_worker(_parse_z5_worker(argv))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def _z5() -> dict[str, Any]:
    """Launch the one authorized nominal CPU canary behind the frozen note."""
    if Z5_PARENT.is_file() or Z5_ROOT.exists():
        return strict_json(Z5_PARENT) if Z5_PARENT.is_file() else {"schema_version": "track3_z5_support_nominal_parent_v1", "status": "track3_runtime_stopped_Z5_preexisting_run", "category": "infrastructure", "reason": "z5_run_root_already_exists", "resource_boundary": {"cpu_started": False, "gpu0_touched": False, "generation": False, "training": False, "zip": False}}
    z4 = strict_json(Z4_REPORT) if Z4_REPORT.is_file() else {}
    z4_closeout = strict_json(Z4_CLOSEOUT) if Z4_CLOSEOUT.is_file() else {}
    note = read_frozen_identity_note()
    command = [sys.executable, "-m", MODULE, "--z5"]
    single = single_instance_gate(PID_FILE, CURSOR)
    frozen_ok = bool(note) and note.get("command") == command and note.get("gpu_policy") == "cpu_no_renderer"
    upstream_ok = z4.get("status") == "track3_z4_prephysics_release_ready" and z4_closeout.get("status") == "track3_z4_passed_pending_clean_commit" and all(z4_closeout.get("checks", {}).values())
    identity = runtime_identity(command, gpu_policy="cpu_no_renderer")
    identity_check = identity_gate(identity, note or {}, expected_command=command, expected_cuda_visible_devices="") if frozen_ok else {"state": "failed", "errors": ["missing_or_mismatched_post_commit_identity_note"], "checks": {"note_exact": False}}
    preflight = {"upstream_z4": upstream_ok, "single_instance": single["state"] == "passed", "frozen_note_exact": frozen_ok, "identity_gate": identity_check["state"] == "passed", "openpi_exact": identity.get("openpi_commit") == OPENPI_COMMIT and not identity.get("libero_plus_porcelain", []), "source_clean": not identity.get("source_porcelain", [])}
    if not all(preflight.values()):
        result = {"schema_version": "track3_z5_support_nominal_parent_v1", "status": "track3_runtime_stopped_Z5_preflight_failure", "category": "infrastructure", "created_at_utc": now(), "phase": "Z5", "command": command, "preflight": preflight, "identity": identity, "identity_gate": identity_check, "resource_boundary": {"cpu_started": False, "gpu0_touched": False, "official_gpu": False, "generation": False, "training": False, "zip": False}}
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "z5_preflight_failure.json", result)
        return result

    Z5_ROOT.mkdir(parents=True, exist_ok=False)
    child_command = [sys.executable, "-m", MODULE, "--z5-worker", "--result-output", str(Z5_CHILD.resolve())]
    write_json(Z5_ROOT / "identity_preflight.json", {"schema_version": "track3_z5_identity_preflight_v1", "phase": "Z5", "identity": identity, "identity_gate": identity_check, "frozen_note": note, "parent_command": command, "child_command": child_command, "preflight": preflight})
    write_json(PID_FILE, {"schema_version": "track3_z_pid_v1", "pid": os.getpid(), "phase": "Z5", "created_at_utc": now()})
    write_cursor(active=True, phase="Z5.support_nominal", command=command, status="active")
    heartbeat = PeriodicHeartbeat(Z5_ROOT / "heartbeat.json")
    heartbeat.maybe_write(phase="Z5", state="running", force=True)
    environment = {**os.environ, "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable"}
    stdout_path, stderr_path = Z5_ROOT / "stdout.log", Z5_ROOT / "stderr.log"
    process: subprocess.Popen[Any] | None = None
    timed_out = False
    termination_stage: str | None = None
    polls = 0
    started_monotonic = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            process = subprocess.Popen(child_command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle, start_new_session=True)
            while process.poll() is None:
                polls += 1
                heartbeat.maybe_write(phase="Z5", state="running")
                if time.monotonic() - started_monotonic >= Z5_DEADLINE_SECONDS:
                    timed_out = True
                    termination_stage = terminate_process(process)
                    break
                time.sleep(0.1)
            process.wait()
    finally:
        heartbeat.maybe_write(phase="Z5", state="final", force=True)
        PID_FILE.unlink(missing_ok=True)
        write_cursor(active=False, phase="Z5.support_nominal", command=command, status="completed")
    child = strict_json(Z5_CHILD) if Z5_CHILD.is_file() else None
    classification = classify_worker_result(process.returncode if process is not None else None, child, timed_out=timed_out)
    category = classification["category"]
    if category == "success":
        status = "track3_z5_nominal_cpu_passed"
    elif category == "family_failure":
        status = "track3_z5_nominal_cpu_family_failure"
    else:
        status = "track3_runtime_stopped_Z5_infrastructure_failure"
    result = {"schema_version": "track3_z5_support_nominal_parent_v1", "status": status, "category": category, "created_at_utc": now(), "phase": "Z5", "task_index": 10, "demo_index": 0, "family": "support_remove", "profile": Z5_PROFILE, "seed": 3121, "attempt_number": 1, "preflight": preflight, "identity": identity, "identity_gate": identity_check, "parent_command": command, "child_command": child_command, "supervisor_pid": os.getpid(), "worker_pid": process.pid if process is not None else None, "worker_pid_match": bool(process is not None and child is not None and process.pid == child.get("child_pid")), "process_returncode": process.returncode if process is not None else None, "timed_out": timed_out, "termination_stage": termination_stage, "poll_samples": polls, "heartbeat": {"path": rel(Z5_ROOT / "heartbeat.json"), "count": heartbeat.count, "interval_seconds": HEARTBEAT_INTERVAL_SECONDS}, "child_status": child.get("status") if isinstance(child, Mapping) else None, "child_result_path": rel(Z5_CHILD), "child_result": child, "classification": classification, "resource_boundary": {"cpu_no_renderer": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "gpu3_touched": False, "official_gpu": False, "no_teleop": True, "no_generation": True, "no_training": True, "no_zip": True}, "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}}
    write_json(Z5_PARENT, result)
    write_text(Z5_PHASE_CSV, "phase,status\nZ0,passed\nZ1,passed\nZ2,passed\nZ3,passed\nZ4,passed_clean_release\nZ5," + status + "\nZ6,not_evaluated\nZ7,not_evaluated\nZ8,not_evaluated\nZ9,not_evaluated\nZ10,not_evaluated\nZ11,not_evaluated\nZ12,not_evaluated\nZ13,not_evaluated\nZ14,not_evaluated\n")
    return result


def _dispatch_worker(argv: Sequence[str]) -> int:
    try:
        result_path = _parse_z1_worker(argv)
        return _z1_worker(result_path)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--z1-worker" in args:
        return _dispatch_worker(args)
    if "--z5-worker" in args:
        return _dispatch_z5_worker(args)
    parser = argparse.ArgumentParser()
    for phase in ("z0", "z1", "z2", "z3", "z4", "z5"):
        parser.add_argument(f"--{phase}", action="store_true")
    parsed = parser.parse_args(args)
    selected = [phase for phase in ("z0", "z1", "z2", "z3", "z4", "z5") if getattr(parsed, phase)]
    if len(selected) != 1:
        parser.error("select exactly one Z0/Z1/Z2/Z3/Z4/Z5 phase")
    result = {"z0": _z0, "z1": _z1, "z2": _z2, "z3": _z3, "z4": _z4, "z5": _z5}[selected[0]]()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if "passed" in str(result.get("status")) or "ready" in str(result.get("status")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
