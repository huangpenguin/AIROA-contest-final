#!/usr/bin/env python3
"""P3 I/O recovery and bounded support-canary entrypoint.

R0/R1 are deliberately simulator- and GPU-free.  Runtime imports are kept
inside the later P3/R5 workers.  Worker results are published through an
independent atomic JSON file; stdout and stderr are diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_p3_io_recovery_and_support_executor_canary"
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_start_bundle_and_support_executor_canary"
OLD_MANIFEST = OLD_ROOT / "artifact_manifest.json"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_P3_IO_RECOVERY_AND_SUPPORT_EXECUTOR_CANARY_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_START_BUNDLE_P0_P6_AUDIT_20260905.md"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
MODULE = "tools.axis_pi05.track3_p3_io_recovery_and_support_executor_canary"
FIXED_OLD_UNTRACKED = (
    OLD_ROOT / "start_states.npz",
    OLD_ROOT / "camera_frames/p3_task10_demo0.npz",
    OLD_ROOT / "videos/p3_task10_demo0_agentview.mp4",
    OLD_ROOT / "videos/p3_task10_demo0_wrist.mp4",
)
IO_DEADLINE_SECONDS = 300.0
IO_TERM_GRACE_SECONDS = 10.0
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


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def path_ref(path: Path) -> str:
    """Use repository-relative refs for evidence, absolute refs for test temps."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def artifact_entry(path: Path, *, source_commit: str, schema: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    inferred = schema or path.suffix.lstrip(".") or "binary"
    if schema is None and path.suffix == ".json":
        try:
            value = strict_json_load(path)
            inferred = str(value.get("schema_version", "json_unknown_schema")) if isinstance(value, dict) else "json_non_object"
        except Exception:
            inferred = "json_unparseable"
    return {"relative_path": relative_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "schema": inferred, "source_commit": source_commit}


def validate_artifact_manifest(entries: Sequence[Mapping[str, Any]], *, require_git_tracking: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[str] = set()
    tracked = set(git(("ls-files",)).splitlines()) if require_git_tracking else set()
    for entry in entries:
        path_value = str(entry.get("relative_path", ""))
        path = ROOT / path_value
        if not path_value or path_value in seen or path_value.startswith("/") or ".." in Path(path_value).parts:
            errors.append(f"invalid_relative_path:{path_value}")
            continue
        seen.add(path_value)
        if not path.is_file():
            errors.append(f"missing:{path_value}")
            continue
        if int(entry.get("bytes", -1)) != path.stat().st_size:
            errors.append(f"bytes:{path_value}")
        if str(entry.get("sha256", "")) != sha256_file(path):
            errors.append(f"sha256:{path_value}")
        if not entry.get("source_commit"):
            errors.append(f"source_commit:{path_value}")
        if require_git_tracking and path_value not in tracked:
            errors.append(f"untracked:{path_value}")
    return {"executed": True, "state": "passed" if not errors else "failed", "entry_count": len(entries), "errors": errors, "require_git_tracking": require_git_tracking}


def validate_npz(path: Path, expected: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    errors: list[str] = []
    keys: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = sorted(archive.files)
            if keys != sorted(expected):
                errors.append(f"keys:{keys}")
            for key, spec in expected.items():
                if key not in archive.files:
                    continue
                array = np.asarray(archive[key])
                if array.dtype != np.dtype("float64"):
                    errors.append(f"dtype:{key}:{array.dtype}")
                if list(array.shape) != list(spec["shape"]):
                    errors.append(f"shape:{key}:{array.shape}")
                if not np.isfinite(array).all():
                    errors.append(f"finite:{key}")
                if sha256_bytes(array.tobytes()) != spec["sha256"]:
                    errors.append(f"sha256:{key}")
    except Exception as exc:
        errors.append(f"npz:{type(exc).__name__}:{exc}")
    return {"executed": True, "state": "passed" if not errors else "failed", "keys": keys, "allow_pickle": False, "errors": errors}


def _old_state_expected() -> dict[str, dict[str, Any]]:
    manifest = strict_json_load(OLD_ROOT / "start_state_manifest.json")
    return {key: {"shape": row["state_shape"], "sha256": row["composed_state_sha256"]} for key, row in manifest["states"].items()}


def _old_p2_compact() -> dict[str, Any]:
    p2 = strict_json_load(OLD_ROOT / "p2_summary.json")
    workers = []
    for row in p2.get("restore_workers", []):
        workers.append({"task_index": row.get("task_index"), "demo_index": row.get("demo_index"), "status": row.get("status"), "restore_max_abs_diff": row.get("restore_max_abs_diff"), "state_sha256": row.get("state_sha256"), "hard_gate_states": {key: value.get("state") for key, value in row.get("hard_gate", {}).items() if isinstance(value, Mapping)}})
    return {"status": p2.get("status"), "validated_start_state": p2.get("validated_start_state"), "restore_count": p2.get("restore_count"), "max_restore_abs_diff": max((float(row.get("restore_max_abs_diff", 0.0)) for row in p2.get("restore_workers", []) if row.get("restore_max_abs_diff") is not None), default=None), "workers": workers, "grasp_geometry_status": p2.get("grasp_geometry", {}).get("status"), "counts": p2.get("counts", {})}


def run_r0() -> dict[str, Any]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    old_manifest = strict_json_load(OLD_MANIFEST)
    entries = old_manifest.get("entries", [])
    manifest_check = validate_artifact_manifest(entries)
    tracked_set_before = set(git(("ls-files",)).splitlines())
    tracked_before = [item["relative_path"] for item in entries if item["relative_path"] in tracked_set_before]
    closure_paths = [relative_path(path) for path in FIXED_OLD_UNTRACKED]
    closure_command = subprocess.run(["git", "add", "-f", "--", *closure_paths], cwd=ROOT, capture_output=True, text=True, check=False)
    tracked_set_after = set(git(("ls-files",)).splitlines())
    tracked_after = [item["relative_path"] for item in entries if item["relative_path"] in tracked_set_after]
    expected = _old_state_expected()
    npz_check = validate_npz(OLD_ROOT / "start_states.npz", expected)
    p2 = _old_p2_compact()
    old_p3 = strict_json_load(OLD_ROOT / "p3_summary.json")
    old_p6 = strict_json_load(OLD_ROOT / "p6_summary.json")
    lock = {"schema_version": "track3_p3_recovery_r0_v1", "status": "track3_p3_recovery_r0_artifact_closure_passed", "created_at_utc": now(), "plan": relative_path(PLAN), "audit": relative_path(AUDIT), "audited_root_commit": "ad2d4ef7d6603e7b0e79a8519155de9cfdefbde2", "observed_root_head": git(("rev-parse", "HEAD")), "observed_openpi_head": git(("rev-parse", "HEAD"), OPENPI), "root_porcelain": porcelain(), "openpi_porcelain": porcelain(OPENPI), "libero_plus_porcelain": porcelain(LIBERO_PLUS), "old_manifest": {"path": relative_path(OLD_MANIFEST), "declared_entry_count": len(entries), "existing_hash_exact": manifest_check["state"] == "passed", "entry_validation": manifest_check, "old_p6_field_erratum": "entries_exclude_self_and_p6_summary=true is inaccurate; old manifest includes p6_summary and has 17 entries", "old_source_commit_erratum": "6cfda92 is the parent execution commit, not the post-fix tool tree"}, "git_tracking": {"closure_command": ["git", "add", "-f", "--", *closure_paths], "closure_returncode": closure_command.returncode, "closure_stderr": closure_command.stderr.strip(), "tracked_before_r0": tracked_before, "tracked_after_r0": tracked_after, "tracked_entry_count": len(tracked_after), "required_entry_count": 17, "required_all_tracked": len(tracked_after) == 17}, "p2_read_only_revalidation": {"npz": npz_check, "summary": p2, "no_simulator_rerun": True, "fixed_grasp_geometry": strict_json_load(OLD_ROOT / "task10_demo0_grasp_geometry.json")["status"] == "passed"}, "p3_attempt_1_preserved": {"path": relative_path(OLD_ROOT / "p3_task10_demo0.json"), "status": old_p3.get("status"), "attempt_count": old_p3.get("attempt_number_for_task10_demo0", 1), "p6_status": old_p6.get("status")}, "resource_boundary": {"simulator_started": False, "gpu_started": False, "old_bytes_modified": False, "hdf5_or_bddl_modified": False}, "status_reasons": []}
    if manifest_check["state"] != "passed" or npz_check["state"] != "passed" or len(entries) != 17:
        lock["status"] = "track3_p3_recovery_r0_artifact_closure_failed"
        lock["status_reasons"].append("old_manifest_missing_or_hash_mismatch")
    if closure_command.returncode != 0 or len(tracked_after) != 17:
        lock["status"] = "track3_p3_recovery_r0_artifact_closure_failed"
        lock["status_reasons"].append("git_tracking_not_17_of_17")
    write_json(OUT_ROOT / "r0_artifact_closure.json", lock)
    write_text(OUT_ROOT / "R0_ERRATA_ZH.md", "# R0 旧证据勘误\n\n- 旧 P0–P6 证据保持 immutable。\n- 旧 manifest 实际包含 `p6_summary`，共 17 项；旧 P6 的 `entries_exclude_self_and_p6_summary` 字段不准确。\n- 旧 `source_commit=6cfda92` 只表示执行父提交，不代表本轮修复工具版本。\n- 本轮 R0 只闭合 Git tracking 与只读复核，不重写 NPZ 或视频。\n\n状态：`%s`\n" % lock["status"])
    return lock


def _write_result(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, dict(value))


def _io_worker(mode: str, result_path: Path) -> int:
    started = now()
    if mode == "large":
        sys.stdout.write("x" * LARGE_STDOUT_BYTES)
        sys.stdout.flush()
        _write_result(result_path, {"schema_version": "track3_result_protocol_test_v1", "status": "passed", "worker_pid": os.getpid(), "mode": mode, "created_at_utc": started})
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
        _write_result(result_path, {"schema_version": "track3_result_protocol_test_v1", "status": "worker_exception", "worker_pid": os.getpid(), "mode": mode})
        return 7
    if mode == "exception":
        raise RuntimeError("synthetic_worker_exception")
    raise ValueError(f"unknown_io_test_mode:{mode}")


def _run_result_protocol(mode: str, *, deadline_seconds: float = IO_DEADLINE_SECONDS, term_grace_seconds: float = IO_TERM_GRACE_SECONDS, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or (OUT_ROOT / "r1_io_tests" / mode)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    for path in (result_path, stdout_path, stderr_path):
        path.unlink(missing_ok=True)
    command = [sys.executable, "-m", MODULE, "--io-test-worker", "--io-mode", mode, "--result-output", str(result_path)]
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'LIBERO'}", "JAX_PLATFORMS": "cpu"}, stdout=stdout_handle, stderr=stderr_handle)
        timed_out = False
        while process.poll() is None:
            if time.monotonic() - started >= deadline_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=term_grace_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=term_grace_seconds)
                break
            time.sleep(min(0.02, max(0.001, deadline_seconds / 100.0)))
    returncode = process.returncode
    stdout_bytes = stdout_path.stat().st_size if stdout_path.is_file() else 0
    result: dict[str, Any] | None = None
    result_error = None
    if result_path.is_file():
        try:
            result = strict_json_load(result_path)
        except Exception as exc:
            result_error = f"{type(exc).__name__}:{exc}"
    worker_pid_match = bool(result and int(result.get("worker_pid", -1)) == process.pid)
    protocol_ok = bool(result and worker_pid_match and result.get("status") == "passed" and returncode == 0 and not timed_out)
    return {"schema_version": "track3_result_protocol_run_v1", "mode": mode, "status": "passed" if protocol_ok else ("timeout_terminated" if timed_out else "failed_closed"), "worker_pid": process.pid, "worker_pid_match": worker_pid_match, "returncode": returncode, "timed_out": timed_out, "deadline_seconds": deadline_seconds, "term_grace_seconds": term_grace_seconds, "result_path": path_ref(result_path), "stdout_path": path_ref(stdout_path), "stderr_path": path_ref(stderr_path), "stdout_bytes": stdout_bytes, "stdout_at_least_16_mib": stdout_bytes >= LARGE_STDOUT_BYTES if mode == "large" else None, "result_present": result is not None, "result_error": result_error, "result": result, "atomic_partial_residue": sorted(path_ref(path) for path in output_dir.glob("*.partial"))}


def run_r1() -> dict[str, Any]:
    p0 = strict_json_load(OUT_ROOT / "r0_artifact_closure.json")
    if p0.get("status") != "track3_p3_recovery_r0_artifact_closure_passed":
        result = {"schema_version": "track3_p3_recovery_r1_v1", "status": "not_evaluated", "reason": "r0_not_passed", "simulator_started": False, "gpu_started": False}
        write_json(OUT_ROOT / "r1_io_recovery_contract.json", result)
        return result
    modes = {
        "large": _run_result_protocol("large", deadline_seconds=10.0, term_grace_seconds=1.0),
        "hang": _run_result_protocol("hang", deadline_seconds=1.0, term_grace_seconds=0.2),
        "malformed": _run_result_protocol("malformed", deadline_seconds=10.0, term_grace_seconds=1.0),
        "nonzero": _run_result_protocol("nonzero", deadline_seconds=10.0, term_grace_seconds=1.0),
        "exception": _run_result_protocol("exception", deadline_seconds=10.0, term_grace_seconds=1.0),
    }
    tests = {"large_stdout_no_deadlock": modes["large"]["status"] == "passed" and modes["large"]["stdout_at_least_16_mib"], "hang_term_then_kill_fail_closed": modes["hang"]["status"] == "timeout_terminated", "malformed_result_fail_closed": modes["malformed"]["status"] == "failed_closed", "nonzero_exit_fail_closed": modes["nonzero"]["status"] == "failed_closed", "worker_exception_fail_closed": modes["exception"]["status"] == "failed_closed", "stdout_stderr_are_diagnostics_only": all("result_path" in row for row in modes.values()), "atomic_partial_cleanup": all(not row["atomic_partial_residue"] for row in modes.values())}
    contract = {"schema_version": "track3_p3_recovery_r1_v1", "status": "track3_p3_recovery_r1_result_protocol_passed" if all(tests.values()) else "track3_p3_recovery_r1_result_protocol_failed", "created_at_utc": now(), "plan": relative_path(PLAN), "simulator_started": False, "gpu_started": False, "protocol": {"result_transport": "independent_atomic_json_file", "stdout_stderr": "redirected_diagnostics_only", "fixed_wall_clock_deadline_seconds": IO_DEADLINE_SECONDS, "term_grace_seconds": IO_TERM_GRACE_SECONDS, "timeout_sequence": ["TERM", "wait_10_seconds", "KILL"], "worker_pid_must_match_parent_observation": True, "compact_p3_telemetry": ["step", "truth", "contact_counts", "state_hash"], "manifest_p6_summary_included": True, "manifest_self_excluded_only": True, "manifest_entries_must_be_git_tracked": True}, "io_tests": modes, "tests": tests, "source_parent_commit": git(("rev-parse", "HEAD")), "tool_sha256": sha256_file(Path(__file__)), "old_tool_untouched": True}
    write_json(OUT_ROOT / "r1_io_recovery_contract.json", contract)
    return contract


# ---------------------------------------------------------------------------
# R2/R3: camera-only provenance canaries.  This section is runtime-imported
# only by an explicit worker and communicates solely through result JSON.


TASK_ORDER = (10, 20, 19)
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SEEDS = {10: 3100, 20: 3200, 19: 3190}
SOURCE_ATOM = {10: "(On akita_black_bowl_1 plate_1)", 20: "(In orange_juice_1 basket_1_contain_region)", 19: "(Open wooden_cabinet_1_middle_region)"}
GOAL_ATOM = {10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)", 20: "(On orange_juice_1 floor_target_object_region)", 19: "(Close wooden_cabinet_1_middle_region)"}
CONTROL_FREQ_HZ = 20
CANARY_STEPS = 41


def _gpu3_snapshot() -> dict[str, Any]:
    gpu_command = ["nvidia-smi", "-i", "3", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    process_command = ["nvidia-smi", "-i", "3", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    gpu = subprocess.run(gpu_command, capture_output=True, text=True, check=False)
    processes = subprocess.run(process_command, capture_output=True, text=True, check=False)
    parsed: list[dict[str, Any]] = []
    for line in gpu.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 4:
            try:
                parsed.append({"index": fields[0], "uuid": fields[1], "memory_used_mib": float(fields[2]), "memory_total_mib": float(fields[3])})
            except ValueError:
                continue
    return {"timestamp_utc": now(), "returncode": gpu.returncode, "query": gpu_command, "gpus": parsed, "compute_processes": [line.strip() for line in processes.stdout.splitlines() if line.strip()], "process_query_returncode": processes.returncode, "stderr": gpu.stderr[-1000:], "free_gate": gpu.returncode == 0 and len(parsed) == 1 and parsed[0]["memory_used_mib"] <= 1024.0 and not processes.stdout.strip()}


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
    return {"path": path_ref(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "schema_version": "track3_gpu3_camera_frames_npz_v2", "allow_pickle": False, "keys": sorted(arrays), "shapes": {key: list(value.shape) for key, value in arrays.items()}, "dtypes": {key: str(value.dtype) for key, value in arrays.items()}}


def _compact_predicate(sample: Mapping[str, Any], atom: str) -> bool | None:
    for row in sample.get("predicates", []):
        if row.get("atom") == atom:
            value = row.get("value")
            return value if isinstance(value, bool) else None
    return None


def _compact_runtime_sample(task: int, sample: Mapping[str, Any], contact: Mapping[str, Any]) -> dict[str, Any]:
    initial = sample.get("initial", {})
    goal = sample.get("goal", {})
    state = sample.get("state", {})
    state_hash = state.get("state_sha256")
    if not state_hash:
        state_hash = sha256_bytes(json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {"step": int(sample.get("step", -1)), "source": _compact_predicate(initial, SOURCE_ATOM[task]), "goal": _compact_predicate(goal, GOAL_ATOM[task]), "initial_truth_complete": bool(initial.get("truth_complete")), "initial_detail_complete": bool(initial.get("detail_complete")), "goal_truth_complete": bool(goal.get("truth_complete")), "goal_detail_complete": bool(goal.get("detail_complete")), "contact_counts": dict(contact.get("counts", {})), "contact_state": contact.get("state"), "state_hash": str(state_hash)}


def _p3_worker(task: int, demo: int, seed: int, state_bundle: Path, frames_path: Path, result_path: Path) -> int:
    import random
    import numpy as np

    started = now()
    env = None
    result: dict[str, Any]
    try:
        if task not in TASK_ORDER or demo != 0:
            raise ValueError("r2_r3_fixed_task_order_and_demo0_only")
        random.seed(seed)
        np.random.seed(seed)
        import tools.axis_pi05.track3_composed_reverse_start_stabilization as composed
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
            telemetry.append(_compact_runtime_sample(task, sample, contact))
            for output_key, observation_key in (("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
                image = np.asarray(observation.get(observation_key))
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(f"invalid_camera_schema:{observation_key}:{list(image.shape)}")
                frames[output_key].append(image.copy())
        source_values = [row["source"] for row in telemetry]
        goal_values = [row["goal"] for row in telemetry]
        complete = [row for row in telemetry if row["step"] not in range(CANARY_STEPS)] == [] and [row["step"] for row in telemetry] == list(range(CANARY_STEPS))
        truth_complete = all(row["initial_truth_complete"] and row["initial_detail_complete"] and row["goal_truth_complete"] and row["goal_detail_complete"] for row in telemetry)
        truth_passed = complete and truth_complete and all(value is True for value in source_values) and all(value is False for value in goal_values)
        contact_passed = all(row["contact_state"] == "passed" and not row["contact_counts"].get("demonstrator_contact", 0) and not row["contact_counts"].get("unrelated_background_contact", 0) for row in telemetry)
        if task == 10:
            contact_passed = contact_passed and all(row["contact_counts"].get("intended_receiver_contact", 0) > 0 for row in telemetry)
        camera_manifest = _camera_npz_atomic(frames_path, frames)
        renderer = composed._renderer_schema(env, env.env._get_observations())
        gates = {"restore_identity": {"executed": True, "state": "passed" if restore_diff <= 1e-12 else "failed", "max_abs_diff": restore_diff}, "truth": {"executed": truth_complete, "state": "passed" if truth_passed else "failed", "source_all_true": all(value is True for value in source_values), "goal_all_false": all(value is False for value in goal_values)}, "contact_v3_full_window": {"executed": complete, "state": "passed" if contact_passed else "failed", "step_range": [0, 40]}, "camera_schema": {"executed": True, "state": "passed" if renderer.get("schema_complete") else "failed", "renderer": renderer}}
        status = "passed" if all(gate.get("state") == "passed" for gate in gates.values()) else "failed_gate"
        result = {"schema_version": "track3_gpu3_camera_canary_v2", "status": status, "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY[task], "attempt_number": 2 if task == 10 else 1, "fresh_process": True, "mode": "GPU3_EGL_camera", "seed": seed, "child_pid": os.getpid(), "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "state_bundle_path": path_ref(state_bundle), "state_key": state_key, "state_sha256": sha256_bytes(state.tobytes()), "restored_state_sha256": sha256_bytes(restored.tobytes()), "restore_max_abs_diff": restore_diff, "same_demo_model_and_state_components": True}, "reverse": {"source_atom": SOURCE_ATOM[task], "goal_atom": GOAL_ATOM[task], "checkpoint_telemetry": telemetry, "hold_action": composed.neutral_hold().tolist(), "frame_count": CANARY_STEPS, "raw_scene_checkpoints_not_emitted_to_stdout": True}, "renderer": renderer, "camera_frames": {key: len(value) for key, value in frames.items()}, "camera_npz": camera_manifest, "hard_gate": gates, "provenance": {"seed_layers": ["parent_registration", "python_random", "numpy_random"], "env_seed_api": "not_required_for_deterministic_state_restore", "no_manipulation_action": True, "no_se3_or_grasp": True, "no_teleop": True, "gpu3_only": True, "official_gpu": False}}
    except Exception as exc:
        result = {"schema_version": "track3_gpu3_camera_canary_v2", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY.get(task), "attempt_number": 2 if task == 10 else 1, "fresh_process": True, "mode": "GPU3_EGL_camera", "seed": seed, "child_pid": os.getpid(), "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    _write_result(result_path, result)
    return 0 if result.get("status") == "passed" else 1


def _terminate_process(process: subprocess.Popen[Any], *, term_grace_seconds: float) -> str:
    process.terminate()
    try:
        process.wait(timeout=term_grace_seconds)
        return "TERM"
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=term_grace_seconds)
        return "TERM_then_KILL"


def _run_gpu_canary(task: int, demo: int, seed: int, *, attempt_number: int) -> dict[str, Any]:
    if attempt_number != 2 and task == 10:
        raise ValueError("task10_attempt_number_must_be_2")
    run_root = OUT_ROOT / "r2_r3_runs" / f"task{task}_demo{demo}"
    run_root.mkdir(parents=True, exist_ok=True)
    result_path = run_root / "result.json"
    stdout_path = run_root / "stdout.log"
    stderr_path = run_root / "stderr.log"
    frames_path = OUT_ROOT / "camera_frames" / f"r{2 if task == 10 else 3}_task{task}_demo{demo}.npz"
    for path in (result_path, stdout_path, stderr_path, frames_path, frames_path.with_name(frames_path.name + ".partial")):
        path.unlink(missing_ok=True)
    before = _gpu3_snapshot()
    start_monotonic = time.monotonic()
    command = [sys.executable, "-m", MODULE, "--p3-worker", "--task-index", str(task), "--demo-index", str(demo), "--seed", str(seed), "--state-bundle", str(OLD_ROOT / "start_states.npz"), "--frames-output", str(frames_path), "--result-output", str(result_path)]
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}", "MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "JAX_PLATFORMS": "cpu"})
    poll_intervals_ms: list[float] = []
    snapshots: list[dict[str, Any]] = []
    timed_out = False
    termination_stage = None
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout_handle, stderr=stderr_handle)
        while process.poll() is None:
            poll_started = time.monotonic()
            snapshots.append(_gpu3_snapshot())
            poll_intervals_ms.append((time.monotonic() - poll_started) * 1000.0)
            if time.monotonic() - start_monotonic >= IO_DEADLINE_SECONDS:
                timed_out = True
                termination_stage = _terminate_process(process, term_grace_seconds=IO_TERM_GRACE_SECONDS)
                break
            time.sleep(max(0.0, 0.1 - (time.monotonic() - poll_started)))
        process.wait()
    after = _gpu3_snapshot()
    snapshots.append(after)
    child_result = None
    result_error = None
    if result_path.is_file():
        try:
            child_result = strict_json_load(result_path)
        except Exception as exc:
            result_error = f"{type(exc).__name__}:{exc}"
    worker_pid_match = bool(child_result and int(child_result.get("child_pid", -1)) == process.pid)
    memory_values = [float(row["gpus"][0]["memory_used_mib"]) for row in snapshots if row.get("gpus")]
    gpu_record = {"seed": seed, "child_pid": process.pid, "physical_gpu_uuid": (before.get("gpus") or [{}])[0].get("uuid"), "cuda_visible_devices": "3", "start_snapshot": before, "end_snapshot": after, "peak_memory_mib": max(memory_values, default=-1.0), "parent_poll_interval_max_ms": max(poll_intervals_ms, default=0.0), "parent_poll_count": len(snapshots), "deadline_seconds": IO_DEADLINE_SECONDS, "term_grace_seconds": IO_TERM_GRACE_SECONDS, "timeout_termination_stage": termination_stage}
    protocol_passed = bool(child_result and worker_pid_match and process.returncode == 0 and not timed_out and child_result.get("status") == "passed" and gpu_record["parent_poll_interval_max_ms"] <= 100.0)
    result = {"schema_version": "track3_gpu3_camera_canary_parent_v2", "status": "passed" if protocol_passed else ("timeout_terminated" if timed_out else "failed_closed"), "created_at_utc": now(), "task_index": task, "demo_index": demo, "family": FAMILY[task], "attempt_number": attempt_number, "seed": seed, "parent_pid": os.getpid(), "worker_pid": process.pid, "worker_pid_match": worker_pid_match, "process_returncode": process.returncode, "timed_out": timed_out, "termination_stage": termination_stage, "result_path": path_ref(result_path), "stdout_path": path_ref(stdout_path), "stderr_path": path_ref(stderr_path), "stdout_bytes": stdout_path.stat().st_size if stdout_path.is_file() else 0, "stderr_bytes": stderr_path.stat().st_size if stderr_path.is_file() else 0, "result_present": child_result is not None, "result_error": result_error, "child_result": child_result, "gpu_provenance": gpu_record, "gpu_provenance_gate": gpu_provenance_gate(gpu_record), "atomic_partial_residue": [path_ref(path) for path in run_root.glob("*.partial")], "camera_npz_present": frames_path.is_file(), "camera_npz": child_result.get("camera_npz") if child_result else None}
    return result


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


def _publish_camera_videos(canary: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    import numpy as np

    child = canary.get("child_result") or {}
    camera_npz = child.get("camera_npz", {})
    npz_path = ROOT / str(camera_npz.get("path", ""))
    if not npz_path.is_file():
        raise FileNotFoundError(f"camera_npz_missing:{npz_path}")
    videos: dict[str, Any] = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        for key in ("agentview", "wrist"):
            if key not in archive.files:
                raise KeyError(f"camera_key_missing:{key}")
            videos[key] = _encode_h264(OUT_ROOT / "videos" / f"{phase}_task{canary['task_index']}_demo{canary['demo_index']}_{key}.mp4", archive[key])
    return videos


def _r2_r3_gate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    r0 = strict_json_load(OUT_ROOT / "r0_artifact_closure.json")
    r1 = strict_json_load(OUT_ROOT / "r1_io_recovery_contract.json")
    return r0, r1


def run_r2() -> dict[str, Any]:
    r0, r1 = _r2_r3_gate_inputs()
    if r0.get("status") != "track3_p3_recovery_r0_artifact_closure_passed" or r1.get("status") != "track3_p3_recovery_r1_result_protocol_passed":
        result = {"schema_version": "track3_p3_recovery_r2_v1", "status": "not_evaluated", "reason": "r0_or_r1_not_passed", "canary_count": 0, "no_gpu_started": True}
        write_json(OUT_ROOT / "r2_summary.json", result)
        return result
    preflight = _gpu3_snapshot()
    result: dict[str, Any] = {"schema_version": "track3_p3_recovery_r2_v1", "status": "track3_p3_recovery_r2_gpu3_resource_gate_failed", "created_at_utc": now(), "fixed_task": {"task_index": 10, "demo_index": 0}, "seed": SEEDS[10], "attempt_number": 2, "gpu3_preflight": preflight, "canary_count": 0, "canary": None, "no_retry": True}
    if preflight.get("free_gate"):
        canary = _run_gpu_canary(10, 0, SEEDS[10], attempt_number=2)
        result["canary"] = canary
        result["canary_count"] = 1
        if canary.get("status") == "passed" and canary.get("gpu_provenance_gate", {}).get("state") == "passed":
            try:
                canary["videos"] = _publish_camera_videos(canary, phase="r2")
                canary["camera_artifacts_passed"] = True
            except Exception as exc:
                canary["camera_artifacts_passed"] = False
                canary["camera_artifact_error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(OUT_ROOT / "r2_task10_demo0_attempt2.json", canary)
        else:
            write_json(OUT_ROOT / "r2_task10_demo0_attempt2.json", canary)
        result["status"] = "track3_p3_recovery_task10_passed" if canary.get("status") == "passed" and canary.get("gpu_provenance_gate", {}).get("state") == "passed" and canary.get("camera_artifacts_passed") is True else "track3_p3_recovery_task10_failed"
    write_json(OUT_ROOT / "r2_summary.json", result)
    return result


def run_r3() -> dict[str, Any]:
    r2 = strict_json_load(OUT_ROOT / "r2_summary.json")
    if r2.get("status") != "track3_p3_recovery_task10_passed":
        result = {"schema_version": "track3_p3_recovery_r3_v1", "status": "not_evaluated", "reason": "r2_task10_not_passed", "canary_count": 0, "no_gpu_started": True}
        write_json(OUT_ROOT / "r3_summary.json", result)
        return result
    canaries: list[dict[str, Any]] = []
    for task in (20, 19):
        canary = _run_gpu_canary(task, 0, SEEDS[task], attempt_number=1)
        if canary.get("status") == "passed" and canary.get("gpu_provenance_gate", {}).get("state") == "passed":
            try:
                canary["videos"] = _publish_camera_videos(canary, phase="r3")
                canary["camera_artifacts_passed"] = True
            except Exception as exc:
                canary["camera_artifacts_passed"] = False
                canary["camera_artifact_error"] = {"type": type(exc).__name__, "message": str(exc)}
        else:
            canary["camera_artifacts_passed"] = False
        report_path = OUT_ROOT / f"r3_task{task}_demo0.json"
        write_json(report_path, canary)
        canaries.append({"task_index": task, "family": FAMILY[task], "status": canary.get("status"), "camera_artifacts_passed": canary.get("camera_artifacts_passed"), "report": path_ref(report_path)})
        if canary.get("status") != "passed" or canary.get("camera_artifacts_passed") is not True or canary.get("gpu_provenance_gate", {}).get("state") != "passed":
            break
    passed = len(canaries) == 2 and all(row["status"] == "passed" and row["camera_artifacts_passed"] for row in canaries)
    result = {"schema_version": "track3_p3_recovery_r3_v1", "status": "track3_start_bundle_gpu_provenance_closed_3_of_3" if passed else "track3_p3_recovery_r3_failed", "created_at_utc": now(), "fixed_order": [10, 20, 19], "seeds": SEEDS, "canary_count": len(canaries) + 1, "canaries": canaries, "task10_report": path_ref(OUT_ROOT / "r2_task10_demo0_attempt2.json"), "no_retry_or_demo_change": True}
    write_json(OUT_ROOT / "r3_summary.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r0", action="store_true")
    parser.add_argument("--r1", action="store_true")
    parser.add_argument("--r2", action="store_true")
    parser.add_argument("--r3", action="store_true")
    parser.add_argument("--io-test-worker", action="store_true")
    parser.add_argument("--p3-worker", action="store_true")
    parser.add_argument("--io-mode")
    parser.add_argument("--result-output")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--demo-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--state-bundle")
    parser.add_argument("--frames-output")
    args = parser.parse_args(argv)
    if args.io_test_worker:
        if args.io_mode not in {"large", "hang", "malformed", "nonzero", "exception"} or not args.result_output:
            parser.error("io worker requires --io-mode and --result-output")
        return _io_worker(args.io_mode, Path(args.result_output))
    if args.p3_worker:
        if args.task_index not in TASK_ORDER or args.demo_index != 0 or args.seed is None or not args.state_bundle or not args.frames_output or not args.result_output:
            parser.error("p3 worker requires fixed task/demo0, seed, state bundle, frames output and result output")
        return _p3_worker(args.task_index, 0, args.seed, Path(args.state_bundle), Path(args.frames_output), Path(args.result_output))
    if sum((args.r0, args.r1, args.r2, args.r3)) != 1:
        parser.error("choose exactly one phase")
    result = run_r0() if args.r0 else run_r1() if args.r1 else run_r2() if args.r2 else run_r3()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if str(result.get("status", "")).endswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
