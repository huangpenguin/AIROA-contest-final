"""Run the CPU lock and the single authorized allocator-fraction recovery path.

The runner owns the experiment boundary.  It copies the already-finalized
step-0 checkpoint into a fresh namespace, launches a fresh process for 0.90,
and launches 0.95 only when the 0.90 failure is classified as a BFC allocator
OOM during compute.  It never changes the dense recipe or deletes evidence.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any

import numpy as np

from .common import ROOT, canonical_json, sha256_file, sha256_tree, write_json
from .dense_fullft_preflight import (
    AXIS_PARAMS,
    AXIS_ROOT,
    EXPECTED_AXIS_REVISION,
    EXPECTED_LIBERO_REVISION,
    EXPECTED_OPENPI_BASE,
    EXPECTED_PARAM_NUMEL,
    FRESH_NORM,
    NESTED_ROOT,
    PAYLOAD_MANIFEST,
    SAMPLING_MANIFEST,
    STEP0_LOCK,
    _git_state,
    validate_config_contract,
)


REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/dense_fullft_allocator_fraction_recovery"
SOURCE_STEP0 = ROOT / "outputs/axis_pi05/dense_fullft_sharded_init_recovery/train_10/0"
OLD_REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_sharded_init_recovery"
TRAIN_SCRIPT = NESTED_ROOT / "scripts/train.py"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"
FRACTIONS = ("0.90", "0.95")
PHYSICAL_VRAM_MIB = 32768
OLD_NVML_PEAK_MIB = 24897
OLD_OOM_REQUEST_BYTES = 143327232
FORBIDDEN_ALLOCATOR_TOKENS = ("platform", "vmm", "address", "cuda_malloc_async")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(command: list[str], *, cwd: Path = ROOT) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256_file(path)}


def _tree_record(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "tree_sha256": sha256_tree(path),
        "metadata_files": {
            str(item.relative_to(path)): _file_record(item)
            for item in sorted(files)
            if item.name in {"_CHECKPOINT_METADATA", "_METADATA", "_sharding", "manifest.ocdbt"}
        },
    }


def _metadata_groups(path: Path) -> dict[str, Any]:
    metadata = _load_json(path)["tree_metadata"]
    groups: dict[str, dict[str, Any]] = {}
    for encoded_path, value in metadata.items():
        key_path = ast.literal_eval(encoded_path)
        group = str(key_path[0])
        entry = groups.setdefault(group, {"leaf_count": 0, "shapes": {}})
        value_metadata = value.get("value_metadata", {})
        value_type = value_metadata.get("value_type")
        shape = tuple(value_metadata.get("write_shape", ()))
        descriptor = f"{value_type}:{shape}"
        entry["leaf_count"] += 1
        entry["shapes"][descriptor] = entry["shapes"].get(descriptor, 0) + 1
    return groups


def _checkpoint_lock() -> dict[str, Any]:
    if not SOURCE_STEP0.is_dir():
        raise FileNotFoundError(f"immutable step-0 checkpoint missing: {SOURCE_STEP0}")
    required = [
        SOURCE_STEP0 / "_CHECKPOINT_METADATA",
        SOURCE_STEP0 / "params" / "_METADATA",
        SOURCE_STEP0 / "train_state" / "_METADATA",
        SOURCE_STEP0 / "params" / "_sharding",
        SOURCE_STEP0 / "train_state" / "_sharding",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"step-0 checkpoint incomplete: {missing}")
    temporary = [
        str(path.relative_to(SOURCE_STEP0))
        for path in SOURCE_STEP0.rglob("*")
        if path.name.endswith((".tmp", ".partial", ".incomplete"))
    ]
    if temporary:
        raise ValueError(f"step-0 checkpoint has temporary files: {temporary[:5]}")
    old_summary = _load_json(OLD_REPORT_ROOT / "r4_r5_train_10.json")
    if old_summary["checkpoint_steps_observed"] != [0] or old_summary["return_code"] != 1:
        raise ValueError("old step-0 source provenance is not the reviewed finalized checkpoint")
    train_groups = _metadata_groups(SOURCE_STEP0 / "train_state" / "_METADATA")
    params_groups = _metadata_groups(SOURCE_STEP0 / "params" / "_METADATA")
    if train_groups.get("step", {}).get("leaf_count") != 1 or train_groups.get("rng", {}).get("leaf_count") != 1:
        raise ValueError("step-0 train_state metadata does not contain exactly one step and rng")
    if params_groups.get("params", {}).get("leaf_count") != 51:
        raise ValueError("step-0 params metadata leaf count drifted")
    sentinel_rows = _load_json(OLD_REPORT_ROOT / "r3_init_only.json")["sentinels"]
    small_sentinels = [row for row in sentinel_rows if np.prod(row["shape"]) <= 1_000_000]
    return {
        "schema_version": 1,
        "status": "step0_checkpoint_locked",
        "internal_step": 0,
        "checkpoint": _tree_record(SOURCE_STEP0),
        "checkpoint_metadata": _file_record(SOURCE_STEP0 / "_CHECKPOINT_METADATA"),
        "train_state_metadata": _file_record(SOURCE_STEP0 / "train_state" / "_METADATA"),
        "params_metadata": _file_record(SOURCE_STEP0 / "params" / "_METADATA"),
        "metadata_groups": {"train_state": train_groups, "params_item_ema": params_groups},
        "dtype_contract": {
            "params_item_ema": "float32",
            "optimizer_adam": "float32",
            "train_state_step": "int32-or-jax-default-scalar",
            "rng": "uint32-or-jax-key",
            "ema_params_field": "None because EMA params are stored in the params item",
        },
        "parameter_metadata": {"leaf_count": 51, "numel": EXPECTED_PARAM_NUMEL, "dtype": "float32"},
        "restore_sentinels": sentinel_rows,
        "small_restore_sentinels": small_sentinels,
        "source_is_immutable": True,
    }


def _old_evidence_hashes() -> dict[str, str]:
    files = sorted(path for path in OLD_REPORT_ROOT.iterdir() if path.is_file())
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in files}


def _allocator_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "allowed_fractions": list(FRACTIONS),
        "preallocate": True,
        "fraction_variable": "XLA_CLIENT_MEM_FRACTION",
        "deprecated_fraction_variable": "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "deprecated_fraction_must_be_absent": True,
        "forbidden_xla_tokens": list(FORBIDDEN_ALLOCATOR_TOKENS),
        "forbidden_modes": ["preallocate_false", "platform", "vmm", "address", "cuda_malloc_async"],
        "fixed_recipe": {
            "precision": "float32",
            "dense": True,
            "freeze_filter": "nnx.Nothing",
            "ema_decay": 0.999,
            "optimizer": "AdamW",
            "microbatch_per_device": 1,
            "accumulation_steps": 16,
            "effective_global_batch": 64,
        },
    }


def cpu_phase() -> dict[str, Any]:
    if (REPORT_ROOT / "cpu_lock.json").exists():
        raise FileExistsError(f"refusing to overwrite CPU lock: {REPORT_ROOT / 'cpu_lock.json'}")
    config = validate_config_contract()
    if config["status"] != "passed":
        raise ValueError(json.dumps(config, ensure_ascii=False))
    for path in (SOURCE_STEP0, AXIS_PARAMS, FRESH_NORM, PAYLOAD_MANIFEST, SAMPLING_MANIFEST, STEP0_LOCK):
        if not path.exists():
            raise FileNotFoundError(path)
    root_state = _git_state(ROOT)
    nested_state = _git_state(NESTED_ROOT)
    if root_state["dirty"] or nested_state["dirty"]:
        raise RuntimeError(f"CPU lock requires clean root/nested worktrees: {root_state}, {nested_state}")
    checkpoint = _checkpoint_lock()
    old_r4 = _load_json(OLD_REPORT_ROOT / "r4_r5_train_10.json")
    old_input = _load_json(OLD_REPORT_ROOT / "cpu_lock.json")
    root_cause = _load_json(OLD_REPORT_ROOT / "root_cause_lock.json")
    if old_r4["allocator_environment"] != "XLA_PYTHON_CLIENT_PREALLOCATE/MEM_FRACTION unset (JAX default)":
        raise ValueError("old R4 allocator environment is not the locked default")
    if old_r4["gpu_monitor_path"] and not Path(old_r4["gpu_monitor_path"]).is_file():
        raise FileNotFoundError(old_r4["gpu_monitor_path"])
    allocator = _allocator_contract()
    recipe = allocator["fixed_recipe"] | {
        "axis_revision": EXPECTED_AXIS_REVISION,
        "libero_revision": EXPECTED_LIBERO_REVISION,
        "fresh_norm_sha256": sha256_file(FRESH_NORM),
        "payload_manifest_sha256": _load_json(PAYLOAD_MANIFEST)["manifest_sha256"],
        "model_remat_sharding_nested_commit": "92604d6d07d5bbf636827ee903f758538c8ba015",
    }
    pool_mib = int(PHYSICAL_VRAM_MIB * 0.75)
    observed_ratio = OLD_NVML_PEAK_MIB / PHYSICAL_VRAM_MIB
    root = {
        "schema_version": 1,
        "status": "default_75pct_allocator_pool_oom_requires_fraction_retest",
        "previous_root_cause_lock": root_cause,
        "previous_cpu_lock_status": old_input.get("status"),
        "old_evidence_immutable": True,
        "old_evidence_sha256": _old_evidence_hashes(),
        "old_r4": {
            "allocator_environment": old_r4["allocator_environment"],
            "nvml_peak_mib_per_card": OLD_NVML_PEAK_MIB,
            "oom_request_bytes_per_card": OLD_OOM_REQUEST_BYTES,
            "oom_request_mib_per_card": round(OLD_OOM_REQUEST_BYTES / 2**20, 2),
            "compute_started": True,
            "optimizer_updates": 0,
        },
        "allocator_arithmetic": {
            "physical_vram_mib_per_card": PHYSICAL_VRAM_MIB,
            "jax_default_pool_fraction": 0.75,
            "jax_default_pool_mib_per_card": pool_mib,
            "nvml_peak_mib_per_card": OLD_NVML_PEAK_MIB,
            "nvml_peak_fraction": observed_ratio,
            "pool_external_mib_per_card": PHYSICAL_VRAM_MIB - pool_mib,
            "observed_peak_minus_pool_mib": OLD_NVML_PEAK_MIB - pool_mib,
            "estimated_context_driver_nccl_mib": OLD_NVML_PEAK_MIB - pool_mib,
        },
        "step0": checkpoint,
        "recipe": recipe,
        "recipe_sha256": _json_hash(recipe),
        "input_hashes": {
            "axis_params_tree_sha256": sha256_tree(AXIS_PARAMS),
            "axis_revision_file_sha256": sha256_file(AXIS_ROOT / ".axis_pi05_revision"),
            "fresh_norm_sha256": sha256_file(FRESH_NORM),
            "payload_manifest_file_sha256": sha256_file(PAYLOAD_MANIFEST),
            "sampling_manifest_sha256": sha256_file(SAMPLING_MANIFEST),
            "step0_protocol_lock_sha256": sha256_file(STEP0_LOCK),
        },
        "git": {"root": root_state, "nested_openpi": nested_state, "openpi_base": EXPECTED_OPENPI_BASE},
        "gpu_policy": {"official_gpu": False, "gpu0_external_process": "untouched", "local_gpu_ids": [0, 1, 2, 3]},
    }
    lock = {
        "schema_version": 1,
        "status": "dense_fullft_allocator_fraction_cpu_locked",
        "report_root": str(REPORT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "root_cause_lock": root,
        "allocator_contract": allocator,
        "config_contract": config,
        "step0_source": str(SOURCE_STEP0),
        "step0_tree_sha256": checkpoint["checkpoint"]["tree_sha256"],
        "step0_internal_step": 0,
        "step0_source_immutable": True,
        "git": {"root": root_state, "nested_openpi": nested_state},
        "next_allowed": ["A2 fraction 0.90 first update", "A3 fraction 0.95 only after allocator OOM"],
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_ROOT / "allocator_contract.json", allocator)
    write_json(REPORT_ROOT / "step0_lock.json", checkpoint)
    write_json(REPORT_ROOT / "root_cause_lock.json", root)
    write_json(REPORT_ROOT / "cpu_lock.json", lock)
    _write_text(
        REPORT_ROOT / "CPU_LOCK_REPORT_ZH.md",
        "\n".join(
            [
                "# Dense full-FT allocator fraction CPU 锁",
                "",
                "- 状态：`dense_fullft_allocator_fraction_cpu_locked`。",
                f"- 旧 R4 NVML 峰值：`{OLD_NVML_PEAK_MIB}/{PHYSICAL_VRAM_MIB} MiB`；默认 75% pool：`{pool_mib} MiB`。",
                f"- 旧单次 OOM allocation：`{OLD_OOM_REQUEST_BYTES} bytes/card`；旧证据 hash 见 `root_cause_lock.json`。",
                f"- step-0 source tree：`{checkpoint['checkpoint']['tree_sha256']}`，internal step `0`；源目录只读锁定，不会 fresh-init 绕过。",
                "- 固定 FP32、dense、AdamW、EMA `0.999`、microbatch `1`、4 卡 accumulation `16`、global batch `64`。",
                "- 唯一 allocator 自变量：`XLA_CLIENT_MEM_FRACTION=0.90`；只有 allocator/BFC OOM 才允许一次 `0.95`。",
                "- 禁止比赛 GPU、preallocate false、其他 allocator、C20/50/100/1K、FAST、ZIP、上传和提交。",
            ]
        )
        + "\n",
    )
    return lock


def _gpu_snapshot() -> list[dict[str, Any]]:
    try:
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 5:
            continue
        rows.append(
            {
                "index": int(row[0].strip()),
                "name": row[1].strip(),
                "memory_used_mib": int(row[2].strip()),
                "memory_total_mib": int(row[3].strip()),
                "utilization_gpu": int(row[4].strip()),
                "processes": [],
            }
        )
    for gpu in rows:
        try:
            processes = _run(
                [
                    "nvidia-smi",
                    "--id",
                    str(gpu["index"]),
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ]
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        for row in csv.reader(processes.splitlines()):
            if len(row) >= 3 and row[0].strip().isdigit():
                gpu["processes"].append(
                    {"pid": int(row[0].strip()), "name": row[1].strip(), "memory_mib": row[2].strip()}
                )
    return rows


def _monitor(pid: int, path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            row = {"timestamp": time.time(), "monitor_pid": pid, "gpus": _gpu_snapshot()}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            stop.wait(0.25)


def _validate_fraction_environment(env: dict[str, str], fraction: str) -> None:
    if fraction not in FRACTIONS:
        raise ValueError(f"fraction is not pre-registered: {fraction}")
    if env.get("XLA_CLIENT_MEM_FRACTION") != fraction:
        raise ValueError("XLA_CLIENT_MEM_FRACTION does not match requested fraction")
    if env.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "true":
        raise ValueError("preallocation must remain enabled")
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in env:
        raise ValueError("deprecated XLA_PYTHON_CLIENT_MEM_FRACTION must be absent")
    flags = env.get("XLA_FLAGS", "").lower()
    forbidden = [token for token in FORBIDDEN_ALLOCATOR_TOKENS if token in flags]
    if forbidden:
        raise ValueError(f"forbidden allocator XLA_FLAGS: {forbidden}")


def _base_gpu_env(fraction: str, stats_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "JAX_PLATFORMS": "cuda",
            "PYTHONPATH": str(NESTED_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
            "XLA_CLIENT_MEM_FRACTION": fraction,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "AXIS_PI05_ALLOCATOR_STATS_PATH": str(stats_path),
            "AXIS_PI05_EXECUTABLE_MEMORY_ANALYSIS": "1",
            "AXIS_PI05_FULLFT_FSDP_DEVICES": "4",
            "AXIS_PI05_FULLFT_MICROBATCH": "1",
            "AXIS_PI05_FULLFT_ACCUMULATION": "16",
            "AXIS_PI05_FULLFT_EFFECTIVE_BATCH": "64",
            "AXIS_PI05_FULLFT_GLOBAL_BATCH": "64",
            "AXIS_PI05_FULLFT_LR_DECAY_STEPS": "30000",
            "WANDB_MODE": "disabled",
            "JAX_TRACEBACK_FILTERING": "off",
        }
    )
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    _validate_fraction_environment(env, fraction)
    return env


def _copy_step0(target: Path, locked_tree_sha256: str) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint namespace: {target}")
    if sha256_tree(SOURCE_STEP0) != locked_tree_sha256:
        raise RuntimeError("immutable step-0 source changed before copy")
    temporary = target.with_name(f".{target.name}.copying")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse incomplete checkpoint copy: {temporary}")
    started = time.time()
    shutil.copytree(SOURCE_STEP0, temporary)
    if sha256_tree(temporary) != locked_tree_sha256:
        raise RuntimeError(f"copied step-0 tree hash mismatch: {temporary}")
    temporary.rename(target)
    return {
        "source": str(SOURCE_STEP0),
        "target": str(target),
        "tree_sha256": locked_tree_sha256,
        "bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
        "duration_seconds": time.time() - started,
        "atomic_directory_rename": True,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def _checkpoint_steps(path: Path) -> list[int]:
    return sorted(int(item.name) for item in path.iterdir() if item.is_dir() and item.name.isdigit())


def _checkpoint_record(path: Path, step: int) -> dict[str, Any]:
    checkpoint = path / str(step)
    required = [checkpoint / "_CHECKPOINT_METADATA", checkpoint / "params" / "_METADATA", checkpoint / "train_state" / "_METADATA"]
    complete = checkpoint.is_dir() and all(item.is_file() for item in required)
    record: dict[str, Any] = {
        "step": step,
        "path": str(checkpoint),
        "complete": complete,
        "temporary_files": [],
    }
    if not checkpoint.exists():
        return record
    record["temporary_files"] = [
        str(item.relative_to(checkpoint))
        for item in checkpoint.rglob("*")
        if item.name.endswith((".tmp", ".partial", ".incomplete"))
    ]
    if complete and not record["temporary_files"]:
        record.update(_tree_record(checkpoint))
        record["atomic_save_evidence"] = True
    else:
        record["atomic_save_evidence"] = False
    return record


def _classify_failure(log_path: Path, stats: list[dict[str, Any]], timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    lowered = text.lower()
    oom_markers = ("resource_exhausted", "out of memory", "allocator", "bfc")
    oom = any(marker in lowered for marker in oom_markers)
    compute_phases = {row.get("phase") for row in stats if row.get("phase") in {"microbatch_exception", "microbatch_complete"}}
    if oom and "microbatch_exception" in compute_phases:
        return "allocator_bfc_oom_during_compute"
    if oom:
        return "oom_without_complete_compute_phase_evidence"
    if any(row.get("phase") == "checkpoint_restore" for row in stats):
        return "non_oom_after_restore_or_training_error"
    return "non_oom_or_early_process_error"


def _run_train_process(
    *, fraction: str, run_dir: Path, label: str, steps: int, expected_sentinels: list[dict[str, Any]], timeout_seconds: int
) -> dict[str, Any]:
    report_path = REPORT_ROOT / f"{label}.json"
    log_path = REPORT_ROOT / f"{label}.log"
    monitor_path = REPORT_ROOT / f"{label}_nvml.jsonl"
    stats_path = REPORT_ROOT / f"{label}_allocator.jsonl"
    validation_path = run_dir / ("restore_step0_validation.json" if label.endswith("canary") else "restore_step1_validation.json")
    command = [str(PYTHON), str(TRAIN_SCRIPT), "axis_pi05_libero_fullft"]
    env = _base_gpu_env(fraction, stats_path)
    env.update(
        {
            "AXIS_PI05_FULLFT_RUN_ID": f"allocator_fraction_{fraction}_{label}",
            "AXIS_PI05_FULLFT_RUN_DIR": str(run_dir),
            "AXIS_PI05_FULLFT_STEPS": str(steps),
            "AXIS_PI05_FULLFT_SAVE_INTERVAL": str(steps),
            "AXIS_PI05_FULLFT_LOG_INTERVAL": "1",
            "AXIS_PI05_FULLFT_OVERWRITE": "0",
            "AXIS_PI05_FULLFT_RESUME": "1",
            "OPENPI_KEEP_STEPS": "0,1,10",
            "AXIS_PI05_SAVE_STEP0": "0",
            "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
            "AXIS_PI05_CHECKPOINT_ASYNC": "0",
            "AXIS_PI05_EXPECTED_PARAMS_SENTINELS": json.dumps(expected_sentinels, separators=(",", ":")),
            "AXIS_PI05_RESTORE_VALIDATION_PATH": str(validation_path),
        }
    )
    _validate_fraction_environment(env, fraction)
    before = _gpu_snapshot()
    before_processes = sorted({process["pid"] for gpu in before for process in gpu.get("processes", [])})
    if set(row["index"] for row in before) != {0, 1, 2, 3} or any(row["memory_used_mib"] > 256 for row in before):
        raise RuntimeError(f"four local GPUs are not naturally idle before {label}: {before}")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()
    monitor_thread = threading.Thread(target=_monitor, args=(process.pid, monitor_path, stop), daemon=True)
    monitor_thread.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=30)
    finally:
        stop.set()
        monitor_thread.join(timeout=10)
    monitor = _read_jsonl(monitor_path)
    stats = _read_jsonl(stats_path)
    metrics_path = run_dir / "training_metrics.jsonl"
    metrics = _read_jsonl(metrics_path)
    steps_observed = _checkpoint_steps(run_dir)
    validation = _load_json(validation_path) if validation_path.is_file() else {}
    pids_during = sorted({process_row["pid"] for row in monitor for gpu in row.get("gpus", []) for process_row in gpu.get("processes", [])})
    external_pids = sorted(set(before_processes) | (set(pids_during) - {process.pid}))
    finite_metrics = all(
        all(np.isfinite(float(value)) for key, value in row.items() if key != "optimizer_step")
        for row in metrics
    )
    checkpoint_records = {str(step): _checkpoint_record(run_dir, step) for step in steps_observed if step in {0, 1, 10}}
    report = {
        "schema_version": 1,
        "phase": label,
        "fraction": fraction,
        "command": command,
        "command_fraction": env["XLA_CLIENT_MEM_FRACTION"],
        "allocator_environment": {
            key: env.get(key)
            for key in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS")
        },
        "visible_devices": [0, 1, 2, 3],
        "fsdp_devices": 4,
        "microbatch_per_device": 1,
        "accumulation_steps": 16,
        "effective_global_batch": 64,
        "precision": "float32",
        "ema_decay": 0.999,
        "steps_target": steps,
        "started_at_unix": started,
        "duration_seconds": time.time() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "failure_classification": _classify_failure(log_path, stats, timed_out),
        "before_gpus": before,
        "after_gpus": _gpu_snapshot(),
        "external_pids": external_pids,
        "gpu0_external_process_touched": any(pid not in {process.pid} for pid in external_pids),
        "log_path": str(log_path),
        "nvml_trace_path": str(monitor_path),
        "allocator_stats_path": str(stats_path),
        "allocator_stats_samples": len(stats),
        "nvml_samples": len(monitor),
        "restore_validation_path": str(validation_path),
        "restore_validation": validation,
        "checkpoint_dir": str(run_dir),
        "checkpoint_steps_observed": steps_observed,
        "checkpoint_records": checkpoint_records,
        "metrics": metrics,
        "metric_steps_observed": sorted({int(row["optimizer_step"]) for row in metrics if "optimizer_step" in row}),
        "finite_metrics": finite_metrics,
        "recipe_changed_to_fit": False,
        "prohibited_phases_started": [],
    }
    write_json(report_path, report)
    return report


def _child_restore_only() -> None:
    import jax
    import openpi.training.checkpoints as checkpoints
    import openpi.training.config as config_module
    import openpi.training.sharding as sharding

    train = _load_train_module()
    config = config_module.get_config("axis_pi05_libero_fullft")
    mesh = sharding.make_mesh(4)
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
        enable_async_checkpointing=False,
    )
    if not resuming:
        raise RuntimeError("restore-only child did not see an existing checkpoint")
    state, _ = train.init_train_state(config, init_rng, mesh, resume=True)
    jax.block_until_ready(state)
    restored = checkpoints.restore_state(manager, state, None)
    jax.block_until_ready(restored)
    train._record_allocator_stats("checkpoint_restore_only", train_state_step=int(restored.step))
    train._write_restore_validation(restored)
    report = {
        "schema_version": 1,
        "status": "restore_only_passed",
        "restored_step": int(restored.step),
        "checkpoint_dir": str(config.checkpoint_dir),
        "allocator_fraction": os.environ.get("XLA_CLIENT_MEM_FRACTION"),
    }
    write_json(Path(os.environ["AXIS_PI05_RECOVERY_REPORT"]), report)


def _load_train_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("axis_openpi_train_allocator_recovery", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_restore_only(*, fraction: str, run_dir: Path, label: str, timeout_seconds: int) -> dict[str, Any]:
    report_path = REPORT_ROOT / f"{label}.json"
    log_path = REPORT_ROOT / f"{label}.log"
    stats_path = REPORT_ROOT / f"{label}_allocator.jsonl"
    validation_path = run_dir / f"{label}_validation.json"
    env = _base_gpu_env(fraction, stats_path)
    env.update(
        {
            "AXIS_PI05_FULLFT_RUN_DIR": str(run_dir),
            "AXIS_PI05_FULLFT_RESUME": "1",
            "AXIS_PI05_FULLFT_OVERWRITE": "0",
            "AXIS_PI05_FULLFT_RUN_ID": f"allocator_fraction_{fraction}_{label}",
            "AXIS_PI05_EXPECTED_PARAMS_SENTINELS": "[]",
            "AXIS_PI05_RESTORE_VALIDATION_PATH": str(validation_path),
            "AXIS_PI05_RECOVERY_CHILD_PHASE": "restore_only",
            "AXIS_PI05_RECOVERY_REPORT": str(report_path),
        }
    )
    _validate_fraction_environment(env, fraction)
    before = _gpu_snapshot()
    if set(row["index"] for row in before) != {0, 1, 2, 3} or any(row["memory_used_mib"] > 256 for row in before):
        raise RuntimeError(f"four local GPUs are not naturally idle before {label}: {before}")
    command = [str(PYTHON), "-m", "tools.axis_pi05.dense_fullft_allocator_fraction_recovery", "--phase", "_child"]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()
    monitor_path = REPORT_ROOT / f"{label}_nvml.jsonl"
    monitor_thread = threading.Thread(target=_monitor, args=(process.pid, monitor_path, stop), daemon=True)
    monitor_thread.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        return_code = process.wait(timeout=30)
    finally:
        stop.set()
        monitor_thread.join(timeout=10)
    child = _load_json(report_path) if report_path.is_file() else {}
    validation = _load_json(validation_path) if validation_path.is_file() else {}
    report = {
        "schema_version": 1,
        "phase": label,
        "fraction": fraction,
        "command": command,
        "command_fraction": env["XLA_CLIENT_MEM_FRACTION"],
        "allocator_environment": {key: env.get(key) for key in ("XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS")},
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": time.time() - started,
        "before_gpus": before,
        "after_gpus": _gpu_snapshot(),
        "child_report": child,
        "restore_validation": validation,
        "restore_validation_path": str(validation_path),
        "allocator_stats_path": str(stats_path),
        "nvml_trace_path": str(monitor_path),
        "allocator_stats_samples": len(_read_jsonl(stats_path)),
        "nvml_samples": len(_read_jsonl(monitor_path)),
        "log_path": str(log_path),
    }
    report["status"] = "restore_only_passed" if return_code == 0 and child.get("status") == "restore_only_passed" and validation.get("status") == "passed" else "restore_only_failed"
    write_json(report_path, report)
    return report


def _full_report_status(report: dict[str, Any], expected_step: int) -> bool:
    return (
        report.get("return_code") == 0
        and not report.get("timed_out")
        and not report.get("external_pids")
        and report.get("restore_validation", {}).get("status") == "passed"
        and expected_step in report.get("checkpoint_steps_observed", [])
        and expected_step in report.get("metric_steps_observed", [])
        and report.get("finite_metrics") is True
        and report.get("checkpoint_records", {}).get(str(expected_step), {}).get("atomic_save_evidence") is True
    )


def _source_still_locked(lock: dict[str, Any]) -> bool:
    return sha256_tree(SOURCE_STEP0) == lock["step0_tree_sha256"]


def run_recovery(timeout_seconds: int = 7200) -> dict[str, Any]:
    lock = _load_json(REPORT_ROOT / "cpu_lock.json")
    if lock.get("status") != "dense_fullft_allocator_fraction_cpu_locked":
        raise RuntimeError("CPU allocator lock is missing or invalid")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    step0_tree = lock["step0_tree_sha256"]
    expected_small = lock["root_cause_lock"]["step0"]["small_restore_sentinels"]
    run090 = OUTPUT_ROOT / "fraction_090"
    copy090 = _copy_step0(run090, step0_tree)
    report090 = _run_train_process(
        fraction="0.90",
        run_dir=run090,
        label="a2_fraction_090_canary",
        steps=1,
        expected_sentinels=expected_small,
        timeout_seconds=timeout_seconds,
    )
    selected_fraction = None
    report095: dict[str, Any] | None = None
    copy095: dict[str, Any] | None = None
    passing_canary: dict[str, Any] | None = None
    restore1: dict[str, Any] | None = None
    continuation: dict[str, Any] | None = None
    restore10: dict[str, Any] | None = None
    if _full_report_status(report090, 1):
        selected_fraction = "0.90"
        passing_canary = report090
    elif report090.get("failure_classification") == "allocator_bfc_oom_during_compute" and not report090.get("external_pids") and _source_still_locked(lock):
        run095 = OUTPUT_ROOT / "fraction_095"
        copy095 = _copy_step0(run095, step0_tree)
        report095 = _run_train_process(
            fraction="0.95",
            run_dir=run095,
            label="a3_fraction_095_canary",
            steps=1,
            expected_sentinels=expected_small,
            timeout_seconds=timeout_seconds,
        )
        if _full_report_status(report095, 1):
            selected_fraction = "0.95"
            passing_canary = report095
    if selected_fraction is not None and passing_canary is not None:
        selected_dir = run090 if selected_fraction == "0.90" else OUTPUT_ROOT / "fraction_095"
        restore1 = _run_restore_only(
            fraction=selected_fraction,
            run_dir=selected_dir,
            label=f"a4_restore_step1_{selected_fraction.replace('.', '')}",
            timeout_seconds=timeout_seconds,
        )
        if restore1.get("status") == "restore_only_passed":
            continuation = _run_train_process(
                fraction=selected_fraction,
                run_dir=selected_dir,
                label=f"a4_fraction_{selected_fraction.replace('.', '')}_continue_10",
                steps=10,
                expected_sentinels=[],
                timeout_seconds=timeout_seconds,
            )
            if _full_report_status(continuation, 10):
                restore10 = _run_restore_only(
                    fraction=selected_fraction,
                    run_dir=selected_dir,
                    label=f"a4_restore_step10_{selected_fraction.replace('.', '')}",
                    timeout_seconds=timeout_seconds,
                )
    if selected_fraction is not None and continuation is not None and restore10 is not None and restore10.get("status") == "restore_only_passed" and _full_report_status(continuation, 10):
        status = "dense_fullft_local_10step_allocator_passed_pending_review"
    elif report095 is not None and report095.get("failure_classification") == "allocator_bfc_oom_during_compute":
        status = "dense_fullft_local_95pct_pool_true_capacity_oom_pending_official_gpu_review"
    else:
        status = "dense_fullft_allocator_fraction_recovery_failed"
    summary = {
        "schema_version": 1,
        "status": status,
        "plan": "AXIS_PI05_DENSE_FULLFT_ALLOCATOR_FRACTION_RECOVERY_PLAN",
        "cpu_lock": str(REPORT_ROOT / "cpu_lock.json"),
        "source_step0_tree_sha256": step0_tree,
        "copy_090": copy090,
        "a2_fraction_090": report090,
        "copy_095": copy095,
        "a3_fraction_095": report095,
        "selected_fraction": selected_fraction,
        "a4_restore_step1": restore1,
        "a4_continue_10": continuation,
        "a4_restore_step10": restore10,
        "prohibited_phases_started": [],
        "official_gpu_started": False,
        "competition_gpu_started": False,
        "c20_started": False,
        "long_training_started": False,
        "fast_zip_upload_submission": False,
        "source_step0_unchanged_at_end": _source_still_locked(lock),
        "after_gpus": _gpu_snapshot(),
        "git": {"root": _git_state(ROOT), "nested_openpi": _git_state(NESTED_ROOT)},
    }
    write_json(REPORT_ROOT / "summary.json", summary)
    _write_text(
        REPORT_ROOT / "RECOVERY_REPORT_ZH.md",
        "\n".join(
            [
                "# Dense full-FT allocator fraction recovery",
                "",
                f"- 最终状态：`{status}`。",
                f"- 0.90：`{report090.get('failure_classification')}`，return code `{report090.get('return_code')}`。",
                f"- 0.95：`{report095.get('failure_classification') if report095 else '未授权/未运行'}`。",
                f"- 通过 fraction：`{selected_fraction or 'none'}`。",
                "- 训练配方保持 FP32/dense/AdamW/EMA 0.999/microbatch 1/global batch 64；没有为适配显存而改 recipe。",
                "- 本轮没有启动比赛 GPU、C20、长程训练、FAST、ZIP、上传或提交。",
                "- allocator internal stats、XLA memory analysis（若后端提供）和 0.25s NVML trace 由同名 phase artifacts 提供。",
                "- 现在停止，等待人工审核。",
            ]
        )
        + "\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "run", "_child"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    if args.phase == "cpu":
        result = cpu_phase()
        print(json.dumps({"status": result["status"], "step0_tree_sha256": result["step0_tree_sha256"]}, indent=2))
    elif args.phase == "run":
        result = run_recovery(args.timeout_seconds)
        print(json.dumps({"status": result["status"], "selected_fraction": result["selected_fraction"]}, indent=2))
    else:
        if os.environ.get("AXIS_PI05_RECOVERY_CHILD_PHASE") != "restore_only":
            raise RuntimeError("unknown recovery child phase")
        _child_restore_only()


if __name__ == "__main__":
    main()
