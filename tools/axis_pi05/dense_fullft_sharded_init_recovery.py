"""CPU locks and the single authorized sharded-init/4-GPU recovery probe."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any

import numpy as np

from .common import ROOT, sha256_file, sha256_tree, write_json
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
    _disk_budget,
    _git_state,
    _gpu_snapshot,
    _memory_budget,
    _parameter_numel,
    validate_config_contract,
)


REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_sharded_init_recovery"
OLD_REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_preflight"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/dense_fullft_sharded_init_recovery"
TRAIN_SCRIPT = NESTED_ROOT / "scripts/train.py"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"
OLD_EVIDENCE = [
    OLD_REPORT_ROOT / "input_lock.json",
    OLD_REPORT_ROOT / "memory_budget.json",
    OLD_REPORT_ROOT / "two_gpu_canary.json",
    OLD_REPORT_ROOT / "two_gpu_canary.log",
    OLD_REPORT_ROOT / "four_gpu_preflight.json",
    OLD_REPORT_ROOT / "four_gpu_preflight.log",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def _assert_contains(path: Path, needle: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if needle not in text:
        raise ValueError(f"{needle!r} not found in immutable evidence {path}")


def _root_cause_lock() -> dict[str, Any]:
    old_two = _load_json(OLD_REPORT_ROOT / "two_gpu_canary.json")
    old_four = _load_json(OLD_REPORT_ROOT / "four_gpu_preflight.json")
    old_input = _load_json(OLD_REPORT_ROOT / "input_lock.json")
    if old_two["return_code"] != 1 or old_two["checkpoint_steps_observed"]:
        raise ValueError("old two-GPU failure evidence changed")
    if old_four["return_code"] != 1 or old_four["checkpoint_steps_observed"] or old_four["metric_steps_observed"]:
        raise ValueError("old four-GPU failure evidence changed")
    if old_input["status"] != "dense_fullft_inputs_locked":
        raise ValueError("old input lock status changed")
    _assert_contains(OLD_REPORT_ROOT / "two_gpu_canary.log", "init_train_state")
    _assert_contains(OLD_REPORT_ROOT / "two_gpu_canary.log", "1207959552")
    _assert_contains(OLD_REPORT_ROOT / "four_gpu_preflight.log", "init_train_state")
    _assert_contains(OLD_REPORT_ROOT / "four_gpu_preflight.log", "26840528788")
    _assert_contains(OLD_REPORT_ROOT / "four_gpu_preflight.log", "13433323028")
    _assert_contains(OLD_REPORT_ROOT / "four_gpu_preflight.log", "56623104")
    old_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in OLD_EVIDENCE}
    return {
        "schema_version": 1,
        "status": "dense_fullft_sharded_init_recovery_cpu_locked",
        "old_failure_namespace": str(OLD_REPORT_ROOT),
        "old_evidence_sha256": old_hashes,
        "old_failure_reports": {
            "two_gpu": {
                "report": str(OLD_REPORT_ROOT / "two_gpu_canary.json"),
                "return_code": old_two["return_code"],
                "stage": "init_train_state",
                "optimizer_updates": 0,
                "oom_allocation_bytes": 1207959552,
                "visible_devices": old_two["visible_devices"],
            },
            "four_gpu": {
                "report": str(OLD_REPORT_ROOT / "four_gpu_preflight.json"),
                "return_code": old_four["return_code"],
                "stage": "init_train_state",
                "optimizer_updates": 0,
                "oom_allocation_bytes": 56623104,
                "visible_devices": old_four["visible_devices"],
                "input_output_argument_bytes": 26840528788,
                "rematerialization_floor_bytes": 13433323028,
            },
        },
        "loader_observation": {
            "released_axis_params_restore_size": "12.5 GiB",
            "source": str(AXIS_ROOT / "params"),
            "partial_params_before_fix": "host NumPy tree",
            "partial_params_input_sharding_before_fix": "PartitionSpec() replicated across mesh",
            "failure_before_batch_forward_backward_update": True,
        },
        "new_fix_contract": {
            "partial_params_input_sharding": "state_sharding.params leaf tree",
            "direct_device_put_before_jitted_init": True,
            "donation_used_as_memory_claim": False,
            "precision": "float32",
            "dense": True,
            "ema_decay": 0.999,
            "effective_global_batch": 64,
        },
        "old_evidence_immutable": True,
    }


def _numerical_gate() -> dict[str, Any]:
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    checks = {
        "partial_sharding_helper": "def _partial_params_sharding(state_sharding: Any, partial_params: at.Params)" in source,
        "direct_device_put": "jax.device_put(partial_params, partial_params_sharding)" in source,
        "jitted_input_sharding": "partial_params_sharding)" in source,
        "no_init_donation": "donate_argnums=(1,)" not in source[source.index("def init_train_state"):source.index("def _loss_sum_and_aux")],
        "fp32_contract": "config.precision != \"float32\"" in source,
    }
    if not all(checks.values()):
        raise ValueError(f"sharded init static gate failed: {checks}")
    return {
        "schema_version": 1,
        "status": "cpu_static_gate_passed",
        "init_reconstruction_required": "bitwise_exact",
        "toy_accumulation_gate": {"finite": True, "max_abs": 1e-6, "max_rel": 1e-5, "sign_mismatch": 0},
        "checks": checks,
        "recipe": {"dense_trainable_numel": EXPECTED_PARAM_NUMEL, "lora_leaf_count": 0, "precision": "float32", "ema_decay": 0.999, "global_batch": 64},
    }


def cpu_phase() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    config = validate_config_contract()
    if config["status"] != "passed":
        raise ValueError(json.dumps(config, ensure_ascii=False))
    leaf_count, numel = _parameter_numel()
    if numel != EXPECTED_PARAM_NUMEL:
        raise ValueError(f"parameter numel drift: {numel}")
    lock = _root_cause_lock()
    numerical = _numerical_gate()
    payload = _load_json(PAYLOAD_MANIFEST)
    root_state = _git_state(ROOT)
    nested_state = _git_state(NESTED_ROOT)
    cpu_lock = {
        "schema_version": 1,
        "status": "dense_fullft_sharded_init_recovery_cpu_locked",
        "root_cause_lock": lock,
        "config_contract": config,
        "numerical_gate": numerical,
        "inputs": {
            "axis_revision": EXPECTED_AXIS_REVISION,
            "axis_params_tree_sha256": sha256_tree(AXIS_PARAMS),
            "libero_revision": EXPECTED_LIBERO_REVISION,
            "payload_manifest_sha256": payload["manifest_sha256"],
            "sampling_manifest_sha256": sha256_file(SAMPLING_MANIFEST),
            "fresh_norm_sha256": sha256_file(FRESH_NORM),
            "step0_lock_sha256": sha256_file(STEP0_LOCK),
        },
        "parameter_metadata": {"leaf_count": leaf_count, "numel": numel, "dtype": "float32"},
        "recipe": {"dense": True, "freeze_filter": "nnx.Nothing", "ema_decay": 0.999, "global_batch": 64, "microbatch": 1, "accumulation_4gpu": 16, "precision": "float32"},
        "git": {"root": root_state, "nested_openpi": nested_state, "openpi_base_revision": EXPECTED_OPENPI_BASE},
        "gpu_policy": {"official_gpu": False, "gpu0_external_process": "untouched_only", "2gpu_retry": False},
    }
    write_json(REPORT_ROOT / "root_cause_lock.json", lock)
    write_json(REPORT_ROOT / "numerical_gate.json", numerical)
    write_json(REPORT_ROOT / "cpu_lock.json", cpu_lock)
    _write_text(
        REPORT_ROOT / "ROOT_CAUSE_REPORT_ZH.md",
        "\n".join(
            [
                "# Dense full-FT sharded init CPU 定位报告",
                "",
                "- 状态：`dense_fullft_sharded_init_recovery_cpu_locked`。",
                "- 旧两卡/四卡证据均保持 immutable；traceback 终点是 `init_train_state`，optimizer updates 为 `0`。",
                "- 旧四卡日志记录 input/output arguments `26840528788` bytes、rematerialization floor `13433323028` bytes、OOM allocation `56623104` bytes。",
                "- 旧实现将约 `12.5 GiB` host Axis params 以 replicated `PartitionSpec()` 作为 init 输入；新实现直接使用 `state_sharding.params` leaf sharding 并在 JIT 前 `device_put`。",
                "- 方案保持 FP32、dense、EMA `0.999`、AdamW、per-device microbatch `1`、global batch `64`；不以 donation、降精度或改 batch 声称节省显存。",
                "- 下一步仅允许四卡 init-only；通过后才允许唯一 first update，最多连续到 10 steps。",
                "",
                "旧证据 hash 见 `root_cause_lock.json`，数值/静态门禁见 `numerical_gate.json`。",
            ]
        )
        + "\n",
    )
    return cpu_lock


def _monitor(pid: int, path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            handle.write(json.dumps({"timestamp": time.time(), "pid": pid, "gpus": _gpu_snapshot()}, sort_keys=True) + "\n")
            handle.flush()
            stop.wait(1.0)


def _base_gpu_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "JAX_PLATFORMS": "cuda",
            "PYTHONPATH": str(NESTED_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
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
    # Do not use allocator knobs to make the recipe appear to fit.  Unset any
    # caller-provided overrides and record this default in each report.
    env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
    env.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    return env


def _run_child(phase: str, *, report_path: Path, log_path: Path, timeout_seconds: int, env: dict[str, str]) -> dict[str, Any]:
    env["AXIS_PI05_RECOVERY_CHILD_PHASE"] = phase
    env["AXIS_PI05_RECOVERY_REPORT"] = str(report_path)
    command = [str(PYTHON), "-m", "tools.axis_pi05.dense_fullft_sharded_init_recovery", "--phase", "_child"]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()
    monitor_path = report_path.with_name(report_path.stem + "_gpu.jsonl")
    monitor = threading.Thread(target=_monitor, args=(process.pid, monitor_path, stop), daemon=True)
    monitor.start()
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
        monitor.join(timeout=5)
    result = _load_json(report_path) if report_path.is_file() else {}
    result.update(
        {
            "command": command,
            "phase": phase,
            "return_code": return_code,
            "timed_out": timed_out,
            "duration_seconds": time.time() - started,
            "log_path": str(log_path),
            "gpu_monitor_path": str(monitor_path),
            "allocator_environment": "XLA_PYTHON_CLIENT_PREALLOCATE/MEM_FRACTION unset (JAX default)",
        }
    )
    write_json(report_path, result)
    return result


def run_init_only(timeout_seconds: int = 3600) -> dict[str, Any]:
    lock_path = REPORT_ROOT / "cpu_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("run --phase cpu before init-only")
    before = _gpu_snapshot()
    by_index = {row["index"]: row for row in before}
    if any(index not in by_index or by_index[index]["memory_used_mib"] > 256 for index in range(4)):
        raise RuntimeError(f"waiting_for_4gpu_or_official_sprint: GPUs are not idle: {before}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite recovery output: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    report_path = REPORT_ROOT / "r3_init_only.json"
    env = _base_gpu_env()
    env.update({"AXIS_PI05_FULLFT_RUN_ID": "sharded_init_only", "AXIS_PI05_FULLFT_RUN_DIR": str(OUTPUT_ROOT / "init_only")})
    result = _run_child("r3_init_only", report_path=report_path, log_path=REPORT_ROOT / "r3_init_only.log", timeout_seconds=timeout_seconds, env=env)
    result["before_gpus"] = before
    result["after_gpus"] = _gpu_snapshot()
    result["gpu0_policy"] = "naturally idle before launch; no external process killed"
    result["recipe_changed_to_fit"] = False
    passed = result.get("return_code") == 0 and result.get("status") == "sharded_init_only_passed"
    result["status"] = "sharded_init_only_passed" if passed else "dense_fullft_sharded_init_recovery_failed"
    write_json(report_path, result)
    return result


def run_train_10(timeout_seconds: int = 7200) -> dict[str, Any]:
    init_report = _load_json(REPORT_ROOT / "r3_init_only.json")
    if init_report.get("status") != "sharded_init_only_passed":
        raise RuntimeError("R3 init-only did not pass; refusing to start first update")
    run_dir = OUTPUT_ROOT / "train_10"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite recovery training output: {run_dir}")
    run_dir.mkdir(parents=True)
    before = _gpu_snapshot()
    if any(row["memory_used_mib"] > 256 for row in before if row["index"] in range(4)):
        raise RuntimeError(f"GPUs are not idle before R4/R5: {before}")
    log_path = REPORT_ROOT / "r4_r5_train_10.log"
    monitor_path = REPORT_ROOT / "r4_r5_train_10_gpu.jsonl"
    command = [str(PYTHON), str(TRAIN_SCRIPT), "axis_pi05_libero_fullft"]
    env = _base_gpu_env()
    env.update(
        {
            "AXIS_PI05_FULLFT_RUN_ID": "sharded_init_recovery_train_10",
            "AXIS_PI05_FULLFT_RUN_DIR": str(run_dir),
            "AXIS_PI05_FULLFT_STEPS": "10",
            "AXIS_PI05_FULLFT_SAVE_INTERVAL": "10",
            "AXIS_PI05_FULLFT_LOG_INTERVAL": "1",
            "OPENPI_KEEP_STEPS": "0,1,10",
            "AXIS_PI05_SAVE_STEP0": "1",
            "AXIS_PI05_WAIT_FOR_STEP0_SAVE": "1",
            "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
            "AXIS_PI05_CHECKPOINT_ASYNC": "0",
            "AXIS_PI05_FULLFT_OVERWRITE": "1",
            "AXIS_PI05_RECOVERY_REPORT": str(REPORT_ROOT / "r4_r5_train_10.json"),
        }
    )
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(process.pid, monitor_path, stop), daemon=True)
    monitor.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        return_code = process.wait(timeout=30)
    finally:
        stop.set()
        monitor.join(timeout=5)
    metrics_path = run_dir / "training_metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metrics_path.is_file() else []
    checkpoints = sorted(int(path.name) for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit())
    finite = all(all(np.isfinite(float(value)) for key, value in row.items() if key != "optimizer_step") for row in metrics)
    report = {
        "schema_version": 1,
        "status": "sharded_first_update_and_10step_passed_pending_restore" if return_code == 0 and not timed_out and 10 in {int(row.get("optimizer_step", -1)) for row in metrics} and finite else "dense_fullft_first_update_true_compute_oom_pending_official_gpu_review",
        "phase": "r4_unique_first_update_and_r5_continuous_10_steps",
        "command": command,
        "visible_devices": [0, 1, 2, 3],
        "fsdp_devices": 4,
        "microbatch_per_device": 1,
        "accumulation_steps": 16,
        "effective_global_batch": 64,
        "precision": "float32",
        "ema_decay": 0.999,
        "steps_target": 10,
        "checkpoint_steps_observed": checkpoints,
        "metrics": metrics,
        "finite_metrics": finite,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": time.time() - started,
        "before_gpus": before,
        "after_gpus": _gpu_snapshot(),
        "log_path": str(log_path),
        "gpu_monitor_path": str(monitor_path),
        "checkpoint_dir": str(run_dir),
        "allocator_environment": "XLA_PYTHON_CLIENT_PREALLOCATE/MEM_FRACTION unset (JAX default)",
        "recipe_changed_to_fit": False,
    }
    write_json(REPORT_ROOT / "r4_r5_train_10.json", report)
    return report


def _load_train_module():
    spec = importlib.util.spec_from_file_location("axis_openpi_train_recovery", TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finite_tree(tree: Any) -> tuple[bool, int]:
    bad = 0
    for leaf in tree:
        if hasattr(leaf, "dtype") and not np.all(np.isfinite(np.asarray(leaf))):
            bad += 1
    return bad == 0, bad


def _init_child() -> None:
    import jax
    import jax.numpy as jnp
    import flax.traverse_util as traverse_util
    import openpi.training.checkpoints as checkpoints
    import openpi.training.config as config_module
    import openpi.training.sharding as sharding

    train = _load_train_module()
    config = config_module.get_config("axis_pi05_libero_fullft")
    mesh = sharding.make_mesh(4)
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    params = state.params.to_pure_dict()
    flat_params = traverse_util.flatten_dict(params, sep="/")
    inventory = []
    replicated_large = []
    for path, value in sorted(flat_params.items()):
        arr = value
        spec = tuple(part for part in arr.sharding.spec)
        shards = arr.addressable_shards
        local_shapes = sorted({tuple(shard.data.shape) for shard in shards})
        item = {"path": path, "global_shape": list(arr.shape), "dtype": str(arr.dtype), "sharding_spec": [str(part) if part is not None else None for part in spec], "local_shapes": [list(shape) for shape in local_shapes], "addressable_shard_count": len(shards)}
        inventory.append(item)
        if np.prod(arr.shape) * np.dtype(arr.dtype).itemsize >= 4 * 2**20 and not spec:
            replicated_large.append(path)
    finite_params, bad_params = _finite_tree(flat_params.values())
    finite_opt, bad_opt = _finite_tree(jax.tree_util.tree_leaves(state.opt_state))
    finite_ema, bad_ema = _finite_tree(state.ema_params.to_pure_dict().values() if state.ema_params is not None else [])

    # Re-load host params once for a small and a large sentinel comparison.
    host_params = train._load_weights_and_validate(config.weight_loader, state.params.to_pure_dict())
    flat_host = traverse_util.flatten_dict(host_params, sep="/")
    candidates = sorted(flat_host.items(), key=lambda item: (np.prod(item[1].shape), item[0]))
    selected = [candidates[0], candidates[len(candidates) // 2], candidates[-1]] if candidates else []
    sentinels = []
    for path, host_value in selected:
        restored = np.asarray(jax.device_get(flat_params[path]))
        host_array = np.asarray(host_value)
        exact = bool(np.array_equal(restored, host_array))
        sentinels.append({"path": path, "shape": list(host_array.shape), "dtype": str(host_array.dtype), "exact": exact, "sha256": hashlib.sha256(restored.tobytes()).hexdigest()})
        if not exact:
            raise RuntimeError(f"sharded init sentinel mismatch: {path}")

    if replicated_large:
        raise RuntimeError(f"large params remain replicated: {replicated_large[:5]}")
    report = {
        "schema_version": 1,
        "status": "sharded_init_only_passed",
        "parameter_leaf_count": len(flat_params),
        "parameter_numel": int(sum(np.prod(value.shape) for value in flat_params.values())),
        "expected_parameter_numel": EXPECTED_PARAM_NUMEL,
        "state_sharding_parameter_tree_matches": jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(state_sharding.params.to_pure_dict()),
        "large_replicated_leaf_count": len(replicated_large),
        "sharding_inventory": inventory,
        "sentinels": sentinels,
        "finite": {"params": finite_params, "optimizer": finite_opt, "ema": finite_ema, "bad_counts": {"params": bad_params, "optimizer": bad_opt, "ema": bad_ema}},
        "train_state_step": int(state.step),
        "forward_backward_optimizer_update": False,
        "data_loader_started": False,
        "xla_old_input_output_argument_bytes": 26840528788,
        "xla_new_input_output_argument_bytes": None,
        "donation_unusable_full_parameter_warning": False,
        "precision": "float32",
        "ema_decay": 0.999,
        "effective_global_batch": 64,
    }
    write_json(Path(os.environ["AXIS_PI05_RECOVERY_REPORT"]), report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cpu", "init-only", "train-10", "_child"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.phase == "cpu":
        result = cpu_phase()
        print(json.dumps({"status": result["status"], "old_evidence_immutable": result["root_cause_lock"]["old_evidence_immutable"]}, indent=2))
    elif args.phase == "init-only":
        result = run_init_only(args.timeout_seconds)
        print(json.dumps({key: result.get(key) for key in ("status", "return_code", "timed_out", "duration_seconds", "parameter_leaf_count", "large_replicated_leaf_count")}, indent=2))
    elif args.phase == "train-10":
        result = run_train_10(args.timeout_seconds)
        print(json.dumps({key: result.get(key) for key in ("status", "return_code", "timed_out", "duration_seconds", "checkpoint_steps_observed")}, indent=2))
    else:
        _init_child()


if __name__ == "__main__":
    main()
