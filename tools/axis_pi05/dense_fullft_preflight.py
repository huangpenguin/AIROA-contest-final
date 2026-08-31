"""CPU locks and the single authorized V100 dense full-FT canary.

This module deliberately keeps the preflight separate from the historical
LoRA runner.  The canary invokes the locked OpenPI trainer with an explicit
``axis_pi05_libero_fullft`` config and never changes the dense recipe to make
an allocation fit.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any

from .common import ROOT, canonical_json, git_info, load_yaml, sha256_file, sha256_tree, write_json


REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_preflight"
STANDARD_CONFIG = ROOT / "configs/axis_pi05/standard_libero_30k.yaml"
DENSE_CONFIG = ROOT / "configs/axis_pi05/dense_fullft_preflight.yaml"
PARAM_METADATA = ROOT / "data/reports/axis_parameter_metadata.json"
FIXED_FORWARD = ROOT / "data/reports/axis_libero_30k/phase1/step0/fixed_forward_report.json"
PAYLOAD_MANIFEST = ROOT / "data/reports/axis_libero_30k/canonical_payload_manifest.json"
NORM_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/norm_manifest.json"
SAMPLING_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/official_sampling_manifest.jsonl"
STEP0_LOCK = ROOT / "data/reports/axis_libero_30k/phase1/rollout/manifest.lock.json"
AXIS_ROOT = ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1"
AXIS_PARAMS = AXIS_ROOT / "params"
FRESH_NORM = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
DATA_ROOT = ROOT / "data/raw/hf/physical-intelligence/libero"
ZIP_PATH = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
NESTED_ROOT = ROOT / "third_party/axis_openpi"
EXPECTED_ROOT_COMMIT = "97f3a35"
EXPECTED_AXIS_REVISION = "c342608c489515efa9ba8281400dc648bf3970e1"
EXPECTED_LIBERO_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
EXPECTED_OPENPI_BASE = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
EXPECTED_PARAM_NUMEL = 3_353_433_872


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd or ROOT, text=True).strip()


def _git_state(path: Path) -> dict[str, Any]:
    status = _run(["git", "status", "--porcelain"], cwd=path)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=path),
        "branch": _run(["git", "branch", "--show-current"], cwd=path),
        "dirty": bool(status),
        "status": status.splitlines(),
        "tree": _run(["git", "write-tree"], cwd=path),
    }


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256_file(path)}


def _parameter_numel() -> tuple[int, int]:
    metadata = json.loads(PARAM_METADATA.read_text(encoding="utf-8"))
    entries = metadata["parameters"]
    total = 0
    for entry in entries:
        numel = 1
        for dim in entry["shape"]:
            numel *= int(dim)
        total += numel
    return len(entries), total


def _disk_budget(param_bytes: int, *, available_bytes: int | None = None) -> dict[str, Any]:
    stat = os.statvfs(ROOT)
    available = int(stat.f_bavail * stat.f_frsize) if available_bytes is None else int(available_bytes)
    # A full train-state save contains params, Adam m/v and EMA.  This is the
    # atomic temporary write allowance; it excludes activations and runtime
    # caches and is intentionally conservative for the disk gate.
    full_state_temp = 4 * param_bytes
    safe_margin = 200 * 1024**3
    zip_bytes = ZIP_PATH.stat().st_size if ZIP_PATH.is_file() else 0
    required = full_state_temp + zip_bytes + safe_margin
    return {
        "schema_version": 1,
        "available_bytes": available,
        "available_gib": round(available / 1024**3, 3),
        "full_state_temp_bytes": full_state_temp,
        "full_state_temp_gib": round(full_state_temp / 1024**3, 3),
        "current_zip_bytes": zip_bytes,
        "safe_margin_bytes": safe_margin,
        "required_bytes": required,
        "required_gib": round(required / 1024**3, 3),
        "headroom_bytes": available - required,
        "headroom_gib": round((available - required) / 1024**3, 3),
        "status": "passed" if available >= required else "disk_budget_failed",
        "automatic_cleanup": False,
    }


def _memory_budget(param_numel: int) -> dict[str, Any]:
    fp32_bytes = param_numel * 4
    categories = {
        "params": fp32_bytes,
        "gradients": fp32_bytes,
        "adam_m": fp32_bytes,
        "adam_v": fp32_bytes,
        "ema": fp32_bytes,
    }
    lower_bound_total = sum(categories.values())
    topologies = {}
    for devices in (2, 4):
        per_card = lower_bound_total / devices
        capacity = 32 * 1024**3
        topologies[str(devices)] = {
            "devices": devices,
            "lower_bound_bytes_total": lower_bound_total,
            "lower_bound_gb_total": round(lower_bound_total / 1e9, 3),
            "lower_bound_bytes_per_card": int(per_card),
            "lower_bound_gb_per_card": round(per_card / 1e9, 3),
            "lower_bound_gib_per_card": round(per_card / 1024**3, 3),
            "vram_capacity_bytes_per_card": capacity,
            "theoretical_headroom_gib_per_card": round((capacity - per_card) / 1024**3, 3),
            "minimum_required_headroom_gib": 2.0,
            "activation_and_replication_included": False,
            "status": "theoretical_headroom_below_2GiB"
            if capacity - per_card < 2 * 1024**3
            else "theoretical_pass_pending_canary",
        }
    return {
        "schema_version": 1,
        "parameter_numel": param_numel,
        "dtype": "float32",
        "bytes_per_element": 4,
        "state_categories": categories,
        "lower_bound_total_bytes": lower_bound_total,
        "lower_bound_total_gb": round(lower_bound_total / 1e9, 3),
        "lower_bound_excludes": ["activations", "temporary_update_buffers", "replicated_small_leaves", "allocator_fragmentation"],
        "topologies": topologies,
        "recipe_preserved": {
            "ema_decay": 0.999,
            "effective_global_batch": 64,
            "precision": "float32",
            "freeze_filter": "nnx.Nothing",
        },
    }


def validate_config_contract() -> dict[str, Any]:
    standard = load_yaml(STANDARD_CONFIG)
    dense = load_yaml(DENSE_CONFIG)
    data = dense["data"]
    model = dense["model"]
    training = dense["training"]
    expected = {
        "data.repo_id": (data["repo_id"], "physical-intelligence/libero"),
        "data.revision": (data["revision"], EXPECTED_LIBERO_REVISION),
        "data.fps": (data["fps"], 10),
        "data.resampling": (data["resampling"], "none"),
        "data.episodes": (data["episodes"], 1693),
        "data.tasks": (data["tasks"], 40),
        "data.frames": (data["frames"], 273465),
        "model.revision": (model["revision"], EXPECTED_AXIS_REVISION),
        "model.dtype": (model["dtype"], "float32"),
        "training.freeze_filter": (training["freeze_filter"], "nnx.Nothing"),
        "training.global_batch": (training["global_batch"], 64),
        "training.ema_decay": (training["ema_decay"], 0.999),
        "training.warmup_steps": (training["warmup_steps"], 10000),
        "training.post_warmup_lr": (training["post_warmup_lr"], 5.0e-5),
    }
    mismatches = {key: {"actual": actual, "expected": wanted} for key, (actual, wanted) in expected.items() if actual != wanted}
    if standard["canonical_dataset"]["repo_id"] != data["repo_id"] or standard["canonical_dataset"]["revision"] != data["revision"]:
        mismatches["standard_config.canonical_dataset"] = "dense config does not match standard config"
    if standard["recipe"]["global_batch"] != training["global_batch"] or standard["recipe"]["ema_decay"] != training["ema_decay"]:
        mismatches["standard_config.recipe"] = "dense training recipe diverges from standard config"
    source_text = (NESTED_ROOT / "src/openpi/training/config.py").read_text(encoding="utf-8")
    config_name_count = source_text.count('name="axis_pi05_libero_fullft"')
    if config_name_count != 1:
        mismatches["openpi.config_name"] = {"occurrences": config_name_count, "expected": 1}
    if 'paligemma_variant="gemma_2b"' not in source_text or 'action_expert_variant="gemma_300m"' not in source_text:
        mismatches["openpi.dense_variants"] = "dense variants not present"
    if 'freeze_filter=nnx.Nothing' not in source_text:
        mismatches["openpi.freeze_filter"] = "dense config does not use nnx.Nothing"
    return {
        "schema_version": 1,
        "config_name": "axis_pi05_libero_fullft",
        "standard_config": str(STANDARD_CONFIG),
        "dense_config": str(DENSE_CONFIG),
        "checks": expected,
        "openpi_config_name_occurrences": config_name_count,
        "mismatches": mismatches,
        "status": "passed" if not mismatches else "config_contract_failed",
    }


def build_input_lock() -> dict[str, Any]:
    payload = json.loads(PAYLOAD_MANIFEST.read_text(encoding="utf-8"))
    norm_manifest = json.loads(NORM_MANIFEST.read_text(encoding="utf-8"))
    fixed = json.loads(FIXED_FORWARD.read_text(encoding="utf-8"))
    step0 = json.loads(STEP0_LOCK.read_text(encoding="utf-8"))
    leaf_count, param_numel = _parameter_numel()
    if param_numel != EXPECTED_PARAM_NUMEL:
        raise ValueError(f"parameter numel drift: {param_numel} != {EXPECTED_PARAM_NUMEL}")
    params_tree = sha256_tree(AXIS_PARAMS)
    expected_tree = fixed["axis_params"]["tree_sha256_after"]
    if params_tree != expected_tree:
        raise ValueError(f"Axis params tree drift: {params_tree} != {expected_tree}")
    root_state = _git_state(ROOT)
    nested_state = _git_state(NESTED_ROOT)
    lock = {
        "schema_version": 1,
        "status": "dense_fullft_inputs_locked",
        "baseline": {
            "track": 1,
            "official_score": 0.47,
            "zip": _file_record(ZIP_PATH),
        },
        "axis": {
            "repo_id": "axisrobotics/Pi0.5-Axis",
            "revision": EXPECTED_AXIS_REVISION,
            "params_root": str(AXIS_PARAMS),
            "revision_file": _file_record(AXIS_ROOT / ".axis_pi05_revision"),
            "params_tree_sha256": params_tree,
            "parameter_leaf_count": leaf_count,
            "parameter_numel": param_numel,
            "released_norm_sha256": sha256_file(AXIS_ROOT / "norm_stats.json"),
            "released_norm_policy": "record_hash_only_never_load",
        },
        "canonical_dataset": {
            "repo_id": "physical-intelligence/libero",
            "revision": EXPECTED_LIBERO_REVISION,
            "local_root": str(DATA_ROOT),
            "payload_manifest_sha256": payload["manifest_sha256"],
            "payload_manifest_file_sha256": sha256_file(PAYLOAD_MANIFEST),
            "payload_files": payload["file_count"],
            "episodes": payload["expected_episodes"],
            "tasks": 40,
            "frames": 273465,
            "fps": 10,
            "resampling": "none",
            "sampling_manifest_sha256": sha256_file(SAMPLING_MANIFEST),
        },
        "fresh_norm": {
            "path": str(FRESH_NORM),
            "sha256": sha256_file(FRESH_NORM),
            "kind": "QUANTILES",
            "policy": "fresh_canonical_only",
            "axis_released_norm_loaded": False,
        },
        "step0_protocol": {
            "manifest_path": str(STEP0_LOCK),
            "manifest_sha256": sha256_file(STEP0_LOCK),
            "panel_sha256": step0["panel_sha256"],
            "protocols": step0["protocols"],
            "fixed_forward_seed": fixed["fixed_seed"],
            "fixed_forward_transform_fingerprint": fixed["transform_fingerprint"],
        },
        "openpi": {
            "base_revision": EXPECTED_OPENPI_BASE,
            "implementation": nested_state,
            "patch_dir": str(ROOT / "patches/axis_openpi"),
        },
        "git": {"root": root_state, "nested_openpi": nested_state},
        "status_gate": {
            "root_clean": not root_state["dirty"],
            "nested_clean": not nested_state["dirty"],
            "params_tree_exact": params_tree == expected_tree,
            "canonical_payload_complete": bool(payload["complete"]),
            "fresh_norm_exact": sha256_file(FRESH_NORM) == norm_manifest["stats"]["sha256"],
        },
    }
    if not all(lock["status_gate"].values()):
        raise ValueError(f"input lock gate failed: {lock['status_gate']}")
    return lock


def write_cpu_locks() -> dict[str, Any]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    config_report = validate_config_contract()
    if config_report["status"] != "passed":
        raise ValueError(json.dumps(config_report, ensure_ascii=False))
    input_lock = build_input_lock()
    _, param_numel = _parameter_numel()
    memory = _memory_budget(param_numel)
    disk = _disk_budget(param_numel * 4)
    resolved = {
        "schema_version": 1,
        "config_name": "axis_pi05_libero_fullft",
        "model": {
            "pi05": True,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "dtype": "float32",
            "action_dim_internal": 32,
            "action_dim_external": 7,
            "action_horizon": 10,
        },
        "data": {
            "repo_id": "physical-intelligence/libero",
            "revision": EXPECTED_LIBERO_REVISION,
            "root": str(DATA_ROOT),
            "fps": 10,
            "resampling": "none",
            "norm": str(FRESH_NORM),
            "norm_sha256": sha256_file(FRESH_NORM),
            "use_quantile_norm": True,
        },
        "training": {
            "parameters": "dense_full_fine_tuning",
            "freeze_filter": "nnx.Nothing",
            "global_batch": 64,
            "microbatch_per_device": 1,
            "accumulation": 32,
            "effective_global_batch": 64,
            "optimizer": "AdamW",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "weight_decay": 1e-10,
            "gradient_clip_norm": 1.0,
            "peak_lr": 5e-5,
            "warmup_steps": 10000,
            "post_warmup_lr": 5e-5,
            "ema_decay": 0.999,
            "precision": "float32",
            "optimizer_steps_target": 30000,
            "reset_optimizer_scheduler_step_ema": True,
        },
        "preflight": {
            "two_gpu_visible_devices": [1, 2],
            "two_gpu_fsdp_devices": 2,
            "two_gpu_max_optimizer_steps": 1,
            "four_gpu_visible_devices": [0, 1, 2, 3],
            "four_gpu_fsdp_devices": 4,
            "four_gpu_max_optimizer_steps": 100,
            "official_gpu": False,
            "zip": False,
        },
        "source": {"root_commit_at_lock": input_lock["git"]["root"]["commit"], "nested_commit_at_lock": input_lock["git"]["nested_openpi"]["commit"]},
    }
    write_json(REPORT_ROOT / "official_baseline.json", input_lock["baseline"] | {"submission_id": None, "submission_time": None, "official_log": None})
    write_json(REPORT_ROOT / "input_lock.json", input_lock)
    write_json(REPORT_ROOT / "memory_budget.json", memory)
    write_json(REPORT_ROOT / "disk_budget.json", disk)
    write_json(REPORT_ROOT / "resolved_config.json", resolved)
    write_json(REPORT_ROOT / "config_contract.json", config_report)
    (REPORT_ROOT / "THREAD_REPORT_ZH.md").write_text(
        "\n".join(
            [
                "# Dense full-FT preflight CPU lock",
                "",
                f"- 状态：`{input_lock['status']}`。",
                f"- Axis params：`{input_lock['axis']['revision']}`，tree SHA-256 `{input_lock['axis']['params_tree_sha256']}`。",
                f"- canonical LIBERO：`{input_lock['canonical_dataset']['revision']}`，native 10Hz，{input_lock['canonical_dataset']['episodes']} episodes / {input_lock['canonical_dataset']['frames']} frames。",
                f"- fresh QUANTILES norm：`{input_lock['fresh_norm']['sha256']}`；Axis norm 仅记录 hash，未加载。",
                f"- dense 参数：{input_lock['axis']['parameter_numel']:,} FP32 params，`nnx.Nothing`，EMA `0.999`，global batch `64`。",
                f"- 磁盘门禁：`{disk['status']}`，可用 {disk['available_gib']} GiB，预留需求 {disk['required_gib']} GiB。",
                "- 下一步：只在 GPU1+GPU2 做一次 microbatch=1 / accumulation=32 的 first-update canary。",
                "- GPU0 被外部进程占用时不等待、不杀进程、不走四卡路径。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"config": config_report, "input_lock": input_lock, "memory": memory, "disk": disk}


def _gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = _run(command).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    result = []
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        result.append({"index": int(fields[0]), "name": fields[1], "memory_used_mib": int(fields[2]), "memory_total_mib": int(fields[3]), "utilization_gpu": int(fields[4])})
    return result


def _monitor(pid: int, path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            row = {"timestamp": time.time(), "pid": pid, "gpus": _gpu_snapshot()}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            stop.wait(2.0)


def run_two_gpu_canary(timeout_seconds: int = 3600) -> dict[str, Any]:
    lock_path = REPORT_ROOT / "input_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("run --phase lock before canary")
    canary_dir = ROOT / "outputs/axis_pi05/dense_fullft_preflight/two_gpu_canary"
    if canary_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior canary: {canary_dir}")
    canary_dir.mkdir(parents=True)
    log_path = REPORT_ROOT / "two_gpu_canary.log"
    monitor_path = REPORT_ROOT / "two_gpu_canary_gpu.jsonl"
    python = ROOT / ".venv_axis_pi05/bin/python"
    command = [str(python), "third_party/axis_openpi/scripts/train.py", "axis_pi05_libero_fullft"]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "1,2",
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONPATH": str(ROOT / "third_party/axis_openpi/src") + os.pathsep + env.get("PYTHONPATH", ""),
            "AXIS_PI05_FULLFT_FSDP_DEVICES": "2",
            "AXIS_PI05_FULLFT_MICROBATCH": "1",
            "AXIS_PI05_FULLFT_ACCUMULATION": "32",
            "AXIS_PI05_FULLFT_EFFECTIVE_BATCH": "64",
            "AXIS_PI05_FULLFT_GLOBAL_BATCH": "64",
            "AXIS_PI05_FULLFT_STEPS": "1",
            "AXIS_PI05_FULLFT_LR_DECAY_STEPS": "30000",
            "AXIS_PI05_FULLFT_RUN_ID": "two_gpu_first_update",
            "AXIS_PI05_FULLFT_RUN_DIR": str(canary_dir),
            "AXIS_PI05_FULLFT_OVERWRITE": "1",
            "AXIS_PI05_FULLFT_SAVE_INTERVAL": "1",
            "AXIS_PI05_FULLFT_LOG_INTERVAL": "1",
            "OPENPI_KEEP_STEPS": "0,1",
            "AXIS_PI05_SAVE_STEP0": "1",
            "AXIS_PI05_WAIT_FOR_STEP0_SAVE": "1",
            "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
            "WANDB_MODE": "disabled",
            "JAX_TRACEBACK_FILTERING": "off",
        }
    )
    before = _gpu_snapshot()
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
        monitor_thread.join(timeout=5)
    after = _gpu_snapshot()
    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
    metrics_path = canary_dir / "training_metrics.jsonl"
    metrics = []
    if metrics_path.is_file():
        metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint_steps = []
    if canary_dir.is_dir():
        checkpoint_steps = sorted(int(path.name) for path in canary_dir.iterdir() if path.is_dir() and path.name.isdigit())
    passed = return_code == 0 and not timed_out and any(int(row.get("optimizer_step", -1)) == 1 for row in metrics)
    report = {
        "schema_version": 1,
        "phase": "F2.1_two_gpu_first_update_canary",
        "status": "first_update_passed_pending_restore" if passed else "dense_fullft_two_gpu_canary_failed",
        "command": command,
        "visible_devices": [1, 2],
        "fsdp_devices": 2,
        "microbatch_per_device": 1,
        "accumulation_steps": 32,
        "effective_global_batch": 64,
        "precision": "float32",
        "ema_decay": 0.999,
        "timeout_seconds": timeout_seconds,
        "started_at_unix": started,
        "duration_seconds": time.time() - started,
        "return_code": return_code,
        "timed_out": timed_out,
        "before_gpus": before,
        "after_gpus": after,
        "log_path": str(log_path),
        "gpu_monitor_path": str(monitor_path),
        "checkpoint_dir": str(canary_dir),
        "checkpoint_steps_observed": checkpoint_steps,
        "metrics": metrics,
        "log_tail": log_tail,
        "gpu0_policy": "not_visible_to_canary; external process untouched",
        "recipe_changed_to_fit": False,
    }
    write_json(REPORT_ROOT / "two_gpu_canary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("lock", "canary"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    if args.phase == "lock":
        result = write_cpu_locks()
        print(json.dumps({"status": result["input_lock"]["status"], "memory": result["memory"]["topologies"], "disk": result["disk"]}, indent=2))
    else:
        result = run_two_gpu_canary(args.timeout_seconds)
        print(json.dumps({key: result[key] for key in ("status", "return_code", "timed_out", "duration_seconds", "checkpoint_steps_observed")}, indent=2))


if __name__ == "__main__":
    main()
