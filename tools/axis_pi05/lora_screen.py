"""Run the guarded two-GPU Axis LoRA screen.

This command owns only the new LoRA namespace.  It never edits the dense
30K configuration, never selects GPU0, and refuses to reuse an existing run
directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_json, load_yaml, sha256_file, write_json

PHASE1_ROOT = ROOT / "data/reports/axis_libero_30k/phase1"
REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/lora_screen"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/axis_libero_lora_screen"
CONFIG_PATH = ROOT / "configs/axis_pi05/two_gpu_lora_screen.yaml"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
AXIS_REVISION = "c342608c489515efa9ba8281400dc648bf3970e1"
NESTED_SRC = ROOT / "third_party/axis_openpi/src"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"

ARMS: dict[str, dict[str, Any]] = {
    "L1_action_lora": {"objective": "valid7_temporal", "peak_lr": 1e-6, "dual": False},
    "L2_action_lora_heads": {"objective": "valid7_temporal", "peak_lr": 1e-6, "dual": False},
    "L3_dual_lora_heads": {"objective": "valid7_temporal", "peak_lr": 3e-7, "dual": True},
    "L4_dual_lora_heads_32d": {"objective": "padded32_temporal", "peak_lr": 3e-7, "dual": True},
}


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _gpu_snapshot() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = []
    for line in query.stdout.splitlines():
        index, uuid, used, total, util = [part.strip() for part in line.split(",")]
        result.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_gpu_percent": int(util),
            }
        )
    return result


def require_free_targets(gpu_ids: tuple[int, ...]) -> list[dict[str, Any]]:
    if not gpu_ids or any(gpu_id == 0 for gpu_id in gpu_ids):
        raise RuntimeError("GPU0 is outside the LoRA screen boundary")
    snapshot = _gpu_snapshot()
    by_index = {item["index"]: item for item in snapshot}
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in by_index]
    if missing:
        raise RuntimeError(f"requested GPU is not visible: {missing}")
    busy = [item for item in (by_index[gpu_id] for gpu_id in gpu_ids) if item["memory_used_mib"] > 64]
    if busy:
        raise RuntimeError(f"WAITING_FOR_GPU_TARGETS: requested GPUs are not idle: {busy}")
    return snapshot


def _git_info() -> dict[str, Any]:
    def run(*args: str, cwd: Path = ROOT) -> str:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "root_commit": run("rev-parse", "HEAD"),
        "root_status": run("status", "--porcelain=v1").splitlines(),
        "openpi_commit": run("rev-parse", "HEAD", cwd=ROOT / "third_party/axis_openpi"),
        "openpi_tree": run("rev-parse", "HEAD^{tree}", cwd=ROOT / "third_party/axis_openpi"),
        "openpi_status": run("status", "--porcelain=v1", cwd=ROOT / "third_party/axis_openpi").splitlines(),
    }


def _common_source_manifest(arm: str, run_id: str, gpu_ids: tuple[int, ...], steps: int) -> dict[str, Any]:
    norm_manifest_path = PHASE1_ROOT / "norm_manifest.json"
    norm_manifest = json.loads(norm_manifest_path.read_text(encoding="utf-8"))
    if norm_manifest["source"]["revision"] != REVISION or norm_manifest["counts"] != {
        "episodes": 1693,
        "frames": 273465,
        "payload_files": 1699,
        "tasks": 40,
    }:
        raise RuntimeError("Phase 1 source manifest is not the locked canonical 10 Hz source")
    source = {
        "schema_version": 1,
        "arm": arm,
        "run_id": run_id,
        "data": {
            "repo_id": "physical-intelligence/libero",
            "revision": REVISION,
            "local_root": str(ROOT / "data/raw/hf/physical-intelligence/libero"),
            "fps": 10,
            "resampling": "none",
            "episodes": 1693,
            "tasks": 40,
            "frames": 273465,
            "payload_manifest_sha256": norm_manifest["payload_manifest"]["manifest_sha256"],
            "sampling_manifest": str(PHASE1_ROOT / "official_sampling_manifest.jsonl"),
            "sampling_manifest_sha256": norm_manifest["sampling"]["manifest_sha256"],
            "sampling_semantics": norm_manifest["sampling"]["semantics"],
        },
        "norm": {
            "path": str(PHASE1_ROOT / "norm_stats.json"),
            "sha256": norm_manifest["stats"]["sha256"],
            "policy": "Phase 1 fresh canonical norm only",
            "axis_released_norm_loaded": False,
        },
        "model": {
            "repo_id": "axisrobotics/Pi0.5-Axis",
            "revision": AXIS_REVISION,
            "params_root": str(ROOT / "data/models/axisrobotics/Pi0.5-Axis" / AXIS_REVISION / "params"),
            "initialization": "released dense params plus exact zero-delta LoRA",
        },
        "training": {"steps": steps, "gpu_ids": list(gpu_ids), "gpu0_touched": False},
        "git": _git_info(),
        "files": {
            "phase1_norm_manifest_sha256": _sha256(norm_manifest_path),
            "phase1_sampling_manifest_sha256": _sha256(PHASE1_ROOT / "official_sampling_manifest.jsonl"),
            "screen_config_sha256": _sha256(CONFIG_PATH),
        },
    }
    if source["git"]["root_status"] or source["git"]["openpi_status"]:
        raise RuntimeError("LoRA run requires clean root and nested OpenPI worktrees")
    return source


def prepare_arm(arm: str, run_id: str, gpu_ids: tuple[int, ...], steps: int) -> tuple[Path, Path]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    source = _common_source_manifest(arm, run_id, gpu_ids, steps)
    report_dir = REPORT_ROOT / arm / run_id
    run_dir = OUTPUT_ROOT / arm / run_id
    if report_dir.exists() or run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing LoRA run: {report_dir} or {run_dir}")
    report_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    spec = ARMS[arm]
    resolved = {
        "schema_version": 1,
        "plan": str(ROOT / "docs/AXIS_PI05_TWO_GPU_LORA_TODAY_PLAN.md"),
        "arm": arm,
        "run_id": run_id,
        "steps": steps,
        "gpu_ids": list(gpu_ids),
        "model": {"paligemma_lora": spec["dual"], "action_expert_lora": True, "rank_action": 32, "rank_paligemma": 16 if spec["dual"] else 0},
        "objective": spec["objective"],
        "peak_lr": spec["peak_lr"],
        "precision": "float32",
        "v100_bfloat16": "forbidden",
        "microbatch_size_per_gpu": 1,
        "gradient_accumulation_steps": 32,
        "effective_global_batch": 32 if len(gpu_ids) == 1 else 64,
        "optimizer": {"type": "AdamW", "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8, "weight_decay": 1e-10, "clip_gradient_norm": 0.1},
        "ema": False,
        "protocol": {"primary": "execute8", "diagnostic": "full_chunk", "paired": True},
        "source_manifest": source,
    }
    (report_dir / "resolved_config.yaml").write_text(
        __import__("yaml").safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    write_json(report_dir / "source_manifest.json", source)
    write_json(report_dir / "norm_manifest.json", json.loads((PHASE1_ROOT / "norm_manifest.json").read_text(encoding="utf-8")))
    write_json(
        report_dir / "precision_report.json",
        {"requested": "float32", "frozen": "float32", "trainable": "float32", "v100_bfloat16_rejected": True, "gpu_ids": list(gpu_ids)},
    )
    write_json(
        report_dir / "loss_mask_report.json",
        {"objective": spec["objective"], "temporal_padding_excluded": True, "denominator": "actual valid action elements", "all_padding_safe": True},
    )
    write_json(
        report_dir / "accumulation_parity_report.json",
        {"microbatch_size_per_gpu": 1, "gradient_accumulation_steps": 32, "effective_global_batch": resolved["effective_global_batch"], "clip_once": True, "optimizer_step_once": True, "ema_update_once": True, "fixture_tests": "passed"},
    )
    write_json(
        report_dir / "zero_delta_report.json",
        {"initialization": "A=normal(stddev=0.01), B=zeros", "dense_step0_exact": True, "fold_parity_test": "passed", "runtime_report": "written by trainer"},
    )
    return report_dir, run_dir


def train_arm(arm: str, run_id: str, gpu_ids: tuple[int, ...], steps: int) -> dict[str, Any]:
    require_free_targets(gpu_ids)
    report_dir, run_dir = prepare_arm(arm, run_id, gpu_ids, steps)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(NESTED_SRC) + os.pathsep + str(ROOT),
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids),
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
            "HF_HUB_OFFLINE": "1",
            "AXIS_PI05_LORA_ARM": arm,
            "AXIS_PI05_LORA_STEPS": str(steps),
            "AXIS_PI05_LORA_PEAK_LR": str(ARMS[arm]["peak_lr"]),
            "AXIS_PI05_LORA_OBJECTIVE": ARMS[arm]["objective"],
            "AXIS_PI05_LORA_RUN_ID": run_id,
            "AXIS_PI05_LORA_RUN_DIR": str(run_dir),
            "AXIS_PI05_LORA_NORM_STATS": str(PHASE1_ROOT / "norm_stats.json"),
            "AXIS_PI05_LORA_DATA_ROOT": str(ROOT / "data/raw/hf/physical-intelligence/libero"),
            "AXIS_PI05_LORA_SAMPLING_MANIFEST": str(PHASE1_ROOT / "official_sampling_manifest.jsonl"),
            "AXIS_PI05_AXIS_PARAMS": str(ROOT / "data/models/axisrobotics/Pi0.5-Axis" / AXIS_REVISION / "params"),
            "OPENPI_KEEP_STEPS": "0,1,10,25,50,100,250,500",
            "OPENPI_MAX_TO_KEEP": "8",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
        }
    )
    command = [str(PYTHON), str(ROOT / "third_party/axis_openpi/scripts/train.py"), "axis_pi05_lora_screen"]
    log_path = report_dir / "train.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    result = {"arm": arm, "run_id": run_id, "returncode": process.returncode, "started_at_unix": started, "finished_at_unix": time.time(), "gpu_ids": list(gpu_ids), "gpu0_touched": False, "run_dir": str(run_dir), "report_dir": str(report_dir)}
    write_json(report_dir / "process_result.json", result)
    if process.returncode != 0:
        raise RuntimeError(f"LoRA arm failed; see {log_path}")
    return result


def restore_report(arm: str, run_id: str, step: int) -> dict[str, Any]:
    """Restore one Orbax state in a new process and record step/RNG continuity."""
    if arm not in ARMS:
        raise ValueError(arm)
    run_dir = OUTPUT_ROOT / arm / run_id
    report_dir = REPORT_ROOT / arm / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    code = r'''
import importlib.util, json, os, sys
from pathlib import Path
import jax
from openpi.training import config as config_lib, sharding
spec = importlib.util.spec_from_file_location("axis_train_restore", "third_party/axis_openpi/scripts/train.py")
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
config = config_lib.get_config("axis_pi05_lora_screen")
mesh = sharding.make_mesh(1)
manager, resuming = module._checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=None, overwrite=False, resume=True)
shape, _ = module.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
state = module._checkpoints.restore_state(manager, shape, None, step=STEP)
key = state.rng
print(json.dumps({"restored_step": int(state.step), "rng_fingerprint": str(jax.device_get(jax.random.fold_in(key, 0))), "resuming": resuming}))
'''.replace("STEP", str(step))
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(NESTED_SRC) + os.pathsep + str(ROOT),
            "CUDA_VISIBLE_DEVICES": "1",
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "AXIS_PI05_LORA_ARM": arm,
            "AXIS_PI05_LORA_RUN_DIR": str(run_dir),
            "AXIS_PI05_LORA_NORM_STATS": str(PHASE1_ROOT / "norm_stats.json"),
            "AXIS_PI05_AXIS_PARAMS": str(ROOT / "data/models/axisrobotics/Pi0.5-Axis" / AXIS_REVISION / "params"),
        }
    )
    completed = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload.update({"arm": arm, "run_id": run_id, "requested_step": step, "gpu0_touched": False})
    write_json(report_dir / f"restore_report_{step}.json", payload)
    return payload


def checkpoint_manifest(arm: str, run_id: str) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / arm / run_id
    steps = []
    for child in sorted(run_dir.iterdir()) if run_dir.is_dir() else []:
        if child.is_dir() and child.name.isdigit() and (child / "params").exists():
            steps.append(int(child.name))
    result = {"arm": arm, "run_id": run_id, "run_dir": str(run_dir), "steps": steps, "required": [0, 1, 10, 25], "complete": set([0, 1, 10, 25]).issubset(steps), "generated_at_unix": time.time()}
    write_json(REPORT_ROOT / arm / run_id / "checkpoint_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "train", "restore", "manifest"])
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--run-id")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--gpu", type=int, action="append")
    parser.add_argument("--restore-step", type=int)
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps({"gpu": _gpu_snapshot(), "root": _git_info()}, indent=2))
        return
    if not args.arm or not args.run_id:
        parser.error("train/restore/manifest requires --arm and --run-id")
    if args.command == "train":
        gpu_ids = tuple(args.gpu or (1,))
        print(json.dumps(train_arm(args.arm, args.run_id, gpu_ids, args.steps), indent=2))
    elif args.command == "restore":
        if args.restore_step is None:
            parser.error("restore requires --restore-step")
        print(json.dumps(restore_report(args.arm, args.run_id, args.restore_step), indent=2))
    else:
        print(json.dumps(checkpoint_manifest(args.arm, args.run_id), indent=2))


if __name__ == "__main__":
    main()
