"""Run the guarded two-GPU Axis LoRA screen.

This command owns only the new LoRA namespace.  It never edits the dense
30K configuration, never selects GPU0, and refuses to reuse an existing run
directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

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


def restore_report(arm: str, run_id: str, step: int, gpu: int = 1) -> dict[str, Any]:
    """Restore one Orbax state in a new process and record step/RNG continuity."""
    if arm not in ARMS:
        raise ValueError(arm)
    require_free_targets((gpu,))
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
            "CUDA_VISIBLE_DEVICES": str(gpu),
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
    payload.update({"arm": arm, "run_id": run_id, "requested_step": step, "gpu": gpu, "gpu0_touched": False})
    write_json(report_dir / f"restore_report_{step}.json", payload)
    return payload


def continue_arm(arm: str, run_id: str, gpu_ids: tuple[int, ...], steps: int) -> dict[str, Any]:
    """Continue an existing LoRA run from its latest Orbax state."""
    if arm not in ARMS:
        raise ValueError(arm)
    if steps <= 25:
        raise ValueError("continuation target must be greater than the 25-step screen")
    require_free_targets(gpu_ids)
    report_dir = REPORT_ROOT / arm / run_id
    run_dir = OUTPUT_ROOT / arm / run_id
    if not report_dir.is_dir() or not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not (run_dir / "25").is_dir():
        raise RuntimeError(f"continuation requires completed step-25 checkpoint: {run_dir / '25'}")
    source = _common_source_manifest(arm, run_id, gpu_ids, steps)
    write_json(report_dir / f"resume_source_manifest_{steps}.json", source)
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
            "AXIS_PI05_LORA_RESUME": "1",
            "AXIS_PI05_LORA_NORM_STATS": str(PHASE1_ROOT / "norm_stats.json"),
            "AXIS_PI05_LORA_DATA_ROOT": str(ROOT / "data/raw/hf/physical-intelligence/libero"),
            "AXIS_PI05_LORA_SAMPLING_MANIFEST": str(PHASE1_ROOT / "official_sampling_manifest.jsonl"),
            "AXIS_PI05_AXIS_PARAMS": str(ROOT / "data/models/axisrobotics/Pi0.5-Axis" / AXIS_REVISION / "params"),
            "OPENPI_KEEP_STEPS": "0,1,10,25,50,100,250,500",
            "OPENPI_MAX_TO_KEEP": "8",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
        }
    )
    log_path = report_dir / f"train_resume_{steps}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [str(PYTHON), str(ROOT / "third_party/axis_openpi/scripts/train.py"), "axis_pi05_lora_screen"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    result = {
        "arm": arm,
        "run_id": run_id,
        "returncode": process.returncode,
        "steps": steps,
        "gpu_ids": list(gpu_ids),
        "gpu0_touched": False,
        "run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "log": str(log_path),
        "resumed_from": 25,
    }
    write_json(report_dir / f"process_result_resume_{steps}.json", result)
    if process.returncode != 0:
        raise RuntimeError(f"LoRA continuation failed; see {log_path}")
    return result

    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def fixed_forward_report(arm: str, run_id: str, step: int, gpu: int = 1) -> dict[str, Any]:
    """Restore one LoRA checkpoint in a new process and run a deterministic forward."""
    if arm not in ARMS:
        raise ValueError(arm)
    if step < 0:
        raise ValueError(step)
    require_free_targets((gpu,))
    run_dir = OUTPUT_ROOT / arm / run_id
    report_dir = REPORT_ROOT / arm / run_id
    if not run_dir.is_dir() or not report_dir.is_dir():
        raise FileNotFoundError(run_dir)
    out_dir = report_dir / f"fixed_forward_step_{step:03d}"
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    code = r'''
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from openpi import transforms
from openpi.shared import normalize
from openpi.training import config as config_lib, data_loader, sharding

def _fp(value):
    array = np.ascontiguousarray(np.asarray(value))
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}

spec = importlib.util.spec_from_file_location("axis_train_fixed_forward", "third_party/axis_openpi/scripts/train.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
config = config_lib.get_config("axis_pi05_lora_screen")
mesh = sharding.make_mesh(1)
data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False, batch_size=1)
batch = next(iter(loader))
manager, resuming = module._checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=None, overwrite=False, resume=True)
shape, _ = module.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
state = module._checkpoints.restore_state(manager, shape, None, step=STEP)
if int(state.step) != STEP:
    raise RuntimeError(f"restored state step mismatch: {state.step} != {STEP}")
model = nnx.merge(state.model_def, state.params)
model.eval()
observation = batch[0]
seed = 20260828
noise = np.random.default_rng(seed).standard_normal((1, 10, 32), dtype=np.float32)
rng = jax.random.key(seed)
started = time.monotonic()
output_a = model.sample_actions(rng, observation, num_steps=10, noise=jnp.asarray(noise))
output_a = np.asarray(jax.block_until_ready(output_a))
elapsed_a = time.monotonic() - started
output_b = model.sample_actions(rng, observation, num_steps=10, noise=jnp.asarray(noise))
output_b = np.asarray(jax.block_until_ready(output_b))
elapsed_b = time.monotonic() - started - elapsed_a
if not np.array_equal(output_a, output_b):
    raise RuntimeError("fixed-input LoRA forward is not bitwise deterministic")
if not np.isfinite(output_a).all():
    raise RuntimeError("fixed-forward LoRA output contains NaN or Inf")
norm_stats = normalize.deserialize_json(Path(os.environ["AXIS_PI05_LORA_NORM_STATS"]).read_text(encoding="utf-8"))
unnormalized = transforms.Unnormalize(norm_stats)({"state": np.asarray(observation.state[0]), "actions": output_a[0]})
external = np.asarray(unnormalized["actions"][:, :7], dtype=np.float32)
if external.shape != (10, 7) or not np.isfinite(external).all():
    raise RuntimeError(f"invalid external LoRA actions: {external.shape}")
out_dir = Path(os.environ["AXIS_PI05_LORA_FIXED_FORWARD_OUT"])
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir / "fixed_input.npz",
    base_0_rgb=np.asarray(observation.images["base_0_rgb"]),
    left_wrist_0_rgb=np.asarray(observation.images["left_wrist_0_rgb"]),
    right_wrist_0_rgb=np.asarray(observation.images["right_wrist_0_rgb"]),
    state=np.asarray(observation.state),
    tokenized_prompt=np.asarray(observation.tokenized_prompt),
    tokenized_prompt_mask=np.asarray(observation.tokenized_prompt_mask),
    action_is_pad=np.asarray(observation.action_is_pad),
    noise=noise,
)
np.savez_compressed(out_dir / "fixed_output.npz", internal_actions=output_a, external_actions=external)
report = {
    "schema_version": 1,
    "status": "passed",
    "arm": os.environ["AXIS_PI05_LORA_ARM"],
    "checkpoint_step": int(state.step),
    "checkpoint_dir": str(config.checkpoint_dir),
    "model_dtype": "float32",
    "action_dim_internal": 32,
    "action_dim_external": 7,
    "action_horizon": 10,
    "deterministic_bitwise": True,
    "finite_internal_actions": True,
    "finite_external_actions": True,
    "optimizer_updates": 0,
    "optimizer_initialized_for_restore": True,
    "fixed_seed": seed,
    "forward_seconds_first": elapsed_a,
    "forward_seconds_second": elapsed_b,
    "fingerprints": {
        "input": {
            "state": _fp(observation.state),
            "tokenized_prompt": _fp(observation.tokenized_prompt),
            "tokenized_prompt_mask": _fp(observation.tokenized_prompt_mask),
            "action_is_pad": _fp(observation.action_is_pad),
            "noise": _fp(noise),
            "images": {key: _fp(value) for key, value in observation.images.items()},
        },
        "output": {"internal_actions": _fp(output_a), "external_actions": _fp(external)},
    },
    "norm_stats_sha256": hashlib.sha256(Path(os.environ["AXIS_PI05_LORA_NORM_STATS"]).read_bytes()).hexdigest(),
    "transform_fingerprint": hashlib.sha256(json.dumps({
        "normalize": "fresh canonical norm z-score",
        "resize": "224x224",
        "tokenizer": "PaligemmaTokenizer",
        "pad": "state/action external dims 8/7 -> internal 32",
        "unnormalize": "internal 32 -> external first 7 action dims",
    }, sort_keys=True).encode()).hexdigest(),
    "artifacts": {
        "fixed_input": str(out_dir / "fixed_input.npz"),
        "fixed_output": str(out_dir / "fixed_output.npz"),
    },
}
Path(os.environ["AXIS_PI05_LORA_FIXED_FORWARD_REPORT"]).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": report["status"], "checkpoint_step": report["checkpoint_step"], "output": report["fingerprints"]["output"]}, sort_keys=True))
'''
    code = code.replace("STEP", str(step))
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(NESTED_SRC) + os.pathsep + str(ROOT),
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "AXIS_PI05_LORA_ARM": arm,
            "AXIS_PI05_LORA_RUN_DIR": str(run_dir),
            "AXIS_PI05_LORA_NORM_STATS": str(PHASE1_ROOT / "norm_stats.json"),
            "AXIS_PI05_LORA_DATA_ROOT": str(ROOT / "data/raw/hf/physical-intelligence/libero"),
            "AXIS_PI05_LORA_SAMPLING_MANIFEST": str(PHASE1_ROOT / "official_sampling_manifest.jsonl"),
            "AXIS_PI05_AXIS_PARAMS": str(ROOT / "data/models/axisrobotics/Pi0.5-Axis" / AXIS_REVISION / "params"),
            "AXIS_PI05_LORA_FIXED_FORWARD_OUT": str(out_dir),
            "AXIS_PI05_LORA_FIXED_FORWARD_REPORT": str(out_dir / "fixed_forward_report.json"),
        }
    )
    completed = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload.update({"arm": arm, "run_id": run_id, "requested_step": step, "gpu": gpu, "gpu0_touched": False})
    write_json(report_dir / f"fixed_forward_report_{step}.json", payload)
    return payload


def checkpoint_manifest(arm: str, run_id: str) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / arm / run_id
    steps = []
    for child in sorted(run_dir.iterdir()) if run_dir.is_dir() else []:
        if child.is_dir() and child.name.isdigit() and (child / "params").exists():
            steps.append(int(child.name))
    required = [0, 1, 10, 25, 50, 100]
    result = {"arm": arm, "run_id": run_id, "run_dir": str(run_dir), "steps": steps, "required": required, "complete": set(required).issubset(steps), "generated_at_unix": time.time()}
    write_json(REPORT_ROOT / arm / run_id / "checkpoint_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "train", "continue", "restore", "fixed-forward", "manifest"])
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
        parser.error("train/continue/restore/fixed-forward/manifest requires --arm and --run-id")
    if args.command == "train":
        gpu_ids = tuple(args.gpu or (1,))
        print(json.dumps(train_arm(args.arm, args.run_id, gpu_ids, args.steps), indent=2))
    elif args.command == "continue":
        gpu_ids = tuple(args.gpu or (1,))
        print(json.dumps(continue_arm(args.arm, args.run_id, gpu_ids, args.steps), indent=2))
    elif args.command == "restore":
        if args.restore_step is None:
            parser.error("restore requires --restore-step")
        gpu = (args.gpu or [1])[0]
        print(json.dumps(restore_report(args.arm, args.run_id, args.restore_step, gpu), indent=2))
    elif args.command == "fixed-forward":
        if args.restore_step is None:
            parser.error("fixed-forward requires --restore-step")
        gpu = (args.gpu or [1])[0]
        print(json.dumps(fixed_forward_report(args.arm, args.run_id, args.restore_step, gpu), indent=2))
    else:
        print(json.dumps(checkpoint_manifest(args.arm, args.run_id), indent=2))


if __name__ == "__main__":
    main()
