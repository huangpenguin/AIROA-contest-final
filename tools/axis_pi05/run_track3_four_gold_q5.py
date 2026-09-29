"""Run the guarded local GPU3 Q5 canary as one persistent supervisor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, write_json


EVIDENCE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q5_local_gpu3"
RUN_DIR = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_FOUR_GOLD_80_20_QUANTILES_AND_R6_SHORT_CONTINUATION_PLAN_20260910.md"
Q2_ROOT = ROOT / "data/manifests_track3/four_gold_80_20_v1"
NORM = Q2_ROOT / "norm_stats.json"
SAMPLING = Q2_ROOT / "sampling_weights.parquet"
MANIFEST = Q2_ROOT / "manifest.json"
SOURCE_ROOTS = Q2_ROOT / "source_roots.json"
FREEZE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/evaluation_freeze.json"
R6_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
FORWARD_ROOT = ROOT / "data/raw/hf/lerobot/libero_plus"
REVERSE_ROOT = ROOT / "data/processed/axis_pi05_track3_reverse_gold_v1"
OPENPI_ROOT = ROOT / "third_party/axis_openpi"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"
TRAIN = OPENPI_ROOT / "scripts/train.py"
CONFIG_PATH = OPENPI_ROOT / "src/openpi/training/config.py"

EXPECTED_R6_PARAMS_SHA = "d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f"


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _json_hash(path: Path) -> str:
    return sha256_file(path)


def _q0_params_sha(path: Path) -> str:
    """Reproduce Q0: sha256(sort -z files | xargs -0 sha256sum)."""
    digest = hashlib.sha256()
    for file_path in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: str(item).encode()):
        # Q0's file list was emitted from the repository root, so sha256sum
        # saw paths such as ``outputs/...`` rather than absolute paths.
        relative = file_path.relative_to(ROOT).as_posix()
        digest.update(f"{sha256_file(file_path)}  {relative}\n".encode())
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, uuid, used, total, util = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
                "utilization_gpu_percent": int(util),
            }
        )
    return rows


def _compute_app_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [part.strip() for part in line.split(",", 3)]
        rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(memory)})
    return rows


def _active_blocking_processes() -> list[dict[str, Any]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
    blocking = []
    needles = (
        "third_party/axis_openpi/scripts/train.py",
        "track3_public_teleop",
        "track3_reverse",
        "mujoco",
        "replay",
    )
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, args = int(fields[0]), fields[1]
        if pid == os.getpid():
            continue
        if any(needle in args.lower() for needle in needles):
            blocking.append({"pid": pid, "args": args})
    return blocking


def _free_gib(path: Path) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3


def _require_inputs() -> dict[str, Any]:
    required = (PLAN, NORM, SAMPLING, MANIFEST, SOURCE_ROOTS, FREEZE, R6_PARAMS, FORWARD_ROOT, REVERSE_ROOT, TRAIN, CONFIG_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Q5 input(s): " + ", ".join(missing))
    params_sha = _q0_params_sha(R6_PARAMS)
    if params_sha != EXPECTED_R6_PARAMS_SHA:
        raise ValueError(f"R6 params tree hash mismatch: {params_sha} != {EXPECTED_R6_PARAMS_SHA}")
    return {
        "plan_sha256": sha256_file(PLAN),
        "manifest_sha256": sha256_file(MANIFEST),
        "sampling_manifest_sha256": sha256_file(SAMPLING),
        "source_roots_sha256": sha256_file(SOURCE_ROOTS),
        "norm_stats_sha256": sha256_file(NORM),
        "evaluation_freeze_sha256": sha256_file(FREEZE),
        "r6_params_tree_sha256": params_sha,
        "r6_params_expected_tree_sha256": EXPECTED_R6_PARAMS_SHA,
        "trainer_sha256": sha256_file(TRAIN),
        "q5_config_sha256": sha256_file(CONFIG_PATH),
    }


def _common_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(OPENPI_ROOT / "src") + os.pathsep + str(ROOT),
            "CUDA_VISIBLE_DEVICES": "3",
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
            "HF_HUB_OFFLINE": "1",
            "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
            "AXIS_PI05_TRACK3_REPO_ID": "local/track3_four_gold_80_20",
            "AXIS_PI05_TRACK3_LOCAL_REPO_IDS": "lerobot/libero_plus,axis-pi05/track3_reverse_gold_v1",
            "AXIS_PI05_TRACK3_LOCAL_REPO_ROOTS": str(FORWARD_ROOT) + os.pathsep + str(REVERSE_ROOT),
            "AXIS_PI05_TRACK3_SOURCE_WEIGHTS": "0.8,0.2",
            "AXIS_PI05_TRACK3_SAMPLING_MANIFEST": str(SAMPLING),
            "AXIS_PI05_TRACK3_NORM_STATS": str(NORM),
            "AXIS_PI05_TRACK3_PARAMS": str(R6_PARAMS),
            "AXIS_PI05_TRACK3_STEPS": "10",
            "AXIS_PI05_TRACK3_GLOBAL_BATCH": "64",
            "AXIS_PI05_TRACK3_MICROBATCH": "1",
            "AXIS_PI05_TRACK3_ACCUMULATION": "64",
            "AXIS_PI05_TRACK3_EFFECTIVE_BATCH": "64",
            "AXIS_PI05_TRACK3_WARMUP_STEPS": "100",
            "AXIS_PI05_TRACK3_PEAK_LR": "1e-6",
            "AXIS_PI05_TRACK3_DECAY_LR": "1e-7",
            "AXIS_PI05_TRACK3_DATALOADER_WORKERS": "0",
            "AXIS_PI05_TRACK3_LOG_INTERVAL": "1",
            "AXIS_PI05_TRACK3_RUN_ID": "q5_local_gpu3",
            "AXIS_PI05_TRACK3_OUTPUT_DIR": str(RUN_DIR.parent),
            "AXIS_PI05_TRACK3_RUN_DIR": str(RUN_DIR),
            "OPENPI_KEEP_STEPS": "0,1,10",
            "OPENPI_MAX_TO_KEEP": "3",
            "AXIS_PI05_SAVE_STEP0": "1",
            "AXIS_PI05_WAIT_FOR_STEP0_SAVE": "1",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
            "AXIS_PI05_AUDIT_METADATA": "1",
            "AXIS_PI05_AUDIT_REPLAY_DIR": str(EVIDENCE / "audit"),
            "AXIS_PI05_AUDIT_SOURCE_MANIFEST": str(EVIDENCE / "source_manifest.json"),
            "AXIS_PI05_AUDIT_RESOLVED_CONFIG": str(EVIDENCE / "resolved_config.json"),
            "AXIS_PI05_AUDIT_ROOT_COMMIT": _git("rev-parse", "HEAD"),
            "AXIS_PI05_AUDIT_OPENPI_COMMIT": _git("rev-parse", "HEAD", cwd=OPENPI_ROOT),
            "AXIS_PI05_AUDIT_SOURCE_MANIFEST": str(EVIDENCE / "source_manifest.json"),
        }
    )
    return env


def _restore_one(step: int, env: dict[str, str], output: Path) -> dict[str, Any]:
    code = r'''
import importlib.util, json, sys
import jax
from openpi.training import config as config_lib, sharding
spec = importlib.util.spec_from_file_location("axis_train_restore_q5", "third_party/axis_openpi/scripts/train.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
config = config_lib.get_config("axis_pi05_track3_q5")
mesh = sharding.make_mesh(1)
manager, resuming = module._checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=None, overwrite=False, resume=True)
shape, _ = module.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
state = module._checkpoints.restore_state(manager, shape, None, step=STEP)
jax.block_until_ready(state)
print(json.dumps({"restored_step": int(state.step), "requested_step": STEP, "resuming": bool(resuming), "device_count": jax.device_count(), "devices": [str(device) for device in jax.devices()]}))
'''.replace("STEP", str(step))
    completed = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    output.write_text(completed.stdout + ("\nSTDERR:\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"fresh restore step {step} failed; see {output}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"fresh restore step {step} produced no JSON")
    result = json.loads(lines[-1])
    if result.get("restored_step") != step or result.get("device_count") != 1:
        raise RuntimeError(f"fresh restore step {step} contract mismatch: {result}")
    return result


def _validate_result(input_hashes: dict[str, Any], pre_gpu: list[dict[str, Any]], post_gpu: list[dict[str, Any]], app_rows: list[dict[str, Any]], restore_reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_path = RUN_DIR / "training_metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row.get("optimizer_step") for row in metrics] != list(range(1, 11)):
        raise ValueError(f"optimizer steps are not contiguous 1..10: {[row.get('optimizer_step') for row in metrics]}")
    for row in metrics:
        numeric = [value for value in row.values() if isinstance(value, (int, float))]
        if not all(__import__("math").isfinite(float(value)) for value in numeric):
            raise ValueError(f"non-finite metrics at step {row.get('optimizer_step')}")
        if row.get("micro_steps") != 64 or row.get("examples_seen") != 64:
            raise ValueError(f"global batch accounting mismatch at step {row.get('optimizer_step')}: {row}")
    scope = json.loads((RUN_DIR / "params_scope.json").read_text(encoding="utf-8"))
    if scope.get("arm") != "L2_action_lora_heads" or scope.get("trainable_leaf_count", 0) <= 0:
        raise ValueError(f"trainable scope mismatch: {scope.get('arm')}")
    required_steps = [0, 1, 10]
    missing = [str(step) for step in required_steps if not (RUN_DIR / str(step) / "params").is_dir() or not (RUN_DIR / str(step) / "train_state").is_dir()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing))
    partials = [str(path.relative_to(RUN_DIR)) for path in RUN_DIR.rglob("*") if ".partial" in path.name]
    if partials:
        raise ValueError(f"partial checkpoint residue: {partials}")
    gpu3_uuid = next(row["uuid"] for row in post_gpu if row["index"] == 3)
    gpu3_pids = [row["pid"] for row in app_rows if row["gpu_uuid"] == gpu3_uuid]
    return {
        "schema_version": "track3_four_gold_q5_summary_v1",
        "status": "track3_four_gold_local_single_gpu_canary_passed_ready_for_official_bundle",
        "contract": {
            "physical_gpu": 3,
            "cuda_visible_devices": "3",
            "fsdp_devices": 1,
            "topology": "L2_action_lora_heads",
            "initialization": "r6_dual_l2_fresh500_r1_step500_params_only",
            "precision": "float32",
            "bf16_forbidden": True,
            "peak_lr": 1e-6,
            "warmup_steps": 100,
            "optimizer": {"type": "AdamW", "b1": 0.9, "b2": 0.95, "eps": 1e-8, "weight_decay": 1e-10, "clip_gradient_norm": 0.1},
            "ema": False,
            "microbatch": 1,
            "accumulation": 64,
            "effective_global_batch": 64,
            "steps": 10,
            "checkpoint_steps": required_steps,
        },
        "input_hashes": input_hashes,
        "train_metrics": {
            "records": len(metrics),
            "optimizer_steps": [row["optimizer_step"] for row in metrics],
            "micro_steps_all_64": True,
            "examples_seen_all_64": True,
            "finite_all_numeric": True,
            "trainable_scope": {"arm": scope["arm"], "trainable_leaf_count": scope["trainable_leaf_count"], "trainable_numel": scope.get("trainable_numel")},
        },
        "fresh_restore": restore_reports,
        "gpu_boundary": {
            "gpu0_1_2_touched": False,
            "pre_snapshot": pre_gpu,
            "post_snapshot": post_gpu,
            "post_compute_app_rows": app_rows,
            "gpu3_compute_pids_after_completion": gpu3_pids,
        },
        "process_boundary": {"supervisor_pid": os.getpid(), "orphan_processes_after_completion": _active_blocking_processes()},
        "disk": {"free_gib_after": _free_gib(ROOT), "threshold_gib": 500.0},
        "artifacts": {"run_dir": str(RUN_DIR), "evidence_dir": str(EVIDENCE), "metrics": str(metrics_path), "params_scope": str(RUN_DIR / "params_scope.json")},
    }


def run() -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if (EVIDENCE / "q5_attempt.lock.json").exists() or RUN_DIR.exists():
        raise FileExistsError(f"Q5 namespace already exists: {EVIDENCE} or {RUN_DIR}")
    inputs = _require_inputs()
    pre_gpu = _gpu_snapshot()
    gpu3 = next(row for row in pre_gpu if row["index"] == 3)
    if gpu3["memory_used_mib"] > 64 or gpu3["utilization_gpu_percent"] != 0:
        raise RuntimeError(f"GPU3 is not idle: {gpu3}")
    blockers = _active_blocking_processes()
    if blockers:
        raise RuntimeError(f"blocking process exists: {blockers}")
    if _free_gib(ROOT) < 500.0:
        raise RuntimeError(f"disk free below 500 GiB: {_free_gib(ROOT):.2f}")

    env = _common_env()
    source_manifest = {
        "schema_version": "track3_four_gold_q5_source_manifest_v1",
        "status": "frozen_before_training",
        "root_commit": _git("rev-parse", "HEAD"),
        "openpi_commit": _git("rev-parse", "HEAD", cwd=OPENPI_ROOT),
        "root_dirty": _git("status", "--porcelain=v1").splitlines(),
        "openpi_dirty": _git("status", "--porcelain=v1", cwd=OPENPI_ROOT).splitlines(),
        "inputs": inputs,
        "source_roots": {"forward": str(FORWARD_ROOT), "reverse": str(REVERSE_ROOT)},
        "contract": {"gpu": 3, "precision": "float32", "microbatch": 1, "accumulation": 64, "effective_global_batch": 64, "steps": 10},
    }
    write_json(EVIDENCE / "source_manifest.json", source_manifest)
    write_json(
        EVIDENCE / "resolved_config.json",
        {
            "config": "axis_pi05_track3_q5",
            "local_repo_ids": env["AXIS_PI05_TRACK3_LOCAL_REPO_IDS"].split(","),
            "local_repo_roots": env["AXIS_PI05_TRACK3_LOCAL_REPO_ROOTS"].split(os.pathsep),
            "sampling_manifest": env["AXIS_PI05_TRACK3_SAMPLING_MANIFEST"],
            "norm_stats": env["AXIS_PI05_TRACK3_NORM_STATS"],
            "params": env["AXIS_PI05_TRACK3_PARAMS"],
            "arm": env["AXIS_PI05_LORA_ARM"],
            "precision": "float32",
            "global_batch": 64,
            "microbatch": 1,
            "accumulation": 64,
            "steps": 10,
        },
    )
    lock_path = EVIDENCE / "q5_attempt.lock.json"
    lock = {
        "schema_version": "track3_four_gold_q5_attempt_lock_v1",
        "status": "running",
        "supervisor_pid": os.getpid(),
        "started_at_unix": time.time(),
        "command": [str(PYTHON), str(TRAIN), "axis_pi05_track3_q5"],
        "root_commit": source_manifest["root_commit"],
        "openpi_commit": source_manifest["openpi_commit"],
        "input_hashes": inputs,
        "gpu3_uuid": gpu3["uuid"],
        "pre_gpu_snapshot": pre_gpu,
    }
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    cursor_path = EVIDENCE / "cursor.json"
    heartbeat_path = EVIDENCE / "heartbeat.json"
    log_path = EVIDENCE / "train.log"
    child: subprocess.Popen[str] | None = None
    try:
        RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
        child = subprocess.Popen(
            [str(PYTHON), str(TRAIN), "axis_pi05_track3_q5"],
            cwd=ROOT,
            env=env,
            stdout=log_path.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        while child.poll() is None:
            checkpoints = sorted(int(path.name) for path in RUN_DIR.iterdir() if path.is_dir() and path.name.isdigit()) if RUN_DIR.is_dir() else []
            heartbeat = {"schema_version": 1, "status": "running", "supervisor_pid": os.getpid(), "child_pid": child.pid, "timestamp": time.time(), "checkpoints_seen": checkpoints}
            _atomic_write_json(cursor_path, heartbeat)
            _atomic_write_json(heartbeat_path, heartbeat)
            time.sleep(5)
        returncode = child.returncode
        lock.update({"status": "trainer_exited", "child_pid": child.pid, "trainer_returncode": returncode, "trainer_finished_at_unix": time.time()})
        _atomic_write_json(lock_path, lock)
        if returncode != 0:
            raise RuntimeError(f"Q5 trainer failed with return code {returncode}; see {log_path}")

        post_gpu = _gpu_snapshot()
        app_rows = _compute_app_snapshot()
        restore_reports = []
        for step in (1, 10):
            _atomic_write_json(cursor_path, {"schema_version": 1, "status": f"restoring_step_{step}", "supervisor_pid": os.getpid(), "timestamp": time.time()})
            restore_env = dict(env)
            restore_env["AXIS_PI05_TRACK3_STEPS"] = "10"
            restore_reports.append(_restore_one(step, restore_env, EVIDENCE / f"fresh_restore_step_{step}.log"))
        summary = _validate_result(inputs, pre_gpu, post_gpu, app_rows, restore_reports)
        write_json(EVIDENCE / "q5_summary.json", summary)
        lock.update({"status": "passed", "finished_at_unix": time.time(), "summary_sha256": sha256_file(EVIDENCE / "q5_summary.json")})
        _atomic_write_json(lock_path, lock)
        _atomic_write_json(cursor_path, {"schema_version": 1, "status": "passed", "supervisor_pid": os.getpid(), "timestamp": time.time(), "summary": str(EVIDENCE / "q5_summary.json")})
        _atomic_write_json(heartbeat_path, {"schema_version": 1, "status": "stopped", "supervisor_pid": os.getpid(), "timestamp": time.time()})
        return summary
    except BaseException as exc:
        failure = {"schema_version": "track3_four_gold_q5_failure_v1", "status": "failed", "error": f"{type(exc).__name__}: {exc}", "supervisor_pid": os.getpid(), "child_pid": child.pid if child else None, "timestamp": time.time()}
        write_json(EVIDENCE / "q5_failure.json", failure)
        lock.update({"status": "failed", "finished_at_unix": time.time(), "failure": failure})
        _atomic_write_json(lock_path, lock)
        _atomic_write_json(cursor_path, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="run Track3 Q5 local GPU3 single-card canary")
    parser.parse_args()
    result = run()
    print(json.dumps({"status": result["status"], "summary": str(EVIDENCE / "q5_summary.json")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
