"""Run the frozen Fast400 T3 H300 LoRA recipe on four local V100s."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file


RUN_ID = "t3_fast400_r6_h300_4gpu_r1"
RUN_DIR = ROOT / "outputs/axis_pi05/track3_fast400_h300" / RUN_ID
EVIDENCE = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_4gpu_r1_local4"
PLAN = ROOT / "docs/AXIS_PI05_T3_FAST400_LOCAL_4GPU_H300_EXECUTION_PLAN_20260914.md"
OPENPI = ROOT / "third_party/axis_openpi"
TRAIN = OPENPI / "scripts/train.py"
CONFIG = OPENPI / "src/openpi/training/config.py"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"
R6_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
MIX = ROOT / "data/manifests_track3/fast400_gold_80_20_20260913"
MANIFEST = MIX / "manifest.json"
NORM = MIX / "norm_stats.json"
SAMPLING = MIX / "sampling_weights.parquet"
SOURCE_ROOTS = MIX / "source_roots.json"
Q3 = ROOT / "data/reports/axis_libero_30k/track3_fast400_manifest_freeze_20260913/q3_summary.json"
FORWARD = ROOT / "data/raw/hf/lerobot/libero_plus"
REVERSE = ROOT / "data/processed/axis_pi05_track3_reverse_gold_fast400_20260913"

EXPECTED = {
    "manifest": "ba940bbcaff15395803c5a942f004e491bb5a6fce69be8b9b55bb4f759fac97e",
    "norm": "7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333",
    "sampling": "aed3380f646bf4c2635090b2b6e994b60a492ee064cda6067cbdc4071e6c90a1",
    "source_roots": "2082c27207bf3b73a2139803a7923a21e8837b02b97c01961469fc1dc7115bdc",
    "reverse_tree": "966303c7d128ffd169cb7a10f5fbb017bda24a9181f64f05bf1eeb39bfd66e50",
    "r6_params_q0": "d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f",
}


def strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(result, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return result


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def q0_params_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: str(p).encode()):
        relative = item.relative_to(ROOT).as_posix()
        digest.update(f"{sha256_file(item)}  {relative}\n".encode())
    return digest.hexdigest()


def gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        index, name, uuid, total, used, util = [part.strip() for part in line.split(",", 5)]
        rows.append({"index": int(index), "name": name, "uuid": uuid, "total_mib": int(total), "used_mib": int(used), "util_percent": int(util)})
    return rows


def last_step() -> int:
    path = RUN_DIR / "training_metrics.jsonl"
    if not path.is_file():
        return 0
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            return int(json.loads(line)["optimizer_step"])
    return 0


def input_contract() -> dict[str, Any]:
    required = (PLAN, TRAIN, CONFIG, R6_PARAMS, MANIFEST, NORM, SAMPLING, SOURCE_ROOTS, Q3, FORWARD, REVERSE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Fast400 H300 inputs: " + ", ".join(missing))
    actual = {
        "manifest": sha256_file(MANIFEST),
        "norm": sha256_file(NORM),
        "sampling": sha256_file(SAMPLING),
        "source_roots": sha256_file(SOURCE_ROOTS),
    }
    mismatch = {key: {"actual": value, "expected": EXPECTED[key]} for key, value in actual.items() if value != EXPECTED[key]}
    if mismatch:
        raise ValueError(f"frozen input hash mismatch: {mismatch}")
    q3 = strict_json(Q3)
    if q3.get("status") != "fast400_manifest_quantiles_frozen_t3_not_started" or q3.get("q4_started") is not False or q3.get("q5_started") is not False:
        raise ValueError(f"Fast400 Q3 state is not the frozen non-started contract: {q3.get('status')}")
    sources = strict_json(SOURCE_ROOTS)
    reverse_record = sources.get("axis-pi05/track3_reverse_gold_fast400_20260913", {})
    if reverse_record.get("tree_sha256") != EXPECTED["reverse_tree"] or reverse_record.get("episodes") != 8:
        raise ValueError("Fast400 reverse source identity mismatch")
    params_sha = q0_params_sha(R6_PARAMS)
    if params_sha != EXPECTED["r6_params_q0"]:
        raise ValueError(f"R6 params-only hash mismatch: {params_sha}")
    return {
        **actual,
        "r6_params_q0": params_sha,
        "trainer": sha256_file(TRAIN),
        "config": sha256_file(CONFIG),
        "plan": sha256_file(PLAN),
        "forward_root": str(FORWARD.resolve()),
        "reverse_root": str(REVERSE.resolve()),
    }


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join((str(OPENPI / "src"), str(ROOT))),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
        "HF_HUB_OFFLINE": "1",
        "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
        "AXIS_PI05_TRACK3_REPO_ID": "local/track3_fast400_gold_80_20_20260913",
        "AXIS_PI05_TRACK3_LOCAL_REPO_IDS": "lerobot/libero_plus,axis-pi05/track3_reverse_gold_fast400_20260913",
        "AXIS_PI05_TRACK3_LOCAL_REPO_ROOTS": os.pathsep.join((str(FORWARD), str(REVERSE))),
        "AXIS_PI05_TRACK3_SOURCE_WEIGHTS": "0.8,0.2",
        "AXIS_PI05_TRACK3_SAMPLING_MANIFEST": str(SAMPLING),
        "AXIS_PI05_TRACK3_NORM_STATS": str(NORM),
        "AXIS_PI05_TRACK3_PARAMS": str(R6_PARAMS),
        "AXIS_PI05_TRACK3_STEPS": "300",
        "AXIS_PI05_TRACK3_GLOBAL_BATCH": "64",
        "AXIS_PI05_TRACK3_MICROBATCH": "1",
        "AXIS_PI05_TRACK3_ACCUMULATION": "16",
        "AXIS_PI05_TRACK3_EFFECTIVE_BATCH": "64",
        "AXIS_PI05_TRACK3_WARMUP_STEPS": "100",
        "AXIS_PI05_TRACK3_PEAK_LR": "1e-6",
        "AXIS_PI05_TRACK3_DECAY_LR": "1e-7",
        "AXIS_PI05_TRACK3_DATALOADER_WORKERS": "0",
        "AXIS_PI05_TRACK3_LOG_INTERVAL": "1",
        "AXIS_PI05_TRACK3_RUN_ID": RUN_ID,
        "AXIS_PI05_TRACK3_OUTPUT_DIR": str(RUN_DIR.parent),
        "AXIS_PI05_TRACK3_RUN_DIR": str(RUN_DIR),
        "OPENPI_KEEP_STEPS": "1,100,300",
        "OPENPI_MAX_TO_KEEP": "3",
        "AXIS_PI05_SAVE_STEP0": "0",
        "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1",
        "AXIS_PI05_FAIL_ON_NONFINITE": "1",
        "AXIS_PI05_AUDIT_METADATA": "1",
        "AXIS_PI05_AUDIT_REPLAY_DIR": str(EVIDENCE / "audit"),
        "AXIS_PI05_AUDIT_SOURCE_MANIFEST": str(EVIDENCE / "source_manifest.json"),
        "AXIS_PI05_AUDIT_RESOLVED_CONFIG": str(EVIDENCE / "resolved_config.json"),
        "AXIS_PI05_AUDIT_ROOT_COMMIT": git("rev-parse", "HEAD"),
        "AXIS_PI05_AUDIT_OPENPI_COMMIT": git("rev-parse", "HEAD", cwd=OPENPI),
    })
    return env


def preflight(*, require_idle: bool = True) -> dict[str, Any]:
    inputs = input_contract()
    gpus = gpu_snapshot()
    if [row["index"] for row in gpus] != [0, 1, 2, 3] or any("V100" not in row["name"] for row in gpus):
        raise RuntimeError(f"expected exactly local V100 GPUs 0..3: {gpus}")
    if require_idle and any(row["used_mib"] > 64 or row["util_percent"] != 0 for row in gpus):
        raise RuntimeError(f"all four GPUs must be idle before start: {gpus}")
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < 300:
        raise RuntimeError(f"less than 300 GiB free: {free_gib:.2f}")
    return {
        "status": "passed",
        "run_id": RUN_ID,
        "inputs": inputs,
        "gpus": gpus,
        "free_gib": free_gib,
        "contract": {"devices": 4, "mesh": {"batch": 4, "fsdp": 1}, "microbatch_per_device": 1, "accumulation": 16, "effective_global_batch": 64, "steps": 300},
    }


def fresh_restore(env: dict[str, str]) -> dict[str, Any]:
    code = f'''
import importlib.util, json, sys
import jax
from openpi.training import config as config_lib, sharding
spec = importlib.util.spec_from_file_location("axis_t3_h300_restore", {str(TRAIN)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
config = config_lib.get_config("axis_pi05_track3_q5")
mesh = sharding.make_mesh(1)
manager, resuming = module._checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=None, overwrite=False, resume=True)
shape, _ = module.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
state = module._checkpoints.restore_state(manager, shape, None, step=300)
jax.block_until_ready(state)
print(json.dumps({{"restored_step": int(state.step), "device_count": jax.device_count(), "resuming": bool(resuming)}}))
'''
    result = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    (EVIDENCE / "fresh_restore_step300.log").write_text(result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("fresh restore step300 failed")
    payload = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    if payload.get("restored_step") != 300 or payload.get("device_count") != 4:
        raise RuntimeError(f"fresh restore contract mismatch: {payload}")
    return payload


def validate_metrics() -> dict[str, Any]:
    path = RUN_DIR / "training_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [int(row["optimizer_step"]) for row in rows] != list(range(1, 301)):
        raise ValueError("optimizer steps are not contiguous 1..300")
    for row in rows:
        if int(row.get("micro_steps", -1)) != 16 or int(row.get("examples_seen", -1)) != 64:
            raise ValueError(f"four-card batch accounting mismatch at step {row.get('optimizer_step')}")
        if not all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))):
            raise FloatingPointError(f"non-finite metric at step {row.get('optimizer_step')}")
    missing = [step for step in (1, 100, 300) if not (RUN_DIR / str(step) / "params").is_dir() or not (RUN_DIR / str(step) / "train_state").is_dir()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    partials = [str(path) for path in RUN_DIR.rglob("*") if ".partial" in path.name]
    if partials:
        raise ValueError(f"partial checkpoint residue: {partials[:5]}")
    return {"records": len(rows), "last_step": 300, "micro_steps": 16, "examples_seen": 64, "metrics_sha256": sha256_file(path)}


def run() -> dict[str, Any]:
    if RUN_DIR.exists() or EVIDENCE.exists():
        raise FileExistsError(f"isolated H300 namespace already exists: {RUN_DIR} or {EVIDENCE}")
    check = preflight()
    EVIDENCE.mkdir(parents=True)
    RUN_DIR.parent.mkdir(parents=True, exist_ok=True)
    env = environment()
    source = {
        "schema_version": "t3_fast400_local4_source_v1",
        "status": "frozen_before_training",
        "run_id": RUN_ID,
        "root_commit": git("rev-parse", "HEAD"),
        "openpi_commit": git("rev-parse", "HEAD", cwd=OPENPI),
        "root_dirty": git("status", "--porcelain=v1").splitlines(),
        "openpi_dirty": git("status", "--porcelain=v1", cwd=OPENPI).splitlines(),
        "preflight": check,
    }
    resolved = {key: env[key] for key in sorted(env) if key.startswith("AXIS_PI05_") or key in {"CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "HF_HUB_OFFLINE"}}
    atomic_json(EVIDENCE / "source_manifest.json", source)
    atomic_json(EVIDENCE / "resolved_config.json", resolved)
    started = time.time()
    log_handle = (EVIDENCE / "train.log").open("w", encoding="utf-8")
    child = subprocess.Popen([str(PYTHON), str(TRAIN), "axis_pi05_track3_q5"], cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)

    def forward(signum: int, _frame: Any) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        while child.poll() is None:
            record = {"status": "running", "phase": "train_h300", "run_id": RUN_ID, "supervisor_pid": os.getpid(), "child_pid": child.pid, "last_optimizer_step": last_step(), "started_at_unix": started, "updated_at_unix": time.time(), "gpus": gpu_snapshot()}
            atomic_json(EVIDENCE / "cursor.json", record)
            atomic_json(EVIDENCE / "heartbeat.json", record)
            time.sleep(10)
        log_handle.close()
        if child.returncode != 0:
            raise RuntimeError(f"trainer exited {child.returncode}; see {EVIDENCE / 'train.log'}")
        atomic_json(EVIDENCE / "cursor.json", {"status": "running", "phase": "validate_and_restore", "run_id": RUN_ID, "supervisor_pid": os.getpid(), "last_optimizer_step": last_step(), "updated_at_unix": time.time()})
        metrics = validate_metrics()
        restore = fresh_restore(env)
        summary = {"schema_version": "t3_fast400_local4_h300_summary_v1", "status": "training_passed_ready_for_behavior_eval", "official_result": False, "behavior_eval": "not_evaluated", "run_id": RUN_ID, "run_dir": str(RUN_DIR), "evidence_dir": str(EVIDENCE), "started_at_unix": started, "finished_at_unix": time.time(), "elapsed_seconds": time.time() - started, "metrics": metrics, "fresh_restore": restore, "input_hashes": check["inputs"]}
        atomic_json(EVIDENCE / "summary.json", summary)
        atomic_json(EVIDENCE / "cursor.json", {"status": summary["status"], "phase": "await_behavior_eval", "run_id": RUN_ID, "last_optimizer_step": 300, "summary": str(EVIDENCE / "summary.json"), "updated_at_unix": time.time()})
        atomic_json(EVIDENCE / "heartbeat.json", {"status": "stopped", "phase": "training_complete", "run_id": RUN_ID, "updated_at_unix": time.time()})
        return summary
    except BaseException as exc:
        log_handle.close()
        failure = {"status": "failed", "run_id": RUN_ID, "error": f"{type(exc).__name__}: {exc}", "last_optimizer_step": last_step(), "updated_at_unix": time.time()}
        atomic_json(EVIDENCE / "failure.json", failure)
        atomic_json(EVIDENCE / "cursor.json", failure)
        atomic_json(EVIDENCE / "heartbeat.json", {**failure, "status": "stopped"})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    args = parser.parse_args()
    result = preflight() if args.command == "preflight" else run()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
