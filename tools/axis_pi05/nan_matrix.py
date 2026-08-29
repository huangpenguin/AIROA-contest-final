"""Run the guarded fixed-replay GPU matrix from the NaN recovery plan."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, write_json

PYTHON = ROOT / ".venv_axis_pi05/bin/python"
NESTED_SRC = ROOT / "third_party/axis_openpi/src"
FIXTURE = ROOT / "data/reports/axis_libero_30k/nan_localization/fixed_replay/fixed_replay.npz"
FIXTURE_MANIFEST = FIXTURE.with_name("fixed_replay_manifest.json")
REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/lora_nan_replay"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/axis_libero_lora_nan_replay"
NORM = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
DATA_ROOT = ROOT / "data/raw/hf/physical-intelligence/libero"
SAMPLING = ROOT / "data/reports/axis_libero_30k/phase1/official_sampling_manifest.jsonl"
PARAMS = ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1/params"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
AXIS_REVISION = "c342608c489515efa9ba8281400dc648bf3970e1"

UNITS: dict[str, dict[str, Any]] = {
    "S1_FB": {"gpus": (1,), "accum": 64, "skip_update": True, "save_step0": False, "async": False, "wait": True},
    "S2_FB": {"gpus": (2,), "accum": 64, "skip_update": True, "save_step0": False, "async": False, "wait": True},
    "D12_FB": {"gpus": (1, 2), "accum": 32, "skip_update": True, "save_step0": False, "async": False, "wait": True},
    "S1_U1": {"gpus": (1,), "accum": 64, "skip_update": False, "save_step0": False, "async": False, "wait": True},
    "S2_U1": {"gpus": (2,), "accum": 64, "skip_update": False, "save_step0": False, "async": False, "wait": True},
    "D12_U1": {"gpus": (1, 2), "accum": 32, "skip_update": False, "save_step0": False, "async": False, "wait": True},
    "D12_SYNC_U1": {"gpus": (1, 2), "accum": 32, "skip_update": False, "save_step0": True, "async": False, "wait": True},
    "D12_ASYNC_WAIT_U1": {"gpus": (1, 2), "accum": 32, "skip_update": False, "save_step0": True, "async": True, "wait": True},
    "D12_ASYNC_OVERLAP_U1": {"gpus": (1, 2), "accum": 32, "skip_update": False, "save_step0": True, "async": True, "wait": False},
}


def _snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
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


def _require_free(gpus: tuple[int, ...]) -> list[dict[str, Any]]:
    if not gpus or 0 in gpus:
        raise RuntimeError("GPU0 is outside the NaN matrix boundary")
    snapshot = _snapshot()
    by_index = {row["index"]: row for row in snapshot}
    if any(gpu not in by_index for gpu in gpus):
        raise RuntimeError(f"requested GPU is not visible: {gpus}")
    busy = [by_index[gpu] for gpu in gpus if by_index[gpu]["memory_used_mib"] > 64]
    if busy:
        raise RuntimeError(f"WAITING_FOR_GPU_TARGETS: requested GPUs are not idle: {busy}")
    return snapshot


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _require_clean() -> tuple[str, str]:
    root_status = _git(ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    nested = ROOT / "third_party/axis_openpi"
    nested_status = _git(nested, "status", "--porcelain=v1", "--untracked-files=all")
    if root_status or nested_status:
        raise RuntimeError("R3 requires clean root and nested OpenPI worktrees")
    return _git(ROOT, "rev-parse", "HEAD"), _git(nested, "rev-parse", "HEAD")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(unit: str, repeat: int, root_commit: str, openpi_commit: str) -> dict[str, Any]:
    spec = UNITS[unit]
    gpus = tuple(spec["gpus"])
    before = _require_free(gpus)
    run_id = f"{unit.lower()}_r{repeat}_{root_commit[:8]}_{openpi_commit[:8]}"
    report_dir = REPORT_ROOT / run_id
    output_dir = OUTPUT_ROOT / run_id
    if report_dir.exists() or output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing R3 run: {report_dir} or {output_dir}")
    report_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    tensor_dir = output_dir / "replay_tensors"
    resolved = {
        "schema_version": 1,
        "unit": unit,
        "repeat": repeat,
        "run_id": run_id,
        "gpus": list(gpus),
        "microbatch_size_per_gpu": 1,
        "accumulation_steps": spec["accum"],
        "effective_global_batch": 64,
        "fixed_replay": str(FIXTURE),
        "fixed_replay_sha256": sha256_file(FIXTURE),
        "optimizer_update": not spec["skip_update"],
        "checkpoint": {
            "save_step0": spec["save_step0"],
            "async": spec["async"],
            "wait_for_step0_save": spec["wait"],
        },
        "root_commit": root_commit,
        "openpi_commit": openpi_commit,
        "norm_sha256": sha256_file(NORM),
        "sampling_sha256": sha256_file(SAMPLING),
        "axis_revision": AXIS_REVISION,
        "source_revision": REVISION,
    }
    (report_dir / "resolved_config.yaml").write_text(
        __import__("yaml").safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    source_manifest = {
        "schema_version": 1,
        "source_repo": "physical-intelligence/libero",
        "source_revision": REVISION,
        "local_root": str(DATA_ROOT),
        "sampling_manifest": str(SAMPLING),
        "sampling_manifest_sha256": sha256_file(SAMPLING),
        "norm_path": str(NORM),
        "norm_sha256": sha256_file(NORM),
        "axis_params": str(PARAMS),
        "axis_revision": AXIS_REVISION,
        "root_commit": root_commit,
        "openpi_commit": openpi_commit,
    }
    write_json(report_dir / "source_manifest.json", source_manifest)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(NESTED_SRC) + os.pathsep + str(ROOT),
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in gpus),
            "JAX_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
            "HF_HUB_OFFLINE": "1",
            "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
            "AXIS_PI05_LORA_STEPS": "1",
            "AXIS_PI05_LORA_MICROBATCH": "1",
            "AXIS_PI05_LORA_ACCUM": str(spec["accum"]),
            "AXIS_PI05_LORA_EFFECTIVE_BATCH": "64",
            "AXIS_PI05_LORA_BATCH_SIZE": "1",
            "AXIS_PI05_LORA_PEAK_LR": "1e-6",
            "AXIS_PI05_LORA_OBJECTIVE": "valid7_temporal",
            "AXIS_PI05_LORA_RUN_ID": run_id,
            "AXIS_PI05_LORA_RUN_DIR": str(output_dir),
            "AXIS_PI05_LORA_NORM_STATS": str(NORM),
            "AXIS_PI05_LORA_DATA_ROOT": str(DATA_ROOT),
            "AXIS_PI05_LORA_SAMPLING_MANIFEST": str(SAMPLING),
            "AXIS_PI05_AXIS_PARAMS": str(PARAMS),
            "AXIS_PI05_AUDIT_METADATA": "1",
            "AXIS_PI05_AUDIT_REPLAY_DIR": str(report_dir),
            "AXIS_PI05_AUDIT_TENSOR_OUTPUT_DIR": str(tensor_dir),
            "AXIS_PI05_AUDIT_ROOT_COMMIT": root_commit,
            "AXIS_PI05_AUDIT_OPENPI_COMMIT": openpi_commit,
            "AXIS_PI05_AUDIT_SOURCE_MANIFEST": str(report_dir / "source_manifest.json"),
            "AXIS_PI05_AUDIT_RESOLVED_CONFIG": str(report_dir / "resolved_config.yaml"),
            "AXIS_PI05_FIXED_REPLAY": str(FIXTURE),
            "AXIS_PI05_SKIP_OPTIMIZER_UPDATE": "1" if spec["skip_update"] else "0",
            "AXIS_PI05_SAVE_STEP0": "1" if spec["save_step0"] else "0",
            "AXIS_PI05_CHECKPOINT_ASYNC": "1" if spec["async"] else "0",
            "AXIS_PI05_WAIT_FOR_STEP0_SAVE": "1" if spec["wait"] else "0",
            "OPENPI_KEEP_STEPS": "",
            "OPENPI_MAX_TO_KEEP": "2",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
            "AXIS_PI05_WANDB_MODE": "disabled",
        }
    )
    log_path = report_dir / "train.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [str(PYTHON), str(ROOT / "third_party/axis_openpi/scripts/train.py"), "axis_pi05_lora_screen"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    after = _snapshot()
    metrics = _read_metrics(output_dir / "training_metrics.jsonl")
    failure_path = report_dir / "failure.json"
    result = {
        "schema_version": 1,
        "status": "passed" if completed.returncode == 0 else "failed",
        "unit": unit,
        "repeat": repeat,
        "run_id": run_id,
        "returncode": completed.returncode,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "gpus": list(gpus),
        "gpu0_touched": False,
        "optimizer_updates_completed": sum(1 for item in metrics if int(item.get("optimizer_step", 0)) > 0),
        "metrics": metrics,
        "failure": json.loads(failure_path.read_text()) if failure_path.is_file() else None,
        "log": str(log_path),
        "report_dir": str(report_dir),
        "output_dir": str(output_dir),
        "gpu_before": before,
        "gpu_after": after,
    }
    write_json(report_dir / "run_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", default=",".join(UNITS), help="comma-separated R3 unit IDs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--continue-on-failure", action="store_true")
    args = parser.parse_args()
    if not FIXTURE.is_file() or not FIXTURE_MANIFEST.is_file():
        raise FileNotFoundError("build the fixed 64-sample replay fixture before R3")
    units = [value.strip() for value in args.units.split(",") if value.strip()]
    unknown = [value for value in units if value not in UNITS]
    if unknown:
        raise ValueError(f"unknown R3 units: {unknown}")
    root_commit, openpi_commit = _require_clean()
    runs = []
    for unit in units:
        repeat_count = 1 if unit == "D12_ASYNC_OVERLAP_U1" else args.repeats
        for repeat in range(1, repeat_count + 1):
            if args.max_runs is not None and len(runs) >= args.max_runs:
                break
            result = _run(unit, repeat, root_commit, openpi_commit)
            runs.append(result)
            if result["status"] == "failed" and not args.continue_on_failure:
                write_json(REPORT_ROOT / "matrix_partial_summary.json", {"status": "stopped_on_failure", "runs": runs})
                raise RuntimeError(f"R3 unit failed: {unit} repeat {repeat}; see {result['log']}")
        if args.max_runs is not None and len(runs) >= args.max_runs:
            break
    passed = all(item["status"] == "passed" for item in runs) and len(runs) > 0
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "root_commit": root_commit,
        "openpi_commit": openpi_commit,
        "fixed_replay_manifest": str(FIXTURE_MANIFEST),
        "fixed_replay_sha256": sha256_file(FIXTURE),
        "runs": runs,
    }
    write_json(REPORT_ROOT / "matrix_summary.json", summary)
    print(json.dumps({"status": summary["status"], "runs": len(runs)}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
