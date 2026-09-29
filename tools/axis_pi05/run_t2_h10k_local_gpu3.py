#!/usr/bin/env python3
"""Download official T2 H10K folded weights, FAST-export, then C20 on local GPU3.

Does not restore JAX 0.11 Orbax on this machine. Does not upload or submit.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, write_json
from . import audit_fast_submission as audit
from . import run_track12_step100_paired_eval as t12
from . import run_track3_n5_official_paired_eval as n5
from . import run_track3_q5_step10_frozen_eval as q7
from .export_candidate_fast_zip import (
    FIXTURES,
    R6_FOLDED,
    SERVER_TEMPLATE,
    VENDOR_SOURCE,
    mechanical_audit,
    stamp,
)
from .export_fast_submission_weights import export_fast, inventory_from_model_dir
from .prepare_lerobot_assets import prepare


# Placeholder only. This repo was never created (HF 404 as of 2026-09-14).
# Token works; canary/R6 private repos exist. Do not --use-hf until a real
# private repo is created AND the 14GB file is uploaded from local after S3
# finishes. Never interrupt the live S3 curl to switch channels.
HF_REPO = "Huangpenguin/parc2026-t2-p30-r6-h10k-r1-folded-10000"
EXPECTED_MODEL_SHA = "bfc678f96190c922a84aa893af111db6ed7126047274257eee449993aa2d7cd3"
EXPECTED_NORM_SHA = "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41"
EXPECTED_BYTES = 14467165872
GPU = "3"
FOLDED_DIR = ROOT / "outputs/axis_pi05/t2_p30_r6_h10k_r1_folded"
NORM = ROOT / "data/manifests_track12/T2-P30/norm_stats.json"
R6_C20 = ROOT / "data/reports/axis_libero_30k/track12_t2_p30_step100_c20_eval/paired_summary.json"
EVIDENCE = ROOT / "data/reports/axis_libero_30k/t2_p30_r6_h10k_r1_local_gpu3_c20_20260914"
OUTPUT = ROOT / "outputs/axis_pi05/t2_p30_r6_h10k_r1_local_gpu3_c20"


def _gpu_guard() -> dict[str, Any]:
    selected = n5.assert_eval_gpu_allowed(int(GPU))
    for row in n5._gpu_rows():
        if row["index"] in {0, 1, 2} and row["memory_used_mib"] > 256:
            print(f"note: GPU{row['index']} busy {row['memory_used_mib']} MiB; not touching it", flush=True)
    return selected


def wait_for_model(*, timeout_sec: int, use_hf: bool) -> Path:
    FOLDED_DIR.mkdir(parents=True, exist_ok=True)
    dest = FOLDED_DIR / "model.safetensors"
    deadline = time.time() + timeout_sec
    while True:
        if dest.is_file() and dest.stat().st_size == EXPECTED_BYTES:
            digest = sha256_file(dest)
            if digest == EXPECTED_MODEL_SHA:
                print(f"local model ok sha256={digest}", flush=True)
                return dest
            print(f"local model sha mismatch {digest} size={dest.stat().st_size}; waiting", flush=True)
        elif dest.is_file():
            print(f"local model size {dest.stat().st_size} expected {EXPECTED_BYTES}; waiting", flush=True)
        if use_hf:
            try:
                from huggingface_hub import hf_hub_download

                print(f"trying HF {HF_REPO}", flush=True)
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="model",
                    filename="model.safetensors",
                    local_dir=str(FOLDED_DIR),
                )
            except Exception as exc:
                print(f"HF not ready: {type(exc).__name__}: {exc}", flush=True)
        if dest.is_file() and dest.stat().st_size == EXPECTED_BYTES:
            digest = sha256_file(dest)
            if digest == EXPECTED_MODEL_SHA:
                print(f"model ok sha256={digest}", flush=True)
                return dest
        if time.time() >= deadline:
            raise FileNotFoundError(
                f"missing folded weights at {dest}. Copy official model.safetensors here (sha {EXPECTED_MODEL_SHA})."
            )
        time.sleep(30)


def prepare_processors() -> None:
    actual_norm = sha256_file(NORM)
    if actual_norm != EXPECTED_NORM_SHA:
        raise ValueError(f"T2 norm SHA {actual_norm} != {EXPECTED_NORM_SHA}")
    prepare(
        FOLDED_DIR,
        R6_FOLDED / "config.json",
        NORM,
        source_checkpoint="official t2_p30_r6_h10k_r1 step 10000 folded on jax 0.11.1",
    )
    if sha256_file(FOLDED_DIR / "model.safetensors") != EXPECTED_MODEL_SHA:
        raise ValueError("prepare() must not rewrite model.safetensors")


def fast_export(report_root: Path, staging: Path) -> dict[str, Any]:
    identity = {
        "source_kind": "on_disk_t2_p30_r6_h10k_r1_folded_pytorch",
        "candidate": "t2_p30_r6_h10k_r1",
        "topology": "L2_action_lora_heads",
        "training_steps": 10000,
        "model_sha256": EXPECTED_MODEL_SHA,
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "processor_sha256": {
            name: sha256_file(FOLDED_DIR / name)
            for name in (
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_2_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
        },
        "lock_mode": "candidate_locked",
    }
    write_json(report_root / "folded_identity.json", identity)
    inventory_from_model_dir(
        FOLDED_DIR,
        staging,
        report_root / "source_inventory.json",
        server_template=SERVER_TEMPLATE,
        vendor_source=VENDOR_SOURCE,
        identity=identity,
    )
    export_report = export_fast(staging, report_root)
    audit_report = mechanical_audit(staging, report_root, identity)
    if audit_report.get("status") != "passed":
        raise RuntimeError(f"mechanical FAST audit failed: {audit_report}")
    parity = audit.run_fixed_parity(staging, FIXTURES, report_root, FOLDED_DIR, GPU)
    if parity.get("status") != "passed":
        raise RuntimeError(f"FAST vs folded parity failed: {parity.get('metrics')}")
    if float(parity["metrics"]["max_abs_error"]) > 1e-3 and parity["metrics"]["gripper_sign_agreement"] < 1.0:
        raise RuntimeError("FAST parity exceeded deployment gate")
    if parity["metrics"]["gripper_sign_agreement"] < 1.0 or parity["metrics"]["finite_rate"] != 1.0:
        raise RuntimeError("FAST gripper/finite gate failed")
    latency = audit.run_server_latency(staging, report_root, FIXTURES, GPU)
    if latency.get("status") != "passed":
        raise RuntimeError("policy server smoke/latency failed")
    return identity


class FastPolicyServer:
    def __init__(self, staging: Path, output_dir: Path) -> None:
        self.staging = staging
        self.output_dir = output_dir
        self.port = audit._free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.client: Any = None

    def start(self) -> Any:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.output_dir / "fast_policy_server.log").open("w", encoding="utf-8")
        env = audit._fast_server_environment(GPU)
        env["LIBERO_CONTROL_FREQ"] = "20"
        env["MUJOCO_GL"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = GPU
        env["AXIS_PI05_EVAL_GPU"] = GPU
        self.process = subprocess.Popen(
            audit._local_fast_server_command(self.staging, self.port),
            cwd=self.staging,
            env=env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        from pipeline.remote_policy import RemotePolicyClient

        self.client = RemotePolicyClient(f"http://127.0.0.1:{self.port}", timeout_sec=90.0)
        try:
            self.client.wait_for_server(timeout=900.0)
        except Exception:
            self.close()
            raise
        return self.client

    def close(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            return {"pid": None, "returncode": None, "clean_shutdown": True}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        return {"pid": process.pid, "returncode": process.returncode, "clean_shutdown": process.poll() is not None}


def _ensure_eval_imports() -> None:
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".eval_libero"))
    for import_path in (ROOT, ROOT / "LIBERO-plus", q7.OPENPI_ROOT / "src"):
        text = str(import_path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _episode_identity(identity: dict[str, Any]) -> dict[str, Any]:
    row = dict(identity)
    row.setdefault("path", str(FOLDED_DIR))
    row.setdefault("role", "t2_h10k_fast")
    row.setdefault("params_tree_sha256", row.get("model_sha256") or EXPECTED_MODEL_SHA)
    row.setdefault("metadata_sha256", None)
    return row


def run_c20(staging: Path, identity: dict[str, Any]) -> dict[str, Any]:
    _ensure_eval_imports()
    from .libero_asset_bridge import activate as activate_libero_asset_bridge
    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.rollout import RolloutExecutor

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ["AXIS_PI05_EVAL_GPU"] = GPU
    os.environ["LIBERO_CONTROL_FREQ"] = "20"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = GPU
    activate_libero_asset_bridge(ROOT)
    cases = t12.load_c20_cases()
    r6 = json.loads(R6_C20.read_text(encoding="utf-8"))
    r6_by_id = {row["case_id"]: row for row in r6["pairs"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    q7.EVIDENCE_ROOT = EVIDENCE
    q7.OUTPUT_ROOT = OUTPUT
    q7.NORM_STATS = NORM
    q7.CONFIG_NAME = "axis_pi05_track12_lora"
    cursor = {
        "schema_version": "t2_h10k_gpu3_c20_cursor_v1",
        "status": "running",
        "pid": os.getpid(),
        "completed_case_ids": [],
        "model_sha256": EXPECTED_MODEL_SHA,
    }
    q7.atomic_json(EVIDENCE / "cursor.json", cursor)
    manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=300, seed=1000))
    step_root = OUTPUT / "full" / "step_10000"
    step_root.mkdir(parents=True, exist_ok=True)
    os.environ["T3_VIDEO_DIR"] = str(step_root / "videos")
    os.environ["T3_VIDEO_FPS"] = "20"
    executor = RolloutExecutor(manager, manager.config)
    server = FastPolicyServer(staging, step_root)
    client = server.start()
    policy = q7.Execute8Policy(client)
    rows: list[dict[str, Any]] = []
    try:
        for index, case in enumerate(cases):
            case_path = q7._case_file(step_root, str(case["case_id"]))
            completed = q7._load_completed(case_path, case, 10000)
            if completed is not None:
                rows.append(completed)
                cursor["completed_case_ids"] = sorted(set(cursor.get("completed_case_ids", [])) | {case["case_id"]})
                q7.atomic_json(EVIDENCE / "cursor.json", cursor)
                continue
            print(f"C20 {index + 1}/20 {case['case_id']}", flush=True)
            task_info = q7._task_info_for_case(case, manager, {})
            init_state = manager.get_init_state(task_info, int(case["init_state_index"])) if case["benchmark"] != "libero_mix" else __import__("numpy").asarray(task_info.init_states[int(case["init_state_index"])])
            episode = executor.evaluate_case(
                policy,
                task_info,
                init_state=init_state,
                env_seed=int(case["env_seed"]),
                policy_seed=int(case["policy_seed"]),
                max_steps=int(case["max_steps"]),
                case_id=str(case["case_id"]),
                perturbation=PerturbationConfig(),
            )
            row = q7._episode_row(case, 10000, _episode_identity(identity), episode, step_root)
            q7.atomic_json(case_path, row)
            rows.append(row)
            cursor["completed_case_ids"] = sorted(set(cursor.get("completed_case_ids", [])) | {case["case_id"]})
            q7.atomic_json(EVIDENCE / "cursor.json", cursor)
    finally:
        q7.atomic_json(step_root / "server_shutdown.json", server.close())
        os.environ.pop("T3_VIDEO_DIR", None)

    pairs = []
    for row in rows:
        base = r6_by_id[row["case_id"]]
        pairs.append(
            {
                "case_id": row["case_id"],
                "panel": "C20",
                "baseline_success": bool(base["baseline_success"]),
                "candidate_success": bool(row.get("success")),
                "success_delta": int(bool(row.get("success"))) - int(bool(base["baseline_success"])),
                "baseline_collision": bool(base["baseline_collision"]),
                "candidate_collision": bool(row.get("collided")),
                "baseline_timeout": bool(base["baseline_timeout"]),
                "candidate_timeout": bool(row.get("timeout")),
            }
        )
    base_n = sum(p["baseline_success"] for p in pairs)
    cand_n = sum(p["candidate_success"] for p in pairs)
    delta = cand_n - base_n
    summary = {
        "schema_version": "t2_h10k_gpu3_c20_paired_summary_v1",
        "track": "T2",
        "status": "evaluated" if len(pairs) == 20 else "failed",
        "official_result": False,
        "auto_submit": False,
        "training_steps": 10000,
        "run_id": "t2_p30_r6_h10k_r1",
        "model_sha256": EXPECTED_MODEL_SHA,
        "norm_sha256": EXPECTED_NORM_SHA,
        "r6_baseline_source": str(R6_C20),
        "paired_case_count": len(pairs),
        "c20": {"baseline_success": base_n, "candidate_success": cand_n, "delta": delta, "n": 20},
        "gates": {
            "coverage_20_paired": len(pairs) == 20,
            "c20_drop_at_most_2_of_20": delta >= -2,
            "beats_r6_official_floor_0.33": "not_evaluated",
        },
        "pairs": pairs,
        "gpu": int(GPU),
    }
    q7.atomic_json(EVIDENCE / "paired_summary.json", summary)
    cursor["status"] = summary["status"]
    q7.atomic_json(EVIDENCE / "cursor.json", cursor)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--skip-c20", action="store_true")
    parser.add_argument("--c20-staging", type=Path, default=None)
    parser.add_argument("--identity-json", type=Path, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    selected = _gpu_guard()
    print("gpu3", selected, flush=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if args.c20_staging is not None:
        staging = args.c20_staging.resolve()
        identity_path = (args.identity_json or (EVIDENCE / "fast_20260914T191345" / "folded_identity.json")).resolve()
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        print(f"c20_resume staging={staging} identity={identity_path}", flush=True)
    else:
        wait_for_model(timeout_sec=args.wait_seconds, use_hf=args.use_hf)
        prepare_processors()
        stamp_id = stamp()
        report_root = EVIDENCE / f"fast_{stamp_id}"
        staging = ROOT / "outputs/axis_pi05" / f"t2_p30_r6_h10k_r1_fast_staging_{stamp_id}"
        report_root.mkdir(parents=True, exist_ok=True)
        identity = fast_export(report_root, staging)
        write_json(report_root / "gpu.json", selected)
        if args.skip_c20:
            print(json.dumps({"status": "fast_only", "staging": str(staging)}, indent=2))
            return 0
    summary = run_c20(staging, identity)
    print(json.dumps({"status": summary["status"], "c20": summary["c20"], "evidence": str(EVIDENCE / "paired_summary.json")}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "evaluated" else 1


if __name__ == "__main__":
    sys.exit(main())
