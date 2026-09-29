"""Independent Track3 Blackwell runner for the post-Q7 official handoff.

The command is intentionally subcommand-based. ``preflight`` is read-only
apart from its evidence file. ``step1`` and ``step100`` require an explicit
Track1/2 release proof and use a new run directory. They never use the local
Q5 checkpoint namespace; the trainer loads the original R6 params-only tree
and ``step100`` resumes the same official run with a fresh process.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_json, sha256_file


Q7_MANIFEST = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/q7_official_transfer_manifest.json"
Q7_SIDECAR = Q7_MANIFEST.with_suffix(Q7_MANIFEST.suffix + ".sha256")
Q7_EVIDENCE = Q7_MANIFEST.parent
DEFAULT_OUTPUT_BASE = ROOT / "outputs/axis_pi05/track3_official_blackwell"
DEFAULT_RUN_ID = "track3_four_gold_official_blackwell"
TRAIN = ROOT / "third_party/axis_openpi/scripts/train.py"
CONFIG = ROOT / "third_party/axis_openpi/src/openpi/training/config.py"
R6_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
NORM = ROOT / "data/manifests_track3/four_gold_80_20_v1/norm_stats.json"
SAMPLING = ROOT / "data/manifests_track3/four_gold_80_20_v1/sampling_weights.parquet"
MANIFEST = ROOT / "data/manifests_track3/four_gold_80_20_v1/manifest.json"
FORWARD_ROOT = ROOT / "data/raw/hf/lerobot/libero_plus"
REVERSE_ROOT = ROOT / "data/processed/axis_pi05_track3_reverse_gold_v1"
PYTHON = Path(os.environ.get("AXIS_PI05_PYTHON", str(ROOT / ".venv_axis_pi05/bin/python")))
EXPECTED_R6_TREE_SHA = "d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f"
EXPECTED_NORM_SHA = "faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab"
EXPECTED_MIX_SHA = "cc6c7206458277512a4f8a33a7e43a49044d8d845cea796f70f1fc292b1d8244"
EXPECTED_SAMPLING_SHA = "2276a6bf2fa5b5741cb11f69c3076a865b7aa4b6ff7800baffa40897f8bd1cb9"


def strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def q0_params_tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(ROOT).as_posix().encode()):
        digest.update(f"{sha256_file(file_path)}  {file_path.relative_to(ROOT).as_posix()}\n".encode())
    return digest.hexdigest()


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, name, uuid, total, used, util = [part.strip() for part in line.split(",", 5)]
        rows.append({"index": int(index), "name": name, "uuid": uuid, "memory_total_mib": int(total), "memory_used_mib": int(used), "utilization_gpu_percent": int(util)})
    return rows


def compute_apps() -> list[dict[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            uuid, pid, name, memory = [part.strip() for part in line.split(",", 3)]
            rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(memory)})
    return rows


def _params_and_inputs(*, manifest: Path, params: Path, norm: Path, sampling: Path, mix_manifest: Path, forward: Path, reverse: Path) -> dict[str, Any]:
    required = (manifest, params, norm, sampling, mix_manifest, forward, reverse, TRAIN, CONFIG)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing official Track3 input(s): " + ", ".join(missing))
    manifest_digest = sha256_file(manifest)
    sidecar = manifest.with_suffix(manifest.suffix + ".sha256")
    if not sidecar.is_file() or not sidecar.read_text(encoding="utf-8").split()[0] == manifest_digest:
        raise ValueError("q7 official transfer manifest sidecar mismatch")
    q7 = strict_json_load(manifest)
    if q7.get("status") != "ready_for_official_transfer_after_track12_release" or q7.get("official_training_started") is not False:
        raise ValueError(f"q7 transfer manifest is not a ready, non-started contract: {q7.get('status')}")
    evaluation = q7.get("evaluation", {}).get("e2_paired", {})
    if evaluation.get("status") != "passed" or evaluation.get("cases") != 48:
        raise ValueError("q7 E2 paired evaluation is not passed 48-case evidence")
    params_sha = q0_params_tree_sha(params)
    if params_sha != EXPECTED_R6_TREE_SHA:
        raise ValueError(f"R6 params-only tree mismatch: {params_sha} != {EXPECTED_R6_TREE_SHA}")
    if sha256_file(norm) != EXPECTED_NORM_SHA:
        raise ValueError("Track3 norm stats hash mismatch")
    if sha256_file(sampling) != EXPECTED_SAMPLING_SHA:
        raise ValueError("Track3 sampling weights hash mismatch")
    if sha256_file(mix_manifest) != EXPECTED_MIX_SHA:
        raise ValueError("Track3 mix manifest hash mismatch")
    return {
        "q7_transfer_manifest_sha256": manifest_digest,
        "q7_e2_paired_sha256": evaluation.get("sha256"),
        "r6_params_tree_sha256": params_sha,
        "norm_stats_sha256": sha256_file(norm),
        "sampling_manifest_sha256": sha256_file(sampling),
        "mix_manifest_sha256": sha256_file(mix_manifest),
        "trainer_sha256": sha256_file(TRAIN),
        "config_sha256": sha256_file(CONFIG),
        "forward_root": str(forward.resolve()),
        "reverse_root": str(reverse.resolve()),
    }


def _blackwell_gate() -> dict[str, Any]:
    rows = gpu_snapshot()
    blackwell = [row for row in rows if "blackwell" in row["name"].lower() or "rtx pro 6000" in row["name"].lower()]
    if len(blackwell) != 1:
        raise RuntimeError(f"expected exactly one visible RTX PRO 6000 Blackwell, got {rows}")
    apps = compute_apps()
    if apps:
        raise RuntimeError(f"Blackwell must be free before Track3 official start: {apps}")
    return {"gpu_snapshot": rows, "compute_apps": apps, "selected": blackwell[0]}


def _release_proof(path: Path) -> dict[str, Any]:
    proof = strict_json_load(path)
    if proof.get("status") not in {"passed", "released", "finished", "track12_released"}:
        raise ValueError(f"Track1/2 release proof is not complete: {proof.get('status')}")
    if proof.get("active") is True or proof.get("training_active") is True:
        raise ValueError("Track1/2 release proof still reports an active trainer")
    return proof


def _trainer_environment(*, run_id: str, run_dir: Path, params: Path, norm: Path, sampling: Path, forward: Path, reverse: Path, cuda_visible_devices: str, steps: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT / "third_party/axis_openpi/src"), str(ROOT))),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
        "HF_HUB_OFFLINE": "1",
        "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
        "AXIS_PI05_TRACK3_REPO_ID": "local/track3_four_gold_80_20",
        "AXIS_PI05_TRACK3_LOCAL_REPO_IDS": "lerobot/libero_plus,axis-pi05/track3_reverse_gold_v1",
        "AXIS_PI05_TRACK3_LOCAL_REPO_ROOTS": os.pathsep.join((str(forward), str(reverse))),
        "AXIS_PI05_TRACK3_SOURCE_WEIGHTS": "0.8,0.2",
        "AXIS_PI05_TRACK3_SAMPLING_MANIFEST": str(sampling),
        "AXIS_PI05_TRACK3_NORM_STATS": str(norm),
        "AXIS_PI05_TRACK3_PARAMS": str(params),
        "AXIS_PI05_TRACK3_STEPS": str(steps),
        "AXIS_PI05_TRACK3_GLOBAL_BATCH": "64",
        "AXIS_PI05_TRACK3_MICROBATCH": "1",
        "AXIS_PI05_TRACK3_ACCUMULATION": "64",
        "AXIS_PI05_TRACK3_EFFECTIVE_BATCH": "64",
        "AXIS_PI05_TRACK3_WARMUP_STEPS": "100",
        "AXIS_PI05_TRACK3_PEAK_LR": "1e-6",
        "AXIS_PI05_TRACK3_DECAY_LR": "1e-7",
        "AXIS_PI05_TRACK3_DATALOADER_WORKERS": "0",
        "AXIS_PI05_TRACK3_LOG_INTERVAL": "1",
        "AXIS_PI05_TRACK3_RUN_ID": run_id,
        "AXIS_PI05_TRACK3_OUTPUT_DIR": str(run_dir.parent),
        "AXIS_PI05_TRACK3_RUN_DIR": str(run_dir),
        "OPENPI_KEEP_STEPS": "0,1,100",
        "OPENPI_MAX_TO_KEEP": "3",
        "AXIS_PI05_SAVE_STEP0": "1",
        "AXIS_PI05_WAIT_FOR_STEP0_SAVE": "1",
        "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1",
        "AXIS_PI05_FAIL_ON_NONFINITE": "1",
        "AXIS_PI05_AUDIT_METADATA": "1",
        "AXIS_PI05_AUDIT_REPLAY_DIR": str(run_dir / "audit"),
        "AXIS_PI05_AUDIT_ROOT_COMMIT": "official-transfer-manifest",
        "AXIS_PI05_AUDIT_OPENPI_COMMIT": "official-transfer-manifest",
        "AXIS_PI05_AUDIT_SOURCE_MANIFEST": str(run_dir / "source_manifest.json"),
    })
    return env


def _train_code(*, resume: bool, run_id: str) -> str:
    return f'''\nimport dataclasses, importlib.util, sys\nfrom openpi.training import config as config_lib\nspec = importlib.util.spec_from_file_location("axis_pi05_official_train", {str(TRAIN)!r})\nmodule = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = module\nspec.loader.exec_module(module)\nconfig = config_lib.get_config("axis_pi05_track3_q5")\nmetadata = dict(config.policy_metadata or {{}})\nmetadata.update({{"phase": "track3_official_blackwell", "initialization": "original_r6_params_only_fresh_optimizer", "run_id": {run_id!r}}})\nconfig = dataclasses.replace(config, resume={resume!r}, policy_metadata=metadata)\nmodule.main(config)\n'''


def _restore_code(step: int) -> str:
    return f'''\nimport importlib.util, json, sys\nimport jax\nfrom openpi.training import config as config_lib, sharding\nspec = importlib.util.spec_from_file_location("axis_pi05_official_restore", {str(TRAIN)!r})\nmodule = importlib.util.module_from_spec(spec)\nsys.modules[spec.name] = module\nspec.loader.exec_module(module)\nconfig = config_lib.get_config("axis_pi05_track3_q5")\nmesh = sharding.make_mesh(1)\nmanager, resuming = module._checkpoints.initialize_checkpoint_dir(config.checkpoint_dir, keep_period=None, overwrite=False, resume=True)\nshape, _ = module.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)\nstate = module._checkpoints.restore_state(manager, shape, None, step={step})\njax.block_until_ready(state)\nprint(json.dumps({{"restored_step": int(state.step), "requested_step": {step}, "resuming": bool(resuming), "device_count": jax.device_count(), "devices": [str(device) for device in jax.devices()]}}))\n'''


def _run_train(*, env: dict[str, str], resume: bool, run_id: str, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run([str(PYTHON), "-c", _train_code(resume=resume, run_id=run_id)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return {"returncode": result.returncode, "log": str(log_path)}


def _fresh_restore(*, env: dict[str, str], step: int, output: Path) -> dict[str, Any]:
    result = subprocess.run([str(PYTHON), "-c", _restore_code(step)], cwd=ROOT, env=env, capture_output=True, text=True)
    output.write_text(result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"fresh restore step {step} failed; see {output}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"fresh restore step {step} produced no JSON")
    restored = json.loads(lines[-1])
    if restored.get("restored_step") != step or restored.get("device_count") != 1:
        raise RuntimeError(f"fresh restore step {step} contract mismatch: {restored}")
    return restored


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    run_id = args.run_id
    return {
        "manifest": Path(args.manifest), "params": Path(args.params), "norm": Path(args.norm), "sampling": Path(args.sampling), "mix_manifest": Path(args.mix_manifest),
        "forward": Path(args.forward_root), "reverse": Path(args.reverse_root), "run_dir": Path(args.run_dir) if args.run_dir else DEFAULT_OUTPUT_BASE / run_id,
        "evidence": Path(args.evidence_dir) if args.evidence_dir else ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q8_official_blackwell" / run_id,
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    inputs = _params_and_inputs(manifest=paths["manifest"], params=paths["params"], norm=paths["norm"], sampling=paths["sampling"], mix_manifest=paths["mix_manifest"], forward=paths["forward"], reverse=paths["reverse"])
    gpu = _blackwell_gate()
    result = {"schema_version": "track3_official_blackwell_preflight_v1", "status": "passed", "run_id": args.run_id, "inputs": inputs, "gpu": gpu, "contract": {"config": "axis_pi05_track3_q5", "physical_device_count": 1, "initialization": "original_r6_params_only_fresh_optimizer", "lora_arm": "L2_action_lora_heads", "precision": "float32", "global_batch": 64, "microbatch": 1, "accumulation": 64, "peak_lr": 1e-6, "warmup_steps": 100, "ema": False, "official_result": False, "track12_release_required_before_step1": True}, "training_started": False, "timestamp": time.time()}
    atomic_json(paths["evidence"] / "preflight.json", result)
    return result


def train_phase(args: argparse.Namespace, *, target_step: int, resume: bool) -> dict[str, Any]:
    paths = _paths(args)
    if not args.release_proof:
        raise ValueError("step1/step100 require --release-proof from the naturally finished Track1/2 run")
    release = _release_proof(Path(args.release_proof))
    inputs = _params_and_inputs(manifest=paths["manifest"], params=paths["params"], norm=paths["norm"], sampling=paths["sampling"], mix_manifest=paths["mix_manifest"], forward=paths["forward"], reverse=paths["reverse"])
    if target_step == 1 and paths["run_dir"].exists():
        raise FileExistsError(f"official Track3 run directory already exists: {paths['run_dir']}")
    if target_step == 100:
        prior = strict_json_load(paths["evidence"] / "step1.json")
        if prior.get("status") != "passed" or prior.get("run_dir") != str(paths["run_dir"].resolve()):
            raise RuntimeError("step100 requires a passed step1 report for the same run")
    gpu = _blackwell_gate()
    paths["run_dir"].mkdir(parents=True, exist_ok=(target_step == 100))
    paths["evidence"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["evidence"] / "cursor.json", {"status": "running", "phase": f"step{target_step}", "run_id": args.run_id, "pid": os.getpid(), "timestamp": time.time()})
    env = _trainer_environment(run_id=args.run_id, run_dir=paths["run_dir"], params=paths["params"], norm=paths["norm"], sampling=paths["sampling"], forward=paths["forward"], reverse=paths["reverse"], cuda_visible_devices=args.cuda_visible_devices, steps=target_step)
    report: dict[str, Any] = {"schema_version": "track3_official_blackwell_train_phase_v1", "status": "running", "phase": f"step{target_step}", "run_id": args.run_id, "run_dir": str(paths["run_dir"].resolve()), "inputs": inputs, "release_proof": release, "gpu_before": gpu, "fresh_optimizer": not resume}
    try:
        execution = _run_train(env=env, resume=resume, run_id=args.run_id, log_path=paths["evidence"] / f"step{target_step}.train.log")
        report["execution"] = execution
        if execution["returncode"] != 0:
            raise RuntimeError(f"official Track3 {target_step}-step trainer failed: {execution}")
        restore = _fresh_restore(env=env, step=target_step, output=paths["evidence"] / f"fresh_restore_step{target_step}.log")
        report["fresh_restore"] = restore
        report["gpu_after"] = _blackwell_gate()
        report["status"] = "passed"
        report["official_result"] = False
        report["training_started"] = True
        atomic_json(paths["evidence"] / f"step{target_step}.json", report)
        atomic_json(paths["evidence"] / "cursor.json", {"status": "passed", "phase": f"step{target_step}", "run_id": args.run_id, "timestamp": time.time()})
        atomic_json(paths["evidence"] / "heartbeat.json", {"status": "stopped", "phase": f"step{target_step}", "timestamp": time.time()})
        return report
    except BaseException as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "training_started": True})
        atomic_json(paths["evidence"] / f"step{target_step}.json", report)
        atomic_json(paths["evidence"] / "cursor.json", {"status": "failed", "phase": f"step{target_step}", "error": report["error"], "timestamp": time.time()})
        atomic_json(paths["evidence"] / "heartbeat.json", {"status": "stopped", "phase": f"step{target_step}", "timestamp": time.time()})
        raise


def validate(args: argparse.Namespace) -> dict[str, Any]:
    paths = _paths(args)
    step1 = strict_json_load(paths["evidence"] / "step1.json")
    step100 = strict_json_load(paths["evidence"] / "step100.json")
    if step1.get("status") != "passed" or step100.get("status") != "passed":
        raise ValueError("official Track3 step1/step100 reports are not both passed")
    metrics_path = paths["run_dir"] / "training_metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row.get("optimizer_step") for row in metrics] != list(range(1, 101)):
        raise ValueError("official metrics are not contiguous optimizer steps 1..100")
    if any(not all(__import__("math").isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))) for row in metrics):
        raise ValueError("official metrics contain non-finite values")
    missing = [str(step) for step in (0, 1, 100) if not (paths["run_dir"] / str(step) / "params").is_dir() or not (paths["run_dir"] / str(step) / "train_state").is_dir()]
    if missing:
        raise FileNotFoundError("official checkpoint(s) missing: " + ", ".join(missing))
    partials = [str(path) for path in paths["run_dir"].rglob("*") if ".partial" in path.name]
    if partials:
        raise ValueError(f"official run has partial checkpoint residue: {partials[:5]}")
    result = {"schema_version": "track3_official_blackwell_validation_v1", "status": "passed", "run_id": args.run_id, "run_dir": str(paths["run_dir"].resolve()), "optimizer_steps": 100, "metrics_sha256": sha256_file(metrics_path), "step1_report_sha256": sha256_file(paths["evidence"] / "step1.json"), "step100_report_sha256": sha256_file(paths["evidence"] / "step100.json"), "official_result": False, "stop_after_step100": True}
    atomic_json(paths["evidence"] / "validation.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "step1", "step100", "validate"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--run-id", default=DEFAULT_RUN_ID)
        sub.add_argument("--run-dir", default=None)
        sub.add_argument("--evidence-dir", default=None)
        sub.add_argument("--manifest", default=str(Q7_MANIFEST))
        sub.add_argument("--params", default=str(R6_PARAMS))
        sub.add_argument("--norm", default=str(NORM))
        sub.add_argument("--sampling", default=str(SAMPLING))
        sub.add_argument("--mix-manifest", default=str(MANIFEST))
        sub.add_argument("--forward-root", default=str(FORWARD_ROOT))
        sub.add_argument("--reverse-root", default=str(REVERSE_ROOT))
        sub.add_argument("--cuda-visible-devices", default=os.environ.get("AXIS_PI05_OFFICIAL_CUDA_VISIBLE_DEVICES", "0"))
        if name in {"step1", "step100"}:
            sub.add_argument("--release-proof", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = {"preflight": preflight, "step1": lambda value: train_phase(value, target_step=1, resume=False), "step100": lambda value: train_phase(value, target_step=100, resume=True), "validate": validate}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
