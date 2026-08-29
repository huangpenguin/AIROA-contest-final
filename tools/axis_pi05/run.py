from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import ROOT, git_info, load_yaml, read_table, resolve_path, sha256_file, sha256_tree, write_json
from .lock import load_and_validate


STAGE_ARMS = {
    "T0": [("F0", "M1", 10)],
    "T1": [("F0", "M1", 500), ("F1", "M1", 500), ("F2", "M1", 500)],
    "T2": [("F0", "M0", 1000), ("F0", "M1", 1000), ("F0", "M2", 1000)],
    "T3": [("F0", "M1", 3000)],
}


def parse_arm(value: str) -> tuple[str, str]:
    try:
        scope, mix = value.upper().split("/", 1)
    except ValueError as exc:
        raise ValueError(f"arm 必须是 F0/M1 形式: {value}") from exc
    if scope not in {"F0", "F1", "F2"} or mix not in {"M0", "M1", "M2"}:
        raise ValueError(f"非法 arm: {value}")
    return scope, mix


def stage_arms(stage: str, requested: str | None) -> list[tuple[str, str, int]]:
    if requested:
        scope, mix = parse_arm(requested)
        default_steps = next((steps for f, m, steps in STAGE_ARMS[stage] if f == scope and m == mix), STAGE_ARMS[stage][0][2])
        return [(scope, mix, default_steps)]
    if stage in {"T2", "T3"}:
        winner = os.environ.get("AXIS_PI05_WINNER")
        if not winner:
            raise RuntimeError(f"{stage} 需要操作者先确认 winner；用 --arm F*/M* 或设置 AXIS_PI05_WINNER")
        scope, mix = parse_arm(winner)
        return [(scope, mix, next((steps for f, m, steps in STAGE_ARMS[stage] if f == scope and m == mix), STAGE_ARMS[stage][0][2]))]
    return STAGE_ARMS[stage]


def load_automatic_winner(root: Path) -> tuple[str, dict[str, Any]]:
    decision_path = resolve_path(
        os.environ.get(
            "AXIS_PI05_WINNER_DECISION",
            root / "reviews/axis_pi05/model_selection_v1/winner_decision.json",
        ),
        root,
    )
    if not decision_path.is_file():
        raise RuntimeError(f"T3 需要自动 winner 决策文件: {decision_path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("decision") != "automatic" or not decision.get("winner"):
        raise RuntimeError("T3 只接受 decision=automatic 且包含 winner 的决策文件")
    return str(decision["winner"]), decision


def paths(root: Path, mix: str, *, data_version: int = 1) -> dict[str, Path]:
    lock = load_and_validate(root / "configs/axis_pi05/data_sources.lock.yaml")
    model = lock["model"]
    org, repo = model["repo_id"].split("/", 1)
    manifest_root = root / ("data/manifests_v2" if data_version == 2 else "data/manifests")
    return {
        "catalog": root / "data/processed/libero_axis_mix_v1/catalog.jsonl",
        "train_manifest": manifest_root / mix / "train_episodes.parquet",
        "validation_manifest": manifest_root / mix / "validation_episodes.parquet",
        "sampling_weights": manifest_root / mix / "sampling_weights.parquet",
        "mix_manifest": manifest_root / mix / "manifest.json",
        "norm": manifest_root / mix / "norm_stats.json",
        "axis_params": root / "data/models" / org / repo / model["revision"] / "params",
    }


def preflight(root: Path, mix: str, *, dry_run: bool, data_version: int = 1) -> dict[str, Any]:
    required = paths(root, mix, data_version=data_version)
    missing = [str(path.relative_to(root)) for path in required.values() if not path.exists()]
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).returncode == 0 if os.name != "nt" else False
    return {"missing": missing, "gpu_available": gpu, "required": {key: str(path) for key, path in required.items()}, "dry_run": dry_run}


def source_settings(root: Path, mix: str, *, data_version: int = 1) -> tuple[list[str], list[str], list[float]]:
    """Resolve local roots and source masses from the locked episode manifest."""
    lock = load_and_validate(root / "configs/axis_pi05/data_sources.lock.yaml")
    rows = read_table(paths(root, mix, data_version=data_version)["sampling_weights"])
    totals: dict[str, float] = {}
    for row in rows:
        source = str(row["source_repo"])
        totals[source] = totals.get(source, 0.0) + float(row["sampling_weight"])
    ordered = [str(spec["repo_id"]) for spec in lock["datasets"] if str(spec["repo_id"]) in totals]
    if not ordered:
        raise RuntimeError(f"sampling manifest has no locked source rows: {mix}")
    weights = [totals[source] for source in ordered]
    return (
        ordered,
        [str(root / "data/raw/hf" / source) for source in ordered],
        weights,
    )


def resolved_config(root: Path, train_config: dict[str, Any], scope: str, mix: str, steps: int, run_id: str) -> dict[str, Any]:
    result = json.loads(json.dumps(train_config))
    configured_warmup = int(train_config["training"].get("warmup_steps", 300))
    warmup_steps = min(configured_warmup, max(0, steps - 1))
    result["resolved"] = {
        "run_id": run_id,
        "scope": scope,
        "mix": mix,
        "steps": steps,
        "precision": os.environ.get("AXIS_PI05_PRECISION", train_config["training"].get("precision", "auto")),
        "global_batch": os.environ.get("AXIS_PI05_GLOBAL_BATCH", train_config["training"].get("global_batch")),
        "peak_lr": float(os.environ.get("AXIS_PI05_PEAK_LR", train_config["training"].get("peak_lr", 5e-5))),
        "decay_lr": float(os.environ.get("AXIS_PI05_DECAY_LR", train_config["training"].get("final_lr", 5e-6))),
        "warmup_steps": int(os.environ.get("AXIS_PI05_WARMUP_STEPS", warmup_steps)),
        "clip_grad_norm": float(os.environ.get("AXIS_PI05_CLIP_GRAD_NORM", train_config["training"].get("gradient_clip_norm", 1.0))),
        "fail_on_nonfinite": os.environ.get("AXIS_PI05_FAIL_ON_NONFINITE", "0") == "1",
        "cast_trainable_bf16": os.environ.get("AXIS_PI05_CAST_TRAINABLE_BF16", "0") == "1",
        "seed": int(train_config["training"].get("seed", 1000)),
        "root": str(root),
        "keep_steps": [500, 1000, 3000],
        "axis_norm_stats_loaded": False,
    }
    return result


def execute(root: Path, train_config: dict[str, Any], stage: str, scope: str, mix: str, steps: int, run_id: str, *, data_version: int = 1) -> Path:
    check = preflight(root, mix, dry_run=False, data_version=data_version)
    if check["missing"]:
        raise RuntimeError(f"训练输入缺失: {check['missing']}")
    if not check["gpu_available"]:
        raise RuntimeError("没有可用 NVIDIA GPU；A12 不能在当前机器上冒充完成")
    git = git_info(root)
    allow_dirty_diagnostic = os.environ.get("AXIS_PI05_ALLOW_DIRTY_DIAGNOSTIC", "0") == "1"
    if data_version == 2 and git["dirty"] and not allow_dirty_diagnostic:
        raise RuntimeError("v2 formal run 必须从 clean Git worktree 启动；请先提交变更或恢复 worktree")
    resolved = resolved_config(root, train_config, scope, mix, steps, run_id)
    if resolved["resolved"]["global_batch"] in (None, "", "null"):
        raise RuntimeError("global batch 未锁定；先执行统一显存 probe 并设置 AXIS_PI05_GLOBAL_BATCH")
    winner_decision = None
    if stage == "T3":
        winner_id, winner_decision = load_automatic_winner(root)
        if not winner_id.lower().startswith(f"{scope.lower()}_{mix.lower()}"):
            raise RuntimeError(f"T3 arm {scope}/{mix} 与 automatic winner {winner_id} 不一致")
    wandb_mode = os.environ.get("AXIS_PI05_WANDB_MODE", train_config["training"].get("wandb_mode", "online"))
    if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("wandb online 需要 WANDB_API_KEY（只从环境变量读取）")
    command_text = os.environ.get("AXIS_PI05_TRAIN_COMMAND")
    if not command_text:
        raise RuntimeError("未设置 AXIS_PI05_TRAIN_COMMAND；必须指向 patched OpenPI trainer，不默认调用旧 LeRobot 入口")
    out = root / ("outputs/axis_pi05_v2" if data_version == 2 else "outputs/axis_pi05") / run_id
    if out.exists():
        raise RuntimeError(f"输出目录已存在，拒绝覆盖: {out}")
    out.mkdir(parents=True)
    resolved_paths = paths(root, mix, data_version=data_version)
    resolved["resolved"]["paths"] = {key: str(value) for key, value in resolved_paths.items()}
    resolved["resolved"]["command"] = command_text
    resolved["resolved"]["diagnostic_only"] = bool(git["dirty"] and allow_dirty_diagnostic)
    if stage == "T3" and winner_decision is not None:
        winner_path = resolve_path(
            os.environ.get(
                "AXIS_PI05_WINNER_DECISION",
                root / "reviews/axis_pi05/model_selection_v1/winner_decision.json",
            ),
            root,
        )
        resolved["resolved"].update({
            "winner": winner_decision["winner"],
            "winner_decision_sha256": sha256_file(winner_path),
            "initialization": "axis_base_params_only",
            "resume_from_t2_checkpoint": False,
            "mixture_norm_policy": "winner_mix_norm",
        })
    write_json(out / "resolved_config.json", resolved)
    provenance = {
        "schema_version": 1,
        "stage": stage,
        "scope": scope,
        "mix": mix,
        "run_id": run_id,
        "command": command_text,
        "git": git,
        "lock_sha256": sha256_file(root / "configs/axis_pi05/data_sources.lock.yaml"),
        "catalog_sha256": sha256_file(resolved_paths["catalog"]),
        "mix_manifest_sha256": sha256_file(resolved_paths["mix_manifest"]),
        "norm_sha256": sha256_file(resolved_paths["norm"]),
        "axis_params_sha256": sha256_tree(resolved_paths["axis_params"]),
        "seed": resolved["resolved"]["seed"],
        "keep_steps": [500, 1000, 3000],
        "created_at_unix": time.time(),
        "diagnostic_only": bool(git["dirty"] and allow_dirty_diagnostic),
        "official_result": False,
    }
    write_json(out / "provenance.json", provenance)
    env = os.environ.copy()
    source_ids, source_roots, source_weights = source_settings(root, mix, data_version=data_version)
    env.update({
        "AXIS_PI05_RUN_ID": run_id,
        "AXIS_PI05_SCOPE": scope,
        "AXIS_PI05_MIX": mix,
        "AXIS_PI05_STEPS": str(steps),
        "AXIS_PI05_NORM_STATS": str(resolved_paths["norm"]),
        "AXIS_PI05_KEEP_STEPS": "500,1000,3000",
        "OPENPI_KEEP_STEPS": "500,1000,3000",
        "OPENPI_MAX_TO_KEEP": "4",
        "AXIS_PI05_AXIS_PARAMS": str(resolved_paths["axis_params"]),
        "AXIS_PI05_OUTPUT_DIR": str(out),
        "AXIS_PI05_WANDB_MODE": wandb_mode,
        "AXIS_PI05_ASSETS_ROOT": str(root / ("data/manifests_v2" if data_version == 2 else "data/manifests")),
        "AXIS_PI05_TRAIN_MANIFEST": str(resolved_paths["train_manifest"]),
        "AXIS_PI05_SAMPLING_MANIFEST": str(resolved_paths["sampling_weights"]),
        "AXIS_PI05_LOCAL_REPO_IDS": ",".join(source_ids),
        "AXIS_PI05_LOCAL_REPO_ROOTS": os.pathsep.join(source_roots),
        "AXIS_PI05_SOURCE_WEIGHTS": ",".join(str(value) for value in source_weights),
        "AXIS_PI05_FAIL_ON_NONFINITE": os.environ.get("AXIS_PI05_FAIL_ON_NONFINITE", "0"),
        "AXIS_PI05_CAST_TRAINABLE_BF16": os.environ.get("AXIS_PI05_CAST_TRAINABLE_BF16", "0"),
        "AXIS_PI05_WARMUP_STEPS": str(resolved["resolved"]["warmup_steps"]),
        "AXIS_PI05_INITIALIZATION": "axis_base_params_only" if stage == "T3" else "axis_base_params",
        "AXIS_PI05_RESUME_FROM_CHECKPOINT": "0" if stage == "T3" else os.environ.get("AXIS_PI05_RESUME_FROM_CHECKPOINT", "0"),
        "AXIS_PI05_DIAGNOSTIC_ONLY": "1" if allow_dirty_diagnostic else "0",
    })
    command = shlex.split(command_text)
    with (out / "train.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        write_json(out / "run_state.json", {"status": "failed", "returncode": result.returncode})
        raise RuntimeError(f"训练命令失败，日志: {out / 'train.log'}")
    write_json(out / "run_state.json", {"status": "completed", "returncode": 0, "checkpoints": sorted(str(p) for p in out.rglob("checkpoint_manifest.json"))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Axis Pi0.5 staged training orchestrator")
    ap.add_argument("--stage", required=True, choices=sorted(STAGE_ARMS))
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/train.yaml")
    ap.add_argument("--arm", help="one arm, e.g. F0/M1")
    ap.add_argument("--steps", type=int, help="override the selected arm steps for a bounded diagnostic run")
    ap.add_argument("--run-id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    root = config_path.parents[2]
    load_and_validate(root / "configs/axis_pi05/data_sources.lock.yaml")
    if args.stage == "T3" and not args.arm:
        winner_id, _ = load_automatic_winner(root)
        winner_match = re.search(r"f([012])[_/-]?m([012])", winner_id.lower())
        if not winner_match:
            raise RuntimeError(f"winner id 无法解析为 F/M arm: {winner_id}")
        selections = [(f"F{winner_match.group(1)}", f"M{winner_match.group(2)}", 3000)]
    else:
        selections = stage_arms(args.stage, args.arm)
    plans = []
    for scope, mix, steps in selections:
        if args.steps is not None:
            if args.steps <= 0:
                raise ValueError("--steps 必须为正整数")
            steps = args.steps
        data_version = int(config.get("data_version", 1))
        check = preflight(root, mix, dry_run=args.dry_run, data_version=data_version)
        run_id = args.run_id or f"{args.stage.lower()}_{scope.lower()}_{mix.lower()}"
        plans.append({"scope": scope, "mix": mix, "steps": steps, "run_id": run_id, "preflight": check})
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "plans": plans, "writes": False, "downloads": False, "training": False}, ensure_ascii=False, indent=2))
        return
    if len(plans) != 1:
        raise RuntimeError("正式执行一次只允许一个 arm；用 --arm 逐个运行，比较报告后再进入下一阶段")
    plan = plans[0]
    data_version = int(config.get("data_version", 1))
    out = execute(root, config, args.stage, plan["scope"], plan["mix"], plan["steps"], plan["run_id"], data_version=data_version)
    print(json.dumps({"status": "completed", "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
