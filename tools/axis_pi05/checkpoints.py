from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, resolve_path, sha256_file, sha256_tree, write_json


KEEP_STEPS = {500, 1000, 3000}


def validate_internal_step(path: Path, expected: int | None = None) -> int:
    try:
        step = int(path.name)
    except ValueError as exc:
        raise ValueError(f"checkpoint 目录必须用 internal train_state.step 命名: {path}") from exc
    if step <= 0:
        raise ValueError(f"checkpoint step 必须为正数: {path}")
    if expected is not None and step != expected:
        raise ValueError(f"checkpoint step mismatch: expected={expected}, actual={step}")
    return step


def checkpoint_manifest(checkpoint: Path, *, norm: Path, mix_manifest: Path, trainable_set: Path, axis_params: Path, optimizer_state: str, restore_smoke: bool) -> dict[str, Any]:
    step = validate_internal_step(checkpoint)
    params_path = checkpoint / "params"
    if not params_path.exists():
        params_path = checkpoint / "train_state"
    result = {
        "schema_version": 1,
        "internal_step": step,
        "params_sha256": sha256_tree(params_path) if params_path.is_dir() else sha256_file(params_path) if params_path.is_file() else None,
        "norm_sha256": sha256_file(norm),
        "mix_sha256": sha256_file(mix_manifest),
        "trainable_set_sha256": sha256_file(trainable_set),
        "optimizer_state": optimizer_state,
        "axis_params_sha256": sha256_tree(axis_params) if axis_params.is_dir() else sha256_file(axis_params) if axis_params.is_file() else None,
        "restore_smoke": bool(restore_smoke),
        "checkpoint_tree_sha256": sha256_tree(checkpoint),
    }
    if step not in KEEP_STEPS:
        result["retention_warning"] = f"step {step} is not one of {sorted(KEEP_STEPS)}"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="write or validate an internal-step checkpoint manifest")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--norm", required=True, type=Path)
    ap.add_argument("--mix-manifest", required=True, type=Path)
    ap.add_argument("--trainable-set", required=True, type=Path)
    ap.add_argument("--axis-params", required=True, type=Path)
    ap.add_argument("--optimizer-state", choices=["present", "absent"], default="present")
    ap.add_argument("--restore-smoke", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = checkpoint_manifest(
        resolve_path(args.checkpoint), norm=resolve_path(args.norm), mix_manifest=resolve_path(args.mix_manifest),
        trainable_set=resolve_path(args.trainable_set), axis_params=resolve_path(args.axis_params),
        optimizer_state=args.optimizer_state, restore_smoke=args.restore_smoke,
    )
    out = resolve_path(args.out) if args.out else resolve_path(args.checkpoint) / "checkpoint_manifest.json"
    write_json(out, result)
    print(json.dumps(result, indent=2))
    if not result["restore_smoke"] or result["params_sha256"] is None:
        raise SystemExit("checkpoint manifest gate failed")


if __name__ == "__main__":
    main()
