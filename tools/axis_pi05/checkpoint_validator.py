from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_json, read_table, resolve_path, sha256_file, sha256_tree, write_json


def _flatten_array_entries(value: Any) -> list[dict[str, Any]]:
    import jax
    import numpy as np

    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        if not hasattr(leaf, "shape"):
            continue
        array = np.asarray(leaf)
        entries.append({
            "path": "/".join(str(getattr(part, "key", getattr(part, "idx", part))) for part in path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "numel": int(array.size),
        })
    return sorted(entries, key=lambda item: item["path"])


def stable_checkpoint_hash(checkpoint: Path) -> str:
    return sha256_tree(
        checkpoint,
        exclude=(
            "checkpoint_manifest.json",
            "checkpoint_manifest_v2.json",
            "restore_validation.json",
            "LOCK",
            "_LOCK",
        ),
    )


def _canonical_norm_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Project norm JSON to the OpenPI NormStats contract.

    The v2 manifest also records min/max/median for auditability, while
    OpenPI's checkpoint asset serializer intentionally stores only the four
    fields consumed by ``NormStats``.  Compare that semantic contract rather
    than JSON formatting or the audit-only fields.
    """
    stats = value.get("norm_stats", value)
    if not isinstance(stats, dict):
        raise ValueError("norm stats must contain a mapping")
    fields = ("mean", "std", "q01", "q99")
    return {
        str(name): {field: entry.get(field) for field in fields}
        for name, entry in sorted(stats.items())
        if isinstance(entry, dict)
    }


def _load_trainer():
    train_path = ROOT / "third_party/axis_openpi/scripts/train.py"
    spec = importlib.util.spec_from_file_location("axis_pi05_checkpoint_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer: {train_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_env(scope: str, mix: str, axis_params: Path, sampling_manifest: Path, norm: Path) -> None:
    rows = read_table(sampling_manifest, columns=["source_repo", "episode_sampling_probability"])
    source_ids = sorted({str(row["source_repo"]) for row in rows})
    source_mass = {
        source: sum(float(row["episode_sampling_probability"]) for row in rows if str(row["source_repo"]) == source)
        for source in source_ids
    }
    os.environ.update({
        "AXIS_PI05_SCOPE": scope,
        "AXIS_PI05_MIX": mix,
        "AXIS_PI05_AXIS_PARAMS": str(axis_params),
        "AXIS_PI05_SAMPLING_MANIFEST": str(sampling_manifest),
        "AXIS_PI05_TRAIN_MANIFEST": str(sampling_manifest.with_name("train_episodes.parquet")),
        # AssetsConfig appends AXIS_PI05_MIX, so pass the manifests root and
        # not the per-mix directory (otherwise it searches M1/M1).
        "AXIS_PI05_ASSETS_ROOT": str(norm.parent.parent),
        "AXIS_PI05_LOCAL_REPO_IDS": ",".join(source_ids),
        "AXIS_PI05_LOCAL_REPO_ROOTS": os.pathsep.join(str(ROOT / "data/raw/hf" / source) for source in source_ids),
        "AXIS_PI05_SOURCE_WEIGHTS": ",".join(str(source_mass[source]) for source in source_ids),
        "AXIS_PI05_CAST_TRAINABLE_BF16": "0",
        "AXIS_PI05_GLOBAL_BATCH": "3",
        "AXIS_PI05_STEPS": "3000",
        "AXIS_PI05_WANDB_MODE": "disabled",
    })


def validate(
    checkpoint: Path,
    *,
    scope: str,
    mix: str,
    axis_params: Path,
    sampling_manifest: Path,
    norm: Path,
    seed: int = 1000,
) -> dict[str, Any]:
    if not checkpoint.is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").exists():
        raise FileNotFoundError(f"not an Orbax checkpoint directory: {checkpoint}")

    # The Axis config registry is constructed at module import time, so set
    # all run-specific environment variables before importing any OpenPI
    # training module (checkpoints/data_loader transitively import config).
    _prepare_env(scope, mix, axis_params, sampling_manifest, norm)
    import jax
    import numpy as np
    from flax import nnx
    import openpi.training.checkpoints as checkpoints
    import openpi.training.sharding as sharding

    import openpi.training.config as config_lib
    import openpi.training.data_loader as data_loader
    train_module = _load_trainer()
    config = config_lib.get_config("axis_pi05_libero_mix")
    mesh = sharding.make_mesh(config.fsdp_devices)
    shape_state, _ = train_module.init_train_state(config, jax.random.key(seed), mesh, resume=True)
    manager, resuming = checkpoints.initialize_checkpoint_dir(checkpoint.parent, keep_period=None, overwrite=False, resume=True)
    if not resuming:
        raise RuntimeError(f"checkpoint manager did not find checkpoints in {checkpoint.parent}")
    restored = checkpoints.restore_state(manager, shape_state, None, int(checkpoint.name))
    jax.block_until_ready(restored)
    if int(np.asarray(restored.step)) != int(checkpoint.name):
        raise ValueError(f"restored step mismatch: {restored.step} != {checkpoint.name}")

    params_entries = _flatten_array_entries(restored.params)
    opt_entries = _flatten_array_entries(restored.opt_state)
    if not params_entries or not opt_entries:
        raise ValueError("restored checkpoint lacks params or optimizer state")
    checkpoint_norm = checkpoint / "assets" / mix / "norm_stats.json"
    if not checkpoint_norm.exists():
        raise FileNotFoundError(f"checkpoint norm asset missing: {checkpoint_norm}")
    external_norm = json.loads(norm.read_text(encoding="utf-8"))
    restored_norm = json.loads(checkpoint_norm.read_text(encoding="utf-8"))
    norm_equal = canonical_json(_canonical_norm_payload(external_norm)) == canonical_json(
        _canonical_norm_payload(restored_norm)
    )
    if not norm_equal:
        raise ValueError("checkpoint norm asset is not structurally identical to canonical norm stats")

    # Use the first deterministic, unshuffled training batch as the fixed
    # forward sample. This exercises the restored parameters through the real
    # model/data transforms rather than a synthetic shape-only check.
    fixed_loader = data_loader.create_data_loader(config, sharding=None, shuffle=False, num_batches=1)
    observation, actions = next(iter(fixed_loader))
    model = nnx.merge(restored.model_def, restored.params)
    model.eval()
    with sharding.set_mesh(mesh):
        loss = model.compute_loss(jax.random.key(seed + 1), observation, actions, train=False)
    loss = jax.device_get(loss)
    if not np.all(np.isfinite(np.asarray(loss))):
        raise FloatingPointError(f"fixed sample forward produced non-finite loss: {loss}")

    tree_hash = stable_checkpoint_hash(checkpoint)
    report = {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "internal_step": int(checkpoint.name),
        "scope": scope,
        "mix": mix,
        "params_restore": "passed",
        "train_state_restore": "passed",
        "optimizer_state_present": True,
        "optimizer_state_leaf_count": len(opt_entries),
        "restored_param_leaf_count": len(params_entries),
        "optimizer_state_dtypes": sorted({entry["dtype"] for entry in opt_entries}),
        "norm_asset_canonical_equal": norm_equal,
        "fixed_sample_forward": "passed",
        "fixed_sample_loss_mean": float(np.mean(np.asarray(loss))),
        "stable_checkpoint_tree_sha256": tree_hash,
        "excluded_from_stable_hash": ["checkpoint_manifest.json", "checkpoint_manifest_v2.json", "restore_validation.json", "LOCK", "_LOCK"],
        "sampling_manifest_sha256": sha256_file(sampling_manifest),
        "norm_sha256": sha256_file(norm),
        "axis_params_sha256": sha256_tree(axis_params),
        "params_metadata_sha256": hashlib.sha256(canonical_json(params_entries)).hexdigest(),
        "optimizer_metadata_sha256": hashlib.sha256(canonical_json(opt_entries)).hexdigest(),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="restore and validate an Axis Pi0.5 v2 checkpoint")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--scope", required=True, choices=["F0", "F1", "F2"])
    ap.add_argument("--mix", required=True, choices=["M0", "M1", "M2"])
    ap.add_argument("--axis-params", required=True, type=Path)
    ap.add_argument("--sampling-manifest", required=True, type=Path)
    ap.add_argument("--norm", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    report = validate(
        resolve_path(args.checkpoint),
        scope=args.scope,
        mix=args.mix,
        axis_params=resolve_path(args.axis_params),
        sampling_manifest=resolve_path(args.sampling_manifest),
        norm=resolve_path(args.norm),
        seed=args.seed,
    )
    out = resolve_path(args.out) if args.out else resolve_path(args.checkpoint) / "checkpoint_manifest_v2.json"
    write_json(out, report)
    write_json(resolve_path(args.checkpoint) / "restore_validation.json", {"schema_version": 2, "status": "passed", "report_sha256": sha256_file(out)})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
