from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_json, load_yaml, resolve_path, write_json


def _path_string(path: tuple[Any, ...]) -> str:
    return "/".join(str(part.key if hasattr(part, "key") else part.idx if hasattr(part, "idx") else part) for part in path)


def _state_entries(state: Any) -> list[dict[str, Any]]:
    import jax
    import numpy as np

    entries: list[dict[str, Any]] = []
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        if not hasattr(value, "shape"):
            continue
        array = np.asarray(value)
        entries.append({
            "path": _path_string(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "numel": int(array.size),
        })
    return sorted(entries, key=lambda item: item["path"])


def _pure_entries(state: Any) -> list[dict[str, Any]]:
    import jax
    import numpy as np

    pure = state.to_pure_dict() if hasattr(state, "to_pure_dict") else state
    entries: list[dict[str, Any]] = []
    for path, value in jax.tree_util.tree_flatten_with_path(pure)[0]:
        if not hasattr(value, "shape"):
            continue
        array = np.asarray(value)
        entries.append({
            "path": _path_string(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "numel": int(array.size),
        })
    return sorted(entries, key=lambda item: item["path"])


def audit_scope(scope: str, *, axis_params: Path, out: Path, seed: int = 1000) -> dict[str, Any]:
    # Imports are deliberately delayed so callers can set JAX_PLATFORMS before
    # loading the large Axis checkpoint.
    import jax
    import importlib.util

    os.environ["AXIS_PI05_SCOPE"] = scope
    os.environ["AXIS_PI05_AXIS_PARAMS"] = str(axis_params)
    os.environ["AXIS_PI05_CAST_TRAINABLE_BF16"] = "0"
    import openpi.training.config as config_lib
    import openpi.training.sharding as sharding

    train_path = ROOT / "third_party/axis_openpi/scripts/train.py"
    spec = importlib.util.spec_from_file_location("axis_pi05_runtime_train", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer: {train_path}")
    train_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = train_module
    spec.loader.exec_module(train_module)

    config = config_lib.get_config("axis_pi05_libero_mix")
    mesh = sharding.make_mesh(1)
    state, _ = train_module.init_train_state(config, jax.random.key(seed), mesh, resume=False)
    jax.block_until_ready(state)

    all_entries = _pure_entries(state.params)
    trainable_state = state.params.filter(config.trainable_filter)
    trainable_entries = _pure_entries(trainable_state)
    frozen_state = state.params.filter(config.freeze_filter)
    frozen_entries = _pure_entries(frozen_state)
    opt_entries = _state_entries(state.opt_state)
    trainable_paths = {entry["path"] for entry in trainable_entries}
    params_by_path = {entry["path"]: entry for entry in all_entries}
    missing = sorted(path for path in trainable_paths if path not in params_by_path)
    optimizer_dtypes = sorted({entry["dtype"] for entry in opt_entries if entry["path"]})
    # Hash only metadata, never array contents; this is stable and reviewable.
    trainable_set_sha = hashlib.sha256(canonical_json(trainable_entries)).hexdigest()
    report = {
        "schema_version": 2,
        "scope": scope,
        "seed": seed,
        "axis_params_path": str(axis_params),
        "axis_restore_executed": True,
        "dtype_cast_trainable_bf16": os.environ.get("AXIS_PI05_CAST_TRAINABLE_BF16") == "1",
        "trainable_parameter_count": sum(entry["numel"] for entry in trainable_entries),
        "all_parameter_count": sum(entry["numel"] for entry in all_entries),
        "frozen_parameter_count": sum(entry["numel"] for entry in frozen_entries),
        "trainable_dtypes": sorted({entry["dtype"] for entry in trainable_entries}),
        "frozen_dtypes": sorted({entry["dtype"] for entry in frozen_entries}),
        "optimizer_state_dtypes": optimizer_dtypes,
        "optimizer_state_present": bool(opt_entries),
        "optimizer_state_leaf_count": len(opt_entries),
        "trainable_paths": [entry["path"] for entry in trainable_entries],
        "trainable_set_sha256": trainable_set_sha,
        "parameters": all_entries,
        "trainable_parameters": trainable_entries,
        "optimizer_state": opt_entries,
        "missing_trainable_paths": missing,
        "runtime_sequence": ["model_create", "axis_partial_restore", "frozen_dtype_cast", "trainable_filter", "optimizer_init"],
    }
    if missing or not trainable_entries or not opt_entries:
        raise RuntimeError(json.dumps({"missing": missing, "trainable": len(trainable_entries), "optimizer": len(opt_entries)}))
    write_json(out, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="runtime Axis Pi0.5 parameter/filter/optimizer audit")
    ap.add_argument("--scope", required=True, choices=["F0", "F1", "F2"])
    ap.add_argument("--axis-params", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    report = audit_scope(args.scope, axis_params=resolve_path(args.axis_params), out=resolve_path(args.out), seed=args.seed)
    print(json.dumps({key: report[key] for key in ("scope", "trainable_parameter_count", "all_parameter_count", "trainable_dtypes", "optimizer_state_dtypes", "trainable_set_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
