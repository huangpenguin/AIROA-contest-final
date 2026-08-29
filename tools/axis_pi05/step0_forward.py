"""Run the deterministic dense Axis Pi0.5 fixed-forward audit.

This is inference only. It loads the released dense Axis parameter tree,
injects the fresh canonical norm through the current OpenPI transforms, and
does not create an optimizer or mutate checkpoint files.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, canonical_json, git_info, resolve_path, sha256_file, sha256_tree, write_json


def _openpi_path() -> None:
    path = ROOT / "third_party" / "axis_openpi" / "src"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _flat_keys(tree: Any) -> set[str]:
    import flax.traverse_util

    return set(flax.traverse_util.flatten_dict(tree, sep="/").keys())


def _array_fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def run(axis_params: Path, norm_path: Path, out_root: Path, *, gpu_label: str = "1", seed: int = 20260828) -> dict[str, Any]:
    _openpi_path()
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.shared import normalize
    import openpi.training.config as config_lib
    import openpi.training.data_loader as data_loader
    from openpi import transforms

    norm_stats = normalize.deserialize_json(norm_path.read_text(encoding="utf-8"))
    model_config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    data_root = ROOT / "data" / "raw" / "hf" / "physical-intelligence" / "libero"
    phase1_root = ROOT / "data" / "reports" / "axis_libero_30k" / "phase1"
    factory = config_lib.LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        assets=config_lib.AssetsConfig(assets_dir=str(phase1_root), asset_id="canonical_10hz"),
        base_config=config_lib.DataConfig(prompt_from_task=True),
        v3_keys=False,
        local_repo_ids=("physical-intelligence/libero",),
        local_repo_roots=(str(data_root),),
        local_sampling_weights=(1.0,),
    )
    data_config = dataclasses.replace(factory.create(phase1_root, model_config), norm_stats=norm_stats)
    dataset = data_loader.create_torch_dataset(data_config, model_config.action_horizon, model_config)
    transformed = data_loader.transform_dataset(dataset, data_config)
    sample = transformed[0]
    obs_data = {
        key: sample[key]
        for key in ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask", "action_is_pad")
        if key in sample
    }
    batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], obs_data)
    observation = model_lib.Observation.from_dict(batched)

    # This config has no LoRA leaves: it is the dense full-FT architecture.
    template = model_config.create(jax.random.key(0))
    expected = nnx.state(template).to_pure_dict()
    params_before = sha256_tree(axis_params)
    loaded = model_lib.restore_params(axis_params, restore_type=np.ndarray, dtype=np.float32)
    expected_keys = _flat_keys(expected)
    loaded_keys = _flat_keys(loaded)
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Axis dense params tree mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model = model_config.load(loaded)
    del template, loaded
    noise = np.random.default_rng(seed).standard_normal((1, 10, 32), dtype=np.float32)
    rng = jax.random.key(seed)
    started = time.monotonic()
    output_a = model.sample_actions(rng, observation, num_steps=10, noise=jnp.asarray(noise))
    output_a = np.asarray(jax.block_until_ready(output_a))
    elapsed_a = time.monotonic() - started
    output_b = model.sample_actions(rng, observation, num_steps=10, noise=jnp.asarray(noise))
    output_b = np.asarray(jax.block_until_ready(output_b))
    elapsed_b = time.monotonic() - started - elapsed_a
    params_after = sha256_tree(axis_params)
    if not np.array_equal(output_a, output_b):
        raise RuntimeError("fixed-input forward is not bitwise deterministic")
    if not np.isfinite(output_a).all():
        raise RuntimeError("fixed-forward produced NaN or Inf")

    # Verify the exact 32D -> 7D external action conversion used by the
    # inference output path with the fresh canonical stats.
    unnormalized = transforms.Unnormalize(norm_stats)({
        "state": np.asarray(observation.state[0]),
        "actions": output_a[0],
    })
    external = np.asarray(unnormalized["actions"][:, :7], dtype=np.float32)
    if not np.isfinite(external).all() or external.shape != (10, 7):
        raise RuntimeError(f"invalid external fixed-forward actions: {external.shape}")

    out_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_root / "fixed_input.npz",
        base_0_rgb=np.asarray(observation.images["base_0_rgb"]),
        left_wrist_0_rgb=np.asarray(observation.images["left_wrist_0_rgb"]),
        right_wrist_0_rgb=np.asarray(observation.images["right_wrist_0_rgb"]),
        state=np.asarray(observation.state),
        tokenized_prompt=np.asarray(observation.tokenized_prompt),
        tokenized_prompt_mask=np.asarray(observation.tokenized_prompt_mask),
        action_is_pad=np.asarray(observation.action_is_pad),
        noise=noise,
    )
    np.savez_compressed(out_root / "fixed_output.npz", internal_actions=output_a, external_actions=external)
    input_fingerprint = {
        "image": {key: _array_fingerprint(value) for key, value in observation.images.items()},
        "state": _array_fingerprint(observation.state),
        "tokenized_prompt": _array_fingerprint(observation.tokenized_prompt),
        "tokenized_prompt_mask": _array_fingerprint(observation.tokenized_prompt_mask),
        "action_is_pad": _array_fingerprint(observation.action_is_pad),
        "noise": _array_fingerprint(noise),
    }
    output_fingerprint = {
        "internal_actions": _array_fingerprint(output_a),
        "external_actions": _array_fingerprint(external),
    }
    transform_fingerprint = hashlib.sha256(canonical_json({
        "repack": "official v2.0 image/wrist_image/state/actions -> OpenPI LIBERO names",
        "normalize": "fresh canonical norm z-score",
        "resize": "224x224",
        "tokenizer": "PaligemmaTokenizer",
        "pad": "state/action external dims 8/7 -> internal 32",
        "unnormalize": "internal 32 -> external first 7 action dims",
        "norm_sha256": sha256_file(norm_path),
    })).hexdigest()
    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint_kind": "released Axis dense params; no LoRA leaves",
        "axis_params": {"path": str(axis_params), "tree_sha256_before": params_before, "tree_sha256_after": params_after, "unchanged": params_before == params_after},
        "model": {"dtype": "float32", "action_dim_internal": 32, "action_dim_external": 7, "action_horizon": 10, "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m", "missing_leaf_count": len(missing), "unexpected_leaf_count": len(unexpected)},
        "fixed_seed": int(seed),
        "forward_repeated": 2,
        "deterministic_bitwise": True,
        "finite_internal_actions": True,
        "finite_external_actions": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "gpu_label": gpu_label,
        "forward_seconds_first": elapsed_a,
        "forward_seconds_second": elapsed_b,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": output_fingerprint,
        "transform_fingerprint": transform_fingerprint,
        "norm_stats_sha256": sha256_file(norm_path),
        "git": git_info(ROOT),
        "artifacts": {"fixed_input": str(out_root / "fixed_input.npz"), "fixed_output": str(out_root / "fixed_output.npz")},
    }
    write_json(out_root / "fixed_forward_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis-params", type=Path, default=ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1/params")
    parser.add_argument("--norm", type=Path, default=ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/reports/axis_libero_30k/phase1/step0")
    parser.add_argument("--gpu-label", default="1")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    report = run(resolve_path(args.axis_params), resolve_path(args.norm), resolve_path(args.out), gpu_label=args.gpu_label, seed=args.seed)
    print(json.dumps({key: report[key] for key in ("status", "model", "deterministic_bitwise", "finite_internal_actions", "axis_params", "forward_seconds_first")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
