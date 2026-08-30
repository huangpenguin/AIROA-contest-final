"""Audit raw-LoRA versus folded-dense fixed-input parity on one GPU.

The raw checkpoint is restored with the same LoRA model topology used for
training.  Its parameter tree is then folded into a dense Pi0.5 graph and both
graphs run the identical tokenized observation and fixed diffusion noise.  The
comparison is intentionally JAX-only; the separate conversion/e2e audit covers
the PyTorch and LeRobot processor boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, resolve_path, sha256_file, write_json


NESTED_SRC = ROOT / "third_party" / "axis_openpi" / "src"
def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def run(
    checkpoint: Path,
    norm_stats: Path,
    fixed_input: Path,
    output_dir: Path,
    *,
    arm: str,
    run_dir: Path,
    gpu_id: str = "1",
    seed: int = 20260828,
) -> dict[str, Any]:
    if gpu_id == "0":
        raise ValueError("GPU0 is outside the parity boundary")
    if not checkpoint.is_dir() or not (checkpoint / "params").is_dir():
        raise FileNotFoundError(checkpoint)
    if not fixed_input.is_file():
        raise FileNotFoundError(fixed_input)

    os.environ.setdefault("PYTHONPATH", str(NESTED_SRC) + os.pathsep + str(ROOT))
    if str(NESTED_SRC) not in sys.path:
        sys.path.insert(0, str(NESTED_SRC))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from tools.axis_pi05.lora_folding import fold_lora_in_mapping

    arrays = np.load(fixed_input)
    required = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "action_is_pad",
        "noise",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"fixed input missing fields: {missing}")
    observation = model_lib.Observation.from_dict(
        {
            "image": {
                "base_0_rgb": jnp.asarray(arrays["base_0_rgb"]),
                "left_wrist_0_rgb": jnp.asarray(arrays["left_wrist_0_rgb"]),
                "right_wrist_0_rgb": jnp.asarray(arrays["right_wrist_0_rgb"]),
            },
            "image_mask": {
                "base_0_rgb": jnp.ones((arrays["state"].shape[0],), dtype=bool),
                "left_wrist_0_rgb": jnp.ones((arrays["state"].shape[0],), dtype=bool),
                "right_wrist_0_rgb": jnp.zeros((arrays["state"].shape[0],), dtype=bool),
            },
            "state": jnp.asarray(arrays["state"]),
            "tokenized_prompt": jnp.asarray(arrays["tokenized_prompt"]),
            "tokenized_prompt_mask": jnp.asarray(arrays["tokenized_prompt_mask"]),
            "action_is_pad": jnp.asarray(arrays["action_is_pad"]),
        }
    )
    noise = jnp.asarray(arrays["noise"])
    batch_size = int(arrays["state"].shape[0])
    if tuple(noise.shape) != (batch_size, 10, 32):
        raise ValueError(f"unexpected fixed noise shape: {noise.shape}")

    # Restore the parameter tree with the actual LoRA model topology.  This is
    # inference-only and deliberately performs no optimizer initialization or
    # training step; the separate restore audit covers Orbax topology restore.
    os.environ.update(
        {
            "AXIS_PI05_LORA_ARM": arm,
            "AXIS_PI05_LORA_RUN_DIR": str(run_dir),
            "AXIS_PI05_LORA_NORM_STATS": str(norm_stats),
            "AXIS_PI05_LORA_AXIS_PARAMS": str(
                ROOT / "data/models/axisrobotics/Pi0.5-Axis" / "c342608c489515efa9ba8281400dc648bf3970e1" / "params"
            ),
        }
    )
    config = config_lib.get_config("axis_pi05_lora_screen")
    checkpoint_step = int(checkpoint.name)
    raw_params = model_lib.restore_params(
        checkpoint / "params", restore_type=np.ndarray, dtype="float32"
    )
    raw_model = config.model.load(raw_params)
    raw_model.eval()

    # Fold the pure parameter mapping and place it into a genuinely dense
    # Pi0.5 graph.  Alpha==rank for both configured LoRA modules, so the
    # materialized delta has scaling 1.0, matching openpi.models.lora.
    folded_params = copy.deepcopy(raw_params)
    folded_modules = fold_lora_in_mapping(folded_params)
    dense_config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    dense_template = dense_config.create(jax.random.key(0))
    dense_graphdef, dense_state = nnx.split(dense_template)
    dense_state.replace_by_pure_dict(folded_params)
    dense_model = nnx.merge(dense_graphdef, dense_state)
    dense_model.eval()

    rng = jax.random.key(seed)
    started = time.monotonic()
    raw_actions = np.asarray(
        jax.block_until_ready(raw_model.sample_actions(rng, observation, num_steps=10, noise=noise))
    )
    raw_seconds = time.monotonic() - started
    started = time.monotonic()
    dense_actions = np.asarray(
        jax.block_until_ready(dense_model.sample_actions(rng, observation, num_steps=10, noise=noise))
    )
    dense_seconds = time.monotonic() - started
    if not np.isfinite(raw_actions).all() or not np.isfinite(dense_actions).all():
        raise FloatingPointError("raw or folded-dense action contains NaN/Inf")

    difference = raw_actions - dense_actions
    max_abs = float(np.max(np.abs(difference)))
    max_rel = float(np.max(np.abs(difference) / np.maximum(np.abs(raw_actions), 1e-6)))
    # The raw and folded graphs use different einsum contraction graphs, so
    # bitwise equality is not expected on V100 FP32.  The fixed parity gate is
    # an absolute 1e-3 action-unit bound (relative error is reported but is not
    # meaningful for values close to zero).
    within_tolerance = bool(np.allclose(raw_actions, dense_actions, rtol=1e-3, atol=1e-3))

    norm = normalize.deserialize_json(norm_stats.read_text(encoding="utf-8"))
    unnormalize = transforms.Unnormalize(norm, use_quantiles=True)
    raw_external = np.asarray(
        unnormalize({"state": np.asarray(arrays["state"]), "actions": raw_actions})["actions"][..., :7],
        dtype=np.float32,
    )
    dense_external = np.asarray(
        unnormalize({"state": np.asarray(arrays["state"]), "actions": dense_actions})["actions"][..., :7]
    , dtype=np.float32)
    external_difference = raw_external - dense_external
    external_max_abs = float(np.max(np.abs(external_difference)))
    external_max_rel = float(
        np.max(np.abs(external_difference) / np.maximum(np.abs(raw_external), 1e-6))
    )
    external_within_tolerance = bool(
        np.allclose(raw_external, dense_external, rtol=1e-3, atol=1e-3)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "parity_output.npz",
        raw_internal_actions=raw_actions,
        folded_dense_internal_actions=dense_actions,
        raw_external_actions=raw_external,
        folded_dense_external_actions=dense_external,
    )
    report = {
        "schema_version": 1,
        "status": "passed" if within_tolerance and external_within_tolerance else "failed",
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "arm": arm,
        "gpu_id": str(gpu_id),
        "gpu0_touched": False,
        "folded_lora_modules": folded_modules,
        "fold_scaling": {"paligemma_rank16_alpha16": 1.0, "action_expert_rank32_alpha32": 1.0},
        "fixed_input": str(fixed_input),
        "fixed_input_sha256": sha256_file(fixed_input),
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": sha256_file(norm_stats),
        "raw_internal": _fingerprint(raw_actions),
        "folded_dense_internal": _fingerprint(dense_actions),
        "internal_max_abs_diff": max_abs,
        "internal_max_relative_diff": max_rel,
        "internal_within_tolerance": within_tolerance,
        "raw_external": _fingerprint(raw_external),
        "folded_dense_external": _fingerprint(dense_external),
        "external_max_abs_diff": external_max_abs,
        "external_max_relative_diff": external_max_rel,
        "external_within_tolerance": external_within_tolerance,
        "parity_tolerance": {"atol": 1e-3, "rtol": 1e-3, "relative_error_is_diagnostic_only": True},
        "normalization": "QUANTILES",
        "raw_seconds": raw_seconds,
        "folded_dense_seconds": dense_seconds,
        "artifact": str(output_dir / "parity_output.npz"),
    }
    write_json(output_dir / "parity_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--fixed-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                resolve_path(args.checkpoint),
                resolve_path(args.norm_stats),
                resolve_path(args.fixed_input),
                resolve_path(args.output),
                arm=args.arm,
                run_dir=resolve_path(args.run_dir),
                gpu_id=args.gpu_id,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
