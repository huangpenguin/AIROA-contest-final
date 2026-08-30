"""Audit OpenPI/JAX -> folded PyTorch -> LeRobot processor action parity.

The fixture is intentionally the same fixed observation/noise used by the raw
LoRA versus folded-dense audit.  The LeRobot preprocessor must recreate the
normalized state and prompt consumed by OpenPI, and the converted PyTorch
model must agree with the folded JAX model before the LeRobot postprocessor
externalizes the first seven action dimensions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, resolve_path, sha256_file, write_json


LEROBOT_SRC = Path(
    os.environ.get(
        "AXIS_LEROBOT_SRC",
        str(ROOT.parent / "AIROA-contest-libero-plus" / "workspace" / "training" / "lerobot" / "src"),
    )
)
NESTED_SRC = ROOT / "third_party" / "axis_openpi" / "src"


def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _max_abs(left: Any, right: Any) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def run(
    model_dir: Path,
    fixed_input: Path,
    norm_stats: Path,
    output_dir: Path,
    *,
    task: str,
    gpu_id: str = "1",
    seed: int = 0,
) -> dict[str, Any]:
    if gpu_id == "0":
        raise ValueError("GPU0 is outside the parity boundary")
    if not model_dir.is_dir() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"folded model is missing: {model_dir}")
    if not fixed_input.is_file():
        raise FileNotFoundError(fixed_input)
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)

    # Set this before importing JAX or torch.  The logical cuda:0 below is
    # physical GPU<gpu_id>; physical GPU0 is therefore not visible.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PALIGEMMA_TOKENIZER_DIR", str(
        ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging/vendor/paligemma_tokenizer"
    ))
    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(1, str(NESTED_SRC))
    sys.path.insert(2, str(ROOT))

    import sentencepiece  # noqa: F401  # preload before the vendored tokenizer path is added
    import torch
    import safetensors.torch

    from lerobot.processor import DataProcessorPipeline

    arrays = np.load(fixed_input)
    required = {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "noise",
    }
    missing = sorted(required.difference(arrays.files))
    if missing:
        raise ValueError(f"fixed input missing fields: {missing}")
    batch_size = int(arrays["state"].shape[0])
    if tuple(arrays["noise"].shape) != (batch_size, 10, 32):
        raise ValueError(f"unexpected fixed noise shape: {arrays['noise'].shape}")

    with norm_stats.open(encoding="utf-8") as handle:
        norm = json.load(handle)["norm_stats"]
    q01_state = np.asarray(norm["state"]["q01"], dtype=np.float32)
    q99_state = np.asarray(norm["state"]["q99"], dtype=np.float32)
    q01_action = np.asarray(norm["actions"]["q01"], dtype=np.float32)
    q99_action = np.asarray(norm["actions"]["q99"], dtype=np.float32)
    normalized_state = np.asarray(arrays["state"][:, :8], dtype=np.float32)
    raw_state = (normalized_state + 1.0) * (q99_state - q01_state) / 2.0 + q01_state

    # Feed the same normalized [-1, 1] image convention as OpenPI's fixed
    # observation.  Visual normalization is IDENTITY in the R6 artifact.
    preprocessor_input = {
        "observation.images.image": torch.from_numpy(np.asarray(arrays["base_0_rgb"]).transpose(0, 3, 1, 2)),
        "observation.images.image2": torch.from_numpy(np.asarray(arrays["left_wrist_0_rgb"]).transpose(0, 3, 1, 2)),
        "observation.images.empty_camera_0": torch.from_numpy(np.asarray(arrays["right_wrist_0_rgb"]).transpose(0, 3, 1, 2)),
        "observation.state": torch.from_numpy(raw_state),
        "task": [task] * batch_size,
    }
    preprocessor = DataProcessorPipeline.from_pretrained(
        model_dir, config_filename="policy_preprocessor.json", local_files_only=True
    )
    processed = preprocessor(preprocessor_input)
    processed_images = {
        key: processed[key].detach().cpu()
        for key in (
            "observation.images.image",
            "observation.images.image2",
            "observation.images.empty_camera_0",
        )
    }
    processed_state = processed["observation.state"].detach().cpu().numpy()
    processed_tokens_tensor = processed["observation.language.tokens"].detach().cpu()
    processed_token_mask_tensor = processed["observation.language.attention_mask"].detach().cpu()
    processed_tokens = processed["observation.language.tokens"].detach().cpu().numpy().astype(np.int32)
    processed_token_mask = processed["observation.language.attention_mask"].detach().cpu().numpy().astype(bool)
    fixed_tokens = np.asarray(arrays["tokenized_prompt"], dtype=np.int32)
    fixed_token_mask = np.asarray(arrays["tokenized_prompt_mask"], dtype=bool)
    state_diff = _max_abs(processed_state, normalized_state)
    token_diff = int(np.max(np.abs(processed_tokens.astype(np.int64) - fixed_tokens.astype(np.int64))))
    token_mask_diff = int(np.count_nonzero(processed_token_mask != fixed_token_mask))
    if processed_tokens.shape != fixed_tokens.shape:
        raise ValueError(f"processor token shape mismatch: {processed_tokens.shape} != {fixed_tokens.shape}")
    # The processor's configured device is CUDA.  Keep only CPU snapshots so
    # its normalization state and batch tensors do not overlap the model peak.
    del processed, preprocessor, preprocessor_input
    gc.collect()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("physical GPU for e2e parity is unavailable")
    model_config = __import__("openpi.models.pi0_config", fromlist=["Pi0Config"]).Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        pytorch_compile_mode=None,
    )
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    pytorch_model = PI0Pytorch(model_config).to(device).eval()
    safetensors.torch.load_model(pytorch_model, str(model_dir / "model.safetensors"))
    pt_observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(
        {
            "image": {
                "base_0_rgb": processed_images["observation.images.image"].to(device),
                "left_wrist_0_rgb": processed_images["observation.images.image2"].to(device),
                "right_wrist_0_rgb": processed_images["observation.images.empty_camera_0"].to(device),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
                "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=device),
            },
            "state": F_torch_pad(processed_state, device=device),
            "tokenized_prompt": processed_tokens_tensor.to(device=device, dtype=torch.int32),
            "tokenized_prompt_mask": processed_token_mask_tensor.to(device=device),
        }
    )
    noise_torch = torch.from_numpy(np.asarray(arrays["noise"], dtype=np.float32)).to(device)
    with torch.inference_mode():
        pytorch_actions = pytorch_model.sample_actions(
            device, pt_observation, noise=noise_torch, num_steps=10
        )
    pytorch_actions_np = pytorch_actions.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(pytorch_actions_np).all():
        raise FloatingPointError("folded PyTorch action contains NaN/Inf")
    del pytorch_model, pt_observation, model_config, noise_torch
    gc.collect()
    torch.cuda.empty_cache()

    # Build the folded JAX model from the same R6 raw parameter tree used by
    # the fixed raw/folded audit.  This is the OpenPI/JAX reference boundary.
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from tools.axis_pi05.lora_folding import fold_lora_in_mapping

    os.environ.update(
        {
            "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
            "AXIS_PI05_LORA_RUN_DIR": str(
                ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
            ),
            "AXIS_PI05_LORA_NORM_STATS": str(norm_stats),
            "AXIS_PI05_LORA_AXIS_PARAMS": str(
                ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1/params"
            ),
        }
    )
    training_cfg = config_lib.get_config("axis_pi05_lora_screen")
    raw_params = model_lib.restore_params(
        ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params",
        restore_type=np.ndarray,
        dtype="float32",
    )
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
    jax_observation = model_lib.Observation.from_dict(
        {
            "image": {
                "base_0_rgb": jnp.asarray(processed_images["observation.images.image"].numpy().transpose(0, 2, 3, 1)),
                "left_wrist_0_rgb": jnp.asarray(processed_images["observation.images.image2"].numpy().transpose(0, 2, 3, 1)),
                "right_wrist_0_rgb": jnp.asarray(processed_images["observation.images.empty_camera_0"].numpy().transpose(0, 2, 3, 1)),
            },
            "image_mask": {
                "base_0_rgb": jnp.ones((batch_size,), dtype=bool),
                "left_wrist_0_rgb": jnp.ones((batch_size,), dtype=bool),
                "right_wrist_0_rgb": jnp.zeros((batch_size,), dtype=bool),
            },
            "state": jnp.asarray(np.asarray(processed_state, dtype=np.float32).copy().tolist()),
            "tokenized_prompt": jnp.asarray(processed_tokens),
            "tokenized_prompt_mask": jnp.asarray(processed_token_mask),
        }
    )
    jax_actions = np.asarray(
        jax.block_until_ready(
            dense_model.sample_actions(
                jax.random.key(seed),
                jax_observation,
                num_steps=10,
                noise=jnp.asarray(arrays["noise"], dtype=jnp.float32),
            )
        ),
        dtype=np.float32,
    )
    if not np.isfinite(jax_actions).all():
        raise FloatingPointError("folded JAX action contains NaN/Inf")

    postprocessor = DataProcessorPipeline.from_pretrained(
        model_dir, config_filename="policy_postprocessor.json", local_files_only=True
    )
    postprocessed = postprocessor({"action": pytorch_actions[..., :7].detach().cpu()})["action"]
    postprocessed_np = postprocessed.detach().cpu().numpy().astype(np.float32)
    direct_external = (
        (pytorch_actions_np[..., :7] + 1.0)
        * (q99_action - q01_action)
        / 2.0
        + q01_action
    ).astype(np.float32)
    jax_external = (
        (jax_actions[..., :7] + 1.0) * (q99_action - q01_action) / 2.0 + q01_action
    ).astype(np.float32)
    pt_jax_diff = _max_abs(pytorch_actions_np, jax_actions)
    post_direct_diff = _max_abs(postprocessed_np, direct_external)
    post_jax_diff = _max_abs(postprocessed_np, jax_external)
    tolerance = {"atol": 1e-3, "rtol": 1e-3, "relative_error_is_diagnostic_only": True}
    model_within = bool(np.allclose(pytorch_actions_np, jax_actions, **{k: tolerance[k] for k in ("atol", "rtol")}))
    processor_within = state_diff <= 1e-6 and token_diff == 0 and token_mask_diff == 0
    post_within = bool(np.allclose(postprocessed_np, jax_external, **{k: tolerance[k] for k in ("atol", "rtol")}))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "e2e_action_parity.npz",
        processor_state=processed_state,
        fixed_state=normalized_state,
        processor_tokens=processed_tokens,
        fixed_tokens=fixed_tokens,
        folded_jax_actions=jax_actions,
        folded_pytorch_actions=pytorch_actions_np,
        lerobot_postprocessed_actions=postprocessed_np,
        folded_jax_external_actions=jax_external,
    )
    report = {
        "schema_version": 1,
        "status": "passed" if processor_within and model_within and post_within else "failed",
        "model_dir": str(model_dir),
        "model_sha256": sha256_file(model_dir / "model.safetensors"),
        "fixed_input": str(fixed_input),
        "fixed_input_sha256": sha256_file(fixed_input),
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": sha256_file(norm_stats),
        "task": task,
        "gpu_id": str(gpu_id),
        "gpu0_touched": False,
        "folded_lora_modules": folded_modules,
        "normalization": "QUANTILES",
        "discrete_state_input": False,
        "processor_state_max_abs_diff": state_diff,
        "processor_token_max_abs_diff": token_diff,
        "processor_token_mask_mismatch_count": token_mask_diff,
        "processor_within_tolerance": processor_within,
        "folded_jax": _fingerprint(jax_actions),
        "folded_pytorch": _fingerprint(pytorch_actions_np),
        "folded_pytorch_vs_jax_max_abs_diff": pt_jax_diff,
        "folded_pytorch_vs_jax_within_tolerance": model_within,
        "lerobot_postprocessed": _fingerprint(postprocessed_np),
        "lerobot_postprocessor_vs_direct_quantiles_max_abs_diff": post_direct_diff,
        "lerobot_postprocessor_vs_folded_jax_external_max_abs_diff": post_jax_diff,
        "lerobot_postprocessor_within_tolerance": post_within,
        "parity_tolerance": tolerance,
        "artifact": str(output_dir / "e2e_action_parity.npz"),
    }
    write_json(output_dir / "e2e_action_parity_report.json", report)
    return report


def F_torch_pad(state: np.ndarray, *, device: Any) -> Any:
    """Pad the eight-dimensional LeRobot state to Pi05's 32-wide contract."""

    import torch

    return torch.nn.functional.pad(torch.from_numpy(np.asarray(state, dtype=np.float32)), (0, 24)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixed-input", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(
        resolve_path(args.model_dir),
        resolve_path(args.fixed_input),
        resolve_path(args.norm_stats),
        resolve_path(args.output),
        task=args.task,
        gpu_id=args.gpu_id,
        seed=args.seed,
    ), indent=2))


if __name__ == "__main__":
    main()
