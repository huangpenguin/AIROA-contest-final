"""Convert an Axis OpenPI Orbax checkpoint to a LeRobot-compatible model.

Axis's training model keeps LoRA parameters in the JAX tree and uses a 32-wide
internal action head.  The competition's LeRobot server has no Axis LoRA
module, so this converter folds the LoRA delta into the dense JAX weights,
then uses the upstream OpenPI PyTorch conversion routines.  The resulting
model still has a 32-wide internal head; the LeRobot LIBERO policy exposes the
first seven action dimensions.

This is an offline conversion tool.  It never modifies the source checkpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import safetensors.torch
import torch

from flax.nnx import traversals
import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import config as training_config

from .lora_folding import assert_no_lora as _assert_no_lora
from .lora_folding import fold_lora_in_mapping as _fold_lora_in_mapping


_UPSTREAM_CONVERTER = Path(__file__).resolve().parents[2] / "third_party" / "axis_openpi" / "examples" / "convert_jax_model_to_pytorch.py"
_UPSTREAM_SPEC = importlib.util.spec_from_file_location("axis_openpi_upstream_converter", _UPSTREAM_CONVERTER)
if _UPSTREAM_SPEC is None or _UPSTREAM_SPEC.loader is None:
    raise ImportError(f"cannot load upstream converter: {_UPSTREAM_CONVERTER}")
upstream = importlib.util.module_from_spec(_UPSTREAM_SPEC)
_UPSTREAM_SPEC.loader.exec_module(upstream)


def convert(checkpoint_dir: Path, output_dir: Path, *, precision: str = "float32") -> dict[str, Any]:
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"checkpoint has no params directory: {checkpoint_dir / 'params'}")
    if precision != "float32":
        raise ValueError("competition V100 conversion is intentionally limited to float32")

    print(f"restoring {checkpoint_dir}")
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params", restore_type=np.ndarray, dtype="float32"
    )
    folded = _fold_lora_in_mapping(params)
    _assert_no_lora(params)
    print(f"folded_lora_modules={folded}")

    paligemma_params = traversals.flatten_mapping(params["PaliGemma"], sep="/")
    projection_params = params
    model_config = openpi.models.pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
    )

    class PaliGemmaConfig:
        def __init__(self) -> None:
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    projection_keys = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
    converted_projection: dict[str, torch.Tensor] = {}
    for key in projection_keys:
        kernel = projection_params[key]["kernel"]
        bias = projection_params[key]["bias"]
        if isinstance(kernel, dict):
            kernel = kernel["value"]
            bias = bias["value"]
        converted_projection[f"{key}.weight"] = torch.from_numpy(np.asarray(kernel).T)
        converted_projection[f"{key}.bias"] = torch.from_numpy(np.asarray(bias))

    paligemma, expert = upstream.slice_paligemma_state_dict(
        paligemma_params, PaliGemmaConfig()
    )
    # The path contains pi05, so the upstream converter selects adaptive RMSNorm.
    expert = upstream.slice_gemma_state_dict(
        expert,
        openpi.models.gemma.get_config("gemma_300m"),
        num_expert=1,
        checkpoint_dir=str(checkpoint_dir),
        pi05=True,
    )
    all_params = {**paligemma, **expert, **converted_projection}

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    missing, unexpected = model.load_state_dict(all_params, strict=False)
    allowed_missing = {
        "paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_with_expert.gemma_expert.lm_head.weight",
    }
    unresolved_missing = [key for key in missing if key not in allowed_missing]
    if unresolved_missing or unexpected:
        raise RuntimeError(
            f"converted model key mismatch: missing={unresolved_missing[:8]} ({len(unresolved_missing)}), "
            f"unexpected={list(unexpected)[:8]} ({len(unexpected)})"
        )
    model = model.to(torch.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, str(output_dir / "model.safetensors"))

    assets_source = checkpoint_dir / "assets"
    if assets_source.is_dir():
        import shutil

        shutil.copytree(assets_source, output_dir / "assets", dirs_exist_ok=True)
    config = {
        "type": "pi05",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "discrete_state_input": False,
        "dtype": "float32",
        "chunk_size": 10,
        "n_action_steps": 8,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "image_resolution": [224, 224],
        "empty_cameras": 1,
        "normalization_mapping": {"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"},
        "compile_model": False,
        "gradient_checkpointing": False,
        "axis_source_checkpoint": str(checkpoint_dir),
        "axis_lora_folded_modules": folded,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return {
        "output_dir": str(output_dir),
        "folded_lora_modules": folded,
        "allowed_tied_missing": list(missing),
        "missing": len(unresolved_missing),
        "unexpected": len(unexpected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
