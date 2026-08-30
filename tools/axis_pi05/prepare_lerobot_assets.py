"""Attach LeRobot metadata and R6 fresh QUANTILES stats to a converted Axis model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def _replace_stats(tensors: dict[str, np.ndarray], prefix: str, stats: dict[str, list[float]]) -> None:
    for name, values in stats.items():
        key = f"{prefix}.{name}"
        if key in tensors:
            tensors[key] = np.asarray(values, dtype=np.float32)


def _set_quantiles(value: object) -> None:
    """Update every processor normalization map in a legacy JSON template."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"norm_map", "normalization_mapping"} and isinstance(child, dict):
                for feature_type in ("ACTION", "STATE"):
                    if feature_type in child:
                        child[feature_type] = "QUANTILES"
            _set_quantiles(child)
    elif isinstance(value, list):
        for child in value:
            _set_quantiles(child)


def _configure_r6_preprocessor(processor: dict) -> None:
    """Align the serialized processor with R6's non-discrete-state Pi05 model."""

    steps = processor.get("steps")
    if not isinstance(steps, list):
        raise ValueError("processor config has no steps list")
    steps[:] = [
        step
        for step in steps
        if step.get("registry_name") != "pi05_prepare_state_tokenizer_processor_step"
    ]
    tokenizer_index = next(
        (index for index, step in enumerate(steps) if step.get("registry_name") == "tokenizer_processor"),
        None,
    )
    if tokenizer_index is None:
        raise ValueError("R6 preprocessor template has no tokenizer step")
    if tokenizer_index == 0 or steps[tokenizer_index - 1].get("registry_name") != "smolvla_new_line_processor":
        steps.insert(
            tokenizer_index,
            {"registry_name": "smolvla_new_line_processor", "config": {}},
        )


def prepare(
    model_dir: Path,
    legacy_template: Path,
    norm_path: Path,
    source_checkpoint: str = "R6 step-500 Orbax checkpoint, LoRA folded",
) -> None:
    template = json.loads(legacy_template.read_text(encoding="utf-8"))
    _set_quantiles(template)
    template.update(
        {
            "type": "pi05",
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "dtype": "float32",
            "chunk_size": 10,
            "n_action_steps": 8,
            "max_state_dim": 32,
            "max_action_dim": 32,
            "num_inference_steps": 10,
            "empty_cameras": 1,
            "compile_model": False,
            "gradient_checkpointing": False,
            "push_to_hub": False,
            "repo_id": "axisrobotics/Pi0.5-Axis",
            "pretrained_path": "axisrobotics/Pi0.5-Axis",
            "normalization_mapping": {
                "ACTION": "QUANTILES",
                "STATE": "QUANTILES",
                "VISUAL": "IDENTITY",
            },
        }
    )
    (model_dir / "config.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")

    norm = json.loads(norm_path.read_text(encoding="utf-8"))["norm_stats"]
    pre_name = "policy_preprocessor_step_2_normalizer_processor.safetensors"
    post_name = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        source = legacy_template.parent / name
        processor = json.loads(source.read_text(encoding="utf-8"))
        _set_quantiles(processor)
        if name == "policy_preprocessor.json":
            _configure_r6_preprocessor(processor)
        (model_dir / name).write_text(json.dumps(processor, indent=2) + "\n", encoding="utf-8")

    for name in (pre_name, post_name):
        source = legacy_template.parent / name
        tensors = {key: value.copy() for key, value in load_file(str(source)).items()}
        _replace_stats(tensors, "action", norm["actions"])
        _replace_stats(tensors, "observation.state", norm["state"])
        save_file(tensors, str(model_dir / name), metadata={"source": "axis_r6_fresh_norm_stats", "normalization": "QUANTILES"})

    provenance = {
        "source": str(norm_path),
        "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        "normalization": "QUANTILES",
        "state_dim": 8,
        "action_dim": 7,
        "model_internal_action_dim": 32,
        "source_checkpoint": source_checkpoint,
    }
    (model_dir / "axis_norm_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--legacy-template", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--source-checkpoint", default="R6 step-500 Orbax checkpoint, LoRA folded")
    args = parser.parse_args()
    prepare(args.model_dir, args.legacy_template, args.norm_stats, args.source_checkpoint)
    print(f"prepared {args.model_dir}")


if __name__ == "__main__":
    main()
