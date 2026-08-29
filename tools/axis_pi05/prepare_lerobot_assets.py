"""Attach LeRobot metadata and M1 normalizer stats to a converted Axis model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def _replace_stats(tensors: dict[str, np.ndarray], prefix: str, stats: dict[str, list[float]]) -> None:
    for name, values in stats.items():
        key = f"{prefix}.{name}"
        if key in tensors:
            tensors[key] = np.asarray(values, dtype=np.float32)


def prepare(
    model_dir: Path,
    legacy_template: Path,
    norm_path: Path,
    source_checkpoint: str = "Axis F2/M1 500-step Orbax checkpoint, LoRA folded",
) -> None:
    template = json.loads(legacy_template.read_text(encoding="utf-8"))
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
                "ACTION": "MEAN_STD",
                "STATE": "MEAN_STD",
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
        (model_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    for name in (pre_name, post_name):
        source = legacy_template.parent / name
        tensors = {key: value.copy() for key, value in load_file(str(source)).items()}
        _replace_stats(tensors, "action", norm["actions"])
        _replace_stats(tensors, "observation.state", norm["state"])
        save_file(tensors, str(model_dir / name), metadata={"source": "axis_m1_norm_stats"})

    provenance = {
        "source": "data/manifests_v2/M1/norm_stats.json",
        "normalization": "MEAN_STD",
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
    parser.add_argument("--source-checkpoint", default="Axis F2/M1 500-step Orbax checkpoint, LoRA folded")
    args = parser.parse_args()
    prepare(args.model_dir, args.legacy_template, args.norm_stats, args.source_checkpoint)
    print(f"prepared {args.model_dir}")


if __name__ == "__main__":
    main()
