#!/usr/bin/env python3
"""Run the local dual-root loader, norm, padding and sampler smoke for Q4."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import ROOT, resolve_path, sha256_file, write_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(*, manifest_root: Path, norm_path: Path, evidence_root: Path, draws: int = 20_000) -> dict[str, Any]:
    from openpi import transforms as _transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import normalize
    from openpi.training import config as training_config
    from openpi.training import data_loader

    sampling_path = manifest_root / "sampling_weights.parquet"
    reverse_path = manifest_root / "reverse_train_episodes.parquet"
    _require(sampling_path.is_file() and norm_path.is_file(), "Q2 sampling manifest or Q3 norm missing")
    _require(sha256_file(norm_path) == json.loads((manifest_root / "norm_stats.sha256").read_text(encoding="utf-8"))["sha256"], "norm hash file mismatch")
    norm_stats = normalize.deserialize_json(norm_path.read_text(encoding="utf-8"))
    model_config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        dtype="float32",
    )
    plus_root = ROOT / "data/raw/hf/lerobot/libero_plus"
    reverse_root = ROOT / "data/processed/axis_pi05_track3_reverse_gold_v1"
    factory = training_config.LeRobotLiberoDataConfig(
        repo_id="local/track3_four_gold_80_20",
        assets=training_config.AssetsConfig(assets_dir=str(norm_path), asset_id="four_gold_80_20_v1"),
        base_config=training_config.DataConfig(prompt_from_task=True),
        v3_keys=True,
        local_repo_ids=("lerobot/libero_plus", "axis-pi05/track3_reverse_gold_v1"),
        local_repo_roots=(str(plus_root), str(reverse_root)),
        local_sampling_weights=(0.8, 0.2),
        local_sampling_manifest=str(sampling_path),
    )
    data_config = factory.create(norm_path.parent, model_config)
    _require(data_config.norm_stats is not None, "Q3 norm was not loaded into DataConfig")
    _require(data_config.use_quantile_norm is True, "Q4 requires QUANTILES normalization")
    # Tokenization is outside the loader/norm contract and can require model
    # assets. Keep the real repack + LIBERO image transform + normalization.
    smoke_config = dataclasses.replace(data_config, model_transforms=_transforms.Group())
    raw_dataset = data_loader.create_torch_dataset(data_config, 10, model_config)
    transformed = data_loader.transform_dataset(raw_dataset, smoke_config)
    _require(len(raw_dataset) == 1_999_965, f"mixed frame count mismatch: {len(raw_dataset)}")
    _require(len(raw_dataset.datasets) == 2, "expected two source datasets")
    forward_len = len(raw_dataset.datasets[0])
    reverse_len = len(raw_dataset.datasets[1])
    _require(forward_len == 1_998_905 and reverse_len == 1_060, "source frame count mismatch")
    _require(len(raw_dataset.frame_weights) == len(raw_dataset), "frame weight length mismatch")
    _require(math.isclose(float(raw_dataset.frame_weights.sum()), 1.0, rel_tol=1e-9, abs_tol=1e-9), "loader frame weight sum mismatch")

    samples: list[dict[str, Any]] = []
    for source, start, length in (("forward", 0, forward_len), ("reverse", forward_len, reverse_len)):
        for local_index in sorted({0, length // 2, length - 1}):
            global_index = start + local_index
            raw = raw_dataset[global_index]
            sample = transformed[global_index]
            _require(np.asarray(raw["observation.state"]).shape == (8,), f"{source} state shape")
            _require(np.asarray(raw["action"]).shape == (10, 7), f"{source} raw action horizon shape")
            _require(np.asarray(raw["action_is_pad"]).shape == (10,), f"{source} action mask shape")
            _require(np.asarray(sample["state"]).shape == (8,), f"{source} transformed state shape")
            _require(np.asarray(sample["actions"]).shape == (10, 7), f"{source} transformed action shape")
            _require(np.asarray(sample["action_is_pad"]).shape == (10,), f"{source} transformed mask shape")
            _require(np.asarray(sample["action_is_pad"], dtype=bool).any() == (local_index == length - 1), f"{source} tail mask semantics")
            _require(np.asarray(sample["action_is_pad"], dtype=bool)[0] is False or not bool(np.asarray(sample["action_is_pad"], dtype=bool)[0]), f"{source} first action unexpectedly padded")
            _require(np.asarray(sample["state"]).dtype.kind == "f" and np.isfinite(sample["state"]).all(), f"{source} normalized state finite")
            _require(np.asarray(sample["actions"]).dtype.kind == "f" and np.isfinite(sample["actions"]).all(), f"{source} normalized action finite")
            for camera in ("base_0_rgb", "left_wrist_0_rgb"):
                image = np.asarray(sample["image"][camera])
                _require(image.shape == (256, 256, 3) and image.dtype == np.uint8, f"{source} image contract: {camera}")
            _require("prompt" in sample and str(sample["prompt"]).strip(), f"{source} prompt missing")
            samples.append({"source": source, "local_index": local_index, "global_index": global_index, "task_index": int(raw["task_index"]), "prompt": str(sample["prompt"])})

    generator = torch.Generator().manual_seed(1000)
    sampled_indices = list(torch.utils.data.WeightedRandomSampler(raw_dataset.sample_weights, draws, replacement=True, generator=generator))
    sampled_forward = sum(int(index) < forward_len for index in sampled_indices)
    sampled_reverse = draws - sampled_forward
    observed_mass = {"forward": sampled_forward / draws, "reverse": sampled_reverse / draws}
    _require(sampled_forward > 0 and sampled_reverse > 0, "fixed sampler did not exercise both sources")
    for source_dataset in raw_dataset.datasets:
        underlying = getattr(source_dataset, "_dataset", None)
        if underlying is not None and hasattr(underlying, "close"):
            underlying.close()
    result = {
        "schema_version": "track3_four_gold_q4_summary_v1",
        "status": "track3_four_gold_80_20_quantiles_ready_for_local_gpu3_single_canary",
        "sampling_manifest": str(sampling_path.resolve()),
        "sampling_manifest_sha256": sha256_file(sampling_path),
        "norm_stats": str(norm_path.resolve()),
        "norm_stats_sha256": sha256_file(norm_path),
        "local_repo_ids": list(data_config.local_repo_ids),
        "local_repo_roots": [str(Path(root).resolve()) for root in data_config.local_repo_roots],
        "source_frame_counts": {"lerobot/libero_plus": forward_len, "axis-pi05/track3_reverse_gold_v1": reverse_len},
        "mixed_frame_count": len(raw_dataset),
        "action_horizon": 10,
        "state_dim": 8,
        "action_dim": 7,
        "normalization": "QUANTILES",
        "padding_mask": "action_is_pad; repeated terminal action is masked",
        "exact_source_mass": {"forward": 0.8, "reverse": 0.2},
        "sampler_smoke": {"seed": 1000, "draws": draws, "forward_draws": sampled_forward, "reverse_draws": sampled_reverse, "observed_mass": observed_mass},
        "sample_probes": samples,
        "q5_started": False,
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    write_json(evidence_root / "q4_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="run Track3 four-gold Q4 dual-root loader/norm smoke")
    parser.add_argument("--manifest-root", type=Path, default=ROOT / "data/manifests_track3/four_gold_80_20_v1")
    parser.add_argument("--norm", type=Path, default=ROOT / "data/manifests_track3/four_gold_80_20_v1/norm_stats.json")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910")
    parser.add_argument("--draws", type=int, default=20_000)
    args = parser.parse_args()
    print(json.dumps(run(manifest_root=resolve_path(args.manifest_root), norm_path=resolve_path(args.norm), evidence_root=resolve_path(args.evidence_root), draws=args.draws), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
