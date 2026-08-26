from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import ROOT, read_table, resolve_path, sha256_file, sha256_tree, write_json


def audit_loader(mix: str, *, manifest: Path, norm: Path, axis_params: Path, seed: int = 1000) -> dict[str, Any]:
    rows = read_table(manifest)
    sources = sorted({str(row["source_repo"]) for row in rows})
    mass = {
        source: sum(float(row["episode_sampling_probability"]) for row in rows if str(row["source_repo"]) == source)
        for source in sources
    }
    os.environ.update({
        "AXIS_PI05_SCOPE": "F0",
        "AXIS_PI05_MIX": mix,
        "AXIS_PI05_AXIS_PARAMS": str(axis_params),
        "AXIS_PI05_ASSETS_ROOT": str(norm.parent),
        "AXIS_PI05_TRAIN_MANIFEST": str(manifest.with_name("train_episodes.parquet")),
        "AXIS_PI05_SAMPLING_MANIFEST": str(manifest),
        "AXIS_PI05_LOCAL_REPO_IDS": ",".join(sources),
        "AXIS_PI05_LOCAL_REPO_ROOTS": os.pathsep.join(str(ROOT / "data/raw/hf" / source) for source in sources),
        "AXIS_PI05_SOURCE_WEIGHTS": ",".join(str(mass[source]) for source in sources),
        "AXIS_PI05_DATALOADER_WORKERS": "0",
        "AXIS_PI05_GLOBAL_BATCH": "3",
        "AXIS_PI05_STEPS": "10",
        "AXIS_PI05_WANDB_MODE": "disabled",
    })
    import torch
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as config_lib
    import openpi.training.data_loader as data_loader

    config = config_lib.get_config("axis_pi05_libero_mix")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    if not hasattr(dataset, "sample_weights"):
        raise RuntimeError("manifest-aware dataset did not expose frame sample weights")
    actual_sum = float(dataset.sample_weights.sum().item())
    expected_len = sum(int(row["observed_frame_count"]) for row in rows)
    if len(dataset) != expected_len:
        raise ValueError(f"loader length mismatch: {len(dataset)} != {expected_len}")
    if abs(actual_sum - 1.0) > 1e-9:
        raise ValueError(f"loader sample weights do not sum to one: {actual_sum}")
    child_lengths = [len(child) for child in dataset.datasets] if hasattr(dataset, "datasets") else []
    child_mass = {}
    start = 0
    for source, length in zip(sources, child_lengths, strict=True):
        child_mass[source] = float(dataset.sample_weights[start:start + length].sum().item())
        start += length
    generator_a = torch.Generator().manual_seed(seed)
    generator_b = torch.Generator().manual_seed(seed)
    sampler_kwargs = {"num_samples": 128, "replacement": True}
    sample_a = list(torch.utils.data.WeightedRandomSampler(dataset.sample_weights, generator=generator_a, **sampler_kwargs))
    sample_b = list(torch.utils.data.WeightedRandomSampler(dataset.sample_weights, generator=generator_b, **sampler_kwargs))
    if sample_a != sample_b:
        raise RuntimeError("seeded sampler replay is not deterministic")
    # A real sample access verifies that the selected episode/frame path is
    # consumable through the same dataset object used by training.
    first = dataset[0]
    required_keys = {"observation.images.image", "observation.images.wrist_image", "observation.state", "action"}
    missing = sorted(required_keys - set(first))
    if missing:
        raise ValueError(f"loader sample missing contract keys: {missing}")
    preview_sha = hashlib.sha256(json.dumps(sample_a, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 2,
        "mix": mix,
        "manifest_sha256": sha256_file(manifest),
        "norm_sha256": sha256_file(norm),
        "episode_count": len(rows),
        "loader_length": len(dataset),
        "expected_loader_length": expected_len,
        "sample_weight_sum": actual_sum,
        "source_expected_mass": mass,
        "source_loader_mass": child_mass,
        "seed": seed,
        "sampler_replay_deterministic": True,
        "sample_index_preview_sha256": preview_sha,
        "first_sample_keys": sorted(first),
        "first_state_shape": list(first["observation.state"].shape),
        "first_action_shape": list(first["action"].shape),
        "axis_params_sha256": sha256_file(axis_params) if axis_params.is_file() else sha256_tree(axis_params),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="instantiate the real Axis manifest-aware loader and audit weights")
    ap.add_argument("--mix", required=True, choices=["M0", "M1", "M2"])
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--norm", type=Path)
    ap.add_argument("--axis-params", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args()
    manifest = resolve_path(args.manifest or ROOT / "data/manifests_v2" / args.mix / "sampling_weights.parquet")
    norm = resolve_path(args.norm or ROOT / "data/manifests_v2" / args.mix / "norm_stats.json")
    report = audit_loader(args.mix, manifest=manifest, norm=norm, axis_params=resolve_path(args.axis_params), seed=args.seed)
    write_json(resolve_path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
