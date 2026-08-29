from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_json, read_table, resolve_path, sha256_file, write_json


REQUIRED_COLUMNS = {
    "source_repo", "source_episode_index", "episode_sampling_probability",
    "observed_frame_count", "frame_weight",
}


def _draw_preview(weights: list[float], seed: int, draws: int) -> list[int]:
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(draws),
        replacement=True,
        generator=generator,
    )
    return [int(index) for index in sampler]


def _expand_frame_weights(rows: list[dict[str, Any]]) -> tuple[list[float], list[int]]:
    """Expand episode manifest weights to the frame-level loader contract.

    OpenPI's manifest-aware dataset exposes one sampler item per frame.  The
    episode probability is conserved by assigning ``probability / frame_count``
    to every frame in that episode.  Sampling the episode rows directly is a
    tempting shortcut, but it over-samples short episodes and is not an audit
    of the training loader.
    """

    weights: list[float] = []
    episode_indices: list[int] = []
    for episode_index, row in enumerate(rows):
        observed = int(row["observed_frame_count"])
        if observed <= 0:
            raise ValueError(
                f"observed_frame_count must be positive for row {episode_index}"
            )
        frame_weight = float(row["frame_weight"])
        weights.extend([frame_weight] * observed)
        episode_indices.extend([episode_index] * observed)
    return weights, episode_indices


def _binomial_99_half_width(probability: float, draws: int) -> float:
    if draws <= 0:
        return 1.0
    return 2.576 * math.sqrt(max(0.0, probability * (1.0 - probability)) / draws)


def audit_manifest(path: Path, *, seed: int, draws: int = 100_000, preview: int = 32) -> dict[str, Any]:
    rows = read_table(path)
    if not rows:
        raise ValueError(f"empty sampling manifest: {path}")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"sampling manifest missing columns: {sorted(missing)}")
    probabilities = [float(row["episode_sampling_probability"]) for row in rows]
    frame_weights = [float(row["frame_weight"]) for row in rows]
    if any(not math.isfinite(value) or value <= 0 for value in probabilities + frame_weights):
        raise ValueError("sampling probabilities and frame weights must be positive finite values")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"episode probability mass is not one: {sum(probabilities)}")
    frame_mass = sum(weight * int(row["observed_frame_count"]) for weight, row in zip(frame_weights, rows, strict=True))
    if not math.isclose(frame_mass, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"frame probability mass is not one: {frame_mass}")
    for row, probability, frame_weight in zip(rows, probabilities, frame_weights, strict=True):
        observed = int(row["observed_frame_count"])
        if observed <= 0 or not math.isclose(frame_weight, probability / observed, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"frame weight formula mismatch for {row['source_repo']}:{row['source_episode_index']}")
    expanded_weights, episode_indices = _expand_frame_weights(rows)
    indices = _draw_preview(expanded_weights, seed, draws)
    source_by_episode = [str(row["source_repo"]) for row in rows]
    source_counts = Counter(source_by_episode[episode_indices[index]] for index in indices)
    preview_indices = indices[:preview]
    preview_rows = [
        {
            "frame_index": index,
            "episode_index": episode_indices[index],
            "source_repo": source_by_episode[episode_indices[index]],
            "source_episode_index": int(rows[episode_indices[index]]["source_episode_index"]),
            "frame_weight": expanded_weights[index],
        }
        for index in preview_indices
    ]
    expected_mass = {
        source: sum(
            probability
            for source_name, probability in zip(source_by_episode, probabilities, strict=True)
            if source_name == source
        )
        for source in sorted(set(source_by_episode))
    }
    empirical_mass = {
        source: source_counts.get(source, 0) / max(1, draws)
        for source in sorted(expected_mass)
    }
    empirical_error = {
        source: empirical_mass[source] - expected_mass[source]
        for source in sorted(expected_mass)
    }
    empirical_gate = {
        source: {
            "pass": (
                abs(empirical_error[source]) <= _binomial_99_half_width(expected_mass[source], draws)
                and abs(empirical_error[source]) <= 0.005
            ),
            "absolute_error": abs(empirical_error[source]),
            "99pct_half_width": _binomial_99_half_width(expected_mass[source], draws),
            "max_absolute_error": 0.005,
        }
        for source in sorted(expected_mass)
    }
    return {
        "schema_version": 3,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "episode_count": len(rows),
        "frame_count": len(expanded_weights),
        "episode_probability_sum": sum(probabilities),
        "frame_probability_mass_sum": frame_mass,
        "seed": int(seed),
        "draw_count": int(draws),
        "sampler": "torch.utils.data.WeightedRandomSampler",
        "sampling_unit": "frame",
        "frame_weight_formula": "episode_sampling_probability / observed_frame_count",
        "replacement": True,
        "explicit_generator": True,
        "source_expected_mass": expected_mass,
        "source_empirical_mass": empirical_mass,
        "source_empirical_error": empirical_error,
        "source_empirical_gate": empirical_gate,
        "empirical_gate_pass": all(item["pass"] for item in empirical_gate.values()),
        "sample_index_preview": preview_rows,
        "sample_index_preview_sha256": hashlib.sha256(canonical_json(preview_rows)).hexdigest(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="audit manifest-aware frame sampler conservation and seeded draws")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=100_000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = audit_manifest(resolve_path(args.manifest), seed=args.seed, draws=args.draws)
    out = resolve_path(args.out) if args.out else resolve_path(args.manifest).with_name("sampler_audit.json")
    write_json(out, report)
    print(json.dumps({key: report[key] for key in ("manifest_sha256", "episode_count", "episode_probability_sum", "frame_probability_mass_sum", "sample_index_preview_sha256")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
