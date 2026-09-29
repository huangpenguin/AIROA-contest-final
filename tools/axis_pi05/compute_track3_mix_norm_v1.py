#!/usr/bin/env python3
"""Compute exact weighted QUANTILES for the Track3 multi-root mixture.

The Q2 sampler manifest is the only source of frame weights.  Unlike the
older single-root norm tools, this reader resolves each source repository from
``source_roots.json`` and scans only the selected frame shards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .common import ROOT, canonical_json, read_table, resolve_path, sha256_bytes, sha256_file, write_json
from .compute_norm import WeightedStats, _vector
from .contract import validate_norm_stats


QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


def strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no frame parquet files under {root / 'data'}")
    return files


def _iter_frame_rows(path: Path) -> Iterator[tuple[int, list[float], list[float]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = ["episode_index", "observation.state", "action"]
    for batch in parquet.iter_batches(columns=columns, batch_size=65536):
        values = {name: batch.column(index).to_pylist() for index, name in enumerate(columns)}
        for episode, state, action in zip(
            values["episode_index"], values["observation.state"], values["action"], strict=True
        ):
            yield (
                int(episode),
                _vector(state, f"{path}:observation.state", 8),
                _vector(action, f"{path}:action", 7),
            )


def _finish(stats: WeightedStats) -> dict[str, list[float]]:
    if stats.weight <= 0:
        raise ValueError("weighted stats have no samples")
    mean = [value / stats.weight for value in stats.sum]
    std = [
        math.sqrt(max(0.0, stats.sum_sq[index] / stats.weight - mean[index] ** 2))
        for index in range(stats.dim)
    ]
    quantile_values = {name: [] for name, _ in QUANTILES}
    for index in range(stats.dim):
        ordered = sorted(stats.values[index], key=lambda item: item[0])
        for name, fraction in QUANTILES:
            target = stats.weight * fraction
            running = 0.0
            selected = ordered[-1][0]
            for value, weight in ordered:
                running += weight
                if running >= target:
                    selected = value
                    break
            quantile_values[name].append(float(selected))
    return {
        "mean": mean,
        "std": std,
        **quantile_values,
        "min": stats.minimum,
        "max": stats.maximum,
    }


def _validate_stats(stats: dict[str, dict[str, list[float]]]) -> None:
    for name, values in stats.items():
        for key, _ in QUANTILES:
            _require(len(values[key]) > 0 and all(math.isfinite(value) for value in values[key]), f"{name}.{key} nonfinite")
        for index in range(len(values["q01"])):
            ordered = [values[key][index] for key, _ in QUANTILES]
            _require(ordered == sorted(ordered), f"{name} quantiles not monotone at dimension {index}")
            _require(math.isfinite(values["mean"][index]), f"{name}.mean nonfinite at dimension {index}")
            _require(math.isfinite(values["std"][index]), f"{name}.std nonfinite at dimension {index}")
    action_gripper = stats["actions"]
    _require(abs(action_gripper["min"][-1] + 1.0) <= 1e-6, "action gripper minimum is not -1")
    _require(abs(action_gripper["max"][-1] - 1.0) <= 1e-6, "action gripper maximum is not +1")


def _load_weights(manifest_path: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    rows = read_table(manifest_path)
    weighted: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_repo"]), int(row["source_episode_index"]))
        _require(key not in weighted, f"duplicate sampling identity {key}")
        probability = float(row["episode_sampling_probability"])
        observed = int(row["observed_frame_count"])
        frame_weight = float(row["frame_weight"])
        _require(math.isfinite(probability) and probability > 0, f"invalid probability for {key}")
        _require(observed > 0, f"invalid observed frame count for {key}")
        _require(
            math.isfinite(frame_weight)
            and frame_weight > 0
            and math.isclose(frame_weight, probability / observed, rel_tol=1e-10, abs_tol=1e-12),
            f"invalid frame weight for {key}",
        )
        weighted[key] = {"row": row, "probability": probability, "observed": observed, "frame_weight": frame_weight}
    _require(math.isclose(sum(item["probability"] for item in weighted.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9), "episode mass mismatch")
    _require(math.isclose(sum(item["frame_weight"] * item["observed"] for item in weighted.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9), "frame mass mismatch")
    return weighted, {"rows": rows, "episode_count": len(rows)}


def compute(
    *,
    manifest_path: Path,
    source_roots_path: Path,
    output_root: Path,
    evidence_root: Path,
    state_dim: int = 8,
    action_dim: int = 7,
) -> dict[str, Any]:
    _require(state_dim == 8 and action_dim == 7, "Track3 norm contract is state8/action7")
    weighted, manifest_info = _load_weights(manifest_path)
    q2_manifest_path = manifest_path.with_name("manifest.json")
    _require(q2_manifest_path.is_file(), f"Q2 manifest missing: {q2_manifest_path}")
    q2_manifest_sha256 = sha256_file(q2_manifest_path)
    source_roots = strict_json(source_roots_path)
    source_roots_sha256 = sha256_file(source_roots_path)
    source_map = {
        key: Path(str(value["root"]))
        for key, value in source_roots.items()
        if isinstance(value, dict) and "root" in value
    }
    required_sources = sorted({key[0] for key in weighted})
    _require(set(required_sources).issubset(source_map), f"source roots missing: {sorted(set(required_sources) - set(source_map))}")

    state_stats = WeightedStats(state_dim)
    action_stats = WeightedStats(action_dim)
    observed: Counter[tuple[str, int]] = Counter()
    source_frames: Counter[str] = Counter()
    source_mass: defaultdict[str, float] = defaultdict(float)
    family_mass: defaultdict[str, float] = defaultdict(float)
    for (source_repo, episode), item in weighted.items():
        source_mass[source_repo] += item["probability"]
        family = str(item["row"].get("family", "forward_libero_plus"))
        family_mass[family] += item["probability"]

    for source_repo in required_sources:
        source_root = source_map[source_repo]
        selected = {episode for source, episode in weighted if source == source_repo}
        for data_file in _data_files(source_root):
            for episode, state, action in _iter_frame_rows(data_file):
                key = (source_repo, episode)
                if key not in weighted:
                    continue
                item = weighted[key]
                state_stats.update(state, item["frame_weight"])
                action_stats.update(action, item["frame_weight"])
                observed[key] += 1
                source_frames[source_repo] += 1
        found = {episode for source, episode in observed if source == source_repo}
        _require(found == selected, f"missing selected episodes for {source_repo}: {sorted(selected - found)[:5]}")

    mismatches = {
        key: {"expected": item["observed"], "observed": observed.get(key, 0)}
        for key, item in weighted.items()
        if observed.get(key, 0) != item["observed"]
    }
    _require(not mismatches, f"frame boundary mismatch: {list(mismatches.items())[:5]}")
    norm_stats = {"state": _finish(state_stats), "actions": _finish(action_stats)}
    _validate_stats(norm_stats)
    validation = validate_norm_stats(output_root / "norm_stats.json", state_dim=state_dim, action_dim=action_dim) if (output_root / "norm_stats.json").exists() else None

    reverse_rows_path = manifest_path.with_name("reverse_train_episodes.parquet")
    reverse_rows = read_table(reverse_rows_path)
    reverse_by_key = {
        (str(row["source_repo"]), int(row["source_episode_index"])): row for row in reverse_rows
    }
    reverse_items = [
        (key, item)
        for key, item in weighted.items()
        if str(item["row"].get("mix_source")) == "reverse"
    ]
    _require(len(reverse_by_key) == len(reverse_items), "reverse provenance row count mismatch")
    reverse_raw = {
        str(reverse_by_key[key]["source_episode_id"]): str(reverse_by_key[key]["raw_trajectory_sha256"])
        for key, _ in reverse_items
    }
    materializer_hashes = sorted({str(reverse_by_key[key]["materializer_sha256"]) for key, _ in reverse_items})
    _require(len(materializer_hashes) == 1, f"reverse materializer hashes are not unique: {materializer_hashes}")
    tool_sha256 = sha256_file(Path(__file__))
    output_root.mkdir(parents=True, exist_ok=True)
    final_paths = [
        output_root / "norm_stats.json",
        output_root / "norm_stats.provenance.json",
        output_root / "norm_stats.sha256",
        output_root / "sampler_audit.json",
    ]
    _require(not any(path.exists() for path in final_paths), "Q3 output already exists; refusing overwrite")
    staging = output_root / ".q3_outputs.partial"
    _require(not staging.exists(), f"Q3 staging path already exists: {staging}")
    staging.mkdir()
    try:
        norm_path = staging / "norm_stats.json"
        write_json(norm_path, {"norm_stats": norm_stats})
        norm_sha256 = sha256_file(norm_path)
        sampler_audit = {
            "schema_version": "track3_four_gold_sampler_audit_v1",
            "manifest_sha256": q2_manifest_sha256,
            "sampling_weights_sha256": sha256_file(manifest_path),
            "source_roots_sha256": source_roots_sha256,
            "selected_episode_count": len(weighted),
            "selected_frame_count": sum(observed.values()),
            "expected_frame_count": sum(item["observed"] for item in weighted.values()),
            "source_frames": dict(sorted(source_frames.items())),
            "source_mass": dict(sorted(source_mass.items())),
            "family_mass": dict(sorted(family_mass.items())),
            "episode_probability_sum": sum(item["probability"] for item in weighted.values()),
            "frame_weight_mass_sum": sum(item["frame_weight"] * item["observed"] for item in weighted.values()),
            "stats_weight_sum": state_stats.weight,
            "norm_stats_sha256": norm_sha256,
            "actual_frame_count_matches_manifest": True,
            "validation_frames_included": False,
            "axis_norm_stats_loaded": False,
        }
        sampler_path = staging / "sampler_audit.json"
        write_json(sampler_path, sampler_audit)
        provenance = {
            "schema_version": "track3_four_gold_norm_provenance_v1",
            "mix_id": "four_gold_80_20_v1",
            "manifest_sha256": q2_manifest_sha256,
            "sampling_weights_sha256": sha256_file(manifest_path),
            "source_roots_sha256": source_roots_sha256,
            "forward_revision": str(source_roots["lerobot/libero_plus"]["revision"]),
            "forward_source_repo": "lerobot/libero_plus",
            "forward_train_frame_count": sum(item["observed"] for item in weighted.values() if item["row"].get("mix_source") == "forward"),
            "reverse_frame_count": sum(item["observed"] for item in weighted.values() if item["row"].get("mix_source") == "reverse"),
            "reverse_raw_trajectory_sha256": dict(sorted(reverse_raw.items())),
            "materializer_sha256": materializer_hashes[0],
            "sampler_audit_sha256": sha256_file(sampler_path),
            "norm_stats_sha256": norm_sha256,
            "quantile_method": "deterministic exact weighted sort",
            "effective_distribution": "manifest frame_weight = episode_sampling_probability / observed_frame_count",
            "validation_frames_included": False,
            "axis_norm_stats_loaded": False,
            "implementation_sha256": tool_sha256,
        }
        write_json(staging / "norm_stats.provenance.json", provenance)
        write_json(staging / "norm_stats.sha256", {"sha256": norm_sha256})
        for staged, final in (
            (norm_path, final_paths[0]),
            (staging / "norm_stats.provenance.json", final_paths[1]),
            (staging / "norm_stats.sha256", final_paths[2]),
            (sampler_path, final_paths[3]),
        ):
            os.replace(staged, final)
        shutil.rmtree(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_validation = validate_norm_stats(final_paths[0], state_dim=state_dim, action_dim=action_dim)
    _require(final_validation["ok"], json.dumps(final_validation, ensure_ascii=False))
    evidence_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "track3_four_gold_q3_summary_v1",
        "status": "q3_quantiles_built_ready_for_q4",
        "manifest_path": str(q2_manifest_path.resolve()),
        "manifest_sha256": q2_manifest_sha256,
        "sampling_weights_path": str(manifest_path.resolve()),
        "sampling_weights_sha256": sha256_file(manifest_path),
        "source_roots_path": str(source_roots_path.resolve()),
        "source_roots_sha256": source_roots_sha256,
        "norm_stats_path": str(final_paths[0].resolve()),
        "norm_stats_sha256": sha256_file(final_paths[0]),
        "norm_stats_provenance_path": str(final_paths[1].resolve()),
        "sampler_audit_path": str(final_paths[3].resolve()),
        "selected_episode_count": len(weighted),
        "selected_frame_count": sum(observed.values()),
        "forward_train_frames": sum(item["observed"] for item in weighted.values() if item["row"].get("mix_source") == "forward"),
        "reverse_frames": sum(item["observed"] for item in weighted.values() if item["row"].get("mix_source") == "reverse"),
        "source_mass": dict(sorted(source_mass.items())),
        "family_mass": dict(sorted(family_mass.items())),
        "validation_frames_included": False,
        "validation": final_validation,
        "q4_started": False,
        "q5_started": False,
    }
    write_json(evidence_root / "q3_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="compute exact weighted QUANTILES for Track3 four-gold mixture")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests_track3/four_gold_80_20_v1/sampling_weights.parquet")
    parser.add_argument("--source-roots", type=Path, default=ROOT / "data/manifests_track3/four_gold_80_20_v1/source_roots.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/manifests_track3/four_gold_80_20_v1")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910")
    args = parser.parse_args()
    result = compute(
        manifest_path=resolve_path(args.manifest),
        source_roots_path=resolve_path(args.source_roots),
        output_root=resolve_path(args.output_root),
        evidence_root=resolve_path(args.evidence_root),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
