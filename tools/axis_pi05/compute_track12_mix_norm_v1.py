#!/usr/bin/env python3
"""Exact weighted QUANTILES for a Track12 mix. No Track3 reverse files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


QUANTILES = (("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _vector(value: Any, name: str, dim: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != dim:
        raise ValueError(f"{name}: expected {dim} values, got {type(value)}")
    output = [float(item) for item in value]
    if not all(math.isfinite(item) for item in output):
        raise ValueError(f"{name}: NaN/Inf detected")
    return output


class WeightedStats:
    def __init__(self, dim: int):
        self.dim = dim
        self.weight = 0.0
        self.sum = [0.0] * dim
        self.sum_sq = [0.0] * dim
        self.minimum = [float("inf")] * dim
        self.maximum = [float("-inf")] * dim
        self.values: list[list[tuple[float, float]]] = [[] for _ in range(dim)]

    def update(self, values: list[float], weight: float) -> None:
        if weight <= 0 or not math.isfinite(weight):
            raise ValueError(f"invalid sample weight {weight}")
        self.weight += weight
        for index, value in enumerate(values):
            self.sum[index] += weight * value
            self.sum_sq[index] += weight * value * value
            self.minimum[index] = min(self.minimum[index], value)
            self.maximum[index] = max(self.maximum[index], value)
            self.values[index].append((value, weight))

    def finish(self) -> dict[str, list[float]]:
        if self.weight <= 0:
            raise ValueError("weighted stats have no samples")
        mean = [value / self.weight for value in self.sum]
        std = [math.sqrt(max(0.0, self.sum_sq[i] / self.weight - mean[i] ** 2)) for i in range(self.dim)]
        quantile_values = {name: [] for name, _ in QUANTILES}
        for index in range(self.dim):
            ordered = sorted(self.values[index], key=lambda item: item[0])
            for name, fraction in QUANTILES:
                target = self.weight * fraction
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
            "min": self.minimum,
            "max": self.maximum,
        }


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


def _load_weights(manifest_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    import pandas as pd

    rows = pd.read_parquet(manifest_path).to_dict(orient="records")
    weighted: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["source_repo"]), int(row["source_episode_index"]))
        if key in weighted:
            raise ValueError(f"duplicate sampling identity {key}")
        probability = float(row["episode_sampling_probability"])
        observed = int(row["observed_frame_count"])
        frame_weight = float(row["frame_weight"])
        if not (math.isfinite(probability) and probability > 0):
            raise ValueError(f"invalid probability for {key}")
        if observed <= 0:
            raise ValueError(f"invalid observed frame count for {key}")
        if not (
            math.isfinite(frame_weight)
            and frame_weight > 0
            and math.isclose(frame_weight, probability / observed, rel_tol=1e-10, abs_tol=1e-12)
        ):
            raise ValueError(f"invalid frame weight for {key}")
        weighted[key] = {"row": row, "probability": probability, "observed": observed, "frame_weight": frame_weight}
    if not math.isclose(sum(item["probability"] for item in weighted.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("episode mass mismatch")
    if not math.isclose(
        sum(item["frame_weight"] * item["observed"] for item in weighted.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("frame mass mismatch")
    return weighted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-roots", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--mix-id", default=None)
    args = parser.parse_args()
    mix_id = args.mix_id or args.manifest.parent.name

    weighted = _load_weights(args.manifest)
    source_roots = json.loads(args.source_roots.read_text(encoding="utf-8"))
    source_map = {
        key: Path(str(value["root"]))
        for key, value in source_roots.items()
        if isinstance(value, dict) and "root" in value
    }
    required_sources = sorted({key[0] for key in weighted})
    missing = sorted(set(required_sources) - set(source_map))
    if missing:
        raise SystemExit(f"source roots missing: {missing}")

    state_stats = WeightedStats(8)
    action_stats = WeightedStats(7)
    observed: Counter[tuple[str, int]] = Counter()
    source_frames: Counter[str] = Counter()
    source_mass: defaultdict[str, float] = defaultdict(float)
    for (source_repo, _episode), item in weighted.items():
        source_mass[source_repo] += item["probability"]

    for source_repo in required_sources:
        source_root = source_map[source_repo]
        selected = {episode for source, episode in weighted if source == source_repo}
        print(f"scan {source_repo} n_ep={len(selected)} root={source_root}", flush=True)
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
        missing_eps = sorted(selected - found)[:5]
        if found != selected:
            raise SystemExit(f"missing selected episodes for {source_repo}: {missing_eps}")

    mismatches = {
        key: {"expected": item["observed"], "observed": observed.get(key, 0)}
        for key, item in weighted.items()
        if observed.get(key, 0) != item["observed"]
    }
    if mismatches:
        raise SystemExit(f"frame boundary mismatch: {list(mismatches.items())[:5]}")

    for name, stats in (("state", state_stats), ("actions", action_stats)):
        finished = stats.finish()
        for qname, _ in QUANTILES:
            if not all(math.isfinite(v) for v in finished[qname]):
                raise SystemExit(f"{name}.{qname} nonfinite")
        for index in range(stats.dim):
            ordered = [finished[qname][index] for qname, _ in QUANTILES]
            if ordered != sorted(ordered):
                raise SystemExit(f"{name} quantiles not monotone at dimension {index}")

    action_finished = action_stats.finish()
    if abs(action_finished["min"][-1] + 1.0) > 1e-6:
        raise SystemExit("action gripper minimum is not -1")
    if abs(action_finished["max"][-1] - 1.0) > 1e-6:
        raise SystemExit("action gripper maximum is not +1")

    norm_stats = {"state": state_stats.finish(), "actions": action_finished}
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    norm_path = args.output_root / "norm_stats.json"
    if norm_path.exists():
        raise SystemExit(f"refusing overwrite: {norm_path}")
    write_json(norm_path, {"norm_stats": norm_stats})
    norm_sha = sha256_file(norm_path)
    write_json(args.output_root / "norm_stats.sha256", {"sha256": norm_sha})
    summary = {
        "status": "t12_quantiles_ready",
        "mix_id": mix_id,
        "selected_episode_count": len(weighted),
        "selected_frame_count": sum(observed.values()),
        "source_mass": dict(sorted(source_mass.items())),
        "source_frames": dict(sorted(source_frames.items())),
        "norm_stats_path": str(norm_path),
        "norm_stats_sha256": norm_sha,
        "manifest_sha256": sha256_file(args.manifest),
        "source_roots_sha256": sha256_file(args.source_roots),
        "validation_frames_included": False,
    }
    summary_name = mix_id.lower().replace("-", "_") + "_quantiles_summary.json"
    write_json(args.evidence_root / summary_name, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
