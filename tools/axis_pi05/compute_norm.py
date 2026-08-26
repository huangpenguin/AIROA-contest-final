from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .common import ROOT, read_table, resolve_path, sha256_file, write_json
from .contract import validate_norm_stats
from .lock import load_and_validate


def _vector(value: Any, name: str, dim: int) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != dim:
        raise ValueError(f"{name}: expected {dim} values, got {value!r}")
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
        # Keeping values makes the weighted quantile exact and deterministic.
        # It is intentionally explicit; for very large jobs the server should
        # replace this with a deterministic external-sort implementation rather
        # than silently switching to unweighted quantiles.
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
        return {
            "mean": mean,
            "std": std,
            "q01": [self.quantile(i, 0.01) for i in range(self.dim)],
            "q50": [self.quantile(i, 0.50) for i in range(self.dim)],
            "q99": [self.quantile(i, 0.99) for i in range(self.dim)],
            "min": self.minimum,
            "max": self.maximum,
        }

    def quantile(self, index: int, fraction: float) -> float:
        values = sorted(self.values[index], key=lambda item: item[0])
        target = self.weight * fraction
        running = 0.0
        for value, weight in values:
            running += weight
            if running >= target:
                return value
        return values[-1][0]


def _data_files(root: Path) -> list[Path]:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 data/**/*.parquet: {root}")
    return files


def _episode_from_filename(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("episode_"):
        return None
    try:
        return int(stem.removeprefix("episode_"))
    except ValueError:
        return None


def _read_frame_rows(path: Path) -> list[dict[str, Any]]:
    """Read only the contract columns from a LeRobot frame shard."""
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return read_table(path)
    return pq.read_table(path, columns=["episode_index", "observation.state", "action"]).to_pylist()


def _episode_index(row: dict[str, Any]) -> int | None:
    value = row.get("episode_index")
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute(mix: str, *, root: Path, manifest_path: Path, lock_path: Path, out: Path, state_dim: int = 8, action_dim: int = 7) -> dict[str, Any]:
    from .common import read_table

    manifest_rows = read_table(manifest_path)
    weighted = {(row["source_repo"], int(row["source_episode_index"])): float(row["sampling_weight"]) for row in manifest_rows}
    expected_frames = {(row["source_repo"], int(row["source_episode_index"])): int(row["frame_count"]) for row in manifest_rows}
    source_repo = {(row["source_repo"], int(row["source_episode_index"])): row["source_repo"] for row in manifest_rows}
    state_stats = WeightedStats(state_dim)
    action_stats = WeightedStats(action_dim)
    frame_counts: dict[int, int] = {}
    for source_repo_name in sorted(set(source_repo.values())):
        org, repo = source_repo_name.split("/", 1)
        source_root = root / org / repo
        selected_episodes = {
            episode for source, episode in weighted if source == source_repo_name
        }
        for file in _data_files(source_root):
            filename_episode = _episode_from_filename(file)
            if filename_episode is not None and filename_episode not in selected_episodes:
                continue
            rows = _read_frame_rows(file)
            for row in rows:
                episode = _episode_index(row)
                key = (source_repo_name, episode) if episode is not None else None
                if key is None or key not in weighted:
                    continue
                state = row.get("observation.state", row.get("state"))
                action = row.get("action", row.get("actions"))
                state_values = _vector(state, f"{file}:observation.state", state_dim)
                action_values = _vector(action, f"{file}:action", action_dim)
                frame_counts[key] = frame_counts.get(key, 0) + 1
                # Per-frame weight is the episode's sampling probability divided
                # by its actual number of frames. This is the effective mixture.
                frame_weight = weighted[key] / max(1, expected_frames[key])
                state_stats.update(state_values, frame_weight)
                action_stats.update(action_values, frame_weight)
    mismatches = {
        episode: {"expected": expected_frames[episode], "observed": count}
        for episode, count in frame_counts.items()
        if count != expected_frames[episode]
    }
    missing = sorted(set(expected_frames) - set(frame_counts))
    if mismatches or missing:
        raise RuntimeError(f"frame boundary mismatch; mismatches={list(mismatches.items())[:5]}, missing={missing[:5]}")
    norm_stats = {
        "state": state_stats.finish(),
        "actions": action_stats.finish(),
    }
    provenance = {
        "schema_version": 1,
        "mix_id": mix,
        "manifest_sha256": sha256_file(manifest_path),
        "lock_sha256": sha256_file(lock_path),
        "source_repositories": sorted(set(source_repo.values())),
        "effective_distribution": "sampling_weight / observed_episode_frame_count",
        "quantile_method": "deterministic exact weighted sort",
        "axis_norm_stats_loaded": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"norm_stats": norm_stats})
    write_json(out.with_name("norm_stats.provenance.json"), provenance)
    validation = validate_norm_stats(out, state_dim=state_dim, action_dim=action_dim)
    if not validation["ok"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False))
    return {"mix": mix, "output": str(out), "sha256": sha256_file(out), "validation": validation}


def main() -> None:
    ap = argparse.ArgumentParser(description="compute weighted norm stats for one effective mixture")
    ap.add_argument("--mix", required=True, choices=["M0", "M1", "M2"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/data.yaml")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[2]
    manifest = args.manifest or (root / "data/manifests" / args.mix / "sampling_weights.parquet")
    lock = root / "configs/axis_pi05/data_sources.lock.yaml"
    axis_stats = resolve_path(config.get("axis_norm_stats", "data/models/axisrobotics/unused"), root)
    # Deliberately do not open axis_stats. Presence is not a license to load it.
    result = compute(
        args.mix,
        root=resolve_path(config["raw_root"], root),
        manifest_path=resolve_path(manifest, root),
        lock_path=lock,
        out=root / "data/manifests" / args.mix / "norm_stats.json",
        state_dim=int(config["expected"]["state_dim"]),
        action_dim=int(config["expected"]["action_dim"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
