from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, read_table, resolve_path, sha256_file, write_json
from .compute_norm import (
    WeightedStats,
    _data_files,
    _episode_index,
    _read_frame_rows,
    _vector,
)
from .contract import validate_norm_stats
from .lock import load_and_validate


def compute(
    mix: str,
    *,
    root: Path,
    manifest_path: Path,
    lock_path: Path,
    out: Path,
    state_dim: int = 8,
    action_dim: int = 7,
    sampler_audit_path: Path | None = None,
) -> dict[str, Any]:
    manifest_rows = read_table(manifest_path)
    weighted = {}
    for row in manifest_rows:
        key = (str(row["source_repo"]), int(row["source_episode_index"]))
        probability = float(row["episode_sampling_probability"])
        observed = int(row["observed_frame_count"])
        frame_weight = float(row["frame_weight"])
        if observed <= 0 or abs(frame_weight - probability / observed) > max(1e-12, abs(frame_weight) * 1e-10):
            raise ValueError(f"invalid v2 frame weight for {key}")
        weighted[key] = (frame_weight, observed)
    if abs(sum(float(row["episode_sampling_probability"]) for row in manifest_rows) - 1.0) > 1e-8:
        raise ValueError("v2 episode probabilities do not sum to one")
    state_stats = WeightedStats(state_dim)
    action_stats = WeightedStats(action_dim)
    frame_counts: dict[tuple[str, int], int] = {}
    for source_repo_name in sorted({key[0] for key in weighted}):
        org, repo = source_repo_name.split("/", 1)
        source_root = root / org / repo
        selected_episodes = {episode for source, episode in weighted if source == source_repo_name}
        for file in _data_files(source_root):
            rows = _read_frame_rows(file)
            for row in rows:
                episode = _episode_index(row)
                key = (source_repo_name, episode) if episode is not None else None
                if key is None or key not in weighted:
                    continue
                state_values = _vector(row.get("observation.state", row.get("state")), f"{file}:observation.state", state_dim)
                action_values = _vector(row.get("action", row.get("actions")), f"{file}:action", action_dim)
                frame_counts[key] = frame_counts.get(key, 0) + 1
                state_stats.update(state_values, weighted[key][0])
                action_stats.update(action_values, weighted[key][0])
        if selected_episodes != {episode for source, episode in frame_counts if source == source_repo_name}:
            missing = sorted(selected_episodes - {episode for source, episode in frame_counts if source == source_repo_name})
            raise RuntimeError(f"missing selected episodes for {source_repo_name}: {missing[:5]}")
    mismatches = {
        key: {"expected": observed, "observed": frame_counts.get(key, 0)}
        for key, (_, observed) in weighted.items()
        if frame_counts.get(key, 0) != observed
    }
    if mismatches:
        raise RuntimeError(f"frame boundary mismatch: {list(mismatches.items())[:5]}")
    norm_stats = {"state": state_stats.finish(), "actions": action_stats.finish()}
    provenance = {
        "schema_version": 2,
        "mix_id": mix,
        "manifest_sha256": sha256_file(manifest_path),
        "lock_sha256": sha256_file(lock_path),
        "sampler_audit_sha256": sha256_file(sampler_audit_path) if sampler_audit_path and sampler_audit_path.exists() else None,
        "source_repositories": sorted({key[0] for key in weighted}),
        "effective_distribution": "manifest frame_weight = episode_sampling_probability / observed_frame_count",
        "observed_frame_count_verified": True,
        "quantile_method": "deterministic exact weighted sort",
        "axis_norm_stats_loaded": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"norm_stats": norm_stats})
    write_json(out.with_name("norm_stats.provenance.json"), provenance)
    validation = validate_norm_stats(out, state_dim=state_dim, action_dim=action_dim)
    if not validation["ok"]:
        raise RuntimeError(json.dumps(validation, ensure_ascii=False))
    return {"mix": mix, "output": str(out), "sha256": sha256_file(out), "validation": validation, "provenance": provenance}


def main() -> None:
    ap = argparse.ArgumentParser(description="compute v2 manifest-weighted norm stats")
    ap.add_argument("--mix", required=True, choices=["M0", "M1", "M2"])
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/data.yaml")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sampler-audit", type=Path)
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[2]
    manifest = args.manifest or (root / "data/manifests_v2" / args.mix / "sampling_weights.parquet")
    lock = root / "configs/axis_pi05/data_sources.lock.yaml"
    load_and_validate(lock)
    out = args.out or root / "data/manifests_v2" / args.mix / "norm_stats.json"
    result = compute(
        args.mix,
        root=resolve_path(config["raw_root"], root),
        manifest_path=resolve_path(manifest, root),
        lock_path=lock,
        out=resolve_path(out, root),
        state_dim=int(config["expected"]["state_dim"]),
        action_dim=int(config["expected"]["action_dim"]),
        sampler_audit_path=resolve_path(args.sampler_audit, root) if args.sampler_audit else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
