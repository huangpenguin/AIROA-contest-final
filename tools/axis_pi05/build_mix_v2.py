from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_mix import _hierarchical_mass, _quantile, bucket_key, episode_key
from .common import ROOT, git_info, load_yaml, read_jsonl, resolve_path, sha256_file, write_csv, write_json, write_table
from .lock import load_and_validate


def _group_id(row: dict[str, Any]) -> str:
    # The Plus snapshot exposes no base-trajectory/init-state provenance key.
    # Storage files are not treated as leakage groups; episode identity is the
    # conservative deterministic unit for this release.
    return "episode:" + ":".join(str(value) for value in episode_key(row))


def split_rows_v2(
    rows: list[dict[str, Any]], seed: int, fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_bucket: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[bucket_key(row)].append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    grouping_audit: list[dict[str, Any]] = []
    for key, bucket in sorted(by_bucket.items()):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            groups[_group_id(row)].append(row)
        target = max(1, int(math.ceil(len(bucket) * fraction)))
        ranked = sorted(
            groups.items(),
            key=lambda item: hashlib.sha256(f"{item[0]}:{seed}".encode()).hexdigest(),
        )
        chosen: list[dict[str, Any]] = []
        count = 0
        for _, group_rows in ranked:
            if count >= target and chosen:
                break
            chosen.extend(group_rows)
            count += len(group_rows)
        chosen_ids = {episode_key(row) for row in chosen}
        validation.extend(chosen)
        train.extend(row for row in bucket if episode_key(row) not in chosen_ids)
        grouping_quality = "episode_only" if key[0] == "libero_plus" else "episode_only"
        grouping_audit.append({
            "source": key[0],
            "suite": key[1],
            "task": key[2],
            "category": key[3],
            "grouping_quality": grouping_quality,
            "group_key": "source_repo/revision/source_episode_index",
            "episode_count": len(bucket),
            "group_count": len(groups),
            "validation_episode_count": len(chosen),
            "train_episode_count": len(bucket) - len(chosen),
            "source_file_unique_count": len({str(row.get("source_file")) for row in bucket}),
            "source_file_used_as_group": False,
            "reason": "Plus has no base trajectory/init state provenance; deterministic episode holdout" if key[0] == "libero_plus" else "deterministic episode holdout",
        })
        if len(bucket) < 5 or len(groups) < 2:
            exceptions.append({
                "bucket": key,
                "reason": "no_holdout",
                "episode_count": len(bucket),
                "group_count": len(groups),
            })
    train_ids = {episode_key(row) for row in train}
    validation_ids = {episode_key(row) for row in validation}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation episode overlap")
    if len(train_ids | validation_ids) != len({episode_key(row) for row in rows}):
        raise RuntimeError("split lost or duplicated episode identities")
    return sorted(train, key=episode_key), sorted(validation, key=episode_key), exceptions, grouping_audit


def make_weights_v2(train_rows: list[dict[str, Any]], mix: str) -> list[dict[str, Any]]:
    target = _hierarchical_mass(train_rows, mix)
    groups: Counter[tuple[str, str, str, str]] = Counter(bucket_key(row) for row in train_rows)
    weights: list[dict[str, Any]] = []
    for row in train_rows:
        bucket = bucket_key(row)
        probability = float(target.get(bucket, 0.0)) / groups[bucket]
        observed = int(row["frame_count"])
        if observed <= 0:
            raise ValueError(f"non-positive frame_count for {episode_key(row)}")
        weights.append({
            **row,
            "episode_sampling_probability": probability,
            "sampling_probability": probability,
            "sampling_weight": probability,
            "observed_frame_count": observed,
            "frame_weight": probability / observed,
            "effective_exposures": probability * len(train_rows),
        })
    total = sum(float(row["episode_sampling_probability"]) for row in weights)
    if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-10):
        if total <= 0:
            raise RuntimeError(f"{mix} sampling weights sum to zero")
        for row in weights:
            row["episode_sampling_probability"] /= total
            row["sampling_probability"] = row["episode_sampling_probability"]
            row["sampling_weight"] = row["episode_sampling_probability"]
            row["frame_weight"] = row["episode_sampling_probability"] / row["observed_frame_count"]
            row["effective_exposures"] = row["episode_sampling_probability"] * len(train_rows)
    if not math.isclose(sum(float(row["frame_weight"]) * int(row["observed_frame_count"]) for row in weights), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(f"{mix} frame weights do not conserve episode mass")
    return weights


def build_one(
    catalog_path: Path,
    output_root: Path,
    mix: str,
    seed: int,
    validation_fraction: float,
    lock_path: Path,
    implementation_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(catalog_path)
    train, validation, exceptions, grouping_audit = split_rows_v2(rows, seed, validation_fraction)
    weighted = make_weights_v2(train, mix)
    out = output_root / mix
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "train_episodes.parquet", train)
    write_table(out / "validation_episodes.parquet", validation)
    write_table(out / "sampling_weights.parquet", weighted)
    before = Counter((row["source"], row["suite"], row["perturb_category"]) for row in train)
    after: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in weighted:
        key = (row["source"], row["suite"], row["perturb_category"])
        after[key] += float(row["episode_sampling_probability"])
    distribution = [
        {
            "source": key[0], "suite": key[1], "category": key[2],
            "train_count": before.get(key, 0),
            "sampling_probability": after.get(key, 0.0),
        }
        for key in sorted(set(before) | set(after))
    ]
    write_csv(out / "distribution_before_after.csv", distribution, ["source", "suite", "category", "train_count", "sampling_probability"])
    write_csv(out / "grouping_audit.csv", grouping_audit, [
        "source", "suite", "task", "category", "grouping_quality", "group_key", "episode_count", "group_count",
        "validation_episode_count", "train_episode_count", "source_file_unique_count", "source_file_used_as_group", "reason",
    ])
    write_csv(out / "shortfalls_and_exceptions.csv", exceptions, ["bucket", "reason", "episode_count", "group_count"])
    manifest = {
        "schema_version": 2,
        "mix_id": mix,
        "mix_name": {"M0": "source_suite_balanced", "M1": "competition_prior", "M2": "natural_capped"}[mix],
        "source_revisions": {row["source_repo"]: row["revision"] for row in rows},
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "split_unit": "source_repo/revision/source_episode_index",
        "grouping_policy": {"libero_plus": "episode_only", "libero_standard": "episode_only"},
        "grouping_quality": {"libero_plus": "episode_only", "libero_standard": "episode_only"},
        "plus_base_trajectory_provenance_available": False,
        "plus_validation_episode_count": sum(1 for row in validation if row["source"] == "libero_plus"),
        "plus_validation_required": True,
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "shortfalls_and_exceptions": exceptions,
        "catalog_sha256": sha256_file(catalog_path),
        "implementation_sha256": sha256_file(implementation_path),
        "implementation_commit": git_info()["commit"],
        "sampling_weight_sum": sum(float(row["episode_sampling_probability"]) for row in weighted),
        "frame_weight_mass_sum": sum(float(row["frame_weight"]) * int(row["observed_frame_count"]) for row in weighted),
        "repeat_exposure_quantiles": {
            quantile: _quantile(sorted(float(row["effective_exposures"]) for row in weighted), quantile)
            for quantile in (0.5, 0.95, 0.99)
        },
        "lock_sha256": sha256_file(lock_path),
        "sampler_contract": {
            "episode_sampling_probability_column": "episode_sampling_probability",
            "observed_frame_count_column": "observed_frame_count",
            "frame_weight_column": "frame_weight",
            "frame_weight_formula": "episode_sampling_probability / observed_frame_count",
            "weighted_random_sampler_replacement": True,
            "explicit_generator_seed": seed,
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "manifest.sha256", {"sha256": sha256_file(out / "manifest.json")})
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="build v2 manifest-aware Axis Pi0.5 splits and sampler weights")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/sampling_v2.yaml")
    ap.add_argument("--mix", choices=["M0", "M1", "M2"], action="append")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    root = config_path.parents[2]
    mixes = ["M0", "M1", "M2"] if args.all or not args.mix else args.mix
    lock_path = root / "configs/axis_pi05/data_sources.lock.yaml"
    load_and_validate(lock_path)
    implementation_path = root / "tools/axis_pi05/build_mix_v2.py"
    for mix in mixes:
        result = build_one(
            resolve_path(config["catalog_path"], root),
            resolve_path(config["output_root"], root),
            mix,
            int(config["split_seed"]),
            float(config["validation_fraction"]),
            lock_path,
            implementation_path,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
