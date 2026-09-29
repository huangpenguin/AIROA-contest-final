#!/usr/bin/env python3
"""Build Track1 T1-P20 and Track2 T2-P30 episode-balanced 20Hz mixes.

These mixes replace the old M0/M1/M2 50% Plus recipes.  Split is
source_repo/revision/source_episode_index, then suite x task x category
episode mass is assigned and converted to frame_weight = p / frame_count.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_mix import bucket_key, episode_key
from .build_mix_v2 import split_rows_v2
from .common import ROOT, git_info, read_jsonl, sha256_file, write_csv, write_json, write_table


CATALOG = ROOT / "data/processed/libero_axis_mix_v1/catalog.jsonl"
OUTPUT_ROOT = ROOT / "data/manifests_track12"
SPLIT_SEED = 1000
VALIDATION_FRACTION = 0.10

MIXES: dict[str, dict[str, Any]] = {
    "T1-P20": {
        "standard_mass": 0.80,
        "plus_categories": {"lighting": 0.10, "env": 0.10},
        "env_label_quality": "recovered-coarse-not-background-texture",
    },
    "T2-P30": {
        "standard_mass": 0.70,
        "plus_categories": {"camera_pose": 0.10, "sensor_noise": 0.10, "env": 0.10},
        "env_label_quality": "recovered-coarse-not-object-layout-or-robot-init",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def eligible_rows(rows: list[dict[str, Any]], mix_id: str) -> list[dict[str, Any]]:
    spec = MIXES[mix_id]
    plus_categories = set(spec["plus_categories"])
    selected = []
    for row in rows:
        _require(int(row["source_fps"]) == 20, f"non-20Hz row {episode_key(row)}")
        if row["source"] == "libero_standard" and row["perturb_category"] == "base":
            selected.append(row)
        elif row["source"] == "libero_plus" and row["perturb_category"] in plus_categories:
            selected.append(row)
    _require(selected, f"{mix_id} selected no episodes")
    return selected


def apply_holdout(rows: list[dict[str, Any]], denylist: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blocked = {
        (str(item["source_repo"]), str(item["revision"]), int(item["source_episode_index"]))
        for item in denylist
    }
    kept = []
    held = []
    for row in rows:
        if episode_key(row) in blocked:
            held.append(row)
        else:
            kept.append(row)
    return kept, held


def hierarchical_mass(rows: list[dict[str, Any]], mix_id: str) -> dict[tuple[str, str, str, str], float]:
    spec = MIXES[mix_id]
    mass: dict[tuple[str, str, str, str], float] = {}
    standard = [row for row in rows if row["source"] == "libero_standard"]
    plus = [row for row in rows if row["source"] == "libero_plus"]
    _require(standard and plus, f"{mix_id} missing standard or plus rows")

    suites = sorted({str(row["suite"]) for row in standard})
    for suite in suites:
        suite_rows = [row for row in standard if row["suite"] == suite]
        suite_mass = spec["standard_mass"] / len(suites)
        tasks = sorted({str(row["task"]) for row in suite_rows})
        for task in tasks:
            mass[("libero_standard", suite, task, "base")] = suite_mass / len(tasks)

    for category, category_mass in spec["plus_categories"].items():
        cat_rows = [row for row in plus if row["perturb_category"] == category]
        _require(cat_rows, f"{mix_id} has no plus/{category} episodes")
        buckets = sorted({(str(row["suite"]), str(row["task"])) for row in cat_rows})
        for suite, task in buckets:
            mass[("libero_plus", suite, task, category)] = category_mass / len(buckets)

    total = sum(mass.values())
    _require(math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-10), f"{mix_id} mass sums to {total}")
    return mass


def make_weights(train_rows: list[dict[str, Any]], mix_id: str) -> list[dict[str, Any]]:
    target = hierarchical_mass(train_rows, mix_id)
    groups: Counter[tuple[str, str, str, str]] = Counter(bucket_key(row) for row in train_rows)
    weights: list[dict[str, Any]] = []
    for row in train_rows:
        bucket = bucket_key(row)
        observed = int(row["frame_count"])
        _require(observed > 0, f"non-positive frame_count for {episode_key(row)}")
        _require(bucket in target, f"{mix_id} unexpected bucket {bucket}")
        probability = float(target[bucket]) / groups[bucket]
        weights.append(
            {
                **row,
                "episode_sampling_probability": probability,
                "sampling_probability": probability,
                "sampling_weight": probability,
                "observed_frame_count": observed,
                "frame_weight": probability / observed,
                "effective_exposures": probability * len(train_rows),
            }
        )
    total = sum(float(row["episode_sampling_probability"]) for row in weights)
    _require(math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-10), f"{mix_id} episode mass {total}")
    frame_mass = sum(float(row["frame_weight"]) * int(row["observed_frame_count"]) for row in weights)
    _require(math.isclose(frame_mass, 1.0, rel_tol=1e-10, abs_tol=1e-10), f"{mix_id} frame mass {frame_mass}")
    return weights


def category_mass(weights: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in weights:
        key = "standard" if row["source"] == "libero_standard" else str(row["perturb_category"])
        totals[key] += float(row["episode_sampling_probability"])
    return dict(totals)


def build_one(
    rows: list[dict[str, Any]],
    mix_id: str,
    output_root: Path,
    denylist: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = eligible_rows(rows, mix_id)
    kept, held = apply_holdout(eligible, denylist)
    train, validation, exceptions, grouping_audit = split_rows_v2(kept, SPLIT_SEED, VALIDATION_FRACTION)
    train_ids = {episode_key(row) for row in train}
    validation_ids = {episode_key(row) for row in validation}
    held_ids = {episode_key(row) for row in held}
    _require(not (train_ids & validation_ids), "train/validation overlap")
    _require(not (train_ids & held_ids), "train/holdout overlap")
    weighted = make_weights(train, mix_id)
    out = output_root / mix_id
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "train_episodes.parquet", train)
    write_table(out / "validation_episodes.parquet", validation)
    write_table(out / "sampling_weights.parquet", weighted)
    write_table(out / "holdout_episodes.parquet", held)
    write_csv(
        out / "grouping_audit.csv",
        grouping_audit,
        [
            "source",
            "suite",
            "task",
            "category",
            "grouping_quality",
            "group_key",
            "episode_count",
            "group_count",
            "validation_episode_count",
            "train_episode_count",
            "source_file_unique_count",
            "source_file_used_as_group",
            "reason",
        ],
    )
    write_csv(
        out / "shortfalls_and_exceptions.csv",
        [{"bucket": repr(item["bucket"]), "reason": item["reason"], "episode_count": item["episode_count"], "group_count": item["group_count"]} for item in exceptions],
        ["bucket", "reason", "episode_count", "group_count"],
    )
    observed = category_mass(weighted)
    distribution = [
        {"stratum": key, "sampling_probability": value}
        for key, value in sorted(observed.items())
    ]
    write_csv(out / "distribution_before_after.csv", distribution, ["stratum", "sampling_probability"])
    source_roots = {
        repo: {"root": str(Path(next(row["dataset_root"] for row in train if row["source_repo"] == repo)))}
        for repo in sorted({str(row["source_repo"]) for row in train})
    }
    write_json(out / "source_roots.json", source_roots)
    git = git_info()
    manifest = {
        "schema_version": 1,
        "mix_id": mix_id,
        "plan": "docs/AXIS_PI05_TRACK1_TRACK2_OFFICIAL_BLACKWELL_LORA_PLAN_20260910.md",
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "split_unit": "source_repo/revision/source_episode_index",
        "standard_mass": MIXES[mix_id]["standard_mass"],
        "plus_categories": MIXES[mix_id]["plus_categories"],
        "env_label_quality": MIXES[mix_id]["env_label_quality"],
        "reused_m0_m1_m2": False,
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "holdout_episode_count": len(held),
        "sampling_weight_sum": 1.0,
        "frame_weight_mass_sum": 1.0,
        "observed_stratum_mass": observed,
        "source_revisions": {
            repo: next(str(row["revision"]) for row in train if row["source_repo"] == repo)
            for repo in sorted({str(row["source_repo"]) for row in train})
        },
        "git": git,
        "catalog_sha256": sha256_file(CATALOG) if CATALOG.is_file() else None,
        "holdout_denylist_count": len(denylist),
    }
    write_json(out / "manifest.json", manifest)
    write_json(
        out / "manifest.sha256",
        {"path": "manifest.json", "sha256": sha256_file(out / "manifest.json")},
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Track1/Track2 20Hz mixes")
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--mix", action="append", choices=sorted(MIXES), default=None)
    parser.add_argument("--holdout", type=Path, default=ROOT / "configs/axis_pi05/track1_example_holdout.json")
    args = parser.parse_args()
    rows = read_jsonl(args.catalog)
    denylist: list[dict[str, Any]] = []
    if args.holdout.is_file():
        payload = json.loads(args.holdout.read_text(encoding="utf-8"))
        denylist = list(payload.get("episodes", []))
    mixes = args.mix or ["T1-P20", "T2-P30"]
    reports = [build_one(rows, mix_id, args.out, denylist) for mix_id in mixes]
    print(json.dumps({"mixes": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
