from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .common import ROOT, git_info, read_jsonl, resolve_path, sha256_file, write_csv, write_json, write_table
from .lock import load_and_validate


def episode_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["source_repo"]), str(row["revision"]), int(row["source_episode_index"]))


def bucket_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row["source"]), str(row["suite"]), str(row["task"]), str(row["perturb_category"]))


def split_rows(rows: list[dict[str, Any]], seed: int, fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_bucket: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[bucket_key(row)].append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    for key, bucket in sorted(by_bucket.items()):
        # For Plus, group a base source_file before choosing a holdout.  If the
        # label is unavailable, the episode itself is the safest non-leaking unit.
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            source_file = row.get("source_file")
            group = str(source_file) if row.get("source") == "libero_plus" and source_file else repr(episode_key(row))
            groups[group].append(row)
        if len(bucket) < 5 or len(groups) < 2:
            train.extend(bucket)
            exceptions.append({"bucket": key, "reason": "no_holdout", "episode_count": len(bucket), "group_count": len(groups)})
            continue
        ranked = sorted(
            groups.items(),
            key=lambda item: hashlib.sha256(f"{item[0]}:{seed}".encode()).hexdigest(),
        )
        target = max(1, int(math.ceil(len(bucket) * fraction)))
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
    if {episode_key(row) for row in train} & {episode_key(row) for row in validation}:
        raise RuntimeError("train/validation episode overlap")
    return sorted(train, key=episode_key), sorted(validation, key=episode_key), exceptions


def _counts(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> Counter[Any]:
    return Counter(key(row) for row in rows)


def _hierarchical_mass(rows: list[dict[str, Any]], mix: str) -> dict[tuple[str, str, str, str], float]:
    if not rows:
        return {}
    source_groups = {source: [row for row in rows if row["source"] == source] for source in {row["source"] for row in rows}}
    source_mass: dict[str, float]
    if mix == "M1":
        source_mass = {"libero_plus": 0.5, "libero_standard": 0.5}
    else:
        source_mass = {"libero_plus": 0.5, "libero_standard": 0.5}
    mass: dict[tuple[str, str, str, str], float] = {}
    for source, source_rows in source_groups.items():
        source_weight = source_mass.get(source, 0.0)
        suites = sorted({str(row["suite"]) for row in source_rows})
        if mix == "M1" and source == "libero_standard":
            suite_mass = {suite: (0.25 if suite == "libero_90" else 0.15 if suite == "libero_10" else 0.10 / 3.0) for suite in suites}
        else:
            suite_mass = {suite: source_weight / max(1, len(suites)) for suite in suites}
        for suite, suite_rows in ((suite, [r for r in source_rows if r["suite"] == suite]) for suite in suites):
            suite_target = suite_mass.get(suite, 0.0)
            categories = sorted({str(row["perturb_category"]) for row in suite_rows})
            if source == "libero_plus" and mix in {"M0", "M1"}:
                category_mass = {category: suite_target / max(1, len(categories)) for category in categories}
            else:
                category_mass = {category: suite_target / max(1, len(categories)) for category in categories}
            for category in categories:
                cat_rows = [r for r in suite_rows if r["perturb_category"] == category]
                tasks = sorted({str(row["task"]) for row in cat_rows})
                task_target = category_mass[category] / max(1, len(tasks))
                for task in tasks:
                    mass[(source, suite, task, category)] = task_target
    if mix == "M2":
        # Natural source distribution with a 2x task cap; Plus category floors
        # are applied at source level before task allocation.
        mass = {}
        for source, source_rows in source_groups.items():
            source_target = source_mass[source]
            suites = sorted({str(r["suite"]) for r in source_rows})
            suite_counts = Counter(str(r["suite"]) for r in source_rows)
            for suite in suites:
                suite_rows = [r for r in source_rows if r["suite"] == suite]
                suite_target = source_target * suite_counts[suite] / len(source_rows)
                categories = sorted({str(r["perturb_category"]) for r in suite_rows})
                cat_counts = Counter(str(r["perturb_category"]) for r in suite_rows)
                if source == "libero_plus":
                    cat_target = {cat: 0.15 * suite_target + 0.25 * suite_target * cat_counts[cat] / len(suite_rows) for cat in categories}
                else:
                    cat_target = {cat: suite_target * cat_counts[cat] / len(suite_rows) for cat in categories}
                for category in categories:
                    cat_rows = [r for r in suite_rows if r["perturb_category"] == category]
                    task_counts = Counter(str(r["task"]) for r in cat_rows)
                    natural = [value / len(cat_rows) for value in task_counts.values()]
                    cap = 2.0 * statistics.median(natural) if natural else 1.0
                    clipped = {task: min(count / len(cat_rows), cap) for task, count in task_counts.items()}
                    norm = sum(clipped.values()) or 1.0
                    for task, task_prob in clipped.items():
                        mass[(source, suite, task, category)] = cat_target[category] * task_prob / norm
    return mass


def make_weights(train_rows: list[dict[str, Any]], mix: str) -> list[dict[str, Any]]:
    target = _hierarchical_mass(train_rows, mix)
    groups: Counter[tuple[str, str, str, str]] = Counter()
    for row in train_rows:
        groups[(row["source"], row["suite"], row["task"], row["perturb_category"])] += 1
    weights: list[dict[str, Any]] = []
    for row in train_rows:
        key = (row["source"], row["suite"], row["task"], row["perturb_category"])
        probability = target.get(key, 0.0) / groups[key]
        weights.append({
            **row,
            "sampling_probability": probability,
            "sampling_weight": probability,
            "effective_exposures": probability * len(train_rows),
        })
    total = sum(float(row["sampling_weight"]) for row in weights)
    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        if total <= 0:
            raise RuntimeError(f"{mix} sampling weights sum to zero")
        for row in weights:
            row["sampling_weight"] /= total
            row["sampling_probability"] = row["sampling_weight"]
            row["effective_exposures"] = row["sampling_weight"] * len(train_rows)
    return weights


def build_one(catalog_path: Path, output_root: Path, mix: str, seed: int, validation_fraction: float, lock_path: Path) -> dict[str, Any]:
    rows = read_jsonl(catalog_path)
    train, validation, exceptions = split_rows(rows, seed, validation_fraction)
    weighted = make_weights(train, mix)
    out = output_root / mix
    out.mkdir(parents=True, exist_ok=True)
    write_table(out / "train_episodes.parquet", train)
    write_table(out / "validation_episodes.parquet", validation)
    write_table(out / "sampling_weights.parquet", weighted)
    before = Counter((row["source"], row["suite"], row["perturb_category"]) for row in train)
    after: dict[tuple[str, str, str], float] = defaultdict(float)
    for row in weighted:
        key = (row["source"], row["suite"], row["perturb_category"])
        after[key] += float(row["sampling_weight"])
    distribution = []
    for key in sorted(set(before) | set(after)):
        distribution.append({"source": key[0], "suite": key[1], "category": key[2], "train_count": before.get(key, 0), "sampling_probability": after.get(key, 0.0)})
    write_csv(out / "distribution_before_after.csv", distribution, ["source", "suite", "category", "train_count", "sampling_probability"])
    write_csv(out / "shortfalls_and_exceptions.csv", exceptions, ["bucket", "reason", "episode_count", "group_count"])
    catalog_sha = sha256_file(catalog_path)
    manifest = {
        "schema_version": 1,
        "mix_id": mix,
        "mix_name": {"M0": "source_suite_balanced", "M1": "competition_prior", "M2": "natural_capped"}[mix],
        "source_revisions": {row["source_repo"]: row["revision"] for row in rows},
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "train_episode_count": len(train),
        "validation_episode_count": len(validation),
        "shortfalls_and_exceptions": exceptions,
        "catalog_sha256": catalog_sha,
        "implementation_commit": git_info()["commit"],
        "sampling_weight_sum": sum(row["sampling_weight"] for row in weighted),
        "repeat_exposure_quantiles": {
            quantile: _quantile(sorted(row["effective_exposures"] for row in weighted), quantile)
            for quantile in (0.5, 0.95, 0.99)
        },
        "lock_sha256": sha256_file(lock_path),
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "manifest.sha256", {"sha256": sha256_file(out / "manifest.json")})
    return manifest


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def main() -> None:
    ap = argparse.ArgumentParser(description="build deterministic split and M0/M1/M2 manifests")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/sampling.yaml")
    ap.add_argument("--mix", choices=["M0", "M1", "M2"], action="append")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    config = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[2]
    mixes = ["M0", "M1", "M2"] if args.all or not args.mix else args.mix
    lock_path = resolve_path("configs/axis_pi05/data_sources.lock.yaml", root)
    for mix in mixes:
        manifest = build_one(
            resolve_path(config["catalog_path"], root),
            resolve_path(config["output_root"], root),
            mix,
            int(config["split_seed"]),
            float(config["validation_fraction"]),
            lock_path,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
