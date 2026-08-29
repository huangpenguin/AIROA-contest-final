"""Canonical evaluation result v2, coverage gates, and paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from .common import ROOT, read_jsonl, resolve_path, sha256_file, write_csv, write_json, write_jsonl, write_table


EPISODE_FIELDS = (
    "candidate",
    "checkpoint",
    "checkpoint_sha256",
    "manifest_sha256",
    "protocol_sha256",
    "case_id",
    "benchmark",
    "suite",
    "task_id",
    "task_name",
    "instruction",
    "init_state_index",
    "env_seed",
    "policy_seed",
    "perturbation_id",
    "perturbation_family",
    "perturbation_params",
    "success",
    "collision",
    "goal_reached",
    "goal_reached_but_collision",
    "timeout",
    "steps",
    "elapsed_sec",
    "failure_reason",
    "failure_class",
    "infra_retry",
    "video_path",
    "trajectory_path",
    "error_summary",
)


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def canonical_episode(row: dict[str, Any]) -> dict[str, Any]:
    """Fill the result schema while preserving structured error information."""

    result = {key: row.get(key) for key in EPISODE_FIELDS}
    for key in ("success", "collision", "goal_reached", "goal_reached_but_collision", "timeout", "infra_retry"):
        result[key] = _bool(result[key])
    for key in ("task_id", "init_state_index", "env_seed", "policy_seed", "steps"):
        if result[key] is not None:
            result[key] = int(result[key])
    result["elapsed_sec"] = float(result["elapsed_sec"] or 0.0)
    result["perturbation_params"] = dict(result["perturbation_params"] or {})
    result["failure_reason"] = result["failure_reason"] or (None if result["success"] else "unknown")
    if not result["failure_class"]:
        if result["success"]:
            result["failure_class"] = None
        elif result["failure_reason"] in {"exception", "server_error", "nonfinite_action", "policy_timeout"}:
            result["failure_class"] = "model_error"
        else:
            result["failure_class"] = "episode_failure"
    result["error_summary"] = result["error_summary"] or None
    return result


def coverage_report(expected_case_ids: Iterable[str], observed_episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    expected = list(expected_case_ids)
    observed = [str(row.get("case_id")) for row in observed_episodes]
    expected_set = set(expected)
    observed_set = set(observed)
    duplicates = sorted({case for case in observed if observed.count(case) > 1})
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    return {
        "expected_case_count": len(expected),
        "observed_case_count": len(observed),
        "unique_observed_case_count": len(observed_set),
        "missing_case_ids": missing,
        "extra_case_ids": extra,
        "duplicate_case_ids": duplicates,
        "complete": not missing and not extra and not duplicates and len(expected) == len(observed),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    ordered = sorted(values)
    pos = 0.95 * (len(ordered) - 1)
    low, high = math.floor(pos), math.ceil(pos)
    p95 = ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
    return {"mean": _mean(values), "median": statistics.median(values), "p95": p95}


def bootstrap_ci(values: list[float], statistic: Callable[[list[float]], float] = lambda xs: sum(xs) / len(xs), *, seed: int = 1000, draws: int = 10_000) -> dict[str, Any]:
    if not values:
        return {"estimate": None, "lower": None, "upper": None, "draws": 0, "seed": seed}
    rng = random.Random(seed)
    estimate = float(statistic(values))
    samples: list[float] = []
    for _ in range(int(draws)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(float(statistic(sample)))
    samples.sort()
    return {
        "estimate": estimate,
        "lower": samples[int(0.025 * (len(samples) - 1))],
        "upper": samples[int(0.975 * (len(samples) - 1))],
        "draws": int(draws),
        "seed": int(seed),
    }


def paired_delta(candidate: Iterable[dict[str, Any]], baseline: Iterable[dict[str, Any]], *, seed: int = 1000, draws: int = 10_000) -> dict[str, Any]:
    baseline_by_id = {str(row.get("case_id")): _bool(row.get("success")) for row in baseline}
    deltas = [
        (1.0 if _bool(row.get("success")) else 0.0) - (1.0 if baseline_by_id[ str(row.get("case_id"))] else 0.0)
        for row in candidate
        if str(row.get("case_id")) in baseline_by_id
    ]
    return {
        "paired_case_count": len(deltas),
        "success_delta": bootstrap_ci(deltas, seed=seed, draws=draws),
    }


def _group_success(episodes: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in episodes:
        groups.setdefault(key_fn(row), []).append(1.0 if row["success"] else 0.0)
    return {key: float(sum(values) / len(values)) for key, values in sorted(groups.items())}


def summarize_episodes(
    episodes: list[dict[str, Any]],
    *,
    candidate: str,
    panel: str,
    checkpoint: str | None,
    checkpoint_sha256: str | None,
    manifest_sha256: str | None,
    protocol_sha256: str | None,
    official_result: bool = False,
    result_scope: str = "local_model_selection",
    baseline_episodes: list[dict[str, Any]] | None = None,
    bootstrap_seed: int = 1000,
    bootstrap_draws: int = 10_000,
) -> dict[str, Any]:
    episodes = [canonical_episode(row) for row in episodes]
    clean = all(row["perturbation_family"] in (None, "clean") for row in episodes)
    task = _group_success(episodes, lambda row: str(row["task_name"]))
    task_perturb = _group_success(episodes, lambda row: f"{row['task_name']}::{row['perturbation_family']}")
    suite = _group_success(episodes, lambda row: str(row["suite"]))
    family = _group_success(episodes, lambda row: str(row["perturbation_family"]))
    primary_values = list(task.values()) if clean else list(task_perturb.values())
    success_values = [1.0 if row["success"] else 0.0 for row in episodes]
    collision_values = [1.0 if row["collision"] else 0.0 for row in episodes]
    goal_collision_values = [1.0 if row["goal_reached_but_collision"] else 0.0 for row in episodes]
    timeout_values = [1.0 if row["timeout"] else 0.0 for row in episodes]
    model_error_values = [1.0 if row["failure_class"] == "model_error" else 0.0 for row in episodes]
    infra_error_values = [1.0 if row["failure_class"] == "infrastructure" else 0.0 for row in episodes]
    successful_steps = [float(row["steps"]) for row in episodes if row["success"]]
    all_steps = [float(row["steps"]) for row in episodes]
    latency = [float(row["elapsed_sec"]) for row in episodes]
    result = {
        "schema_version": 2,
        "candidate": candidate,
        "panel": panel,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": protocol_sha256,
        "official_result": bool(official_result),
        "result_scope": result_scope,
        "diagnostic_only": not official_result,
        "case_count": len(episodes),
        "metrics": {
            "micro_success": _mean(success_values),
            "task_macro_success": _mean(list(task.values())),
            "task_perturbation_macro_success": _mean(list(task_perturb.values())),
            "suite_macro_success": _mean(list(suite.values())),
            "worst_suite_success": min(suite.values()) if suite else None,
            "perturbation_family_macro_success": _mean(list(family.values())),
            "worst_perturbation_family_success": min(family.values()) if family else None,
            "collision_rate": _mean(collision_values),
            "goal_reached_but_collision_rate": _mean(goal_collision_values),
            "timeout_stall_rate": _mean(timeout_values),
            "runtime_model_error_rate": _mean(model_error_values),
            "infrastructure_error_rate": _mean(infra_error_values),
            "success_steps": _quantiles(successful_steps),
            "all_steps": _quantiles(all_steps),
            "latency_sec": _quantiles(latency),
        },
        "confidence_intervals": {
            "micro_success": bootstrap_ci(success_values, seed=bootstrap_seed, draws=bootstrap_draws),
            "primary": bootstrap_ci(primary_values, seed=bootstrap_seed + 1, draws=bootstrap_draws),
        },
        "groups": {"task": task, "task_perturbation": task_perturb, "suite": suite, "perturbation_family": family},
        "paired_delta_to_axis_init": paired_delta(episodes, baseline_episodes or [], seed=bootstrap_seed + 2, draws=bootstrap_draws) if baseline_episodes is not None else None,
    }
    return result


def write_result_bundle(
    output_dir: Path,
    episodes: list[dict[str, Any]],
    *,
    expected_case_ids: Iterable[str],
    summary_kwargs: dict[str, Any],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = [canonical_episode(row) for row in episodes]
    coverage = coverage_report(expected_case_ids, canonical)
    summary = summarize_episodes(canonical, **summary_kwargs)
    summary["coverage_complete"] = coverage["complete"]
    summary["qualification_blockers"] = [] if coverage["complete"] else ["coverage_incomplete"]
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "coverage.json", coverage)
    write_jsonl(output_dir / "episodes.jsonl", canonical)
    fields = list(EPISODE_FIELDS)
    csv_rows = [{key: json.dumps(row[key], ensure_ascii=False, sort_keys=True) if isinstance(row[key], (dict, list)) else row[key] for key in fields} for row in canonical]
    write_csv(output_dir / "episodes.csv", csv_rows, fields)
    task_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in canonical:
        key = (str(row.get("suite")), str(row.get("task_id")), str(row.get("task_name")))
        task_groups.setdefault(key, []).append(row)
    task_rows = []
    for (suite, task_id, task_name), group in sorted(task_groups.items()):
        task_rows.append({
            "suite": suite,
            "task_id": task_id,
            "task_name": task_name,
            "case_count": len(group),
            "success_count": sum(bool(row["success"]) for row in group),
            "success_rate": sum(bool(row["success"]) for row in group) / len(group),
            "collision_rate": sum(bool(row["collision"]) for row in group) / len(group),
            "timeout_rate": sum(bool(row["timeout"]) for row in group) / len(group),
        })
    write_csv(output_dir / "tasks.csv", task_rows, ["suite", "task_id", "task_name", "case_count", "success_count", "success_rate", "collision_rate", "timeout_rate"])
    # Keep the parquet schema stable even when clean cases have an empty
    # perturbation object (Arrow cannot materialize a struct with zero child
    # fields).  JSON is still lossless and preserves the result contract.
    parquet_rows = [
        {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        }
        for row in canonical
    ]
    write_table(output_dir / "episodes.parquet", parquet_rows)
    failures = [row for row in canonical if not row["success"]]
    write_csv(output_dir / "failure_index.csv", [{key: row.get(key) for key in ("case_id", "candidate", "suite", "task_name", "perturbation_family", "failure_reason", "failure_class", "video_path", "trajectory_path")} for row in failures], ["case_id", "candidate", "suite", "task_name", "perturbation_family", "failure_reason", "failure_class", "video_path", "trajectory_path"])
    if run_metadata is not None:
        write_json(output_dir / "run_metadata.json", run_metadata)
    return {"summary": summary, "coverage": coverage}


def load_episode_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if isinstance(row.get("perturbation_params"), str):
                row["perturbation_params"] = json.loads(row["perturbation_params"] or "{}")
        return rows
    raise ValueError(f"unsupported episode input: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path, help="JSONL manifest containing case_id")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    expected = read_jsonl(resolve_path(args.expected))
    rows = load_episode_rows(resolve_path(args.episodes))
    result = write_result_bundle(
        resolve_path(args.out), rows,
        expected_case_ids=[row["case_id"] for row in expected],
        summary_kwargs={
            "candidate": args.candidate,
            "panel": args.panel,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": args.checkpoint_sha256,
            "manifest_sha256": args.manifest_sha256 or sha256_file(resolve_path(args.expected)),
            "protocol_sha256": args.protocol_sha256,
        },
    )
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2))
    if not result["coverage"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
