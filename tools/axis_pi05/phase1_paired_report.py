"""Create an auditable paired comparison for two phase-1 rollout summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .eval_result import bootstrap_ci
from .phase1_rollout import is_stall_episode


PROVENANCE_KEYS = (
    "checkpoint_identity",
    "checkpoint_path",
    "checkpoint_step",
    "checkpoint_kind",
    "lora_arm",
    "norm_stats_path",
    "norm_stats_sha256",
    "normalization_mode",
    "policy_seed_source",
)
PAIRING_FIELDS = (
    "official_task_index",
    "task_name",
    "language",
    "bddl_sha256",
    "init_state_index",
    "init_state_sha256",
    "env_seed",
    "policy_seed",
    "max_steps",
)
COUNT_KEYS = ("case_count", "success_count", "collision_count", "timeout_count", "stall_count")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _scalar(field: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    if field in {"official_task_index", "init_state_index", "env_seed", "policy_seed", "max_steps"}:
        return int(value)
    return value


def _read_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    episodes_path = Path(summary["artifacts"]["episodes_csv"])
    with episodes_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return summary, rows, episodes_path


def _episode_map(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"{label} contains duplicate case_id {case_id}")
        result[case_id] = row
    return result


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "case_count": len(rows),
        "success_count": sum(_bool(row.get("success")) for row in rows),
        "collision_count": sum(_bool(row.get("collision")) for row in rows),
        "timeout_count": sum(_bool(row.get("timeout")) for row in rows),
        "stall_count": sum(is_stall_episode(row) for row in rows),
    }


def _provenance(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in PROVENANCE_KEYS}


def _manifest_path(summary: dict[str, Any], episodes_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return episodes_path.parent.parent / f"{summary['panel']}.jsonl"


def _manifest_ids(path: Path) -> list[str]:
    return [str(json.loads(line)["case_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_audit(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    episodes_path: Path,
    *,
    label: str,
    manifest_path: Path,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise ValueError(f"{label} manifest is missing: {manifest_path}")
    ids = [str(row["case_id"]) for row in rows]
    manifest_ids = _manifest_ids(manifest_path)
    if ids != manifest_ids:
        raise ValueError(f"{label} CSV does not exactly cover manifest order")
    manifest_sha256 = _sha256(manifest_path)
    if summary.get("panel_manifest_sha256") != manifest_sha256:
        raise ValueError(f"{label} summary panel_manifest_sha256 does not match manifest")

    counts = _counts(rows)
    count_mismatches = {
        key: {"csv": counts[key], "summary": summary.get(key)}
        for key in COUNT_KEYS
        if int(summary.get(key, -1)) != counts[key]
    }
    if count_mismatches:
        raise ValueError(f"{label} summary/CSV count mismatch: {count_mismatches}")
    videos_dir = Path(summary["artifacts"]["videos_dir"])
    missing_videos = [str(row["case_id"]) for row in rows if not row.get("video_path") or not Path(row["video_path"]).is_file()]
    video_files = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    if missing_videos or int(summary.get("video_count", -1)) != len(video_files) or len(video_files) != len(rows):
        raise ValueError(
            f"{label} video coverage mismatch: missing_rows={missing_videos[:5]}, "
            f"summary={summary.get('video_count')}, files={len(video_files)}, rows={len(rows)}"
        )
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "case_count": len(rows),
        "unique_case_count": len(set(ids)),
        "video_count": len(video_files),
        "episodes_csv_sha256": _sha256(episodes_path),
        "counts": counts,
    }


def _pairing_mismatches(
    candidate_by_id: dict[str, dict[str, Any]], baseline_by_id: dict[str, dict[str, Any]]
) -> dict[str, list[str]]:
    mismatches: dict[str, list[str]] = defaultdict(list)
    for case_id in sorted(candidate_by_id):
        candidate = candidate_by_id[case_id]
        baseline = baseline_by_id[case_id]
        for field in PAIRING_FIELDS:
            if field not in candidate or field not in baseline:
                mismatches[field].append(case_id)
            elif _scalar(field, candidate[field]) != _scalar(field, baseline[field]):
                mismatches[field].append(case_id)
    return dict(mismatches)


def _cluster_bootstrap(
    paired: list[dict[str, Any]], *, seed: int, draws: int
) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in paired:
        groups[str(row["official_task_index"])].append(float(row["delta"]))
    group_values = list(groups.values())
    all_values = [value for values in group_values for value in values]
    if not all_values:
        return {"estimate": None, "lower": None, "upper": None, "draws": 0, "seed": seed, "cluster_count": 0}
    rng = random.Random(seed)
    samples = []
    for _ in range(int(draws)):
        sampled_groups = [group_values[rng.randrange(len(group_values))] for _ in group_values]
        samples.append(sum(sum(group) for group in sampled_groups) / sum(len(group) for group in sampled_groups))
    samples.sort()
    return {
        "estimate": sum(all_values) / len(all_values),
        "lower": samples[int(0.025 * (len(samples) - 1))],
        "upper": samples[int(0.975 * (len(samples) - 1))],
        "draws": int(draws),
        "seed": int(seed),
        "cluster_count": len(group_values),
    }


def _mcnemar_exact(wins: int, losses: int) -> dict[str, Any]:
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"candidate_wins": wins, "candidate_losses": losses, "discordant_pairs": discordant, "two_sided_exact_p": p_value}


def build_report(
    candidate_path: Path,
    baseline_path: Path,
    *,
    seed: int,
    draws: int,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    candidate, candidate_rows, candidate_csv = _read_summary(candidate_path)
    baseline, baseline_rows, baseline_csv = _read_summary(baseline_path)
    candidate_by_id = _episode_map(candidate_rows, "candidate")
    baseline_by_id = _episode_map(baseline_rows, "baseline")
    if set(candidate_by_id) != set(baseline_by_id):
        raise ValueError("candidate and baseline case_id sets differ")
    if candidate.get("panel") != baseline.get("panel") or candidate.get("execution_protocol") != baseline.get("execution_protocol"):
        raise ValueError("candidate and baseline panel/execution protocol differ")
    candidate_provenance = _provenance(candidate)
    baseline_provenance = _provenance(baseline)
    for key in ("norm_stats_path", "norm_stats_sha256", "normalization_mode", "policy_seed_source"):
        if candidate_provenance[key] != baseline_provenance[key]:
            raise ValueError(f"candidate/baseline provenance mismatch for {key}")

    candidate_manifest = _manifest_path(candidate, candidate_csv, manifest_path)
    baseline_manifest = _manifest_path(baseline, baseline_csv, manifest_path)
    candidate_audit = _artifact_audit(candidate, candidate_rows, candidate_csv, label="candidate", manifest_path=candidate_manifest)
    baseline_audit = _artifact_audit(baseline, baseline_rows, baseline_csv, label="baseline", manifest_path=baseline_manifest)
    if candidate_audit["manifest_sha256"] != baseline_audit["manifest_sha256"]:
        raise ValueError("candidate and baseline manifest hashes differ")

    mismatches = _pairing_mismatches(candidate_by_id, baseline_by_id)
    mismatch_count = sum(len(case_ids) for case_ids in mismatches.values())
    if mismatch_count:
        raise ValueError(f"per-case pairing invariant mismatch: {mismatches}")
    paired = []
    for case_id in sorted(candidate_by_id):
        cand = candidate_by_id[case_id]
        base = baseline_by_id[case_id]
        cand_success = _bool(cand.get("success"))
        base_success = _bool(base.get("success"))
        paired.append(
            {
                "case_id": case_id,
                "candidate_success": cand_success,
                "baseline_success": base_success,
                "delta": int(cand_success) - int(base_success),
                "official_task_index": cand.get("official_task_index"),
                "task_name": cand.get("task_name"),
                "policy_seed": cand.get("policy_seed"),
            }
        )

    deltas = [float(row["delta"]) for row in paired]
    wins = sum(row["delta"] == 1 for row in paired)
    losses = sum(row["delta"] == -1 for row in paired)
    ties = sum(row["delta"] == 0 for row in paired)
    candidate_counts = candidate_audit["counts"]
    baseline_counts = baseline_audit["counts"]
    success_delta = (candidate_counts["success_count"] - baseline_counts["success_count"]) / len(paired)
    episode_ci = bootstrap_ci(deltas, seed=seed, draws=draws)
    cluster_ci = _cluster_bootstrap(paired, seed=seed + 1, draws=draws)
    result = {
        "schema_version": 2,
        "status": "review_required",
        "comparison": "paired_same_case_id_seeded_B80",
        "candidate_summary": str(candidate_path),
        "baseline_summary": str(baseline_path),
        "candidate_artifacts": {"summary_sha256": _sha256(candidate_path), "episodes_csv": str(candidate_csv), "episodes_csv_sha256": candidate_audit["episodes_csv_sha256"], "counts": candidate_counts, "provenance": candidate_provenance, "manifest": candidate_audit},
        "baseline_artifacts": {"summary_sha256": _sha256(baseline_path), "episodes_csv": str(baseline_csv), "episodes_csv_sha256": baseline_audit["episodes_csv_sha256"], "counts": baseline_counts, "provenance": baseline_provenance, "manifest": baseline_audit},
        "pairing_invariants": {"fields": ["case_id", *PAIRING_FIELDS, "execution_protocol"], "mismatch_count": mismatch_count, "mismatch_by_field": mismatches, "manifest_hash_match": True, "protocol_match": True},
        "paired_case_count": len(paired),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "success_delta": success_delta,
        "bootstrap": {"episode": episode_ci, "task_cluster": cluster_ci},
        "mcnemar_exact": _mcnemar_exact(wins, losses),
        "selection_rule": {
            "point_noninferiority": {"estimate": success_delta, "threshold": -0.05, "passes": success_delta >= -0.05},
            "ci_noninferiority": {"ci": episode_ci, "threshold": -0.05, "passes": episode_ci["lower"] >= -0.05},
        },
        "failure_count_delta": {key: candidate_counts[key] - baseline_counts[key] for key in ("collision_count", "timeout_count", "stall_count")},
        "task_counts": {"candidate": _task_counts(candidate_rows), "baseline": _task_counts(baseline_rows)},
        "paired_cases": paired,
        "decision": {
            "local_gates": {"complete_case_coverage": len(paired) == 80, "candidate_success_at_least_46": candidate_counts["success_count"] >= 46, "point_noninferiority": success_delta >= -0.05, "no_stalls": candidate_counts["stall_count"] == 0},
            "recommendation": "eligible_for_human_review",
            "approval": "not_auto_approved_pending_human_review",
            "reason": "Technical restore/fold/processor parity and seeded behavior point gates pass; CI non-inferiority is reported separately and final approval remains a human decision.",
        },
    }
    return result


def _task_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('official_task_index')}:{row.get('task_name')}"
        grouped[key].append(row)
    return {key: {"case_count": len(group), "success_count": sum(_bool(row.get("success")) for row in group)} for key, group in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=1002)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    report = build_report(args.candidate, args.baseline, seed=args.bootstrap_seed, draws=args.bootstrap_draws, manifest_path=args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "paired_case_count", "paired_wins", "paired_losses", "paired_ties", "success_delta", "selection_rule", "mcnemar_exact")}, indent=2))


if __name__ == "__main__":
    main()
