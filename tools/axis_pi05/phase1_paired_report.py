"""Create an auditable paired comparison for two phase-1 rollout summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .eval_result import bootstrap_ci


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _read_summary(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    episodes_path = Path(summary["artifacts"]["episodes_csv"])
    rows = list(csv.DictReader(episodes_path.open(newline="", encoding="utf-8")))
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
        "stall_count": sum(row.get("failure_reason") == "stall" for row in rows),
    }


def _provenance(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in PROVENANCE_KEYS}


def _task_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('official_task_index')}:{row.get('task_name')}"
        grouped[key].append(row)
    return {
        key: {
            "case_count": len(group),
            "success_count": sum(_bool(row.get("success")) for row in group),
        }
        for key, group in sorted(grouped.items())
    }


def build_report(candidate_path: Path, baseline_path: Path, *, seed: int, draws: int) -> dict[str, Any]:
    candidate, candidate_rows, candidate_csv = _read_summary(candidate_path)
    baseline, baseline_rows, baseline_csv = _read_summary(baseline_path)
    candidate_by_id = _episode_map(candidate_rows, "candidate")
    baseline_by_id = _episode_map(baseline_rows, "baseline")
    if set(candidate_by_id) != set(baseline_by_id):
        raise ValueError("candidate and baseline case_id sets differ")

    candidate_provenance = _provenance(candidate)
    baseline_provenance = _provenance(baseline)
    for key in ("norm_stats_path", "norm_stats_sha256", "normalization_mode", "policy_seed_source"):
        if candidate_provenance[key] != baseline_provenance[key]:
            raise ValueError(f"candidate/baseline provenance mismatch for {key}")

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
    candidate_counts = _counts(candidate_rows)
    baseline_counts = _counts(baseline_rows)
    result = {
        "schema_version": 1,
        "status": "review_required",
        "comparison": "paired_same_case_id_seeded_B80",
        "candidate_summary": str(candidate_path),
        "baseline_summary": str(baseline_path),
        "candidate_artifacts": {
            "summary_sha256": _sha256(candidate_path),
            "episodes_csv": str(candidate_csv),
            "episodes_csv_sha256": _sha256(candidate_csv),
            "counts": candidate_counts,
            "provenance": candidate_provenance,
        },
        "baseline_artifacts": {
            "summary_sha256": _sha256(baseline_path),
            "episodes_csv": str(baseline_csv),
            "episodes_csv_sha256": _sha256(baseline_csv),
            "counts": baseline_counts,
            "provenance": baseline_provenance,
        },
        "paired_case_count": len(paired),
        "paired_wins": wins,
        "paired_losses": losses,
        "paired_ties": ties,
        "success_delta": (candidate_counts["success_count"] - baseline_counts["success_count"]) / len(paired),
        "success_delta_bootstrap_95ci": bootstrap_ci(deltas, seed=seed, draws=draws),
        "failure_count_delta": {
            key: candidate_counts[key] - baseline_counts[key]
            for key in ("collision_count", "timeout_count", "stall_count")
        },
        "task_counts": {
            "candidate": _task_counts(candidate_rows),
            "baseline": _task_counts(baseline_rows),
        },
        "paired_cases": paired,
        "decision": {
            "local_gates": {
                "complete_case_coverage": len(paired) == 80,
                "candidate_success_at_least_46": candidate_counts["success_count"] >= 46,
                "candidate_success_delta_at_least_minus_005": (candidate_counts["success_count"] - baseline_counts["success_count"]) / len(paired) >= -0.05,
                "no_stalls": candidate_counts["stall_count"] == 0,
            },
            "recommendation": "eligible_for_human_review",
            "approval": "not_auto_approved_pending_human_review",
            "reason": "Restore, finite-checkpoint, raw-LoRA/folded-dense parity, processor E2E parity, seeded B80 coverage, and merged provenance gates pass; final candidate approval remains a human decision.",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=1002)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    report = build_report(args.candidate, args.baseline, seed=args.bootstrap_seed, draws=args.bootstrap_draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "paired_case_count", "paired_wins", "paired_losses", "paired_ties", "success_delta", "decision")}, indent=2))


if __name__ == "__main__":
    main()
