"""Select the T2 winner from fixed, case-aligned result bundles."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from .common import ROOT, git_info, resolve_path, write_json
from .eval_result import load_episode_rows, paired_delta


def _number(value: Any, default: float = -1.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _metric(summary: dict[str, Any], name: str, default: float = -1.0) -> float:
    return _number(summary.get("metrics", {}).get(name), default)


def _step_mean(summary: dict[str, Any]) -> float:
    value = summary.get("metrics", {}).get("success_steps", {})
    return _number(value.get("mean") if isinstance(value, dict) else None, 1e12)


def _recipe_rank(candidate: str) -> int:
    # All current candidates use F0.  The explicit rank is only the final
    # deterministic recipe tie-break and never replaces the primary metric.
    match = re.search(r"m([012])", candidate.lower())
    return int(match.group(1)) if match else 99


def _qualified(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if not summary.get("coverage_complete", False):
        blockers.append("coverage_incomplete")
    if int(summary.get("case_count", 0)) <= 0:
        blockers.append("empty_result")
    metrics = summary.get("metrics", {})
    if _number(metrics.get("infrastructure_error_rate"), 1.0) > 0.0:
        blockers.append("infrastructure_error")
    if _number(metrics.get("runtime_model_error_rate"), 1.0) > 0.0:
        blockers.append("runtime_model_error")
    return not blockers, blockers


def _load_candidate(candidate: str, result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    episodes_path = result_dir / "episodes.jsonl"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary for {candidate}: {summary_path}")
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing raw episodes for {candidate}: {episodes_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = load_episode_rows(episodes_path)
    qualified, blockers = _qualified(summary)
    return {
        "candidate": candidate,
        "result_dir": str(result_dir),
        "summary": summary,
        "episodes": episodes,
        "qualified": qualified,
        "qualification_blockers": blockers,
    }


def _pairwise(candidate_a: dict[str, Any], candidate_b: dict[str, Any], *, seed: int, draws: int) -> dict[str, Any]:
    delta = paired_delta(candidate_a["episodes"], candidate_b["episodes"], seed=seed, draws=draws)
    return {
        "candidate_a": candidate_a["candidate"],
        "candidate_b": candidate_b["candidate"],
        "paired": delta,
    }


def _comparison_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, int]:
    summary = candidate["summary"]
    return (
        _metric(summary, "task_perturbation_macro_success"),
        _metric(summary, "worst_suite_success"),
        _metric(summary, "worst_perturbation_family_success"),
        -_metric(summary, "collision_rate", 1.0),
        -_metric(summary, "timeout_stall_rate", 1.0) - _metric(summary, "runtime_model_error_rate", 1.0),
        -_step_mean(summary),
        -_recipe_rank(candidate["candidate"]),
    )


def select(
    candidates: list[tuple[str, Path]],
    *,
    out: Path,
    protocol_sha256: str | None = None,
    manifest_sha256: str | None = None,
    bootstrap_seed: int = 1000,
    bootstrap_draws: int = 10_000,
    override_candidate: str | None = None,
    override_reason: str | None = None,
    operator: str | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    loaded = [_load_candidate(candidate, path) for candidate, path in candidates]
    manifest_hashes = {item["summary"].get("manifest_sha256") for item in loaded}
    protocol_hashes = {item["summary"].get("protocol_sha256") for item in loaded}
    if len(manifest_hashes) > 1:
        raise ValueError("candidates do not share one manifest hash")
    if len(protocol_hashes) > 1:
        raise ValueError("candidates do not share one protocol hash")
    if manifest_sha256 is not None and manifest_hashes != {manifest_sha256}:
        raise ValueError("candidate manifest hash does not match requested lock")
    if protocol_sha256 is not None and protocol_hashes != {protocol_sha256}:
        raise ValueError("candidate protocol hash does not match requested lock")
    qualified = [item for item in loaded if item["qualified"]]
    comparisons: list[dict[str, Any]] = []
    for index, first in enumerate(qualified):
        for second in qualified[index + 1:]:
            comparisons.append(_pairwise(first, second, seed=bootstrap_seed + len(comparisons), draws=bootstrap_draws))

    decision = "automatic"
    metric_path = ["qualification"]
    winner: dict[str, Any] | None = None
    if not qualified:
        decision = "blocked"
    else:
        ranked = sorted(qualified, key=_comparison_key, reverse=True)
        winner = ranked[0]
        metric_path.append("primary")
        if len(ranked) > 1:
            first, second = ranked[0], ranked[1]
            primary_diff = abs(
                _metric(first["summary"], "task_perturbation_macro_success")
                - _metric(second["summary"], "task_perturbation_macro_success")
            )
            pair = next(
                (
                    item["paired"]
                    for item in comparisons
                    if {item["candidate_a"], item["candidate_b"]} == {first["candidate"], second["candidate"]}
                ),
                None,
            )
            ci = (pair or {}).get("success_delta", {})
            ci_contains_zero = _number(ci.get("lower"), 1.0) <= 0.0 <= _number(ci.get("upper"), -1.0)
            if primary_diff < 0.02 or ci_contains_zero:
                metric_path.append("statistical_tie")
                winner = sorted(
                    ranked,
                    key=lambda item: (
                        _metric(item["summary"], "worst_suite_success"),
                        _metric(item["summary"], "worst_perturbation_family_success"),
                        -_metric(item["summary"], "collision_rate", 1.0),
                        -_metric(item["summary"], "timeout_stall_rate", 1.0),
                        -_metric(item["summary"], "runtime_model_error_rate", 1.0),
                        -_step_mean(item["summary"]),
                        -_recipe_rank(item["candidate"]),
                    ),
                    reverse=True,
                )[0]
                metric_path.extend(["worst_suite", "worst_perturbation_family", "collision", "timeout_model_error", "success_steps", "recipe"])

    winner_id = winner["candidate"] if winner else None
    result = {
        "schema_version": 1,
        "decision": decision,
        "winner": winner_id,
        "candidate_checkpoint_sha256": winner["summary"].get("checkpoint_sha256") if winner else None,
        "manifest_sha256": manifest_sha256 or (winner["summary"].get("manifest_sha256") if winner else None),
        "protocol_sha256": protocol_sha256 or (winner["summary"].get("protocol_sha256") if winner else None),
        "metric_path": metric_path,
        "qualification": {
            item["candidate"]: {"qualified": item["qualified"], "blockers": item["qualification_blockers"], "case_count": item["summary"].get("case_count")}
            for item in loaded
        },
        "summaries": {
            item["candidate"]: item["summary"] for item in loaded
        },
        "comparisons": comparisons,
        "runner_git_commit": git_info(ROOT)["commit"],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "official_result": False,
        "result_scope": "local_model_selection",
    }
    write_json(out, result)
    if override_candidate is not None:
        if override_candidate not in {candidate for candidate, _ in candidates}:
            raise ValueError(f"override candidate is not in candidates: {override_candidate}")
        if not override_reason or not operator:
            raise ValueError("manual override requires --reason and --operator")
        write_json(out.with_name("winner_override.json"), {
            "schema_version": 1,
            "automatic_winner": winner_id,
            "override_winner": override_candidate,
            "reason": override_reason,
            "operator": operator,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "decision": "manual_override",
            "risk": "manual choice is not an automatic selection result",
        })
        result["decision"] = "manual_override"
        result["winner"] = override_candidate
        write_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, help="candidate_id=result_dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--bootstrap-seed", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--override-candidate")
    parser.add_argument("--reason")
    parser.add_argument("--operator")
    args = parser.parse_args()
    candidates: list[tuple[str, Path]] = []
    for value in args.candidate:
        if "=" not in value:
            raise SystemExit("--candidate must use candidate_id=result_dir")
        candidate, path = value.split("=", 1)
        candidates.append((candidate, resolve_path(path)))
    result = select(
        candidates,
        out=resolve_path(args.out),
        manifest_sha256=args.manifest_sha256,
        protocol_sha256=args.protocol_sha256,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_draws=args.bootstrap_draws,
        override_candidate=args.override_candidate,
        override_reason=args.reason,
        operator=args.operator,
    )
    print(json.dumps({"decision": result["decision"], "winner": result["winner"]}, ensure_ascii=False, indent=2))
    if result["decision"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
