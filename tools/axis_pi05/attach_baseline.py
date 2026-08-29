"""Attach a case-aligned Axis-init paired delta to completed result bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import ROOT, resolve_path, write_json
from .eval_result import load_episode_rows, summarize_episodes


def attach(result_dirs: list[Path], baseline_dir: Path) -> None:
    baseline = load_episode_rows(baseline_dir / "episodes.jsonl")
    baseline_ids = {str(row.get("case_id")) for row in baseline}
    for result_dir in result_dirs:
        summary_path = result_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        episodes = load_episode_rows(result_dir / "episodes.jsonl")
        if {str(row.get("case_id")) for row in episodes} != baseline_ids:
            raise ValueError(f"baseline case IDs do not match: {result_dir}")
        updated = summarize_episodes(
            episodes,
            candidate=summary["candidate"],
            panel=summary["panel"],
            checkpoint=summary.get("checkpoint"),
            checkpoint_sha256=summary.get("checkpoint_sha256"),
            manifest_sha256=summary.get("manifest_sha256"),
            protocol_sha256=summary.get("protocol_sha256"),
            official_result=False,
            result_scope=summary.get("result_scope", "local_model_selection"),
            baseline_episodes=baseline,
            bootstrap_seed=1000,
            bootstrap_draws=10000,
        )
        updated["coverage_complete"] = summary.get("coverage_complete", False)
        updated["qualification_blockers"] = summary.get("qualification_blockers", [])
        write_json(summary_path, updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", action="append", type=Path, required=True)
    args = parser.parse_args()
    attach([resolve_path(path) for path in args.result], resolve_path(args.baseline))


if __name__ == "__main__":
    main()
