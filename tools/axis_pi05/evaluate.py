from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from .common import ROOT, git_info, load_yaml, resolve_path, sha256_file, write_csv, write_json
from .checkpoints import validate_internal_step


def summarize(result: dict[str, Any], *, run_id: str, step: int) -> dict[str, Any]:
    episodes = result.get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []
    def mean(key: str, rows: list[dict[str, Any]] = episodes) -> float | None:
        values = [float(row[key]) for row in rows if isinstance(row, dict) and isinstance(row.get(key), (int, float))]
        return sum(values) / len(values) if values else None
    sources = {}
    for source in sorted({row.get("source", "unavailable") for row in episodes if isinstance(row, dict)}):
        subset = [row for row in episodes if row.get("source", "unavailable") == source]
        sources[source] = {"success": mean("success", subset), "collision_rate": mean("collision", subset), "count": len(subset)}
    return {
        "schema_version": 1,
        "run_id": run_id,
        "internal_step": step,
        "episode_count": len(episodes),
        "success": mean("success"),
        "collision_rate": mean("collision"),
        "timeout_rate": mean("timeout"),
        "goal_reached_but_collision": mean("goal_reached_but_collision"),
        "source_macro": sources,
        "official_result": result.get("official_result", False),
        "diagnostic_only": not bool(result.get("official_result", False)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="evaluate fixed checkpoints through the current official pipeline adapter")
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", default="500,1000,3000")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/eval.yaml")
    ap.add_argument("--command", help="override AXIS_PI05_EVAL_COMMAND")
    args = ap.parse_args()
    root = ROOT
    run_root = resolve_path(args.run, root)
    eval_config = load_yaml(resolve_path(args.config))
    command_text = args.command or os.environ.get("AXIS_PI05_EVAL_COMMAND")
    summaries = []
    for raw_step in args.steps.split(","):
        step = validate_internal_step(Path(str(raw_step)))
        checkpoint = run_root / str(step)
        if not checkpoint.exists():
            checkpoint = run_root / "checkpoints" / str(step)
        if not checkpoint.exists():
            raise SystemExit(f"找不到 internal step checkpoint: {step} under {run_root}")
        result_path = checkpoint / "evaluation_result.json"
        if command_text:
            env = os.environ.copy()
            env.update({"AXIS_PI05_CHECKPOINT": str(checkpoint), "AXIS_PI05_EVAL_RUN": str(run_root), "AXIS_PI05_EVAL_STEP": str(step)})
            subprocess.run(shlex.split(command_text), cwd=root, env=env, check=True)
        if not result_path.is_file():
            raise SystemExit(f"evaluator 未写出 {result_path}；不会生成虚假的 0 分报告")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summaries.append(summarize(result, run_id=run_root.name, step=step))
        review = root / "reviews" / "axis_pi05" / run_root.name / str(step)
        review.mkdir(parents=True, exist_ok=True)
        write_json(review / "summary.json", summaries[-1])
        write_json(review / "compare_to_axis_init.json", {"status": "requires_fixed_axis_init_result", "source": "official evaluator adapter"})
        episodes = result.get("episodes", []) if isinstance(result, dict) else []
        write_csv(review / "episodes.csv", episodes, sorted({key for row in episodes if isinstance(row, dict) for key in row}))
    write_json(run_root / "evaluation_summary.json", {"schema_version": 1, "config": eval_config, "git": git_info(root), "summaries": summaries})
    print(json.dumps({"run": str(run_root), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
