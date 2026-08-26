from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import ROOT, resolve_path, sha256_file, write_json


STEP_RE = re.compile(r"Step (?P<step>\d+): (?P<metrics>.*)")
METRIC_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>[-+0-9.eE]+)")


def build_report(run_dir: Path, expected_steps: int) -> dict:
    log_path = run_dir / "train.log"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    steps = []
    first_nonfinite = None
    for line in lines:
        match = STEP_RE.search(line)
        if not match:
            continue
        metrics = {}
        for metric in METRIC_RE.finditer(match.group("metrics")):
            metrics[metric.group("key")] = float(metric.group("value"))
        step = int(match.group("step"))
        steps.append({"step": step, **metrics})
        if first_nonfinite is None and any(
            key.startswith("nonfinite_") and value != 0 for key, value in metrics.items()
        ):
            first_nonfinite = step

    losses = [row["loss"] for row in steps if "loss" in row]
    action_grad = [row["action_proj_grad_norm"] for row in steps if "action_proj_grad_norm" in row]
    lora_grad = [row["lora_grad_norm"] for row in steps if "lora_grad_norm" in row]
    return {
        "schema_version": 2,
        "run_dir": str(run_dir),
        "log_sha256": sha256_file(log_path),
        "expected_steps": expected_steps,
        "observed_step_count": len(steps),
        "observed_first_step": steps[0]["step"] if steps else None,
        "observed_last_step": steps[-1]["step"] if steps else None,
        "all_steps_finite": first_nonfinite is None and len(steps) == expected_steps,
        "first_nonfinite_step": first_nonfinite,
        "loss_min": min(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "action_proj_grad_norm_max": max(action_grad) if action_grad else None,
        "lora_grad_norm_max": max(lora_grad) if lora_grad else None,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-step Axis Pi0.5 diagnostics")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(resolve_path(args.run_dir), args.expected_steps)
    write_json(resolve_path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
