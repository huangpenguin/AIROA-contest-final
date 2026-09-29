"""Run the final R6 inference-only execute-horizon screen.

This adapter intentionally reuses the frozen 20 Hz 4-reverse + C20 worker,
but binds a new adapter/plan identity and writes one namespace per horizon.
It never trains, folds, packages, uploads, or submits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import run_track3_fast400_h300_paired_eval as base


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/AXIS_PI05_R6_FINAL_INFERENCE_HORIZON_SWEEP_PLAN_20260916.md"
ADAPTER = Path(__file__).resolve()
EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/r6_final_inference_horizon_sweep_20260916_attempt_0001"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/r6_final_inference_horizon_sweep_20260916_attempt_0001"
HORIZONS = (4, 6, 8, 10)
BASELINE_HORIZON = 8


def configure(horizon: int) -> None:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    base.EXECUTE_ACTIONS = int(horizon)
    base.ADAPTER_PATH = ADAPTER
    base.PLAN = PLAN


def variant_roots(horizon: int) -> tuple[Path, Path]:
    configure(horizon)
    return EVIDENCE_ROOT / f"execute{horizon}", OUTPUT_ROOT / f"execute{horizon}"


def preflight(horizon: int, gpu: int) -> dict[str, Any]:
    evidence, output = variant_roots(horizon)
    report = base.preflight(role="r6", gpu=gpu, evidence_dir=evidence, output_dir=output)
    if report["protocol"]["executed_actions_per_chunk"] != horizon:
        raise RuntimeError("preflight protocol did not bind requested horizon")
    return report


def worker(horizon: int, gpu: int, *, resume: bool = False) -> dict[str, Any]:
    evidence, output = variant_roots(horizon)
    result = base.run_worker(
        role="r6",
        gpu=gpu,
        resume=resume,
        evidence_dir=evidence,
        output_dir=output,
        python=base.PYTHON,
    )
    if result.get("protocol", {}).get("executed_actions_per_chunk") != horizon:
        raise RuntimeError("worker summary protocol did not bind requested horizon")
    return result


def _variant_rows(root: Path, horizon: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    role_root = root / f"execute{horizon}" / "r6"
    summary = base.strict_json_load(role_root / "summary.json")
    if summary.get("status") != "completed" or summary.get("case_count") != 24:
        raise ValueError(f"execute{horizon} is not a completed 24-case worker")
    if summary.get("protocol", {}).get("executed_actions_per_chunk") != horizon:
        raise ValueError(f"execute{horizon} summary protocol mismatch")
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted((role_root / "cases").glob("*.json")):
        row = base.strict_json_load(path)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in rows:
            raise ValueError(f"invalid or duplicate case in {path}")
        trace = Path(str(row.get("action_trace_path", "")))
        actions = np.load(trace, allow_pickle=False)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"invalid trace for {case_id}: {trace}")
        row = dict(row)
        row["trace_nonzero"] = bool(actions.size and np.any(np.abs(actions) > 1e-8))
        rows[case_id] = row
    if len(rows) != 24:
        raise ValueError(f"execute{horizon} has {len(rows)} case files, expected 24")
    return summary, rows


def _panel_metrics(rows: dict[str, dict[str, Any]], panel: str) -> dict[str, int]:
    selected = [row for row in rows.values() if row.get("panel") == panel]
    return {
        "n": len(selected),
        "success": sum(bool(row.get("success")) for row in selected),
        "collision": sum(bool(row.get("collision")) for row in selected),
        "timeout": sum(bool(row.get("timeout")) for row in selected),
        "exception": sum(row.get("failure_reason") == "exception" for row in selected),
    }


def compare(root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    loaded = {horizon: _variant_rows(root, horizon) for horizon in HORIZONS}
    baseline_rows = loaded[BASELINE_HORIZON][1]
    baseline_ids = set(baseline_rows)
    baseline_fingerprints = {case_id: row.get("case_fingerprint") for case_id, row in baseline_rows.items()}
    baseline_c20 = _panel_metrics(baseline_rows, "C20")
    baseline_reverse = _panel_metrics(baseline_rows, "reverse_dev")
    variants: dict[str, Any] = {}
    passing: list[int] = []
    for horizon, (_summary, rows) in loaded.items():
        if set(rows) != baseline_ids:
            raise ValueError(f"execute{horizon} case IDs differ from execute8")
        if {case_id: row.get("case_fingerprint") for case_id, row in rows.items()} != baseline_fingerprints:
            raise ValueError(f"execute{horizon} case fingerprints differ from execute8")
        c20 = _panel_metrics(rows, "C20")
        reverse = _panel_metrics(rows, "reverse_dev")
        wins = sum(not bool(baseline_rows[cid].get("success")) and bool(rows[cid].get("success")) for cid in baseline_ids if rows[cid].get("panel") == "C20")
        losses = sum(bool(baseline_rows[cid].get("success")) and not bool(rows[cid].get("success")) for cid in baseline_ids if rows[cid].get("panel") == "C20")
        mechanical = {
            "coverage_24": len(rows) == 24,
            "panel_counts": c20["n"] == 20 and reverse["n"] == 4,
            "no_exception": c20["exception"] == 0 and reverse["exception"] == 0,
            "finite_nonzero": all(bool(row.get("actions_finite")) and bool(row.get("trace_nonzero")) for row in rows.values()),
        }
        gates = {
            **mechanical,
            "c20_at_least_plus_3": c20["success"] - baseline_c20["success"] >= 3,
            "paired_net_wins_at_least_3": wins - losses >= 3,
            "collision_not_higher": c20["collision"] <= baseline_c20["collision"],
            "timeout_not_more_than_plus_1": c20["timeout"] <= baseline_c20["timeout"] + 1,
            "reverse_not_lower": reverse["success"] >= baseline_reverse["success"],
        }
        passed = horizon != BASELINE_HORIZON and all(gates.values())
        if passed:
            passing.append(horizon)
        variants[str(horizon)] = {
            "c20": c20,
            "reverse": reverse,
            "c20_delta": c20["success"] - baseline_c20["success"],
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_net_wins": wins - losses,
            "gates": gates,
            "passes_screen": passed,
        }
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda h: (
                variants[str(h)]["c20"]["success"],
                variants[str(h)]["paired_net_wins"],
                -variants[str(h)]["c20"]["collision"],
                -variants[str(h)]["c20"]["timeout"],
                -abs(h - BASELINE_HORIZON),
            ),
        )
    result = {
        "schema_version": "r6_final_inference_horizon_sweep_v1",
        "status": "screen_candidate_ready_for_b80_confirmation" if selected is not None else "screen_failed_keep_existing_r6_zip",
        "adapter": str(ADAPTER),
        "adapter_sha256": base.sha256_file(ADAPTER),
        "plan": str(PLAN),
        "plan_sha256": base.sha256_file(PLAN),
        "baseline_horizon": BASELINE_HORIZON,
        "baseline": {"c20": baseline_c20, "reverse": baseline_reverse},
        "variants": variants,
        "selected_horizon": selected,
        "training_performed": False,
        "fold_performed": False,
        "zip_generated": False,
        "auto_upload": False,
        "auto_submit": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    base.atomic_json(root / "comparison.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
        command.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
        if name == "worker":
            command.add_argument("--resume", action="store_true")
    sub.add_parser("compare")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.horizon, args.gpu)
    elif args.command == "worker":
        result = worker(args.horizon, args.gpu, resume=args.resume)
    else:
        result = compare()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

