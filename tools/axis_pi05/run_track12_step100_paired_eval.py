#!/usr/bin/env python3
"""T1/T2 step100 vs R6 C20 paired eval on local GPU3.

Dedicated Plus panels are not frozen, so continue-to-250 stays unauthorized
even if C20 retention passes.  Training steps remain 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file
from . import run_track3_n5_official_paired_eval as n5
from . import run_track3_q5_step10_frozen_eval as q7


# Lock the OpenPI checkpoint assets copy from official canary100, not the
# pre-train source JSON. Those source files hashed to e4f41c9d… (T1) and
# a98b923e… (T2); OpenPI dropped unused q10/q50/q90 and pretty-printed.
TRACKS = {
    "T1": {
        "config_name": "axis_pi05_track12_lora",
        "candidate": ROOT / "outputs/axis_pi05/official_canary100/t1_p20_canary100/100",
        "evidence": ROOT / "data/reports/axis_libero_30k/track12_t1_p20_step100_c20_eval",
        "output": ROOT / "outputs/axis_pi05/track12_t1_p20_step100_c20_eval",
        "expected_norm_sha256": "f49d37dac32533048a14039570ab99ef2e8fd693f0081544807e72a7ceffa50c",
    },
    "T2": {
        "config_name": "axis_pi05_track12_lora",
        "candidate": ROOT / "outputs/axis_pi05/official_canary100/t2_p30_canary100/100",
        "evidence": ROOT / "data/reports/axis_libero_30k/track12_t2_p30_step100_c20_eval",
        "output": ROOT / "outputs/axis_pi05/track12_t2_p30_step100_c20_eval",
        "expected_norm_sha256": "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41",
    },
}


def find_norm(candidate: Path) -> Path:
    hits = [path for path in candidate.rglob("norm_stats.json") if ".cache" not in path.parts]
    if len(hits) != 1:
        raise FileNotFoundError(f"need exactly one norm_stats.json under {candidate}, found {hits}")
    return hits[0]


def load_c20_cases() -> list[dict[str, Any]]:
    freeze = q7.strict_json_load(q7.EVALUATION_FREEZE)
    if freeze.get("status") != "evaluation_set_frozen_before_q5":
        raise ValueError(f"evaluation freeze is not locked: {freeze.get('status')}")
    q7._assert_hash(q7.C20_MANIFEST, q7.EXPECTED_C20_SHA)
    source = {row["case_id"]: row for row in q7.strict_jsonl_load(q7.C20_MANIFEST)}
    cases = []
    for frozen in freeze.get("c20_retention", {}).get("cases", []):
        row = source.get(frozen["case_id"])
        if row is None:
            raise ValueError(f"C20 freeze case missing: {frozen['case_id']}")
        cases.append(q7._normalise_case(row, panel="C20", panel_manifest_sha=q7.EXPECTED_C20_SHA, benchmark="libero_mix", suite="C20", task_id=int(row["official_task_index"])))
    if len(cases) != 20:
        raise ValueError(f"C20 freeze must contain 20 cases, got {len(cases)}")
    return cases


def c20_gates(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    c20 = [pair for pair in pairs if pair["panel"] == "C20"]
    base = sum(pair["baseline_success"] for pair in c20)
    cand = sum(pair["candidate_success"] for pair in c20)
    delta = cand - base
    return {
        "c20": {"baseline_success": base, "candidate_success": cand, "delta": delta, "n": 20},
        "gates": {
            "coverage_20_paired": len(pairs) == 20 and len(c20) == 20,
            "c20_drop_at_most_2_of_20": delta >= -2,
            "dedicated_panel_frozen": False,
            "dedicated_panel_plus_2": False,
        },
        "continue_to_step250_authorized": False,
        "continue_blocked_reason": "dedicated_plus_panel_not_frozen",
    }


def run_track(track: str, *, gpu: int = 3) -> dict[str, Any]:
    spec = TRACKS[track]
    candidate = spec["candidate"]
    evidence = spec["evidence"]
    output = spec["output"]
    selected = n5.assert_eval_gpu_allowed(gpu)
    baseline = n5.validate_r6_params(n5.DEFAULT_R6)
    identity = n5.validate_step100(candidate)
    identity["role"] = f"{track.lower()}_official_canary100"
    norm = find_norm(candidate)
    actual_norm = sha256_file(norm)
    if actual_norm != spec["expected_norm_sha256"]:
        raise ValueError(f"{track} norm SHA mismatch: {actual_norm} != {spec['expected_norm_sha256']}")
    cases = load_c20_cases()
    evidence.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    q7.CONFIG_NAME = spec["config_name"]
    q7.NORM_STATS = norm
    q7.PYTHON = Path(str(q7.PYTHON))
    q7.EVIDENCE_ROOT = evidence
    q7.OUTPUT_ROOT = output
    os.environ["AXIS_PI05_EVAL_GPU"] = str(gpu)
    os.environ.update(q7.server_environment())
    for import_path in (ROOT, ROOT / "LIBERO-plus", q7.OPENPI_ROOT / "src"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    from .libero_asset_bridge import activate as activate_libero_asset_bridge
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    activate_libero_asset_bridge(ROOT)
    manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=300, seed=1000))
    cursor = {"schema_version": "track12_step100_c20_cursor_v1", "status": "running", "track": track, "pid": os.getpid(), "completed_case_ids": []}
    q7.atomic_json(evidence / "cursor.json", cursor)
    baseline_rows = q7._run_cases(cases, 0, "full", baseline, manager, {}, output, cursor)
    candidate_rows = q7._run_cases(cases, 100, "full", identity, manager, {}, output, cursor)
    pairs = n5.paired_summary(baseline_rows, candidate_rows, cases)
    decision = c20_gates(pairs["pairs"])
    summary = {
        "schema_version": "track12_step100_c20_paired_summary_v1",
        "track": track,
        "status": "evaluated" if len(pairs["pairs"]) == 20 else "failed",
        "continue_to_step250_authorized": False,
        "continue_blocked_reason": decision["continue_blocked_reason"],
        "official_result": False,
        "diagnostic_only": True,
        "training_steps": 0,
        "paired_case_count": len(pairs["pairs"]),
        "rollout_count": 40,
        "pairs": pairs["pairs"],
        "c20": decision["c20"],
        "gates": {**decision["gates"], "no_infrastructure_model_or_nonfinite_exception": q7._no_infra(baseline_rows + candidate_rows)},
        "norm_stats_sha256": actual_norm,
        "config_name": spec["config_name"],
        "gpu": selected,
        "dedicated_panel": "not_frozen",
    }
    q7.atomic_json(evidence / "paired_summary.json", summary)
    cursor.update({"status": summary["status"], "continue_to_step250_authorized": False})
    q7.atomic_json(evidence / "cursor.json", cursor)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("T1", "T2"), required=True)
    parser.add_argument("--gpu", type=int, default=3)
    args = parser.parse_args(argv)
    result = run_track(args.track, gpu=args.gpu)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("status") == "evaluated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
