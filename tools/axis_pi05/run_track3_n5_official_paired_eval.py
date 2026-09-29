#!/usr/bin/env python3
"""N5 paired frozen eval: R6 baseline vs official T3 step-100.

This is not Q5 step-10 eval, not a training launcher, and not an official
competition result.  It reuses the frozen reverse-dev 4 + C20 20 identities
and execute8 protocol.  Continuation to optimizer step 300 is authorized only
when reverse gains at least +1/4 and C20 drops by at most 2/20.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file
from . import run_track3_q5_step10_frozen_eval as q7


N5_PLAN = ROOT / "docs/AXIS_PI05_TRACK3_POST_Q5_SEMANTIC_SIGNOFF_AND_OFFICIAL_STEP100_PLAN_20260910.md"
DEFAULT_EVIDENCE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/n5_official_t3_step100_paired_eval"
DEFAULT_OUTPUT = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/n5_official_t3_step100_paired_eval"
DEFAULT_R6 = q7.R6_PARAMS
DEFAULT_CANDIDATE = ROOT / "outputs/axis_pi05/official_canary100/t3_four_gold_canary100/100"
Q7_NAMESPACE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval"


def _live_train() -> str:
    result = subprocess.run(["pgrep", "-af", "scripts/train.py"], capture_output=True, text=True)
    return result.stdout.strip()


def _gpu_rows() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, name, uuid, used, util = [part.strip() for part in line.split(",", 4)]
        rows.append({"index": int(index), "name": name, "uuid": uuid, "memory_used_mib": int(used), "utilization_gpu_percent": int(util)})
    return rows


def assert_eval_gpu_allowed(gpu: int) -> dict[str, Any]:
    rows = _gpu_rows()
    selected = next((row for row in rows if row["index"] == gpu), None)
    if selected is None:
        raise RuntimeError(f"GPU {gpu} is not visible: {rows}")
    name = selected["name"].lower()
    blackwell = "blackwell" in name or "rtx pro 6000" in name
    if gpu == 0 and not blackwell:
        raise RuntimeError("refusing local GPU0 (YOLO boundary); N5 on this machine must use GPU3, or run on official Blackwell")
    if selected["memory_used_mib"] > 64 or selected["utilization_gpu_percent"] != 0:
        raise RuntimeError(f"eval GPU is not idle: {selected}")
    live = _live_train()
    if "train.py" in live:
        raise RuntimeError(f"refusing eval while train.py is running:\n{live}")
    return selected


def validate_r6_params(path: Path) -> dict[str, Any]:
    params = path / "params" if (path / "params").is_dir() else path
    if not params.is_dir():
        raise FileNotFoundError(f"R6 params directory missing: {path}")
    checkpoint = params.parent if params.name == "params" else path
    if any(".partial" in item.name for item in params.rglob("*")):
        raise ValueError(f"partial residue in R6 params: {params}")
    return {
        "step": 0,
        "role": "r6_baseline",
        "path": str(checkpoint.resolve()),
        "metadata_sha256": None,
        "params_tree_sha256": q7.sha256_tree(params),
        "train_state_present": False,
    }


def validate_step100(path: Path) -> dict[str, Any]:
    params = path / "params"
    train_state = path / "train_state"
    metadata = path / "_CHECKPOINT_METADATA"
    if not params.is_dir() or not train_state.is_dir():
        raise FileNotFoundError(f"incomplete T3 step100 checkpoint: {path}")
    if any(".partial" in item.name for item in path.rglob("*")):
        raise ValueError(f"partial residue in T3 step100: {path}")
    return {
        "step": 100,
        "role": "t3_official_canary100",
        "path": str(path.resolve()),
        "metadata_sha256": sha256_file(metadata) if metadata.is_file() else None,
        "params_tree_sha256": q7.sha256_tree(params),
        "train_state_present": True,
    }


def load_frozen_cases() -> dict[str, Any]:
    freeze = q7.strict_json_load(q7.EVALUATION_FREEZE)
    if freeze.get("status") != "evaluation_set_frozen_before_q5":
        raise ValueError(f"evaluation freeze is not locked: {freeze.get('status')}")
    q7._assert_hash(q7.NORM_STATS, q7.EXPECTED_NORM_SHA)
    q7._assert_hash(q7.REVERSE_MANIFEST, q7.EXPECTED_REVERSE_SHA)
    q7._assert_hash(q7.C20_MANIFEST, q7.EXPECTED_C20_SHA)
    reverse_source = {row["case_id"]: row for row in q7.strict_jsonl_load(q7.REVERSE_MANIFEST)}
    c20_source = {row["case_id"]: row for row in q7.strict_jsonl_load(q7.C20_MANIFEST)}
    reverse_cases = []
    for frozen in freeze.get("reverse_dev", {}).get("rows", []):
        source = reverse_source.get(frozen["case_id"])
        if source is None:
            raise ValueError(f"reverse freeze case missing: {frozen['case_id']}")
        reverse_cases.append(q7._normalise_case(source, panel="reverse_dev", panel_manifest_sha=q7.EXPECTED_REVERSE_SHA, benchmark="libero_t3", suite="libero_t3", task_id=int(frozen["task_index"])))
    if len(reverse_cases) != 4:
        raise ValueError(f"reverse freeze must contain 4 cases, got {len(reverse_cases)}")
    c20_cases = []
    for frozen in freeze.get("c20_retention", {}).get("cases", []):
        source = c20_source.get(frozen["case_id"])
        if source is None:
            raise ValueError(f"C20 freeze case missing: {frozen['case_id']}")
        c20_cases.append(q7._normalise_case(source, panel="C20", panel_manifest_sha=q7.EXPECTED_C20_SHA, benchmark="libero_mix", suite="C20", task_id=int(source["official_task_index"])))
    if len(c20_cases) != 20:
        raise ValueError(f"C20 freeze must contain 20 cases, got {len(c20_cases)}")
    return {"reverse_cases": reverse_cases, "c20_cases": c20_cases, "all_cases": reverse_cases + c20_cases, "freeze": freeze}


def continuation_gates(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    reverse = [pair for pair in pairs if pair["panel"] == "reverse_dev"]
    c20 = [pair for pair in pairs if pair["panel"] == "C20"]
    reverse_base = sum(pair["baseline_success"] for pair in reverse)
    reverse_cand = sum(pair["candidate_success"] for pair in reverse)
    c20_base = sum(pair["baseline_success"] for pair in c20)
    c20_cand = sum(pair["candidate_success"] for pair in c20)
    reverse_delta = reverse_cand - reverse_base
    c20_delta = c20_cand - c20_base
    gate = {
        "coverage_24_paired": len(pairs) == 24 and len(reverse) == 4 and len(c20) == 20,
        "reverse_at_least_plus_1_of_4": reverse_delta >= 1,
        "c20_drop_at_most_2_of_20": c20_delta >= -2,
    }
    return {
        "reverse": {"baseline_success": reverse_base, "candidate_success": reverse_cand, "delta": reverse_delta, "n": 4},
        "c20": {"baseline_success": c20_base, "candidate_success": c20_cand, "delta": c20_delta, "n": 20},
        "gates": gate,
        "continue_to_step300_authorized": all(gate.values()),
    }


def paired_summary(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in baseline_rows}
    candidate = {row["case_id"]: row for row in candidate_rows}
    expected = [row["case_id"] for row in cases]
    if list(baseline) != expected or list(candidate) != expected:
        raise ValueError("paired case order does not match frozen case order")
    pairs = [{
        "case_id": case_id,
        "panel": baseline[case_id]["panel"],
        "baseline_success": bool(baseline[case_id]["success"]),
        "candidate_success": bool(candidate[case_id]["success"]),
        "success_delta": int(bool(candidate[case_id]["success"])) - int(bool(baseline[case_id]["success"])),
        "baseline_collision": bool(baseline[case_id]["collision"]),
        "candidate_collision": bool(candidate[case_id]["collision"]),
        "baseline_timeout": bool(baseline[case_id]["timeout"]),
        "candidate_timeout": bool(candidate[case_id]["timeout"]),
    } for case_id in expected]
    decision = continuation_gates(pairs)
    all_rows = baseline_rows + candidate_rows
    no_infra = q7._no_infra(all_rows)
    return {
        "schema_version": "track3_n5_r6_vs_step100_paired_summary_v1",
        "status": "evaluated" if len(pairs) == 24 else "failed",
        "continue_to_step300_authorized": bool(decision["continue_to_step300_authorized"] and no_infra),
        "official_result": False,
        "diagnostic_only": True,
        "training_steps": 0,
        "paired_case_count": len(pairs),
        "rollout_count": len(all_rows),
        "pairs": pairs,
        "reverse_dev": decision["reverse"],
        "c20": decision["c20"],
        "gates": {**decision["gates"], "no_infrastructure_model_or_nonfinite_exception": no_infra},
        "protocol": {"control_frequency_hz": q7.CONTROL_HZ, "chunk_size": q7.CHUNK_SIZE, "executed_actions_per_chunk": q7.EXECUTE_ACTIONS},
    }


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.evidence_dir.resolve() == Q7_NAMESPACE.resolve() or args.output_dir.resolve() == q7.OUTPUT_ROOT.resolve():
        raise ValueError("N5 must not write into the Q7 GPU3 frozen-eval namespace")
    selected = assert_eval_gpu_allowed(args.gpu)
    baseline = validate_r6_params(args.baseline)
    candidate = validate_step100(args.candidate)
    cases = load_frozen_cases()
    libero = (ROOT / "LIBERO-plus").is_dir()
    result = {
        "schema_version": "track3_n5_preflight_v1",
        "status": "passed",
        "training_steps": 0,
        "gpu": selected,
        "libero_plus_present": libero,
        "baseline": baseline,
        "candidate": candidate,
        "case_counts": {"reverse_dev": len(cases["reverse_cases"]), "c20": len(cases["c20_cases"]), "paired": 24, "rollouts": 48},
        "evidence_dir": str(args.evidence_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "q7_namespace": str(Q7_NAMESPACE),
        "official_result": False,
    }
    if not libero:
        result["status"] = "blocked_missing_libero"
        result["next"] = "N5 needs LIBERO-plus / EGL. Official trainvenv may not have it; then run on local GPU3 after copying T3 step100, never GPU0."
    return result


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    report = preflight(args)
    if report["status"] != "passed":
        return report
    if args.preflight_only:
        return report
    evidence = args.evidence_dir.resolve()
    output = args.output_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["AXIS_PI05_EVAL_GPU"] = str(args.gpu)
    q7.PYTHON = Path(args.python)
    q7.EVIDENCE_ROOT = evidence
    q7.OUTPUT_ROOT = output
    cases = load_frozen_cases()
    identities = {0: validate_r6_params(args.baseline), 100: validate_step100(args.candidate)}
    q7.atomic_json(evidence / "preflight.json", report)
    q7.atomic_json(evidence / "input_lock.json", {
        "baseline": identities[0],
        "candidate": identities[100],
        "evaluation_freeze_sha256": sha256_file(q7.EVALUATION_FREEZE),
        "norm_stats_sha256": q7.EXPECTED_NORM_SHA,
        "reverse_manifest_sha256": q7.EXPECTED_REVERSE_SHA,
        "c20_manifest_sha256": q7.EXPECTED_C20_SHA,
        "plan_sha256": sha256_file(N5_PLAN),
    })
    os.environ.update(q7.server_environment())
    for import_path in (ROOT, ROOT / "LIBERO-plus", q7.OPENPI_ROOT / "src"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    from .libero_asset_bridge import activate as activate_libero_asset_bridge
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    activate_libero_asset_bridge(ROOT)
    manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=300, seed=1000))
    task_cache: dict[str, Any] = {}
    cursor = {"schema_version": "track3_n5_cursor_v1", "status": "running", "pid": os.getpid(), "completed_case_ids": []}
    q7.atomic_json(evidence / "cursor.json", cursor)
    baseline_rows = q7._run_cases(cases["all_cases"], 0, "full", identities[0], manager, task_cache, output, cursor)
    candidate_rows = q7._run_cases(cases["all_cases"], 100, "full", identities[100], manager, task_cache, output, cursor)
    paired = paired_summary(baseline_rows, candidate_rows, cases["all_cases"])
    q7.atomic_json(evidence / "paired_summary.json", paired)
    cursor.update({"status": "passed" if paired["status"] == "evaluated" else "failed", "continue_to_step300_authorized": paired["continue_to_step300_authorized"]})
    q7.atomic_json(evidence / "cursor.json", cursor)
    return paired


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--python", default=str(q7.PYTHON))
    parser.add_argument("--baseline", type=Path, default=DEFAULT_R6)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_eval(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    if result.get("status") in {"passed", "evaluated", "blocked_missing_libero"}:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
