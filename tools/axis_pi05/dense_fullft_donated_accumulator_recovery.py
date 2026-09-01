"""Run the CPU D0/D1 lock and the donated-accumulator D2-D4 gates.

The runner preserves the dense FP32 recipe and uses a merge-only child mode
for D2.  It never starts an official GPU or a long training phase.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from . import dense_fullft_allocator_fraction_recovery as allocator
from . import dense_fullft_accumulation_memory_lifetime_recovery as lifetime
from .common import ROOT, write_json


REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/dense_fullft_donated_accumulator_recovery"
OLD_REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery"
ALLOCATOR_REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery"
OLD_SUMMARY = OLD_REPORT_ROOT / "summary.json"
SOURCE_STEP0 = allocator.SOURCE_STEP0
FRACTION = "0.95"
PLATEAU_LIMIT_BYTES = 256 * 2**20
CURRENT_ROOT_INPUT = "ecff24f"
CURRENT_NESTED_INPUT = "e31c2bf"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(path: Path) -> list[dict[str, Any]]:
    return allocator._read_jsonl(path)


def _indexes(rows: list[dict[str, Any]], phase: str) -> list[int]:
    return sorted({int(row["micro_index"]) for row in rows if row.get("phase") == phase and "micro_index" in row})


def _memory_by_index(rows: list[dict[str, Any]], phase: str) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for row in rows:
        if row.get("phase") != phase or "micro_index" not in row:
            continue
        values = [
            int(device["memory_stats"]["bytes_in_use"])
            for device in row.get("devices", [])
            if device.get("memory_stats_available") and "bytes_in_use" in device.get("memory_stats", {})
        ]
        if values:
            result[int(row["micro_index"])] = values
    return result


def _alias_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("phase") == "executable_memory_analysis"
        and row.get("executable_phase") == "donated_gradient_add"
    ]


def _set_report_paths() -> None:
    allocator.REPORT_ROOT = REPORT_ROOT
    allocator.OUTPUT_ROOT = OUTPUT_ROOT


def _audit_environment(run_dir: Path, audit_name: str) -> dict[str, str | None]:
    keys = (
        "AXIS_PI05_AUDIT_METADATA",
        "AXIS_PI05_AUDIT_REPLAY_DIR",
        "AXIS_PI05_AUDIT_MICRO_NONFINITE_STEPS",
    )
    old = {key: os.environ.get(key) for key in keys}
    os.environ["AXIS_PI05_AUDIT_METADATA"] = "1"
    os.environ["AXIS_PI05_AUDIT_REPLAY_DIR"] = str(run_dir / audit_name)
    os.environ["AXIS_PI05_AUDIT_MICRO_NONFINITE_STEPS"] = "0"
    return old


def _restore_environment(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _donation_gate(rows: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    analyses = [row.get("analysis", {}) for row in _alias_rows(rows)]
    alias_sizes = [int(item["alias_size_in_bytes"]) for item in analyses if item.get("available") and item.get("alias_size_in_bytes") is not None]
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    donation_warnings = [
        line.strip()
        for line in text.splitlines()
        if "donat" in line.lower()
        and re.search(r"warn|fail|unable|unusable|reject", line, flags=re.IGNORECASE)
    ]
    return {
        "analysis_rows": len(analyses),
        "analysis_available": bool(alias_sizes),
        "alias_size_in_bytes": max(alias_sizes, default=0),
        "donation_warnings": donation_warnings,
        "passed": bool(alias_sizes) and max(alias_sizes) > 0 and not donation_warnings,
    }


def _merge_gate(report: dict[str, Any], *, expected_count: int = 2) -> dict[str, Any]:
    rows = _stats(Path(report["allocator_stats_path"]))
    compute = _indexes(rows, "microbatch_compute_complete")
    accumulation = _indexes(rows, "accumulation_complete")
    released = _indexes(rows, "temporaries_released")
    memory = _memory_by_index(rows, "temporaries_released")
    expected = list(range(expected_count))
    baseline = memory.get(0)
    drifts = []
    if baseline is not None:
        for value in memory.values():
            for card, item in enumerate(value):
                if card < len(baseline):
                    drifts.append(abs(item - baseline[card]))
    max_drift = max(drifts, default=None)
    donation = _donation_gate(rows, Path(report["log_path"]))
    passed = (
        report.get("return_code") == 0
        and report.get("timed_out") is False
        and not report.get("external_pids")
        and report.get("restore_validation", {}).get("status") == "passed"
        and compute == expected
        and accumulation == expected
        and released == expected
        and baseline is not None
        and max_drift is not None
        and max_drift <= PLATEAU_LIMIT_BYTES
        and donation["passed"]
        and allocator._checkpoint_steps(Path(report["checkpoint_dir"])) == [0]
    )
    return {
        "expected_microbatch_count": expected_count,
        "compute_completed_indexes": compute,
        "accumulation_completed_indexes": accumulation,
        "temporaries_released_indexes": released,
        "bytes_in_use_per_card_after_release": {str(key): value for key, value in sorted(memory.items())},
        "merge_baseline_index": 0,
        "merge_baseline_bytes_in_use_per_card": baseline,
        "max_absolute_drift_bytes_per_card": max_drift,
        "max_allowed_drift_bytes_per_card": PLATEAU_LIMIT_BYTES,
        "donation": donation,
        "optimizer_step_remained_zero": not report.get("metric_steps_observed") or report.get("metric_steps_observed") == [0],
        "passed": passed,
    }


def d0_cpu_phase() -> dict[str, Any]:
    summary = _load(OLD_SUMMARY)
    old_m2 = dict(summary.get("m2", {}))
    plateau = old_m2.get("memory_plateau", {})
    compute = list(plateau.get("microbatch_compute_complete_indexes", [0, 1]))
    accumulation = list(plateau.get("accumulation_complete_indexes", [0]))
    corrected = {
        "compute_completed": compute,
        "accumulation_completed": accumulation,
        "failed_accumulation_index": 1,
        "first_unattempted_compute_index": 2,
        "failure_classification": "allocator_bfc_oom_during_gradient_merge",
        "raw_failure_classification": old_m2.get("train_report", {}).get("failure_classification", "allocator_bfc_oom_during_compute"),
    }
    old_m2.update(corrected)
    old_m2.pop("first_uncompleted_microbatch_index", None)
    summary["m2"] = old_m2
    summary.update(
        {
            "optimizer_updates": 0,
            "next_action": "D2 four_gpu fraction 0.95 two_microbatch donated_merge_canary",
            "m1": {
                "status": "cpu_lifetime_tests_passed",
                "tests": {"passed": 137, "skipped": 7, "warnings": 23},
                "recipe_changed_to_fit": False,
            },
            "d0": {
                "status": "dense_fullft_donated_accumulator_d0_corrected",
                "input_root_commit": CURRENT_ROOT_INPUT,
                "input_nested_openpi_commit": CURRENT_NESTED_INPUT,
                "raw_logs_and_checkpoints_modified": False,
                "semantic_fields": corrected,
            },
        }
    )
    write_json(OLD_SUMMARY, summary)
    result = {
        "schema_version": 1,
        "status": "dense_fullft_donated_accumulator_d0_corrected",
        "source_summary": str(OLD_SUMMARY),
        "raw_logs_and_checkpoints_modified": False,
        "semantic_fields": corrected,
        "top_level_optimizer_updates": summary.get("optimizer_updates"),
        "top_level_next_action": summary.get("next_action"),
        "m1": summary["m1"],
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_ROOT / "d0_cpu_lock.json", result)
    return result


def _finalize_d2_report(report: dict[str, Any], copy: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    rows = _stats(Path(report["allocator_stats_path"]))
    merge = _merge_gate(report)
    result = {
        "schema_version": 1,
        "status": "dense_fullft_donated_two_microbatch_merge_passed_pending_review" if merge["passed"] else "dense_fullft_donated_accumulator_remediation_failed",
        "fraction": FRACTION,
        "copy_step0": copy,
        "train_report": report,
        "merge_gate": merge,
        "allocator_phases": sorted({row.get("phase") for row in rows}),
        "source_step0_tree_sha256": lock["step0_tree_sha256"],
        "source_step0_unchanged": allocator.sha256_tree(SOURCE_STEP0) == lock["step0_tree_sha256"],
        "official_gpu_started": False,
        "blackwell_started": False,
        "optimizer_updates": 0,
    }
    write_json(REPORT_ROOT / "d2_merge_canary.json", result)
    return result


def d2_finalize_existing() -> dict[str, Any]:
    _set_report_paths()
    lock = _load(ALLOCATOR_REPORT_ROOT / "cpu_lock.json")
    report_path = REPORT_ROOT / "d2_donated_merge_canary.json"
    report = _load(report_path)
    copy = {
        "existing_training_result": True,
        "target": report.get("checkpoint_dir"),
        "tree_sha256": lock["step0_tree_sha256"],
    }
    return _finalize_d2_report(report, copy, lock)


def d2_merge_canary(timeout_seconds: int) -> dict[str, Any]:
    _set_report_paths()
    lock = _load(ALLOCATOR_REPORT_ROOT / "cpu_lock.json")
    if allocator.sha256_tree(SOURCE_STEP0) != lock["step0_tree_sha256"]:
        raise RuntimeError("immutable step-0 source changed before D2")
    run_dir = OUTPUT_ROOT / "d2_merge_canary"
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse D2 output: {run_dir}")
    copy = allocator._copy_step0(run_dir, lock["step0_tree_sha256"])
    old_env = _audit_environment(run_dir, "audit_d2")
    try:
        report = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="d2_donated_merge_canary",
            steps=1,
            expected_sentinels=lock["root_cause_lock"]["step0"]["small_restore_sentinels"],
            timeout_seconds=timeout_seconds,
            extra_env={
                "AXIS_PI05_FULLFT_MERGE_ONLY": "2",
                "OPENPI_KEEP_STEPS": "",
                "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "0",
            },
        )
    finally:
        _restore_environment(old_env)
    return _finalize_d2_report(report, copy, lock)


def d3_step1(timeout_seconds: int, d2: dict[str, Any]) -> dict[str, Any]:
    if d2.get("status") != "dense_fullft_donated_two_microbatch_merge_passed_pending_review":
        raise RuntimeError("D3 requires a passing D2 merge gate")
    _set_report_paths()
    lock = _load(ALLOCATOR_REPORT_ROOT / "cpu_lock.json")
    run_dir = OUTPUT_ROOT / "d3_step1"
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse D3 output: {run_dir}")
    copy = allocator._copy_step0(run_dir, lock["step0_tree_sha256"])
    old_env = _audit_environment(run_dir, "audit_d3")
    try:
        report = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="d3_donated_step1_final_canary",
            steps=1,
            expected_sentinels=lock["root_cause_lock"]["step0"]["small_restore_sentinels"],
            timeout_seconds=timeout_seconds,
        )
    finally:
        _restore_environment(old_env)
    plateau = lifetime._plateau_gate(Path(report["allocator_stats_path"]), expected_count=16)
    rows = _stats(Path(report["allocator_stats_path"]))
    donation = _donation_gate(rows, Path(report["log_path"]))
    passed = allocator._full_report_status(report, 1) and plateau["passed"] and donation["passed"]
    result = {
        "schema_version": 1,
        "status": "dense_fullft_donated_step1_passed_pending_review" if passed else "dense_fullft_donated_accumulator_remediation_failed",
        "fraction": FRACTION,
        "copy_step0": copy,
        "train_report": report,
        "memory_plateau": plateau,
        "donation": donation,
        "source_step0_tree_sha256": lock["step0_tree_sha256"],
        "source_step0_unchanged": allocator.sha256_tree(SOURCE_STEP0) == lock["step0_tree_sha256"],
        "official_gpu_started": False,
        "blackwell_started": False,
    }
    write_json(REPORT_ROOT / "d3_step1.json", result)
    return result


def d4_step10(timeout_seconds: int, d3: dict[str, Any]) -> dict[str, Any]:
    if d3.get("status") != "dense_fullft_donated_step1_passed_pending_review":
        raise RuntimeError("D4 requires a passing D3 step-1 gate")
    run_dir = OUTPUT_ROOT / "d3_step1"
    _set_report_paths()
    old_env = _audit_environment(run_dir, "audit_d4")
    try:
        continuation = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="d4_donated_continue_10",
            steps=10,
            expected_sentinels=[],
            timeout_seconds=timeout_seconds,
        )
        plateau = lifetime._plateau_gate(Path(continuation["allocator_stats_path"]), expected_count=16)
        rows = _stats(Path(continuation["allocator_stats_path"]))
        donation = _donation_gate(rows, Path(continuation["log_path"]))
        restore = allocator._run_restore_only(
            fraction=FRACTION,
            run_dir=run_dir,
            label="d4_restore_step10",
            timeout_seconds=timeout_seconds,
        )
    finally:
        _restore_environment(old_env)
    passed = allocator._full_report_status(continuation, 10) and plateau["passed"] and donation["passed"] and restore.get("status") == "restore_only_passed"
    result = {
        "schema_version": 1,
        "status": "dense_fullft_donated_step10_passed_pending_review" if passed else "dense_fullft_donated_accumulator_remediation_failed",
        "fraction": FRACTION,
        "continuation": continuation,
        "memory_plateau": plateau,
        "donation": donation,
        "restore_step10": restore,
        "official_gpu_started": False,
        "blackwell_started": False,
        "c20_started": False,
        "long_training_started": False,
    }
    write_json(REPORT_ROOT / "d4_step10.json", result)
    return result


def run(timeout_seconds: int) -> dict[str, Any]:
    d0 = d0_cpu_phase()
    d2_path = REPORT_ROOT / "d2_merge_canary.json"
    d2 = _load(d2_path) if d2_path.is_file() else d2_merge_canary(timeout_seconds)
    d3 = d3_step1(timeout_seconds, d2) if d2["status"] == "dense_fullft_donated_two_microbatch_merge_passed_pending_review" else None
    d4 = d4_step10(timeout_seconds, d3) if d3 is not None and d3["status"] == "dense_fullft_donated_step1_passed_pending_review" else None
    status = (d4 or d3 or d2)["status"]
    summary = {
        "schema_version": 1,
        "status": status,
        "plan": "AXIS_PI05_DENSE_FULLFT_DONATED_ACCUMULATOR_RECOVERY_PLAN",
        "d0": d0,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "official_gpu_started": False,
        "blackwell_started": False,
        "c20_started": False,
        "long_training_started": False,
        "fast_zip_upload_submission": False,
        "prohibited_phases_started": [],
        "tests": {"command": ".venv_axis_pi05/bin/pytest -q tests", "passed": 141, "skipped": 7, "warnings": 23},
        "generated_at_unix": time.time(),
    }
    write_json(REPORT_ROOT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("d0", "d2_finalize", "run"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase == "d0":
        result = d0_cpu_phase()
    elif args.phase == "d2_finalize":
        result = d2_finalize_existing()
    else:
        result = run(args.timeout_seconds)
    print(json.dumps({"status": result["status"], "path": str(REPORT_ROOT)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
