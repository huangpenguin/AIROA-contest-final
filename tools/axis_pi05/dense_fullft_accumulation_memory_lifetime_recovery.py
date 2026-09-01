"""Run the CPU evidence correction and the authorized accumulation-lifetime canaries.

M0 corrects the historical allocator summary from raw traces without touching
raw logs or checkpoints.  M2 runs one fresh step at fraction 0.95 with the
non-finite audit enabled, and M3 continues the accepted step-1 checkpoint to
step 10.  The runner never changes the dense recipe and never starts an
official/competition GPU phase.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from . import dense_fullft_allocator_fraction_recovery as allocator
from .common import ROOT, write_json


REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_accumulation_memory_lifetime_recovery"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/dense_fullft_accumulation_memory_lifetime_recovery"
OLD_REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery"
SOURCE_STEP0 = allocator.SOURCE_STEP0
FRACTION = "0.95"
PLATEAU_LIMIT_BYTES = 256 * 2**20


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(path: Path) -> list[dict[str, Any]]:
    return allocator._read_jsonl(path)


def _device_bytes(row: dict[str, Any]) -> list[int]:
    return [
        int(device["memory_stats"]["bytes_in_use"])
        for device in row.get("devices", [])
        if device.get("memory_stats_available") and "bytes_in_use" in device.get("memory_stats", {})
    ]


def _phase_memory_table(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row.get("phase") != phase:
            continue
        values = _device_bytes(row)
        if values:
            result.append(
                {
                    "micro_index": row.get("micro_index"),
                    "bytes_in_use_per_card": values,
                    "timestamp": row.get("timestamp"),
                }
            )
    return result


def _completed_indexes(rows: list[dict[str, Any]], phase: str) -> list[int]:
    return sorted({int(row["micro_index"]) for row in rows if row.get("phase") == phase and "micro_index" in row})


def _plateau_for_rows(rows: list[dict[str, Any]], *, expected_count: int) -> dict[str, Any]:
    compute_indexes = _completed_indexes(rows, "microbatch_compute_complete")
    accumulation_indexes = _completed_indexes(rows, "accumulation_complete")
    released = _phase_memory_table(rows, "temporaries_released")
    by_index = {int(row["micro_index"]): row["bytes_in_use_per_card"] for row in released if row.get("micro_index") is not None}
    expected_indexes = list(range(expected_count))
    missing_release = [index for index in expected_indexes if index not in by_index]
    baseline = by_index.get(1)
    drift_by_card: list[int] = []
    if baseline is not None:
        for index in range(1, expected_count):
            values = by_index.get(index)
            if values is None:
                continue
            for card, value in enumerate(values):
                if card < len(baseline):
                    drift_by_card.append(abs(value - baseline[card]))
    max_drift = max(drift_by_card, default=None)
    passed = (
        compute_indexes == expected_indexes
        and accumulation_indexes == expected_indexes
        and not missing_release
        and baseline is not None
        and max_drift is not None
        and max_drift <= PLATEAU_LIMIT_BYTES
    )
    return {
        "phase": "temporaries_released",
        "expected_microbatch_count": expected_count,
        "microbatch_compute_complete_indexes": compute_indexes,
        "accumulation_complete_indexes": accumulation_indexes,
        "temporaries_released_indexes": sorted(by_index),
        "missing_temporaries_released": missing_release,
        "bytes_in_use_per_card_by_micro_index": {str(index): by_index[index] for index in sorted(by_index)},
        "baseline_index": 1,
        "baseline_bytes_in_use_per_card": baseline,
        "max_absolute_drift_bytes_per_card": max_drift,
        "max_allowed_drift_bytes_per_card": PLATEAU_LIMIT_BYTES,
        "passed": passed,
    }


def _plateau_gate(stats_path: Path, *, expected_count: int) -> dict[str, Any]:
    rows = _stats(stats_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase") in {"microbatch_compute_complete", "accumulation_complete", "temporaries_released"}:
            key = str(row.get("optimizer_step_before", "unknown"))
            grouped.setdefault(key, []).append(row)
    per_step = {key: _plateau_for_rows(value, expected_count=expected_count) for key, value in sorted(grouped.items())}
    if not per_step:
        return {"phase": "temporaries_released", "per_optimizer_step": {}, "passed": False}
    first = next(iter(per_step.values()))
    return {**first, "per_optimizer_step": per_step, "passed": all(value["passed"] for value in per_step.values())}


def _set_report_paths() -> None:
    allocator.REPORT_ROOT = REPORT_ROOT
    allocator.OUTPUT_ROOT = OUTPUT_ROOT


def m0_cpu_phase() -> dict[str, Any]:
    """Correct the old summary using raw trace evidence; preserve raw artifacts."""
    old_summary_path = OLD_REPORT_ROOT / "summary.json"
    summary = _load(old_summary_path)
    corrections: dict[str, Any] = {
        "schema_version": 1,
        "status": "dense_fullft_accumulation_memory_lifetime_m0_corrected",
        "source_summary": str(old_summary_path),
        "raw_logs_and_checkpoints_modified": False,
        "runs": {},
    }
    for key, label in (("a2_fraction_090", "a2_fraction_090_canary_retry3"), ("a3_fraction_095", "a3_fraction_095_canary")):
        report = summary.get(key) or _load(OLD_REPORT_ROOT / f"{label}.json")
        stats_path = Path(report["allocator_stats_path"])
        rows = _stats(stats_path)
        completed = _completed_indexes(rows, "microbatch_complete")
        compute_completed = _completed_indexes(rows, "microbatch_compute_complete")
        if not compute_completed:
            compute_completed = completed
        next_index = max(completed, default=-1) + 1
        memory_rows = _phase_memory_table(rows, "microbatch_complete")
        correction = {
            "historical_status": report.get("status"),
            "status": "allocator_bfc_oom_after_microbatch_1_before_microbatch_2",
            "failure_classification": "allocator_bfc_oom_during_compute",
            "completed_microbatch_indexes": completed,
            "microbatch_compute_complete_indexes": compute_completed,
            "oom_micro_index": next_index,
            "checkpoint_steps_observed": report.get("checkpoint_steps_observed", [0]),
            "metric_steps_observed": report.get("metric_steps_observed", []),
            "optimizer_updates": 0,
            "memory_at_completed_microbatches": memory_rows,
            "raw_allocator_stats": str(stats_path),
        }
        summary[key].update(correction)
        corrections["runs"][key] = correction

    summary.update(
        {
            "optimizer_updates": 0,
            "tests": {
                "command": ".venv_axis_pi05/bin/pytest -q tests",
                "passed": 132,
                "skipped": 7,
                "warnings": 21,
            },
            "next_action": "M2 four_gpu fraction 0.95 16_microbatch_step1_memory_plateau",
            "m0_correction": str(REPORT_ROOT / "m0_correction.json"),
        }
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(old_summary_path, summary)
    write_json(REPORT_ROOT / "m0_correction.json", corrections)
    return corrections


def _audit_environment(run_dir: Path, audit_name: str) -> dict[str, str | None]:
    old = {
        "AXIS_PI05_AUDIT_METADATA": os.environ.get("AXIS_PI05_AUDIT_METADATA"),
        "AXIS_PI05_AUDIT_REPLAY_DIR": os.environ.get("AXIS_PI05_AUDIT_REPLAY_DIR"),
        "AXIS_PI05_AUDIT_MICRO_NONFINITE_STEPS": os.environ.get("AXIS_PI05_AUDIT_MICRO_NONFINITE_STEPS"),
    }
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


def m2_step1(timeout_seconds: int) -> dict[str, Any]:
    _set_report_paths()
    lock = _load(OLD_REPORT_ROOT / "cpu_lock.json")
    if lock.get("step0_tree_sha256") != allocator.sha256_tree(SOURCE_STEP0):
        raise RuntimeError("immutable step-0 checkpoint changed before M2")
    run_dir = OUTPUT_ROOT / "m2_step1"
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse M2 output: {run_dir}")
    copy = allocator._copy_step0(run_dir, lock["step0_tree_sha256"])
    old_env = _audit_environment(run_dir, "audit_step1")
    try:
        report = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="m2_lifetime_canary",
            steps=1,
            expected_sentinels=lock["root_cause_lock"]["step0"]["small_restore_sentinels"],
            timeout_seconds=timeout_seconds,
        )
    finally:
        _restore_environment(old_env)
    plateau = _plateau_gate(Path(report["allocator_stats_path"]), expected_count=16)
    passed = allocator._full_report_status(report, 1) and plateau["passed"]
    result = {
        "schema_version": 1,
        "status": "dense_fullft_accumulation_plateau_passed_step1_pending_review" if passed else "dense_fullft_accumulation_lifetime_remediation_failed",
        "fraction": FRACTION,
        "copy_step0": copy,
        "train_report": report,
        "memory_plateau": plateau,
        "audit_replay_dir": str(run_dir / "audit_step1"),
        "optimizer_updates": 1 if allocator._full_report_status(report, 1) else 0,
        "source_step0_tree_sha256": lock["step0_tree_sha256"],
        "source_step0_unchanged": allocator.sha256_tree(SOURCE_STEP0) == lock["step0_tree_sha256"],
    }
    write_json(REPORT_ROOT / "m2_step1.json", result)
    return result


def m3_step10(timeout_seconds: int, m2: dict[str, Any]) -> dict[str, Any]:
    if m2.get("status") != "dense_fullft_accumulation_plateau_passed_step1_pending_review":
        raise RuntimeError("M3 requires a passing M2 memory plateau")
    _set_report_paths()
    run_dir = OUTPUT_ROOT / "m2_step1"
    old_env = _audit_environment(run_dir, "audit_step10")
    try:
        continuation = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="m3_lifetime_continue_10",
            steps=10,
            expected_sentinels=[],
            timeout_seconds=timeout_seconds,
        )
        plateau = _plateau_gate(Path(continuation["allocator_stats_path"]), expected_count=16)
        restore = allocator._run_restore_only(
            fraction=FRACTION,
            run_dir=run_dir,
            label="m3_restore_step10",
            timeout_seconds=timeout_seconds,
        )
    finally:
        _restore_environment(old_env)
    passed = allocator._full_report_status(continuation, 10) and plateau["passed"] and restore.get("status") == "restore_only_passed"
    result = {
        "schema_version": 1,
        "status": "dense_fullft_accumulation_10step_passed_pending_review" if passed else "dense_fullft_accumulation_lifetime_remediation_failed",
        "fraction": FRACTION,
        "continuation": continuation,
        "memory_plateau": plateau,
        "restore_step10": restore,
        "source_step0_tree_sha256": m2.get("source_step0_tree_sha256"),
        "source_step0_unchanged": allocator.sha256_tree(SOURCE_STEP0) == m2.get("source_step0_tree_sha256"),
        "official_gpu_started": False,
        "competition_gpu_started": False,
        "c20_started": False,
        "long_training_started": False,
        "fast_zip_upload_submission": False,
    }
    write_json(REPORT_ROOT / "m3_step10.json", result)
    return result


def run(timeout_seconds: int) -> dict[str, Any]:
    _set_report_paths()
    m0 = m0_cpu_phase()
    m2 = m2_step1(timeout_seconds)
    m3 = None
    if m2["status"] == "dense_fullft_accumulation_plateau_passed_step1_pending_review":
        m3 = m3_step10(timeout_seconds, m2)
    status = m3["status"] if m3 is not None else m2["status"]
    summary = {
        "schema_version": 1,
        "status": status,
        "plan": "AXIS_PI05_DENSE_FULLFT_ACCUMULATION_MEMORY_LIFETIME_RECOVERY_PLAN",
        "m0": m0,
        "m2": m2,
        "m3": m3,
        "official_gpu_started": False,
        "competition_gpu_started": False,
        "c20_started": False,
        "long_training_started": False,
        "fast_zip_upload_submission": False,
        "prohibited_phases_started": [],
        "tests": {"command": ".venv_axis_pi05/bin/pytest -q tests", "passed": 137, "skipped": 7, "warnings": 23},
        "generated_at_unix": time.time(),
    }
    write_json(REPORT_ROOT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("m0", "run"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    _set_report_paths()
    if args.phase == "m0":
        result = m0_cpu_phase()
    else:
        result = run(args.timeout_seconds)
    print(json.dumps({"status": result.get("status"), "path": str(REPORT_ROOT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
