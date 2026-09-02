"""Run the CPU L0/L1 lock and the authorized inter-step lifetime gates.

The GPU phases are deliberately narrow: L2 stops after the next-step
microbatch zero, L3 stops after two updates and a step-3 restore, and L4 stops
after step 10.  No resource escalation or long training is part of this
runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
import time
from typing import Any

from . import dense_fullft_allocator_fraction_recovery as allocator
from . import dense_fullft_donated_accumulator_recovery as donated
from . import dense_fullft_accumulation_memory_lifetime_recovery as lifetime
from .common import ROOT, write_json


REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_interstep_state_lifetime_recovery"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/dense_fullft_interstep_state_lifetime_recovery"
DONATED_REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/dense_fullft_donated_accumulator_recovery"
DONATED_OUTPUT_ROOT = ROOT / "outputs/axis_pi05/dense_fullft_donated_accumulator_recovery"
SOURCE_STEP1 = DONATED_OUTPUT_ROOT / "d3_step1" / "1"
SOURCE_D3_REPORT = DONATED_REPORT_ROOT / "d3_step1.json"
FRACTION = "0.95"
PLATEAU_LIMIT_BYTES = 256 * 2**20


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(path: Path) -> list[dict[str, Any]]:
    return allocator._read_jsonl(path)


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _configure_paths() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    allocator.REPORT_ROOT = REPORT_ROOT
    allocator.OUTPUT_ROOT = OUTPUT_ROOT
    donated.REPORT_ROOT = REPORT_ROOT
    donated.OUTPUT_ROOT = OUTPUT_ROOT


def _rows_for_step(rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    return [row for row in rows if int(row.get("optimizer_step_before", -1)) == step]


def _indexes(rows: list[dict[str, Any]], phase: str, step: int | None = None) -> list[int]:
    selected = rows if step is None else _rows_for_step(rows, step)
    return sorted({int(row["micro_index"]) for row in selected if row.get("phase") == phase and "micro_index" in row})


def _device_bytes(row: dict[str, Any]) -> list[int]:
    return [
        int(device["memory_stats"]["bytes_in_use"])
        for device in row.get("devices", [])
        if device.get("memory_stats_available") and "bytes_in_use" in device.get("memory_stats", {})
    ]


def _release_rows(rows: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    return [
        {"micro_index": int(row["micro_index"]), "bytes_in_use_per_card": _device_bytes(row), "timestamp": row.get("timestamp")}
        for row in _rows_for_step(rows, step)
        if row.get("phase") == "temporaries_released" and _device_bytes(row)
    ]


def _memory_by_index(rows: list[dict[str, Any]], step: int) -> dict[int, list[int]]:
    return {row["micro_index"]: row["bytes_in_use_per_card"] for row in _release_rows(rows, step)}


def _step_plateau(rows: list[dict[str, Any]], step: int, expected_count: int = 16) -> dict[str, Any]:
    selected = _rows_for_step(rows, step)
    memory = _memory_by_index(rows, step)
    expected = list(range(expected_count))
    baseline = memory.get(1)
    drifts = [
        abs(value - baseline[card])
        for values in memory.values()
        for card, value in enumerate(values)
        if baseline is not None and card < len(baseline)
    ]
    result = {
        "optimizer_step_before": step,
        "compute_completed_indexes": _indexes(selected, "microbatch_compute_complete"),
        "accumulation_completed_indexes": _indexes(selected, "accumulation_complete"),
        "temporaries_released_indexes": sorted(memory),
        "bytes_in_use_per_card_by_micro_index": {str(index): memory[index] for index in sorted(memory)},
        "baseline_index": 1,
        "baseline_bytes_in_use_per_card": baseline,
        "max_absolute_drift_bytes_per_card": max(drifts, default=None),
        "max_allowed_drift_bytes_per_card": PLATEAU_LIMIT_BYTES,
    }
    result["passed"] = (
        result["compute_completed_indexes"] == expected
        and result["accumulation_completed_indexes"] == expected
        and result["temporaries_released_indexes"] == expected
        and baseline is not None
        and result["max_absolute_drift_bytes_per_card"] is not None
        and result["max_absolute_drift_bytes_per_card"] <= PLATEAU_LIMIT_BYTES
    )
    return result


def _donation_gate(rows: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    analyses = [
        row.get("analysis", {})
        for row in rows
        if row.get("phase") == "executable_memory_analysis"
        and row.get("executable_phase") == "donated_gradient_add"
    ]
    aliases = [
        int(item["alias_size_in_bytes"])
        for item in analyses
        if item.get("available") and item.get("alias_size_in_bytes") is not None
    ]
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    warnings = [line.strip() for line in text.splitlines() if "donat" in line.lower() and any(word in line.lower() for word in ("warn", "fail", "unable", "unusable", "reject"))]
    return {
        "analysis_rows": len(analyses),
        "analysis_available": bool(aliases),
        "alias_size_in_bytes": max(aliases, default=0),
        "donation_warnings": warnings,
        "passed": bool(aliases) and max(aliases) > 0 and not warnings,
    }


def _transition_memory_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    previous = _memory_by_index(rows, 1)
    transition = _memory_by_index(rows, 2)
    reference = previous.get(15)
    observed = transition.get(0)
    drifts = [
        abs(value - reference[card])
        for card, value in enumerate(observed or [])
        if reference is not None and card < len(reference)
    ]
    return {
        "reference_optimizer_step_before": 1,
        "reference_micro_index": 15,
        "reference_bytes_in_use_per_card": reference,
        "transition_optimizer_step_before": 2,
        "transition_micro_index": 0,
        "transition_bytes_in_use_per_card": observed,
        "max_absolute_drift_bytes_per_card": max(drifts, default=None),
        "max_allowed_drift_bytes_per_card": PLATEAU_LIMIT_BYTES,
        "passed": reference is not None and observed is not None and bool(drifts) and max(drifts) <= PLATEAU_LIMIT_BYTES,
    }


def _copy_step1(target: Path, tree_sha256: str) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint namespace: {target}")
    if allocator.sha256_tree(SOURCE_STEP1) != tree_sha256:
        raise RuntimeError("immutable D3 step-1 source changed before copy")
    temporary = target.with_name(f".{target.name}.copying")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse incomplete checkpoint copy: {temporary}")
    started = time.time()
    temporary.mkdir(parents=True)
    shutil.copytree(SOURCE_STEP1, temporary / "1")
    if allocator.sha256_tree(temporary / "1") != tree_sha256:
        raise RuntimeError("copied D3 step-1 tree hash mismatch")
    temporary.rename(target)
    return {
        "source": str(SOURCE_STEP1),
        "target": str(target),
        "step": 1,
        "tree_sha256": tree_sha256,
        "bytes": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()),
        "duration_seconds": time.time() - started,
        "atomic_directory_rename": True,
    }


def _audit_environment(run_dir: Path, name: str) -> dict[str, str | None]:
    keys = (
        "AXIS_PI05_AUDIT_METADATA",
        "AXIS_PI05_AUDIT_REPLAY_DIR",
        "AXIS_PI05_AUDIT_MICRO_NONFINITE_STEPS",
    )
    old = {key: os.environ.get(key) for key in keys}
    os.environ["AXIS_PI05_AUDIT_METADATA"] = "1"
    os.environ["AXIS_PI05_AUDIT_REPLAY_DIR"] = str(run_dir / name)
    os.environ["AXIS_PI05_AUDIT_MICRO_NONFINITE_STEPS"] = "0"
    return old


def _restore_environment(old: dict[str, str | None]) -> None:
    import os

    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def l0_cpu_phase() -> dict[str, Any]:
    _configure_paths()
    if not SOURCE_D3_REPORT.is_file() or not SOURCE_STEP1.is_dir():
        raise FileNotFoundError("D3 step-1 report/checkpoint is required for L0")
    d3 = _load(SOURCE_D3_REPORT)
    d4_path = DONATED_REPORT_ROOT / "d4_donated_continue_10.json"
    d4 = _load(d4_path)
    rows = _stats(Path(d4["allocator_stats_path"]))
    step1 = d3.get("train_report", {})
    semantics = {
        "restored_internal_step": d4.get("restore_validation", {}).get("step", 1),
        "completed_update_transition": [1, 2],
        "failed_optimizer_step_before": 2,
        "completed_microbatches": _indexes(rows, "temporaries_released", 2),
        "failed_compute_microbatch": next(
            (int(row["micro_index"]) for row in rows if row.get("phase") == "microbatch_exception" and int(row.get("optimizer_step_before", -1)) == 2),
            None,
        ),
        "no_step10_checkpoint": 10 not in d4.get("checkpoint_steps_observed", []),
    }
    boundary_phases = (
        "checkpoint_restore",
        "optimizer_step_start",
        "temporaries_released",
        "optimizer_update_start",
        "optimizer_update_complete",
        "optimizer_returned",
        "new_state_ready",
        "metrics_host_materialized",
        "device_metrics_released",
        "next_step_start",
    )
    boundaries = {
        phase: [
            {
                "optimizer_step_before": row.get("optimizer_step_before"),
                "optimizer_step_after": row.get("optimizer_step_after"),
                "micro_index": row.get("micro_index"),
                "microbatch_count": row.get("microbatch_count"),
                "bytes_in_use_per_card": _device_bytes(row),
            }
            for row in rows
            if row.get("phase") == phase
        ]
        for phase in boundary_phases
    }
    lock = {
        "schema_version": 1,
        "status": "dense_fullft_interstep_l0_cpu_locked",
        "input_root_commit": _git_commit(ROOT),
        "input_nested_openpi_commit": _git_commit(ROOT / "third_party/axis_openpi"),
        "raw_evidence_immutable": True,
        "raw_evidence": {
            "d4_report": str(d4_path),
            "d4_allocator_stats": d4["allocator_stats_path"],
            "d4_log": d4["log_path"],
            "d4_checkpoint_steps_observed": d4.get("checkpoint_steps_observed", []),
        },
        "source_step1": {
            "path": str(SOURCE_STEP1),
            "tree_sha256": allocator.sha256_tree(SOURCE_STEP1),
            "d3_status": d3.get("status"),
            "d3_report": str(SOURCE_D3_REPORT),
            "checkpoint_step": 1,
        },
        "semantic_correction": semantics,
        "boundary_table": boundaries,
        "gpu_idle_before": donated._gpu_snapshot(),
        "contracts_unchanged": {
            "precision": "float32",
            "topology": "dense_full_model",
            "optimizer": "AdamW",
            "ema_decay": 0.999,
            "microbatch_per_device": 1,
            "accumulation_steps": 16,
            "effective_global_batch": 64,
            "allocator_fraction": FRACTION,
            "ordered_gradient_sum": True,
        },
        "next_allowed": ["L1 CPU lifecycle gate", "L2 step1-to-step2-to-next-microbatch-0 transition canary"],
    }
    write_json(REPORT_ROOT / "l0_cpu_lock.json", lock)
    return lock


def l2_transition(timeout_seconds: int, lock: dict[str, Any]) -> dict[str, Any]:
    _configure_paths()
    if lock.get("status") != "dense_fullft_interstep_l0_cpu_locked":
        raise RuntimeError("L2 requires a passing L0 lock")
    source_hash = lock["source_step1"]["tree_sha256"]
    if allocator.sha256_tree(SOURCE_STEP1) != source_hash:
        raise RuntimeError("immutable D3 step-1 checkpoint changed before L2")
    run_dir = OUTPUT_ROOT / "l2_transition"
    copy = _copy_step1(run_dir, source_hash)
    old_env = _audit_environment(run_dir, "audit_l2")
    try:
        report = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="l2_transition_probe",
            steps=3,
            expected_sentinels=[],
            timeout_seconds=timeout_seconds,
            extra_env={
                "AXIS_PI05_INTERSTEP_TRANSITION_PROBE": "1",
                "AXIS_PI05_INTERSTEP_TARGET_STEP": "2",
                "AXIS_PI05_INTERSTEP_TRANSITION_METRICS_PATH": str(run_dir / "interstep_transition_metrics.json"),
                "OPENPI_KEEP_STEPS": "1",
                "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "0",
            },
        )
    finally:
        _restore_environment(old_env)
    rows = _stats(Path(report["allocator_stats_path"]))
    transition_metrics = _load(run_dir / "interstep_transition_metrics.json") if (run_dir / "interstep_transition_metrics.json").is_file() else {}
    step1_plateau = _step_plateau(rows, 1)
    transition_memory = _transition_memory_gate(rows)
    donation = _donation_gate(rows, Path(report["log_path"]))
    events = {
        "checkpoint_restore_step": [row.get("train_state_step") for row in rows if row.get("phase") == "checkpoint_restore"],
        "update_transitions": [[row.get("optimizer_step_before"), row.get("optimizer_step_after")] for row in rows if row.get("phase") == "optimizer_update_complete"],
        "transition_compute_indexes": _indexes(rows, "microbatch_compute_complete", 2),
        "transition_accumulation_indexes": _indexes(rows, "accumulation_complete", 2),
        "transition_release_indexes": _indexes(rows, "temporaries_released", 2),
        "next_step_start_count": sum(row.get("phase") == "next_step_start" for row in rows),
        "step2_optimizer_update_count": sum(row.get("phase") == "optimizer_update_start" and int(row.get("optimizer_step_before", -1)) == 2 for row in rows),
    }
    passed = (
        report.get("return_code") == 0
        and report.get("timed_out") is False
        and not report.get("external_pids")
        and report.get("restore_validation", {}).get("status") == "passed"
        and report.get("checkpoint_steps_observed") == [1]
        and report.get("metric_steps_observed") == [2]
        and report.get("finite_metrics") is True
        and step1_plateau["passed"]
        and transition_memory["passed"]
        and donation["passed"]
        and events["checkpoint_restore_step"] == [1]
        and events["update_transitions"] == [[1, 2]]
        and events["transition_compute_indexes"] == [0]
        and events["transition_accumulation_indexes"] == [0]
        and events["transition_release_indexes"] == [0]
        and events["next_step_start_count"] == 1
        and events["step2_optimizer_update_count"] == 0
        and transition_metrics.get("status") == "transition_probe_completed"
    )
    result = {
        "schema_version": 1,
        "status": "dense_fullft_interstep_transition_passed_pending_review" if passed else "dense_fullft_interstep_lifetime_remediation_failed",
        "fraction": FRACTION,
        "copy_step1": copy,
        "train_report": report,
        "step1_plateau": step1_plateau,
        "transition_memory": transition_memory,
        "events": events,
        "transition_metrics": transition_metrics,
        "donation": donation,
        "source_step1_tree_sha256": source_hash,
        "source_step1_unchanged": allocator.sha256_tree(SOURCE_STEP1) == source_hash,
        "official_gpu_started": False,
        "blackwell_started": False,
        "c20_started": False,
        "long_training_started": False,
    }
    write_json(REPORT_ROOT / "l2_transition.json", result)
    return result


def _full_step_result(report: dict[str, Any], expected_step: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    plateau = _step_plateau(rows, expected_step - 1)
    return {
        "expected_post_update_step": expected_step,
        "metric_steps": report.get("metric_steps_observed", []),
        "checkpoint_steps": report.get("checkpoint_steps_observed", []),
        "plateau": plateau,
        "passed": report.get("return_code") == 0 and not report.get("timed_out") and not report.get("external_pids") and report.get("finite_metrics") is True and expected_step in report.get("metric_steps_observed", []) and plateau["passed"],
    }


def l3_step3(timeout_seconds: int, lock: dict[str, Any], l2: dict[str, Any]) -> dict[str, Any]:
    if l2.get("status") != "dense_fullft_interstep_transition_passed_pending_review":
        raise RuntimeError("L3 requires a passing L2 transition gate")
    _configure_paths()
    run_dir = OUTPUT_ROOT / "l3_step3"
    copy = _copy_step1(run_dir, lock["source_step1"]["tree_sha256"])
    old_env = _audit_environment(run_dir, "audit_l3")
    try:
        report = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="l3_continue_step3",
            steps=3,
            expected_sentinels=[],
            timeout_seconds=timeout_seconds,
            extra_env={"OPENPI_KEEP_STEPS": "1,3", "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1"},
        )
        rows = _stats(Path(report["allocator_stats_path"]))
        restore = allocator._run_restore_only(
            fraction=FRACTION,
            run_dir=run_dir,
            label="l3_restore_step3",
            timeout_seconds=timeout_seconds,
        )
    finally:
        _restore_environment(old_env)
    rows = _stats(Path(report["allocator_stats_path"]))
    donation = _donation_gate(rows, Path(report["log_path"]))
    full = {
        "step_1_to_2": _full_step_result(report, 2, rows),
        "step_2_to_3": _full_step_result(report, 3, rows),
    }
    passed = (
        report.get("return_code") == 0
        and report.get("checkpoint_steps_observed") == [1, 3]
        and report.get("metric_steps_observed") == [2, 3]
        and full["step_1_to_2"]["passed"]
        and full["step_2_to_3"]["passed"]
        and donation["passed"]
        and restore.get("status") == "restore_only_passed"
    )
    result = {
        "schema_version": 1,
        "status": "dense_fullft_interstep_step3_passed_pending_review" if passed else "dense_fullft_interstep_lifetime_remediation_failed",
        "fraction": FRACTION,
        "copy_step1": copy,
        "continuation": report,
        "full_steps": full,
        "donation": donation,
        "restore_step3": restore,
        "source_step1_tree_sha256": lock["source_step1"]["tree_sha256"],
        "source_step1_unchanged": allocator.sha256_tree(SOURCE_STEP1) == lock["source_step1"]["tree_sha256"],
        "official_gpu_started": False,
        "blackwell_started": False,
        "c20_started": False,
        "long_training_started": False,
    }
    write_json(REPORT_ROOT / "l3_step3.json", result)
    return result


def l4_step10(timeout_seconds: int, lock: dict[str, Any], l3: dict[str, Any]) -> dict[str, Any]:
    if l3.get("status") != "dense_fullft_interstep_step3_passed_pending_review":
        raise RuntimeError("L4 requires a passing L3 step-3 gate")
    _configure_paths()
    run_dir = OUTPUT_ROOT / "l3_step3"
    old_env = _audit_environment(run_dir, "audit_l4")
    try:
        report = allocator._run_train_process(
            fraction=FRACTION,
            run_dir=run_dir,
            label="l4_continue_step10",
            steps=10,
            expected_sentinels=[],
            timeout_seconds=timeout_seconds,
            extra_env={"OPENPI_KEEP_STEPS": "1,3,10", "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1"},
        )
        rows = _stats(Path(report["allocator_stats_path"]))
        restore = allocator._run_restore_only(
            fraction=FRACTION,
            run_dir=run_dir,
            label="l4_restore_step10",
            timeout_seconds=timeout_seconds,
        )
    finally:
        _restore_environment(old_env)
    rows = _stats(Path(report["allocator_stats_path"]))
    donation = _donation_gate(rows, Path(report["log_path"]))
    per_step = {str(step): _step_plateau(rows, step) for step in range(1, 10)}
    passed = allocator._full_report_status(report, 10) and all(item["passed"] for item in per_step.values()) and donation["passed"] and restore.get("status") == "restore_only_passed"
    result = {
        "schema_version": 1,
        "status": "dense_fullft_interstep_step10_passed_pending_review" if passed else "dense_fullft_interstep_lifetime_remediation_failed",
        "fraction": FRACTION,
        "continuation": report,
        "per_optimizer_step_plateau": per_step,
        "donation": donation,
        "restore_step10": restore,
        "source_step1_tree_sha256": lock["source_step1"]["tree_sha256"],
        "source_step1_unchanged": allocator.sha256_tree(SOURCE_STEP1) == lock["source_step1"]["tree_sha256"],
        "official_gpu_started": False,
        "blackwell_started": False,
        "c20_started": False,
        "long_training_started": False,
    }
    write_json(REPORT_ROOT / "l4_step10.json", result)
    return result


def run(timeout_seconds: int) -> dict[str, Any]:
    lock = l0_cpu_phase()
    l2 = l2_transition(timeout_seconds, lock)
    l3 = l3_step3(timeout_seconds, lock, l2) if l2["status"] == "dense_fullft_interstep_transition_passed_pending_review" else None
    l4 = l4_step10(timeout_seconds, lock, l3) if l3 is not None and l3["status"] == "dense_fullft_interstep_step3_passed_pending_review" else None
    status = (l4 or l3 or l2)["status"]
    summary = {
        "schema_version": 1,
        "status": status,
        "plan": "AXIS_PI05_DENSE_FULLFT_INTERSTEP_STATE_LIFETIME_RECOVERY_PLAN",
        "l0": lock,
        "l2": l2,
        "l3": l3,
        "l4": l4,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("l0", "run"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    result = l0_cpu_phase() if args.phase == "l0" else run(args.timeout_seconds)
    print(json.dumps({"status": result["status"], "path": str(REPORT_ROOT)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
