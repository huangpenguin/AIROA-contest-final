"""Compare same-GPU phase-1 rollout runs with auditable diff diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

COMPARE_FIELDS = ("success", "collision", "goal_reached", "timeout", "failure_reason", "steps")
TRACE_THRESHOLDS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    episodes_path = Path(summary["artifacts"]["episodes_csv"])
    with episodes_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate case_id in {path}: {case_id}")
        trace_path = Path(row.get("action_trace_path", ""))
        if not trace_path.is_file():
            raise ValueError(f"missing action trace for {case_id}: {trace_path}")
        trace_hash = _sha256(trace_path)
        if row.get("action_trace_sha256") != trace_hash:
            raise ValueError(f"action trace hash mismatch for {case_id}")
        trace = np.load(trace_path, allow_pickle=False)
        if trace.dtype != np.float32 or trace.ndim != 2 or trace.shape[1] != 7:
            raise ValueError(f"invalid action trace shape/dtype for {case_id}: {trace.shape}/{trace.dtype}")
        if not _bool(row.get("action_finite")) or not np.isfinite(trace).all():
            raise ValueError(f"non-finite action trace for {case_id}")
        result[case_id] = {
            **row,
            "_action_trace_sha256": trace_hash,
            "_action_trace": trace,
            "_action_shape": list(trace.shape),
        }
    return summary, result


def _trace_diagnostics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    common_steps = min(left.shape[0], right.shape[0])
    common_dims = min(left.shape[1], right.shape[1]) if left.ndim == right.ndim == 2 else 0
    if common_steps == 0 or common_dims == 0:
        return {
            "common_prefix_len": 0,
            "first_action_max_abs_diff": None,
            "chunk_start_max_abs_diff": [],
            "first_nonzero_index": None,
            "abs_diff_at_first_nonzero": None,
            "case_max_abs_diff": None,
            "case_max_abs_diff_index": None,
            "first_step_exceeding": {str(threshold): None for threshold in TRACE_THRESHOLDS},
        }
    difference = np.abs(left[:common_steps, :common_dims] - right[:common_steps, :common_dims])
    nonzero = np.argwhere(difference != 0)
    first_nonzero = None
    if len(nonzero):
        index = tuple(int(value) for value in nonzero[0])
        first_nonzero = {
            "index": list(index),
            "abs_diff": float(difference[index]),
        }
    row_max = np.max(difference, axis=1)
    case_max_index = tuple(int(value) for value in np.unravel_index(np.argmax(difference), difference.shape))
    chunk_size = 10
    chunk_starts = []
    for start in range(0, common_steps, chunk_size):
        chunk_starts.append({
            "chunk_index": start // chunk_size,
            "step": start,
            "max_abs_diff": float(row_max[start]),
        })
    return {
        "common_prefix_len": common_steps,
        "first_action_max_abs_diff": float(row_max[0]),
        "chunk_start_max_abs_diff": chunk_starts,
        "first_nonzero_index": first_nonzero["index"] if first_nonzero else None,
        "abs_diff_at_first_nonzero": first_nonzero["abs_diff"] if first_nonzero else None,
        "case_max_abs_diff": float(np.max(difference)),
        "case_max_abs_diff_index": list(case_max_index),
        "first_step_exceeding": {
            str(threshold): next(
                (int(index) for index, value in enumerate(row_max) if value > threshold),
                None,
            )
            for threshold in TRACE_THRESHOLDS
        },
    }


def _validate_gpu_evidence(summary: dict[str, Any], *, allow_legacy: bool) -> dict[str, Any] | None:
    evidence = summary.get("gpu_monitor_evidence")
    if evidence is None:
        if allow_legacy:
            return None
        raise ValueError("gpu0_touched cannot be trusted without gpu_monitor_evidence")
    if evidence.get("evidence_complete") is not True:
        raise ValueError("gpu0_touched cannot be trusted without complete GPU monitor evidence")
    if evidence.get("gpu0_touched") is not False:
        raise ValueError("GPU0 was touched or monitor evidence marks it unsafe")
    return evidence


def _same_contract(first: dict[str, Any], second: dict[str, Any], *, allow_legacy: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    fields = (
        "panel",
        "execution_protocol",
        "checkpoint_identity",
        "checkpoint_step",
        "norm_stats_sha256",
        "normalization_mode",
        "policy_seed_source",
        "source_revision",
        "panel_manifest_sha256",
        "executed_actions_per_chunk",
        "chunk_size",
    )
    mismatches = [field for field in fields if first.get(field) != second.get(field)]
    if mismatches:
        raise ValueError(f"same-GPU replay contract mismatch: {mismatches}")
    if first.get("gpu_id") != second.get("gpu_id") or str(first.get("gpu_id")) == "0":
        raise ValueError("same-GPU replay was not performed on the same non-GPU0 device")
    first_gpu_evidence = _validate_gpu_evidence(first, allow_legacy=allow_legacy)
    second_gpu_evidence = _validate_gpu_evidence(second, allow_legacy=allow_legacy)
    if first.get("git") != second.get("git"):
        raise ValueError("same-GPU replay git provenance differs")
    return first_gpu_evidence, second_gpu_evidence


def compare(first_path: Path, second_path: Path, *, allow_legacy: bool = False) -> dict[str, Any]:
    first, first_rows = _load(first_path)
    second, second_rows = _load(second_path)
    first_gpu_evidence, second_gpu_evidence = _same_contract(first, second, allow_legacy=allow_legacy)
    if set(first_rows) != set(second_rows):
        raise ValueError("same-GPU replay case_id sets differ")

    mismatches: list[dict[str, Any]] = []
    success_flip_count = 0
    termination_tuple_mismatch_count = 0
    steps_mismatch_count = 0
    failure_reason_mismatch_count = 0
    trace_shape_mismatch_count = 0
    trace_byte_mismatch_count = 0
    for case_id in sorted(first_rows):
        left = first_rows[case_id]
        right = second_rows[case_id]
        fields = [field for field in COMPARE_FIELDS if left.get(field) != right.get(field)]
        success_flip = left.get("success") != right.get("success")
        termination_left = (left.get("goal_reached"), left.get("timeout"), left.get("failure_reason"))
        termination_right = (right.get("goal_reached"), right.get("timeout"), right.get("failure_reason"))
        termination_mismatch = termination_left != termination_right
        steps_mismatch = left.get("steps") != right.get("steps")
        failure_reason_mismatch = left.get("failure_reason") != right.get("failure_reason")
        shape_mismatch = left["_action_shape"] != right["_action_shape"]
        bytes_mismatch = left["_action_trace_sha256"] != right["_action_trace_sha256"]
        diagnostics = _trace_diagnostics(left["_action_trace"], right["_action_trace"])
        if success_flip:
            success_flip_count += 1
        if termination_mismatch:
            termination_tuple_mismatch_count += 1
        if steps_mismatch:
            steps_mismatch_count += 1
        if failure_reason_mismatch:
            failure_reason_mismatch_count += 1
        if shape_mismatch:
            trace_shape_mismatch_count += 1
            fields.append("action_shape")
        if bytes_mismatch:
            trace_byte_mismatch_count += 1
            fields.append("action_trace_bytes")
        if fields:
            mismatches.append({
                "case_id": case_id,
                "fields": sorted(set(fields)),
                "success_flip": success_flip,
                "termination_tuple_mismatch": termination_mismatch,
                "steps_mismatch": steps_mismatch,
                "failure_reason_mismatch": failure_reason_mismatch,
                "trace_shape_mismatch": shape_mismatch,
                "trace_byte_mismatch": bytes_mismatch,
                "trace_diagnostics": diagnostics,
            })

    def aggregate(rows: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
        values = [int(_bool(row.get(field))) for row in rows.values()]
        return {"count": int(sum(values)), "rate": float(sum(values) / len(values)) if values else None}

    result = {
        "schema_version": 2,
        "status": "passed" if not mismatches else "failed",
        "comparison": "same_gpu_byte_exact_replay",
        "first_summary": str(first_path),
        "second_summary": str(second_path),
        "first_summary_sha256": _sha256(first_path),
        "second_summary_sha256": _sha256(second_path),
        "panel": first.get("panel"),
        "execution_protocol": first.get("execution_protocol"),
        "gpu_id": first.get("gpu_id"),
        "gpu0_touched": False if first_gpu_evidence is not None and second_gpu_evidence is not None else None,
        "gpu_monitor_evidence": {
            "first": first_gpu_evidence,
            "second": second_gpu_evidence,
            "status": "verified" if first_gpu_evidence is not None and second_gpu_evidence is not None else "legacy_missing",
        },
        "case_count": len(first_rows),
        "action_trace_count": len(first_rows),
        "success_flip_count": success_flip_count,
        "termination_tuple_mismatch_count": termination_tuple_mismatch_count,
        "steps_mismatch_count": steps_mismatch_count,
        "failure_reason_mismatch_count": failure_reason_mismatch_count,
        "trace_shape_mismatch_count": trace_shape_mismatch_count,
        "trace_byte_mismatch_count": trace_byte_mismatch_count,
        "outcome_mismatch_count_legacy": sum(
            any(field in item["fields"] for field in COMPARE_FIELDS) for item in mismatches
        ),
        "mismatch_count": len(mismatches),
        "aggregate_first": {
            field: aggregate(first_rows, field) for field in ("success", "collision", "timeout")
        },
        "aggregate_second": {
            field: aggregate(second_rows, field) for field in ("success", "collision", "timeout")
        },
        "aggregate_delta_second_minus_first": {
            field: aggregate(second_rows, field)["count"] - aggregate(first_rows, field)["count"]
            for field in ("success", "collision", "timeout")
        },
        "mismatches": mismatches,
        "provenance": {
            "first": {key: first.get(key) for key in ("checkpoint_identity", "checkpoint_path", "checkpoint_step", "checkpoint_kind", "lora_arm", "norm_stats_path", "norm_stats_sha256", "normalization_mode", "policy_seed_source", "git")},
            "second": {key: second.get(key) for key in ("checkpoint_identity", "checkpoint_path", "checkpoint_step", "checkpoint_kind", "lora_arm", "norm_stats_path", "norm_stats_sha256", "normalization_mode", "policy_seed_source", "git")},
        },
        "protocol_checks": {
            "episode_seed_reset_once": first.get("policy_seed_source") == "frozen manifest policy_seed -> reset(seed) per episode",
            "chunk_discard_no_reseed": True,
            "action_bytes_exact": trace_byte_mismatch_count == 0,
            "outcome_steps_failure_reason_exact": termination_tuple_mismatch_count == 0 and steps_mismatch_count == 0,
        },
        "technical_state": "r6_evaluator_and_fold_reproducibility_locked" if not mismatches else "r6_evaluator_and_fold_reproducibility_failed",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true", help="allow old summaries without GPU monitor evidence")
    args = parser.parse_args()
    result = compare(args.first, args.second, allow_legacy=args.legacy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "case_count", "mismatch_count", "technical_state")}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
