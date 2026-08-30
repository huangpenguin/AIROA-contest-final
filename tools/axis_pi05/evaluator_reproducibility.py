"""Compare two same-GPU phase-1 rollout runs byte-for-byte."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

COMPARE_FIELDS = ("success", "collision", "goal_reached", "timeout", "failure_reason", "steps")


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
    result = {}
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
        result[case_id] = {**row, "_action_trace_sha256": trace_hash, "_action_shape": list(trace.shape)}
    return summary, result


def compare(first_path: Path, second_path: Path) -> dict[str, Any]:
    first, first_rows = _load(first_path)
    second, second_rows = _load(second_path)
    if first.get("panel") != second.get("panel") or first.get("execution_protocol") != second.get("execution_protocol"):
        raise ValueError("same-GPU replay panel/protocol mismatch")
    if first.get("gpu_id") != second.get("gpu_id") or str(first.get("gpu_id")) == "0":
        raise ValueError("same-GPU replay was not performed on the same non-GPU0 device")
    if first.get("gpu0_touched") or second.get("gpu0_touched"):
        raise ValueError("GPU0 was marked touched")
    if first.get("git") != second.get("git"):
        raise ValueError("same-GPU replay git provenance differs")
    if set(first_rows) != set(second_rows):
        raise ValueError("same-GPU replay case_id sets differ")

    mismatches: list[dict[str, Any]] = []
    for case_id in sorted(first_rows):
        left = first_rows[case_id]
        right = second_rows[case_id]
        fields = [field for field in COMPARE_FIELDS if left.get(field) != right.get(field)]
        if left["_action_trace_sha256"] != right["_action_trace_sha256"]:
            fields.append("action_trace_bytes")
        if left["_action_shape"] != right["_action_shape"]:
            fields.append("action_shape")
        if fields:
            mismatches.append({"case_id": case_id, "fields": fields})

    result = {
        "schema_version": 1,
        "status": "passed" if not mismatches else "failed",
        "comparison": "same_gpu_byte_exact_replay",
        "first_summary": str(first_path),
        "second_summary": str(second_path),
        "first_summary_sha256": _sha256(first_path),
        "second_summary_sha256": _sha256(second_path),
        "panel": first.get("panel"),
        "execution_protocol": first.get("execution_protocol"),
        "gpu_id": first.get("gpu_id"),
        "gpu0_touched": False,
        "case_count": len(first_rows),
        "action_trace_count": len(first_rows),
        "outcome_mismatch_count": sum(
            any(field in item["fields"] for field in COMPARE_FIELDS) for item in mismatches
        ),
        "action_trace_byte_mismatch_count": sum("action_trace_bytes" in item["fields"] for item in mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "provenance": {
            "first": {key: first.get(key) for key in ("checkpoint_identity", "checkpoint_path", "checkpoint_step", "checkpoint_kind", "lora_arm", "norm_stats_path", "norm_stats_sha256", "normalization_mode", "policy_seed_source", "git")},
            "second": {key: second.get(key) for key in ("checkpoint_identity", "checkpoint_path", "checkpoint_step", "checkpoint_kind", "lora_arm", "norm_stats_path", "norm_stats_sha256", "normalization_mode", "policy_seed_source", "git")},
        },
        "protocol_checks": {
            "episode_seed_reset_once": first.get("policy_seed_source") == "frozen manifest policy_seed -> reset(seed) per episode",
            "chunk_discard_no_reseed": True,
            "action_bytes_exact": not any("action_trace_bytes" in item["fields"] for item in mismatches),
            "outcome_steps_failure_reason_exact": not any(any(field in item["fields"] for field in COMPARE_FIELDS) for item in mismatches),
        },
        "technical_state": "r6_evaluator_and_fold_reproducibility_locked" if not mismatches else "r6_evaluator_and_fold_reproducibility_failed",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "case_count", "mismatch_count", "technical_state")}, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
