"""Run the authorized, read-only R1 integration against frozen R6/step100.

This adapter does not start an evaluator.  It translates the already-written
legacy case records into the strict worker-result envelope expected by the
remediated finalizer, reuses their real trace/video paths, and records a new
namespace.  The frozen namespaces are never opened for writing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import track3_paired_eval_finalizer_v2 as finalizer


ROOT = Path(__file__).resolve().parents[2]
R6_EVIDENCE = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
STEP100_EVIDENCE = ROOT / "data/reports/axis_libero_30k/t3_fast400_h300_step100_salvage_20260915_attempt_0001"
R6_OUTPUT = ROOT / "outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002/r6"
STEP100_OUTPUT = ROOT / "outputs/axis_pi05/t3_fast400_h300_step100_salvage_20260915_attempt_0001/step100"
COMPARATOR = ROOT / "tools/axis_pi05/compare.py"
WORKER_ADAPTER = ROOT / "tools/axis_pi05/run_track3_fast400_h300_paired_eval.py"
WORKER_PLAN = ROOT / "docs/AXIS_PI05_T3_ONE_DAY_FINALIZER_R2_DATA_GATE_AND_GPU_SCREEN_PLAN_20260916.md"
COMPARISON_PLAN = ROOT / "docs/AXIS_PI05_T3_D1_EVALUATION_FINALIZER_REMEDIATION_PLAN_20260916.md"
DEFAULT_OUT = ROOT / "data/reports/axis_libero_30k/t3_d1_finalizer_r1_reject_integration_20260916_attempt_0001"


def _case_filename(case_id: str) -> str:
    return f"sha256_{case_id.split(':', 1)[1]}.json" if case_id.startswith("sha256:") else f"{case_id}.json"


def _load_case(root: Path, case_id: str) -> dict[str, Any]:
    path = root / "cases" / _case_filename(case_id)
    value = finalizer.strict_json_load(path)
    if value.get("case_id") != case_id or value.get("status") != "completed":
        raise RuntimeError(f"invalid frozen case record: {path}")
    return value


def _source_identity(cases: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    source_paths = {
        "comparator_source_sha256": COMPARATOR,
        "worker_adapter_sha256": WORKER_ADAPTER,
        "worker_plan_sha256": WORKER_PLAN,
        "comparison_plan_sha256": COMPARISON_PLAN,
    }
    identity = {field: finalizer.sha256_file(path) for field, path in source_paths.items()}
    identity["input_identity_sha256"] = finalizer.input_identity_for_cases(cases)
    return identity, {field: str(path.resolve()) for field, path in source_paths.items()}


def _worker_payload(role: str, identity: dict[str, str], cases: dict[str, str], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": finalizer.WORKER_RESULT_SCHEMA,
        "status": "completed",
        "role": role,
        "identity": identity,
        "infrastructure_error": False,
        "source_kind": "frozen_legacy_case_records_read_only_adapter",
        "cases": [
            {
                "case_id": case_id,
                "case_fingerprint": cases[case_id],
                "status": "completed",
                "infrastructure_error": False,
                "success": records[case_id].get("success"),
                "panel": records[case_id].get("panel"),
            }
            for case_id in sorted(cases)
        ],
    }


def _comparison(identity: dict[str, str], cases: dict[str, str], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    reverse = [row for row in pairs if row.get("panel") == "reverse_dev"]
    c20 = [row for row in pairs if row.get("panel") == "C20"]
    reverse_baseline = sum(bool(row.get("baseline_success")) for row in reverse)
    reverse_candidate = sum(bool(row.get("candidate_success")) for row in reverse)
    c20_baseline = sum(bool(row.get("baseline_success")) for row in c20)
    c20_candidate = sum(bool(row.get("candidate_success")) for row in c20)
    return {
        "schema_version": finalizer.COMPARISON_RESULT_SCHEMA,
        "status": "completed",
        "comparator_source_sha256": identity["comparator_source_sha256"],
        "comparison_plan_sha256": identity["comparison_plan_sha256"],
        "input_identity_sha256": identity["input_identity_sha256"],
        "zero_trace_policy": finalizer.ZERO_TRACE_POLICY,
        "behavior_gate": bool(reverse_candidate - reverse_baseline >= 1),
        "retention_gate": bool(c20_candidate - c20_baseline >= -2),
        "derived_from_frozen_pairs": True,
        "reverse": {"baseline_success": reverse_baseline, "candidate_success": reverse_candidate, "n": len(reverse)},
        "c20": {"baseline_success": c20_baseline, "candidate_success": c20_candidate, "n": len(c20)},
        "cases": [{"case_id": case_id, "case_fingerprint": cases[case_id]} for case_id in sorted(cases)],
    }


def run(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"R1 output namespace already exists: {out}")
    frozen_summary = finalizer.strict_json_load(R6_EVIDENCE / "paired_summary.json")
    step100_summary = finalizer.strict_json_load(STEP100_EVIDENCE / "paired_summary.json")
    pairs = frozen_summary.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise RuntimeError("frozen R6 paired summary does not contain exactly 24 pairs")
    step100_pairs = step100_summary.get("pairs")
    if not isinstance(step100_pairs, list) or len(step100_pairs) != 24:
        raise RuntimeError("frozen step100 paired summary does not contain exactly 24 pairs")
    if [row.get("case_id") for row in pairs] != [row.get("case_id") for row in step100_pairs]:
        raise RuntimeError("R6 and step100 frozen case ordering differs")

    case_ids = [str(row["case_id"]) for row in pairs]
    r6_records = {case_id: _load_case(R6_EVIDENCE / "r6", case_id) for case_id in case_ids}
    step100_records = {case_id: _load_case(STEP100_EVIDENCE / "step100", case_id) for case_id in case_ids}
    cases = {case_id: str(r6_records[case_id]["case_fingerprint"]) for case_id in case_ids}
    if any(step100_records[case_id].get("case_fingerprint") != fingerprint for case_id, fingerprint in cases.items()):
        raise RuntimeError("R6 and step100 case fingerprints differ")
    identity, source_paths = _source_identity(cases)

    traces: dict[str, list[dict[str, str]]] = {role: [] for role in finalizer.PAYLOAD_NAMES}
    videos: dict[str, list[dict[str, str]]] = {role: [] for role in finalizer.PAYLOAD_NAMES}
    for role, records, output_root in (
        ("baseline", r6_records, R6_OUTPUT),
        ("candidate", step100_records, STEP100_OUTPUT),
    ):
        for case_id in sorted(cases):
            record = records[case_id]
            trace = Path(str(record["action_trace_path"]))
            video = Path(str(record["video_path"]))
            if not trace.is_file() or not video.is_file():
                raise FileNotFoundError(f"frozen artifact missing for {role}/{case_id}: {trace} {video}")
            if trace.parent != output_root / "action_traces" or video.parent != output_root / "videos":
                raise RuntimeError(f"frozen artifact escaped expected output root for {role}/{case_id}")
            traces[role].append({"case_id": case_id, "case_fingerprint": cases[case_id], "path": str(trace)})
            videos[role].append({"case_id": case_id, "case_fingerprint": cases[case_id], "path": str(video)})

    payloads = {
        "baseline": _worker_payload("baseline", identity, cases, r6_records),
        "candidate": _worker_payload("candidate", identity, cases, step100_records),
    }
    comparison = _comparison(identity, cases, pairs)
    old_sha = {
        "r6_paired_summary": finalizer.sha256_file(R6_EVIDENCE / "paired_summary.json"),
        "step100_paired_summary": finalizer.sha256_file(STEP100_EVIDENCE / "paired_summary.json"),
    }
    result = finalizer.FinalizerV2(out / "attempt", identity, source_paths).finalize(
        payloads,
        cases=cases,
        traces=traces,
        videos=videos,
        comparison_result=comparison,
        summary_metadata={
            "integration": "R1_finalizer_only",
            "frozen_baseline": str(R6_EVIDENCE),
            "frozen_candidate": str(STEP100_EVIDENCE),
            "simulator_started": False,
            "policy_server_started": False,
            "gpu_used": False,
        },
    )
    new_sha = {
        "r6_paired_summary": finalizer.sha256_file(R6_EVIDENCE / "paired_summary.json"),
        "step100_paired_summary": finalizer.sha256_file(STEP100_EVIDENCE / "paired_summary.json"),
    }
    if old_sha != new_sha:
        raise RuntimeError("frozen paired summaries changed during R1")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "track3_d1_finalizer_r1_reject_integration_v1",
        "status": "completed",
        "release": result["release"],
        "expected_release": False,
        "reverse": comparison["reverse"],
        "c20": comparison["c20"],
        "gates": result["gates"],
        "identity": identity,
        "source_paths": source_paths,
        "finalizer_module": {"path": str((ROOT / "tools/axis_pi05/track3_paired_eval_finalizer_v2.py").resolve()), "sha256": finalizer.sha256_file(ROOT / "tools/axis_pi05/track3_paired_eval_finalizer_v2.py")},
        "focused_tests": {"path": str((ROOT / "tests/test_track3_paired_eval_finalizer_v2.py").resolve()), "sha256": finalizer.sha256_file(ROOT / "tests/test_track3_paired_eval_finalizer_v2.py")},
        "frozen_sha_before_after": {"before": old_sha, "after": new_sha},
        "attempt": str((out / "attempt" / "completed").resolve()),
        "no_gpu_or_runtime": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
