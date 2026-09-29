"""Independent, fail-closed comparator for the existing H300 step100 checkpoint.

The closed Fast400/H300 attempt is deliberately treated as an input fixture:
its R6 role is read from ``attempt_0002`` and is never resumed or merged in
place.  A fresh step100 worker may write only to the salvage namespace.  This
module owns the step100-specific checkpoint locks and the final paired gate;
the simulator worker remains the existing paired-evaluation runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import run_track3_fast400_h300_paired_eval as runner


ROOT = runner.ROOT
PLAN = ROOT / "docs/AXIS_PI05_T3_POST_H300_STEP100_SALVAGE_AND_R6_STYLE_RECOVERY_PLAN_20260915.md"
OLD_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
OLD_OUTPUT_ROOT = ROOT / "outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
SALVAGE_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_h300_step100_salvage_20260915_attempt_0001"
SALVAGE_OUTPUT_ROOT = ROOT / "outputs/axis_pi05/t3_fast400_h300_step100_salvage_20260915_attempt_0001"

EXPECTED_ADAPTER_SHA = "9c736c0ac1e3809df0ee35710781ffe8011929220794c58a75cf756e5b8a6c97"
EXPECTED_PLAN_SHA = "d034d45528e94a4a4d2a0c868782d5e73c35023e4a3fd9bb759b765a334e09c7"
EXPECTED_STEP100_NATIVE_SHA = "870022fc61f4e00d283adcc10371ac4ecf53e15c93c871b415a126bdf0bce314"
EXPECTED_STEP100_Q0_SHA = "0a8ee72f7bb58490ecade86ca2dad351273ba0938116ee25d5c300a34f8657ad"
EXPECTED_STEP100_METADATA_SHA = "d6b64050fe3c82dc7e78236cc00eb6bf5409dfaffab1043d3e242fa7c04b5bde"

ROLE = "step100"
EXPECTED_PROTOCOL = {
    "control_frequency_hz": runner.CONTROL_HZ,
    "chunk_size": runner.CHUNK_SIZE,
    "executed_actions_per_chunk": runner.EXECUTE_ACTIONS,
    "max_steps": runner.MAX_STEPS,
}


def strict_json_load(path: Path) -> dict[str, Any]:
    """Load JSON with duplicate-key rejection at every nesting level."""

    return runner.strict_json_load(path)


def _assert_no_partial(path: Path) -> None:
    runner._assert_no_partial(path)


def _assert_hash(path: Path, expected: str, label: str) -> None:
    actual = runner.sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def _assert_new_salvage_namespace(path: Path) -> None:
    """Reject all historical roots, including the closed attempt_0002 root."""

    resolved = path.resolve()
    old_roots = (OLD_EVIDENCE_ROOT.resolve(), OLD_OUTPUT_ROOT.resolve())
    if any(resolved == old or old in resolved.parents for old in old_roots):
        raise ValueError(f"refusing historical step100 salvage input namespace: {path}")
    runner._assert_new_namespace(path)


def _assert_merge_artifacts_absent(evidence_dir: Path) -> None:
    existing = [evidence_dir / name for name in ("paired_summary.json", "release_decision.json") if (evidence_dir / name).exists()]
    if existing:
        raise FileExistsError(f"paired salvage artifacts already exist; refusing repeat merge: {existing}")


def _expected_adapter_source() -> dict[str, str]:
    source = runner.adapter_source_identity()
    if source["sha256"] != EXPECTED_ADAPTER_SHA:
        raise ValueError(f"paired-eval adapter hash mismatch: {source['sha256']} != {EXPECTED_ADAPTER_SHA}")
    return source


def load_frozen_inputs() -> dict[str, Any]:
    """Load the same frozen 24-case panel used by the closed R6 attempt."""

    source = _expected_adapter_source()
    if not PLAN.is_file():
        raise FileNotFoundError(f"missing salvage plan: {PLAN}")
    _assert_hash(PLAN, EXPECTED_PLAN_SHA, "salvage plan")
    inputs = runner.load_frozen_inputs(adapter_source=source)
    if inputs["input_hashes"].get("adapter_sha256") != EXPECTED_ADAPTER_SHA:
        raise ValueError("frozen input adapter hash is not the closed paired-eval adapter")
    if inputs["input_hashes"].get("plan_sha256") != runner.sha256_file(runner.PLAN):
        raise ValueError("paired-eval input plan hash is inconsistent with its worker lock")
    for case in inputs["all_cases"]:
        for key, expected in EXPECTED_PROTOCOL.items():
            if key == "max_steps" and case.get(key, runner.MAX_STEPS) == expected:
                continue
            if key != "max_steps" and case.get(key) not in (None, expected):
                raise ValueError(f"frozen case protocol mismatch: {case['case_id']} {key}")
    return inputs


def validate_step100_checkpoint() -> dict[str, Any]:
    """Validate the three immutable hashes recorded in the salvage plan."""

    identity = runner.validate_checkpoint(ROLE)
    hashes = identity.get("params_hashes", {})
    observed = {
        "params_native_sha256": hashes.get(runner.PARAMS_RELATIVE_NUL_SHA256_V1),
        "params_q0_sha256": hashes.get(runner.Q0_REPO_RELATIVE_SHA256_V1),
        "metadata_sha256": identity.get("metadata_sha256"),
    }
    expected = {
        "params_native_sha256": EXPECTED_STEP100_NATIVE_SHA,
        "params_q0_sha256": EXPECTED_STEP100_Q0_SHA,
        "metadata_sha256": EXPECTED_STEP100_METADATA_SHA,
    }
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"step100 checkpoint {key} mismatch: {observed.get(key)} != {value}")
    for key, value in {
        "checkpoint_step": 100,
        "training_steps": 100,
        "diagnostic_only": True,
        "role": "h300_step100_diagnostic",
        "run_id": runner.RUN_ID,
    }.items():
        if identity.get(key) != value:
            raise ValueError(f"step100 checkpoint identity mismatch: {key}")
    return {**identity, "frozen_hashes": expected}


def _r6_lock_and_identity(inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and verify the closed R6 lock without writing to its namespace."""

    _assert_no_partial(OLD_EVIDENCE_ROOT)
    _assert_no_partial(OLD_OUTPUT_ROOT)
    lock_path = OLD_EVIDENCE_ROOT / "r6" / "attempt.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"missing closed R6 worker lock: {lock_path}")
    lock = strict_json_load(lock_path)
    identity = runner.validate_checkpoint("r6")
    expected_source = {"path": str(runner.ADAPTER_PATH.resolve()), "sha256": EXPECTED_ADAPTER_SHA}
    if lock.get("status") != "completed" or lock.get("role") != "r6":
        raise ValueError("closed R6 worker lock is not completed")
    if lock.get("adapter_source") != expected_source:
        raise ValueError("closed R6 worker adapter source mismatch")
    if lock.get("identity") != identity:
        raise ValueError("closed R6 worker checkpoint identity mismatch")
    if lock.get("input_hashes") != inputs["input_hashes"]:
        raise ValueError("closed R6 worker frozen input hash mismatch")
    expected_fingerprints = {case["case_id"]: case["case_fingerprint"] for case in inputs["all_cases"]}
    if lock.get("case_fingerprints") != expected_fingerprints:
        raise ValueError("closed R6 worker case fingerprint lock mismatch")
    if lock.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("closed R6 worker protocol lock mismatch")
    return lock, identity


def _assert_row_contract(row: dict[str, Any], case: dict[str, Any], identity: dict[str, Any], role_root: Path, output_root: Path) -> None:
    if row.get("status") != "completed":
        raise ValueError(f"incomplete {identity['role']} case: {case['case_id']}")
    for key, expected in {
        "case_id": case["case_id"],
        "case_fingerprint": case["case_fingerprint"],
        "role": identity["role"],
        "norm_stats_sha256": runner.EXPECTED_NORM_SHA,
        "checkpoint_params_tree_sha256": identity["params_tree_sha256"],
        "env_seed": case.get("env_seed"),
        "policy_seed": case.get("policy_seed"),
        "init_state_index": case.get("init_state_index"),
        "init_state_sha256": case.get("init_state_sha256"),
    }.items():
        if expected is not None and row.get(key) != expected:
            raise ValueError(f"case/seed/input mismatch for {case['case_id']}: {key}")
    for key, expected in EXPECTED_PROTOCOL.items():
        if row.get(key) != expected:
            raise ValueError(f"protocol mismatch for {identity['role']}/{case['case_id']}: {key}")
    trace_path = Path(str(row.get("action_trace_path", ""))).resolve()
    video_path = Path(str(row.get("video_path", ""))).resolve()
    if not trace_path.is_relative_to(output_root.resolve()):
        raise ValueError(f"action trace escaped role output root: {trace_path}")
    if not video_path.is_relative_to(output_root.resolve()):
        raise ValueError(f"video escaped role output root: {video_path}")
    if not trace_path.is_file():
        raise FileNotFoundError(f"missing action trace: {trace_path}")
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise ValueError(f"missing or empty video: {video_path}")
    if not runner._case_file(role_root, case["case_id"]).is_file():
        raise FileNotFoundError(f"missing case file: {case['case_id']}")


def _load_role_rows(role: str, role_root: Path, output_root: Path, cases: list[dict[str, Any]], identity: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = runner._load_role_rows(role_root, role, cases, identity)
    nonzero_count = 0
    finite_count = 0
    video_count = 0
    for row, case in zip(rows, cases):
        _assert_row_contract(row, case, identity, role_root, output_root)
        trace = runner._validate_trace(row)
        finite_count += int(trace["finite"])
        nonzero_count += int(trace["nonzero"])
        video_count += 1
        row.update({"trace_finite": trace["finite"], "trace_nonzero": trace["nonzero"]})
    checks = {
        "case_count": len(rows),
        "finite_trace_count": finite_count,
        "nonzero_trace_count": nonzero_count,
        "video_count": video_count,
        "coverage_complete": len(rows) == len(cases),
        "all_zero": nonzero_count == 0,
    }
    return rows, checks


def _paired_summary(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], cases: list[dict[str, Any]], inputs: dict[str, Any], baseline_identity: dict[str, Any], candidate_identity: dict[str, Any], r6_lock: dict[str, Any]) -> dict[str, Any]:
    paired = runner.paired_summary(baseline_rows, candidate_rows, cases, input_hashes=inputs["input_hashes"])
    for pair, baseline, candidate in zip(paired["pairs"], baseline_rows, candidate_rows):
        pair["actions_finite"] = bool(baseline["trace_finite"] and candidate["trace_finite"])
        pair["baseline_nonzero"] = bool(baseline["trace_nonzero"])
        pair["candidate_nonzero"] = bool(candidate["trace_nonzero"])
    gates = runner.continuation_gates(paired["pairs"])
    trace_gate = all(row["trace_finite"] for row in baseline_rows + candidate_rows)
    video_gate = all(Path(str(row["video_path"])).is_file() and Path(str(row["video_path"])).stat().st_size > 0 for row in baseline_rows + candidate_rows)
    gates["gates"].update({
        "trace_shape_finite_nonzero_checks": trace_gate and any(row["trace_nonzero"] for row in baseline_rows) and any(row["trace_nonzero"] for row in candidate_rows),
        "video_files_present_nonempty": video_gate,
        "r6_lock_completed_and_exact": True,
        "step100_checkpoint_hashes_exact": True,
        "historical_inputs_read_only": True,
    })
    passed = all(gates["gates"].values())
    return {
        **paired,
        "schema_version": "track3_fast400_r6_vs_h300_step100_salvage_paired_summary_v1",
        "candidate": runner.RUN_ID,
        "candidate_role": candidate_identity["role"],
        "training_steps": 100,
        "diagnostic_only": True,
        "rollout_count": len(baseline_rows) + len(candidate_rows),
        "baseline_identity": baseline_identity,
        "candidate_identity": candidate_identity,
        "r6_lock_sha256": hashlib.sha256(runner.canonical_json(r6_lock).replace(b"\"pid\":", b"\"pid\":").strip()).hexdigest(),
        "trace_checks": {
            "r6": {"case_count": len(baseline_rows), "finite": sum(row["trace_finite"] for row in baseline_rows), "nonzero": sum(row["trace_nonzero"] for row in baseline_rows)},
            "step100": {"case_count": len(candidate_rows), "finite": sum(row["trace_finite"] for row in candidate_rows), "nonzero": sum(row["trace_nonzero"] for row in candidate_rows)},
        },
        "video_checks": {"r6": len(baseline_rows), "step100": len(candidate_rows), "all_present_nonempty": video_gate},
        "gates": gates["gates"],
        "behavior_gate_passed": passed,
        "status": "evaluated",
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }


def preflight(*, gpu: int = 0, evidence_dir: Path = SALVAGE_EVIDENCE_ROOT, output_dir: Path = SALVAGE_OUTPUT_ROOT) -> dict[str, Any]:
    """Perform S0 validation without creating a salvage artifact."""

    if gpu != 0:
        raise ValueError("step100 salvage is authorized only on GPU0")
    _assert_new_salvage_namespace(evidence_dir)
    _assert_new_salvage_namespace(output_dir)
    inputs = load_frozen_inputs()
    r6_lock, r6_identity = _r6_lock_and_identity(inputs)
    step100_identity = validate_step100_checkpoint()
    runtime = runner._runtime_preflight(ROLE, gpu)
    parent = runner.parent_environment(gpu)
    return {
        "schema_version": "track3_fast400_h300_step100_salvage_preflight_v1",
        "status": "passed",
        "gpu": runtime["gpu"],
        "case_counts": {"reverse_dev": 4, "c20": 20, "paired": 24},
        "protocol": {**EXPECTED_PROTOCOL, "norm": "QUANTILES"},
        "norm_stats_sha256": runner.EXPECTED_NORM_SHA,
        "adapter_source": _expected_adapter_source(),
        "plan_sha256": EXPECTED_PLAN_SHA,
        "r6_lock_status": r6_lock["status"],
        "r6_identity": r6_identity,
        "step100_identity": step100_identity,
        "parent_renderer_runtime": {key: parent[key] for key in runner.PARENT_RENDERER_ENV_KEYS},
        "historical_evidence_root": str(OLD_EVIDENCE_ROOT.resolve()),
        "historical_output_root": str(OLD_OUTPUT_ROOT.resolve()),
        "evidence_dir": str(evidence_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }


def compare(*, evidence_dir: Path = SALVAGE_EVIDENCE_ROOT, output_dir: Path = SALVAGE_OUTPUT_ROOT) -> dict[str, Any]:
    """Read the old R6 and new step100 roles, then write only new artifacts."""

    _assert_new_salvage_namespace(evidence_dir)
    _assert_new_salvage_namespace(output_dir)
    _assert_merge_artifacts_absent(evidence_dir)
    old_evidence_before = runner.namespace_fingerprint(OLD_EVIDENCE_ROOT)
    old_output_before = runner.namespace_fingerprint(OLD_OUTPUT_ROOT)
    inputs = load_frozen_inputs()
    r6_lock, r6_identity = _r6_lock_and_identity(inputs)
    step100_identity = validate_step100_checkpoint()
    candidate_root = evidence_dir.resolve() / ROLE
    candidate_output = output_dir.resolve() / ROLE
    candidate_lock_path = candidate_root / "attempt.lock.json"
    if not candidate_lock_path.is_file():
        raise FileNotFoundError(f"missing step100 worker lock: {candidate_lock_path}")
    candidate_lock = strict_json_load(candidate_lock_path)
    if candidate_lock.get("status") != "completed" or candidate_lock.get("role") != ROLE:
        raise ValueError("step100 worker lock is not completed")
    if candidate_lock.get("adapter_source", {}).get("sha256") != EXPECTED_ADAPTER_SHA:
        raise ValueError("step100 worker adapter source mismatch")
    locked_step100_identity = {key: value for key, value in step100_identity.items() if key != "frozen_hashes"}
    if candidate_lock.get("identity") != locked_step100_identity:
        raise ValueError("step100 worker checkpoint identity mismatch")
    if candidate_lock.get("input_hashes") != inputs["input_hashes"]:
        raise ValueError("step100 worker frozen input hash mismatch")
    if candidate_lock.get("case_fingerprints") != {case["case_id"]: case["case_fingerprint"] for case in inputs["all_cases"]}:
        raise ValueError("step100 worker case fingerprint lock mismatch")
    if candidate_lock.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("step100 worker protocol lock mismatch")
    _assert_no_partial(candidate_root)
    _assert_no_partial(candidate_output)
    baseline_rows, baseline_checks = _load_role_rows("r6", OLD_EVIDENCE_ROOT / "r6", OLD_OUTPUT_ROOT / "r6", inputs["all_cases"], r6_identity)
    candidate_rows, candidate_checks = _load_role_rows(ROLE, candidate_root, candidate_output, inputs["all_cases"], step100_identity)
    paired = _paired_summary(baseline_rows, candidate_rows, inputs["all_cases"], inputs, r6_identity, step100_identity, r6_lock)
    paired["role_checks"] = {"r6": baseline_checks, "step100": candidate_checks}
    old_evidence_after = runner.namespace_fingerprint(OLD_EVIDENCE_ROOT)
    old_output_after = runner.namespace_fingerprint(OLD_OUTPUT_ROOT)
    if old_evidence_before != old_evidence_after or old_output_before != old_output_after:
        raise RuntimeError("historical R6 evidence/output changed during read-only comparison")
    paired["historical_namespace_fingerprints"] = {"evidence": old_evidence_before["sha256"], "output": old_output_before["sha256"]}
    runner.atomic_json(evidence_dir / "paired_summary.json", paired)
    passed = bool(paired["behavior_gate_passed"])
    release = {
        "schema_version": "track3_fast400_h300_step100_salvage_release_decision_v1",
        "status": "step100_salvage_passed_release_authorized" if passed else "step100_behavior_gate_failed_no_zip",
        "behavior_gate_passed": passed,
        "fold_authorized": passed,
        "step100_salvage_promoted_for_release": passed,
        "zip_created": False,
        "fold_started": False,
        "candidate": runner.RUN_ID,
        "candidate_role": step100_identity["role"],
        "training_steps": 100,
        "diagnostic_only": True,
        "adapter_source": _expected_adapter_source(),
        "baseline": r6_identity,
        "candidate_identity": step100_identity,
        "paired_summary": str((evidence_dir / "paired_summary.json").resolve()),
        "norm_stats_sha256": runner.EXPECTED_NORM_SHA,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
        "reason": "reverse delta must be >= +1/4 and C20 delta must be >= -2/20; exact coverage, finite traces, nonzero roles and nonempty videos are required",
        "created_at_unix": time.time(),
    }
    runner.atomic_json(evidence_dir / "release_decision.json", release)
    return release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "compare"):
        child = sub.add_parser(command)
        child.add_argument("--evidence-dir", type=Path, default=SALVAGE_EVIDENCE_ROOT)
        child.add_argument("--output-dir", type=Path, default=SALVAGE_OUTPUT_ROOT)
    sub.add_parser("status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(evidence_dir=args.evidence_dir, output_dir=args.output_dir)
    elif args.command == "compare":
        result = compare(evidence_dir=args.evidence_dir, output_dir=args.output_dir)
    else:
        result = {
            "schema_version": "track3_fast400_h300_step100_salvage_status_v1",
            "status": "read_only",
            "evidence_dir": str(SALVAGE_EVIDENCE_ROOT.resolve()),
            "output_dir": str(SALVAGE_OUTPUT_ROOT.resolve()),
            "salvage_exists": SALVAGE_EVIDENCE_ROOT.exists(),
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("status") in {"passed", "read_only", "step100_salvage_passed_release_authorized", "step100_behavior_gate_failed_no_zip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
