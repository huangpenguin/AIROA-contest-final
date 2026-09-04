#!/usr/bin/env python3
"""Role-aware background-predicate convergence release gates.

K0/K2 consume only the committed J2 raw worker reports.  No simulator is
started in this module's CPU phases, and the previous J2/J4 namespace remains
read-only input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.axis_pi05 import track3_operator_scoped_stability as prior

ROOT = prior.ROOT
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_BACKGROUND_PREDICATE_CONVERGENCE_RELEASE_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_OPERATOR_SCOPED_J0_J4_AUDIT_20260904.md"
PRIOR_ROOT = ROOT / "data/reports/axis_libero_30k/track3_operator_scoped_stability"
PRIOR_J0 = PRIOR_ROOT / "j0_release_lock.json"
PRIOR_J1 = PRIOR_ROOT / "j1_contract.json"
PRIOR_J2 = PRIOR_ROOT / "j2_summary.json"
PRIOR_J2_REVERSE = PRIOR_ROOT / "j2_reverse_release.json"
PRIOR_J2_FORWARD = PRIOR_ROOT / "j2_forward_controls.json"
PRIOR_J4 = PRIOR_ROOT / "summary.json"
OLD_I2 = prior.OLD_I2
OLD_I2_SHA256 = prior.OLD_I2_SHA256
TASKS = prior.TASKS
FAMILIES = prior.FAMILIES
CHECKPOINTS = tuple(range(0, 21))
RELEASE_STEPS = tuple(range(5, 21))
WORKER_SCHEMA = "track3_operator_scoped_stability_j2_worker_v1"
EXPECTED_ROOT_COMMIT = "f969393f3e16dc94985675083d09daa712cc7c04"
EXPECTED_EXECUTION_COMMIT = "bc2419e62ff05a9584d1d6033d22b0e55d03a240"
EXPECTED_AUDIT_PARENT = "3acbde8046563e3e550f78feaf19ad97cb391b79"
EXPECTED_AUDIT_COMMIT = "417b977d8f8eed9c72c1e6f95eda28ecd3889593"
EXPECTED_OPENPI = "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with partial.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def git_output(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path) -> list[str]:
    return [line for line in git_output(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def finite_tree(value: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return errors
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(path)
        return errors
    if isinstance(value, Mapping):
        for key, child in value.items():
            errors.extend(finite_tree(child, f"{path}.{key}"))
        return errors
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            errors.extend(finite_tree(child, f"{path}[{index}]"))
        return errors
    return errors


def gate(executed: bool, value: bool | None, *, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    result = {"executed": executed, "state": "not_evaluated" if not executed else ("passed" if value else "failed"), "required": required}
    if reason:
        result["reason"] = reason
    return result


def all_required(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in gates.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def _prior_summary() -> dict[str, Any]:
    return json.loads(PRIOR_J2.read_text(encoding="utf-8"))


def raw_paths_from_prior() -> list[Path]:
    paths = _prior_summary().get("raw_fresh_process_reports")
    if not isinstance(paths, list) or len(paths) != 15 or any(not isinstance(path, str) for path in paths):
        raise RuntimeError("J2 raw_fresh_process_reports must contain exactly 15 explicit paths")
    result = []
    for path_string in paths:
        path = Path(path_string).resolve()
        if path.parent != (PRIOR_ROOT / "runs").resolve() or not path.name.startswith("j2_task"):
            raise RuntimeError(f"raw path outside approved J2 raw directory: {path}")
        result.append(path)
    return result


def expected_identities() -> set[tuple[int, str, int]]:
    return {(task, "reverse", run) for task in TASKS for run in range(1, 4)} | {(task, "forward", run) for task in (20, 19) for run in range(1, 4)}


def validate_raw(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"raw report missing: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema_version") != WORKER_SCHEMA:
        raise RuntimeError(f"schema mismatch: {path}")
    if row.get("status") != "passed" or row.get("process_returncode") != 0 or row.get("worker_returncode_contract") is not True:
        raise RuntimeError(f"raw process gate failed: {path}")
    identity = (int(row["task_index"]), str(row["kind"]), int(row["run_index"]))
    checkpoints = row.get("checkpoints")
    if not isinstance(checkpoints, list) or [item.get("step") for item in checkpoints] != list(CHECKPOINTS):
        raise RuntimeError(f"checkpoint shape mismatch: {path}")
    for sample in checkpoints:
        if not sample.get("initial", {}).get("truth_complete") or not sample.get("goal", {}).get("truth_complete"):
            raise RuntimeError(f"truth incomplete: {path} step {sample.get('step')}")
        if not sample.get("initial", {}).get("detail_complete") or not sample.get("goal", {}).get("detail_complete"):
            raise RuntimeError(f"detail incomplete: {path} step {sample.get('step')}")
        if sample.get("contact_matrix", {}).get("complete") is not True:
            raise RuntimeError(f"contact matrix incomplete: {path} step {sample.get('step')}")
        for predicate_set in (sample["initial"], sample["goal"]):
            for predicate in predicate_set.get("predicates", []):
                if predicate.get("truth_executed") is not True or predicate.get("detail_executed") is not True or not isinstance(predicate.get("value"), bool):
                    raise RuntimeError(f"predicate truth/detail shape mismatch: {path} step {sample.get('step')}")
        errors = finite_tree(sample)
        if errors:
            raise RuntimeError(f"non-finite raw telemetry {path}: {errors[0]}")
    digest = {
        "relative_path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "task_index": identity[0],
        "kind": identity[1],
        "run_index": identity[2],
        "native_reset_state_sha256": row.get("native_reset_state_sha256"),
        "checkpoint_steps": [item["step"] for item in checkpoints],
    }
    return row, digest


def _input_lock() -> dict[str, Any]:
    paths = [PLAN, AUDIT, PRIOR_J0, PRIOR_J1, PRIOR_J2, PRIOR_J2_REVERSE, PRIOR_J2_FORWARD, PRIOR_J4, OLD_I2]
    missing = [str(path.resolve()) for path in paths if not path.is_file()]
    locked = {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths if path.is_file()}
    return {"missing": missing, "locked": locked}


def gpu_read_only_snapshot() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-2000:], "read_only": True}
    except Exception as exc:
        return {"returncode": None, "error": f"{type(exc).__name__}: {exc}", "read_only": True}


def run_k0() -> dict[str, Any]:
    paths = raw_paths_from_prior()
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    identities: set[tuple[int, str, int]] = set()
    for path in paths:
        try:
            row, record = validate_raw(path)
            rows.append(row)
            records.append(record)
            identity = (record["task_index"], record["kind"], record["run_index"])
            if identity in identities:
                errors.append(f"duplicate identity: {identity}")
            identities.add(identity)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    errors.extend(f"missing identity: {identity}" for identity in sorted(expected_identities() - identities))
    errors.extend(f"unexpected identity: {identity}" for identity in sorted(identities - expected_identities()))
    manifest_core = {"schema_version": "track3_background_convergence_k0_raw_manifest_v1", "created_at_utc": now(), "raw_reports": records, "identity_set": [list(identity) for identity in sorted(identities)], "expected_identity_set": [list(identity) for identity in sorted(expected_identities())]}
    manifest = dict(manifest_core)
    manifest["manifest_canonical_sha256"] = canonical_hash(manifest_core)
    write_json(OUT_ROOT / "k0_raw_manifest.json", manifest)
    prior_j4 = json.loads(PRIOR_J4.read_text(encoding="utf-8"))
    old_summary = json.loads(prior.OLD_SUMMARY.read_text(encoding="utf-8"))
    root_status = porcelain(ROOT)
    allowed = {Path(__file__).resolve(), ROOT / "tests/test_track3_background_predicate_convergence.py"}
    preexisting = [line for line in root_status if (ROOT / line[3:].strip()).resolve() not in {path.resolve() for path in allowed}]
    nested = ROOT / "third_party/axis_openpi"
    lock_input = _input_lock()
    hard_gate = {
        "input_files_exist": not lock_input["missing"],
        "root_input_commit_is_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", EXPECTED_ROOT_COMMIT, git_output(ROOT, ("rev-parse", "HEAD"))], check=False).returncode == 0,
        "execution_commit_is_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", EXPECTED_EXECUTION_COMMIT, git_output(ROOT, ("rev-parse", "HEAD"))], check=False).returncode == 0,
        "audit_parent_is_ancestor": subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", EXPECTED_AUDIT_PARENT, git_output(ROOT, ("rev-parse", "HEAD"))], check=False).returncode == 0,
        "audit_commit_locked": git_output(ROOT, ("rev-parse", "HEAD")) == EXPECTED_AUDIT_COMMIT,
        "nested_openpi_locked_and_clean": git_output(nested, ("rev-parse", "HEAD")) == EXPECTED_OPENPI and not porcelain(nested),
        "root_preexisting_changes_absent": not preexisting,
        "old_i2_hash_locked": sha256_file(OLD_I2) == OLD_I2_SHA256,
        "raw_identity_and_schema_closed": not errors and len(records) == 15,
        "prior_j3_not_evaluated": prior_j4.get("phases", {}).get("J3", {}).get("state") == "not_evaluated",
        "reverse_demo_counts_zero": all(int(old_summary.get("counts", {}).get(key, 0)) == 0 for key in ("exact_reverse_demo", "pending_autogen", "accepted_reverse_demo")),
    }
    report = {
        "schema_version": "track3_background_convergence_k0_v1",
        "status": "track3_background_convergence_input_locked" if all(hard_gate.values()) else "track3_background_convergence_evidence_closure_failed",
        "created_at_utc": now(),
        "plan": str(PLAN.resolve()),
        "audit": str(AUDIT.resolve()),
        "input_root_commit": EXPECTED_ROOT_COMMIT,
        "input_execution_commit": EXPECTED_EXECUTION_COMMIT,
        "input_audit_parent": EXPECTED_AUDIT_PARENT,
        "audit_commit_at_k0": git_output(ROOT, ("rev-parse", "HEAD")),
        "nested_openpi_commit": EXPECTED_OPENPI,
        "root_status_at_k0": root_status,
        "preexisting_root_status_at_k0": preexisting,
        "libero_plus_dirty_read_only": porcelain(ROOT / "LIBERO-plus"),
        "locked_inputs": lock_input["locked"],
        "missing_inputs": lock_input["missing"],
        "prior_j2_summary_sha256": sha256_file(PRIOR_J2),
        "prior_j4_summary_sha256": sha256_file(PRIOR_J4),
        "old_i2_sha256": sha256_file(OLD_I2),
        "raw_manifest": str((OUT_ROOT / "k0_raw_manifest.json").resolve()),
        "raw_report_count": len(records),
        "raw_bytes_total": sum(record["bytes"] for record in records),
        "raw_manifest_canonical_sha256": manifest["manifest_canonical_sha256"],
        "raw_errors": errors,
        "prior_j2_exact_vector": {"status": prior_j4.get("j2_outcome", {}).get("status"), "reverse_pass": prior_j4.get("j2_outcome", {}).get("new_reverse", {}).get("pass_count"), "failed_run": prior_j4.get("j2_outcome", {}).get("failed_runs")},
        "prior_j3": {"state": prior_j4.get("phases", {}).get("J3", {}).get("state"), "reason": prior_j4.get("phases", {}).get("J3", {}).get("reason")},
        "gpu_read_only_snapshot": gpu_read_only_snapshot(),
        "hard_gate": hard_gate,
        "fresh_checkout_validation": {"state": "not_evaluated", "required_after_k0_evidence_commit": True},
        "resource_boundary": {"cpu_filesystem_only": True, "gpu_touched": False, "se3": False, "training": False, "raw_reports_only": True},
    }
    write_json(OUT_ROOT / "k0_input_lock.json", report)
    return report


def run_k0_verify(worktree: Path) -> dict[str, Any]:
    """Verify the committed raw closure from an independent checkout."""

    manifest_path = OUT_ROOT / "k0_raw_manifest.json"
    lock_path = OUT_ROOT / "k0_input_lock.json"
    if not manifest_path.is_file() or not lock_path.is_file():
        raise RuntimeError("K0 manifest and lock must exist before fresh-checkout verification")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("raw_reports", [])
    checks: list[dict[str, Any]] = []
    for record in records:
        relative = Path(str(record["relative_path"]))
        path = worktree / relative
        exists = path.is_file()
        checks.append({
            "relative_path": str(relative),
            "exists": exists,
            "bytes": path.stat().st_size if exists else None,
            "expected_bytes": record.get("bytes"),
            "sha256": sha256_file(path) if exists else None,
            "expected_sha256": record.get("sha256"),
            "passed": exists and path.stat().st_size == record.get("bytes") and sha256_file(path) == record.get("sha256"),
        })
    observed_commit = git_output(worktree, ("rev-parse", "HEAD"))
    result = {
        "state": "passed" if len(checks) == 15 and all(item["passed"] for item in checks) else "failed",
        "worktree": str(worktree.resolve()),
        "observed_commit": observed_commit,
        "manifest_canonical_sha256": manifest.get("manifest_canonical_sha256"),
        "file_count": len(checks),
        "checks": checks,
    }
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["fresh_checkout_validation"] = result
    lock["status"] = "track3_background_convergence_input_locked" if result["state"] == "passed" and all(lock.get("hard_gate", {}).values()) else lock.get("status")
    write_json(lock_path, lock)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k0", action="store_true")
    parser.add_argument("--k0-verify")
    args = parser.parse_args()
    if args.k0:
        result = run_k0()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "track3_background_convergence_input_locked" else 1
    if args.k0_verify:
        result = run_k0_verify(Path(args.k0_verify))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["state"] == "passed" else 1
    parser.error("use --k0")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
