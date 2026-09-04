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
import traceback
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
K3_ROOT = OUT_ROOT / "k3_gpu3_release"
K3_TASK_ORDER = (10, 20, 19)
K3_WORKER_TIMEOUT_S = 600


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


def checkpoint_map(row: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    checkpoints = row.get("checkpoints")
    if not isinstance(checkpoints, list):
        return {}
    result: dict[int, Mapping[str, Any]] = {}
    for sample in checkpoints:
        if isinstance(sample, Mapping) and isinstance(sample.get("step"), int):
            result[int(sample["step"])] = sample
    return result


def _predicate_series(
    *,
    atom: str,
    role: str,
    values: Sequence[bool],
    steps: Sequence[int],
    strict_expected: bool | None,
) -> dict[str, Any]:
    transitions = []
    for left_step, right_step, left, right in zip(steps, steps[1:], values, values[1:]):
        if left == right:
            continue
        transitions.append({
            "from_step": int(left_step),
            "to_step": int(right_step),
            "from_value": bool(left),
            "to_value": bool(right),
            "direction": "false_to_true" if not left and right else "true_to_false",
        })
    false_to_true = [row for row in transitions if row["direction"] == "false_to_true"]
    true_to_false = [row for row in transitions if row["direction"] == "true_to_false"]
    if role in {"operator_critical", "operator_critical_goal"}:
        passed = strict_expected is not None and all(value is strict_expected for value in values)
        reason = None if passed else "critical_strict_truth_mismatch"
    elif role == "background_initializer":
        passed = not true_to_false and len(false_to_true) <= 1
        reason = None if passed else "background_non_monotonic_or_repeated_rise"
    else:
        passed = False
        reason = "unknown_role"
    return {
        "atom": atom,
        "role": role,
        "steps": [int(step) for step in steps],
        "values": [bool(value) for value in values],
        "transitions": transitions,
        "false_to_true": false_to_true,
        "true_to_false": true_to_false,
        "strict_expected": strict_expected,
        "monotonic_non_decreasing": not true_to_false,
        "passed": passed,
        "reason": reason,
    }


def predicate_ledger(
    row: Mapping[str, Any],
    *,
    task_index: int,
    kind: str,
    section: str,
    critical_atom: str,
    critical_role: str,
    strict_expected: bool | None,
) -> dict[str, Any]:
    """Recompute predicate roles and transitions from raw checkpoint telemetry."""

    samples_by_step = checkpoint_map(row)
    samples = [samples_by_step.get(step) for step in RELEASE_STEPS]
    errors: list[str] = []
    if len(samples) != len(RELEASE_STEPS) or any(sample is None for sample in samples):
        return {
            "task_index": task_index,
            "kind": kind,
            "section": section,
            "status": "failed",
            "steps": list(RELEASE_STEPS),
            "atoms": [],
            "series": [],
            "errors": ["release_window_incomplete"],
        }
    first_predicates = samples[0].get(section, {}).get("predicates", [])
    if not isinstance(first_predicates, list) or not first_predicates:
        errors.append("predicate_list_missing")
        first_predicates = []
    atoms = [item.get("atom") for item in first_predicates if isinstance(item, Mapping)]
    if any(not isinstance(atom, str) for atom in atoms):
        errors.append("atom_name_missing_or_not_string")
    if len(atoms) != len(set(atoms)):
        errors.append("duplicate_atom")
    if critical_atom not in atoms or atoms.count(critical_atom) != 1:
        errors.append("critical_atom_missing_or_not_unique")
    rows_by_step: dict[int, dict[str, Mapping[str, Any]]] = {}
    for step, sample in zip(RELEASE_STEPS, samples):
        predicates = sample.get(section, {}).get("predicates", [])
        if not isinstance(predicates, list):
            errors.append(f"step_{step}:predicate_list_missing")
            continue
        step_atoms = [item.get("atom") for item in predicates if isinstance(item, Mapping)]
        if step_atoms != atoms:
            errors.append(f"step_{step}:atom_order_or_membership_changed")
        rows_by_step[step] = {str(item.get("atom")): item for item in predicates if isinstance(item, Mapping)}
    series: list[dict[str, Any]] = []
    for atom in atoms:
        expected_role = critical_role if atom == critical_atom else "background_initializer"
        values: list[bool] = []
        row_roles: list[str] = []
        for step in RELEASE_STEPS:
            item = rows_by_step.get(step, {}).get(atom)
            if item is None:
                errors.append(f"{atom}:missing_at_step_{step}")
                continue
            value = item.get("value")
            role = item.get("role")
            row_roles.append(str(role))
            if item.get("truth_executed") is not True or item.get("detail_executed") is not True:
                errors.append(f"{atom}:truth_or_detail_not_executed_at_step_{step}")
            if not isinstance(value, bool):
                errors.append(f"{atom}:non_bool_value_at_step_{step}")
            else:
                values.append(value)
            if role != expected_role:
                errors.append(f"{atom}:role_mismatch_expected_{expected_role}_got_{role}")
        if len(values) != len(RELEASE_STEPS):
            continue
        if len(set(row_roles)) != 1:
            errors.append(f"{atom}:role_changed")
        series.append(_predicate_series(atom=atom, role=expected_role, values=values, steps=RELEASE_STEPS, strict_expected=strict_expected if atom == critical_atom else None))
    critical = next((item for item in series if item["atom"] == critical_atom), None)
    background = [item for item in series if item["atom"] != critical_atom]
    passed = not errors and critical is not None and critical["passed"] and all(item["passed"] for item in background)
    return {
        "task_index": task_index,
        "kind": kind,
        "section": section,
        "critical_atom": critical_atom,
        "critical_role": critical_role,
        "strict_expected": strict_expected,
        "steps": list(RELEASE_STEPS),
        "atoms": atoms,
        "series": series,
        "critical_transition_count": len(critical["transitions"]) if critical else None,
        "background_false_to_true_count": sum(len(item["false_to_true"]) for item in background),
        "background_true_to_false_count": sum(len(item["true_to_false"]) for item in background),
        "background_repeated_rise_count": sum(max(0, len(item["false_to_true"]) - 1) for item in background),
        "errors": errors,
        "status": "passed" if passed else "failed",
    }


def k1_contract() -> dict[str, Any]:
    core = {
        "schema_version": "track3_background_convergence_k1_monotonic_contract_core_v1",
        "release_window": {"start_step": 5, "end_step": 20, "steps": list(RELEASE_STEPS), "frequency_hz": 20},
        "roles": {
            "source": {"role": "operator_critical", "strict_value": True},
            "reverse_goal": {"role": "operator_critical_goal", "strict_value": False},
            "background": {"role": "background_initializer", "allowed_transitions": ["none", "false_to_true"], "max_false_to_true": 1, "forbidden_transitions": ["true_to_false", "backtracking"]},
        },
        "operator_atoms": {
            str(task): {"source_atom": prior.SOURCE_ATOM[task], "reverse_goal_atom": prior.GOAL_ATOM[task], "family": FAMILIES[task]}
            for task in TASKS
        },
        "fail_closed": ["unknown_role", "missing_atom", "duplicate_atom", "atom_order_change", "missing_step", "truth_not_executed", "detail_not_executed", "non_bool_value"],
        "physical_gates_unchanged": {"robot": "prior_J1_component_thresholds", "participant": "prior_J1_operator_participant_thresholds", "contact": "prior_J1_step5_baseline_body_pair_contract", "finite": True, "scene_names": True},
        "old_evidence_policy": {"old_exact_vector_is_read_only": True, "old_j2_reverse_result": "8/9", "old_j2_failed_run": [19, "reverse", 1], "old_i2_immutable": True},
        "provenance": {"simulator_in_k1": False, "raw_reports_in_k1": False, "hdf5_state_used": False, "bddl_modified": False},
    }
    report = dict(core)
    report["contract_canonical_sha256"] = canonical_hash(core)
    return report


def run_k1() -> dict[str, Any]:
    k0 = json.loads((OUT_ROOT / "k0_input_lock.json").read_text(encoding="utf-8"))
    old_summary = json.loads(PRIOR_J4.read_text(encoding="utf-8"))
    contract = k1_contract()
    hard_gate = {
        "k0_locked": k0.get("status") == "track3_background_convergence_input_locked" and k0.get("fresh_checkout_validation", {}).get("state") == "passed",
        "source_atoms_fixed": all(contract["operator_atoms"][str(task)]["source_atom"] == prior.SOURCE_ATOM[task] for task in TASKS),
        "reverse_goal_atoms_fixed": all(contract["operator_atoms"][str(task)]["reverse_goal_atom"] == prior.GOAL_ATOM[task] for task in TASKS),
        "role_rules_complete": set(contract["roles"]) == {"source", "reverse_goal", "background"},
        "fail_closed_rules_complete": set(contract["fail_closed"]) >= {"unknown_role", "missing_atom", "duplicate_atom", "missing_step", "truth_not_executed", "detail_not_executed", "non_bool_value"},
        "old_i2_unchanged": sha256_file(OLD_I2) == OLD_I2_SHA256,
        "old_exact_vector_preserved": old_summary.get("j2_outcome", {}).get("new_reverse", {}).get("pass_count") == 8,
        "reverse_demo_counts_zero": all(int(old_summary.get("reverse_demo_counts", {}).get(key, 0)) == 0 for key in ("exact_reverse_demo", "pending_autogen", "accepted_reverse_demo")),
    }
    report = {
        **contract,
        "schema_version": "track3_background_convergence_k1_monotonic_contract_v1",
        "status": "track3_background_convergence_contract_validated" if all(hard_gate.values()) else "track3_background_convergence_contract_failed",
        "created_at_utc": now(),
        "k0_input_lock_sha256": sha256_file(OUT_ROOT / "k0_input_lock.json"),
        "hard_gate": hard_gate,
        "resource_boundary": {"cpu_filesystem_only": True, "simulator_started": False, "gpu3_touched": False, "se3": False, "training": False},
    }
    write_json(OUT_ROOT / "k1_monotonic_contract.json", report)
    return report


def _manifest_records() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = OUT_ROOT / "k0_raw_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("raw_reports", [])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for record in records:
        path = ROOT / str(record.get("relative_path", ""))
        try:
            row, digest = validate_raw(path)
            if digest != record:
                raise RuntimeError("manifest digest mismatch")
            tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(record["relative_path"])], capture_output=True, text=True, check=False).returncode == 0
            if not tracked:
                raise RuntimeError("raw report is not Git tracked")
            rows.append(row)
        except Exception as exc:
            errors.append({"relative_path": record.get("relative_path"), "error": f"{type(exc).__name__}: {exc}"})
    return manifest, rows, errors


def _physical_gate(row: Mapping[str, Any], task_index: int, kind: str) -> dict[str, Any]:
    samples_by_step = checkpoint_map(row)
    release = [samples_by_step[step] for step in RELEASE_STEPS if step in samples_by_step]
    old = __import__("tools.axis_pi05.track3_dynamic_gate_reconciliation", fromlist=["module"])
    gates: dict[str, Any] = {
        "worker_process": gate(row.get("status") == "passed" and row.get("process_returncode") == 0 and row.get("worker_returncode_contract") is True, True, reason=None if row.get("status") == "passed" else "worker_status_or_returncode"),
        "release_window": gate(len(release) == len(RELEASE_STEPS), len(release) == len(RELEASE_STEPS), reason="release_window_incomplete" if len(release) != len(RELEASE_STEPS) else None),
        "truth_detail": gate(bool(release), all(sample.get("initial", {}).get("truth_complete") and sample.get("goal", {}).get("truth_complete") and sample.get("initial", {}).get("detail_complete") and sample.get("goal", {}).get("detail_complete") for sample in release), reason="truth_or_detail_incomplete"),
        "finite": gate(bool(release), not finite_tree(release), reason="non_finite_telemetry" if finite_tree(release) else None),
        "scene_names_invariant": gate(bool(release), bool(release) and len({tuple(sorted(sample.get("scene", {}).get("object_and_site_poses", {}).keys())) for sample in release}) == 1, reason="scene_name_set_changed"),
    }
    if release:
        baseline_matrix = release[0].get("contact_matrix", {})
        contact_rows = [prior._contact_contract_from_matrix(task_index, sample.get("contact_matrix", {}), baseline_matrix=baseline_matrix) for sample in release]
        contact_pass = all(item.get("state") == "passed" and item.get("complete") is True for item in contact_rows)
        participant = prior.participant_stability(release, task_index, old)
        robot = prior._robot_stability(release, old)
        gates["contact_matrix_and_family_rule"] = gate(True, contact_pass, reason="family_contact_rule_failed" if not contact_pass else None)
        gates["operator_participant_stability"] = gate(True, participant.get("passed") is True, reason="operator_participant_drift" if participant.get("passed") is not True else None)
        gates["robot_component_stability"] = gate(True, robot.get("state") == "passed", reason="robot_component_drift" if robot.get("state") != "passed" else None)
        return {"status": "passed" if all(item.get("state") == "passed" for item in gates.values()) else "failed", "gates": gates, "contact_samples": contact_rows, "participant": participant, "robot": robot}
    gates["contact_matrix_and_family_rule"] = gate(False, None, reason="release_window_incomplete")
    gates["operator_participant_stability"] = gate(False, None, reason="release_window_incomplete")
    gates["robot_component_stability"] = gate(False, None, reason="release_window_incomplete")
    return {"status": "failed", "gates": gates, "contact_samples": [], "participant": gate(False, None), "robot": gate(False, None)}


def _compact_raw_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"task_index": row.get("task_index"), "operator_family": row.get("operator_family"), "kind": row.get("kind"), "run_index": row.get("run_index"), "source_raw_report": f"data/reports/axis_libero_30k/track3_operator_scoped_stability/runs/j2_task{row.get('task_index')}_{row.get('kind')}_run_{row.get('run_index')}.json"}


def run_k2() -> dict[str, Any]:
    k0 = json.loads((OUT_ROOT / "k0_input_lock.json").read_text(encoding="utf-8"))
    k1 = json.loads((OUT_ROOT / "k1_monotonic_contract.json").read_text(encoding="utf-8"))
    if k0.get("status") != "track3_background_convergence_input_locked" or k0.get("fresh_checkout_validation", {}).get("state") != "passed":
        raise RuntimeError("K0 gate is not passed")
    if k1.get("status") != "track3_background_convergence_contract_validated":
        raise RuntimeError("K1 gate is not passed")
    manifest, rows, manifest_errors = _manifest_records()
    ledger_rows: list[dict[str, Any]] = []
    reverse_rows: list[dict[str, Any]] = []
    forward_rows: list[dict[str, Any]] = []
    old = json.loads(PRIOR_J4.read_text(encoding="utf-8"))
    for row in rows:
        task = int(row["task_index"])
        kind = str(row["kind"])
        if kind == "reverse":
            initial = predicate_ledger(row, task_index=task, kind=kind, section="initial", critical_atom=prior.SOURCE_ATOM[task], critical_role="operator_critical", strict_expected=True)
            goal = predicate_ledger(row, task_index=task, kind=kind, section="goal", critical_atom=prior.GOAL_ATOM[task], critical_role="operator_critical_goal", strict_expected=False)
            physical = _physical_gate(row, task, kind)
            result = {**_compact_raw_identity(row), "status": "passed" if initial["status"] == "passed" and goal["status"] == "passed" and physical["status"] == "passed" else "failed", "predicate": {"initial": initial, "goal": goal}, "physical": physical}
            reverse_rows.append(result)
        else:
            initial = predicate_ledger(row, task_index=task, kind=kind, section="initial", critical_atom=prior.SOURCE_ATOM[task], critical_role="operator_critical", strict_expected=None)
            goal = predicate_ledger(row, task_index=task, kind=kind, section="goal", critical_atom=prior.SOURCE_ATOM[task], critical_role="operator_critical", strict_expected=None)
            physical = _physical_gate(row, task, kind)
            forward_rows.append({**_compact_raw_identity(row), "status": "passed" if physical["gates"]["worker_process"]["state"] == "passed" and physical["gates"]["release_window"]["state"] == "passed" and physical["gates"]["truth_detail"]["state"] == "passed" and physical["gates"]["finite"]["state"] == "passed" and physical["gates"]["scene_names_invariant"]["state"] == "passed" and physical["gates"]["contact_matrix_and_family_rule"]["state"] == "passed" else "failed", "predicate_diagnostics": {"initial": initial, "goal": goal}, "physical": physical})
        ledger_rows.append({**_compact_raw_identity(row), "sections": {"initial": initial, "goal": goal}})
    reverse_pass = len(reverse_rows) == 9 and all(row["status"] == "passed" for row in reverse_rows)
    forward_pass = len(forward_rows) == 6 and all(row["status"] == "passed" for row in forward_rows)
    reverse_ledgers = [row["sections"] for row in ledger_rows if row["kind"] == "reverse"]
    critical_transitions = sum(section.get("critical_transition_count", 0) for row in reverse_ledgers for section in (row["initial"], row["goal"]))
    goal_transitions = sum(row["sections"]["goal"].get("critical_transition_count", 0) for row in ledger_rows if row["kind"] == "reverse")
    background_true_to_false = sum(section.get("background_true_to_false_count", 0) for row in ledger_rows for section in (row["sections"]["initial"], row["sections"]["goal"]))
    background_repeated_rise = sum(section.get("background_repeated_rise_count", 0) for row in ledger_rows for section in (row["sections"]["initial"], row["sections"]["goal"]))
    background_false_to_true = sum(section.get("background_false_to_true_count", 0) for row in ledger_rows for section in (row["sections"]["initial"], row["sections"]["goal"]))
    old_exact = {"status": old.get("status"), "reverse_pass": old.get("j2_outcome", {}).get("new_reverse", {}).get("pass_count"), "required": "8/9", "failed_run": old.get("j2_outcome", {}).get("failed_runs")}
    hard_gate = {
        "k0_manifest_fresh_checkout_15_of_15": k0.get("fresh_checkout_validation", {}).get("state") == "passed" and len(manifest.get("raw_reports", [])) == 15 and not manifest_errors,
        "raw_identity_closed": len(rows) == 15 and not manifest_errors,
        "reverse_9_of_9": reverse_pass,
        "forward_6_of_6": forward_pass,
        "worker_exception_zero": sum(row.get("status") == "worker_exception" for row in rows) == 0,
        "critical_transition_zero": critical_transitions == 0,
        "goal_transition_zero": goal_transitions == 0,
        "background_true_to_false_zero": background_true_to_false == 0,
        "background_repeated_rise_zero": background_repeated_rise == 0,
        "old_i2_hash_unchanged": sha256_file(OLD_I2) == OLD_I2_SHA256,
        "old_j1_hash_unchanged": sha256_file(PRIOR_J1) == json.loads((OUT_ROOT / "k0_input_lock.json").read_text(encoding="utf-8"))["locked_inputs"][str(PRIOR_J1.resolve())]["sha256"],
        "old_j2_hash_unchanged": sha256_file(PRIOR_J2) == json.loads((OUT_ROOT / "k0_input_lock.json").read_text(encoding="utf-8"))["locked_inputs"][str(PRIOR_J2.resolve())]["sha256"],
        "old_j4_hash_unchanged": sha256_file(PRIOR_J4) == json.loads((OUT_ROOT / "k0_input_lock.json").read_text(encoding="utf-8"))["locked_inputs"][str(PRIOR_J4.resolve())]["sha256"],
        "reverse_demo_counts_zero": all(int(old.get("reverse_demo_counts", {}).get(key, 0)) == 0 for key in ("exact_reverse_demo", "pending_autogen", "accepted_reverse_demo")),
    }
    status = "track3_background_convergence_cpu_release_passed" if all(hard_gate.values()) else "track3_background_convergence_cpu_release_failed"
    summary = {
        "schema_version": "track3_background_convergence_k2_v1",
        "status": status,
        "created_at_utc": now(),
        "k0_raw_manifest_sha256": sha256_file(OUT_ROOT / "k0_raw_manifest.json"),
        "k1_contract_sha256": sha256_file(OUT_ROOT / "k1_monotonic_contract.json"),
        "old_exact_vector": old_exact,
        "new_reverse": {"runs": len(reverse_rows), "pass_count": sum(row["status"] == "passed" for row in reverse_rows), "required": "9/9", "release": reverse_pass},
        "new_forward": {"runs": len(forward_rows), "pass_count": sum(row["status"] == "passed" for row in forward_rows), "required": "6/6", "release": forward_pass},
        "transition_counts": {"critical": critical_transitions, "goal": goal_transitions, "background_false_to_true": background_false_to_true, "background_true_to_false": background_true_to_false, "background_repeated_rise": background_repeated_rise},
        "raw_manifest_errors": manifest_errors,
        "hard_gate": hard_gate,
        "resource_boundary": {"cpu_no_renderer_only": True, "simulator_started": False, "gpu3_touched": False, "gpu0_touched": False, "se3": False, "grasp": False, "demo_generation": False, "training_or_norm": False, "fast_or_zip": False, "pending_autogen": 0, "accepted_reverse_demo": 0},
    }
    write_json(OUT_ROOT / "k2_transition_ledger.json", {"schema_version": "track3_background_convergence_k2_transition_ledger_v1", "status": status, "steps": list(RELEASE_STEPS), "rows": ledger_rows})
    write_json(OUT_ROOT / "k2_reverse_release.json", {"schema_version": "track3_background_convergence_k2_reverse_v1", "status": "passed" if reverse_pass else "failed", "old_exact_vector": old_exact, "new_reverse": summary["new_reverse"], "runs": reverse_rows})
    write_json(OUT_ROOT / "k2_forward_controls.json", {"schema_version": "track3_background_convergence_k2_forward_v1", "status": "passed" if forward_pass else "failed", "new_forward": summary["new_forward"], "runs": forward_rows})
    write_json(OUT_ROOT / "k2_summary.json", summary)
    return summary


def worker_exit_code(status: str) -> int:
    return 0 if status in {"passed", "failed_gate"} else 1


def _k3_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "3",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "3",
        "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}",
        "JAX_PLATFORMS": "cpu",
    })
    return environment


def _encode_h264(path: Path, frames: Sequence[Any], *, fps: int = 20) -> dict[str, Any]:
    if not frames:
        return {"status": "failed", "reason": "no_camera_frames"}
    import numpy as np

    arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    first = arrays[0]
    if first.ndim != 3 or first.shape[2] != 3:
        return {"status": "failed", "reason": f"invalid_frame_shape:{first.shape}"}
    if any(frame.shape != first.shape for frame in arrays):
        return {"status": "failed", "reason": "camera_frame_shape_changed"}
    height, width = first.shape[:2]
    payload = b"".join(frame.tobytes() for frame in arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    command = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p", str(partial)]
    completed = subprocess.run(command, input=payload, capture_output=True, check=False)
    if completed.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        return {"status": "failed", "reason": "ffmpeg_failed", "returncode": completed.returncode, "stderr_tail": completed.stderr.decode(errors="replace")[-2000:]}
    os.replace(partial, path)
    return {"status": "passed", "codec": "h264", "fps": fps, "path": str(path.resolve()), "sha256": sha256_file(path), "frame_count": len(arrays), "width": width, "height": height}


def _camera_frame(obs: Mapping[str, Any], key: str) -> Any:
    import numpy as np

    if key not in obs:
        raise RuntimeError(f"camera observation missing: {key}")
    frame = np.asarray(obs[key])
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not np.isfinite(frame).all():
        raise RuntimeError(f"invalid camera observation: {key}:{frame.shape}:{frame.dtype}")
    return frame.copy()


def _k3_direct_sensitivity(env: Any, old: Any, task_index: int, stable_state: Any) -> dict[str, Any]:
    import numpy as np

    source_atom = prior.SOURCE_ATOM[task_index]
    stable = np.asarray(stable_state, dtype=np.float64).copy()
    env.set_state(stable.copy())
    env.sim.forward()
    before = old.evaluate_predicates(env, [source_atom], label="k3_sensitivity_before")
    before_value = before["predicates"][0].get("value") if before.get("predicates") else None
    mutation = old._mutate_for_controlled_destruction(env, task_index, stable.copy())
    destroyed = old.evaluate_predicates(env, [source_atom], label="k3_sensitivity_destroyed")
    destroyed_value = destroyed["predicates"][0].get("value") if destroyed.get("predicates") else None
    env.set_state(stable.copy())
    env.sim.forward()
    restored = old.evaluate_predicates(env, [source_atom], label="k3_sensitivity_restored")
    restored_value = restored["predicates"][0].get("value") if restored.get("predicates") else None
    passed = before_value is True and destroyed_value is False and restored_value is True
    return {"status": "passed" if passed else "failed", "source_atom": source_atom, "before_true": before_value is True, "destroyed_false": destroyed_value is False, "restored_true": restored_value is True, "mutation": mutation, "before": before, "destroyed": destroyed, "restored": restored, "action_used": False, "state_sensitivity_only": True}


def k3_worker(task_index: int, run_index: int, phase: str) -> dict[str, Any]:
    """Run one camera/EGL native reverse stability process."""

    import numpy as np

    started = now()
    env = None
    try:
        from tools.axis_pi05 import track3_public_teleop as recorder

        recorder.configure_public_libero()
        from libero.libero.envs import OffScreenRenderEnv

        old = __import__("tools.axis_pi05.track3_dynamic_gate_reconciliation", fromlist=["module"])
        bddl_path = prior.base.REVERSE_BDDL[task_index]
        initial_atoms = prior.base.public_predicates(bddl_path, "init")
        goal_atoms = prior.base.public_predicates(bddl_path, "goal")
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            control_freq=20,
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=128,
            camera_widths=128,
            render_gpu_device_id=0,
        )
        obs, native_state = old._native_reset(env)
        old.controller_reset_sequence(old._controller(env), env.sim)
        frames_front = [_camera_frame(obs, "agentview_image")]
        frames_wrist = [_camera_frame(obs, "robot0_eye_in_hand_image")]
        samples: list[dict[str, Any]] = [prior._runtime_sample(env, old, task_index, "reverse", 0, initial_atoms, goal_atoms)]
        for step in range(1, 21):
            action = np.zeros(7, dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all() or float(action[-1]) != 0.0:
                raise RuntimeError("invalid 7D zero hold action")
            obs, _reward, _done, _info = env.step(action)
            frames_front.append(_camera_frame(obs, "agentview_image"))
            frames_wrist.append(_camera_frame(obs, "robot0_eye_in_hand_image"))
            samples.append(prior._runtime_sample(env, old, task_index, "reverse", step, initial_atoms, goal_atoms))
        baseline_matrix = next(sample["contact_matrix"] for sample in samples if sample["step"] == 5)
        for sample in samples:
            sample["contacts"] = prior._contact_contract_from_matrix(task_index, sample["contact_matrix"], baseline_matrix=baseline_matrix if sample["step"] >= 5 else None)
        row_base = {
            "schema_version": "track3_background_convergence_k3_worker_v1",
            "status": "passed",
            "process_returncode": 0,
            "worker_returncode_contract": True,
            "task_index": task_index,
            "operator_family": FAMILIES[task_index],
            "kind": "reverse",
            "run_index": run_index,
            "native_reset_state_sha256": old.state_hash(native_state),
            "schema": prior.base._runtime_schema(env, old, bddl_path=bddl_path),
            "action_contract": {"dimension": 7, "values": [0.0] * 7, "gripper": 0.0, "control_freq_hz": 20, "recorded_steps": list(range(21)), "minus_one_used": False},
            "checkpoints": samples,
            "provenance": {"hdf5_state_used_for_predicate_oracle": False, "no_grasp_or_demo_generation": True, "no_pending_autogen": True, "camera_evidence": True},
        }
        physical = _physical_gate(row_base, task_index, "reverse")
        initial_ledger = predicate_ledger(row_base, task_index=task_index, kind="reverse", section="initial", critical_atom=prior.SOURCE_ATOM[task_index], critical_role="operator_critical", strict_expected=True)
        goal_ledger = predicate_ledger(row_base, task_index=task_index, kind="reverse", section="goal", critical_atom=prior.GOAL_ATOM[task_index], critical_role="operator_critical_goal", strict_expected=False)
        stable_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        sensitivity = _k3_direct_sensitivity(env, old, task_index, stable_state)
        video_dir = K3_ROOT / "videos"
        videos = {
            "agentview": _encode_h264(video_dir / f"task{task_index}_reverse_run{run_index}_{phase}_agentview.mp4", frames_front),
            "robot0_eye_in_hand": _encode_h264(video_dir / f"task{task_index}_reverse_run{run_index}_{phase}_wrist.mp4", frames_wrist),
        }
        renderer_schema = row_base["schema"]
        gates = {
            "renderer_schema": renderer_schema.get("use_camera_obs") is True and renderer_schema.get("has_offscreen_renderer") is True and renderer_schema.get("has_renderer") is False,
            "predicate_initial_monotonic": initial_ledger["status"] == "passed",
            "predicate_goal_monotonic": goal_ledger["status"] == "passed",
            "physical_contact_participant_robot": physical["status"] == "passed",
            "direct_state_sensitivity": sensitivity["status"] == "passed",
            "agentview_h264": videos["agentview"].get("status") == "passed",
            "wrist_h264": videos["robot0_eye_in_hand"].get("status") == "passed",
            "camera_frame_count_21": len(frames_front) == 21 and len(frames_wrist) == 21,
        }
        passed = all(gates.values())
        return {**row_base, "status": "passed" if passed else "failed_gate", "created_at_utc": started, "completed_at_utc": now(), "phase": phase, "renderer_schema": renderer_schema, "predicate": {"initial": initial_ledger, "goal": goal_ledger}, "physical": physical, "direct_state_sensitivity": sensitivity, "videos": videos, "hard_gate": gates, "gpu_policy": {"physical_gpu": 3, "cuda_visible_devices": "3", "mujoco_egl_device_id": "3", "mujoco_render_gpu_device_id": 0}, "no_action_for_sensitivity": True}
    except Exception as exc:
        return {"schema_version": "track3_background_convergence_k3_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "phase": phase, "task_index": task_index, "operator_family": FAMILIES.get(task_index), "run_index": run_index, "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}, "gpu_policy": {"physical_gpu": 3, "cuda_visible_devices": "3", "mujoco_egl_device_id": "3"}}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _run_k3_worker(task_index: int, run_index: int, phase: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "tools.axis_pi05.track3_background_predicate_convergence", "--k3-worker", str(task_index), str(run_index), phase]
    completed = subprocess.run(command, cwd=ROOT, env=_k3_environment(), capture_output=True, text=True, timeout=K3_WORKER_TIMEOUT_S, check=False)
    parsed: dict[str, Any] | None = None
    for line in reversed(completed.stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
    if parsed is None:
        parsed = {"schema_version": "track3_background_convergence_k3_worker_wrapper_v1", "status": "worker_output_invalid"}
    parsed["process_returncode"] = int(completed.returncode)
    parsed["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(parsed.get("status")))
    parsed["stdout_tail"] = completed.stdout[-2000:]
    parsed["stderr_tail"] = completed.stderr[-4000:]
    if parsed.get("status") == "passed" and completed.returncode != 0:
        parsed["status"] = "worker_contract_failure"
    return parsed


def run_k3() -> dict[str, Any]:
    k2 = json.loads((OUT_ROOT / "k2_summary.json").read_text(encoding="utf-8"))
    k1 = json.loads((OUT_ROOT / "k1_monotonic_contract.json").read_text(encoding="utf-8"))
    if k2.get("status") != "track3_background_convergence_cpu_release_passed":
        raise RuntimeError("K2 gate is not passed; GPU3 is forbidden")
    if k1.get("status") != "track3_background_convergence_contract_validated":
        raise RuntimeError("K1 contract is not passed; GPU3 is forbidden")
    guard = gpu_read_only_snapshot()
    gpu3_rows = [line for line in str(guard.get("stdout", "")).splitlines() if line.startswith("3,")]
    gpu3_idle = bool(gpu3_rows and int(gpu3_rows[0].split(",")[2].strip()) <= 1024)
    if not gpu3_idle:
        report = {"schema_version": "track3_background_convergence_k3_v1", "status": "track3_background_convergence_gpu3_preflight_blocked", "created_at_utc": now(), "gpu_read_only_snapshot": guard, "hard_stop": True, "runs": [], "resource_boundary": {"gpu3_touched": False, "gpu0_touched": False}}
        write_json(OUT_ROOT / "k3_summary.json", report)
        return report
    runs: list[dict[str, Any]] = []
    canary = _run_k3_worker(19, 1, "task19_canary")
    canary_path = K3_ROOT / "task19_canary.json"
    write_json(canary_path, canary)
    runs.append(canary)
    if canary.get("status") != "passed" or canary.get("process_returncode") != 0 or canary.get("worker_returncode_contract") is not True:
        status = "track3_background_convergence_gpu_failed_task19_canary"
    else:
        status = "track3_background_convergence_gpu_running_family_matrix"
        for task_index in K3_TASK_ORDER:
            for run_index in range(1, 4):
                result = _run_k3_worker(task_index, run_index, "family_matrix")
                report_path = K3_ROOT / f"task{task_index}_run_{run_index}.json"
                write_json(report_path, result)
                runs.append(result)
                if result.get("status") != "passed" or result.get("process_returncode") != 0 or result.get("worker_returncode_contract") is not True:
                    status = f"track3_background_convergence_gpu_failed_task{task_index}_run{run_index}"
                    break
            if status != "track3_background_convergence_gpu_running_family_matrix":
                break
        if status == "track3_background_convergence_gpu_running_family_matrix":
            status = "track3_predicate_convergence_dynamic_validated_pending_se3_review"
    by_family = {str(task): [row for row in runs if int(row.get("task_index", -1)) == task] for task in K3_TASK_ORDER}
    summary = {
        "schema_version": "track3_background_convergence_k3_v1",
        "status": status,
        "created_at_utc": now(),
        "k2_summary_sha256": sha256_file(OUT_ROOT / "k2_summary.json"),
        "k1_contract_sha256": sha256_file(OUT_ROOT / "k1_monotonic_contract.json"),
        "gpu_read_only_snapshot": guard,
        "gpu_policy": {"allowed_physical_gpu": 3, "cuda_visible_devices": "3", "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "mujoco_render_gpu_device_id": 0},
        "task19_canary": {"runs": 1, "passed": canary.get("status") == "passed", "report": str(canary_path.resolve())},
        "family_matrix": {str(task): {"required": "3/3", "runs": len(by_family[str(task)]), "passed": sum(row.get("status") == "passed" for row in by_family[str(task)])} for task in K3_TASK_ORDER},
        "runs": [{"task_index": row.get("task_index"), "operator_family": row.get("operator_family"), "run_index": row.get("run_index"), "phase": row.get("phase"), "status": row.get("status"), "process_returncode": row.get("process_returncode"), "worker_returncode_contract": row.get("worker_returncode_contract")} for row in runs],
        "hard_gate": {"k2_passed": True, "task19_canary_passed": canary.get("status") == "passed", "family_3x3": all(len(by_family[str(task)]) == 3 and all(row.get("status") == "passed" for row in by_family[str(task)]) for task in K3_TASK_ORDER), "worker_exception_zero": all(row.get("status") != "worker_exception" and row.get("worker_returncode_contract") is True for row in runs), "no_retry": True, "no_seed_change": True},
        "resource_boundary": {"gpu3_touched": True, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "se3": False, "grasp": False, "demo_generation": False, "training_or_norm": False, "fast_or_zip": False, "pending_autogen": 0, "accepted_reverse_demo": 0},
    }
    write_json(OUT_ROOT / "k3_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k0", action="store_true")
    parser.add_argument("--k0-verify")
    parser.add_argument("--k1", action="store_true")
    parser.add_argument("--k2", action="store_true")
    parser.add_argument("--k3", action="store_true")
    parser.add_argument("--k3-worker", nargs=3)
    args = parser.parse_args()
    if args.k0:
        result = run_k0()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "track3_background_convergence_input_locked" else 1
    if args.k0_verify:
        result = run_k0_verify(Path(args.k0_verify))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["state"] == "passed" else 1
    if args.k1:
        result = run_k1()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "track3_background_convergence_contract_validated" else 1
    if args.k2:
        result = run_k2()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "track3_background_convergence_cpu_release_passed" else 1
    if args.k3_worker:
        task_string, run_string, phase = args.k3_worker
        if task_string not in {str(task) for task in TASKS} or run_string not in {"1", "2", "3"}:
            parser.error("--k3-worker requires TASK_INDEX 1|2|3 PHASE")
        result = k3_worker(int(task_string), int(run_string), phase)
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return worker_exit_code(str(result.get("status")))
    if args.k3:
        result = run_k3()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "track3_predicate_convergence_dynamic_validated_pending_se3_review" else 1
    parser.error("use --k0")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
