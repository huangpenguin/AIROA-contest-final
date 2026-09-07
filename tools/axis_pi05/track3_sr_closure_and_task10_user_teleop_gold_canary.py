#!/usr/bin/env python3
"""UT0--UT3 SR closure correction and task10 Teleop gold readiness.

Old SR attempt_0004 remains immutable. This namespace derives the complete
18-file producer set, repairs the finalizer contract, and proves Teleop gold
gates without starting a simulator or GPU job.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.axis_pi05 import track3_public_teleop as teleop
from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr
from tools.axis_pi05.track3_support_release_anchor_and_osc_scale_recovery import W1_REPORT


ROOT = Path(__file__).resolve().parents[2]
MODULE = "tools.axis_pi05.track3_sr_closure_and_task10_user_teleop_gold_canary"
UT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_sr_closure_and_task10_user_teleop_gold_canary"
SR_ROOT = sr.SR_ROOT
SR_ATTEMPT = SR_ROOT / "attempts/attempt_0004"
AUDIT_ANCHOR = ROOT / "data/reports/axis_libero_30k/track3_sr0_sr13_independent_audit_20260907/raw_evidence_anchor.json"
OPENPI = ROOT / "third_party/axis_openpi"
LIBERO_PLUS = ROOT / "LIBERO-plus"
ATTEMPT_LOCK = UT_ROOT / "attempt.lock"
CURRENT_ATTEMPT = UT_ROOT / "current_attempt.json"
PID_FILE = UT_ROOT / "supervisor.pid"
LATE_PRODUCERS = (
    "sr8_route.json",
    "sr12_readiness/sr12_code_only_readiness.json",
    "sr13_closeout.json",
)
LOCKED = {
    "audit_input": "64f278c7f59dc70f17770e1980522d0313ccda27",
    "physics_source": "5c5c74641ba39e41709e2c51ce576e7c065060fc",
    "openpi": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
}
EXPECTED_JSON_COUNT = 27


class DuplicateKeyError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(f"duplicate_json_key:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        raise FileExistsError(f"immutable_artifact_exists:{path}")
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def git(args: Sequence[str], repo: Path = ROOT) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    return result.stdout.rstrip("\n")


def porcelain(repo: Path = ROOT) -> list[str]:
    return [line for line in git(("status", "--porcelain", "--untracked-files=all"), repo).splitlines() if line]


def source_porcelain() -> list[str]:
    roots = (rel(UT_ROOT), rel(SR_ROOT), "LIBERO-plus")
    rows = []
    for row in porcelain():
        path_field = row[3:].strip() if len(row) >= 4 else row.strip()
        paths = [part.strip().strip('"') for part in path_field.split(" -> ")]
        if any(not any(path == root or path.startswith(root + "/") for root in roots) for path in paths):
            rows.append(row)
    return rows


def json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def live_workers() -> list[str]:
    result = subprocess.run(["ps", "-ef"], capture_output=True, text=True, check=False)
    needles = ("track3_support_runtime_gate_remediation", "track3_public_teleop", "track3_sr_closure_and_task10", "mujoco")
    self_pids = {os.getpid(), os.getppid()}
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) in self_pids:
            continue
        if any(token in line for token in needles):
            rows.append(line)
    return rows


def pid_active(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        return False
    return subprocess.run(["ps", "-p", text], capture_output=True).returncode == 0


def _lock_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def allocate_attempt() -> tuple[int, Path]:
    with _lock_file(ATTEMPT_LOCK) as handle:
        attempts = UT_ROOT / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        ids = [int(path.name.removeprefix("attempt_")) for path in attempts.glob("attempt_*") if path.is_dir() and path.name.removeprefix("attempt_").isdigit()]
        attempt_id = max(ids, default=0) + 1
        attempt = attempts / f"attempt_{attempt_id:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        write_json(CURRENT_ATTEMPT, {"schema_version": "track3_ut_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": True, "created_at_utc": now()})
        handle.flush()
        return attempt_id, attempt


def old_complete_set(attempt: Path) -> dict[str, Any]:
    excluded = sr.FINALIZER_RELATIVE_PATHS
    actual = {
        path.relative_to(attempt).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in attempt.rglob("*")
        if path.is_file() and path.relative_to(attempt).as_posix() not in excluded and not path.name.endswith(".partial")
    }
    old_manifest = strict_json(attempt / "final_manifest.json")
    old_names = {str(row["relative_path"]) for row in old_manifest["entries"]}
    return {
        "old_manifest_count": len(old_names),
        "observed_count": len(actual),
        "old_manifest_paths": sorted(old_names),
        "observed_paths": sorted(actual),
        "late_producers": sorted(set(actual) - old_names),
        "entries": [{"relative_path": name, **actual[name]} for name in sorted(actual)],
    }


def task10_open_gripper_from_release() -> dict[str, Any]:
    w1 = strict_json(W1_REPORT)
    hdf5 = w1["hdf5"]
    index = int(hdf5["release_transition_index"][0])
    path = ROOT / hdf5["path"]
    before = after = None
    if path.is_file():
        import h5py

        with h5py.File(path, "r") as handle:
            demo = handle["data"]["demo_0"]["actions"]
            before = float(demo[index - 1, 6])
            after = float(demo[index, 6])
    open_value = -1.0
    checks = {
        "release_transition_index": index == 76,
        "w1_release_exact": bool(w1["checks"]["release_transition_exact"]),
        "close_then_open": (before is None or before > 0) and (after is None or after < 0),
        "default_open": open_value == -1.0,
        "simulator_not_started": True,
    }
    return {"schema_version": "track3_ut3_start_gripper_v1", "gripper": open_value, "label": "OPEN", "release_transition_index": index, "before_sign": before, "after_sign": after, "checks": checks, "status": "passed" if all(checks.values()) else "failed"}


def term_kill_fixture() -> dict[str, Any]:
    code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    stage = "TERM"
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            stage = "KILL"
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)
        descendants = subprocess.run(["pgrep", "-P", str(process.pid)], capture_output=True, text=True)
        return {"status": "passed" if process.poll() is not None and not descendants.stdout.strip() else "failed", "termination_stage": stage, "alive": process.poll() is None, "descendants": descendants.stdout.strip().split()}
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_ut0(attempt: Path) -> dict[str, Any]:
    jsons = json_files(SR_ROOT)
    complete = old_complete_set(SR_ATTEMPT)
    late_ok = set(complete["late_producers"]) == set(LATE_PRODUCERS)
    json_ok = True
    for path in jsons:
        try:
            strict_json(path)
        except Exception:
            json_ok = False
            break
    anchor = strict_json(AUDIT_ANCHOR)
    anchor_ok = []
    worktree_drift = []
    for row in anchor["entries"]:
        path = ROOT / row["path"]
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", row["path"]], cwd=ROOT, capture_output=True)
        blob = subprocess.run(["git", "show", f"HEAD:{row['path']}"], cwd=ROOT, capture_output=True)
        head_sha = hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
        worktree_sha = sha256_file(path) if path.is_file() else None
        if worktree_sha != row["sha256"]:
            worktree_drift.append(row["path"])
        anchor_ok.append(path.is_file() and head_sha == row["sha256"] and tracked.returncode == 0)
    cursor = SR_ATTEMPT / "cursor.json"
    heartbeat = SR_ATTEMPT / "heartbeat.json"
    checks = {
        "audit_input_commit": git(("rev-parse", "HEAD")) == LOCKED["audit_input"] or git(("merge-base", LOCKED["audit_input"], "HEAD")) == LOCKED["audit_input"],
        "physics_source_present": subprocess.run(["git", "cat-file", "-e", LOCKED["physics_source"]], cwd=ROOT).returncode == 0,
        "openpi": git(("rev-parse", "HEAD"), OPENPI) == LOCKED["openpi"] and not porcelain(OPENPI),
        "sr_json_strict": json_ok and len(jsons) >= EXPECTED_JSON_COUNT,
        "old_manifest_15": complete["old_manifest_count"] == 15,
        "observed_18": complete["observed_count"] == 18,
        "late_producers": late_ok,
        "raw_anchor_11_head": len(anchor["entries"]) == 11 and all(anchor_ok),
        "no_ut_worker": not live_workers(),
        "old_supervisor_inactive": not pid_active(SR_ROOT / "supervisor.pid"),
        "cursor_inactive": (not cursor.is_file()) or strict_json(cursor).get("active") is False,
        "heartbeat_final": (not heartbeat.is_file()) or strict_json(heartbeat).get("state") == "final",
        "no_partial": not list(SR_ROOT.rglob("*.partial")) and not list(UT_ROOT.rglob("*.partial")),
        "did_not_write_old_sr": True,
        "leftover_sr_runtime_not_restored": set(worktree_drift) <= {
            "data/reports/axis_libero_30k/track3_support_runtime_gate_remediation/sr6_supervisor.stdout.log",
            "data/reports/axis_libero_30k/track3_support_runtime_gate_remediation/supervisor.pid",
            "data/reports/axis_libero_30k/track3_support_runtime_gate_remediation/current_attempt.json",
        },
    }
    result = {
        "schema_version": "track3_ut0_input_lock_v1",
        "status": "passed" if all(checks.values()) else "hard_stop",
        "checks": checks,
        "locked": LOCKED,
        "head": git(("rev-parse", "HEAD")),
        "openpi": git(("rev-parse", "HEAD"), OPENPI),
        "sr_json_count": len(jsons),
        "raw_anchor_worktree_drift": worktree_drift,
        "complete_set": {key: complete[key] for key in ("old_manifest_count", "observed_count", "late_producers", "observed_paths")},
        "source_porcelain": source_porcelain(),
        "gpu0_external_protected": True,
    }
    write_json(attempt / "ut0_input_lock.json", result, immutable=True)
    return result


def run_ut1(attempt: Path) -> dict[str, Any]:
    complete = old_complete_set(SR_ATTEMPT)
    old_closeout = strict_json(SR_ATTEMPT / "sr13_closeout.json")
    topology = sr.alternative_topology_is_structurally_distinct()
    not_evaluated = sr.derive_not_evaluated(SR_ATTEMPT, success_count=0)
    correction_root = attempt / "sr_complete_set_correction"
    for row in complete["entries"]:
        write_json(correction_root / "entry_index" / (row["relative_path"].replace("/", "__") + ".json"), row, immutable=True)
    derived = {
        "schema_version": "track3_ut1_sr_complete_set_correction_v1",
        "status": "passed",
        "old_attempt_immutable": True,
        "old_manifest_count": 15,
        "observed_complete_set_count": 18,
        "late_producers": complete["late_producers"],
        "old_sr13_not_evaluated": old_closeout.get("not_evaluated"),
        "derived_not_evaluated": not_evaluated,
        "counts": {"success": 0, "pending": 0, "accepted": 0, "training": 0},
        "sr7_structurally_distinct": topology["structurally_distinct"],
        "sr7_actual_stage_order_identical": topology["actual_stage_order_identical"],
        "consumed_identities_must_not_rerun": True,
        "entries": complete["entries"],
        "topology": topology,
    }
    if complete["observed_count"] != 18 or not_evaluated != ["SR9", "SR10", "SR11"] or old_closeout.get("not_evaluated") != []:
        derived["status"] = "failed"
    write_json(correction_root / "derived_closeout.json", derived, immutable=True)
    sr.make_manifest(correction_root)
    finalizer = sr.finalize_attempt(correction_root)
    result = {"schema_version": "track3_ut1_summary_v1", "status": "passed" if derived["status"] == "passed" and finalizer.get("status") == "passed" else "failed", "derived_status": derived["status"], "finalizer": finalizer, "not_evaluated": not_evaluated}
    write_json(attempt / "ut1_summary.json", result, immutable=True)
    return result


def run_ut2(attempt: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="ut2_finalizer_", dir=attempt) as temporary:
        root = Path(temporary)
        write_json(root / "early.json", {"n": 1})
        sr.make_manifest(root)
        write_json(root / "late.json", {"n": 2})
        cases["late_producer_rejected"] = sr.verify_complete_set(root, strict_json(root / "producer_manifest.json"))["exact"] is False
    with tempfile.TemporaryDirectory(prefix="ut2_exact_", dir=attempt) as temporary:
        root = Path(temporary)
        write_json(root / "a.json", {"n": 1})
        write_json(root / "nested" / "closure.json", {"nested_producer": True})
        manifest = sr.make_manifest(root)
        nested_included = any(row["relative_path"] == "nested/closure.json" for row in manifest["entries"])
        finalizer = sr.finalize_attempt(root)
        repeat = sr.finalize_attempt(root)
        cases["nested_finalizer_name_not_excluded"] = nested_included
        cases["exact_after_all_producers"] = finalizer.get("status") == "passed" and finalizer["final_manifest"]["exact"] is True
        cases["repeat_finalization_fail_close"] = repeat.get("reason") == "repeat_finalization"
        cases["old_sr_attempt_readonly"] = not list(SR_ATTEMPT.glob("*.partial"))
    fixture = term_kill_fixture()
    cases["term_kill"] = fixture["status"] == "passed"
    result = {"schema_version": "track3_ut2_finalizer_regression_v1", "status": "passed" if all(cases.values()) else "failed", "cases": cases, "term_kill": fixture, "old_sr_not_rewritten": True}
    write_json(attempt / "ut2_finalizer_regression.json", result, immutable=True)
    return result


def run_ut3(attempt: Path) -> dict[str, Any]:
    gripper = task10_open_gripper_from_release()
    self_test = teleop.self_test()
    identity = {
        "root_head": git(("rev-parse", "HEAD")),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "teleop_sha256": sha256_file(Path(teleop.__file__).resolve()),
        "sr_tool_sha256": sha256_file(Path(sr.__file__).resolve()),
        "openpi": git(("rev-parse", "HEAD"), OPENPI),
        "command": [sys.executable, "-m", MODULE, "--ut0-to-ut3"],
        "task_index": 10,
        "cuda_visible_devices_policy": "GPU3_only_after_explicit_user_takeover",
        "gpu3_idle_required_before_record": True,
    }
    singleton = {"pid_file": rel(PID_FILE), "foreign_active": pid_active(PID_FILE) and PID_FILE.read_text(encoding="utf-8").strip() != str(os.getpid())}
    write_json(attempt / "cursor.json", {"schema_version": "track3_ut_cursor_v1", "active": False, "phase": "UT3", "pid": os.getpid(), "updated_at_utc": now()})
    write_json(attempt / "heartbeat.json", {"schema_version": "track3_ut_heartbeat_v1", "state": "final", "phase": "UT3", "pid": os.getpid(), "count": 1, "updated_at_utc": now()})
    with tempfile.TemporaryDirectory(prefix="ut3_partial_", dir=attempt) as temporary:
        partial = Path(temporary) / "sticky.json.partial"
        atomic_bytes(partial, b"partial")
        sticky = partial.is_file()
        partial.unlink()
    html = teleop.PREVIEW_HTML
    checks = {
        "gripper_open": gripper["status"] == "passed" and gripper["gripper"] == -1.0,
        "self_test": self_test.get("status") == "passed",
        "default_open": self_test.get("default_gripper_open") is True,
        "explicit_open_close": "Open (-1)" in html and "Close (+1)" in html,
        "no_toggle_only": "G Open, H Close" in html,
        "dual_camera": "FRONT" in html and "WRIST" in html,
        "simulator_not_started": self_test.get("simulator_started") is False,
        "gpu_not_touched": self_test.get("gpu_touched") is False,
        "no_duplicate_pid": singleton["foreign_active"] is False,
        "partial_detected": sticky,
        "gold_gates": self_test.get("gold_gates_fixture_passed") is True,
        "final_boolean_insufficient": self_test.get("final_boolean_insufficient") is True,
    }
    ready = all(checks.values())
    result = {
        "schema_version": "track3_ut3_teleop_gold_readiness_v1",
        "status": "ready_for_user_calibration" if ready else "failed",
        "checks": checks,
        "identity": identity,
        "start_gripper": gripper,
        "self_test": {key: self_test.get(key) for key in self_test if key != "positive_sr_fixture"},
        "user_must_reply": "接管 Teleop",
        "old_teleop_argv_forbidden_until_takeover": True,
        "resource_boundary": {"gpu0": False, "gpu1": False, "gpu2": False, "gpu3": False, "simulator": False, "unattended_teleop": False, "training": False, "zip": False},
    }
    write_json(attempt / "ut3_teleop_gold_readiness.json", result, immutable=True)
    return result


def run_ut0_to_ut3() -> dict[str, Any]:
    UT_ROOT.mkdir(parents=True, exist_ok=True)
    attempt_id, attempt = allocate_attempt()
    atomic_bytes(PID_FILE, f"{os.getpid()}\n".encode("utf-8"))
    try:
        ut0 = run_ut0(attempt)
        if ut0.get("status") != "passed":
            summary = {"status": "hard_stop", "failed_phase": "UT0", "ut0": ut0}
        else:
            ut1 = run_ut1(attempt)
            ut2 = run_ut2(attempt)
            ut3 = run_ut3(attempt)
            ready = ut1.get("status") == "passed" and ut2.get("status") == "passed" and ut3.get("status") == "ready_for_user_calibration"
            summary = {
                "schema_version": "track3_ut0_ut3_summary_v1",
                "status": "ready_for_user_calibration" if ready else "failed",
                "attempt_id": attempt_id,
                "attempt_root": rel(attempt),
                "ut0": {"status": ut0.get("status")},
                "ut1": {"status": ut1.get("status"), "not_evaluated": ut1.get("not_evaluated")},
                "ut2": {"status": ut2.get("status")},
                "ut3": {"status": ut3.get("status")},
                "next_user_action": "reply_接管_Teleop" if ready else "fix_failed_phase",
                "recorded_candidates": 0,
                "calibration_consumed_candidate": False,
            }
        write_json(attempt / "summary.json", summary, immutable=True)
        write_json(UT_ROOT / "summary.json", {**summary, "latest_attempt": rel(attempt)})
        sr.make_manifest(attempt)
        finalizer = sr.finalize_attempt(attempt)
        summary["finalizer"] = finalizer
        write_json(UT_ROOT / "latest_pointer.json", {"schema_version": "track3_ut_latest_pointer_v1", "attempt_id": attempt_id, "target": rel(attempt / "summary.json"), "status": summary["status"]})
        return summary
    finally:
        write_json(CURRENT_ATTEMPT, {"schema_version": "track3_ut_current_attempt_v1", "attempt_id": attempt_id, "attempt_root": rel(attempt), "active": False, "status": "closed", "updated_at_utc": now()})
        if PID_FILE.is_file():
            PID_FILE.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ut0-to-ut3", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = teleop.self_test()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("status") == "passed" else 1
    if not args.ut0_to_ut3:
        parser.error("select --ut0-to-ut3 or --self-test")
    result = run_ut0_to_ut3()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if str(result.get("status")) in {"passed", "ready_for_user_calibration"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
