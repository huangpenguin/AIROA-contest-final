"""Fast400-norm paired behavior evaluation adapter for R6 and H300.

This adapter is intentionally a new evaluation identity.  It does not mutate
or monkey-patch the historical Q5/N5 runners and it never writes into their
evidence or rollout namespaces.  The worker protocol is:

* ``preflight`` validates the frozen inputs, checkpoint identity, and runtime
  GPU boundary without writing evidence;
* ``worker`` evaluates one role (``r6``, ``step300`` or optional ``step100``)
  into its own cursor/heartbeat/case/video namespace and can resume completed
  cases without re-running them;
* ``merge`` pairs R6 and H300 step300 rows and writes a fail-closed release
  decision; it never folds or creates a ZIP;
* ``status`` is read-only and reports cursor/heartbeat/PID liveness;
* ``renderer-smoke`` is a no-policy/no-action fresh-process EGL probe for the
  parent runtime contract only;
* ``policy-server-smoke`` checks policy-server readiness/close without a
  simulator environment, reset, action, or behavior case.

The actual E1 workers are deliberately not started by this module's import or
tests.  The release gate is the matched Fast400-norm panel: reverse-dev 4 and
C20 20, with reverse improvement of at least one and C20 loss no greater than
two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import ROOT, canonical_json, git_info, sha256_file


PLAN = ROOT / "docs/AXIS_PI05_T3_FAST400_H300_POSTTRAIN_EVAL_RELEASE_HANDOFF_20260915.md"
EVALUATION_FREEZE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/evaluation_freeze.json"
REVERSE_MANIFEST = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"
C20_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
NORM_STATS = ROOT / "data/manifests_track3/fast400_gold_80_20_20260913/norm_stats.json"
R6_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
H300_RUN_ROOT = ROOT / "outputs/axis_pi05/track3_fast400_h300/t3_fast400_r6_h300_4gpu_r1"
H300_CHECKPOINT = H300_RUN_ROOT / "300"
STEP100_CHECKPOINT = H300_RUN_ROOT / "100"
H300_SUMMARY = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_4gpu_r1_local4/summary.json"
ADAPTER_PATH = ROOT / "tools/axis_pi05/run_track3_fast400_h300_paired_eval.py"
FAILED_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915"
FAILED_OUTPUT_ROOT = ROOT / "outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915"
EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
REMEDIATION_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_e1_r6_free_port_remediation_20260915"
DL0_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_dl0_signal_norm_attestation_20260915"
DL0_RETRY_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_dl0_signal_norm_attestation_20260915_attempt_0002"
OPENPI_ROOT = ROOT / "third_party/axis_openpi"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"
SERVER_MODULE = "tools.axis_pi05.checkpoint_http_server"
MODULE = "tools.axis_pi05.run_track3_fast400_h300_paired_eval"
CONFIG_NAME = "axis_pi05_track3_q5"

CONTROL_HZ = 20
CHUNK_SIZE = 10
EXECUTE_ACTIONS = 8
MAX_STEPS = 300
RUN_ID = "t3_fast400_r6_h300_4gpu_r1"
EXPECTED_EVALUATION_FREEZE_SHA = "2c8d2eca6b5138ddc59d9d00837c95e774f4a2e14687cd656662ee6e9dd77b77"
EXPECTED_REVERSE_SHA = "64a1124ef17a866e259ba7e8bce96351cd183dff1ebf76c1afbd5694dfe0ec95"
EXPECTED_C20_SHA = "1dad183301e255e4287348470d962b9016420c2365f0ac8a8c2eef7ed0052231"
EXPECTED_NORM_SHA = "7e4e24dacb24d6333d32cb982d301761c2f00e8f5daca418109bea226268f333"
Q0_REPO_RELATIVE_SHA256_V1 = "q0_repo_relative_sha256_v1"
PARAMS_RELATIVE_NUL_SHA256_V1 = "params_relative_nul_sha256_v1"
EXPECTED_R6_PARAMS_Q0_SHA = "d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f"
EXPECTED_R6_PARAMS_NATIVE_SHA = "e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc"
EXPECTED_R6_PARAMS_FILE_COUNT = 16
EXPECTED_R6_PARAMS_BYTES = 12505927866
EXPECTED_H300_SUMMARY_SHA = "d9f81e6b1fde66a26e4d659647674785c0815d2cfd489f745bfc9f0df2912bed"

HEARTBEAT_INTERVAL_SEC = 30.0
HEARTBEAT_STALE_AFTER_SEC = HEARTBEAT_INTERVAL_SEC * 3.0
PARENT_EGL_EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_e1_parent_egl_remediation_20260915"
RENDERER_SMOKE_CASE_ID = "sha256:13ec6c60961387b083a78896858d70e98bacc645467ceb0e68a9d9f12bf9a7e7"
RENDERER_SMOKE_HOLD_SEC = 2.0
PARENT_RENDERER_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_GL",
    "MUJOCO_EGL_DEVICE_ID",
    "AXIS_PI05_EVAL_GPU",
    "LIBERO_CONTROL_FREQ",
    "LIBERO_CONFIG_PATH",
    "AXIS_PI05_LIBERO_ASSET_BRIDGE",
)
POLICY_NORM_ENV_KEYS = (
    "AXIS_PI05_NORM_STATS",
    "AXIS_PI05_TRACK3_NORM_STATS",
    "AXIS_PI05_NORM_STATS_SHA256",
)

ROLE_SPECS: dict[str, dict[str, Any]] = {
    "r6": {
        "role": "r6_params_only_baseline",
        "checkpoint": R6_PARAMS,
        "checkpoint_step": 500,
        "training_steps": 500,
        "diagnostic_only": False,
        "require_train_state": False,
    },
    "step300": {
        "role": "h300_step300_candidate",
        "checkpoint": H300_CHECKPOINT,
        "checkpoint_step": 300,
        "training_steps": 300,
        "diagnostic_only": False,
        "require_train_state": True,
    },
    "step100": {
        "role": "h300_step100_diagnostic",
        "checkpoint": STEP100_CHECKPOINT,
        "checkpoint_step": 100,
        "training_steps": 100,
        "diagnostic_only": True,
        "require_train_state": True,
    },
}

HISTORICAL_NAMESPACES = (
    ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval",
    ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/n5_official_t3_step100_paired_eval",
    ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q7_gpu3_frozen_eval",
    ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/n5_official_t3_step100_paired_eval",
)

IMMUTABLE_FAILED_NAMESPACES = (FAILED_EVIDENCE_ROOT, FAILED_OUTPUT_ROOT)


def adapter_source_identity() -> dict[str, str]:
    """Return the exact adapter path and bytes identity used by this process."""

    return {"path": str(ADAPTER_PATH.resolve()), "sha256": sha256_file(ADAPTER_PATH)}


def assert_adapter_source_identity(expected: dict[str, str]) -> dict[str, str]:
    observed = adapter_source_identity()
    if expected != observed:
        raise ValueError(f"adapter source identity mismatch: {observed} != {expected}")
    return observed


def _assert_locked_adapter_source(lock: dict[str, Any], current: dict[str, str]) -> None:
    if lock.get("adapter_source") != current:
        raise ValueError("adapter source identity mismatch in worker lock")


def strict_json_load(path: Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys at every depth."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return value


def strict_jsonl_load(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key!r} at {path}:{line_number}")
                result[key] = value
            return result

        value = json.loads(line, object_pairs_hook=reject_duplicates)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, canonical_json(value))


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sorted_files(path: Path) -> list[Path]:
    return sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    ) if path.exists() else []


def params_relative_nul_sha256_v1(path: Path) -> str:
    """Hash ``params/`` files using params-relative NUL-delimited records."""

    digest = hashlib.sha256()
    for item in _sorted_files(path):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def q0_repo_relative_sha256_v1(path: Path, *, repo_root: Path = ROOT) -> str:
    """Reproduce the frozen Q0 ``sha256sum`` text stream from repo-relative paths."""

    path = path.resolve()
    repo_root = repo_root.resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Q0 path must be inside repo root: {path}") from exc
    digest = hashlib.sha256()
    for item in sorted(_sorted_files(path), key=lambda candidate: candidate.relative_to(repo_root).as_posix()):
        digest.update(sha256_file(item).encode())
        digest.update(b"  ")
        digest.update(item.relative_to(repo_root).as_posix().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Backward-compatible alias for the explicitly versioned native params hash."""

    return params_relative_nul_sha256_v1(path)


def namespace_fingerprint(path: Path) -> dict[str, Any]:
    """Hash a small evidence namespace without including any aggregate file."""

    entries = [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in _sorted_files(path)
    ] if path.exists() else []
    return {"file_count": len(entries), "sha256": hashlib.sha256(canonical_json(entries)).hexdigest(), "files": entries}


def _params_identity(path: Path) -> dict[str, Any]:
    """Hash one params tree once while recording both frozen hash domains."""

    path = path.resolve()
    repo_root = ROOT.resolve()
    try:
        repo_relative_root = path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"params path must be inside repo root: {path}") from exc
    native_digest = hashlib.sha256()
    q0_digest = hashlib.sha256()
    files = _sorted_files(path)
    total_bytes = 0
    for item in files:
        relative_params = item.relative_to(path).as_posix()
        relative_repo = f"{repo_relative_root}/{relative_params}"
        file_sha = sha256_file(item)
        native_digest.update(relative_params.encode())
        native_digest.update(b"\0")
        native_digest.update(file_sha.encode())
        native_digest.update(b"\n")
        q0_digest.update(file_sha.encode())
        q0_digest.update(b"  ")
        q0_digest.update(relative_repo.encode())
        q0_digest.update(b"\n")
        total_bytes += item.stat().st_size
    return {
        Q0_REPO_RELATIVE_SHA256_V1: q0_digest.hexdigest(),
        PARAMS_RELATIVE_NUL_SHA256_V1: native_digest.hexdigest(),
        "file_count": len(files),
        "bytes": total_bytes,
    }


def _assert_frozen_params_identity(role: str, actual: dict[str, Any]) -> None:
    if role != "r6":
        return
    expected = {
        Q0_REPO_RELATIVE_SHA256_V1: EXPECTED_R6_PARAMS_Q0_SHA,
        PARAMS_RELATIVE_NUL_SHA256_V1: EXPECTED_R6_PARAMS_NATIVE_SHA,
        "file_count": EXPECTED_R6_PARAMS_FILE_COUNT,
        "bytes": EXPECTED_R6_PARAMS_BYTES,
    }
    for field, expected_value in expected.items():
        actual_value = actual.get(field)
        if actual_value != expected_value:
            raise ValueError(f"R6 params {field} mismatch: {actual_value!r} != {expected_value!r}")


class _HeartbeatUpdater:
    """Atomically refresh a worker heartbeat while a case can run for minutes."""

    def __init__(self, path: Path, initial: dict[str, Any], *, interval_sec: float = HEARTBEAT_INTERVAL_SEC):
        self.path = path
        self.base = dict(initial)
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = int(self.base.get("sequence", 0))
        self.error: BaseException | None = None

    def _write(self, status: str, **updates: Any) -> None:
        self._sequence += 1
        payload = {
            **self.base,
            **updates,
            "status": status,
            "pid": os.getpid(),
            "updated_at_unix": time.time(),
            "sequence": self._sequence,
        }
        atomic_json(self.path, payload)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_sec):
                self._write("running")
        except BaseException as exc:  # surfaced by the owning worker at a case boundary
            self.error = exc
            self._stop.set()

    def start(self) -> None:
        self._write("running")
        self._thread = threading.Thread(target=self._run, name="track3-fast400-heartbeat", daemon=True)
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"heartbeat updater failed: {self.error}") from self.error

    def stop(self, *, terminal_phase: str) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec + 1.0))
        self._write("stopped", terminal_phase=terminal_phase)


def _safe_case_id(case_id: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in case_id)


def _assert_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid, used, total, utilization = [part.strip() for part in line.split(",")]
        rows.append({
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_gpu_percent": int(utilization),
        })
    return rows


def compute_app_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, used_memory = [part.strip() for part in line.split(",", 3)]
        rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": process_name, "used_memory_mib": int(used_memory)})
    return rows


def _process_rows() -> list[tuple[int, str]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
    rows: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2:
            rows.append((int(fields[0]), fields[1]))
    return rows


def active_conflicting_processes(*, role: str | None = None, gpu: int | None = None) -> list[dict[str, Any]]:
    """Return live processes that would collide with this adapter invocation."""

    current_pid = os.getpid()
    ancestor_pids: set[int] = set()
    ancestor = os.getppid()
    while ancestor > 1 and ancestor not in ancestor_pids:
        ancestor_pids.add(ancestor)
        try:
            ancestor = int(subprocess.check_output(["ps", "-o", "ppid=", "-p", str(ancestor)], text=True).strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            break
    conflicts: list[dict[str, Any]] = []
    for pid, command in _process_rows():
        if pid == current_pid or pid in ancestor_pids:
            continue
        old_runner = any(
            needle in command
            for needle in (
                "run_track3_q5_step10_frozen_eval",
                "run_track3_n5_official_paired_eval",
                "track3_four_gold_mix_v1/q7_gpu3_frozen_eval",
                "track3_four_gold_mix_v1/n5_official_t3_step100_paired_eval",
            )
        )
        new_runner = "run_track3_fast400_h300_paired_eval" in command
        train = "scripts/train.py" in command or "run_track3_fast400_h300_local4" in command
        if old_runner or train:
            conflicts.append({"pid": pid, "args": command, "reason": "historical_eval_or_training"})
            continue
        if new_runner:
            same_role = role is not None and f"--role {role}" in command
            same_gpu = gpu is not None and f"--gpu {gpu}" in command
            if same_role or same_gpu:
                conflicts.append({"pid": pid, "args": command, "reason": "same_adapter_role_or_gpu"})
    return conflicts


def _assert_new_namespace(path: Path) -> None:
    resolved = path.resolve()
    for historical in HISTORICAL_NAMESPACES:
        if resolved == historical.resolve() or historical.resolve() in resolved.parents:
            raise ValueError(f"refusing historical Q7/N5 namespace: {path}")
    if any(token in resolved.as_posix() for token in ("q7_gpu3_frozen_eval", "n5_official_t3_step100_paired_eval")):
        raise ValueError(f"refusing historical Q7/N5 namespace: {path}")
    for immutable in IMMUTABLE_FAILED_NAMESPACES:
        immutable_resolved = immutable.resolve()
        if resolved == immutable_resolved or immutable_resolved in resolved.parents:
            raise ValueError(f"refusing immutable failed E1 attempt namespace: {path}")


def _assert_status_namespace(path: Path) -> None:
    """Allow read-only status inspection of the immutable failed attempt."""

    resolved = path.resolve()
    if any(token in resolved.as_posix() for token in ("q7_gpu3_frozen_eval", "n5_official_t3_step100_paired_eval")):
        raise ValueError(f"refusing historical Q7/N5 namespace: {path}")


def _assert_no_partial(path: Path) -> None:
    partials = [item for item in path.rglob("*") if ".partial" in item.name] if path.exists() else []
    if partials:
        raise ValueError(f"partial residue under {path}: {partials[:3]}")


def _normalise_case(row: dict[str, Any], *, panel: str, panel_manifest_sha: str, benchmark: str, suite: str, task_id: int) -> dict[str, Any]:
    result = dict(row)
    result.update({
        "panel": panel,
        "benchmark": benchmark,
        "suite": suite,
        "task_id": int(task_id),
        "perturbation_id": "clean",
        "perturbation_family": "clean",
        "perturbation_params": {},
        "selection_eligible": False,
        "panel_manifest_sha256": panel_manifest_sha,
        "max_steps": int(row.get("max_steps", MAX_STEPS)),
    })
    result["case_fingerprint"] = hashlib.sha256(
        canonical_json({key: result[key] for key in sorted(result) if key != "case_fingerprint"})
    ).hexdigest()
    return result


def _unique_manifest_rows(rows: Iterable[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"manifest row has invalid case_id in {path}: {row}")
        if case_id in indexed:
            raise ValueError(f"duplicate case_id {case_id!r} in {path}")
        indexed[case_id] = row
    return indexed


def load_frozen_inputs(*, adapter_source: dict[str, str] | None = None) -> dict[str, Any]:
    adapter_source = adapter_source or adapter_source_identity()
    required = (PLAN, EVALUATION_FREEZE, REVERSE_MANIFEST, C20_MANIFEST, NORM_STATS, R6_PARAMS, H300_CHECKPOINT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Fast400 paired-eval input(s): " + ", ".join(missing))
    _assert_hash(EVALUATION_FREEZE, EXPECTED_EVALUATION_FREEZE_SHA)
    _assert_hash(REVERSE_MANIFEST, EXPECTED_REVERSE_SHA)
    _assert_hash(C20_MANIFEST, EXPECTED_C20_SHA)
    _assert_hash(NORM_STATS, EXPECTED_NORM_SHA)
    freeze = strict_json_load(EVALUATION_FREEZE)
    if freeze.get("status") != "evaluation_set_frozen_before_q5":
        raise ValueError(f"evaluation freeze is not locked: {freeze.get('status')}")
    reverse_source = _unique_manifest_rows(strict_jsonl_load(REVERSE_MANIFEST), REVERSE_MANIFEST)
    c20_source = _unique_manifest_rows(strict_jsonl_load(C20_MANIFEST), C20_MANIFEST)
    reverse_cases: list[dict[str, Any]] = []
    for frozen in freeze.get("reverse_dev", {}).get("rows", []):
        case_id = frozen["case_id"]
        source = reverse_source.get(case_id)
        if source is None:
            raise ValueError(f"reverse freeze case missing from source manifest: {case_id}")
        for key in ("env_seed", "policy_seed", "init_state_index", "init_state_sha256", "bddl_sha256"):
            if source.get(key) != frozen.get(key):
                raise ValueError(f"reverse freeze mismatch for {case_id}: {key}")
        reverse_cases.append(_normalise_case(source, panel="reverse_dev", panel_manifest_sha=EXPECTED_REVERSE_SHA, benchmark="libero_t3", suite="libero_t3", task_id=int(frozen["task_index"])))
    if len(reverse_cases) != 4:
        raise ValueError(f"reverse freeze must contain 4 cases, got {len(reverse_cases)}")
    c20_cases: list[dict[str, Any]] = []
    for frozen in freeze.get("c20_retention", {}).get("cases", []):
        case_id = frozen["case_id"]
        source = c20_source.get(case_id)
        if source is None:
            raise ValueError(f"C20 freeze case missing from source manifest: {case_id}")
        for key in ("env_seed", "policy_seed", "init_state_index", "init_state_sha256"):
            if source.get(key) != frozen.get(key):
                raise ValueError(f"C20 freeze mismatch for {case_id}: {key}")
        c20_cases.append(_normalise_case(source, panel="C20", panel_manifest_sha=EXPECTED_C20_SHA, benchmark="libero_mix", suite="C20", task_id=int(source["official_task_index"])))
    if len(c20_cases) != 20:
        raise ValueError(f"C20 freeze must contain 20 cases, got {len(c20_cases)}")
    all_cases = reverse_cases + c20_cases
    case_ids = [case["case_id"] for case in all_cases]
    if len(set(case_ids)) != 24:
        raise ValueError("frozen reverse and C20 panels must have 24 unique case IDs")
    return {
        "freeze": freeze,
        "reverse_cases": reverse_cases,
        "c20_cases": c20_cases,
        "all_cases": all_cases,
        "input_hashes": {
            "adapter_sha256": adapter_source["sha256"],
            "plan_sha256": sha256_file(PLAN),
            "evaluation_freeze_sha256": EXPECTED_EVALUATION_FREEZE_SHA,
            "reverse_manifest_sha256": EXPECTED_REVERSE_SHA,
            "c20_manifest_sha256": EXPECTED_C20_SHA,
            "norm_stats_sha256": EXPECTED_NORM_SHA,
        },
    }


def _normalise_checkpoint_path(path: Path) -> Path:
    path = path.resolve()
    return path.parent if path.name == "params" else path


def validate_checkpoint(role: str, path: Path | None = None) -> dict[str, Any]:
    if role not in ROLE_SPECS:
        raise ValueError(f"unknown role: {role}")
    spec = ROLE_SPECS[role]
    checkpoint = _normalise_checkpoint_path(path or spec["checkpoint"])
    expected = _normalise_checkpoint_path(spec["checkpoint"])
    if role != "r6" and checkpoint != expected:
        raise ValueError(f"{role} checkpoint must be the frozen H300 run path: {checkpoint}")
    if role == "r6" and checkpoint != expected:
        raise ValueError(f"r6 baseline must be the frozen params-only checkpoint: {checkpoint}")
    metadata = checkpoint / "_CHECKPOINT_METADATA"
    params = checkpoint / "params"
    if not metadata.is_file() or not params.is_dir():
        raise FileNotFoundError(f"incomplete {role} checkpoint: {checkpoint}")
    if spec["require_train_state"] and not (checkpoint / "train_state").is_dir():
        raise FileNotFoundError(f"{role} checkpoint is missing train_state: {checkpoint}")
    _assert_no_partial(checkpoint)
    params_identity = _params_identity(params)
    _assert_frozen_params_identity(role, params_identity)
    identity = {
        "role": spec["role"],
        "checkpoint_step": int(spec["checkpoint_step"]),
        "training_steps": int(spec["training_steps"]),
        "path": str(checkpoint),
        "params_path": str(params),
        "metadata_sha256": sha256_file(metadata),
        "params_tree_sha256": params_identity[PARAMS_RELATIVE_NUL_SHA256_V1],
        "params_tree_sha256_algorithm": PARAMS_RELATIVE_NUL_SHA256_V1,
        "params_hashes": {
            Q0_REPO_RELATIVE_SHA256_V1: params_identity[Q0_REPO_RELATIVE_SHA256_V1],
            PARAMS_RELATIVE_NUL_SHA256_V1: params_identity[PARAMS_RELATIVE_NUL_SHA256_V1],
        },
        "params_file_count": params_identity["file_count"],
        "params_bytes": params_identity["bytes"],
        "train_state_present": (checkpoint / "train_state").is_dir(),
        "diagnostic_only": bool(spec["diagnostic_only"]),
        "official_result": False,
    }
    if role in {"step300", "step100"}:
        if not checkpoint.is_relative_to(H300_RUN_ROOT.resolve()):
            raise ValueError(f"{role} checkpoint escaped frozen H300 run root: {checkpoint}")
        identity["run_id"] = RUN_ID
    return identity


def validate_h300_training_summary() -> dict[str, Any]:
    _assert_hash(H300_SUMMARY, EXPECTED_H300_SUMMARY_SHA)
    summary = strict_json_load(H300_SUMMARY)
    if summary.get("run_id") != RUN_ID or summary.get("status") != "training_passed_ready_for_behavior_eval":
        raise ValueError(f"H300 training summary is not closed for behavior eval: {summary.get('status')}")
    if summary.get("behavior_eval") != "not_evaluated":
        raise ValueError(f"H300 behavior eval is already classified unexpectedly: {summary.get('behavior_eval')}")
    if summary.get("input_hashes", {}).get("norm") != EXPECTED_NORM_SHA:
        raise ValueError("H300 summary norm hash does not match Fast400 norm")
    if summary.get("metrics", {}).get("metrics_sha256") != "f36a4cfacb5eba7112dd0106b74be78a0d3fca5d795e9b608b85b2c7385f648a":
        raise ValueError("H300 summary metrics hash mismatch")
    if summary.get("fresh_restore", {}).get("restored_step") != 300:
        raise ValueError("H300 summary does not prove fresh step300 restore")
    if summary.get("metrics", {}).get("last_step") != 300:
        raise ValueError("H300 summary does not prove 300 optimizer steps")
    return {"summary_sha256": EXPECTED_H300_SUMMARY_SHA, "run_id": RUN_ID, "status": summary["status"]}


def _validate_gpu_id(gpu: int) -> str:
    if gpu not in {0, 1, 2, 3}:
        raise ValueError(f"GPU must be one of 0,1,2,3, got {gpu}")
    return str(gpu)


def _python_path() -> str:
    python_path = [str(ROOT), str(ROOT / "LIBERO-plus"), str(OPENPI_ROOT / "src")]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(python_path)


def _renderer_environment_updates(gpu: int) -> dict[str, str]:
    gpu_id = _validate_gpu_id(gpu)
    config_path = ROOT / ".eval_libero"
    if not (config_path / "config.yaml").is_file():
        raise FileNotFoundError(f"frozen LIBERO config is missing: {config_path / 'config.yaml'}")
    return {
        "PYTHONPATH": _python_path(),
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": gpu_id,
        "AXIS_PI05_EVAL_GPU": gpu_id,
        "LIBERO_CONTROL_FREQ": str(CONTROL_HZ),
        "LIBERO_CONFIG_PATH": str(config_path),
        "AXIS_PI05_LIBERO_ASSET_BRIDGE": "1",
        "HF_HUB_OFFLINE": "1",
    }


def parent_environment(gpu: int) -> dict[str, str]:
    """Return the complete environment for a simulator-owning parent process."""

    env = os.environ.copy()
    env.update(_renderer_environment_updates(gpu))
    return env


def assert_parent_renderer_environment(gpu: int, environment: Any | None = None) -> dict[str, str]:
    """Fail closed unless all parent-side renderer keys target one physical GPU."""

    gpu_id = _validate_gpu_id(gpu)
    observed = os.environ if environment is None else environment
    expected = {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": gpu_id,
        "AXIS_PI05_EVAL_GPU": gpu_id,
        "LIBERO_CONTROL_FREQ": str(CONTROL_HZ),
        "LIBERO_CONFIG_PATH": str(ROOT / ".eval_libero"),
        "AXIS_PI05_LIBERO_ASSET_BRIDGE": "1",
    }
    mismatches = {key: {"expected": value, "observed": observed.get(key)} for key, value in expected.items() if observed.get(key) != value}
    if mismatches:
        raise RuntimeError(f"parent renderer runtime binding mismatch for GPU {gpu}: {mismatches}")
    return {key: str(observed[key]) for key in PARENT_RENDERER_ENV_KEYS}


def bind_parent_renderer_environment(gpu: int) -> dict[str, str]:
    """Apply and verify the parent-side EGL contract before simulator import."""

    updates = _renderer_environment_updates(gpu)
    os.environ.update({key: updates[key] for key in PARENT_RENDERER_ENV_KEYS})
    os.environ.update({"PYTHONPATH": updates["PYTHONPATH"], "HF_HUB_OFFLINE": updates["HF_HUB_OFFLINE"]})
    return assert_parent_renderer_environment(gpu)


def assert_parent_child_gpu_identity(parent_gpu: int, child_environment: Any) -> None:
    """Reject a policy child whose physical GPU identity differs from its parent."""

    gpu_id = _validate_gpu_id(parent_gpu)
    expected = {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "MUJOCO_EGL_DEVICE_ID": gpu_id,
        "AXIS_PI05_EVAL_GPU": gpu_id,
    }
    mismatches = {key: {"expected": value, "observed": child_environment.get(key)} for key, value in expected.items() if child_environment.get(key) != value}
    if mismatches:
        raise RuntimeError(f"parent/child GPU identity mismatch: {mismatches}")


def server_environment(gpu: int) -> dict[str, str]:
    """Return the parent contract plus policy-server-specific runtime settings."""

    env = parent_environment(gpu)
    _validate_gpu_id(gpu)
    env.update({
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
        "AXIS_PI05_NORM_STATS": str(NORM_STATS),
        "AXIS_PI05_TRACK3_NORM_STATS": str(NORM_STATS),
        "AXIS_PI05_NORM_STATS_SHA256": EXPECTED_NORM_SHA,
    })
    return env


def _runtime_preflight(role: str, gpu: int) -> dict[str, Any]:
    if active_conflicting_processes(role=role, gpu=gpu):
        raise RuntimeError(f"conflicting evaluation/training process is active: {active_conflicting_processes(role=role, gpu=gpu)}")
    rows = gpu_snapshot()
    selected = next((row for row in rows if row["index"] == gpu), None)
    if selected is None:
        raise RuntimeError(f"GPU {gpu} is not visible: {rows}")
    apps = [row for row in compute_app_snapshot() if row["gpu_uuid"] == selected["uuid"]]
    if selected["memory_used_mib"] > 64 or selected["utilization_gpu_percent"] != 0 or apps:
        raise RuntimeError(f"GPU {gpu} is not idle: {selected}; apps={apps}")
    return {"gpu": selected, "compute_apps": apps}


def preflight(*, role: str, gpu: int, evidence_dir: Path = EVIDENCE_ROOT, output_dir: Path = OUTPUT_ROOT, adapter_source: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate one role without creating evidence or starting a server."""

    if role not in ROLE_SPECS:
        raise ValueError(f"unknown role: {role}")
    adapter_source = adapter_source or adapter_source_identity()
    _assert_new_namespace(evidence_dir)
    _assert_new_namespace(output_dir)
    inputs = load_frozen_inputs(adapter_source=adapter_source)
    summary = validate_h300_training_summary()
    identity = validate_checkpoint(role)
    runtime = _runtime_preflight(role, gpu)
    parent_runtime = parent_environment(gpu)
    return {
        "schema_version": "track3_fast400_h300_paired_eval_preflight_v1",
        "status": "passed",
        "adapter_source": adapter_source,
        "role": role,
        "identity": identity,
        "h300_training_summary": summary,
        "case_counts": {"reverse_dev": 4, "c20": 20, "paired": 24},
        "protocol": {"control_frequency_hz": CONTROL_HZ, "chunk_size": CHUNK_SIZE, "executed_actions_per_chunk": EXECUTE_ACTIONS, "max_steps": MAX_STEPS, "norm": "QUANTILES"},
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "input_hashes": inputs["input_hashes"],
        "gpu": runtime["gpu"],
        "parent_renderer_runtime": {key: parent_runtime[key] for key in PARENT_RENDERER_ENV_KEYS},
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
        "evidence_dir": str(evidence_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
    }


class Execute8Policy:
    """Make the HTTP action queue obey chunk=10 / execute-first-8."""

    def __init__(self, client: Any):
        self.client = client
        self.instruction = ""
        self.needs_chunk_reset = False
        self.actions_seen = 0
        self.chunk_index = 0

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.instruction = str(instruction)
        self.needs_chunk_reset = False
        self.actions_seen = 0
        self.chunk_index = 0
        self.client.reset(instruction=self.instruction, seed=seed, reset_kind="episode_reset", chunk_index=0)

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if self.needs_chunk_reset:
            self.chunk_index += 1
            self.client.reset(instruction=self.instruction, reset_kind="chunk_discard", chunk_index=self.chunk_index)
            self.needs_chunk_reset = False
            self.actions_seen = 0
        action = np.asarray(self.client.get_action(observation), dtype=np.float32)
        if action.shape != (7,):
            raise ValueError(f"policy action shape must be (7,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("policy action contains NaN or Inf")
        self.actions_seen += 1
        if self.actions_seen >= EXECUTE_ACTIONS:
            self.needs_chunk_reset = True
        return action


class PolicyServer:
    """Per-role policy server with an explicitly frozen Fast400 norm."""

    def __init__(self, checkpoint: Path, output_dir: Path, gpu: int, python: Path = PYTHON):
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.gpu = gpu
        self.python = python
        self.port = _free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.health_attestation: dict[str, Any] | None = None

    def _verify_health(self) -> dict[str, Any]:
        """Read the final runtime identity before any reset/action."""

        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=10) as response:  # noqa: S310 - loopback child only
            if response.status != 200:
                raise RuntimeError(f"policy server health returned HTTP {response.status}")
            health = _strict_json_text(response.read().decode("utf-8"), "policy server health")
        expected = {
            "status": "ok",
            "active_norm_path": str(NORM_STATS.resolve()),
            "active_norm_sha256": EXPECTED_NORM_SHA,
            "checkpoint_path": str(self.checkpoint.resolve()),
            "config_name": CONFIG_NAME,
        }
        mismatches = {
            key: {"expected": value, "observed": health.get(key)}
            for key, value in expected.items()
            if health.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"policy server health attestation mismatch: {mismatches}")
        return health

    def start(self) -> Any:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.output_dir / "policy_server.log").open("w", encoding="utf-8")
        assert_parent_renderer_environment(self.gpu)
        child_environment = server_environment(self.gpu)
        assert_parent_child_gpu_identity(self.gpu, child_environment)
        self.process = subprocess.Popen(
            [str(self.python), "-m", SERVER_MODULE, "--checkpoint", str(self.checkpoint), "--port", str(self.port), "--config", CONFIG_NAME],
            cwd=ROOT,
            env=child_environment,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        from pipeline.remote_policy import RemotePolicyClient

        client = RemotePolicyClient(f"http://127.0.0.1:{self.port}", timeout_sec=90.0)
        try:
            client.wait_for_server(timeout=900.0)
            self.health_attestation = self._verify_health()
        except Exception:
            self.close()
            raise
        return client

    def close(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            return {"pid": None, "returncode": None, "clean_shutdown": True, "pid_exited": True, "termination_requested": False, "termination_signal": None, "kill_required": False, "port": self.port}
        termination_requested = False
        termination_signal = None
        kill_required = False
        if process.poll() is None:
            termination_requested = True
            termination_signal = "SIGTERM"
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                kill_required = True
                process.kill()
                process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        return {"pid": process.pid, "returncode": process.returncode, "clean_shutdown": process.poll() is not None, "pid_exited": process.poll() is not None, "termination_requested": termination_requested, "termination_signal": termination_signal, "kill_required": kill_required, "port": self.port}


def controlled_server_shutdown(shutdown: dict[str, Any], *, orphan_free: bool) -> bool:
    """Accept normal exit or only the fully evidenced parent-requested TERM."""

    if not (shutdown.get("clean_shutdown") and shutdown.get("pid_exited") and orphan_free):
        return False
    if shutdown.get("returncode") == 0:
        return True
    return (
        shutdown.get("returncode") == -signal.SIGTERM
        and shutdown.get("termination_requested") is True
        and shutdown.get("termination_signal") == "SIGTERM"
        and shutdown.get("kill_required") is False
    )


def _task_info_for_case(case: dict[str, Any], manager: Any, cache: dict[str, Any]) -> Any:
    from pipeline.environment import TaskInfo

    if case["benchmark"] == "libero_mix":
        import torch

        init_states_path = Path(case["init_states_path"])
        init_states = torch.load(init_states_path, weights_only=False)
        state = np.asarray(init_states[int(case["init_state_index"])])
        from .phase1_rollout import _state_hash

        if _state_hash(state) != case["init_state_sha256"]:
            raise ValueError(f"C20 init state changed: {case['case_id']}")
        if sha256_file(Path(case["bddl_path"])) != case["bddl_sha256"]:
            raise ValueError(f"C20 BDDL changed: {case['case_id']}")
        return TaskInfo(task_id=int(case["task_id"]), name=str(case["task_name"]), language=str(case["language"]), bddl_file=str(case["bddl_path"]), init_states=init_states, benchmark_name="libero_mix")
    benchmark = str(case["benchmark"])
    if benchmark not in cache:
        cache[benchmark] = {task.task_id: task for task in manager.get_task_infos(benchmark)}
    task_info = cache[benchmark].get(int(case["task_id"]))
    if task_info is None or str(task_info.name) != str(case["task_name"]):
        raise ValueError(f"Track3 task registry mismatch: {case['case_id']}")
    if sha256_file(Path(task_info.bddl_file)) != case["bddl_sha256"]:
        raise ValueError(f"Track3 BDDL changed: {case['case_id']}")
    from .phase1_rollout import _state_hash

    init_state = manager.get_init_state(task_info, int(case["init_state_index"]))
    if _state_hash(init_state) != case["init_state_sha256"]:
        raise ValueError(f"Track3 init state changed: {case['case_id']}")
    return task_info


def _current_render_observation(env: Any, fallback: Any) -> Any:
    """Read one rendered observation after direct state restore without stepping."""

    getter = getattr(getattr(env, "env", None), "_get_observations", None)
    return getter() if callable(getter) else fallback


def _observation_schema(observation: Any) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    if not isinstance(observation, dict):
        raise TypeError(f"renderer observation must be a dict, got {type(observation).__name__}")
    for key, value in sorted(observation.items()):
        if not isinstance(value, np.ndarray):
            continue
        array = np.asarray(value)
        try:
            finite = bool(np.all(np.isfinite(array)))
        except TypeError as exc:
            raise TypeError(f"renderer observation is not numeric: {key}") from exc
        arrays[str(key)] = {"shape": list(array.shape), "dtype": str(array.dtype), "finite": finite}
    required = ("agentview_image", "robot0_eye_in_hand_image")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise RuntimeError(f"renderer observation is missing camera keys: {missing}")
    if not all(item["finite"] for item in arrays.values()):
        raise FloatingPointError("renderer observation contains NaN or Inf")
    return {"arrays": arrays, "required_camera_keys": list(required), "finite": True}


def _gpu_runtime_snapshot() -> dict[str, Any]:
    return {"captured_at_unix": time.time(), "gpus": gpu_snapshot(), "compute_apps": compute_app_snapshot()}


def _strict_json_text(text: str, source: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {source}")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON top level must be an object: {source}")
    return value


def renderer_smoke_child(*, gpu: int, case_id: str = RENDERER_SMOKE_CASE_ID, hold_sec: float = RENDERER_SMOKE_HOLD_SEC) -> dict[str, Any]:
    """Create, restore and render one frozen case in a fresh parent process."""

    runtime = bind_parent_renderer_environment(gpu)
    inputs = load_frozen_inputs()
    cases = {str(case["case_id"]): case for case in inputs["reverse_cases"]}
    if case_id not in cases:
        raise ValueError(f"renderer smoke case is not in the frozen reverse panel: {case_id}")
    case = cases[case_id]
    for import_path in (ROOT, ROOT / "LIBERO-plus", OPENPI_ROOT / "src"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    from .libero_asset_bridge import activate as activate_libero_asset_bridge
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    bridge = activate_libero_asset_bridge(ROOT)
    manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=MAX_STEPS, seed=1000))
    task_info = _task_info_for_case(case, manager, {})
    env = None
    opened = False
    closed = False
    try:
        env = manager.create_env(task_info, env_seed=int(case["env_seed"]))
        opened = True
        reset_observation = env.reset()
        init_state = manager.get_init_state(task_info, int(case["init_state_index"]))
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()
        observation = _current_render_observation(env, reset_observation)
        schema = _observation_schema(observation)
        time.sleep(max(0.0, float(hold_sec)))
        return {
            "schema_version": "track3_fast400_h300_parent_renderer_smoke_child_v1",
            "status": "passed",
            "fresh_process": True,
            "pid": os.getpid(),
            "gpu": int(gpu),
            "case_id": case_id,
            "case_fingerprint": case["case_fingerprint"],
            "init_state_sha256": case["init_state_sha256"],
            "runtime_environment": runtime,
            "bridge": bridge,
            "renderer": schema,
            "environment_created": opened,
            "environment_closed": True,
            "action_calls": 0,
            "physics_step_calls": 0,
            "policy_server_started": False,
            "behavior_episode_started": False,
        }
    finally:
        if env is not None:
            env.close()
            closed = True
        if not closed and opened:
            raise RuntimeError("renderer smoke environment did not close cleanly")


def _pid_gpu_binding(pid: int, snapshots: list[dict[str, Any]], selected_gpu: int) -> dict[str, Any]:
    selected_uuid = None
    for snapshot in snapshots:
        for row in snapshot.get("gpus", []):
            if int(row["index"]) == int(selected_gpu):
                selected_uuid = row["uuid"]
                break
        if selected_uuid is not None:
            break
    selected_seen = False
    nonselected_seen = False
    for snapshot in snapshots:
        gpu_by_uuid = {row["uuid"]: int(row["index"]) for row in snapshot.get("gpus", [])}
        for app in snapshot.get("compute_apps", []):
            if int(app.get("pid", -1)) != int(pid):
                continue
            index = gpu_by_uuid.get(app.get("gpu_uuid"))
            if index == int(selected_gpu):
                selected_seen = True
            elif index is not None:
                nonselected_seen = True
    return {
        "selected_gpu_uuid": selected_uuid,
        "pid_seen_on_selected_gpu": selected_seen,
        "pid_seen_on_nonselected_gpu": nonselected_seen,
    }


def renderer_smoke(*, gpu: int, case_id: str = RENDERER_SMOKE_CASE_ID, evidence_dir: Path = PARENT_EGL_EVIDENCE_ROOT, python: Path = PYTHON, hold_sec: float = RENDERER_SMOKE_HOLD_SEC) -> dict[str, Any]:
    """Run one fresh-process parent EGL smoke and persist only remediation evidence."""

    if gpu not in {0, 1}:
        raise ValueError(f"E1P0 smoke is authorized only on GPU0/GPU1, got {gpu}")
    _assert_new_namespace(evidence_dir)
    if EVIDENCE_ROOT.exists() or OUTPUT_ROOT.exists():
        raise RuntimeError("E1 behavior evidence/output root must remain absent during E1P0 smoke")
    inputs = load_frozen_inputs()
    if case_id not in {str(case["case_id"]) for case in inputs["reverse_cases"]}:
        raise ValueError(f"renderer smoke case is not in the frozen reverse panel: {case_id}")
    environment = parent_environment(gpu)
    command = [str(python), "-m", MODULE, "_renderer-smoke-child", "--gpu", str(gpu), "--case-id", case_id, "--hold-sec", str(float(hold_sec))]
    before = _gpu_runtime_snapshot()
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    during: list[dict[str, Any]] = []
    while process.poll() is None:
        during.append(_gpu_runtime_snapshot())
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    after = _gpu_runtime_snapshot()
    child = None
    child_error = None
    try:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        child = _strict_json_text(lines[-1], f"renderer-smoke-child GPU{gpu} stdout") if lines else None
    except Exception as exc:
        child_error = f"{type(exc).__name__}: {exc}"
    pid_binding = _pid_gpu_binding(process.pid, [before, *during, after], gpu)
    selected_before = next(row for row in before["gpus"] if row["index"] == gpu)
    selected_during = [row for snapshot in during for row in snapshot["gpus"] if row["index"] == gpu]
    max_selected_memory = max([row["memory_used_mib"] for row in selected_during] or [selected_before["memory_used_mib"]])
    renderer_resource_seen = bool(pid_binding["pid_seen_on_selected_gpu"] or max_selected_memory > selected_before["memory_used_mib"])
    result = {
        "schema_version": "track3_fast400_h300_parent_renderer_smoke_v1",
        "status": "passed" if (
            process.returncode == 0
            and child_error is None
            and child is not None
            and child.get("status") == "passed"
            and child.get("environment_closed") is True
            and child.get("policy_server_started") is False
            and child.get("action_calls") == 0
            and child.get("physics_step_calls") == 0
            and renderer_resource_seen
            and not pid_binding["pid_seen_on_nonselected_gpu"]
        ) else "failed",
        "fresh_process": True,
        "gpu": gpu,
        "case_id": case_id,
        "pid": process.pid,
        "command": command,
        "runtime_environment": {key: environment.get(key) for key in PARENT_RENDERER_ENV_KEYS},
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "process_returncode": process.returncode,
        "child": child,
        "child_stdout_tail": stdout[-4000:],
        "child_stderr_tail": stderr[-4000:],
        "child_parse_error": child_error,
        "gpu_snapshots": {"before": before, "during": during, "after": after},
        "gpu_binding": {**pid_binding, "renderer_resource_seen": renderer_resource_seen, "max_selected_memory_mib": max_selected_memory, "selected_memory_before_mib": selected_before["memory_used_mib"]},
        "no_policy_server": bool(child and child.get("policy_server_started") is False),
        "no_action": bool(child and child.get("action_calls") == 0 and child.get("physics_step_calls") == 0),
        "e1_behavior_roots_absent": not EVIDENCE_ROOT.exists() and not OUTPUT_ROOT.exists(),
        "orphan_free_after_close": process.poll() is not None,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(evidence_dir / f"renderer_smoke_gpu{gpu}.json", result)
    return result


def renderer_smoke_pair(*, evidence_dir: Path = PARENT_EGL_EVIDENCE_ROOT, python: Path = PYTHON, hold_sec: float = RENDERER_SMOKE_HOLD_SEC) -> dict[str, Any]:
    """Run GPU0 then GPU1, never concurrently, and write the E1P0 summary."""

    _assert_new_namespace(evidence_dir)
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise FileExistsError(f"E1P0 evidence namespace is not empty: {evidence_dir}")
    results = [renderer_smoke(gpu=gpu, evidence_dir=evidence_dir, python=python, hold_sec=hold_sec) for gpu in (0, 1)]
    summary = {
        "schema_version": "track3_fast400_h300_e1_parent_egl_remediation_v1",
        "status": "passed" if all(result["status"] == "passed" for result in results) else "failed",
        "next_phase_only": "E1_worker_launch_review" if all(result["status"] == "passed" for result in results) else "E1P0_failure_review",
        "hard_stop_before_behavior_workers": True,
        "serial_gpu_order": [0, 1],
        "smokes": results,
        "e1_behavior_roots_absent": not EVIDENCE_ROOT.exists() and not OUTPUT_ROOT.exists(),
        "behavior_workers_started": False,
        "policy_servers_started": False,
        "training_started": False,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }
    atomic_json(evidence_dir / "summary.json", summary)
    return summary


def policy_server_smoke_child(*, gpu: int, output_dir: Path, python: Path = PYTHON) -> dict[str, Any]:
    """Start and close one policy server without creating an environment or acting."""

    if gpu != 0:
        raise ValueError(f"free-port remediation smoke is authorized only on GPU0, got {gpu}")
    runtime = bind_parent_renderer_environment(gpu)
    adapter_source = adapter_source_identity()
    identity = validate_checkpoint("r6")
    _assert_new_namespace(output_dir)
    output_dir = output_dir.resolve()
    server = PolicyServer(Path(identity["path"]), output_dir, gpu, python)
    ready = False
    start_error = None
    shutdown: dict[str, Any] = {"pid": None, "returncode": None, "clean_shutdown": False, "pid_exited": False, "termination_requested": False, "termination_signal": None, "kill_required": False, "port": server.port}
    try:
        server.start()
        ready = True
    except BaseException as exc:
        start_error = f"{type(exc).__name__}: {exc}"
    finally:
        shutdown = server.close()
    log_path = output_dir / "policy_server.log"
    orphan_free = not _pid_alive(shutdown.get("pid"))
    controlled_shutdown = controlled_server_shutdown(shutdown, orphan_free=orphan_free)
    return {
        "schema_version": "track3_fast400_h300_e1_r6_free_port_policy_server_smoke_child_v1",
        "status": "passed" if ready and controlled_shutdown and server.health_attestation is not None and start_error is None else "failed",
        "fresh_process": True,
        "pid": os.getpid(),
        "gpu": gpu,
        "adapter_source": adapter_source,
        "identity": identity,
        "runtime_environment": {key: server_environment(gpu)[key] for key in (*PARENT_RENDERER_ENV_KEYS, *POLICY_NORM_ENV_KEYS)},
        "policy_server_ready": ready,
        "policy_server_started": ready,
        "policy_server_pid": shutdown.get("pid"),
        "port": shutdown.get("port"),
        "server_shutdown": shutdown,
        "controlled_shutdown": controlled_shutdown,
        "server_health_attestation": server.health_attestation,
        "orphan_free_after_close": orphan_free,
        "server_returncode": shutdown.get("returncode"),
        "server_clean_shutdown": bool(shutdown.get("clean_shutdown")),
        "server_log_path": str(log_path.resolve()),
        "server_log_sha256": sha256_file(log_path) if log_path.is_file() else None,
        "server_start_error": start_error,
        "simulator_environment_created": False,
        "action_calls": 0,
        "physics_step_calls": 0,
        "behavior_case_started": False,
    }


def policy_server_smoke(*, gpu: int = 0, evidence_dir: Path = DL0_RETRY_EVIDENCE_ROOT, python: Path = PYTHON) -> dict[str, Any]:
    """Run one fresh GPU0 policy-server readiness/close smoke for remediation only."""

    if gpu != 0:
        raise ValueError(f"free-port remediation smoke is authorized only on GPU0, got {gpu}")
    _assert_new_namespace(evidence_dir)
    if evidence_dir.exists() and any(evidence_dir.iterdir()):
        raise FileExistsError(f"remediation evidence namespace is not empty: {evidence_dir}")
    if EVIDENCE_ROOT.exists() or OUTPUT_ROOT.exists():
        raise RuntimeError("fresh behavior attempt_0002 roots must remain absent during remediation smoke")
    old_evidence_before = namespace_fingerprint(FAILED_EVIDENCE_ROOT)
    old_output_before = namespace_fingerprint(FAILED_OUTPUT_ROOT)
    output_dir = evidence_dir.resolve() / "policy_server_smoke"
    environment = parent_environment(gpu)
    command = [
        str(python), "-m", MODULE, "_policy-server-smoke-child",
        "--gpu", str(gpu), "--output-dir", str(output_dir), "--python", str(python),
    ]
    before = _gpu_runtime_snapshot()
    started = time.time()
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    during: list[dict[str, Any]] = []
    while process.poll() is None:
        during.append(_gpu_runtime_snapshot())
        time.sleep(0.25)
    stdout, stderr = process.communicate()
    after = _gpu_runtime_snapshot()
    child = None
    child_error = None
    try:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        child = _strict_json_text(lines[-1], "policy-server-smoke-child stdout") if lines else None
    except Exception as exc:
        child_error = f"{type(exc).__name__}: {exc}"
    server_pid = int(child["policy_server_pid"]) if child and child.get("policy_server_pid") is not None else -1
    pid_binding = _pid_gpu_binding(server_pid, [before, *during, after], gpu) if server_pid > 0 else {
        "selected_gpu_uuid": None,
        "pid_seen_on_selected_gpu": False,
        "pid_seen_on_nonselected_gpu": False,
    }
    orphan_free = not _pid_alive(server_pid)
    controlled_shutdown = controlled_server_shutdown(child.get("server_shutdown", {}) if child else {}, orphan_free=orphan_free)
    selected_before = next(row for row in before["gpus"] if row["index"] == gpu)
    selected_during = [row for snapshot in during for row in snapshot["gpus"] if row["index"] == gpu]
    max_selected_memory = max([row["memory_used_mib"] for row in selected_during] or [selected_before["memory_used_mib"]])
    policy_resource_seen = bool(pid_binding["pid_seen_on_selected_gpu"] or max_selected_memory > selected_before["memory_used_mib"])
    old_evidence_after = namespace_fingerprint(FAILED_EVIDENCE_ROOT)
    old_output_after = namespace_fingerprint(FAILED_OUTPUT_ROOT)
    behavior_roots_absent = not EVIDENCE_ROOT.exists() and not OUTPUT_ROOT.exists()
    server_log = Path(child["server_log_path"]) if child and child.get("server_log_path") else output_dir / "policy_server.log"
    result = {
        "schema_version": "track3_fast400_h300_e1_r6_free_port_policy_server_smoke_v1",
        "status": "passed" if (
            process.returncode == 0
            and child_error is None
            and child is not None
            and child.get("status") == "passed"
            and child.get("policy_server_ready") is True
            and child.get("policy_server_started") is True
            and child.get("action_calls") == 0
            and child.get("physics_step_calls") == 0
            and child.get("simulator_environment_created") is False
            and child.get("behavior_case_started") is False
            and controlled_shutdown
            and child.get("server_health_attestation") is not None
            and policy_resource_seen
            and not pid_binding["pid_seen_on_nonselected_gpu"]
            and orphan_free
            and behavior_roots_absent
            and old_evidence_before == old_evidence_after
            and old_output_before == old_output_after
        ) else "failed",
        "fresh_process": True,
        "gpu": gpu,
        "adapter_source": adapter_source_identity(),
        "command": command,
        "runtime_environment": {key: environment.get(key) for key in PARENT_RENDERER_ENV_KEYS},
        "policy_norm_environment": {key: server_environment(gpu).get(key) for key in POLICY_NORM_ENV_KEYS},
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "process_pid": process.pid,
        "process_returncode": process.returncode,
        "child": child,
        "child_stdout_tail": stdout[-4000:],
        "child_stderr_tail": stderr[-4000:],
        "child_parse_error": child_error,
        "gpu_snapshots": {"before": before, "during": during, "after": after},
        "gpu_binding": {**pid_binding, "policy_resource_seen": policy_resource_seen, "max_selected_memory_mib": max_selected_memory, "selected_memory_before_mib": selected_before["memory_used_mib"]},
        "policy_server_log_path": str(server_log.resolve()),
        "policy_server_log_sha256": sha256_file(server_log) if server_log.is_file() else None,
        "old_failed_evidence_before": old_evidence_before,
        "old_failed_evidence_after": old_evidence_after,
        "old_failed_output_before": old_output_before,
        "old_failed_output_after": old_output_after,
        "failed_attempt_immutable": old_evidence_before == old_evidence_after and old_output_before == old_output_after,
        "behavior_attempt_0002_roots_absent": behavior_roots_absent,
        "controlled_shutdown": controlled_shutdown,
        "server_health_attestation": child.get("server_health_attestation") if child else None,
        "orphan_free_after_close": orphan_free,
    }
    evidence_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(evidence_dir / "policy_server_smoke.json", result)
    return result


def _classify_exception(error: Exception) -> str:
    if isinstance(error, FloatingPointError):
        return "policy_nonfinite"
    text = str(error).lower()
    if any(token in text for token in ("server", "connection", "bddl", "init_state", "mujoco", "environment", "asset", "scene", "texture", "file not found")):
        return "infrastructure"
    return "model_error"


def _write_trace(path: Path, actions: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite action trace: {path}")
    _atomic_npy(path, actions)


def _episode_row(case: dict[str, Any], role: str, identity: dict[str, Any], episode: Any | None, output_dir: Path, *, error: Exception | None = None) -> dict[str, Any]:
    actions = np.asarray(getattr(episode, "actions", []), dtype=np.float32).reshape((-1, 7)) if error is None and episode is not None else np.empty((0, 7), dtype=np.float32)
    trace_path = output_dir / "action_traces" / f"{_safe_case_id(str(case['case_id']))}.npy"
    _write_trace(trace_path, actions)
    norms = np.linalg.norm(actions, axis=1) if actions.size else np.empty((0,), dtype=np.float32)
    finite = bool(np.all(np.isfinite(actions)))
    if error is not None:
        failure_class = _classify_exception(error)
        payload = {
            "success": False,
            "collision": False,
            "goal_reached": False,
            "timeout": False,
            "steps": 0,
            "elapsed_sec": 0.0,
            "failure_reason": "exception",
            "failure_class": failure_class,
            "error_summary": f"{type(error).__name__}: {error}",
            "video_path": None,
        }
    else:
        payload = {
            "success": bool(episode.success),
            "collision": bool(episode.collided),
            "goal_reached": bool(episode.goal_reached),
            "goal_reached_but_collision": bool(episode.goal_reached and episode.collided),
            "timeout": bool(episode.timeout),
            "steps": int(episode.total_steps),
            "elapsed_sec": float(episode.elapsed_time_sec),
            "failure_reason": episode.failure_reason,
            "failure_class": None if episode.success else "episode_failure",
            "error_summary": None,
            "video_path": getattr(episode, "video_path", None),
        }
    return {
        **case,
        **payload,
        "status": "completed",
        "role": role,
        "checkpoint_step": identity["checkpoint_step"],
        "training_steps": identity["training_steps"],
        "checkpoint": identity["path"],
        "checkpoint_params_tree_sha256": identity["params_tree_sha256"],
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "evaluation_freeze_sha256": EXPECTED_EVALUATION_FREEZE_SHA,
        "control_frequency_hz": CONTROL_HZ,
        "chunk_size": CHUNK_SIZE,
        "executed_actions_per_chunk": EXECUTE_ACTIONS,
        "max_steps": MAX_STEPS,
        "action_shape": list(actions.shape),
        "actions_finite": finite,
        "action_norm_mean": float(norms.mean()) if norms.size else 0.0,
        "action_norm_max": float(norms.max()) if norms.size else 0.0,
        "action_trace_path": str(trace_path.resolve()),
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }


def _validate_trace(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("action_trace_path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing action trace for {row.get('case_id')}: {path}")
    try:
        actions = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"action trace is not no-pickle readable: {path}") from exc
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"invalid action trace shape for {row.get('case_id')}: {actions.shape}")
    finite = bool(np.all(np.isfinite(actions)))
    if not finite:
        raise FloatingPointError(f"nonfinite action trace for {row.get('case_id')}")
    return {"finite": finite, "nonzero": bool(actions.size and np.any(np.abs(actions) > 1e-8)), "shape": list(actions.shape)}


def _case_file(role_root: Path, case_id: str) -> Path:
    return role_root / "cases" / f"{_safe_case_id(case_id)}.json"


def _load_completed(path: Path, case: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = strict_json_load(path)
    if value.get("status") != "completed":
        raise ValueError(f"case file is not complete; refusing implicit retry: {path}")
    for key in ("case_id", "case_fingerprint", "role", "norm_stats_sha256", "checkpoint_params_tree_sha256"):
        expected = {
            "case_id": case["case_id"],
            "case_fingerprint": case["case_fingerprint"],
            "role": identity["role"],
            "norm_stats_sha256": EXPECTED_NORM_SHA,
            "checkpoint_params_tree_sha256": identity["params_tree_sha256"],
        }[key]
        if value.get(key) != expected:
            raise ValueError(f"completed case input mismatch at {path}: {key}")
    _validate_trace(value)
    return value


def _run_cases(cases: list[dict[str, Any]], role: str, identity: dict[str, Any], manager: Any, task_cache: dict[str, Any], role_root: Path, output_root: Path, cursor: dict[str, Any], gpu: int, python: Path, heartbeat_updater: _HeartbeatUpdater | None = None) -> list[dict[str, Any]]:
    from pipeline.config import PerturbationConfig
    from pipeline.rollout import RolloutExecutor

    role_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["T3_VIDEO_DIR"] = str(output_root / "videos")
    os.environ["T3_VIDEO_FPS"] = str(CONTROL_HZ)
    executor = RolloutExecutor(manager, manager.config)
    result_rows: list[dict[str, Any]] = []
    server = PolicyServer(Path(identity["path"]), output_root / "server", gpu, python)
    client = server.start()
    policy = Execute8Policy(client)
    try:
        for index, case in enumerate(cases):
            if heartbeat_updater is not None:
                heartbeat_updater.raise_if_failed()
            case_path = _case_file(role_root, str(case["case_id"]))
            partial_path = case_path.with_suffix(case_path.suffix + ".partial")
            if partial_path.exists():
                raise RuntimeError(f"active/partial case residue requires manual inspection: {partial_path}")
            completed = _load_completed(case_path, case, identity)
            if completed is not None:
                result_rows.append(completed)
                cursor["completed_case_ids"] = sorted(set(cursor.get("completed_case_ids", [])) | {case["case_id"]})
                cursor.update({"case_index": index, "case_id": case["case_id"], "updated_at_unix": time.time()})
                atomic_json(role_root / "cursor.json", cursor)
                continue
            atomic_json(partial_path, {"status": "running", "case_id": case["case_id"], "case_fingerprint": case["case_fingerprint"], "role": identity["role"], "started_at_unix": time.time()})
            cursor.update({"status": "running", "case_index": index, "case_id": case["case_id"], "updated_at_unix": time.time()})
            atomic_json(role_root / "cursor.json", cursor)
            try:
                task_info = _task_info_for_case(case, manager, task_cache)
                if case["benchmark"] == "libero_mix":
                    import torch

                    init_state = np.asarray(torch.load(Path(case["init_states_path"]), weights_only=False)[int(case["init_state_index"])])
                else:
                    init_state = manager.get_init_state(task_info, int(case["init_state_index"]))
                episode = executor.evaluate_case(policy, task_info, init_state=init_state, env_seed=int(case["env_seed"]), policy_seed=int(case["policy_seed"]), max_steps=MAX_STEPS, case_id=str(case["case_id"]), perturbation=PerturbationConfig())
                row = _episode_row(case, identity["role"], identity, episode, output_root)
            except Exception as exc:
                row = _episode_row(case, identity["role"], identity, None, output_root, error=exc)
            atomic_json(case_path, row)
            partial_path.unlink(missing_ok=True)
            _validate_trace(row)
            result_rows.append(row)
            cursor["completed_case_ids"] = sorted(set(cursor.get("completed_case_ids", [])) | {case["case_id"]})
            cursor.update({"case_index": index, "case_id": case["case_id"], "updated_at_unix": time.time()})
            atomic_json(role_root / "cursor.json", cursor)
            print(f"{role} {index + 1}/{len(cases)} {case['case_id']}", flush=True)
            if heartbeat_updater is not None:
                heartbeat_updater.raise_if_failed()
    finally:
        atomic_json(role_root / "server_shutdown.json", server.close())
        os.environ.pop("T3_VIDEO_DIR", None)
    return result_rows


def _role_root(evidence_dir: Path, role: str) -> Path:
    return evidence_dir / role


def _refresh_runtime_state(cursor: dict[str, Any], heartbeat: dict[str, Any], *, role: str, gpu: int, resume: bool, pid: int, now: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return atomically publishable runtime state bound to this worker process."""

    timestamp = time.time() if now is None else float(now)
    previous_pid = cursor.get("pid")
    cursor = {
        **cursor,
        "status": "running",
        "role": role,
        "pid": int(pid),
        "gpu": int(gpu),
        "runtime_started_at_unix": timestamp,
        "updated_at_unix": timestamp,
    }
    heartbeat = {
        **heartbeat,
        "status": "running",
        "role": role,
        "pid": int(pid),
        "gpu": int(gpu),
        "runtime_started_at_unix": timestamp,
        "updated_at_unix": timestamp,
    }
    if resume:
        cursor["resumed_at_unix"] = timestamp
        heartbeat["resumed_at_unix"] = timestamp
        if previous_pid is not None and int(previous_pid) != int(pid):
            cursor["resumed_from_pid"] = int(previous_pid)
            heartbeat["resumed_from_pid"] = int(previous_pid)
    return cursor, heartbeat


def run_worker(*, role: str, gpu: int, resume: bool = False, evidence_dir: Path = EVIDENCE_ROOT, output_dir: Path = OUTPUT_ROOT, python: Path = PYTHON) -> dict[str, Any]:
    # This must happen before any LIBERO/MuJoCo import, environment creation,
    # or E1 worker namespace creation.  The policy child receives the same
    # physical-GPU identity below in PolicyServer.start().
    parent_runtime = bind_parent_renderer_environment(gpu)
    adapter_source = adapter_source_identity()
    report = preflight(role=role, gpu=gpu, evidence_dir=evidence_dir, output_dir=output_dir, adapter_source=adapter_source)
    inputs = load_frozen_inputs(adapter_source=adapter_source)
    identity = validate_checkpoint(role)
    role_root = _role_root(evidence_dir.resolve(), role)
    role_output_root = output_dir.resolve() / role
    _assert_new_namespace(role_root)
    _assert_new_namespace(role_output_root)
    if role_root.exists() and not resume and any(role_root.iterdir()):
        raise FileExistsError(f"{role} namespace already exists; use --resume only for the same input lock: {role_root}")
    if role_output_root.exists() and not resume and any(role_output_root.iterdir()):
        raise FileExistsError(f"{role} rollout namespace already exists; use --resume only for the same input lock: {role_output_root}")
    role_root.mkdir(parents=True, exist_ok=True)
    lock_path = role_root / "attempt.lock.json"
    if resume:
        if not lock_path.is_file():
            raise FileNotFoundError(f"cannot resume without input lock: {lock_path}")
        locked = strict_json_load(lock_path)
        _assert_locked_adapter_source(locked, adapter_source)
        if locked.get("input_hashes") != inputs["input_hashes"] or locked.get("identity") != identity:
            raise ValueError("resume input/identity mismatch")
        if locked.get("role") != role:
            raise ValueError("resume role mismatch")
        lock = locked
    else:
        lock = {
            "schema_version": "track3_fast400_h300_worker_lock_v1",
            "status": "running",
            "role": role,
            "pid": os.getpid(),
            "gpu": gpu,
            "adapter_source": adapter_source,
            "identity": identity,
            "input_hashes": inputs["input_hashes"],
            "case_fingerprints": {case["case_id"]: case["case_fingerprint"] for case in inputs["all_cases"]},
            "protocol": {"control_frequency_hz": CONTROL_HZ, "chunk_size": CHUNK_SIZE, "executed_actions_per_chunk": EXECUTE_ACTIONS, "max_steps": MAX_STEPS},
            "parent_renderer_runtime": parent_runtime,
            "official_result": False,
            "auto_upload": False,
            "auto_submit": False,
            "started_at_unix": time.time(),
        }
    cursor = strict_json_load(role_root / "cursor.json") if resume and (role_root / "cursor.json").is_file() else {"schema_version": "track3_fast400_h300_cursor_v1", "status": "running", "role": role, "pid": os.getpid(), "gpu": gpu, "completed_case_ids": [], "updated_at_unix": time.time()}
    heartbeat = strict_json_load(role_root / "heartbeat.json") if resume and (role_root / "heartbeat.json").is_file() else {"schema_version": "track3_fast400_h300_heartbeat_v1", "status": "running", "role": role, "pid": os.getpid(), "gpu": gpu, "updated_at_unix": time.time()}
    now = time.time()
    cursor, heartbeat = _refresh_runtime_state(cursor, heartbeat, role=role, gpu=gpu, resume=resume, pid=os.getpid(), now=now)
    lock.update({"status": "running", "pid": os.getpid(), "gpu": gpu, "runtime_started_at_unix": now})
    if resume:
        lock["resumed_at_unix"] = now
        if lock.get("previous_pid") is None and cursor.get("resumed_from_pid") is not None:
            lock["previous_pid"] = cursor["resumed_from_pid"]
    atomic_json(lock_path, lock)
    atomic_json(role_root / "cursor.json", cursor)
    atomic_json(role_root / "heartbeat.json", heartbeat)
    atomic_json(role_root / "preflight.json", report)
    heartbeat_updater = _HeartbeatUpdater(role_root / "heartbeat.json", heartbeat)
    heartbeat_updater.start()
    terminal_phase = "worker_exception"
    try:
        for import_path in (ROOT, ROOT / "LIBERO-plus", OPENPI_ROOT / "src"):
            if str(import_path) not in sys.path:
                sys.path.insert(0, str(import_path))
        from .libero_asset_bridge import activate as activate_libero_asset_bridge
        from pipeline.config import EvalConfig
        from pipeline.environment import EnvironmentManager

        activate_libero_asset_bridge(ROOT)
        manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=MAX_STEPS, seed=1000))
        rows = _run_cases(inputs["all_cases"], role, identity, manager, {}, role_root, role_output_root, cursor, gpu, python, heartbeat_updater)
        heartbeat_updater.raise_if_failed()
        summary = {
            "schema_version": "track3_fast400_h300_worker_summary_v1",
            "status": "completed",
            "role": role,
            "adapter_source": adapter_source,
            "role_identity": identity,
            "case_count": len(rows),
            "expected_case_count": 24,
            "coverage_complete": len(rows) == 24 and [row["case_id"] for row in rows] == [case["case_id"] for case in inputs["all_cases"]],
            "infrastructure_or_model_exception_count": sum(row.get("failure_class") in {"infrastructure", "model_error", "policy_nonfinite"} for row in rows),
            "success_count": sum(bool(row.get("success")) for row in rows),
            "input_hashes": inputs["input_hashes"],
            "protocol": {"control_frequency_hz": CONTROL_HZ, "chunk_size": CHUNK_SIZE, "executed_actions_per_chunk": EXECUTE_ACTIONS, "max_steps": MAX_STEPS},
            "official_result": False,
            "diagnostic_only": bool(ROLE_SPECS[role]["diagnostic_only"]),
            "git": git_info(ROOT),
            "parent_renderer_runtime": parent_runtime,
        }
        atomic_json(role_root / "summary.json", summary)
        lock = strict_json_load(lock_path)
        lock.update({"status": "completed", "finished_at_unix": time.time(), "completed_case_count": len(rows)})
        atomic_json(lock_path, lock)
        cursor.update({"status": "completed", "finished_at_unix": time.time(), "completed_case_count": len(rows)})
        atomic_json(role_root / "cursor.json", cursor)
        terminal_phase = "completed"
        return summary
    except BaseException as exc:
        failure = {"schema_version": "track3_fast400_h300_worker_failure_v1", "status": "failed", "role": role, "adapter_source": adapter_source, "error": f"{type(exc).__name__}: {exc}", "timestamp": time.time()}
        atomic_json(role_root / "failure.json", failure)
        locked = strict_json_load(lock_path)
        locked.update({"status": "failed", "finished_at_unix": time.time(), "failure": failure})
        atomic_json(lock_path, locked)
        atomic_json(role_root / "cursor.json", {**cursor, "status": "failed", "failure": failure})
        raise
    finally:
        heartbeat_updater.stop(terminal_phase=terminal_phase)


def _load_role_rows(role_root: Path, role: str, cases: list[dict[str, Any]], identity: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_ids = [case["case_id"] for case in cases]
    for case in cases:
        path = _case_file(role_root, case["case_id"])
        if path.with_suffix(path.suffix + ".partial").exists():
            raise ValueError(f"partial case residue in {role}: {path}")
        row = _load_completed(path, case, identity)
        if row is None:
            raise ValueError(f"missing completed case for {role}: {case['case_id']}")
        for key, expected in (("control_frequency_hz", CONTROL_HZ), ("chunk_size", CHUNK_SIZE), ("executed_actions_per_chunk", EXECUTE_ACTIONS), ("max_steps", MAX_STEPS)):
            if row.get(key) != expected:
                raise ValueError(f"protocol mismatch for {role}/{case['case_id']}: {key}")
        rows.append(row)
    actual_ids = [row["case_id"] for row in rows]
    if actual_ids != expected_ids or len(set(actual_ids)) != 24:
        raise ValueError(f"{role} case coverage/order is not exact")
    return rows


def continuation_gates(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    reverse = [pair for pair in pairs if pair["panel"] == "reverse_dev"]
    c20 = [pair for pair in pairs if pair["panel"] == "C20"]
    reverse_base = sum(bool(pair["baseline_success"]) for pair in reverse)
    reverse_candidate = sum(bool(pair["candidate_success"]) for pair in reverse)
    c20_base = sum(bool(pair["baseline_success"]) for pair in c20)
    c20_candidate = sum(bool(pair["candidate_success"]) for pair in c20)
    reverse_delta = reverse_candidate - reverse_base
    c20_delta = c20_candidate - c20_base
    gates = {
        "coverage_24_paired": len(pairs) == 24 and len(reverse) == 4 and len(c20) == 20,
        "reverse_at_least_plus_1_of_4": reverse_delta >= 1,
        "c20_drop_at_most_2_of_20": c20_delta >= -2,
        "no_infrastructure_model_or_nonfinite_exception": all(pair.get("infrastructure_ok", True) for pair in pairs),
        "case_fingerprints_and_seeds_match": all(pair.get("case_match", True) for pair in pairs),
        "action_traces_finite": all(pair.get("actions_finite", True) for pair in pairs),
        # This is a role-level gate: one non-zero trace is sufficient to show
        # that the complete 24-case role is not systemically all-zero.  A
        # single zero-action episode remains a valid per-case observation.
        "no_systemwide_all_zero_actions": (
            any(bool(pair.get("baseline_nonzero", False)) for pair in pairs)
            and any(bool(pair.get("candidate_nonzero", False)) for pair in pairs)
        ),
        "execute8_protocol": all(pair.get("execute8_protocol", True) for pair in pairs),
    }
    return {
        "reverse": {"baseline_success": reverse_base, "candidate_success": reverse_candidate, "delta": reverse_delta, "n": 4},
        "c20": {"baseline_success": c20_base, "candidate_success": c20_candidate, "delta": c20_delta, "n": 20},
        "gates": gates,
        "behavior_gate_passed": all(gates.values()),
    }


def paired_summary(baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], cases: list[dict[str, Any]], *, input_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    expected = [case["case_id"] for case in cases]
    baseline = {row["case_id"]: row for row in baseline_rows}
    candidate = {row["case_id"]: row for row in candidate_rows}
    if list(baseline) != expected or list(candidate) != expected:
        raise ValueError("paired case-ID order does not match the frozen 24-case panel")
    pairs: list[dict[str, Any]] = []
    for case_id in expected:
        base = baseline[case_id]
        cand = candidate[case_id]
        case_match = base.get("case_fingerprint") == cand.get("case_fingerprint") and all(base.get(key) == cand.get(key) for key in ("env_seed", "policy_seed", "init_state_index", "init_state_sha256"))
        infra_ok = base.get("failure_class") not in {"infrastructure", "model_error", "policy_nonfinite"} and cand.get("failure_class") not in {"infrastructure", "model_error", "policy_nonfinite"}
        pairs.append({
            "case_id": case_id,
            "panel": base["panel"],
            "baseline_success": bool(base.get("success")),
            "candidate_success": bool(cand.get("success")),
            "success_delta": int(bool(cand.get("success"))) - int(bool(base.get("success"))),
            "baseline_collision": bool(base.get("collision")),
            "candidate_collision": bool(cand.get("collision")),
            "baseline_timeout": bool(base.get("timeout")),
            "candidate_timeout": bool(cand.get("timeout")),
            "baseline_action_norm_max": float(base.get("action_norm_max", 0.0)),
            "candidate_action_norm_max": float(cand.get("action_norm_max", 0.0)),
            "case_match": case_match,
            "infrastructure_ok": infra_ok,
            "actions_finite": bool(base.get("actions_finite", False)) and bool(cand.get("actions_finite", False)),
            "baseline_nonzero": float(base.get("action_norm_max", 0.0)) > 1e-8,
            "candidate_nonzero": float(cand.get("action_norm_max", 0.0)) > 1e-8,
            "execute8_protocol": all(base.get(key) == expected_value and cand.get(key) == expected_value for key, expected_value in (("control_frequency_hz", CONTROL_HZ), ("chunk_size", CHUNK_SIZE), ("executed_actions_per_chunk", EXECUTE_ACTIONS))),
        })
    decision = continuation_gates(pairs)
    return {
        "schema_version": "track3_fast400_r6_vs_h300_step300_paired_summary_v1",
        "status": "evaluated",
        "candidate": "t3_fast400_r6_h300_4gpu_r1",
        "training_steps": 300,
        "baseline": "r6_params_only_fast400_norm",
        "paired_case_count": len(pairs),
        "rollout_count": len(baseline_rows) + len(candidate_rows),
        "pairs": pairs,
        **decision,
        "protocol": {"control_frequency_hz": CONTROL_HZ, "chunk_size": CHUNK_SIZE, "executed_actions_per_chunk": EXECUTE_ACTIONS, "max_steps": MAX_STEPS, "norm": "QUANTILES", "norm_stats_sha256": EXPECTED_NORM_SHA},
        "input_hashes": input_hashes or {},
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }


def merge(*, evidence_dir: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    """Merge two completed workers and write only behavior-gate evidence."""

    _assert_new_namespace(evidence_dir)
    adapter_source = adapter_source_identity()
    inputs = load_frozen_inputs(adapter_source=adapter_source)
    baseline_identity = validate_checkpoint("r6")
    candidate_identity = validate_checkpoint("step300")
    baseline_root = _role_root(evidence_dir.resolve(), "r6")
    candidate_root = _role_root(evidence_dir.resolve(), "step300")
    for lock_root, lock_role, lock_identity in ((baseline_root, "r6", baseline_identity), (candidate_root, "step300", candidate_identity)):
        lock_path = lock_root / "attempt.lock.json"
        if not lock_path.is_file():
            raise FileNotFoundError(f"missing worker input lock: {lock_path}")
        lock = strict_json_load(lock_path)
        if lock.get("status") != "completed" or lock.get("role") != lock_role:
            raise ValueError(f"worker is not a completed {lock_role} attempt: {lock_path}")
        try:
            _assert_locked_adapter_source(lock, adapter_source)
        except ValueError as exc:
            raise ValueError(f"worker adapter source identity mismatch: {lock_path}") from exc
        if lock.get("identity") != lock_identity or lock.get("input_hashes") != inputs["input_hashes"]:
            raise ValueError(f"worker input identity/hash mismatch: {lock_path}")
    if (evidence_dir / "paired_summary.json").exists() or (evidence_dir / "release_decision.json").exists():
        raise FileExistsError("paired merge artifacts already exist; refusing to overwrite release evidence")
    baseline_rows = _load_role_rows(baseline_root, "r6", inputs["all_cases"], baseline_identity)
    candidate_rows = _load_role_rows(candidate_root, "step300", inputs["all_cases"], candidate_identity)
    for base, cand in zip(baseline_rows, candidate_rows):
        base_trace = _validate_trace(base)
        cand_trace = _validate_trace(cand)
        base.update({"trace_finite": base_trace["finite"], "trace_nonzero": base_trace["nonzero"]})
        cand.update({"trace_finite": cand_trace["finite"], "trace_nonzero": cand_trace["nonzero"]})
    paired = paired_summary(baseline_rows, candidate_rows, inputs["all_cases"], input_hashes=inputs["input_hashes"])
    # Recompute the trace-specific gates after loading no-pickle arrays, rather
    # than trusting row metadata written by a worker.
    for pair, base, cand in zip(paired["pairs"], baseline_rows, candidate_rows):
        pair["actions_finite"] = bool(base["trace_finite"] and cand["trace_finite"])
        pair["baseline_nonzero"] = bool(base["trace_nonzero"])
        pair["candidate_nonzero"] = bool(cand["trace_nonzero"])
    paired.update(continuation_gates(paired["pairs"]))
    paired["behavior_gate_passed"] = bool(paired["behavior_gate_passed"])
    atomic_json(evidence_dir / "paired_summary.json", paired)
    release = {
        "schema_version": "track3_fast400_h300_release_decision_v1",
        "status": "behavior_gate_passed_fold_authorized" if paired["behavior_gate_passed"] else "behavior_gate_failed_no_zip",
        "behavior_gate_passed": paired["behavior_gate_passed"],
        "fold_authorized": paired["behavior_gate_passed"],
        "zip_created": False,
        "fold_started": False,
        "candidate": RUN_ID,
        "training_steps": 300,
        "adapter_source": adapter_source,
        "baseline": baseline_identity,
        "candidate_identity": candidate_identity,
        "paired_summary": str((evidence_dir / "paired_summary.json").resolve()),
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
        "reason": "reverse delta must be >= +1/4 and C20 delta must be >= -2/20; no infrastructure/model/nonfinite errors and exact coverage are required",
        "created_at_unix": time.time(),
    }
    atomic_json(evidence_dir / "release_decision.json", release)
    return release


def _pid_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        os.kill(value, 0)
    except (TypeError, ValueError, OSError):
        return False
    return True


def status(*, evidence_dir: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    """Read-only worker progress and PID liveness summary."""

    _assert_status_namespace(evidence_dir)
    roles: dict[str, Any] = {}
    for role in ROLE_SPECS:
        role_root = _role_root(evidence_dir.resolve(), role)
        item: dict[str, Any] = {"role": role, "exists": role_root.exists()}
        for name in ("attempt.lock.json", "cursor.json", "heartbeat.json", "summary.json", "failure.json"):
            path = role_root / name
            if path.is_file():
                item[name.removesuffix(".json")] = strict_json_load(path)
        pid = item.get("cursor", {}).get("pid") or item.get("attempt.lock", {}).get("pid")
        heartbeat = item.get("heartbeat")
        cursor_status = item.get("cursor", {}).get("status")
        heartbeat_status = heartbeat.get("status") if heartbeat is not None else None
        worker_running = cursor_status == "running" or heartbeat_status == "running"
        item["pid_alive"] = bool(worker_running and _pid_alive(pid))
        if heartbeat is None:
            item["heartbeat_age_sec"] = None
            item["heartbeat_fresh"] = False
        else:
            heartbeat_age = max(0.0, time.time() - float(heartbeat.get("updated_at_unix", 0.0)))
            item["heartbeat_age_sec"] = heartbeat_age
            item["heartbeat_fresh"] = bool(
                heartbeat.get("status") == "running" and heartbeat_age <= HEARTBEAT_STALE_AFTER_SEC
            )
        roles[role] = item
    return {"schema_version": "track3_fast400_h300_status_v1", "status": "read_only", "evidence_dir": str(evidence_dir.resolve()), "roles": roles, "release_decision": strict_json_load(evidence_dir / "release_decision.json") if (evidence_dir / "release_decision.json").is_file() else None}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "worker"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--role", choices=tuple(ROLE_SPECS), required=True)
        sub.add_argument("--gpu", type=int, required=True)
        sub.add_argument("--evidence-dir", type=Path, default=EVIDENCE_ROOT)
        sub.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
        sub.add_argument("--python", type=Path, default=PYTHON)
        if command == "worker":
            sub.add_argument("--resume", action="store_true")
    sub = subparsers.add_parser("merge")
    sub.add_argument("--evidence-dir", type=Path, default=EVIDENCE_ROOT)
    sub = subparsers.add_parser("status")
    sub.add_argument("--evidence-dir", type=Path, default=EVIDENCE_ROOT)
    sub = subparsers.add_parser("renderer-smoke")
    sub.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    sub.add_argument("--case-id", default=RENDERER_SMOKE_CASE_ID)
    sub.add_argument("--evidence-dir", type=Path, default=PARENT_EGL_EVIDENCE_ROOT)
    sub.add_argument("--python", type=Path, default=PYTHON)
    sub.add_argument("--hold-sec", type=float, default=RENDERER_SMOKE_HOLD_SEC)
    sub = subparsers.add_parser("renderer-smoke-pair")
    sub.add_argument("--evidence-dir", type=Path, default=PARENT_EGL_EVIDENCE_ROOT)
    sub.add_argument("--python", type=Path, default=PYTHON)
    sub.add_argument("--hold-sec", type=float, default=RENDERER_SMOKE_HOLD_SEC)
    sub = subparsers.add_parser("policy-server-smoke")
    sub.add_argument("--gpu", type=int, choices=(0,), default=0)
    sub.add_argument("--evidence-dir", type=Path, default=DL0_RETRY_EVIDENCE_ROOT)
    sub.add_argument("--python", type=Path, default=PYTHON)
    sub = subparsers.add_parser("_renderer-smoke-child", help=argparse.SUPPRESS)
    sub.add_argument("--gpu", type=int, required=True)
    sub.add_argument("--case-id", default=RENDERER_SMOKE_CASE_ID)
    sub.add_argument("--hold-sec", type=float, default=RENDERER_SMOKE_HOLD_SEC)
    sub = subparsers.add_parser("_policy-server-smoke-child", help=argparse.SUPPRESS)
    sub.add_argument("--gpu", type=int, choices=(0,), required=True)
    sub.add_argument("--output-dir", type=Path, required=True)
    sub.add_argument("--python", type=Path, default=PYTHON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(role=args.role, gpu=args.gpu, evidence_dir=args.evidence_dir, output_dir=args.output_dir)
    elif args.command == "worker":
        result = run_worker(role=args.role, gpu=args.gpu, resume=args.resume, evidence_dir=args.evidence_dir, output_dir=args.output_dir, python=args.python)
    elif args.command == "merge":
        result = merge(evidence_dir=args.evidence_dir)
    elif args.command == "renderer-smoke":
        result = renderer_smoke(gpu=args.gpu, case_id=args.case_id, evidence_dir=args.evidence_dir, python=args.python, hold_sec=args.hold_sec)
    elif args.command == "renderer-smoke-pair":
        result = renderer_smoke_pair(evidence_dir=args.evidence_dir, python=args.python, hold_sec=args.hold_sec)
    elif args.command == "policy-server-smoke":
        result = policy_server_smoke(gpu=args.gpu, evidence_dir=args.evidence_dir, python=args.python)
    elif args.command == "_renderer-smoke-child":
        try:
            result = renderer_smoke_child(gpu=args.gpu, case_id=args.case_id, hold_sec=args.hold_sec)
        except BaseException as exc:
            result = {"schema_version": "track3_fast400_h300_parent_renderer_smoke_child_v1", "status": "failed", "fresh_process": True, "pid": os.getpid(), "gpu": args.gpu, "case_id": args.case_id, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return 1
    elif args.command == "_policy-server-smoke-child":
        try:
            result = policy_server_smoke_child(gpu=args.gpu, output_dir=args.output_dir, python=args.python)
        except BaseException as exc:
            result = {"schema_version": "track3_fast400_h300_e1_r6_free_port_policy_server_smoke_child_v1", "status": "failed", "fresh_process": True, "pid": os.getpid(), "gpu": args.gpu, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            return 1
    else:
        result = status(evidence_dir=args.evidence_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("status") in {"passed", "completed", "read_only", "behavior_gate_passed_fold_authorized", "behavior_gate_failed_no_zip"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
