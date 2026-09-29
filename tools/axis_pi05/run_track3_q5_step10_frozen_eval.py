"""Run the immutable Track3 Q5 step-0/step-10 paired frozen evaluation.

This module is deliberately separate from ``phase1_rollout`` and
``eval_runner``.  Those runners encode older panel/norm contracts.  The
Track3 runner below owns the frozen Q5 config, 20 Hz protocol, execute8 chunk
discard semantics, and a new output namespace.  It never writes to the Q5
checkpoint namespace and never marks the result as an official competition
result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, canonical_json, git_info, sha256_file, write_json
from .eval_result import canonical_episode


PLAN = ROOT / "docs/AXIS_PI05_TRACK3_Q5_STEP10_FROZEN_EVAL_TO_OFFICIAL_TRAINING_HANDOFF_20260910.md"
Q5_EVIDENCE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q5_local_gpu3"
Q5_SUMMARY = Q5_EVIDENCE / "q5_summary.json"
Q5_LOCK = Q5_EVIDENCE / "q5_attempt.lock.json"
Q5_CURSOR = Q5_EVIDENCE / "cursor.json"
Q5_HEARTBEAT = Q5_EVIDENCE / "heartbeat.json"
Q6_SIGNOFF = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q6_semantic_signoff.json"
EVALUATION_FREEZE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/evaluation_freeze.json"
REVERSE_MANIFEST = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"
C20_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
NORM_STATS = ROOT / "data/manifests_track3/four_gold_80_20_v1/norm_stats.json"
R6_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
CHECKPOINT_ROOT = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3"
OPENPI_ROOT = ROOT / "third_party/axis_openpi"
PYTHON = ROOT / ".venv_axis_pi05/bin/python"
EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q7_gpu3_frozen_eval"
SERVER_MODULE = "tools.axis_pi05.checkpoint_http_server"
CONFIG_NAME = "axis_pi05_track3_q5"
GPU_ID = "3"
CONTROL_HZ = 20
CHUNK_SIZE = 10
EXECUTE_ACTIONS = 8
EXPECTED_NORM_SHA = "faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab"
EXPECTED_REVERSE_SHA = "64a1124ef17a866e259ba7e8bce96351cd183dff1ebf76c1afbd5694dfe0ec95"
EXPECTED_C20_SHA = "1dad183301e255e4287348470d962b9016420c2365f0ac8a8c2eef7ed0052231"
EXPECTED_R6_TREE_SHA = "d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f"


def strict_json_load(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject duplicate keys."""

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


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value))


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False) as handle:
        temp_path = Path(handle.name)
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix().encode()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def file_hash_map(path: Path) -> dict[str, str]:
    return {
        file_path.relative_to(path).as_posix(): sha256_file(file_path)
        for file_path in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix().encode())
    }



def restored_params_identity(path: Path) -> dict[str, Any]:
    """Hash the restored parameter PyTree without allocating on a project GPU."""
    code = r"""
import hashlib
import json
import numpy as np
from openpi.models import model

def walk(value, path=()):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from walk(value[key], path + (str(key),))
        return
    array = np.ascontiguousarray(np.asarray(value))
    yield {
        "path": list(path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }

leaves = list(walk(model.restore_params(PATH, restore_type=np.ndarray)))
payload = json.dumps(leaves, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
print(json.dumps({"leaf_count": len(leaves), "identity_sha256": hashlib.sha256(payload).hexdigest()}))
""".replace("PATH", json.dumps(str(path.resolve())))
    env = server_environment()
    env.update({"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})
    completed = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"no params identity emitted for {path}: {completed.stderr[-1000:]}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict) or not value.get("identity_sha256"):
        raise ValueError(f"invalid params identity for {path}: {value}")
    return value

def _free_gib(path: Path = ROOT) -> float:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize / 1024**3


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
    rows = []
    for line in result.stdout.splitlines():
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
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, used_memory = [part.strip() for part in line.split(",", 3)]
        rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": process_name, "used_memory_mib": int(used_memory)})
    return rows


def active_track3_processes() -> list[dict[str, Any]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
    needles = ("run_track3_q5_step10_frozen_eval", "checkpoint_http_server", "track3_four_gold_mix_v1/q7_gpu3_frozen_eval")
    rows = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        if any(needle in fields[1] for needle in needles):
            rows.append({"pid": int(fields[0]), "args": fields[1]})
    return rows


def _safe_case_id(case_id: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in case_id)


def _assert_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")


def _validate_checkpoint(step: int, *, compare_step0_to_r6: bool = False) -> dict[str, Any]:
    if step not in {0, 10}:
        raise ValueError(f"frozen evaluation only accepts step 0 or 10, got {step}")
    checkpoint = CHECKPOINT_ROOT / str(step)
    metadata = checkpoint / "_CHECKPOINT_METADATA"
    params = checkpoint / "params"
    train_state = checkpoint / "train_state"
    if not metadata.is_file() or not params.is_dir() or not train_state.is_dir():
        raise FileNotFoundError(f"incomplete Q5 checkpoint step {step}: {checkpoint}")
    if any(".partial" in path.name for path in checkpoint.rglob("*")):
        raise ValueError(f"partial checkpoint residue in {checkpoint}")
    params_tree_sha = sha256_tree(params)
    if compare_step0_to_r6:
        step0_identity = restored_params_identity(params)
        r6_identity = restored_params_identity(R6_PARAMS)
        if step0_identity != r6_identity:
            raise ValueError(f"Q5 step 0 restored params differ from frozen R6 params-only tree: {step0_identity} != {r6_identity}")
    return {
        "step": step,
        "path": str(checkpoint.resolve()),
        "metadata_sha256": sha256_file(metadata),
        "params_tree_sha256": params_tree_sha,
        "restored_params_identity": step0_identity if compare_step0_to_r6 else None,
        "params_file_count": sum(1 for path in params.rglob("*") if path.is_file()),
        "params_bytes": sum(path.stat().st_size for path in params.rglob("*") if path.is_file()),
        "train_state_present": True,
    }


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
        "max_steps": int(row.get("max_steps", 300)),
    })
    result["case_fingerprint"] = hashlib.sha256(canonical_json({key: result[key] for key in sorted(result) if key != "case_fingerprint"})).hexdigest()
    return result


def load_frozen_inputs(*, resume: bool = False) -> dict[str, Any]:
    required = (
        PLAN, Q5_SUMMARY, Q5_LOCK, Q5_CURSOR, Q5_HEARTBEAT, Q6_SIGNOFF,
        EVALUATION_FREEZE, REVERSE_MANIFEST, C20_MANIFEST, NORM_STATS,
        R6_PARAMS, CHECKPOINT_ROOT / "0", CHECKPOINT_ROOT / "10",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen evaluation input(s): " + ", ".join(missing))

    q5 = strict_json_load(Q5_SUMMARY)
    q5_lock = strict_json_load(Q5_LOCK)
    q5_cursor = strict_json_load(Q5_CURSOR)
    q5_heartbeat = strict_json_load(Q5_HEARTBEAT)
    q6 = strict_json_load(Q6_SIGNOFF)
    freeze = strict_json_load(EVALUATION_FREEZE)
    if q5.get("status") != "track3_four_gold_local_single_gpu_canary_passed_ready_for_official_bundle":
        raise ValueError(f"unexpected Q5 status: {q5.get('status')}")
    if q5_lock.get("status") != "passed" or q5_lock.get("trainer_returncode") != 0:
        raise ValueError("Q5 lock is not a passed, zero-returncode closeout")
    if q5_cursor.get("status") != "passed" or q5_heartbeat.get("status") != "stopped":
        raise ValueError("Q5 cursor/heartbeat is not closed")
    if q6.get("status") != "passed_shifted_pre_action_semantic_signoff":
        raise ValueError(f"Q6 semantic signoff is not passed: {q6.get('status')}")
    if freeze.get("status") != "evaluation_set_frozen_before_q5":
        raise ValueError(f"evaluation freeze is not locked: {freeze.get('status')}")

    _assert_hash(NORM_STATS, EXPECTED_NORM_SHA)
    _assert_hash(REVERSE_MANIFEST, EXPECTED_REVERSE_SHA)
    _assert_hash(C20_MANIFEST, EXPECTED_C20_SHA)
    r6_tree_sha = q5.get("input_hashes", {}).get("r6_params_tree_sha256")
    if r6_tree_sha != EXPECTED_R6_TREE_SHA:
        raise ValueError(f"Q5 R6 tree identity mismatch: {r6_tree_sha}")
    locked_input = EVIDENCE_ROOT / "input_lock.json"
    if resume and locked_input.is_file():
        locked = strict_json_load(locked_input)
        checkpoints = {int(key): value for key, value in locked.get("checkpoints", {}).items()}
        if sorted(checkpoints) != [0, 10]:
            raise ValueError("resume input lock has incomplete checkpoint identity")
        for step in (0, 10):
            checkpoint = CHECKPOINT_ROOT / str(step)
            if not (checkpoint / "_CHECKPOINT_METADATA").is_file() or not (checkpoint / "params").is_dir() or not (checkpoint / "train_state").is_dir():
                raise FileNotFoundError(f"incomplete Q5 checkpoint step {step}: {checkpoint}")
            if any(".partial" in path.name for path in checkpoint.rglob("*")):
                raise ValueError(f"partial checkpoint residue in {checkpoint}")
    else:
        checkpoints = {
            0: _validate_checkpoint(0, compare_step0_to_r6=True),
            10: _validate_checkpoint(10),
        }

    reverse_source = {row["case_id"]: row for row in strict_jsonl_load(REVERSE_MANIFEST)}
    c20_source = {row["case_id"]: row for row in strict_jsonl_load(C20_MANIFEST)}
    reverse_cases = []
    for frozen in freeze.get("reverse_dev", {}).get("rows", []):
        case_id = frozen["case_id"]
        source = reverse_source.get(case_id)
        if source is None:
            raise ValueError(f"reverse freeze case missing from source manifest: {case_id}")
        for key in ("env_seed", "policy_seed", "init_state_index", "init_state_sha256", "bddl_sha256"):
            if source.get(key) != frozen.get(key):
                raise ValueError(f"reverse freeze mismatch for {case_id}: {key}")
        reverse_cases.append(_normalise_case(
            source,
            panel="reverse_dev",
            panel_manifest_sha=EXPECTED_REVERSE_SHA,
            benchmark="libero_t3",
            suite="libero_t3",
            task_id=int(frozen["task_index"]),
        ))
    if len(reverse_cases) != 4:
        raise ValueError(f"reverse freeze must contain 4 cases, got {len(reverse_cases)}")

    c20_cases = []
    for frozen in freeze.get("c20_retention", {}).get("cases", []):
        case_id = frozen["case_id"]
        source = c20_source.get(case_id)
        if source is None:
            raise ValueError(f"C20 freeze case missing from source manifest: {case_id}")
        for key in ("env_seed", "policy_seed", "init_state_index", "init_state_sha256"):
            if source.get(key) != frozen.get(key):
                raise ValueError(f"C20 freeze mismatch for {case_id}: {key}")
        c20_cases.append(_normalise_case(
            source,
            panel="C20",
            panel_manifest_sha=EXPECTED_C20_SHA,
            benchmark="libero_mix",
            suite="C20",
            task_id=int(source["official_task_index"]),
        ))
    if len(c20_cases) != 20:
        raise ValueError(f"C20 freeze must contain 20 cases, got {len(c20_cases)}")

    input_hashes = {
        "plan_sha256": sha256_file(PLAN),
        "q5_summary_sha256": sha256_file(Q5_SUMMARY),
        "q6_signoff_sha256": sha256_file(Q6_SIGNOFF),
        "evaluation_freeze_sha256": sha256_file(EVALUATION_FREEZE),
        "reverse_manifest_sha256": EXPECTED_REVERSE_SHA,
        "c20_manifest_sha256": EXPECTED_C20_SHA,
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "r6_params_tree_sha256": EXPECTED_R6_TREE_SHA,
    }
    if resume and locked_input.is_file() and {key: strict_json_load(locked_input).get(key) for key in input_hashes} != input_hashes:
        raise ValueError("resume input hash mismatch")

    return {
        "q5": q5,
        "q5_lock": q5_lock,
        "q6": q6,
        "freeze": freeze,
        "checkpoints": checkpoints,
        "reverse_cases": reverse_cases,
        "c20_cases": c20_cases,
        "all_cases": reverse_cases + c20_cases,
        "input_hashes": input_hashes,
    }


def server_environment() -> dict[str, str]:
    python_path = [str(ROOT), str(ROOT / "LIBERO-plus"), str(OPENPI_ROOT / "src")]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    gpu_id = os.environ.get("AXIS_PI05_EVAL_GPU", GPU_ID)
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": os.pathsep.join(python_path),
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
        "HF_HUB_OFFLINE": "1",
        "AXIS_PI05_NORM_STATS": str(NORM_STATS),
        "LIBERO_CONTROL_FREQ": str(CONTROL_HZ),
        "LIBERO_CONFIG_PATH": str(ROOT / ".eval_libero"),
        "AXIS_PI05_LIBERO_ASSET_BRIDGE": "1",
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": gpu_id,
        "AXIS_PI05_EVAL_GPU": gpu_id,
    })
    return env


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
    def __init__(self, checkpoint: Path, output_dir: Path):
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.port = _free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.client: Any = None

    def start(self) -> Any:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_handle = (self.output_dir / "policy_server.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(PYTHON), "-m", SERVER_MODULE, "--checkpoint", str(self.checkpoint), "--port", str(self.port), "--config", CONFIG_NAME],
            cwd=ROOT,
            env=server_environment(),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        from pipeline.remote_policy import RemotePolicyClient

        self.client = RemotePolicyClient(f"http://127.0.0.1:{self.port}", timeout_sec=90.0)
        try:
            self.client.wait_for_server(timeout=900.0)
        except Exception:
            self.close()
            raise
        return self.client

    def close(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            return {"pid": None, "returncode": None, "clean_shutdown": True}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        return {"pid": process.pid, "returncode": process.returncode, "clean_shutdown": process.poll() is not None}


def _task_info_for_case(case: dict[str, Any], manager: Any, cache: dict[str, Any]) -> Any:
    from pipeline.environment import TaskInfo

    if case["benchmark"] == "libero_mix":
        import torch

        init_states_path = Path(case["init_states_path"])
        init_states = torch.load(init_states_path, weights_only=False)
        state = np.asarray(init_states[int(case["init_state_index"])])
        if hashlib.sha256(canonical_json({"dtype": str(state.dtype), "shape": list(state.shape), "data_sha256": hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()})).hexdigest() != case["init_state_sha256"]:
            # C20 uses the same canonical state hash as phase1_rollout; use its
            # implementation for the exact comparison before constructing info.
            from .phase1_rollout import _state_hash

            if _state_hash(state) != case["init_state_sha256"]:
                raise ValueError(f"C20 init state changed: {case['case_id']}")
        if sha256_file(Path(case["bddl_path"])) != case["bddl_sha256"]:
            raise ValueError(f"C20 BDDL changed: {case['case_id']}")
        return TaskInfo(
            task_id=int(case["task_id"]),
            name=str(case["task_name"]),
            language=str(case["language"]),
            bddl_file=str(case["bddl_path"]),
            init_states=init_states,
            benchmark_name="libero_mix",
        )

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


def _write_trace(path: Path, actions: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite action trace: {path}")
    atomic_npy(path, actions)


def _episode_row(case: dict[str, Any], step: int, identity: dict[str, Any], episode: Any, output_dir: Path, *, error: Exception | None = None) -> dict[str, Any]:
    actions = np.asarray(getattr(episode, "actions", []), dtype=np.float32).reshape((-1, 7)) if error is None else np.empty((0, 7), dtype=np.float32)
    trace_path = output_dir / "action_traces" / f"step_{step}" / f"{_safe_case_id(str(case['case_id']))}.npy"
    if trace_path.exists():
        preserved_trace = trace_path.with_name(trace_path.name + f".previous_{int(time.time())}")
        shutil.move(trace_path, preserved_trace)
    _write_trace(trace_path, actions)
    finite = bool(np.all(np.isfinite(actions)))
    norms = np.linalg.norm(actions, axis=1) if actions.size else np.empty((0,), dtype=np.float32)
    gripper = actions[:, 6] if actions.size else np.empty((0,), dtype=np.float32)
    unique_signs = sorted({int(np.sign(value)) for value in gripper})
    if error is not None:
        text = f"{type(error).__name__}: {error}"
        if isinstance(error, FloatingPointError):
            failure_class = "policy_nonfinite"
        elif any(token in text.lower() for token in ("server", "connection", "bddl", "init_state", "mujoco", "environment", "asset", "scene", "texture", "file not found")):
            failure_class = "infrastructure"
        else:
            failure_class = "model_error"
        result = canonical_episode({
            **case,
            "candidate": f"q5_step{step}",
            "checkpoint": identity["path"],
            "checkpoint_sha256": identity["params_tree_sha256"],
            "manifest_sha256": case["panel_manifest_sha256"],
            "protocol_sha256": None,
            "instruction": case.get("language"),
            "success": False,
            "collision": False,
            "goal_reached": False,
            "timeout": False,
            "steps": 0,
            "elapsed_sec": 0.0,
            "failure_reason": "exception",
            "failure_class": failure_class,
            "error_summary": text,
            "video_path": None,
            "trajectory_path": None,
        })
    else:
        video_path = getattr(episode, "video_path", None)
        result = canonical_episode({
            **case,
            "candidate": f"q5_step{step}",
            "checkpoint": identity["path"],
            "checkpoint_sha256": identity["params_tree_sha256"],
            "manifest_sha256": case["panel_manifest_sha256"],
            "protocol_sha256": None,
            "instruction": case.get("language"),
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
            "video_path": video_path,
            "trajectory_path": None,
        })
    result.update({
        "panel": case["panel"],
        "case_fingerprint": case["case_fingerprint"],
        "checkpoint_step": int(step),
        "checkpoint_metadata_sha256": identity["metadata_sha256"],
        "checkpoint_params_tree_sha256": identity["params_tree_sha256"],
        "norm_stats_sha256": EXPECTED_NORM_SHA,
        "evaluation_freeze_sha256": sha256_file(EVALUATION_FREEZE),
        "control_frequency_hz": CONTROL_HZ,
        "chunk_size": CHUNK_SIZE,
        "executed_actions_per_chunk": EXECUTE_ACTIONS,
        "action_shape": list(actions.shape),
        "actions_finite": finite,
        "action_norm_mean": float(norms.mean()) if norms.size else 0.0,
        "action_norm_max": float(norms.max()) if norms.size else 0.0,
        "gripper_unique_signs": unique_signs,
        "action_trace_path": str(trace_path.resolve()),
        "official_result": False,
        "diagnostic_only": True,
    })
    return result


def _case_file(root: Path, case_id: str) -> Path:
    return root / "cases" / f"{_safe_case_id(case_id)}.json"


def _load_completed(path: Path, case: dict[str, Any], step: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = strict_json_load(path)
    if value.get("case_fingerprint") != case["case_fingerprint"] or value.get("checkpoint_step") != step:
        raise ValueError(f"completed case input mismatch: {path}")
    if value.get("failure_reason") == "exception":
        return None
    return value


def _run_cases(cases: list[dict[str, Any]], step: int, phase: str, identity: dict[str, Any], manager: Any, task_cache: dict[str, Any], output_dir: Path, cursor: dict[str, Any]) -> list[dict[str, Any]]:
    from pipeline.config import PerturbationConfig
    from pipeline.rollout import RolloutExecutor

    step_root = output_dir / phase / f"step_{step}"
    step_root.mkdir(parents=True, exist_ok=True)
    os.environ["T3_VIDEO_DIR"] = str(step_root / "videos")
    os.environ["T3_VIDEO_FPS"] = str(CONTROL_HZ)
    config = manager.config
    executor = RolloutExecutor(manager, config)
    result_rows: list[dict[str, Any]] = []
    server = PolicyServer(Path(identity["path"]), step_root)
    client = server.start()
    policy = Execute8Policy(client)
    try:
        for index, case in enumerate(cases):
            case_path = _case_file(step_root, str(case["case_id"]))
            completed = _load_completed(case_path, case, step)
            if completed is not None:
                result_rows.append(completed)
                cursor["completed_case_ids"] = sorted(set(cursor.get("completed_case_ids", [])) | {case["case_id"]})
                atomic_json(EVIDENCE_ROOT / "cursor.json", cursor)
                continue
            if case_path.exists():
                preserved = case_path.with_name(case_path.name + f".previous_{int(time.time())}")
                shutil.move(case_path, preserved)
            partial_path = case_path.with_suffix(case_path.suffix + ".partial")
            if partial_path.exists():
                preserved = partial_path.with_name(partial_path.name + f".previous_{int(time.time())}")
                shutil.move(partial_path, preserved)
            atomic_json(partial_path, {"status": "running", "case_id": case["case_id"], "checkpoint_step": step, "case_fingerprint": case["case_fingerprint"], "started_at_unix": time.time()})
            cursor.update({"status": "running", "phase": phase, "checkpoint_step": step, "case_index": index, "case_id": case["case_id"]})
            atomic_json(EVIDENCE_ROOT / "cursor.json", cursor)
            try:
                task_info = _task_info_for_case(case, manager, task_cache)
                init_state = manager.get_init_state(task_info, int(case["init_state_index"])) if case["benchmark"] != "libero_mix" else np.asarray(task_info.init_states[int(case["init_state_index"])])
                episode = executor.evaluate_case(
                    policy,
                    task_info,
                    init_state=init_state,
                    env_seed=int(case["env_seed"]),
                    policy_seed=int(case["policy_seed"]),
                    max_steps=int(case["max_steps"]),
                    case_id=str(case["case_id"]),
                    perturbation=PerturbationConfig(),
                )
                row = _episode_row(case, step, identity, episode, step_root)
            except Exception as exc:
                row = _episode_row(case, step, identity, type("FailedEpisode", (), {"actions": [], "success": False, "collided": False, "goal_reached": False, "timeout": False, "total_steps": 0, "elapsed_time_sec": 0.0, "failure_reason": "exception", "video_path": None})(), step_root, error=exc)
            atomic_json(case_path, row)
            partial_path.unlink(missing_ok=True)
            result_rows.append(row)
            cursor["completed_case_ids"] = sorted(set(cursor.get("completed_case_ids", [])) | {case["case_id"]})
            atomic_json(EVIDENCE_ROOT / "cursor.json", cursor)
            print(f"{phase}/step_{step} {index + 1}/{len(cases)} {case['case_id']}", flush=True)
    finally:
        server_meta = server.close()
        atomic_json(step_root / "server_shutdown.json", server_meta)
        os.environ.pop("T3_VIDEO_DIR", None)
    return result_rows


def _no_infra(rows: list[dict[str, Any]]) -> bool:
    return not any(row.get("failure_class") in {"infrastructure", "model_error", "policy_nonfinite"} for row in rows)


def _summary(rows: list[dict[str, Any]], *, phase: str, expected_count: int, input_hashes: dict[str, str], pre_gpu: list[dict[str, Any]], post_gpu: list[dict[str, Any]], pre_apps: list[dict[str, Any]], post_apps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "track3_q5_frozen_eval_summary_v1",
        "status": "passed" if len(rows) == expected_count and _no_infra(rows) else "failed",
        "phase": phase,
        "case_count": len(rows),
        "expected_case_count": expected_count,
        "coverage_complete": len(rows) == expected_count and len({row["case_id"] for row in rows}) == expected_count,
        "infrastructure_or_model_exception_count": sum(row.get("failure_class") in {"infrastructure", "model_error", "policy_nonfinite"} for row in rows),
        "success_count": sum(bool(row.get("success")) for row in rows),
        "collision_count": sum(bool(row.get("collision")) for row in rows),
        "timeout_count": sum(bool(row.get("timeout")) for row in rows),
        "action_norm_mean": float(np.mean([row.get("action_norm_mean", 0.0) for row in rows])) if rows else 0.0,
        "action_norm_max": float(np.max([row.get("action_norm_max", 0.0) for row in rows])) if rows else 0.0,
        "input_hashes": input_hashes,
        "protocol": {"control_frequency_hz": CONTROL_HZ, "chunk_size": CHUNK_SIZE, "executed_actions_per_chunk": EXECUTE_ACTIONS},
        "gpu_boundary": {"physical_gpu": 3, "gpu0_1_2_touched": False, "pre_snapshot": pre_gpu, "post_snapshot": post_gpu, "pre_compute_apps": pre_apps, "post_compute_apps": post_apps},
        "disk": {"free_gib_after": _free_gib(), "training_threshold_unchanged_gib": 500.0},
        "official_result": False,
        "diagnostic_only": True,
        "git": git_info(ROOT),
        "artifacts": {"rows": [str(row) for row in rows]},
    }


def _paired_summary(step0: list[dict[str, Any]], step10: list[dict[str, Any]], inputs: dict[str, Any], pre_gpu: list[dict[str, Any]], post_gpu: list[dict[str, Any]], pre_apps: list[dict[str, Any]], post_apps: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = {row["case_id"]: row for row in step0}
    candidate = {row["case_id"]: row for row in step10}
    expected = [row["case_id"] for row in inputs["all_cases"]]
    if list(baseline) != expected or list(candidate) != expected:
        # Dict insertion order is intentional: it detects a changed case order
        # rather than silently sorting away a protocol mismatch.
        raise ValueError("paired case order does not match frozen case order")
    pairs = [{
        "case_id": case_id,
        "panel": baseline[case_id]["panel"],
        "step0_success": bool(baseline[case_id]["success"]),
        "step10_success": bool(candidate[case_id]["success"]),
        "success_delta": int(bool(candidate[case_id]["success"])) - int(bool(baseline[case_id]["success"])),
        "step0_collision": bool(baseline[case_id]["collision"]),
        "step10_collision": bool(candidate[case_id]["collision"]),
        "step0_timeout": bool(baseline[case_id]["timeout"]),
        "step10_timeout": bool(candidate[case_id]["timeout"]),
        "step0_action_norm_mean": baseline[case_id]["action_norm_mean"],
        "step10_action_norm_mean": candidate[case_id]["action_norm_mean"],
    } for case_id in expected]
    c20_pairs = [pair for pair in pairs if pair["panel"] == "C20"]
    c20_step0 = sum(pair["step0_success"] for pair in c20_pairs)
    c20_step10 = sum(pair["step10_success"] for pair in c20_pairs)
    all_rows = step0 + step10
    gripper_signs = sorted({sign for row in all_rows for sign in row.get("gripper_unique_signs", [])})
    nonzero_cases = sum(row.get("action_norm_max", 0.0) > 1e-8 for row in all_rows)
    gate = {
        "coverage_48_of_48": len(pairs) == 24 and len(step0) == 24 and len(step10) == 24,
        "no_infrastructure_model_or_nonfinite_exception": _no_infra(all_rows),
        "c20_step10_drop_at_most_2": c20_step10 >= c20_step0 - 2,
        "no_systemwide_all_zero_actions": nonzero_cases == len(all_rows),
        "gripper_has_nonconstant_sign_or_zero_transition": len(gripper_signs) >= 2,
    }
    return {
        "schema_version": "track3_q5_step0_step10_paired_summary_v1",
        "status": "passed" if all(gate.values()) else "failed",
        "case_count": len(pairs) * 2,
        "paired_case_count": len(pairs),
        "step0_case_count": len(step0),
        "step10_case_count": len(step10),
        "pairs": pairs,
        "reverse_dev": {"case_count": 8, "step0_success": sum(pair["step0_success"] for pair in pairs if pair["panel"] == "reverse_dev"), "step10_success": sum(pair["step10_success"] for pair in pairs if pair["panel"] == "reverse_dev")},
        "c20": {"case_count": 40, "step0_success": c20_step0, "step10_success": c20_step10, "success_delta": c20_step10 - c20_step0},
        "gates": gate,
        "input_hashes": inputs["input_hashes"],
        "protocol": {"control_frequency_hz": CONTROL_HZ, "chunk_size": CHUNK_SIZE, "executed_actions_per_chunk": EXECUTE_ACTIONS, "same_case_order": True, "official_result": False, "diagnostic_only": True},
        "gpu_boundary": {"physical_gpu": 3, "gpu0_1_2_touched": False, "pre_snapshot": pre_gpu, "post_snapshot": post_gpu, "pre_compute_apps": pre_apps, "post_compute_apps": post_apps},
        "disk": {"free_gib_after": _free_gib(), "training_threshold_unchanged_gib": 500.0},
        "official_result": False,
        "diagnostic_only": True,
        "git": git_info(ROOT),
    }


def _new_attempt_guard(resume: bool) -> None:
    if not OUTPUT_ROOT.exists() and not EVIDENCE_ROOT.exists():
        return
    lock_path = EVIDENCE_ROOT / "q7_attempt.lock.json"
    if not resume or not lock_path.is_file():
        raise FileExistsError("q7 frozen-evaluation namespace already exists; use --resume only for the same incomplete input-locked attempt")
    lock = strict_json_load(lock_path)
    if lock.get("status") == "passed" and lock.get("result_keys") != ["canary"]:
        raise FileExistsError("q7 frozen-evaluation attempt is already complete")


def run(*, resume: bool = False, phase: str = "all") -> dict[str, Any]:
    if phase not in {"canary", "full", "all"}:
        raise ValueError(phase)
    inputs = load_frozen_inputs(resume=resume)
    if active_track3_processes():
        raise RuntimeError("another Track3 frozen-eval/server process is active")
    pre_gpu = gpu_snapshot()
    pre_apps = compute_app_snapshot()
    gpu3 = next(row for row in pre_gpu if row["index"] == 3)
    gpu3_apps = [row for row in pre_apps if row["gpu_uuid"] == gpu3["uuid"]]
    if gpu3["memory_used_mib"] > 64 or gpu3["utilization_gpu_percent"] != 0 or gpu3_apps:
        raise RuntimeError(f"GPU3 is not idle: {gpu3}; apps={gpu3_apps}")
    _new_attempt_guard(resume)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = EVIDENCE_ROOT / "q7_attempt.lock.json"
    cursor_path = EVIDENCE_ROOT / "cursor.json"
    heartbeat_path = EVIDENCE_ROOT / "heartbeat.json"
    lock = {
        "schema_version": "track3_q5_frozen_eval_attempt_v1",
        "status": "running",
        "pid": os.getpid(),
        "started_at_unix": time.time(),
        "phase": phase,
        "gpu": 3,
        "input_hashes": inputs["input_hashes"],
        "checkpoints": inputs["checkpoints"],
        "official_result": False,
    }
    if resume:
        existing = strict_json_load(lock_path)
        if existing.get("input_hashes") != lock["input_hashes"]:
            raise ValueError("resume input hash mismatch")
    else:
        atomic_json(lock_path, lock)
    cursor = {"schema_version": "track3_q5_frozen_eval_cursor_v1", "status": "running", "pid": os.getpid(), "completed_case_ids": []}
    heartbeat = {"schema_version": "track3_q5_frozen_eval_heartbeat_v1", "status": "running", "pid": os.getpid(), "timestamp": time.time()}
    atomic_json(cursor_path, cursor)
    atomic_json(heartbeat_path, heartbeat)
    atomic_json(EVIDENCE_ROOT / "input_lock.json", inputs["input_hashes"] | {"checkpoints": inputs["checkpoints"], "case_fingerprints": {row["case_id"]: row["case_fingerprint"] for row in inputs["all_cases"]}})

    # The parent owns the simulator and is also restricted to the same physical
    # card.  The server receives the same environment explicitly.
    os.environ.update(server_environment())
    for import_path in (ROOT, ROOT / "LIBERO-plus", OPENPI_ROOT / "src"):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    os.environ["T3_VIDEO_FPS"] = str(CONTROL_HZ)
    from .libero_asset_bridge import activate as activate_libero_asset_bridge

    asset_bridge = activate_libero_asset_bridge(ROOT)
    atomic_json(EVIDENCE_ROOT / "runtime_asset_bridge.json", asset_bridge)
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager

    manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=300, seed=1000))
    task_cache: dict[str, Any] = {}
    all_results: dict[str, Any] = {}
    try:
        if phase in {"canary", "all"}:
            canary_cases = [inputs["reverse_cases"][0], inputs["c20_cases"][0]]
            canary_rows = []
            for step in (0, 10):
                identity = inputs["checkpoints"][step]
                canary_rows.extend(_run_cases(canary_cases, step, "canary", identity, manager, task_cache, OUTPUT_ROOT, cursor))
            canary = {"schema_version": "track3_q5_frozen_eval_canary_v1", "status": "passed" if len(canary_rows) == 4 and _no_infra(canary_rows) else "failed", "case_count": len(canary_rows), "expected_case_count": 4, "rows": canary_rows, "official_result": False, "diagnostic_only": True}
            atomic_json(EVIDENCE_ROOT / "e1_canary_summary.json", canary)
            all_results["canary"] = canary
            if canary["status"] != "passed":
                raise RuntimeError("E1 GPU3 restore/inference canary failed")
        if phase in {"full", "all"}:
            full_rows_by_step: dict[int, list[dict[str, Any]]] = {}
            for step in (0, 10):
                full_rows_by_step[step] = _run_cases(inputs["all_cases"], step, "full", inputs["checkpoints"][step], manager, task_cache, OUTPUT_ROOT, cursor)
            post_gpu = gpu_snapshot()
            post_apps = compute_app_snapshot()
            for step, rows in full_rows_by_step.items():
                summary = _summary(rows, phase="full", expected_count=24, input_hashes=inputs["input_hashes"], pre_gpu=pre_gpu, post_gpu=post_gpu, pre_apps=pre_apps, post_apps=post_apps)
                atomic_json(EVIDENCE_ROOT / f"step_{step}_summary.json", summary)
            paired = _paired_summary(full_rows_by_step[0], full_rows_by_step[10], inputs, pre_gpu, post_gpu, pre_apps, post_apps)
            atomic_json(EVIDENCE_ROOT / "paired_summary.json", paired)
            all_results["paired"] = paired
            if paired["status"] != "passed":
                raise RuntimeError("E2 48-case paired frozen evaluation failed its gate")
        lock.update({"status": "passed", "finished_at_unix": time.time(), "result_keys": sorted(all_results)})
        atomic_json(lock_path, lock)
        cursor.update({"status": "passed", "finished_at_unix": time.time()})
        atomic_json(cursor_path, cursor)
        heartbeat.update({"status": "stopped", "timestamp": time.time()})
        atomic_json(heartbeat_path, heartbeat)
        return all_results
    except BaseException as exc:
        failure = {"schema_version": "track3_q5_frozen_eval_failure_v1", "status": "failed", "error": f"{type(exc).__name__}: {exc}", "pid": os.getpid(), "timestamp": time.time()}
        atomic_json(EVIDENCE_ROOT / "failure.json", failure)
        lock.update({"status": "failed", "finished_at_unix": time.time(), "failure": failure})
        atomic_json(lock_path, lock)
        atomic_json(cursor_path, {**cursor, "status": "failed", "failure": failure})
        atomic_json(heartbeat_path, {**heartbeat, "status": "stopped", "timestamp": time.time()})
        raise
    finally:
        os.environ.pop("T3_VIDEO_DIR", None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("canary", "full", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(resume=args.resume, phase=args.phase)
    print(json.dumps({"status": "passed", "keys": sorted(result)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
