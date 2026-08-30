"""Lossless, auditable snapshots for N1 environment and policy probes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, canonical_json

SNAPSHOT_SCHEMA_VERSION = 1
RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MUJOCO_GL",
    "MUJOCO_EGL_DEVICE_ID",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ("numpy", "torch", "jax", "jaxlib", "mujoco", "robosuite", "libero"):
        try:
            module = __import__(name)
            modules[name] = {
                "version": getattr(module, "__version__", None),
                "file": str(getattr(module, "__file__", "")),
            }
        except Exception as exc:
            modules[name] = {"version": None, "file": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "thread_count": threading.active_count(),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {key: os.environ.get(key) for key in RUNTIME_ENV_KEYS},
        "packages": {name: _package_version(name) for name in ("numpy", "torch", "jax", "jaxlib", "mujoco", "robosuite", "libero")},
        "modules": modules,
    }


def gpu_snapshot() -> dict[str, Any]:
    """Read GPU/PID state without initializing a CUDA context."""

    result: dict[str, Any] = {"timestamp_unix": time.time(), "pid": os.getpid(), "gpus": [], "error": None}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        by_uuid: dict[str, dict[str, Any]] = {}
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                continue
            index, uuid, memory = parts
            by_uuid[uuid] = {"index": int(index), "uuid": uuid, "memory_used_mib": int(float(memory)), "pids": []}
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in apps.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4 or parts[0] not in by_uuid:
                continue
            by_uuid[parts[0]]["pids"].append({"pid": int(parts[1]), "process_name": parts[2], "used_memory_mib": int(float(parts[3]))})
        result["gpus"] = [by_uuid[key] for key in sorted(by_uuid, key=lambda value: by_uuid[value]["index"])]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def gpu_boundary_evidence(samples: list[dict[str, Any]], *, target_pids: set[int], allowed_gpu_ids: set[int]) -> dict[str, Any]:
    """Derive GPU0 evidence from explicit before/during/after samples."""

    if len(samples) < 2 or any(not sample.get("gpus") for sample in samples):
        raise ValueError("gpu0_touched cannot be determined without GPU monitor samples")
    observed_target: dict[int, list[int]] = {}
    for sample in samples:
        for gpu in sample["gpus"]:
            observed_target.setdefault(int(gpu["index"]), []).extend(
                int(item["pid"]) for item in gpu.get("pids", []) if int(item["pid"]) in target_pids
            )
    target_gpu_ids = {gpu_id for gpu_id, pids in observed_target.items() if pids}
    forbidden_target_gpu_ids = sorted(target_gpu_ids - allowed_gpu_ids)
    return {
        "sample_count": len(samples),
        "target_pids": sorted(target_pids),
        "allowed_gpu_ids": sorted(allowed_gpu_ids),
        "target_gpu_ids_observed": sorted(target_gpu_ids),
        "forbidden_target_gpu_ids": forbidden_target_gpu_ids,
        "gpu0_external_pids": sorted({
            int(item["pid"])
            for sample in samples
            for gpu in sample["gpus"]
            if int(gpu["index"]) == 0
            for item in gpu.get("pids", [])
            if int(item["pid"]) not in target_pids
        }),
        "gpu0_touched": 0 in target_gpu_ids,
        "evidence_complete": True,
    }


def array_descriptor(array: Any, *, path: str | None = None) -> dict[str, Any]:
    value = np.ascontiguousarray(np.asarray(array))
    descriptor = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data_sha256": sha256_bytes(value.tobytes()),
    }
    if path is not None:
        descriptor["path"] = path
    return descriptor


def save_array(root: Path, name: str, array: Any) -> dict[str, Any]:
    """Save an exact .npy artifact and return its hash/shape descriptor."""

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.ascontiguousarray(np.asarray(array))
    np.save(path, value, allow_pickle=False)
    descriptor = array_descriptor(value, path=str(path))
    descriptor["file_sha256"] = sha256_file(path)
    return descriptor


def model_structural_descriptor(env: Any) -> dict[str, Any]:
    sim = env.sim
    model = getattr(sim, "model", None)
    fields: dict[str, Any] = {}
    for name in ("nq", "nv", "nu", "nbody", "ngeom", "njnt", "nsite", "ncam"):
        try:
            fields[name] = int(getattr(model, name))
        except Exception:
            fields[name] = None
    for name in ("body_pos", "body_quat", "geom_pos", "geom_quat", "jnt_range"):
        try:
            fields[name] = array_descriptor(getattr(model, name))
        except Exception:
            fields[name] = None
    payload = canonical_json(fields)
    return {"fields": fields, "structural_sha256": sha256_bytes(payload)}


def simulator_state_descriptor(env: Any, root: Path, prefix: str) -> dict[str, Any]:
    sim = env.sim
    data = getattr(sim, "data", None)
    descriptor: dict[str, Any] = {}
    try:
        descriptor["flattened"] = save_array(root, f"{prefix}/sim_state_flattened.npy", sim.get_state().flatten())
    except Exception:
        descriptor["flattened"] = None
    for name in ("qpos", "qvel", "qacc", "ctrl", "act", "mocap_pos", "mocap_quat"):
        try:
            descriptor[name] = save_array(root, f"{prefix}/sim_{name}.npy", getattr(data, name))
        except Exception:
            descriptor[name] = None
    return descriptor


def observation_descriptor(obs: dict[str, Any], root: Path, prefix: str) -> dict[str, Any]:
    return {
        str(key): save_array(root, f"{prefix}/obs_{key}.npy", value)
        for key, value in sorted(obs.items())
        if isinstance(value, np.ndarray)
    }


def snapshot_record(
    *,
    env: Any,
    obs: dict[str, Any],
    root: Path,
    case_id: str,
    label: str,
    step: int,
    manifest_row: dict[str, Any],
    policy_seed: int | None = None,
) -> dict[str, Any]:
    prefix = f"snapshots/{case_id}/{label}_{step:04d}"
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "case_id": case_id,
        "label": label,
        "step": int(step),
        "manifest": {
            key: manifest_row.get(key)
            for key in ("task_name", "language", "bddl_sha256", "init_state_index", "init_state_sha256", "env_seed", "policy_seed", "max_steps")
        },
        "policy_seed": policy_seed,
        "runtime": runtime_provenance(),
        "model": model_structural_descriptor(env),
        "simulator": simulator_state_descriptor(env, root, prefix),
        "observation": observation_descriptor(obs, root, prefix),
    }


def write_probe_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
