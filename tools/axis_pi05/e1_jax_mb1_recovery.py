"""Resumable, microbatch-one folded-JAX recovery for the E1 OOM remediation.

This module intentionally has a separate entry point from the original E1
envelope runner.  Its CPU helpers are testable without JAX; the GPU path loads
one folded model per process and writes only validated, contiguous 16-sample
shards.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, canonical_json, sha256_file, write_json
from .e1_action_envelope import (
    E1_INPUTS,
    E1_LOCK,
    E1_MANIFEST,
    E1_ROOT,
    FOLDED_MODEL,
    FOLDED_MODEL_SHA256,
    NORM_STATS,
    NORM_STATS_SHA256,
    RAW_PARAMS_TREE_SHA256,
    RAW_PARAMS,
    _array_hash,
    _fingerprint,
    validate_locked_inputs,
)
from .n1_snapshot import gpu_boundary_evidence, gpu_snapshot


RECOVERY_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_jax_mb1_recovery"
INPUT_AUDIT = RECOVERY_ROOT / "input_audit.json"
CANARY_ROOT = RECOVERY_ROOT / "canary"
FULL_ROOT = RECOVERY_ROOT / "full"
SHARDS_ROOT = FULL_ROOT / "shards"
PRIOR_SENTINEL = E1_ROOT / "jax_source/sentinel/actions.npz"
PRIOR_SENTINEL_SHA256 = "f3977cebecbd186c37ee5f4a5de53e7ce43bcef8141ed93586cd16e335422e65"
OLD_E1_SUMMARY = E1_ROOT / "summary.json"
SHARD_SIZE = 16
FULL_COUNT = 340
INTERNAL_SHAPE = (10, 32)
EXTERNAL_SHAPE = (10, 7)


def _json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _runtime_provenance() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "numpy": np.__version__,
    }


def _model_contract() -> dict[str, Any]:
    config_path = FOLDED_MODEL / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor_paths = sorted(path for pattern in ("policy_preprocessor*", "policy_postprocessor*") for path in FOLDED_MODEL.glob(pattern) if path.is_file())
    return {
        "model_sha256": sha256_file(FOLDED_MODEL / "model.safetensors"),
        "norm_stats_sha256": sha256_file(NORM_STATS),
        "raw_params_tree_sha256": RAW_PARAMS_TREE_SHA256,
        "source_config_sha256": sha256_file(config_path),
        "source_config_contract": {key: config.get(key) for key in ("type", "dtype", "chunk_size", "n_action_steps", "max_action_dim")},
        "processor_hashes": {path.name: sha256_file(path) for path in processor_paths},
    }


def _expected_provenance(*, gpu_id: str | None = None, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "lock_sha256": sha256_file(E1_LOCK),
        "manifest_sha256": sha256_file(E1_MANIFEST),
        "inputs_sha256": sha256_file(E1_INPUTS),
        "model_sha256": FOLDED_MODEL_SHA256,
        "norm_stats_sha256": NORM_STATS_SHA256,
        "raw_params_tree_sha256": RAW_PARAMS_TREE_SHA256,
        "microbatch_size": 1,
        "backend": "r6_folded_jax_source",
    }
    if gpu_id is not None:
        result["gpu_id"] = str(gpu_id)
    if runtime is not None:
        result["runtime"] = runtime
    return result


def _shard_stem(start: int, stop: int) -> str:
    return f"{start:04d}_{stop:04d}"


def _shard_paths(shard_dir: Path, start: int, stop: int) -> tuple[Path, Path]:
    stem = _shard_stem(start, stop)
    return shard_dir / f"{stem}.npz", shard_dir / f"{stem}.json"


def _metadata_without_hash(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "metadata_sha256"}


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_atomic_shard(
    shard_dir: Path,
    start: int,
    stop: int,
    internal_actions: np.ndarray,
    external_actions: np.ndarray,
    provenance: dict[str, Any],
    *,
    gpu_monitor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write npz then metadata with two atomic renames; no partial pair is valid."""
    if stop <= start or stop - start > SHARD_SIZE and stop != FULL_COUNT:
        raise ValueError("invalid shard interval")
    internal = np.ascontiguousarray(np.asarray(internal_actions, dtype=np.float32))
    external = np.ascontiguousarray(np.asarray(external_actions, dtype=np.float32))
    if internal.shape != (stop - start, *INTERNAL_SHAPE) or external.shape != (stop - start, *EXTERNAL_SHAPE):
        raise ValueError("invalid shard action shape")
    if not np.isfinite(internal).all() or not np.isfinite(external).all():
        raise ValueError("non-finite shard action")
    shard_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = _shard_paths(shard_dir, start, stop)
    fd, temporary_npz = tempfile.mkstemp(prefix=f".{npz_path.name}.", suffix=".tmp", dir=str(shard_dir))
    os.close(fd)
    try:
        # Pass an open file object: numpy otherwise appends '.npz' to a temp
        # filename and the subsequent rename could publish an empty file.
        with open(temporary_npz, "wb") as handle:
            np.savez_compressed(handle, internal_actions=internal, external_actions=external)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_npz, npz_path)
    except BaseException:
        try:
            os.unlink(temporary_npz)
        except FileNotFoundError:
            pass
        raise
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "start": start,
        "stop": stop,
        "observation_count": stop - start,
        "internal": {"shape": list(internal.shape), "dtype": str(internal.dtype), "sha256": _array_hash(internal)},
        "external": {"shape": list(external.shape), "dtype": str(external.dtype), "sha256": _array_hash(external)},
        "npz_sha256": sha256_file(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "finite": True,
        "provenance": provenance,
        "gpu_monitor": gpu_monitor,
    }
    metadata["metadata_sha256"] = _json_hash(metadata)
    _atomic_write_bytes(json_path, canonical_json(metadata))
    return metadata


def validate_shard(shard_dir: Path, start: int, stop: int, expected_provenance: dict[str, Any]) -> dict[str, Any]:
    npz_path, json_path = _shard_paths(shard_dir, start, stop)
    if npz_path.is_file() != json_path.is_file():
        raise ValueError(f"partial shard: {start}_{stop}")
    if not npz_path.is_file():
        raise ValueError(f"missing shard: {start}_{stop}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("metadata_sha256") != _json_hash(_metadata_without_hash(metadata)):
        raise ValueError(f"shard metadata hash mismatch: {start}_{stop}")
    if metadata.get("start") != start or metadata.get("stop") != stop or metadata.get("observation_count") != stop - start:
        raise ValueError(f"shard interval/order mismatch: {start}_{stop}")
    if metadata.get("provenance") != expected_provenance:
        raise ValueError(f"shard provenance drift: {start}_{stop}")
    if metadata.get("npz_sha256") != sha256_file(npz_path):
        raise ValueError(f"shard file hash mismatch: {start}_{stop}")
    with np.load(npz_path, allow_pickle=False) as file:
        if set(file.files) != {"internal_actions", "external_actions"}:
            raise ValueError(f"shard schema mismatch: {start}_{stop}")
        internal = np.asarray(file["internal_actions"])
        external = np.asarray(file["external_actions"])
    if internal.shape != (stop - start, *INTERNAL_SHAPE) or external.shape != (stop - start, *EXTERNAL_SHAPE):
        raise ValueError(f"shard shape mismatch: {start}_{stop}")
    if internal.dtype != np.dtype("float32") or external.dtype != np.dtype("float32"):
        raise ValueError(f"shard dtype mismatch: {start}_{stop}")
    if not np.isfinite(internal).all() or not np.isfinite(external).all():
        raise ValueError(f"shard non-finite: {start}_{stop}")
    if metadata["internal"]["sha256"] != _array_hash(internal) or metadata["external"]["sha256"] != _array_hash(external):
        raise ValueError(f"shard action hash mismatch: {start}_{stop}")
    return {"start": start, "stop": stop, "npz": npz_path, "json": json_path, "metadata": metadata, "internal": internal, "external": external}


def _expected_intervals(expected_stop: int) -> list[tuple[int, int]]:
    if expected_stop < 1:
        raise ValueError("expected stop must be positive")
    intervals = []
    for start in range(0, expected_stop, SHARD_SIZE):
        intervals.append((start, min(start + SHARD_SIZE, expected_stop)))
    return intervals


def validate_contiguous_shards(shard_dir: Path, expected_stop: int, expected_provenance: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Return the next index only for a complete prefix; reject gaps and debris."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    intervals = _expected_intervals(expected_stop)
    known_stems = {_shard_stem(start, stop) for start, stop in intervals}
    allowed_names = {f"{stem}{suffix}" for stem in known_stems for suffix in (".npz", ".json")}
    for path in shard_dir.iterdir():
        if path.is_file() and path.name not in allowed_names:
            raise ValueError(f"unexpected shard file: {path.name}")
    prefix: list[dict[str, Any]] = []
    missing_at: int | None = None
    for index, (start, stop) in enumerate(intervals):
        npz_path, json_path = _shard_paths(shard_dir, start, stop)
        if not npz_path.exists() and not json_path.exists():
            missing_at = index
            break
        if missing_at is not None:
            raise ValueError("shard gap")
        prefix.append(validate_shard(shard_dir, start, stop, expected_provenance))
    if missing_at is not None:
        for start, stop in intervals[missing_at + 1 :]:
            npz_path, json_path = _shard_paths(shard_dir, start, stop)
            if npz_path.exists() or json_path.exists():
                raise ValueError("isolated shard after first missing shard")
    return (intervals[missing_at][0] if missing_at is not None else expected_stop), prefix


def aggregate_shards(shard_dir: Path, expected_stop: int, expected_provenance: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    next_start, shards = validate_contiguous_shards(shard_dir, expected_stop, expected_provenance)
    if next_start != expected_stop:
        raise ValueError(f"incomplete shard coverage: {next_start}/{expected_stop}")
    internal = np.concatenate([item["internal"] for item in shards], axis=0)
    external = np.concatenate([item["external"] for item in shards], axis=0)
    if internal.shape != (expected_stop, *INTERNAL_SHAPE) or external.shape != (expected_stop, *EXTERNAL_SHAPE):
        raise ValueError("aggregate shape/coverage mismatch")
    return internal, external, shards


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_input_audit(output: Path = INPUT_AUDIT) -> dict[str, Any]:
    """R0/R1 CPU-only audit; never imports a model backend."""
    validated = validate_locked_inputs(E1_LOCK, E1_MANIFEST, E1_INPUTS)
    if not PRIOR_SENTINEL.is_file() or sha256_file(PRIOR_SENTINEL) != PRIOR_SENTINEL_SHA256:
        raise ValueError("e1_jax_mb1_recovery_input_drifted: prior sentinel")
    old_summary = json.loads(OLD_E1_SUMMARY.read_text(encoding="utf-8"))
    if old_summary.get("status") != "e1_jax_full_inference_failed":
        raise ValueError("e1_jax_mb1_recovery_input_drifted: old OOM summary status")
    audit = {
        "schema_version": 1,
        "phase": "E1_JAX_MB1_RECOVERY",
        "status": "e1_jax_mb1_recovery_inputs_locked",
        "actions_observed": False,
        "lock_sha256": validated["lock_sha256"],
        "manifest_sha256": validated["manifest_sha256"],
        "inputs_sha256": validated["inputs_sha256"],
        "prior_sentinel": str(PRIOR_SENTINEL.resolve()),
        "prior_sentinel_sha256": sha256_file(PRIOR_SENTINEL),
        "old_e1_oom_summary": str(OLD_E1_SUMMARY.resolve()),
        "old_e1_oom_summary_sha256": sha256_file(OLD_E1_SUMMARY),
        "old_e1_oom_status": old_summary["status"],
        "pair_count": 170,
        "observation_count": 340,
        "snapshot_pair_count": 13,
        "trajectory_pair_count": 157,
        "runtime_contract": {"backend": "r6_folded_jax_source", "microbatch_size": 1, "shard_size": 16, "internal_shape": [10, 32], "external_shape": [10, 7], "dtype": "float32"},
        "model_contract": _model_contract(),
        "gpu0_touched": False,
        "gpu_execution_started": False,
    }
    write_json(output, audit)
    return audit


def _load_runtime_model(gpu_id: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PALIGEMMA_TOKENIZER_DIR", str(ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging/vendor/paligemma_tokenizer"))
    sys.path.insert(0, str(ROOT / "third_party/axis_openpi/src"))
    sys.path.insert(1, str(ROOT))
    import copy
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from tools.axis_pi05.lora_folding import fold_lora_in_mapping

    raw_params = model_lib.restore_params(RAW_PARAMS, restore_type=np.ndarray, dtype="float32")
    folded_params = copy.deepcopy(raw_params)
    folded_modules = fold_lora_in_mapping(folded_params)
    config = pi0_config.Pi0Config(pi05=True, dtype="float32", action_dim=32, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    template = config.create(jax.random.key(0))
    graphdef, state = nnx.split(template)
    state.replace_by_pure_dict(folded_params)
    model = nnx.merge(graphdef, state)
    model.eval()
    return jax, jnp, model_lib, model, folded_modules, raw_params, folded_params, template


def _run_gpu(mode: str, output: Path, gpu_id: str, *, resume: bool = False, seed: int = 0, microbatch_size: int = 1) -> dict[str, Any]:
    if mode not in {"canary", "full"}:
        raise ValueError(mode)
    if microbatch_size != 1:
        raise ValueError("E1 JAX recovery is fixed to microbatch_size=1")
    validated = validate_locked_inputs(E1_LOCK, E1_MANIFEST, E1_INPUTS)
    audit = json.loads(INPUT_AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "e1_jax_mb1_recovery_inputs_locked" or audit.get("lock_sha256") != validated["lock_sha256"] or audit.get("prior_sentinel_sha256") != PRIOR_SENTINEL_SHA256:
        raise ValueError("e1_jax_mb1_recovery_input_drifted: input audit")
    if mode == "full" and not resume and SHARDS_ROOT.is_dir() and any(SHARDS_ROOT.iterdir()):
        raise ValueError("full recovery has existing shards; pass --resume")
    if mode == "full" and not json.loads((CANARY_ROOT / "report.json").read_text(encoding="utf-8")).get("canary_passed", False):
        raise ValueError("canary has not passed")
    if gpu_id == "0":
        raise ValueError("GPU0 is outside recovery boundary")
    output.mkdir(parents=True, exist_ok=True)
    monitor = [gpu_snapshot()]
    runtime = _runtime_provenance()
    # Runtime identity (PID/timestamps) is recorded in each shard but is not
    # part of the immutable cross-process provenance used for resume.
    provenance = _expected_provenance(gpu_id=gpu_id)
    start_index = 0
    prior_shards: list[dict[str, Any]] = []
    if mode == "full":
        start_index, prior_shards = validate_contiguous_shards(SHARDS_ROOT, FULL_COUNT, provenance)
    arrays = validated["arrays"]
    stop_index = 16 if mode == "canary" else FULL_COUNT
    if start_index > stop_index:
        raise ValueError("resume boundary exceeds requested mode")
    jax, jnp, model_lib, model, folded_modules, raw_params, folded_params, template = _load_runtime_model(gpu_id)
    monitor.append(gpu_snapshot())
    latency: list[float] = []
    written: list[dict[str, Any]] = []
    for shard_start in range(start_index, stop_index, SHARD_SIZE):
        shard_stop = min(shard_start + SHARD_SIZE, stop_index)
        internal_values: list[np.ndarray] = []
        external_values: list[np.ndarray] = []
        for index in range(shard_start, shard_stop):
            observation = model_lib.Observation.from_dict({
                "image": {key: jnp.asarray(arrays[key][index:index + 1]) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
                "image_mask": {"base_0_rgb": jnp.ones((1,), dtype=bool), "left_wrist_0_rgb": jnp.ones((1,), dtype=bool), "right_wrist_0_rgb": jnp.zeros((1,), dtype=bool)},
                "state": jnp.asarray(arrays["state"][index:index + 1]), "tokenized_prompt": jnp.asarray(arrays["tokenized_prompt"][index:index + 1]), "tokenized_prompt_mask": jnp.asarray(arrays["tokenized_prompt_mask"][index:index + 1]),
            })
            started = time.perf_counter()
            value = np.asarray(jax.block_until_ready(model.sample_actions(jax.random.key(seed), observation, num_steps=10, noise=jnp.asarray(arrays["noise"][index:index + 1]))), dtype=np.float32)
            latency.append(time.perf_counter() - started)
            internal_values.append(value[0])
            q = json.loads(NORM_STATS.read_text(encoding="utf-8"))["norm_stats"]["actions"]
            q01, q99 = np.asarray(q["q01"], dtype=np.float32), np.asarray(q["q99"], dtype=np.float32)
            external_values.append(((value[0, ..., :7] + 1.0) * (q99 - q01) / 2.0 + q01).astype(np.float32))
        shard_internal = np.stack(internal_values)
        shard_external = np.stack(external_values)
        monitor.append(gpu_snapshot())
        written.append(write_atomic_shard(SHARDS_ROOT, shard_start, shard_stop, shard_internal, shard_external, provenance, gpu_monitor={"runtime": runtime, "snapshot": monitor[-1]}))
    del model, template, raw_params, folded_params
    gc.collect()
    monitor.append(gpu_snapshot())
    evidence = gpu_boundary_evidence(monitor, target_pids={os.getpid()}, allowed_gpu_ids={int(gpu_id)})
    all_internal = all_external = None
    all_shards = prior_shards
    if mode == "canary":
        all_internal = np.concatenate([written[0]["internal"]], axis=0) if written else np.empty((0, *INTERNAL_SHAPE), dtype=np.float32)
        all_external = np.concatenate([written[0]["external"]], axis=0) if written else np.empty((0, *EXTERNAL_SHAPE), dtype=np.float32)
        sentinel_match = False
        with np.load(PRIOR_SENTINEL, allow_pickle=False) as prior:
            prior_internal = np.asarray(prior["internal_actions"])
            prior_external = np.asarray(prior["external_actions"])
        sentinel_match = bool(np.array_equal(all_internal[:4], prior_internal[:4]) and np.array_equal(all_external[:4], prior_external[:4]))
        _atomic_save_npz(output / "actions.npz", internal_actions=all_internal, external_actions=all_external)
        report = {"schema_version": 1, "phase": "E1_JAX_MB1_RECOVERY", "mode": "canary", "status": "passed" if sentinel_match and evidence["gpu0_touched"] is False else "e1_jax_mb1_canary_failed", "canary_passed": bool(sentinel_match and evidence["gpu0_touched"] is False), "microbatch_size": 1, "observation_count": 16, "internal": _fingerprint(all_internal), "external": _fingerprint(all_external), "first_four_internal_bitwise_exact": sentinel_match, "first_four_external_bitwise_exact": sentinel_match, "latency_seconds": latency, "folded_lora_modules": folded_modules, "model_contract": _model_contract(), "provenance": provenance, "gpu_monitor_evidence": {"samples": monitor, **evidence}, "gpu0_touched": evidence["gpu0_touched"], "artifact": str((output / "actions.npz").resolve())}
    else:
        all_internal, all_external, all_shards = aggregate_shards(SHARDS_ROOT, FULL_COUNT, provenance)
        if not np.array_equal(all_internal[:4], np.load(PRIOR_SENTINEL, allow_pickle=False)["internal_actions"][:4]) or not np.array_equal(all_external[:4], np.load(PRIOR_SENTINEL, allow_pickle=False)["external_actions"][:4]):
            raise ValueError("first four actions differ from prior sentinel")
        _atomic_save_npz(output / "actions.npz", internal_actions=all_internal, external_actions=all_external)
        report = {"schema_version": 1, "phase": "E1_JAX_MB1_RECOVERY", "mode": "full", "status": "e1_jax_full_recovered_pending_review" if evidence["gpu0_touched"] is False else "e1_jax_mb1_full_failed", "microbatch_size": 1, "observation_count": FULL_COUNT, "coverage": {"expected": FULL_COUNT, "observed": int(all_internal.shape[0]), "continuous": True, "shard_count": len(all_shards)}, "internal": _fingerprint(all_internal), "external": _fingerprint(all_external), "aggregate_bitwise_from_shards": True, "first_four_internal_bitwise_exact": True, "first_four_external_bitwise_exact": True, "latency_seconds": latency, "resume": {"requested": resume, "resume_start": start_index, "resume_count": 1 if resume and start_index else 0, "shards_preexisting": len(prior_shards), "shards_written_this_run": len(written)}, "folded_lora_modules": folded_modules, "model_contract": _model_contract(), "provenance": provenance, "gpu_monitor_evidence": {"samples": monitor, **evidence}, "gpu0_touched": evidence["gpu0_touched"], "artifact": str((output / "actions.npz").resolve()), "shards": [{"start": item["start"], "stop": item["stop"], "npz_sha256": item["metadata"]["npz_sha256"], "json_sha256": sha256_file(item["json"])} for item in all_shards]}
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("input-audit")
    audit.add_argument("--output", type=Path, default=INPUT_AUDIT)
    shard = sub.add_parser("backend")
    shard.add_argument("--mode", choices=("canary", "full"), required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--gpu-id", default="1")
    shard.add_argument("--microbatch-size", type=int, default=1)
    shard.add_argument("--resume", action="store_true")
    shard.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "input-audit":
        print(json.dumps(write_input_audit(args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_run_gpu(args.mode, args.output, args.gpu_id, resume=args.resume, seed=args.seed, microbatch_size=args.microbatch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
