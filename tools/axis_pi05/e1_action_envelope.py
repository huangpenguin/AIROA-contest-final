"""E1 frozen-observation EGL action-response envelope.

The CPU part materializes the already-approved E0 content-unique pairs before
any model/backend import.  The GPU part has deliberately separate JAX and
LeRobot/PyTorch entry points so that the two source backends can be run in
fresh processes on one physical GPU.  No environment or closed-loop code is
used here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .checkpoint_http_server import _resize_for_openpi
from .common import ROOT, canonical_json, sha256_file, sha256_tree, write_json, write_jsonl
from .n1_snapshot import gpu_boundary_evidence, gpu_snapshot


E0_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e0_closure"
E0_MANIFEST = E0_ROOT / "content_unique_pair_manifest.json"
E1_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release/e1_action_envelope"
E1_LOCK = E1_ROOT / "e1_input_lock.json"
E1_MANIFEST = E1_ROOT / "e1_frozen_pair_manifest.json"
E1_INPUTS = E1_ROOT / "e1_frozen_inputs_and_noise.npz"
E1_SUMMARY = E1_ROOT / "summary.json"
E1_PAIR_METRICS = E1_ROOT / "pair_metrics.jsonl"
E1_JAX = E1_ROOT / "jax_source"
E1_PYTORCH = E1_ROOT / "lerobot_pytorch_source"
Q1_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q1_environment_full_c20"
C20_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
NORM_STATS = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
TOKENIZER_MODEL = ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging/vendor/paligemma_tokenizer/tokenizer.model"
FOLDED_MODEL = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500"
RAW_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
AXIS_PARAMS = ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1/params"
RUN_DIR = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1"
LEROBOT_SRC = Path(os.environ.get("AXIS_LEROBOT_SRC", str(ROOT.parent / "AIROA-contest-libero-plus" / "workspace" / "training" / "lerobot" / "src")))
NESTED_SRC = ROOT / "third_party" / "axis_openpi" / "src"

E0_MANIFEST_SHA256 = "b4ea8d3bed7f9cfc60f2f7f53613471e222d9857a8d6857f39b854720905c7bf"
E0_SUMMARY_SHA256 = "088376936c2ecba61258316bcafe48c70c04c7c115c62917222f7add2aba3c43"
FOLDED_MODEL_SHA256 = "5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b"
NORM_STATS_SHA256 = "4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5"
RAW_PARAMS_TREE_SHA256 = "d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f"
GLOBAL_NOISE_SEED = 20260831
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_TO_INPUT = {"agentview_image": "base_0_rgb", "robot0_eye_in_hand_image": "left_wrist_0_rgb"}
REPLICAS = ("replica0", "replica1", "replica2")
SNAPSHOT_PHASES = ("E1", "E2", *(f"E3-{i}" for i in range(1, 11)))


def _array_hash(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": _array_hash(array)}


def _bundle_hash(fields: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in fields:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_descriptor(descriptor: dict[str, Any], *, frame_index: int | None = None) -> np.ndarray:
    path = Path(str(descriptor.get("path", "")))
    if not path.is_file():
        raise ValueError(f"missing source array: {path}")
    value = np.asarray(np.load(path, allow_pickle=False))
    expected = descriptor.get("data_sha256")
    if frame_index is not None:
        if value.ndim == 0 or frame_index < 0 or frame_index >= value.shape[0]:
            raise ValueError(f"frame out of range: {path} frame={frame_index}")
        value = np.asarray(value[frame_index])
    if expected and frame_index is None and _array_hash(value) != expected:
        raise ValueError(f"source array data hash mismatch: {path}")
    return value


def _load_q1_cases() -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for replica in REPLICAS:
        path = Q1_ROOT / replica / "summary.json"
        if not path.is_file():
            raise ValueError(f"missing Q1 replica summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[replica] = {str(case["case_id"]): case for case in payload["records"][0]["cases"]}
    return result


def _snapshot_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in case.get("snapshots", []):
        stage = str(item.get("stage"))
        if stage in {"E1", "E2"}:
            result[stage] = item
        elif stage == "E3" and int(item.get("warmup_step", -1)) in range(1, 11):
            result[f"E3-{int(item['warmup_step'])}"] = item
    return result


def _trajectory_policy_state(arrays: dict[str, np.ndarray], frame: int) -> np.ndarray:
    pos = np.asarray(arrays["obs:robot0_eef_pos"][frame], dtype=np.float32).reshape(3)
    quat = np.asarray(arrays["obs:robot0_eef_quat"][frame], dtype=np.float32).reshape(4)
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - scalar * scalar)))
    axisangle = np.zeros(3, dtype=np.float32) if np.isclose(denominator, 0.0) else (quat[:3] * (2.0 * np.arccos(scalar)) / denominator).astype(np.float32)
    gripper = np.asarray(arrays["obs:robot0_gripper_qpos"][frame], dtype=np.float32).reshape(2)
    state = np.concatenate((pos, axisangle, gripper)).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("invalid trajectory policy state")
    return state


def _deployment_tokens(prompt: str) -> tuple[np.ndarray, np.ndarray]:
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = tokenizer.encode(cleaned, add_bos=True) + tokenizer.encode("\n")
    tokens = tokens[:200]
    padded = np.zeros((200,), dtype=np.int32)
    padded[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
    mask = np.zeros((200,), dtype=np.bool_)
    mask[: len(tokens)] = True
    return padded, mask


def _source_occurrence_hash(occurrence: dict[str, Any]) -> str:
    payload = {key: value for key, value in occurrence.items() if key != "source_occurrence_hash"}
    return _sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _noise_for_pair(content_hash: str, global_seed: int) -> tuple[int, np.ndarray]:
    material = f"axis-pi05-e1-noise-v1|{global_seed}|{content_hash}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)
    return seed, np.random.default_rng(seed).standard_normal((10, 32), dtype=np.float32)


def _prompt_map() -> dict[str, str]:
    rows = [json.loads(line) for line in C20_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["case_id"]): str(row["language"]) for row in rows}


def _source_pair_values(pair: dict[str, Any], q1: dict[str, dict[str, dict[str, Any]]], prompts: dict[str, str]) -> dict[str, Any]:
    occurrences = pair.get("source_occurrences")
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError(f"pair has no source occurrence: {pair.get('content_hash')}")
    occurrence = occurrences[0]
    if _source_occurrence_hash(occurrence) != occurrence.get("source_occurrence_hash"):
        raise ValueError("source occurrence hash mismatch")
    case_id = str(pair["case_id"])
    prompt = prompts.get(case_id)
    if prompt is None or _sha256_text(prompt) != pair["prompt_hash"]:
        raise ValueError(f"prompt hash mismatch: {case_id}")
    phase = str(pair["phase_or_trajectory_frame"])
    camera = str(pair["camera"])
    left_replica = str(occurrence["left_replica"])
    right_replica = str(occurrence["right_replica"])
    if left_replica not in q1 or right_replica not in q1:
        raise ValueError("unknown Q1 replica")
    left_case, right_case = q1[left_replica][case_id], q1[right_replica][case_id]

    if pair["source_kind"] == "snapshot":
        left_snapshot = _snapshot_map(left_case)[phase]
        right_snapshot = _snapshot_map(right_case)[phase]
        left_obs, right_obs = left_snapshot["observation"], right_snapshot["observation"]
        left_images = {name: _load_descriptor(left_obs[name]) for name in CAMERAS}
        right_images = {name: _load_descriptor(right_obs[name]) for name in CAMERAS}
        left_state = _load_descriptor(left_snapshot["policy_state"])
        right_state = _load_descriptor(right_snapshot["policy_state"])
        source_frame = None
        left_paths = {name: str(left_obs[name]["path"]) for name in CAMERAS}
        right_paths = {name: str(right_obs[name]["path"]) for name in CAMERAS}
        state_paths = {"left": str(left_snapshot["policy_state"]["path"]), "right": str(right_snapshot["policy_state"]["path"])}
    elif pair["source_kind"] == "trajectory":
        mode, frame_text = phase.split("/frame-", 1)
        source_frame = int(frame_text)
        left_desc = left_case["trajectories"][mode]["arrays"]
        right_desc = right_case["trajectories"][mode]["arrays"]
        left_full = {key: _load_descriptor(value) for key, value in left_desc.items() if key in {f"obs:{name}" for name in CAMERAS} | {"obs:robot0_eef_pos", "obs:robot0_eef_quat", "obs:robot0_gripper_qpos"}}
        right_full = {key: _load_descriptor(value) for key, value in right_desc.items() if key in {f"obs:{name}" for name in CAMERAS} | {"obs:robot0_eef_pos", "obs:robot0_eef_quat", "obs:robot0_gripper_qpos"}}
        left_images = {name: np.asarray(left_full[f"obs:{name}"][source_frame]) for name in CAMERAS}
        right_images = {name: np.asarray(right_full[f"obs:{name}"][source_frame]) for name in CAMERAS}
        left_state = _trajectory_policy_state(left_full, source_frame)
        right_state = _trajectory_policy_state(right_full, source_frame)
        left_paths = {name: str(left_desc[f"obs:{name}"]["path"]) for name in CAMERAS}
        right_paths = {name: str(right_desc[f"obs:{name}"]["path"]) for name in CAMERAS}
        state_paths = {"left": f"{left_replica}:{case_id}:{mode}:frame-{source_frame:04d}:derived_policy_state", "right": f"{right_replica}:{case_id}:{mode}:frame-{source_frame:04d}:derived_policy_state"}
    else:
        raise ValueError(f"unexpected source kind: {pair['source_kind']}")

    if not np.array_equal(left_state, right_state) or _array_hash(left_state) != pair["policy_state_hash"]:
        raise ValueError(f"policy state mismatch: {case_id} {phase}")
    selected_left_hash = _array_hash(left_images[camera])
    selected_right_hash = _array_hash(right_images[camera])
    if selected_left_hash != pair["left_image_data_sha256"] or selected_right_hash != pair["right_image_data_sha256"]:
        raise ValueError(f"selected image hash mismatch: {case_id} {phase} {camera}")
    other_camera = next(name for name in CAMERAS if name != camera)
    if not np.array_equal(left_images[other_camera], right_images[other_camera]):
        raise ValueError(f"other-camera mismatch: {case_id} {phase} {other_camera}")
    return {
        "prompt": prompt,
        "source_frame": source_frame,
        "camera": camera,
        "other_camera": other_camera,
        "images": {"left": left_images, "right": right_images},
        "states": {"left": np.asarray(left_state, dtype=np.float32), "right": np.asarray(right_state, dtype=np.float32)},
        "image_paths": {"left": left_paths, "right": right_paths},
        "state_paths": state_paths,
        "source_occurrence": occurrence,
    }


def _normalized_state(raw: np.ndarray) -> np.ndarray:
    norm = json.loads(NORM_STATS.read_text(encoding="utf-8"))["norm_stats"]["state"]
    q01, q99 = np.asarray(norm["q01"], dtype=np.float32), np.asarray(norm["q99"], dtype=np.float32)
    return ((np.asarray(raw, dtype=np.float32) - q01) / (q99 - q01 + np.float32(1e-6)) * np.float32(2.0) - np.float32(1.0)).astype(np.float32)


def _transformed_image(raw: np.ndarray) -> np.ndarray:
    value = _resize_for_openpi(np.asarray(raw))
    return (value.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)).astype(np.float32)


def _validate_e0_manifest() -> dict[str, Any]:
    if not E0_MANIFEST.is_file():
        raise FileNotFoundError(E0_MANIFEST)
    if sha256_file(E0_MANIFEST) != E0_MANIFEST_SHA256:
        raise ValueError("e1_input_or_runtime_drifted: E0 manifest hash")
    payload = json.loads(E0_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("pair_count") != 170 or len(payload.get("pairs", [])) != 170:
        raise ValueError("E0 manifest is not the approved 170-pair input")
    if sha256_file(E0_ROOT / "summary.json") != E0_SUMMARY_SHA256:
        raise ValueError("e1_input_or_runtime_drifted: E0 summary hash")
    if sum(item.get("source_kind") == "snapshot" for item in payload["pairs"]) != 13 or sum(item.get("source_kind") == "trajectory" for item in payload["pairs"]) != 157:
        raise ValueError("E0 source-kind coverage drifted")
    return payload


def build_lock(output_dir: Path = E1_ROOT, *, global_seed: int = GLOBAL_NOISE_SEED) -> dict[str, Any]:
    """CPU-only E1.1 materialization; must run before backend loading."""
    e0 = _validate_e0_manifest()
    q1 = _load_q1_cases()
    prompts = _prompt_map()
    norm = json.loads(NORM_STATS.read_text(encoding="utf-8"))
    if sha256_file(NORM_STATS) != NORM_STATS_SHA256:
        raise ValueError("e1_input_or_runtime_drifted: norm stats hash")
    output_dir.mkdir(parents=True, exist_ok=True)
    base: dict[str, list[np.ndarray]] = {key: [] for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask", "noise")}
    tasks: list[str] = []
    pairs: list[dict[str, Any]] = []
    occurrence_hash_mismatches = 0
    semantic_keys: set[str] = set()
    for index, pair in enumerate(e0["pairs"]):
        source = _source_pair_values(pair, q1, prompts)
        semantic_key_json = json.dumps(pair["semantic_key"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if semantic_key_json in semantic_keys:
            raise ValueError("E0 semantic-key duplicate")
        semantic_keys.add(semantic_key_json)
        occurrence_hashes = [str(item.get("source_occurrence_hash")) for item in pair["source_occurrences"]]
        occurrence_hash_mismatches += sum(_source_occurrence_hash(item) != item.get("source_occurrence_hash") for item in pair["source_occurrences"])
        token, token_mask = _deployment_tokens(source["prompt"])
        noise_seed, noise = _noise_for_pair(str(pair["content_hash"]), global_seed)
        normalized_states = {side: _normalized_state(source["states"][side]) for side in ("left", "right")}
        if not np.array_equal(normalized_states["left"], normalized_states["right"]):
            raise ValueError("normalized state mismatch")
        side_records = []
        for side in ("left", "right"):
            images = source["images"][side]
            base_image = _transformed_image(images["agentview_image"])
            wrist_image = _transformed_image(images["robot0_eye_in_hand_image"])
            empty_image = np.full((224, 224, 3), -1.0, dtype=np.float32)
            state32 = np.pad(normalized_states[side], (0, 24)).astype(np.float32)
            for key, value in (("base_0_rgb", base_image), ("left_wrist_0_rgb", wrist_image), ("right_wrist_0_rgb", empty_image), ("state", state32), ("tokenized_prompt", token), ("tokenized_prompt_mask", token_mask), ("noise", noise)):
                base[key].append(value)
            observation_index = len(base["state"]) - 1
            tasks.append(source["prompt"])
            side_records.append({
                "side": side,
                "observation_index": observation_index,
                "replica": source["source_occurrence"][f"{side}_replica"],
                "image_paths": source["image_paths"][side],
                "image_data_sha256": {name: _array_hash(value) for name, value in images.items()},
                "transformed_input_fingerprints": {"base_0_rgb": _fingerprint(base_image), "left_wrist_0_rgb": _fingerprint(wrist_image), "right_wrist_0_rgb": _fingerprint(empty_image)},
                "state_raw_8d": np.asarray(source["states"][side], dtype=np.float32).tolist(),
                "state_raw_8d_sha256": _array_hash(source["states"][side]),
                "state_normalized_32d": _fingerprint(state32),
                "tokenized_prompt": _fingerprint(token),
                "tokenized_prompt_mask": _fingerprint(token_mask),
                "policy_input_sha256": _bundle_hash((("base_0_rgb", base_image), ("left_wrist_0_rgb", wrist_image), ("right_wrist_0_rgb", empty_image), ("state", state32), ("tokenized_prompt", token), ("tokenized_prompt_mask", token_mask))),
            })
        pairs.append({
            "pair_id": f"e1-egl-{index:03d}",
            "pair_index": index,
            "observation_indices": [side_records[0]["observation_index"], side_records[1]["observation_index"]],
            "content_hash": pair["content_hash"],
            "semantic_key": pair["semantic_key"],
            "case_id": pair["case_id"],
            "instruction": source["prompt"],
            "prompt_hash": pair["prompt_hash"],
            "source_kind": pair["source_kind"],
            "phase_or_trajectory_frame": pair["phase_or_trajectory_frame"],
            "frame_index": pair.get("frame_index"),
            "camera": pair["camera"],
            "source_occurrences": pair["source_occurrences"],
            "source_occurrence_hashes": occurrence_hashes,
            "source_comparison": source["source_occurrence"]["source_comparison"],
            "selected_camera": {"name": pair["camera"], "input_key": CAMERA_TO_INPUT[pair["camera"]], "left_data_sha256": pair["left_image_data_sha256"], "right_data_sha256": pair["right_image_data_sha256"]},
            "other_camera": {"name": source["other_camera"], "exact": True, "left_data_sha256": _array_hash(source["images"]["left"][source["other_camera"]]), "right_data_sha256": _array_hash(source["images"]["right"][source["other_camera"]]), "left_path": source["image_paths"]["left"][source["other_camera"]], "right_path": source["image_paths"]["right"][source["other_camera"]]},
            "policy_state": {"raw_8d_sha256": pair["policy_state_hash"], "left_path_or_reference": source["state_paths"]["left"], "right_path_or_reference": source["state_paths"]["right"], "left_equals_right": True, "values": side_records[0]["state_raw_8d"]},
            "sides": side_records,
            "noise": {"algorithm": "sha256('axis-pi05-e1-noise-v1|global_seed|content_hash')[:8] little-endian -> default_rng.standard_normal", "global_seed": global_seed, "seed": noise_seed, "shape": [10, 32], "dtype": "float32", "sha256": _array_hash(noise), "left_right_exact": True},
        })
    arrays = {key: np.stack(value, axis=0) for key, value in base.items()}
    arrays["pair_index"] = np.repeat(np.arange(len(pairs), dtype=np.int32), 2)
    arrays["side_index"] = np.tile(np.asarray([0, 1], dtype=np.int32), len(pairs))
    np.savez_compressed(output_dir / E1_INPUTS.name, **arrays)
    input_path = output_dir / E1_INPUTS.name
    manifest = {
        "schema_version": 1,
        "phase": "E1",
        "artifact": "frozen_runnable_egl_inputs",
        "actions_observed": False,
        "pair_count": len(pairs),
        "snapshot_pair_count": sum(item["source_kind"] == "snapshot" for item in pairs),
        "trajectory_pair_count": sum(item["source_kind"] == "trajectory" for item in pairs),
        "observation_count": len(pairs) * 2,
        "semantic_key_duplicate_count": len(pairs) - len(semantic_keys),
        "source_occurrence_hash_mismatch_count": occurrence_hash_mismatches,
        "e0_manifest": str(E0_MANIFEST.resolve()),
        "e0_manifest_sha256": sha256_file(E0_MANIFEST),
        "e0_summary_sha256": sha256_file(E0_ROOT / "summary.json"),
        "normalization": "QUANTILES",
        "state_contract": "raw 8D -> QUANTILES normalized 8D -> zero-pad 32D",
        "action_contract": "internal 32D, external first 7D, chunk 10, execute 8",
        "pairs": pairs,
        "tasks_sha256": _array_hash(np.asarray(tasks, dtype=f"U{max(len(value) for value in tasks)}")),
    }
    manifest_path = output_dir / E1_MANIFEST.name
    write_json(manifest_path, manifest)
    lock = {
        "schema_version": 1,
        "phase": "E1.1",
        "status": "e1_inputs_locked_before_action",
        "actions_observed": False,
        "lock_commit_required": True,
        "pair_count": len(pairs),
        "observation_count": len(pairs) * 2,
        "snapshot_pair_count": 13,
        "trajectory_pair_count": 157,
        "semantic_key_duplicate_count": 0,
        "source_occurrence_hash_mismatch_count": occurrence_hash_mismatches,
        "other_camera_mismatch_count": 0,
        "not_evaluable_count": 0,
        "prompt_state_image_hash_mismatch_count": 0,
        "global_noise_seed": global_seed,
        "e0_manifest_sha256": sha256_file(E0_MANIFEST),
        "e0_summary_sha256": sha256_file(E0_ROOT / "summary.json"),
        "norm_stats_sha256": sha256_file(NORM_STATS),
        "e1_frozen_pair_manifest": str(manifest_path.resolve()),
        "e1_frozen_pair_manifest_sha256": sha256_file(manifest_path),
        "e1_frozen_inputs_and_noise": str(input_path.resolve()),
        "e1_frozen_inputs_and_noise_sha256": sha256_file(input_path),
        "e1_frozen_inputs_and_noise_bytes": input_path.stat().st_size,
        "input_schema": {key: _fingerprint(value) for key, value in arrays.items()},
        "runnable_input_order": "E0 pairs order; each pair left then right",
        "backend_load_or_inference_started": False,
    }
    write_json(output_dir / E1_LOCK.name, lock)
    return lock


def validate_locked_inputs(lock_path: Path = E1_LOCK, manifest_path: Path = E1_MANIFEST, inputs_path: Path = E1_INPUTS) -> dict[str, Any]:
    """Fail-closed validation used by CPU comparator and every GPU process."""
    if not lock_path.is_file() or not manifest_path.is_file() or not inputs_path.is_file():
        raise ValueError("e1_input_or_runtime_drifted: locked input artifact is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if lock.get("actions_observed") is not False:
        raise ValueError("locked input cannot be reopened after action observation")
    if sha256_file(manifest_path) != lock.get("e1_frozen_pair_manifest_sha256"):
        raise ValueError("e1_input_or_runtime_drifted: frozen manifest")
    if sha256_file(inputs_path) != lock.get("e1_frozen_inputs_and_noise_sha256"):
        raise ValueError("e1_input_or_runtime_drifted: frozen NPZ")
    if manifest.get("pair_count") != 170 or manifest.get("observation_count") != 340 or len(manifest.get("pairs", [])) != 170:
        raise ValueError("frozen manifest coverage is not 170/340")
    with np.load(inputs_path, allow_pickle=False) as arrays_file:
        arrays = {key: np.asarray(arrays_file[key]) for key in arrays_file.files}
    expected_shapes = {"base_0_rgb": (340, 224, 224, 3), "left_wrist_0_rgb": (340, 224, 224, 3), "right_wrist_0_rgb": (340, 224, 224, 3), "state": (340, 32), "tokenized_prompt": (340, 200), "tokenized_prompt_mask": (340, 200), "noise": (340, 10, 32), "pair_index": (340,), "side_index": (340,)}
    for key, shape in expected_shapes.items():
        if key not in arrays or tuple(arrays[key].shape) != shape:
            raise ValueError(f"frozen input shape missing/mismatch: {key}")
    if any(arrays[key].dtype != np.dtype(dtype) for key, dtype in {"base_0_rgb": "float32", "left_wrist_0_rgb": "float32", "right_wrist_0_rgb": "float32", "state": "float32", "tokenized_prompt": "int32", "tokenized_prompt_mask": "bool", "noise": "float32"}.items()):
        raise ValueError("frozen input dtype mismatch")
    if not all(np.isfinite(arrays[key]).all() for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "noise")):
        raise ValueError("frozen input contains non-finite values")
    for index, pair in enumerate(manifest["pairs"]):
        left, right = pair["observation_indices"]
        if [left, right] != [2 * index, 2 * index + 1] or arrays["pair_index"][left] != index or arrays["pair_index"][right] != index:
            raise ValueError("frozen pair order/index mismatch")
        for observation_index, side in ((left, pair["sides"][0]), (right, pair["sides"][1])):
            expected = side["policy_input_sha256"]
            actual = _bundle_hash(tuple((key, arrays[key][observation_index]) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask")))
            if actual != expected:
                raise ValueError(f"policy input hash mismatch at {observation_index}")
        if not np.array_equal(arrays["noise"][left], arrays["noise"][right]) or _array_hash(arrays["noise"][left]) != pair["noise"]["sha256"]:
            raise ValueError(f"pair noise mismatch: {pair['pair_id']}")
    return {"lock": lock, "manifest": manifest, "arrays": arrays, "lock_sha256": sha256_file(lock_path), "manifest_sha256": sha256_file(manifest_path), "inputs_sha256": sha256_file(inputs_path)}


def _model_contract() -> dict[str, Any]:
    config_path = FOLDED_MODEL / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processor_paths = sorted(path for pattern in ("policy_preprocessor*", "policy_postprocessor*") for path in FOLDED_MODEL.glob(pattern) if path.is_file())
    required = {"policy_preprocessor.json", "policy_postprocessor.json", "policy_preprocessor_step_2_normalizer_processor.safetensors", "policy_postprocessor_step_0_unnormalizer_processor.safetensors"}
    if required.difference(path.name for path in processor_paths):
        raise ValueError("on-disk LeRobot processor set incomplete")
    return {
        "model_sha256": sha256_file(FOLDED_MODEL / "model.safetensors"),
        "norm_stats_sha256": sha256_file(NORM_STATS),
        "raw_params_tree_sha256": sha256_tree(RAW_PARAMS),
        "source_config_sha256": sha256_file(config_path),
        "source_config_contract": {key: config.get(key) for key in ("type", "dtype", "chunk_size", "n_action_steps", "max_action_dim")},
        "processor_hashes": {path.name: sha256_file(path) for path in processor_paths},
    }


def _backend_common(backend: str, validated: dict[str, Any], output_dir: Path, gpu_id: str, mode: str) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    if str(gpu_id) == "0":
        raise ValueError("GPU0 is outside the E1 boundary")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PALIGEMMA_TOKENIZER_DIR", str(TOKENIZER_MODEL.parent))
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = validated["arrays"]
    limit = 4 if mode == "sentinel" else 340
    monitor = [gpu_snapshot()]
    contract = _model_contract()
    report = {"schema_version": 1, "phase": "E1", "backend": backend, "mode": mode, "status": "running", "gpu_id": str(gpu_id), "pair_count": 170 if limit == 340 else 2, "observation_count": limit, "lock_sha256": validated["lock_sha256"], "manifest_sha256": validated["manifest_sha256"], "frozen_inputs_sha256": validated["inputs_sha256"], "model_contract": contract, "input_contract": {key: _fingerprint(arrays[key]) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask", "noise")}, "latency_seconds": [], "warnings": [], "gpu0_touched": None}
    return report, {key: value[:limit] for key, value in arrays.items()}, monitor


def _jax_backend(validated: dict[str, Any], output_dir: Path, gpu_id: str, mode: str, microbatch_size: int, seed: int) -> dict[str, Any]:
    report, arrays, monitor = _backend_common("r6_folded_jax_source", validated, output_dir, gpu_id, mode)
    sys.path.insert(0, str(NESTED_SRC))
    sys.path.insert(1, str(ROOT))
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from tools.axis_pi05.lora_folding import fold_lora_in_mapping

    raw_params = model_lib.restore_params(RAW_PARAMS, restore_type=np.ndarray, dtype="float32")
    folded_params = copy.deepcopy(raw_params)
    folded_modules = fold_lora_in_mapping(folded_params)
    dense_config = pi0_config.Pi0Config(pi05=True, dtype="float32", action_dim=32, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    template = dense_config.create(jax.random.key(0))
    graphdef, state = nnx.split(template)
    state.replace_by_pure_dict(folded_params)
    model = nnx.merge(graphdef, state)
    model.eval()
    monitor.append(gpu_snapshot())
    limit = arrays["state"].shape[0]
    actions: list[np.ndarray] = []
    for start in range(0, limit, microbatch_size):
        stop = min(start + microbatch_size, limit)
        width = stop - start
        observation = model_lib.Observation.from_dict({
            "image": {key: jnp.asarray(arrays[key][start:stop]) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
            "image_mask": {"base_0_rgb": jnp.ones((width,), dtype=bool), "left_wrist_0_rgb": jnp.ones((width,), dtype=bool), "right_wrist_0_rgb": jnp.zeros((width,), dtype=bool)},
            "state": jnp.asarray(arrays["state"][start:stop]), "tokenized_prompt": jnp.asarray(arrays["tokenized_prompt"][start:stop]), "tokenized_prompt_mask": jnp.asarray(arrays["tokenized_prompt_mask"][start:stop]),
        })
        started = time.perf_counter()
        value = np.asarray(jax.block_until_ready(model.sample_actions(jax.random.key(seed), observation, num_steps=10, noise=jnp.asarray(arrays["noise"][start:stop]))), dtype=np.float32)
        report["latency_seconds"].append(time.perf_counter() - started)
        actions.append(value)
    internal = np.concatenate(actions, axis=0)
    q = json.loads(NORM_STATS.read_text(encoding="utf-8"))["norm_stats"]["actions"]
    q01, q99 = np.asarray(q["q01"], dtype=np.float32), np.asarray(q["q99"], dtype=np.float32)
    external = ((internal[..., :7] + 1.0) * (q99 - q01) / 2.0 + q01).astype(np.float32)
    monitor.append(gpu_snapshot())
    del model, template, raw_params, folded_params
    gc.collect()
    monitor.append(gpu_snapshot())
    evidence = gpu_boundary_evidence(monitor, target_pids={os.getpid()}, allowed_gpu_ids={int(gpu_id)})
    report.update({"status": "passed" if np.isfinite(internal).all() and np.isfinite(external).all() else "failed", "folded_lora_modules": folded_modules, "internal": _fingerprint(internal), "external": _fingerprint(external), "all_finite": bool(np.isfinite(internal).all() and np.isfinite(external).all()), "gpu_monitor_evidence": {"samples": monitor, **evidence}, "gpu0_touched": evidence["gpu0_touched"], "artifact": str((output_dir / "actions.npz").resolve())})
    np.savez_compressed(output_dir / "actions.npz", internal_actions=internal, external_actions=external, tokenized_prompt=arrays["tokenized_prompt"], tokenized_prompt_mask=arrays["tokenized_prompt_mask"], state=arrays["state"], noise=arrays["noise"])
    write_json(output_dir / "report.json", report)
    return report


def _torch_pad(state: np.ndarray, device: Any) -> Any:
    import torch

    return torch.nn.functional.pad(torch.from_numpy(np.asarray(state, dtype=np.float32)), (0, 24)).to(device)


def _pytorch_backend(validated: dict[str, Any], output_dir: Path, gpu_id: str, mode: str, microbatch_size: int) -> dict[str, Any]:
    report, arrays, monitor = _backend_common("r6_on_disk_lerobot_folded_pytorch_source", validated, output_dir, gpu_id, mode)
    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(1, str(NESTED_SRC))
    sys.path.insert(2, str(ROOT))
    import sentencepiece  # noqa: F401
    import torch
    import safetensors.torch
    from lerobot.processor import DataProcessorPipeline
    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    tasks = [str(item["instruction"]) for pair in validated["manifest"]["pairs"] for item in pair["sides"]]
    tasks = tasks[: arrays["state"].shape[0]]
    norm = json.loads(NORM_STATS.read_text(encoding="utf-8"))["norm_stats"]
    q01_state, q99_state = np.asarray(norm["state"]["q01"], dtype=np.float32), np.asarray(norm["state"]["q99"], dtype=np.float32)
    normalized_state = arrays["state"][:, :8]
    raw_state = (normalized_state + 1.0) * (q99_state - q01_state) / 2.0 + q01_state
    processor_input = {"observation.images.image": torch.from_numpy(arrays["base_0_rgb"].transpose(0, 3, 1, 2)), "observation.images.image2": torch.from_numpy(arrays["left_wrist_0_rgb"].transpose(0, 3, 1, 2)), "observation.images.empty_camera_0": torch.from_numpy(arrays["right_wrist_0_rgb"].transpose(0, 3, 1, 2)), "observation.state": torch.from_numpy(raw_state), "task": tasks}
    preprocessor = DataProcessorPipeline.from_pretrained(FOLDED_MODEL, config_filename="policy_preprocessor.json", local_files_only=True)
    processed = preprocessor(processor_input)
    processed_images = {key: processed[key].detach().cpu() for key in ("observation.images.image", "observation.images.image2", "observation.images.empty_camera_0")}
    processed_state = processed["observation.state"].detach().cpu().numpy().astype(np.float32)
    processed_tokens_tensor = processed["observation.language.tokens"].detach().cpu()
    processed_mask_tensor = processed["observation.language.attention_mask"].detach().cpu()
    processed_tokens = processed_tokens_tensor.numpy().astype(np.int32)
    processed_mask = processed_mask_tensor.numpy().astype(bool)
    state_diff = float(np.max(np.abs(processed_state - normalized_state)))
    token_diff = int(np.max(np.abs(processed_tokens.astype(np.int64) - arrays["tokenized_prompt"].astype(np.int64))))
    mask_diff = int(np.count_nonzero(processed_mask != arrays["tokenized_prompt_mask"]))
    report.update({"processor_state_max_abs_diff": state_diff, "processor_token_max_abs_diff": token_diff, "processor_token_mask_mismatch_count": mask_diff, "processor_contract_exact": bool(state_diff <= 1e-6 and token_diff == 0 and mask_diff == 0)})
    del processed, preprocessor, processor_input
    gc.collect()
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("selected E1 GPU is unavailable")
    config = Pi0Config(pi05=True, action_dim=32, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", dtype="float32", pytorch_compile_mode=None)
    model = PI0Pytorch(config).to(device).eval()
    safetensors.torch.load_model(model, str(FOLDED_MODEL / "model.safetensors"))
    monitor.append(gpu_snapshot())
    internal_batches: list[np.ndarray] = []
    for start in range(0, arrays["state"].shape[0], microbatch_size):
        stop = min(start + microbatch_size, arrays["state"].shape[0])
        width = stop - start
        observation = Observation.from_dict({"image": {"base_0_rgb": processed_images["observation.images.image"][start:stop].to(device), "left_wrist_0_rgb": processed_images["observation.images.image2"][start:stop].to(device), "right_wrist_0_rgb": processed_images["observation.images.empty_camera_0"][start:stop].to(device)}, "image_mask": {"base_0_rgb": torch.ones(width, dtype=torch.bool, device=device), "left_wrist_0_rgb": torch.ones(width, dtype=torch.bool, device=device), "right_wrist_0_rgb": torch.zeros(width, dtype=torch.bool, device=device)}, "state": _torch_pad(processed_state[start:stop], device), "tokenized_prompt": processed_tokens_tensor[start:stop].to(device=device, dtype=torch.int32), "tokenized_prompt_mask": processed_mask_tensor[start:stop].to(device=device)})
        noise = torch.from_numpy(np.asarray(arrays["noise"][start:stop], dtype=np.float32)).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            value = model.sample_actions(device, observation, noise=noise, num_steps=10)
        report["latency_seconds"].append(time.perf_counter() - started)
        internal_batches.append(value.detach().cpu().numpy().astype(np.float32))
    internal = np.concatenate(internal_batches, axis=0)
    postprocessor = DataProcessorPipeline.from_pretrained(FOLDED_MODEL, config_filename="policy_postprocessor.json", local_files_only=True)
    external = postprocessor({"action": torch.from_numpy(internal[..., :7])})["action"].detach().cpu().numpy().astype(np.float32)
    q = json.loads(NORM_STATS.read_text(encoding="utf-8"))["norm_stats"]["actions"]
    q01, q99 = np.asarray(q["q01"], dtype=np.float32), np.asarray(q["q99"], dtype=np.float32)
    direct = ((internal[..., :7] + 1.0) * (q99 - q01) / 2.0 + q01).astype(np.float32)
    monitor.append(gpu_snapshot())
    del model, config, postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    monitor.append(gpu_snapshot())
    evidence = gpu_boundary_evidence(monitor, target_pids={os.getpid()}, allowed_gpu_ids={int(gpu_id)})
    finite = bool(np.isfinite(internal).all() and np.isfinite(external).all())
    report.update({"status": "passed" if finite else "failed", "internal": _fingerprint(internal), "external": _fingerprint(external), "postprocessor_vs_direct_max_abs": float(np.max(np.abs(external - direct))), "all_finite": finite, "gpu_monitor_evidence": {"samples": monitor, **evidence}, "gpu0_touched": evidence["gpu0_touched"], "artifact": str((output_dir / "actions.npz").resolve())})
    np.savez_compressed(output_dir / "actions.npz", internal_actions=internal, external_actions=external, tokenized_prompt=processed_tokens, tokenized_prompt_mask=processed_mask, state=processed_state, noise=arrays["noise"])
    write_json(output_dir / "report.json", report)
    return report


def _diff(a: Any, b: Any) -> dict[str, Any]:
    left, right = np.asarray(a), np.asarray(b)
    if left.shape != right.shape:
        return {"shape_match": False, "dtype_match": left.dtype == right.dtype, "finite": False, "max_abs_diff": None, "mean_abs_diff": None, "p50_abs_diff": None, "p95_abs_diff": None, "p99_abs_diff": None, "exact": False}
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"shape_match": True, "dtype_match": left.dtype == right.dtype, "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()), "max_abs_diff": float(np.max(delta)), "mean_abs_diff": float(np.mean(delta)), "p50_abs_diff": float(np.percentile(delta, 50)), "p95_abs_diff": float(np.percentile(delta, 95)), "p99_abs_diff": float(np.percentile(delta, 99)), "max_abs_by_horizon": np.max(delta, axis=(0, 2)).tolist() if delta.ndim == 3 else None, "max_abs_by_dimension": np.max(delta, axis=(0, 1)).tolist() if delta.ndim == 3 else None, "exact": bool(left.dtype == right.dtype and np.array_equal(left, right))}


def _sign_flips(a: Any, b: Any) -> int:
    left, right = np.asarray(a), np.asarray(b)
    return int(np.count_nonzero(np.sign(left) != np.sign(right)))


def compare_envelope(*, lock_path: Path = E1_LOCK, manifest_path: Path = E1_MANIFEST, inputs_path: Path = E1_INPUTS, jax_dir: Path = E1_JAX, pytorch_dir: Path = E1_PYTORCH, output_dir: Path = E1_ROOT) -> dict[str, Any]:
    validated = validate_locked_inputs(lock_path, manifest_path, inputs_path)
    jax_report = json.loads((jax_dir / "report.json").read_text(encoding="utf-8"))
    pt_report = json.loads((pytorch_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(jax_dir / "actions.npz", allow_pickle=False) as file:
        jax = {key: np.asarray(file[key]) for key in file.files}
    with np.load(pytorch_dir / "actions.npz", allow_pickle=False) as file:
        pt = {key: np.asarray(file[key]) for key in file.files}
    expected = (340, 10, 32)
    expected_external = (340, 10, 7)
    contract_exact = all(tuple(data["internal_actions"].shape) == expected and tuple(data["external_actions"].shape) == expected_external and data["internal_actions"].dtype == np.dtype("float32") and data["external_actions"].dtype == np.dtype("float32") for data in (jax, pt))
    input_hashes_exact = all(report.get("lock_sha256") == validated["lock_sha256"] and report.get("manifest_sha256") == validated["manifest_sha256"] and report.get("frozen_inputs_sha256") == validated["inputs_sha256"] for report in (jax_report, pt_report))
    model_contract_exact = jax_report["model_contract"]["model_sha256"] == FOLDED_MODEL_SHA256 and pt_report["model_contract"]["model_sha256"] == FOLDED_MODEL_SHA256 and jax_report["model_contract"]["norm_stats_sha256"] == NORM_STATS_SHA256 and pt_report["model_contract"]["norm_stats_sha256"] == NORM_STATS_SHA256 and jax_report["model_contract"]["raw_params_tree_sha256"] == RAW_PARAMS_TREE_SHA256
    pair_metrics: list[dict[str, Any]] = []
    for index, pair in enumerate(validated["manifest"]["pairs"]):
        left, right = 2 * index, 2 * index + 1
        metric = {"pair_id": pair["pair_id"], "pair_index": index, "content_hash": pair["content_hash"], "case_id": pair["case_id"], "source_kind": pair["source_kind"], "phase_or_trajectory_frame": pair["phase_or_trajectory_frame"], "camera": pair["camera"], "observation_indices": [left, right], "jax_internal_full_chunk": _diff(jax["internal_actions"][left], jax["internal_actions"][right]), "jax_external_full_chunk": _diff(jax["external_actions"][left], jax["external_actions"][right]), "pytorch_internal_full_chunk": _diff(pt["internal_actions"][left], pt["internal_actions"][right]), "pytorch_external_full_chunk": _diff(pt["external_actions"][left], pt["external_actions"][right]), "jax_internal_execute8": _diff(jax["internal_actions"][left, :8], jax["internal_actions"][right, :8]), "jax_external_execute8": _diff(jax["external_actions"][left, :8], jax["external_actions"][right, :8]), "pytorch_internal_execute8": _diff(pt["internal_actions"][left, :8], pt["internal_actions"][right, :8]), "pytorch_external_execute8": _diff(pt["external_actions"][left, :8], pt["external_actions"][right, :8])}
        metric["jax_gripper_sign_flip_execute8"] = _sign_flips(jax["external_actions"][left, :8, 6], jax["external_actions"][right, :8, 6])
        metric["pytorch_gripper_sign_flip_execute8"] = _sign_flips(pt["external_actions"][left, :8, 6], pt["external_actions"][right, :8, 6])
        pair_metrics.append(metric)
    jax_external_diff = _diff(jax["external_actions"], pt["external_actions"])
    jax_internal_diff = _diff(jax["internal_actions"], pt["internal_actions"])
    sentinel_replay = None
    sentinel_a = E1_PYTORCH / "sentinel_a" / "actions.npz"
    sentinel_b = E1_PYTORCH / "sentinel_b" / "actions.npz"
    if sentinel_a.is_file() and sentinel_b.is_file():
        with np.load(sentinel_a, allow_pickle=False) as first, np.load(sentinel_b, allow_pickle=False) as second:
            sentinel_replay = {"internal": _diff(first["internal_actions"], second["internal_actions"]), "external": _diff(first["external_actions"], second["external_actions"])}
    max_metric = lambda key: float(max(item[key]["max_abs_diff"] for item in pair_metrics))
    gates = {
        "pair_input_coverage_170": len(pair_metrics) == 170,
        "observation_coverage_340_per_backend": jax_report.get("observation_count") == 340 and pt_report.get("observation_count") == 340 and jax["internal_actions"].shape[0] == 340 and pt["internal_actions"].shape[0] == 340,
        "all_actions_finite": bool(np.isfinite(jax["internal_actions"]).all() and np.isfinite(jax["external_actions"]).all() and np.isfinite(pt["internal_actions"]).all() and np.isfinite(pt["external_actions"]).all()),
        "shape_dtype_contract_exact": contract_exact,
        "model_norm_processor_contract_exact": model_contract_exact and bool(pt_report.get("processor_contract_exact")) and jax_report.get("gpu0_touched") is False and pt_report.get("gpu0_touched") is False,
        "input_state_prompt_noise_hash_mismatch_zero": input_hashes_exact and bool(np.array_equal(jax["tokenized_prompt"], validated["arrays"]["tokenized_prompt"])) and bool(np.array_equal(pt["tokenized_prompt"], validated["arrays"]["tokenized_prompt"])) and bool(np.array_equal(jax["noise"], validated["arrays"]["noise"])) and bool(np.array_equal(pt["noise"], validated["arrays"]["noise"])),
        "token_mask_mismatch_zero": pt_report.get("processor_token_max_abs_diff") == 0 and pt_report.get("processor_token_mask_mismatch_count") == 0,
        "pytorch_sentinel_fresh_replay_bitwise_exact": sentinel_replay is not None and sentinel_replay["internal"]["exact"] and sentinel_replay["external"]["exact"],
        "jax_egl_external_max_le_1e_4": max_metric("jax_external_execute8") <= 1e-4 and max_metric("jax_external_full_chunk") <= 1e-4,
        "pytorch_egl_external_max_le_1e_4": max_metric("pytorch_external_execute8") <= 1e-4 and max_metric("pytorch_external_full_chunk") <= 1e-4,
        "egl_execute8_gripper_sign_flip_zero": sum(item["jax_gripper_sign_flip_execute8"] + item["pytorch_gripper_sign_flip_execute8"] for item in pair_metrics) == 0,
        "folded_jax_pytorch_external_max_le_1e_3": jax_external_diff["max_abs_diff"] is not None and jax_external_diff["max_abs_diff"] <= 1e-3,
        "backend_gripper_sign_agreement_100_percent": _sign_flips(jax["external_actions"][..., :8, 6], pt["external_actions"][..., :8, 6]) == 0,
    }
    status = "egl_action_response_bounded_pending_statistical_closed_loop" if all(gates.values()) else "e1_action_response_hard_gate_failed"
    first_failure = next((key for key, value in gates.items() if not value), None)
    summary = {"schema_version": 1, "phase": "E1", "status": status, "actions_observed": True, "input_lock_immutable": True, "pair_count": 170, "observation_count": 340, "snapshot_pair_count": 13, "trajectory_pair_count": 157, "trajectory_affected_frame_occurrences": 302, "backends": {"jax": jax_report, "lerobot_pytorch": pt_report}, "hard_gates": gates, "first_failure": first_failure, "sentinel_replay": sentinel_replay, "correspondence": {"internal_full_chunk": jax_internal_diff, "external_full_chunk": jax_external_diff, "external_execute8": _diff(jax["external_actions"][..., :8], pt["external_actions"][..., :8]), "gripper_sign_flip_execute8": _sign_flips(jax["external_actions"][..., :8, 6], pt["external_actions"][..., :8, 6])}, "egl_envelope": {"jax_external_execute8_max_abs": max_metric("jax_external_execute8"), "jax_external_full_chunk_max_abs": max_metric("jax_external_full_chunk"), "pytorch_external_execute8_max_abs": max_metric("pytorch_external_execute8"), "pytorch_external_full_chunk_max_abs": max_metric("pytorch_external_full_chunk"), "jax_pytorch_external_max_abs": jax_external_diff["max_abs_diff"]}, "artifacts": {"pair_metrics": str(E1_PAIR_METRICS.resolve()), "jax_actions": str((jax_dir / "actions.npz").resolve()), "pytorch_actions": str((pytorch_dir / "actions.npz").resolve())}}
    write_jsonl(output_dir / E1_PAIR_METRICS.name, pair_metrics)
    write_json(output_dir / E1_SUMMARY.name, summary)
    return summary


def write_reports(summary: dict[str, Any], output_dir: Path = E1_ROOT) -> None:
    gates = summary["hard_gates"]
    report = f"""# E1 EGL Action-Response Envelope

状态：`{summary['status']}`

本 section 仅执行 E1.0–E1.6：冻结 E0 的 170 个 content-unique pair（340 observations），在同一张本地 GPU 上以 fresh process 顺序运行 folded JAX 与真实 on-disk LeRobot/PyTorch。未执行 environment、closed-loop、E2、FAST、ZIP、训练、比赛 GPU、push 或上传。

## 覆盖

- snapshot / trajectory content-unique pairs：`13 / 157`
- trajectory affected-frame occurrences（仅 provenance 计数）：`302`
- pair / observation coverage：`170 / 340`
- frozen input/noise：lock commit 中固定，action 观测后未修改

## Hard gates

```json
{json.dumps(gates, ensure_ascii=False, indent=2, sort_keys=True)}
```

首次失败：`{summary.get('first_failure')}`

EGL envelope 和 backend correspondence 的完整 pair-level 结果见 `pair_metrics.jsonl` 与 `summary.json`；本结果不宣称模型稳定、性能提升或正式 winner。完成 clean commit 后停止，等待人工审核。
"""
    (output_dir / "E1_ACTION_ENVELOPE_REPORT_ZH.md").write_text(report, encoding="utf-8")
    (output_dir / "THREAD_REPORT_ZH.md").write_text(report + "\n本 section 已 hard stop，未进入 E2 或 closed-loop。\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lock = sub.add_parser("build-lock")
    lock.add_argument("--output", type=Path, default=E1_ROOT)
    lock.add_argument("--global-noise-seed", type=int, default=GLOBAL_NOISE_SEED)
    validate = sub.add_parser("validate-lock")
    validate.add_argument("--lock", type=Path, default=E1_LOCK)
    validate.add_argument("--manifest", type=Path, default=E1_MANIFEST)
    validate.add_argument("--inputs", type=Path, default=E1_INPUTS)
    backend = sub.add_parser("backend")
    backend.add_argument("--backend", choices=("jax", "pytorch"), required=True)
    backend.add_argument("--mode", choices=("sentinel", "full"), required=True)
    backend.add_argument("--lock", type=Path, default=E1_LOCK)
    backend.add_argument("--manifest", type=Path, default=E1_MANIFEST)
    backend.add_argument("--inputs", type=Path, default=E1_INPUTS)
    backend.add_argument("--output", type=Path, required=True)
    backend.add_argument("--gpu-id", default="1")
    backend.add_argument("--microbatch-size", type=int, default=1)
    backend.add_argument("--seed", type=int, default=0)
    compare = sub.add_parser("compare")
    compare.add_argument("--output", type=Path, default=E1_ROOT)
    args = parser.parse_args()
    if args.command == "build-lock":
        print(json.dumps(build_lock(args.output, global_seed=args.global_noise_seed), ensure_ascii=False, indent=2))
    elif args.command == "validate-lock":
        result = validate_locked_inputs(args.lock, args.manifest, args.inputs)
        print(json.dumps({key: result[key] for key in ("lock_sha256", "manifest_sha256", "inputs_sha256")}, ensure_ascii=False, indent=2))
    elif args.command == "backend":
        validated = validate_locked_inputs(args.lock, args.manifest, args.inputs)
        result = _jax_backend(validated, args.output, args.gpu_id, args.mode, args.microbatch_size, args.seed) if args.backend == "jax" else _pytorch_backend(validated, args.output, args.gpu_id, args.mode, args.microbatch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "compare":
        summary = compare_envelope(output_dir=args.output)
        write_reports(summary, args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
