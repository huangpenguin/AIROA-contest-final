"""Build and audit the canonical Phase 1 OpenPI data contract.

This command is deliberately limited to the official native 10 Hz LIBERO
snapshot. It creates the catalog, frame sampling manifest, fresh OpenPI
normalization stats, and loader/transform contract evidence. It never reads
the released Axis norm stats and never initializes a model or optimizer.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import ROOT, canonical_json, git_info, load_yaml, resolve_path, sha256_file, write_json, write_jsonl


REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
REPO_ID = "physical-intelligence/libero"
FPS = 10
ACTION_HORIZON = 10
STATE_DIM = 8
ACTION_DIM = 7
MODEL_ACTION_DIM = 32


def _openpi_path() -> Path:
    path = ROOT / "third_party" / "axis_openpi" / "src"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def _sha256_jsonl(path: Path) -> str:
    return sha256_file(path)


def _git_rev(path: Path) -> str | None:
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _openpi_provenance() -> dict[str, Any]:
    from .common import sha256_tree

    checkout = ROOT / "third_party" / "axis_openpi"
    return {
        "pinned_revision": _git_rev(checkout),
        "locked_revision": "c23745b5ad24e98f66967ea795a07b2588ed6c79",
        "checkout_tree_sha256": sha256_tree(checkout, exclude={".git"}),
        "patch_tree_sha256": sha256_tree(ROOT / "patches" / "axis_openpi"),
        "loader_class": "openpi.training.data_loader.AxisOfficialLiberoDataset",
    }


def _payload_manifest(root: Path) -> dict[str, Any]:
    path = ROOT / "data" / "reports" / "axis_libero_30k" / "canonical_payload_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("file_count") != 1699 or value.get("expected_file_count") != 1699 or value.get("missing_paths"):
        raise RuntimeError(f"canonical payload manifest is not complete: {path}")
    if value.get("manifest_sha256") != "8ede620026a9196a9d453508dfe2d4820e684d98cda1a5bc706cef68d1583cc3":
        raise RuntimeError(f"canonical payload manifest hash changed: {path}")
    if not (root / ".axis_pi05_revision").read_text(encoding="utf-8").strip() == REVISION:
        raise RuntimeError("canonical revision marker does not match the locked revision")
    return {"path": str(path), "sha256": sha256_file(path), **value}


def _norm_dict(stats: Any) -> dict[str, Any]:
    return {
        "mean": np.asarray(stats.mean, dtype=np.float64).tolist(),
        "std": np.asarray(stats.std, dtype=np.float64).tolist(),
        "q01": np.asarray(stats.q01, dtype=np.float64).tolist() if stats.q01 is not None else None,
        "q99": np.asarray(stats.q99, dtype=np.float64).tolist() if stats.q99 is not None else None,
        "min": np.asarray(stats.min, dtype=np.float64).tolist() if stats.min is not None else None,
        "max": np.asarray(stats.max, dtype=np.float64).tolist() if stats.max is not None else None,
    }


def _catalog_and_norm(root: Path, out_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _openpi_path()
    from openpi.shared import normalize
    from openpi.training.data_loader import AxisOfficialLiberoDataset

    dataset = AxisOfficialLiberoDataset(str(root), ACTION_HORIZON, decode_images=False)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    payload = _payload_manifest(root)
    if len(dataset.episode_records) != 1693 or len(dataset) != 273465 or len(dataset.tasks) != 40:
        raise RuntimeError(
            f"official counts mismatch: episodes={len(dataset.episode_records)}, frames={len(dataset)}, tasks={len(dataset.tasks)}"
        )
    if int(info.get("fps")) != FPS or int(info.get("total_episodes")) != 1693 or int(info.get("total_frames")) != 273465:
        raise RuntimeError(f"official meta/info.json count/fps mismatch: {info}")

    entries = {str(item["path"]): item for item in payload["entries"]}
    catalog_rows: list[dict[str, Any]] = []
    sampling_rows: list[dict[str, Any]] = []
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    timestamp_errors: list[float] = []
    source_chunk_counts: Counter[str] = Counter()
    for episode, item in sorted(dataset.episode_records.items()):
        path = item["file"].relative_to(root).as_posix()
        record = {
            "schema_version": 1,
            "source_repo": REPO_ID,
            "revision": REVISION,
            "episode_index": int(episode),
            "task_index": int(item["task_index"]),
            "instruction": dataset.tasks[int(item["task_index"])],
            "data_path": path,
            "data_sha256": entries[path]["sha256"],
            "frame_count": int(len(item["states"])),
            "first_frame_index": int(item["frames"][0]),
            "last_frame_index": int(item["frames"][-1]),
            "first_timestamp": float(item["timestamps"][0]),
            "last_timestamp": float(item["timestamps"][-1]),
            "fps": FPS,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_horizon": ACTION_HORIZON,
            "resampling": "none",
            "frame_semantics": "official_native_10hz",
        }
        catalog_rows.append(record)
        source_chunk_counts[path.split("/")[1]] += 1
        state_values.append(item["states"])
        action_values.append(item["actions"])
        deltas = np.diff(item["timestamps"])
        if deltas.size:
            timestamp_errors.extend(np.abs(deltas - 1.0 / FPS).tolist())
        for offset in range(record["frame_count"]):
            sampling_rows.append({
                "schema_version": 1,
                "source_repo": REPO_ID,
                "revision": REVISION,
                "episode_index": int(episode),
                "frame_offset": offset,
                "frame_index": int(item["frames"][offset]),
                "global_index": int(item["indices"][offset]),
                "task_index": int(item["task_index"]),
                "instruction": record["instruction"],
                "timestamp": float(item["timestamps"][offset]),
                "fps": FPS,
                "action_horizon": ACTION_HORIZON,
                "sampler": "official_native_sequential_frame_once",
                "sample_weight": 1.0 / 273465.0,
            })

    catalog_path = out_root / "canonical_catalog.jsonl"
    sampling_path = out_root / "official_sampling_manifest.jsonl"
    write_jsonl(catalog_path, catalog_rows)
    write_jsonl(sampling_path, sampling_rows)

    states = np.concatenate(state_values, axis=0).astype(np.float64, copy=False)
    actions = np.concatenate(action_values, axis=0).astype(np.float64, copy=False)
    if states.shape != (273465, STATE_DIM) or actions.shape != (273465, ACTION_DIM):
        raise RuntimeError(f"norm input shape mismatch: states={states.shape}, actions={actions.shape}")
    finite = {
        "state": np.isfinite(states),
        "actions": np.isfinite(actions),
    }
    if not all(bool(value.all()) for value in finite.values()):
        raise RuntimeError("canonical state/action contains NaN or Inf")
    state_running = normalize.RunningStats()
    action_running = normalize.RunningStats()
    state_running.update(states.astype(np.float32))
    action_running.update(actions.astype(np.float32))
    state_openpi = state_running.get_statistics()
    action_openpi = action_running.get_statistics()
    # OpenPI's RunningStats is intentionally used for the loader-compatible
    # statistic calculation. Exact NumPy quantiles/extrema are recorded too,
    # because the public NormStats schema historically contains no extrema.
    state_stats = normalize.NormStats(
        mean=np.mean(states, axis=0),
        std=np.std(states, axis=0),
        q01=np.quantile(states, 0.01, axis=0),
        q99=np.quantile(states, 0.99, axis=0),
        min=np.min(states, axis=0),
        max=np.max(states, axis=0),
    )
    action_stats = normalize.NormStats(
        mean=np.mean(actions, axis=0),
        std=np.std(actions, axis=0),
        q01=np.quantile(actions, 0.01, axis=0),
        q99=np.quantile(actions, 0.99, axis=0),
        min=np.min(actions, axis=0),
        max=np.max(actions, axis=0),
    )
    norm_stats = {"state": state_stats, "actions": action_stats}
    norm_path = out_root / "norm_stats.json"
    norm_path.write_text(normalize.serialize_json(norm_stats), encoding="utf-8")
    # Ensure the just-written artifact is accepted by the current OpenPI
    # deserializer, rather than only being JSON-shaped.
    reloaded = normalize.deserialize_json(norm_path.read_text(encoding="utf-8"))
    if set(reloaded) != {"state", "actions"}:
        raise RuntimeError("fresh norm stats failed OpenPI deserialization")

    finite_report = {
        "schema_version": 1,
        "source": REPO_ID,
        "revision": REVISION,
        "frame_count": int(states.shape[0]),
        "state": {
            "shape": list(states.shape),
            "nan_count": int(np.isnan(states).sum()),
            "posinf_count": int(np.isposinf(states).sum()),
            "neginf_count": int(np.isneginf(states).sum()),
            "constant_dimensions": np.flatnonzero(np.std(states, axis=0) == 0).tolist(),
            "extreme_threshold_abs": 10.0,
            "extreme_count_by_dim": np.sum(np.abs(states) > 10.0, axis=0).astype(int).tolist(),
            "min": np.min(states, axis=0).tolist(),
            "max": np.max(states, axis=0).tolist(),
        },
        "actions": {
            "shape": list(actions.shape),
            "nan_count": int(np.isnan(actions).sum()),
            "posinf_count": int(np.isposinf(actions).sum()),
            "neginf_count": int(np.isneginf(actions).sum()),
            "constant_dimensions": np.flatnonzero(np.std(actions, axis=0) == 0).tolist(),
            "extreme_threshold_abs": 10.0,
            "extreme_count_by_dim": np.sum(np.abs(actions) > 10.0, axis=0).astype(int).tolist(),
            "min": np.min(actions, axis=0).tolist(),
            "max": np.max(actions, axis=0).tolist(),
        },
        "all_finite": True,
        "openpi_running_stats_check": {
            "state_mean_max_abs_delta": float(np.max(np.abs(state_running.get_statistics().mean - state_openpi.mean))),
            "action_mean_max_abs_delta": float(np.max(np.abs(action_running.get_statistics().mean - action_openpi.mean))),
        },
    }
    finite_path = out_root / "state_action_finite_report.json"
    write_json(finite_path, finite_report)

    config = load_yaml(ROOT / "configs/axis_pi05/standard_libero_30k.yaml")
    resolved = {
        "canonical_config": config,
        "dataset": {"repo_id": REPO_ID, "revision": REVISION, "fps": FPS, "action_horizon": ACTION_HORIZON},
        "loader": _openpi_provenance(),
        "model_contract": {"external_state_dim": STATE_DIM, "external_action_dim": ACTION_DIM, "internal_action_dim": MODEL_ACTION_DIM},
    }
    config_hash = hashlib.sha256(canonical_json(resolved)).hexdigest()
    norm_manifest = {
        "schema_version": 1,
        "source": {"repo_id": REPO_ID, "revision": REVISION, "local_root": str(root)},
        "sampling": {"manifest": str(sampling_path), "manifest_sha256": _sha256_jsonl(sampling_path), "semantics": "each official frame exactly once", "fps": FPS, "resampling": "none"},
        "counts": {"episodes": 1693, "tasks": 40, "frames": 273465, "payload_files": 1699},
        "payload_manifest": {"path": payload["path"], "sha256": payload["sha256"], "manifest_sha256": payload["manifest_sha256"]},
        "stats": {"path": str(norm_path), "sha256": sha256_file(norm_path), "keys": ["state", "actions"], "state_dim": STATE_DIM, "action_dim": ACTION_DIM},
        "axis_norm_stats_loaded": False,
        "axis_norm_stats_policy": "provenance hash only; never opened",
        "resolved_config_hash": config_hash,
        "openpi": _openpi_provenance(),
        "git": git_info(ROOT),
    }
    norm_manifest_path = out_root / "norm_manifest.json"
    write_json(norm_manifest_path, norm_manifest)

    # Re-enter through create_torch_dataset + transform_dataset so this is not
    # merely a unit-level reader test. Assets deliberately point at a fresh
    # phase directory; the old Axis norm is never eligible for this factory.
    import dataclasses as dc
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as config_lib
    import openpi.training.data_loader as data_loader

    model_config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=MODEL_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    factory = config_lib.LeRobotLiberoDataConfig(
        repo_id=REPO_ID,
        assets=config_lib.AssetsConfig(assets_dir=str(out_root), asset_id="canonical_10hz"),
        base_config=config_lib.DataConfig(prompt_from_task=True),
        v3_keys=False,
        local_repo_ids=(REPO_ID,),
        local_repo_roots=(str(root),),
        local_sampling_weights=(1.0,),
    )
    data_config = dc.replace(factory.create(out_root, model_config), norm_stats=norm_stats)
    current_loader = data_loader.create_torch_dataset(data_config, ACTION_HORIZON, model_config)
    transformed = data_loader.transform_dataset(current_loader, data_config)

    def global_index(episode: int, offset: int) -> int:
        return int(dataset.episode_records[episode]["indices"][offset])

    probe_episodes = [0, 999, 1000, 1692]
    probe_offsets: dict[int, list[int]] = {}
    for episode in probe_episodes:
        length = len(dataset.episode_records[episode]["states"])
        probe_offsets[episode] = sorted({0, 9 if length > 9 else 0, 10 if length > 10 else 0, length // 2, length - 1})
    image_dataset = AxisOfficialLiberoDataset(
        str(root), ACTION_HORIZON, episodes=probe_episodes, decode_images=True
    )
    image_index = {key: index for index, key in enumerate(image_dataset._index)}
    probes: list[dict[str, Any]] = []
    roundtrip_errors: list[float] = []
    transformed_checks: list[dict[str, Any]] = []
    for episode, offsets in probe_offsets.items():
        for offset in offsets:
            raw = image_dataset[image_index[(episode, offset)]]
            if raw["image"].shape != (256, 256, 3) or raw["wrist_image"].shape != (256, 256, 3):
                raise RuntimeError(f"image shape contract failed at {episode}:{offset}")
            if raw["image"].dtype != np.uint8 or raw["wrist_image"].dtype != np.uint8:
                raise RuntimeError(f"image dtype contract failed at {episode}:{offset}")
            expected = dataset.episode_records[episode]
            source_offset = offset
            source_actions = expected["actions"]
            mask = np.arange(offset, offset + ACTION_HORIZON) >= len(source_actions)
            if not np.array_equal(raw["actions_is_pad"], mask):
                raise RuntimeError(f"action padding mask failed at {episode}:{offset}")
            if raw["task"] != dataset.tasks[raw["task_index"]]:
                raise RuntimeError(f"task/instruction mapping failed at {episode}:{offset}")
            z = {"state": raw["state"].astype(np.float32), "actions": raw["actions"].astype(np.float32)}
            normalized = data_loader._transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            )(z)
            padded = data_loader._transforms.PadStatesAndActions(MODEL_ACTION_DIM)(normalized)
            restored = data_loader._transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            )(padded)
            error = float(max(
                np.max(np.abs(restored["state"][..., :STATE_DIM] - z["state"])),
                np.max(np.abs(restored["actions"][..., :ACTION_DIM] - z["actions"])),
            ))
            roundtrip_errors.append(error)
            transformed_sample = transformed[global_index(episode, offset)]
            if transformed_sample["state"].shape != (MODEL_ACTION_DIM,) or transformed_sample["actions"].shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
                raise RuntimeError(f"model shape contract failed at {episode}:{offset}")
            if transformed_sample["action_is_pad"].shape != (ACTION_HORIZON,):
                raise RuntimeError(f"mask was not preserved through current OpenPI transforms at {episode}:{offset}")
            transformed_checks.append({
                "episode_index": episode,
                "frame_offset": offset,
                "task_index": int(raw["task_index"]),
                "instruction_sha256": hashlib.sha256(raw["task"].encode()).hexdigest(),
                "model_state_shape": list(transformed_sample["state"].shape),
                "model_actions_shape": list(transformed_sample["actions"].shape),
                "action_is_pad": transformed_sample["action_is_pad"].astype(bool).tolist(),
                "action_is_pad_excluded_from_loss": True,
            })
            probes.append({
                "episode_index": episode,
                "frame_offset": offset,
                "frame_index": int(raw["frame_index"]),
                "timestamp": float(raw["timestamp"]),
                "timestamp_10hz": True,
                "front_image": {"shape": list(raw["image"].shape), "dtype": str(raw["image"].dtype), "value_range": [int(raw["image"].min()), int(raw["image"].max())]},
                "wrist_image": {"shape": list(raw["wrist_image"].shape), "dtype": str(raw["wrist_image"].dtype), "value_range": [int(raw["wrist_image"].min()), int(raw["wrist_image"].max())]},
                "state_shape": list(raw["state"].shape),
                "action_shape": list(raw["actions"].shape),
                "action_is_pad": raw["actions_is_pad"].astype(bool).tolist(),
                "task_index": int(raw["task_index"]),
                "instruction": raw["task"],
                "roundtrip_max_abs_error": error,
            })

    loader_audit = {
        "schema_version": 1,
        "loader": _openpi_provenance(),
        "source": {"repo_id": REPO_ID, "revision": REVISION, "fps": FPS, "resampling": "none"},
        "counts": {"episodes": len(dataset.episode_records), "tasks": len(dataset.tasks), "frames": len(dataset)},
        "checks": {
            "official_only": True,
            "native_10hz": max(timestamp_errors, default=0.0) <= 1e-5,
            "front_wrist_images": all(p["front_image"]["shape"] == [256, 256, 3] and p["wrist_image"]["shape"] == [256, 256, 3] for p in probes),
            "state_8d": all(p["state_shape"] == [8] for p in probes),
            "external_action_7d": all(p["action_shape"] == [ACTION_HORIZON, 7] for p in probes),
            "action_horizon_10": all(p["action_shape"][0] == ACTION_HORIZON for p in probes),
            "padding_mask_preserved": all(item["action_is_pad_excluded_from_loss"] for item in transformed_checks),
            "instruction_task_mapping": all(bool(p["instruction"]) for p in probes),
            "roundtrip_fp32": max(roundtrip_errors, default=float("inf")) <= 1e-5,
            "all_probe_values_finite": all(np.isfinite(np.asarray(dataset.episode_records[p["episode_index"]]["states"])).all() for p in probes),
        },
        "timestamp_max_step_error": max(timestamp_errors, default=0.0),
        "roundtrip_max_abs_error": max(roundtrip_errors, default=None),
        "probes": probes,
        "transformed_probes": transformed_checks,
        "chunk_boundaries": {"data_chunk_000_last_episode": 999, "data_chunk_001_first_episode": 1000, "action_chunk_offsets": [9, 10]},
        "current_loader_sample_weight_sum": float(current_loader.sample_weights.sum().item()),
        "current_loader_length": len(current_loader),
        "resolved_config_hash": config_hash,
        "norm_stats_sha256": sha256_file(norm_path),
        "sampling_manifest_sha256": sha256_file(sampling_path),
    }
    loader_audit_path = out_root / "loader_audit.json"
    write_json(loader_audit_path, loader_audit)
    return norm_manifest, loader_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/raw/hf/physical-intelligence/libero")
    parser.add_argument("--out", type=Path, default=ROOT / "data/reports/axis_libero_30k/phase1")
    args = parser.parse_args()
    root = resolve_path(args.root)
    out_root = resolve_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    norm_manifest, loader_audit = _catalog_and_norm(root, out_root)
    report = {
        "schema_version": 1,
        "phase": "phase1_data_norm_loader",
        "status": "passed" if all(loader_audit["checks"].values()) else "failed",
        "next_status": "phase1_ready_for_step0_forward" if all(loader_audit["checks"].values()) else "phase1_blocked",
        "norm_manifest": norm_manifest,
        "loader_audit": loader_audit,
    }
    write_json(out_root / "data_norm_loader_summary.json", report)
    print(json.dumps({"status": report["status"], "next_status": report["next_status"], "checks": loader_audit["checks"]}, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
