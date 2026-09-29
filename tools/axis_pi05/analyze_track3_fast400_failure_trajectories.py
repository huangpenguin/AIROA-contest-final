#!/usr/bin/env python3
"""Read-only D0 analysis for the rejected Track3 Fast400 run.

This module intentionally has no simulator, policy-server, GPU, training, or
promotion path.  It reads the immutable R6/step100/step300 evaluation roots
and the immutable eight-episode Fast400 reverse source, then atomically writes
an analysis-only evidence namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import ROOT, canonical_json, read_table, sha256_bytes, sha256_file, write_json


R6_EVIDENCE = ROOT / "data/reports/axis_libero_30k/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
STEP100_EVIDENCE = ROOT / "data/reports/axis_libero_30k/t3_fast400_h300_step100_salvage_20260915_attempt_0001"
STEP300_EVIDENCE = R6_EVIDENCE
R6_OUTPUT = ROOT / "outputs/axis_pi05/t3_fast400_r6_h300_paired_eval_20260915_attempt_0002"
STEP100_OUTPUT = ROOT / "outputs/axis_pi05/t3_fast400_h300_step100_salvage_20260915_attempt_0001"
STEP300_OUTPUT = R6_OUTPUT
REVERSE_ACCEPTED = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted"
REVERSE_PROCESSED = ROOT / "data/processed/axis_pi05_track3_reverse_gold_fast400_20260913"
FAST400_MANIFEST = ROOT / "data/manifests_track3/fast400_gold_80_20_20260913"
ACTIVE_PLAN = ROOT / "docs/AXIS_PI05_T3_POST_STEP100_FAILURE_DATA_FIRST_R6_SCREEN_PLAN_20260916.md"
LATEST_AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_STEP100_SALVAGE_CLOSEOUT_REVIEW_20260916.md"
MACHINE_HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
CLOSEOUT_SUMMARY = ROOT / "data/reports/axis_libero_30k/t3_fast400_h300_behavior_gate_closeout_20260915/summary.json"

ROLE_CASE_ROOTS = {
    "r6": R6_EVIDENCE / "r6/cases",
    "step100": STEP100_EVIDENCE / "step100/cases",
    "step300": STEP300_EVIDENCE / "step300/cases",
}
ROLE_OUTPUT_ROOTS = {
    "r6": R6_OUTPUT / "r6",
    "step100": STEP100_OUTPUT / "step100",
    "step300": STEP300_OUTPUT / "step300",
}
REVERSE_TRAIN = FAST400_MANIFEST / "reverse_train_episodes.parquet"
FORWARD_TRAIN = FAST400_MANIFEST / "forward_train_episodes.parquet"
FORWARD_DEV = FAST400_MANIFEST / "forward_validation_episodes.parquet"
SAMPLING_WEIGHTS = FAST400_MANIFEST / "sampling_weights.parquet"
MANIFEST_JSON = FAST400_MANIFEST / "manifest.json"
SAMPLER_AUDIT = FAST400_MANIFEST / "sampler_audit.json"
SOURCE_ROOTS = FAST400_MANIFEST / "source_roots.json"
REVERSE_PROVENANCE = REVERSE_PROCESSED / "provenance.json"

HISTORICAL_GLOBAL_BATCH = 64
PLANNED_D4_GLOBAL_BATCH = 32
EXPOSURE_STEPS = (25, 100, 300)
NEAR_DUPLICATE_RMSE_THRESHOLD = 0.05
NEAR_DUPLICATE_MEAN_L2_THRESHOLD = 0.10


def strict_json(path: Path) -> Any:
    """Load JSON while rejecting duplicate keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _file_inventory(path: Path) -> dict[str, Any]:
    path = path.resolve()
    _require(path.exists(), f"missing D0 input: {path}")
    if path.is_file():
        files = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}]
    else:
        files = []
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
    fingerprint = sha256_bytes(canonical_json(files))
    return {
        "root": str(path),
        "kind": "file" if path.is_file() else "directory",
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "fingerprint_sha256": fingerprint,
        "files": files,
    }


def _fingerprint(path: Path) -> tuple[str, int, int]:
    inventory = _file_inventory(path)
    return str(inventory["fingerprint_sha256"]), int(inventory["file_count"]), int(inventory["bytes"])


def _strict_json_count(inventories: Iterable[dict[str, Any]]) -> int:
    count = 0
    for inventory in inventories:
        root = Path(str(inventory["root"]))
        for item in inventory["files"]:
            rel = str(item["path"])
            if rel.endswith(".json"):
                strict_json(root if inventory["kind"] == "file" else root / rel)
                count += 1
    return count


def _snapshot_inputs() -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        "r6_evidence": R6_EVIDENCE,
        "step100_evidence": STEP100_EVIDENCE,
        "step100_output": STEP100_OUTPUT,
        "r6_output": R6_OUTPUT,
        "reverse_accepted": REVERSE_ACCEPTED,
        "reverse_processed": REVERSE_PROCESSED,
        "fast400_manifest": FAST400_MANIFEST,
        "active_plan": ACTIVE_PLAN,
        "latest_audit": LATEST_AUDIT,
        "machine_handoff": MACHINE_HANDOFF,
        "h300_closeout_summary": CLOSEOUT_SUMMARY,
    }
    # step300 and R6 are the same closed namespace, but the role is named in
    # the analysis so that the shared immutable input is not accidentally
    # mistaken for a newly evaluated step300 namespace.
    return {label: _file_inventory(path) for label, path in paths.items()}


def _case_files(case_root: Path, panel: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(case_root.glob("*.json")):
        value = strict_json(path)
        if not isinstance(value, dict) or value.get("panel") != panel:
            continue
        case_id = str(value.get("case_id"))
        _require(case_id not in result, f"duplicate {panel} case id {case_id}")
        result[case_id] = value
    return result


def _case_identity(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "case_fingerprint": case.get("case_fingerprint"),
        "env_seed": case.get("env_seed"),
        "policy_seed": case.get("policy_seed"),
        "init_state_index": case.get("init_state_index"),
        "init_state_sha256": case.get("init_state_sha256"),
        "bddl_sha256": case.get("bddl_sha256"),
        "evaluation_freeze_sha256": case.get("evaluation_freeze_sha256"),
        "panel_manifest_sha256": case.get("panel_manifest_sha256"),
        "control_frequency_hz": case.get("control_frequency_hz"),
        "chunk_size": case.get("chunk_size"),
        "executed_actions_per_chunk": case.get("executed_actions_per_chunk"),
        "max_steps": case.get("max_steps"),
        "norm_stats_sha256": case.get("norm_stats_sha256"),
    }


def _trace(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    _require(path.exists(), f"missing action trace: {path}")
    with path.open("rb") as handle:
        value = np.load(handle, allow_pickle=False)
        array = np.asarray(value)
    _require(array.ndim == 2 and array.shape[1] == 7, f"action trace is not N x 7: {path}: {array.shape}")
    _require(bool(np.isfinite(array).all()), f"nonfinite action trace: {path}")
    _require(bool(np.any(array != 0.0)), f"all-zero action trace: {path}")
    return array, {"shape": [int(v) for v in array.shape], "sha256": sha256_file(path)}


def _aligned_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    length = min(len(first), len(second))
    _require(length > 0, "cannot align empty action traces")
    diff = np.asarray(first[:length], dtype=np.float64) - np.asarray(second[:length], dtype=np.float64)
    return {
        "aligned_frames": length,
        "first_frames": int(len(first)),
        "second_frames": int(len(second)),
        "length_delta": int(len(first) - len(second)),
        "action_rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_l2": float(np.mean(np.linalg.norm(diff, axis=1))),
    }


def analyze_trajectories() -> dict[str, Any]:
    cases = {role: _case_files(root, "reverse_dev") for role, root in ROLE_CASE_ROOTS.items()}
    case_ids = set(cases["r6"])
    _require(case_ids == set(cases["step100"]) == set(cases["step300"]), "reverse case sets are not aligned")
    _require(len(case_ids) == 4, f"expected four reverse cases, got {len(case_ids)}")
    for case_id in sorted(case_ids):
        identities = [_case_identity(cases[role][case_id]) for role in ("r6", "step100", "step300")]
        _require(identities[0] == identities[1] == identities[2], f"case identity drift: {case_id}")

    rows: list[dict[str, Any]] = []
    role_counts: dict[str, dict[str, Any]] = {}
    for role in ("r6", "step100", "step300"):
        counts = Counter()
        for case_id in sorted(case_ids):
            case = cases[role][case_id]
            trace, trace_meta = _trace(Path(str(case["action_trace_path"])))
            counts["success"] += int(bool(case.get("success")))
            counts["collision"] += int(bool(case.get("collision")))
            counts["timeout"] += int(bool(case.get("timeout")))
            row = {
                "case_id": case_id,
                "task_id": case.get("task_id"),
                "task_name": case.get("task_name"),
                "language": case.get("language"),
                "env_seed": case.get("env_seed"),
                "case_fingerprint": case.get("case_fingerprint"),
                "success": bool(case.get("success")),
                "collision": bool(case.get("collision")),
                "timeout": bool(case.get("timeout")),
                "goal_reached": bool(case.get("goal_reached")),
                "steps": int(case.get("steps")),
                "action_shape_recorded": case.get("action_shape"),
                "failure_class": case.get("failure_class"),
                "failure_reason": case.get("failure_reason"),
                "trace": trace_meta,
                "semantic_stage_annotation": {
                    "status": "not_recorded",
                    "source": "none",
                    "used_for_success_or_failure": False,
                },
            }
            rows.append({"role": role, **row})
        role_counts[role] = {
            "case_count": len(case_ids),
            "success_count": int(counts["success"]),
            "collision_count": int(counts["collision"]),
            "timeout_count": int(counts["timeout"]),
            "failure_reason_counts": dict(sorted(Counter(str(r["failure_reason"]) for r in rows if r["role"] == role).items())),
        }

    by_case_role = {(row["case_id"], row["role"]): row for row in rows}
    aligned: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        base_case = cases["r6"][case_id]
        base_trace, _ = _trace(Path(str(base_case["action_trace_path"])))
        for role in ("step100", "step300"):
            candidate_case = cases[role][case_id]
            candidate_trace, _ = _trace(Path(str(candidate_case["action_trace_path"])))
            aligned.append({"case_id": case_id, "candidate_role": role, **_aligned_metrics(base_trace, candidate_trace)})

    return {
        "schema_version": "track3_d0_failure_trajectory_analysis_v1",
        "roles": role_counts,
        "reverse_case_count": len(case_ids),
        "case_results": rows,
        "aligned_action_comparisons_vs_r6": aligned,
        "semantic_stage_policy": "No stage meaning is inferred from action norm, trace shape, or video presence; no human stage annotation was supplied in these case records.",
    }


def _recursive_named_fields(value: Any, path: str = "") -> dict[str, Any]:
    found: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            lowered = str(key).lower()
            if any(token in lowered for token in ("lineage", "recording_id", "capture_id", "session_id")):
                found[child_path] = child
            found.update(_recursive_named_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_recursive_named_fields(child, f"{path}[{index}]"))
    return found


def _resample(array: np.ndarray, points: int = 128) -> np.ndarray:
    if len(array) == 1:
        return np.repeat(array, points, axis=0)
    source_x = np.linspace(0.0, 1.0, len(array))
    target_x = np.linspace(0.0, 1.0, points)
    return np.stack([np.interp(target_x, source_x, array[:, dim]) for dim in range(array.shape[1])], axis=1)


def _duplicate_groups(records: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is not None:
            groups[str(value)].append(str(record["episode_id"]))
    return {key: values for key, values in sorted(groups.items()) if len(values) > 1}


def _exposure_counts(probability: float, batch: int) -> dict[str, float]:
    return {f"steps_{steps}": float(probability * batch * steps) for steps in EXPOSURE_STEPS}


def _data_audit(closeout: dict[str, Any]) -> dict[str, Any]:
    manifest = strict_json(MANIFEST_JSON)
    sampler_audit = strict_json(SAMPLER_AUDIT)
    source_roots = strict_json(SOURCE_ROOTS)
    provenance = strict_json(REVERSE_PROVENANCE)
    reverse_rows = read_table(REVERSE_TRAIN)
    forward_train = read_table(FORWARD_TRAIN)
    forward_dev = read_table(FORWARD_DEV)
    sampling_rows = read_table(SAMPLING_WEIGHTS)
    _require(len(reverse_rows) == 8, f"expected eight reverse rows, got {len(reverse_rows)}")
    _require(len(set(str(row["source_episode_id"]) for row in reverse_rows)) == 8, "reverse source episode ids are duplicated")

    provenance_by_id = {str(row["source_episode"]): row for row in provenance["episodes"]}
    reverse_records: list[dict[str, Any]] = []
    action_arrays: dict[str, np.ndarray] = {}
    for row in sorted(reverse_rows, key=lambda item: int(item["source_episode_index"])):
        episode_id = str(row["source_episode_id"])
        metadata_path = REVERSE_ACCEPTED / episode_id / "metadata.json"
        promotion_path = REVERSE_ACCEPTED / episode_id / "promotion.json"
        replay_path = REVERSE_ACCEPTED / episode_id / "replay_summary.json"
        metadata = strict_json(metadata_path)
        promotion = strict_json(promotion_path)
        replay = strict_json(replay_path)
        provenance_row = provenance_by_id.get(episode_id)
        _require(provenance_row is not None, f"missing provenance for {episode_id}")
        trajectory_path = REVERSE_ACCEPTED / episode_id / "trajectory.npz"
        with np.load(trajectory_path, allow_pickle=False) as arrays:
            actions = np.asarray(arrays["actions"], dtype=np.float64)
            states = np.asarray(arrays["states"], dtype=np.float64)
        _require(actions.ndim == 2 and actions.shape[1] == 7, f"invalid raw action shape for {episode_id}: {actions.shape}")
        _require(bool(np.isfinite(actions).all()) and bool(np.isfinite(states).all()), f"nonfinite raw arrays for {episode_id}")
        action_arrays[episode_id] = actions
        explicit_lineage = _recursive_named_fields(metadata)
        bddl_sha = str(metadata.get("lock", {}).get("official_t3_bddl_sha256"))
        context = {
            "source": metadata.get("source"),
            "operator": metadata.get("operator"),
            "input_device": metadata.get("input_device"),
            "bddl_sha256": bddl_sha,
        }
        reverse_records.append(
            {
                "episode_id": episode_id,
                "task_id": metadata.get("task_id"),
                "task": row.get("task"),
                "family": row.get("family"),
                "source_repo": row.get("source_repo"),
                "revision": row.get("revision"),
                "source_episode_index": row.get("source_episode_index"),
                "frame_count": row.get("frame_count"),
                "source_fps": row.get("source_fps"),
                "source_file": row.get("source_file"),
                "raw_trajectory_sha256": row.get("raw_trajectory_sha256"),
                "raw_metadata_sha256": row.get("raw_metadata_sha256"),
                "start_state_sha256": row.get("start_state_sha256"),
                "end_state_sha256": metadata.get("end_state_sha256"),
                "computed_trajectory_sha256": sha256_file(trajectory_path),
                "computed_metadata_sha256": sha256_file(metadata_path),
                "computed_start_state_sha256": metadata.get("start_state_sha256"),
                "status": metadata.get("status"),
                "human_review": metadata.get("human_review"),
                "replay": metadata.get("replay"),
                "promotion_status": promotion.get("status"),
                "promotion_human_review_sha256": promotion.get("human_review_sha256"),
                "promotion_replay_summary_sha256": promotion.get("replay_summary_sha256"),
                "promotion_source_exists": Path(str(promotion.get("source_pending", ""))).exists(),
                "training_holdout": metadata.get("training_holdout"),
                "official_public_example": metadata.get("official_public_example"),
                "metadata_created_at_utc": metadata.get("created_at_utc"),
                "recording_lineage": {
                    "explicit_fields": explicit_lineage,
                    "explicit_lineage_id": None,
                    "status": "not_recorded" if not explicit_lineage else "recorded_field_requires_review",
                    "conservative_recording_context": context,
                    "promotion_source": metadata.get("promotion_source"),
                    "replay_summary_episode_dir": replay.get("episode_dir"),
                },
                "raw_array_shapes": {
                    "actions": [int(v) for v in actions.shape],
                    "states": [int(v) for v in states.shape],
                },
            }
        )

    sampler_by_id = {str(row.get("source_file")): row for row in sampling_rows if row.get("source") == "track3_reverse"}
    _require(set(sampler_by_id) == {str(record["episode_id"]) for record in reverse_records}, "sampler/reverse episode set mismatch")
    forward_sampler = [row for row in sampling_rows if row.get("source") == "libero_plus"]
    forward_probabilities = [float(row["episode_sampling_probability"]) for row in forward_sampler]
    forward_median = float(np.median(np.asarray(forward_probabilities, dtype=np.float64)))
    nominal_epoch_draws = len(sampling_rows)

    for record in reverse_records:
        sample = sampler_by_id[str(record["episode_id"])]
        probability = float(sample["episode_sampling_probability"])
        record["sampling"] = {
            "episode_sampling_probability": probability,
            "sampling_probability": float(sample["sampling_probability"]),
            "sampling_weight": float(sample["sampling_weight"]),
            "observed_frame_count": int(sample["observed_frame_count"]),
            "frame_weight": float(sample["frame_weight"]),
            "stored_effective_exposures": float(sample["effective_exposures"]),
            "expected_examples_per_nominal_manifest_epoch": float(probability * nominal_epoch_draws),
            "relative_to_forward_episode_probability_median": float(probability / forward_median),
            "historical_h300_expected_examples": _exposure_counts(probability, HISTORICAL_GLOBAL_BATCH),
            "planned_d4_expected_examples": _exposure_counts(probability, PLANNED_D4_GLOBAL_BATCH),
            "stored_exposure_formula": "episode_sampling_probability * reverse_row_count (8) in existing Fast400 builder",
            "recomputed_exposure_formula": "episode_sampling_probability * nominal_manifest_draw_count (12840), or * batch * optimizer_steps",
        }

    forward_train_ids = {
        (str(row.get("source_repo")), str(row.get("revision")), int(row.get("source_episode_index")))
        for row in forward_train
    }
    forward_dev_ids = {
        (str(row.get("source_repo")), str(row.get("revision")), int(row.get("source_episode_index")))
        for row in forward_dev
    }
    reverse_ids = {
        (str(row.get("source_repo")), str(row.get("revision")), int(row.get("source_episode_index")))
        for row in reverse_rows
    }
    reverse_task_texts = {str(row.get("task", "")).strip().lower() for row in reverse_rows}
    forward_task_texts = {str(row.get("task", "")).strip().lower() for row in forward_train + forward_dev}
    reverse_dev_path = FAST400_MANIFEST / "reverse_validation_episodes.parquet"

    duplicate_fields = {
        field: _duplicate_groups(reverse_records, field)
        for field in (
            "raw_trajectory_sha256",
            "raw_metadata_sha256",
            "start_state_sha256",
            "end_state_sha256",
            "promotion_human_review_sha256",
            "promotion_replay_summary_sha256",
        )
    }
    near_duplicate_pairs: list[dict[str, Any]] = []
    for first, second in itertools.combinations(reverse_records, 2):
        if first["task_id"] != second["task_id"]:
            continue
        first_array = _resample(action_arrays[str(first["episode_id"])])
        second_array = _resample(action_arrays[str(second["episode_id"])])
        difference = first_array - second_array
        rmse = float(np.sqrt(np.mean(difference * difference)))
        mean_l2 = float(np.mean(np.linalg.norm(difference, axis=1)))
        near_duplicate_pairs.append(
            {
                "first_episode_id": first["episode_id"],
                "second_episode_id": second["episode_id"],
                "task_id": first["task_id"],
                "resample_points": 128,
                "resampled_action_rmse": rmse,
                "resampled_action_mean_l2": mean_l2,
                "candidate_near_duplicate": rmse <= NEAR_DUPLICATE_RMSE_THRESHOLD and mean_l2 <= NEAR_DUPLICATE_MEAN_L2_THRESHOLD,
            }
        )

    by_task: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in reverse_records:
        by_task[str(record["task_id"])].append(record)
    task_summary: dict[str, Any] = {}
    for task_id, records in sorted(by_task.items()):
        context_ids = {
            sha256_bytes(canonical_json(record["recording_lineage"]["conservative_recording_context"]))
            for record in records
        }
        explicit_lineage_ids = {
            str(record["recording_lineage"]["explicit_lineage_id"])
            for record in records
            if record["recording_lineage"]["explicit_lineage_id"] is not None
        }
        task_summary[task_id] = {
            "accepted_episode_count": len(records),
            "accepted_episode_ids": [record["episode_id"] for record in records],
            "explicit_lineage_count": len(explicit_lineage_ids),
            "provable_independent_lineage_count": len(explicit_lineage_ids),
            "conservative_recording_context_count": len(context_ids),
            "required_accepted_count": 3,
            "required_independent_lineage_count": 2,
            "passes_d2_task_data_gate": len(records) >= 3 and len(explicit_lineage_ids) >= 2,
            "sampling_probability_sum": float(sum(record["sampling"]["episode_sampling_probability"] for record in records)),
            "historical_h300_expected_examples": {
                f"steps_{steps}": float(sum(record["sampling"]["historical_h300_expected_examples"][f"steps_{steps}"] for record in records))
                for steps in EXPOSURE_STEPS
            },
            "planned_d4_expected_examples": {
                f"steps_{steps}": float(sum(record["sampling"]["planned_d4_expected_examples"][f"steps_{steps}"] for record in records))
                for steps in EXPOSURE_STEPS
            },
        }

    source_summary = {
        "track3_reverse": {
            "episode_count": len(reverse_records),
            "task_count": len(task_summary),
            "frame_count": int(sum(int(record["frame_count"]) for record in reverse_records)),
            "sampling_probability": float(sum(record["sampling"]["episode_sampling_probability"] for record in reverse_records)),
        },
        "libero_plus_forward_train": {
            "episode_count": len(forward_train),
            "frame_count": int(sum(int(row["frame_count"]) for row in forward_train)),
            "sampling_probability": float(sum(float(row["episode_sampling_probability"]) for row in forward_sampler)),
        },
    }
    for source, record in source_summary.items():
        probability = float(record["sampling_probability"])
        record["historical_h300_expected_examples"] = _exposure_counts(probability, HISTORICAL_GLOBAL_BATCH)
        record["planned_d4_expected_examples"] = _exposure_counts(probability, PLANNED_D4_GLOBAL_BATCH)

    return {
        "schema_version": "track3_d0_reverse_lineage_sampling_audit_v1",
        "fast400_manifest": {
            "path": str(MANIFEST_JSON.resolve()),
            "sha256": sha256_file(MANIFEST_JSON),
            "reverse_episode_count": manifest.get("reverse_episode_count"),
            "reverse_task_count": manifest.get("reverse_task_count"),
            "reverse_frame_count": manifest.get("reverse_frame_count"),
            "reverse_selection": manifest.get("reverse_selection"),
            "sampling_rows": manifest.get("sampling_rows"),
            "forward_train_episode_count": manifest.get("forward_train_episode_count"),
            "forward_validation_episode_count": manifest.get("forward_validation_episode_count"),
        },
        "source_roots_sha256": sha256_file(SOURCE_ROOTS),
        "sampler_audit_sha256": sha256_file(SAMPLER_AUDIT),
        "sampler_audit_claims": {
            "episode_probability_sum": sampler_audit.get("episode_probability_sum"),
            "frame_weight_mass_sum": sampler_audit.get("frame_weight_mass_sum"),
            "validation_frames_included": sampler_audit.get("validation_frames_included"),
            "source_mass": sampler_audit.get("source_mass"),
        },
        "reverse_records": reverse_records,
        "task_summary": task_summary,
        "source_summary": source_summary,
        "duplicate_hash_groups": duplicate_fields,
        "near_duplicate_definition": {
            "scope": "same task only",
            "resample_points": 128,
            "action_rmse_max": NEAR_DUPLICATE_RMSE_THRESHOLD,
            "mean_l2_max": NEAR_DUPLICATE_MEAN_L2_THRESHOLD,
            "interpretation": "heuristic diagnostic only; not a promotion criterion",
        },
        "near_duplicate_pairs": near_duplicate_pairs,
        "train_dev_leakage": {
            "forward_train_episode_count": len(forward_train),
            "forward_dev_episode_count": len(forward_dev),
            "forward_train_dev_identity_overlap_count": len(forward_train_ids & forward_dev_ids),
            "reverse_train_episode_count": len(reverse_rows),
            "reverse_dev_path": str(reverse_dev_path.resolve()),
            "reverse_dev_exists": reverse_dev_path.exists(),
            "reverse_train_dev_split_status": "not_constructed",
            "reverse_vs_forward_train_identity_overlap_count": len(reverse_ids & forward_train_ids),
            "reverse_vs_forward_dev_identity_overlap_count": len(reverse_ids & forward_dev_ids),
            "reverse_vs_forward_exact_task_text_overlap_count": len(reverse_task_texts & forward_task_texts),
            "direct_identity_leakage_detected": False,
            "gate_passed": False,
            "reason": "Fast400 has reverse_train only; no reverse dev lineage-disjoint panel exists. The eight accepted records also declare training_holdout=false and official_public_example=true, which is provenance context rather than proof of evaluation leakage.",
        },
        "lineage_gate": {
            "reverse_episode_count": len(reverse_records),
            "task_count": len(task_summary),
            "explicit_lineage_field_count": sum(len(record["recording_lineage"]["explicit_fields"]) for record in reverse_records),
            "provable_independent_lineages_total": len({record["recording_lineage"]["explicit_lineage_id"] for record in reverse_records if record["recording_lineage"]["explicit_lineage_id"] is not None}),
            "every_target_task_has_three_accepted": all(item["accepted_episode_count"] >= 3 for item in task_summary.values()),
            "every_target_task_has_two_provable_lineages": all(item["provable_independent_lineage_count"] >= 2 for item in task_summary.values()),
            "gate_passed": False,
            "reason": "No explicit recording_lineage, recording_id, capture_id, or session_id was present in the eight metadata records; distinct episode names/timestamps are not promoted to independent lineage evidence.",
        },
        "sampling_contract": {
            "nominal_manifest_draws": nominal_epoch_draws,
            "historical_h300_effective_global_batch": HISTORICAL_GLOBAL_BATCH,
            "historical_batch_source": str(CLOSEOUT_SUMMARY.resolve()),
            "historical_batch_source_sha256": sha256_file(CLOSEOUT_SUMMARY),
            "planned_d4_global_batch": PLANNED_D4_GLOBAL_BATCH,
            "optimizer_steps_reported": list(EXPOSURE_STEPS),
            "expected_count_model": "probability * independent sampling draws; historical H300 rows used batch 64, future D4 plan specifies batch 32",
            "forward_episode_probability_median": forward_median,
            "reverse_stored_effective_exposures_recomputed": True,
            "stored_reverse_effective_exposure_formula_is_inconsistent_with_nominal_epoch": True,
        },
        "counts": {
            "reverse_rows": len(reverse_rows),
            "reverse_tasks": len(task_summary),
            "forward_train_rows": len(forward_train),
            "forward_dev_rows": len(forward_dev),
            "sampling_rows": len(sampling_rows),
        },
    }


def _c20_flips() -> dict[str, Any]:
    cases = {role: _case_files(root, "C20") for role, root in ROLE_CASE_ROOTS.items()}
    ids = set(cases["r6"])
    _require(ids == set(cases["step100"]) == set(cases["step300"]), "C20 case sets are not aligned")
    _require(len(ids) == 20, f"expected 20 C20 cases, got {len(ids)}")
    outcomes: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    for case_id in sorted(ids):
        values = {role: bool(cases[role][case_id].get("success")) for role in ("r6", "step100", "step300")}
        row = {
            "case_id": case_id,
            "task_id": cases["r6"][case_id].get("task_id"),
            "task_name": cases["r6"][case_id].get("task_name"),
            **values,
        }
        outcomes.append(row)
        if len(set(values.values())) > 1:
            flips.append({"flip": row, "pattern": [values[role] for role in ("r6", "step100", "step300")]})
    return {
        "schema_version": "track3_d0_c20_flip_analysis_v1",
        "success_counts": {role: sum(int(row[role]) for row in outcomes) for role in ("r6", "step100", "step300")},
        "case_count": len(outcomes),
        "success_flips": flips,
        "all_case_outcomes": outcomes,
    }


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def analyze(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    input_paths = [R6_EVIDENCE, STEP100_EVIDENCE, R6_OUTPUT, STEP100_OUTPUT, REVERSE_ACCEPTED, REVERSE_PROCESSED, FAST400_MANIFEST, ACTIVE_PLAN, LATEST_AUDIT, MACHINE_HANDOFF, CLOSEOUT_SUMMARY]
    for path in input_paths:
        _require(output_root != path.resolve() and path.resolve() not in output_root.parents, f"D0 output collides with immutable input: {output_root}")
    _require(not output_root.exists(), f"refusing to overwrite D0 evidence root: {output_root}")
    partial = output_root.with_name(output_root.name + ".partial")
    _require(not partial.exists(), f"refusing to reuse D0 partial root: {partial}")

    before = _snapshot_inputs()
    strict_json_count = _strict_json_count(before.values())
    trajectories = analyze_trajectories()
    c20 = _c20_flips()
    closeout = strict_json(CLOSEOUT_SUMMARY)
    data_audit = _data_audit(closeout)
    after_fingerprints: dict[str, dict[str, Any]] = {}
    for label, inventory in before.items():
        fingerprint, file_count, byte_count = _fingerprint(Path(str(inventory["root"])))
        after_fingerprints[label] = {
            "fingerprint_sha256_after": fingerprint,
            "file_count_after": file_count,
            "bytes_after": byte_count,
            "unchanged": fingerprint == inventory["fingerprint_sha256"] and file_count == inventory["file_count"] and byte_count == inventory["bytes"],
        }
    no_write = all(item["unchanged"] for item in after_fingerprints.values())
    script_sha = sha256_file(Path(__file__).resolve())
    summary = {
        "schema_version": "track3_d0_failure_taxonomy_and_lineage_inventory_summary_v1",
        "status": "d0_read_only_analysis_complete_hard_stop",
        "analysis_scope": "R6/step100/step300 immutable trajectory alignment, C20 flip, and eight Fast400 reverse data lineage/sampling inventory",
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": script_sha,
        "active_plan": str(ACTIVE_PLAN.resolve()),
        "active_plan_sha256": sha256_file(ACTIVE_PLAN),
        "latest_audit": str(LATEST_AUDIT.resolve()),
        "latest_audit_sha256": sha256_file(LATEST_AUDIT),
        "root_git_head": _git_revision(ROOT),
        "nested_openpi_git_head": _git_revision(ROOT / "third_party/axis_openpi"),
        "strict_json_files_checked": strict_json_count,
        "trajectory_analysis": "trajectory_analysis.json",
        "c20_flip_analysis": "c20_flip_analysis.json",
        "reverse_data_audit": "reverse_data_audit.json",
        "input_inventory": "input_inventory.json",
        "input_namespace_no_write_check": {
            "passed": no_write,
            "namespaces": after_fingerprints,
        },
        "behavior": {
            "reverse_r6_step100_step300": trajectories["roles"],
            "c20_success_counts": c20["success_counts"],
            "c20_flip_count": len(c20["success_flips"]),
            "known_flip_case_ids": [item["flip"]["case_id"] for item in c20["success_flips"]],
        },
        "data": {
            "reverse_episode_count": data_audit["counts"]["reverse_rows"],
            "reverse_task_count": data_audit["counts"]["reverse_tasks"],
            "accepted_by_task": {task: value["accepted_episode_count"] for task, value in data_audit["task_summary"].items()},
            "provable_lineages_by_task": {task: value["provable_independent_lineage_count"] for task, value in data_audit["task_summary"].items()},
            "reverse_dev_exists": data_audit["train_dev_leakage"]["reverse_dev_exists"],
            "direct_identity_leakage_detected": data_audit["train_dev_leakage"]["direct_identity_leakage_detected"],
            "duplicate_hash_groups": data_audit["duplicate_hash_groups"],
        },
        "gates": {
            "four_reverse_cases_aligned": trajectories["reverse_case_count"] == 4,
            "c20_cases_aligned": c20["case_count"] == 20,
            "no_semantic_stage_inference": True,
            "all_reverse_payload_hashes_unique": not data_audit["duplicate_hash_groups"]["raw_trajectory_sha256"],
            "all_reverse_start_state_hashes_unique": not data_audit["duplicate_hash_groups"]["start_state_sha256"],
            "forward_train_dev_identity_overlap_zero": data_audit["train_dev_leakage"]["forward_train_dev_identity_overlap_count"] == 0,
            "reverse_dev_lineage_disjoint_panel_present": data_audit["train_dev_leakage"]["reverse_dev_exists"],
            "every_task_has_three_accepted": data_audit["lineage_gate"]["every_target_task_has_three_accepted"],
            "every_task_has_two_provable_lineages": data_audit["lineage_gate"]["every_target_task_has_two_provable_lineages"],
            "sampling_exposure_recomputed": data_audit["sampling_contract"]["reverse_stored_effective_exposures_recomputed"],
            "d2_data_gate_passed": False,
        },
        "authorization_boundary": {
            "simulator_started": False,
            "policy_server_started": False,
            "gpu_started": False,
            "training_started": False,
            "evaluation_started": False,
            "data_collection_started": False,
            "promotion_started": False,
            "d1_started": False,
            "d3_to_d6_started": False,
            "fold_fast_zip_upload_submission": False,
            "hard_stop_before_d1": True,
        },
        "input_hashes": {
            "machine_handoff_sha256": sha256_file(MACHINE_HANDOFF),
            "fast400_manifest_sha256": sha256_file(MANIFEST_JSON),
            "fast400_sampling_weights_sha256": sha256_file(SAMPLING_WEIGHTS),
            "reverse_provenance_sha256": sha256_file(REVERSE_PROVENANCE),
            "h300_closeout_summary_sha256": sha256_file(CLOSEOUT_SUMMARY),
        },
    }
    payloads = {
        "trajectory_analysis.json": trajectories,
        "c20_flip_analysis.json": c20,
        "reverse_data_audit.json": data_audit,
        "input_inventory.json": {"schema_version": "track3_d0_input_inventory_v1", "namespaces": before},
        "summary.json": summary,
    }
    partial.mkdir(parents=True)
    try:
        for name, value in payloads.items():
            write_json(partial / name, _jsonable(value))
        output_manifest = {
            "schema_version": "track3_d0_output_manifest_v1",
            "files": {name: sha256_file(partial / name) for name in sorted(payloads)},
        }
        write_json(partial / "output_manifest.json", output_manifest)
        os.replace(partial, output_root)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="D0 read-only Track3 Fast400 failure/lineage/sampling analysis")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/reports/axis_libero_30k/t3_post_salvage_failure_analysis_20260916_attempt_0001")
    args = parser.parse_args()
    result = analyze(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
