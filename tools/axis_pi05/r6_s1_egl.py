"""Lock and gate real Q1 EGL image pairs through the R6 folded deployment path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_http_server import _resize_for_openpi
from .common import ROOT, sha256_file, write_json
from .n1_snapshot import _policy_state
from .r6_s1_frozen import DEFAULT_OUTPUT, TOKENIZER_MODEL, _diff_summary


Q1_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q1_environment_full_c20"
S1_ROOT = DEFAULT_OUTPUT
PAIR_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_folded_submission/s1_real_egl_pairs"
E0_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_egl_statistical_release"
CLOSURE_ROOT = E0_ROOT / "e0_closure"
OLD_S1_SUMMARY = PAIR_ROOT / "summary.json"
OLD_S1_MANIFEST = PAIR_ROOT / "pair_manifest.json"
OLD_S1_REPORT = PAIR_ROOT / "S1_REAL_EGL_REPORT_ZH.md"
OLD_S1_RUN_REPORT = PAIR_ROOT / "e2e_gpu1/e2e_action_parity_report.json"
OLD_S1_RUN_ARTIFACT = PAIR_ROOT / "e2e_gpu1/e2e_action_parity.npz"
BASE_INPUT = S1_ROOT / "frozen_input.npz"
BASE_TASKS = S1_ROOT / "tasks.json"
NORM_STATS = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
C20_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_TO_INPUT = {
    "agentview_image": "base_0_rgb",
    "robot0_eye_in_hand_image": "left_wrist_0_rgb",
}
REPLICA_NAMES = ("replica0", "replica1", "replica2")
REPLICA_PAIR_SPECS = (
    ("replica0", "replica1", "replica0_vs_replica1.json"),
    ("replica0", "replica2", "replica0_vs_replica2.json"),
    ("replica1", "replica2", "replica1_vs_replica2.json"),
)
SNAPSHOT_PHASES = ("E1", "E2", *(f"E3-{step}" for step in range(1, 11)))
TRAJECTORY_MODES = ("zero64", "fixed64")
TRAJECTORY_REQUIRED_ARRAYS = (
    "sim_state_flattened",
    "qpos",
    "qvel",
    "ctrl",
    "obs:robot0_eef_pos",
    "obs:robot0_eef_quat",
    "obs:robot0_gripper_qpos",
    *(f"obs:{camera}" for camera in CAMERAS),
)
LOCKED_OLD_S1_SUMMARY_SHA256 = "96f6c55f2b42ea55d3e103820e52912a4fcfad79794b686411e0e1659ca1a952"
LOCKED_OLD_S1_MANIFEST_SHA256 = "dacb81bcc6216c0b86be9ff350d731cdc199612b5e8f6f662560598cd588e972"


def _raw_descriptor(snapshot: dict[str, Any], camera: str) -> dict[str, Any] | None:
    value = snapshot.get("observation", {}).get(camera)
    return value if isinstance(value, dict) and value.get("path") else None


def _snapshot_maps() -> list[dict[tuple[str, str], dict[str, Any]]]:
    maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    for replica in REPLICA_NAMES:
        path = Q1_ROOT / replica / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for case in summary["records"][0]["cases"]:
            case_id = str(case["case_id"])
            for snapshot in case["snapshots"]:
                stage = str(snapshot.get("stage"))
                if stage == "E3":
                    phase = f"E3-{int(snapshot['warmup_step'])}"
                elif stage in {"E1", "E2"}:
                    phase = stage
                else:
                    continue
                result[(case_id, phase)] = snapshot
        maps.append(result)
    return maps


def _transformed_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(np.load(path, allow_pickle=False))
    transformed_uint8 = _resize_for_openpi(raw)
    transformed_float = transformed_uint8.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
    return raw, transformed_float


def _image_metrics(left_raw: np.ndarray, right_raw: np.ndarray) -> dict[str, Any]:
    left_raw = np.asarray(left_raw)
    right_raw = np.asarray(right_raw)
    left_transformed_uint8 = _resize_for_openpi(left_raw)
    right_transformed_uint8 = _resize_for_openpi(right_raw)
    left_transformed = left_transformed_uint8.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
    right_transformed = right_transformed_uint8.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
    raw_diff = np.abs(left_raw.astype(np.int16) - right_raw.astype(np.int16))
    transformed_diff = np.abs(left_transformed.astype(np.float64) - right_transformed.astype(np.float64))
    return {
        "raw": {
            "max_abs_diff": float(np.max(raw_diff)),
            "changed_element_count": int(np.count_nonzero(raw_diff)),
            "changed_pixel_count": int(np.count_nonzero(np.any(raw_diff != 0, axis=-1))),
            "mean_abs_diff": float(np.mean(raw_diff)),
            "p99_abs_diff": float(np.percentile(raw_diff, 99)),
        },
        "transformed_float32_minus1_plus1": {
            "max_abs_diff": float(np.max(transformed_diff)),
            "changed_element_count": int(np.count_nonzero(transformed_diff != 0)),
            "changed_pixel_count": int(np.count_nonzero(np.any(transformed_diff != 0, axis=-1))),
            "mean_abs_diff": float(np.mean(transformed_diff)),
            "p99_abs_diff": float(np.percentile(transformed_diff, 99)),
        },
    }


def _pair_metrics(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_raw, _ = _transformed_image(left_path)
    right_raw, _ = _transformed_image(right_path)
    return _image_metrics(left_raw, right_raw)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _corrected_reference_label(name: str) -> str:
    prefix = "axis_reference_jax_"
    if name.startswith(prefix):
        return "r6_folded_jax_reference_" + name[len(prefix) :]
    return name


def _array_from_descriptor_local(descriptor: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(descriptor, dict) or not descriptor.get("path"):
        return None
    path = Path(str(descriptor["path"]))
    if not path.is_file():
        return None
    try:
        return np.asarray(np.load(path, allow_pickle=False))
    except (OSError, ValueError):
        return None


def _frame_descriptor(array: np.ndarray | None, frame_index: int) -> dict[str, Any] | None:
    if array is None or array.ndim == 0 or frame_index < 0 or frame_index >= array.shape[0]:
        return None
    value = np.asarray(array[frame_index])
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data_sha256": _sha256_array(value),
    }


def _exact_array_pair(left: np.ndarray | None, right: np.ndarray | None) -> dict[str, Any]:
    present_left = left is not None
    present_right = right is not None
    if not present_left or not present_right:
        if not present_left and not present_right:
            reason = "missing_both"
        elif not present_left:
            reason = "missing_left"
        else:
            reason = "missing_right"
        return {
            "present_left": present_left,
            "present_right": present_right,
            "exact": False,
            "evaluable": False,
            "shape_match": False,
            "dtype_match": False,
            "max_abs_diff": None,
            "reason": reason,
        }
    shape_match = left.shape == right.shape
    dtype_match = left.dtype == right.dtype
    if not shape_match:
        return {
            "present_left": True,
            "present_right": True,
            "exact": False,
            "evaluable": False,
            "shape_match": False,
            "dtype_match": dtype_match,
            "max_abs_diff": None,
            "reason": "shape_mismatch",
        }
    if not dtype_match:
        return {
            "present_left": True,
            "present_right": True,
            "exact": False,
            "evaluable": False,
            "shape_match": True,
            "dtype_match": False,
            "max_abs_diff": None,
            "reason": "dtype_mismatch",
        }
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
        max_abs_diff = float(np.max(diff)) if diff.size else 0.0
    else:
        max_abs_diff = None
    return {
        "present_left": True,
        "present_right": True,
        "exact": bool(np.array_equal(left, right)),
        "evaluable": True,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "max_abs_diff": max_abs_diff,
        "reason": None,
    }


def _snapshot_key(stage: str, warmup_step: int | None = None) -> str:
    if stage == "E3":
        return f"E3-{int(warmup_step)}"
    return stage


def _snapshot_maps_for_reanalysis() -> list[dict[str, dict[str, Any]]]:
    result: list[dict[str, dict[str, Any]]] = []
    for replica in REPLICA_NAMES:
        summary_path = Q1_ROOT / replica / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result.append({str(case["case_id"]): case for case in summary["records"][0]["cases"]})
    return result


def _snapshot_by_phase(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for snapshot in case.get("snapshots", []):
        stage = str(snapshot.get("stage"))
        if stage not in {"E1", "E2", "E3"}:
            continue
        if stage == "E3" and snapshot.get("warmup_step") not in range(1, 11):
            continue
        result[_snapshot_key(stage, snapshot.get("warmup_step"))] = snapshot
    return result


def _coverage_group_key(source_comparison: str, case_id: str, phase_or_mode: str, camera: str) -> str:
    return "|".join((source_comparison, case_id, phase_or_mode, camera))


def _coverage_issue(issues: list[dict[str, Any]], code: str, **details: Any) -> None:
    issues.append({"code": code, **details})


def _checked_descriptor_array(
    descriptor: Any,
    *,
    issues: list[dict[str, Any]],
    location: str,
) -> np.ndarray | None:
    if not isinstance(descriptor, dict) or not descriptor.get("path"):
        _coverage_issue(issues, "missing_image_or_evidence_descriptor", location=location)
        return None
    path = Path(str(descriptor["path"]))
    if not path.is_file():
        _coverage_issue(issues, "missing_image_or_evidence_path", location=location, path=str(path))
        return None
    try:
        return np.asarray(np.load(path, allow_pickle=False))
    except (OSError, ValueError, EOFError, TypeError) as exc:
        _coverage_issue(issues, "unreadable_image_or_evidence_array", location=location, path=str(path), error=str(exc))
        return None


def _comparison_metric(
    mapping: Any,
    key: str,
    *,
    issues: list[dict[str, Any]],
    location: str,
) -> dict[str, Any] | None:
    if not isinstance(mapping, dict) or key not in mapping:
        _coverage_issue(issues, "comparison_group_missing_or_schema_error", location=location, group_key=key)
        return None
    metric = mapping[key]
    if not isinstance(metric, dict) or not isinstance(metric.get("exact"), bool):
        _coverage_issue(issues, "comparison_group_missing_or_schema_error", location=location, group_key=key)
        return None
    return metric


def _metric_changed_frame_count(metric: dict[str, Any], *, issues: list[dict[str, Any]], location: str) -> int | None:
    count = metric.get("affected_frame_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        _coverage_issue(issues, "comparison_metric_missing_changed_frame_count", location=location)
        return None
    return count


def _validate_snapshot_coverage(
    *,
    comparison: dict[str, Any],
    left_cases: dict[str, dict[str, Any]],
    right_cases: dict[str, dict[str, Any]],
    source_comparison: str,
    expected: set[str],
    observed: set[str],
    issues: list[dict[str, Any]],
) -> None:
    cases = comparison.get("cases")
    if not isinstance(cases, list):
        _coverage_issue(issues, "comparison_cases_missing_or_schema_error", source_comparison=source_comparison)
        return
    comparison_case_ids: set[str] = set()
    for comparison_case in cases:
        if not isinstance(comparison_case, dict) or not comparison_case.get("case_id"):
            _coverage_issue(issues, "comparison_case_missing_or_schema_error", source_comparison=source_comparison)
            continue
        case_id = str(comparison_case["case_id"])
        comparison_case_ids.add(case_id)
        left_case = left_cases.get(case_id)
        right_case = right_cases.get(case_id)
        if left_case is None:
            _coverage_issue(issues, "comparison_case_missing_in_replica", source_comparison=source_comparison, case_id=case_id, side="left")
        if right_case is None:
            _coverage_issue(issues, "comparison_case_missing_in_replica", source_comparison=source_comparison, case_id=case_id, side="right")
        renderer_mapping = comparison_case.get("renderer_exact_by_camera_and_stage")
        trajectory_mapping = comparison_case.get("trajectory_renderer_exact_by_camera_and_stage")
        for phase in SNAPSHOT_PHASES:
            for camera in CAMERAS:
                group_key = f"{camera}:{phase}"
                location = f"{source_comparison}:{case_id}:{group_key}"
                metric = _comparison_metric(renderer_mapping, group_key, issues=issues, location=location)
                if metric is None:
                    continue
                is_non_exact = not metric["exact"]
                coverage_key = _coverage_group_key(source_comparison, case_id, phase, camera)
                if is_non_exact:
                    expected.add(coverage_key)
                    changed_count = _metric_changed_frame_count(metric, issues=issues, location=location)
                    if changed_count is not None and changed_count == 0:
                        _coverage_issue(issues, "source_non_exact_without_changed_frame", location=location)

                if left_case is None or right_case is None:
                    continue
                left_snapshots = _snapshot_by_phase(left_case)
                right_snapshots = _snapshot_by_phase(right_case)
                left_snapshot = left_snapshots.get(phase)
                right_snapshot = right_snapshots.get(phase)
                if left_snapshot is None:
                    _coverage_issue(issues, "expected_snapshot_phase_missing", location=location, side="left")
                if right_snapshot is None:
                    _coverage_issue(issues, "expected_snapshot_phase_missing", location=location, side="right")
                if left_snapshot is None or right_snapshot is None:
                    continue
                left_observation = left_snapshot.get("observation")
                right_observation = right_snapshot.get("observation")
                if not isinstance(left_observation, dict):
                    _coverage_issue(issues, "snapshot_observation_missing_or_schema_error", location=location, side="left")
                    left_observation = {}
                if not isinstance(right_observation, dict):
                    _coverage_issue(issues, "snapshot_observation_missing_or_schema_error", location=location, side="right")
                    right_observation = {}
                left_image = _checked_descriptor_array(left_observation.get(camera), issues=issues, location=f"{location}:left:image")
                right_image = _checked_descriptor_array(right_observation.get(camera), issues=issues, location=f"{location}:right:image")
                image_evidence = _exact_array_pair(left_image, right_image)
                if not image_evidence["evaluable"]:
                    _coverage_issue(issues, "snapshot_image_pair_not_evaluable", location=location, evidence=image_evidence)
                elif is_non_exact and not image_evidence["exact"]:
                    observed.add(coverage_key)
                elif is_non_exact:
                    _coverage_issue(issues, "source_non_exact_but_image_pair_exact", location=location)
                elif not image_evidence["exact"]:
                    _coverage_issue(issues, "source_exact_but_image_pair_non_exact", location=location)

                for evidence_name, descriptor_key in (("left", camera), ("right", camera)):
                    observation = left_observation if evidence_name == "left" else right_observation
                    if descriptor_key not in observation:
                        _coverage_issue(issues, "missing_image_descriptor", location=f"{location}:{evidence_name}:{descriptor_key}")

                other_camera = next(value for value in CAMERAS if value != camera)
                other_left = _checked_descriptor_array(left_observation.get(other_camera), issues=issues, location=f"{location}:left:{other_camera}")
                other_right = _checked_descriptor_array(right_observation.get(other_camera), issues=issues, location=f"{location}:right:{other_camera}")
                other_evidence = _exact_array_pair(other_left, other_right)
                if not other_evidence["evaluable"]:
                    _coverage_issue(issues, "other_camera_not_evaluable", location=location, evidence=other_evidence)

                dynamics = _snapshot_dynamics_evidence(left_snapshot, right_snapshot)
                for evidence_name, evidence in dynamics.items():
                    if evidence_name == "exact":
                        continue
                    if not evidence["evaluable"]:
                        _coverage_issue(issues, "missing_or_invalid_snapshot_dynamics_evidence", location=f"{location}:{evidence_name}", evidence=evidence)

        if not isinstance(trajectory_mapping, dict):
            _coverage_issue(issues, "trajectory_comparison_groups_missing_or_schema_error", source_comparison=source_comparison, case_id=case_id)
    if comparison_case_ids != set(left_cases):
        _coverage_issue(issues, "comparison_case_set_mismatch", source_comparison=source_comparison, side="left")
    if comparison_case_ids != set(right_cases):
        _coverage_issue(issues, "comparison_case_set_mismatch", source_comparison=source_comparison, side="right")


def _validated_trajectory_arrays(
    case: dict[str, Any],
    trajectory_name: str,
    *,
    issues: list[dict[str, Any]],
    location: str,
) -> tuple[dict[str, np.ndarray], int | None]:
    trajectories = case.get("trajectories")
    if not isinstance(trajectories, dict) or not isinstance(trajectories.get(trajectory_name), dict):
        _coverage_issue(issues, "trajectory_missing_or_schema_error", location=location)
        return {}, None
    trajectory = trajectories[trajectory_name]
    declared_length = trajectory.get("length")
    if not isinstance(declared_length, int) or isinstance(declared_length, bool) or declared_length < 0:
        _coverage_issue(issues, "trajectory_declared_length_missing_or_invalid", location=location)
        declared_length = None
    descriptors = trajectory.get("arrays")
    if not isinstance(descriptors, dict):
        _coverage_issue(issues, "trajectory_arrays_missing_or_schema_error", location=location)
        return {}, declared_length
    arrays: dict[str, np.ndarray] = {}
    for key in TRAJECTORY_REQUIRED_ARRAYS:
        array = _checked_descriptor_array(descriptors.get(key), issues=issues, location=f"{location}:{key}")
        if array is None:
            continue
        if array.ndim == 0:
            _coverage_issue(issues, "trajectory_array_has_no_frame_axis", location=f"{location}:{key}")
            continue
        if declared_length is not None and array.shape[0] != declared_length:
            _coverage_issue(
                issues,
                "trajectory_length_mismatch",
                location=f"{location}:{key}",
                declared_length=declared_length,
                observed_length=int(array.shape[0]),
            )
        arrays[key] = array
    return arrays, declared_length


def _validate_trajectory_coverage(
    *,
    comparison: dict[str, Any],
    left_cases: dict[str, dict[str, Any]],
    right_cases: dict[str, dict[str, Any]],
    source_comparison: str,
    expected: set[str],
    observed: set[str],
    issues: list[dict[str, Any]],
) -> None:
    cases = comparison.get("cases")
    if not isinstance(cases, list):
        return
    camera_names = {
        "agentview": "agentview_image",
        "robot0_eye_in_hand": "robot0_eye_in_hand_image",
    }
    for comparison_case in cases:
        if not isinstance(comparison_case, dict) or not comparison_case.get("case_id"):
            continue
        case_id = str(comparison_case["case_id"])
        left_case = left_cases.get(case_id)
        right_case = right_cases.get(case_id)
        mapping = comparison_case.get("trajectory_renderer_exact_by_camera_and_stage")
        for trajectory_name in TRAJECTORY_MODES:
            left_arrays: dict[str, np.ndarray] = {}
            right_arrays: dict[str, np.ndarray] = {}
            left_length: int | None = None
            right_length: int | None = None
            if left_case is not None:
                left_arrays, left_length = _validated_trajectory_arrays(
                    left_case,
                    trajectory_name,
                    issues=issues,
                    location=f"{source_comparison}:{case_id}:{trajectory_name}:left",
                )
            else:
                _coverage_issue(issues, "comparison_case_missing_in_replica", source_comparison=source_comparison, case_id=case_id, side="left")
            if right_case is not None:
                right_arrays, right_length = _validated_trajectory_arrays(
                    right_case,
                    trajectory_name,
                    issues=issues,
                    location=f"{source_comparison}:{case_id}:{trajectory_name}:right",
                )
            else:
                _coverage_issue(issues, "comparison_case_missing_in_replica", source_comparison=source_comparison, case_id=case_id, side="right")
            if left_length is not None and right_length is not None and left_length != right_length:
                _coverage_issue(
                    issues,
                    "trajectory_replica_length_mismatch",
                    source_comparison=source_comparison,
                    case_id=case_id,
                    trajectory=trajectory_name,
                    left_length=left_length,
                    right_length=right_length,
                )
            for camera_alias, camera in camera_names.items():
                group_key = f"{trajectory_name}:{camera_alias}"
                location = f"{source_comparison}:{case_id}:{group_key}"
                metric = _comparison_metric(mapping, group_key, issues=issues, location=location)
                if metric is None:
                    continue
                is_non_exact = not metric["exact"]
                coverage_key = _coverage_group_key(source_comparison, case_id, trajectory_name, camera)
                if is_non_exact:
                    expected.add(coverage_key)
                    changed_count = _metric_changed_frame_count(metric, issues=issues, location=location)
                    if changed_count is not None and changed_count == 0:
                        _coverage_issue(issues, "source_non_exact_without_changed_frame", location=location)
                if left_length is not None and metric.get("total_frame_count") not in (None, left_length):
                    _coverage_issue(issues, "trajectory_length_mismatch", location=location, metric_length=metric.get("total_frame_count"), declared_length=left_length)
                left_image = left_arrays.get(f"obs:{camera}")
                right_image = right_arrays.get(f"obs:{camera}")
                image_evidence = _exact_array_pair(left_image, right_image)
                if not image_evidence["evaluable"]:
                    _coverage_issue(issues, "trajectory_image_pair_not_evaluable", location=location, evidence=image_evidence)
                    continue
                if is_non_exact:
                    changed_frames = [
                        frame_index
                        for frame_index in range(min(left_image.shape[0], right_image.shape[0]))
                        if not np.array_equal(left_image[frame_index], right_image[frame_index])
                    ]
                    if changed_frames:
                        observed.add(coverage_key)
                    else:
                        _coverage_issue(issues, "source_non_exact_but_no_changed_frame", location=location)
                elif not image_evidence["exact"]:
                    _coverage_issue(issues, "source_exact_but_image_pair_non_exact", location=location)


def _coverage_set_audit(expected: set[str], observed: set[str], issues: list[dict[str, Any]]) -> dict[str, Any]:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    return {
        "expected": sorted(expected),
        "observed": sorted(observed),
        "missing": missing,
        "extra": extra,
        "complete": not missing and not extra and not issues,
    }


def _trajectory_policy_state(arrays: dict[str, np.ndarray], frame_index: int) -> np.ndarray | None:
    required = (
        "obs:robot0_eef_pos",
        "obs:robot0_eef_quat",
        "obs:robot0_gripper_qpos",
    )
    if any(key not in arrays for key in required):
        return None
    return _policy_state(
        {
            "robot0_eef_pos": arrays["obs:robot0_eef_pos"][frame_index],
            "robot0_eef_quat": arrays["obs:robot0_eef_quat"][frame_index],
            "robot0_gripper_qpos": arrays["obs:robot0_gripper_qpos"][frame_index],
        }
    )


def _semantic_key(item: dict[str, Any]) -> tuple[str, str, str, str, str, tuple[str, str]]:
    return (
        str(item["case_id"]),
        str(item["phase_or_trajectory_frame"]),
        str(item["camera"]),
        str(item["policy_state_hash"]),
        str(item["prompt_hash"]),
        tuple(sorted((str(item["left_image_data_sha256"]), str(item["right_image_data_sha256"]))),
        ),
    )


def _semantic_key_record(key: tuple[str, str, str, str, str, tuple[str, str]]) -> dict[str, Any]:
    return {
        "case_id": key[0],
        "phase_or_trajectory_frame": key[1],
        "camera": key[2],
        "policy_state_hash": key[3],
        "prompt_hash": key[4],
        "canonical_image_data_sha256_pair": list(key[5]),
    }


def _dedupe_semantic_pairs(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, str, str, tuple[str, str]], dict[str, Any]] = {}
    for candidate in candidates:
        key = _semantic_key(candidate)
        occurrence = {
            "source_comparison": candidate["source_comparison"],
            "left_replica": candidate["left_replica"],
            "right_replica": candidate["right_replica"],
            "left_image_path": candidate["left_image_path"],
            "right_image_path": candidate["right_image_path"],
            "left_image_data_sha256": candidate["left_image_data_sha256"],
            "right_image_data_sha256": candidate["right_image_data_sha256"],
            "policy_state_hash": candidate["policy_state_hash"],
            "prompt_hash": candidate["prompt_hash"],
            "frame_index": candidate.get("frame_index"),
        }
        occurrence["source_occurrence_hash"] = _sha256_text(
            json.dumps(occurrence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        if key not in by_key:
            item = dict(candidate)
            item["semantic_key"] = _semantic_key_record(key)
            item["source_occurrences"] = [occurrence]
            by_key[key] = item
            unique.append(item)
        else:
            target = by_key[key]
            target.setdefault("source_occurrences", []).append(occurrence)
            duplicates.append(
                {
                    "semantic_key": _semantic_key_record(key),
                    "duplicate_source": occurrence,
                    "canonical_source": target["source_occurrences"][0],
                }
            )
    return unique, duplicates


def _content_hash(candidate: dict[str, Any]) -> str:
    return _sha256_text(
        json.dumps(_semantic_key_record(_semantic_key(candidate)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _content_unique_manifest(
    candidates: list[dict[str, Any]],
    *,
    source_reanalysis: str,
    status: str,
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for candidate in candidates:
        item = {
            "case_id": candidate["case_id"],
            "phase_or_trajectory_frame": candidate["phase_or_trajectory_frame"],
            "camera": candidate["camera"],
            "source_kind": candidate["source_kind"],
            "frame_index": candidate.get("frame_index"),
            "prompt_hash": candidate["prompt_hash"],
            "policy_state_hash": candidate["policy_state_hash"],
            "left_image_data_sha256": candidate["left_image_data_sha256"],
            "right_image_data_sha256": candidate["right_image_data_sha256"],
            "content_hash": _content_hash(candidate),
            "source_occurrences": candidate.get("source_occurrences", []),
            "semantic_key": candidate["semantic_key"],
        }
        if candidate["source_kind"] == "trajectory" and not isinstance(item["frame_index"], int):
            raise ValueError("trajectory content-unique manifest item is missing frame_index")
        item["source_occurrence_hashes"] = [
            occurrence["source_occurrence_hash"] for occurrence in item["source_occurrences"]
        ]
        pairs.append(item)
    return {
        "schema_version": 1,
        "phase": "E0",
        "status": status,
        "read_only": True,
        "source_reanalysis": source_reanalysis,
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def _candidate_common(
    *,
    case_id: str,
    phase_or_trajectory_frame: str,
    camera: str,
    prompt: str,
    left_replica: str,
    right_replica: str,
    left_image_path: str,
    right_image_path: str,
    left_image_data_sha256: str,
    right_image_data_sha256: str,
    left_policy_state_hash: str | None,
    right_policy_state_hash: str | None,
    dynamics: dict[str, Any],
    other_camera_exact: bool,
    source_kind: str,
    source_comparison: str,
    frame_index: int | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    state_hash = left_policy_state_hash if left_policy_state_hash == right_policy_state_hash else (
        f"left:{left_policy_state_hash};right:{right_policy_state_hash}"
    )
    return {
        "case_id": case_id,
        "phase_or_trajectory_frame": phase_or_trajectory_frame,
        "camera": camera,
        "source_kind": source_kind,
        "source_comparison": source_comparison,
        "frame_index": frame_index,
        "left_replica": left_replica,
        "right_replica": right_replica,
        "left_image_path": left_image_path,
        "right_image_path": right_image_path,
        "left_image_data_sha256": left_image_data_sha256,
        "right_image_data_sha256": right_image_data_sha256,
        "left_policy_state_hash": left_policy_state_hash,
        "right_policy_state_hash": right_policy_state_hash,
        "policy_state_hash": state_hash,
        "prompt_hash": _sha256_text(prompt),
        "dynamics_exact": bool(dynamics["exact"]),
        "dynamics_evidence": dynamics,
        "other_camera_exact": bool(other_camera_exact),
        "renderer_only_evaluable": bool(dynamics["exact"] and other_camera_exact),
        "raw_and_transformed_input_metrics": metrics,
        "frame_hashes": {
            "left_image_data_sha256": left_image_data_sha256,
            "right_image_data_sha256": right_image_data_sha256,
            "left_policy_state_hash": left_policy_state_hash,
            "right_policy_state_hash": right_policy_state_hash,
        },
    }


def _snapshot_dynamics_evidence(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    simulator_left = left.get("simulator", {})
    simulator_right = right.get("simulator", {})
    if not isinstance(simulator_left, dict):
        simulator_left = {}
    if not isinstance(simulator_right, dict):
        simulator_right = {}
    for key in ("flattened", "qpos", "qvel"):
        evidence[f"simulator_{key}"] = _exact_array_pair(
            _array_from_descriptor_local(simulator_left.get(key)),
            _array_from_descriptor_local(simulator_right.get(key)),
        )
    evidence["policy_state"] = _exact_array_pair(
        _array_from_descriptor_local(left.get("policy_state")),
        _array_from_descriptor_local(right.get("policy_state")),
    )
    evidence["exact"] = bool(all(item["exact"] for key, item in evidence.items() if key != "exact"))
    return evidence


def _trajectory_dynamics_evidence(
    left_arrays: dict[str, np.ndarray],
    right_arrays: dict[str, np.ndarray],
    frame_index: int,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("sim_state_flattened", "qpos", "qvel", "ctrl"):
        left = left_arrays.get(key)
        right = right_arrays.get(key)
        evidence[key] = _exact_array_pair(
            None if left is None else left[frame_index],
            None if right is None else right[frame_index],
        )
    left_state = _trajectory_policy_state(left_arrays, frame_index)
    right_state = _trajectory_policy_state(right_arrays, frame_index)
    evidence["policy_state"] = _exact_array_pair(left_state, right_state)
    evidence["exact"] = bool(all(item["exact"] for key, item in evidence.items() if key != "exact"))
    return evidence


def _snapshot_candidates_from_comparison(
    *,
    comparison: dict[str, Any],
    left_cases: dict[str, dict[str, Any]],
    right_cases: dict[str, dict[str, Any]],
    left_replica: str,
    right_replica: str,
    source_comparison: str,
    coverage_issues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for comparison_case in comparison.get("cases", []):
        case_id = str(comparison_case["case_id"])
        left_case = left_cases[case_id]
        right_case = right_cases[case_id]
        left_snapshots = _snapshot_by_phase(left_case)
        right_snapshots = _snapshot_by_phase(right_case)
        prompt = str(left_case["manifest"]["language"])
        for phase in SNAPSHOT_PHASES:
            left_snapshot = left_snapshots.get(phase)
            right_snapshot = right_snapshots.get(phase)
            if left_snapshot is None or right_snapshot is None:
                if coverage_issues is not None:
                    _coverage_issue(coverage_issues, "expected_snapshot_phase_missing", source_comparison=source_comparison, case_id=case_id, phase=phase)
                continue
            for camera in CAMERAS:
                image_key = camera
                comparison_key = f"{image_key}:{phase}"
                group_metrics = comparison_case.get("renderer_exact_by_camera_and_stage", {}).get(comparison_key)
                if not isinstance(group_metrics, dict) or group_metrics.get("exact", True):
                    if coverage_issues is not None and not isinstance(group_metrics, dict):
                        _coverage_issue(coverage_issues, "comparison_group_missing_or_schema_error", source_comparison=source_comparison, case_id=case_id, group_key=comparison_key)
                    continue
                left_image = _array_from_descriptor_local(left_snapshot.get("observation", {}).get(image_key))
                right_image = _array_from_descriptor_local(right_snapshot.get("observation", {}).get(image_key))
                if left_image is None or right_image is None:
                    if coverage_issues is not None:
                        _coverage_issue(coverage_issues, "snapshot_image_pair_not_evaluable", source_comparison=source_comparison, case_id=case_id, phase=phase, camera=camera)
                    continue
                other_camera = next(value for value in CAMERAS if value != camera)
                other_exact = bool(
                    _exact_array_pair(
                        _array_from_descriptor_local(left_snapshot.get("observation", {}).get(other_camera)),
                        _array_from_descriptor_local(right_snapshot.get("observation", {}).get(other_camera)),
                    )["exact"]
                )
                left_state = _array_from_descriptor_local(left_snapshot.get("policy_state"))
                right_state = _array_from_descriptor_local(right_snapshot.get("policy_state"))
                dynamics = _snapshot_dynamics_evidence(left_snapshot, right_snapshot)
                candidates.append(
                    _candidate_common(
                        case_id=case_id,
                        phase_or_trajectory_frame=phase,
                        camera=camera,
                        prompt=prompt,
                        left_replica=left_replica,
                        right_replica=right_replica,
                        left_image_path=str(left_snapshot["observation"][image_key]["path"]),
                        right_image_path=str(right_snapshot["observation"][image_key]["path"]),
                        left_image_data_sha256=_sha256_array(left_image),
                        right_image_data_sha256=_sha256_array(right_image),
                        left_policy_state_hash=None if left_state is None else _sha256_array(left_state),
                        right_policy_state_hash=None if right_state is None else _sha256_array(right_state),
                        dynamics=dynamics,
                        other_camera_exact=other_exact,
                        source_kind="snapshot",
                        source_comparison=source_comparison,
                        frame_index=None,
                        metrics=_image_metrics(left_image, right_image),
                    )
                )
    return candidates


def _trajectory_arrays(case: dict[str, Any], trajectory_name: str) -> dict[str, np.ndarray]:
    descriptors = case.get("trajectories", {}).get(trajectory_name, {}).get("arrays", {})
    keys = {
        "sim_state_flattened",
        "qpos",
        "qvel",
        "ctrl",
        "obs:robot0_eef_pos",
        "obs:robot0_eef_quat",
        "obs:robot0_gripper_qpos",
        *(f"obs:{camera}" for camera in CAMERAS),
    }
    result: dict[str, np.ndarray] = {}
    for key in keys:
        value = _array_from_descriptor_local(descriptors.get(key))
        if value is not None:
            result[key] = value
    return result


def _trajectory_candidates_from_comparison(
    *,
    comparison: dict[str, Any],
    left_cases: dict[str, dict[str, Any]],
    right_cases: dict[str, dict[str, Any]],
    left_replica: str,
    right_replica: str,
    source_comparison: str,
    coverage_issues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for comparison_case in comparison.get("cases", []):
        case_id = str(comparison_case["case_id"])
        left_case = left_cases[case_id]
        right_case = right_cases[case_id]
        prompt = str(left_case["manifest"]["language"])
        for trajectory_name in TRAJECTORY_MODES:
            left_arrays = _trajectory_arrays(left_case, trajectory_name)
            right_arrays = _trajectory_arrays(right_case, trajectory_name)
            for camera in CAMERAS:
                image_key = f"obs:{camera}"
                group_key = f"{trajectory_name}:{'agentview' if camera == 'agentview_image' else 'robot0_eye_in_hand'}"
                group_metrics = comparison_case.get("trajectory_renderer_exact_by_camera_and_stage", {}).get(group_key)
                if not isinstance(group_metrics, dict) or group_metrics.get("exact", True):
                    if coverage_issues is not None and not isinstance(group_metrics, dict):
                        _coverage_issue(coverage_issues, "comparison_group_missing_or_schema_error", source_comparison=source_comparison, case_id=case_id, group_key=group_key)
                    continue
                left_image = left_arrays.get(image_key)
                right_image = right_arrays.get(image_key)
                if left_image is None or right_image is None or left_image.shape != right_image.shape:
                    if coverage_issues is not None:
                        _coverage_issue(coverage_issues, "trajectory_image_pair_not_evaluable", source_comparison=source_comparison, case_id=case_id, trajectory=trajectory_name, camera=camera)
                    continue
                other_camera = next(value for value in CAMERAS if value != camera)
                other_key = f"obs:{other_camera}"
                other_left = left_arrays.get(other_key)
                other_right = right_arrays.get(other_key)
                frame_count = min(left_image.shape[0], right_image.shape[0])
                for frame_index in range(frame_count):
                    if np.array_equal(left_image[frame_index], right_image[frame_index]):
                        continue
                    left_state = _trajectory_policy_state(left_arrays, frame_index)
                    right_state = _trajectory_policy_state(right_arrays, frame_index)
                    other_exact = bool(
                        _exact_array_pair(
                            None if other_left is None else other_left[frame_index],
                            None if other_right is None else other_right[frame_index],
                        )["exact"]
                    )
                    dynamics = _trajectory_dynamics_evidence(left_arrays, right_arrays, frame_index)
                    descriptors_left = left_case["trajectories"][trajectory_name]["arrays"]
                    descriptors_right = right_case["trajectories"][trajectory_name]["arrays"]
                    candidates.append(
                        _candidate_common(
                            case_id=case_id,
                            phase_or_trajectory_frame=f"{trajectory_name}/frame-{frame_index:04d}",
                            camera=camera,
                            prompt=prompt,
                            left_replica=left_replica,
                            right_replica=right_replica,
                            left_image_path=str(descriptors_left[image_key]["path"]),
                            right_image_path=str(descriptors_right[image_key]["path"]),
                            left_image_data_sha256=_sha256_array(left_image[frame_index]),
                            right_image_data_sha256=_sha256_array(right_image[frame_index]),
                            left_policy_state_hash=None if left_state is None else _sha256_array(left_state),
                            right_policy_state_hash=None if right_state is None else _sha256_array(right_state),
                            dynamics=dynamics,
                            other_camera_exact=other_exact,
                            source_kind="trajectory",
                            source_comparison=source_comparison,
                            frame_index=frame_index,
                            metrics=_image_metrics(left_image[frame_index], right_image[frame_index]),
                        )
                    )
    return candidates


def _legacy_pair_key_record(
    pair: dict[str, Any],
    snapshot_maps: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    replica_to_index = {name: index for index, name in enumerate(REPLICA_NAMES)}
    case_id = str(pair["case_id"])
    phase = str(pair["phase"])
    camera = str(pair["camera"])
    left_case = snapshot_maps[replica_to_index[str(pair["left_replica"])]] [case_id]
    right_case = snapshot_maps[replica_to_index[str(pair["right_replica"])]] [case_id]
    left_snapshot = _snapshot_by_phase(left_case)[phase]
    right_snapshot = _snapshot_by_phase(right_case)[phase]
    left_image = _array_from_descriptor_local(left_snapshot["observation"][camera])
    right_image = _array_from_descriptor_local(right_snapshot["observation"][camera])
    left_state = _array_from_descriptor_local(left_snapshot.get("policy_state"))
    right_state = _array_from_descriptor_local(right_snapshot.get("policy_state"))
    candidate = {
        "case_id": case_id,
        "phase_or_trajectory_frame": phase,
        "camera": camera,
        "policy_state_hash": None if left_state is None else _sha256_array(left_state),
        "prompt_hash": _sha256_text(str(left_case["manifest"]["language"])),
        "left_image_data_sha256": _sha256_array(left_image),
        "right_image_data_sha256": _sha256_array(right_image),
    }
    return {
        "pair_id": pair["pair_id"],
        "semantic_key": _semantic_key_record(_semantic_key(candidate)),
        "left_image_data_sha256": candidate["left_image_data_sha256"],
        "right_image_data_sha256": candidate["right_image_data_sha256"],
        "left_replica": pair["left_replica"],
        "right_replica": pair["right_replica"],
    }


def _capture_locked_old_s1_3() -> dict[str, Any]:
    """Capture the immutable S1.3 namespace before a read-only closure pass."""
    if not OLD_S1_SUMMARY.is_file() or not OLD_S1_MANIFEST.is_file():
        raise FileNotFoundError("locked old S1.3 summary/manifest is missing")
    summary = json.loads(OLD_S1_SUMMARY.read_text(encoding="utf-8"))
    manifest = json.loads(OLD_S1_MANIFEST.read_text(encoding="utf-8"))
    summary_sha256 = sha256_file(OLD_S1_SUMMARY)
    manifest_sha256 = sha256_file(OLD_S1_MANIFEST)
    if summary.get("status") != "s1_real_egl_pairs_failed":
        raise ValueError("old S1.3 failure status was changed")
    if summary_sha256 != LOCKED_OLD_S1_SUMMARY_SHA256:
        raise ValueError("old S1.3 summary hash drifted")
    if manifest_sha256 != LOCKED_OLD_S1_MANIFEST_SHA256:
        raise ValueError("old S1.3 pair manifest hash drifted")
    if not manifest.get("pairs") or int(manifest.get("pair_count", -1)) != 4:
        raise ValueError("old S1.3 pair manifest is not the locked four-pair artifact")
    return {
        "summary": summary,
        "manifest": manifest,
        "summary_sha256": summary_sha256,
        "manifest_sha256": manifest_sha256,
        "status": str(summary["status"]),
    }


def _assert_locked_old_s1_3_unchanged(lock: dict[str, Any]) -> None:
    after = _capture_locked_old_s1_3()
    if after["status"] != lock["status"] or after["summary_sha256"] != lock["summary_sha256"] or after["manifest_sha256"] != lock["manifest_sha256"]:
        raise AssertionError("read-only E0 closure changed the locked S1.3 namespace")


def _legacy_selection_audit(old_manifest: dict[str, Any], snapshot_maps: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    legacy_records = [_legacy_pair_key_record(pair, snapshot_maps) for pair in old_manifest["pairs"]]
    legacy_by_key: dict[str, list[str]] = {}
    pair_by_id = {str(pair["pair_id"]): pair for pair in old_manifest["pairs"]}
    for record in legacy_records:
        key = json.dumps(record["semantic_key"], sort_keys=True)
        legacy_by_key.setdefault(key, []).append(str(record["pair_id"]))
    duplicate_groups = [
        {"semantic_key": json.loads(key), "pair_ids": pair_ids}
        for key, pair_ids in legacy_by_key.items()
        if len(pair_ids) > 1
    ]
    per_camera = []
    for camera in sorted({str(pair["camera"]) for pair in old_manifest["pairs"]}):
        camera_records = [record for record in legacy_records if str(pair_by_id[record["pair_id"]]["camera"]) == camera]
        unique_count = len({json.dumps(record["semantic_key"], sort_keys=True) for record in camera_records})
        representative = next((pair for pair in old_manifest["pairs"] if pair["camera"] == camera and pair["selection"] == "representative_nonzero"), None)
        worst = next((pair for pair in old_manifest["pairs"] if pair["camera"] == camera and pair["selection"] == "worst_transformed"), None)
        representative_unavailable = bool(
            representative
            and worst
            and json.dumps(next(record for record in camera_records if record["pair_id"] == representative["pair_id"])["semantic_key"], sort_keys=True)
            == json.dumps(next(record for record in camera_records if record["pair_id"] == worst["pair_id"])["semantic_key"], sort_keys=True)
        )
        per_camera.append(
            {
                "camera": camera,
                "old_pair_count": len(camera_records),
                "old_content_unique_pair_count": unique_count,
                "representative_unavailable": representative_unavailable,
                "reason": "representative content key equals worst content key; no second distinct pair was available" if representative_unavailable else None,
            }
        )
    return {
        "old_pair_count": len(old_manifest["pairs"]),
        "old_content_unique_pair_count": len(legacy_by_key),
        "duplicate_groups": duplicate_groups,
        "per_camera": per_camera,
    }


def _source_occurrence_expectations(comparison: dict[str, Any]) -> tuple[int, int]:
    snapshot_count = 0
    trajectory_count = 0
    for case in comparison.get("cases", []):
        if not isinstance(case, dict):
            continue
        renderer_mapping = case.get("renderer_exact_by_camera_and_stage") or {}
        for phase in SNAPSHOT_PHASES:
            for camera in CAMERAS:
                metric = renderer_mapping.get(f"{camera}:{phase}")
                if isinstance(metric, dict) and metric.get("exact") is False:
                    snapshot_count += 1
        for metric in (case.get("trajectory_renderer_exact_by_camera_and_stage") or {}).values():
            if isinstance(metric, dict) and metric.get("exact") is False and isinstance(metric.get("affected_frame_count"), int):
                trajectory_count += int(metric["affected_frame_count"])
    return snapshot_count, trajectory_count


def build_e0_reanalysis(output_dir: Path = CLOSURE_ROOT) -> dict[str, Any]:
    """Build a CPU-only, read-only E0 closure artifact; never runs a model."""
    lock = _capture_locked_old_s1_3()
    old_manifest = lock["manifest"]
    snapshot_maps = _snapshot_maps_for_reanalysis()
    all_snapshot_candidates: list[dict[str, Any]] = []
    all_trajectory_candidates: list[dict[str, Any]] = []
    expected_snapshot_groups: set[str] = set()
    observed_snapshot_groups: set[str] = set()
    expected_trajectory_groups: set[str] = set()
    observed_trajectory_groups: set[str] = set()
    coverage_issues: list[dict[str, Any]] = []
    comparison_files: list[str] = []
    expected_snapshot_occurrences = 0
    expected_trajectory_occurrences = 0

    for left_replica, right_replica, filename in REPLICA_PAIR_SPECS:
        path = Q1_ROOT / filename
        source_comparison = str(path.resolve())
        comparison_files.append(source_comparison)
        if not path.is_file():
            _coverage_issue(coverage_issues, "comparison_file_missing", path=source_comparison)
            continue
        try:
            comparison = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _coverage_issue(coverage_issues, "comparison_file_unreadable", path=source_comparison, error=str(exc))
            continue
        if not isinstance(comparison, dict):
            _coverage_issue(coverage_issues, "comparison_file_schema_error", path=source_comparison)
            continue
        left_cases = snapshot_maps[REPLICA_NAMES.index(left_replica)]
        right_cases = snapshot_maps[REPLICA_NAMES.index(right_replica)]
        _validate_snapshot_coverage(
            comparison=comparison,
            left_cases=left_cases,
            right_cases=right_cases,
            source_comparison=source_comparison,
            expected=expected_snapshot_groups,
            observed=observed_snapshot_groups,
            issues=coverage_issues,
        )
        _validate_trajectory_coverage(
            comparison=comparison,
            left_cases=left_cases,
            right_cases=right_cases,
            source_comparison=source_comparison,
            expected=expected_trajectory_groups,
            observed=observed_trajectory_groups,
            issues=coverage_issues,
        )
        snapshot_expected, trajectory_expected = _source_occurrence_expectations(comparison)
        expected_snapshot_occurrences += snapshot_expected
        expected_trajectory_occurrences += trajectory_expected
        try:
            all_snapshot_candidates.extend(
                _snapshot_candidates_from_comparison(
                    comparison=comparison,
                    left_cases=left_cases,
                    right_cases=right_cases,
                    left_replica=left_replica,
                    right_replica=right_replica,
                    source_comparison=source_comparison,
                    coverage_issues=coverage_issues,
                )
            )
            all_trajectory_candidates.extend(
                _trajectory_candidates_from_comparison(
                    comparison=comparison,
                    left_cases=left_cases,
                    right_cases=right_cases,
                    left_replica=left_replica,
                    right_replica=right_replica,
                    source_comparison=source_comparison,
                    coverage_issues=coverage_issues,
                )
            )
        except (AttributeError, KeyError, OSError, TypeError, ValueError, IndexError) as exc:
            _coverage_issue(coverage_issues, "candidate_enumeration_error", source_comparison=source_comparison, error=str(exc))

    all_candidates = all_snapshot_candidates + all_trajectory_candidates
    if len(all_snapshot_candidates) != expected_snapshot_occurrences:
        _coverage_issue(
            coverage_issues,
            "snapshot_candidate_occurrence_count_mismatch",
            expected=expected_snapshot_occurrences,
            observed=len(all_snapshot_candidates),
        )
    if len(all_trajectory_candidates) != expected_trajectory_occurrences:
        _coverage_issue(
            coverage_issues,
            "trajectory_affected_frame_occurrence_count_mismatch",
            expected=expected_trajectory_occurrences,
            observed=len(all_trajectory_candidates),
        )

    unique_candidates, duplicates = _dedupe_semantic_pairs(all_candidates)
    dynamics_mismatches = [item for item in unique_candidates if not item["dynamics_exact"]]
    not_evaluable = [item for item in unique_candidates if not item["renderer_only_evaluable"]]
    snapshot_set_audit = _coverage_set_audit(expected_snapshot_groups, observed_snapshot_groups, [])
    trajectory_set_audit = _coverage_set_audit(expected_trajectory_groups, observed_trajectory_groups, [])
    coverage_complete = bool(snapshot_set_audit["complete"] and trajectory_set_audit["complete"] and not coverage_issues)
    closure_complete = bool(coverage_complete and not dynamics_mismatches and not not_evaluable)
    status = "e0_closure_complete_pending_review" if closure_complete else "e0_closure_coverage_incomplete"

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "content_unique_pair_manifest.json"
    manifest = _content_unique_manifest(
        unique_candidates,
        source_reanalysis=str(summary_path.resolve()),
        status=status,
    )
    write_json(manifest_path, manifest)
    result = {
        "schema_version": 3,
        "phase": "E0",
        "status": status,
        "closure_complete": closure_complete,
        "read_only_reanalysis": True,
        "model_run_started": False,
        "gpu_started": False,
        "old_s1_3": {
            "summary": str(OLD_S1_SUMMARY.resolve()),
            "summary_sha256": lock["summary_sha256"],
            "pair_manifest": str(OLD_S1_MANIFEST.resolve()),
            "pair_manifest_sha256": lock["manifest_sha256"],
            "status_preserved": lock["status"],
            "old_threshold": 1e-6,
            "old_max_abs_diff": 5.125999450683594e-5,
            "raw_lora_q2_status_preserved": "n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay",
            "corrected_reference_backend": "R6 folded JAX external actions",
        },
        "source_q1": {
            "root": str(Q1_ROOT.resolve()),
            "summary": str((Q1_ROOT / "q1_environment_full_c20_summary.json").resolve()),
            "summary_sha256": sha256_file(Q1_ROOT / "q1_environment_full_c20_summary.json"),
            "comparison_files": comparison_files,
            "manifest_sha256": json.loads((Q1_ROOT / "q1_environment_full_c20_summary.json").read_text(encoding="utf-8"))["manifest_sha256"],
        },
        "legacy_selection_audit": _legacy_selection_audit(old_manifest, snapshot_maps),
        "coverage": {
            "case_count": len(snapshot_maps[0]),
            "replica_pairs": [list(spec[:2]) for spec in REPLICA_PAIR_SPECS],
            "snapshot_phases": list(SNAPSHOT_PHASES),
            "snapshot_camera_count": len(CAMERAS),
            "snapshot_comparison_count": len(REPLICA_PAIR_SPECS) * len(snapshot_maps[0]) * len(SNAPSHOT_PHASES) * len(CAMERAS),
            "snapshot_non_exact_group_count": len(expected_snapshot_groups),
            "snapshot_candidate_occurrence_count": len(all_snapshot_candidates),
            "trajectory_modes": list(TRAJECTORY_MODES),
            "trajectory_frame_count_per_mode": 64,
            "trajectory_camera_count": len(CAMERAS),
            "trajectory_group_comparison_count": len(REPLICA_PAIR_SPECS) * len(snapshot_maps[0]) * len(TRAJECTORY_MODES) * len(CAMERAS),
            "trajectory_frame_comparison_count": len(REPLICA_PAIR_SPECS) * len(snapshot_maps[0]) * len(TRAJECTORY_MODES) * len(CAMERAS) * 64,
            "trajectory_non_exact_group_count": len(expected_trajectory_groups),
            "trajectory_affected_frame_occurrence_count": len(all_trajectory_candidates),
            "q1_comparison_count": len(REPLICA_PAIR_SPECS) * len(snapshot_maps[0]) * (len(SNAPSHOT_PHASES) + len(TRAJECTORY_MODES)) * len(CAMERAS),
            "non_exact_comparison_group_count": len(expected_snapshot_groups) + len(expected_trajectory_groups),
            "snapshot_and_trajectory_candidate_count": len(all_candidates),
            "snapshot_content_unique_pair_count": sum(1 for item in unique_candidates if item["source_kind"] == "snapshot"),
            "trajectory_content_unique_pair_count": sum(1 for item in unique_candidates if item["source_kind"] == "trajectory"),
            "content_unique_pair_count": len(unique_candidates),
            "content_duplicate_occurrence_count": len(duplicates),
            "renderer_only_evaluable_count": sum(1 for item in unique_candidates if item["renderer_only_evaluable"]),
            "not_evaluable_count": len(not_evaluable),
            "dynamics_mismatch_count": len(dynamics_mismatches),
            "expected_snapshot_group_keys": snapshot_set_audit["expected"],
            "observed_snapshot_group_keys": snapshot_set_audit["observed"],
            "missing_snapshot_group_keys": snapshot_set_audit["missing"],
            "extra_snapshot_group_keys": snapshot_set_audit["extra"],
            "expected_trajectory_group_keys": trajectory_set_audit["expected"],
            "observed_trajectory_group_keys": trajectory_set_audit["observed"],
            "missing_trajectory_group_keys": trajectory_set_audit["missing"],
            "extra_trajectory_group_keys": trajectory_set_audit["extra"],
            "snapshot_coverage_complete": snapshot_set_audit["complete"],
            "trajectory_coverage_complete": trajectory_set_audit["complete"],
            "coverage_complete": coverage_complete,
            "coverage_issues": coverage_issues,
        },
        "dynamics_audit": {
            "dynamics_mismatch_count": len(dynamics_mismatches),
            "dynamics_mismatch_pairs": dynamics_mismatches,
            "not_evaluable_count": len(not_evaluable),
            "renderer_only_evaluable_count": sum(1 for item in unique_candidates if item["renderer_only_evaluable"]),
            "q1_dynamics_mismatch_cases": [],
        },
        "closure_artifacts": {
            "summary": str(summary_path.resolve()),
            "content_unique_pair_manifest": str(manifest_path.resolve()),
            "content_unique_pair_manifest_sha256": sha256_file(manifest_path),
        },
        "content_unique_pair_count": len(unique_candidates),
        "content_duplicate_occurrence_count": len(duplicates),
        "not_started_phases": ["E1", "E2", "E3", "E4", "E5", "E6", "E7"],
    }
    write_json(summary_path, result)
    report = f"""# E0 Closure：只读 fail-closed 证据

状态：`{status}`

本 closure 只读取 Q1/S1.3 JSON 与数组证据，不运行模型、不启动 GPU。旧 S1.3 failure、summary hash 与 pair manifest hash 已锁定并在写入前后复核；初次 E0 namespace 未覆盖。

## 覆盖与计数

- Q1 comparison：`{result['coverage']['q1_comparison_count']}`；non-exact group：`{result['coverage']['non_exact_comparison_group_count']}`
- snapshot：`{result['coverage']['snapshot_non_exact_group_count']}` groups，`{result['coverage']['snapshot_candidate_occurrence_count']}` occurrences，`{result['coverage']['snapshot_content_unique_pair_count']}` content-unique
- trajectory：`{result['coverage']['trajectory_non_exact_group_count']}` groups，`{result['coverage']['trajectory_affected_frame_occurrence_count']}` affected-frame occurrences，`{result['coverage']['trajectory_content_unique_pair_count']}` content-unique
- total：`{result['coverage']['content_unique_pair_count']}` content-unique，`{result['coverage']['content_duplicate_occurrence_count']}` duplicate occurrences
- dynamics mismatch：`{result['coverage']['dynamics_mismatch_count']}`；not evaluable：`{result['coverage']['not_evaluable_count']}`

`trajectory affected-frame occurrences` 与 `trajectory content-unique pairs` 是不同计数维度，不能互换。

## Fail-closed 门禁

- snapshot expected/observed set equal：`{result['coverage']['snapshot_coverage_complete']}`
- trajectory expected/observed set equal：`{result['coverage']['trajectory_coverage_complete']}`
- coverage issues：`{len(result['coverage']['coverage_issues'])}`
- old S1.3 status：`{result['old_s1_3']['status_preserved']}`
- corrected reference backend：`{result['old_s1_3']['corrected_reference_backend']}`

未启动：E1、E2、E3、E4、E5、E6、E7；下一阶段只能是 `E1_pending_manual_review`。
"""
    (output_dir / "E0_CLOSURE_REPORT_ZH.md").write_text(report, encoding="utf-8")
    (output_dir / "THREAD_REPORT_ZH.md").write_text(
        report + "\n本 section 按 E0 Closure 计划在 clean commit 后停止；未进行 GPU、closed-loop、FAST、ZIP、上传或比赛提交。\n",
        encoding="utf-8",
    )
    _assert_locked_old_s1_3_unchanged(lock)
    return result


def _candidate_pairs(snapshot_maps: list[dict[tuple[str, str], dict[str, Any]]], camera: str) -> list[dict[str, Any]]:
    common = set(snapshot_maps[0]) & set(snapshot_maps[1]) & set(snapshot_maps[2])
    candidates: list[dict[str, Any]] = []
    for case_id, phase in sorted(common):
        for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
            left_descriptor = _raw_descriptor(snapshot_maps[left_index][(case_id, phase)], camera)
            right_descriptor = _raw_descriptor(snapshot_maps[right_index][(case_id, phase)], camera)
            if left_descriptor is None or right_descriptor is None:
                continue
            left_path = Path(str(left_descriptor["path"]))
            right_path = Path(str(right_descriptor["path"]))
            if not left_path.is_file() or not right_path.is_file():
                raise FileNotFoundError(f"Q1 pair image missing: {left_path} / {right_path}")
            metrics = _pair_metrics(left_path, right_path)
            candidates.append(
                {
                    "case_id": case_id,
                    "phase": phase,
                    "camera": camera,
                    "left_replica": REPLICA_NAMES[left_index],
                    "right_replica": REPLICA_NAMES[right_index],
                    "left_path": str(left_path.resolve()),
                    "right_path": str(right_path.resolve()),
                    "left_sha256": sha256_file(left_path),
                    "right_sha256": sha256_file(right_path),
                    "metrics": metrics,
                }
            )
    if not candidates:
        raise ValueError(f"no Q1 candidates for camera {camera}")
    return candidates


def _selection_key(item: dict[str, Any]) -> tuple[float, int, float, str, str, str]:
    metric = item["metrics"]["transformed_float32_minus1_plus1"]
    return (
        float(metric["max_abs_diff"]),
        int(metric["changed_element_count"]),
        float(metric["mean_abs_diff"]),
        str(item["case_id"]),
        str(item["phase"]),
        str(item["left_replica"]),
    )


def _load_base() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not BASE_INPUT.is_file() or not BASE_TASKS.is_file():
        raise FileNotFoundError("S1.2 frozen input/tasks are required before S1.3")
    arrays_file = np.load(BASE_INPUT, allow_pickle=False)
    arrays = {key: np.asarray(arrays_file[key]) for key in arrays_file.files}
    tasks = json.loads(BASE_TASKS.read_text(encoding="utf-8"))
    if len(tasks.get("fixture_ids", [])) != 32 or len(tasks.get("tasks", [])) != 32:
        raise ValueError("S1.2 task fixture is not 32 samples")
    return arrays, tasks


def _deployment_tokens(prompt: str, tokenizer: Any) -> tuple[np.ndarray, np.ndarray]:
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = tokenizer.encode(cleaned, add_bos=True) + tokenizer.encode("\n")
    tokens = tokens[:200]
    padded = np.zeros((200,), dtype=np.int32)
    padded[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
    mask = np.zeros((200,), dtype=np.bool_)
    mask[: len(tokens)] = True
    return padded, mask


def _manifest_prompts() -> dict[str, str]:
    return {
        str(row["case_id"]): str(row["language"])
        for row in (json.loads(line) for line in C20_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _q1_base_fixture(case_id: str, snapshot_maps: list[dict[tuple[str, str], dict[str, Any]]], tokenizer: Any) -> tuple[dict[str, np.ndarray], str]:
    snapshot = snapshot_maps[0].get((case_id, "E3-10"))
    if snapshot is None:
        raise ValueError(f"Q1 has no E3-10 base snapshot for {case_id}")
    prompt = _manifest_prompts()[case_id]
    with NORM_STATS.open(encoding="utf-8") as handle:
        norm = json.load(handle)["norm_stats"]["state"]
    state_raw = np.asarray(np.load(snapshot["policy_state"]["path"], allow_pickle=False), dtype=np.float32)
    state_norm = ((state_raw - np.asarray(norm["q01"], dtype=np.float32)) / (np.asarray(norm["q99"], dtype=np.float32) - np.asarray(norm["q01"], dtype=np.float32) + np.float32(1e-6)) * np.float32(2.0) - np.float32(1.0))
    tokenized, tokenized_mask = _deployment_tokens(prompt, tokenizer)
    base_image = _normalize_raw(Path(str(snapshot["observation"]["agentview_image"]["path"])))
    wrist_image = _normalize_raw(Path(str(snapshot["observation"]["robot0_eye_in_hand_image"]["path"])))
    return {
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": wrist_image,
        "right_wrist_0_rgb": np.full((224, 224, 3), -1.0, dtype=np.float32),
        "state": np.pad(state_norm, (0, 24)).astype(np.float32),
        "tokenized_prompt": tokenized,
        "tokenized_prompt_mask": tokenized_mask,
        "noise": np.random.default_rng(63000).standard_normal((10, 32), dtype=np.float32),
    }, f"{case_id}::Q1-replica0::E3-10"


def _normalize_raw(path: Path) -> np.ndarray:
    raw = np.asarray(np.load(path, allow_pickle=False))
    transformed = _resize_for_openpi(raw)
    return transformed.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)


def build_egl_pairs(output_dir: Path = PAIR_ROOT) -> dict[str, Any]:
    arrays, base_tasks = _load_base()
    snapshot_maps = _snapshot_maps()
    tokenizer = __import__("sentencepiece").SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    base_index_by_case = {
        str(fixture_id).rsplit("-s", 1)[0]: index
        for index, fixture_id in enumerate(base_tasks["fixture_ids"])
        if str(fixture_id).endswith("-s0")
    }
    selections: list[dict[str, Any]] = []
    for camera in CAMERAS:
        candidates = _candidate_pairs(snapshot_maps, camera)
        worst = max(candidates, key=_selection_key)
        nonzero = [item for item in candidates if item["metrics"]["transformed_float32_minus1_plus1"]["max_abs_diff"] > 0]
        if not nonzero:
            raise ValueError(f"Q1 has no nonzero transformed pair for {camera}")
        representative = next((item for item in sorted(nonzero, key=lambda value: (value["case_id"], value["phase"], value["left_replica"])) if item != worst), nonzero[0])
        for selection_name, item in (("worst_transformed", worst), ("representative_nonzero", representative)):
            selected = dict(item)
            selected["selection"] = selection_name
            selected["selection_rule"] = "worst=max(transformed max, changed elements, mean; stable case/phase/replica tie-break)" if selection_name == "worst_transformed" else "first stable nonzero transformed pair distinct from worst when available"
            selections.append(selected)

    input_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask", "noise")}
    tasks: list[str] = []
    pair_ids: list[str] = []
    variants: list[str] = []
    pair_rows: list[dict[str, Any]] = []
    for pair_index, selected in enumerate(selections):
        case_id = str(selected["case_id"])
        if case_id in base_index_by_case:
            base_index = base_index_by_case[case_id]
            base_values = {key: arrays[key][base_index] for key in input_arrays}
            base_fixture_id = str(base_tasks["fixture_ids"][base_index])
            base_task = str(base_tasks["tasks"][base_index])
        else:
            base_values, base_fixture_id = _q1_base_fixture(case_id, snapshot_maps, tokenizer)
            base_task = _manifest_prompts()[case_id]
        camera_key = CAMERA_TO_INPUT[str(selected["camera"])]
        pair_id = f"pair{pair_index:02d}_{selected['camera']}_{selected['selection']}"
        left_image = _normalize_raw(Path(str(selected["left_path"])))
        right_image = _normalize_raw(Path(str(selected["right_path"])))
        pair_values: dict[str, np.ndarray] = {}
        for key in input_arrays:
            if key == camera_key:
                pair_values[key] = np.stack((left_image, right_image), axis=0)
            else:
                pair_values[key] = np.repeat(np.asarray(base_values[key])[None], 2, axis=0)
            input_arrays[key].extend(list(pair_values[key]))
        tasks.extend([base_task] * 2)
        pair_ids.extend([pair_id, pair_id])
        variants.extend(["left", "right"])
        pair_rows.append(
            {
                **selected,
                "pair_id": pair_id,
                "input_indices": [2 * pair_index, 2 * pair_index + 1],
                "base_fixture_id": base_fixture_id,
                "held_fixed_camera": next(camera for camera in CAMERAS if camera != selected["camera"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "egl_input.npz"
    np.savez_compressed(input_path, **{key: np.stack(value, axis=0) for key, value in input_arrays.items()})
    tasks_path = output_dir / "tasks.json"
    write_json(tasks_path, {"pair_ids": pair_ids, "variants": variants, "tasks": tasks})
    q1_paths = [Q1_ROOT / replica / "summary.json" for replica in REPLICA_NAMES]
    manifest = {
        "schema_version": 1,
        "phase": "S1.3",
        "status": "egl_pair_input_built",
        "selection_rule": "per camera: transformed maximum pair plus one representative nonzero pair; p99 is not used because Q1 reports sparse differences",
        "camera_count": len(CAMERAS),
        "pair_count": len(pair_rows),
        "sample_count": len(pair_ids),
        "source_q1_summary": str((Q1_ROOT / "q1_environment_full_c20_summary.json").resolve()),
        "source_q1_summary_sha256": sha256_file(Q1_ROOT / "q1_environment_full_c20_summary.json"),
        "source_q1_replica_summary_sha256": {replica: sha256_file(Q1_ROOT / replica / "summary.json") for replica in REPLICA_NAMES},
        "source_s1_frozen_input": str(BASE_INPUT.resolve()),
        "source_s1_frozen_input_sha256": sha256_file(BASE_INPUT),
        "source_q1_replicas": {replica: str(path.resolve()) for replica, path in zip(REPLICA_NAMES, q1_paths)},
        "pairs": pair_rows,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "tasks": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "deployment_image_convention": "float32 [-1, 1], _resize_for_openpi uint8 / 255 * 2 - 1",
    }
    write_json(output_dir / "pair_manifest.json", manifest)
    write_json(output_dir / "summary.json", manifest)
    return manifest


def compare_egl_pairs(run_dir: Path, manifest_path: Path, output_dir: Path = PAIR_ROOT) -> dict[str, Any]:
    report_path = run_dir / "e2e_action_parity_report.json"
    artifact_path = run_dir / "e2e_action_parity.npz"
    if not report_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError("S1.3 E2E run report/artifact is incomplete")
    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = np.load(artifact_path, allow_pickle=False)
    failures: list[dict[str, Any]] = []

    def require(condition: bool, gate: str, evidence: Any) -> None:
        if not condition:
            failures.append({"gate": gate, "evidence": evidence})

    require(run_report.get("backend") == "lerobot_disk_processors_and_model_safetensors_with_openpi_pytorch_core", "actual_disk_lerobot_backend", run_report.get("backend"))
    require(run_report.get("batch_size") == 8, "coverage_4_pairs_x_2", run_report.get("batch_size"))
    require(run_report.get("normalization") == "QUANTILES", "quantiles_normalization", run_report.get("normalization"))
    require(run_report.get("processor_token_mask_mismatch_count") == 0, "token_mask_mismatch", run_report.get("processor_token_mask_mismatch_count"))
    require(run_report.get("processor_token_max_abs_diff") == 0, "token_mismatch", run_report.get("processor_token_max_abs_diff"))
    require(run_report.get("processor_state_max_abs_diff", float("inf")) <= 1e-6, "state_processor_diff", run_report.get("processor_state_max_abs_diff"))
    for name in ("folded_jax_external_actions", "lerobot_postprocessed_actions", "folded_jax_actions", "folded_pytorch_actions"):
        require(name in artifact and bool(np.isfinite(artifact[name]).all()), f"finite_{name}", name in artifact)

    pair_results: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        left, right = map(int, pair["input_indices"])
        pt = artifact["lerobot_postprocessed_actions"]
        jax = artifact["folded_jax_external_actions"]
        pt_full = _diff_summary(pt[left], pt[right])
        pt_execute8 = _diff_summary(pt[left, :8], pt[right, :8])
        jax_full = _diff_summary(jax[left], jax[right])
        jax_execute8 = _diff_summary(jax[left, :8], jax[right, :8])
        pt_flip = int(np.count_nonzero(np.sign(pt[left, ..., -1]) != np.sign(pt[right, ..., -1])))
        jax_flip = int(np.count_nonzero(np.sign(jax[left, ..., -1]) != np.sign(jax[right, ..., -1])))
        result = {
            "pair_id": pair["pair_id"],
            "camera": pair["camera"],
            "selection": pair["selection"],
            "case_id": pair["case_id"],
            "phase": pair["phase"],
            "raw_and_transformed_input_metrics": pair["metrics"],
            "deployment_pytorch_external_full_chunk": pt_full,
            "deployment_pytorch_external_execute8": pt_execute8,
            _corrected_reference_label("axis_reference_jax_external_full_chunk"): jax_full,
            _corrected_reference_label("axis_reference_jax_external_execute8"): jax_execute8,
            "deployment_pytorch_gripper_sign_flip": pt_flip,
            _corrected_reference_label("axis_reference_jax_gripper_sign_flip"): jax_flip,
        }
        require(pt_full["finite"] and pt_full["max_abs_diff"] <= 1e-6, f"{pair['pair_id']}_pytorch_full_chunk_le_1e-6", pt_full)
        require(pt_execute8["finite"] and pt_execute8["max_abs_diff"] <= 1e-6, f"{pair['pair_id']}_pytorch_execute8_le_1e-6", pt_execute8)
        require(pt_flip == 0, f"{pair['pair_id']}_pytorch_gripper_sign_flip_zero", pt_flip)
        require(jax_full["finite"] and jax_execute8["finite"] and jax_flip == 0, f"{pair['pair_id']}_jax_reference_finite_no_flip", {"full": jax_full, "execute8": jax_execute8, "gripper_flip": jax_flip})
        pair_results.append(result)

    status = "r6_lerobot_folded_policy_sensitivity_locked" if not failures else "s1_real_egl_pairs_failed"
    result = {
        "schema_version": 1,
        "phase": "S1.3",
        "status": status,
        "deployment_backend": run_report.get("backend"),
        "model_sha256": run_report.get("model_sha256"),
        "norm_stats_sha256": run_report.get("norm_stats_sha256"),
        "run_report": str(report_path.resolve()),
        "run_artifact": str(artifact_path.resolve()),
        "pair_manifest": str(manifest_path.resolve()),
        "pairs": pair_results,
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "affected_artifact": str((run_dir if failures else output_dir).resolve()),
        "not_started_phases": ["S2", "S3", "S4", "S5", "S6", "S7"] if failures else [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)
    (output_dir / "S1_REAL_EGL_REPORT_ZH.md").write_text(
        "# S1.3 R6 folded real EGL pair sensitivity\n\n"
        f"状态：`{status}`\n\n"
        f"- pairs：{len(pair_results)}（每 camera 最差 transformed pair + representative nonzero pair）\n"
        f"- backend：`{result['deployment_backend']}`\n"
        f"- first_failure：{json.dumps(result['first_failure'], ensure_ascii=False)}\n"
        f"- not_started_phases：{', '.join(result['not_started_phases']) or 'none'}\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, default=PAIR_ROOT)
    compare = sub.add_parser("compare")
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, default=PAIR_ROOT / "pair_manifest.json")
    compare.add_argument("--output", type=Path, default=PAIR_ROOT)
    reanalyse = sub.add_parser("reanalyse")
    reanalyse.add_argument("--output", type=Path, default=CLOSURE_ROOT)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_egl_pairs(args.output), ensure_ascii=False, indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_egl_pairs(args.run_dir, args.manifest, args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build_e0_reanalysis(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
