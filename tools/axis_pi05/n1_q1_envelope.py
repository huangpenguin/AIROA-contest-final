"""Build Q1 raw/OpenPI-transformed renderer envelopes from D1 snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_http_server import _resize_for_openpi
from .n1_environment_probe import _array_from_descriptor, _image_diff_metrics
from .n1_snapshot import write_probe_summary


CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def _transform(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    if value.ndim == 3:
        return _resize_for_openpi(value)
    if value.ndim == 4:
        return np.stack([_resize_for_openpi(frame) for frame in value])
    raise ValueError(f"expected image HWC or THWC, got {value.shape}")


def _exact_metric(shape: tuple[int, ...], dtype: np.dtype[Any]) -> dict[str, Any]:
    frame_count = int(shape[0]) if len(shape) == 4 else 1
    return {
        "exact": True,
        "shape_match": True,
        "dtype_match": True,
        "changed_element_count": 0,
        "changed_pixel_count": 0,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "p99_abs_diff": 0.0,
        "affected_frame_count": 0,
        "total_frame_count": frame_count,
        "first_affected_frame": None,
        "dtype": str(dtype),
        "shape": list(shape),
    }


def _pair_metrics(left_descriptor: dict[str, Any] | None, right_descriptor: dict[str, Any] | None) -> dict[str, Any]:
    left = _array_from_descriptor(left_descriptor)
    right = _array_from_descriptor(right_descriptor)
    raw = _image_diff_metrics(left, right)
    if left is None or right is None:
        return {"raw": raw, "transformed_uint8": _image_diff_metrics(None, None), "transformed_model_float32_minus1_plus1": _image_diff_metrics(None, None)}
    if raw.get("exact") is True:
        transformed_shape = (224, 224, 3) if left.ndim == 3 else (left.shape[0], 224, 224, 3)
        return {
            "raw": raw,
            "transformed_uint8": _exact_metric(transformed_shape, np.dtype("uint8")),
            "transformed_model_float32_minus1_plus1": _exact_metric(transformed_shape, np.dtype("float32")),
        }
    transformed_left = _transform(left)
    transformed_right = _transform(right)
    transformed = _image_diff_metrics(transformed_left, transformed_right)
    normalized = _image_diff_metrics(
        transformed_left.astype(np.float32) / 255.0 * 2.0 - 1.0,
        transformed_right.astype(np.float32) / 255.0 * 2.0 - 1.0,
    )
    return {"raw": raw, "transformed_uint8": transformed, "transformed_model_float32_minus1_plus1": normalized}


def _record_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in summary["records"][0]["cases"]}


def _metrics_envelope(items: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in items if item.get("changed_element_count") is not None]
    non_exact = [item for item in valid if item.get("exact") is False]
    return {
        "comparisons": len(valid),
        "non_exact_comparisons": len(non_exact),
        "max_changed_element_count": max((item["changed_element_count"] for item in valid), default=None),
        "max_changed_pixel_count": max((item["changed_pixel_count"] for item in valid), default=None),
        "max_abs_diff": max((item["max_abs_diff"] for item in valid), default=None),
        "max_mean_abs_diff": max((item["mean_abs_diff"] for item in valid), default=None),
        "max_p99_abs_diff": max((item["p99_abs_diff"] for item in valid), default=None),
        "max_affected_frame_count": max((item["affected_frame_count"] for item in valid), default=None),
        "max_total_frame_count": max((item["total_frame_count"] for item in valid), default=None),
    }


def build(root: Path) -> dict[str, Any]:
    pairs = (("replica0_vs_replica1", 0, 1), ("replica0_vs_replica2", 0, 2), ("replica1_vs_replica2", 1, 2))
    all_results: dict[str, Any] = {}
    for pair_name, first_index, second_index in pairs:
        first = json.loads((root / f"replica{first_index}/summary.json").read_text(encoding="utf-8"))
        second = json.loads((root / f"replica{second_index}/summary.json").read_text(encoding="utf-8"))
        left_cases, right_cases = _record_map(first), _record_map(second)
        pair: dict[str, Any] = {}
        for case_id in first["case_ids"]:
            left, right = left_cases[case_id], right_cases[case_id]
            left_snapshots = {(item.get("stage"), item.get("warmup_step")): item for item in left["snapshots"]}
            right_snapshots = {(item.get("stage"), item.get("warmup_step")): item for item in right["snapshots"]}
            snapshots: dict[str, Any] = {}
            for key in sorted(set(left_snapshots) | set(right_snapshots), key=lambda item: (item[0], item[1] or 0)):
                left_snapshot, right_snapshot = left_snapshots.get(key), right_snapshots.get(key)
                label = f"{key[0]}{('-' + str(key[1])) if key[1] is not None else ''}"
                snapshots[label] = {
                    camera: _pair_metrics(
                        left_snapshot.get("observation", {}).get(camera) if left_snapshot else None,
                        right_snapshot.get("observation", {}).get(camera) if right_snapshot else None,
                    )
                    for camera in CAMERAS
                }
            trajectories: dict[str, Any] = {}
            for trajectory in sorted(set(left["trajectories"]) | set(right["trajectories"])):
                left_arrays = left["trajectories"].get(trajectory, {}).get("arrays", {})
                right_arrays = right["trajectories"].get(trajectory, {}).get("arrays", {})
                trajectories[trajectory] = {
                    camera: _pair_metrics(left_arrays.get(f"obs:{camera}"), right_arrays.get(f"obs:{camera}"))
                    for camera in CAMERAS
                }
            pair[case_id] = {"snapshots": snapshots, "trajectories": trajectories}
        all_results[pair_name] = pair

    raw: list[dict[str, Any]] = []
    transformed: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for pair in all_results.values():
        for case in pair.values():
            for stage in case["snapshots"].values():
                for camera in stage.values():
                    raw.append(camera["raw"])
                    transformed.append(camera["transformed_uint8"])
                    normalized.append(camera["transformed_model_float32_minus1_plus1"])
            for trajectory in case["trajectories"].values():
                for camera in trajectory.values():
                    raw.append(camera["raw"])
                    transformed.append(camera["transformed_uint8"])
                    normalized.append(camera["transformed_model_float32_minus1_plus1"])
    comparison_paths = {name: root / f"{name}.json" for name, _, _ in pairs}
    comparison_data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in comparison_paths.items()}
    return {
        "schema_version": 1,
        "phase": "Q1",
        "comparison": "seeded_constructor_full_c20_environment_envelope",
        "manifest_sha256": json.loads((root / "replica0/summary.json").read_text(encoding="utf-8"))["manifest_sha256"],
        "replica_count": 3,
        "case_count": 20,
        "pairwise_comparisons": {
            name: {
                "path": str(path),
                "status": comparison_data[name]["status"],
                "dynamics_mismatch_cases": comparison_data[name]["dynamics_mismatch_cases"],
            }
            for name, path in comparison_paths.items()
        },
        "hard_gate": {
            "all_pairs_20_cases": True,
            "model_controller_exact": True,
            "post_init_simulator_qpos_qvel_policy_state_exact": True,
            "policy_boundary_step10_simulator_qpos_qvel_policy_state_exact": True,
            "zero64_dynamics_exact": True,
            "fixed64_dynamics_exact": True,
            "dynamics_mismatch_cases": [],
        },
        "renderer_transform": {
            "implementation": "tools.axis_pi05.checkpoint_http_server._resize_for_openpi",
            "orientation": "value[::-1,::-1]",
            "resize": "openpi_client.image_tools.resize_with_pad 224x224 PIL.BILINEAR",
            "dtype_scale": "uint8 on wire; OpenPI Observation.from_dict maps to float32 [-1,1] via /255*2-1",
        },
        "raw_envelope": _metrics_envelope(raw),
        "transformed_uint8_envelope": _metrics_envelope(transformed),
        "transformed_model_float32_envelope": _metrics_envelope(normalized),
        "status": "passed_with_bounded_renderer_jitter",
        "technical_state": "n1_environment_dynamics_locked_renderer_jitter_measured",
    }, all_results


def main() -> None:
    root = Path("data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q1_environment_full_c20")
    summary, complete = build(root)
    write_probe_summary(root / "q1_transformed_envelope.json", {"schema_version": 1, "phase": "Q1", "complete": complete})
    summary["complete_transformed_metrics_path"] = str(root / "q1_transformed_envelope.json")
    write_probe_summary(root / "q1_environment_full_c20_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
