"""D1 environment-only determinism probes for the frozen C20 manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, git_info, sha256_file
from .n1_snapshot import (
    gpu_boundary_evidence,
    gpu_snapshot,
    save_array,
    snapshot_record,
    validate_gpu_namespace,
    write_probe_summary,
)


DEFAULT_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
DEFAULT_OUT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_nondeterminism/environment_probe"
DEFAULT_FIXED_TRACE_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/evaluator_reproducibility/axis_step0_c20_replay1_gpu1_v2/action_traces"
DEFAULT_CASES = ("C20-task00-init0", "C20-task02-init0", "C20-task11-init0")
IMAGE_OBSERVATION_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
TRAJECTORY_DYNAMICS_KEYS = ("sim_state_flattened", "qpos", "qvel", "ctrl")


def _load_cases(path: Path, case_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["case_id"]): row for row in rows}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"missing C20 cases: {missing}")
    return [by_id[case_id] for case_id in case_ids]


def _seed_globals(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))


def _current_observation(env: Any) -> dict[str, np.ndarray]:
    getter = getattr(getattr(env, "env", None), "_get_observations", None)
    if getter is None:
        raise RuntimeError("LIBERO wrapper does not expose _get_observations for direct init probe")
    return getter()


def _controller_descriptor(env: Any) -> dict[str, Any]:
    robot = env.robots[0]
    controller = getattr(robot, "controller", None)
    if controller is None:
        return {"class": None}
    result: dict[str, Any] = {"class": f"{type(controller).__module__}.{type(controller).__name__}"}
    for name in ("control_dim", "input_min", "input_max", "output_min", "output_max", "goal_pos", "goal_ori"):
        value = getattr(controller, name, None)
        if value is None:
            continue
        if np.isscalar(value):
            result[name] = float(value)
        else:
            array = np.asarray(value)
            result[name] = {"dtype": str(array.dtype), "shape": list(array.shape), "data_sha256": __import__("hashlib").sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}
    return result


def _env_factory(row: dict[str, Any], *, seed_before_constructor: bool) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    env_seed = int(row["env_seed"])
    if seed_before_constructor:
        _seed_globals(env_seed)
    env = OffScreenRenderEnv(
        bddl_file_name=str(row["bddl_path"]),
        camera_heights=128,
        camera_widths=128,
        control_freq=10,
    )
    env.seed(env_seed)
    if seed_before_constructor:
        _seed_globals(env_seed)
    return env


def _init_observation(env: Any, row: dict[str, Any], *, init_mode: str) -> dict[str, np.ndarray]:
    import torch

    init_states = torch.load(row["init_states_path"], weights_only=False)
    init_state = np.asarray(init_states[int(row["init_state_index"])])
    if init_mode == "official":
        return env.set_init_state(init_state)
    if init_mode != "direct":
        raise ValueError(f"unknown init mode: {init_mode}")
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    return _current_observation(env)


def _trajectory(env: Any, root: Path, case_id: str, name: str, actions: np.ndarray) -> dict[str, Any]:
    states: list[np.ndarray] = []
    qpos: list[np.ndarray] = []
    qvel: list[np.ndarray] = []
    ctrl: list[np.ndarray] = []
    observations: dict[str, list[np.ndarray]] = {}
    action_rows: list[np.ndarray] = []
    for action in np.asarray(actions, dtype=np.float32):
        obs, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
        action_rows.append(np.asarray(action, dtype=np.float32).copy())
        states.append(np.asarray(env.sim.get_state().flatten()).copy())
        qpos.append(np.asarray(env.sim.data.qpos).copy())
        qvel.append(np.asarray(env.sim.data.qvel).copy())
        ctrl.append(np.asarray(env.sim.data.ctrl).copy())
        for key, value in sorted(obs.items()):
            if isinstance(value, np.ndarray):
                observations.setdefault(str(key), []).append(np.asarray(value).copy())
    prefix = f"trajectories/{case_id}/{name}"
    arrays: dict[str, Any] = {
        "actions": save_array(root, f"{prefix}/actions.npy", np.stack(action_rows)),
        "sim_state_flattened": save_array(root, f"{prefix}/sim_state_flattened.npy", np.stack(states)),
        "qpos": save_array(root, f"{prefix}/qpos.npy", np.stack(qpos)),
        "qvel": save_array(root, f"{prefix}/qvel.npy", np.stack(qvel)),
        "ctrl": save_array(root, f"{prefix}/ctrl.npy", np.stack(ctrl)),
    }
    for key, values in observations.items():
        arrays[f"obs:{key}"] = save_array(root, f"{prefix}/obs_{key}.npy", np.stack(values))
    return {"length": len(action_rows), "arrays": arrays}


def _fixed_script(case_id: str, root: Path, length: int = 64) -> np.ndarray:
    source = DEFAULT_FIXED_TRACE_ROOT / f"{case_id}.npy"
    if source.is_file():
        values = np.load(source, allow_pickle=False).astype(np.float32, copy=False)
        if values.ndim == 2 and values.shape[1] == 7 and len(values) >= length:
            return np.ascontiguousarray(values[:length])
    base = np.array([0.02, -0.015, 0.01, 0.005, -0.004, 0.003, 0.0], dtype=np.float32)
    return np.repeat(base[None, :], length, axis=0)


def _probe_case(row: dict[str, Any], root: Path, *, replica: int, seed_before_constructor: bool, init_mode: str, monitor_samples: list[dict[str, Any]]) -> dict[str, Any]:
    case_id = str(row["case_id"])
    env = _env_factory(row, seed_before_constructor=seed_before_constructor)
    monitor_samples.append(gpu_snapshot())
    snapshots: list[dict[str, Any]] = []
    try:
        snapshots.append({"stage": "E0", **snapshot_record(env=env, obs={}, root=root, case_id=case_id, label=f"replica{replica}_E0", step=0, manifest_row=row)})
        obs = env.reset()
        snapshots.append({"stage": "E1", **snapshot_record(env=env, obs=obs, root=root, case_id=case_id, label=f"replica{replica}_E1", step=0, manifest_row=row)})
        obs = _init_observation(env, row, init_mode=init_mode)
        snapshots.append({"stage": "E2", **snapshot_record(env=env, obs=obs, root=root, case_id=case_id, label=f"replica{replica}_E2", step=0, manifest_row=row)})
        snapshots[-1]["controller"] = _controller_descriptor(env)
        zero = np.zeros((7,), dtype=np.float32)
        warmup_states: list[np.ndarray] = []
        for warmup_step in range(10):
            obs, _, _, _ = env.step(zero)
            warmup_states.append(np.asarray(env.sim.get_state().flatten()).copy())
            snapshots.append({"stage": "E3", "warmup_step": warmup_step + 1, **snapshot_record(env=env, obs=obs, root=root, case_id=case_id, label=f"replica{replica}_E3", step=warmup_step + 1, manifest_row=row)})
        zero_trajectory = _trajectory(env, root, case_id, f"replica{replica}_zero64", np.zeros((64, 7), dtype=np.float32))

        obs = _init_observation(env, row, init_mode=init_mode)
        for _ in range(10):
            obs, _, _, _ = env.step(zero)
        fixed_trajectory = _trajectory(env, root, case_id, f"replica{replica}_fixed64", _fixed_script(case_id, root))
        return {
            "case_id": case_id,
            "manifest": {key: row.get(key) for key in ("task_name", "language", "bddl_sha256", "init_state_index", "init_state_sha256", "env_seed", "policy_seed", "max_steps")},
            "model": snapshots[0]["model"],
            "controller": snapshots[2].get("controller"),
            "snapshots": snapshots,
            "trajectories": {"zero64": zero_trajectory, "fixed64": fixed_trajectory},
        }
    finally:
        env.close()


def run_probe(*, manifest: Path, output: Path, gpu_id: int, mode: str, init_mode: str, replica_count: int, case_ids: tuple[str, ...]) -> dict[str, Any]:
    if mode not in {"baseline", "seeded_constructor"}:
        raise ValueError(mode)
    if init_mode not in {"direct", "official"}:
        raise ValueError(init_mode)
    rows = _load_cases(manifest, case_ids)
    before = gpu_snapshot()
    monitor_samples = [before]
    started = time.time()
    records = []
    for replica in range(replica_count):
        records.append({
            "replica": replica,
            "cases": [
                _probe_case(row, output, replica=replica, seed_before_constructor=mode == "seeded_constructor", init_mode=init_mode, monitor_samples=monitor_samples)
                for row in rows
            ],
        })
    monitor_samples.append(gpu_snapshot())
    evidence = gpu_boundary_evidence(monitor_samples, target_pids={os.getpid()}, allowed_gpu_ids={int(gpu_id)})
    summary = {
        "schema_version": 1,
        "phase": "D1",
        "status": "completed",
        "probe": "environment_only",
        "mode": mode,
        "init_mode": init_mode,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "git": {
            "root": git_info(ROOT),
            "nested_openpi": git_info(ROOT / "third_party/axis_openpi"),
        },
        "case_ids": list(case_ids),
        "replica_count": replica_count,
        "gpu_id_physical": int(gpu_id),
        "gpu_id_visible": 0,
        "gpu0_touched": evidence["gpu0_touched"],
        "gpu_monitor_evidence": evidence,
        "runtime_environment": {key: os.environ.get(key) for key in ("LIBERO_CONFIG_PATH", "CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "JAX_PLATFORMS", "XLA_FLAGS", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "records": records,
    }
    write_probe_summary(output / "summary.json", summary)
    return summary


def _descriptor_signature(value: Any) -> Any:
    if isinstance(value, dict):
        if {"dtype", "shape", "data_sha256"}.issubset(value):
            return {key: value.get(key) for key in ("dtype", "shape", "data_sha256")}
        return {key: _descriptor_signature(item) for key, item in sorted(value.items()) if key not in {"path", "file_sha256", "runtime", "label", "replica"}}
    if isinstance(value, list):
        return [_descriptor_signature(item) for item in value]
    return value


def _array_from_descriptor(descriptor: dict[str, Any] | None) -> np.ndarray | None:
    """Load one of the lossless arrays referenced by a probe descriptor."""

    if not descriptor:
        return None
    path = descriptor.get("path")
    if not path:
        return None
    value = np.load(Path(path), allow_pickle=False)
    expected = descriptor.get("data_sha256")
    if expected:
        actual = __import__("hashlib").sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"probe array hash mismatch: {path}")
    return value


def _image_diff_metrics(left: np.ndarray | None, right: np.ndarray | None) -> dict[str, Any]:
    """Return pixel- and element-level metrics without applying a tolerance."""

    if left is None or right is None:
        return {
            "exact": left is None and right is None,
            "shape_match": left is None and right is None,
            "dtype_match": left is None and right is None,
            "changed_element_count": None,
            "changed_pixel_count": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "p99_abs_diff": None,
            "affected_frame_count": None,
            "total_frame_count": None,
            "first_affected_frame": None,
        }
    shape_match = left.shape == right.shape
    dtype_match = left.dtype == right.dtype
    if not shape_match:
        return {
            "exact": False,
            "shape_match": False,
            "dtype_match": dtype_match,
            "changed_element_count": None,
            "changed_pixel_count": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "p99_abs_diff": None,
            "affected_frame_count": None,
            "total_frame_count": None,
            "first_affected_frame": None,
        }
    # Images in the probe are uint8.  Use a signed wide type so subtraction
    # cannot wrap around; the same code remains valid for transformed floats.
    difference = np.abs(left.astype(np.int64, copy=False) - right.astype(np.int64, copy=False))
    changed = difference != 0
    if difference.ndim >= 3:
        # [H,W,C] and [T,H,W,C] are both supported.  A 2-D image has one
        # scalar per pixel; a channel axis is collapsed for pixel counts.
        if difference.ndim == 3:
            pixel_changed = np.any(changed, axis=-1)
            frame_changed = bool(np.any(pixel_changed))
            affected_frame_count = int(frame_changed)
            total_frame_count = 1
            first_frame = 0 if frame_changed else None
        else:
            pixel_changed = np.any(changed, axis=-1)
            frame_axes = tuple(range(1, pixel_changed.ndim))
            frame_changed_values = np.any(pixel_changed, axis=frame_axes)
            affected_frame_count = int(np.count_nonzero(frame_changed_values))
            total_frame_count = int(frame_changed_values.shape[0])
            first_frame = int(np.flatnonzero(frame_changed_values)[0]) if affected_frame_count else None
        changed_pixel_count = int(np.count_nonzero(pixel_changed))
    else:
        changed_pixel_count = int(np.count_nonzero(changed))
        affected_frame_count = int(np.count_nonzero(changed))
        total_frame_count = int(changed.size)
        first_frame = int(np.flatnonzero(changed)[0]) if np.any(changed) else None
    flat = difference.reshape(-1)
    return {
        "exact": bool(shape_match and dtype_match and not np.any(changed)),
        "shape_match": True,
        "dtype_match": dtype_match,
        "changed_element_count": int(np.count_nonzero(changed)),
        "changed_pixel_count": changed_pixel_count,
        "max_abs_diff": int(np.max(flat)) if flat.size else 0,
        "mean_abs_diff": float(np.mean(flat)) if flat.size else 0.0,
        "p99_abs_diff": float(np.percentile(flat, 99)) if flat.size else 0.0,
        "affected_frame_count": affected_frame_count,
        "total_frame_count": total_frame_count,
        "first_affected_frame": first_frame,
    }


def _camera_from_key(key: str) -> str:
    if key == "agentview_image":
        return "agentview"
    if key == "robot0_eye_in_hand_image":
        return "robot0_eye_in_hand"
    return key


def _compare_descriptor_exact(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.get("dtype") == right.get("dtype")
        and left.get("shape") == right.get("shape")
        and left.get("data_sha256") == right.get("data_sha256")
    )


def _snapshot_by_stage(snapshots: list[dict[str, Any]]) -> dict[tuple[str, int | None], dict[str, Any]]:
    return {(str(item.get("stage")), item.get("warmup_step")): item for item in snapshots}


def _compare_snapshot_semantics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_snapshots = _snapshot_by_stage(left.get("snapshots", []))
    right_snapshots = _snapshot_by_stage(right.get("snapshots", []))
    stages = [
        ("E0", None, "constructor_exact"),
        ("E1", None, "post_reset_exact"),
        ("E2", None, "post_init_exact"),
        ("E3", 10, "policy_boundary_exact"),
    ]
    stage_results: dict[str, Any] = {}
    renderer: dict[str, Any] = {}
    dynamics_all = True
    compared_snapshot_keys: set[tuple[str, int | None]] = set()
    for stage, warmup_step, result_key in stages:
        compared_snapshot_keys.add((stage, warmup_step))
        l_snap = left_snapshots.get((stage, warmup_step))
        r_snap = right_snapshots.get((stage, warmup_step))
        if l_snap is None or r_snap is None:
            stage_results[result_key] = False
            dynamics_all = False
            continue
        model_exact = _descriptor_signature(l_snap.get("model")) == _descriptor_signature(r_snap.get("model"))
        controller_exact = _descriptor_signature(l_snap.get("controller")) == _descriptor_signature(r_snap.get("controller"))
        simulator = l_snap.get("simulator", {})
        other_simulator = r_snap.get("simulator", {})
        simulator_exact = all(
            _compare_descriptor_exact(simulator.get(key), other_simulator.get(key))
            for key in ("flattened", "qpos", "qvel")
        )
        policy_exact = _compare_descriptor_exact(l_snap.get("policy_state"), r_snap.get("policy_state"))
        non_image_keys = sorted(
            (set(l_snap.get("observation", {})) | set(r_snap.get("observation", {})))
            - set(IMAGE_OBSERVATION_KEYS)
        )
        non_image_exact = all(
            _compare_descriptor_exact(l_snap.get("observation", {}).get(key), r_snap.get("observation", {}).get(key))
            for key in non_image_keys
        )
        stage_results[result_key] = bool(model_exact and controller_exact and simulator_exact and policy_exact and non_image_exact)
        dynamics_all = dynamics_all and stage_results[result_key]
        for image_key in IMAGE_OBSERVATION_KEYS:
            metrics = _image_diff_metrics(
                _array_from_descriptor(l_snap.get("observation", {}).get(image_key)),
                _array_from_descriptor(r_snap.get("observation", {}).get(image_key)),
            )
            renderer[f"{image_key}:{stage}{'' if warmup_step is None else f'-{warmup_step}'}"] = metrics
    # Keep every warm-up image in the audit view.  The policy-boundary gate is
    # specifically step 10, but earlier frames are needed to expose sparse
    # renderer jitter without misclassifying it as dynamics divergence.
    for snapshot_key in sorted(set(left_snapshots) | set(right_snapshots), key=lambda value: (value[0], value[1] or 0)):
        stage, warmup_step = snapshot_key
        if stage != "E3" or snapshot_key in compared_snapshot_keys:
            continue
        l_snap = left_snapshots.get(snapshot_key)
        r_snap = right_snapshots.get(snapshot_key)
        for image_key in IMAGE_OBSERVATION_KEYS:
            metrics = _image_diff_metrics(
                _array_from_descriptor(l_snap.get("observation", {}).get(image_key)) if l_snap else None,
                _array_from_descriptor(r_snap.get("observation", {}).get(image_key)) if r_snap else None,
            )
            renderer[f"{image_key}:E3-{warmup_step}"] = metrics
    return {
        **stage_results,
        "dynamics_exact": dynamics_all,
        "renderer_exact_by_camera_and_stage": renderer,
    }


def _compare_trajectory_semantics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    dynamics: dict[str, bool] = {}
    renderer: dict[str, Any] = {}
    for trajectory_name in sorted(set(left.get("trajectories", {})) | set(right.get("trajectories", {}))):
        l_arrays = left.get("trajectories", {}).get(trajectory_name, {}).get("arrays", {})
        r_arrays = right.get("trajectories", {}).get(trajectory_name, {}).get("arrays", {})
        dynamics[trajectory_name] = all(
            _compare_descriptor_exact(l_arrays.get(f"{key}"), r_arrays.get(f"{key}"))
            for key in TRAJECTORY_DYNAMICS_KEYS
        )
        for image_key in IMAGE_OBSERVATION_KEYS:
            name = f"obs:{image_key}"
            renderer[f"{trajectory_name}:{_camera_from_key(image_key)}"] = _image_diff_metrics(
                _array_from_descriptor(l_arrays.get(name)),
                _array_from_descriptor(r_arrays.get(name)),
            )
    return {
        "trajectory_dynamics_exact": dynamics,
        "zero64_dynamics_exact": dynamics.get("zero64", False),
        "fixed64_dynamics_exact": dynamics.get("fixed64", False),
        "trajectory_renderer_exact_by_camera_and_stage": renderer,
        "zero64_renderer_exact_by_camera": {
            camera: metrics.get("changed_element_count") == 0
            for name, metrics in renderer.items()
            if name.startswith("zero64:")
            for camera in (name.split(":", 1)[1],)
        },
        "fixed64_renderer_exact_by_camera": {
            camera: metrics.get("changed_element_count") == 0
            for name, metrics in renderer.items()
            if name.startswith("fixed64:")
            for camera in (name.split(":", 1)[1],)
        },
    }


def compare_probe_summaries_semantic(
    first_path: Path,
    second_path: Path,
    output: Path,
    *,
    first_replica: int = 0,
    second_replica: int = 0,
) -> dict[str, Any]:
    """Compare D1 evidence while separating dynamics, policy boundary and renderer."""

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    if first.get("probe") != "environment_only" or second.get("probe") != "environment_only":
        raise ValueError("semantic comparison requires environment-only summaries")
    if first.get("manifest_sha256") != second.get("manifest_sha256") or first.get("case_ids") != second.get("case_ids"):
        raise ValueError("D1 manifest/case mismatch")
    if first.get("gpu0_touched") is not False or second.get("gpu0_touched") is not False:
        raise ValueError("D1 comparison lacks verified GPU0 boundary")
    try:
        left_cases = {record["case_id"]: record for record in first["records"][int(first_replica)]["cases"]}
        right_cases = {record["case_id"]: record for record in second["records"][int(second_replica)]["cases"]}
    except (IndexError, KeyError) as exc:
        raise ValueError("D1 replica index is not present in the summary") from exc
    cases: list[dict[str, Any]] = []
    for case_id in first["case_ids"]:
        left = left_cases[case_id]
        right = right_cases[case_id]
        snapshot = _compare_snapshot_semantics(left, right)
        trajectory = _compare_trajectory_semantics(left, right)
        cases.append({
            "case_id": case_id,
            **snapshot,
            **trajectory,
            "renderer_exact_by_camera_and_stage": {
                **snapshot["renderer_exact_by_camera_and_stage"],
                **trajectory["trajectory_renderer_exact_by_camera_and_stage"],
            },
        })
    dynamics_failed = [case["case_id"] for case in cases if not (
        case["dynamics_exact"]
        and case["policy_boundary_exact"]
        and case["zero64_dynamics_exact"]
        and case["fixed64_dynamics_exact"]
    )]
    renderer_metrics = [
        metrics
        for case in cases
        for metrics in case["renderer_exact_by_camera_and_stage"].values()
    ]
    renderer_jitter = [metric for metric in renderer_metrics if metric.get("exact") is False]
    if dynamics_failed:
        technical_state = "n1_environment_dynamics_failed"
    elif renderer_jitter:
        technical_state = "n1_seeded_constructor_effective_renderer_jitter_pending_full_c20"
    else:
        technical_state = "n1_environment_strict_exact_existing_evidence"
    result = {
        "schema_version": 2,
        "phase": "Q0",
        "comparison": "environment_only_semantic_reanalysis",
        "first_summary": str(first_path),
        "second_summary": str(second_path),
        "first_summary_sha256": sha256_file(first_path),
        "second_summary_sha256": sha256_file(second_path),
        "first_replica": int(first_replica),
        "second_replica": int(second_replica),
        "case_count": len(cases),
        "dynamics_mismatch_cases": dynamics_failed,
        "renderer_jitter_present": bool(renderer_jitter),
        "cases": cases,
        "status": "passed" if not dynamics_failed else "failed",
        "technical_state": technical_state,
    }
    write_probe_summary(output, result)
    return result


def _first_snapshot_difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, (left_snapshot, right_snapshot) in enumerate(zip(left, right)):
        for section in ("model", "controller", "simulator", "observation", "policy_state"):
            left_value = left_snapshot.get(section)
            right_value = right_snapshot.get(section)
            if _descriptor_signature(left_value) == _descriptor_signature(right_value):
                continue
            if section in {"simulator", "observation"} and isinstance(left_value, dict) and isinstance(right_value, dict):
                keys = sorted(set(left_value) | set(right_value))
                field = next((key for key in keys if _descriptor_signature(left_value.get(key)) != _descriptor_signature(right_value.get(key))), None)
                return {
                    "stage": left_snapshot.get("stage"),
                    "step": left_snapshot.get("step"),
                    "warmup_step": left_snapshot.get("warmup_step"),
                    "section": section,
                    "field": field,
                    "first": _descriptor_signature(left_value.get(field)) if field else _descriptor_signature(left_value),
                    "second": _descriptor_signature(right_value.get(field)) if field else _descriptor_signature(right_value),
                }
            return {
                "stage": left_snapshot.get("stage"),
                "step": left_snapshot.get("step"),
                "warmup_step": left_snapshot.get("warmup_step"),
                "section": section,
                "field": None,
                "first": _descriptor_signature(left_value),
                "second": _descriptor_signature(right_value),
            }
    if len(left) != len(right):
        return {"section": "snapshots", "field": "length", "first": len(left), "second": len(right)}
    return None


def _first_trajectory_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    for trajectory_name in sorted(set(left) | set(right)):
        left_arrays = left.get(trajectory_name, {}).get("arrays", {})
        right_arrays = right.get(trajectory_name, {}).get("arrays", {})
        for array_name in sorted(set(left_arrays) | set(right_arrays)):
            if _descriptor_signature(left_arrays.get(array_name)) != _descriptor_signature(right_arrays.get(array_name)):
                return {"trajectory": trajectory_name, "array": array_name, "first": _descriptor_signature(left_arrays.get(array_name)), "second": _descriptor_signature(right_arrays.get(array_name))}
    return None


def compare_probe_summaries(first_path: Path, second_path: Path, output: Path, *, first_replica: int = 0, second_replica: int = 0) -> dict[str, Any]:
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    if first.get("probe") != "environment_only" or second.get("probe") != "environment_only":
        raise ValueError("D1 comparison requires environment-only summaries")
    if first.get("manifest_sha256") != second.get("manifest_sha256") or first.get("case_ids") != second.get("case_ids"):
        raise ValueError("D1 manifest/case mismatch")
    if first.get("gpu0_touched") is not False or second.get("gpu0_touched") is not False:
        raise ValueError("D1 comparison lacks verified GPU0 boundary")
    try:
        left = {record["case_id"]: record for record in first["records"][int(first_replica)]["cases"]}
        right = {record["case_id"]: record for record in second["records"][int(second_replica)]["cases"]}
    except (IndexError, KeyError) as exc:
        raise ValueError("D1 replica index is not present in the summary") from exc
    mismatches: list[dict[str, Any]] = []
    for case_id in first["case_ids"]:
        lrecord, rrecord = left[case_id], right[case_id]
        sections = {
            "model": (lrecord.get("model"), rrecord.get("model")),
            "controller": (lrecord.get("controller"), rrecord.get("controller")),
            "snapshots": (lrecord.get("snapshots"), rrecord.get("snapshots")),
            "trajectories": (lrecord.get("trajectories"), rrecord.get("trajectories")),
        }
        for section, (left_value, right_value) in sections.items():
            if _descriptor_signature(left_value) != _descriptor_signature(right_value):
                detail = _first_snapshot_difference(left_value, right_value) if section == "snapshots" else _first_trajectory_difference(left_value, right_value) if section == "trajectories" else None
                mismatches.append({"case_id": case_id, "section": section, "first_difference": detail, "first": _descriptor_signature(left_value), "second": _descriptor_signature(right_value)})
                break
    result = {
        "schema_version": 1,
        "phase": "D1",
        "comparison": "environment_only_exact_replay",
        "first_summary": str(first_path),
        "second_summary": str(second_path),
        "first_summary_sha256": sha256_file(first_path),
        "second_summary_sha256": sha256_file(second_path),
        "first_replica": int(first_replica),
        "second_replica": int(second_replica),
        "case_count": len(first["case_ids"]),
        "mismatch_count": len(mismatches),
        "status": "passed" if not mismatches else "failed",
        "first_difference": mismatches[0] if mismatches else None,
        "mismatches": mismatches,
        "technical_state": "evaluator_environment_determinism_locked" if not mismatches else "n1_nondeterminism_root_cause_unresolved",
    }
    write_probe_summary(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--output", type=Path, default=DEFAULT_OUT / "baseline_summary")
    run.add_argument("--gpu-id", type=int, default=1)
    run.add_argument("--mode", choices=("baseline", "seeded_constructor"), required=True)
    run.add_argument("--init-mode", choices=("direct", "official"), default="direct")
    run.add_argument("--replica-count", type=int, default=1)
    run.add_argument("--case-id", action="append", dest="case_ids")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--first-replica", type=int, default=0)
    compare.add_argument("--second-replica", type=int, default=0)
    semantic = subparsers.add_parser("compare-semantic")
    semantic.add_argument("--first", type=Path, required=True)
    semantic.add_argument("--second", type=Path, required=True)
    semantic.add_argument("--output", type=Path, required=True)
    semantic.add_argument("--first-replica", type=int, default=0)
    semantic.add_argument("--second-replica", type=int, default=0)
    args = parser.parse_args()
    if args.command == "run":
        case_ids = tuple(args.case_ids or DEFAULT_CASES)
        result = run_probe(manifest=args.manifest, output=args.output, gpu_id=args.gpu_id, mode=args.mode, init_mode=args.init_mode, replica_count=args.replica_count, case_ids=case_ids)
    elif args.command == "compare-semantic":
        result = compare_probe_summaries_semantic(args.first, args.second, args.output, first_replica=args.first_replica, second_replica=args.second_replica)
    else:
        result = compare_probe_summaries(args.first, args.second, args.output, first_replica=args.first_replica, second_replica=args.second_replica)
    print(json.dumps({key: result.get(key) for key in ("status", "mismatch_count", "technical_state") if key in result}, ensure_ascii=False, indent=2))
    if result.get("status") == "failed" and args.command == "compare":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
