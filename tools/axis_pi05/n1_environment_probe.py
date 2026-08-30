"""D1 environment-only determinism probes for the frozen C20 manifest."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, sha256_file
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


def _probe_case(row: dict[str, Any], root: Path, *, replica: int, seed_before_constructor: bool, init_mode: str) -> dict[str, Any]:
    case_id = str(row["case_id"])
    env = _env_factory(row, seed_before_constructor=seed_before_constructor)
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
    started = time.time()
    records = []
    for replica in range(replica_count):
        records.append({
            "replica": replica,
            "cases": [
                _probe_case(row, output, replica=replica, seed_before_constructor=mode == "seeded_constructor", init_mode=init_mode)
                for row in rows
            ],
        })
    during = gpu_snapshot()
    after = gpu_snapshot()
    evidence = gpu_boundary_evidence([before, during, after], target_pids={__import__("os").getpid()}, allowed_gpu_ids={int(gpu_id)})
    summary = {
        "schema_version": 1,
        "phase": "D1",
        "status": "completed",
        "probe": "environment_only",
        "mode": mode,
        "init_mode": init_mode,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "case_ids": list(case_ids),
        "replica_count": replica_count,
        "gpu_id_physical": int(gpu_id),
        "gpu_id_visible": 0,
        "gpu0_touched": evidence["gpu0_touched"],
        "gpu_monitor_evidence": evidence,
        "runtime_environment": {key: __import__("os").environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID", "JAX_PLATFORMS", "XLA_FLAGS", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
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
        return {key: _descriptor_signature(item) for key, item in sorted(value.items()) if key not in {"path", "file_sha256", "runtime"}}
    if isinstance(value, list):
        return [_descriptor_signature(item) for item in value]
    return value


def compare_probe_summaries(first_path: Path, second_path: Path, output: Path) -> dict[str, Any]:
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    if first.get("probe") != "environment_only" or second.get("probe") != "environment_only":
        raise ValueError("D1 comparison requires environment-only summaries")
    if first.get("manifest_sha256") != second.get("manifest_sha256") or first.get("case_ids") != second.get("case_ids"):
        raise ValueError("D1 manifest/case mismatch")
    if first.get("gpu0_touched") is not False or second.get("gpu0_touched") is not False:
        raise ValueError("D1 comparison lacks verified GPU0 boundary")
    left = {record["case_id"]: record for record in first["records"][0]["cases"]}
    right = {record["case_id"]: record for record in second["records"][0]["cases"]}
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
                mismatches.append({"case_id": case_id, "section": section, "first": _descriptor_signature(left_value), "second": _descriptor_signature(right_value)})
                break
    result = {
        "schema_version": 1,
        "phase": "D1",
        "comparison": "environment_only_exact_replay",
        "first_summary": str(first_path),
        "second_summary": str(second_path),
        "first_summary_sha256": sha256_file(first_path),
        "second_summary_sha256": sha256_file(second_path),
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
    args = parser.parse_args()
    if args.command == "run":
        case_ids = tuple(args.case_ids or DEFAULT_CASES)
        result = run_probe(manifest=args.manifest, output=args.output, gpu_id=args.gpu_id, mode=args.mode, init_mode=args.init_mode, replica_count=args.replica_count, case_ids=case_ids)
    else:
        result = compare_probe_summaries(args.first, args.second, args.output)
    print(json.dumps({key: result.get(key) for key in ("status", "mismatch_count", "technical_state") if key in result}, ensure_ascii=False, indent=2))
    if result.get("status") == "failed" and args.command == "compare":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
