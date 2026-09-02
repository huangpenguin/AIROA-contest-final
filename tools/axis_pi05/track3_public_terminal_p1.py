#!/usr/bin/env python3
"""P1 public HDF5 full-state restore canary.

The parent process only schedules fresh workers.  Each worker explicitly uses
the clean public LIBERO checkout under ``LIBERO/`` and GPU3; it never imports
the ignored/dirty LIBERO-plus checkout or any Track 3-specific asset tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P0_LOCK = TRACK3_ROOT / "p0_lock.json"
P0_MAPPING = TRACK3_ROOT / "p0_task_mapping.json"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SNAPSHOT_ROOT = P1_ROOT / "snapshots"
PUBLIC_LIBERO = ROOT / "LIBERO"
CANARY_TASKS = (10, 20, 19)  # support, containment, articulation; task 16 activation remains separately blocked


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_mapping() -> dict[int, dict[str, Any]]:
    rows = json.loads(P0_MAPPING.read_text(encoding="utf-8"))
    return {int(row["task_index"]): row for row in rows}


def make_public_config(directory: Path) -> Path:
    config = directory / "config.yaml"
    root = PUBLIC_LIBERO / "libero/libero"
    config.write_text(
        "\n".join(
            [
                f"benchmark_root: {root}",
                f"bddl_files: {root / 'bddl_files'}",
                f"init_states: {root / 'init_files'}",
                f"datasets: {root / 'datasets'}",
                f"assets: {root / 'assets'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


def read_demo(hdf5_path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data/demo_0"]
        states = np.asarray(demo["states"][()])
        actions = np.asarray(demo["actions"][()])
        rewards = np.asarray(demo["rewards"][()]) if "rewards" in demo else None
        dones = np.asarray(demo["dones"][()]) if "dones" in demo else None
        attrs = {}
        for key, value in handle["data"].attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            attrs[str(key)] = value if isinstance(value, (str, int, float, bool)) else repr(value)
    if states.ndim != 2 or states.shape[0] < 2:
        raise RuntimeError(f"invalid full state shape: {states.shape}")
    return {"states": states, "actions": actions, "rewards": rewards, "dones": dones, "attrs": attrs}


def candidate_state_indices(demo: dict[str, Any]) -> list[int]:
    import numpy as np

    states = demo["states"]
    rewards = demo["rewards"]
    indices: list[int] = []
    if rewards is not None and rewards.size:
        positive = np.flatnonzero(np.asarray(rewards).reshape(-1) > 0)
        for action_index in positive[::-1]:
            state_index = int(action_index) + 1 if states.shape[0] == len(demo["actions"]) + 1 else int(action_index)
            if 0 <= state_index < states.shape[0] and state_index not in indices:
                indices.append(state_index)
    if states.shape[0] - 1 not in indices:
        indices.append(states.shape[0] - 1)
    return indices


def image_from_obs(obs: dict[str, Any], suffix: str) -> Any:
    import numpy as np

    for key in (f"{suffix}_image", suffix):
        if key in obs:
            image = np.asarray(obs[key])
            if image.ndim == 3:
                return image
    return None


def save_png(path: Path, image: Any) -> str | None:
    if image is None:
        return None
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), array[..., ::-1])
    return sha256_file(path)


def capture_live_forward_success(demo: dict[str, Any], public_bddl: Path, render_env: Any) -> dict[str, Any] | None:
    """Capture the first success from a public demonstration in this env."""
    import numpy as np

    env = render_env(
        bddl_file_name=str(public_bddl),
        control_freq=20,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
        render_gpu_device_id=0,
    )
    try:
        env.reset()
        for action_index, action in enumerate(demo["actions"]):
            env.step(np.asarray(action, dtype=np.float64))
            if env.check_success():
                return {
                    "state": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
                    "action_index": action_index,
                }
    finally:
        env.close()
    return None


def restore_once(task_index: int, attempt: int, mapping: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    # These imports occur only inside the explicit P1 worker.
    from libero.libero.envs import OffScreenRenderEnv

    row = mapping[task_index]
    hdf5_path = Path(row["hdf5_local_path"])
    reverse_bddl = Path(row["reverse_bddl_path"])
    public_bddl = Path(row["bddl_path"])
    demo = read_demo(hdf5_path)
    errors: list[str] = []

    def evaluate_candidate(state: np.ndarray, source: str, state_index: int | None, source_action_index: int | None = None) -> dict[str, Any] | None:
        env = None
        reverse_env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=str(public_bddl),
                control_freq=20,
                use_camera_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=128,
                camera_widths=128,
                render_gpu_device_id=0,
            )
            env.reset()
            env.set_init_state(state)
            forward_before = bool(env.check_success())
            source_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            settled = []
            last_action = np.asarray(demo["actions"][-1], dtype=np.float64) if len(demo["actions"]) else np.zeros(7)
            settle_action = np.zeros(7, dtype=np.float64)
            settle_action[-1] = 1.0 if last_action[-1] >= 0 else -1.0
            for _ in range(20):
                obs, _, _, _ = env.step(settle_action)
                settled.append(bool(env.check_success()))
            forward_after = bool(env.check_success())
            stable_state = np.asarray(env.get_sim_state(), dtype=np.float64)
            pre_images = {
                "front": image_from_obs(obs, "agentview"),
                "wrist": image_from_obs(obs, "robot0_eye_in_hand"),
            }
            reverse_env = OffScreenRenderEnv(
                bddl_file_name=str(reverse_bddl),
                control_freq=20,
                use_camera_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=128,
                camera_widths=128,
                render_gpu_device_id=0,
            )
            reverse_env.reset()
            reverse_env.set_init_state(stable_state)
            reverse_before = bool(reverse_env.check_success())
            reverse_settled = []
            for _ in range(20):
                reverse_obs, _, _, _ = reverse_env.step(settle_action)
                reverse_settled.append(bool(reverse_env.check_success()))
            reverse_after = bool(reverse_env.check_success())
            if not (forward_before and forward_after and all(settled) and not reverse_before and not reverse_after and all(not value for value in reverse_settled)):
                errors.append(f"source={source}, state={state_index}, action={source_action_index}: forward={forward_before}/{forward_after}, reverse={reverse_before}/{reverse_after}")
                return None

            snapshot_dir = P1_SNAPSHOT_ROOT / f"task_{task_index:02d}" / f"attempt_{attempt}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            np.save(snapshot_dir / "source_terminal_state.npy", source_state)
            np.save(snapshot_dir / "terminal_state.npy", stable_state)
            post_images = {
                "front": image_from_obs(reverse_obs, "agentview"),
                "wrist": image_from_obs(reverse_obs, "robot0_eye_in_hand"),
            }
            image_hashes = {
                "forward_front": save_png(snapshot_dir / "forward_front.png", pre_images["front"]),
                "forward_wrist": save_png(snapshot_dir / "forward_wrist.png", pre_images["wrist"]),
                "reverse_front": save_png(snapshot_dir / "reverse_front.png", post_images["front"]),
                "reverse_wrist": save_png(snapshot_dir / "reverse_wrist.png", post_images["wrist"]),
            }
            return {
                "task_index": task_index,
                "attempt": attempt,
                "state_index": state_index,
                "source_action_index": source_action_index,
                "source": source,
                "hdf5_path": str(hdf5_path),
                "hdf5_sha256": row["hdf5_sha256"],
                "public_bddl_path": str(public_bddl),
                "public_bddl_sha256": row["bddl_sha256"],
                "reverse_bddl_path": str(reverse_bddl),
                "forward_success_before_settle": forward_before,
                "forward_success_after_settle": forward_after,
                "forward_settle_success": all(settled),
                "reverse_success_before_settle": reverse_before,
                "reverse_success_after_settle": reverse_after,
                "reverse_settle_not_success": all(not value for value in reverse_settled),
                "source_terminal_state_sha256": sha256_file(snapshot_dir / "source_terminal_state.npy"),
                "terminal_state_sha256": sha256_file(snapshot_dir / "terminal_state.npy"),
                "image_hashes": image_hashes,
                "video_codec": "not_yet_recorded",
                "status": "accepted_p1_snapshot_pending_video_review",
            }
        except Exception as exc:
            errors.append(f"source={source}, state={state_index}, action={source_action_index}: {type(exc).__name__}: {exc}")
            return None
        finally:
            if reverse_env is not None:
                reverse_env.close()
            if env is not None:
                env.close()

    for state_index in candidate_state_indices(demo):
        selected = evaluate_candidate(
            np.asarray(demo["states"][state_index], dtype=np.float64),
            "official_hdf5_terminal_state",
            state_index,
        )
        if selected is not None:
            return selected

    live = capture_live_forward_success(demo, public_bddl, OffScreenRenderEnv)
    if live is not None:
        selected = evaluate_candidate(
            live["state"],
            "live_public_forward_success_capture",
            None,
            int(live["action_index"]),
        )
        if selected is not None:
            return selected
        errors.append(f"live capture action={live['action_index']} failed restore/stability")
    else:
        errors.append("live public forward replay did not reach success")
    return {
        "task_index": task_index,
        "attempt": attempt,
        "source": "official_hdf5_terminal_state_or_live_public_forward_success_capture",
        "hdf5_path": str(hdf5_path),
        "hdf5_sha256": row["hdf5_sha256"],
        "status": "rejected_restore_or_stability",
        "errors": errors,
    }


def worker(task_index: int, attempt: int) -> int:
    mapping = load_mapping()
    result = restore_once(task_index, attempt, mapping)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"].startswith("accepted") else 2


def run_canary() -> dict[str, Any]:
    if not P0_LOCK.is_file() or not P0_MAPPING.is_file():
        raise RuntimeError("P0 lock/mapping is required before P1")
    mapping = load_mapping()
    results = []
    for task_index in CANARY_TASKS:
        if task_index not in mapping:
            raise RuntimeError(f"missing P0 mapping for task {task_index}")
        for attempt in range(1, 4):
            env = os.environ.copy()
            config_dir = Path(tempfile.mkdtemp(prefix="axis_libero_public_config_"))
            make_public_config(config_dir)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "3",
                    "MUJOCO_EGL_DEVICE_ID": "3",
                    "MUJOCO_GL": "egl",
                    "LIBERO_CONFIG_PATH": str(config_dir),
                    "PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}",
                }
            )
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(task_index), str(attempt)]
            completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
            if completed.stdout.strip():
                try:
                    result = json.loads(completed.stdout.strip().splitlines()[-1])
                except json.JSONDecodeError:
                    result = {"status": "worker_output_invalid", "stdout": completed.stdout[-4000:]}
            else:
                result = {"status": "worker_no_output", "stderr": completed.stderr[-4000:]}
            result["worker_returncode"] = completed.returncode
            results.append(result)
            if not result.get("status", "").startswith("accepted"):
                break
    accepted = [row for row in results if row.get("status", "").startswith("accepted")]
    by_task = {int(row["task_index"]): row for row in accepted}
    summary = {
        "schema_version": "track3_public_terminal_p1_v1",
        "status": "track3_terminal_snapshot_canary_passed_pending_review" if len(by_task) == len(CANARY_TASKS) else "track3_public_terminal_source_blocked",
        "created_at_utc": utc_now(),
        "gpu_policy": {"gpu0_touched": False, "gpu3_only": True, "gpu1_gpu2_training_started": False},
        "canary_task_indices": list(CANARY_TASKS),
        "required_fresh_process_replay": "3/3",
        "accepted_snapshot_count": len(by_task),
        "accepted_task_indices": sorted(by_task),
        "attempt_results": results,
        "next_phase_authorization": "P2_recorder_and_manual_input_only" if len(by_task) == len(CANARY_TASKS) else "stop_and_remediate_source",
    }
    write_json(P1_ROOT / "p1_restore_canary_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, type=int, metavar=("TASK_INDEX", "ATTEMPT"))
    parser.add_argument("--run-canary", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(*args.worker)
    if args.run_canary:
        summary = run_canary()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["accepted_snapshot_count"] == len(CANARY_TASKS) else 2
    parser.error("choose --run-canary or --worker TASK_INDEX ATTEMPT")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
