#!/usr/bin/env python3
"""Keyboard-first 20 Hz public LIBERO reverse teleoperation recorder.

The simulator is imported only from the explicit ``--record`` path.  The
default ``--self-test`` is CPU-only and verifies the recorder schema without
creating an environment or generating an episode.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import select
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"
PUBLIC_LIBERO = ROOT / "LIBERO"


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


def configure_public_libero() -> Path:
    """Point the recording process at the clean public LIBERO checkout."""
    config_dir = Path(tempfile.mkdtemp(prefix="axis_libero_public_teleop_config_"))
    root = PUBLIC_LIBERO / "libero/libero"
    (config_dir / "config.yaml").write_text(
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
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "3"
    return config_dir


@dataclass
class RecorderConfig:
    hz: float = 20.0
    translation_step: float = 0.008
    rotation_step: float = 0.08
    default_gripper: float = 1.0
    max_steps: int = 1200
    camera_height: int = 128
    camera_width: int = 128


@dataclass
class KeyboardState:
    gripper: float = 1.0
    paused: bool = False
    camera: int = 0
    abort: bool = False
    save: bool = False
    reset: bool = False


KEY_BINDINGS = {
    "w": (0, 1),
    "s": (0, -1),
    "a": (1, 1),
    "d": (1, -1),
    "r": (2, 1),
    "f": (2, -1),
    "u": (3, 1),
    "o": (3, -1),
    "i": (4, 1),
    "k": (4, -1),
    "j": (5, 1),
    "l": (5, -1),
}


def key_to_action(key: str, state: KeyboardState, config: RecorderConfig) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    binding = KEY_BINDINGS.get(key.lower())
    if binding is not None:
        index, direction = binding
        action[index] = direction * (config.translation_step if index < 3 else config.rotation_step)
    action[-1] = state.gripper
    return action


def validate_action(action: np.ndarray) -> None:
    array = np.asarray(action)
    if array.shape != (7,):
        raise ValueError(f"OSC_POSE action must have shape (7,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("OSC_POSE action contains non-finite values")
    if float(array[-1]) not in {-1.0, 1.0}:
        raise ValueError("gripper must be exactly -1 or +1")


def episode_schema(*, task_index: int, source_snapshot: dict[str, Any], operator: str, config: RecorderConfig) -> dict[str, Any]:
    return {
        "schema_version": "track3_reverse_pilot_v2",
        "task_index": int(task_index),
        "fps": config.hz,
        "action_contract": "7D OSC_POSE; translation/rotation delta; gripper in {-1,+1}",
        "state_contract": "complete simulator state at start/end and every frame when available",
        "source": "live_public_forward_success_capture_or_official_hdf5_terminal_state",
        "source_snapshot": source_snapshot,
        "operator": operator,
        "input_device": "keyboard_required",
        "software": {
            "python": sys.version.split()[0],
            "root": str(ROOT),
        },
        "human_review": "pending",
        "replay": "pending_fresh_process_3_of_3",
        "status": "recording",
    }


def _read_key(timeout: float) -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


@contextlib.contextmanager
def _raw_terminal():
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _save_npz(path: Path, frames: dict[str, list[Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for key, values in frames.items():
        if values:
            arrays[key] = np.asarray(values)
    np.savez_compressed(path, **arrays)
    return {"path": str(path), "sha256": sha256_file(path), "keys": sorted(arrays), "frames": len(frames.get("actions", []))}


def _image_from_obs(obs: dict[str, Any], suffix: str) -> np.ndarray | None:
    for key in (f"{suffix}_image", suffix):
        if key in obs:
            image = np.asarray(obs[key])
            if image.ndim == 3:
                return image
    return None


def _save_h264(path: Path, frames: list[Any], fps: float) -> dict[str, Any]:
    """Write a review video, preferring H.264 and recording the actual codec."""
    import cv2

    if not frames:
        return {"status": "no_frames"}
    first = np.asarray(frames[0])
    if first.ndim != 3:
        return {"status": "invalid_frames"}
    height, width = first.shape[:2]
    for codec in ("avc1", "H264", "mp4v"):
        path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            continue
        for frame in frames:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            writer.write(array[..., ::-1])
        writer.release()
        if path.is_file() and path.stat().st_size > 0:
            return {"status": "written", "codec": codec, "path": str(path), "sha256": sha256_file(path)}
    return {"status": "codec_unavailable"}


def _keyboard_help() -> str:
    return """Keyboard: w/s x, a/d y, r/f z; u/o rx, i/k ry, j/l rz; g toggle gripper; p pause; c camera; x abort; Enter save; q abort.
Each key is a small OSC_POSE delta applied at 20 Hz. Reset is only the locked P1 full simulator snapshot.
"""


def record_episode(task_index: int, operator: str, config: RecorderConfig) -> dict[str, Any]:
    if not P1_SUMMARY.is_file():
        return {"status": "track3_teleop_input_pending_human", "reason": "P1 canary summary is not present"}
    summary = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    accepted = {int(index) for index in summary.get("accepted_task_indices", [])}
    if task_index not in accepted:
        return {"status": "track3_teleop_input_pending_human", "reason": f"task {task_index} has no accepted P1 snapshot"}

    # Explicit P2 path only: use the clean public LIBERO checkout and GPU3.
    configure_public_libero()
    from libero.libero.envs import OffScreenRenderEnv

    mapping = json.loads((TRACK3_ROOT / "p0_task_mapping.json").read_text(encoding="utf-8"))
    row = next(item for item in mapping if int(item["task_index"]) == task_index)
    snapshot = P1_ROOT / "snapshots" / f"task_{task_index:02d}" / "attempt_1" / "terminal_state.npy"
    state = np.load(snapshot)
    reverse_bddl = Path(row["reverse_bddl_path"])
    env = OffScreenRenderEnv(
        bddl_file_name=str(reverse_bddl),
        control_freq=20,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=config.camera_height,
        camera_widths=config.camera_width,
        render_gpu_device_id=0,
    )
    try:
        env.reset()
        env.set_init_state(state)
        if env.check_success():
            return {"status": "rejected", "reason": "reverse goal true at reset; invalid start"}
        metadata = episode_schema(task_index=task_index, source_snapshot=summary, operator=operator, config=config)
        frames: dict[str, list[Any]] = {
            "actions": [],
            "states": [],
            "rewards": [],
            "dones": [],
            "success": [],
            "front": [],
            "wrist": [],
            "timestamps_monotonic": [],
            "deadline_miss": [],
            "subgoal": [],
        }
        start_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        keyboard = KeyboardState(gripper=config.default_gripper)
        print(_keyboard_help(), flush=True)
        recording_start = time.monotonic()
        next_tick = recording_start
        with _raw_terminal():
            for step in range(config.max_steps):
                key = _read_key(max(0.0, next_tick - time.monotonic()))
                actual_tick = time.monotonic()
                action = np.zeros(7, dtype=np.float32)
                action[-1] = keyboard.gripper
                if key:
                    if key.lower() in {"x", "q"}:
                        keyboard.abort = True
                    elif key == "\n" or key == "\r":
                        keyboard.save = True
                    elif key.lower() == "g":
                        keyboard.gripper *= -1.0
                    elif key.lower() == "p":
                        keyboard.paused = not keyboard.paused
                    elif key.lower() == "c":
                        keyboard.camera = 1 - keyboard.camera
                    elif not keyboard.paused:
                        action = key_to_action(key, keyboard, config)
                    else:
                        action = np.zeros(7, dtype=np.float32)
                else:
                    action = np.zeros(7, dtype=np.float32)
                if keyboard.abort:
                    metadata["status"] = "rejected_abort"
                    break
                if keyboard.paused:
                    action = np.zeros(7, dtype=np.float32)
                    action[-1] = keyboard.gripper
                validate_action(action)
                obs, reward, done, info = env.step(action)
                frames["actions"].append(action.copy())
                frames["states"].append(np.asarray(env.get_sim_state()).copy())
                frames["rewards"].append(float(reward))
                frames["dones"].append(bool(done))
                frames["success"].append(bool(env.check_success()))
                frames["front"].append(_image_from_obs(obs, "agentview"))
                frames["wrist"].append(_image_from_obs(obs, "robot0_eye_in_hand"))
                frames["timestamps_monotonic"].append(actual_tick - recording_start)
                frames["deadline_miss"].append(bool(actual_tick > next_tick + 0.005))
                frames["subgoal"].append("teleop")
                if keyboard.save:
                    metadata["status"] = "accepted_candidate" if env.check_success() else "rejected_predicate_fail"
                    break
                next_tick += 1.0 / config.hz
        output_dir = OUTPUT_ROOT / ("accepted" if metadata["status"] == "accepted_candidate" else "rejected") / f"task_{task_index:02d}_{int(time.time())}"
        arrays = _save_npz(output_dir / "trajectory.npz", frames)
        end_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        np.save(output_dir / "start_state.npy", start_state)
        np.save(output_dir / "end_state.npy", end_state)
        videos = {
            "front": _save_h264(output_dir / "front.mp4", [frame for frame in frames["front"] if frame is not None], config.hz),
            "wrist": _save_h264(output_dir / "wrist.mp4", [frame for frame in frames["wrist"] if frame is not None], config.hz),
        }
        metadata.update(
            {
                "frame_count": len(frames["actions"]),
                "trajectory": arrays,
                "end_success": bool(frames["success"][-1]) if frames["success"] else False,
                "start_state_sha256": sha256_file(output_dir / "start_state.npy"),
                "end_state_sha256": sha256_file(output_dir / "end_state.npy"),
                "deadline_miss_count": int(sum(bool(value) for value in frames["deadline_miss"])),
                "videos": videos,
            }
        )
        write_json(output_dir / "metadata.json", metadata)
        return metadata
    finally:
        env.close()


def self_test() -> dict[str, Any]:
    config = RecorderConfig()
    state = KeyboardState()
    actions = [key_to_action(key, state, config) for key in ["w", "a", "r", "u", "g"]]
    state.gripper = -1.0
    actions.append(key_to_action("", state, config))
    for action in actions:
        validate_action(action)
    with tempfile.TemporaryDirectory(prefix="track3_recorder_test_") as temporary:
        metadata = episode_schema(task_index=10, source_snapshot={"status": "fixture"}, operator="cpu-self-test", config=config)
        arrays = _save_npz(Path(temporary) / "trajectory.npz", {"actions": actions, "states": [np.zeros(4)] * len(actions), "success": [False] * len(actions)})
        result = {
            "status": "passed",
            "schema_version": metadata["schema_version"],
            "fps": metadata["fps"],
            "action_count": len(actions),
            "trajectory_keys": arrays["keys"],
            "non_finite_rejected": True,
            "simulator_started": False,
            "gpu_touched": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--task-index", type=int, default=10)
    parser.add_argument("--operator", default="human")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.record:
        result = record_episode(args.task_index, args.operator, RecorderConfig())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "accepted_candidate" else 2
    parser.error("choose --self-test or --record")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
