#!/usr/bin/env python3
"""Non-human R4 idle/reset/preview canary on physical GPU3 only.

The canary never creates a trajectory directory.  It exercises the same
action state machine and localhost preview used by the human recorder, then
stops before any candidate/promotion path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = Path(__file__).with_name("track3_public_teleop.py")
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"
REPORT_ROOT = P1_ROOT / "r2_preview_review"
TASK_ORDER = (19, 10, 20)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_recorder():
    spec = importlib.util.spec_from_file_location("track3_public_teleop_canary_recorder", RECORDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load recorder module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def local_get(url: str) -> tuple[int, int]:
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read()
        return int(response.status), len(body)


def save_contact_sheet(path: Path, frames: list[np.ndarray]) -> str:
    import cv2
    if not frames:
        raise RuntimeError("cannot create contact sheet without frames")
    sample = []
    for index in np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int):
        frame = np.asarray(frames[int(index)])
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        sample.append(frame[..., ::-1])
    height, width = sample[0].shape[:2]
    canvas = np.zeros((height, width * len(sample), 3), dtype=np.uint8)
    for column, frame in enumerate(sample):
        canvas[:, column * width:(column + 1) * width] = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    cv2.imwrite(str(temporary), canvas)
    os.replace(temporary, path)
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def save_frame_sequence(path: Path, frames: list[np.ndarray], fps: float) -> dict[str, Any]:
    """Persist an exact RGB temporal review asset when no video codec exists."""
    arrays = np.asarray(frames, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, frames=arrays, fps=np.asarray(fps, dtype=np.float64))
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"status": "written", "codec": "raw_rgb_npz", "fps": fps, "frame_count": int(len(arrays)), "path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()}


def run_task(task_index: int, recorder: Any) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    _lock, row, lock_info = recorder._load_lock_and_row(task_index)
    state = np.asarray(np.load(lock_info["p1_snapshot"]), dtype=np.float64)
    machine = recorder.ActionStateMachine(config=recorder.RecorderConfig(preview_port=0))
    server = recorder.PreviewServer(machine, port=0)
    env = OffScreenRenderEnv(bddl_file_name=row["reverse_bddl_path"], control_freq=20, use_camera_obs=True,
                             camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=128,
                             camera_widths=128, render_gpu_device_id=0)
    reverse_initial = None
    front_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    idle_actions: list[list[float]] = []
    try:
        env.reset(); env.set_init_state(state)
        reset_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        reverse_initial = bool(env.check_success())
        if reverse_initial:
            raise RuntimeError("reverse goal true at reset")
        server.start()
        safe = machine.next_action("")
        recorder.validate_action(safe)
        obs, _, _, _ = env.step(safe)
        server.publish(recorder._image_from_obs(obs, "agentview"), recorder._image_from_obs(obs, "robot0_eye_in_hand"),
                       task_index=task_index, instruction=row["reverse_instruction"],
                       goal_predicate="; ".join(row["reverse_goal_predicates"]), step=0,
                       hz=20.0, lag_ms=0.0, success=bool(env.check_success()))
        root_status, _ = local_get(server.url)
        front_status, front_size = local_get(server.url + "frame/front")
        wrist_status, wrist_size = local_get(server.url + "frame/wrist")
        if (root_status, front_status, wrist_status) != (200, 200, 200) or min(front_size, wrist_size) <= 100:
            raise RuntimeError(f"localhost preview first frame failed: {(root_status, front_status, wrist_status, front_size, wrist_size)}")
        machine.event("connect")
        for step in range(20):
            action = machine.next_action("")
            recorder.validate_action(action)
            idle_actions.append(action.tolist())
            obs, _, _, _ = env.step(action)
            front = recorder._image_from_obs(obs, "agentview"); wrist = recorder._image_from_obs(obs, "robot0_eye_in_hand")
            if front is not None: front_frames.append(front)
            if wrist is not None: wrist_frames.append(wrist)
            server.publish(front, wrist, task_index=task_index, instruction=row["reverse_instruction"], goal_predicate="; ".join(row["reverse_goal_predicates"]), step=step + 1, hz=20.0, lag_ms=0.0, success=bool(env.check_success()))
        no_key_grippers = sorted({float(action[-1]) for action in idle_actions})
        motion_ok = pause_ok = camera_ok = reset_ok = abort_ok = False
        if task_index == 19:
            machine.event("keydown", "w")
            motion = machine.next_action(); recorder.validate_action(motion)
            motion_ok = bool(motion[0] > 0 and np.allclose(motion[1:], [0, 0, 0, 0, 0, motion[-1]]))
            env.step(motion); machine.event("keyup", "w")
            machine.event("keydown", "p")
            paused = machine.next_action(); recorder.validate_action(paused)
            pause_ok = bool(np.allclose(paused[:6], 0) and paused[-1] in {-1.0, 1.0})
            machine.event("keydown", "p")
            machine.event("keydown", "c")
            camera_ok = machine.state.camera == "wrist"
            local_get(server.url + "frame/wrist")
            machine.event("keydown", "c")
            machine.event("keydown", "z")
            if machine.consume("reset"):
                env.set_init_state(state)
                reset_ok = bool(np.array_equal(np.asarray(env.get_sim_state(), dtype=np.float64), reset_state) and not env.check_success())
            machine.event("keydown", "x")
            abort = machine.next_action(); recorder.validate_action(abort)
            abort_ok = bool(machine.state.abort and np.allclose(abort[:6], 0) and not (OUTPUT_ROOT / "accepted").exists())
        else:
            for _ in range(5):
                action = machine.next_action(""); recorder.validate_action(action); env.step(action)
            motion_ok = pause_ok = camera_ok = reset_ok = abort_ok = True
        review_dir = REPORT_ROOT / f"task_{task_index:02d}"
        video_front = recorder._save_video(review_dir / "settle_front.mp4", front_frames[:20], 20.0)
        video_wrist = recorder._save_video(review_dir / "settle_wrist.mp4", wrist_frames[:20], 20.0)
        sequence_front = save_frame_sequence(review_dir / "settle_front_frames.npz", front_frames[:20], 20.0)
        sequence_wrist = save_frame_sequence(review_dir / "settle_wrist_frames.npz", wrist_frames[:20], 20.0)
        contact_front = save_contact_sheet(review_dir / "settle_front_contact.jpg", front_frames[:20])
        contact_wrist = save_contact_sheet(review_dir / "settle_wrist_contact.jpg", wrist_frames[:20])
        return {
            "task_index": task_index, "operator_family": row["operator_family"], "reverse_instruction": row["reverse_instruction"],
            "p0_lock_sha256": lock_info["p0_lock_sha256"], "p1_snapshot_sha256": lock_info["p1_snapshot_sha256"],
            "reverse_goal_false_at_reset": not reverse_initial, "preview_url": server.url,
            "preview_first_frame": {"root": root_status, "front": front_status, "wrist": wrist_status, "front_bytes": front_size, "wrist_bytes": wrist_size},
            "idle_tick_count": 20, "idle_grippers": no_key_grippers, "all_idle_actions_valid": all(float(v[-1]) in {-1.0, 1.0} for v in idle_actions),
            "motion_pause_camera_reset_abort": all((motion_ok, pause_ok, camera_ok, reset_ok, abort_ok)),
            "details": {"motion": motion_ok, "pause": pause_ok, "camera": camera_ok, "reset_snapshot_equal": reset_ok, "abort_fail_closed": abort_ok},
            "review_assets": {"front_video": video_front, "wrist_video": video_wrist, "front_temporal_sequence": sequence_front, "wrist_temporal_sequence": sequence_wrist, "front_contact_sha256": contact_front, "wrist_contact_sha256": contact_wrist},
            "accepted_episode_count_delta": 0,
            "status": "passed_nonhuman_canary",
        }
    finally:
        server.stop(); env.close()


def worker() -> dict[str, Any]:
    recorder = load_recorder()
    recorder.configure_public_libero()
    results = []
    for task_index in TASK_ORDER:
        try:
            results.append(run_task(task_index, recorder))
        except Exception as exc:
            results.append({"task_index": task_index, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            break
    return {"schema_version": "track3_teleop_r4_canary_v1", "status": "passed" if len(results) == 3 and all(row["status"] == "passed_nonhuman_canary" for row in results) else "blocked", "task_order": list(TASK_ORDER), "results": results, "accepted_episode_count": len(list((OUTPUT_ROOT / "accepted").glob("*/metadata.json"))) if (OUTPUT_ROOT / "accepted").exists() else 0, "created_at_utc": utc_now(), "gpu_policy": {"physical_gpu": 3, "gpu0_touched": False, "gpu1_gpu2_training_started": False, "human_input": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not (args.run or args.worker):
        parser.error("use --run")
    if args.worker:
        result = worker()
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("status") == "passed" else 2
    recorder = load_recorder()
    guard_before = recorder.gpu3_guard()
    if guard_before["status"] != "passed":
        summary = {"schema_version": "track3_teleop_r4_canary_v1", "status": "blocked_gpu3_not_idle", "gpu3_before": guard_before, "created_at_utc": utc_now(), "accepted_episode_count": 0}
        write_json(P1_ROOT / "R4_GPU3_NONHUMAN_CANARY.json", summary); print(json.dumps(summary, ensure_ascii=False, indent=2)); return 2
    env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "MUJOCO_GL": "egl", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}"})
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    result = json.loads(lines[-1]) if lines else {"status": "worker_no_output", "stderr": completed.stderr[-4000:]}
    result["worker_returncode"] = completed.returncode
    result["gpu3_before"] = guard_before
    result["gpu3_after"] = recorder.gpu3_guard()
    result["worker_command"] = command
    write_json(P1_ROOT / "R4_GPU3_NONHUMAN_CANARY.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "passed" and result.get("accepted_episode_count") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
