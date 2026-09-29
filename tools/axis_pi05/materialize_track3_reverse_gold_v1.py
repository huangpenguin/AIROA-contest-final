#!/usr/bin/env python3
"""Materialize accepted Track3 reverse episodes as pre-action LeRobot-v3 data.

Historical accepted episodes are read-only inputs.  The executable path uses a
fresh MuJoCo environment for every episode, emits the observation *before*
each recorded action, and verifies the resulting post-action simulator state
against the historical ``states[t]`` array before publishing a derived repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted"
DEFAULT_OUT = ROOT / "data/processed/axis_pi05_track3_reverse_gold_v1"
SOURCE_REVISION = "track3-accepted-pre-action-v1"
CONTROL_HZ = 20
# Used only to report continuous replay diagnostics.  It is not a publication
# gate for the pre-action materializer because MuJoCo contact solver state is
# not fully represented by the saved finite simulator state.
STATE_TOLERANCE = 1e-8
GRIPPER_CONTROLLER_SEEDS: dict[str, tuple[float, float]] = {
    "official_t3_alphabet_soup": (1.0, -1.0),
    "official_t3_red_mug": (1.0, -1.0),
    "official_t3_two_mugs": (-1.0, 1.0),
    "official_t3_yellow_book": (1.0, -1.0),
}

TASKS: dict[str, dict[str, Any]] = {
    "official_t3_alphabet_soup": {
        "episode": "official_t3_alphabet_soup_1788946170",
        "task_index": 0,
        "prompt": "take the alphabet soup out of the tray and place it on the table in front of the ketchup",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table.bddl",
    },
    "official_t3_red_mug": {
        "episode": "official_t3_red_mug_1788976069",
        "task_index": 1,
        "prompt": "take the red mug off the left plate and put it back on the table behind the other two mugs",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_red_mug_off_the_left_plate_and_put_it_on_the_table.bddl",
    },
    "official_t3_two_mugs": {
        "episode": "official_t3_two_mugs_1788976820",
        "task_index": 2,
        "prompt": "take the white mug off the left plate and the yellow and white mug off the right plate, then place the white mug to the left of the red mug and the yellow and white mug to the right of the red mug",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_two_mugs_off_the_plates_and_place_them_on_the_table.bddl",
    },
    "official_t3_yellow_book": {
        "episode": "official_t3_yellow_book_1788971728",
        "task_index": 3,
        "prompt": "take the yellow book off the top of the shelf and put it back on the table on the left",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def axis_angle_from_quat_xyzw(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.isfinite(value).all():
        raise ValueError("non-finite quaternion")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("zero quaternion")
    value = value / norm
    if value[3] < 0.0:
        value = -value
    angle = 2.0 * np.arccos(np.clip(value[3], -1.0, 1.0))
    scale = np.sqrt(max(0.0, 1.0 - float(value[3]) ** 2))
    if scale < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (value[:3] * angle / scale).astype(np.float32)


def state8_from_observation(observation: dict[str, Any]) -> np.ndarray:
    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"observation missing state fields: {missing}")
    state = np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(3),
            axis_angle_from_quat_xyzw(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        ]
    ).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError(f"invalid state8: {state.shape}")
    return state


def frame_from_observation(observation: dict[str, Any], suffix: str) -> np.ndarray:
    for key in (f"{suffix}_image", suffix):
        if key in observation:
            frame = np.asarray(observation[key])
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if frame.shape != (256, 256, 3):
                    raise ValueError(f"{suffix} frame shape is {frame.shape}, expected (256,256,3)")
                return frame
    raise ValueError(f"observation missing camera frame: {suffix}")


def validate_raw_episode(task_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    spec = TASKS[task_id]
    directory = RAW_ROOT / spec["episode"]
    metadata = strict_json(directory / "metadata.json")
    if metadata.get("status") != "accepted" or metadata.get("replay") != "3/3_passed":
        raise ValueError(f"raw episode is not accepted/replayed: {directory}")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"actions", "states", "success", "front", "wrist"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"raw trajectory missing {sorted(missing)}")
    frame_count = int(arrays["actions"].shape[0])
    if metadata.get("frame_count") != frame_count:
        raise ValueError("metadata/trajectory frame count mismatch")
    if arrays["actions"].shape != (frame_count, 7) or arrays["states"].shape[0] != frame_count:
        raise ValueError("raw action/state shapes are invalid")
    for key in ("actions", "states"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"raw {key} contains non-finite values")
    for key in ("front", "wrist"):
        if arrays[key].shape != (frame_count, 256, 256, 3) or arrays[key].dtype != np.uint8:
            raise ValueError(f"raw {key} schema mismatch: {arrays[key].shape} {arrays[key].dtype}")
    return metadata, arrays, directory


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    temporary = path.with_name(path.name + ".partial")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def encode_video(path: Path, frames: np.ndarray) -> None:
    if frames.ndim != 4 or frames.shape[1:] != (256, 256, 3) or frames.dtype != np.uint8:
        raise ValueError(f"video input schema mismatch: {frames.shape} {frames.dtype}")
    # Keep the media suffix visible to ffmpeg while retaining atomic rename.
    temporary = path.with_name(f"{path.stem}.partial{path.suffix}")
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "256x256", "-r", str(CONTROL_HZ), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    completed = subprocess.run(command, input=frames.tobytes(), capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.decode(errors='replace')[-2000:]}")
    os.replace(temporary, path)


def resolved_model_xml(env: Any) -> bytes:
    xml = env.sim.model.get_xml()
    return xml.encode("utf-8") if isinstance(xml, str) else bytes(xml)


def state8_from_sim_state(env: Any, full_state: np.ndarray) -> np.ndarray:
    """Restore one historical full state and extract the policy's 8D state."""
    observation = env.set_init_state(np.asarray(full_state, dtype=np.float64))
    if observation is None:
        observation = env.env._get_observations()
    if not isinstance(observation, dict):
        raise ValueError("state restore did not return an observation mapping")
    return state8_from_observation(observation)


def make_replay_env(spec: dict[str, Any], frame_count: int, *, use_camera_obs: bool) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(ROOT / spec["bddl"]),
        control_freq=CONTROL_HZ,
        use_camera_obs=use_camera_obs,
        camera_names=["agentview", "robot0_eye_in_hand"] if use_camera_obs else None,
        camera_heights=256 if use_camera_obs else None,
        camera_widths=256 if use_camera_obs else None,
        render_gpu_device_id=0,
        horizon=frame_count + 8,
        ignore_done=True,
    )


def historical_yellow_asset_overlay() -> Path:
    """Build a temporary LIBERO view with only the dirty shelf restored from HEAD."""
    overlay = Path(tempfile.mkdtemp(prefix="axis_t3_yellow_asset_overlay_")) / "LIBERO"
    package = overlay / "libero/libero"
    package.mkdir(parents=True)
    source_package = ROOT / "LIBERO/libero/libero"
    for child in source_package.iterdir():
        if child.name == "assets":
            continue
        os.symlink(child, package / child.name, target_is_directory=child.is_dir())
    source_assets = source_package / "assets"
    target_assets = package / "assets"
    target_assets.mkdir()
    for child in source_assets.iterdir():
        if child.name != "turbosquid_objects":
            os.symlink(child, target_assets / child.name, target_is_directory=child.is_dir())
    source_objects = source_assets / "turbosquid_objects"
    target_objects = target_assets / "turbosquid_objects"
    target_objects.mkdir()
    shelf_name = "wooden_two_layer_shelf"
    for child in source_objects.iterdir():
        if child.name != shelf_name:
            os.symlink(child, target_objects / child.name, target_is_directory=child.is_dir())
    shelf_target = target_objects / shelf_name
    shelf_target.mkdir()
    for child in (source_objects / shelf_name).iterdir():
        os.symlink(child, shelf_target / child.name, target_is_directory=child.is_dir())
    original = subprocess.check_output(
        ["git", "-C", str(ROOT / "LIBERO"), "show", "HEAD:libero/libero/assets/turbosquid_objects/wooden_two_layer_shelf/wooden_two_layer_shelf.xml"]
    )
    (shelf_target / "wooden_two_layer_shelf.xml").unlink()
    (shelf_target / "wooden_two_layer_shelf.xml").write_bytes(original)
    return overlay


def configure_replay(recorder: Any, task_id: str) -> tuple[Path | None, list[float]]:
    overlay: Path | None = None
    original_root = recorder.PUBLIC_LIBERO
    if task_id == "official_t3_yellow_book":
        overlay = historical_yellow_asset_overlay()
        recorder.PUBLIC_LIBERO = overlay
    recorder.configure_public_libero()
    recorder.PUBLIC_LIBERO = original_root
    return overlay, list(GRIPPER_CONTROLLER_SEEDS[task_id])


def replay_one(task_id: str, *, output_root: Path) -> dict[str, Any]:
    """Materialize pre-action samples and retain continuous replay as diagnostics."""
    import tools.axis_pi05.track3_public_teleop as recorder

    metadata, raw, raw_directory = validate_raw_episode(task_id)
    spec = TASKS[task_id]
    overlay, gripper_seed = configure_replay(recorder, task_id)
    from compe.t3 import register_t3

    register_t3()
    frame_count = int(raw["actions"].shape[0])
    start_state = np.asarray(np.load(raw_directory / "start_state.npy", allow_pickle=False), dtype=np.float64)
    pre_front: list[np.ndarray]
    pre_wrist: list[np.ndarray]
    pre_states: list[np.ndarray]
    model_xml = b""
    max_abs_error = 0.0
    first_mismatch_frame: int | None = None
    diagnostic_shape_match = True
    final_success = False
    probe_env = make_replay_env(spec, frame_count, use_camera_obs=True)
    try:
        probe_env.reset()
        observation = recorder.restore_frozen_start(probe_env, start_state)
        if observation is None:
            observation = probe_env.env._get_observations()
        model_xml = resolved_model_xml(probe_env)

        # The first sample is genuinely pre-action.  Every later sample uses
        # the historical post-action observation from the preceding frame.
        pre_front = [frame_from_observation(observation, "agentview")]
        pre_front.extend(np.asarray(frame, dtype=np.uint8).copy() for frame in raw["front"][:-1])
        pre_wrist = [frame_from_observation(observation, "robot0_eye_in_hand")]
        pre_wrist.extend(np.asarray(frame, dtype=np.uint8).copy() for frame in raw["wrist"][:-1])
        pre_states = [state8_from_observation(observation)]
        pre_states.extend(state8_from_sim_state(probe_env, full_state) for full_state in raw["states"][:-1])
    finally:
        probe_env.close()

    # A separate fresh process-equivalent environment retains the original
    # action replay as a diagnostic.  Its full-state error is recorded but is
    # deliberately not used to reject the shifted pre-action materialization.
    replay_env = make_replay_env(spec, frame_count, use_camera_obs=False)
    try:
        replay_env.reset()
        recorder.restore_frozen_start(replay_env, start_state)
        replay_env.env.robots[0].gripper.current_action = np.asarray(gripper_seed, dtype=np.float64)
        for index, action in enumerate(np.asarray(raw["actions"], dtype=np.float32)):
            replay_env.step(action)
            actual = np.asarray(replay_env.get_sim_state(), dtype=np.float64)
            expected = np.asarray(raw["states"][index], dtype=np.float64)
            if actual.shape != expected.shape:
                diagnostic_shape_match = False
                raise RuntimeError(f"{task_id} state shape mismatch at {index}: {actual.shape} != {expected.shape}")
            error = float(np.max(np.abs(actual - expected)))
            max_abs_error = max(max_abs_error, error)
            if error > STATE_TOLERANCE and first_mismatch_frame is None:
                first_mismatch_frame = index
        final_success = bool(replay_env.check_success())
        if not final_success:
            raise RuntimeError(f"{task_id} fresh replay did not finish in success state")
    finally:
        replay_env.close()
        if overlay is not None:
            shutil.rmtree(overlay.parent, ignore_errors=True)
    if len(pre_front) != frame_count or len(pre_wrist) != frame_count or len(pre_states) != frame_count:
        raise RuntimeError(f"{task_id} shifted materialization length mismatch")
    return {
        "task_id": task_id,
        "task_index": int(spec["task_index"]),
        "prompt": spec["prompt"],
        "metadata": metadata,
        "raw_directory": raw_directory,
        "actions": np.asarray(raw["actions"], dtype=np.float32),
        "states": np.asarray(pre_states, dtype=np.float32),
        "front": np.asarray(pre_front, dtype=np.uint8),
        "wrist": np.asarray(pre_wrist, dtype=np.uint8),
        "model_xml": model_xml,
        "max_abs_error": max_abs_error,
        "first_mismatch_frame": first_mismatch_frame,
        "diagnostic_state_shape_match": diagnostic_shape_match,
        "final_success": final_success,
        "gripper_controller_seed": gripper_seed,
        "alignment_mode": "start_state_initial_plus_raw_post_action_frame_shift",
    }


def episode_rows(result: dict[str, Any], episode_index: int, start_index: int) -> list[dict[str, Any]]:
    length = int(result["actions"].shape[0])
    return [
        {
            "observation.state": result["states"][offset].tolist(),
            "action": result["actions"][offset].tolist(),
            "timestamp": float(offset / CONTROL_HZ),
            "frame_index": offset,
            "episode_index": episode_index,
            "index": start_index + offset,
            "task_index": int(result["task_index"]),
        }
        for offset in range(length)
    ]


def write_repo(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite derived repo: {output_root}")
    temporary = output_root.with_name(output_root.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse partial derived repo: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "data/chunk-000").mkdir(parents=True)
    (temporary / "videos/observation.images.front/chunk-000").mkdir(parents=True)
    (temporary / "videos/observation.images.wrist/chunk-000").mkdir(parents=True)
    (temporary / "meta/episodes/chunk-000").mkdir(parents=True)
    (temporary / "provenance").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    episode_rows_meta: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    start = 0
    for episode_index, result in enumerate(results):
        episode_rows_values = episode_rows(result, episode_index, start)
        length = len(episode_rows_values)
        rows.extend(episode_rows_values)
        write_parquet(temporary / f"data/chunk-000/file-{episode_index:03d}.parquet", episode_rows_values)
        front_path = temporary / f"videos/observation.images.front/chunk-000/file-{episode_index:03d}.mp4"
        wrist_path = temporary / f"videos/observation.images.wrist/chunk-000/file-{episode_index:03d}.mp4"
        encode_video(front_path, result["front"])
        encode_video(wrist_path, result["wrist"])
        episode_rows_meta.append({
            "episode_index": episode_index,
            "data/chunk_index": 0,
            "data/file_index": episode_index,
            "dataset_from_index": start,
            "dataset_to_index": start + length - 1,
            "videos/observation.images.front/chunk_index": 0,
            "videos/observation.images.front/file_index": episode_index,
            "videos/observation.images.front/from_timestamp": 0.0,
            "videos/observation.images.front/to_timestamp": (length - 1) / CONTROL_HZ,
            "videos/observation.images.wrist/chunk_index": 0,
            "videos/observation.images.wrist/file_index": episode_index,
            "videos/observation.images.wrist/from_timestamp": 0.0,
            "videos/observation.images.wrist/to_timestamp": (length - 1) / CONTROL_HZ,
            "tasks": [result["prompt"]],
            "length": length,
        })
        model_path = temporary / f"provenance/{result['task_id']}.resolved_model.xml"
        model_path.write_bytes(result["model_xml"])
        provenance.append({
            "task_id": result["task_id"],
            "source_episode": result["raw_directory"].name,
            "raw_episode_path": str(result["raw_directory"].relative_to(ROOT)),
            "raw_metadata_sha256": sha256_file(result["raw_directory"] / "metadata.json"),
            "raw_trajectory_sha256": sha256_file(result["raw_directory"] / "trajectory.npz"),
            "start_state_sha256": sha256_file(result["raw_directory"] / "start_state.npy"),
            "bddl_path": TASKS[result["task_id"]]["bddl"],
            "bddl_sha256": sha256_file(ROOT / TASKS[result["task_id"]]["bddl"]),
            "resolved_model_xml": model_path.relative_to(temporary).as_posix(),
            "resolved_model_xml_sha256": sha256_file(model_path),
            "materializer_source": str(Path(__file__).relative_to(ROOT)),
            "max_abs_post_action_state_error": result["max_abs_error"],
            "first_post_action_state_mismatch_frame": result["first_mismatch_frame"],
            "continuous_replay_state_check": "diagnostic_only",
            "continuous_replay_state_tolerance": STATE_TOLERANCE,
            "diagnostic_state_shape_match": result["diagnostic_state_shape_match"],
            "gripper_controller_seed": result["gripper_controller_seed"],
            "pre_action_alignment": True,
            "alignment_mode": result["alignment_mode"],
            "final_success": result["final_success"],
        })
        start += length
    write_parquet(temporary / "meta/episodes/chunk-000/file-000.parquet", episode_rows_meta)
    write_parquet(
        temporary / "meta/tasks.parquet",
        [{"task_index": int(result["task_index"]), "task": result["prompt"]} for result in results],
    )
    states = np.concatenate([result["states"] for result in results], axis=0)
    actions = np.concatenate([result["actions"] for result in results], axis=0)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "panda",
        "total_episodes": len(results),
        "total_frames": int(len(rows)),
        "total_tasks": len(results),
        "chunks_size": 1000,
        "fps": CONTROL_HZ,
        "splits": {"train": f"0:{len(results)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.front": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "info": {"video.height": 256, "video.width": 256, "video.fps": CONTROL_HZ, "video.channels": 3}},
            "observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "info": {"video.height": 256, "video.width": 256, "video.fps": CONTROL_HZ, "video.channels": 3}},
            "observation.state": {"dtype": "float32", "shape": [8], "names": [f"state_{i}" for i in range(8)], "fps": CONTROL_HZ},
            "action": {"dtype": "float32", "shape": [7], "names": [f"action_{i}" for i in range(7)], "fps": CONTROL_HZ},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None, "fps": CONTROL_HZ},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": CONTROL_HZ},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": CONTROL_HZ},
            "index": {"dtype": "int64", "shape": [1], "names": None, "fps": CONTROL_HZ},
            "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": CONTROL_HZ},
        },
    }
    stats = {
        "observation.state": {"min": states.min(axis=0).tolist(), "max": states.max(axis=0).tolist(), "mean": states.mean(axis=0).tolist(), "std": states.std(axis=0).tolist(), "count": [int(len(states))]},
        "action": {"min": actions.min(axis=0).tolist(), "max": actions.max(axis=0).tolist(), "mean": actions.mean(axis=0).tolist(), "std": actions.std(axis=0).tolist(), "count": [int(len(actions))]},
    }
    write_json(temporary / "meta/info.json", info)
    write_json(temporary / "meta/stats.json", stats)
    write_json(temporary / "provenance.json", {"schema_version": "track3_reverse_pre_action_v1", "source_revision": SOURCE_REVISION, "materializer_sha256": sha256_file(Path(__file__)), "episodes": provenance, "total_frames": len(rows), "state_alignment_tolerance": STATE_TOLERANCE})
    (temporary / ".axis_pi05_revision").write_text(SOURCE_REVISION + "\n", encoding="utf-8")
    os.replace(temporary, output_root)
    return {"status": "published", "output": str(output_root), "episodes": len(results), "frames": len(rows), "provenance": str(output_root / "provenance.json")}


def self_test() -> dict[str, Any]:
    quaternion = axis_angle_from_quat_xyzw([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    state = state8_from_observation({"robot0_eef_pos": [1, 2, 3], "robot0_eef_quat": [0, 0, 0, 1], "robot0_gripper_qpos": [0.1, -0.1]})
    passed = bool(np.allclose(quaternion, [0.0, 0.0, np.pi / 2.0], atol=1e-6) and state.shape == (8,) and np.isfinite(state).all())
    return {"status": "passed" if passed else "failed", "state8_shape": list(state.shape), "axis_angle_z": float(quaternion[2]), "simulator_started": False, "gpu_touched": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--execute", action="store_true", help="run fresh MuJoCo replay on GPU3")
    parser.add_argument("--task", choices=sorted(TASKS), action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        parser.error("choose --self-test or --execute")
    tasks = args.task or sorted(TASKS)
    if set(tasks) != set(TASKS):
        raise SystemExit("Q1 publish requires all four tasks; use --task only for isolated diagnostics")
    results = [replay_one(task_id, output_root=args.output_root) for task_id in tasks]
    print(json.dumps(write_repo(results, args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
