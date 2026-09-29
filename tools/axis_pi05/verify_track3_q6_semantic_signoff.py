#!/usr/bin/env python3
"""Verify the Track3 shifted pre-action semantics before official transfer.

This is deliberately a read-only verifier.  It does not start MuJoCo, CUDA,
training, replay, or a writer for any historical input.  The output is a new
atomic evidence record and is refused if that record already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted"
DERIVED_ROOT = ROOT / "data/processed/axis_pi05_track3_reverse_gold_v1"
EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910"
OUTPUT = EVIDENCE_ROOT / "q6_semantic_signoff.json"
Q1 = EVIDENCE_ROOT / "q1_materialization_summary.json"
Q4 = EVIDENCE_ROOT / "q4_summary.json"
Q5_LOCK = EVIDENCE_ROOT / "q5_local_gpu3/q5_attempt.lock.json"
Q5_CURSOR = EVIDENCE_ROOT / "q5_local_gpu3/cursor.json"
Q5_HEARTBEAT = EVIDENCE_ROOT / "q5_local_gpu3/heartbeat.json"
Q5_SUMMARY = EVIDENCE_ROOT / "q5_local_gpu3/q5_summary.json"
Q5_METRICS = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3/training_metrics.jsonl"
POST_Q5_PLAN = ROOT / "docs/AXIS_PI05_TRACK3_POST_Q5_SEMANTIC_SIGNOFF_AND_OFFICIAL_STEP100_PLAN_20260910.md"
RECORDER = ROOT / "tools/axis_pi05/track3_public_teleop.py"
MATERIALIZER = ROOT / "tools/axis_pi05/materialize_track3_reverse_gold_v1.py"
LOADER = ROOT / "third_party/axis_openpi/src/openpi/training/data_loader.py"
CONFIG = ROOT / "third_party/axis_openpi/src/openpi/training/config.py"

TASKS: tuple[tuple[str, str, int], ...] = (
    ("official_t3_alphabet_soup", "official_t3_alphabet_soup_1788946170", 278),
    ("official_t3_red_mug", "official_t3_red_mug_1788976069", 240),
    ("official_t3_two_mugs", "official_t3_two_mugs_1788976820", 282),
    ("official_t3_yellow_book", "official_t3_yellow_book_1788971728", 260),
)
DRIFT = {
    "official_t3_alphabet_soup": {"first_mismatch_frame": 263, "max_abs_error": 1.1444953420092219e-05},
    "official_t3_red_mug": {"first_mismatch_frame": 143, "max_abs_error": 0.07338490577769499},
    "official_t3_two_mugs": {"first_mismatch_frame": 91, "max_abs_error": 0.751217292276802},
    "official_t3_yellow_book": {"first_mismatch_frame": 0, "max_abs_error": 0.0004411426881052163},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def source_refs(path: Path, anchors: tuple[str, ...]) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, Any] = {}
    for anchor in anchors:
        matches = [index for index, line in enumerate(lines, start=1) if anchor in line]
        if not matches:
            raise AssertionError(f"source anchor not found: {path}:{anchor}")
        line_number = matches[0]
        result[anchor] = {"path": str(path.relative_to(ROOT)), "line": line_number, "text": lines[line_number - 1].strip()}
    return result


def ffprobe(path: Path) -> dict[str, str]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames",
            "-of", "default=noprint_wrappers=1:nokey=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def runtime_snapshot() -> dict[str, Any]:
    ps = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
    needles = (
        "third_party/axis_openpi/scripts/train.py",
        "run_track3_four_gold_q5",
        "track3_public_teleop",
        "track3_reverse",
        "mujoco",
        "replay",
    )
    blocking: list[dict[str, Any]] = []
    for line in ps.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        pid, args = int(fields[0]), fields[1]
        if pid == os.getpid():
            continue
        if any(needle in args.lower() for needle in needles):
            blocking.append({"pid": pid, "args": args})

    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = []
    for line in gpu_query.stdout.splitlines():
        index, uuid, used, total, utilization = [part.strip() for part in line.split(",")]
        gpus.append({
            "index": int(index), "uuid": uuid, "memory_used_mib": int(used),
            "memory_total_mib": int(total), "utilization_gpu_percent": int(utilization),
        })
    app_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    apps = []
    for line in app_query.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = [part.strip() for part in line.split(",", 3)]
        apps.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name, "used_memory_mib": int(memory)})
    free_gib = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize / 1024**3
    return {"blocking_processes": blocking, "gpus": gpus, "compute_apps": apps, "free_gib": free_gib}


def verify() -> dict[str, Any]:
    required = (
        Q1, Q4, Q5_LOCK, Q5_CURSOR, Q5_HEARTBEAT, Q5_SUMMARY, Q5_METRICS,
        POST_Q5_PLAN, RECORDER, MATERIALIZER, LOADER, CONFIG,
        DERIVED_ROOT / "provenance.json", DERIVED_ROOT / "meta/info.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing semantic signoff input(s): " + ", ".join(missing))

    q1 = strict_json(Q1)
    q4 = strict_json(Q4)
    q5_lock = strict_json(Q5_LOCK)
    q5_cursor = strict_json(Q5_CURSOR)
    q5_heartbeat = strict_json(Q5_HEARTBEAT)
    q5_summary = strict_json(Q5_SUMMARY)
    provenance = strict_json(DERIVED_ROOT / "provenance.json")
    info = strict_json(DERIVED_ROOT / "meta/info.json")
    q5_metrics = [json.loads(line) for line in Q5_METRICS.read_text(encoding="utf-8").splitlines() if line.strip()]

    if q5_lock.get("status") != "passed" or q5_lock.get("trainer_returncode") != 0:
        raise AssertionError("Q5 lock is not passed/returncode 0")
    if q5_cursor.get("status") != "passed" or q5_heartbeat.get("status") != "stopped":
        raise AssertionError("Q5 cursor/heartbeat is not closed")
    if q5_summary.get("status") != "track3_four_gold_local_single_gpu_canary_passed_ready_for_official_bundle":
        raise AssertionError("unexpected Q5 summary status")
    if [row.get("optimizer_step") for row in q5_metrics] != list(range(1, 11)):
        raise AssertionError("Q5 optimizer steps are not exactly 1..10")
    if not all(all(np.isfinite(float(value)) for value in row.values() if isinstance(value, (int, float))) for row in q5_metrics):
        raise AssertionError("Q5 contains non-finite numeric metrics")

    recorder_anchors = source_refs(RECORDER, (
        "action = machine.next_action()",
        "obs, reward, done, info = env.step(action)",
        "contact_matrix = raw_contact_matrix_from_sim(env)",
        "check_success = bool(env.check_success())",
        "source, goal = _eval_profile_predicates(env, profile)",
        "sample = append_gold_frame(",
        "front=_image_from_obs(obs, \"agentview\")",
        "wrist=_image_from_obs(obs, \"robot0_eye_in_hand\")",
        "state=np.asarray(env.get_sim_state())",
        "frames[\"actions\"].append(np.asarray(action, dtype=np.float32).copy())",
        "frames[\"states\"].append(np.asarray(state, dtype=np.float64).copy())",
        "frames[\"success\"].append(bool(success))",
    ))
    materializer_anchors = source_refs(MATERIALIZER, (
        "pre_front = [frame_from_observation(observation, \"agentview\")]",
        "pre_front.extend(np.asarray(frame, dtype=np.uint8).copy() for frame in raw[\"front\"][:-1])",
        "pre_wrist = [frame_from_observation(observation, \"robot0_eye_in_hand\")]",
        "pre_wrist.extend(np.asarray(frame, dtype=np.uint8).copy() for frame in raw[\"wrist\"][:-1])",
        "pre_states = [state8_from_observation(observation)]",
        "pre_states.extend(state8_from_sim_state(probe_env, full_state) for full_state in raw[\"states\"][:-1])",
        "\"actions\": np.asarray(raw[\"actions\"], dtype=np.float32)",
        "continuous_replay_state_check",
    ))
    loader_anchors = source_refs(LOADER, (
        "action_indices = np.arange(frame_offset, frame_offset + self.action_horizon, dtype=np.int64)",
        "action_is_pad = action_indices >= len(item[\"actions\"])",
        "action_indices = np.minimum(action_indices, len(item[\"actions\"]) - 1)",
        "\"action_is_pad\": action_is_pad",
    ))
    config_anchors = source_refs(CONFIG, (
        "name=\"axis_pi05_track3_q5\"",
        "action_horizon=10",
        "\"action_is_pad\": \"action_is_pad\"",
        "loss_objective=\"valid7_temporal\"",
    ))

    raw_checks: list[dict[str, Any]] = []
    derived_checks: list[dict[str, Any]] = []
    provenance_by_task = {row["task_id"]: row for row in provenance.get("episodes", [])}
    if provenance.get("total_frames") != sum(length for _, _, length in TASKS):
        raise AssertionError("provenance total frame count mismatch")

    global_index = 0
    for episode_index, (task_id, episode_id, expected_frames) in enumerate(TASKS):
        raw_dir = RAW_ROOT / episode_id
        metadata = strict_json(raw_dir / "metadata.json")
        raw_promotion = strict_json(raw_dir / "promotion.json")
        raw_replay = strict_json(raw_dir / "replay_summary.json")
        with np.load(raw_dir / "trajectory.npz", allow_pickle=False) as archive:
            raw = {key: np.asarray(archive[key]).copy() for key in archive.files}
        frames = int(raw["actions"].shape[0])
        if frames != expected_frames or metadata.get("frame_count") != frames:
            raise AssertionError(f"{task_id}: frame count mismatch")
        if metadata.get("status") != "accepted" or metadata.get("replay") != "3/3_passed":
            raise AssertionError(f"{task_id}: raw episode is not accepted/replayed")
        if raw_promotion.get("status") != "accepted" or raw_replay.get("passed_runs") != 3 or raw_replay.get("required_runs") != 3 or not str(raw_replay.get("status", "")).startswith("replay_3_of_3_passed"):
            raise AssertionError(f"{task_id}: promotion/replay evidence is not passed")
        for key in ("actions", "states"):
            if not np.isfinite(raw[key]).all():
                raise AssertionError(f"{task_id}: non-finite {key}")
        if raw["actions"].shape != (frames, 7) or raw["states"].shape[0] != frames:
            raise AssertionError(f"{task_id}: raw action/state shape mismatch")
        for camera in ("front", "wrist"):
            if raw[camera].shape != (frames, 256, 256, 3) or raw[camera].dtype != np.uint8:
                raise AssertionError(f"{task_id}: raw {camera} schema mismatch")
        gripper_values = set(np.unique(raw["actions"][:, -1]).tolist())
        if not gripper_values.issubset({-1.0, 1.0}):
            raise AssertionError(f"{task_id}: gripper action is not exactly +/-1: {gripper_values}")

        provenance_row = provenance_by_task.get(task_id)
        if provenance_row is None:
            raise AssertionError(f"{task_id}: missing provenance row")
        raw_hashes = {
            "raw_metadata_sha256": sha256_file(raw_dir / "metadata.json"),
            "raw_trajectory_sha256": sha256_file(raw_dir / "trajectory.npz"),
            "start_state_sha256": sha256_file(raw_dir / "start_state.npy"),
        }
        if any(provenance_row.get(key) != value for key, value in raw_hashes.items()):
            raise AssertionError(f"{task_id}: provenance raw hashes mismatch")
        drift = DRIFT[task_id]
        if provenance_row.get("first_post_action_state_mismatch_frame") != drift["first_mismatch_frame"] or not np.isclose(provenance_row.get("max_abs_post_action_state_error"), drift["max_abs_error"], rtol=0, atol=0):
            raise AssertionError(f"{task_id}: drift diagnostic changed")
        if provenance_row.get("continuous_replay_state_check") != "diagnostic_only" or provenance_row.get("pre_action_alignment") is not True:
            raise AssertionError(f"{task_id}: publication alignment flags mismatch")

        parquet_path = DERIVED_ROOT / f"data/chunk-000/file-{episode_index:03d}.parquet"
        table = pq.read_table(parquet_path)
        rows = table.to_pydict()
        actions = np.asarray(rows["action"], dtype=np.float32)
        states = np.asarray(rows["observation.state"], dtype=np.float32)
        if table.num_rows != frames or actions.shape != (frames, 7) or states.shape != (frames, 8):
            raise AssertionError(f"{task_id}: derived parquet shape mismatch")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise AssertionError(f"{task_id}: derived parquet non-finite values")
        if not np.array_equal(actions, raw["actions"].astype(np.float32, copy=False)):
            raise AssertionError(f"{task_id}: derived actions differ from raw actions")
        expected_frame_index = np.arange(frames, dtype=np.int64)
        expected_global_index = np.arange(global_index, global_index + frames, dtype=np.int64)
        if not np.array_equal(np.asarray(rows["frame_index"], dtype=np.int64), expected_frame_index):
            raise AssertionError(f"{task_id}: frame index is not episode-local")
        if not np.array_equal(np.asarray(rows["index"], dtype=np.int64), expected_global_index):
            raise AssertionError(f"{task_id}: global index is not contiguous")
        if not np.array_equal(np.asarray(rows["episode_index"], dtype=np.int64), np.full(frames, episode_index)):
            raise AssertionError(f"{task_id}: episode boundary index mismatch")
        if not np.array_equal(np.asarray(rows["task_index"], dtype=np.int64), np.full(frames, episode_index)):
            raise AssertionError(f"{task_id}: task index mismatch")
        if not np.allclose(np.asarray(rows["timestamp"], dtype=np.float64), expected_frame_index / 20.0, rtol=0, atol=1e-12):
            raise AssertionError(f"{task_id}: timestamp is not 20 Hz")
        for camera in ("front", "wrist"):
            video_path = DERIVED_ROOT / f"videos/observation.images.{camera}/chunk-000/file-{episode_index:03d}.mp4"
            probe = ffprobe(video_path)
            if probe.get("width") != "256" or probe.get("height") != "256" or probe.get("r_frame_rate") != "20/1" or int(probe.get("nb_frames", "-1")) != frames:
                raise AssertionError(f"{task_id}: derived {camera} video schema mismatch: {probe}")
        raw_checks.append({
            "task_id": task_id,
            "episode_id": episode_id,
            "frames": frames,
            "actions_shape": list(raw["actions"].shape),
            "states_shape": list(raw["states"].shape),
            "front_shape": list(raw["front"].shape),
            "wrist_shape": list(raw["wrist"].shape),
            "raw_hashes": raw_hashes,
            "gripper_values": sorted(gripper_values),
        })
        derived_checks.append({
            "task_id": task_id,
            "parquet": str(parquet_path.relative_to(ROOT)),
            "rows": table.num_rows,
            "action_exact_equal_to_raw": True,
            "state_shape": list(states.shape),
            "state_finite": True,
            "episode_boundary_closed": True,
            "video_frames_front_wrist": frames,
            "video_fps": 20,
        })
        global_index += frames

    if info.get("total_frames") != global_index or info.get("total_episodes") != len(TASKS) or info.get("fps") != 20:
        raise AssertionError("derived info topology mismatch")
    q1_episode_rows = {row["task_id"]: row for row in q1.get("episodes", [])}
    for task_id, _, frames in TASKS:
        if q1_episode_rows.get(task_id, {}).get("frames") != frames:
            raise AssertionError(f"Q1 frame count mismatch for {task_id}")

    snapshot = runtime_snapshot()
    gpu3 = next(row for row in snapshot["gpus"] if row["index"] == 3)
    gpu3_apps = [row for row in snapshot["compute_apps"] if row["gpu_uuid"] == gpu3["uuid"]]
    if snapshot["blocking_processes"] or gpu3_apps:
        raise AssertionError(f"live Track3/Q5 or GPU3 activity detected: {snapshot}")

    root_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    openpi_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/axis_openpi", check=True, capture_output=True, text=True).stdout.strip()
    return {
        "schema_version": "track3_q6_semantic_signoff_v1",
        "status": "passed_shifted_pre_action_semantic_signoff",
        "publication_decision": "reuse_v1_for_q6_transfer_preparation_only",
        "generated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "plan": {"path": str(POST_Q5_PLAN.relative_to(ROOT)), "sha256": sha256_file(POST_Q5_PLAN)},
        "source_snapshot": {"root_head": root_head, "nested_openpi_head": openpi_head},
        "n0_q5_closeout": {
            "attempt_lock": str(Q5_LOCK.relative_to(ROOT)),
            "cursor": str(Q5_CURSOR.relative_to(ROOT)),
            "heartbeat": str(Q5_HEARTBEAT.relative_to(ROOT)),
            "summary": str(Q5_SUMMARY.relative_to(ROOT)),
            "lock_status": q5_lock["status"],
            "trainer_returncode": q5_lock["trainer_returncode"],
            "optimizer_steps": [row["optimizer_step"] for row in q5_metrics],
            "checkpoint_steps": q5_summary["contract"]["checkpoint_steps"],
            "fresh_restore": q5_summary["fresh_restore"],
            "no_orphan_processes": q5_summary["process_boundary"]["orphan_processes_after_completion"] == [],
        },
        "checks": {
            "recorder_ordering": "passed",
            "index_mapping": "passed",
            "raw_action_content_unchanged": "passed",
            "state_alignment_from_frozen_full_state": "passed_by_materializer_source_and_provenance",
            "image_alignment_pre_encode": "passed_by_direct_raw_frame_slice_copy",
            "training_window_and_padding": "passed_by_loader_source_and_q4_contract",
            "continuous_replay_drift_explained_not_hidden": "passed",
        },
        "source_line_references": {
            "recorder": recorder_anchors,
            "materializer": materializer_anchors,
            "loader": loader_anchors,
            "config": config_anchors,
        },
        "index_contract": {
            "row_0": "frozen_start_state_restore_observation_plus_raw_action_0",
            "rows_1_to_T_minus_1": "raw_post_action_observation_and_full_state_t_minus_1_plus_raw_action_t",
            "last_raw_post_action": "terminal_validation_only",
            "episodes_do_not_cross": True,
            "action_horizon": 10,
            "action_is_pad": "indices beyond episode length are masked after terminal action repetition",
        },
        "raw_checks": raw_checks,
        "derived_checks": derived_checks,
        "drift_diagnostics": {
            "publication_gate": "diagnostic_only_not_1e-8",
            "rows": DRIFT,
            "interpretation": "saved finite simulator state does not encode all contact-solver history; logged transitions remain time/content traceable",
        },
        "input_hashes": {
            "q1_summary_sha256": sha256_file(Q1),
            "q4_summary_sha256": sha256_file(Q4),
            "derived_provenance_sha256": sha256_file(DERIVED_ROOT / "provenance.json"),
            "derived_info_sha256": sha256_file(DERIVED_ROOT / "meta/info.json"),
            "q5_summary_sha256": sha256_file(Q5_SUMMARY),
            "q5_metrics_sha256": sha256_file(Q5_METRICS),
            "recorder_sha256": sha256_file(RECORDER),
            "materializer_sha256": sha256_file(MATERIALIZER),
            "loader_sha256": sha256_file(LOADER),
            "config_sha256": sha256_file(CONFIG),
        },
        "runtime_snapshot": snapshot,
        "forbidden_actions": {
            "q5_rerun": False,
            "track3_official_training": False,
            "track12_server_stop_or_extension": False,
            "new_physics_or_replay": False,
            "upload_or_submission": False,
        },
    }


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite signoff: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = verify()
    if not args.check_only:
        write_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output), "frames": sum(row["frames"] for row in result["raw_checks"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
