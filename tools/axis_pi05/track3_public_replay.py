#!/usr/bin/env python3
"""Fresh-process replay and gated promotion for Track 3 public candidates.

This tool intentionally has no command that writes ``accepted/`` without both
fresh-process 3/3 replay and an explicit human-review sidecar.  It is not used
by the R4 idle canary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P0_MAPPING = TRACK3_ROOT / "p0_task_mapping.json"
RECORDER = Path(__file__).with_name("track3_public_teleop.py")
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib
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


def load_recorder():
    spec = importlib.util.spec_from_file_location("track3_public_teleop_replay", RECORDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load recorder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_row(task_index: int) -> dict[str, Any]:
    rows = json.loads(P0_MAPPING.read_text(encoding="utf-8"))
    for row in rows:
        if int(row["task_index"]) == task_index:
            return row
    raise RuntimeError(f"task {task_index} is absent from locked public mapping")


def replay_worker(episode_dir: Path, run_index: int) -> dict[str, Any]:
    recorder = load_recorder()
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    task_index = int(metadata["task_index"])
    row = load_row(task_index)
    data = np.load(episode_dir / "trajectory.npz", allow_pickle=False)
    actions = np.asarray(data["actions"], dtype=np.float32)
    start_state = np.asarray(np.load(episode_dir / "start_state.npy"), dtype=np.float64)
    for action in actions:
        recorder.validate_action(action)
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=row["reverse_bddl_path"], control_freq=20, use_camera_obs=False, render_gpu_device_id=0)
    try:
        env.reset()
        env.set_init_state(start_state)
        start_success = bool(env.check_success())
        for action in actions:
            env.step(action)
        end_success = bool(env.check_success())
        return {
            "status": "passed" if not start_success and end_success else "failed_predicate",
            "run_index": run_index,
            "task_index": task_index,
            "frame_count": int(len(actions)),
            "start_success": start_success,
            "end_success": end_success,
            "worker_gpu_policy": "physical_gpu3_only",
        }
    except Exception as exc:
        return {"status": "failed_exception", "run_index": run_index, "task_index": task_index, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def run_replay(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    if episode_dir.parent.name != "pending":
        raise RuntimeError("replay accepts only a pending episode")
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") not in {"pending_replay", "replay_failed"}:
        raise RuntimeError(f"episode status is not replayable: {metadata.get('status')}")
    results = []
    for run_index in range(1, 4):
        config_dir = Path(tempfile.mkdtemp(prefix="axis_libero_public_replay_config_"))
        recorder = load_recorder()
        recorder.configure_public_libero()
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "MUJOCO_GL": "egl", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}"})
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(episode_dir), str(run_index)]
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if lines:
            try:
                result = json.loads(lines[-1])
            except json.JSONDecodeError:
                result = {"status": "worker_output_invalid", "stdout": completed.stdout[-4000:]}
        else:
            result = {"status": "worker_no_output", "stderr": completed.stderr[-4000:]}
        result["worker_returncode"] = completed.returncode
        results.append(result)
        if result.get("status") != "passed":
            break
    passed = sum(row.get("status") == "passed" for row in results)
    summary = {
        "schema_version": "track3_reverse_replay_v1",
        "status": "replay_3_of_3_passed_human_review_pending" if passed == 3 else "replay_failed",
        "episode_dir": str(episode_dir),
        "episode_metadata_sha256": sha256_file(episode_dir / "metadata.json"),
        "required_runs": 3,
        "passed_runs": passed,
        "runs": results,
        "created_at_utc": utc_now(),
        "accepted_promotion": False,
    }
    write_json(episode_dir / "replay_summary.json", summary)
    metadata["replay"] = "3/3_passed" if passed == 3 else f"{passed}/3_passed"
    metadata["status"] = "replay_3_of_3_passed_human_review_pending" if passed == 3 else "replay_failed"
    write_json(episode_dir / "metadata.json", metadata)
    return summary


def promote_pending(episode_dir: Path) -> dict[str, Any]:
    episode_dir = episode_dir.resolve()
    if episode_dir.parent.name != "pending":
        raise RuntimeError("only pending episodes may be promoted")
    metadata_path = episode_dir / "metadata.json"
    replay_path = episode_dir / "replay_summary.json"
    review_path = episode_dir / "human_review.json"
    if not metadata_path.is_file() or not replay_path.is_file() or not review_path.is_file():
        raise RuntimeError("metadata, replay_summary and human_review are all required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if replay.get("status") != "replay_3_of_3_passed_human_review_pending" or replay.get("passed_runs") != 3:
        raise RuntimeError("fresh-process replay is not 3/3")
    required_review = {"operator", "instruction_match", "objects_correct", "no_teleport_or_failure_trim", "front_wrist_confirmed", "approved_at_utc"}
    if review.get("status") != "approved" or not required_review.issubset(review) or not all(bool(review[key]) for key in required_review - {"operator", "approved_at_utc"}):
        raise RuntimeError("human review is not explicitly approved with all required fields")
    name = episode_dir.name
    target = OUTPUT_ROOT / "accepted" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    metadata["status"] = "accepted"
    metadata["human_review"] = "approved"
    metadata["promotion_source"] = str(episode_dir)
    write_json(metadata_path, metadata)
    os.replace(episode_dir, target)
    write_json(target / "promotion.json", {"status": "accepted", "source_pending": str(episode_dir), "replay_summary_sha256": sha256_file(target / "replay_summary.json"), "human_review_sha256": sha256_file(target / "human_review.json"), "promoted_at_utc": utc_now()})
    return {"status": "accepted", "episode_dir": str(target), "source_pending": str(episode_dir)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--promote", type=Path)
    parser.add_argument("--worker", nargs=2, metavar=("EPISODE_DIR", "RUN"))
    args = parser.parse_args()
    if args.worker:
        result = replay_worker(Path(args.worker[0]), int(args.worker[1]))
        print(json.dumps(result, ensure_ascii=False)); return 0 if result.get("status") == "passed" else 2
    if args.verify:
        result = run_replay(args.verify)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0 if result["passed_runs"] == 3 else 2
    if args.promote:
        result = promote_pending(args.promote)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    parser.error("choose --verify PENDING_EPISODE or --promote PENDING_EPISODE")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
