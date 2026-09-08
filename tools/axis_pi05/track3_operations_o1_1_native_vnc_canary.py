#!/usr/bin/env python3
"""O1.1 native MuJoCo/TurboVNC no-save canary for the red-mug profile.

The canary starts one native MuJoCo environment on physical GPU3, exercises
the same v4 recorder primitives used by the future human session, and writes
only a small derived summary.  It never calls ``record_episode`` and never
creates an accepted, pending, or rejected episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
NAMESPACE = REPO / os.environ.get(
    "TRACK3_O11_NAMESPACE",
    "data/reports/axis_libero_30k/track3_operations_o1_1_native_vnc_canary_20260909",
)
PLAN = "docs/AXIS_PI05_TRACK3_OPERATIONS_O1_O2_RED_MUG_NATIVE_MUJOCO_PLAN.md"
TASK_ID = "official_t3_red_mug"
PREFERRED_START_STATE = REPO / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/pending/official_t3_red_mug_1788838632/start_state.npy"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_head(directory: Path = REPO) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def x11_probe() -> dict[str, Any]:
    command = ["xdpyinfo", "-display", ":3"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    first_lines = completed.stdout.splitlines()[:8]
    return {"command": command, "returncode": completed.returncode, "available": completed.returncode == 0, "stdout_head": first_lines, "stderr": completed.stderr[-1000:]}


def candidate_inventory(recorder: Any) -> dict[str, Any]:
    root = recorder.OUTPUT_ROOT
    rows: list[dict[str, Any]] = []
    for bucket in ("accepted", "pending", "rejected", "diagnostic"):
        directory = root / bucket
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                rows.append({"path": path.relative_to(REPO).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"root": root.relative_to(REPO).as_posix(), "files": rows, "episode_metadata_count": sum(row["path"].endswith("metadata.json") for row in rows)}


def http_probe(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        body = response.read()
        return {"status": int(response.status), "bytes": len(body), "content_type": response.headers.get("Content-Type", "")}


def run() -> dict[str, Any]:
    os.environ["DISPLAY"] = ":3"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "3"

    from tools.axis_pi05 import track3_public_teleop as recorder

    x11 = x11_probe()
    gpu_before = recorder.gpu3_guard()
    if not x11["available"]:
        raise RuntimeError(f"TurboVNC display :3 unavailable: {x11}")
    if gpu_before["status"] != "passed":
        raise RuntimeError(f"GPU3 is not idle: {gpu_before}")

    before_inventory = candidate_inventory(recorder)
    recorder.configure_public_libero()
    profile = recorder.resolve_record_profile(task_id=TASK_ID)
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(profile["bddl_path"]),
        control_freq=20,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=0,
        horizon=320,
        ignore_done=True,
    )
    server = None
    temporary_root = None
    try:
        env.reset()
        bddl_reset_source, bddl_reset_goal = recorder._eval_profile_predicates(env, profile)
        frozen_start_state = np.asarray(np.load(PREFERRED_START_STATE, allow_pickle=False), dtype=np.float64)
        start_state = frozen_start_state.copy()
        start_hash = hashlib.sha256(start_state.tobytes()).hexdigest()
        reset_obs = env.set_init_state(start_state)
        source0, goal0 = recorder._eval_profile_predicates(env, profile)
        initial_success = bool(env.check_success())
        if source0 is not True or goal0 is not False or initial_success:
            raise RuntimeError(f"invalid frozen red-mug start predicates: source={source0} goal={goal0} success={initial_success}")

        reset_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        reset_hash = hashlib.sha256(reset_state.tobytes()).hexdigest()
        reset_source, reset_goal = recorder._eval_profile_predicates(env, profile)
        reset_equal = bool(np.array_equal(reset_state, start_state))

        config = recorder.RecorderConfig(preview_port=0, max_steps=300)
        machine = recorder.ActionStateMachine(config=config)
        machine.state.paused = True
        server = recorder.PreviewServer(machine, port=0)
        server.start()
        server.publish(
            recorder._image_from_obs(reset_obs, "agentview"),
            recorder._image_from_obs(reset_obs, "robot0_eye_in_hand"),
            env=env,
            profile=profile,
            task_id=TASK_ID,
            instruction=profile["instruction"],
            step=0,
            hz=0.0,
            success=False,
            source=True,
            goal=False,
            candidate_gate="not_evaluated",
            physics_guard="ok",
        )
        preview = {
            "root": http_probe(server.url),
            "state": http_probe(server.url + "state"),
            "front": http_probe(server.url + "frame/front"),
            "wrist": http_probe(server.url + "frame/wrist"),
            "url": server.url,
        }
        machine.event("connect")
        safe_open = machine.next_action("")
        recorder.validate_action(safe_open)
        safe_action_valid = bool(float(safe_open[-1]) == -1.0 and np.allclose(safe_open[:6], 0.0))
        obs, reward, done, _info = env.step(safe_open)
        contact_matrix = recorder.raw_contact_matrix_from_sim(env)
        source, goal = recorder._eval_profile_predicates(env, profile)
        eef = None
        try:
            eef = list(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])
        except Exception:
            pass
        frames = recorder.empty_gold_buffers()
        sample = recorder.append_gold_frame(
            frames,
            action=safe_open,
            state=np.asarray(env.get_sim_state()),
            reward=float(reward),
            done=bool(done),
            success=bool(env.check_success()),
            front=recorder._image_from_obs(obs, "agentview"),
            wrist=recorder._image_from_obs(obs, "robot0_eye_in_hand"),
            timestamp=0.0,
            deadline_miss=False,
            contact_matrix=contact_matrix,
            source=source,
            goal=goal,
            eef_pos=eef,
            gripper=float(safe_open[-1]),
            check_success=bool(env.check_success()),
            contact_roles=profile.get("roles"),
        )

        machine.event("keydown", "h")
        close_action = machine.next_action("")
        recorder.validate_action(close_action)
        machine.event("keydown", "g")
        open_action = machine.next_action("")
        recorder.validate_action(open_action)
        open_close = {
            "close_action_valid": bool(float(close_action[-1]) == 1.0),
            "open_action_valid": bool(float(open_action[-1]) == -1.0),
            "close_action": close_action.tolist(),
            "open_action": open_action.tolist(),
        }

        machine.event("keydown", "z")
        reset_event = bool(machine.consume("reset"))
        reset_obs_2 = env.set_init_state(start_state)
        reset_state_2 = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        candidate_reset = bool(reset_event and np.array_equal(reset_state_2, start_state) and not env.check_success())
        machine.event("keydown", "x")
        abort_action = machine.next_action("")
        recorder.validate_action(abort_action)
        abort_fail_closed = bool(machine.state.abort and np.allclose(abort_action[:6], 0.0))
        save_seen = bool(machine.consume("save"))

        temporary_root = Path(tempfile.mkdtemp(prefix="axis_o1_1_canary_", dir="/tmp"))
        temporary_npz = temporary_root / "trajectory.npz"
        writer_result = recorder._save_npz(temporary_npz, frames)
        with np.load(temporary_npz, allow_pickle=False) as archive:
            no_object_arrays = not any(archive[name].dtype == object for name in archive.files)
            contact_offsets_equal = bool(np.array_equal(np.diff(archive["contact_offsets"]), archive["contact_n"]))
            writer_keys = sorted(archive.files)

        return {
            "schema_version": "track3_o1_1_native_vnc_no_save_canary_v1",
            "status": "passed_native_vnc_no_save_canary",
            "phase": "O1.1",
            "created_at_utc": utc_now(),
            "active_plan": PLAN,
            "root_commit": git_head(),
            "nested_openpi_commit": git_head(REPO / "third_party/axis_openpi"),
            "display": {"display": ":3", "x11": x11, "vnc_x11_available": True},
            "gpu": {"policy": "physical_gpu3_only", "before": gpu_before, "after": recorder.gpu3_guard()},
            "task": {"task_id": TASK_ID, "instruction": profile["instruction"], "fps": 20.0},
            "reset": {
                "bddl_reset_source_true": bool(bddl_reset_source),
                "bddl_reset_goal_false": not bool(bddl_reset_goal),
                "frozen_start_state_path": PREFERRED_START_STATE.relative_to(REPO).as_posix(),
                "frozen_start_state_sha256": sha256(PREFERRED_START_STATE),
                "source_true": bool(source0),
                "goal_false": not bool(goal0),
                "success_false": not initial_success,
                "start_state_sha256": start_hash,
                "first_reset_state_sha256": reset_hash,
                "first_reset_equal": reset_equal,
                "first_reset_source_true": bool(reset_source),
                "first_reset_goal_false": not bool(reset_goal),
                "second_reset_equal": candidate_reset,
            },
            "preview": preview,
            "control": {"safe_open_action_valid": safe_action_valid, "open_close": open_close, "abort_fail_closed": abort_fail_closed},
            "writer": {
                "temporary_only": True,
                "trajectory_writer_result": writer_result,
                "no_object_arrays": no_object_arrays,
                "contact_offsets_equal_contact_n": contact_offsets_equal,
                "writer_keys": writer_keys,
                "sample_contact_n": int(frames["contact_n"][-1]),
            },
            "save": {"save_event_seen": save_seen, "record_episode_called": False, "candidate_created": False, "pending_created": False},
            "candidate_inventory": {"before": before_inventory, "after": candidate_inventory(recorder), "unchanged": before_inventory == candidate_inventory(recorder)},
            "replay": "not_evaluated",
            "formal_recording": "not_started",
            "next_phase": "O1.2_requires_explicit_user_takeover_after_canary_reset",
        }
    finally:
        if server is not None:
            server.stop()
        env.close()
        if temporary_root is not None:
            for child in temporary_root.iterdir():
                child.unlink(missing_ok=True)
            temporary_root.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("O1.1 requires --run")
    if NAMESPACE.exists():
        raise SystemExit(f"refusing to reuse existing O1.1 namespace: {NAMESPACE}")
    try:
        result = run()
    except Exception as exc:
        result = {
            "schema_version": "track3_o1_1_native_vnc_no_save_canary_v1",
            "status": "blocked_or_failed_before_closeout",
            "created_at_utc": utc_now(),
            "active_plan": PLAN,
            "root_commit": git_head(),
            "error": f"{type(exc).__name__}: {exc}",
            "replay": "not_evaluated",
            "formal_recording": "not_started",
            "candidate_created": False,
        }
        NAMESPACE.mkdir(parents=True, exist_ok=True)
        write_json(NAMESPACE / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    NAMESPACE.mkdir(parents=True, exist_ok=True)
    write_json(NAMESPACE / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed_native_vnc_no_save_canary" else 2


if __name__ == "__main__":
    raise SystemExit(main())
