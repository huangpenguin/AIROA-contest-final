#!/usr/bin/env python3
"""Fail-closed 20 Hz public-LIBERO reverse teleoperation recorder.

The module is deliberately CPU-safe at import time.  Simulator and image
libraries are imported only by ``--record``/``--canary`` callers.  A human
candidate is always written to ``pending/`` first; replay and human review
are separate gates before an atomic promotion can create ``accepted/``.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import html
import json
import os
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"
PUBLIC_LIBERO = ROOT / "LIBERO"
P0_LOCK = TRACK3_ROOT / "p0_lock.json"
P0_MAPPING = TRACK3_ROOT / "p0_task_mapping.json"
T3_BDDL_DIR = ROOT / "compe/t3/assets/bddl_files/libero_t3"
ALLOWED_TASKS = (19, 10, 20)
DEFAULT_TASK_ID = "official_t3_red_mug"
OFFICIAL_T3_TASKS = {
    "official_t3_red_mug": {
        "task_id": "official_t3_red_mug",
        "stem": "take_the_red_mug_off_the_left_plate_and_put_it_on_the_table",
        "instruction": "take the red mug off the left plate and put it back on the table behind the other two mugs",
        "recordable": True,
        "language_gold_ok": True,
        "official_public_example": True,
        "training_holdout": False,
        "source_eval": ["on", "red_coffee_mug_1", "plate_1"],
        "goal_eval": ["on", "red_coffee_mug_1", "living_room_table_red_coffee_mug_init_region"],
        "goal_predicates": ["(On red_coffee_mug_1 living_room_table_red_coffee_mug_init_region)"],
        "roles": {
            "subject": ("red_coffee_mug_1",),
            "source_receiver": ("plate_1",),
            "goal_receiver": ("living_room_table",),
            "demonstrator": ("robot0", "gripper0", "panda"),
        },
        "upright_subject": "red_coffee_mug_1",
        "upright_support": "plate_1",
    },
    "official_t3_two_mugs": {
        "task_id": "official_t3_two_mugs",
        "stem": "take_the_two_mugs_off_the_plates_and_place_them_on_the_table",
        "instruction": "take the white mug off the left plate and the yellow and white mug off the right plate, then place the white mug to the left of the red mug and the yellow and white mug to the right of the red mug",
        "recordable": True,
        "language_gold_ok": True,
        "official_public_example": True,
        "training_holdout": False,
        "source_eval": ["on", "porcelain_mug_1", "plate_1"],
        "source_evals": [
            ["on", "porcelain_mug_1", "plate_1"],
            ["on", "white_yellow_mug_1", "plate_2"],
        ],
        "goal_eval": ["on", "porcelain_mug_1", "living_room_table_porcelain_mug_init_region"],
        "goal_evals": [
            ["on", "porcelain_mug_1", "living_room_table_porcelain_mug_init_region"],
            ["on", "white_yellow_mug_1", "living_room_table_white_yellow_mug_init_region"],
        ],
        "goal_predicates": [
            "(On porcelain_mug_1 living_room_table_porcelain_mug_init_region)",
            "(On white_yellow_mug_1 living_room_table_white_yellow_mug_init_region)",
        ],
        "roles": {
            "subject": ("porcelain_mug_1", "white_yellow_mug_1"),
            "source_receiver": ("plate_1", "plate_2"),
            "goal_receiver": ("living_room_table",),
            "demonstrator": ("robot0", "gripper0", "panda"),
        },
        "upright_pairs": (("porcelain_mug_1", "plate_1"), ("white_yellow_mug_1", "plate_2")),
        "require_upright": True,
    },
    "official_t3_yellow_book": {
        "task_id": "official_t3_yellow_book",
        "stem": "take_the_yellow_book_off_the_shelf_and_put_it_on_the_table",
        "instruction": "take the yellow book off the top of the shelf and put it back on the table on the left",
        "recordable": True,
        "language_gold_ok": True,
        "official_public_example": True,
        "training_holdout": False,
        "source_eval": ["on", "yellow_book_2", "wooden_two_layer_shelf_1_top_side"],
        "goal_eval": ["on", "yellow_book_2", "study_table_yellow_book_left_init_region"],
        "goal_predicates": ["(On yellow_book_2 study_table_yellow_book_left_init_region)"],
        "roles": {
            "subject": ("yellow_book_2",),
            "source_receiver": ("wooden_two_layer_shelf_1",),
            "goal_receiver": ("study_table",),
            "demonstrator": ("robot0", "gripper0", "panda"),
        },
        "upright_subject": "yellow_book_2",
        "require_upright": False,
        "init_height_sites": (
            ("yellow_book_2", "wooden_two_layer_shelf_1_top_side"),
            ("yellow_book_1", "study_table_yellow_book_right_init_region"),
            ("black_book_1", "study_table_black_book_init_region"),
        ),
    },
    "official_t3_alphabet_soup": {
        "task_id": "official_t3_alphabet_soup",
        "stem": "take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table",
        "instruction": "take the alphabet soup out of the tray and place it on the table in front of the ketchup",
        "recordable": True,
        "language_gold_ok": True,
        "official_public_example": True,
        "training_holdout": False,
        "source_eval": ["in", "alphabet_soup_1", "wooden_tray_1_contain_region"],
        "goal_eval": ["on", "alphabet_soup_1", "living_room_table_alphabet_soup_init_region"],
        "goal_predicates": ["(On alphabet_soup_1 living_room_table_alphabet_soup_init_region)"],
        "roles": {
            "subject": ("alphabet_soup_1",),
            "source_receiver": ("wooden_tray_1",),
            "goal_receiver": ("living_room_table",),
            "demonstrator": ("robot0", "gripper0", "panda"),
        },
        "upright_subject": "alphabet_soup_1",
        "require_upright": True,
    },
}
MOTION_KEYS = frozenset({"w", "s", "a", "d", "r", "f", "u", "o", "i", "k", "j", "l"})
KEY_BINDINGS = {
    "w": (0, -1), "s": (0, 1), "a": (1, 1), "d": (1, -1),
    "r": (2, 1), "f": (2, -1), "u": (3, 1), "o": (3, -1),
    "i": (4, 1), "k": (4, -1), "j": (5, 1), "l": (5, -1),
}
KEY_AXIS = {
    "w": ("X", "-"), "s": ("X", "+"), "a": ("Y", "+"), "d": ("Y", "-"),
    "r": ("Z", "+"), "f": ("Z", "-"), "u": ("Rx", "+"), "o": ("Rx", "-"),
    "i": ("Ry", "+"), "k": ("Ry", "-"), "j": ("Rz", "+"), "l": ("Rz", "-"),
}
GOLD_NPZ_KEYS = (
    "actions", "states", "rewards", "dones", "success", "front", "wrist",
    "timestamps_monotonic", "deadline_miss", "contact_n", "contact_body1", "contact_body2",
    "contact_geom1", "contact_geom2", "contact_role", "required_contact", "demonstrator_contact",
    "source_receiver_contact", "goal_receiver_contact", "unrelated_contact", "source_predicate",
    "goal_predicate", "eef_pos", "gripper", "phase", "check_success", "contact_offsets",
)
CONTACT_ARRAY_KEYS = ("contact_body1", "contact_body2", "contact_geom1", "contact_geom2", "contact_role")
UNICODE_DTYPE = "U64"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_episode_relative_path(value: str) -> str:
    """Validate the only paths that may be persisted in episode metadata."""
    candidate = Path(str(value))
    if not str(value) or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"episode artifact path must be relative and confined: {value!r}")
    if any(".partial" in part for part in candidate.parts):
        raise ValueError(f"episode artifact path may not reference a partial artifact: {value!r}")
    return candidate.as_posix()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def natural_instruction(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if not cleaned or cleaned.lower().startswith("reverse"):
        raise ValueError("instruction must be natural language without a Reverse prefix")
    return cleaned


def official_t3_bddl(stem: str) -> Path:
    path = T3_BDDL_DIR / f"{stem}.bddl"
    if not path.is_file():
        raise RuntimeError(f"missing official Track3 BDDL: {path}")
    return path


def flip_front_preview(image: Any) -> np.ndarray:
    """Rotate FRONT 180° for viewing only. Saved frames stay unflipped."""
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("front preview must be HxWxC")
    return np.ascontiguousarray(np.rot90(array, 2))


GOAL_OVERLAY_COLOR = (40, 255, 90)


def goal_eval_triples(profile: Mapping[str, Any]) -> list[list[str]]:
    triples = [list(row) for row in (profile.get("goal_evals") or [])]
    if not triples and profile.get("goal_eval"):
        triples = [list(profile["goal_eval"])]
    return triples


def source_eval_triples(profile: Mapping[str, Any]) -> list[list[str]]:
    triples = [list(row) for row in (profile.get("source_evals") or [])]
    if not triples and profile.get("source_eval"):
        triples = [list(profile["source_eval"])]
    return triples


def upright_rest_pairs(profile: Mapping[str, Any]) -> list[tuple[str, str]]:
    pairs = [(str(subject), str(support)) for subject, support in (profile.get("upright_pairs") or ())]
    if not pairs and profile.get("upright_subject") and profile.get("upright_support"):
        pairs = [(str(profile["upright_subject"]), str(profile["upright_support"]))]
    return pairs


def goal_site_name_candidates(profile: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for triple in goal_eval_triples(profile):
        goal = str(triple[-1] if triple else "")
        if not goal:
            continue
        names.append(goal)
        for prefix in ("living_room_table_", "main_table_", "kitchen_table_", "study_table_"):
            if goal.startswith(prefix):
                names.append(goal[len(prefix):])
    return list(dict.fromkeys(names))


def resolve_goal_site_names(env: Any, profile: Mapping[str, Any]) -> list[str]:
    problem = getattr(env, "env", env)
    sim = getattr(problem, "sim", None)
    if sim is None:
        return []
    model_names: list[str] = []
    for index in range(int(sim.model.nsite)):
        with contextlib.suppress(Exception):
            model_names.append(str(sim.model.site_id2name(index)))
    known = set(getattr(problem, "object_sites_dict", {}) or {})
    resolved: list[str] = []
    for candidate in goal_site_name_candidates(profile):
        if candidate in model_names or candidate in known:
            resolved.append(candidate)
            continue
        for name in model_names:
            if candidate and candidate in name:
                resolved.append(name)
                break
    return list(dict.fromkeys(resolved))


def resolve_goal_site_name(env: Any, profile: Mapping[str, Any]) -> str | None:
    names = resolve_goal_site_names(env, profile)
    return names[0] if names else None


def goal_region_world_corners(env: Any, site_name: str) -> np.ndarray | None:
    sim = getattr(getattr(env, "env", env), "sim", None)
    if sim is None:
        return None
    try:
        pos = np.asarray(sim.data.get_site_xpos(site_name), dtype=np.float64)
        mat = np.asarray(sim.data.get_site_xmat(site_name), dtype=np.float64).reshape(3, 3)
        size = np.asarray(sim.model.site_size[sim.model.site_name2id(site_name)], dtype=np.float64)
    except Exception:
        return None
    local = np.array(
        [[-size[0], -size[1], size[2]], [size[0], -size[1], size[2]], [size[0], size[1], size[2]], [-size[0], size[1], size[2]]],
        dtype=np.float64,
    )
    return (mat @ local.T).T + pos


def project_world_to_pixels(sim: Any, points: np.ndarray, *, camera_name: str, height: int, width: int) -> np.ndarray | None:
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    transform = get_camera_transform_matrix(sim, camera_name, height, width)
    ones = np.ones((len(points), 1), dtype=np.float64)
    pix = transform @ np.concatenate([np.asarray(points, dtype=np.float64), ones], axis=1).T
    pix = pix.T
    if np.any(np.abs(pix[:, 2]) < 1e-8):
        return None
    return pix[:, :2] / pix[:, 2:3]


def overlay_goal_box(image: np.ndarray, corners_xy: np.ndarray) -> np.ndarray:
    """Draw a translucent GOAL rectangle. Does not mutate ``image``."""
    canvas = np.ascontiguousarray(np.asarray(image).copy())
    pts = np.round(np.asarray(corners_xy, dtype=np.float64)).astype(np.int32)
    if pts.shape != (4, 2):
        return canvas
    try:
        import cv2
        poly = pts.reshape(-1, 1, 2)
        washed = canvas.copy()
        cv2.fillPoly(washed, [poly], GOAL_OVERLAY_COLOR)
        canvas = cv2.addWeighted(washed, 0.22, canvas, 0.78, 0)
        cv2.polylines(canvas, [poly], True, GOAL_OVERLAY_COLOR, 2, cv2.LINE_AA)
        label = (int(pts[0, 0]), max(int(pts[0, 1]) - 6, 12))
        cv2.putText(canvas, "GOAL", label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, GOAL_OVERLAY_COLOR, 1, cv2.LINE_AA)
    except Exception:
        return np.ascontiguousarray(np.asarray(image).copy())
    return canvas


def make_front_preview(image: Any, env: Any = None, profile: Mapping[str, Any] | None = None) -> np.ndarray:
    """180° FRONT preview plus the official place box. Saved gold frames stay unmarked."""
    flipped = flip_front_preview(image)
    if env is None or profile is None:
        return flipped
    array = np.asarray(image)
    height, width = int(array.shape[0]), int(array.shape[1])
    sim = getattr(getattr(env, "env", env), "sim", None)
    if sim is None:
        return flipped
    for site in resolve_goal_site_names(env, profile):
        corners = goal_region_world_corners(env, site)
        if corners is None:
            continue
        xy = project_world_to_pixels(sim, corners, camera_name="agentview", height=height, width=width)
        if xy is None or len(xy) != 4:
            continue
        flipped_xy = np.stack([width - 1 - xy[:, 0], height - 1 - xy[:, 1]], axis=1)
        flipped = overlay_goal_box(flipped, flipped_xy)
    return flipped


def local_z_world(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat_wxyz)
    return np.array([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dtype=np.float64)


def _joint_qpos_slice(sim: Any, joint_name: str) -> slice:
    if hasattr(sim.model, "get_joint_qpos_addr"):
        addr = sim.model.get_joint_qpos_addr(joint_name)
        if isinstance(addr, tuple):
            return slice(int(addr[0]), int(addr[1]))
        return slice(int(addr), int(addr) + 7)
    joint_id = int(sim.model.joint_name2id(joint_name))
    start = int(sim.model.jnt_qposadr[joint_id])
    return slice(start, start + 7)


def _joint_qvel_slice(sim: Any, joint_name: str) -> slice:
    if hasattr(sim.model, "get_joint_qvel_addr"):
        addr = sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(addr, tuple):
            return slice(int(addr[0]), int(addr[1]))
        return slice(int(addr), int(addr) + 6)
    joint_id = int(sim.model.joint_name2id(joint_name))
    start = int(sim.model.jnt_dofadr[joint_id])
    return slice(start, start + 6)


def zero_sim_velocity(env: Any) -> None:
    env.sim.data.qvel[:] = 0.0
    env.sim.forward()


def object_speed(env: Any, subject: str) -> float:
    problem = _inner_env(env)
    joint = problem.get_object(subject).joints[0]
    vel = np.asarray(env.sim.data.qvel[_joint_qvel_slice(env.sim, joint)], dtype=np.float64)
    return float(np.linalg.norm(vel))


def object_half_height_z(env: Any, subject: str) -> float:
    """World-Z half extent of the subject's box/mesh geoms. Used to seat objects without a drop."""
    sim = env.sim
    problem = _inner_env(env)
    body_id = int(problem.obj_body_id[subject])
    best = 0.012
    for geom_id in range(int(sim.model.ngeom)):
        if int(sim.model.geom_bodyid[geom_id]) != body_id:
            continue
        geom_type = int(sim.model.geom_type[geom_id])
        if geom_type not in {6, 7}:
            continue
        size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float64)
        mat = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        corners = np.array(
            [[x, y, z] for x in (size[0], -size[0]) for y in (size[1], -size[1]) for z in (size[2], -size[2])],
            dtype=np.float64,
        )
        best = max(best, float(np.max(np.abs((corners @ mat.T)[:, 2]))))
    return best


def _quat_local_axis_to_world_z(axis: int) -> np.ndarray:
    """wxyz quaternion that maps a local box axis onto world +Z so a book lies flat."""
    half = float(np.sqrt(0.5))
    if int(axis) == 0:
        return np.array([half, 0.0, -half, 0.0], dtype=np.float64)
    if int(axis) == 1:
        return np.array([half, half, 0.0, 0.0], dtype=np.float64)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _subject_box_thin_axis(env: Any, subject: str) -> tuple[int, float]:
    sim = env.sim
    problem = _inner_env(env)
    body_id = int(problem.obj_body_id[subject])
    for geom_id in range(int(sim.model.ngeom)):
        if int(sim.model.geom_bodyid[geom_id]) != body_id or int(sim.model.geom_type[geom_id]) != 6:
            continue
        size = np.asarray(sim.model.geom_size[geom_id], dtype=np.float64)
        axis = int(np.argmin(size[:3]))
        return axis, float(size[axis])
    return 2, float(object_half_height_z(env, subject))


def rest_object_to_site_height(env: Any, subject: str, site_name: str, *, clearance_m: float = 0.004) -> dict[str, Any]:
    """Lay a free object flat on a BDDL site instead of dropping/tumbling it onto the support."""
    sim = env.sim
    problem = _inner_env(env)
    joint = problem.get_object(subject).joints[0]
    qpos_slice = _joint_qpos_slice(sim, joint)
    qpos = np.asarray(sim.data.qpos[qpos_slice], dtype=np.float64).copy()
    site_z = float(sim.data.get_site_xpos(site_name)[2])
    axis, thin = _subject_box_thin_axis(env, subject)
    qpos[3:7] = _quat_local_axis_to_world_z(axis)
    qpos[2] = site_z + thin + float(clearance_m)
    sim.data.qpos[qpos_slice] = qpos
    zero_sim_velocity(env)
    return {"subject": subject, "site": site_name, "z": float(qpos[2]), "thin_axis": axis, "thin_half": thin}


def scene_bodies_ok(env: Any, names: Sequence[str], *, min_z: float = 0.4) -> bool:
    problem = _inner_env(env)
    if not np.isfinite(np.asarray(env.sim.data.qpos)).all() or not np.isfinite(np.asarray(env.sim.data.qacc)).all():
        return False
    for name in names:
        body_id = int(problem.obj_body_id[str(name)])
        pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64)
        if not np.isfinite(pos).all() or float(pos[2]) < float(min_z):
            return False
    return True


def rest_object_on_support(env: Any, subject: str, support: str, *, lift_m: float = 0.022) -> dict[str, Any]:
    """Place a free object onto its support without a high drop, keeping it upright."""
    problem = _inner_env(env)
    sim = env.sim
    joint = problem.get_object(subject).joints[0]
    qpos_slice = _joint_qpos_slice(sim, joint)
    support_pos = np.asarray(sim.data.body_xpos[problem.obj_body_id[support]], dtype=np.float64)
    qpos = np.asarray(sim.data.qpos[qpos_slice], dtype=np.float64).copy()
    qpos[0:2] = support_pos[0:2]
    qpos[2] = float(support_pos[2] + lift_m)
    if float(local_z_world(qpos[3:7])[2]) < 0.95:
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sim.data.qpos[qpos_slice] = qpos
    zero_sim_velocity(env)
    up = local_z_world(qpos[3:7])
    return {"subject": subject, "support": support, "up_z": float(up[2]), "position": qpos[0:3].tolist()}


def upright_support_object(env: Any, subject: str, support: str, *, lift_m: float = 0.028) -> dict[str, Any]:
    """Force a free object upright on its support so a mug cannot start on its side."""
    problem = _inner_env(env)
    sim = env.sim
    obj = problem.get_object(subject)
    joint = obj.joints[0]
    qpos_slice = _joint_qpos_slice(sim, joint)
    support_pos = np.asarray(sim.data.body_xpos[problem.obj_body_id[support]], dtype=np.float64)
    qpos = np.asarray(sim.data.qpos[qpos_slice], dtype=np.float64).copy()
    qpos[0:2] = support_pos[0:2]
    qpos[2] = float(support_pos[2] + lift_m)
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sim.data.qpos[qpos_slice] = qpos
    if hasattr(sim.model, "get_joint_qvel_addr"):
        vel_addr = sim.model.get_joint_qvel_addr(joint)
        vel_slice = slice(int(vel_addr[0]), int(vel_addr[1])) if isinstance(vel_addr, tuple) else slice(int(vel_addr), int(vel_addr) + 6)
        sim.data.qvel[vel_slice] = 0.0
    sim.forward()
    up = local_z_world(qpos[3:7])
    return {"subject": subject, "support": support, "up_z": float(up[2]), "position": qpos[0:3].tolist()}


def contact_role_for(roles: dict[str, tuple[str, ...]], body1: str, body2: str) -> str:
    from tools.axis_pi05.track3_support_runtime_gate_remediation import _is_robot_name, _name_matches

    subjects = set(roles["subject"])
    source_receivers = set(roles["source_receiver"])
    goal_receivers = set(roles["goal_receiver"])
    left_subject = any(_name_matches(body1, name) for name in subjects)
    right_subject = any(_name_matches(body2, name) for name in subjects)
    left_source = any(_name_matches(body1, name) for name in source_receivers)
    right_source = any(_name_matches(body2, name) for name in source_receivers)
    left_goal = any(_name_matches(body1, name) for name in goal_receivers)
    right_goal = any(_name_matches(body2, name) for name in goal_receivers)
    left_robot, right_robot = _is_robot_name(body1), _is_robot_name(body2)
    if (left_subject and right_source) or (right_subject and left_source):
        return "source_receiver_contact"
    if (left_subject and right_goal) or (right_subject and left_goal):
        return "goal_receiver_contact"
    if (left_subject and right_robot) or (right_subject and left_robot):
        return "demonstrator_contact"
    if left_subject or right_subject:
        return "unrelated_background_contact"
    return "background_contact"


def contact_truth_for_roles(roles: dict[str, tuple[str, ...]], matrix: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(matrix, dict) or matrix.get("complete") is not True:
        return {"contact_telemetry_complete": False, "required_contact": False, "unrelated_contact_runtime": False, "demonstrator_contact": False, "source_receiver_contact": False, "goal_receiver_contact": False}
    raw_contacts = matrix.get("contacts")
    if not isinstance(raw_contacts, list):
        return {"contact_telemetry_complete": False, "required_contact": False, "unrelated_contact_runtime": False, "demonstrator_contact": False, "source_receiver_contact": False, "goal_receiver_contact": False}
    categories: dict[str, list[dict[str, Any]]] = {"demonstrator_contact": [], "source_receiver_contact": [], "goal_receiver_contact": [], "unrelated_background_contact": [], "background_contact": []}
    for raw in raw_contacts:
        if not isinstance(raw, dict) or not raw.get("body1_name") or not raw.get("body2_name"):
            return {"contact_telemetry_complete": False, "required_contact": False, "unrelated_contact_runtime": False, "demonstrator_contact": False, "source_receiver_contact": False, "goal_receiver_contact": False}
        role = contact_role_for(roles, str(raw["body1_name"]), str(raw["body2_name"]))
        categories[role].append(raw)
    demonstrator = bool(categories["demonstrator_contact"])
    return {
        "contact_telemetry_complete": True,
        "required_contact": demonstrator,
        "demonstrator_contact": demonstrator,
        "source_receiver_contact": bool(categories["source_receiver_contact"]),
        "goal_receiver_contact": bool(categories["goal_receiver_contact"]),
        "unrelated_contact_runtime": bool(categories["unrelated_background_contact"]),
    }


def configure_public_libero() -> Path:
    """Create an isolated public config and bind visible CUDA to physical GPU3."""
    config_dir = Path(tempfile.mkdtemp(prefix="axis_libero_public_teleop_config_"))
    root = PUBLIC_LIBERO / "libero/libero"
    atomic_write_bytes(
        config_dir / "config.yaml",
        ("\n".join([
            f"benchmark_root: {root}",
            f"bddl_files: {root / 'bddl_files'}",
            f"init_states: {root / 'init_files'}",
            f"datasets: {root / 'datasets'}",
            f"assets: {root / 'assets'}",
            "",
        ])).encode("utf-8"),
    )
    os.environ.update({
        "LIBERO_CONFIG_PATH": str(config_dir),
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "3",
        "MUJOCO_EGL_DEVICE_ID": "3",
    })
    return config_dir


def gpu3_guard(max_used_mib: int = 1024) -> dict[str, Any]:
    """Read-only GPU3 guard; it never kills or changes an external process."""
    command = ["nvidia-smi", "-i", "3", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    except Exception as exc:
        return {"status": "blocked", "reason": f"nvidia-smi unavailable: {type(exc).__name__}: {exc}", "command": command}
    if completed.returncode != 0:
        return {"status": "blocked", "reason": completed.stderr.strip()[-1000:], "command": command}
    fields = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(fields) != 3:
        return {"status": "blocked", "reason": f"unexpected nvidia-smi output: {completed.stdout!r}", "command": command}
    try:
        used = int(float(fields[1]))
        total = int(float(fields[2]))
    except ValueError as exc:
        return {"status": "blocked", "reason": f"invalid nvidia-smi output: {exc}", "command": command}
    result = {"status": "passed" if used <= max_used_mib else "blocked", "gpu": 3, "used_mib": used, "total_mib": total, "threshold_mib": max_used_mib, "command": command}
    if used > max_used_mib:
        result["reason"] = "GPU3 has an external/non-idle allocation; fail closed without terminating it"
    return result


@dataclass
class RecorderConfig:
    hz: float = 20.0
    operator_hz: float = 2.0
    translation_step: float = 0.30
    rotation_step: float = 0.36
    default_gripper: float = -1.0
    max_steps: int = 1200
    eval_step_budget: int = 300
    camera_height: int = 256
    camera_width: int = 256
    motion_pulse_ticks: int = 4
    session_timeout_seconds: float = 1800.0
    preview_host: str = "127.0.0.1"
    preview_port: int = 8765
    preview_wait_seconds: float = 45.0
    preview_refresh_ms: int = 50

    def tick_period(self) -> float:
        """Wall-clock seconds between operator ticks; saved data stays at ``hz``."""
        return 1.0 / max(float(self.operator_hz), 1e-6)

    def tick_ms(self) -> int:
        return int(round(self.tick_period() * 1000))


@dataclass
class KeyboardState:
    gripper: float = -1.0
    paused: bool = False
    camera: str = "front"
    abort: bool = False
    save: bool = False
    reset: bool = False
    preview_connected: bool = False
    first_frame_delivered: bool = False
    held_keys: set[str] = field(default_factory=set)
    last_motion_key: str | None = None
    reset_count: int = 0
    pulse_key: str | None = None
    pulse_remaining: int = 0


def _safe_gripper(value: float) -> float:
    return 1.0 if float(value) >= 0 else -1.0


def key_to_action(key: str, state: KeyboardState, config: RecorderConfig) -> np.ndarray:
    """Translate one motion/no-key tick, always retaining a valid gripper."""
    action = np.zeros(7, dtype=np.float32)
    binding = KEY_BINDINGS.get(str(key).lower())
    if binding is not None:
        index, direction = binding
        action[index] = direction * (config.translation_step if index < 3 else config.rotation_step)
    action[-1] = _safe_gripper(state.gripper)
    return action


def validate_action(action: np.ndarray) -> None:
    array = np.asarray(action)
    if array.shape != (7,):
        raise ValueError(f"OSC_POSE action must have shape (7,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("OSC_POSE action contains non-finite values")
    if float(array[-1]) not in {-1.0, 1.0}:
        raise ValueError("gripper must be exactly -1 or +1")


def empty_gold_buffers() -> dict[str, list[Any]]:
    return {key: [] for key in GOLD_NPZ_KEYS if key != "contact_offsets"}


def infer_phase(source: bool, goal: bool, required: bool, gripper: float) -> str:
    if source and required:
        return "grasp_dwell"
    if source and not required:
        return "precontact_xy"
    if not source and not goal and float(gripper) > 0:
        return "lift"
    if goal and float(gripper) > 0:
        return "place"
    if goal and float(gripper) < 0:
        return "release"
    return "transport"


def _flatten_contact_rows(rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    flattened: list[str] = []
    offsets = [0]
    for row in rows:
        values = np.asarray(row, dtype=UNICODE_DTYPE).reshape(-1).tolist()
        flattened.extend(str(value) for value in values)
        offsets.append(len(flattened))
    return np.asarray(flattened, dtype=UNICODE_DTYPE), np.asarray(offsets, dtype=np.int64)


def typed_npz_arrays(frames: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, values in frames.items():
        if key in CONTACT_ARRAY_KEYS:
            arrays[key], offsets = _flatten_contact_rows(values)
            if key == "contact_body1":
                arrays["contact_offsets"] = offsets
        elif key == "phase":
            arrays[key] = np.asarray(values, dtype=UNICODE_DTYPE)
        elif key in {"required_contact", "demonstrator_contact", "source_receiver_contact", "goal_receiver_contact", "unrelated_contact", "source_predicate", "goal_predicate", "success", "dones", "deadline_miss", "check_success"}:
            arrays[key] = np.asarray(values, dtype=np.bool_)
        elif key == "contact_n":
            arrays[key] = np.asarray(values, dtype=np.int32)
        elif values:
            arrays[key] = np.asarray(values)
    if "contact_offsets" not in arrays:
        _, arrays["contact_offsets"] = _flatten_contact_rows(frames.get("contact_body1", []))
    return arrays


def acquisition_contact_gate_v4(samples: list[dict[str, Any]], stable_ticks: int = 4) -> dict[str, Any]:
    """Find the first stable gripper↔subject acquisition window.

    This gate is intentionally forward-only.  A final-frame contact or a
    subject↔goal-receiver contact cannot manufacture an acquisition barrier.
    """
    if stable_ticks < 1:
        raise ValueError("stable_ticks must be positive")
    for start in range(0, max(0, len(samples) - stable_ticks + 1)):
        window = samples[start:start + stable_ticks]
        if not all(
            str(row.get("phase") or "") in {"close", "grasp_dwell"}
            and bool(row.get("source_predicate"))
            and bool(row.get("demonstrator_contact"))
            and row.get("contact_telemetry_complete") is not False
            and row.get("contact_raw_complete") is not False
            for row in window
        ):
            continue
        if any(not bool(row.get("source_predicate")) for row in samples[:start]):
            continue
        return {
            "state": "passed",
            "schema": "acquisition_contact_gate_v4",
            "barrier_index": start + stable_ticks - 1,
            "window_start": start,
            "window_end": start + stable_ticks - 1,
            "stable_ticks": stable_ticks,
            "contact_kind": "demonstrator_subject",
            "source_before_release": True,
        }
    return {
        "state": "failed",
        "schema": "acquisition_contact_gate_v4",
        "reason": "no_stable_gripper_subject_contact_during_close_or_dwell_before_source_release",
        "stable_ticks": stable_ticks,
        "contact_kind": "demonstrator_subject_only",
        "barrier_index": None,
    }


def evaluate_gold_gates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr

    if not samples:
        return {"status": "failed", "reason": "empty_buffer", "candidate_gate": "failed", "save_allowed": False}
    phases = [str(row.get("phase") or infer_phase(bool(row.get("source_predicate")), bool(row.get("goal_predicate")), bool(row.get("required_contact")), float(row.get("gripper", -1)))) for row in samples]
    source_values = [bool(row.get("source_predicate")) for row in samples]
    goal_values = [bool(row.get("goal_predicate")) for row in samples]
    initial_source = source_values[0] is True
    initial_goal_false = goal_values[0] is False
    required = acquisition_contact_gate_v4(samples, stable_ticks=4)
    released = sr.source_released_gate(source_values, phases, stable_ticks=4)
    check_success = bool(samples[-1].get("check_success") or samples[-1].get("success"))
    goal = sr.goal_continuous_gate(goal_values, phases, check_success=check_success, stable_ticks=4)
    unrelated = any(bool(row.get("unrelated_contact")) for row in samples)
    unknown = any(row.get("contact_telemetry_complete") is False or row.get("contact_raw_complete") is False for row in samples)
    finite = all(np.isfinite(np.asarray(row.get("action", [0] * 7))).all() for row in samples)
    passed = all((
        initial_source,
        initial_goal_false,
        required.get("state") == "passed",
        released.get("state") == "passed",
        goal.get("state") == "passed",
        not unrelated,
        not unknown,
        finite,
    ))
    return {
        "status": "passed" if passed else "failed",
        "candidate_gate": "passed" if passed else "failed",
        "save_allowed": passed,
        "initial_source_true": initial_source,
        "initial_goal_false": initial_goal_false,
        "required_contact": required,
        "source_released": released,
        "goal_continuous_true": goal,
        "unrelated_any": unrelated,
        "unknown_any": unknown,
        "finite_actions": finite,
        "final_success_boolean_insufficient": True,
    }


def raw_contact_matrix_from_sim(env: Any) -> dict[str, Any]:
    model = env.sim.model
    data = env.sim.data
    contacts: list[dict[str, Any]] = []
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        contacts.append({
            "body1_name": str(model.body_id2name(body1) or f"body_{body1}"),
            "body2_name": str(model.body_id2name(body2) or f"body_{body2}"),
            "geom1_name": str(model.geom_id2name(geom1) or f"geom_{geom1}"),
            "geom2_name": str(model.geom_id2name(geom2) or f"geom_{geom2}"),
            "distance_m": float(contact.dist),
        })
    return {"complete": True, "contacts": contacts}


def append_gold_frame(frames: dict[str, list[Any]], *, action: np.ndarray, state: np.ndarray, reward: float, done: bool, success: bool, front: Any, wrist: Any, timestamp: float, deadline_miss: bool, contact_matrix: dict[str, Any], source: bool, goal: bool, eef_pos: Sequence[float] | None, gripper: float, check_success: bool, contact_roles: dict[str, tuple[str, ...]] | None = None) -> dict[str, Any]:
    from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr

    truth = contact_truth_for_roles(contact_roles, contact_matrix) if contact_roles is not None else sr.contact_truth_v3(10, contact_matrix)
    contacts = list(contact_matrix.get("contacts") or [])
    roles = []
    for row in contacts:
        if contact_roles is not None:
            roles.append(contact_role_for(contact_roles, str(row.get("body1_name", "")), str(row.get("body2_name", ""))))
        else:
            roles.append(sr.contact_role(str(row.get("body1_name", "")), str(row.get("body2_name", ""))))
    phase = infer_phase(source, goal, bool(truth.get("required_contact")), gripper)
    frames["actions"].append(np.asarray(action, dtype=np.float32).copy())
    frames["states"].append(np.asarray(state, dtype=np.float64).copy())
    frames["rewards"].append(float(reward))
    frames["dones"].append(bool(done))
    frames["success"].append(bool(success))
    frames["front"].append(front)
    frames["wrist"].append(wrist)
    frames["timestamps_monotonic"].append(float(timestamp))
    frames["deadline_miss"].append(bool(deadline_miss))
    frames["contact_n"].append(len(contacts))
    frames["contact_body1"].append(np.asarray([str(row.get("body1_name", "")) for row in contacts], dtype=UNICODE_DTYPE))
    frames["contact_body2"].append(np.asarray([str(row.get("body2_name", "")) for row in contacts], dtype=UNICODE_DTYPE))
    frames["contact_geom1"].append(np.asarray([str(row.get("geom1_name", "")) for row in contacts], dtype=UNICODE_DTYPE))
    frames["contact_geom2"].append(np.asarray([str(row.get("geom2_name", "")) for row in contacts], dtype=UNICODE_DTYPE))
    frames["contact_role"].append(np.asarray(roles, dtype=UNICODE_DTYPE))
    frames["required_contact"].append(bool(truth.get("required_contact")))
    frames["demonstrator_contact"].append(bool(truth.get("demonstrator_contact")))
    frames["source_receiver_contact"].append(bool(truth.get("source_receiver_contact")))
    frames["goal_receiver_contact"].append(bool(truth.get("goal_receiver_contact")))
    frames["unrelated_contact"].append(bool(truth.get("unrelated_contact_runtime")))
    frames["source_predicate"].append(bool(source))
    frames["goal_predicate"].append(bool(goal))
    frames["eef_pos"].append(np.asarray(eef_pos if eef_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64))
    frames["gripper"].append(float(gripper))
    frames["phase"].append(phase)
    frames["check_success"].append(bool(check_success))
    sample = {
        "contact_matrix": contact_matrix,
        "contact_truth_v3": truth,
        "source_predicate": source,
        "goal_predicate": goal,
        "required_contact": bool(truth.get("required_contact")),
        "demonstrator_contact": bool(truth.get("demonstrator_contact")),
        "unrelated_contact": bool(truth.get("unrelated_contact_runtime")),
        "contact_telemetry_complete": bool(truth.get("contact_telemetry_complete")),
        "contact_raw_complete": True,
        "gripper": gripper,
        "phase": phase,
        "action": np.asarray(action, dtype=np.float32),
        "check_success": check_success,
        "success": success,
    }
    return sample


class ActionStateMachine:
    """Single input path shared by terminal keys and browser events."""

    def __init__(self, state: KeyboardState | None = None, config: RecorderConfig | None = None):
        self.state = state or KeyboardState()
        self.config = config or RecorderConfig()
        self._lock = threading.RLock()

    def event(self, event: str, key: str = "") -> dict[str, Any]:
        normalized = str(key).lower()
        with self._lock:
            if event in {"connect", "first_frame"}:
                self.state.preview_connected = True
                if event == "first_frame":
                    self.state.first_frame_delivered = True
            elif event == "disconnect":
                self.state.preview_connected = False
                self.state.held_keys.clear()
                self.state.paused = True
                self.state.pulse_remaining = 0
                self.state.pulse_key = None
            elif event == "release_motion":
                self.state.held_keys.clear()
            elif event == "keydown":
                if normalized in MOTION_KEYS:
                    self.state.held_keys.add(normalized)
                    self.state.last_motion_key = normalized
                    self.state.pulse_key = normalized
                    self.state.pulse_remaining = max(int(self.config.motion_pulse_ticks), 1)
                elif normalized in {"g", "[", "open"}:
                    self.state.gripper = -1.0
                elif normalized in {"h", "]", "close"}:
                    self.state.gripper = 1.0
                elif normalized == "p":
                    self.state.paused = not self.state.paused
                elif normalized == "c":
                    self.state.camera = "wrist" if self.state.camera == "front" else "front"
                elif normalized in {"z", "reset"}:
                    self.state.reset = True
                    self.state.held_keys.clear()
                elif normalized in {"enter", "save"}:
                    self.state.save = True
                elif normalized in {"x", "q", "abort"}:
                    self.state.abort = True
            elif event == "keyup" and normalized in MOTION_KEYS:
                self.state.held_keys.discard(normalized)
            elif event == "button":
                return self.event("keydown", normalized)
            else:
                raise ValueError(f"unknown control event: {event}")
            return self.snapshot()

    def next_action(self, key: str | None = None) -> np.ndarray:
        with self._lock:
            if key is not None:
                normalized = str(key).lower()
                if normalized in MOTION_KEYS:
                    return key_to_action(normalized, self.state, self.config) if not self.state.paused else key_to_action("", self.state, self.config)
                return key_to_action("", self.state, self.config)
            if self.state.paused or self.state.abort or not self.state.preview_connected:
                return key_to_action("", self.state, self.config)
            motion = ""
            if self.state.last_motion_key in self.state.held_keys:
                motion = self.state.last_motion_key
            elif self.state.pulse_remaining > 0 and self.state.pulse_key:
                motion = self.state.pulse_key
                self.state.pulse_remaining -= 1
            return key_to_action(motion, self.state, self.config)

    def consume(self, field_name: str) -> bool:
        with self._lock:
            value = bool(getattr(self.state, field_name))
            if value:
                setattr(self.state, field_name, False)
            return value

    def snapshot(self) -> dict[str, Any]:
        motion = self.state.last_motion_key if self.state.last_motion_key in self.state.held_keys else (self.state.pulse_key if self.state.pulse_remaining > 0 else "")
        axis, sign = KEY_AXIS.get(motion, ("none", "0"))
        gripper = _safe_gripper(self.state.gripper)
        return {
            "gripper": gripper,
            "gripper_label": "OPEN" if gripper < 0 else "CLOSED",
            "paused": self.state.paused,
            "camera": self.state.camera,
            "abort": self.state.abort,
            "save": self.state.save,
            "reset": self.state.reset,
            "preview_connected": self.state.preview_connected,
            "first_frame_delivered": self.state.first_frame_delivered,
            "held_keys": sorted(self.state.held_keys),
            "reset_count": self.state.reset_count,
            "axis": axis,
            "sign": sign,
        }


def _encode_jpeg(image: Any) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError("preview image must be HxWxC")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        import cv2
        ok, encoded = cv2.imencode(".jpg", array[..., ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RuntimeError("cv2.imencode returned false")
        return bytes(encoded)
    except ImportError:
        from PIL import Image
        import io
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()


PREVIEW_HTML = """<!doctype html>
<meta charset="utf-8"><title>AXIS Track3 official teleop</title>
<style>body{font:14px sans-serif;background:#171717;color:#eee;margin:12px}main{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}img{width:640px;max-width:48vw;image-rendering:auto;background:#333}button{margin:3px;padding:8px 12px}#state{white-space:pre;font-family:monospace;background:#222;padding:8px}#hud,#instruction{font:16px monospace;margin:8px 0}.open{background:#14532d;color:#fff}.close{background:#7f1d1d;color:#fff}.go{background:#1e3a5f;color:#fff}</style>
<h2>AXIS Track3 official teleop — localhost only</h2>
<p>FRONT 预览旋转 180° 方便看；W/S 已对调。绿色半透明框是官方放置目标区。框只画在预览上，保存的帧不带框。语言用官方原文，不要 Reverse 前缀。</p>
<p>先看 <b>paused=false</b> 再动。短按方向也会走一步。切走浏览器标签会暂停；点按钮不会。瞄准时请暂停：未暂停即使不按键也会记步。当前墙钟约 2 Hz（每步 500ms），预览按新帧刷新；存盘仍是 20 Hz。官方评测约 300 步，录制上限 1200。</p>
<div id="instruction">waiting instruction</div>
<div id="hud">waiting</div>
<div id="state">waiting for simulator</div>
<main><figure><figcaption>FRONT 256px preview 180° + GOAL box</figcaption><img id="front" alt="front"></figure><figure><figcaption>WRIST 256px</figcaption><img id="wrist" alt="wrist"></figure></main>
<p>
<button class="go" onclick="button('w')">W -X</button>
<button class="go" onclick="button('s')">S +X</button>
<button class="go" onclick="button('a')">A +Y</button>
<button class="go" onclick="button('d')">D -Y</button>
<button class="go" onclick="button('r')">R +Z</button>
<button class="go" onclick="button('f')">F -Z</button>
</p>
<p>
<button class="open" onclick="button('open')">Open (-1)</button>
<button class="close" onclick="button('close')">Close (+1)</button>
<button onclick="button('p')">Pause</button>
<button onclick="button('z')">Reset</button>
<button onclick="button('enter')">Save Candidate</button>
<button onclick="button('x')">Abort</button>
</p>
<p>Keyboard: W/S ±X, A/D ±Y, R/F ±Z; U/O ±Rx, I/K ±Ry, J/L ±Rz; G Open, H Close; P pause; Z reset; Enter save; X abort.</p>
<script>
const CODES={KeyW:'w',KeyS:'s',KeyA:'a',KeyD:'d',KeyR:'r',KeyF:'f',KeyU:'u',KeyO:'o',KeyI:'i',KeyK:'k',KeyJ:'j',KeyL:'l',KeyG:'g',KeyH:'h',KeyP:'p',KeyZ:'z',KeyX:'x',KeyQ:'q',Enter:'enter'};
const post=(event,key='')=>fetch('/event',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({event,key})});
const button=(key)=>post('button',key);
const mapKey=e=>e.key==='Enter'?'enter':(CODES[e.code]||e.key.toLowerCase());
window.addEventListener('keydown',e=>{if(!e.repeat){post('keydown',mapKey(e));e.preventDefault();}});
window.addEventListener('keyup',e=>{post('keyup',mapKey(e));});
document.addEventListener('visibilitychange',()=>{if(document.hidden)post('disconnect'); else post('connect');});
window.addEventListener('blur',()=>post('release_motion'));
window.addEventListener('focus',()=>post('connect'));
let frameSeq=-1,busy=false;
async function refresh(){if(busy)return;busy=true;try{const s=await fetch('/state').then(r=>r.json());document.getElementById('instruction').textContent=s.instruction||'';document.getElementById('hud').textContent=`paused=${s.paused} gripper=${s.gripper_label} axis=${s.axis}${s.sign} connected=${s.preview_connected} step=${s.step}/${s.max_steps||'?'} eval<=${s.eval_step_budget||300} tick=${s.tick_ms||500}ms gate=${s.candidate_gate} phys=${s.physics_guard||'ok'}`;document.getElementById('state').textContent=JSON.stringify(s,null,2);const seq=Number(s.frame_seq);if(Number.isFinite(seq)&&seq!==frameSeq){frameSeq=seq;for(const k of ['front','wrist'])document.getElementById(k).src='/frame/'+k+'?t='+seq;}}catch(e){}finally{busy=false;}}
setInterval(refresh,50); refresh();
</script>"""


class _PreviewHandler(BaseHTTPRequestHandler):
    server: "PreviewServer"

    def log_message(self, *_args: Any) -> None:
        return

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/":
            self.server.machine.event("connect")
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", PREVIEW_HTML.encode("utf-8"))
            return
        if self.path.startswith("/frame/"):
            camera = self.path.split("/", 2)[-1].split("?", 1)[0]
            body = self.server.frames.get(camera)
            if body is None:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain", b"first frame not available")
            else:
                self.server.machine.event("first_frame")
                self._send(HTTPStatus.OK, "image/jpeg", body)
            return
        if self.path.split("?", 1)[0] == "/state":
            self._send(HTTPStatus.OK, "application/json", json.dumps(self.server.payload(), sort_keys=True).encode("utf-8"))
            return
        if self.path.split("?", 1)[0] == "/health":
            self._send(HTTPStatus.OK, "application/json", json.dumps({"status": "ok", **self.server.payload()}, sort_keys=True).encode("utf-8"))
            return
        self._send(HTTPStatus.NOT_FOUND, "text/plain", b"not found")

    def do_POST(self) -> None:
        if self.path != "/event":
            self._send(HTTPStatus.NOT_FOUND, "text/plain", b"not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(size) or b"{}")
            state = self.server.machine.event(str(body.get("event", "")), str(body.get("key", "")))
            self._send(HTTPStatus.OK, "application/json", json.dumps(state, sort_keys=True).encode("utf-8"))
        except Exception as exc:
            self._send(HTTPStatus.BAD_REQUEST, "application/json", json.dumps({"error": str(exc)}).encode("utf-8"))


class PreviewServer:
    def __init__(self, machine: ActionStateMachine, host: str = "127.0.0.1", port: int = 8765):
        if host != "127.0.0.1":
            raise ValueError("preview must bind to 127.0.0.1")
        self.machine = machine
        self.frames: dict[str, bytes | None] = {"front": None, "wrist": None}
        self.frame_seq = 0
        self.status: dict[str, Any] = {
            "task_index": None, "instruction": "", "goal_predicate": "", "step": 0, "hz": 0.0, "lag_ms": 0.0,
            "success": False, "source": None, "goal": None, "required_contact": False, "demonstrator_contact": False,
            "source_receiver_contact": False, "goal_receiver_contact": False, "unrelated_contact": False,
            "candidate_gate": "not_evaluated", "cameras": ["FRONT", "WRIST"],
            "max_steps": 1200, "eval_step_budget": 300, "operator_hz": 2.0, "tick_ms": 500, "frame_seq": 0,
            "physics_guard": "ok",
        }
        self.httpd = ThreadingHTTPServer((host, port), _PreviewHandler)
        # The request handler is constructed by ThreadingHTTPServer and uses
        # these explicit aliases; no socket is ever exposed beyond loopback.
        self.httpd.machine = machine
        self.httpd.frames = self.frames
        self.httpd.payload = self.payload
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="track3-preview", daemon=True)
        self._started = False

    @property
    def port(self) -> int:
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> None:
        self._started = True
        self.thread.start()

    def stop(self) -> None:
        if self._started:
            self.httpd.shutdown()
        self.httpd.server_close()
        if self._started:
            self.thread.join(timeout=5)

    def publish(self, front: Any, wrist: Any, env: Any = None, profile: Mapping[str, Any] | None = None, **status: Any) -> None:
        self.frames["front"] = _encode_jpeg(make_front_preview(front, env, profile)) if front is not None else None
        self.frames["wrist"] = _encode_jpeg(wrist) if wrist is not None else None
        self.frame_seq += 1
        self.status.update(status)
        self.status["frame_seq"] = self.frame_seq

    def payload(self) -> dict[str, Any]:
        return {**self.status, **self.machine.snapshot(), "preview_url": self.url}

    def wait_for_first_frame(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.machine.snapshot()
            if state["preview_connected"] and state["first_frame_delivered"]:
                return True
            time.sleep(0.05)
        return False


def _raw_terminal():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return fd, old


@contextlib.contextmanager
def raw_terminal():
    """Enable cbreak only on a real TTY; browser-only launches skip ioctl."""
    if not sys.stdin.isatty():
        yield
        return
    fd, old = _raw_terminal()
    try:
        yield
    finally:
        import termios
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key(timeout: float) -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if ready else None


def _image_from_obs(obs: dict[str, Any], suffix: str) -> np.ndarray | None:
    for key in (f"{suffix}_image", suffix):
        if key in obs and np.asarray(obs[key]).ndim == 3:
            return np.asarray(obs[key])
    return None


def _save_npz(path: Path, frames: dict[str, list[Any]]) -> dict[str, Any]:
    arrays = typed_npz_arrays(frames)
    contact_n = np.asarray(arrays.get("contact_n", []), dtype=np.int32)
    contact_offsets = np.asarray(arrays.get("contact_offsets", []), dtype=np.int64)
    if contact_offsets.shape != (contact_n.shape[0] + 1,):
        raise ValueError("contact_offsets must have one more entry than contact_n")
    if not np.array_equal(np.diff(contact_offsets), contact_n):
        raise ValueError("saved_contact_n must equal each contact_offsets span")
    total_contacts = int(contact_offsets[-1]) if contact_offsets.size else 0
    for key in CONTACT_ARRAY_KEYS:
        if len(arrays.get(key, [])) != total_contacts:
            raise ValueError(f"{key} length does not match dynamic contact table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        if any(archive[name].dtype == object for name in keys):
            raise ValueError("gold trajectory contains object arrays")
        saved_n = np.asarray(archive["contact_n"], dtype=np.int32)
        saved_offsets = np.asarray(archive["contact_offsets"], dtype=np.int64)
        if not np.array_equal(np.diff(saved_offsets), saved_n):
            raise ValueError("archive contact_n/contact_offsets mismatch")
    return {"path": validate_episode_relative_path(path.name), "sha256": sha256_file(path), "keys": keys, "frames": len(frames.get("actions", []))}


def _save_npy(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_video(path: Path, frames: list[Any], fps: float) -> dict[str, Any]:
    relative_path = validate_episode_relative_path(path.name)
    if not frames:
        return {"status": "no_frames", "path": relative_path}
    import cv2
    first = np.asarray(frames[0])
    height, width = first.shape[:2]
    for codec in ("avc1", "H264", "mp4v"):
        temporary = path.with_name(path.stem + ".partial" + path.suffix)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            temporary.unlink(missing_ok=True)
            continue
        for frame in frames:
            array = np.asarray(frame)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            writer.write(array[..., ::-1])
        writer.release()
        if temporary.is_file() and temporary.stat().st_size:
            os.replace(temporary, path)
            return {"status": "written", "codec": codec, "path": relative_path, "sha256": sha256_file(path)}
    return {"status": "codec_unavailable", "path": relative_path}


def _load_lock_and_row(task_index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if task_index not in ALLOWED_TASKS:
        raise RuntimeError(f"task {task_index} is not in the public R4 allowlist {ALLOWED_TASKS}")
    if not P0_LOCK.is_file() or not P0_MAPPING.is_file() or not P1_SUMMARY.is_file():
        raise RuntimeError("P0 lock, mapping and P1 summary are required")
    lock_sha = sha256_file(P0_LOCK)
    lock = json.loads(P0_LOCK.read_text(encoding="utf-8"))
    mapping = json.loads(P0_MAPPING.read_text(encoding="utf-8"))
    row = next((item for item in mapping if int(item["task_index"]) == task_index), None)
    if row is None or not Path(row["reverse_bddl_path"]).is_file():
        raise RuntimeError(f"missing locked reverse BDDL for task {task_index}")
    summary = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    if task_index not in {int(value) for value in summary.get("accepted_task_indices", [])}:
        raise RuntimeError(f"task {task_index} has no P1 full-state snapshot")
    snapshot = P1_ROOT / "snapshots" / f"task_{task_index:02d}" / "attempt_1" / "terminal_state.npy"
    if not snapshot.is_file():
        raise RuntimeError(f"missing P1 snapshot for task {task_index}")
    p1_row = next(item for item in summary["attempt_results"] if int(item.get("task_index", -1)) == task_index and item.get("attempt") == 1)
    if sha256_file(snapshot) != p1_row.get("terminal_state_sha256"):
        raise RuntimeError(f"P1 snapshot hash mismatch for task {task_index}")
    return lock, row, {"p0_lock_sha256": lock_sha, "p1_snapshot": str(snapshot), "p1_snapshot_sha256": sha256_file(snapshot), "p1_row": p1_row}


def _metadata(profile: dict[str, Any], operator: str, config: RecorderConfig, lock_info: dict[str, Any]) -> dict[str, Any]:
    instruction = natural_instruction(profile["instruction"])
    return {
        "schema_version": "track3_reverse_pilot_v4",
        "task_index": profile.get("task_index"),
        "task_id": profile["task_id"],
        "operator": operator,
        "input_device": "keyboard_or_localhost_browser",
        "fps": config.hz,
        "action_contract": "7D OSC_POSE; translation/rotation delta; gripper exactly -1 or +1",
        "state_contract": "complete simulator state at start/end and every frame",
        "source": profile.get("start_source", "official_t3_bddl_reset"),
        "instruction": instruction,
        "reverse_instruction": instruction,
        "reverse_goal_predicates": list(profile.get("goal_predicates") or []),
        "language_gold_ok": bool(profile.get("language_gold_ok")),
        "official_public_example": bool(profile.get("official_public_example")),
        "training_holdout": bool(profile.get("training_holdout")),
        "lock": lock_info,
        "replay": "pending_fresh_process_3_of_3",
        "human_review": "pending",
        "status": "recording",
        "created_at_utc": utc_now(),
    }


def resolve_record_profile(*, task_id: str | None = None, task_index: int | None = None) -> dict[str, Any]:
    if task_index is not None:
        _lock, row, lock_info = _load_lock_and_row(int(task_index))
        instruction = str(row.get("reverse_instruction") or "")
        language_ok = False
        try:
            instruction = natural_instruction(instruction)
            language_ok = True
        except ValueError:
            language_ok = False
        return {
            "task_id": f"public_task{int(task_index):02d}",
            "task_index": int(task_index),
            "instruction": instruction or str(row.get("reverse_instruction") or ""),
            "language_gold_ok": False if int(task_index) == 10 else language_ok,
            "official_public_example": False,
            "training_holdout": False,
            "bddl_path": Path(row["reverse_bddl_path"]),
            "start_source": "official_public_hdf5_full_state_p1_snapshot",
            "start_state_path": lock_info["p1_snapshot"],
            "source_eval": ["on", "akita_black_bowl_1", "plate_1"] if int(task_index) == 10 else None,
            "goal_eval": ["on", "akita_black_bowl_1", "main_table"] if int(task_index) == 10 else None,
            "goal_predicates": list(row.get("reverse_goal_predicates") or []),
            "roles": None,
            "lock_info": {**lock_info, "p0_lock": True},
            "language_unqualified": int(task_index) == 10,
        }
    chosen = str(task_id or DEFAULT_TASK_ID)
    spec = OFFICIAL_T3_TASKS.get(chosen)
    if spec is None:
        raise RuntimeError(f"unknown task id {chosen}; recordable official example is {DEFAULT_TASK_ID}")
    if not spec.get("recordable"):
        raise RuntimeError(f"{chosen} is a reserved official holdout and cannot be recorded")
    instruction = natural_instruction(spec["instruction"])
    return {
        **spec,
        "instruction": instruction,
        "bddl_path": official_t3_bddl(str(spec["stem"])),
        "start_source": "official_t3_bddl_reset_frozen",
        "start_state_path": None,
        "lock_info": {
            "official_t3_bddl": str(official_t3_bddl(str(spec["stem"]))),
            "official_t3_bddl_sha256": sha256_file(official_t3_bddl(str(spec["stem"]))),
            "training_holdout_examples": [key for key, row in OFFICIAL_T3_TASKS.items() if row.get("training_holdout")],
        },
    }


def _atomic_episode(parent: Path, name: str, writer: Any) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{name}.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        writer(temporary)
        target = parent / name
        if target.exists():
            raise FileExistsError(target)
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_episode(directory: Path, metadata: dict[str, Any], frames: dict[str, list[Any]], start_state: Any, end_state: Any, config: RecorderConfig) -> None:
    arrays = _save_npz(directory / "trajectory.npz", frames)
    _save_npy(directory / "start_state.npy", start_state)
    _save_npy(directory / "end_state.npy", end_state)
    videos = {
        "front": _save_video(directory / "front.mp4", [v for v in frames["front"] if v is not None], config.hz),
        "wrist": _save_video(directory / "wrist.mp4", [v for v in frames["wrist"] if v is not None], config.hz),
    }
    expected_paths = {
        "trajectory": "trajectory.npz",
        "front": "front.mp4",
        "wrist": "wrist.mp4",
    }
    if validate_episode_relative_path(str(arrays["path"])) != expected_paths["trajectory"]:
        raise ValueError("trajectory metadata path must be episode-relative")
    for camera in ("front", "wrist"):
        video_path = videos[camera].get("path")
        if validate_episode_relative_path(str(video_path)) != expected_paths[camera]:
            raise ValueError(f"{camera} metadata path must be episode-relative")
    metadata["schema_version"] = "track3_reverse_pilot_v4"
    metadata.update({
        "frame_count": len(frames["actions"]),
        "trajectory": arrays,
        "start_state_sha256": sha256_file(directory / "start_state.npy"),
        "end_state_sha256": sha256_file(directory / "end_state.npy"),
        "end_success": bool(frames["success"][-1]) if frames["success"] else False,
        "deadline_miss_count": int(sum(bool(value) for value in frames["deadline_miss"])),
        "videos": videos,
    })
    write_json(directory / "metadata.json", metadata)


def write_diagnostic(task_label: Any, reason: str, details: dict[str, Any] | None = None) -> Path:
    """Persist a small atomic failure record without creating an episode."""
    name = f"{task_label}_{int(time.time())}.json"
    path = OUTPUT_ROOT / "diagnostic" / name
    write_json(path, {"schema_version": "track3_reverse_pilot_v4_diagnostic", "status": "diagnostic", "task_label": task_label, "reason": reason, "details": details or {}, "created_at_utc": utc_now()})
    return path


def _inner_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _eval_one_predicate(problem: Any, triple: Sequence[Any]) -> bool:
    pred = [str(triple[0]).lower(), *[str(part) for part in triple[1:]]]
    return bool(problem._eval_predicate(pred))


def _eval_profile_predicates(env: Any, profile: dict[str, Any]) -> tuple[bool, bool]:
    sources = source_eval_triples(profile)
    goals = goal_eval_triples(profile)
    problem = _inner_env(env)
    if not sources or not goals:
        return False, bool(env.check_success())
    try:
        source = all(_eval_one_predicate(problem, row) for row in sources)
        goal = all(_eval_one_predicate(problem, row) for row in goals)
        return bool(source), bool(goal)
    except Exception:
        return False, bool(env.check_success())


def record_episode(task_index: int | None, operator: str, config: RecorderConfig, *, task_id: str | None = None) -> dict[str, Any]:
    """Run only with explicit human ``--record``; no automatic input is used."""
    try:
        profile = resolve_record_profile(task_id=task_id, task_index=task_index)
        guard = gpu3_guard()
        if guard["status"] != "passed":
            return {"status": "track3_teleop_preview_blocked", "reason": guard.get("reason"), "gpu3_guard": guard}
    except Exception as exc:
        return {"status": "track3_teleop_recorder_remediation_failed", "reason": str(exc)}
    configure_public_libero()
    if profile.get("official_public_example"):
        from compe.t3 import register_t3
        register_t3()
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=str(profile["bddl_path"]), control_freq=20, use_camera_obs=True,
                             camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=config.camera_height,
                             camera_widths=config.camera_width, render_gpu_device_id=0,
                             horizon=max(int(config.max_steps) + 20, 1200), ignore_done=True)
    server: PreviewServer | None = None
    task_label = profile.get("task_index") if profile.get("task_index") is not None else profile["task_id"]
    try:
        initial_obs = env.reset()
        if profile.get("start_state_path"):
            state = np.load(profile["start_state_path"])
            initial_obs = env.set_init_state(state)
            start_state = np.asarray(state, dtype=np.float64).copy()
        else:
            hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            height_sites = [(str(subject), str(site)) for subject, site in (profile.get("init_height_sites") or ())]
            pairs = upright_rest_pairs(profile)
            require_upright = bool(profile.get("require_upright", True))
            start_state = None
            initial_obs = None
            for _attempt in range(8):
                if _attempt:
                    initial_obs = env.reset()
                for subject, site in height_sites:
                    rest_object_to_site_height(env, subject, site)
                for subject, support in pairs:
                    rest_object_on_support(env, str(subject), str(support))
                settle_steps = 16 if height_sites else (24 if pairs else 8)
                stable_ticks = 0
                subjects = [subject for subject, _ in height_sites] or [subject for subject, _ in pairs] or ([str(profile["upright_subject"])] if profile.get("upright_subject") else [])
                exploded = False
                for _ in range(settle_steps):
                    initial_obs, _, _, _ = env.step(hold)
                    if not scene_bodies_ok(env, subjects):
                        exploded = True
                        break
                    source_now, _ = _eval_profile_predicates(env, profile)
                    upright = True
                    slow = True
                    for subject in subjects:
                        body_id = _inner_env(env).obj_body_id[str(subject)]
                        if require_upright and float(local_z_world(env.sim.data.body_xquat[body_id])[2]) < 0.95:
                            upright = False
                        if object_speed(env, str(subject)) >= 0.02:
                            slow = False
                    if source_now and upright and slow:
                        stable_ticks += 1
                        if stable_ticks >= 4:
                            break
                    else:
                        stable_ticks = 0
                if exploded:
                    continue
                if pairs and (stable_ticks < 4 or (require_upright and any(float(local_z_world(env.sim.data.body_xquat[_inner_env(env).obj_body_id[str(subject)]])[2]) < 0.95 for subject, _ in pairs))):
                    for subject, support in pairs:
                        rest_object_on_support(env, str(subject), str(support))
                    for _ in range(16):
                        initial_obs, _, _, _ = env.step(hold)
                        if not scene_bodies_ok(env, subjects):
                            exploded = True
                            break
                if exploded:
                    continue
                zero_sim_velocity(env)
                start_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
                if not np.isfinite(start_state).all() or not scene_bodies_ok(env, subjects):
                    continue
                initial_obs = env.set_init_state(start_state)
                source0, goal0 = _eval_profile_predicates(env, profile)
                if source0 is True and goal0 is False and scene_bodies_ok(env, subjects):
                    break
                start_state = None
            if start_state is None or initial_obs is None:
                raise RuntimeError("failed to obtain a stable official start state without dropping objects through the table")
        if env.check_success():
            raise RuntimeError("goal is already true at the start state")
        source0, goal0 = _eval_profile_predicates(env, profile)
        if source0 is not True or goal0 is not False:
            names = sorted(getattr(_inner_env(env), "object_states_dict", {}) or {})
            raise RuntimeError(f"start predicates invalid: source={source0} goal={goal0} objects={names}")
        machine = ActionStateMachine(config=config)
        machine.state.paused = True
        machine.state.gripper = _safe_gripper(config.default_gripper)
        server = PreviewServer(machine, config.preview_host, config.preview_port)
        server.start()
        print(json.dumps({"status": "preview_listening", "preview_url": server.url, "task_id": profile["task_id"], "task_index": profile.get("task_index"), "paused": True, "gripper_label": "OPEN", "instruction": profile["instruction"], "operator_hz": config.operator_hz, "tick_ms": config.tick_ms()}, ensure_ascii=False), flush=True)
        server.publish(_image_from_obs(initial_obs, "agentview"), _image_from_obs(initial_obs, "robot0_eye_in_hand"),
                       env=env, profile=profile,
                       task_index=profile.get("task_index"), task_id=profile["task_id"], instruction=profile["instruction"],
                       goal_predicate="; ".join(profile.get("goal_predicates") or []), step=0,
                       hz=0.0, lag_ms=0.0, success=bool(env.check_success()), source=True, goal=False,
                       required_contact=False, demonstrator_contact=False, source_receiver_contact=True,
                       goal_receiver_contact=False, unrelated_contact=False, candidate_gate="not_evaluated",
                       max_steps=config.max_steps, eval_step_budget=config.eval_step_budget,
                       operator_hz=config.operator_hz, tick_ms=config.tick_ms(), physics_guard="ok")
        if not server.wait_for_first_frame(config.preview_wait_seconds):
            return {"status": "track3_teleop_preview_blocked", "reason": "preview did not connect and fetch a first frame", "preview_url": server.url}
        machine.event("connect")
        metadata = _metadata(profile, operator, config, {**(profile.get("lock_info") or {}), "preview_url": server.url})
        frames = empty_gold_buffers()
        samples: list[dict[str, Any]] = []
        recording_start = time.monotonic()
        next_tick = recording_start
        session_deadline = recording_start + float(config.session_timeout_seconds)
        step = 0
        with raw_terminal():
            while step < int(config.max_steps):
                if time.monotonic() >= session_deadline:
                    metadata["status"] = "rejected_timeout"
                    metadata["timeout_reason"] = "session_wall_clock"
                    break
                terminal_key = _read_key(max(0.0, next_tick - time.monotonic()))
                if terminal_key is not None:
                    if terminal_key in {"\n", "\r"}:
                        machine.event("keydown", "enter")
                    elif terminal_key == "\x03":
                        machine.event("keydown", "x")
                    else:
                        key = terminal_key.lower()
                        if key in MOTION_KEYS:
                            machine.event("keydown", key)
                        elif key in {"g", "h", "[", "]", "p", "c", "z", "x", "q"}:
                            machine.event("keydown", key)
                if machine.consume("reset"):
                    reset_obs = env.set_init_state(start_state)
                    frames = empty_gold_buffers()
                    samples = []
                    step = 0
                    machine.state.reset_count += 1
                    machine.state.gripper = _safe_gripper(config.default_gripper)
                    server.publish(_image_from_obs(reset_obs, "agentview"), _image_from_obs(reset_obs, "robot0_eye_in_hand"),
                                   env=env, profile=profile,
                                   task_index=profile.get("task_index"), task_id=profile["task_id"], instruction=profile["instruction"],
                                   goal_predicate="; ".join(profile.get("goal_predicates") or []), step=0,
                                   hz=0.0, lag_ms=0.0, success=bool(env.check_success()), source=True, goal=False,
                                   required_contact=False, candidate_gate="not_evaluated",
                                   max_steps=config.max_steps, eval_step_budget=config.eval_step_budget,
                                   operator_hz=config.operator_hz, tick_ms=config.tick_ms(), physics_guard="ok")
                action = machine.next_action()
                validate_action(action)
                if machine.state.abort:
                    metadata["status"] = "rejected_abort"
                    break
                if machine.state.paused:
                    time.sleep(min(0.05, max(0.0, next_tick - time.monotonic())))
                    if time.monotonic() >= next_tick:
                        next_tick += config.tick_period()
                    continue
                tick = time.monotonic()
                obs, reward, done, info = env.step(action)
                now = time.monotonic()
                watched = [str(subject) for subject, _ in (profile.get("init_height_sites") or ())] or [str(name) for name in ((profile.get("roles") or {}).get("subject") or ())]
                if watched and not scene_bodies_ok(env, watched):
                    reset_obs = env.set_init_state(start_state)
                    machine.state.paused = True
                    machine.state.held_keys.clear()
                    machine.state.pulse_remaining = 0
                    machine.state.pulse_key = None
                    server.publish(
                        _image_from_obs(reset_obs, "agentview"), _image_from_obs(reset_obs, "robot0_eye_in_hand"),
                        env=env, profile=profile,
                        task_index=profile.get("task_index"), task_id=profile["task_id"], instruction=profile["instruction"],
                        goal_predicate="; ".join(profile.get("goal_predicates") or []), step=step,
                        hz=0.0, lag_ms=0.0, success=bool(env.check_success()), source=True, goal=False,
                        required_contact=False, candidate_gate="not_evaluated",
                        max_steps=config.max_steps, eval_step_budget=config.eval_step_budget,
                        operator_hz=config.operator_hz, tick_ms=config.tick_ms(),
                        physics_guard="restored_after_unstable",
                    )
                    next_tick += config.tick_period()
                    continue
                contact_matrix = raw_contact_matrix_from_sim(env)
                check_success = bool(env.check_success())
                source, goal = _eval_profile_predicates(env, profile)
                eef = None
                with contextlib.suppress(Exception):
                    eef = list(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])
                sample = append_gold_frame(
                    frames, action=action, state=np.asarray(env.get_sim_state()), reward=float(reward), done=bool(done),
                    success=check_success, front=_image_from_obs(obs, "agentview"), wrist=_image_from_obs(obs, "robot0_eye_in_hand"),
                    timestamp=now - recording_start, deadline_miss=bool(now > next_tick + 0.005), contact_matrix=contact_matrix,
                    source=source, goal=goal, eef_pos=eef, gripper=float(action[-1]), check_success=check_success,
                    contact_roles=profile.get("roles"),
                )
                samples.append(sample)
                gates = evaluate_gold_gates(samples)
                if not profile.get("language_gold_ok"):
                    gates = {**gates, "save_allowed": False, "candidate_gate": "failed", "language_unqualified": True}
                server.publish(
                    frames["front"][-1], frames["wrist"][-1], env=env, profile=profile,
                    task_index=profile.get("task_index"), task_id=profile["task_id"], instruction=profile["instruction"],
                    goal_predicate="; ".join(profile.get("goal_predicates") or []), step=step + 1, hz=1.0 / max(now - tick, 1e-6),
                    lag_ms=max(0.0, (now - next_tick) * 1000), success=check_success, source=source, goal=goal,
                    required_contact=bool(sample["required_contact"]), demonstrator_contact=bool(frames["demonstrator_contact"][-1]),
                    source_receiver_contact=bool(frames["source_receiver_contact"][-1]), goal_receiver_contact=bool(frames["goal_receiver_contact"][-1]),
                    unrelated_contact=bool(frames["unrelated_contact"][-1]), candidate_gate=gates["candidate_gate"],
                    max_steps=config.max_steps, eval_step_budget=config.eval_step_budget,
                    operator_hz=config.operator_hz, tick_ms=config.tick_ms(), physics_guard="ok",
                )
                if terminal_key is not None and terminal_key.lower() in MOTION_KEYS:
                    machine.event("keyup", terminal_key.lower())
                if machine.consume("save"):
                    metadata["gold_gates"] = gates
                    if not profile.get("language_gold_ok"):
                        metadata["status"] = "rejected_language_unqualified"
                    else:
                        metadata["status"] = "pending_replay" if gates.get("save_allowed") else "rejected_predicate_fail"
                    break
                if machine.state.abort:
                    metadata["status"] = "rejected_abort"
                    break
                step += 1
                next_tick += config.tick_period()
        if metadata["status"] == "recording":
            metadata["status"] = "rejected_timeout"
        bucket = "pending" if metadata["status"] == "pending_replay" else ("diagnostic" if metadata["status"] == "rejected_abort" else "rejected")
        name = f"{profile['task_id']}_{int(time.time())}"
        end_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        target = _atomic_episode(OUTPUT_ROOT / bucket, name, lambda directory: _write_episode(directory, metadata, frames, start_state, end_state, config))
        metadata["episode_dir"] = str(target)
        return metadata
    except KeyboardInterrupt:
        diagnostic = write_diagnostic(task_label, "keyboard_interrupt")
        return {"status": "rejected_interrupt", "task_id": profile["task_id"], "task_index": profile.get("task_index"), "diagnostic": str(diagnostic)}
    except Exception as exc:
        diagnostic = write_diagnostic(task_label, f"{type(exc).__name__}: {exc}")
        return {"status": "track3_teleop_recorder_remediation_failed", "task_id": profile.get("task_id"), "task_index": profile.get("task_index"), "reason": f"{type(exc).__name__}: {exc}", "diagnostic": str(diagnostic)}
    finally:
        if server is not None:
            server.stop()
        env.close()


def self_test() -> dict[str, Any]:
    config = RecorderConfig()
    assert config.default_gripper == -1.0
    machine = ActionStateMachine(config=config)
    no_key_grippers = []
    for _ in range(50):
        action = machine.next_action("")
        validate_action(action); no_key_grippers.append(float(action[-1]))
    assert set(no_key_grippers) == {-1.0}
    assert machine.snapshot()["gripper_label"] == "OPEN"
    machine.event("keydown", "h"); assert machine.next_action("")[-1] == 1
    assert machine.snapshot()["gripper_label"] == "CLOSED"
    machine.event("keydown", "g"); assert machine.next_action("")[-1] == -1
    machine.event("keydown", "p"); assert machine.next_action("w")[0] == 0
    machine.event("keydown", "p"); machine.event("connect"); machine.event("keydown", "w"); assert machine.next_action()[0] < 0
    machine.event("keyup", "w")
    assert machine.next_action()[0] < 0
    machine.event("keydown", "s"); assert machine.next_action()[0] > 0
    assert machine.snapshot()["axis"] == "X" and machine.snapshot()["sign"] == "+"
    machine.event("disconnect"); assert machine.snapshot()["paused"] is True and machine.snapshot()["held_keys"] == []
    machine.event("connect"); machine.event("keydown", "c"); assert machine.state.camera == "wrist"
    machine.event("keydown", "z"); assert machine.consume("reset"); machine.event("keydown", "enter"); assert machine.consume("save")
    machine.event("keydown", "x"); assert np.allclose(machine.next_action()[:6], 0)
    try:
        validate_action(np.zeros(7, dtype=np.float32))
    except ValueError:
        non_finite_or_invalid_rejected = True
    else:
        non_finite_or_invalid_rejected = False
    from tools.axis_pi05.track3_support_runtime_gate_remediation import success_path_fixture
    positive = success_path_fixture()
    grip = {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "robot0_right_hand", "geom1_name": "bowl_g", "geom2_name": "hand_g", "distance_m": -0.001}]}
    plate = {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "plate_1", "geom1_name": "bowl_g", "geom2_name": "plate_g", "distance_m": -0.001}]}
    table = {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "main_table", "geom1_name": "bowl_g", "geom2_name": "table_g", "distance_m": -0.001}]}
    buffers = empty_gold_buffers()
    samples = []
    action_open = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    action_close = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    state = np.zeros(79, dtype=np.float64)
    for _ in range(4):
        samples.append(append_gold_frame(buffers, action=action_open, state=state, reward=0, done=False, success=False, front=None, wrist=None, timestamp=0, deadline_miss=False, contact_matrix=plate, source=True, goal=False, eef_pos=[0, 0, 0], gripper=-1, check_success=False))
    for _ in range(4):
        samples.append(append_gold_frame(buffers, action=action_close, state=state, reward=0, done=False, success=False, front=None, wrist=None, timestamp=0, deadline_miss=False, contact_matrix=grip, source=True, goal=False, eef_pos=[0, 0, 0], gripper=1, check_success=False))
    for _ in range(4):
        samples.append(append_gold_frame(buffers, action=action_close, state=state, reward=0, done=False, success=False, front=None, wrist=None, timestamp=0, deadline_miss=False, contact_matrix=grip, source=False, goal=False, eef_pos=[0, 0, 0], gripper=1, check_success=False))
    for _ in range(4):
        samples.append(append_gold_frame(buffers, action=action_open, state=state, reward=0, done=False, success=True, front=None, wrist=None, timestamp=0, deadline_miss=False, contact_matrix=table, source=False, goal=True, eef_pos=[0, 0, 0], gripper=-1, check_success=True))
    passed_gates = evaluate_gold_gates(samples)
    assert passed_gates["save_allowed"] is True
    buffers_reset = empty_gold_buffers()
    assert all(len(value) == 0 for value in buffers_reset.values())
    final_only = evaluate_gold_gates(samples[:4] + samples[-1:])
    assert final_only["save_allowed"] is False
    with tempfile.TemporaryDirectory(prefix="track3_recorder_test_") as temporary:
        sample = Path(temporary) / "sample.npz"
        _save_npz(sample, {key: buffers[key] for key in buffers if key not in {"front", "wrist"}})
        with np.load(sample, allow_pickle=False) as archive:
            assert "required_contact" in archive.files
            assert archive["contact_body1"].dtype.kind in {"U", "S"}
        assert "Open (-1)" in PREVIEW_HTML and "Close (+1)" in PREVIEW_HTML
        assert "Reverse" not in OFFICIAL_T3_TASKS[DEFAULT_TASK_ID]["instruction"]
        red = resolve_record_profile(task_id=DEFAULT_TASK_ID)
        assert red["instruction"] == natural_instruction(OFFICIAL_T3_TASKS[DEFAULT_TASK_ID]["instruction"])
        assert red["language_gold_ok"] is True
        assert red["training_holdout"] is False
        two = resolve_record_profile(task_id="official_t3_two_mugs")
        book = resolve_record_profile(task_id="official_t3_yellow_book")
        soup = resolve_record_profile(task_id="official_t3_alphabet_soup")
        assert two["recordable"] is True and two["training_holdout"] is False
        assert two["instruction"].startswith("take the white mug")
        assert book["instruction"].startswith("take the yellow book")
        assert soup["instruction"].startswith("take the alphabet soup")
        assert book["init_height_sites"][0][0] == "yellow_book_2"
        assert np.allclose(_quat_local_axis_to_world_z(2), [1.0, 0.0, 0.0, 0.0])
        assert rest_object_to_site_height
        assert scene_bodies_ok
        assert all(not str(row["instruction"]).lower().startswith("reverse") for row in OFFICIAL_T3_TASKS.values())
        assert all(bool(row["recordable"]) and not bool(row["training_holdout"]) for row in OFFICIAL_T3_TASKS.values())
        assert "porcelain_mug_init_region" in " ".join(goal_site_name_candidates(two))
        assert "white_yellow_mug_init_region" in " ".join(goal_site_name_candidates(two))
        flipped = flip_front_preview(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
        assert flipped.shape == (2, 2, 3) and int(flipped[0, 0, 0]) == 9
        blank = np.zeros((48, 48, 3), dtype=np.uint8)
        marked = overlay_goal_box(blank, np.array([[8, 8], [40, 8], [40, 40], [8, 40]], dtype=np.int32))
        assert marked[24, 24].sum() > 0 and int(blank[24, 24].sum()) == 0
        assert "GOAL" in PREVIEW_HTML and "绿色" in PREVIEW_HTML
        assert "red_coffee_mug_init_region" in goal_site_name_candidates(OFFICIAL_T3_TASKS[DEFAULT_TASK_ID])[-1]
        preview_no_env = make_front_preview(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
        assert int(preview_no_env[0, 0, 0]) == 9
        assert float(local_z_world((1.0, 0.0, 0.0, 0.0))[2]) == 1.0
        assert config.hz == 20.0
        assert config.operator_hz == 2.0
        assert 1.0 <= config.operator_hz < config.hz
        assert config.tick_ms() == 500
        assert config.preview_refresh_ms <= 50
        assert config.translation_step >= 0.30
        assert config.motion_pulse_ticks <= 4
        assert config.eval_step_budget == 300
        assert config.max_steps >= 1200
        server = PreviewServer(ActionStateMachine(config=config), port=0)
        server.start(); server.stop()
        result = {
            "status": "passed", "schema_version": "track3_reverse_pilot_v4", "fps": config.hz,
            "no_key_ticks": 50, "gripper_values": sorted(set(no_key_grippers)),
            "default_gripper_open": True, "explicit_open_close": True,
            "pause_camera_reset_save_abort": True, "invalid_action_rejected": non_finite_or_invalid_rejected,
            "preview_bind_host": "127.0.0.1", "accepted_episode_count": 0,
            "simulator_started": False, "gpu_touched": False,
            "gold_gates_fixture_passed": passed_gates["status"] == "passed",
            "final_boolean_insufficient": final_only["save_allowed"] is False,
            "positive_sr_fixture": positive["required_contact"]["state"] == "passed",
            "official_red_mug_natural_language": True,
            "official_all_examples_recordable": True,
            "front_preview_rotated_180": True,
            "goal_overlay_preview_only": True,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID, choices=sorted(OFFICIAL_T3_TASKS))
    parser.add_argument("--task-index", type=int, default=None, choices=ALLOWED_TASKS)
    parser.add_argument("--operator", default="operator-local")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--preview-wait-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.record:
        result = record_episode(args.task_index, args.operator, RecorderConfig(preview_port=args.preview_port, preview_wait_seconds=args.preview_wait_seconds), task_id=None if args.task_index is not None else args.task_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "pending_replay" else 2
    parser.error("choose --self-test or --record")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
