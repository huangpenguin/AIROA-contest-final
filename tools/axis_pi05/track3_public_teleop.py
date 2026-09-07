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
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"
PUBLIC_LIBERO = ROOT / "LIBERO"
P0_LOCK = TRACK3_ROOT / "p0_lock.json"
P0_MAPPING = TRACK3_ROOT / "p0_task_mapping.json"
ALLOWED_TASKS = (19, 10, 20)
MOTION_KEYS = frozenset({"w", "s", "a", "d", "r", "f", "u", "o", "i", "k", "j", "l"})
KEY_BINDINGS = {
    "w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1),
    "r": (2, 1), "f": (2, -1), "u": (3, 1), "o": (3, -1),
    "i": (4, 1), "k": (4, -1), "j": (5, 1), "l": (5, -1),
}
KEY_AXIS = {
    "w": ("X", "+"), "s": ("X", "-"), "a": ("Y", "+"), "d": ("Y", "-"),
    "r": ("Z", "+"), "f": ("Z", "-"), "u": ("Rx", "+"), "o": ("Rx", "-"),
    "i": ("Ry", "+"), "k": ("Ry", "-"), "j": ("Rz", "+"), "l": ("Rz", "-"),
}
GOLD_NPZ_KEYS = (
    "actions", "states", "rewards", "dones", "success", "front", "wrist",
    "timestamps_monotonic", "deadline_miss", "contact_n", "contact_body1", "contact_body2",
    "contact_geom1", "contact_geom2", "contact_role", "required_contact", "demonstrator_contact",
    "source_receiver_contact", "goal_receiver_contact", "unrelated_contact", "source_predicate",
    "goal_predicate", "eef_pos", "gripper", "phase", "check_success",
)
MAX_CONTACTS = 32
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
    translation_step: float = 0.008
    rotation_step: float = 0.08
    default_gripper: float = -1.0
    max_steps: int = 1200
    camera_height: int = 128
    camera_width: int = 128
    preview_host: str = "127.0.0.1"
    preview_port: int = 8765
    preview_wait_seconds: float = 45.0


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
    return {key: [] for key in GOLD_NPZ_KEYS}


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


def pad_contacts(names: list[str]) -> np.ndarray:
    padded = list(names[:MAX_CONTACTS]) + [""] * max(0, MAX_CONTACTS - len(names))
    return np.asarray(padded, dtype=UNICODE_DTYPE)


def typed_npz_arrays(frames: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, values in frames.items():
        if key in {"contact_body1", "contact_body2", "contact_geom1", "contact_geom2", "contact_role", "phase"}:
            arrays[key] = np.asarray(values, dtype=UNICODE_DTYPE)
        elif key in {"required_contact", "demonstrator_contact", "source_receiver_contact", "goal_receiver_contact", "unrelated_contact", "source_predicate", "goal_predicate", "success", "dones", "deadline_miss", "check_success"}:
            arrays[key] = np.asarray(values, dtype=np.bool_)
        elif key == "contact_n":
            arrays[key] = np.asarray(values, dtype=np.int32)
        elif values:
            arrays[key] = np.asarray(values)
    return arrays


def evaluate_gold_gates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr

    if not samples:
        return {"status": "failed", "reason": "empty_buffer", "candidate_gate": "failed", "save_allowed": False}
    phases = [str(row.get("phase") or infer_phase(bool(row.get("source_predicate")), bool(row.get("goal_predicate")), bool(row.get("required_contact")), float(row.get("gripper", -1)))) for row in samples]
    source_values = [bool(row.get("source_predicate")) for row in samples]
    goal_values = [bool(row.get("goal_predicate")) for row in samples]
    initial_source = source_values[0] is True
    initial_goal_false = goal_values[0] is False
    barrier = None
    for index in range(len(samples) - 1, -1, -1):
        if bool(samples[index].get("required_contact")):
            barrier = index
            break
    required = sr.required_contact_gate(samples, barrier, stable_ticks=4)
    released = sr.source_released_gate(source_values, phases, stable_ticks=4)
    check_success = bool(samples[-1].get("check_success") or samples[-1].get("success"))
    goal = sr.goal_continuous_gate(goal_values, phases, check_success=check_success, stable_ticks=4)
    unrelated = any(bool(row.get("unrelated_contact")) for row in samples)
    unknown = any(row.get("contact_telemetry_complete") is False for row in samples)
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


def append_gold_frame(frames: dict[str, list[Any]], *, action: np.ndarray, state: np.ndarray, reward: float, done: bool, success: bool, front: Any, wrist: Any, timestamp: float, deadline_miss: bool, contact_matrix: dict[str, Any], source: bool, goal: bool, eef_pos: Sequence[float] | None, gripper: float, check_success: bool) -> dict[str, Any]:
    from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr

    truth = sr.contact_truth_v3(10, contact_matrix)
    contacts = list(contact_matrix.get("contacts") or [])
    roles = []
    for row in contacts[:MAX_CONTACTS]:
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
    frames["contact_body1"].append(pad_contacts([str(row.get("body1_name", "")) for row in contacts]))
    frames["contact_body2"].append(pad_contacts([str(row.get("body2_name", "")) for row in contacts]))
    frames["contact_geom1"].append(pad_contacts([str(row.get("geom1_name", "")) for row in contacts]))
    frames["contact_geom2"].append(pad_contacts([str(row.get("geom2_name", "")) for row in contacts]))
    frames["contact_role"].append(pad_contacts(roles))
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
        "unrelated_contact": bool(truth.get("unrelated_contact_runtime")),
        "contact_telemetry_complete": bool(truth.get("contact_telemetry_complete")),
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
            elif event == "keydown":
                if normalized in MOTION_KEYS:
                    self.state.held_keys.add(normalized)
                    self.state.last_motion_key = normalized
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
            motion = self.state.last_motion_key if self.state.last_motion_key in self.state.held_keys else ""
            return key_to_action(motion, self.state, self.config)

    def consume(self, field_name: str) -> bool:
        with self._lock:
            value = bool(getattr(self.state, field_name))
            if value:
                setattr(self.state, field_name, False)
            return value

    def snapshot(self) -> dict[str, Any]:
        motion = self.state.last_motion_key if self.state.last_motion_key in self.state.held_keys else ""
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
        ok, encoded = cv2.imencode(".jpg", array[..., ::-1])
        if not ok:
            raise RuntimeError("cv2.imencode returned false")
        return bytes(encoded)
    except ImportError:
        from PIL import Image
        import io
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


PREVIEW_HTML = """<!doctype html>
<meta charset="utf-8"><title>AXIS Track3 task10 gold teleop</title>
<style>body{font:14px sans-serif;background:#171717;color:#eee;margin:12px}main{display:flex;gap:12px}figure{margin:0}img{width:480px;max-width:46vw;background:#333}button{margin:3px;padding:8px 12px}#state{white-space:pre;font-family:monospace;background:#222;padding:8px}.open{background:#14532d;color:#fff}.close{background:#7f1d1d;color:#fff}</style>
<h2>AXIS Track3 task10 gold teleop — localhost only</h2>
<p>FRONT and WRIST are both live. Gripper is explicit OPEN/CLOSED. G/Open opens, H/Close closes. Disconnect pauses and clears held keys.</p>
<div id="state">waiting for simulator</div>
<main><figure><figcaption>FRONT</figcaption><img id="front" alt="front"></figure><figure><figcaption>WRIST</figcaption><img id="wrist" alt="wrist"></figure></main>
<p><button class="open" onclick="button('open')">Open (-1)</button><button class="close" onclick="button('close')">Close (+1)</button><button onclick="button('p')">Pause</button><button onclick="button('z')">Reset</button><button onclick="button('enter')">Save Candidate</button><button onclick="button('x')">Abort</button></p>
<p>Keyboard: W/S ±X, A/D ±Y, R/F ±Z; U/O ±Rx, I/K ±Ry, J/L ±Rz; G Open, H Close; P pause; Z reset; Enter save; X abort.</p>
<script>
const post=(event,key='')=>fetch('/event',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({event,key})});
const button=(key)=>post('button',key);
window.addEventListener('keydown',e=>{if(!e.repeat){post('keydown',e.key==='Enter'?'enter':e.key.toLowerCase());e.preventDefault();}});
window.addEventListener('keyup',e=>{post('keyup',e.key.toLowerCase());});
window.addEventListener('blur',()=>post('disconnect'));
window.addEventListener('focus',()=>post('connect'));
async function refresh(){try{const s=await fetch('/state').then(r=>r.json());document.getElementById('state').textContent=JSON.stringify(s,null,2);for(const k of ['front','wrist'])document.getElementById(k).src='/frame/'+k+'?t='+Date.now();}catch(e){}}
setInterval(refresh,150); refresh();
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
        self.status: dict[str, Any] = {
            "task_index": None, "instruction": "", "goal_predicate": "", "step": 0, "hz": 0.0, "lag_ms": 0.0,
            "success": False, "source": None, "goal": None, "required_contact": False, "demonstrator_contact": False,
            "source_receiver_contact": False, "goal_receiver_contact": False, "unrelated_contact": False,
            "candidate_gate": "not_evaluated", "cameras": ["FRONT", "WRIST"],
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

    def publish(self, front: Any, wrist: Any, **status: Any) -> None:
        self.frames["front"] = _encode_jpeg(front) if front is not None else None
        self.frames["wrist"] = _encode_jpeg(wrist) if wrist is not None else None
        self.status.update(status)

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
    return {"path": str(path), "sha256": sha256_file(path), "keys": keys, "frames": len(frames.get("actions", []))}


def _save_npy(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_video(path: Path, frames: list[Any], fps: float) -> dict[str, Any]:
    if not frames:
        return {"status": "no_frames"}
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
            return {"status": "written", "codec": codec, "path": str(path), "sha256": sha256_file(path)}
    return {"status": "codec_unavailable"}


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


def _metadata(task_index: int, operator: str, config: RecorderConfig, lock_info: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "track3_reverse_pilot_v3",
        "task_index": int(task_index),
        "operator": operator,
        "input_device": "keyboard_or_localhost_browser",
        "fps": config.hz,
        "action_contract": "7D OSC_POSE; translation/rotation delta; gripper exactly -1 or +1",
        "state_contract": "complete simulator state at start/end and every frame",
        "source": "official_public_hdf5_full_state_p1_snapshot",
        "reverse_instruction": row["reverse_instruction"],
        "reverse_goal_predicates": row["reverse_goal_predicates"],
        "lock": lock_info,
        "replay": "pending_fresh_process_3_of_3",
        "human_review": "pending",
        "status": "recording",
        "created_at_utc": utc_now(),
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


def write_diagnostic(task_index: int, reason: str, details: dict[str, Any] | None = None) -> Path:
    """Persist a small atomic failure record without creating an episode."""
    name = f"task_{task_index:02d}_{int(time.time())}.json"
    path = OUTPUT_ROOT / "diagnostic" / name
    write_json(path, {"schema_version": "track3_reverse_pilot_v3_diagnostic", "status": "diagnostic", "task_index": task_index, "reason": reason, "details": details or {}, "created_at_utc": utc_now()})
    return path


def record_episode(task_index: int, operator: str, config: RecorderConfig) -> dict[str, Any]:
    """Run only with explicit human ``--record``; no automatic input is used."""
    try:
        _lock, row, lock_info = _load_lock_and_row(task_index)
        guard = gpu3_guard()
        if guard["status"] != "passed":
            return {"status": "track3_teleop_preview_blocked", "reason": guard.get("reason"), "gpu3_guard": guard}
    except Exception as exc:
        return {"status": "track3_teleop_recorder_remediation_failed", "reason": str(exc)}
    configure_public_libero()
    from libero.libero.envs import OffScreenRenderEnv

    state = np.load(lock_info["p1_snapshot"])
    env = OffScreenRenderEnv(bddl_file_name=row["reverse_bddl_path"], control_freq=20, use_camera_obs=True,
                             camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=config.camera_height,
                             camera_widths=config.camera_width, render_gpu_device_id=0)
    server: PreviewServer | None = None
    try:
        env.reset()
        initial_obs = env.set_init_state(state)
        if env.check_success():
            raise RuntimeError("reverse goal is already true at the locked start state")
        # ``set_init_state`` refreshes observables without advancing physics.
        # Use that observation for the preview handshake so the saved start
        # state remains exactly the first state replay will restore.
        start_state = np.asarray(state, dtype=np.float64).copy()
        machine = ActionStateMachine(config=config)
        machine.state.paused = True
        machine.state.gripper = _safe_gripper(config.default_gripper)
        server = PreviewServer(machine, config.preview_host, config.preview_port)
        server.start()
        # The initial observation is not an action and is never written to the
        # trajectory. Recording remains blocked until the browser fetches it.
        server.publish(_image_from_obs(initial_obs, "agentview"), _image_from_obs(initial_obs, "robot0_eye_in_hand"),
                       task_index=task_index, instruction=row["reverse_instruction"],
                       goal_predicate="; ".join(row["reverse_goal_predicates"]), step=0,
                       hz=0.0, lag_ms=0.0, success=bool(env.check_success()), source=True, goal=False,
                       required_contact=False, demonstrator_contact=False, source_receiver_contact=True,
                       goal_receiver_contact=False, unrelated_contact=False, candidate_gate="not_evaluated")
        if not server.wait_for_first_frame(config.preview_wait_seconds):
            return {"status": "track3_teleop_preview_blocked", "reason": "preview did not connect and fetch a first frame", "preview_url": server.url}
        machine.event("connect")
        metadata = _metadata(task_index, operator, config, {**lock_info, "preview_url": server.url}, row)
        frames = empty_gold_buffers()
        samples: list[dict[str, Any]] = []
        recording_start = time.monotonic()
        next_tick = recording_start
        with raw_terminal():
            for step in range(config.max_steps):
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
                    reset_obs = env.set_init_state(state)
                    frames = empty_gold_buffers()
                    samples = []
                    machine.state.reset_count += 1
                    machine.state.gripper = _safe_gripper(config.default_gripper)
                    start_state = np.asarray(state, dtype=np.float64).copy()
                    server.publish(_image_from_obs(reset_obs, "agentview"), _image_from_obs(reset_obs, "robot0_eye_in_hand"),
                                   task_index=task_index, instruction=row["reverse_instruction"],
                                   goal_predicate="; ".join(row["reverse_goal_predicates"]), step=0,
                                   hz=0.0, lag_ms=0.0, success=bool(env.check_success()), source=True, goal=False,
                                   required_contact=False, candidate_gate="not_evaluated")
                action = machine.next_action()
                validate_action(action)
                if machine.state.abort:
                    metadata["status"] = "rejected_abort"
                    break
                tick = time.monotonic()
                obs, reward, done, info = env.step(action)
                now = time.monotonic()
                contact_matrix = raw_contact_matrix_from_sim(env)
                check_success = bool(env.check_success())
                try:
                    source = bool(env._eval_predicate(["On", "akita_black_bowl_1", "plate_1"]))
                    goal = bool(env._eval_predicate(["On", "akita_black_bowl_1", "main_table"]))
                except Exception:
                    source = bool(contact_matrix) and bool(check_success) is False
                    goal = check_success
                eef = None
                with contextlib.suppress(Exception):
                    eef = list(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])
                sample = append_gold_frame(
                    frames, action=action, state=np.asarray(env.get_sim_state()), reward=float(reward), done=bool(done),
                    success=check_success, front=_image_from_obs(obs, "agentview"), wrist=_image_from_obs(obs, "robot0_eye_in_hand"),
                    timestamp=now - recording_start, deadline_miss=bool(now > next_tick + 0.005), contact_matrix=contact_matrix,
                    source=source, goal=goal, eef_pos=eef, gripper=float(action[-1]), check_success=check_success,
                )
                samples.append(sample)
                gates = evaluate_gold_gates(samples)
                server.publish(
                    frames["front"][-1], frames["wrist"][-1], task_index=task_index, instruction=row["reverse_instruction"],
                    goal_predicate="; ".join(row["reverse_goal_predicates"]), step=step + 1, hz=1.0 / max(now - tick, 1e-6),
                    lag_ms=max(0.0, (now - next_tick) * 1000), success=check_success, source=source, goal=goal,
                    required_contact=bool(sample["required_contact"]), demonstrator_contact=bool(frames["demonstrator_contact"][-1]),
                    source_receiver_contact=bool(frames["source_receiver_contact"][-1]), goal_receiver_contact=bool(frames["goal_receiver_contact"][-1]),
                    unrelated_contact=bool(frames["unrelated_contact"][-1]), candidate_gate=gates["candidate_gate"],
                )
                if terminal_key is not None and terminal_key.lower() in MOTION_KEYS:
                    machine.event("keyup", terminal_key.lower())
                if machine.consume("save"):
                    metadata["gold_gates"] = gates
                    metadata["status"] = "pending_replay" if gates.get("save_allowed") else "rejected_predicate_fail"
                    break
                if machine.state.abort:
                    metadata["status"] = "rejected_abort"
                    break
                next_tick += 1.0 / config.hz
        if metadata["status"] == "recording":
            metadata["status"] = "rejected_timeout"
        bucket = "pending" if metadata["status"] == "pending_replay" else ("diagnostic" if metadata["status"] == "rejected_abort" else "rejected")
        name = f"task_{task_index:02d}_{int(time.time())}"
        end_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        target = _atomic_episode(OUTPUT_ROOT / bucket, name, lambda directory: _write_episode(directory, metadata, frames, start_state, end_state, config))
        metadata["episode_dir"] = str(target)
        return metadata
    except KeyboardInterrupt:
        diagnostic = write_diagnostic(task_index, "keyboard_interrupt")
        return {"status": "rejected_interrupt", "task_index": task_index, "diagnostic": str(diagnostic)}
    except Exception as exc:
        diagnostic = write_diagnostic(task_index, f"{type(exc).__name__}: {exc}")
        return {"status": "track3_teleop_recorder_remediation_failed", "task_index": task_index, "reason": f"{type(exc).__name__}: {exc}", "diagnostic": str(diagnostic)}
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
    machine.event("keydown", "p"); machine.event("connect"); machine.event("keydown", "w"); assert machine.next_action()[0] > 0
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
        server = PreviewServer(ActionStateMachine(config=config), port=0)
        server.start(); server.stop()
        result = {
            "status": "passed", "schema_version": "track3_reverse_pilot_v3", "fps": config.hz,
            "no_key_ticks": 50, "gripper_values": sorted(set(no_key_grippers)),
            "default_gripper_open": True, "explicit_open_close": True,
            "pause_camera_reset_save_abort": True, "invalid_action_rejected": non_finite_or_invalid_rejected,
            "preview_bind_host": "127.0.0.1", "accepted_episode_count": 0,
            "simulator_started": False, "gpu_touched": False,
            "gold_gates_fixture_passed": passed_gates["status"] == "passed",
            "final_boolean_insufficient": final_only["save_allowed"] is False,
            "positive_sr_fixture": positive["required_contact"]["state"] == "passed",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--task-index", type=int, default=19, choices=ALLOWED_TASKS)
    parser.add_argument("--operator", default="operator-local")
    parser.add_argument("--preview-port", type=int, default=8765)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.record:
        result = record_episode(args.task_index, args.operator, RecorderConfig(preview_port=args.preview_port))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "pending_replay" else 2
    parser.error("choose --self-test or --record")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
