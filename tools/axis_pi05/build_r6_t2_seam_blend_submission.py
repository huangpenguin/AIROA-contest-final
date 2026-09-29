#!/usr/bin/env python3
"""Build the conditionally-authorized R6 T2 seam-blend submission.

This builder is separate from the historical execute4 builder. It consumes only
the frozen R6 FAST staging and the passed local seam-blend screen, creates a new
staging/package namespace, and never uploads or submits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import requests

from .audit_fast_submission import (
    _add_local_asgi_runtime,
    _local_fast_server_command,
    _package_entries,
    _policy_static_audit,
    audit_package,
    write_package_sidecars,
)
from .common import ROOT

SOURCE_STAGING = ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging"
SOURCE_ZIP = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
PLAN = ROOT / "docs/AXIS_PI05_R6_T2_LIGHT_SMOOTHING_SCREEN_PLAN_20260917.md"
COMPARISON = ROOT / (
    "data/reports/axis_libero_30k/"
    "r6_t2_light_smoothing_screen_20260917_attempt_0001/comparison.json"
)
REPORT_ROOT = ROOT / (
    "data/reports/axis_libero_30k/"
    "r6_t2_seam_blend_submission_20260917_attempt_0001"
)
STAGING = ROOT / (
    "outputs/axis_pi05/"
    "r6_t2_seam_blend_submission_20260917_attempt_0001/staging"
)
OUTPUT = ROOT / (
    "artifacts/"
    "submission_axis_r6_step500_fast_seam_blend_t2_20260917.zip"
)
VALIDATOR = ROOT / "submission_template/validate_submission.py"

EXPECTED_SOURCE_ZIP_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
EXPECTED_FAST_SHA = "e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6"
EXPECTED_PLAN_SHA = "564062cc7e95bea845ef1251c5c3c951e7fa0cd0283da2d3ff97b00deb955ee7"
EXPECTED_SCREEN_STATUS = "candidate_passed_zip_authorized"
ALLOWED_CHANGED_ENTRIES = {
    "policy_server.py",
    "model_weights/config.json",
    "submission_manifest.json",
}
CHUNK = 8 * 1024 * 1024
OLD_ACTION_WEIGHT = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )


def strict_json_bytes(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def rewrite_policy_source(source: str) -> str:
    """Lock execute4 and add arm-only seam blending at the queue boundary."""
    replacements = {
        'n_action_steps.strip() != "8"': 'n_action_steps.strip() != "4"',
        "N_ACTION_STEPS must be 8": "N_ACTION_STEPS must be 4",
        "        n = 8\n": "        n = 4\n",
        "execute=8 ": "execute=4 ",
        "n_action_steps=8 action_ema=0.0 rtc=False ":
            "n_action_steps=4 action_ema=0.0 rtc=False seam_blend_arm_weight=0.1 ",
        "the first 8 actions enter": "the first 4 actions enter",
    }
    result = source
    for old, new in replacements.items():
        count = result.count(old)
        if count != 1:
            raise ValueError(
                f"policy rewrite expected one occurrence of {old!r}, got {count}"
            )
        result = result.replace(old, new)

    old = """        raw_action_np = None
        raw_chunk = None
        if self._rtc_enabled:
"""
    new = """        raw_action_np = None
        raw_chunk = None
        is_new_chunk = False
        if self._rtc_enabled:
"""
    if result.count(old) != 1:
        raise ValueError("policy rewrite could not add chunk-boundary state")
    result = result.replace(old, new)

    old = """                if len(self.policy._action_queue) == 0:
                    raw_chunk = self.policy.predict_action_chunk(processed)
"""
    new = """                is_new_chunk = len(self.policy._action_queue) == 0
                if is_new_chunk:
                    raw_chunk = self.policy.predict_action_chunk(processed)
"""
    if result.count(old) != 1:
        raise ValueError("policy rewrite could not identify the action queue boundary")
    result = result.replace(old, new)

    old = """        if action_np.shape != (7,) or not np.isfinite(action_np).all():
            raise ValueError(f"policy action must be finite shape (7,), got {action_np.shape}")
        self._last_trace = {
"""
    new = """        if action_np.shape != (7,) or not np.isfinite(action_np).all():
            raise ValueError(f"policy action must be finite shape (7,), got {action_np.shape}")
        # T2 seam blend: only the six arm dimensions are blended on the first
        # emitted frame of a freshly predicted chunk. The gripper is untouched.
        if is_new_chunk and self._prev_action is not None:
            action_np[:6] = (1.0 - 0.1) * action_np[:6] + 0.1 * self._prev_action[:6]
            action_np[6] = np.asarray(action_np[6], dtype=np.float32)
        self._prev_action = action_np.copy()
        self._last_trace = {
"""
    if result.count(old) != 1:
        raise ValueError("policy rewrite could not add seam blend")
    result = result.replace(old, new)

    if "execute=8 " in result or 'n_action_steps.strip() != "8"' in result:
        raise ValueError("execute8 policy literal survived rewrite")
    if "seam_blend_arm_weight=0.1" not in result:
        raise ValueError("seam blend marker missing from rewritten policy")
    compile(result, "policy_server.py", "exec")
    return result


def _copy_or_link(relative: str, source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative in {"policy_server.py", "model_weights/config.json"}:
        shutil.copy2(source, target)
        return "copied_for_seam_blend_rewrite"
    os.link(source, target)
    return "hardlinked_unchanged"


def _load_screen() -> dict[str, Any]:
    screen = strict_json(COMPARISON)
    if screen.get("status") != EXPECTED_SCREEN_STATUS:
        raise ValueError(f"screen status is not authorized: {screen.get('status')!r}")
    if screen.get("selected_variant") != "seam_blend":
        raise ValueError(f"screen selected {screen.get('selected_variant')!r}")
    if screen.get("release_authorized") is not True:
        raise ValueError("screen did not authorize release")
    candidate = screen["variants"]["seam_blend"]
    if candidate.get("passes_screen") is not True:
        raise ValueError("seam_blend candidate does not pass its recorded screen")
    if not all(candidate.get("gates", {}).values()):
        raise ValueError("seam_blend has a false gate")
    return screen


def prepare() -> dict[str, Any]:
    if (
        STAGING.exists()
        or REPORT_ROOT.exists()
        or OUTPUT.exists()
        or OUTPUT.with_name(OUTPUT.name + ".partial").exists()
    ):
        raise FileExistsError("refusing to overwrite existing seam-blend package namespace")
    for required in (SOURCE_STAGING, SOURCE_ZIP, PLAN, COMPARISON, VALIDATOR):
        if not required.exists():
            raise FileNotFoundError(required)
    screen = _load_screen()
    source_zip_sha = sha256_file(SOURCE_ZIP)
    if source_zip_sha != EXPECTED_SOURCE_ZIP_SHA:
        raise ValueError(f"source ZIP hash mismatch: {source_zip_sha}")
    fast_source = SOURCE_STAGING / "model_weights/model_fast.safetensors"
    fast_sha = sha256_file(fast_source)
    if fast_sha != EXPECTED_FAST_SHA:
        raise ValueError(f"FAST hash mismatch: {fast_sha}")
    plan_sha = sha256_file(PLAN)
    if plan_sha != EXPECTED_PLAN_SHA:
        raise ValueError(f"plan hash mismatch: {plan_sha}")

    REPORT_ROOT.mkdir(parents=True)
    STAGING.mkdir(parents=True)
    linked: list[dict[str, Any]] = []
    for relative, source in _package_entries(SOURCE_STAGING):
        target = STAGING / relative
        mode = _copy_or_link(relative, source, target)
        linked.append({
            "path": relative,
            "mode": mode,
            "bytes": source.stat().st_size,
            "same_inode": source.stat().st_ino == target.stat().st_ino,
        })

    config_path = STAGING / "model_weights/config.json"
    config = strict_json(config_path)
    if (
        config.get("type") != "pi05"
        or config.get("chunk_size") != 10
        or config.get("n_action_steps") != 8
    ):
        raise ValueError("source config is not locked pi05/chunk10/execute8")
    config["n_action_steps"] = 4
    write_json_atomic(config_path, config)

    server_path = STAGING / "policy_server.py"
    rewritten = rewrite_policy_source(server_path.read_text(encoding="utf-8"))
    server_path.write_text(rewritten, encoding="utf-8")
    server_audit = _policy_static_audit(server_path)
    checks = {
        "screen_status_authorized": screen["status"] == EXPECTED_SCREEN_STATUS,
        "source_zip_sha_locked": source_zip_sha == EXPECTED_SOURCE_ZIP_SHA,
        "fast_sha_locked": fast_sha == EXPECTED_FAST_SHA,
        "plan_sha_locked": plan_sha == EXPECTED_PLAN_SHA,
        "config_pi05": config.get("type") == "pi05",
        "config_chunk10": config.get("chunk_size") == 10,
        "config_execute4": config.get("n_action_steps") == 4,
        "server_static": server_audit.get("passed") is True,
        "server_accepts_only_execute4": (
            'n_action_steps.strip() != "4"' in rewritten
            and "N_ACTION_STEPS must be 4" in rewritten
        ),
        "server_has_seam_blend": (
            "seam_blend_arm_weight=0.1" in rewritten
            and "action_np[:6]" in rewritten
        ),
        "server_leaves_gripper_raw": (
            "action_np[6] = np.asarray(action_np[6]" in rewritten
        ),
        "no_slow_model_in_staging": not (
            STAGING / "model_weights/model.safetensors"
        ).exists(),
        "unchanged_entries_hardlinked": all(
            row["same_inode"]
            for row in linked
            if row["path"] not in {"policy_server.py", "model_weights/config.json"}
        ),
    }
    report = {
        "schema_version": "r6_t2_seam_blend_prepare_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_unix": time.time(),
        "source_staging": str(SOURCE_STAGING),
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": source_zip_sha,
        "fast_sha256": fast_sha,
        "plan": str(PLAN),
        "plan_sha256": plan_sha,
        "comparison": str(COMPARISON),
        "comparison_sha256": sha256_file(COMPARISON),
        "selected_variant": "seam_blend",
        "staging": str(STAGING),
        "checks": checks,
        "entries": linked,
        "server_static": server_audit,
        "auto_upload": False,
        "auto_submit": False,
    }
    write_json_atomic(REPORT_ROOT / "prepare.json", report)
    if report["status"] != "passed":
        raise RuntimeError(report)
    return report


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _serialize_obs(obs: dict[str, np.ndarray]) -> bytes:
    encoded = {
        key: {
            "data": value.tobytes(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in obs.items()
    }
    return msgpack.packb(encoded, use_bin_type=True)


def _decode_action(content: bytes) -> np.ndarray:
    unpacked = msgpack.unpackb(content, raw=False)
    action = np.frombuffer(unpacked["data"], dtype=np.float32).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"invalid action: shape={action.shape}")
    return action


def smoke(gpu: str = "0", health_timeout: int = 1200) -> dict[str, Any]:
    prepare_report = strict_json(REPORT_ROOT / "prepare.json")
    if prepare_report.get("status") != "passed":
        raise RuntimeError("prepare gate not passed")
    smoke_path = REPORT_ROOT / "runtime_smoke.json"
    if smoke_path.exists():
        raise FileExistsError(smoke_path)

    port = _free_port()
    log_path = REPORT_ROOT / "runtime_smoke.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "MODEL_DIR": str(STAGING / "model_weights"),
        "POLICY_TYPE": "pi05",
        "PI05_FAST": "1",
        "PI05_DTYPE": "float16",
        "N_ACTION_STEPS": "4",
        "ACTION_EMA": "0",
        "RTC_ENABLED": "0",
        "RTC_INFERENCE_DELAY": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PI05_FIDELITY_DUMP": "0",
        "TORCHDYNAMO_DISABLE": "1",
        "TORCH_COMPILE_DISABLE": "1",
        "TORCHINDUCTOR_DISABLE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    started = time.time()
    _add_local_asgi_runtime(env)
    proc: subprocess.Popen[str] | None = None
    error: str | None = None
    health_payload: Any = None
    actions: list[list[float]] = []
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.Popen(
                _local_fast_server_command(STAGING, port),
                cwd=str(STAGING),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            deadline = time.time() + health_timeout
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"policy server exited early: {proc.returncode}")
                try:
                    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
                    if response.status_code == 200:
                        health_payload = response.json()
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"health did not become ready within {health_timeout}s")

            reset = requests.post(
                f"http://127.0.0.1:{port}/reset",
                json={"instruction": "put the bowl on the stove", "seed": 20260917},
                timeout=30,
            )
            reset.raise_for_status()
            obs = {
                "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8),
                "robot0_joint_pos": np.zeros(7, dtype=np.float32),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
            }
            payload = _serialize_obs(obs)
            # Cross one execute4 queue seam (4 + 2 actions) in the fresh server.
            for _ in range(6):
                response = requests.post(
                    f"http://127.0.0.1:{port}/act",
                    data=payload,
                    headers={"Content-Type": "application/x-msgpack"},
                    timeout=30,
                )
                response.raise_for_status()
                actions.append(_decode_action(response.content).tolist())
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    action_array = np.asarray(actions, dtype=np.float32)
    checks = {
        "health_ok": health_payload == {"status": "ok"},
        "six_actions": len(actions) == 6,
        "actions_finite": bool(action_array.size and np.isfinite(action_array).all()),
        "actions_nonzero": bool(action_array.size and np.any(np.abs(action_array) > 0)),
        "log_fast": "FAST_REQUIRED_AND_VERIFIED pi05" in log_text,
        "log_execute4": "execute=4 " in log_text,
        "log_seam_blend": "seam_blend_arm_weight=0.1" in log_text,
        "no_error": error is None,
        "controlled_close": proc is not None and proc.returncode in (-15, 0),
    }
    report = {
        "schema_version": "r6_t2_seam_blend_runtime_smoke_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_unix": time.time(),
        "elapsed_sec": round(time.time() - started, 3),
        "gpu": gpu,
        "port": port,
        "health": health_payload,
        "action_count": len(actions),
        "actions": actions,
        "checks": checks,
        "error": error,
        "process_returncode": None if proc is None else proc.returncode,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
        "auto_upload": False,
        "auto_submit": False,
    }
    write_json_atomic(smoke_path, report)
    if report["status"] != "passed":
        raise RuntimeError(report)
    return report


def _write_zip(entries: list[tuple[str, Path]], output: Path) -> None:
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(output if output.exists() else temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for relative, path in entries:
                mode = path.stat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ValueError(f"non-regular package entry: {path}")
                info = zipfile.ZipInfo(relative)
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = (
                    zipfile.ZIP_STORED
                    if path.name.endswith(".safetensors")
                    else zipfile.ZIP_DEFLATED
                )
                with path.open("rb") as source, archive.open(
                    info, "w", force_zip64=True
                ) as destination:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        destination.write(block)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def compare_zip_entries(candidate: Path) -> dict[str, Any]:
    with zipfile.ZipFile(SOURCE_ZIP, "r") as source_archive, zipfile.ZipFile(
        candidate, "r"
    ) as candidate_archive:
        source = {info.filename: info for info in source_archive.infolist()}
        current = {info.filename: info for info in candidate_archive.infolist()}
    changed: list[str] = []
    unchanged_mismatch: list[str] = []
    for name in sorted(set(source) | set(current)):
        left = source.get(name)
        right = current.get(name)
        same = (
            left is not None
            and right is not None
            and left.file_size == right.file_size
            and left.CRC == right.CRC
        )
        if not same:
            changed.append(name)
            if name not in ALLOWED_CHANGED_ENTRIES:
                unchanged_mismatch.append(name)
    checks = {
        "entry_names_exact": set(source) == set(current),
        "changed_entries_exact": set(changed) == ALLOWED_CHANGED_ENTRIES,
        "unchanged_size_crc_exact": not unchanged_mismatch,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "changed_entries": changed,
        "unchanged_mismatch": unchanged_mismatch,
        "source_entry_count": len(source),
        "candidate_entry_count": len(current),
    }


def build_manifest(
    smoke: dict[str, Any],
    smoke_path: Path,
    screen: dict[str, Any],
) -> dict[str, Any]:
    source_manifest = strict_json(SOURCE_STAGING / "submission_manifest.json")
    manifest = copy.deepcopy(source_manifest)
    manifest.update({
        "schema_version": 2,
        "submission_status": "NOT_SUBMITTED",
        "candidate": "r6_dual_l2_fresh500_r1_seam_blend_t2_last_chance",
        "formal_winner": False,
        "official_result": False,
        "diagnostic_submission_candidate": True,
        "auto_submit": False,
        "auto_upload": False,
        "screen_evidence": {
            "status": screen["status"],
            "selected_variant": screen["selected_variant"],
            "comparison": str(COMPARISON),
            "comparison_sha256": sha256_file(COMPARISON),
            "gates": screen["variants"]["seam_blend"]["gates"],
            "metrics": screen["variants"]["seam_blend"]["metrics"],
        },
    })
    manifest["action_protocol"] = {
        "policy_type": "pi05",
        "model_chunk_size": 10,
        "n_action_steps": 4,
        "action_ema": 0.0,
        "rtc": False,
        "compile_model": False,
        "gradient_checkpointing": False,
        "offline": True,
        "postprocess": {
            "type": "chunk_seam_arm_blend",
            "arm_dimensions": 6,
            "old_action_weight": OLD_ACTION_WEIGHT,
            "gripper_smoothed": False,
            "first_frame_of_new_chunk_only": True,
        },
    }
    manifest["policy_server"] = {
        "sha256": sha256_file(STAGING / "policy_server.py"),
        "fast_required_and_verified": True,
        "execute_horizon": 4,
        "seam_blend_arm_weight": OLD_ACTION_WEIGHT,
        "gripper_smoothed": False,
    }
    manifest["fast_model"] = {
        **dict(source_manifest["fast_model"]),
        "sha256": EXPECTED_FAST_SHA,
    }
    manifest["trial_evidence"] = {
        "plan": str(PLAN),
        "plan_sha256": sha256_file(PLAN),
        "comparison": str(COMPARISON),
        "comparison_sha256": sha256_file(COMPARISON),
        "runtime_smoke": str(smoke_path),
        "runtime_smoke_sha256": sha256_file(smoke_path),
        "runtime_smoke_status": smoke["status"],
    }
    return manifest


def package() -> dict[str, Any]:
    prepare_report = strict_json(REPORT_ROOT / "prepare.json")
    smoke_report = strict_json(REPORT_ROOT / "runtime_smoke.json")
    screen = _load_screen()
    if prepare_report.get("status") != "passed" or smoke_report.get("status") != "passed":
        raise RuntimeError("prepare and runtime smoke must pass before packaging")

    manifest = build_manifest(
        smoke_report,
        REPORT_ROOT / "runtime_smoke.json",
        screen,
    )
    manifest_path = STAGING / "submission_manifest.json"
    write_json_atomic(manifest_path, manifest)
    entries = _package_entries(STAGING) + [("submission_manifest.json", manifest_path)]
    _write_zip(entries, OUTPUT)

    package_validation = audit_package(OUTPUT, REPORT_ROOT)
    delta = compare_zip_entries(OUTPUT)
    write_json_atomic(REPORT_ROOT / "zip_delta_audit.json", delta)
    static_proc = subprocess.run(
        [sys.executable, str(VALIDATOR), str(OUTPUT), "--static", "--json"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    static_report = {
        "status": "passed" if static_proc.returncode == 0 else "failed",
        "returncode": static_proc.returncode,
        "stdout": static_proc.stdout,
        "stderr": static_proc.stderr,
    }
    write_json_atomic(REPORT_ROOT / "validate_submission_static.json", static_report)
    with zipfile.ZipFile(OUTPUT, "r") as archive:
        fast_info = archive.getinfo("model_weights/model_fast.safetensors")
        config_in_zip = strict_json_bytes(archive.read("model_weights/config.json"))
        manifest_in_zip = strict_json_bytes(archive.read("submission_manifest.json"))
        server_in_zip = archive.read("policy_server.py").decode("utf-8")
    checks = {
        "package_validation": package_validation.get("status") == "passed",
        "zip_delta": delta.get("status") == "passed",
        "validate_submission_static": static_report["status"] == "passed",
        "fast_size_exact": (
            fast_info.file_size
            == (STAGING / "model_weights/model_fast.safetensors").stat().st_size
        ),
        "config_execute4": config_in_zip.get("n_action_steps") == 4,
        "manifest_execute4": (
            manifest_in_zip.get("action_protocol", {}).get("n_action_steps") == 4
        ),
        "manifest_screen_authorized": (
            manifest_in_zip.get("screen_evidence", {}).get("status")
            == EXPECTED_SCREEN_STATUS
        ),
        "server_execute4": (
            "execute=4 " in server_in_zip
            and "n_action_steps=4 action_ema=0.0 rtc=False" in server_in_zip
        ),
        "server_seam_blend": (
            "seam_blend_arm_weight=0.1" in server_in_zip
            and "action_np[:6]" in server_in_zip
        ),
        "no_partial": not OUTPUT.with_name(OUTPUT.name + ".partial").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "delta": delta, "static": static_report})

    sidecars = write_package_sidecars(OUTPUT, REPORT_ROOT)
    release = {
        "schema_version": "r6_t2_seam_blend_submission_release_v1",
        "status": "passed",
        "created_unix": time.time(),
        "candidate": manifest["candidate"],
        "zip": str(OUTPUT),
        "zip_sha256": package_validation["sha256"],
        "zip_bytes": package_validation["bytes"],
        "zip_entry_count": package_validation["entry_count"],
        "fast_model_sha256": EXPECTED_FAST_SHA,
        "action_protocol": manifest["action_protocol"],
        "checks": checks,
        "package_validation": package_validation,
        "zip_delta_audit": delta,
        "static_validation": static_report,
        "runtime_smoke": smoke_report,
        "screen_evidence": screen,
        "sidecars": sidecars,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }
    write_json_atomic(REPORT_ROOT / "release.json", release)
    return release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "smoke", "package", "all"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--health-timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.command in ("prepare", "all"):
        prepare()
    if args.command in ("smoke", "all"):
        smoke(args.gpu, args.health_timeout)
    if args.command in ("package", "all"):
        result = package()
        print(json.dumps({
            "status": result["status"],
            "zip": result["zip"],
            "sha256": result["zip_sha256"],
            "bytes": result["zip_bytes"],
            "auto_upload": False,
            "auto_submit": False,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

