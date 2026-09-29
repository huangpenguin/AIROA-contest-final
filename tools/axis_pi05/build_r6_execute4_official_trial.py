#!/usr/bin/env python3
"""Build one R6 FAST submission whose only behavior change is execute8 -> execute4.

This is a user-authorized, high-variance official trial. It never uploads or submits.
Historical R6 staging and ZIP artifacts are read-only inputs.
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
    _package_entries,
    _add_local_asgi_runtime,
    _local_fast_server_command,
    _policy_static_audit,
    audit_package,
    write_package_sidecars,
)
from .common import ROOT

SOURCE_STAGING = ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging"
SOURCE_ZIP = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
PLAN = ROOT / "docs/AXIS_PI05_R6_EXECUTE4_OFFICIAL_TRIAL_PLAN_20260916.md"
COMPARISON = ROOT / (
    "data/reports/axis_libero_30k/"
    "r6_final_inference_horizon_sweep_20260916_attempt_0001/comparison.json"
)
REPORT_ROOT = ROOT / (
    "data/reports/axis_libero_30k/"
    "r6_execute4_official_trial_20260916_attempt_0001"
)
STAGING = ROOT / (
    "outputs/axis_pi05/"
    "r6_execute4_official_trial_20260916_attempt_0001/staging"
)
OUTPUT = ROOT / (
    "artifacts/"
    "submission_axis_r6_step500_fast_execute4_official_trial_20260916.zip"
)
VALIDATOR = ROOT / "submission_template/validate_submission.py"

EXPECTED_SOURCE_ZIP_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
EXPECTED_FAST_SHA = "e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6"
EXPECTED_COMPARISON_SHA = "32cf8efb3a8100329bae53b3b4be2ed09e252350ec6db617c84333ddf04bd15b"
EXPECTED_PLAN_SHA = "c22f21a67cfe1a0538fc9ed8a2c97d495444a595082fe34fd0def001395a5eb1"
ALLOWED_CHANGED_ENTRIES = {
    "policy_server.py",
    "model_weights/config.json",
    "submission_manifest.json",
}
CHUNK = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


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
    replacements = {
        'n_action_steps.strip() != "8"': 'n_action_steps.strip() != "4"',
        "N_ACTION_STEPS must be 8": "N_ACTION_STEPS must be 4",
        "        n = 8\n": "        n = 4\n",
        "execute=8 ": "execute=4 ",
        "n_action_steps=8 action_ema": "n_action_steps=4 action_ema",
        "the first 8 actions enter": "the first 4 actions enter",
    }
    result = source
    for old, new in replacements.items():
        count = result.count(old)
        if count != 1:
            raise ValueError(f"policy rewrite expected one occurrence of {old!r}, got {count}")
        result = result.replace(old, new)
    forbidden = (
        'n_action_steps.strip() != "8"',
        "N_ACTION_STEPS must be 8",
        "        n = 8\n",
        "execute=8 ",
        "n_action_steps=8 action_ema",
        "the first 8 actions enter",
    )
    if any(item in result for item in forbidden):
        raise ValueError("execute8 policy literal survived rewrite")
    compile(result, "policy_server.py", "exec")
    return result


def build_manifest(smoke: dict[str, Any], smoke_path: Path) -> dict[str, Any]:
    source_manifest = json.loads(
        (SOURCE_STAGING / "submission_manifest.json").read_text(encoding="utf-8")
    )
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "schema_version": 2,
            "submission_status": "NOT_SUBMITTED",
            "track": "user_selected_official_track",
            "candidate": "r6_dual_l2_fresh500_r1_execute4_user_override",
            "formal_winner": False,
            "official_result": False,
            "diagnostic_submission_candidate": True,
            "auto_submit": False,
            "auto_upload": False,
            "user_risk_override": {
                "authorized": True,
                "reason": "last official score opportunity would otherwise expire",
                "screen_status": "failed_preregistered_plus3_gate",
                "c20_execute4": 17,
                "c20_execute8": 15,
                "c20_delta": 2,
                "paired_net_wins": 2,
                "b80_performed": False,
            },
        }
    )
    manifest["action_protocol"] = {
        "policy_type": "pi05",
        "model_chunk_size": 10,
        "n_action_steps": 4,
        "action_ema": 0.0,
        "rtc": False,
        "compile_model": False,
        "gradient_checkpointing": False,
        "offline": True,
    }
    manifest["policy_server"] = {
        "sha256": sha256_file(STAGING / "policy_server.py"),
        "fast_required_and_verified": True,
        "execute_horizon": 4,
    }
    manifest["fast_model"] = {
        **dict(source_manifest["fast_model"]),
        "sha256": EXPECTED_FAST_SHA,
    }
    manifest["trial_evidence"] = {
        "plan": str(PLAN),
        "plan_sha256": sha256_file(PLAN),
        "comparison": str(COMPARISON),
        "comparison_sha256": EXPECTED_COMPARISON_SHA,
        "runtime_smoke": str(smoke_path),
        "runtime_smoke_sha256": sha256_file(smoke_path),
        "runtime_smoke_status": smoke["status"],
    }
    return manifest


def _copy_or_link(relative: str, source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if relative in {"policy_server.py", "model_weights/config.json"}:
        shutil.copy2(source, target)
        return "copied_for_execute4_rewrite"
    os.link(source, target)
    return "hardlinked_unchanged"


def prepare() -> dict[str, Any]:
    if STAGING.exists():
        raise FileExistsError(f"refusing to overwrite staging: {STAGING}")
    if REPORT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite report root: {REPORT_ROOT}")
    for required in (SOURCE_STAGING, SOURCE_ZIP, PLAN, COMPARISON, VALIDATOR):
        if not required.exists():
            raise FileNotFoundError(required)

    source_zip_sha = sha256_file(SOURCE_ZIP)
    if source_zip_sha != EXPECTED_SOURCE_ZIP_SHA:
        raise ValueError(f"source ZIP hash mismatch: {source_zip_sha}")
    comparison_sha = sha256_file(COMPARISON)
    if comparison_sha != EXPECTED_COMPARISON_SHA:
        raise ValueError(f"comparison hash mismatch: {comparison_sha}")
    fast_source = SOURCE_STAGING / "model_weights/model_fast.safetensors"
    plan_sha = sha256_file(PLAN)
    if plan_sha != EXPECTED_PLAN_SHA:
        raise ValueError(f"plan hash mismatch: {plan_sha}")
    fast_sha = sha256_file(fast_source)
    if fast_sha != EXPECTED_FAST_SHA:
        raise ValueError(f"FAST hash mismatch: {fast_sha}")

    REPORT_ROOT.mkdir(parents=True)
    STAGING.mkdir(parents=True)
    linked: list[dict[str, Any]] = []
    for relative, source in _package_entries(SOURCE_STAGING):
        target = STAGING / relative
        mode = _copy_or_link(relative, source, target)
        linked.append(
            {
                "path": relative,
                "mode": mode,
                "bytes": source.stat().st_size,
                "same_inode": source.stat().st_ino == target.stat().st_ino,
            }
        )

    config_path = STAGING / "model_weights/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type") != "pi05" or config.get("chunk_size") != 10:
        raise ValueError("source config is not locked pi05/chunk10")
    if config.get("n_action_steps") != 8:
        raise ValueError("source config is not execute8")
    config["n_action_steps"] = 4
    write_json_atomic(config_path, config)

    server_path = STAGING / "policy_server.py"
    rewritten = rewrite_policy_source(server_path.read_text(encoding="utf-8"))
    server_path.write_text(rewritten, encoding="utf-8")

    server_audit = _policy_static_audit(server_path)
    checks = {
        "source_zip_sha_locked": source_zip_sha == EXPECTED_SOURCE_ZIP_SHA,
        "fast_sha_locked": fast_sha == EXPECTED_FAST_SHA,
        "comparison_sha_locked": comparison_sha == EXPECTED_COMPARISON_SHA,
        "plan_sha_locked": plan_sha == EXPECTED_PLAN_SHA,
        "config_pi05": config.get("type") == "pi05",
        "config_chunk10": config.get("chunk_size") == 10,
        "config_execute4": config.get("n_action_steps") == 4,
        "server_static": server_audit.get("passed") is True,
        "server_accepts_only_execute4": (
            'n_action_steps.strip() != "4"' in rewritten
            and "N_ACTION_STEPS must be 4" in rewritten
            and "        n = 4\n" in rewritten
        ),
        "server_logs_execute4": (
            "execute=4 " in rewritten and "n_action_steps=4 action_ema" in rewritten
        ),
        "no_slow_model_in_staging": not (
            STAGING / "model_weights/model.safetensors"
        ).exists(),
        "all_unchanged_entries_hardlinked": all(
            row["same_inode"]
            for row in linked
            if row["path"] not in {"policy_server.py", "model_weights/config.json"}
        ),
    }
    report = {
        "schema_version": "r6_execute4_prepare_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_unix": time.time(),
        "source_staging": str(SOURCE_STAGING),
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": source_zip_sha,
        "fast_sha256": fast_sha,
        "plan": str(PLAN),
        "plan_sha256": plan_sha,
        "comparison": str(COMPARISON),
        "comparison_sha256": comparison_sha,
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
        key: {"data": value.tobytes(), "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in obs.items()
    }
    return msgpack.packb(encoded, use_bin_type=True)


def _decode_action(content: bytes) -> np.ndarray:
    unpacked = msgpack.unpackb(content, raw=False)
    data = unpacked["data"]
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("action data is not bytes")
    action = np.frombuffer(data, dtype=np.float32).copy()
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"invalid action: shape={action.shape} finite={np.isfinite(action).all()}")
    return action


def smoke(gpu: str = "0", health_timeout: int = 1200, attempt: int = 2) -> dict[str, Any]:
    prepare_report = json.loads((REPORT_ROOT / "prepare.json").read_text(encoding="utf-8"))
    if prepare_report.get("status") != "passed":
        raise RuntimeError("prepare gate not passed")
    suffix = "" if attempt == 1 else f"_attempt_{attempt:04d}"
    smoke_path = REPORT_ROOT / f"runtime_smoke{suffix}.json"
    if smoke_path.exists():
        raise FileExistsError(smoke_path)

    port = _free_port()
    log_path = REPORT_ROOT / f"runtime_smoke{suffix}.log"
    env = os.environ.copy()
    env.update(
        {
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
        }
    )
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
                json={"instruction": "put the bowl on the stove", "seed": 20260916},
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
        "log_n_action_steps4": "n_action_steps=4 action_ema=0.0 rtc=False" in log_text,
        "no_error": error is None,
        "controlled_close": proc is not None and proc.returncode in (-15, 0),
    }
    report = {
        "schema_version": "r6_execute4_runtime_smoke_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_unix": time.time(),
        "attempt": attempt,
        "previous_attempt": None if attempt == 1 else str(REPORT_ROOT / "runtime_smoke.json"),
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
    names_exact = set(source) == set(current)
    changed: list[str] = []
    unchanged_mismatch: list[str] = []
    for name in sorted(set(source) & set(current)):
        left, right = source[name], current[name]
        same = left.file_size == right.file_size and left.CRC == right.CRC
        if not same:
            changed.append(name)
            if name not in ALLOWED_CHANGED_ENTRIES:
                unchanged_mismatch.append(name)
    checks = {
        "entry_names_exact": names_exact,
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


def package() -> dict[str, Any]:
    prepare_report = json.loads((REPORT_ROOT / "prepare.json").read_text(encoding="utf-8"))
    passed_smokes: list[tuple[Path, dict[str, Any]]] = []
    for smoke_path in sorted(REPORT_ROOT.glob("runtime_smoke*.json")):
        payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        if payload.get("status") == "passed":
            passed_smokes.append((smoke_path, payload))
    if not passed_smokes:
        raise RuntimeError("no passed immutable runtime smoke attempt")
    smoke_path, smoke_report = passed_smokes[-1]
    if prepare_report.get("status") != "passed" or smoke_report.get("status") != "passed":
        raise RuntimeError("prepare and runtime smoke must pass before packaging")

    manifest = build_manifest(smoke_report, smoke_path)
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
        config_in_zip = json.loads(archive.read("model_weights/config.json"))
        manifest_in_zip = json.loads(archive.read("submission_manifest.json"))
        server_in_zip = archive.read("policy_server.py").decode("utf-8")
    checks = {
        "package_validation": package_validation.get("status") == "passed",
        "zip_delta": delta.get("status") == "passed",
        "validate_submission_static": static_report["status"] == "passed",
        "fast_size_exact": fast_info.file_size
        == (STAGING / "model_weights/model_fast.safetensors").stat().st_size,
        "config_execute4": config_in_zip.get("n_action_steps") == 4,
        "manifest_execute4": manifest_in_zip.get("action_protocol", {}).get(
            "n_action_steps"
        )
        == 4,
        "server_execute4": (
            "execute=4 " in server_in_zip
            and "n_action_steps=4 action_ema=0.0 rtc=False" in server_in_zip
        ),
        "no_partial": not OUTPUT.with_name(OUTPUT.name + ".partial").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "delta": delta, "static": static_report})

    sidecars = write_package_sidecars(OUTPUT, REPORT_ROOT)
    release = {
        "schema_version": "r6_execute4_official_trial_release_v1",
        "status": "passed",
        "created_unix": time.time(),
        "candidate": "r6_dual_l2_fresh500_r1_execute4_user_override",
        "zip": str(OUTPUT),
        "zip_sha256": package_validation["sha256"],
        "zip_bytes": package_validation["bytes"],
        "zip_entry_count": package_validation["entry_count"],
        "fast_model_sha256": EXPECTED_FAST_SHA,
        "action_protocol": {
            "chunk_size": 10,
            "n_action_steps": 4,
            "action_ema": 0.0,
            "rtc": False,
        },
        "screen_evidence": {
            "execute4_c20": 17,
            "execute8_c20": 15,
            "delta": 2,
            "paired_net_wins": 2,
            "preregistered_gate_passed": False,
            "user_risk_override": True,
        },
        "checks": checks,
        "package_validation": package_validation,
        "zip_delta_audit": delta,
        "static_validation": static_report,
        "runtime_smoke": smoke_report,
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
    parser.add_argument("--attempt", type=int, default=2)
    args = parser.parse_args()

    if args.command in ("prepare", "all"):
        prepare()
    if args.command in ("smoke", "all"):
        smoke(args.gpu, args.health_timeout, args.attempt)
    if args.command in ("package", "all"):
        result = package()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "zip": result["zip"],
                    "sha256": result["zip_sha256"],
                    "bytes": result["zip_bytes"],
                    "auto_upload": False,
                    "auto_submit": False,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
