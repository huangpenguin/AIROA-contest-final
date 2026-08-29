#!/usr/bin/env python3
"""Fail-closed audit and packager for the F2/M1@500 FAST-only candidate.

The exporter owns model conversion.  This file owns the independent checks that
must be true before a human can consider the resulting zip: policy-server
protocol, safetensors headers, package contents, and the 20GB validator gate.
It never reads or writes an Orbax checkpoint and it never changes the source
submission zip.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import socket
import stat
import subprocess
import sys
import time
import types
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


MAX_PACKAGE_BYTES = 20_000_000_000
REQUIRED_PACKAGE_FILES = {
    "policy_server.py",
    "requirements.txt",
    "model_weights/config.json",
    "model_weights/model_fast.safetensors",
    "model_weights/pi05_fast_meta.json",
    "model_weights/policy_preprocessor.json",
    "model_weights/policy_postprocessor.json",
    "model_weights/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "model_weights/policy_preprocessor_step_2_normalizer_processor.safetensors",
    "submission_manifest.json",
}
VISION_SUFFIXES = (
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_zip_name(name: str) -> None:
    if "\\" in name or name.startswith("/"):
        raise ValueError(f"unsafe package member path: {name!r}")
    parts = PurePosixPath(name).parts
    if ".." in parts or "." in parts:
        raise ValueError(f"unsafe package member path: {name!r}")


def _header(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
        dtypes = {key: str(handle.get_slice(key).get_dtype()) for key in keys}
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "key_count": len(keys),
        "keys": keys,
        "shapes": shapes,
        "dtype_tensor_count": {
            dtype: sum(value == dtype for value in dtypes.values())
            for dtype in sorted(set(dtypes.values()))
        },
    }


def _policy_static_audit(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    required_literals = ["/health", "/reset", "/act", "FAST_REQUIRED_AND_VERIFIED"]
    forbidden_literals = [
        "pi05 SLOW",
        "PI05Policy.from_pretrained",
        "model_fast.safetensors for FAST",
    ]
    missing_literals = [item for item in required_literals if item not in source]
    forbidden_hits = [item for item in forbidden_literals if item in source]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    strict_false_fast = []
    for node in calls:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict":
            for keyword in node.keywords:
                if keyword.arg == "strict" and isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                    strict_false_fast.append(node.lineno)
    checks = {
        "syntax": True,
        "required_literals": {item: item in source for item in required_literals},
        "missing_literals": missing_literals,
        "forbidden_literals": forbidden_hits,
        "strict_false_load_state_dict_lines": strict_false_fast,
        "fast_only_model_load": "PI05Policy.from_pretrained" not in source,
        "has_reset_seed": "seed: int | None = None" in source and "manual_seed" in source,
        "has_exact_observation_gate": "fixed-input parity observation keys mismatch" in source,
        "has_finite_action_gate": "policy action must be finite" in source,
    }
    checks["passed"] = bool(
        not missing_literals
        and not forbidden_hits
        and not strict_false_fast
        and checks["fast_only_model_load"]
        and checks["has_reset_seed"]
        and checks["has_exact_observation_gate"]
        and checks["has_finite_action_gate"]
    )
    return checks


def audit_staging(staging: Path, report_root: Path) -> dict[str, Any]:
    model_dir = staging / "model_weights"
    source_inventory = report_root / "source_inventory.json"
    export_report = report_root / "export_report.json"
    mapping_report = report_root / "key_mapping.json"
    required = [
        staging / "policy_server.py",
        model_dir / "config.json",
        model_dir / "model.safetensors",
        model_dir / "model_fast.safetensors",
        model_dir / "pi05_fast_meta.json",
        source_inventory,
        export_report,
        mapping_report,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("staging audit missing: " + ", ".join(missing))

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    inventory = json.loads(source_inventory.read_text(encoding="utf-8"))
    export = json.loads(export_report.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_report.read_text(encoding="utf-8"))
    source_header = _header(model_dir / "model.safetensors")
    fast_header = _header(model_dir / "model_fast.safetensors")
    server = _policy_static_audit(staging / "policy_server.py")
    fast_meta = json.loads((model_dir / "pi05_fast_meta.json").read_text(encoding="utf-8"))

    source_set = set(source_header["keys"])
    fast_set = set(fast_header["keys"])
    target_vision = {
        f"model.{suffix}": fast_header["shapes"].get(f"model.{suffix}")
        for suffix in VISION_SUFFIXES
    }
    vision = {
        suffix: {
            "source_key": suffix,
            "source_present": suffix in source_set,
            "source_shape": source_header["shapes"].get(suffix),
            "target_key": f"model.{suffix}",
            "target_present": f"model.{suffix}" in fast_set,
            "target_shape": target_vision[f"model.{suffix}"],
        }
        for suffix in VISION_SUFFIXES
    }
    vision_passed = all(
        value["source_present"]
        and value["target_present"]
        and value["source_shape"] == value["target_shape"]
        for value in vision.values()
    )
    checks = {
        "config_protocol": {
            "type_pi05": config.get("type") == "pi05",
            "chunk_size_10": int(config.get("chunk_size", -1)) == 10,
            "n_action_steps_8": int(config.get("n_action_steps", -1)) == 8,
            "compile_model_false": config.get("compile_model") is False,
            "gradient_checkpointing_false": config.get("gradient_checkpointing") is False,
        },
        "source_identity": {
            "source_is_current_submission_model": export.get("source_is_current_submission_model") is True,
            "historical_fast_asset_not_used": export.get("historical_fast_asset_used") is False,
            "training_not_performed": export.get("training_performed") is False,
            "inventory_source_model_sha256": inventory.get("source_model_sha256") == source_header["sha256"],
        },
        "key_gate": mapping.get("passed") is True and export.get("key_gate", {}).get("passed") is True,
        "strict_reload": export.get("reload", {}).get("strict_key_gate") is True,
        "vision_gate": vision_passed and export.get("vision_gate", {}).get("passed") is True,
        "server_static": server["passed"],
        "fast_header": {
            "has_model_fast": fast_header["key_count"] > 0,
            "fast_differs_from_source": fast_header["sha256"] != source_header["sha256"],
            "key_count_813": fast_header["key_count"] == 813,
        },
        "fast_metadata": {
            "source_hash_matches": fast_meta.get("source_model_sha256") == source_header["sha256"],
            "fast_hash_matches": fast_meta.get("model_fast_sha256") == fast_header["sha256"],
            "historical_fast_asset_not_used": fast_meta.get("historical_fast_asset_used", False) is False,
        },
    }
    checks["passed"] = all(
        value if isinstance(value, bool) else all(value.values())
        for value in checks.values()
    )
    result = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "created_unix": time.time(),
        "staging": str(staging.resolve()),
        "source_inventory": str(source_inventory.resolve()),
        "source_model": source_header,
        "fast_model": fast_header,
        "vision": vision,
        "checks": checks,
        "server_static": server,
    }
    write_json(report_root / "server_static_audit.json", server)
    write_json(report_root / "fast_submission_audit.json", result)
    return result


def _package_entries(staging: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    explicit = [
        "policy_server.py",
        "requirements.txt",
        "model_weights/config.json",
        "model_weights/model_fast.safetensors",
        "model_weights/pi05_fast_meta.json",
        "model_weights/policy_preprocessor.json",
        "model_weights/policy_postprocessor.json",
        "model_weights/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "model_weights/policy_preprocessor_step_2_normalizer_processor.safetensors",
    ]
    for relative in explicit:
        path = staging / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append((relative, path))
    for path in sorted((staging / "vendor").rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(staging).as_posix()
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        entries.append((relative, path))
    return entries


def build_package(staging: Path, report_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite package: {output}")
    audit = json.loads((report_root / "fast_submission_audit.json").read_text(encoding="utf-8"))
    export = json.loads((report_root / "export_report.json").read_text(encoding="utf-8"))
    inventory = json.loads((report_root / "source_inventory.json").read_text(encoding="utf-8"))
    fast_path = staging / "model_weights/model_fast.safetensors"
    converter = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "submission_status": "NOT_SUBMITTED",
        "training_steps": 500,
        "scope": "F2",
        "normalization": "M1",
        "source": {
            "submission_zip": inventory.get("source_submission_zip"),
            "submission_zip_sha256": inventory.get("source_submission_zip_sha256"),
            "model_sha256": inventory.get("source_model_sha256"),
            "model_bytes": inventory.get("source_model_bytes"),
            "config_sha256": inventory.get("source_config_sha256"),
            "preprocessor_sha256": inventory.get("source_preprocessor_sha256"),
            "postprocessor_sha256": inventory.get("source_postprocessor_sha256"),
        },
        "converter": {
            "path": str(converter),
            "sha256": sha256_file(converter),
        },
        "fast_model": {
            "sha256": sha256_file(fast_path),
            "bytes": fast_path.stat().st_size,
            "key_count": export["fast_model"]["key_count"],
            "dtype_tensor_count": export["fast_model"]["dtype_tensor_count"],
            "dtype_bytes": export["fast_model"]["dtype_bytes"],
        },
        "policy_server": {
            "sha256": sha256_file(staging / "policy_server.py"),
            "fast_required_and_verified": True,
        },
        "action_protocol": {
            "policy_type": "pi05",
            "model_chunk_size": 10,
            "n_action_steps": 8,
            "action_ema": 0.0,
            "rtc": False,
            "compile_model": False,
            "gradient_checkpointing": False,
            "offline": True,
        },
        "validation_reports": {
            name: {
                "path": str((report_root / name).resolve()),
                "sha256": sha256_file(report_root / name),
            }
            for name in (
                "source_inventory.json",
                "key_mapping.json",
                "export_report.json",
                "fast_submission_audit.json",
                "fixed_input_parity.json",
                "server_latency.json",
                "track1_calibration/summary.json",
                "official_docker_e2e.json",
            )
        },
    }
    manifest_path = staging / "submission_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("submission_status") != "NOT_SUBMITTED":
            raise FileExistsError(f"refusing to replace a non-recovery manifest: {manifest_path}")
        # This manifest was created by this recovery packager after a previous
        # ZIP64 write attempt; refresh its converter/report hashes, never an
        # arbitrary pre-existing staging file.
        if existing_manifest != manifest:
            write_json(manifest_path, manifest)
    else:
        write_json(manifest_path, manifest)

    entries = _package_entries(staging) + [("submission_manifest.json", manifest_path)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for relative, path in entries:
                info = zipfile.ZipInfo(relative)
                mode = path.stat().st_mode
                if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                    raise ValueError(f"package entry is not a regular file: {path}")
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_STORED if path.name.endswith(".safetensors") else zipfile.ZIP_DEFLATED
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        destination.write(block)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "entries": [relative for relative, _ in entries],
        "manifest_sha256": sha256_file(manifest_path),
    }


def audit_package(package: Path, report_root: Path) -> dict[str, Any]:
    if not package.is_file():
        raise FileNotFoundError(package)
    size = package.stat().st_size
    errors: list[str] = []
    with zipfile.ZipFile(package, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        for info in infos:
            try:
                _safe_zip_name(info.filename)
            except ValueError as exc:
                errors.append(str(exc))
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode) or (mode and not stat.S_ISREG(mode) and not info.is_dir()):
                errors.append(f"non-regular package member: {info.filename}")
            if info.filename.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                errors.append(f"nested archive: {info.filename}")
        if len(names) != len(set(names)):
            errors.append("duplicate package member")
        missing = sorted(REQUIRED_PACKAGE_FILES - set(names))
        if missing:
            errors.append("missing package members: " + ", ".join(missing))
        if "model_weights/model.safetensors" in names:
            errors.append("slow model.safetensors is present")
        if names.count("model_weights/model_fast.safetensors") != 1:
            errors.append("model_fast.safetensors is not unique")
        if archive.testzip() is not None:
            errors.append("zip CRC test failed")
    result = {
        "schema_version": 1,
        "status": "passed" if not errors and size < MAX_PACKAGE_BYTES else "failed",
        "package": str(package.resolve()),
        "sha256": sha256_file(package),
        "bytes": size,
        "size_limit_bytes": MAX_PACKAGE_BYTES,
        "entry_count": len(names),
        "errors": errors,
        "root_level": "policy_server.py" in names,
        "contains_model_safetensors": "model_weights/model.safetensors" in names,
        "contains_model_fast": names.count("model_weights/model_fast.safetensors") == 1,
        "zip_test": not errors or "zip CRC test failed" not in errors,
    }
    write_json(report_root / "package_validation.json", result)
    return result


def _axisangle(quat: Any) -> Any:
    import numpy as np

    value = np.asarray(quat, dtype=np.float32).reshape(4)
    scalar = float(np.clip(value[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(0.0, 1.0 - scalar * scalar)))
    if denominator < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (value[:3] * (2.0 * np.arccos(scalar)) / denominator).astype(np.float32)


def collect_fixtures(manifest_path: Path, fixture_dir: Path, count: int = 20) -> dict[str, Any]:
    """Freeze exactly the post-bootstrap observations from the locked T1 panel."""

    import numpy as np
    import robosuite.macros as robosuite_macros
    # The official image has no writable source locator for numba's cache.
    # Disable only that cache for this fixture collector; model/server code is
    # unaffected and the submission package is never changed.
    robosuite_macros.CACHE_NUMBA = False
    from pipeline.config import EvalConfig
    from pipeline.environment import EnvironmentManager
    from tools.axis_pi05.eval_manifest import load_locked_panel

    rows = [row for row in load_locked_panel(manifest_path) if row.get("suite") == "libero_t1"]
    if len(rows) < count:
        raise ValueError(f"locked Track 1 panel has only {len(rows)} rows; need {count}")
    rows = rows[:count]
    fixture_dir.mkdir(parents=True, exist_ok=True)
    manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128))
    task_cache: dict[tuple[str, int], Any] = {}
    cases: list[dict[str, Any]] = []
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_joint_pos",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    for index, row in enumerate(rows):
        cache_key = (str(row["suite"]), int(row["task_id"]))
        if cache_key not in task_cache:
            task_cache[cache_key] = manager.get_task_infos(row["suite"])[int(row["task_id"])]
        task = task_cache[cache_key]
        env = manager.create_env(task, env_seed=int(row["env_seed"]))
        try:
            env.reset()
            env.sim.set_state_from_flattened(manager.get_init_state(task, int(row["init_state_index"])))
            env.sim.forward()
            dummy = np.zeros(env.robots[0].action_dim, dtype=np.float32)
            obs = None
            for _ in range(10):
                obs, _, _, _ = env.step(dummy)
            if obs is None:
                raise RuntimeError("environment produced no observation after bootstrap")
            frozen = {key: np.ascontiguousarray(np.asarray(obs[key]).copy()) for key in required}
        finally:
            env.close()
        if frozen["agentview_image"].dtype != np.uint8 or frozen["agentview_image"].shape != (128, 128, 3):
            raise ValueError(f"unexpected agentview fixture shape/dtype: {frozen['agentview_image'].shape} {frozen['agentview_image'].dtype}")
        if frozen["robot0_eye_in_hand_image"].dtype != np.uint8 or frozen["robot0_eye_in_hand_image"].shape != (128, 128, 3):
            raise ValueError("unexpected wrist fixture shape/dtype")
        fixture_name = f"fixture_{index:02d}.npz"
        fixture_path = fixture_dir / fixture_name
        np.savez_compressed(fixture_path, **frozen)
        cases.append(
            {
                "index": index,
                "case_id": row["case_id"],
                "row": row,
                "file": fixture_name,
                "sha256": sha256_file(fixture_path),
                "image_sha256": {
                    "agentview_image": hashlib.sha256(frozen["agentview_image"].tobytes()).hexdigest(),
                    "robot0_eye_in_hand_image": hashlib.sha256(frozen["robot0_eye_in_hand_image"].tobytes()).hexdigest(),
                },
                "state8": np.concatenate(
                    [
                        frozen["robot0_eef_pos"].astype(np.float32).reshape(3),
                        _axisangle(frozen["robot0_eef_quat"]),
                        frozen["robot0_gripper_qpos"].astype(np.float32).reshape(2),
                    ]
                ).tolist(),
            }
        )
    result = {
        "schema_version": 1,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "fixture_count": len(cases),
        "bootstrap_dummy_steps": 10,
        "required_keys": list(required),
        "cases": cases,
    }
    write_json(fixture_dir / "fixtures.json", result)
    return result


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_fast_policy(staging: Path) -> Any:
    """Load MyPolicy without installing FastAPI; parity uses direct calls."""

    host_vendor = Path("/tmp/axis_host_vendor")
    if host_vendor.is_dir() and str(host_vendor) not in sys.path:
        sys.path.insert(0, str(host_vendor))
    # The host audit venv has the matching torchvision for its torch build.
    # Preload it before policy_server adds the package's official-image vendor
    # directory (which intentionally contains torchvision 0.26 for torch 2.11).
    # This keeps host-side direct parity independent of that image-only wheel.
    import torchvision  # noqa: F401
    import sentencepiece  # noqa: F401

    class _App:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def get(self, *_args: Any, **_kwargs: Any):
            return lambda function: function

        def post(self, *_args: Any, **_kwargs: Any):
            return lambda function: function

    class _HTTPException(RuntimeError):
        def __init__(self, status_code: int, detail: str = "") -> None:
            super().__init__(detail)
            self.status_code = status_code

    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _App
    fastapi.HTTPException = _HTTPException
    fastapi.Request = object
    fastapi.Response = object
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.run = lambda *_args, **_kwargs: None
    sys.modules["fastapi"] = fastapi
    sys.modules["uvicorn"] = uvicorn
    spec = importlib.util.spec_from_file_location("axis_fast_staged_policy", staging / "policy_server.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load staged policy_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyPolicy()


def _load_fixture(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def run_fixed_parity(
    staging: Path,
    fixture_dir: Path,
    report_root: Path,
    checkpoint: Path,
    gpu: str = "1",
) -> dict[str, Any]:
    """Compare the staged policy with the authoritative Orbax HTTP adapter."""

    import numpy as np

    # When this file is executed by path, Python puts tools/axis_pi05 on
    # sys.path rather than the repository root.  Make the repository-local
    # pipeline client available without relying on the caller's PYTHONPATH.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pipeline.remote_policy import RemotePolicyClient

    fixture_manifest = json.loads((fixture_dir / "fixtures.json").read_text(encoding="utf-8"))
    cases = fixture_manifest.get("cases", [])
    if len(cases) != 20:
        raise ValueError(f"fixed-input parity requires 20 fixtures, got {len(cases)}")
    seeds = (2000, 2001, 2002)
    source_port = _free_port()
    source_log_path = report_root / "source_parity_server.log"
    source_log = source_log_path.open("w", encoding="utf-8")
    source_env = os.environ.copy()
    source_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "AXIS_PI05_SCOPE": "F2",
            "AXIS_PI05_MIX": "M1",
            "AXIS_PI05_ASSETS_ROOT": str(Path(__file__).resolve().parents[2] / "data/manifests_v2"),
            "AXIS_PI05_NORM_STATS": str(Path(__file__).resolve().parents[2] / "data/manifests_v2/M1/norm_stats.json"),
        }
    )
    source_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.axis_pi05.checkpoint_http_server",
            "--checkpoint",
            str(checkpoint),
            "--port",
            str(source_port),
            "--config",
            "axis_pi05_libero_mix",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=source_env,
        stdout=source_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    source_client = RemotePolicyClient(f"http://127.0.0.1:{source_port}", timeout_sec=30.0)
    source_rows: dict[tuple[int, int], dict[str, Any]] = {}
    source_start = time.time()
    try:
        source_client.wait_for_server(timeout=float(os.environ.get("AXIS_PI05_SERVER_STARTUP_TIMEOUT", "600")))
        source_health_seconds = time.time() - source_start
        for case in cases:
            fixture = _load_fixture(fixture_dir / case["file"])
            instruction = str(case["row"]["language"])
            for seed in seeds:
                source_client.reset(instruction=instruction, seed=seed)
                t0 = time.perf_counter()
                action = np.asarray(source_client.get_action(fixture), dtype=np.float32)
                elapsed = time.perf_counter() - t0
                source_rows[(int(case["index"]), seed)] = {
                    "action": action.tolist(),
                    "seconds": elapsed,
                    "finite": bool(action.shape == (7,) and np.isfinite(action).all()),
                }
    finally:
        if source_process.poll() is None:
            source_process.terminate()
            try:
                source_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                source_process.kill()
                source_process.wait(timeout=10)
        source_log.close()

    fast_env = os.environ.copy()
    fast_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "POLICY_TYPE": "pi05",
            "PI05_FAST": "1",
            "PI05_DTYPE": "float16" if gpu != "cpu" else "float32",
            "N_ACTION_STEPS": "8",
            "ACTION_EMA": "0",
            "RTC_ENABLED": "0",
            "RTC_INFERENCE_DELAY": "0",
            "PI05_FIDELITY_DUMP": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    old_env = os.environ.copy()
    os.environ.update(fast_env)
    fast_policy = None
    fast_rows: dict[tuple[int, int], dict[str, Any]] = {}
    fast_start = time.time()
    try:
        fast_policy = _load_fast_policy(staging)
        fast_health_seconds = time.time() - fast_start
        for case in cases:
            fixture = _load_fixture(fixture_dir / case["file"])
            instruction = str(case["row"]["language"])
            for seed in seeds:
                fast_policy.reset(instruction=instruction, seed=seed)
                t0 = time.perf_counter()
                action = np.asarray(fast_policy.get_action(fixture), dtype=np.float32)
                elapsed = time.perf_counter() - t0
                trace = dict(getattr(fast_policy, "_last_trace", {}))
                fast_rows[(int(case["index"]), seed)] = {
                    "action": action.tolist(),
                    "seconds": elapsed,
                    "finite": bool(action.shape == (7,) and np.isfinite(action).all()),
                    "trace": trace,
                }
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        del fast_policy
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    records: list[dict[str, Any]] = []
    errors: list[float] = []
    cosines: list[float] = []
    gripper_matches: list[bool] = []
    finite_values: list[bool] = []
    fast_latencies: list[float] = []
    source_latencies: list[float] = []
    for case in cases:
        for seed in seeds:
            key = (int(case["index"]), seed)
            source = source_rows[key]
            fast = fast_rows[key]
            source_action = np.asarray(source["action"], dtype=np.float32)
            fast_action = np.asarray(fast["action"], dtype=np.float32)
            delta = np.abs(source_action - fast_action)
            errors.extend(delta.reshape(-1).tolist())
            finite_values.extend([source["finite"], fast["finite"]])
            if np.linalg.norm(source_action) > 0 and np.linalg.norm(fast_action) > 0:
                cosines.append(float(np.dot(source_action, fast_action) / (np.linalg.norm(source_action) * np.linalg.norm(fast_action))))
            gripper_matches.append(bool(np.sign(source_action[6]) == np.sign(fast_action[6])))
            fast_latencies.append(float(fast["seconds"]))
            source_latencies.append(float(source["seconds"]))
            trace = fast.get("trace", {})
            raw_chunk = trace.get("raw_internal_chunk")
            denorm_chunk = trace.get("denormalized_chunk")
            if raw_chunk is not None and np.asarray(raw_chunk).shape != (10, 7):
                raise ValueError(f"FAST raw chunk shape is not [10,7]: {np.asarray(raw_chunk).shape}")
            if denorm_chunk is not None and np.asarray(denorm_chunk).shape != (10, 7):
                raise ValueError(f"FAST denormalized chunk shape is not [10,7]: {np.asarray(denorm_chunk).shape}")
            records.append(
                {
                    "fixture_index": int(case["index"]),
                    "case_id": case["case_id"],
                    "seed": seed,
                    "image_sha256": case["image_sha256"],
                    "state8": case["state8"],
                    "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                    "source": source,
                    "fast": fast,
                }
            )
    errors_array = np.asarray(errors, dtype=np.float64)
    finite_rate = float(sum(finite_values) / len(finite_values))
    metrics = {
        "finite_rate": finite_rate,
        "median_abs_error": float(np.median(errors_array)),
        "p95_abs_error": float(np.percentile(errors_array, 95)),
        "max_abs_error": float(np.max(errors_array)),
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "gripper_sign_agreement": float(sum(gripper_matches) / len(gripper_matches)),
        "source_startup_to_health_seconds": source_health_seconds,
        "fast_policy_direct_init_seconds": fast_health_seconds,
        "source_act_seconds": {
            "p50": float(np.percentile(source_latencies, 50)),
            "p95": float(np.percentile(source_latencies, 95)),
            "max": float(np.max(source_latencies)),
        },
        "fast_act_seconds": {
            "p50": float(np.percentile(fast_latencies, 50)),
            "p95": float(np.percentile(fast_latencies, 95)),
            "max": float(np.max(fast_latencies)),
        },
    }
    result = {
        "schema_version": 1,
        "status": "passed" if (
            metrics["finite_rate"] == 1.0
            and metrics["median_abs_error"] <= 0.03
            and metrics["p95_abs_error"] <= 0.10
            and metrics["gripper_sign_agreement"] >= 0.95
        ) else "failed",
        "authoritative_source": str(checkpoint.resolve()),
        "fixture_manifest": str((fixture_dir / "fixtures.json").resolve()),
        "fixture_count": len(cases),
        "policy_seeds": list(seeds),
        "fast_chunk_shape": [10, 7],
        "executed_action_steps": 8,
        "metrics": metrics,
        "records": records,
    }
    write_json(report_root / "fixed_input_parity.json", result)
    return result


def run_server_latency(
    staging: Path,
    report_root: Path,
    fixture_dir: Path,
    gpu: str = "1",
) -> dict[str, Any]:
    """Exercise the real staged FastAPI server and record fail-closed latency."""

    import numpy as np
    import msgpack
    from urllib import request as urllib_request

    fixture_manifest = json.loads((fixture_dir / "fixtures.json").read_text(encoding="utf-8"))
    cases = fixture_manifest.get("cases", [])
    if not cases:
        raise ValueError("server latency requires at least one frozen fixture")
    fixture = _load_fixture(fixture_dir / cases[0]["file"])
    instruction = str(cases[0]["row"]["language"])
    port = _free_port()
    log_path = report_root / "server_latency_process.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    container_gpu = "0" if gpu != "cpu" else ""
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": container_gpu,
            "POLICY_TYPE": "pi05",
            "PI05_FAST": "1",
            "PI05_DTYPE": "float16" if gpu != "cpu" else "float32",
            "N_ACTION_STEPS": "8",
            "ACTION_EMA": "0",
            "RTC_ENABLED": "0",
            "RTC_INFERENCE_DELAY": "0",
            "PI05_FIDELITY_DUMP": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "TORCHINDUCTOR_DISABLE": "1",
        }
    )
    process = subprocess.Popen(
        [sys.executable, "policy_server.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=staging,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    health_seconds = None
    health_error = None
    try:
        deadline = time.time() + float(os.environ.get("AXIS_PI05_SERVER_STARTUP_TIMEOUT", "900"))
        while time.time() < deadline:
            try:
                with urllib_request.urlopen(base_url + "/health", timeout=3.0) as response:
                    if response.status == 200:
                        health_seconds = time.perf_counter() - started
                        break
            except Exception as exc:  # startup is intentionally polled
                health_error = f"{type(exc).__name__}: {exc}"
            if process.poll() is not None:
                health_error = f"server exited with code {process.returncode}"
                break
            time.sleep(1.0)

        reset_ok = False
        reset_error = None
        action_latencies: list[float] = []
        action_errors: list[str] = []
        if health_seconds is not None:
            try:
                body = json.dumps({"instruction": instruction, "seed": 2000}).encode("utf-8")
                reset_request = urllib_request.Request(
                    base_url + "/reset",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib_request.urlopen(reset_request, timeout=15.0) as response:
                    reset_ok = response.status == 200
            except Exception as exc:
                reset_error = f"{type(exc).__name__}: {exc}"

            serialized = {
                key: {
                    "data": np.asarray(value).tobytes(),
                    "shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype),
                }
                for key, value in fixture.items()
            }
            payload = msgpack.packb(serialized, use_bin_type=True)
            for _ in range(24):
                try:
                    action_request = urllib_request.Request(
                        base_url + "/act",
                        data=payload,
                        headers={"Content-Type": "application/x-msgpack"},
                        method="POST",
                    )
                    t0 = time.perf_counter()
                    with urllib_request.urlopen(action_request, timeout=15.0) as response:
                        response.read()
                    action_latencies.append(time.perf_counter() - t0)
                except Exception as exc:
                    action_errors.append(f"{type(exc).__name__}: {exc}")
                    break
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_handle.close()

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "server": "staged policy_server.py via FastAPI/uvicorn",
        "gpu": str(gpu),
        "fixture": cases[0]["file"],
        "fixture_manifest": str((fixture_dir / "fixtures.json").resolve()),
        "process_start_to_health_seconds": health_seconds,
        "health_error": health_error,
        "reset_ok": reset_ok,
        "reset_error": reset_error,
        "act_count": len(action_latencies),
        "act_errors": action_errors,
        "act_seconds": {},
        "process_log": str(log_path.resolve()),
        "gates": {
            "health": health_seconds is not None,
            "reset": reset_ok,
            "all_act_requests_completed": len(action_latencies) == 24,
            "each_act_under_10s": bool(action_latencies) and max(action_latencies) < 10.0,
        },
    }
    if action_latencies:
        result["act_seconds"] = {
            "p50": float(np.percentile(action_latencies, 50)),
            "p95": float(np.percentile(action_latencies, 95)),
            "max": float(np.max(action_latencies)),
        }
    result["status"] = "passed" if all(result["gates"].values()) else "failed"
    write_json(report_root / "server_latency.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--build-package", action="store_true")
    parser.add_argument("--collect-fixtures", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--run-parity", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--parity-gpu", default="1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    if args.collect_fixtures:
        if args.manifest is None or args.fixtures is None:
            raise SystemExit("--manifest and --fixtures are required with --collect-fixtures")
        print(json.dumps(collect_fixtures(args.manifest, args.fixtures), indent=2, sort_keys=True))
        return
    audit = audit_staging(args.staging, report_root)
    if not audit["checks"]["passed"]:
        raise SystemExit("FAST staging audit failed")
    if args.run_parity:
        if args.checkpoint is None or args.fixtures is None:
            raise SystemExit("--checkpoint and --fixtures are required with --run-parity")
        parity = run_fixed_parity(args.staging, args.fixtures, report_root, args.checkpoint, args.parity_gpu)
        print(json.dumps({"audit": audit["status"], "parity": parity["status"], "metrics": parity["metrics"]}, indent=2, sort_keys=True))
        if parity["status"] != "passed":
            raise SystemExit("fixed-input parity failed")
        return
    package = args.package
    if args.build_package:
        if package is None:
            raise SystemExit("--package is required with --build-package")
        build = build_package(args.staging, report_root, package)
        print(json.dumps(build, indent=2, sort_keys=True))
    if package is not None and package.is_file():
        result = audit_package(package, report_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "passed":
            raise SystemExit("package audit failed")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
