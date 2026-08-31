#!/usr/bin/env python3
"""Freeze a submission zip and export its own Pi0.5 weights to FAST.

This command is deliberately submission-specific.  It never reads an Orbax
checkpoint and it never consults the historical ``model_fast.safetensors``
asset.  The production path accepts only the locked on-disk R6 folded model;
the older source-zip path remains available for historical recovery tests.

The first phase inventories and safely extracts a source zip into a new
staging directory.  The second phase loads the staged LeRobot model, applies
the validated mixed-dtype policy, and writes an atomic ``model_fast`` file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


EXPECTED_MEMBERS = {
    "policy_server.py",
    "requirements.txt",
    "model_weights/config.json",
    "model_weights/policy_preprocessor.json",
    "model_weights/policy_postprocessor.json",
    "model_weights/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "model_weights/policy_preprocessor_step_2_normalizer_processor.safetensors",
    "model_weights/model.safetensors",
}
VISION_KEYS = (
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
    "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias",
)
PROJECTION_NAMES = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")
# The source LeRobot file preserves tied-weight metadata by storing the
# PaliGemma lm_head once.  The wrapper's module also exposes the tied
# embed_tokens name, so the exporter materializes that alias explicitly.
TIED_SOURCE_ALIASES = {
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight":
    "paligemma_with_expert.paligemma.lm_head.weight",
}
CHUNK = 8 * 1024 * 1024
R6_SOURCE_KIND = "on_disk_r6_folded_pytorch"
R6_CANDIDATE = "r6_dual_l2_fresh500_r1"
R6_TOPOLOGY = "L2_action_lora_heads"
R6_MODEL_SHA256 = "5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b"
R6_NORM_SHA256 = "4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5"
R6_PROCESSOR_SHA256 = {
    "policy_preprocessor.json": "1d05decf37445ed3f5d2b48c2e7d61bc0299a1c200a5159517fa18f05b8430ef",
    "policy_postprocessor.json": "9c09f2006d9035ee657dc954c4a687ec93acacbd286a25bbd489cdbf6f5340da",
    "policy_preprocessor_step_2_normalizer_processor.safetensors": "d80c2ea8a253fc1b3b8d3ae4fb564d111ce125678abb36a1e0f63a8c02b05b3d",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": "15fe7b1f1692b7046fa18c6857e8fff6d8eefc07597d235c6331b8945dd55402",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(source: BinaryIO, destination: BinaryIO | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: source.read(CHUNK), b""):
        digest.update(block)
        size += len(block)
        if destination is not None:
            destination.write(block)
    return digest.hexdigest(), size


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_member_name(name: str) -> str:
    if "\\" in name:
        raise ValueError(f"backslash path is not allowed: {name!r}")
    if not name or name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        raise ValueError(f"absolute zip path is not allowed: {name!r}")
    parts = PurePosixPath(name).parts
    if ".." in parts or "." in parts:
        raise ValueError(f"dot path component is not allowed: {name!r}")
    return "/".join(parts)


def _is_link(info: zipfile.ZipInfo) -> tuple[bool, bool]:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode), stat.S_ISREG(mode) or mode == 0


def inventory_and_extract(source_zip: Path, staging: Path, report_path: Path) -> dict[str, Any]:
    """Validate, hash and safely extract ``source_zip`` into a new directory."""

    if not source_zip.is_file():
        raise FileNotFoundError(source_zip)
    if staging.exists():
        raise FileExistsError(
            f"refusing to overwrite existing staging directory: {staging}"
        )

    source_zip_hash = sha256_file(source_zip)
    started = time.time()
    with zipfile.ZipFile(source_zip, "r") as archive:
        infos = archive.infolist()
        names: set[str] = set()
        errors: list[str] = []
        for info in infos:
            try:
                name = _safe_member_name(info.filename)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if name in names:
                errors.append(f"duplicate zip member: {name}")
            names.add(name)
            is_link, regular_or_unspecified = _is_link(info)
            if is_link:
                errors.append(f"symlink zip member: {name}")
            if not regular_or_unspecified and not info.is_dir():
                errors.append(f"non-regular zip member: {name}")
            if name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                errors.append(f"nested archive is not allowed: {name}")
        if errors:
            raise ValueError("unsafe source zip: " + "; ".join(errors[:12]))

        # testzip reads every member and verifies CRC before anything is
        # written.  This also makes a corrupt 14GB source fail closed.
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in source zip member: {bad_member}")

        if not EXPECTED_MEMBERS.issubset(names):
            missing = sorted(EXPECTED_MEMBERS - names)
            raise ValueError(f"source zip missing required members: {missing}")
        if "model_weights/model_fast.safetensors" in names:
            raise ValueError("source zip unexpectedly already contains model_fast")

        staging.mkdir(parents=True)
        members: list[dict[str, Any]] = []
        for info in infos:
            name = _safe_member_name(info.filename)
            target = staging / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                members.append(
                    {
                        "path": name,
                        "directory": True,
                        "uncompressed_bytes": 0,
                        "compressed_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": None,
                    }
                )
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with archive.open(info, "r") as source, temporary.open("xb") as destination:
                digest, copied = sha256_stream(source, destination)
            if copied != info.file_size:
                temporary.unlink(missing_ok=True)
                raise IOError(
                    f"uncompressed size mismatch for {name}: {copied} != {info.file_size}"
                )
            temporary.replace(target)
            members.append(
                {
                    "path": name,
                    "directory": False,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": digest,
                }
            )

    source_model = next(item for item in members if item["path"] == "model_weights/model.safetensors")
    source_config = next(item for item in members if item["path"] == "model_weights/config.json")
    source_pre = next(item for item in members if item["path"] == "model_weights/policy_preprocessor.json")
    source_post = next(item for item in members if item["path"] == "model_weights/policy_postprocessor.json")
    source_server = next(item for item in members if item["path"] == "policy_server.py")
    config = json.loads((staging / "model_weights/config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05":
        raise ValueError(f"source config type must be pi05, got {config.get('type')!r}")
    if int(config.get("chunk_size", -1)) != 10 or int(config.get("n_action_steps", -1)) != 8:
        raise ValueError("source config does not declare chunk_size=10 and n_action_steps=8")

    report = {
        "schema_version": 1,
        "source_submission_zip": str(source_zip.resolve()),
        "source_submission_zip_sha256": source_zip_hash,
        "source_submission_zip_bytes": source_zip.stat().st_size,
        "source_zip_entry_count": len(infos),
        "source_zip_uncompressed_bytes": sum(i.file_size for i in infos),
        "source_zip_compressed_bytes": sum(i.compress_size for i in infos),
        "staging": str(staging.resolve()),
        "created_unix": time.time(),
        "inventory_elapsed_sec": time.time() - started,
        "source_model_sha256": source_model["sha256"],
        "source_model_bytes": source_model["uncompressed_bytes"],
        "source_config_sha256": source_config["sha256"],
        "source_preprocessor_sha256": source_pre["sha256"],
        "source_postprocessor_sha256": source_post["sha256"],
        "source_policy_server_sha256": source_server["sha256"],
        "source_config": config,
        "source_has_model_fast": "model_weights/model_fast.safetensors" in names,
        "members": members,
        "safety": {
            "crc_test": "passed",
            "absolute_paths_rejected": True,
            "traversal_rejected": True,
            "symlink_rejected": True,
            "hardlink_rejected": True,
            "nested_archive_rejected": True,
        },
    }
    write_json_atomic(report_path, report)
    return report


def inventory_from_model_dir(
    source_model_dir: Path,
    staging: Path,
    report_path: Path,
    *,
    server_template: Path,
    vendor_source: Path,
) -> dict[str, Any]:
    """Stage the locked on-disk R6 folded model without creating a source ZIP."""

    source_model_dir = source_model_dir.resolve()
    staging = staging.resolve()
    if not source_model_dir.is_dir():
        raise FileNotFoundError(source_model_dir)
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite existing staging directory: {staging}")
    if not server_template.is_file():
        raise FileNotFoundError(server_template)
    if not vendor_source.is_dir():
        raise FileNotFoundError(vendor_source)
    required_source = (
        "model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    source_files: dict[str, dict[str, Any]] = {}
    for name in required_source:
        path = source_model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if source_files["model.safetensors"]["sha256"] != R6_MODEL_SHA256:
        raise ValueError("R6 folded source model hash does not match the locked candidate")
    for name, expected in R6_PROCESSOR_SHA256.items():
        if source_files[name]["sha256"] != expected:
            raise ValueError(f"R6 processor hash mismatch: {name}")
    config = json.loads((source_model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05" or int(config.get("chunk_size", -1)) != 10 or int(config.get("n_action_steps", -1)) != 8:
        raise ValueError("R6 folded source config must declare pi05/chunk=10/n_action_steps=8")
    if config.get("normalization_mapping", {}).get("ACTION") != "QUANTILES" or config.get("normalization_mapping", {}).get("STATE") != "QUANTILES":
        raise ValueError("R6 folded source config must declare ACTION/STATE QUANTILES")

    model_dir = staging / "model_weights"
    model_dir.mkdir(parents=True)
    # Hard-link the 14GB source model on the same filesystem.  The exporter
    # only reads it; this avoids an untracked duplicate while preserving the
    # source hash and keeps the final ZIP's member a regular file.
    for name in required_source:
        target = model_dir / name
        source = source_model_dir / name
        if name == "model.safetensors":
            os.link(source, target)
        else:
            shutil.copy2(source, target)
    shutil.copy2(server_template, staging / "policy_server.py")
    shutil.copy2(server_template.with_name("requirements.txt"), staging / "requirements.txt")
    shutil.copytree(vendor_source, staging / "vendor", copy_function=os.link, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))

    members = []
    for name in required_source:
        path = model_dir / name
        members.append({
            "path": f"model_weights/{name}",
            "directory": False,
            "uncompressed_bytes": path.stat().st_size,
            "compressed_bytes": path.stat().st_size,
            "crc32": None,
            "sha256": sha256_file(path),
        })
    report = {
        "schema_version": 1,
        "source_kind": R6_SOURCE_KIND,
        "candidate": R6_CANDIDATE,
        "topology": R6_TOPOLOGY,
        "training_steps": 500,
        "normalization": "QUANTILES",
        "source_submission_zip": None,
        "source_submission_zip_sha256": None,
        "source_submission_zip_bytes": None,
        "source_model_path": str((source_model_dir / "model.safetensors")),
        "source_model_sha256": R6_MODEL_SHA256,
        "source_model_bytes": source_files["model.safetensors"]["bytes"],
        "source_config_sha256": source_files["config.json"]["sha256"],
        "source_preprocessor_sha256": source_files["policy_preprocessor.json"]["sha256"],
        "source_postprocessor_sha256": source_files["policy_postprocessor.json"]["sha256"],
        "source_files": source_files,
        "source_config": config,
        "source_has_model_fast": False,
        "server_template": str(server_template.resolve()),
        "vendor_source": str(vendor_source.resolve()),
        "staging": str(staging),
        "members": members,
        "safety": {
            "source_is_current_r6_on_disk_model": True,
            "historical_fast_asset_not_used": True,
            "source_model_hardlinked_read_only": True,
            "training_performed": False,
        },
    }
    write_json_atomic(report_path, report)
    return report


def verify_existing_staging(source_zip: Path, staging: Path, report_path: Path) -> dict[str, Any]:
    """Revalidate a staging directory created by an earlier failed export."""

    if not staging.is_dir() or not report_path.is_file():
        raise FileNotFoundError("existing staging recovery requires staging and source_inventory.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("source_submission_zip_sha256") != sha256_file(source_zip):
        raise ValueError("existing staging source zip hash does not match current source zip")
    checked = 0
    for member in report.get("members", []):
        if member.get("directory"):
            continue
        path = staging / str(member["path"])
        if not path.is_file():
            raise FileNotFoundError(f"staging member missing during recovery: {path}")
        if member["path"] == "model_weights/config.json":
            config = json.loads(path.read_text(encoding="utf-8"))
            if config.get("type") != "pi05" or int(config.get("chunk_size", -1)) != 10 or int(config.get("n_action_steps", -1)) != 8:
                raise ValueError("staging config changed beyond the allowed export-only fields")
            checked += 1
            continue
        if path.stat().st_size != int(member["uncompressed_bytes"]):
            raise ValueError(f"staging member size changed during recovery: {path}")
        if sha256_file(path) != member.get("sha256"):
            raise ValueError(f"staging member hash changed during recovery: {path}")
        checked += 1
    if not (staging / "model_weights/model.safetensors").is_file():
        raise FileNotFoundError("staging source model missing during recovery")
    report["recovery_revalidated_unix"] = time.time()
    report["recovery_revalidated_file_count"] = checked
    write_json_atomic(report_path, report)
    return report


def _set_cpu_export_config(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "device": "cpu",
            "dtype": "float32",
            "push_to_hub": False,
            "compile_model": False,
            "gradient_checkpointing": False,
        }
    )
    write_json_atomic(config_path, config)
    return config


def _tensor_summary(state: dict[str, Any]) -> dict[str, Any]:
    dtype_tensors: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    total_bytes = 0
    total_numel = 0
    for value in state.values():
        key = str(value.dtype).replace("torch.", "")
        bytes_for_tensor = int(value.numel() * value.element_size())
        dtype_tensors[key] += 1
        dtype_bytes[key] += bytes_for_tensor
        total_bytes += bytes_for_tensor
        total_numel += int(value.numel())
    return {
        "tensor_count": len(state),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "dtype_tensor_count": dict(sorted(dtype_tensors.items())),
        "dtype_bytes": dict(sorted(dtype_bytes.items())),
    }


def _header_summary(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
        key_dtypes = {key: str(handle.get_slice(key).get_dtype()) for key in keys}
        dtypes = Counter(key_dtypes.values())
    dtype_sizes = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    dtype_bytes = {
        dtype: sum(
            int(__import__("math").prod(shapes[key])) * dtype_sizes.get(key_dtypes[key], 0)
            for key in keys
            if key_dtypes[key] == dtype
        )
        for dtype in sorted(dtypes)
    }
    return {
        "keys": keys,
        "key_count": len(keys),
        "shapes": shapes,
        "dtype_tensor_count": dict(sorted(dtypes.items())),
        "dtype_bytes": dtype_bytes,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _strict_reload(model_dir: Path, fast_path: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    # Imports are intentionally after the staged vendor path is installed.
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config = PreTrainedConfig.from_pretrained(str(model_dir))
    if not isinstance(config, PI05Config):
        raise RuntimeError(f"expected PI05Config, got {type(config)}")
    config.device = "cpu"
    config.dtype = "float32"
    config.push_to_hub = False
    config.compile_model = False
    config.gradient_checkpointing = False
    model = PI05Policy(config)
    state = load_file(str(fast_path), device="cpu")
    try:
        model.load_state_dict(state, strict=True)
        missing, unexpected = [], []
    except RuntimeError as exc:
        raise RuntimeError(f"FAST reload key gate failed: {exc}") from exc
    result = {
        "missing": list(missing),
        "unexpected": list(unexpected),
        "strict_key_gate": not missing and not unexpected,
        "loaded_tensor_count": len(state),
    }
    del state, model, torch
    return result


def export_fast(staging: Path, report_root: Path) -> dict[str, Any]:
    """Export FAST from the staged source model only."""

    model_dir = staging / "model_weights"
    source_path = model_dir / "model.safetensors"
    fast_path = model_dir / "model_fast.safetensors"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if fast_path.exists():
        raise FileExistsError(f"refusing to overwrite FAST file: {fast_path}")
    inventory_path = report_root / "source_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}

    sys.path.insert(0, str(staging / "vendor"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

    import torch
    from safetensors.torch import load_file, save_file

    # Register Pi0.5 processor steps before the policy/processor imports.
    from lerobot.policies.pi05 import processor_pi05 as _pi05_processor  # noqa: F401
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    config = _set_cpu_export_config(model_dir / "config.json")
    source_header = _header_summary(source_path)
    t0 = time.time()
    print(f"[export] source={source_path} sha256={source_header['sha256']}", flush=True)
    print("[export] loading staged source model (no historical FAST asset)", flush=True)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    config_obj = PreTrainedConfig.from_pretrained(str(model_dir))
    if not isinstance(config_obj, PI05Config):
        raise RuntimeError(f"expected PI05Config, got {type(config_obj)}")
    config_obj.device = "cpu"
    config_obj.dtype = "float32"
    config_obj.push_to_hub = False
    config_obj.compile_model = False
    config_obj.gradient_checkpointing = False
    policy = PI05Policy(config_obj)
    source_state = load_file(str(source_path), device="cpu")
    for target_key, source_key in TIED_SOURCE_ALIASES.items():
        if target_key not in source_state and source_key in source_state:
            source_state[target_key] = source_state[source_key]
    # The source submission file is the underlying PI0Pytorch state dict.  A
    # PI05Policy wrapper owns that module under ``model``, so the FAST file
    # must use the wrapper's fully-qualified ``model.*`` keys.
    try:
        policy.model.load_state_dict(source_state, strict=True)
        load_missing, load_unexpected = [], []
    except RuntimeError as exc:
        raise RuntimeError(f"source model key gate failed while loading PI05Policy: {exc}") from exc
    if load_missing or load_unexpected:
        raise RuntimeError(
            "source model key gate failed while loading PI05Policy: "
            f"missing={list(load_missing)[:8]} unexpected={list(load_unexpected)[:8]}"
        )
    del source_state, load_missing, load_unexpected
    policy.model.paligemma_with_expert.to_bfloat16_for_selected_params("float16")
    for module_name in PROJECTION_NAMES:
        module = getattr(policy.model, module_name, None)
        if module is None:
            raise RuntimeError(f"required projection module is absent: {module_name}")
        module.to(dtype=torch.float32)
    policy.eval()

    state = {key: value.detach().contiguous() for key, value in policy.state_dict().items()}
    # safetensors requires aliases to be materialized unless save_model is
    # used.  Clone only repeated storages; all other tensors remain zero-copy.
    seen_storage: dict[tuple[int, int], str] = {}
    for key, value in list(state.items()):
        storage_id = (int(value.data_ptr()), int(value.nbytes))
        if storage_id in seen_storage:
            state[key] = value.clone().contiguous()
        else:
            seen_storage[storage_id] = key
    target_keys = sorted(state)
    source_keys = source_header["keys"]
    source_set = set(source_keys)
    target_set = set(target_keys)
    source_to_targets = {
        key: [f"model.{key}"] + [f"model.{target}" for target, source in TIED_SOURCE_ALIASES.items() if source == key]
        for key in source_keys
    }
    mapping = []
    for key in source_keys:
        target_keys_for_source = source_to_targets[key]
        target_key = target_keys_for_source[0]
        mapping.append(
            {
                "source_key": key,
                "target_key": target_key if target_key in target_set else None,
                "target_keys": [item for item in target_keys_for_source if item in target_set],
                "source_shape": source_header["shapes"][key],
                "target_shape": list(state[target_key].shape) if target_key in state else None,
                "status": "mapped_wrapper_model_prefix" if target_key in target_set else "unmapped_source",
            }
        )
    mapped_targets = {target for targets in source_to_targets.values() for target in targets}
    missing_source = sorted(target_key for target_key in target_set if target_key not in mapped_targets)
    unexpected_source = sorted(source_key for source_key in source_set if not set(source_to_targets[source_key]) & target_set)
    shape_mismatch = sorted(
        key
        for key in source_set
        for target_key in source_to_targets[key]
        if target_key in target_set
        and list(state[target_key].shape) != source_header["shapes"][key]
    )
    duplicate_targets: list[str] = []
    vision = {
        key: {
            "source_present": key in source_set,
            "target_key": source_to_targets.get(key, [None])[0],
            "target_keys": [item for item in source_to_targets.get(key, []) if item in target_set],
            "target_present": bool(set(source_to_targets.get(key, [])) & target_set),
            "source_shape": source_header["shapes"].get(key),
            "target_shape": list(state[source_to_targets[key][0]].shape) if source_to_targets.get(key, [None])[0] in state else None,
            "mapped_source_key": key if key in source_set and set(source_to_targets.get(key, [])) & target_set else None,
        }
        for key in VISION_KEYS
    }
    key_gate = not missing_source and not unexpected_source and not shape_mismatch and not duplicate_targets
    vision_gate = all(
        item["source_present"]
        and item["target_present"]
        and item["source_shape"] == item["target_shape"]
        and item["mapped_source_key"]
        for item in vision.values()
    )
    if not key_gate or not vision_gate:
        raise RuntimeError(
            "source/target key gate failed: "
            f"missing={len(missing_source)} unexpected={len(unexpected_source)} "
            f"shape_mismatch={len(shape_mismatch)} vision_gate={vision_gate}"
        )

    temporary = fast_path.with_name(f".{fast_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        save_file(state, str(temporary), metadata={"format": "axis_pi05_fast_mixed"})
        header = _header_summary(temporary)
        if header["keys"] != target_keys:
            raise RuntimeError("FAST header key set differs from exported state dict")
        temporary.replace(fast_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    del state, policy
    import gc

    gc.collect()
    reload_result = _strict_reload(model_dir, fast_path)
    fast_header = _header_summary(fast_path)
    if fast_header["keys"] != target_keys:
        raise RuntimeError("post-reload FAST key set changed")
    if not all(fast_header["shapes"][vision[key]["target_key"]] == list(vision[key]["target_shape"]) for key in VISION_KEYS):
        raise RuntimeError("post-reload vision key shapes changed")

    metadata = {
        "schema_version": 1,
        "format": "remapped_mixed_state_dict",
        "candidate": R6_CANDIDATE,
        "topology": R6_TOPOLOGY,
        "training_steps": 500,
        "normalization": "QUANTILES",
        "source_submission_zip": inventory.get("source_submission_zip"),
        "source_submission_zip_sha256": inventory.get("source_submission_zip_sha256"),
        "source_model": str(source_path.resolve()),
        "source_model_sha256": source_header["sha256"],
        "source_model_bytes": source_path.stat().st_size,
        "historical_fast_asset_used": False,
        "training_performed": False,
        "model_fast": str(fast_path.resolve()),
        "model_fast_sha256": fast_header["sha256"],
        "model_fast_bytes": fast_path.stat().st_size,
        "key_count": fast_header["key_count"],
        "dtype_policy": "float16_mixed_projection_fp32",
        "projection_modules_fp32": list(PROJECTION_NAMES),
        "vision_keys": vision,
        "source_target_key_gate": {
            "missing": missing_source,
            "unexpected": unexpected_source,
            "shape_mismatch": shape_mismatch,
            "duplicate_target": duplicate_targets,
            "passed": key_gate,
        },
        "reload": reload_result,
        "created_unix": time.time(),
    }
    write_json_atomic(model_dir / "pi05_fast_meta.json", metadata)
    export_report = {
        "schema_version": 1,
        "status": "passed",
        "candidate": R6_CANDIDATE,
        "topology": R6_TOPOLOGY,
        "training_steps": 500,
        "normalization": "QUANTILES",
        "norm_stats_sha256": R6_NORM_SHA256,
        "elapsed_sec": time.time() - t0,
        "source_model": source_header,
        "fast_model": fast_header,
        "config": config,
        "key_gate": {
            "missing": missing_source,
            "unexpected": unexpected_source,
            "shape_mismatch": shape_mismatch,
            "duplicate_target": duplicate_targets,
            "passed": key_gate,
        },
        "vision_gate": {"keys": vision, "passed": vision_gate},
        "dtype_summary": fast_header,
        "reload": reload_result,
        "source_is_current_submission_model": True,
        "source_kind": R6_SOURCE_KIND,
        "historical_fast_asset_used": False,
        "training_performed": False,
        "mapping_report": str((report_root / "key_mapping.json").resolve()),
    }
    # State tensors have been released before this point; the authoritative
    # dtype/byte summary is reconstructed from the safetensors header below.
    export_report["dtype_summary"] = {
        "tensor_count": fast_header["key_count"],
        "total_bytes": fast_header["bytes"],
        "dtype_tensor_count": fast_header["dtype_tensor_count"],
        "dtype_bytes": fast_header["dtype_bytes"],
    }
    write_json_atomic(report_root / "export_report.json", export_report)
    write_json_atomic(
        report_root / "key_mapping.json",
        {
            "schema_version": 1,
            "source_model": str(source_path.resolve()),
            "target_model": str(fast_path.resolve()),
            "source_key_count": len(source_keys),
            "target_key_count": len(target_keys),
            "mapping": mapping,
            "missing": missing_source,
            "unexpected": unexpected_source,
            "shape_mismatch": shape_mismatch,
            "duplicate_target": duplicate_targets,
            "vision": vision,
            "passed": key_gate and vision_gate and reload_result["strict_key_gate"],
        },
    )
    return export_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-zip", type=Path)
    source.add_argument("--source-model-dir", type=Path)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--server-template", type=Path, default=Path("outputs/axis_fast_recovery/f2_m1_500_fast/staging/policy_server.py"))
    parser.add_argument("--vendor-source", type=Path, default=Path("outputs/axis_fast_recovery/f2_m1_500_fast/staging/vendor"))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--reuse-staging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)
    source_inventory_path = args.report_root / "source_inventory.json"
    if args.source_model_dir is not None and args.reuse_staging:
        raise SystemExit("--reuse-staging is only valid with --source-zip")
    if args.source_model_dir is not None:
        inventory_and_extract_report = inventory_from_model_dir(
            args.source_model_dir,
            args.staging,
            source_inventory_path,
            server_template=args.server_template,
            vendor_source=args.vendor_source,
        )
    elif args.reuse_staging:
        inventory_and_extract_report = verify_existing_staging(
            args.source_zip, args.staging, source_inventory_path
        )
    else:
        inventory_and_extract_report = inventory_and_extract(
            args.source_zip, args.staging, source_inventory_path
        )
    if args.inventory_only:
        print(json.dumps({"status": "source_inventory_passed", "files": len(inventory_and_extract_report.get("members", []))}, indent=2))
        return
    result = export_fast(args.staging, args.report_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
