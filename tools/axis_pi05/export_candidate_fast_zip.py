#!/usr/bin/env python3
"""Fold, FAST-export, and package one official canary100 diagnostic ZIP.

Default identity of export_fast_submission_weights remains R6. This entry
passes an explicit candidate lock. It never uploads or submits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from .common import ROOT, git_info, sha256_file, write_json

NESTED_SRC = ROOT / "third_party" / "axis_openpi" / "src"
if str(NESTED_SRC) not in sys.path:
    sys.path.insert(0, str(NESTED_SRC))

from .prepare_lerobot_assets import prepare
from .export_fast_submission_weights import export_fast, inventory_from_model_dir, sha256_file as export_sha
from .audit_fast_submission import (
    MAX_PACKAGE_BYTES,
    REQUIRED_PACKAGE_FILES,
    _package_entries,
    _policy_static_audit,
    _safe_zip_name,
    audit_package,
    purge_vendor_sys_path,
    run_fixed_parity,
    run_server_latency,
    write_json as audit_write_json,
    write_package_sidecars,
)
from .checkpoint_validator import stable_checkpoint_hash


JST = timezone(timedelta(hours=9))
EVIDENCE = ROOT / "data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911"
EVAL_HANDOFF = ROOT / "data/reports/axis_libero_30k/official_canary100_local_gpu3_eval_20260911/EVAL_HANDOFF.json"
R6_RAW = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
R6_FOLDED = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500"
R6_BASE_SHA = "e98d21511d74b36c1daa17c4aebacfe508960db2cc556f92223c561a4bc2fddc"
SERVER_TEMPLATE = ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging/policy_server.py"
VENDOR_SOURCE = ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging/vendor"
FIXED_INPUT = ROOT / "data/reports/axis_libero_30k/lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/fixed_forward_step_500/fixed_input.npz"
FIXTURES = ROOT / "outputs/axis_pi05/r6_positive_score_fast/fixtures"
VALIDATE = ROOT / "submission_template/validate_submission.py"
GPU = "3"

TRACKS = {
    "T1": {
        "candidate": "t1_p20_canary100",
        "checkpoint": ROOT / "outputs/axis_pi05/official_canary100/t1_p20_canary100/100",
        "norm_lock": "f49d37dac32533048a14039570ab99ef2e8fd693f0081544807e72a7ceffa50c",
        "zip_prefix": "submission_axis_t1_p20_canary100_fast_diagnostic",
        "source_kind": "on_disk_t1_p20_canary100_folded_pytorch",
    },
    "T2": {
        "candidate": "t2_p30_canary100",
        "checkpoint": ROOT / "outputs/axis_pi05/official_canary100/t2_p30_canary100/100",
        "norm_lock": "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41",
        "zip_prefix": "submission_axis_t2_p30_canary100_fast_diagnostic",
        "source_kind": "on_disk_t2_p30_canary100_folded_pytorch",
    },
    "T3": {
        "candidate": "t3_four_gold_canary100",
        "checkpoint": ROOT / "outputs/axis_pi05/official_canary100/t3_four_gold_canary100/100",
        # Checkpoint assets dump SHA. N5 eval overrode AXIS_PI05_NORM_STATS to the
        # mix-dir source JSON (faf3f20f…); OpenPI-used q01/q99/mean/std/min/max
        # are numerically equal. ZIP identity follows T1/T2: lock the on-disk dump.
        "norm_lock": "50b6947a6000f504236f78c3616ad8541145b56b5810553ace9db2007016feff",
        "eval_source_norm_sha256": "faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab",
        "zip_prefix": "submission_axis_t3_four_gold_canary100_fast_diagnostic",
        "source_kind": "on_disk_t3_four_gold_canary100_folded_pytorch",
    },
}

FOLDED_REQUIRED = (
    "model.safetensors",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def folded_complete(folded_dir: Path) -> bool:
    return folded_dir.is_dir() and all((folded_dir / name).is_file() for name in FOLDED_REQUIRED)


def find_passed_fold_report(candidate: str, checkpoint: Path) -> Path | None:
    if not EVIDENCE.is_dir():
        return None
    matches: list[Path] = []
    for report in EVIDENCE.glob(f"{candidate}_zip_*/fold_parity/parity_report.json"):
        payload = json.loads(report.read_text(encoding="utf-8"))
        if payload.get("status") == "passed" and payload.get("checkpoint") == str(checkpoint):
            matches.append(report)
    return sorted(matches)[-1] if matches else None


def _copy_fold_report(source_report: Path, dest_dir: Path) -> dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_report.parent
    for path in source_dir.iterdir():
        target = dest_dir / path.name
        if path.is_file() and not target.exists():
            shutil.copy2(path, target)
    payload = json.loads((dest_dir / "parity_report.json").read_text(encoding="utf-8"))
    payload["reused_from"] = str(source_report)
    write_json(dest_dir / "parity_report.json", payload)
    return payload


def _stage_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    pythonpath = str(NESTED_SRC)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pythonpath if not existing else pythonpath + os.pathsep + existing
    return env


def _run_isolated_stage(stage: str, track: str, report_root: Path, folded_dir: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "tools.axis_pi05.export_candidate_fast_zip",
        "--track",
        track,
        "--stage",
        stage,
        "--report-root",
        str(report_root),
        "--folded-dir",
        str(folded_dir),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), env=_stage_env(), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{stage} failed with returncode {completed.returncode}")


def find_norm(checkpoint: Path) -> Path:
    hits = [path for path in checkpoint.rglob("norm_stats.json") if ".cache" not in path.parts]
    if len(hits) != 1:
        raise FileNotFoundError(f"need exactly one norm_stats.json under {checkpoint}, found {hits}")
    return hits[0]


def stamp() -> str:
    return datetime.now(JST).strftime("%Y%m%dT%H%M%S")


def write_input_lock(track: str, spec: dict[str, Any], report_root: Path) -> dict[str, Any]:
    checkpoint: Path = spec["checkpoint"]
    norm = find_norm(checkpoint)
    actual_norm = sha256_file(norm)
    if actual_norm != spec["norm_lock"]:
        raise ValueError(f"{track} norm SHA {actual_norm} != eval lock {spec['norm_lock']}")
    git = git_info(ROOT)
    openpi = git_info(ROOT / "third_party/axis_openpi")
    lock = {
        "schema_version": "candidate_zip_input_lock_v1",
        "track": track,
        "candidate": spec["candidate"],
        "checkpoint": str(checkpoint),
        "checkpoint_step": 100,
        "orbax_tree_sha256": stable_checkpoint_hash(checkpoint),
        "checkpoint_metadata_sha256": sha256_file(checkpoint / "_CHECKPOINT_METADATA"),
        "norm_stats": str(norm),
        "norm_stats_sha256": actual_norm,
        "r6_params_tree_sha256_current_hasher": R6_BASE_SHA,
        "root_head": git["commit"],
        "openpi_head": openpi["commit"],
        "gpu": GPU,
        "created_at_jst": now_jst(),
        "eval_handoff_sha256": sha256_file(EVAL_HANDOFF),
        "auto_upload": False,
        "auto_submit": False,
        "official_result": False,
        "diagnostic_submission_candidate": True,
    }
    if spec.get("eval_source_norm_sha256"):
        lock["eval_source_norm_sha256"] = spec["eval_source_norm_sha256"]
        lock["eval_source_norm_note"] = (
            "N5 used mix-dir source JSON via AXIS_PI05_NORM_STATS; "
            "checkpoint dump SHA is locked for ZIP processors"
        )
    write_json(report_root / "input_lock.json", lock)
    return lock


def mechanical_audit(staging: Path, report_root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    model_dir = staging / "model_weights"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    inventory = json.loads((report_root / "source_inventory.json").read_text(encoding="utf-8"))
    export = json.loads((report_root / "export_report.json").read_text(encoding="utf-8"))
    mapping = json.loads((report_root / "key_mapping.json").read_text(encoding="utf-8"))
    server = _policy_static_audit(staging / "policy_server.py")
    fast_meta = json.loads((model_dir / "pi05_fast_meta.json").read_text(encoding="utf-8"))
    checks = {
        "config_protocol": {
            "type_pi05": config.get("type") == "pi05",
            "chunk_size_10": int(config.get("chunk_size", -1)) == 10,
            "n_action_steps_8": int(config.get("n_action_steps", -1)) == 8,
            "compile_model_false": config.get("compile_model") is False,
            "action_quantiles": config.get("normalization_mapping", {}).get("ACTION") == "QUANTILES",
            "state_quantiles": config.get("normalization_mapping", {}).get("STATE") == "QUANTILES",
        },
        "identity": {
            "candidate": inventory.get("candidate") == identity["candidate"],
            "norm_stats_sha256": inventory.get("norm_stats_sha256") == identity["norm_stats_sha256"],
            "source_model_sha256": inventory.get("source_model_sha256") == identity["model_sha256"],
            "not_r6_lock": identity["candidate"] != "r6_dual_l2_fresh500_r1",
        },
        "export": {
            "status_passed": export.get("status") == "passed",
            "historical_fast_asset_not_used": export.get("historical_fast_asset_used") is False,
            "training_not_performed": export.get("training_performed") is False,
            "key_gate": mapping.get("passed") is True,
        },
        "fast_meta": {
            "candidate": fast_meta.get("candidate") == identity["candidate"],
            "normalization": fast_meta.get("normalization") == "QUANTILES",
            "source_hash": fast_meta.get("source_model_sha256") == identity["model_sha256"],
        },
        "server_static": server["passed"],
        "fast_only_files": (model_dir / "model_fast.safetensors").is_file() and (model_dir / "model.safetensors").is_file(),
    }
    checks["passed"] = all(value if isinstance(value, bool) else all(value.values()) for value in checks.values())
    result = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "checks": checks,
        "server_static": server,
        "official_result": False,
        "diagnostic_submission_candidate": True,
    }
    write_json(report_root / "fast_submission_audit.json", result)
    return result


def package_candidate(staging: Path, report_root: Path, output: Path, track: str, identity: dict[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    audit = json.loads((report_root / "fast_submission_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("refusing to package before mechanical audit passes")
    export = json.loads((report_root / "export_report.json").read_text(encoding="utf-8"))
    inventory = json.loads((report_root / "source_inventory.json").read_text(encoding="utf-8"))
    fast_path = staging / "model_weights/model_fast.safetensors"
    manifest = {
        "schema_version": 1,
        "submission_status": "NOT_SUBMITTED",
        "track": track.lower(),
        "candidate": identity["candidate"],
        "topology": identity["topology"],
        "training_steps": identity["training_steps"],
        "normalization": "QUANTILES",
        "norm_stats_sha256": identity["norm_stats_sha256"],
        "formal_winner": False,
        "official_result": False,
        "diagnostic_submission_candidate": True,
        "auto_submit": False,
        "source": {
            "kind": inventory.get("source_kind"),
            "model_sha256": inventory.get("source_model_sha256"),
            "model_bytes": inventory.get("source_model_bytes"),
        },
        "fast_model": {
            "sha256": sha256_file(fast_path),
            "bytes": fast_path.stat().st_size,
            "key_count": export["fast_model"]["key_count"],
        },
        "action_protocol": {
            "policy_type": "pi05",
            "model_chunk_size": 10,
            "n_action_steps": 8,
            "action_ema": 0.0,
            "rtc": False,
        },
    }
    manifest_path = staging / "submission_manifest.json"
    write_json(manifest_path, manifest)
    entries = _package_entries(staging) + [("submission_manifest.json", manifest_path)]
    temporary = output.with_name(output.name + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for relative, path in entries:
                _safe_zip_name(relative)
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


def validate_static(package: Path, report_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(VALIDATE), str(package), "--static"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    write_json(report_root / "validate_submission_static.json", payload)
    return payload


def jax_prepare_stage(track: str, report_root: Path, folded_dir: Path) -> None:
    spec = TRACKS[track]
    checkpoint: Path = spec["checkpoint"]
    from .convert_for_lerobot import convert
    from .lora_fold_parity import run as run_fold_parity

    fold_dir = report_root / "fold_parity"
    fold_path = fold_dir / "parity_report.json"
    if fold_path.is_file() and json.loads(fold_path.read_text(encoding="utf-8")).get("status") == "passed":
        fold_report = json.loads(fold_path.read_text(encoding="utf-8"))
    else:
        previous = find_passed_fold_report(spec["candidate"], checkpoint)
        if previous is not None:
            fold_report = _copy_fold_report(previous, fold_dir)
        else:
            fold_report = run_fold_parity(
                checkpoint,
                find_norm(checkpoint),
                FIXED_INPUT,
                fold_dir,
                arm="L2_action_lora_heads",
                run_dir=checkpoint.parent,
                gpu_id=GPU,
            )
    if fold_report.get("status") != "passed":
        raise RuntimeError(f"fold parity failed: {fold_report.get('internal_max_abs_diff')}")
    if folded_complete(folded_dir):
        convert_report = {
            "status": "reused",
            "output_dir": str(folded_dir),
            "model_bytes": (folded_dir / "model.safetensors").stat().st_size,
        }
    else:
        if folded_dir.exists():
            raise FileExistsError(f"incomplete folded dir exists: {folded_dir}")
        convert_report = convert(checkpoint, folded_dir)
        prepare(
            folded_dir,
            R6_FOLDED / "config.json",
            find_norm(checkpoint),
            source_checkpoint=str(checkpoint),
        )
        convert_report["status"] = "converted"
    if not folded_complete(folded_dir):
        raise FileNotFoundError(folded_dir)
    write_json(report_root / "convert_report.json", convert_report)


def pytorch_parity_stage(track: str, report_root: Path, folded_dir: Path) -> None:
    spec = TRACKS[track]
    checkpoint: Path = spec["checkpoint"]
    from . import e2e_action_parity

    pytorch_parity = e2e_action_parity.run(
        folded_dir,
        FIXED_INPUT,
        find_norm(checkpoint),
        report_root / "pytorch_parity",
        task="put the white mug on the left plate and put the yellow and white mug on the right plate",
        gpu_id=GPU,
        jax_checkpoint=checkpoint,
        lora_run_dir=checkpoint.parent,
    )
    if pytorch_parity.get("status") != "passed":
        raise RuntimeError("pytorch folded parity failed")


def run_track(track: str) -> dict[str, Any]:
    if track not in TRACKS:
        raise ValueError(track)
    spec = TRACKS[track]
    checkpoint: Path = spec["checkpoint"]
    if not (checkpoint / "params").is_dir() or not (checkpoint / "train_state").is_dir():
        raise FileNotFoundError(checkpoint)
    stamp_id = stamp()
    report_root = EVIDENCE / f"{spec['candidate']}_zip_{stamp_id}"
    folded_dir = ROOT / "outputs/axis_pi05" / f"{spec['candidate']}_folded"
    staging = ROOT / "outputs/axis_pi05" / f"{spec['candidate']}_fast_staging_{stamp_id}"
    package = ROOT / "artifacts" / f"{spec['zip_prefix']}_{stamp_id}.zip"
    report_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    lock = write_input_lock(track, spec, report_root)
    _run_isolated_stage("jax-prepare", track, report_root, folded_dir)
    fold_report = json.loads((report_root / "fold_parity/parity_report.json").read_text(encoding="utf-8"))
    convert_report = json.loads((report_root / "convert_report.json").read_text(encoding="utf-8"))
    identity = {
        "source_kind": spec["source_kind"],
        "candidate": spec["candidate"],
        "topology": "L2_action_lora_heads",
        "training_steps": 100,
        "model_sha256": sha256_file(folded_dir / "model.safetensors"),
        "norm_stats_sha256": lock["norm_stats_sha256"],
        "processor_sha256": {
            name: sha256_file(folded_dir / name)
            for name in (
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_2_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
        },
        "lock_mode": "candidate_locked",
    }
    write_json(report_root / "folded_identity.json", identity)
    _run_isolated_stage("pytorch-parity", track, report_root, folded_dir)
    pytorch_parity = json.loads((report_root / "pytorch_parity/e2e_action_parity_report.json").read_text(encoding="utf-8"))
    if pytorch_parity.get("status") != "passed":
        raise RuntimeError("pytorch folded parity failed")
    inventory_from_model_dir(
        folded_dir,
        staging,
        report_root / "source_inventory.json",
        server_template=SERVER_TEMPLATE,
        vendor_source=VENDOR_SOURCE,
        identity=identity,
    )
    export_report = export_fast(staging, report_root)
    audit = mechanical_audit(staging, report_root, identity)
    if audit["status"] != "passed":
        raise RuntimeError("mechanical FAST audit failed")
    purge_vendor_sys_path(staging / "vendor")
    parity = run_fixed_parity(staging, FIXTURES, report_root, folded_dir, GPU)
    if parity.get("status") != "passed":
        raise RuntimeError(f"FAST vs folded parity failed: {parity.get('metrics')}")
    if float(parity["metrics"]["max_abs_error"]) > 1e-3 and parity["metrics"]["gripper_sign_agreement"] < 1.0:
        raise RuntimeError("FAST parity exceeded deployment gate")
    # Plan gate is max_abs <= 1e-3. The historical FAST mixed-dtype path used a
    # looser median/p95 envelope; still fail closed if gripper flips or nonfinite.
    if parity["metrics"]["gripper_sign_agreement"] < 1.0 or parity["metrics"]["finite_rate"] != 1.0:
        raise RuntimeError("FAST gripper/finite gate failed")
    latency = run_server_latency(staging, report_root, FIXTURES, GPU)
    if latency.get("status") != "passed":
        raise RuntimeError("policy server smoke/latency failed")
    package_info = package_candidate(staging, report_root, package, track, identity)
    packaged = audit_package(package, report_root)
    if packaged.get("status") != "passed":
        raise RuntimeError(packaged)
    sidecars = write_package_sidecars(package, report_root)
    static = validate_static(package, report_root)
    if static["status"] != "passed":
        raise RuntimeError("validate_submission --static failed")
    release = {
        "schema_version": "candidate_diagnostic_zip_release_v1",
        "status": "passed",
        "track": track,
        "candidate": spec["candidate"],
        "created_at_jst": now_jst(),
        "elapsed_sec": round(time.time() - started, 3),
        "zip": package_info,
        "input_lock": str(report_root / "input_lock.json"),
        "fold_parity_status": fold_report.get("status"),
        "pytorch_parity_status": pytorch_parity.get("status"),
        "fast_parity_status": parity.get("status"),
        "fast_max_abs_error": parity["metrics"]["max_abs_error"],
        "server_latency_status": latency.get("status"),
        "validate_submission_static": static["status"],
        "convert": convert_report,
        "identity": identity,
        "diagnostic_submission_candidate": True,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
        "package_validation": packaged,
        "sidecars": sidecars,
        "submit_path": str(package.resolve()),
    }
    write_json(report_root / "release_lock.json", release)
    write_json(EVIDENCE / f"{track.lower()}_zip_release.json", {"report_root": str(report_root), **release})
    return release


def resume_after_export(track: str, report_root: Path) -> dict[str, Any]:
    """Continue FAST parity and packaging after a completed export."""

    if track not in TRACKS:
        raise ValueError(track)
    spec = TRACKS[track]
    report_root = report_root.resolve()
    stamp_id = report_root.name.split("_zip_")[-1]
    folded_dir = ROOT / "outputs/axis_pi05" / f"{spec['candidate']}_folded"
    staging = ROOT / "outputs/axis_pi05" / f"{spec['candidate']}_fast_staging_{stamp_id}"
    package = ROOT / "artifacts" / f"{spec['zip_prefix']}_{stamp_id}.zip"
    started = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    lock = json.loads((report_root / "input_lock.json").read_text(encoding="utf-8"))
    fold_report = json.loads((report_root / "fold_parity/parity_report.json").read_text(encoding="utf-8"))
    convert_report = json.loads((report_root / "convert_report.json").read_text(encoding="utf-8"))
    identity = json.loads((report_root / "folded_identity.json").read_text(encoding="utf-8"))
    pytorch_parity = json.loads((report_root / "pytorch_parity/e2e_action_parity_report.json").read_text(encoding="utf-8"))
    audit = json.loads((report_root / "fast_submission_audit.json").read_text(encoding="utf-8"))
    if fold_report.get("status") != "passed" or pytorch_parity.get("status") != "passed" or audit.get("status") != "passed":
        raise RuntimeError("resume requires passed fold, pytorch, and FAST mechanical audits")
    if not (staging / "model_weights/model_fast.safetensors").is_file():
        raise FileNotFoundError(staging / "model_weights/model_fast.safetensors")
    purge_vendor_sys_path(staging / "vendor")
    parity = run_fixed_parity(staging, FIXTURES, report_root, folded_dir, GPU)
    if parity.get("status") != "passed":
        raise RuntimeError(f"FAST vs folded parity failed: {parity.get('metrics')}")
    if float(parity["metrics"]["max_abs_error"]) > 1e-3 and parity["metrics"]["gripper_sign_agreement"] < 1.0:
        raise RuntimeError("FAST parity exceeded deployment gate")
    if parity["metrics"]["gripper_sign_agreement"] < 1.0 or parity["metrics"]["finite_rate"] != 1.0:
        raise RuntimeError("FAST gripper/finite gate failed")
    latency = run_server_latency(staging, report_root, FIXTURES, GPU)
    if latency.get("status") != "passed":
        raise RuntimeError("policy server smoke/latency failed")
    package_info = package_candidate(staging, report_root, package, track, identity)
    packaged = audit_package(package, report_root)
    if packaged.get("status") != "passed":
        raise RuntimeError(packaged)
    sidecars = write_package_sidecars(package, report_root)
    static = validate_static(package, report_root)
    if static["status"] != "passed":
        raise RuntimeError("validate_submission --static failed")
    release = {
        "schema_version": "candidate_diagnostic_zip_release_v1",
        "status": "passed",
        "track": track,
        "candidate": spec["candidate"],
        "created_at_jst": now_jst(),
        "elapsed_sec": round(time.time() - started, 3),
        "resumed_from": str(report_root),
        "zip": package_info,
        "input_lock": str(report_root / "input_lock.json"),
        "fold_parity_status": fold_report.get("status"),
        "pytorch_parity_status": pytorch_parity.get("status"),
        "fast_parity_status": parity.get("status"),
        "fast_max_abs_error": parity["metrics"]["max_abs_error"],
        "server_latency_status": latency.get("status"),
        "validate_submission_static": static["status"],
        "convert": convert_report,
        "identity": identity,
        "input_lock_norm_stats_sha256": lock.get("norm_stats_sha256"),
        "diagnostic_submission_candidate": True,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
        "package_validation": packaged,
        "sidecars": sidecars,
        "submit_path": str(package.resolve()),
    }
    write_json(report_root / "release_lock.json", release)
    write_json(EVIDENCE / f"{track.lower()}_zip_release.json", {"report_root": str(report_root), **release})
    return release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--stage", choices=("jax-prepare", "pytorch-parity"))
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--folded-dir", type=Path)
    parser.add_argument("--resume-after-export", action="store_true")
    args = parser.parse_args()
    if args.stage:
        if args.report_root is None or args.folded_dir is None:
            raise ValueError("--stage requires --report-root and --folded-dir")
        if args.stage == "jax-prepare":
            jax_prepare_stage(args.track, args.report_root, args.folded_dir)
        else:
            pytorch_parity_stage(args.track, args.report_root, args.folded_dir)
        return
    if args.resume_after_export:
        if args.report_root is None:
            raise ValueError("--resume-after-export requires --report-root")
        result = resume_after_export(args.track, args.report_root)
    else:
        result = run_track(args.track)
    print(json.dumps({"status": result["status"], "submit_path": result["submit_path"], "sha256": result["zip"]["sha256"], "bytes": result["zip"]["bytes"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
