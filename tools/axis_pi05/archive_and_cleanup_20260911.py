#!/usr/bin/env python3
"""S1–S3 archive, restore verification, and enumerated cleanup.

Copy first, verify, then delete only explicit eligible paths. Never upload or
submit. Never touch Fast400 cursor/manifest or EVAL_HANDOFF.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, sha256_tree, write_json
from .s0_storage_release_inventory import ARCHIVE_ROOT, EVIDENCE, STABLE_EXCLUDE, df_path, now_jst


JST = timezone(timedelta(hours=9))
HOME = Path("/home")
TARGET_GIB = 550
EVAL_HANDOFF = ROOT / "data/reports/axis_libero_30k/official_canary100_local_gpu3_eval_20260911/EVAL_HANDOFF.json"
HF_CONTRACT = ROOT / "configs/axis_pi05/official_canary100_hf.json"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def rsync_copy(source: Path, destination: Path) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    _run(
        [
            "rsync",
            "-aH",
            "--info=name0,stats2",
            str(source if source.is_file() else str(source).rstrip("/") + "/"),
            str(destination if source.is_file() else str(destination).rstrip("/") + "/"),
        ]
    )
    return {
        "source": str(source),
        "destination": str(destination),
        "elapsed_sec": round(time.time() - started, 3),
        "source_bytes": source.stat().st_size if source.is_file() else None,
    }


def copy_tree_or_file(source: Path, dest_root: Path) -> Path:
    relative = source.relative_to(ROOT) if str(source).startswith(str(ROOT)) else Path(source.name)
    destination = dest_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    rsync_copy(source, destination)
    return destination


def verify_copy(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_file():
        src_sha = sha256_file(source)
        dst_sha = sha256_file(destination)
        return {
            "kind": "file",
            "source_bytes": source.stat().st_size,
            "destination_bytes": destination.stat().st_size,
            "source_sha256": src_sha,
            "destination_sha256": dst_sha,
            "match": src_sha == dst_sha and source.stat().st_size == destination.stat().st_size,
        }
    src_tree = sha256_tree(source, exclude=STABLE_EXCLUDE)
    dst_tree = sha256_tree(destination, exclude=STABLE_EXCLUDE)
    src_files = sum(1 for path in source.rglob("*") if path.is_file())
    dst_files = sum(1 for path in destination.rglob("*") if path.is_file())
    src_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
    dst_bytes = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    members = {
        "params": (destination / "params").is_dir(),
        "train_state": (destination / "train_state").is_dir(),
        "checkpoint_metadata": (destination / "_CHECKPOINT_METADATA").is_file(),
        "assets_norm": any(destination.rglob("norm_stats.json")),
    }
    return {
        "kind": "dir",
        "source_bytes": src_bytes,
        "destination_bytes": dst_bytes,
        "source_file_count": src_files,
        "destination_file_count": dst_files,
        "source_tree_sha256": src_tree,
        "destination_tree_sha256": dst_tree,
        "tree_hasher": "tools.axis_pi05.common.sha256_tree",
        "members": members,
        "match": src_tree == dst_tree and src_files == dst_files and src_bytes == dst_bytes,
    }


def restore_checkpoint(checkpoint: Path, report_path: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(ROOT / "third_party/axis_openpi/src") + os.pathsep + env.get("PYTHONPATH", "")
    code = r"""
import json, sys
from pathlib import Path
import numpy as np
from openpi.models import model as model_lib

checkpoint = Path(sys.argv[1])
params = model_lib.restore_params(str(checkpoint / "params"), restore_type=np.ndarray, dtype="float32")

def finite(node):
    if isinstance(node, dict):
        return all(finite(value) for value in node.values())
    array = np.asarray(node)
    return bool(np.isfinite(array).all())

train_state_ok = (checkpoint / "train_state").is_dir()
metadata_ok = (checkpoint / "_CHECKPOINT_METADATA").is_file()
report = {
    "checkpoint": str(checkpoint),
    "params_restored": True,
    "params_finite": finite(params),
    "train_state_present": train_state_ok,
    "metadata_present": metadata_ok,
    "status": "passed" if finite(params) and train_state_ok and metadata_ok else "failed",
}
print(json.dumps(report))
"""
    result = subprocess.run(
        [str(ROOT / ".venv_axis_pi05/bin/python"), "-c", code, str(checkpoint)],
        cwd=str(ROOT / "third_party/axis_openpi"),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        payload = {
            "status": "failed",
            "checkpoint": str(checkpoint),
            "returncode": result.returncode,
            "stderr": result.stderr[-4000:],
        }
    else:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    write_json(report_path, payload)
    return payload


def hf_refetch(track: str, dest_root: Path) -> dict[str, Any]:
    handoff = json.loads(EVAL_HANDOFF.read_text(encoding="utf-8"))
    contract = json.loads(HF_CONTRACT.read_text(encoding="utf-8"))
    spec = contract["tracks"][track]
    recorded = handoff["checkpoints"]["hf_recorded_during_upload_not_refetched"][track]
    dest = dest_root / "hf_refetch" / track / "100"
    dest.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="model",
        revision=recorded["commit"],
        local_dir=str(dest),
        cache_dir=str(ARCHIVE_ROOT / "hf_cache"),
    )
    return {
        "track": track,
        "repo_id": spec["repo_id"],
        "revision": recorded["commit"],
        "destination": str(dest),
        "complete": (dest / "params").is_dir() and (dest / "train_state").is_dir() and (dest / "_CHECKPOINT_METADATA").is_file(),
    }


def s1_copy(inventory: dict[str, Any]) -> dict[str, Any]:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    copies = []
    for item in inventory["objects"]:
        if item.get("status") not in {"archive_then_delete_source", "keep_local"}:
            continue
        if not item.get("exists"):
            continue
        role = item.get("role")
        if role in {"reports", "manifests", "fast400_raw", "eval_handoff"}:
            continue
        source = Path(item["path"])
        if role not in {
            "old_f2_m1_zip",
            "pretrained_base_zip",
            "q5_checkpoint",
            "canary100",
            "r6_step500",
            "r6_fallback_zip",
        }:
            continue
        destination = copy_tree_or_file(source, ARCHIVE_ROOT)
        copies.append({"role": role, "source": str(source), "destination": str(destination)})
    hf = []
    for track in ("T1", "T2", "T3"):
        hf.append(hf_refetch(track, ARCHIVE_ROOT))
    report = {
        "schema_version": "s1_archive_copy_v1",
        "status": "passed",
        "created_at_jst": now_jst(),
        "archive_root": str(ARCHIVE_ROOT),
        "copies": copies,
        "hf_refetch": hf,
        "deleted_source": False,
        "eval_handoff_untouched": True,
    }
    write_json(EVIDENCE / "s1_archive_copy.json", report)
    return report


def s2_verify(s1: dict[str, Any]) -> dict[str, Any]:
    rows = []
    eligible = []
    for copy in s1["copies"]:
        source = Path(copy["source"])
        destination = Path(copy["destination"])
        verification = verify_copy(source, destination)
        restore = None
        if copy["role"] in {"canary100", "q5_checkpoint", "r6_step500"}:
            restore = restore_checkpoint(
                destination,
                EVIDENCE / "s2_restore" / f"{source.name}_{copy['role']}_restore.json",
            )
        two_copies = destination.exists() and source.exists()
        archive_verified = bool(verification["match"] and two_copies and (restore is None or restore.get("status") == "passed"))
        can_delete = archive_verified and copy["role"] in {"old_f2_m1_zip", "pretrained_base_zip", "q5_checkpoint"}
        row = {
            "role": copy["role"],
            "source": str(source),
            "destination": str(destination),
            "verification": verification,
            "restore": restore,
            "archive_verified": archive_verified,
            "deletion_eligible": can_delete,
        }
        rows.append(row)
        if can_delete:
            eligible.append(str(source))
    hf_rows = []
    for item in s1.get("hf_refetch", []):
        dest = Path(item["destination"])
        restore = restore_checkpoint(dest, EVIDENCE / "s2_restore" / f"hf_{item['track']}_restore.json") if item.get("complete") else {"status": "failed"}
        hf_rows.append({**item, "restore": restore, "archive_verified": restore.get("status") == "passed"})
    report = {
        "schema_version": "s2_archive_verification_v1",
        "status": "passed" if all(row["verification"]["match"] for row in rows) else "failed",
        "created_at_jst": now_jst(),
        "copies": rows,
        "hf_refetch": hf_rows,
        "deletion_eligible_paths": eligible,
    }
    write_json(EVIDENCE / "s2_archive_verification.json", report)
    write_json(
        EVIDENCE / "deletion_eligibility.json",
        {
            "schema_version": "deletion_eligibility_v1",
            "created_at_jst": now_jst(),
            "paths": [
                {"path": row["source"], "archive_verified": row["archive_verified"], "deletion_eligible": row["deletion_eligible"], "archive": row["destination"]}
                for row in rows
            ],
        },
    )
    return report


def s3_cleanup(eligibility: dict[str, Any]) -> dict[str, Any]:
    before = df_path(HOME)
    deleted = []
    skipped = []
    for item in eligibility["paths"]:
        current = df_path(HOME)
        if current["available_gib"] >= TARGET_GIB:
            skipped.append({"path": item["path"], "reason": "target_capacity_reached"})
            continue
        if not item.get("deletion_eligible") or not item.get("archive_verified"):
            skipped.append({"path": item["path"], "reason": "not_eligible"})
            continue
        path = Path(item["path"])
        if path.is_symlink() or ".." in path.parts:
            skipped.append({"path": str(path), "reason": "symlink_or_traversal"})
            continue
        bytes_before = path.stat().st_size if path.is_file() else sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        deleted.append({"path": str(path), "bytes": bytes_before, "archive": item.get("archive")})
    after = df_path(HOME)
    report = {
        "schema_version": "s3_enumerated_cleanup_v1",
        "status": "passed" if after["available_gib"] >= TARGET_GIB else "incomplete_below_target",
        "created_at_jst": now_jst(),
        "before": before,
        "after": after,
        "deleted": deleted,
        "skipped": skipped,
        "restore_command": f"rsync -aH {ARCHIVE_ROOT}/ <original-path>",
        "eval_handoff_untouched": True,
        "fast400_untouched": True,
        "r6_fallback_not_deleted": True,
    }
    write_json(EVIDENCE / "s3_enumerated_cleanup.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["S1", "S2", "S3", "all"], required=True)
    parser.add_argument("--inventory", type=Path, default=EVIDENCE / "s0_storage_release_inventory.json")
    args = parser.parse_args()
    if args.phase in {"S1", "all"}:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        s1 = s1_copy(inventory)
        print(json.dumps({"phase": "S1", "copies": len(s1["copies"]), "hf": len(s1["hf_refetch"])}, indent=2))
    if args.phase in {"S2", "all"}:
        s1 = json.loads((EVIDENCE / "s1_archive_copy.json").read_text(encoding="utf-8"))
        s2 = s2_verify(s1)
        print(json.dumps({"phase": "S2", "status": s2["status"], "eligible": len(s2["deletion_eligible_paths"])}, indent=2))
    if args.phase in {"S3", "all"}:
        eligibility = json.loads((EVIDENCE / "deletion_eligibility.json").read_text(encoding="utf-8"))
        s3 = s3_cleanup(eligibility)
        print(json.dumps({"phase": "S3", "status": s3["status"], "after_gib": s3["after"]["available_gib"]}, indent=2))


if __name__ == "__main__":
    main()
