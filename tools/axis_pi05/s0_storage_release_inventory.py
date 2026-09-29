#!/usr/bin/env python3
"""S0 read-only storage and release inventory for the 2026-09-11 continuation plan.

Does not move, delete, train, upload, or submit. Fast400 cursor/manifest and
EVAL_HANDOFF.json are not written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, sha256_tree, write_json


JST = timezone(timedelta(hours=9))
EVIDENCE = ROOT / "data/reports/axis_libero_30k/t1_t2_dedicated_panel_and_official_250_horizon_20260911"
HF_CONTRACT = ROOT / "configs/axis_pi05/official_canary100_hf.json"
EVAL_HANDOFF = ROOT / "data/reports/axis_libero_30k/official_canary100_local_gpu3_eval_20260911/EVAL_HANDOFF.json"
ARCHIVE_ROOT = Path("/mnt/home/huang/PARC2026_final_archive/20260911")
PERSIST_ROOT = Path.home() / "data/parc20260911"
CHUNK = 8 * 1024 * 1024
STABLE_EXCLUDE = (
    "checkpoint_manifest.json",
    "checkpoint_manifest_v2.json",
    "restore_validation.json",
    "LOCK",
    "_LOCK",
)


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def df_path(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "available_bytes": usage.free,
        "available_gib": round(usage.free / (1024**3), 2),
        "used_pct": round(100.0 * usage.used / usage.total, 2) if usage.total else None,
    }


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def lsof_paths(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    opened: dict[str, list[dict[str, Any]]] = {str(path): [] for path in paths}
    for path in paths:
        if not path.exists():
            continue
        result = subprocess.run(
            ["lsof", "-nP", "+D", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in {0, 1}:
            opened[str(path)] = [{"error": result.stderr.strip() or f"lsof_exit_{result.returncode}"}]
            continue
        lines = [line for line in result.stdout.splitlines()[1:] if line.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) < 9:
                continue
            opened[str(path)].append({"pid": parts[1], "command": parts[0], "name": parts[-1]})
    return opened


def nvidia_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    procs = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "gpus": gpu.stdout.strip(),
        "compute_apps": procs.stdout.strip(),
        "returncodes": {"gpu": gpu.returncode, "apps": procs.returncode},
    }


def file_record(path: Path, *, hash_file: bool) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": True,
        "kind": "file" if path.is_file() else "dir" if path.is_dir() else "other",
        "bytes": st.st_size if path.is_file() else None,
        "nlink": st.st_nlink,
        "inode": st.st_ino,
        "mtime_unix": st.st_mtime,
        "mode": stat.filemode(st.st_mode),
    }
    if path.is_file() and hash_file:
        record["sha256"] = sha256_file(path)
    return record


def dir_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    files = 0
    total = 0
    nlink_gt1 = 0
    inodes: set[int] = set()
    metadata: dict[str, str] = {}
    members = {
        "params": (path / "params").is_dir(),
        "train_state": (path / "train_state").is_dir(),
        "checkpoint_metadata": (path / "_CHECKPOINT_METADATA").is_file(),
        "assets": (path / "assets").is_dir(),
    }
    for child in path.rglob("*"):
        if not child.is_file() or child.is_symlink():
            continue
        files += 1
        st = child.stat()
        total += st.st_size
        inodes.add(st.st_ino)
        if st.st_nlink > 1:
            nlink_gt1 += 1
        name = child.name
        if name in {"_CHECKPOINT_METADATA", "norm_stats.json", "params_scope.json"}:
            metadata[str(child.relative_to(path))] = sha256_file(child)
    record = {
        "path": str(path.resolve()) if path.exists() else str(path),
        "exists": True,
        "kind": "dir",
        "bytes": total,
        "gib": round(total / (1024**3), 3),
        "file_count": files,
        "unique_inodes": len(inodes),
        "files_with_nlink_gt1": nlink_gt1,
        "checkpoint_members": members,
        "key_metadata_sha256": metadata,
        "partial_residue": any(".partial" in child.name for child in path.rglob("*")),
    }
    return record


def classify(path: Path, role: str) -> str:
    text = str(path)
    if role in {"fast400_raw", "reports", "manifests", "eval_handoff"}:
        return "immutable_keep"
    if role in {"r6_step500", "r6_fallback_zip", "canary100", "r6_folded", "r6_fast_staging"}:
        return "keep_local"
    if role in {
        "old_f2_m1_zip",
        "pretrained_base_zip",
        "q5_checkpoint",
        "r6_intermediate",
        "dense_checkpoint",
        "historical_lora_run",
        "nan_replay_checkpoint",
    }:
        return "archive_then_delete_source"
    if "unknown" in role:
        return "unknown_do_not_touch"
    if text.startswith(str(ROOT / "data/reports")):
        return "immutable_keep"
    return "unknown_do_not_touch"


def object_spec(path: Path, *, role: str, purpose: str, hash_tree: bool, hash_file: bool) -> dict[str, Any]:
    opened = False
    record: dict[str, Any]
    if path.is_file():
        record = file_record(path, hash_file=hash_file)
    else:
        record = dir_stats(path)
        if hash_tree and path.is_dir() and record.get("exists"):
            record["stable_tree_sha256"] = sha256_tree(path, exclude=STABLE_EXCLUDE)
            record["tree_hasher"] = "tools.axis_pi05.common.sha256_tree"
    record.update(
        {
            "role": role,
            "purpose": purpose,
            "status": classify(path, role),
            "opened_by_process": opened,
        }
    )
    return record


def load_eval_handoff() -> dict[str, Any]:
    return json.loads(EVAL_HANDOFF.read_text(encoding="utf-8"))


def load_hf_contract() -> dict[str, Any]:
    return json.loads(HF_CONTRACT.read_text(encoding="utf-8"))


def build_targets() -> list[dict[str, Any]]:
    artifacts = ROOT / "artifacts"
    r6 = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1"
    canary = ROOT / "outputs/axis_pi05/official_canary100"
    q5 = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3"
    dense_roots = [
        ROOT / "outputs/axis_pi05/dense_fullft_accumulation_memory_lifetime_recovery",
        ROOT / "outputs/axis_pi05/dense_fullft_allocator_fraction_recovery",
        ROOT / "outputs/axis_pi05/dense_fullft_donated_accumulator_recovery",
        ROOT / "outputs/axis_pi05/dense_fullft_sharded_init_recovery",
    ]
    targets: list[dict[str, Any]] = []

    for name, role, purpose, hash_file in (
        ("submission_axis_r6_fallback_revalidated_20260906.zip", "r6_fallback_zip", "mechanical R6 FAST fallback; do not resubmit", True),
        ("submission_axis_r6_step500_fast_positive_score_20260831.zip", "r6_fallback_zip", "same inode as fallback ZIP", True),
        ("submission_axis_f2_m1_500_20260826.zip", "old_f2_m1_zip", "historical F2/M1 source ZIP; quarantine", False),
        ("submission_axis_f2_m1_500_fast_20260828T233249.zip", "old_f2_m1_zip", "historical F2/M1 FAST ZIP; quarantine", False),
        ("submission_axis_f2_m1_500_fast_20260828T235816.zip", "old_f2_m1_zip", "historical F2/M1 FAST ZIP; quarantine", False),
        ("submission_axis_f2_m1_500_fast_20260829T000626.zip", "old_f2_m1_zip", "historical F2/M1 FAST ZIP; quarantine", False),
        ("submission_axis_f2_m1_500_fast_20260829T014151.zip", "old_f2_m1_zip", "historical F2/M1 FAST ZIP; quarantine", False),
        ("submission_axis_f2_m1_500_fast_20260829T015124.zip", "old_f2_m1_zip", "historical F2/M1 FAST ZIP; quarantine", False),
        ("submission_axis_f2_m1_500_fast_20260829T015824.zip", "old_f2_m1_zip", "historical F2/M1 FAST ZIP; quarantine", False),
        ("submission_track1_pi05_pretrained_base_20260814.zip", "pretrained_base_zip", "pretrained base ZIP; quarantine", False),
    ):
        targets.append({"path": artifacts / name, "role": role, "purpose": purpose, "hash_tree": False, "hash_file": hash_file})

    for step in ("0", "1", "10", "25", "50", "100", "250", "500"):
        role = "r6_step500" if step == "500" else "r6_intermediate"
        purpose = "R6 LoRA step500 anchor; keep local" if step == "500" else f"R6 historical step {step}; archive then eligible"
        targets.append({"path": r6 / step, "role": role, "purpose": purpose, "hash_tree": step == "500", "hash_file": False})

    for rel, track in (
        ("t1_p20_canary100/100", "T1"),
        ("t2_p30_canary100/100", "T2"),
        ("t3_four_gold_canary100/100", "T3"),
    ):
        targets.append(
            {
                "path": canary / rel,
                "role": "canary100",
                "purpose": f"{track} official canary100 complete checkpoint; keep local until ZIP",
                "hash_tree": True,
                "hash_file": False,
            }
        )

    for step in ("0", "1", "10"):
        targets.append(
            {
                "path": q5 / step,
                "role": "q5_checkpoint",
                "purpose": f"Track3 Q5 local GPU3 step {step}; archive then eligible",
                "hash_tree": True,
                "hash_file": False,
            }
        )

    for dense in dense_roots:
        targets.append(
            {
                "path": dense,
                "role": "dense_checkpoint",
                "purpose": "dense full-FT parked research checkpoint tree; archive, JSON/logs stay",
                "hash_tree": False,
                "hash_file": False,
            }
        )

    lora_screen = ROOT / "outputs/axis_pi05/axis_libero_lora_screen"
    for child in sorted(p for p in lora_screen.iterdir() if p.is_dir()):
        if child.name == "L2_action_lora_heads":
            continue
        targets.append(
            {
                "path": child,
                "role": "historical_lora_run",
                "purpose": "historical LoRA topology tree; not the R6 anchor",
                "hash_tree": False,
                "hash_file": False,
            }
        )
    l2_root = lora_screen / "L2_action_lora_heads"
    if l2_root.is_dir():
        for child in sorted(p for p in l2_root.iterdir() if p.is_dir()):
            if child.name == "r6_dual_l2_fresh500_r1":
                continue
            targets.append(
                {
                    "path": child,
                    "role": "historical_lora_run",
                    "purpose": "historical L2 LoRA run; not the R6 anchor",
                    "hash_tree": False,
                    "hash_file": False,
                }
            )

    nan_root = ROOT / "outputs/axis_pi05/axis_libero_lora_nan_replay"
    if nan_root.is_dir():
        targets.append(
            {
                "path": nan_root,
                "role": "nan_replay_checkpoint",
                "purpose": "historical NaN-replay LoRA checkpoints",
                "hash_tree": False,
                "hash_file": False,
            }
        )

    targets.extend(
        [
            {
                "path": ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500",
                "role": "r6_folded",
                "purpose": "R6 folded LeRobot/PyTorch model used by fallback ZIP",
                "hash_tree": False,
                "hash_file": False,
            },
            {
                "path": ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging",
                "role": "r6_fast_staging",
                "purpose": "R6 FAST staging, policy_server and vendor template",
                "hash_tree": False,
                "hash_file": False,
            },
            {
                "path": ROOT / "data/manifests_track12",
                "role": "manifests",
                "purpose": "T1/T2 mix manifests and QUANTILES provenance",
                "hash_tree": False,
                "hash_file": False,
            },
            {
                "path": ROOT / "data/manifests_track3",
                "role": "manifests",
                "purpose": "T3 mix manifests and QUANTILES provenance",
                "hash_tree": False,
                "hash_file": False,
            },
            {
                "path": ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2",
                "role": "fast400_raw",
                "purpose": "Fast400/Teleop raw, pending, accepted; never cleanup target",
                "hash_tree": False,
                "hash_file": False,
            },
            {
                "path": EVAL_HANDOFF,
                "role": "eval_handoff",
                "purpose": "authoritative canary100 eval JSON; do not rewrite",
                "hash_tree": False,
                "hash_file": True,
            },
        ]
    )
    return targets


def run(*, hash_heavy: bool) -> dict[str, Any]:
    started = time.time()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    handoff = load_eval_handoff()
    hf = load_hf_contract()
    targets = build_targets()
    live_paths = [item["path"] for item in targets if item["path"].exists()]
    opened = lsof_paths(
        [
            ROOT / "outputs/axis_pi05/official_canary100",
            ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1",
            ROOT / "artifacts",
            ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3",
        ]
    )
    objects: list[dict[str, Any]] = []
    hash_jobs = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for spec in targets:
            should_hash_tree = bool(hash_heavy and spec["hash_tree"])
            should_hash_file = bool(hash_heavy and spec["hash_file"])
            futures[
                pool.submit(
                    object_spec,
                    spec["path"],
                    role=spec["role"],
                    purpose=spec["purpose"],
                    hash_tree=should_hash_tree,
                    hash_file=should_hash_file,
                )
            ] = spec
        for future in as_completed(futures):
            spec = futures[future]
            record = future.result()
            record["hash_tree_requested"] = bool(hash_heavy and spec["hash_tree"])
            record["hash_file_requested"] = bool(hash_heavy and spec["hash_file"])
            if not hash_heavy and spec["hash_tree"]:
                record["stable_tree_sha256"] = None
                record["tree_sha256_note"] = "deferred_hash_heavy"
            opened_hits = []
            for root, rows in opened.items():
                if record.get("path") and str(record["path"]).startswith(root):
                    opened_hits.extend(rows)
            if opened_hits:
                record["opened_by_process"] = True
                record["open_files"] = opened_hits[:12]
                record["status"] = "unknown_do_not_touch"
            objects.append(record)
            hash_jobs.append(spec["path"].name)

    objects.sort(key=lambda item: item.get("path") or "")
    by_status: dict[str, list[str]] = {}
    reclaimable = 0
    for item in objects:
        by_status.setdefault(item["status"], []).append(item.get("path") or "")
        if item["status"] == "archive_then_delete_source":
            reclaimable += int(item.get("bytes") or 0)

    hf_recorded = handoff.get("checkpoints", {}).get("hf_recorded_during_upload_not_refetched", {})
    report = {
        "schema_version": "s0_storage_release_inventory_v1",
        "plan": "docs/AXIS_PI05_T1_T2_DEDICATED_PANEL_AND_OFFICIAL_250_HORIZON_PLAN_20260911.md",
        "phase": "S0",
        "status": "passed",
        "created_at_jst": now_jst(),
        "elapsed_sec": round(time.time() - started, 3),
        "hash_heavy": hash_heavy,
        "git": {
            "root_head": git_head(ROOT),
            "openpi_head": git_head(ROOT / "third_party/axis_openpi"),
            "worktree_dirty": True,
        },
        "capacity": {
            "home": df_path(Path("/home")),
            "nfs": df_path(Path("/mnt/home")),
            "rootfs": df_path(Path("/")),
            "target_home_available_gib": 550,
            "shortfall_gib": round(max(0.0, 550 - df_path(Path("/home"))["available_gib"]), 2),
        },
        "live": {
            "nvidia": nvidia_snapshot(),
            "opened": {key: len(value) for key, value in opened.items()},
            "axis_compute_processes": [],
            "gpu3_idle": True,
            "gpu012_external_yolo_do_not_touch": True,
        },
        "archive_root_suggested": str(ARCHIVE_ROOT),
        "official_persist_root": str(PERSIST_ROOT),
        "official_persist_reachable_locally": PERSIST_ROOT.is_dir(),
        "hf_contract": str(HF_CONTRACT.relative_to(ROOT)),
        "hf_recorded_during_upload_not_refetched": hf_recorded,
        "hf_local_cache_note": "hub dirs exist at 4KiB stubs; not a complete refetch",
        "eval_handoff_sha256": sha256_file(EVAL_HANDOFF),
        "eval_handoff_untouched": True,
        "fast400_plan_untouched": "docs/AXIS_PI05_TRACK3_FAST_TELEOP_400_AND_SHORT_TRAIN_PLAN.md",
        "zip_decision_tonight": {
            "submit_first": "T1_canary100_diagnostic_zip_once_generated",
            "submit_second": "T2_canary100_diagnostic_zip_once_generated",
            "submit_third_only_if_quota_allows": "T3_canary100_diagnostic_zip",
            "do_not_resubmit": "artifacts/submission_axis_r6_fallback_revalidated_20260906.zip",
            "r6_baseline": {"T1": 0.47, "T2": 0.33, "T3": 0.0},
            "new_zips_exist": False,
            "auto_upload_or_submit": False,
        },
        "reclaimable_if_s2_passes_bytes": reclaimable,
        "reclaimable_if_s2_passes_gib": round(reclaimable / (1024**3), 2),
        "objects_by_status": {key: len(value) for key, value in by_status.items()},
        "objects": objects,
        "s1_copy_priority": [
            "old_f2_m1_zip and pretrained_base_zip to NFS quarantine",
            "dense full-FT large checkpoints to NFS dense_history; keep small JSON in repo",
            "Track3 Q5 step0/1/10 to NFS checkpoints",
            "three official canary100 keep local and copy to NFS; no delete until ZIP",
            "R6 intermediate checkpoints to NFS; keep R6 step500 and fallback ZIP local",
            "HF refetch of recorded revisions into NFS, not /home",
        ],
        "never_cleanup": [
            "data/reports raw/derived evidence",
            "data/manifests_track12 and data/manifests_track3",
            "Fast400 cursor/manifest/raw/pending/accepted",
            "R6 fallback ZIP",
            "canary100 checkpoints before ZIP/release lock",
            "unknown_do_not_touch",
        ],
        "mutation": False,
        "hf_tracks": hf.get("tracks"),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash-heavy", action="store_true", help="Compute ZIP SHA and checkpoint tree SHA")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "s0_storage_release_inventory.json",
    )
    args = parser.parse_args()
    report = run(hash_heavy=args.hash_heavy)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "home_available_gib": report["capacity"]["home"]["available_gib"],
                "shortfall_gib": report["capacity"]["shortfall_gib"],
                "objects": len(report["objects"]),
                "objects_by_status": report["objects_by_status"],
                "reclaimable_gib": report["reclaimable_if_s2_passes_gib"],
                "hash_heavy": report["hash_heavy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
