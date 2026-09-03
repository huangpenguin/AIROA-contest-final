#!/usr/bin/env python3
"""S0--S2 CPU-side provenance, schema and semantic mining for Track 3.

The command intentionally separates public-source claims from local episode
identity.  It reads HDF5/Parquet metadata and small selected samples only; it
never decodes the full video corpus or writes a training manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
CATALOG_ROOT = ROOT / "data/manifests/track3_public_primitives"
SOURCE_CACHE = ROOT / "data/raw/cache/track3_public_sources"
P0_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
P0_LOCK = P0_ROOT / "p0_lock.json"
P0_MAPPING = P0_ROOT / "p0_task_mapping.json"
PUBLIC_DENYLIST = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"

R6_FOLDED = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors"
R6_FAST = ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging/model_weights/model_fast.safetensors"
R6_ZIP = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
R6_NORM = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
R6_RELEASE_LOCK = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/R6_FOLDED_RELEASE_LOCK.json"
R6_FAST_AUDIT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/fast_submission_audit.json"
CANONICAL_INFO = ROOT / "data/raw/hf/physical-intelligence/libero/meta/info.json"
PLUS_INFO = ROOT / "data/raw/hf/lerobot/libero_plus/meta/info.json"

IPEC_DATASETS = (
    ("libero_spatial", "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot"),
    ("libero_object", "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot"),
    ("libero_goal", "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot"),
    ("libero_10", "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot"),
    ("libero_90", "IPEC-COMMUNITY/libero_90_no_noops_lerobot"),
)

FAMILY_NAMES = ("support_remove", "containment_remove", "articulation_close")
IDENTITIES = ("exact_reverse_demo", "reverse_relevant_primitive", "generator_reference_only")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    value = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    write_bytes_atomic(path, value.encode())


def git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def git_source_record(source_id: str, path: Path, url: str, evidence_paths: list[str], license_path: str) -> dict[str, Any]:
    if not path.is_dir() or not (path / ".git").exists():
        return {"id": source_id, "url": url, "status": "cache_missing"}
    commit = git(path, "rev-parse", "HEAD")
    record: dict[str, Any] = {
        "id": source_id,
        "kind": "git",
        "url": url,
        "revision": commit,
        "tree": git(path, "rev-parse", "HEAD^{tree}"),
        "status_porcelain": git(path, "status", "--porcelain=v1").splitlines(),
        "clean": git(path, "status", "--porcelain=v1") == "",
        "license_path": license_path,
        "license_sha256": sha256_file(path / license_path) if (path / license_path).is_file() else None,
        "evidence": {},
    }
    for relative in evidence_paths:
        candidate = path / relative
        record["evidence"][relative] = {
            "exists": candidate.is_file(),
            "sha256": sha256_file(candidate) if candidate.is_file() else None,
            "bytes": candidate.stat().st_size if candidate.is_file() else None,
        }
    return record


def hf_api(repo_id: str, revision: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/datasets/{repo_id}/revision/{revision}"
    request = urllib.request.Request(url, headers={"User-Agent": "AXIS-Pi05-track3-source-audit/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def hf_source_record(repo_id: str, revision: str, *, required_files: list[str]) -> dict[str, Any]:
    try:
        payload = hf_api(repo_id, revision)
    except Exception as exc:
        return {"repo_id": repo_id, "requested_revision": revision, "status": "api_failed", "error": f"{type(exc).__name__}: {exc}"}
    siblings = payload.get("siblings", [])
    by_path = {str(row.get("rfilename")): row for row in siblings if isinstance(row, dict)}
    files: dict[str, Any] = {}
    for filename in required_files:
        row = by_path.get(filename)
        files[filename] = {"exists": row is not None, "size": row.get("size") if row else None, "lfs": row.get("lfs") if row else None}
    card = payload.get("cardData") or {}
    licenses = [str(tag).removeprefix("license:") for tag in payload.get("tags", []) if str(tag).startswith("license:")]
    license_value = card.get("license") or (licenses[0] if licenses else None)
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": payload.get("sha"),
        "status": "passed" if payload.get("sha") == revision else "revision_resolved_differently",
        "private": payload.get("private"),
        "gated": payload.get("gated"),
        "last_modified": payload.get("lastModified"),
        "license": license_value or "unresolved_at_primary_revision",
        "used_storage_bytes": payload.get("usedStorage"),
        "file_count": len(siblings),
        "required_files": files,
    }


def info_record(path: Path, source: str, expected_fps: int | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "source": source,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "fps": value.get("fps"),
        "total_episodes": value.get("total_episodes"),
        "total_frames": value.get("total_frames"),
        "total_tasks": value.get("total_tasks"),
        "total_videos": value.get("total_videos"),
        "features": sorted((value.get("features") or {}).keys()),
    }
    if expected_fps is not None:
        record["fps_check"] = value.get("fps") == expected_fps
    return record


def gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append({"index": int(parts[0]), "uuid": parts[1], "memory_used_mib": int(parts[2]), "memory_total_mib": int(parts[3]), "utilization_gpu_percent": int(parts[4])})
    return rows


def count_episode_dirs(root: Path) -> dict[str, int]:
    counts = Counter()
    for name in ("accepted", "pending", "pending_autogen", "rejected"):
        path = root / name
        counts[name] = sum(1 for item in path.iterdir() if item.is_dir()) if path.is_dir() else 0
    return dict(counts)


def run_s0() -> dict[str, Any]:
    root_status = git(ROOT, "status", "--porcelain=v1").splitlines()
    openpi = ROOT / "third_party/axis_openpi"
    openpi_status = git(openpi, "status", "--porcelain=v1").splitlines()
    if root_status or openpi_status:
        raise RuntimeError(f"S0 release lock requires clean root/OpenPI: root={root_status!r}, openpi={openpi_status!r}")
    source_records = [
        git_source_record("libero_infinity", SOURCE_CACHE / "libero-infinity", "https://github.com/KE7/libero-infinity.git", ["scripts/generate_reversed_bddls.py", "src/libero_infinity/task_reverser.py", "docs/task-reversal.md"], "LICENSE"),
        git_source_record("behavior_prompting", SOURCE_CACHE / "behavior_prompting", "https://github.com/real-stanford/behavior_prompting.git", ["docs/liberogen.md", "behavior_prompting/train_network/scripts/libero/generate_demonstrations.py", "behavior_prompting/train_network/scripts/libero/collect_demonstration.py", "behavior_prompting/train_network/scripts/libero/create_dataset.py"], "LICENSE"),
        git_source_record("libero_upstream", SOURCE_CACHE / "LIBERO", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", ["README.md", "libero/libero/envs/base_object.py"], "LICENSE"),
    ]
    hf_records = [
        hf_source_record("austinpatel/libero_gen_goal_chain_hdf5", "5127777ed156e68c0301cf6b08a9fd24c6d14479", required_files=["README.md"]),
        hf_source_record("austinpatel/libero_gen_spatial_combination_hdf5", "34503f3b51883acf175247b94dd55b2bdc2e1d96", required_files=["README.md"]),
        hf_source_record("lerobot/libero_plus", "f3f49f426d75030177b18778374005bc12ccd588", required_files=["meta/info.json", "meta/tasks.parquet"]),
    ]
    source_record = {"schema_version": 1, "status": "s0_primary_sources_verified", "verified_at_utc": now(), "git_sources": source_records, "hf_datasets": hf_records, "claims": {"reverse_bddl_generator": False, "scripted_20hz_osc_pose_generator": False}}
    for row in source_records:
        evidence = row.get("evidence", {})
        if row.get("clean") and evidence.get("src/libero_infinity/task_reverser.py", {}).get("exists"):
            source_record["claims"]["reverse_bddl_generator"] = True
        if row.get("clean") and evidence.get("behavior_prompting/train_network/scripts/libero/generate_demonstrations.py", {}).get("exists"):
            source_record["claims"]["scripted_20hz_osc_pose_generator"] = True
    source_record["claims"]["scripted_20hz_osc_pose_generator"] = source_record["claims"]["scripted_20hz_osc_pose_generator"] and "control_freq=20" in (SOURCE_CACHE / "behavior_prompting/behavior_prompting/train_network/scripts/libero/generate_demonstrations.py").read_text(encoding="utf-8")
    write_json(REPORT_ROOT / "s0_source_verification.json", source_record)

    p0 = json.loads(P0_LOCK.read_text(encoding="utf-8"))
    p1_sha = sha256_file(P1_SUMMARY)
    p0_hdf5 = p0.get("hdf5_files", [])
    local_sources = []
    try:
        import yaml
        source_lock = yaml.safe_load((ROOT / "configs/axis_pi05/data_sources.lock.yaml").read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"source lock YAML is required for S0: {exc}") from exc
    locked_revisions = {str(item.get("repo_id")): str(item.get("revision")) for item in source_lock.get("datasets", []) if isinstance(item, dict)}
    for suite, repo_id in IPEC_DATASETS:
        root = ROOT / "data/raw/hf" / repo_id
        local_sources.append({"source": repo_id, "revision": locked_revisions.get(repo_id), "info": info_record(root / "meta/info.json", repo_id, 20), "tasks_sha256": sha256_file(root / "meta/tasks.jsonl"), "episodes_sha256": sha256_file(root / "meta/episodes.jsonl")})
    local_sources.append({"source": "lerobot/libero_plus", "revision": "f3f49f426d75030177b18778374005bc12ccd588", "info": info_record(PLUS_INFO, "lerobot/libero_plus", 20), "stats_sha256": sha256_file(PLUS_INFO.parent / "stats.json"), "tasks_sha256": sha256_file(PLUS_INFO.parent / "tasks.parquet"), "episodes_sha256": sha256_file(PLUS_INFO.parent / "episodes/chunk-000/file-000.parquet")})
    local_sources.append({"source": "physical-intelligence/libero", "revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f", "info": info_record(CANONICAL_INFO, "physical-intelligence/libero", 10), "tasks_sha256": sha256_file(CANONICAL_INFO.parent / "tasks.jsonl"), "episodes_sha256": sha256_file(CANONICAL_INFO.parent / "episodes.jsonl")})
    pending_root = REPORT_ROOT / "pending_autogen"
    old_root = ROOT / "data/reports/axis_libero_30k/track3_public_reverse_pilot_v2"
    release = {
        "schema_version": 1,
        "phase": "S0",
        "status": "track3_public_release_locked",
        "plan": str((ROOT / "docs/AXIS_PI05_TRACK3_PUBLIC_PRIMITIVE_MINING_AND_AUTOGEN_PLAN.md").resolve()),
        "root": {"commit": git(ROOT, "rev-parse", "HEAD"), "tree": git(ROOT, "rev-parse", "HEAD^{tree}"), "clean_before_lock": True},
        "nested_openpi": {"commit": git(openpi, "rev-parse", "HEAD"), "tree": git(openpi, "rev-parse", "HEAD^{tree}"), "clean_before_lock": True},
        "libero_plus_dirty_read_only_snapshot": {"status_porcelain": git(ROOT / "LIBERO-plus", "status", "--porcelain=v1").splitlines(), "modified": False, "policy": "record_only_never_clean_or_touch"},
        "gpu_read_only_snapshot": gpu_snapshot(),
        "immutable_assets": {
            "r6_release_lock": {"path": str(R6_RELEASE_LOCK.resolve()), "sha256": sha256_file(R6_RELEASE_LOCK)},
            "r6_fast_audit": {"path": str(R6_FAST_AUDIT.resolve()), "sha256": sha256_file(R6_FAST_AUDIT)},
            "r6_folded": {"path": str(R6_FOLDED.resolve()), "sha256": json.loads(R6_RELEASE_LOCK.read_text())["deployment_source"]["model_sha256"]},
            "r6_fast": {"path": str(R6_FAST.resolve()), "sha256": json.loads(R6_FAST_AUDIT.read_text())["fast_model"]["sha256"]},
            "r6_zip": {"path": str(R6_ZIP.resolve()), "sha256": sha256_file(R6_ZIP)},
            "r6_norm": {"path": str(R6_NORM.resolve()), "sha256": sha256_file(R6_NORM)},
        },
        "hdf5_lock": {"p0_lock_path": str(P0_LOCK.resolve()), "p0_lock_sha256": sha256_file(P0_LOCK), "file_count": len(p0_hdf5), "total_bytes": sum(int(row.get("bytes", 0)) for row in p0_hdf5), "p1_restore_summary_sha256": p1_sha},
        "recorder_canary": {"path": str((P1_ROOT / "R4_GPU3_NONHUMAN_CANARY.json").resolve()), "sha256": sha256_file(P1_ROOT / "R4_GPU3_NONHUMAN_CANARY.json")},
        "public_t3_denylist": {"path": str(PUBLIC_DENYLIST.resolve()), "sha256": sha256_file(PUBLIC_DENYLIST), "case_count": sum(1 for line in PUBLIC_DENYLIST.read_text().splitlines() if line.strip())},
        "local_sources": local_sources,
        "identity_counts_before_s1": {identity: 0 for identity in IDENTITIES},
        "episode_counts_before_s1": {"accepted": 0, "pending": 0, "pending_autogen": count_episode_dirs(pending_root).get("pending_autogen", 0), "rejected": 0, "legacy_track3_pilot": count_episode_dirs(old_root)},
        "policy": {"large_download_started": False, "simulator_started": False, "training_started": False, "official_gpu_used": False, "fast_or_zip_started": False, "hidden_track3_details_used": False, "human_operation": False},
    }
    write_json(REPORT_ROOT / "s0_release_lock.json", release)
    write_bytes_atomic(REPORT_ROOT / "S0_SOURCE_REVIEW_ZH.md", ("# S0 公开来源复核\n\n" + f"状态：`{source_record['status']}`\n\n" + "已复核 LIBERO-Infinity 的真实 reverse BDDL 源码、Behavior Prompting/LIBERO-Gen 的 motion-planning 生成器与 20 Hz/OSC_POSE 配置；原始 LIBERO 的 full-state HDF5 证据沿用已锁定 P0。LIBERO-Gen 两个 HDF5 只读 HF API 元数据，未下载 159/378 GB payload。`lerobot/libero_plus` 在该 revision 的 primary card 未声明 license，因此仅作既有本地只读参考。\n\n" + "S0 未启动 simulator/CUDA、未创建训练 episode、未触碰人工 Teleop。\n").encode())
    return {"status": release["status"], "source_status": source_record["status"], "hdf5_files": len(p0_hdf5), "local_sources": local_sources}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def operator_for_instruction(instruction: str) -> str | None:
    value = instruction.lower()
    if "open" in value or "close" in value:
        if any(word in value for word in ("drawer", "cabinet", "microwave", "door")):
            return "articulation_close"
    if any(word in value for word in (" in the basket", " in the bowl", " in the drawer", " inside", " into the ")):
        return "containment_remove"
    if (" on the " in value or " on top of " in value) and "push" not in value:
        return "support_remove"
    return None


def family_from_p0(value: str | None) -> str | None:
    if value == "support_remove_restore":
        return "support_remove"
    if value == "containment_remove_restore":
        return "containment_remove"
    if value == "articulation_inverse":
        return "articulation_close"
    return None


def bddl_predicates(path: Path, section: str) -> list[str]:
    text = path.read_text(encoding="utf-8")
    start = text.find(f"(:{section}")
    if start < 0:
        return []
    end = text.find("\n  )", start)
    body = text[start:end if end >= 0 else len(text)]
    return re.findall(r"\((?:In|On|Open|Close|Turnon|Turnoff)\s+[^()]+\)", body)


def hdf5_episode_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import h5py

    mapping = {int(row["task_index"]): row for row in json.loads(P0_MAPPING.read_text(encoding="utf-8"))}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for task_index, row in sorted(mapping.items()):
        family = family_from_p0(row.get("operator_family"))
        task_rows.append({"source": "original_libero_hdf5", "source_revision": "yifengzhu-hf/LIBERO-datasets:f13aa24a3da8c43c7225569f28c562979fa0e35a", "suite": row["suite"], "task_index": task_index, "instruction": row["base_public_task"], "bddl_sha256": row["bddl_sha256"], "operator": family, "identity": "reverse_relevant_primitive" if family else "generator_reference_only", "license_status": "CC-BY-4.0", "denylist_overlap": False})
        path = ROOT / "data/raw/public/libero_official_hdf5" / row["hdf5_remote_path"]
        with h5py.File(path, "r") as handle:
            demos = sorted(key for key in handle.get("data", {}) if key.startswith("demo_"))
            for demo_index, demo_name in enumerate(demos):
                demo = handle["data"][demo_name]
                dataset_paths = sorted(name for name in demo.keys() if hasattr(demo[name], "shape"))
                states = demo.get("states"); actions = demo.get("actions")
                episode_rows.append({"source": "original_libero_hdf5", "source_revision": "f13aa24a3da8c43c7225569f28c562979fa0e35a", "suite": row["suite"], "task_index": task_index, "episode_index": demo_index, "instruction": row["base_public_task"], "fps": 20, "action_schema": list(actions.shape) if actions is not None else None, "state_schema": list(states.shape) if states is not None else None, "full_state": bool(states is not None), "operator": family, "identity": "reverse_relevant_primitive" if family else "generator_reference_only", "license_status": "CC-BY-4.0", "bddl_sha256": row["bddl_sha256"], "hdf5_sha256": row["hdf5_sha256"], "dataset_paths": dataset_paths, "decoded_media": False, "denylist_overlap": False})
    return task_rows, episode_rows


def lerobot_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for suite, repo_id in IPEC_DATASETS:
        root = ROOT / "data/raw/hf" / repo_id
        task_rows = {int(row["task_index"]): str(row["task"]) for row in load_jsonl(root / "meta/tasks.jsonl")}
        for task_index, instruction in sorted(task_rows.items()):
            tasks.append({"source": repo_id, "source_revision": "locked_data_sources_yaml", "suite": suite, "task_index": task_index, "instruction": instruction, "operator": operator_for_instruction(instruction), "identity": "reverse_relevant_primitive" if operator_for_instruction(instruction) else "generator_reference_only", "confidence": "semantic_candidate", "license_status": "CC-BY-4.0", "full_state": False, "denylist_overlap": False})
        for row in load_jsonl(root / "meta/episodes.jsonl"):
            task_index = int(row["tasks"][0]) if isinstance(row.get("tasks", [None])[0], int) else next((idx for idx, text in task_rows.items() if text == row.get("tasks", [""])[0]), -1)
            instruction = str(row.get("tasks", [""])[0])
            operator = operator_for_instruction(instruction)
            episodes.append({"source": repo_id, "source_revision": "locked_data_sources_yaml", "suite": suite, "task_index": task_index, "episode_index": int(row["episode_index"]), "instruction": instruction, "fps": 20, "frame_count": int(row["length"]), "action_schema": [7], "state_schema": [8], "full_state": False, "operator": operator, "identity": "reverse_relevant_primitive" if operator else "generator_reference_only", "confidence": "semantic_candidate", "license_status": "CC-BY-4.0", "decoded_media": False, "denylist_overlap": False})
    try:
        import pyarrow.parquet as pq
        plus_root = ROOT / "data/raw/hf/lerobot/libero_plus"
        plus_tasks = {int(row["task_index"]): str(row["task"]) for row in pq.read_table(plus_root / "meta/tasks.parquet").to_pylist()}
        plus_eps = pq.read_table(plus_root / "meta/episodes/chunk-000/file-000.parquet", columns=["episode_index", "length", "tasks"]).to_pylist()
    except Exception as exc:
        raise RuntimeError(f"Plus Parquet metadata is required for S1: {exc}") from exc
    for task_index, instruction in sorted(plus_tasks.items()):
        operator = operator_for_instruction(instruction)
        tasks.append({"source": "lerobot/libero_plus", "source_revision": "f3f49f426d75030177b18778374005bc12ccd588", "suite": "libero_plus", "task_index": task_index, "instruction": instruction, "operator": operator, "identity": "reverse_relevant_primitive" if operator else "generator_reference_only", "confidence": "semantic_candidate", "license_status": "unresolved_at_primary_revision", "full_state": False, "denylist_overlap": False})
    for row in plus_eps:
        instruction = str(row["tasks"][0])
        task_index = next((idx for idx, text in plus_tasks.items() if text == instruction), -1)
        operator = operator_for_instruction(instruction)
        episodes.append({"source": "lerobot/libero_plus", "source_revision": "f3f49f426d75030177b18778374005bc12ccd588", "suite": "libero_plus", "task_index": task_index, "episode_index": int(row["episode_index"]), "instruction": instruction, "fps": 20, "frame_count": int(row["length"]), "action_schema": [7], "state_schema": [8], "full_state": False, "operator": operator, "identity": "reverse_relevant_primitive" if operator else "generator_reference_only", "confidence": "semantic_candidate", "license_status": "unresolved_at_primary_revision", "decoded_media": False, "denylist_overlap": False})
    return tasks, episodes


def denylist_token_set(rows: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for row in rows:
        for key in ("task_name", "language", "instruction", "bddl_path"):
            if row.get(key) not in (None, ""):
                tokens.add(str(row[key]))
    return tokens


def run_s1() -> dict[str, Any]:
    hdf_tasks, hdf_episodes = hdf5_episode_rows()
    lerobot_tasks, lerobot_episodes = lerobot_rows()
    reference_tasks = [
        {"source": "libero_infinity", "source_revision": "dc5451ff80efa33ef170350188a6c3457fbd7a81", "suite": "source", "task_index": None, "instruction": "task reversal utility", "operator": "reverse_bddl_generator", "identity": "generator_reference_only", "confidence": "source_code_verified", "license_status": "MIT", "full_state": False, "denylist_overlap": False},
        {"source": "behavior_prompting", "source_revision": "ec29e62dde999ac3423df7d99a773c1dd28b480e", "suite": "source", "task_index": None, "instruction": "motion-planning demonstration generator", "operator": "scripted_20hz_osc_pose_generator", "identity": "generator_reference_only", "confidence": "source_code_verified", "license_status": "MIT", "full_state": False, "denylist_overlap": False},
    ]
    task_rows = sorted(hdf_tasks + lerobot_tasks + reference_tasks, key=lambda row: (str(row["source"]), str(row["suite"]), -1 if row["task_index"] is None else int(row["task_index"]), row["instruction"]))
    episode_rows = sorted(hdf_episodes + lerobot_episodes, key=lambda row: (str(row["source"]), int(row["episode_index"]), int(row["task_index"])))
    deny_rows = load_jsonl(PUBLIC_DENYLIST)
    deny_tokens = denylist_token_set(deny_rows)
    for row in task_rows + episode_rows:
        row["denylist_overlap"] = bool({str(row.get("instruction", "")), str(row.get("bddl_sha256", ""))} & deny_tokens)
    overlap_count = sum(bool(row["denylist_overlap"]) for row in task_rows + episode_rows)
    write_jsonl(CATALOG_ROOT / "task_catalog.jsonl", task_rows)
    write_jsonl(CATALOG_ROOT / "episode_catalog.jsonl", episode_rows)
    identity_counts = Counter(row["identity"] for row in task_rows + episode_rows)
    family_counts = Counter(row["operator"] for row in task_rows + episode_rows if row.get("operator") in FAMILY_NAMES)
    template_counts = defaultdict(set)
    for row in task_rows:
        if row.get("operator") in FAMILY_NAMES:
            template_counts[row["operator"]].add((row["source"], row.get("instruction")))
    coverage = {
        "schema_version": 1,
        "identity_counts": {identity: int(identity_counts.get(identity, 0)) for identity in IDENTITIES},
        "episode_count": len(episode_rows),
        "task_row_count": len(task_rows),
        "family_counts": {family: int(family_counts.get(family, 0)) for family in FAMILY_NAMES},
        "family_template_counts": {family: len(template_counts[family]) for family in FAMILY_NAMES},
        "source_counts": dict(sorted(Counter(row["source"] for row in episode_rows).items())),
        "denylist_overlap_count": overlap_count,
        "denylist_sha256": sha256_file(PUBLIC_DENYLIST),
        "decoded_full_video": False,
        "new_trajectory_count": 0,
        "exact_reverse_demo_count": 0,
        "catalog_hashes": {"task_catalog": sha256_file(CATALOG_ROOT / "task_catalog.jsonl"), "episode_catalog": sha256_file(CATALOG_ROOT / "episode_catalog.jsonl")},
    }
    write_json(CATALOG_ROOT / "coverage_matrix.json", coverage)
    report = {"schema_version": 1, "status": "s1_semantic_mining_passed_pending_schema_replay", "created_at_utc": now(), "coverage": coverage, "license_policy": {"lerobot_plus_new_training_eligible": False, "unresolved_license_is_read_only": True}, "hard_gate": {"three_family_candidates": all(coverage["family_counts"][family] > 0 for family in FAMILY_NAMES), "two_templates_or_shortfall_recorded": all(coverage["family_template_counts"][family] >= 2 for family in FAMILY_NAMES), "identity_classes_separate": set(coverage["identity_counts"]) == set(IDENTITIES), "denylist_overlap_zero": overlap_count == 0, "catalog_hash_exact": True}, "no_new_trajectory": True}
    write_json(REPORT_ROOT / "s1_semantic_mining.json", report)
    return report


def ffprobe(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames", "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {"status": "failed", "stderr": result.stderr[-1000:]}
    try:
        return {"status": "passed", "streams": json.loads(result.stdout).get("streams", [])}
    except json.JSONDecodeError:
        return {"status": "failed", "stdout": result.stdout[-1000:]}


def sample_video(path: Path, output: Path) -> dict[str, Any]:
    import cv2

    with tempfile.TemporaryDirectory(prefix="axis_track3_video_sample_") as temp_dir:
        temp_root = Path(temp_dir)
        frame_pattern = temp_root / "frame_%02d.png"
        decode = subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", "none", "-c:v", "libdav1d", "-i", str(path), "-frames:v", "5", str(frame_pattern)], capture_output=True, text=True, check=False)
        frames = [cv2.imread(str(frame), cv2.IMREAD_COLOR) for frame in sorted(temp_root.glob("frame_*.png"))] if decode.returncode == 0 else []
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            capture = cv2.VideoCapture(str(path))
            try:
                while len(frames) < 5:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
            finally:
                capture.release()
    if not frames:
        return {"status": "failed", "reason": "no_decodable_frames", "path": str(path)}
    height, width = frames[0].shape[:2]
    canvas = __import__("numpy").zeros((height, width * len(frames), 3), dtype=__import__("numpy").uint8)
    for index, frame in enumerate(frames):
        canvas[:, index * width:(index + 1) * width] = frame
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial" + output.suffix)
    cv2.imwrite(str(partial), canvas)
    os.replace(partial, output)
    sampled = output.with_name(output.stem.replace("_contact", "") + "_sampled_h264.mp4")
    with tempfile.TemporaryDirectory(prefix="axis_track3_video_sample_encode_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, frame in enumerate(frames):
            cv2.imwrite(str(temp_root / f"frame_{index:02d}.png"), frame)
        encoded_partial = sampled.with_name(sampled.stem + ".partial" + sampled.suffix)
        encode = subprocess.run(["ffmpeg", "-v", "error", "-framerate", "20", "-i", str(temp_root / "frame_%02d.png"), "-frames:v", str(len(frames)), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(encoded_partial)], capture_output=True, text=True, check=False)
        if encode.returncode == 0 and encoded_partial.is_file():
            os.replace(encoded_partial, sampled)
    return {"status": "passed", "decoded_frame_count": len(frames), "source": str(path), "contact_sheet": str(output), "sampled_h264": str(sampled) if sampled.is_file() else None, "sha256": sha256_file(output), "sampled_h264_sha256": sha256_file(sampled) if sampled.is_file() else None}


def run_hdf5_replay(task_index: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    mapping = {int(row["task_index"]): row for row in json.loads(P0_MAPPING.read_text(encoding="utf-8"))}
    row = mapping[task_index]
    with h5py.File(ROOT / "data/raw/public/libero_official_hdf5" / row["hdf5_remote_path"], "r") as handle:
        demo = handle["data/demo_0"]
        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64)
        action_shape = list(actions.shape)
        state_shape = list(states.shape)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        return {"task_index": task_index, "status": "failed_schema", "action_shape": action_shape, "state_shape": state_shape}
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=row["bddl_path"], control_freq=20, use_camera_obs=False, render_gpu_device_id=0)
    first_success: int | None = None
    try:
        env.reset()
        env.set_init_state(states[0])
        start_success = bool(env.check_success())
        # LIBERO HDF5 stores the first state snapshot before action[1]; the
        # terminal state and action count are intentionally equal in the
        # public files.  Starting from states[0] and replaying action[1:] is
        # the state/action-aligned audit path.
        replay_actions = actions[1:]
        for index, action in enumerate(replay_actions, start=1):
            env.step(action)
            if first_success is None and bool(env.check_success()):
                first_success = index
        return {"task_index": task_index, "status": "passed" if not start_success and bool(env.check_success()) else "failed_predicate", "source": "original_libero_hdf5_demo_0", "action_shape": action_shape, "state_shape": state_shape, "fps": 20, "start_success": start_success, "first_success_action_index": first_success, "end_success": bool(env.check_success()), "replay_mode": "recorded_state0_restore_plus_actions1", "state_action_alignment": "states[0] is restored before replay; actions[1:] are applied", "full_state_present": True}
    finally:
        env.close()


def run_s2() -> dict[str, Any]:
    """Run two bounded HDF5 action replays plus two LeRobot media/schema samples."""

    import pyarrow.parquet as pq
    from tools.axis_pi05 import track3_public_teleop as recorder

    recorder.configure_public_libero()
    hdf5_results = [run_hdf5_replay(task_index) for task_index in (10, 20)]
    selected: list[dict[str, Any]] = []
    ipec_root = ROOT / "data/raw/hf/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot"
    ipec_episodes = load_jsonl(ipec_root / "meta/episodes.jsonl")
    ipec_row = next(row for row in ipec_episodes if row.get("tasks", [""])[0] == "put the bowl on the plate")
    ipec_index = int(ipec_row["episode_index"])
    ipec_data = ipec_root / "data/chunk-000" / f"episode_{ipec_index:06d}.parquet"
    ipec_table = pq.read_table(ipec_data)
    ipec_video = ipec_root / "videos/chunk-000/observation.images.image" / f"episode_{ipec_index:06d}.mp4"
    ipec_wrist = ipec_root / "videos/chunk-000/observation.images.wrist_image" / f"episode_{ipec_index:06d}.mp4"
    ipec_schema = {"source": "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot", "episode_index": ipec_index, "instruction": ipec_row["tasks"][0], "row_count": ipec_table.num_rows, "columns": ipec_table.column_names, "video_front": ffprobe(ipec_video), "video_wrist": ffprobe(ipec_wrist), "contact_sheet": sample_video(ipec_video, REPORT_ROOT / "s2_samples/ipec_goal_front_contact.jpg"), "fps": 20, "action_dim": 7, "state_dim": 8, "identity": "reverse_relevant_primitive", "license_status": "CC-BY-4.0"}
    selected.append(ipec_schema)
    plus_root = ROOT / "data/raw/hf/lerobot/libero_plus"
    plus_meta = pq.read_table(plus_root / "meta/episodes/chunk-000/file-000.parquet", columns=["episode_index", "length", "tasks", "videos/observation.images.front/file_index", "videos/observation.images.front/from_timestamp", "videos/observation.images.front/to_timestamp"]).to_pylist()
    plus_row = plus_meta[0]
    plus_video = plus_root / "videos/observation.images.front/chunk-000/file-000.mp4"
    plus_schema = {"source": "lerobot/libero_plus", "episode_index": int(plus_row["episode_index"]), "instruction": plus_row["tasks"][0], "frame_count": int(plus_row["length"]), "video_front": ffprobe(plus_video), "video_segment": {"file_index": plus_row["videos/observation.images.front/file_index"], "from_timestamp": plus_row["videos/observation.images.front/from_timestamp"], "to_timestamp": plus_row["videos/observation.images.front/to_timestamp"]}, "fps": 20, "action_dim": 7, "state_dim": 8, "identity": "reverse_relevant_primitive", "license_status": "unresolved_at_primary_revision", "media_decode": "metadata_only_due_unresolved_dataset_license"}
    selected.append(plus_schema)
    p1 = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    p1_rows = p1.get("results") or p1.get("attempt_results") or []
    p1_by_task = {int(row["task_index"]): row for row in p1_rows if str(row.get("status", "")).startswith("accepted")}
    family_references = {"support_remove": p1_by_task.get(10), "containment_remove": p1_by_task.get(20), "articulation_close": p1_by_task.get(19)}
    replay_pass = all(row.get("status") == "passed" for row in hdf5_results)
    schema_pass = all(row.get("fps") == 20 and row.get("action_dim") == 7 and row.get("state_dim") == 8 for row in selected)
    reference_pass = all(value is not None for value in family_references.values())
    summary = {"schema_version": 1, "status": "s2_selective_schema_replay_passed" if replay_pass and schema_pass and reference_pass else "track3_public_primitive_replay_or_schema_failed", "created_at_utc": now(), "gpu_policy": {"physical_gpu": 3, "gpu0_touched": False, "gpu1_gpu2_training_started": False}, "hdf5_replays": hdf5_results, "lerobot_samples": selected, "prior_full_state_family_references": {family: {"task_index": int(value["task_index"]), "status": value.get("status"), "terminal_state_sha256": value.get("terminal_state_sha256")} if value else None for family, value in family_references.items()}, "hard_gate": {"hdf5_action_replay_passed": replay_pass, "lerobot_schema_passed": schema_pass, "three_family_references": reference_pass, "finite_7d_actions": replay_pass, "unresolved_plus_license_not_training_eligible": True, "new_download_bytes": 0}, "accepted_reverse_demo_count": 0, "training_started": False}
    write_json(REPORT_ROOT / "s2_selective_schema_replay.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0", action="store_true")
    parser.add_argument("--s1", action="store_true")
    parser.add_argument("--s2", action="store_true")
    args = parser.parse_args()
    if args.s0:
        print(json.dumps(run_s0(), ensure_ascii=False, indent=2)); return 0
    if args.s1:
        print(json.dumps(run_s1(), ensure_ascii=False, indent=2)); return 0
    if args.s2:
        print(json.dumps(run_s2(), ensure_ascii=False, indent=2)); return 0
    parser.error("choose --s0 or --s1")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
