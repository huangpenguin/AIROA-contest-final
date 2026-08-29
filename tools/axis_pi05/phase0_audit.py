"""Freeze Phase 0 evidence for the AXIS -> official LIBERO 30k study.

This command is intentionally an audit, not a trainer.  It records the
authoritative Hugging Face revision and metadata, compares it with the local
IPEC no-noops snapshots, audits the pinned OpenPI recipe against the AXIS
paper's LIBERO post-training table, and writes resource/gate evidence.

The command may use the network for small metadata files and one sample
episode.  It never treats an incomplete local official snapshot as usable
data and never loads Axis norm_stats.json into a transform.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    ROOT,
    canonical_json,
    git_info,
    load_yaml,
    resolve_path,
    sha256_bytes,
    sha256_file,
    sha256_tree,
    write_json,
)


PAPER_URL = "https://arxiv.org/html/2607.21588"
HF_API_URL = "https://huggingface.co/api/datasets/physical-intelligence/libero"
MODEL_API_URL = "https://huggingface.co/api/models/axisrobotics/Pi0.5-Axis/revision/c342608c489515efa9ba8281400dc648bf3970e1"
HF_RESOLVE = "https://huggingface.co/datasets/physical-intelligence/libero/resolve/{revision}/{path}"


def fetch_bytes(url: str, *, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "axis-libero-phase0-audit/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str) -> dict[str, Any]:
    value = json.loads(fetch_bytes(url).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object from {url}")
    return value


def remote_paths(api: dict[str, Any]) -> list[str]:
    return sorted(
        str(item.get("rfilename"))
        for item in api.get("siblings", [])
        if isinstance(item, dict) and item.get("rfilename")
    )


def remote_file_listing(api: dict[str, Any]) -> dict[str, Any]:
    names = remote_paths(api)
    return {
        "file_count": len(names),
        "paths_sha256": sha256_bytes(canonical_json(names)),
        "first_paths": names[:10],
        "last_paths": names[-10:],
        "data_parquet_count": sum(name.startswith("data/") and name.endswith(".parquet") for name in names),
        "metadata_paths": [name for name in names if name.startswith("meta/")],
    }


def safe_cmd(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_payload_manifest(root: Path, remote_names: list[str], *, expected_count: int | None = None) -> dict[str, Any]:
    """Hash only the payload paths listed by the locked official HF revision."""
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in remote_names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid remote payload path: {name!r}")
        path = root / relative
        if not path.is_file():
            missing.append(name)
            continue
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = canonical_json(entries)
    return {
        "scope": "locked official Hugging Face sibling payload paths only",
        "file_count": len(entries),
        "expected_file_count": expected_count,
        "missing_paths": missing,
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "paths_sha256": sha256_bytes(canonical_json([entry["path"] for entry in entries])),
        "manifest_sha256": sha256_bytes(payload),
        "complete": not missing and (expected_count is None or len(entries) == expected_count),
        "entries": entries,
    }


def data_source_gate(*, canonical_ready: bool, alternative_equivalent: bool) -> bool:
    """A complete locked canonical source supersedes historical ablation sources."""
    return canonical_ready or alternative_equivalent


def local_snapshot(root: Path, *, expected_revision: str | None = None) -> dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file() and ".incomplete" not in p.name]
    incomplete = [str(p) for p in root.rglob("*.incomplete")]
    marker = root / ".axis_pi05_revision"
    marker_value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "marker": marker_value,
        "marker_matches_revision": expected_revision is None or marker_value == expected_revision,
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "tree_sha256": sha256_tree(root) if root.is_dir() else None,
        "incomplete_files": incomplete,
        "complete_marker": marker.is_file() and not incomplete,
    }


def feature_summary(info: dict[str, Any]) -> dict[str, Any]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        return {"keys": [], "fps": [], "images": [], "vectors": {}}
    fps: set[float] = set()
    images: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    for key, value in features.items():
        if not isinstance(value, dict):
            continue
        shape = value.get("shape")
        if isinstance(shape, list):
            vectors[str(key)] = {"dtype": value.get("dtype"), "shape": shape}
        nested = value.get("info")
        if isinstance(nested, dict) and nested.get("video.fps") is not None:
            fps.add(float(nested["video.fps"]))
        if str(key).lower() in {"image", "wrist_image", "observation.images.front", "observation.images.wrist"}:
            images.append({"key": str(key), "dtype": value.get("dtype"), "shape": shape, "info": nested})
        if value.get("fps") is not None:
            fps.add(float(value["fps"]))
    return {"keys": sorted(str(key) for key in features), "fps": sorted(fps), "images": images, "vectors": vectors}


def local_dataset_summary(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {"path": str(root), "exists": root.is_dir(), "metadata": None}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes: list[dict[str, Any]] = []
    if episodes_path.is_file():
        with episodes_path.open(encoding="utf-8") as handle:
            episodes = [json.loads(line) for line in handle if line.strip()]
    tasks_path = root / "meta" / "tasks.jsonl"
    tasks = []
    if tasks_path.is_file():
        with tasks_path.open(encoding="utf-8") as handle:
            tasks = [json.loads(line) for line in handle if line.strip()]
    return {
        "path": str(root),
        "info_sha256": sha256_file(info_path),
        "info": {key: info.get(key) for key in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "fps")},
        "episodes_jsonl_count": len(episodes),
        "episode_length_sum": sum(int(row.get("length", 0)) for row in episodes),
        "tasks_jsonl_count": len(tasks),
        "feature_summary": feature_summary(info),
    }


def vector_probe(table: Any, names: list[str]) -> dict[str, Any]:
    available = [name for name in names if name in table.column_names]
    if not available:
        return {}
    rows = table.select(available).to_pylist()
    selected = rows[:3] + rows[-3:] if len(rows) > 6 else rows
    timestamps = table["timestamp"].to_pylist() if "timestamp" in table.column_names else []
    diffs = [float(b) - float(a) for a, b in zip(timestamps, timestamps[1:])]
    return {
        "rows": table.num_rows,
        "columns": table.column_names,
        "timestamp_start": timestamps[0] if timestamps else None,
        "timestamp_end": timestamps[-1] if timestamps else None,
        "timestamp_step_min": min(diffs) if diffs else None,
        "timestamp_step_max": max(diffs) if diffs else None,
        "numeric_sample_sha256": sha256_bytes(canonical_json(selected)),
        "sample": selected,
    }


def sample_equivalence(config: dict[str, Any], official_revision: str) -> dict[str, Any]:
    """Compare one authoritative episode to the local Spatial episode 0."""
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        return {"available": False, "reason": f"pyarrow unavailable: {exc}"}

    local_path = ROOT / "data/raw/hf/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot/data/chunk-000/episode_000000.parquet"
    if not local_path.is_file():
        return {"available": False, "reason": f"missing local sample: {local_path}"}
    with tempfile.TemporaryDirectory(prefix="axis_libero_phase0_") as temp:
        official_path = Path(temp) / "episode_000000.parquet"
        try:
            official_path.write_bytes(fetch_bytes(HF_RESOLVE.format(revision=official_revision, path="data/chunk-000/episode_000000.parquet"), timeout=120))
        except Exception as exc:  # pragma: no cover - network dependent
            return {"available": False, "reason": f"official sample download failed: {exc}"}
        official = pq.read_table(official_path)
        local = pq.read_table(local_path)
        return {
            "available": True,
            "official": {"path": "remote:data/chunk-000/episode_000000.parquet", "sha256": sha256_file(official_path), **vector_probe(official, ["timestamp", "frame_index", "episode_index", "task_index", "actions", "state"])},
            "local_ipec_spatial": {"path": str(local_path), **vector_probe(local, ["timestamp", "frame_index", "episode_index", "task_index", "action", "observation.state"])},
            "same_row_count": official.num_rows == local.num_rows,
            "same_timestamp_step": vector_probe(official, ["timestamp"]).get("timestamp_step_min") == vector_probe(local, ["timestamp"]).get("timestamp_step_min"),
        }


def parse_param_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    value = json.loads(path.read_text(encoding="utf-8"))
    metadata = value.get("tree_metadata", {})
    total = 0
    by_root: Counter[str] = Counter()
    for key, item in metadata.items():
        shape = item.get("value_metadata", {}).get("write_shape", [])
        count = math.prod(int(dim) for dim in shape)
        total += count
        parts = str(key).split("', '")
        if len(parts) > 1:
            by_root[parts[1]] += count
    return {
        "exists": True,
        "leaf_count": len(metadata),
        "numel": total,
        "fp32_parameter_bytes": total * 4,
        "by_root_numel": dict(sorted(by_root.items())),
        "metadata_sha256": sha256_file(path),
    }


def gpu_info() -> dict[str, Any]:
    result = safe_cmd(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap", "--format=csv,noheader,nounits"])
    rows = []
    for line in result["stdout"].splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) >= 6:
            rows.append({"index": values[0], "name": values[1], "memory_total_mib": values[2], "memory_free_mib": values[3], "driver": values[4], "compute_cap": values[5]})
    return {"command": result, "gpus": rows}


def recipe_audit(config_path: Path, lock: dict[str, Any], paper_html: bytes) -> dict[str, Any]:
    config = load_yaml(config_path)
    openpi_root = ROOT / "third_party/axis_openpi"
    revision = lock["openpi"]["revision"]
    show = safe_cmd(["git", "show", f"{revision}:src/openpi/training/config.py"], cwd=openpi_root)
    base_text = show["stdout"]
    match = re.search(r'TrainConfig\(\s*name="pi05_libero"(?P<body>.*?\n\s*\),\n\s*#)', base_text, re.S)
    body = match.group("body") if match else ""
    paper_text = paper_html.decode("utf-8", errors="replace")
    table_start = paper_text.find('<figure id="A7.T11"')
    table_end = paper_text.find("</figure>", table_start) if table_start >= 0 else -1
    table_text = paper_text[table_start:table_end if table_end >= 0 else None]
    paper_checks = {
        "official_libero_release": "Official LIBERO LeRobot release" in paper_text,
        "expected_episodes_1693": "1,693" in paper_text,
        "global_batch_64": "Global batch size" in table_text and ">64<" in table_text,
        "train_steps_30000": ">30,000<" in table_text,
        "full_model_finetuning": "full-model finetuning" in paper_text,
        "ema_0999": "EMA decay" in table_text and ">0.999<" in table_text,
    }
    canonical = config["recipe"]
    local_diagnostic = load_yaml(ROOT / "configs/axis_pi05/train_v2.yaml")
    observed_base = {
        "config_name": "pi05_libero",
        "batch_size": 256 if "batch_size=256" in body.replace(" ", "") else "not parsed",
        "num_train_steps": 30000 if "num_train_steps=30_000" in body.replace(" ", "") else "not parsed",
        "peak_lr": "5e-5" in body.replace(" ", "") or "5e-5" in body,
        "warmup_steps": "10_000" in body,
        "ema_decay": "0.999" in body,
        "optimizer": "AdamW" in body,
        "gradient_clip_norm": "clip_gradient_norm=1.0" in body.replace(" ", ""),
        "source_excerpt_sha256": sha256_bytes(body.encode()),
    }
    return {
        "schema_version": 1,
        "generated_at": now(),
        "sources": {"paper": PAPER_URL, "openpi_revision": revision, "config": str(config_path)},
        "paper_html_sha256": sha256_bytes(paper_html),
        "paper_checks": paper_checks,
        "locked_openpi_pi05_libero": observed_base,
        "canonical_resolved_target": canonical,
        "existing_diagnostic_config_not_canonical": {
            "path": str(ROOT / "configs/axis_pi05/train_v2.yaml"),
            "training": local_diagnostic.get("training"),
            "model": local_diagnostic.get("model"),
        },
        "semantic_deltas": [
            {"field": "parameters", "paper": "full-model finetuning", "canonical": "full_fine_tuning", "diagnostic": "LoRA arms F0/F1/F2", "ok": True},
            {"field": "global_batch", "paper": 64, "canonical": canonical["global_batch"], "diagnostic": local_diagnostic.get("training", {}).get("global_batch"), "ok": canonical["global_batch"] == 64},
            {"field": "optimizer_steps", "paper": 30000, "canonical": canonical["optimizer_steps"], "diagnostic": local_diagnostic.get("training", {}).get("max_steps"), "ok": canonical["optimizer_steps"] == 30000},
            {"field": "peak_lr", "paper": 5e-5, "canonical": canonical["peak_lr"], "diagnostic": local_diagnostic.get("training", {}).get("peak_lr"), "ok": canonical["peak_lr"] == 5e-5},
            {"field": "warmup_steps", "paper": 10000, "canonical": canonical["warmup_steps"], "diagnostic": local_diagnostic.get("training", {}).get("warmup_steps"), "ok": canonical["warmup_steps"] == 10000},
            {"field": "ema_decay", "paper": 0.999, "canonical": canonical["ema_decay"], "diagnostic": local_diagnostic.get("training", {}).get("ema_decay"), "ok": canonical["ema_decay"] == 0.999},
        ],
        "ok": all(paper_checks.values()) and canonical["parameters"] == "full_fine_tuning",
    }


def resource_preflight(config: dict[str, Any], params: dict[str, Any], canonical_root: Path, *, canonical_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    param_bytes = int(params.get("fp32_parameter_bytes", 0))
    adam_bytes = param_bytes * 2
    ema_bytes = param_bytes
    full_state = param_bytes + adam_bytes + ema_bytes
    params_ema = param_bytes + ema_bytes
    full_points = len(config["checkpoints"]["full_state_steps"])
    snapshot_points = len(set(config["checkpoints"]["params_ema_steps"]))
    retained = full_points * full_state + snapshot_points * params_ema
    required_with_headroom = int(retained * 1.5)
    mem = safe_cmd(["free", "-b"])
    return {
        "schema_version": 1,
        "generated_at": now(),
        "hardware": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(), "gpu": gpu_info(), "memory_command": mem},
        "disk": {"path": str(ROOT), "total_bytes": usage.total, "used_bytes": usage.total - usage.free, "free_bytes": usage.free, "free_gib": round(usage.free / 2**30, 2)},
        "model": {"params_root": str(resolve_path(config["model"]["params_root"])), "params_snapshot": local_snapshot(resolve_path(config["model"]["params_root"]).parent), "parameter_metadata": params},
        "canonical_dataset_snapshot": canonical_snapshot or local_snapshot(canonical_root),
        "checkpoint_estimate": {
            "assumptions": {"params_dtype": "fp32", "adam_m_dtype": "fp32", "adam_v_dtype": "fp32", "ema_dtype": "fp32", "no_deduplication": True},
            "params_bytes": param_bytes,
            "adam_state_bytes": adam_bytes,
            "ema_bytes": ema_bytes,
            "one_full_resumable_state_bytes": full_state,
            "one_params_plus_ema_snapshot_bytes": params_ema,
            "full_state_point_count": full_points,
            "params_ema_snapshot_count": snapshot_points,
            "retained_bytes_before_headroom": retained,
            "required_bytes_with_1_5x_headroom": required_with_headroom,
            "required_gib_with_1_5x_headroom": round(required_with_headroom / 2**30, 2),
            "fits_current_free_disk": required_with_headroom < usage.free,
        },
        "profile": {"microbatch_probe_required": True, "fsdp_required": True, "gradient_accumulation_required": True, "bf16_on_v100_forbidden": True, "precision_parity_probe_required": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/standard_libero_30k.yaml")
    parser.add_argument("--out-root", type=Path)
    args = parser.parse_args()
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    report_root = resolve_path(args.out_root or config["report_root"])
    report_root.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "configs/axis_pi05/data_sources.lock.yaml"
    lock = load_yaml(lock_path)
    canonical_spec = lock.get("canonical_dataset", config["canonical_dataset"])
    revision = canonical_spec["revision"]
    canonical_root = resolve_path(config["canonical_dataset"]["local_root"])

    paper_html = fetch_bytes(PAPER_URL, timeout=60)
    official_info_bytes = fetch_bytes(HF_RESOLVE.format(revision=revision, path="meta/info.json"), timeout=60)
    official_info = json.loads(official_info_bytes.decode("utf-8"))
    official_api = fetch_json(HF_API_URL)
    model_api = fetch_json(MODEL_API_URL)
    local_specs = [item for item in lock.get("datasets", []) if item.get("source") == "libero_standard" and item.get("suite") in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}]
    local_summaries = {}
    local_snapshots = {}
    for spec in local_specs:
        org, repo = spec["repo_id"].split("/", 1)
        root = ROOT / "data/raw/hf" / org / repo
        local_summaries[spec["suite"]] = local_dataset_summary(root)
        local_snapshots[spec["suite"]] = local_snapshot(root)

    source_provenance = {
        "schema_version": 1,
        "generated_at": now(),
        "research_plan": file_record(ROOT / config["plan"]),
        "lock": file_record(lock_path),
        "canonical": {"spec": canonical_spec, "remote_api": {key: official_api.get(key) for key in ("id", "sha", "lastModified", "private", "downloads")}, "remote_file_listing": remote_file_listing(official_api), "remote_api_url": HF_API_URL, "remote_revision_verified": official_api.get("sha") == revision, "remote_info_json": {"sha256": sha256_bytes(official_info_bytes), "summary": {key: official_info.get(key) for key in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps")}, "feature_summary": feature_summary(official_info)}, "local_snapshot": local_snapshot(canonical_root)},
        "model": {"repo_id": config["model"]["repo_id"], "revision": config["model"]["revision"], "remote_api_url": MODEL_API_URL, "remote_revision_verified": model_api.get("sha") == config["model"]["revision"], "remote_file_listing": remote_file_listing(model_api), "params_root": str(resolve_path(config["model"]["params_root"])), "params_tree_sha256": sha256_tree(resolve_path(config["model"]["params_root"])), "norm_stats": {"path": str(resolve_path(config["model"]["axis_norm_stats"])), "sha256": sha256_file(resolve_path(config["model"]["axis_norm_stats"])), "policy": "record_hash_only_never_load"}},
        "openpi": {"revision": lock["openpi"]["revision"], "checkout": safe_cmd(["git", "rev-parse", "HEAD"], cwd=ROOT / "third_party/axis_openpi"), "checkout_status": safe_cmd(["git", "status", "--short"], cwd=ROOT / "third_party/axis_openpi"), "checkout_tree_sha256": sha256_tree(ROOT / "third_party/axis_openpi", exclude={".git"}), "patch_tree_sha256": sha256_tree(ROOT / "patches/axis_openpi")},
        "paper": {"url": PAPER_URL, "html_sha256": sha256_bytes(paper_html), "verified_sections": ["A7: LIBERO Post-Training from AXIS-Pretrained Parameters", "A7.T11: Default LIBERO post-training settings"]},
        "historical_ipec_sources": {suite: {"summary": local_summaries[suite], "snapshot": local_snapshots[suite]} for suite in sorted(local_summaries)},
        "decisions": {"canonical_source": "physical-intelligence/libero", "historical_ipec_sources_are_ablation_only": True, "axis_norm_stats_loaded": False, "old_checkpoints_loaded": False},
    }
    canonical_snapshot = source_provenance["canonical"]["local_snapshot"]
    canonical_snapshot["marker_matches_revision"] = canonical_snapshot["marker"] == revision
    payload_manifest = canonical_payload_manifest(
        canonical_root,
        remote_paths(official_api),
        expected_count=int(canonical_spec.get("expected_payload_files", config["canonical_dataset"].get("expected_payload_files", 0))),
    )
    write_json(report_root / "canonical_payload_manifest.json", payload_manifest)
    source_provenance["canonical"]["official_payload_manifest"] = {
        key: value for key, value in payload_manifest.items() if key != "entries"
    }
    write_json(report_root / "source_provenance.json", source_provenance)

    official_summary = {"repo_id": canonical_spec["repo_id"], "revision": revision, "metadata": {key: official_info.get(key) for key in ("codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks", "chunks_size", "fps")}, "feature_summary": feature_summary(official_info)}
    equivalence = {"schema_version": 1, "generated_at": now(), "official": official_summary, "local_candidates": local_summaries, "aggregate_episode_count_matches": sum(int(item.get("info", {}).get("total_episodes", 0)) for item in local_summaries.values()) == official_info.get("total_episodes"), "aggregate_frame_count_matches": sum(int(item.get("info", {}).get("total_frames", 0)) for item in local_summaries.values()) == official_info.get("total_frames"), "sample_probe": sample_equivalence(config, revision)}
    reasons = [
        "official fps is 10 while local IPEC snapshots advertise 20",
        "official robot_type is panda while local IPEC snapshots advertise franka",
        "official v2.0 schema uses image/wrist_image and state/actions while local snapshots use video references and observation.* keys",
        "same aggregate episode/frame counts do not establish per-episode identity or content equivalence",
    ]
    probe = equivalence["sample_probe"]
    if probe.get("available") and probe.get("same_row_count") is False:
        reasons.append("official and local Spatial episode 0 have different row counts")
    equivalence["status"] = "not_equivalent"
    equivalence["equivalence_reasons"] = reasons
    equivalence["canonical_selection"] = "official_source_required"
    write_json(report_root / "dataset_equivalence.json", equivalence)

    recipe = recipe_audit(config_path, lock, paper_html)
    write_json(report_root / "reference_recipe_audit.json", recipe)

    params = parse_param_metadata(resolve_path(config["model"]["params_root"]) / "_METADATA")
    resources = resource_preflight(config, params, canonical_root, canonical_snapshot=canonical_snapshot)
    write_json(report_root / "resource_preflight.json", resources)

    root_git = git_info(ROOT)
    remote_revision_verified = source_provenance["canonical"]["remote_revision_verified"]
    canonical_snapshot_complete = canonical_snapshot["complete_marker"] and payload_manifest["complete"]
    canonical_revision_correct = remote_revision_verified and canonical_snapshot["marker_matches_revision"]
    canonical_ready = canonical_snapshot_complete and canonical_revision_correct
    alternative_equivalent = equivalence["status"] == "equivalent"
    data_source_ready = data_source_gate(canonical_ready=canonical_ready, alternative_equivalent=alternative_equivalent)
    expected_fps = int(config["canonical_dataset"]["authoritative_fps"])
    blockers = []
    if not remote_revision_verified and not alternative_equivalent:
        blockers.append("canonical revision is not the current Hugging Face API SHA")
    if not canonical_ready and not alternative_equivalent:
        blockers.append("no usable data source: canonical snapshot/revision is incomplete or an alternative equivalence proof is absent")
    if official_info.get("fps") != expected_fps:
        blockers.append(f"canonical_fps_mismatch: config expects native {expected_fps} Hz but official release metadata is {official_info.get('fps')} Hz")
    if not recipe["ok"]:
        blockers.append("reference recipe audit failed")
    if not resources["checkpoint_estimate"]["fits_current_free_disk"]:
        blockers.append("checkpoint estimate exceeds current free disk")
    phase0_ready = data_source_ready and recipe["ok"] and resources["checkpoint_estimate"]["fits_current_free_disk"] and official_info.get("fps") == expected_fps
    formal_run_blockers = []
    if root_git["dirty"]:
        formal_run_blockers.append("git_dirty: formal 30k run requires a clean implementation commit")
    formal_run_blockers.append("phase1_required_before_any_training: Phase 0 never directly authorizes training")
    next_action = (
        "Execute Phase 1 fresh-data loader, new norm stats, step-0 loss and precision audits; require a clean commit before formal 30k training."
        if phase0_ready
        else "Resolve Phase 0 data/config gates, then execute Phase 1 fresh-data loader, new norm stats, step-0 loss and precision audits."
    )
    summary = {
        "schema_version": 1,
        "generated_at": now(),
        "status": "phase0_complete_ready_for_phase1" if phase0_ready else "phase0_complete_blocked",
        "reports": [str(report_root / name) for name in ("source_provenance.json", "reference_recipe_audit.json", "dataset_equivalence.json", "resource_preflight.json", "canonical_payload_manifest.json")],
        "gates": {"source_provenance": remote_revision_verified or alternative_equivalent, "recipe_audit": recipe["ok"], "dataset_source": data_source_ready, "dataset_equivalence": alternative_equivalent, "resource_preflight": resources["checkpoint_estimate"]["fits_current_free_disk"], "official_snapshot_complete": canonical_snapshot_complete, "canonical_revision_correct": canonical_revision_correct, "clean_git": not root_git["dirty"]},
        "blockers": blockers,
        "formal_run_blockers": formal_run_blockers,
        "formal_run_ready": False,
        "next_action": next_action,
    }
    write_json(report_root / "phase0_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
