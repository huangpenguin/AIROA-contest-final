from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import ROOT, find_single, read_csv_rows, read_table, resolve_path, write_json, write_jsonl
from .contract import CATEGORIES, validate_catalog
from .lock import load_and_validate


def scalar(value: Any, default: Any = None) -> Any:
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return default if value is None else value


def as_int(value: Any, default: int | None = None) -> int | None:
    value = scalar(value, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_text(value: Any, default: str = "") -> str:
    value = scalar(value, default)
    return str(value) if value is not None else default


def task_map(root: Path) -> dict[int, str]:
    path = find_single(root / "meta", ("tasks.parquet", "tasks.jsonl", "tasks.json"))
    if path is None:
        return {}
    if path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else value.get("tasks", [])
    else:
        rows = read_table(path)
    result: dict[int, str] = {}
    for index, row in enumerate(rows):
        task_index = as_int(row.get("task_index"), index)
        task = row.get("task", row.get("instruction", row.get("prompt", "")))
        if task_index is not None:
            result[task_index] = as_text(task)
    return result


def info(root: Path) -> dict[str, Any]:
    path = root / "meta" / "info.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是 mapping")
    return value


def episode_table(root: Path) -> list[dict[str, Any]]:
    candidates = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not candidates:
        candidate = root / "meta" / "episodes.parquet"
        if candidate.is_file():
            candidates = [candidate]
    if not candidates:
        jsonl = root / "meta" / "episodes.jsonl"
        if jsonl.is_file():
            with jsonl.open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        json_path = root / "meta" / "episodes.json"
        if json_path.is_file():
            value = json.loads(json_path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else value.get("episodes", [])
        raise FileNotFoundError(f"{root} 缺少 meta/episodes Parquet/JSONL")
    rows: list[dict[str, Any]] = []
    for path in candidates:
        rows.extend(read_table(path))
    return rows


def _stat_from_episode(row: dict[str, Any], name: str) -> int | None:
    for key in (name, f"stats/{name}/min", f"stats/{name}/max"):
        if key in row:
            return as_int(row[key])
    return None


def _infer_dims(meta: dict[str, Any]) -> tuple[int | None, int | None, list[str]]:
    features = meta.get("features", {})
    if not isinstance(features, dict):
        return None, None, []
    state_dim: int | None = None
    action_dim: int | None = None
    cameras: list[str] = []
    for key, feature in features.items():
        key_text = str(key).lower()
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if key_text == "observation.state" and isinstance(shape, list) and shape:
            state_dim = int(shape[-1])
        if key_text == "action" and isinstance(shape, list) and shape:
            action_dim = int(shape[-1])
        if "image" in key_text:
            if "wrist" in key_text or "eye_in_hand" in key_text:
                cameras.append("wrist")
            elif "front" in key_text or "third_person" in key_text or "agentview" in key_text or "image" in key_text:
                cameras.append("front")
    return state_dim, action_dim, sorted(set(cameras))


def load_plus_labels(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        return {}
    rows = read_csv_rows(path)
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        episode_index = as_int(row.get("episode_index"))
        if episode_index is not None:
            result[episode_index] = {key: str(value) for key, value in row.items()}
    return result


def build_source_catalog(spec: dict[str, Any], root: Path, labels: dict[int, dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = info(root)
    fps = metadata.get("fps", metadata.get("video_fps"))
    try:
        fps_value: int | float | None = float(fps)
        if fps_value.is_integer():
            fps_value = int(fps_value)
    except (TypeError, ValueError):
        fps_value = None
    state_dim, action_dim, cameras = _infer_dims(metadata)
    tasks = task_map(root)
    rows = episode_table(root)
    out: list[dict[str, Any]] = []
    for raw in rows:
        episode_index = as_int(raw.get("episode_index"))
        if episode_index is None:
            raise ValueError(f"{spec['id']} episode 缺少 episode_index")
        task_index = as_int(raw.get("task_index"), _stat_from_episode(raw, "task_index"))
        task = as_text(raw.get("task", raw.get("tasks", tasks.get(task_index, ""))))
        if task_index is None:
            task_index = next((index for index, value in tasks.items() if value == task), 0)
        length = as_int(raw.get("length"), None)
        if length is None:
            start = as_int(raw.get("dataset_from_index"), 0) or 0
            end = as_int(raw.get("dataset_to_index"), 0) or 0
            length = max(0, end - start)
        label = labels.get(episode_index, {}) if spec["source"] == "libero_plus" else {}
        category = label.get("category", "unavailable")
        if category not in CATEGORIES:
            category = "unavailable"
        suite = str(label.get("suite") or spec.get("suite") or "unavailable")
        source_file = label.get("source_file", "")
        row = {
            "source": spec["source"],
            "source_repo": spec["repo_id"],
            "revision": spec["revision"],
            "source_episode_index": episode_index,
            "suite": suite,
            "task": task,
            "task_index": task_index,
            "instruction": task,
            "perturb_category": "base" if spec["source"] == "libero_standard" else category,
            "perturb_label_quality": "recovered-coarse" if label else ("exact-source" if spec["source"] == "libero_standard" else "unavailable"),
            "source_file": source_file or None,
            "frame_count": length,
            "source_fps": fps_value,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "cameras": cameras,
            "dataset_root": str(root),
        }
        out.append(row)
    return out, {
        "id": spec["id"],
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "source": spec["source"],
        "suite": spec.get("suite"),
        "episode_count": len(out),
        "fps_values": sorted({row["source_fps"] for row in out}),
        "state_dims": sorted({row["state_dim"] for row in out}),
        "action_dims": sorted({row["action_dim"] for row in out}),
        "cameras": sorted({camera for row in out for camera in row["cameras"]}),
    }


def build_catalog(lock: dict[str, Any], raw_root: Path, labels_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = load_plus_labels(labels_path)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in lock["datasets"]:
        org, repo = spec["repo_id"].split("/", 1)
        root = raw_root / org / repo
        rows, summary = build_source_catalog(spec, root, labels)
        all_rows.extend(rows)
        summaries.append(summary)
    return all_rows, {"sources": summaries, "label_rows": len(labels)}


def catalog_report(rows: list[dict[str, Any]], source_report: dict[str, Any]) -> dict[str, Any]:
    contract = validate_catalog(rows)
    return {
        "schema_version": 1,
        "contract": contract,
        "source_report": source_report,
        "counts": {
            "source": dict(Counter(row.get("source") for row in rows)),
            "suite": dict(Counter(row.get("suite") for row in rows)),
            "category": dict(Counter(row.get("perturb_category") for row in rows)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="build the virtual canonical LIBERO catalog")
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/axis_pi05/data_sources.lock.yaml")
    ap.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/hf")
    ap.add_argument("--labels", type=Path, default=ROOT / "data/raw/hf/katsukiono/libero-plus-episode-categories/episode_categories.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/libero_axis_mix_v1/catalog.jsonl")
    ap.add_argument("--report", type=Path, default=ROOT / "data/reports/data_audit.json")
    args = ap.parse_args()
    lock = load_and_validate(resolve_path(args.lock))
    rows, source_report = build_catalog(lock, resolve_path(args.raw_root), resolve_path(args.labels))
    report = catalog_report(rows, source_report)
    report["catalog_sha256"] = __import__("hashlib").sha256(
        ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n").encode()
    ).hexdigest()
    write_jsonl(resolve_path(args.out), rows)
    write_json(resolve_path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["contract"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
