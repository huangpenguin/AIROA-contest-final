from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .common import ROOT, canonical_json, load_yaml, resolve_path, sha256_file, write_json


def _match(path: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, path) for pattern in patterns)


def expected_trainable(paths: list[str], scope: str, config: dict[str, Any]) -> set[str]:
    arm = config["arms"][scope]
    trainable = set()
    for path in paths:
        if _match(path, arm.get("frozen_regex", [])):
            continue
        if _match(path, arm.get("trainable_regex", [])):
            trainable.add(path)
    return trainable


def normalize_param_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "parameters" in value:
        value = value["parameters"]
    if isinstance(value, dict):
        entries = []
        for path, meta in value.items():
            if isinstance(meta, dict):
                entries.append({"path": path, **meta})
            else:
                entries.append({"path": path, "shape": list(meta) if isinstance(meta, (list, tuple)) else []})
        return entries
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"path": str(item)} for item in value]
    raise ValueError("parameter metadata 必须是 mapping 或 list")


def audit(entries: list[dict[str, Any]], scope: str, config: dict[str, Any]) -> dict[str, Any]:
    paths = sorted(str(entry["path"]) for entry in entries if entry.get("path"))
    if len(paths) != len(set(paths)):
        raise ValueError("parameter path 重复")
    trainable = expected_trainable(paths, scope, config)
    frozen = sorted(set(paths) - trainable)
    missing_lora = [path for path in paths if "lora" in path.lower() and path not in trainable]
    if missing_lora:
        raise ValueError(f"{scope}: LoRA path 被错误冻结: {missing_lora[:5]}")
    if not trainable:
        raise ValueError(f"{scope}: 没有匹配到 trainable parameter")
    if scope == "F0" and any(re.search(r"action_(in|out)_proj|time_mlp_(in|out)", path) for path in trainable):
        raise ValueError("F0 不得训练 action/time adapters")
    if scope == "F1" and any(re.search(r"time_mlp_(in|out)", path) for path in trainable):
        raise ValueError("F1 不得训练 time MLP")
    total = 0
    details = []
    for entry in entries:
        path = str(entry.get("path"))
        shape = entry.get("shape", [])
        count = entry.get("numel")
        if count is None and isinstance(shape, list):
            count = 1
            for dim in shape:
                count *= int(dim)
        count = int(count or 0)
        total += count if path in trainable else 0
        details.append({"path": path, "shape": shape, "dtype": entry.get("dtype"), "numel": count, "trainable": path in trainable})
    details.sort(key=lambda item: item["path"])
    trainable_hash = hashlib.sha256(canonical_json(details)).hexdigest()
    return {
        "schema_version": 1,
        "scope": scope,
        "trainable_paths": [item["path"] for item in details if item["trainable"]],
        "frozen_paths": [item["path"] for item in details if not item["trainable"]],
        "trainable_parameter_count": total,
        "trainable_set_sha256": trainable_hash,
        "all_parameter_count": sum(int(item["numel"]) for item in details),
        "parameters": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="audit actual OpenPI/JAX trainable paths against F0/F1/F2")
    ap.add_argument("--scope", required=True, choices=["F0", "F1", "F2"])
    ap.add_argument("--params-json", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/train.yaml")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    config = load_yaml(resolve_path(args.config))
    entries = normalize_param_entries(json.loads(resolve_path(args.params_json).read_text(encoding="utf-8")))
    report = audit(entries, args.scope, config)
    out = resolve_path(args.out) if args.out else resolve_path(args.params_json).with_name(f"parameter_audit_{args.scope}.json")
    write_json(out, report)
    print(json.dumps({key: report[key] for key in ("scope", "trainable_parameter_count", "all_parameter_count", "trainable_set_sha256")}, indent=2))


if __name__ == "__main__":
    main()
