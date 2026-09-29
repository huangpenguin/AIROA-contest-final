#!/usr/bin/env python3
"""Download official canary step100 checkpoints from private Hugging Face.

This does not train, does not upload, and does not start N5 evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT


CONTRACT = ROOT / "configs/axis_pi05/official_canary100_hf.json"
PERSIST_ROOT = Path.home() / "data/parc20260911"


def load_contract() -> dict[str, Any]:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if value.get("schema_version") != "official_canary100_hf_v1":
        raise ValueError(f"unexpected canary100 HF contract: {value.get('schema_version')}")
    return value


def checkpoint_inventory(path: Path) -> dict[str, Any]:
    params = path / "params"
    train_state = path / "train_state"
    metadata = path / "_CHECKPOINT_METADATA"
    files = [item for item in path.rglob("*") if item.is_file() and ".partial" not in item.name]
    bytes_by_member = {}
    for name, member in (("params", params), ("train_state", train_state), ("_CHECKPOINT_METADATA", metadata)):
        if member.is_dir():
            bytes_by_member[name] = sum(item.stat().st_size for item in member.rglob("*") if item.is_file())
        elif member.is_file():
            bytes_by_member[name] = member.stat().st_size
        else:
            bytes_by_member[name] = 0
    return {
        "path": str(path.resolve()) if path.exists() else str(path),
        "exists": path.is_dir(),
        "params": params.is_dir(),
        "train_state": train_state.is_dir(),
        "metadata": metadata.is_file(),
        "complete": params.is_dir() and train_state.is_dir() and metadata.is_file(),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "bytes_by_member": bytes_by_member,
        "partial_residue": any(".partial" in item.name for item in path.rglob("*")) if path.exists() else False,
    }


def verify_checkpoint(path: Path) -> dict[str, Any]:
    report = checkpoint_inventory(path)
    if not report["complete"]:
        raise FileNotFoundError(f"incomplete step100 checkpoint: {json.dumps(report, sort_keys=True)}")
    if report["partial_residue"]:
        raise ValueError(f"partial residue in checkpoint: {path}")
    return report


def download_track(track: str) -> dict[str, Any]:
    contract = load_contract()
    spec = contract["tracks"][track]
    local_dir = ROOT / spec["local_dir"]
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type=str(contract["repo_type"]),
        local_dir=str(local_dir),
    )
    inventory = verify_checkpoint(local_dir)
    return {
        "track": track,
        "repo_id": spec["repo_id"],
        "local_dir": str(local_dir),
        "inventory": inventory,
        "training_steps": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("T3", "T1", "T2"), default="T3")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_contract()
    spec = contract["tracks"][args.track]
    path = args.path or (PERSIST_ROOT / spec["persist_rel"] if args.inventory_only else ROOT / spec["local_dir"])
    if args.inventory_only:
        print(json.dumps({"track": args.track, "inventory": checkpoint_inventory(path)}, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    print(json.dumps(download_track(args.track), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
