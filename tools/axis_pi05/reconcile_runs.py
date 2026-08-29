"""Reconcile recorded run state with the checkpoint tree without mutation.

Historical runs may say ``completed`` while their checkpoint list was written
before Orbax manifests were added.  This tool produces a new inventory and a
diff report; it never rewrites historical ``run_state.json`` or checkpoint
contents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, git_info, resolve_path, sha256_file, sha256_tree, write_json


def _checkpoint_dirs(run_root: Path) -> list[Path]:
    return sorted(
        path
        for path in run_root.rglob("[0-9]*")
        if path.is_dir()
        and path.name.isdigit()
        and (path / "_CHECKPOINT_METADATA").exists()
    )


def _checkpoint_record(root: Path, checkpoint: Path, *, hash_tree: bool = False) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint_manifest_v2.json"
    if not manifest_path.exists():
        manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {"_invalid_json": True}
    record = {
        "path": str(checkpoint.relative_to(root)),
        "step": int(checkpoint.name),
        "file_count": sum(1 for item in checkpoint.rglob("*") if item.is_file()),
        "bytes": sum(item.stat().st_size for item in checkpoint.rglob("*") if item.is_file()),
        "manifest": {
            "path": str(manifest_path.relative_to(root)) if manifest_path.exists() else None,
            "sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
            "internal_step": manifest.get("internal_step"),
            "stable_checkpoint_tree_sha256": manifest.get("stable_checkpoint_tree_sha256"),
            "checkpoint_tree_sha256": manifest.get("checkpoint_tree_sha256"),
        },
        "has_params": (checkpoint / "params").is_dir(),
        "has_train_state": (checkpoint / "train_state").is_dir(),
        "has_restore_validation": (checkpoint / "restore_validation.json").is_file(),
    }
    if hash_tree:
        # This is intentionally opt-in: a full Orbax tree can be tens or
        # hundreds of GiB.  Normal reconciliation only needs the checkpoint
        # manifest hash and file inventory; the manifest itself carries the
        # stable parameter-tree hash when it has been validated.
        record["tree_sha256"] = sha256_tree(checkpoint)
    return record


def reconcile_run(root: Path, run_root: Path, *, hash_tree: bool = False) -> dict[str, Any]:
    state_path = run_root / "run_state.json"
    state: dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"_invalid_json": True}
    recorded = sorted(str(item) for item in state.get("checkpoints", []) if isinstance(item, str))
    actual = [_checkpoint_record(root, path, hash_tree=hash_tree) for path in _checkpoint_dirs(run_root)]
    actual_paths = sorted(item["path"] for item in actual)
    return {
        "run_id": run_root.name,
        "path": str(run_root.relative_to(root)),
        "state_path": str(state_path.relative_to(root)),
        "recorded_status": state.get("status"),
        "recorded_checkpoints": recorded,
        "actual_checkpoints": actual,
        "missing_from_state": sorted(set(actual_paths) - set(recorded)),
        "stale_in_state": sorted(set(recorded) - set(actual_paths)),
        "inventory_consistent": recorded == actual_paths,
        "preserve_history": True,
    }


def reconcile(root: Path, output_root: Path, *, hash_tree: bool = False) -> dict[str, Any]:
    runs = [path for path in sorted(output_root.iterdir()) if path.is_dir()] if output_root.exists() else []
    records = [reconcile_run(root, run, hash_tree=hash_tree) for run in runs]
    return {
        "schema_version": 1,
        "inventory_type": "checkpoint_run_reconciliation",
        "root": str(root),
        "output_root": str(output_root.relative_to(root)) if output_root.exists() else str(output_root),
        "git": git_info(root),
        "runs": records,
        "summary": {
            "run_count": len(records),
            "consistent_run_count": sum(bool(item["inventory_consistent"]) for item in records),
            "inconsistent_run_ids": [item["run_id"] for item in records if not item["inventory_consistent"]],
            "checkpoint_count": sum(len(item["actual_checkpoints"]) for item in records),
        },
        "mutation": {"historical_run_state_changed": False, "checkpoint_contents_changed": False},
        "hash_policy": "full_checkpoint_tree" if hash_tree else "manifest_and_file_inventory_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/axis_pi05_v2")
    parser.add_argument("--out", type=Path, default=ROOT / "data/reports/axis_pi05_v2/run_inventory_reconciliation.json")
    parser.add_argument("--hash-checkpoints", action="store_true", help="hash all checkpoint bytes; potentially very slow for large Orbax trees")
    args = parser.parse_args()
    root = resolve_path(args.root)
    report = reconcile(root, resolve_path(args.output_root), hash_tree=args.hash_checkpoints)
    write_json(resolve_path(args.out), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
