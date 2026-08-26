from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .common import ROOT, git_info, load_yaml, resolve_path, sha256_file, sha256_tree, write_json


def _git_files(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


def _path_record(root: Path, path: Path, *, hash_directory: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.relative_to(root)), "exists": path.exists()}
    if path.is_file():
        record.update({"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    elif path.is_dir():
        record.update({"kind": "directory"})
        if hash_directory:
            record["tree_sha256"] = sha256_tree(path)
    else:
        record["kind"] = "missing"
    return record


def _last_log_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    step_lines = [line for line in lines if re.search(r"\bStep\b|\bstep\b", line)]
    nonfinite = [line for line in lines if re.search(r"nan|inf|nonfinite|traceback|error", line, re.I)]
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "line_count": len(lines),
        "last_step_line": step_lines[-1] if step_lines else None,
        "nonfinite_or_error_lines": nonfinite[-20:],
    }


def _directory_size(path: Path) -> tuple[int, int]:
    files = 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            files += 1
            total += child.stat().st_size
    return files, total


def _run_record(root: Path, run_root: Path) -> dict[str, Any]:
    resolved_path = run_root / "resolved_config.json"
    state_path = run_root / "run_state.json"
    provenance_path = run_root / "provenance.json"
    resolved: dict[str, Any] = {}
    state: dict[str, Any] = {}
    if resolved_path.exists():
        try:
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            resolved = {"_invalid_json": True}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"_invalid_json": True}
    settings = resolved.get("resolved", {}) if isinstance(resolved, dict) else {}
    checkpoint_dirs = sorted(
        path for path in run_root.rglob("[0-9]*")
        if path.is_dir() and path.name.isdigit() and (path / "_CHECKPOINT_METADATA").exists()
    )
    checkpoints = []
    for checkpoint in checkpoint_dirs:
        file_count, byte_count = _directory_size(checkpoint)
        checkpoints.append({
            "path": str(checkpoint.relative_to(root)),
            "step": int(checkpoint.name),
            "file_count": file_count,
            "bytes": byte_count,
            "has_params": (checkpoint / "params").exists(),
            "has_train_state": (checkpoint / "train_state").exists(),
        })
    complete_provenance = all(path.exists() for path in (resolved_path, provenance_path, state_path))
    return {
        "run_id": run_root.name,
        "path": str(run_root.relative_to(root)),
        "scope": settings.get("scope"),
        "mix": settings.get("mix"),
        "steps_requested": settings.get("steps"),
        "status": state.get("status"),
        "returncode": state.get("returncode"),
        "resolved": {
            key: settings.get(key)
            for key in ("precision", "global_batch", "peak_lr", "decay_lr", "clip_grad_norm", "fail_on_nonfinite", "cast_trainable_bf16", "seed")
        },
        "complete_provenance": complete_provenance,
        "files": {
            "resolved_config": _path_record(root, resolved_path),
            "provenance": _path_record(root, provenance_path),
            "run_state": _path_record(root, state_path),
            "train_log": _last_log_state(run_root / "train.log"),
        },
        "checkpoints": checkpoints,
        "diagnostic_only_reason": (
            "pre-remediation v1 manifest/norm sampler mismatch; retained for history, not promotion"
            if run_root.name == "t1_f0_m1_lr1e5"
            else None
        ),
    }


def build_inventory(root: Path) -> dict[str, Any]:
    tracked = _git_files(root, "ls-files")
    untracked = _git_files(root, "ls-files", "--others", "--exclude-standard")
    relevant_prefixes = (
        "tools/axis_pi05/", "configs/axis_pi05/", "contracts/axis_pi05/", "patches/axis_openpi/"
    )
    relevant = sorted(path for path in tracked + untracked if path.startswith(relevant_prefixes))
    relevant_records = []
    for relative in relevant:
        path = root / relative
        relevant_records.append({
            "path": relative,
            "tracked": relative in tracked,
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        })
    output_root = root / "outputs/axis_pi05"
    runs = [_run_record(root, path) for path in sorted(output_root.iterdir()) if path.is_dir()] if output_root.exists() else []
    lock_path = root / "configs/axis_pi05/data_sources.lock.yaml"
    catalog_path = root / "data/processed/libero_axis_mix_v1/catalog.jsonl"
    return {
        "schema_version": 2,
        "inventory_type": "pre_remediation_freeze",
        "root": str(root),
        "git": git_info(root),
        "relevant_file_inventory": relevant_records,
        "patch_tree_sha256": sha256_tree(root / "patches/axis_openpi"),
        "lock": _path_record(root, lock_path),
        "catalog": _path_record(root, catalog_path),
        "old_outputs_root": _path_record(root, output_root, hash_directory=False),
        "old_runs": runs,
        "preservation_policy": {
            "old_outputs_mutated": False,
            "old_manifests_mutated": False,
            "old_valid_t1_f0_m1_lr1e5": "diagnostic_only_due_to_v1_sampler_norm_mismatch",
            "new_namespace": "axis_pi05_v2",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze pre-remediation Axis Pi0.5 evidence inventory")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "data/reports/axis_pi05_v2/pre_remediation_inventory.json")
    args = ap.parse_args()
    report = build_inventory(resolve_path(args.root))
    write_json(resolve_path(args.out), report)
    print(json.dumps({"output": str(resolve_path(args.out)), "old_runs": len(report["old_runs"]), "git_dirty": report["git"]["dirty"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
