#!/usr/bin/env python3
"""Safely prune reproducible-but-heavy competition artifacts.

Default mode is a read-only preview. --execute is required for deletion.
The script preserves reports, source data, code, configs, docs, and small
metadata copied from generated output roots before pruning them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/cleanup_20260921"
METADATA_ROOT = REPORT_ROOT / "retained_output_metadata"
OUTPUT_ROOTS = (
    ROOT / "outputs/axis_pi05",
    ROOT / "outputs/axis_pi05_v2",
    ROOT / "outputs/axis_fast_recovery",
)
ARTIFACT_ROOT = ROOT / "artifacts"
REPORT_DATA_ROOT = ROOT / "data/reports/axis_libero_30k"
METADATA_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".md",
    ".txt",
    ".sha256",
}
HEAVY_REPORT_SUFFIXES = {".npy", ".npz", ".mp4", ".png", ".jpg", ".jpeg", ".partial"}
CACHE_ROOTS = (
    ROOT,
    ROOT / "compe",
    ROOT / "pipeline",
    ROOT / "tests",
    ROOT / "tools",
)
EXCLUDED_CACHE_ROOTS = {
    (ROOT / ".venv_axis_pi05").resolve(),
    (ROOT / ".venv_train").resolve(),
    (ROOT / ".git").resolve(),
}
MAX_METADATA_BYTES = 10 * 1024 * 1024
BLOCK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def record_file(path: Path, category: str) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "category": category,
        "bytes": path.stat().st_size,
    }


def collect_metadata() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for source_root in OUTPUT_ROOTS:
        if not source_root.is_dir():
            continue
        root_label = source_root.relative_to(ROOT).as_posix().replace("/", "__")
        for source in sorted(source_root.rglob("*")):
            if not safe_file(source) or source.suffix.lower() not in METADATA_SUFFIXES:
                continue
            size = source.stat().st_size
            if size > MAX_METADATA_BYTES:
                skipped.append({
                    "path": str(source.relative_to(ROOT)),
                    "reason": "metadata_file_over_limit",
                    "bytes": size,
                })
                continue
            relative = source.relative_to(source_root)
            target = METADATA_ROOT / root_label / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append({
                "source": str(source.relative_to(ROOT)),
                "retained_copy": str(target.relative_to(ROOT)),
                "bytes": size,
                "sha256": sha256(source),
            })
    return copied, skipped


def planned_deletions() -> list[dict[str, Any]]:
    deletions: list[dict[str, Any]] = []
    for root in OUTPUT_ROOTS:
        if root.is_dir() and not root.is_symlink():
            for path in sorted(root.rglob("*")):
                if safe_file(path):
                    deletions.append(record_file(path, "generated_output_root"))
            deletions.append({
                "path": str(root.relative_to(ROOT)),
                "category": "generated_output_root_directory",
                "bytes": sum(
                    item["bytes"]
                    for item in deletions
                    if item["category"] == "generated_output_root"
                    and item["path"].startswith(str(root.relative_to(ROOT)) + "/")
                ),
            })
    if ARTIFACT_ROOT.is_dir():
        for path in sorted(ARTIFACT_ROOT.iterdir()):
            if safe_file(path) and (path.name.endswith(".zip") or path.name.endswith(".zip.partial")):
                deletions.append(record_file(path, "regenerable_zip"))
    if REPORT_DATA_ROOT.is_dir():
        for path in sorted(REPORT_DATA_ROOT.rglob("*")):
            if safe_file(path) and path.suffix.lower() in HEAVY_REPORT_SUFFIXES:
                deletions.append(record_file(path, "heavy_report_binary"))
            elif path.is_symlink() and path.suffix.lower() in HEAVY_REPORT_SUFFIXES:
                deletions.append({
                    "path": str(path.relative_to(ROOT)),
                    "category": "heavy_report_symlink",
                    "bytes": 0,
                })
    cache_dirs: set[Path] = set()
    for scan_root in CACHE_ROOTS:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("__pycache__"):
            resolved = path.resolve()
            if resolved in EXCLUDED_CACHE_ROOTS or any(
                str(resolved).startswith(str(excluded) + "/")
                for excluded in EXCLUDED_CACHE_ROOTS
            ):
                continue
            if path.is_dir() and not path.is_symlink():
                cache_dirs.add(path)
        pytest_cache = scan_root / ".pytest_cache"
        if pytest_cache.is_dir() and not pytest_cache.is_symlink():
            cache_dirs.add(pytest_cache)
    for directory in sorted(cache_dirs):
        for path in sorted(directory.rglob("*")):
            if safe_file(path):
                deletions.append(record_file(path, "python_test_cache"))
        deletions.append({
            "path": str(directory.relative_to(ROOT)),
            "category": "python_test_cache_directory",
            "bytes": 0,
        })
    return deletions


def remove_entry(relative: str) -> None:
    path = ROOT / relative
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def summarize(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_category: dict[str, dict[str, int]] = {}
    for item in entries:
        category = item["category"]
        bucket = by_category.setdefault(category, {"files": 0, "bytes": 0})
        if category.endswith("_directory"):
            continue
        bucket["files"] += 1
        bucket["bytes"] += int(item.get("bytes", 0))
    return by_category


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    deletions = planned_deletions()
    preview = {
        "schema_version": "competition_generated_artifact_cleanup_plan_v1",
        "created_unix": time.time(),
        "execute": args.execute,
        "retention": {
            "keep": [
                "source data under data/raw",
                "source code, configs, docs, tests and environments",
                "data/reports text/JSON/CSV/YAML/Markdown experiment records",
                "artifact .sha256 and .manifest.json sidecars",
                "copied small output metadata under data/reports/axis_libero_30k/cleanup_20260921",
            ],
            "delete": [
                "all files under outputs/axis_pi05, outputs/axis_pi05_v2 and outputs/axis_fast_recovery",
                "all artifacts/*.zip and *.zip.partial",
                "report *.npy, *.npz, *.mp4, *.png, *.jpg, *.jpeg and *.partial",
                "Python __pycache__ and pytest caches outside virtual environments",
            ],
        },
        "deletion_summary": summarize(deletions),
        "deletions": deletions,
    }
    if not args.execute:
        print(json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    copied, skipped = collect_metadata()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    preview["execute"] = True
    preview["metadata_copied"] = copied
    preview["metadata_skipped"] = skipped
    preview["metadata_summary"] = {
        "copied_files": len(copied),
        "copied_bytes": sum(item["bytes"] for item in copied),
        "skipped_files": len(skipped),
        "skipped_bytes": sum(item["bytes"] for item in skipped),
    }
    plan_path = REPORT_ROOT / "cleanup_plan.json"
    plan_path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    deleted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    # Delete files first and directories last. A few historical output files
    # may be root-owned; one such file must not abort cleanup of everything
    # else, and must be surfaced explicitly in the result.
    ordered = sorted(
        deletions,
        key=lambda item: (
            item["category"].endswith("_directory"),
            -len(Path(item["path"]).parts),
            item["path"],
        ),
    )
    for item in ordered:
        relative = item["path"]
        path = ROOT / relative
        if not (path.exists() or path.is_symlink()):
            continue
        try:
            remove_entry(relative)
        except (OSError, PermissionError) as exc:
            failed.append({
                **item,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        else:
            deleted.append(item)
    result = {
        "schema_version": "competition_generated_artifact_cleanup_result_v1",
        "created_unix": time.time(),
        "status": "completed" if not failed else "completed_with_permission_denials",
        "cleanup_plan": str(plan_path.relative_to(ROOT)),
        "deleted_summary": summarize(deleted),
        "deleted_paths": len(deleted),
        "failed_summary": summarize(failed),
        "failed_paths": len(failed),
        "failed_deletions": failed,
        "metadata_summary": preview["metadata_summary"],
        "auto_upload": False,
        "auto_submit": False,
    }
    (REPORT_ROOT / "cleanup_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
