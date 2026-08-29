from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .common import FULL_SHA_RE, ROOT, load_yaml, resolve_path, sha256_file, sha256_tree, write_json


class LockError(ValueError):
    pass


def _full_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(FULL_SHA_RE, value):
        raise LockError(f"{field} 必须是完整 40 位小写 SHA，得到 {value!r}")
    return value


def validate_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1:
        raise LockError("不支持的 lock schema_version")
    policy = lock.get("policy", {})
    if policy.get("require_full_sha") is not True:
        raise LockError("lock 必须启用 require_full_sha")
    openpi = lock.get("openpi")
    if not isinstance(openpi, dict):
        raise LockError("缺少 openpi lock")
    _full_sha(openpi.get("revision"), "openpi.revision")
    if "upstream_base_revision" in openpi:
        _full_sha(openpi.get("upstream_base_revision"), "openpi.upstream_base_revision")
    if "implementation_commit" in openpi:
        _full_sha(openpi.get("implementation_commit"), "openpi.implementation_commit")
    if "implementation_git_tree" in openpi:
        _full_sha(openpi.get("implementation_git_tree"), "openpi.implementation_git_tree")
    if "implementation_patch_sha256" in openpi and not re.fullmatch(
        r"^[0-9a-f]{64}$", str(openpi["implementation_patch_sha256"])
    ):
        raise LockError("openpi.implementation_patch_sha256 必须是完整 64 位小写 SHA-256")
    if not openpi.get("url") or not openpi.get("patch_dir"):
        raise LockError("openpi.url 与 openpi.patch_dir 必须存在")
    datasets = lock.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 6:
        raise LockError("必须锁定六个训练数据源")
    ids: set[str] = set()
    for item in datasets:
        if not isinstance(item, dict):
            raise LockError("datasets 项必须是 mapping")
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id in ids:
            raise LockError(f"重复或非法 dataset id: {item_id!r}")
        ids.add(item_id)
        _full_sha(item.get("revision"), f"datasets[{item_id}].revision")
        if not item.get("repo_id") or item.get("repo_type") != "dataset":
            raise LockError(f"dataset {item_id} 缺少 repo_id/repo_type")
    labels = lock.get("labels")
    if not isinstance(labels, list) or not labels:
        raise LockError("必须锁定 Plus coarse label repo")
    for item in labels:
        _full_sha(item.get("revision"), f"labels[{item.get('id')}].revision")
    model = lock.get("model")
    if not isinstance(model, dict):
        raise LockError("缺少 model lock")
    _full_sha(model.get("revision"), "model.revision")
    if model.get("norm_stats_policy") != "record_hash_only_never_load":
        raise LockError("Axis norm_stats policy 必须是 record_hash_only_never_load")
    canonical = lock.get("canonical_dataset")
    if canonical is not None:
        if not isinstance(canonical, dict):
            raise LockError("canonical_dataset 必须是 mapping")
        _full_sha(canonical.get("revision"), "canonical_dataset.revision")
        if not canonical.get("repo_id") or canonical.get("repo_type") != "dataset":
            raise LockError("canonical_dataset 缺少 repo_id/repo_type")
        if not isinstance(canonical.get("expected_payload_files"), int) or canonical["expected_payload_files"] <= 0:
            raise LockError("canonical_dataset.expected_payload_files 必须是正整数")
        if not isinstance(canonical.get("expected_fps"), int) or canonical["expected_fps"] <= 0:
            raise LockError("canonical_dataset.expected_fps 必须是正整数")


def load_and_validate(path: Path) -> dict[str, Any]:
    lock = load_yaml(path)
    validate_lock(lock)
    return lock


def _run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def verify_openpi_checkout(root: Path, openpi: dict[str, Any]) -> dict[str, Any]:
    if not (root / ".git").is_dir():
        raise LockError(f"OpenPI checkout 不存在: {root}")
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    expected = str(openpi.get("implementation_commit", openpi["revision"]))
    if actual != expected:
        raise LockError(f"OpenPI implementation commit 不一致: expected={expected}, actual={actual}")
    base = str(openpi.get("upstream_base_revision", openpi["revision"]))
    if subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{base}^{{commit}}"], check=False
    ).returncode != 0:
        raise LockError(f"OpenPI upstream base revision 不存在于 checkout: {base}")
    if subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", base, actual], check=False
    ).returncode != 0:
        raise LockError(f"OpenPI implementation commit 不是 upstream base 的后代: base={base}, actual={actual}")
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).splitlines()
    if status:
        raise LockError(f"OpenPI checkout 必须 clean: {status[:5]}")
    git_tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    expected_tree = openpi.get("implementation_git_tree")
    if expected_tree is not None and git_tree != expected_tree:
        raise LockError(f"OpenPI Git tree 不一致: expected={expected_tree}, actual={git_tree}")
    return {
        "upstream_base_revision": base,
        "implementation_commit": actual,
        "implementation_git_tree": git_tree,
        "worktree_clean": True,
        "tree_sha256": sha256_tree(root, exclude={".git", "__pycache__"}),
    }


def verify_snapshot(root: Path, *, expected_files: list[str] | None = None) -> dict[str, Any]:
    if not root.is_dir():
        raise LockError(f"snapshot 不存在: {root}")
    files = [path for path in root.rglob("*") if path.is_file() and ".incomplete" not in path.name]
    incomplete = [str(path) for path in root.rglob("*.incomplete")]
    missing = [name for name in (expected_files or []) if not (root / name).is_file()]
    if incomplete or missing:
        raise LockError(f"snapshot 校验失败: incomplete={incomplete[:3]}, missing={missing}")
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "tree_sha256": sha256_tree(root),
        "expected_files": expected_files or [],
    }


def revision_map(lock: dict[str, Any]) -> dict[str, str]:
    specs = lock["datasets"] + lock["labels"] + [lock["model"]]
    canonical = lock.get("canonical_dataset")
    if isinstance(canonical, dict):
        specs.append(canonical)
    return {item["repo_id"]: item["revision"] for item in specs}


def main() -> None:
    ap = argparse.ArgumentParser(description="validate locked full revisions and local snapshots")
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/axis_pi05/data_sources.lock.yaml")
    ap.add_argument("--openpi-root", type=Path, default=ROOT / "third_party/axis_openpi")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    lock_path = resolve_path(args.lock)
    lock = load_and_validate(lock_path)
    report: dict[str, Any] = {"lock": str(lock_path), "revisions": revision_map(lock)}
    if args.openpi_root.exists():
        report["openpi"] = verify_openpi_checkout(args.openpi_root, lock["openpi"])
    else:
        report["openpi"] = {"status": "not_downloaded"}
    if args.json:
        write_json(resolve_path(args.json), report)
    print(report)


if __name__ == "__main__":
    main()
