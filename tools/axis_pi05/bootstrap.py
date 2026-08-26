from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .common import ROOT, git_info, sha256_tree, write_json
from .lock import load_and_validate, verify_openpi_checkout, verify_snapshot


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def download_snapshot(spec: dict[str, Any], raw_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    org, repo = spec["repo_id"].split("/", 1)
    target = raw_root / org / repo
    marker = target / ".axis_pi05_revision"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != spec["revision"]:
        raise RuntimeError(f"revision mismatch，拒绝覆盖: {target}")
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == spec["revision"]:
        return target
    if target.exists() and any(target.iterdir()) and not marker.is_file():
        if os.environ.get("AXIS_PI05_RESUME_PARTIAL") != "1":
            raise RuntimeError(f"发现未标记的既有目录，拒绝覆盖: {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"], repo_type=spec["repo_type"], revision=spec["revision"],
        local_dir=str(target), token=os.environ.get("HF_TOKEN"),
    )
    marker.write_text(spec["revision"] + "\n", encoding="utf-8")
    expected = ["meta/info.json"] if spec["repo_type"] == "dataset" and spec.get("source") else None
    if spec.get("purpose") == "recovered_coarse_categories_only":
        expected = ["episode_categories.csv"]
    return target


def download_model(spec: dict[str, Any], root: Path) -> Path:
    from huggingface_hub import snapshot_download

    org, repo = spec["repo_id"].split("/", 1)
    target = root / "data/models" / org / repo / spec["revision"]
    marker = target / ".axis_pi05_revision"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() != spec["revision"]:
        raise RuntimeError(f"model revision mismatch，拒绝覆盖: {target}")
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == spec["revision"]:
        return target
    if target.exists() and any(target.iterdir()) and not marker.is_file():
        if os.environ.get("AXIS_PI05_RESUME_PARTIAL") != "1":
            raise RuntimeError(f"发现未标记的既有模型目录，拒绝覆盖: {target}")
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec["repo_id"], repo_type="model", revision=spec["revision"], local_dir=str(target), token=os.environ.get("HF_TOKEN"))
    marker.write_text(spec["revision"] + "\n", encoding="utf-8")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description="Axis Pi0.5 bootstrap implementation")
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--env", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    lock_path = args.lock.resolve()
    lock = load_and_validate(lock_path)

    openpi_root = root / "third_party/axis_openpi"
    patch_dir = root / lock["openpi"]["patch_dir"]
    if not (openpi_root / ".git").is_dir():
        openpi_root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", lock["openpi"]["url"], str(openpi_root)])
    status = subprocess.check_output(["git", "-C", str(openpi_root), "status", "--porcelain"], text=True).strip()
    if status:
        unapplied = []
        for patch in sorted(patch_dir.glob("*.patch")):
            reverse = subprocess.run(
                ["git", "-C", str(openpi_root), "apply", "--reverse", "--check", str(patch)],
                capture_output=True,
            )
            if reverse.returncode != 0:
                unapplied.append(patch.name)
        if unapplied:
            raise RuntimeError(f"OpenPI checkout dirty，且并非仅包含锁定 patches: {openpi_root}; unapplied={unapplied}")
    run(["git", "-C", str(openpi_root), "fetch", "--quiet", "origin", lock["openpi"]["revision"]])
    run(["git", "-C", str(openpi_root), "checkout", "--quiet", "--detach", lock["openpi"]["revision"]])
    for patch in sorted(patch_dir.glob("*.patch")):
        reverse = subprocess.run(["git", "-C", str(openpi_root), "apply", "--reverse", "--check", str(patch)], capture_output=True)
        if reverse.returncode == 0:
            continue
        check = subprocess.run(["git", "-C", str(openpi_root), "apply", "--check", str(patch)], capture_output=True, text=True)
        if check.returncode != 0:
            raise RuntimeError(f"OpenPI patch 无法应用: {patch}\n{check.stderr}")
        run(["git", "-C", str(openpi_root), "apply", str(patch)])

    raw_root = root / "data/raw/hf"
    source_reports = []
    for spec in lock["datasets"] + lock["labels"]:
        path = download_snapshot(spec, raw_root)
        source_reports.append({**spec, **verify_snapshot(path, expected_files=["meta/info.json"] if spec in lock["datasets"] else ["episode_categories.csv"])})
    model_path = download_model(lock["model"], root)
    source_reports.append({**lock["model"], **verify_snapshot(model_path, expected_files=lock["model"]["expected_files"])})

    report_root = root / "data/reports"
    report_root.mkdir(parents=True, exist_ok=True)
    write_json(report_root / "bootstrap_sources.json", {
        "schema_version": 1,
        "lock": str(lock_path),
        "openpi": verify_openpi_checkout(openpi_root, lock["openpi"]["revision"]),
        "sources": source_reports,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
    })
    freeze = subprocess.run(
        [str(args.env / "bin/python"), "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    )
    if freeze.returncode == 0:
        requirements = freeze.stdout
    else:
        requirements = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(args.env / "bin/python")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    (report_root / "axis_pi05_bootstrap_requirements.lock.txt").write_text(requirements, encoding="utf-8")
    report = {
        "schema_version": 1,
        "git": git_info(root),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "uv": shutil.which("uv"), "ffmpeg": shutil.which("ffmpeg"), "nvidia_smi": shutil.which("nvidia-smi"), "hf_token_present": bool(os.environ.get("HF_TOKEN"))},
        "lock": str(lock_path),
        "openpi": verify_openpi_checkout(openpi_root, lock["openpi"]["revision"]),
        "patch_tree_sha256": sha256_tree(root / "patches/axis_openpi"),
        "sources": source_reports,
    }
    write_json(report_root / "bootstrap_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
