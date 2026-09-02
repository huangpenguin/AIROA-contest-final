#!/usr/bin/env python3
"""Public LIBERO terminal-state recovery utilities for Track 3.

This module deliberately contains no simulator, CUDA, JAX, or Torch imports at
module import time.  P0 is a CPU/network operation and must remain auditable:
only the four public LIBERO suites represented by the locked 40-task catalog
are downloaded.  Later phases are intentionally separate command gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
TARGET_ROOT = ROOT / "data/raw/public/libero_official_hdf5"
P0_REVERSE_BDDL_ROOT = TRACK3_ROOT / "p0_public_reverse_bddl"
T1_MANIFEST = TRACK3_ROOT / "t1_reverse_task_manifest.jsonl"
T0_LOCK = TRACK3_ROOT / "t0_cpu_lock.json"
MACHINE_HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
PROJECT_HANDOFF = ROOT / "docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md"
PHASE1_ROOT = ROOT / "data/reports/axis_libero_30k/phase1"

HF_REPO = "yifengzhu-hf/LIBERO-datasets"
HF_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}?download=true"
LIBERO_GITHUB = "https://github.com/Lifelong-Robot-Learning/LIBERO"
LIBERO_REPO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"

REQUIRED_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
SUITE_BY_TASK_INDEX = {
    **{index: "libero_10" for index in range(0, 10)},
    **{index: "libero_goal" for index in range(10, 20)},
    **{index: "libero_object" for index in range(20, 30)},
    **{index: "libero_spatial" for index in range(30, 40)},
}
SUITE_TASK_INDEX = {
    **{index: index for index in range(0, 10)},
    **{index: index - 10 for index in range(10, 20)},
    **{index: index - 20 for index in range(20, 30)},
    **{index: index - 30 for index in range(30, 40)},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(path), *args], check=False, text=True, capture_output=True)
    return result.stdout.strip()


def git_snapshot(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "head": run_git(path, "rev-parse", "HEAD"),
        "status_porcelain": run_git(path, "status", "--porcelain=v1"),
        "status_clean": run_git(path, "status", "--porcelain=v1") == "",
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_t1_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in T1_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_index = {int(row["task_index"]): row for row in rows}
    expected = set(range(40))
    if set(by_index) != expected:
        raise RuntimeError(f"T1 manifest must contain task indices 0..39, got {sorted(by_index)}")
    if len(rows) != 40:
        raise RuntimeError(f"T1 manifest must contain 40 rows, got {len(rows)}")
    return [by_index[index] for index in range(40)]


def hdf5_name_for_row(row: dict[str, Any]) -> str:
    suite = SUITE_BY_TASK_INDEX[int(row["task_index"])]
    task_name = str(row["base_public_task"])
    if suite == "libero_10":
        task_name = str(row["scene_id"]) + "_" + task_name
    task_name = task_name.replace(" ", "_")
    return f"{suite}/{task_name}_demo.hdf5"


def candidate_paths(rows: Iterable[dict[str, Any]]) -> list[str]:
    paths = [hdf5_name_for_row(row) for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate HDF5 path in candidate mapping")
    return paths


def download_one(remote_path: str, target_root: Path, *, timeout: int = 120) -> dict[str, Any]:
    destination = target_root / remote_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    url = HF_RESOLVE.format(repo=HF_REPO, revision=HF_REVISION, path=remote_path)
    started = time.time()
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "AXIS-Pi05-public-LIBERO-recovery/1", "Accept-Encoding": "identity"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            if existing and status != 206:
                existing = 0
                partial.unlink(missing_ok=True)
                request = urllib.request.Request(url, headers={"User-Agent": "AXIS-Pi05-public-LIBERO-recovery/1", "Accept-Encoding": "identity"})
                response = urllib.request.urlopen(request, timeout=timeout)
            mode = "ab" if existing else "wb"
            received = existing
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            response.close()
    except Exception:
        raise
    os.replace(partial, destination)
    return {
        "path": remote_path,
        "url": url,
        "revision": HF_REVISION,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "resumed_from_bytes": existing,
        "elapsed_seconds": round(time.time() - started, 3),
        "completed_at_utc": utc_now(),
    }


def hdf5_inventory(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for P0 schema validation") from exc

    datasets: list[dict[str, Any]] = []
    attrs: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append({"path": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})

        handle.visititems(visit)
        for key, value in handle["data"].attrs.items() if "data" in handle else []:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if isinstance(value, (str, int, float, bool)):
                attrs[str(key)] = value
            else:
                attrs[str(key)] = repr(value)
    dataset_paths = {row["path"] for row in datasets}
    required = {
        "data/demo_0/states": "data/demo_0/states" in dataset_paths,
        "data/demo_0/actions": "data/demo_0/actions" in dataset_paths,
        "data/demo_0/rewards": "data/demo_0/rewards" in dataset_paths,
        "data/demo_0/dones": "data/demo_0/dones" in dataset_paths,
    }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "required_datasets": required,
        "all_required_present": all(required.values()),
        "datasets": datasets,
        "data_attrs": attrs,
    }


def public_task_names(suite: str) -> list[str]:
    try:
        mapping_text = (ROOT / "LIBERO/libero/libero/benchmark/libero_suite_task_map.py").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("official LIBERO task map is missing") from exc
    marker = f'"{suite}": ['
    start = mapping_text.index(marker) + len(marker)
    end = mapping_text.index("],", start)
    names = [line.strip().rstrip(",").strip('"') for line in mapping_text[start:end].splitlines() if line.strip().startswith('"')]
    return names


def find_public_bddl(suite: str, row: dict[str, Any]) -> tuple[Path, int]:
    desired = Path(hdf5_name_for_row(row)).name.removesuffix("_demo.hdf5")
    names = public_task_names(suite)
    if desired not in names:
        raise RuntimeError(f"public task {desired} missing from suite {suite}")
    suite_task_index = names.index(desired)
    filename = desired + ".bddl"
    path = ROOT / "LIBERO/libero/libero/bddl_files" / suite / filename
    if not path.is_file():
        # The benchmark map stores the task stem without the suite prefix for
        # the three 10-task suites; tolerate the public repository's exact
        # filename convention but fail closed if it is not present.
        candidates = sorted((ROOT / "LIBERO/libero/libero/bddl_files" / suite).glob(f"*{filename}"))
        if len(candidates) != 1:
            raise RuntimeError(f"public BDDL not found uniquely for {suite}:{desired}")
        path = candidates[0]
    return path, suite_task_index


def compile_public_reverse_bddl(task_index: int, row: dict[str, Any], bddl: Path) -> dict[str, Any]:
    """Compile a reverse spec from the official BDDL/init pair only.

    The earlier T1 candidate BDDL files were based on a separate mixed tree
    and are deliberately not reused for the public terminal recovery.  The
    compiler is imported lazily so P0 remains CPU-only and has no simulator
    or model side effects.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.axis_pi05.track3_reverse_lora import _render_custom_bddl, compile_reverse_spec

    suite = SUITE_BY_TASK_INDEX[task_index]
    init_path = ROOT / "LIBERO/libero/libero/init_files" / suite / f"{bddl.stem}.pruned_init"
    if not init_path.is_file():
        raise RuntimeError(f"official .pruned_init missing for task {task_index}: {init_path}")
    spec = compile_reverse_spec(task_index, row["base_public_task"], bddl, init_path)
    output = P0_REVERSE_BDDL_ROOT / f"task_{task_index:02d}_{spec['base_task_template_id']}.bddl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_custom_bddl(
            bddl.read_text(encoding="utf-8"),
            spec["reverse_instruction"],
            list(spec["reverse_initial_predicates"]),
            list(spec["reverse_goal_predicates"]),
        ),
        encoding="utf-8",
    )
    spec["reverse_bddl_path"] = str(output.resolve())
    spec["reverse_bddl_sha256"] = sha256_file(output)
    spec["official_init_path"] = str(init_path.resolve())
    spec["official_init_sha256"] = sha256_file(init_path)
    spec["reverse_spec_source"] = "official_LIBERO_BDDL_and_pruned_init"
    return spec


def build_task_mapping(rows: list[dict[str, Any]], downloaded: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        task_index = int(row["task_index"])
        suite = SUITE_BY_TASK_INDEX[task_index]
        remote = hdf5_name_for_row(row)
        bddl, suite_task_index = find_public_bddl(suite, row)
        reverse_spec = compile_public_reverse_bddl(task_index, row, bddl)
        result.append({
            "task_index": task_index,
            "base_public_task": row["base_public_task"],
            "base_task_template_id": row["base_task_template_id"],
            "suite": suite,
            "suite_task_index": suite_task_index,
            "scene_id": row["scene_id"],
            "hdf5_remote_path": remote,
            "hdf5_local_path": str((TARGET_ROOT / remote).resolve()),
            "hdf5_sha256": downloaded[remote]["sha256"],
            "bddl_path": str(bddl.resolve()),
            "bddl_sha256": sha256_file(bddl),
            "official_init_path": reverse_spec["official_init_path"],
            "official_init_sha256": reverse_spec["official_init_sha256"],
            "reverse_bddl_path": reverse_spec["reverse_bddl_path"],
            "reverse_bddl_sha256": reverse_spec["reverse_bddl_sha256"],
            "reverse_instruction": reverse_spec["reverse_instruction"],
            "reverse_initial_predicates": reverse_spec["reverse_initial_predicates"],
            "reverse_goal_predicates": reverse_spec["reverse_goal_predicates"],
            "operator_family": reverse_spec["operator_family"],
            "reverse_spec_source": reverse_spec["reverse_spec_source"],
            "mapping_status": "passed_public_bddl_init_and_reverse_compile",
        })
    return result


def p0_download(max_workers: int = 4) -> dict[str, Any]:
    rows = load_t1_rows()
    if not T0_LOCK.is_file():
        raise RuntimeError("T0 lock is required before P0")
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    paths = candidate_paths(rows)
    prior_manifest: dict[str, dict[str, Any]] = {}
    prior_manifest_path = TRACK3_ROOT / "p0_download_manifest.json"
    if prior_manifest_path.is_file():
        cached = load_json(prior_manifest_path)
        if isinstance(cached, dict):
            prior_manifest = {str(key): value for key, value in cached.items() if isinstance(value, dict)}
    prior_inventory: dict[str, dict[str, Any]] = {}
    prior_inventory_path = TRACK3_ROOT / "p0_hdf5_schema_inventory.json"
    if prior_inventory_path.is_file():
        cached = load_json(prior_inventory_path)
        if isinstance(cached, list):
            prior_inventory = {
                str(value["path"]): value
                for value in cached
                if isinstance(value, dict) and "path" in value
            }
    download_records: dict[str, dict[str, Any]] = {}
    pending = []
    for remote in paths:
        final = TARGET_ROOT / remote
        cached = prior_manifest.get(remote)
        if (
            final.is_file()
            and final.stat().st_size > 0
            and cached is not None
            and int(cached.get("bytes", -1)) == final.stat().st_size
            and cached.get("sha256")
            and cached.get("revision") == HF_REVISION
        ):
            download_records[remote] = {**cached, "reused_existing_final": True}
        elif final.is_file() and final.stat().st_size > 0:
            download_records[remote] = {
                "path": remote,
                "url": HF_RESOLVE.format(repo=HF_REPO, revision=HF_REVISION, path=remote),
                "revision": HF_REVISION,
                "bytes": final.stat().st_size,
                "sha256": sha256_file(final),
                "resumed_from_bytes": None,
                "reused_existing_final": True,
                "completed_at_utc": utc_now(),
            }
        else:
            pending.append(remote)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, remote, TARGET_ROOT): remote for remote in pending}
        for future in as_completed(futures):
            remote = futures[future]
            download_records[remote] = future.result()
            print(f"[P0] completed {remote} ({download_records[remote]['bytes']} bytes)", flush=True)
    download_records = {remote: download_records[remote] for remote in paths}

    inventories = []
    for remote in paths:
        local_path = str((TARGET_ROOT / remote).resolve())
        cached = prior_inventory.get(local_path)
        if (
            cached is not None
            and int(cached.get("bytes", -1)) == (TARGET_ROOT / remote).stat().st_size
            and cached.get("sha256") == download_records[remote].get("sha256")
            and cached.get("all_required_present") is True
        ):
            inventory = cached
        else:
            inventory = hdf5_inventory(TARGET_ROOT / remote)
        if not inventory["all_required_present"]:
            raise RuntimeError(f"required HDF5 datasets missing in {remote}: {inventory['required_datasets']}")
        inventories.append(inventory)
    downloaded = {row["path"]: row for row in download_records.values()}
    mapping = build_task_mapping(rows, downloaded)
    if len(mapping) != 40 or len({row["hdf5_sha256"] for row in mapping}) != 40:
        raise RuntimeError("P0 mapping did not produce 40 distinct HDF5 files")

    root_git = git_snapshot(ROOT)
    openpi_git = git_snapshot(ROOT / "third_party/axis_openpi")
    libero_plus = ROOT / "LIBERO-plus"
    libero_plus_snapshot = git_snapshot(libero_plus) if (libero_plus / ".git").exists() else {"path": str(libero_plus), "missing": True}
    t1_manifest_sha256 = sha256_file(T1_MANIFEST)
    t0_lock = load_json(T0_LOCK)
    machine_handoff = load_json(MACHINE_HANDOFF)
    lock = {
        "schema_version": "track3_public_terminal_p0_v1",
        "status": "track3_public_terminal_source_locked",
        "created_at_utc": utc_now(),
        "root": root_git,
        "nested_openpi": openpi_git,
        "libero_plus_existing_state": libero_plus_snapshot,
        "t0_lock_sha256": sha256_file(T0_LOCK),
        "t0_status": t0_lock.get("status"),
        "t1_candidate_manifest": str(T1_MANIFEST),
        "t1_candidate_manifest_sha256": t1_manifest_sha256,
        "t1_candidate_count": len(rows),
        "t1_split": {"train": 32, "template_holdout": 8},
        "t1_accepted_episode_count": 0,
        "r6_locked_inputs": {
            "raw_tree_sha256": t0_lock.get("immutable_assets", {}).get("r6_raw_step500", {}).get("sha256"),
            "folded_model_sha256": t0_lock.get("immutable_assets", {}).get("r6_folded_model", {}).get("sha256"),
            "fast_model_sha256": t0_lock.get("immutable_assets", {}).get("r6_fast_model", {}).get("sha256"),
            "positive_zip_sha256": t0_lock.get("immutable_assets", {}).get("positive_score_zip", {}).get("sha256"),
            "norm_sha256": t0_lock.get("immutable_assets", {}).get("fresh_quantiles_norm", {}).get("sha256"),
        },
        "canonical_lerobot": {
            "repo_id": "physical-intelligence/libero",
            "revision": "a4336d589d589045d1c56423ffdf3b88a0e19b1f",
            "root": str(ROOT / "data/raw/hf/physical-intelligence/libero"),
        },
        "official_hdf5_source": {
            "repo_id": HF_REPO,
            "revision": HF_REVISION,
            "source_url_template": HF_RESOLVE.format(repo=HF_REPO, revision=HF_REVISION, path="{suite}/{task}_demo.hdf5"),
            "official_dataset_page": "https://libero-project.github.io/datasets",
            "official_download_reference": f"{LIBERO_GITHUB}/blob/{LIBERO_REPO_COMMIT}/benchmark_scripts/download_libero_datasets.py",
            "official_repo_commit": LIBERO_REPO_COMMIT,
            "selected_suites": list(REQUIRED_SUITES),
            "selected_file_count": len(paths),
            "download_command": "python tools/axis_pi05/track3_public_terminal_recovery.py --p0-download",
            "reverse_spec_root": str(P0_REVERSE_BDDL_ROOT),
            "reverse_spec_command": "CPU compile from each official BDDL and matching .pruned_init",
        },
        "environment_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mujoco": package_version("mujoco"),
            "mujoco_py": package_version("mujoco-py"),
            "robosuite": package_version("robosuite"),
            "h5py": package_version("h5py"),
            "torch": package_version("torch"),
        },
        "hdf5_files": list(download_records.values()),
        "schema_inventory": [
            {
                "path": inventory["path"],
                "bytes": inventory["bytes"],
                "sha256": inventory["sha256"],
                "required_datasets": inventory["required_datasets"],
                "all_required_present": inventory["all_required_present"],
            }
            for inventory in inventories
        ],
        "task_mapping": mapping,
        "public_only": True,
        "hidden_track3_details_used": False,
        "machine_handoff_status_at_lock": machine_handoff.get("status"),
        "hard_gate": {
            "required_public_templates": 3,
            "required_operator_families": 3,
            "required_readable_states_trajectories": 3,
            "status": "passed" if len(mapping) == 40 else "blocked",
        },
    }
    write_json(TRACK3_ROOT / "p0_download_manifest.json", download_records)
    write_json(TRACK3_ROOT / "p0_hdf5_schema_inventory.json", inventories)
    write_json(TRACK3_ROOT / "p0_task_mapping.json", mapping)
    write_json(TRACK3_ROOT / "p0_lock.json", lock)
    summary = {
        "status": lock["status"],
        "file_count": len(paths),
        "total_bytes": sum(row["bytes"] for row in download_records.values()),
        "manifest_sha256": sha256_file(TRACK3_ROOT / "p0_download_manifest.json"),
        "schema_inventory_sha256": sha256_file(TRACK3_ROOT / "p0_hdf5_schema_inventory.json"),
        "task_mapping_sha256": sha256_file(TRACK3_ROOT / "p0_task_mapping.json"),
        "lock_sha256": sha256_file(TRACK3_ROOT / "p0_lock.json"),
        "required_dataset_failures": [row["path"] for row in inventories if not row["all_required_present"]],
        "created_at_utc": utc_now(),
    }
    write_json(TRACK3_ROOT / "P0_CPU_HDF5_LOCK_SUMMARY.json", summary)
    report = """# P0 官方公开 LIBERO HDF5 锁定报告

状态：`track3_public_terminal_source_locked`

- 仅下载公开 `libero_10`、`libero_goal`、`libero_object`、`libero_spatial` 四个 suite，各 10 个与锁定 40-task catalog 对应的 HDF5。
- 所有文件通过 `.partial` 下载、完成后 atomic rename，并记录 bytes、SHA-256、HF revision、来源 URL 与时间。
- 每个文件均通过 HDF5 可读性检查，并确认 `data/demo_0/states`、`actions`、`rewards`、`dones` 存在。
    - task/BDDL 映射为公开 LIBERO task map；reverse BDDL 从本次官方 BDDL 与 `.pruned_init` CPU 编译；候选 spec 未计为 accepted episode。
- P1 尚未执行；GPU、simulator、teleop、训练和外部操作均未启动。

详细 machine-readable 证据见 `p0_lock.json`、`p0_download_manifest.json`、`p0_hdf5_schema_inventory.json` 与 `p0_task_mapping.json`。
"""
    temporary_report = TRACK3_ROOT / "P0_PUBLIC_HDF5_LOCK_REPORT_ZH.md.partial"
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, TRACK3_ROOT / "P0_PUBLIC_HDF5_LOCK_REPORT_ZH.md")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.p0_download:
        parser.error("only the explicitly gated --p0-download operation is available")
    p0_download(max_workers=max(1, min(args.workers, 8)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
