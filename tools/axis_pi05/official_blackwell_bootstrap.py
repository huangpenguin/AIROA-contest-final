#!/usr/bin/env python3
"""Record official Blackwell environment and audit libero_combined_20hz.

Read-only.  Does not copy the dataset and does not start training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _json_if_possible(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"unparsed": True, "bytes": path.stat().st_size}


def collect(dataset_root: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv",
        ]
    )
    disks = _run(["df", "-h", "/home", "/home/huangpenguin", "/opt/dlami/nvme", str(Path.home() / "data")])
    listing = []
    if dataset_root.exists():
        for child in sorted(dataset_root.iterdir()):
            listing.append(
                {
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "size_bytes": child.stat().st_size if child.is_file() else None,
                }
            )
    meta = {}
    for name in ("info.json", "stats.json", "tasks.json", "episodes.jsonl"):
        candidate = dataset_root / "meta" / name
        if candidate.is_file():
            meta[name] = _json_if_possible(candidate) if candidate.stat().st_size < 2_000_000 else {"bytes": candidate.stat().st_size}
    sample_keys: list[str] = []
    if (dataset_root / "meta").is_dir():
        sample_keys = sorted(path.name for path in (dataset_root / "meta").iterdir())
    return {
        "schema_version": 1,
        "created_at_utc": now,
        "hostname": _run(["hostname"]),
        "whoami": _run(["whoami"]),
        "pwd": str(Path.cwd()),
        "home": str(Path.home()),
        "python": _run(["python3", "-c", "import sys,platform; print(sys.version); print(platform.platform())"]),
        "jax": _run(["python3", "-c", "import jax, jaxlib; print('jax', jax.__version__); print('jaxlib', jaxlib.__version__); print(jax.devices())"]),
        "torch": _run(["python3", "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"]),
        "nvidia_smi": gpu,
        "nvcc": _run(["nvcc", "--version"]),
        "disks": disks,
        "nvme_exists": Path("/opt/dlami/nvme").exists(),
        "dataset_root": str(dataset_root),
        "dataset_exists": dataset_root.exists(),
        "dataset_listing": listing,
        "dataset_meta_names": sample_keys,
        "dataset_meta": meta,
        "copied_dataset": False,
        "training_started": False,
        "which": {name: shutil.which(name) for name in ("git", "python3", "nvidia-smi", "parc-home-sync", "docker")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official Blackwell environment")
    parser.add_argument("--dataset", type=Path, default=Path.home() / "dataset" / "libero_combined_20hz")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = collect(args.dataset)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
