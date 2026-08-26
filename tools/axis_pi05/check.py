from __future__ import annotations

import argparse
import re
from pathlib import Path

from .common import ROOT, load_yaml
from .lock import validate_lock


FORBIDDEN = "/home/huang/code/" + "AIROA-contest-libero-plus"


def run(root: Path) -> list[str]:
    errors: list[str] = []
    lock_path = root / "configs/axis_pi05/data_sources.lock.yaml"
    try:
        validate_lock(load_yaml(lock_path))
    except Exception as exc:
        errors.append(f"lock: {exc}")
    for path in [root / "tools/axis_pi05", root / "configs/axis_pi05", root / "contracts/axis_pi05", root / "patches/axis_openpi"]:
        for file in path.rglob("*"):
            if file.is_file() and file.suffix not in {".pyc", ".pyo"} and FORBIDDEN in file.read_text(encoding="utf-8", errors="ignore"):
                errors.append(f"legacy absolute path: {file}")
    text = (root / ".gitignore").read_text(encoding="utf-8")
    for required in ("/data/raw/", "/data/models/", "/data/processed/", "/third_party/axis_openpi/", "/outputs/axis_pi05/", "/reviews/axis_pi05/"):
        if required not in text:
            errors.append(f".gitignore missing {required}")
    for revision in re.findall(r"revision:\s*([^\s#]+)", (root / "configs/axis_pi05/data_sources.lock.yaml").read_text(encoding="utf-8")):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append(f"non-full revision: {revision}")
    patch_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "patches/axis_openpi").glob("*.patch"))
    if FORBIDDEN in patch_text:
        errors.append("OpenPI patch contains legacy absolute path")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="static Axis Pi0.5 plan gates")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    errors = run(args.root.resolve())
    if errors:
        print("\n".join(f"FAIL: {error}" for error in errors))
        raise SystemExit(1)
    print("PASS: Axis Pi0.5 static gates")


if __name__ == "__main__":
    main()
