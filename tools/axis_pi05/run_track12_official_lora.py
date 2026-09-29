#!/usr/bin/env python3
"""Compatibility wrapper. Prefer tools.axis_pi05.launch_official_blackwell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .launch_official_blackwell import main as launch_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Deprecated wrapper for launch_official_blackwell --track T1-P20|T2-P30")
    parser.add_argument("--mix", choices=("T1-P20", "T2-P30"), required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--source-roots", type=Path, required=True)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--wandb", choices=("disabled", "online"), default="disabled")
    parser.add_argument("--detach", action="store_true")
    args, extra = parser.parse_known_args()
    argv = [
        "--track",
        args.mix,
        "--steps",
        str(args.steps),
        "--microbatch",
        str(args.microbatch),
        "--run-dir",
        str(args.run_dir),
        "--source-roots",
        str(args.source_roots),
        "--wandb",
        args.wandb,
    ]
    if args.params is not None:
        argv.extend(["--params", str(args.params)])
    if args.norm is not None:
        argv.extend(["--norm", str(args.norm)])
    if args.detach:
        argv.append("--detach")
    argv.extend(extra)
    return launch_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
