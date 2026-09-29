#!/usr/bin/env python3
"""Launch a bounded T1 / T2 / T3 Blackwell LoRA canary.

Machine paths, run directories, and wandb secrets stay on the CLI / environment.
This fresh-R6 launcher is intentionally limited to at most 100 optimizer steps.
It does not resume a previous canary optimizer, does not mkdir an empty run
directory, and writes the log beside the run, not inside it.  Post-canary
continuation requires a separate, evaluation-gated resume entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import ROOT, load_yaml, resolve_path


DEFAULT_CONFIG = ROOT / "configs/axis_pi05/official_blackwell_launch.yaml"
DEFAULT_PYTHON = Path(os.environ.get("AXIS_PI05_PYTHON", str(ROOT / ".venv_axis_pi05/bin/python")))
DEFAULT_PARAMS = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500/params"
TRAIN = ROOT / "third_party/axis_openpi/scripts/train.py"
TRACKS = ("T1-P20", "T2-P30", "T3")
MAX_FRESH_CANARY_STEPS = 100


def load_launch_config(path: Path) -> dict[str, Any]:
    value = load_yaml(path)
    if "tracks" not in value or "shared" not in value or "wandb" not in value:
        raise ValueError(f"launch yaml missing tracks/shared/wandb: {path}")
    return value


def _comma(values: list[str] | tuple[str, ...]) -> str:
    return ",".join(str(item) for item in values)


def _keep_steps(values: list[int]) -> str:
    return ",".join(str(int(item)) for item in values)


def wandb_credentials_present(env: dict[str, str]) -> bool:
    if env.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    if netrc.is_file() and "api.wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore"):
        return True
    return (Path.home() / ".config/wandb/settings").is_file()


def _load_source_roots(path: Path) -> tuple[list[str], list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError(f"source_roots must be a non-empty object: {path}")
    ids = list(value)
    roots = []
    for repo in ids:
        item = value[repo]
        if not isinstance(item, dict) or "root" not in item:
            raise ValueError(f"source_roots[{repo!r}] missing root")
        roots.append(str(Path(str(item["root"]))))
    return ids, roots


def build_environment(
    *,
    track: str,
    recipe: dict[str, Any],
    steps: int,
    run_dir: Path,
    run_id: str,
    params: Path,
    norm: Path,
    sampling: Path,
    python: Path,
    wandb_mode: str,
    wandb_project: str,
    source_roots: Path | None,
    forward_root: Path | None,
    reverse_root: Path | None,
    microbatch: int,
    accumulation: int,
) -> dict[str, str]:
    shared = recipe["shared"]
    spec = recipe["tracks"][track]
    prefix = str(spec["env_prefix"])
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    if wandb_mode == "online" and not wandb_credentials_present(env):
        raise RuntimeError("wandb online 需要先 `wandb login` 或设置 WANDB_API_KEY（不要写入 yaml/Git）")

    if track in {"T1-P20", "T2-P30"}:
        if source_roots is None:
            raise ValueError(f"{track} requires --source-roots")
        ids, roots = _load_source_roots(source_roots)
        source_weights = ["1"] * len(ids)
    else:
        ids = [str(item) for item in spec["local_repo_ids"]]
        if forward_root is None or reverse_root is None:
            raise ValueError("T3 requires --forward-root and --reverse-root")
        roots = [str(forward_root), str(reverse_root)]
        source_weights = [str(item) for item in spec["source_weights"]]
        if len(ids) != len(roots) or len(ids) != len(source_weights):
            raise ValueError("T3 local repo ids/roots/weights length mismatch")

    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join((str(ROOT / "third_party/axis_openpi/src"), str(ROOT))),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cuda"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90"),
            "HF_HUB_OFFLINE": "1" if shared.get("hf_hub_offline", True) else "0",
            "WANDB_MODE": "online" if wandb_mode == "online" else "disabled",
            "WANDB_PROJECT": wandb_project,
            "AXIS_PI05_WANDB_MODE": wandb_mode,
            "AXIS_PI05_WANDB_PROJECT": wandb_project,
            "AXIS_PI05_DISABLE_ACTIVATION_SHARDING_CONSTRAINT": "1",
            "AXIS_PI05_PYTHON": str(python),
            "AXIS_PI05_LORA_ARM": str(shared["lora_arm"]),
            "AXIS_PI05_FAIL_ON_NONFINITE": "1" if shared.get("fail_on_nonfinite", True) else "0",
            "OPENPI_KEEP_STEPS": _keep_steps(shared["keep_steps"]),
            f"{prefix}STEPS": str(steps),
            f"{prefix}WARMUP_STEPS": str(spec["warmup_steps"]),
            f"{prefix}PEAK_LR": str(spec["peak_lr"]),
            f"{prefix}DECAY_LR": str(spec["decay_lr"]),
            f"{prefix}MICROBATCH": str(microbatch),
            f"{prefix}ACCUMULATION": str(accumulation),
            f"{prefix}EFFECTIVE_BATCH": str(shared["effective_global_batch"]),
            f"{prefix}GLOBAL_BATCH": str(shared["effective_global_batch"]),
            f"{prefix}NORM_STATS": str(norm),
            f"{prefix}ASSET_ID": str(spec["asset_id"]),
            f"{prefix}LOCAL_REPO_IDS": _comma(ids),
            f"{prefix}LOCAL_REPO_ROOTS": os.pathsep.join(roots),
            f"{prefix}SOURCE_WEIGHTS": _comma(source_weights),
            f"{prefix}SAMPLING_MANIFEST": str(sampling),
            f"{prefix}PARAMS": str(params),
            f"{prefix}RUN_ID": run_id,
            f"{prefix}RUN_DIR": str(run_dir),
        }
    )
    return env


def live_train_processes() -> str:
    result = subprocess.run(["pgrep", "-af", "scripts/train.py"], capture_output=True, text=True)
    return result.stdout.strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--track", choices=TRACKS, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--source-roots", type=Path, default=None)
    parser.add_argument("--forward-root", type=Path, default=None)
    parser.add_argument("--reverse-root", type=Path, default=None)
    parser.add_argument("--microbatch", type=int, default=None)
    parser.add_argument("--wandb", choices=("disabled", "online"), default=None)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--print-env", action="store_true", help="print resolved env and exit without training")
    return parser.parse_args(argv)


def resolve_launch(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.steps > MAX_FRESH_CANARY_STEPS:
        raise ValueError(
            "refusing formal continuation from the fresh-R6 launcher: "
            f"requested {args.steps} steps, maximum is {MAX_FRESH_CANARY_STEPS}; "
            "use tools.axis_pi05.run_track12_long_horizon_lora for H3K/H10K "
            "complete-cosine training; do not resume a canary100 train_state"
        )
    recipe = load_launch_config(args.config)
    spec = recipe["tracks"][args.track]
    mix_dir = resolve_path(spec["mix_dir"])
    run_dir = args.run_dir.expanduser().resolve()
    log = (args.log or run_dir.with_name(run_dir.name + ".train.log")).expanduser().resolve()
    run_id = args.run_id or run_dir.name
    microbatch = int(args.microbatch or recipe["shared"]["microbatch"])
    accumulation = int(recipe["shared"]["effective_global_batch"]) // microbatch
    if microbatch * accumulation != int(recipe["shared"]["effective_global_batch"]):
        raise ValueError("microbatch must divide effective_global_batch 64")
    wandb_mode = args.wandb or str(recipe["wandb"]["default_mode"])
    wandb_project = str(recipe["wandb"]["project"])
    norm = (args.norm or mix_dir / "norm_stats.json").expanduser().resolve()
    sampling = (args.sampling or mix_dir / "sampling_weights.parquet").expanduser().resolve()
    params = args.params.expanduser().resolve()
    python = args.python.expanduser().resolve()
    env = build_environment(
        track=args.track,
        recipe=recipe,
        steps=args.steps,
        run_dir=run_dir,
        run_id=run_id,
        params=params,
        norm=norm,
        sampling=sampling,
        python=python,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        source_roots=None if args.source_roots is None else args.source_roots.expanduser().resolve(),
        forward_root=None if args.forward_root is None else args.forward_root.expanduser().resolve(),
        reverse_root=None if args.reverse_root is None else args.reverse_root.expanduser().resolve(),
        microbatch=microbatch,
        accumulation=accumulation,
    )
    return {
        "recipe": recipe,
        "spec": spec,
        "run_dir": run_dir,
        "log": log,
        "run_id": run_id,
        "python": python,
        "openpi_config": spec["openpi_config"],
        "wandb_mode": wandb_mode,
        "env": env,
    }


def launch(resolved: dict[str, Any], *, detach: bool) -> int:
    run_dir: Path = resolved["run_dir"]
    log: Path = resolved["log"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to start: run directory already exists: {run_dir}")
    try:
        log.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError(f"log must live outside the run directory: {log}")
    live = live_train_processes()
    if "train.py" in live:
        raise RuntimeError(f"refusing to start: train.py already running:\n{live}")
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [str(resolved["python"]), "-u", str(TRAIN), str(resolved["openpi_config"])]
    if detach:
        handle = log.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=resolved["env"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid_path = log.with_suffix(log.suffix + ".pid")
        pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
        print(json.dumps({"status": "started", "pid": proc.pid, "run_dir": str(run_dir), "log": str(log), "wandb": resolved["wandb_mode"]}, ensure_ascii=False))
        return 0
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(ROOT), env=resolved["env"], stdout=handle, stderr=subprocess.STDOUT)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolved = resolve_launch(args)
    if args.print_env:
        public = {
            key: value
            for key, value in resolved["env"].items()
            if key.startswith("AXIS_PI05_") or key in {"WANDB_MODE", "WANDB_PROJECT", "OPENPI_KEEP_STEPS", "PYTHONPATH"}
        }
        print(json.dumps({"track": args.track, "run_dir": str(resolved["run_dir"]), "log": str(resolved["log"]), "wandb": resolved["wandb_mode"], "env": public}, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return launch(resolved, detach=args.detach)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
