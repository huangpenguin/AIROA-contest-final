#!/usr/bin/env python3
"""Official T2 long-horizon LoRA runner for the 4-day plan.

This is the evaluation-gated entrypoint that the fresh-canary launcher
refuses to become.  It separates cosine ``schedule_horizon_steps`` from
``stop_after_step``, freezes a run identity, loads R6 params only, and
rejects canary100 train_state.  Default is resolve/preflight only; training
requires ``--execute``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import ROOT, canonical_json, sha256_file, sha256_bytes
from .launch_official_blackwell import (
    DEFAULT_CONFIG as CANARY_LAUNCH_CONFIG,
    DEFAULT_PARAMS,
    TRAIN,
    live_train_processes,
    load_launch_config,
    wandb_credentials_present,
)


JST = timezone(timedelta(hours=9))
DEFAULT_PYTHON = Path(os.environ.get("AXIS_PI05_PYTHON", str(ROOT / ".venv_axis_pi05/bin/python")))
DEFAULT_RECIPE = ROOT / "configs/axis_pi05/track12_long_horizon_lora.yaml"
DEFAULT_T2_MIX = ROOT / "data/manifests_track12/T2-P30"
LOCKED_T2_NORM = (
    ROOT / "outputs/axis_pi05/official_canary100/t2_p30_canary100/100/assets/T2-P30/norm_stats.json"
)
CANARY_RUN_IDS = {
    "t1_p20_canary100",
    "t2_p30_canary100",
    "t3_four_gold_canary100",
    "t1_p20_official250",
    "t2_p30_official250",
}
OFFICIAL_RUN_IDS = {
    "t2_p30_r6_h10k_r1": {"schedule_horizon_steps": 10000, "stop_after_step": 10000, "save_steps": (1, 10, 100, 1000, 3000, 10000)},
    "t2_p30_r6_h3k_r1": {"schedule_horizon_steps": 3000, "stop_after_step": 3000, "save_steps": (1, 10, 100, 1000, 3000)},
}
PROBE_RUN_IDS = {
    "t2_p30_r6_h10k_throughput_probe_r1": {"schedule_horizon_steps": 10000, "stop_after_step": 100, "save_steps": (1, 10, 100)},
    "t2_p30_r6_h3k_throughput_probe_r1": {"schedule_horizon_steps": 3000, "stop_after_step": 100, "save_steps": (1, 10, 100)},
}
ALLOWED_OFFICIAL_HORIZONS = (3000, 10000)
MAX_PROBE_STEPS = 100
EXPECTED_T2_MANIFEST_SHA = "85700f9d85311cdadbacadcecf2b7c9b6617b77e03f51de42a81681ffc770af4"
EXPECTED_T2_NORM_SHA = "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41"
IDENTITY_COMPARE_FIELDS = (
    "schema_version",
    "run_id",
    "track",
    "mix_id",
    "mode_family",
    "initialization",
    "schedule_horizon_steps",
    "stop_after_step",
    "save_steps",
    "manifest_sha256",
    "norm_stats_sha256",
    "sampling_sha256",
    "peak_lr",
    "warmup_steps",
    "decay_lr",
    "microbatch",
    "accumulation",
    "effective_global_batch",
    "lora_arm",
    "trainer_sha256",
    "config_sha256",
    "runner_sha256",
)


def strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON top level must be an object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def _comma_ints(values: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(int(item)) for item in values)


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


def assert_params_are_r6_only(params: Path) -> None:
    resolved = params.expanduser().resolve()
    # Keep the caller path too: official Blackwell may symlink R6 params,
    # and Path.resolve() would otherwise drop the required identity substring.
    text = resolved.as_posix() + " " + params.expanduser().as_posix()
    if "canary100" in text:
        raise ValueError(f"refusing canary100 params as initialization: {resolved}")
    if resolved.name == "train_state" or "train_state" in resolved.parts[-2:]:
        raise ValueError(f"refusing train_state as params-only init: {resolved}")
    if resolved.name != "params":
        raise ValueError(f"params path must end in /params, got {resolved}")
    if "r6_dual_l2_fresh500_r1" not in text:
        raise ValueError(f"params must be the R6 step500 params-only tree: {resolved}")


def assert_not_canary_run(run_id: str, run_dir: Path) -> None:
    if run_id in CANARY_RUN_IDS or "canary100" in run_id:
        raise ValueError(f"refusing canary100 run identity: {run_id}")
    text = run_dir.as_posix()
    if "canary100" in text:
        raise ValueError(f"refusing canary100 run directory: {run_dir}")


def default_save_steps(*, schedule_horizon_steps: int, stop_after_step: int, mode: str) -> tuple[int, ...]:
    if mode == "throughput-probe":
        return tuple(step for step in (1, 10, 100) if step <= stop_after_step)
    if schedule_horizon_steps == 10000:
        return (1, 10, 100, 1000, 3000, 10000)
    if schedule_horizon_steps == 3000:
        return (1, 10, 100, 1000, 3000)
    raise ValueError(f"unsupported official horizon {schedule_horizon_steps}")


def parse_save_steps(raw: str | None, *, schedule_horizon_steps: int, stop_after_step: int, mode: str) -> tuple[int, ...]:
    if not raw:
        return default_save_steps(
            schedule_horizon_steps=schedule_horizon_steps, stop_after_step=stop_after_step, mode=mode
        )
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("save_steps must be positive integers")
    if any(item > stop_after_step for item in values):
        raise ValueError("save_steps cannot exceed stop_after_step")
    return values


def identity_payload(fields: dict[str, Any]) -> dict[str, Any]:
    payload = {key: fields[key] for key in IDENTITY_COMPARE_FIELDS}
    payload["schema_version"] = "track12_long_horizon_identity_v1"
    digest = sha256_bytes(canonical_json({key: payload[key] for key in IDENTITY_COMPARE_FIELDS}))
    payload["identity_sha256"] = digest
    return payload


def identity_path(run_dir: Path) -> Path:
    return run_dir / "run_identity.json"


def compare_identity(expected: dict[str, Any], observed: dict[str, Any]) -> None:
    for field in IDENTITY_COMPARE_FIELDS:
        if expected.get(field) != observed.get(field):
            raise ValueError(f"identity drift on {field}: expected {expected.get(field)!r}, observed {observed.get(field)!r}")
    if expected.get("identity_sha256") != observed.get("identity_sha256"):
        raise ValueError("frozen identity hash mismatch")


def refuse_canary_resume_source(run_dir: Path) -> None:
    existing = identity_path(run_dir)
    if existing.is_file():
        identity = strict_json_load(existing)
        run_id = str(identity.get("run_id", ""))
        if run_id in CANARY_RUN_IDS or identity.get("mode_family") == "canary100" or int(identity.get("schedule_horizon_steps") or 0) == 100:
            if run_id in CANARY_RUN_IDS or "canary" in run_id:
                raise ValueError(f"refusing canary100 checkpoint as resume source: {run_dir}")
    if "canary100" in run_dir.as_posix():
        raise ValueError(f"refusing canary100 checkpoint as resume source: {run_dir}")
    marker = run_dir / "train_state"
    sibling_identity = run_dir / "run_identity.json"
    if marker.is_dir() and not sibling_identity.is_file():
        # A raw Orbax tree without this runner's identity is not a legal resume source.
        raise ValueError(f"refusing unlabeled checkpoint tree as resume source: {run_dir}")


def resolve_norm(path: Path | None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    mix_norm = DEFAULT_T2_MIX / "norm_stats.json"
    if mix_norm.is_file():
        return mix_norm.resolve()
    if LOCKED_T2_NORM.is_file():
        return LOCKED_T2_NORM.resolve()
    raise FileNotFoundError("T2-P30 norm_stats.json missing from mix dir and canary assets")


def build_identity(
    *,
    mode: str,
    run_id: str,
    track: str,
    schedule_horizon_steps: int,
    stop_after_step: int,
    save_steps: tuple[int, ...],
    manifest: Path,
    norm: Path,
    sampling: Path,
    params: Path,
    microbatch: int,
    accumulation: int,
) -> dict[str, Any]:
    mode_family = "throughput_probe" if mode == "throughput-probe" else "official_complete_cosine"
    payload = identity_payload(
        {
            "schema_version": "track12_long_horizon_identity_v1",
            "run_id": run_id,
            "track": track,
            "mix_id": "T2-P30",
            "mode_family": mode_family,
            "initialization": "r6_dual_l2_fresh500_r1_step500_params_only",
            "schedule_horizon_steps": int(schedule_horizon_steps),
            "stop_after_step": int(stop_after_step),
            "save_steps": list(save_steps),
            "manifest_sha256": sha256_file(manifest),
            "norm_stats_sha256": sha256_file(norm),
            "sampling_sha256": sha256_file(sampling),
            "peak_lr": "3e-7",
            "warmup_steps": 10,
            "decay_lr": "3e-8",
            "microbatch": int(microbatch),
            "accumulation": int(accumulation),
            "effective_global_batch": 64,
            "lora_arm": "L2_action_lora_heads",
            "trainer_sha256": sha256_file(TRAIN),
            "config_sha256": sha256_file(ROOT / "third_party/axis_openpi/src/openpi/training/config.py"),
            "runner_sha256": sha256_file(Path(__file__)),
        }
    )
    payload["params_path"] = str(params)
    payload["manifest_path"] = str(manifest)
    payload["norm_path"] = str(norm)
    payload["sampling_path"] = str(sampling)
    return payload


def validate_contract(
    *,
    mode: str,
    track: str,
    run_id: str,
    schedule_horizon_steps: int,
    stop_after_step: int,
    save_steps: tuple[int, ...],
    manifest: Path,
    norm: Path,
    params: Path,
) -> None:
    if track != "T2-P30":
        raise ValueError("4-day official runner only authorizes T2-P30; T1 is conditional after T2 gain")
    if schedule_horizon_steps not in ALLOWED_OFFICIAL_HORIZONS:
        raise ValueError(
            f"competition horizon must be 3000 or 10000 complete cosine, not {schedule_horizon_steps}; 30k is not selected"
        )
    if stop_after_step <= 0:
        raise ValueError("stop_after_step must be positive")
    if stop_after_step > schedule_horizon_steps:
        raise ValueError("stop_after_step cannot exceed schedule_horizon_steps")
    assert_params_are_r6_only(params)
    if sha256_file(manifest) != EXPECTED_T2_MANIFEST_SHA:
        raise ValueError("T2-P30 manifest hash mismatch")
    if sha256_file(norm) != EXPECTED_T2_NORM_SHA:
        raise ValueError("T2-P30 norm hash mismatch")
    if mode == "throughput-probe":
        if run_id not in PROBE_RUN_IDS:
            raise ValueError(f"throughput probe run_id must be one of {sorted(PROBE_RUN_IDS)}")
        spec = PROBE_RUN_IDS[run_id]
        if stop_after_step > MAX_PROBE_STEPS:
            raise ValueError(f"throughput probe stop_after_step max is {MAX_PROBE_STEPS}")
        if schedule_horizon_steps != spec["schedule_horizon_steps"]:
            raise ValueError("throughput probe schedule_horizon_steps must match the named probe identity")
        return
    if mode in {"new", "resume", "preflight", "print-env"}:
        if run_id in PROBE_RUN_IDS:
            raise ValueError("official modes cannot use a throughput probe run_id")
        if run_id not in OFFICIAL_RUN_IDS:
            raise ValueError(f"official run_id must be one of {sorted(OFFICIAL_RUN_IDS)}")
        spec = OFFICIAL_RUN_IDS[run_id]
        if schedule_horizon_steps != spec["schedule_horizon_steps"] or stop_after_step != spec["stop_after_step"]:
            raise ValueError(
                "official 4-day runs require a complete cosine: "
                f"{run_id} needs schedule={spec['schedule_horizon_steps']} stop={spec['stop_after_step']}"
            )
        if tuple(save_steps) != spec["save_steps"]:
            raise ValueError(f"official save_steps for {run_id} must be {spec['save_steps']}")


def build_environment(resolved: dict[str, Any], *, resume: bool) -> dict[str, str]:
    recipe = resolved["canary_recipe"]
    spec = recipe["tracks"]["T2-P30"]
    ids, roots = _load_source_roots(resolved["source_roots"])
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    wandb_mode = resolved["wandb_mode"]
    if wandb_mode == "online" and not wandb_credentials_present(env):
        raise RuntimeError("wandb online 需要先 `wandb login` 或设置 WANDB_API_KEY（不要写入 yaml/Git）")
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join((str(ROOT / "third_party/axis_openpi/src"), str(ROOT))),
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "JAX_PLATFORMS": env.get("JAX_PLATFORMS", "cuda"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90"),
            # Lerobot still pings huggingface.co for dataset refs even when
            # AXIS_PI05_TRACK12_LOCAL_REPO_ROOTS points at a complete NVMe extract.
            "HF_HUB_OFFLINE": "0",
            "WANDB_MODE": "online" if wandb_mode == "online" else "disabled",
            "WANDB_PROJECT": resolved["wandb_project"],
            "AXIS_PI05_WANDB_MODE": wandb_mode,
            "AXIS_PI05_WANDB_PROJECT": resolved["wandb_project"],
            "AXIS_PI05_DISABLE_ACTIVATION_SHARDING_CONSTRAINT": "1",
            "AXIS_PI05_PYTHON": str(resolved["python"]),
            "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
            "AXIS_PI05_FAIL_ON_NONFINITE": "1",
            "AXIS_PI05_SAVE_FINAL_CHECKPOINT": "1",
            "OPENPI_KEEP_STEPS": _comma_ints(resolved["save_steps"]),
            "OPENPI_MAX_TO_KEEP": str(max(4, len(resolved["save_steps"]))),
            "AXIS_PI05_TRACK12_STEPS": str(resolved["stop_after_step"]),
            "AXIS_PI05_TRACK12_SCHEDULE_HORIZON_STEPS": str(resolved["schedule_horizon_steps"]),
            "AXIS_PI05_TRACK12_STOP_AFTER_STEP": str(resolved["stop_after_step"]),
            # new/probe mkdir the run dir and write identity first. train.py must
            # see resume=True so an empty orbax tree is treated as a fresh start,
            # not FileExistsError. Actual optimizer resume remains args.mode=resume.
            "AXIS_PI05_TRACK12_RESUME": "1" if resume or resolved["mode"] in {"new", "throughput-probe"} else "0",
            "AXIS_PI05_TRACK12_IDENTITY_SHA": resolved["identity"]["identity_sha256"],
            "AXIS_PI05_TRACK12_WARMUP_STEPS": "10",
            "AXIS_PI05_TRACK12_PEAK_LR": "3e-7",
            "AXIS_PI05_TRACK12_DECAY_LR": "3e-8",
            "AXIS_PI05_TRACK12_MICROBATCH": str(resolved["microbatch"]),
            "AXIS_PI05_TRACK12_ACCUMULATION": str(resolved["accumulation"]),
            "AXIS_PI05_TRACK12_EFFECTIVE_BATCH": "64",
            "AXIS_PI05_TRACK12_GLOBAL_BATCH": "64",
            "AXIS_PI05_TRACK12_NORM_STATS": str(resolved["norm"]),
            "AXIS_PI05_TRACK12_ASSET_ID": "T2-P30",
            "AXIS_PI05_TRACK12_LOCAL_REPO_IDS": ",".join(ids),
            "AXIS_PI05_TRACK12_LOCAL_REPO_ROOTS": os.pathsep.join(roots),
            "AXIS_PI05_TRACK12_SOURCE_WEIGHTS": ",".join(["1"] * len(ids)),
            "AXIS_PI05_TRACK12_SAMPLING_MANIFEST": str(resolved["sampling"]),
            "AXIS_PI05_TRACK12_PARAMS": str(resolved["params"]),
            "AXIS_PI05_TRACK12_RUN_ID": resolved["run_id"],
            "AXIS_PI05_TRACK12_RUN_DIR": str(resolved["run_dir"]),
            "AXIS_PI05_TRACK12_OUTPUT_DIR": str(resolved["run_dir"].parent),
        }
    )
    return env


def step_contract(env: dict[str, str]) -> dict[str, Any]:
    schedule = int(env["AXIS_PI05_TRACK12_SCHEDULE_HORIZON_STEPS"])
    stop = int(env["AXIS_PI05_TRACK12_STOP_AFTER_STEP"])
    return {
        "schedule_horizon_steps": schedule,
        "stop_after_step": stop,
        "legacy_steps": int(env["AXIS_PI05_TRACK12_STEPS"]),
        "resume": env.get("AXIS_PI05_TRACK12_RESUME") == "1",
        "stop_does_not_change_horizon": stop != schedule or schedule in ALLOWED_OFFICIAL_HORIZONS,
        "cosine_decay_steps": max(schedule, int(env.get("AXIS_PI05_TRACK12_WARMUP_STEPS", "10")) + 1),
        "num_train_steps": stop,
    }


def estimate_fit(
    *,
    seconds_per_optimizer_step: float,
    horizon_steps: int,
    deadline: datetime,
    now: datetime | None = None,
    freeze_reserve_hours: float = 24.0,
    eval_pack_hours: float = 6.0,
    fluctuation: float = 0.15,
    checkpoint_hours: float = 1.0,
) -> dict[str, Any]:
    now = now or datetime.now(tz=deadline.tzinfo or JST)
    freeze = deadline - timedelta(hours=freeze_reserve_hours)
    train_hours = (seconds_per_optimizer_step * horizon_steps * (1.0 + fluctuation)) / 3600.0 + checkpoint_hours
    remaining_hours = (freeze - now).total_seconds() / 3600.0 - eval_pack_hours
    return {
        "horizon_steps": horizon_steps,
        "seconds_per_optimizer_step": seconds_per_optimizer_step,
        "train_hours_with_margin": train_hours,
        "remaining_hours_before_freeze_minus_eval": remaining_hours,
        "deadline": deadline.isoformat(),
        "freeze": freeze.isoformat(),
        "fits": remaining_hours >= train_hours,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "print-env", "new", "resume", "throughput-probe"), required=True)
    parser.add_argument("--track", choices=("T2-P30",), default="T2-P30")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--schedule-horizon-steps", type=int, required=True)
    parser.add_argument("--stop-after-step", type=int, required=True)
    parser.add_argument("--save-steps", default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--canary-config", type=Path, default=CANARY_LAUNCH_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--norm", type=Path, default=None)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_T2_MIX / "manifest.json")
    parser.add_argument("--source-roots", type=Path, default=DEFAULT_T2_MIX / "source_roots.json")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--wandb", choices=("disabled", "online"), default="disabled")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--execute", action="store_true", help="actually launch train.py; default is resolve only")
    return parser.parse_args(argv)


def resolve_launch(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    assert_not_canary_run(args.run_id, run_dir)
    save_steps = parse_save_steps(
        args.save_steps,
        schedule_horizon_steps=args.schedule_horizon_steps,
        stop_after_step=args.stop_after_step,
        mode=args.mode,
    )
    mix_dir = DEFAULT_T2_MIX
    manifest = args.manifest.expanduser().resolve()
    sampling = (args.sampling or mix_dir / "sampling_weights.parquet").expanduser().resolve()
    source_roots = args.source_roots.expanduser().resolve()
    params = args.params.expanduser().resolve()
    norm = resolve_norm(args.norm)
    validate_contract(
        mode=args.mode,
        track=args.track,
        run_id=args.run_id,
        schedule_horizon_steps=args.schedule_horizon_steps,
        stop_after_step=args.stop_after_step,
        save_steps=save_steps,
        manifest=manifest,
        norm=norm,
        params=params,
    )
    missing = [str(path) for path in (manifest, sampling, source_roots, params, norm, TRAIN) if not path.exists()]
    if missing:
        raise FileNotFoundError("missing long-horizon input(s): " + ", ".join(missing))
    if args.microbatch <= 0 or 64 % args.microbatch != 0:
        raise ValueError("microbatch must divide effective_global_batch 64")
    accumulation = 64 // args.microbatch
    identity = build_identity(
        mode=args.mode,
        run_id=args.run_id,
        track=args.track,
        schedule_horizon_steps=args.schedule_horizon_steps,
        stop_after_step=args.stop_after_step,
        save_steps=save_steps,
        manifest=manifest,
        norm=norm,
        sampling=sampling,
        params=params,
        microbatch=args.microbatch,
        accumulation=accumulation,
    )
    log = (args.log or run_dir.with_name(run_dir.name + ".train.log")).expanduser().resolve()
    try:
        log.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError(f"log must live outside the run directory: {log}")
    if args.mode == "new" and run_dir.exists():
        raise FileExistsError(f"refusing to start: run directory already exists; pass --mode resume: {run_dir}")
    if args.mode == "resume":
        refuse_canary_resume_source(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"resume requires an existing run directory: {run_dir}")
        compare_identity(strict_json_load(identity_path(run_dir)), identity)
    if args.mode not in {"new", "resume", "throughput-probe", "preflight", "print-env"}:
        raise ValueError(args.mode)
    if args.mode in {"new", "resume", "throughput-probe"} and run_dir.exists() and args.mode != "resume":
        raise FileExistsError(f"refusing to start: run directory already exists: {run_dir}")
    canary_recipe = load_launch_config(args.canary_config)
    resolved = {
        "mode": args.mode,
        "track": args.track,
        "run_id": args.run_id,
        "run_dir": run_dir,
        "log": log,
        "schedule_horizon_steps": args.schedule_horizon_steps,
        "stop_after_step": args.stop_after_step,
        "save_steps": save_steps,
        "manifest": manifest,
        "norm": norm,
        "sampling": sampling,
        "source_roots": source_roots,
        "params": params,
        # Do not Path.resolve() the interpreter: venv/bin/python is often a
        # symlink to a uv/standalone CPython, and resolving it drops site-packages.
        "python": args.python.expanduser().absolute(),
        "microbatch": args.microbatch,
        "accumulation": accumulation,
        "wandb_mode": args.wandb,
        "wandb_project": str(canary_recipe["wandb"]["project"]),
        "identity": identity,
        "canary_recipe": canary_recipe,
        "openpi_config": "axis_pi05_track12_lora",
        "execute": bool(args.execute),
        "training_started": False,
    }
    resolved["env"] = build_environment(resolved, resume=args.mode == "resume")
    resolved["step_contract"] = step_contract(resolved["env"])
    return resolved


def preflight_disk() -> dict[str, Any]:
    result = subprocess.run(["df", "-k", "/home"], capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    fields = lines[-1].split()
    avail_kib = int(fields[3])
    return {"home_avail_gib": round(avail_kib / 1024 / 1024, 2), "df": result.stdout.strip()}


def launch(resolved: dict[str, Any], *, detach: bool) -> int:
    run_dir: Path = resolved["run_dir"]
    log: Path = resolved["log"]
    live = live_train_processes()
    if "train.py" in live:
        raise RuntimeError(f"refusing to start: train.py already running:\n{live}")
    if resolved["mode"] == "new" or resolved["mode"] == "throughput-probe":
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(identity_path(run_dir), resolved["identity"])
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
        log.with_suffix(log.suffix + ".pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        print(json.dumps({"status": "started", "pid": proc.pid, "run_dir": str(run_dir), "log": str(log), "identity_sha256": resolved["identity"]["identity_sha256"]}, ensure_ascii=False))
        return 0
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(ROOT), env=resolved["env"], stdout=handle, stderr=subprocess.STDOUT)
    return result.returncode


def public_env(env: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if key.startswith("AXIS_PI05_") or key in {"WANDB_MODE", "WANDB_PROJECT", "OPENPI_KEEP_STEPS", "PYTHONPATH"}
    }


def materialize_locked_t2_norm(*, mix_dir: Path = DEFAULT_T2_MIX) -> dict[str, Any]:
    destination = mix_dir / "norm_stats.json"
    source = LOCKED_T2_NORM
    if not source.is_file():
        raise FileNotFoundError(f"locked T2 norm missing: {source}")
    digest = sha256_file(source)
    if digest != EXPECTED_T2_NORM_SHA:
        raise ValueError(f"locked T2 norm hash mismatch: {digest}")
    if destination.is_file():
        if sha256_file(destination) != EXPECTED_T2_NORM_SHA:
            raise ValueError(f"existing mix-dir T2 norm is not the frozen SHA: {destination}")
        copied = False
    else:
        shutil.copy2(source, destination)
        copied = True
        if sha256_file(destination) != EXPECTED_T2_NORM_SHA:
            destination.unlink()
            raise ValueError("copied T2 norm is not byte-identical to the frozen SHA")
    sidecar = mix_dir / "norm_stats.sha256"
    atomic_json(sidecar, {"path": "norm_stats.json", "sha256": EXPECTED_T2_NORM_SHA})
    return {"path": str(destination), "sha256": EXPECTED_T2_NORM_SHA, "copied": copied, "source": str(source)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resolved = resolve_launch(args)
    public = {
        "mode": resolved["mode"],
        "run_id": resolved["run_id"],
        "run_dir": str(resolved["run_dir"]),
        "log": str(resolved["log"]),
        "schedule_horizon_steps": resolved["schedule_horizon_steps"],
        "stop_after_step": resolved["stop_after_step"],
        "save_steps": list(resolved["save_steps"]),
        "identity_sha256": resolved["identity"]["identity_sha256"],
        "step_contract": resolved["step_contract"],
        "training_started": False,
        "execute": resolved["execute"],
    }
    if args.mode == "preflight":
        public["disk"] = preflight_disk()
        public["params_ok"] = resolved["params"].is_dir()
        public["norm_sha256"] = resolved["identity"]["norm_stats_sha256"]
        public["manifest_sha256"] = resolved["identity"]["manifest_sha256"]
        print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.mode == "print-env" or not args.execute:
        public["env"] = public_env(resolved["env"])
        print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return launch(resolved, detach=args.detach)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
