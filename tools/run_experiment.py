#!/usr/bin/env python3
"""PARC2026 outer runner: config validation, training, local eval and packaging gates.

The official training/evaluation implementation remains the source of truth.  This
file only validates configuration, records provenance and invokes those entrypoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OVERRIDES = {
    "run.tag", "run.output_dir", "run.seed", "train.steps",
    "wandb.mode", "hardware.gpu_ids",
}
EXPECTED_KEYS = {
    "run": {"tag", "output_dir", "seed", "allow_dirty", "smoke_allow_dirty"},
    "model": {"name", "base", "revision", "weights_sha256"},
    "dataset": {"repo_id", "root", "schema", "revision", "manifest_sha256"},
    "mix": {"path", "sha256"},
    "train": {"entry", "smoke_entry", "steps", "smoke_steps", "batch_size",
              "gradient_accumulation", "save_steps", "lora_rank", "learning_rate",
              "num_workers", "video_backend", "action_chunk", "inference_steps",
              "precision"},
    "wandb": {"mode", "project", "disable_artifact"},
    "hardware": {"gpu_ids"},
    "eval": {"tracks", "n_episodes", "max_steps", "timeout_seconds", "output_dir"},
    "submission": {"directory", "zip", "assignment_ids"},
}
SECRET_WORDS = ("token", "api_key", "secret", "password")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    try:
        cfg = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        fail(f"config を読めません: {path}: {exc}")
    if not isinstance(cfg, dict):
        fail("config のトップレベルは mapping である必要があります")
    unknown_sections = set(cfg) - set(EXPECTED_KEYS)
    if unknown_sections:
        fail(f"未知の config セクション: {sorted(unknown_sections)}")
    for section, keys in EXPECTED_KEYS.items():
        if section not in cfg or not isinstance(cfg[section], dict):
            fail(f"config セクションがありません: {section}")
        unknown = set(cfg[section]) - keys
        if unknown:
            fail(f"{section} に未知のフィールドがあります: {sorted(unknown)}")

    for raw in overrides:
        if "=" not in raw:
            fail(f"override は key=value 形式です: {raw}")
        key, raw_value = raw.split("=", 1)
        if key not in ALLOWED_OVERRIDES:
            fail(f"CLI override は許可されていません: {key}")
        section, field = key.split(".", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            fail(f"override の値を YAML として解釈できません: {raw}: {exc}")
        if isinstance(value, (dict, list)):
            fail(f"override は scalar のみ許可されます: {key}")
        cfg[section][field] = value

    cfg = expand(cfg)
    validate_values(cfg)
    return cfg


def validate_values(cfg: dict[str, Any]) -> None:
    if cfg["run"]["seed"] is None:
        fail("run.seed は必須です")
    if cfg["train"]["steps"] <= 0 or cfg["train"]["smoke_steps"] <= 0:
        fail("train.steps と train.smoke_steps は正数である必要があります")
    if cfg["train"]["action_chunk"] < cfg["train"]["inference_steps"]:
        fail("action_chunk は inference_steps 以上である必要があります")
    if cfg["wandb"]["mode"] not in {"online", "offline", "disabled"}:
        fail("wandb.mode は online/offline/disabled のいずれかです")
    if not cfg["eval"]["tracks"] or any(t not in {"track1", "track2", "track3"}
                                            for t in cfg["eval"]["tracks"]):
        fail("eval.tracks は track1/track2/track3 のリストです")
    if cfg["eval"]["timeout_seconds"] > 10:
        fail("eval.timeout_seconds は公式上限 10 秒以下にしてください")


def yaml_text(cfg: dict[str, Any]) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True,
                                capture_output=True, check=True)
        return result.stdout.strip()
    status = git("status", "--porcelain")
    diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--binary"],
                          capture_output=True, check=True).stdout
    cached = subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--binary"],
                            capture_output=True, check=True).stdout
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status": status.splitlines(),
        "diff_sha256": hashlib.sha256(diff + cached).hexdigest(),
    }


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def redact(value: Any, key: str = "") -> Any:
    if any(word in key.lower() for word in SECRET_WORDS):
        return "<redacted>"
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    return value


def path_from_config(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def output_paths(cfg: dict[str, Any], smoke: bool = False) -> tuple[Path, Path]:
    configured = path_from_config(cfg["run"]["output_dir"])
    if smoke:
        actual = configured.parent / f"smoke_pi05_{cfg['run']['tag']}"
    else:
        actual = configured
    return actual, actual.parent


def base_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["RUN_NAME"] = str(cfg["run"]["tag"])
    env["OUT_ROOT"] = str(path_from_config(cfg["run"]["output_dir"]).parent)
    env["LOG_ROOT"] = str(path_from_config(cfg["run"]["output_dir"]).parent / "logs")
    env["PI05_BS"] = str(cfg["train"]["batch_size"])
    env["PI05_GA"] = str(cfg["train"]["gradient_accumulation"])
    env["PI05_STEPS"] = str(cfg["train"]["steps"])
    env["PI05_SAVE_STEPS"] = str(cfg["train"]["save_steps"])
    env["PI05_LORA_R"] = str(cfg["train"]["lora_rank"])
    env["PI05_LR"] = str(cfg["train"]["learning_rate"])
    env["DATALOADER_NUM_WORKERS"] = str(cfg["train"]["num_workers"])
    env["PI05_DATASET_REPO_ID"] = str(cfg["dataset"]["repo_id"])
    env["PI05_DATASET_ROOT"] = str(path_from_config(cfg["dataset"]["root"]))
    env["PI05_VIDEO_BACKEND"] = str(cfg["train"]["video_backend"])
    env["WANDB_MODE"] = str(cfg["wandb"]["mode"])
    env["WANDB_PROJECT"] = str(cfg["wandb"]["project"])
    if cfg["hardware"]["gpu_ids"] not in (None, ""):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["hardware"]["gpu_ids"])
    return env


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def ensure_gate(cfg: dict[str, Any], *, smoke: bool) -> tuple[Path, Path]:
    info = git_info()
    if info["dirty"] and not (smoke and cfg["run"]["smoke_allow_dirty"]):
        fail("git worktree が dirty です。正式実行前に commit/stash するか、smoke のみ明示許可してください")
    actual, _ = output_paths(cfg, smoke=smoke)
    if actual.exists():
        fail(f"出力先が既に存在します（自動上書きしません）: {actual}")
    dataset_root = path_from_config(cfg["dataset"]["root"])
    if not dataset_root.exists():
        fail(f"dataset root がありません。公式手順で展開してください: {dataset_root}")
    if cfg["wandb"]["mode"] == "online" and not os.environ.get("WANDB_API_KEY"):
        fail("wandb.mode=online には環境変数 WANDB_API_KEY が必要です（config には書かない）")
    if cfg["model"]["name"] == "pi05" and not os.environ.get("HF_TOKEN"):
        fail("pi0.5 の gated weight 取得には HF_TOKEN が必要です（config には書かない）")
    return actual, Path(cfg["run"]["output_dir"])


def record_run(cfg: dict[str, Any], command: list[str], actual: Path,
               *, dry_run: bool, smoke: bool) -> tuple[Path, dict[str, Any]]:
    info = git_info()
    record_dir = actual.parent / ".parc_runs" / cfg["run"]["tag"]
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(ROOT),
        "git": info,
        "config": redact(cfg),
        "command": command,
        "command_shell": command_string(command),
        "dry_run": dry_run,
        "smoke": smoke,
        "dataset_root": str(path_from_config(cfg["dataset"]["root"])),
        "dataset_manifest_sha256": cfg["dataset"].get("manifest_sha256", ""),
        "model_weights_sha256": cfg["model"].get("weights_sha256", ""),
        "environment": {
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "wandb_url": os.environ.get("WANDB_RUN_URL", ""),
        },
    }
    if dry_run:
        return record_dir, manifest
    record_dir.mkdir(parents=True, exist_ok=False)
    (record_dir / "resolved_config.yaml").write_text(yaml_text(cfg))
    (record_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return record_dir, manifest


def copy_records(record_dir: Path, actual: Path) -> None:
    if not actual.is_dir():
        return
    for name in ("resolved_config.yaml", "run_manifest.json"):
        source = record_dir / name
        if source.is_file():
            shutil.copy2(source, actual / name)


def run_training(cfg: dict[str, Any], smoke: bool) -> int:
    actual, _ = ensure_gate(cfg, smoke=smoke)
    entry_key = "smoke_entry" if smoke else "entry"
    entry = ROOT / cfg["train"][entry_key]
    if not entry.is_file():
        fail(f"training entry がありません: {entry}")
    command = ["bash", str(entry)]
    record_dir, _ = record_run(cfg, command, actual, dry_run=False, smoke=smoke)
    env = base_env(cfg)
    if smoke:
        env.update({"TAG": cfg["run"]["tag"], "SMOKE_STEPS": str(cfg["train"]["smoke_steps"]),
                    "SMOKE_BS": str(cfg["train"]["batch_size"]),
                    "SMOKE_GA": str(cfg["train"]["gradient_accumulation"])})
    print(f"[runner] training: {command_string(command)}")
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode == 0:
        copy_records(record_dir, actual)
    return result.returncode


def eval_command(cfg: dict[str, Any], server_url: str) -> list[str]:
    output = path_from_config(cfg["eval"]["output_dir"])
    return [sys.executable, "-m", "pipeline", "--server-url", server_url,
            "--track", *cfg["eval"]["tracks"], "--n-episodes", str(cfg["eval"]["n_episodes"]),
            "--max-steps", str(cfg["eval"]["max_steps"]), "--timeout",
            str(cfg["eval"]["timeout_seconds"]), "--seed", str(cfg["run"]["seed"]),
            "--output-dir", str(output)]


def run_eval(cfg: dict[str, Any], server_url: str, dry_run: bool) -> int:
    command = eval_command(cfg, server_url)
    print(command_string(command))
    if dry_run:
        return 0
    env = os.environ.copy()
    paths = [ROOT / "LIBERO-plus", ROOT, ROOT / "compe"]
    env["PYTHONPATH"] = ":".join(str(p) for p in paths if p.exists()) + ":" + env.get("PYTHONPATH", "")
    return subprocess.run(command, cwd=ROOT, env=env).returncode


def run_validate(target: Path, *, static: bool, install: bool, pip_dry_run: bool,
                 dry_run: bool) -> int:
    command = [sys.executable, str(ROOT / "validate_submission.py"), str(target)]
    if static:
        command.append("--static")
    if install:
        command.append("--install")
    if pip_dry_run:
        command.append("--pip-dry-run")
    print(command_string(command))
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


def pack_submission(source: Path, destination: Path) -> None:
    if not source.is_dir():
        fail(f"submission directory がありません: {source}")
    if destination.exists():
        fail(f"zip が既に存在します（上書きしません）: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if path.is_symlink():
            fail(f"提出物に禁止/不要なファイルがあります: {rel}")
        if path.is_dir():
            continue
        if path.stat().st_nlink > 1 or path.name in {".DS_Store"} or "__pycache__" in rel.parts:
            fail(f"提出物に禁止/不要なファイルがあります: {rel}")
        if path.suffix.lower() == ".zip":
            fail(f"nested zip は禁止です: {rel}")
        files.append((path, rel.as_posix()))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, name in files:
            zf.write(path, name)
    print(f"packed: {destination} sha256={sha256_file(destination)}")


def archive_outputs(cfg: dict[str, Any]) -> None:
    output_dir = path_from_config(cfg["run"]["output_dir"])
    zip_path = path_from_config(cfg["submission"]["zip"])
    archive_dir = output_dir.parent / "archives" / cfg["run"]["tag"]
    if archive_dir.exists():
        fail(f"archive が既に存在します（上書きしません）: {archive_dir}")
    archive_dir.mkdir(parents=True)
    copied: list[dict[str, str]] = []

    def copy_file(source: Path, relative: str) -> None:
        if not source.is_file():
            return
        destination = archive_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({"path": relative, "sha256": sha256_file(destination) or ""})

    copy_file(output_dir / "resolved_config.yaml", "resolved_config.yaml")
    copy_file(output_dir / "run_manifest.json", "run_manifest.json")
    copy_file(zip_path, "submission.zip")
    eval_dir = path_from_config(cfg["eval"]["output_dir"])
    if eval_dir.is_dir():
        for result in sorted(eval_dir.glob("*.json")):
            copy_file(result, f"evaluation/{result.name}")
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_info(),
        "files": copied,
    }
    (archive_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"archived: {archive_dir}")


def run_submit_dry_run(assignment_id: str, submission_dir: Path, dry_run: bool) -> int:
    command = ["omni", "submit", "--assignment-id", assignment_id, "--path",
               str(submission_dir), "--dry-run"]
    print(command_string(command))
    if dry_run:
        return 0
    if shutil.which("omni") is None:
        print("[runner] omni コマンドはこの環境にありません。上のコマンドを Omnicampus 環境で実行してください。")
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PARC2026 train/eval/package gate runner")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pi05_lora.yaml")
    parser.add_argument("--action", choices=["dry-run", "preflight", "train", "smoke",
                                               "eval", "validate", "pack", "archive",
                                               "submit-dry-run"],
                        default="dry-run")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="許可された scalar override のみ: run.tag=smoke")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--assignment-id")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--pip-dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config.resolve(), args.overrides)

    if args.action == "dry-run":
        actual, _ = output_paths(cfg, smoke=False)
        command = ["bash", str(ROOT / cfg["train"]["entry"])]
        print("# resolved_config (secrets are never loaded from YAML)")
        print(yaml_text(redact(cfg)), end="")
        print("# planned training command")
        print(command_string(command))
        print("# planned eval command")
        print(command_string(eval_command(cfg, args.server_url)))
        print(f"# output: {actual}")
        return 0
    if args.action == "preflight":
        actual, _ = output_paths(cfg)
        info = git_info()
        print(json.dumps({"git": info, "output_absent": not actual.exists(),
                          "dataset_exists": path_from_config(cfg["dataset"]["root"]).exists(),
                          "hf_token_set": bool(os.environ.get("HF_TOKEN")),
                          "wandb_api_key_set": bool(os.environ.get("WANDB_API_KEY"))},
                         ensure_ascii=False, indent=2))
        return 0
    if args.action == "train":
        return run_training(cfg, smoke=False)
    if args.action == "smoke":
        return run_training(cfg, smoke=True)
    if args.action == "eval":
        return run_eval(cfg, args.server_url, dry_run=False)
    if args.action == "validate":
        target = args.target or path_from_config(cfg["submission"]["zip"])
        return run_validate(target, static=args.static, install=args.install,
                            pip_dry_run=args.pip_dry_run, dry_run=False)
    if args.action == "pack":
        source = args.submission_dir or path_from_config(cfg["submission"]["directory"])
        destination = path_from_config(cfg["submission"]["zip"])
        pack_submission(source, destination)
        return 0
    if args.action == "archive":
        archive_outputs(cfg)
        return 0
    if args.action == "submit-dry-run":
        if not args.assignment_id:
            fail("--assignment-id は必須です")
        source = args.submission_dir or path_from_config(cfg["submission"]["directory"])
        return run_submit_dry_run(args.assignment_id, source, dry_run=False)
    return 2


if __name__ == "__main__":
    sys.exit(main())
