"""Run a frozen manifest through one checkpoint-specific policy session."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .common import ROOT, git_info, load_yaml, resolve_path, sha256_file, write_json
from .eval_manifest import _state_hash, load_locked_panel
from .eval_result import canonical_episode, load_episode_rows, write_result_bundle


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _checkpoint_manifest(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    for name in ("checkpoint_manifest_v2.json", "checkpoint_manifest.json"):
        path = checkpoint / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"checkpoint manifest is not an object: {path}")
            return path, value
    raise FileNotFoundError(f"checkpoint manifest missing: {checkpoint}")


def checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    axis_manifest_path = checkpoint / "axis_base_manifest.json"
    if axis_manifest_path.is_file():
        manifest = json.loads(axis_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_kind") != "axis_base_params":
            raise ValueError(f"unsupported non-Orbax checkpoint kind: {checkpoint}")
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(f"Axis base params tree missing: {checkpoint}")
        stable_hash = manifest.get("stable_checkpoint_tree_sha256")
        if not stable_hash:
            raise ValueError(f"Axis base manifest has no stable params identity: {checkpoint}")
        return {
            "path": str(checkpoint),
            "internal_step": 0,
            "checkpoint_sha256": str(stable_hash),
            "norm_sha256": manifest.get("norm_sha256"),
            "manifest_sha256": sha256_file(axis_manifest_path),
            "manifest": manifest,
        }
    if not checkpoint.is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").exists():
        raise FileNotFoundError(f"not an Orbax checkpoint directory: {checkpoint}")
    try:
        internal_step = int(checkpoint.name)
    except ValueError as exc:
        raise ValueError(f"checkpoint directory must use internal step: {checkpoint}") from exc
    manifest_path, manifest = _checkpoint_manifest(checkpoint)
    if int(manifest.get("internal_step", internal_step)) != internal_step:
        raise ValueError(f"checkpoint manifest step mismatch: {checkpoint}")
    restore = checkpoint / "restore_validation.json"
    if restore.is_file():
        restore_value = json.loads(restore.read_text(encoding="utf-8"))
        if restore_value.get("status") != "passed":
            raise ValueError(f"checkpoint restore validation did not pass: {restore}")
    if not (checkpoint / "params").is_dir():
        raise ValueError(f"checkpoint has no params tree: {checkpoint}")
    stable_hash = manifest.get("stable_checkpoint_tree_sha256") or manifest.get("checkpoint_tree_sha256")
    if not stable_hash:
        raise ValueError(f"checkpoint manifest has no parameter tree identity: {checkpoint}")
    return {
        "path": str(checkpoint),
        "internal_step": internal_step,
        "checkpoint_sha256": str(stable_hash),
        "norm_sha256": manifest.get("norm_sha256"),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
    }


def _candidate_scope_mix(candidate: str) -> tuple[str, str]:
    match = re.search(r"f([012])[_/-]?m([012])", candidate.lower())
    if not match:
        raise ValueError(f"candidate id must encode scope/mix, e.g. f0_m1: {candidate}")
    return f"F{match.group(1)}", f"M{match.group(2)}"


def protocol_payload(manifest_hash: str, *, order_seed: int = 20260827, bootstrap_seed: int = 1000, bootstrap_draws: int = 10_000) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "order_seed": int(order_seed),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_draws": int(bootstrap_draws),
        "reset_before_every_case": True,
        "environment_and_policy_seeds_separate": True,
        "same_case_order_for_all_candidates": True,
        "official_result": False,
    }


def ensure_protocol(path: Path, manifest_hash: str) -> dict[str, Any]:
    expected = protocol_payload(manifest_hash)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"protocol lock mismatch; create a new protocol version: {path}")
        return actual
    write_json(path, expected)
    return expected


class ServerSession:
    def __init__(self, *, root: Path, checkpoint: Path, output_dir: Path, gpu_id: str, config_name: str, scope: str, mix: str, server_python: str | None = None):
        self.root = root
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.gpu_id = gpu_id
        self.config_name = config_name
        self.scope = scope
        self.mix = mix
        self.server_python = server_python or os.environ.get("AXIS_PI05_SERVER_PYTHON", sys.executable)
        self.port = _free_port()
        self.process: subprocess.Popen[str] | None = None
        self.log_path = output_dir / "policy_server.log"

    def __enter__(self):
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": self.gpu_id,
            "HF_HUB_OFFLINE": "1",
            "AXIS_PI05_SCOPE": self.scope,
            "AXIS_PI05_MIX": self.mix,
            "AXIS_PI05_ASSETS_ROOT": str(self.root / "data/manifests_v2"),
            "AXIS_PI05_NORM_STATS": str(self.root / "data/manifests_v2" / self.mix / "norm_stats.json"),
        })
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                self.server_python,
                "-m",
                "tools.axis_pi05.checkpoint_http_server",
                "--checkpoint",
                str(self.checkpoint),
                "--port",
                str(self.port),
                "--config",
                self.config_name,
            ],
            cwd=self.root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._log_handle = log
        from pipeline.remote_policy import RemotePolicyClient

        self.client = RemotePolicyClient(f"http://127.0.0.1:{self.port}", timeout_sec=30.0)
        try:
            # Large Orbax checkpoints can require several minutes to restore
            # from a shared filesystem.  The old fixed 180 s limit turned a
            # slow but valid startup into an infrastructure failure.  Keep a
            # bounded default while allowing the runner to record/override
            # the startup budget per hardware and checkpoint.
            startup_timeout = float(os.environ.get("AXIS_PI05_SERVER_STARTUP_TIMEOUT", "600"))
            self.client.wait_for_server(timeout=startup_timeout)
        except Exception:
            self.close()
            raise
        return self.client

    def close(self) -> dict[str, Any]:
        process = self.process
        if process is None:
            return {"pid": None, "returncode": None, "clean_shutdown": True}
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if hasattr(self, "_log_handle"):
            self._log_handle.close()
        return {"pid": process.pid, "returncode": process.returncode, "clean_shutdown": process.poll() is not None}

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _failure_class(exc: Exception) -> str:
    text = str(exc).lower()
    if any(token in text for token in ("connection", "server", "environment", "bddl", "init_state", "mujoco")):
        return "infrastructure"
    return "model_error"


def _trajectory_path(output_dir: Path, case_id: str, episode: Any) -> str | None:
    path = output_dir / "trajectories" / (case_id.replace(":", "_") + ".npz")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            actions=np.asarray(episode.actions, dtype=np.float32),
            rewards=np.asarray(episode.rewards, dtype=np.float32),
            ee_positions=np.asarray(episode.ee_positions, dtype=np.float32),
            joint_positions=np.asarray(episode.joint_positions, dtype=np.float32),
            gripper_qpos=np.asarray(episode.gripper_qpos, dtype=np.float32),
        )
        return path.relative_to(output_dir).as_posix()
    except Exception:
        return None


def _episode_row(manifest_row: dict[str, Any], *, candidate: str, identity: dict[str, Any], output_dir: Path, episode: Any) -> dict[str, Any]:
    video_path = None
    if episode.video_path:
        try:
            video_path = Path(episode.video_path).resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            video_path = str(episode.video_path)
    return canonical_episode({
        **manifest_row,
        "candidate": candidate,
        "checkpoint": identity["path"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "manifest_sha256": manifest_row.get("manifest_sha256"),
        "protocol_sha256": manifest_row.get("protocol_sha256"),
        "instruction": manifest_row.get("language"),
        "success": episode.success,
        "collision": episode.collided,
        "goal_reached": episode.goal_reached,
        "goal_reached_but_collision": episode.goal_reached and episode.collided,
        "timeout": episode.timeout,
        "steps": episode.total_steps,
        "elapsed_sec": episode.elapsed_time_sec,
        "failure_reason": episode.failure_reason,
        "failure_class": episode.failure_class,
        "video_path": video_path,
        "trajectory_path": _trajectory_path(output_dir, manifest_row["case_id"], episode),
        "error_summary": episode.error_summary,
    })


def run_cases(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    identity: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    policy: Any,
    protocol_sha256: str | None = None,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.rollout import RolloutExecutor

    config = EvalConfig(max_steps_per_episode=300, seed=1000)
    manager = EnvironmentManager(config)
    executor = RolloutExecutor(manager, config)
    task_cache: dict[str, dict[int, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    for manifest_row in rows:
        benchmark = str(manifest_row["benchmark"])
        if benchmark not in task_cache:
            task_cache[benchmark] = {task.task_id: task for task in manager.get_task_infos(benchmark)}
        task_info = task_cache[benchmark].get(int(manifest_row["task_id"]))
        if task_info is None or str(task_info.name) != str(manifest_row["task_name"]):
            raise RuntimeError(f"manifest task no longer matches registry: {manifest_row['case_id']}")
        init_state = manager.get_init_state(task_info, int(manifest_row["init_state_index"]))
        if _state_hash(init_state) != manifest_row["init_state_sha256"]:
            raise RuntimeError(f"init state hash mismatch: {manifest_row['case_id']}")
        if sha256_file(Path(task_info.bddl_file)) != manifest_row["bddl_sha256"]:
            raise RuntimeError(f"BDDL hash mismatch: {manifest_row['case_id']}")
        perturbation = PerturbationConfig(**{
            key: value for key, value in manifest_row["perturbation_params"].items()
            if key in PerturbationConfig.__dataclass_fields__
        })
        row = dict(manifest_row)
        row["manifest_sha256"] = protocol["manifest_sha256"]
        row["protocol_sha256"] = protocol_sha256
        try:
            episode = executor.evaluate_case(
                policy,
                task_info,
                init_state=init_state,
                env_seed=int(manifest_row["env_seed"]),
                policy_seed=int(manifest_row["policy_seed"]),
                max_steps=int(manifest_row["max_steps"]),
                case_id=str(manifest_row["case_id"]),
                perturbation=perturbation,
            )
            result_rows.append(_episode_row(row, candidate=candidate, identity=identity, output_dir=output_dir, episode=episode))
        except Exception as exc:
            result_rows.append(canonical_episode({
                **row,
                "candidate": candidate,
                "checkpoint": identity["path"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "success": False,
                "failure_reason": "exception",
                "failure_class": _failure_class(exc),
                "steps": 0,
                "elapsed_sec": 0.0,
                "error_summary": f"{type(exc).__name__}: {exc}",
            }))
    return result_rows


def run_panel(
    *,
    manifest: Path,
    checkpoint: Path,
    candidate: str,
    output_dir: Path,
    protocol_path: Path,
    root: Path = ROOT,
    gpu_id: str = "1",
    config_name: str = "axis_pi05_libero_mix",
    server_url: str | None = None,
    server_python: str | None = None,
    baseline_result_dir: Path | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> dict[str, Any]:
    # The repository ships the exact LIBERO config used to build its locked
    # manifests.  Prefer it when the caller has not explicitly selected a
    # config; relying on a user's global ~/.libero can silently resolve a
    # different install and make valid T2/T3 cases unavailable.
    if "LIBERO_CONFIG_PATH" not in os.environ:
        bundled_libero_config = root / ".eval_libero"
        if bundled_libero_config.is_dir():
            os.environ["LIBERO_CONFIG_PATH"] = str(bundled_libero_config)
    rows = load_locked_panel(manifest, manifest.parent / "manifest.lock.json")
    start = 0 if row_start is None else int(row_start)
    end = len(rows) if row_end is None else int(row_end)
    if start < 0 or end < start or end > len(rows):
        raise ValueError(f"invalid row range [{start}, {end}) for {len(rows)} manifest rows")
    rows = rows[start:end]
    manifest_hash = sha256_file(manifest)
    protocol = ensure_protocol(protocol_path, manifest_hash)
    identity = checkpoint_identity(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha256_file(protocol_path)
    baseline_episodes = None
    if baseline_result_dir is not None:
        baseline_episodes = load_episode_rows(baseline_result_dir / "episodes.jsonl")
    write_json(output_dir / "run_metadata.json", {
        "schema_version": 2,
        "candidate": candidate,
        "checkpoint": identity,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "git": git_info(root),
        "gpu_id": gpu_id,
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "server_url": server_url,
        "started_at_unix": time.time(),
        "official_result": False,
        "result_scope": "local_model_selection",
        "row_start": start,
        "row_end": end,
        "expected_case_count": len(rows),
    })
    scope, mix = _candidate_scope_mix(candidate)
    old_video_dir = os.environ.get("T3_VIDEO_DIR")
    os.environ["T3_VIDEO_DIR"] = str(output_dir / "videos")
    try:
        if server_url:
            from pipeline.remote_policy import RemotePolicyClient

            policy = RemotePolicyClient(server_url, timeout_sec=30.0)
            policy.wait_for_server(timeout=30.0)
            server_meta = {"external": True, "url": server_url}
            episodes = run_cases(rows, candidate=candidate, identity=identity, protocol=protocol, protocol_sha256=protocol_hash, output_dir=output_dir, policy=policy, root=root)
        else:
            with ServerSession(root=root, checkpoint=checkpoint, output_dir=output_dir, gpu_id=gpu_id, config_name=config_name, scope=scope, mix=mix, server_python=server_python) as policy:
                server_meta = {"external": False, "port": policy.server_url.rsplit(":", 1)[-1]}
                episodes = run_cases(rows, candidate=candidate, identity=identity, protocol=protocol, protocol_sha256=protocol_hash, output_dir=output_dir, policy=policy, root=root)
        result = write_result_bundle(
            output_dir,
            episodes,
            expected_case_ids=[row["case_id"] for row in rows],
            summary_kwargs={
                "candidate": candidate,
                "panel": manifest.stem,
                "checkpoint": identity["path"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "manifest_sha256": manifest_hash,
                "protocol_sha256": protocol_hash,
                "official_result": False,
                "result_scope": "local_model_selection",
                "bootstrap_seed": protocol["bootstrap_seed"],
                "bootstrap_draws": protocol["bootstrap_draws"],
                "baseline_episodes": baseline_episodes,
            },
            run_metadata={
                "schema_version": 2,
                "candidate": candidate,
                "panel": manifest.stem,
                "row_start": start,
                "row_end": end,
                "expected_case_count": len(rows),
                "server": server_meta,
                "checkpoint": identity,
                "manifest_sha256": manifest_hash,
                "protocol_sha256": protocol_hash,
                "official_result": False,
                "result_scope": "local_model_selection",
            },
        )
    finally:
        if old_video_dir is None:
            os.environ.pop("T3_VIDEO_DIR", None)
        else:
            os.environ["T3_VIDEO_DIR"] = old_video_dir
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "reviews/axis_pi05/model_selection_v1/protocol.lock.json")
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--config", default="axis_pi05_libero_mix")
    parser.add_argument("--server-url")
    parser.add_argument("--server-python")
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--row-start", type=int)
    parser.add_argument("--row-end", type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run_panel(
        manifest=resolve_path(args.manifest),
        checkpoint=resolve_path(args.checkpoint),
        candidate=args.candidate,
        output_dir=resolve_path(args.out),
        protocol_path=resolve_path(args.protocol),
        root=resolve_path(args.root),
        gpu_id=args.gpu_id,
        config_name=args.config,
        server_url=args.server_url,
        server_python=args.server_python,
        baseline_result_dir=resolve_path(args.baseline_result) if args.baseline_result else None,
        row_start=args.row_start,
        row_end=args.row_end,
    )
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2))
    if not result["coverage"]["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
