"""Build and run frozen canonical C20/B80 rollout panels.

The simulator side of this command runs in the LIBERO evaluation environment;
the policy side is a separate OpenPI process pinned to one of GPU1--3. The
server loads Orbax directly and never invokes FAST/SLOW conversion.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import html
import io
import json
import os
import re
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, canonical_json, git_info, resolve_path, sha256_file, write_json, write_jsonl


PANEL_ROOT = ROOT / "data" / "reports" / "axis_libero_30k" / "phase1" / "rollout"
CHECKPOINT = ROOT / "artifacts" / "axis_pi05_v2" / "model_selection_v1" / "axis_init_f0_m1"
PHASE1_ROOT = ROOT / "data" / "reports" / "axis_libero_30k" / "phase1"
SIM_ROOT = ROOT / "LIBERO-plus"
OFFICIAL_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def is_stall_episode(row: dict[str, Any]) -> bool:
    """Use one stall definition for run summaries, merges, and comparisons."""

    success = row["success"]
    if isinstance(success, str):
        success = success.strip().lower() in {"1", "true", "yes", "y"}
    return (not bool(success)) and float(row["action_norm_mean"]) < 1e-3


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _state_hash(value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(canonical_json({
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_sha256": _sha256_array(array),
    })).hexdigest()


def _find_base_bddl(task: str) -> tuple[Path, Path]:
    """Resolve the exact base LIBERO-mix BDDL/init pair for one language."""
    safe = task.replace(" ", "_")
    # The checkout has the Python package under LIBERO-plus/libero/libero.
    # Keep SIM_ROOT at the checkout root so the simulator subprocess and
    # generated paths use the same anchor.
    bddl_root = SIM_ROOT / "libero" / "libero" / "bddl_files" / "libero_mix"
    init_root = SIM_ROOT / "libero" / "libero" / "init_files" / "libero_mix"
    direct = bddl_root / f"{safe}.bddl"
    if direct.is_file():
        bddl = direct
    else:
        candidates = sorted(bddl_root.glob(f"*_{safe}*.bddl"))
        if not candidates:
            raise FileNotFoundError(f"no base libero_mix BDDL for official task: {task}")
        bddl = next((p for p in candidates if "_language_" not in p.name and "_add_" not in p.name and "_tb_" not in p.name and "copy" not in p.name), candidates[0])
    stem = re.sub(r" copy(?: \d+)?$", "", bddl.stem)
    stem = re.sub(r"_(?:language|add|tb|view)_\d+$", "", stem)
    # Scene prefixes are present in the first ten base init files; the mix
    # init tree follows the same stem. For non-scene tasks it is just the task.
    init = init_root / f"{stem}.pruned_init"
    if not init.is_file():
        init = init_root / f"{safe}.pruned_init"
    if not init.is_file():
        raise FileNotFoundError(f"missing base init state for {bddl}")
    return bddl, init


def build_panels(out_root: Path = PANEL_ROOT) -> dict[str, Any]:
    import torch

    official = [json.loads(line)["task"] for line in (ROOT / "data/raw/hf/physical-intelligence/libero/meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(official) != 40:
        raise RuntimeError(f"official task catalog must contain 40 tasks, got {len(official)}")
    task_rows: list[dict[str, Any]] = []
    for task_index, language in enumerate(official):
        bddl, init_path = _find_base_bddl(language)
        init_states = torch.load(init_path, weights_only=False)
        init_count = int(len(init_states))
        if init_count < 2:
            raise RuntimeError(f"task has fewer than two fixed init states: {language} ({init_count})")
        task_rows.append({
            "task_index": task_index,
            "task_name": bddl.stem,
            "language": language,
            "bddl_path": str(bddl),
            "bddl_sha256": sha256_file(bddl),
            "init_states_path": str(init_path),
            "init_state_count": init_count,
        })
    manifests: dict[str, list[dict[str, Any]]] = {"C20": [], "B80": []}
    case_number = 0
    for task in task_rows[:20]:
        init_path = Path(task["init_states_path"])
        init_states = torch.load(init_path, weights_only=False)
        state = np.asarray(init_states[0])
        manifests["C20"].append({
            "schema_version": 1,
            "panel": "C20",
            "case_id": f"C20-task{task['task_index']:02d}-init0",
            "official_task_index": task["task_index"],
            "task_name": task["task_name"],
            "language": task["language"],
            "bddl_path": task["bddl_path"],
            "bddl_sha256": task["bddl_sha256"],
            "init_states_path": task["init_states_path"],
            "init_state_index": 0,
            "init_state_sha256": _state_hash(state),
            "env_seed": 51000 + case_number,
            "policy_seed": 52000 + case_number,
            "max_steps": 300,
        })
        case_number += 1
    case_number = 0
    for task in task_rows:
        init_path = Path(task["init_states_path"])
        init_states = torch.load(init_path, weights_only=False)
        for init_index in (0, 1):
            state = np.asarray(init_states[init_index])
            manifests["B80"].append({
                "schema_version": 1,
                "panel": "B80",
                "case_id": f"B80-task{task['task_index']:02d}-init{init_index}",
                "official_task_index": task["task_index"],
                "task_name": task["task_name"],
                "language": task["language"],
                "bddl_path": task["bddl_path"],
                "bddl_sha256": task["bddl_sha256"],
                "init_states_path": task["init_states_path"],
                "init_state_index": init_index,
                "init_state_sha256": _state_hash(state),
                "env_seed": 53000 + case_number,
                "policy_seed": 54000 + case_number,
                "max_steps": 300,
            })
            case_number += 1
    out_root.mkdir(parents=True, exist_ok=True)
    panel_hashes = {}
    for panel, rows in manifests.items():
        path = out_root / f"{panel}.jsonl"
        write_jsonl(path, rows)
        panel_hashes[panel] = sha256_file(path)
    lock = {
        "schema_version": 1,
        "panel_type": "canonical_step0_rollout",
        "source_repo": "physical-intelligence/libero",
        "source_revision": OFFICIAL_REVISION,
        "source_fps": 10,
        "task_count": 40,
        "panels": {"C20": {"cases": 20, "semantics": "20 official tasks x init state 0"}, "B80": {"cases": 80, "semantics": "40 official tasks x init states 0,1"}},
        "panel_sha256": panel_hashes,
        "checkpoint": str(CHECKPOINT),
        "protocols": {"execute8": "chunk=10, execute first 8 then discard remaining 2", "full_chunk": "chunk=10, execute all 10"},
        "direct_orbax": True,
        "fast_slow_conversion": False,
    }
    write_json(out_root / "manifest.lock.json", lock)
    write_json(out_root / "task_catalog.json", {"schema_version": 1, "tasks": task_rows})
    return lock


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _checkpoint_provenance(
    checkpoint: Path,
    *,
    config_name: str,
    arm: str | None,
    checkpoint_step: int | None,
    norm_stats_path: Path,
) -> dict[str, Any]:
    """Build a stable identity for the exact policy artifact used by a shard."""

    step = checkpoint_step
    if step is None and checkpoint.name.isdigit():
        step = int(checkpoint.name)
    axis_base_manifest = checkpoint / "axis_base_manifest.json"
    if axis_base_manifest.is_file():
        manifest_sha256 = sha256_file(axis_base_manifest)
        identity = ":".join(
            (
                config_name,
                arm or "dense_base",
                f"step-{step if step is not None else 0}",
                manifest_sha256,
            )
        )
        return {
            "checkpoint_identity": identity,
            "checkpoint_path": str(checkpoint),
            "checkpoint_step": step if step is not None else 0,
            "checkpoint_params_manifest_sha256": manifest_sha256,
            "checkpoint_metadata_sha256": manifest_sha256,
            "checkpoint_kind": "axis_base_params",
            "lora_arm": arm,
            "normalization_mode": "QUANTILES",
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_sha256": sha256_file(norm_stats_path),
            "policy_seed_source": "frozen manifest policy_seed -> reset(seed) per episode",
        }
    params_manifest = checkpoint / "params" / "manifest.ocdbt"
    checkpoint_metadata = checkpoint / "_CHECKPOINT_METADATA"
    if not params_manifest.is_file() or not checkpoint_metadata.is_file():
        raise FileNotFoundError(
            f"checkpoint provenance requires Orbax metadata and params manifest: {checkpoint}"
        )
    params_manifest_sha256 = sha256_file(params_manifest)
    metadata_sha256 = sha256_file(checkpoint_metadata)
    identity = ":".join(
        (
            config_name,
            arm or "unspecified_arm",
            f"step-{step if step is not None else 'unknown'}",
            params_manifest_sha256,
            metadata_sha256,
        )
    )
    return {
        "checkpoint_identity": identity,
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": step,
        "checkpoint_params_manifest_sha256": params_manifest_sha256,
        "checkpoint_metadata_sha256": metadata_sha256,
        "checkpoint_kind": "raw_lora_orbax",
        "lora_arm": arm,
        "normalization_mode": "QUANTILES",
        "norm_stats_path": str(norm_stats_path),
        "norm_stats_sha256": sha256_file(norm_stats_path),
        "policy_seed_source": "frozen manifest policy_seed -> reset(seed) per episode",
    }


class ChunkPolicy:
    def __init__(self, client: Any, *, mode: str, instruction: str = ""):
        if mode not in {"execute8", "full_chunk"}:
            raise ValueError(mode)
        self.client = client
        self.mode = mode
        self.instruction = instruction
        self.needs_chunk_reset = False
        self.actions_seen = 0
        self.chunk_index = 0

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        self.instruction = str(instruction)
        self.needs_chunk_reset = False
        self.actions_seen = 0
        self.chunk_index = 0
        self.client.reset(
            instruction=self.instruction,
            seed=seed,
            reset_kind="episode_reset",
            chunk_index=self.chunk_index,
        )

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self.needs_chunk_reset:
            self.chunk_index += 1
            self.client.reset(
                instruction=self.instruction,
                reset_kind="chunk_discard",
                chunk_index=self.chunk_index,
            )
            self.needs_chunk_reset = False
            self.actions_seen = 0
        action = np.asarray(self.client.get_action(obs), dtype=np.float32)
        self.actions_seen += 1
        if self.mode == "execute8" and self.actions_seen >= 8:
            self.needs_chunk_reset = True
        elif self.mode == "full_chunk" and self.actions_seen >= 10:
            self.actions_seen = 0
        return action


def _load_task_info(row: dict[str, Any]) -> Any:
    import torch
    from pipeline.environment import TaskInfo

    init_states = torch.load(row["init_states_path"], weights_only=False)
    if _state_hash(np.asarray(init_states[int(row["init_state_index"])])) != row["init_state_sha256"]:
        raise RuntimeError(f"init state changed after manifest build: {row['case_id']}")
    if sha256_file(Path(row["bddl_path"])) != row["bddl_sha256"]:
        raise RuntimeError(f"BDDL changed after manifest build: {row['case_id']}")
    return TaskInfo(
        task_id=int(row["official_task_index"]),
        name=str(row["task_name"]),
        language=str(row["language"]),
        bddl_file=str(row["bddl_path"]),
        init_states=init_states,
        benchmark_name="libero_mix",
    )


def _write_result_tables(run_root: Path, episode_rows: list[dict[str, Any]]) -> None:
    """Write the stable per-episode and per-task tables for one run."""
    fieldnames = sorted({key for row in episode_rows for key in row})
    with (run_root / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episode_rows)

    tasks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        tasks[int(row["official_task_index"])].append(row)
    task_rows = []
    for task_index, task_episodes in sorted(tasks.items()):
        task_rows.append({
            "official_task_index": task_index,
            "task_name": task_episodes[0]["task_name"],
            "instruction": task_episodes[0]["language"],
            "cases": len(task_episodes),
            "successes": sum(bool(row["success"]) for row in task_episodes),
            "success_rate": sum(bool(row["success"]) for row in task_episodes) / len(task_episodes),
            "collision_count": sum(bool(row["collision"]) for row in task_episodes),
            "timeout_count": sum(bool(row["timeout"]) for row in task_episodes),
            "stall_count": sum(is_stall_episode(row) for row in task_episodes),
        })
    task_fields = sorted({key for row in task_rows for key in row})
    with (run_root / "tasks.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(task_rows)


def _write_action_trace(run_root: Path, case_id: str, actions: Any) -> dict[str, Any]:
    """Persist an exact float32 action trace and its content metadata."""

    trace_dir = run_root / "action_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"action trace must have shape [N,7], got {array.shape}")
    path = trace_dir / f"{case_id}.npy"
    np.save(path, array, allow_pickle=False)
    return {
        "action_trace_path": str(path),
        "action_trace_sha256": sha256_file(path),
        "action_shape": list(array.shape),
        "action_finite": bool(np.isfinite(array).all()),
    }


def _git_provenance() -> dict[str, Any]:
    return {
        "root": git_info(ROOT),
        "nested_openpi": git_info(ROOT / "third_party" / "axis_openpi"),
    }


def _write_failure_gallery(run_root: Path, episode_rows: list[dict[str, Any]]) -> Path:
    """Write a self-contained review index for failed H.264 episodes."""
    gallery_root = run_root / "failure_gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in episode_rows:
        if bool(row.get("success")):
            continue
        video = Path(str(row.get("video_path") or ""))
        if not video.is_absolute():
            video = run_root / video
        if not video.is_file():
            continue
        source = html.escape(os.path.relpath(video, gallery_root))
        metadata = html.escape(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))
        title = html.escape(str(row.get("case_id", "unknown")))
        cards.append(
            f'<article><h3>{title}</h3><video controls preload="metadata" src="{source}"></video>'
            f'<details><summary>episode metadata</summary><pre>{metadata}</pre></details></article>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>Axis LoRA failure gallery</title>'
        '<style>body{font:14px sans-serif;margin:2rem;background:#f5f5f5}'
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}'
        'article{background:#fff;padding:1rem;border-radius:8px}video{width:100%}'
        'pre{white-space:pre-wrap}</style><main>'
        + "\n".join(cards)
        + '</main>'
    )
    index = gallery_root / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def run_panel(
    panel: str,
    mode: str,
    gpu_id: str,
    *,
    out_root: Path = PANEL_ROOT,
    server_python: str | None = None,
    start_index: int = 0,
    end_index: int | None = None,
    run_id: str | None = None,
    manifest_root: Path | None = None,
    checkpoint: Path = CHECKPOINT,
    config_name: str = "axis_pi05_libero_phase1",
    norm_stats_path: Path = PHASE1_ROOT / "norm_stats.json",
    arm: str | None = None,
    checkpoint_step: int | None = None,
) -> dict[str, Any]:
    from pipeline.config import EvalConfig, PerturbationConfig
    from pipeline.environment import EnvironmentManager
    from pipeline.remote_policy import RemotePolicyClient
    from pipeline.rollout import RolloutExecutor

    if panel not in {"C20", "B80"} or mode not in {"execute8", "full_chunk"}:
        raise ValueError(f"invalid panel/mode: {panel}/{mode}")
    manifest_root = manifest_root or out_root
    manifest_path = manifest_root / f"{panel}.jsonl"
    checkpoint = resolve_path(checkpoint)
    norm_stats_path = resolve_path(norm_stats_path)
    all_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if start_index < 0 or start_index > len(all_rows):
        raise ValueError(f"invalid start index: {start_index}")
    if end_index is None:
        end_index = len(all_rows)
    if end_index < start_index or end_index > len(all_rows):
        raise ValueError(f"invalid end index: {end_index}")
    rows = all_rows[start_index:end_index]
    run_root = out_root / (run_id or f"{panel}_{mode}")
    run_root.mkdir(parents=True, exist_ok=True)
    video_dir = run_root / "videos"
    os.environ["T3_VIDEO_DIR"] = str(video_dir)
    os.environ["T3_VIDEO_FPS"] = "10"
    os.environ["LIBERO_CONTROL_FREQ"] = "10"
    if "LIBERO_CONFIG_PATH" not in os.environ and (ROOT / ".eval_libero").is_dir():
        os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / ".eval_libero")
    server_python = server_python or str(ROOT / ".venv_axis_pi05" / "bin" / "python")
    port = _free_port()
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "HF_HUB_OFFLINE": "1",
        "AXIS_PI05_NORM_STATS": str(norm_stats_path),
        "AXIS_PI05_PHASE1_ASSETS": str(PHASE1_ROOT),
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".90",
    })
    if arm is not None:
        env.update({
            "AXIS_PI05_LORA_ARM": arm,
            "AXIS_PI05_LORA_RUN_DIR": str(checkpoint.parent),
            "AXIS_PI05_LORA_NORM_STATS": str(norm_stats_path),
            "AXIS_PI05_LORA_DATA_ROOT": str(ROOT / "data/raw/hf/physical-intelligence/libero"),
            "AXIS_PI05_LORA_SAMPLING_MANIFEST": str(PHASE1_ROOT / "official_sampling_manifest.jsonl"),
            "AXIS_PI05_LORA_AXIS_PARAMS": str(ROOT / "data/models/axisrobotics/Pi0.5-Axis" / "c342608c489515efa9ba8281400dc648bf3970e1" / "params"),
        })
    checkpoint_provenance = _checkpoint_provenance(
        checkpoint,
        config_name=config_name,
        arm=arm,
        checkpoint_step=checkpoint_step,
        norm_stats_path=norm_stats_path,
    )
    log_path = run_root / "policy_server.log"
    log = log_path.open("w", encoding="utf-8")
    partial_path = run_root / "episodes.partial.jsonl"
    partial_path.write_text("", encoding="utf-8")
    process = subprocess.Popen(
        [server_python, "-m", "tools.axis_pi05.checkpoint_http_server", "--checkpoint", str(checkpoint), "--port", str(port), "--config", config_name],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    client = RemotePolicyClient(f"http://127.0.0.1:{port}", timeout_sec=90.0)
    manager = None
    started = time.time()
    try:
        client.wait_for_server(timeout=900.0)
        manager = EnvironmentManager(EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=300, seed=51000))
        executor = RolloutExecutor(manager, EvalConfig(camera_height=128, camera_width=128, max_steps_per_episode=300, seed=51000))
        policy = ChunkPolicy(client, mode=mode)
        episode_rows: list[dict[str, Any]] = []
        for number, row in enumerate(rows):
            task_info = _load_task_info(row)
            init_states = task_info.init_states
            init_state = np.asarray(init_states[int(row["init_state_index"])])
            try:
                episode = executor.evaluate_case(
                    policy,
                    task_info,
                    init_state=init_state,
                    env_seed=int(row["env_seed"]),
                    policy_seed=int(row["policy_seed"]),
                    max_steps=int(row["max_steps"]),
                    case_id=row["case_id"],
                    perturbation=PerturbationConfig(),
                )
                actions = np.asarray(episode.actions, dtype=np.float32)
                action_norm = float(np.linalg.norm(actions, axis=1).mean()) if actions.size else 0.0
                episode_rows.append({
                    **row,
                    "success": bool(episode.success),
                    "collision": bool(episode.collided),
                    "goal_reached": bool(episode.goal_reached),
                    "timeout": bool(episode.timeout),
                    "failure_reason": episode.failure_reason,
                    "steps": int(episode.total_steps),
                    "elapsed_sec": float(episode.elapsed_time_sec),
                    "action_norm_mean": action_norm,
                    "action_norm_max": float(np.linalg.norm(actions, axis=1).max()) if actions.size else 0.0,
                    "video_path": episode.video_path,
                })
                episode_rows[-1].update(_write_action_trace(run_root, row["case_id"], actions))
            except Exception as exc:
                episode_rows.append({**row, "success": False, "collision": False, "goal_reached": False, "timeout": False, "failure_reason": "exception", "steps": 0, "elapsed_sec": 0.0, "action_norm_mean": 0.0, "action_norm_max": 0.0, "video_path": None, "error_summary": f"{type(exc).__name__}: {exc}"})
                episode_rows[-1].update(_write_action_trace(run_root, row["case_id"], np.empty((0, 7), dtype=np.float32)))
            with partial_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode_rows[-1], ensure_ascii=False, sort_keys=True) + "\n")
            print(f"{panel}/{mode} {number + 1}/{len(rows)} {row['case_id']}", flush=True)
        success_count = sum(bool(row["success"]) for row in episode_rows)
        stall_count = sum(is_stall_episode(row) for row in episode_rows)
        _write_result_tables(run_root, episode_rows)
        failure_gallery = _write_failure_gallery(run_root, episode_rows)
        videos = sorted(str(path.relative_to(run_root)) for path in video_dir.glob("*.mp4")) if video_dir.is_dir() else []
        summary = {
            "schema_version": 1,
            # A complete artifact set is not a usable baseline if every case
            # fails.  In particular, 0/80 must stop the phase rather than be
            # explained away as insufficient training.
            "status": "passed" if len(episode_rows) == len(rows) and success_count > 0 and stall_count <= 0.9 * len(rows) else "failed",
            "panel": panel,
            "execution_protocol": mode,
            "chunk_size": 10,
            "executed_actions_per_chunk": 8 if mode == "execute8" else 10,
            "direct_orbax": True,
            "fast_slow_conversion": False,
            "source_repo": "physical-intelligence/libero",
            "source_revision": OFFICIAL_REVISION,
            "source_fps": 10,
            "git": _git_provenance(),
            "norm_stats_sha256": checkpoint_provenance["norm_stats_sha256"],
            "panel_manifest_sha256": sha256_file(manifest_path),
            **checkpoint_provenance,
            "config_name": config_name,
            "case_count": len(episode_rows),
            "case_range": [start_index, end_index],
            "success_count": success_count,
            "success_rate": success_count / len(episode_rows) if episode_rows else 0.0,
            "collision_count": sum(bool(row["collision"]) for row in episode_rows),
            "timeout_count": sum(bool(row["timeout"]) for row in episode_rows),
            "stall_count": stall_count,
            "stall_rate": stall_count / len(episode_rows) if episode_rows else 1.0,
            "mean_action_norm": float(np.mean([row["action_norm_mean"] for row in episode_rows])) if episode_rows else 0.0,
            "max_action_norm": float(np.max([row["action_norm_max"] for row in episode_rows])) if episode_rows else 0.0,
            "video_count": len(videos),
            "videos": videos,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "gpu_id": str(gpu_id),
            "gpu0_touched": False,
            "artifacts": {"episodes_csv": str(run_root / "episodes.csv"), "tasks_csv": str(run_root / "tasks.csv"), "videos_dir": str(video_dir), "failure_gallery": str(failure_gallery), "action_traces_dir": str(run_root / "action_traces")},
        }
        write_json(run_root / "summary.json", summary)
        return summary
    finally:
        client_process = process
        if client_process.poll() is None:
            client_process.terminate()
            try:
                client_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                client_process.kill()
                client_process.wait(timeout=10)
        log.close()


def merge_panel_runs(
    panel: str,
    mode: str,
    shard_roots: list[Path],
    *,
    out_root: Path = PANEL_ROOT,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Merge complete independent shards without deleting interrupted output."""
    if panel not in {"C20", "B80"} or mode not in {"execute8", "full_chunk"}:
        raise ValueError(f"invalid panel/mode: {panel}/{mode}")
    manifest_path = out_root / f"{panel}.jsonl"
    manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest_order = {row["case_id"]: index for index, row in enumerate(manifest_rows)}
    episode_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    for shard_root in shard_roots:
        path = shard_root / "episodes.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary_path = shard_root / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        shard_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(shard_summary, dict):
            raise ValueError(f"shard summary must be an object: {summary_path}")
        shard_summaries.append(shard_summary)
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                case_id = row["case_id"]
                if case_id in seen:
                    raise ValueError(f"duplicate case in shards: {case_id}")
                if case_id not in manifest_order:
                    raise ValueError(f"case not in frozen manifest: {case_id}")
                seen.add(case_id)
                for key in ("success", "collision", "goal_reached", "timeout"):
                    row[key] = row.get(key, "False").lower() == "true"
                for key in ("official_task_index", "init_state_index", "env_seed", "policy_seed", "max_steps", "steps"):
                    if key in row and row[key] != "":
                        row[key] = int(row[key])
                for key in ("elapsed_sec", "action_norm_mean", "action_norm_max"):
                    if key in row and row[key] != "":
                        row[key] = float(row[key])
                episode_rows.append(row)
    expected_ids = set(manifest_order)
    if seen != expected_ids:
        raise ValueError(f"shards do not cover frozen manifest: missing={sorted(expected_ids - seen)[:5]}")
    episode_rows.sort(key=lambda row: manifest_order[row["case_id"]])

    provenance_keys = (
        "checkpoint_identity",
        "checkpoint_path",
        "checkpoint_step",
        "checkpoint_params_manifest_sha256",
        "checkpoint_metadata_sha256",
        "checkpoint_kind",
        "lora_arm",
        "normalization_mode",
        "norm_stats_path",
        "norm_stats_sha256",
        "policy_seed_source",
    )
    first_provenance = {
        key: shard_summaries[0].get(key) for key in provenance_keys
    }
    if any(
        {key: summary.get(key) for key in provenance_keys} != first_provenance
        for summary in shard_summaries[1:]
    ):
        raise ValueError("shard summaries disagree on checkpoint or evaluator provenance")
    # A dense step-0 base has no LoRA arm by design; null is meaningful there,
    # while every identity/norm/seed field must still be present.
    missing_provenance = [
        key for key, value in first_provenance.items() if value is None and key != "lora_arm"
    ]
    if missing_provenance:
        raise ValueError(f"shard summary missing required provenance: {missing_provenance}")

    run_root = out_root / (run_id or f"{panel}_{mode}")
    run_root.mkdir(parents=True, exist_ok=True)
    complete_video_dir = run_root / "videos_complete"
    complete_video_dir.mkdir(parents=True, exist_ok=True)
    complete_trace_dir = run_root / "action_traces"
    complete_trace_dir.mkdir(parents=True, exist_ok=True)
    shard_started_at = min((shard_root / "episodes.csv").stat().st_mtime for shard_root in shard_roots)
    shard_gpu_ids = []
    for shard_root in shard_roots:
        match = re.search(r"_retry_(\d+)$", shard_root.name)
        shard_gpu_ids.append(str(int(match.group(1)) + 1) if match else shard_root.name)
    videos: list[str] = []
    for row in episode_rows:
        source = Path(row["video_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = complete_video_dir / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())
        row["video_path"] = str(destination)
        videos.append(str(destination.relative_to(run_root)))
        source_trace = Path(row["action_trace_path"])
        if not source_trace.is_file():
            raise FileNotFoundError(source_trace)
        destination_trace = complete_trace_dir / source_trace.name
        if destination_trace.exists() or destination_trace.is_symlink():
            destination_trace.unlink()
        destination_trace.symlink_to(source_trace.resolve())
        row["action_trace_path"] = str(destination_trace)
    _write_result_tables(run_root, episode_rows)

    success_count = sum(bool(row["success"]) for row in episode_rows)
    stall_count = sum(is_stall_episode(row) for row in episode_rows)
    summary = {
        "schema_version": 1,
        "status": "passed" if success_count > 0 and stall_count <= 0.9 * len(episode_rows) else "failed",
        "panel": panel,
        "execution_protocol": mode,
        "chunk_size": 10,
        "executed_actions_per_chunk": 8 if mode == "execute8" else 10,
        "direct_orbax": True,
        "fast_slow_conversion": False,
        "source_repo": "physical-intelligence/libero",
        "source_revision": OFFICIAL_REVISION,
        "source_fps": 10,
        "git": _git_provenance(),
        "norm_stats_sha256": first_provenance["norm_stats_sha256"],
        "panel_manifest_sha256": sha256_file(manifest_path),
        **first_provenance,
        "case_count": len(episode_rows),
        "case_range": [0, len(episode_rows)],
        "success_count": success_count,
        "success_rate": success_count / len(episode_rows),
        "collision_count": sum(bool(row["collision"]) for row in episode_rows),
        "timeout_count": sum(bool(row["timeout"]) for row in episode_rows),
        "stall_count": stall_count,
        "stall_rate": stall_count / len(episode_rows),
        "mean_action_norm": float(np.mean([float(row["action_norm_mean"]) for row in episode_rows])),
        "max_action_norm": float(np.max([float(row["action_norm_max"]) for row in episode_rows])),
        "video_count": len(videos),
        "videos": sorted(videos),
        "started_at_unix": shard_started_at,
        "finished_at_unix": time.time(),
        "gpu_id": "sharded:" + ",".join(shard_gpu_ids),
        "gpu0_touched": False,
        "artifacts": {"episodes_csv": str(run_root / "episodes.csv"), "tasks_csv": str(run_root / "tasks.csv"), "videos_dir": str(complete_video_dir), "action_traces_dir": str(complete_trace_dir)},
    }
    write_json(run_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "run", "merge"])
    parser.add_argument("--panel", choices=["C20", "B80"])
    parser.add_argument("--mode", choices=["execute8", "full_chunk"])
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--out", type=Path, default=PANEL_ROOT)
    parser.add_argument("--server-python")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", default="axis_pi05_libero_phase1")
    parser.add_argument("--norm-stats", type=Path, default=PHASE1_ROOT / "norm_stats.json")
    parser.add_argument("--arm", choices=["L1_action_lora", "L2_action_lora_heads", "L3_dual_lora_heads", "L4_dual_lora_heads_32d"])
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--shard", type=Path, action="append", default=[])
    args = parser.parse_args()
    out = resolve_path(args.out)
    if args.command == "build":
        print(json.dumps(build_panels(out), ensure_ascii=False, indent=2))
        return
    if args.panel is None or args.mode is None:
        parser.error("run/merge requires --panel and --mode")
    if args.command == "merge":
        if not args.shard:
            parser.error("merge requires one or more --shard paths")
        result = merge_panel_runs(args.panel, args.mode, args.shard, out_root=out, run_id=args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)
        return
    result = run_panel(
        args.panel,
        args.mode,
        args.gpu_id,
        out_root=out,
        server_python=args.server_python,
        start_index=args.start_index,
        end_index=args.end_index,
        run_id=args.run_id,
        manifest_root=resolve_path(args.manifest_root) if args.manifest_root else None,
        checkpoint=resolve_path(args.checkpoint),
        config_name=args.config,
        norm_stats_path=resolve_path(args.norm_stats),
        arm=args.arm,
        checkpoint_step=args.checkpoint_step,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
