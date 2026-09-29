"""Audit OpenPI/JAX -> folded PyTorch -> LeRobot processor action parity.

The fixture is intentionally the same fixed observation/noise used by the raw
LoRA versus folded-dense audit.  The LeRobot preprocessor must recreate the
normalized state and prompt consumed by OpenPI, and the converted PyTorch
model must agree with the folded JAX model before the LeRobot postprocessor
externalizes the first seven action dimensions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, resolve_path, sha256_file, write_json

R6_JAX_CHECKPOINT = (
    ROOT
    / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
)


LEROBOT_SRC = Path(
    os.environ.get(
        "AXIS_LEROBOT_SRC",
        str(ROOT.parent / "AIROA-contest-libero-plus" / "workspace" / "training" / "lerobot" / "src"),
    )
)
NESTED_SRC = ROOT / "third_party" / "axis_openpi" / "src"


def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def _max_abs(left: Any, right: Any) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _diff_summary(left: Any, right: Any) -> dict[str, Any]:
    a = np.asarray(left)
    b = np.asarray(right)
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape_match": a.shape == b.shape,
        "dtype_match": a.dtype == b.dtype,
        "max_abs_diff": float(np.max(difference)),
        "p99_abs_diff": float(np.percentile(difference, 99)),
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "exact": bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)),
    }


def run(
    model_dir: Path,
    fixed_input: Path,
    norm_stats: Path,
    output_dir: Path,
    *,
    task: str | list[str],
    gpu_id: str = "1",
    seed: int = 0,
    microbatch_size: int = 1,
    same_process_replay: bool = False,
    jax_checkpoint: Path | None = None,
    lora_run_dir: Path | None = None,
) -> dict[str, Any]:
    if gpu_id == "0":
        raise ValueError("GPU0 is outside the parity boundary")
    if not model_dir.is_dir() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"folded model is missing: {model_dir}")
    if not fixed_input.is_file():
        raise FileNotFoundError(fixed_input)
    if not norm_stats.is_file():
        raise FileNotFoundError(norm_stats)
    if microbatch_size < 1:
        raise ValueError("microbatch_size must be positive")

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if source_config.get("type") != "pi05":
        raise ValueError(f"unexpected folded source model type: {source_config.get('type')!r}")
    if source_config.get("chunk_size") != 10 or source_config.get("n_action_steps") != 8:
        raise ValueError("folded source contract is not chunk=10/execute8")
    processor_paths = sorted(
        path for pattern in ("policy_preprocessor*", "policy_postprocessor*")
        for path in model_dir.glob(pattern)
        if path.is_file()
    )
    required_processor_names = {
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    }
    if required_processor_names.difference(path.name for path in processor_paths):
        raise ValueError("folded source processor set is incomplete")

    # Set this before importing JAX or torch.  The logical cuda:0 below is
    # physical GPU<gpu_id>; physical GPU0 is therefore not visible.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PALIGEMMA_TOKENIZER_DIR", str(
        ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging/vendor/paligemma_tokenizer"
    ))
    from .n1_snapshot import gpu_boundary_evidence, gpu_snapshot

    gpu_monitor_samples = [gpu_snapshot()]
    sys.path.insert(0, str(LEROBOT_SRC))
    sys.path.insert(1, str(NESTED_SRC))
    sys.path.insert(2, str(ROOT))

    import sentencepiece  # noqa: F401  # preload before the vendored tokenizer path is added
    import torch
    import safetensors.torch

    from lerobot.processor import DataProcessorPipeline

    arrays = np.load(fixed_input)
    required = {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "noise",
    }
    missing = sorted(required.difference(arrays.files))
    if missing:
        raise ValueError(f"fixed input missing fields: {missing}")
    batch_size = int(arrays["state"].shape[0])
    if tuple(arrays["noise"].shape) != (batch_size, 10, 32):
        raise ValueError(f"unexpected fixed noise shape: {arrays['noise'].shape}")
    tasks = [task] * batch_size if isinstance(task, str) else list(task)
    if len(tasks) != batch_size or not all(isinstance(value, str) for value in tasks):
        raise ValueError(f"task list must have one string per sample: {len(tasks)} != {batch_size}")

    with norm_stats.open(encoding="utf-8") as handle:
        norm = json.load(handle)["norm_stats"]
    q01_state = np.asarray(norm["state"]["q01"], dtype=np.float32)
    q99_state = np.asarray(norm["state"]["q99"], dtype=np.float32)
    q01_action = np.asarray(norm["actions"]["q01"], dtype=np.float32)
    q99_action = np.asarray(norm["actions"]["q99"], dtype=np.float32)
    normalized_state = np.asarray(arrays["state"][:, :8], dtype=np.float32)
    raw_state = (normalized_state + 1.0) * (q99_state - q01_state) / 2.0 + q01_state

    # Feed the same normalized [-1, 1] image convention as OpenPI's fixed
    # observation.  Visual normalization is IDENTITY in the R6 artifact.
    preprocessor_input = {
        "observation.images.image": torch.from_numpy(np.asarray(arrays["base_0_rgb"]).transpose(0, 3, 1, 2)),
        "observation.images.image2": torch.from_numpy(np.asarray(arrays["left_wrist_0_rgb"]).transpose(0, 3, 1, 2)),
        "observation.images.empty_camera_0": torch.from_numpy(np.asarray(arrays["right_wrist_0_rgb"]).transpose(0, 3, 1, 2)),
        "observation.state": torch.from_numpy(raw_state),
        "task": tasks,
    }
    preprocessor = DataProcessorPipeline.from_pretrained(
        model_dir, config_filename="policy_preprocessor.json", local_files_only=True
    )
    processed = preprocessor(preprocessor_input)
    processed_images = {
        key: processed[key].detach().cpu()
        for key in (
            "observation.images.image",
            "observation.images.image2",
            "observation.images.empty_camera_0",
        )
    }
    processed_state = processed["observation.state"].detach().cpu().numpy()
    processed_tokens_tensor = processed["observation.language.tokens"].detach().cpu()
    processed_token_mask_tensor = processed["observation.language.attention_mask"].detach().cpu()
    processed_tokens = processed["observation.language.tokens"].detach().cpu().numpy().astype(np.int32)
    processed_token_mask = processed["observation.language.attention_mask"].detach().cpu().numpy().astype(bool)
    fixed_tokens = np.asarray(arrays["tokenized_prompt"], dtype=np.int32)
    fixed_token_mask = np.asarray(arrays["tokenized_prompt_mask"], dtype=bool)
    state_diff = _max_abs(processed_state, normalized_state)
    token_diff = int(np.max(np.abs(processed_tokens.astype(np.int64) - fixed_tokens.astype(np.int64))))
    token_mask_diff = int(np.count_nonzero(processed_token_mask != fixed_token_mask))
    if processed_tokens.shape != fixed_tokens.shape:
        raise ValueError(f"processor token shape mismatch: {processed_tokens.shape} != {fixed_tokens.shape}")
    # The processor's configured device is CUDA.  Keep only CPU snapshots so
    # its normalization state and batch tensors do not overlap the model peak.
    del processed, preprocessor, preprocessor_input
    gc.collect()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("physical GPU for e2e parity is unavailable")
    model_config = __import__("openpi.models.pi0_config", fromlist=["Pi0Config"]).Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="float32",
        pytorch_compile_mode=None,
    )
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    pytorch_model = PI0Pytorch(model_config).to(device).eval()
    safetensors.torch.load_model(pytorch_model, str(model_dir / "model.safetensors"))
    gpu_monitor_samples.append(gpu_snapshot())
    observation_cls = __import__("openpi.models.model", fromlist=["Observation"]).Observation

    def sample_pytorch(start: int, stop: int) -> np.ndarray:
        width = stop - start
        pt_observation = observation_cls.from_dict(
            {
                "image": {
                    "base_0_rgb": processed_images["observation.images.image"][start:stop].to(device),
                    "left_wrist_0_rgb": processed_images["observation.images.image2"][start:stop].to(device),
                    "right_wrist_0_rgb": processed_images["observation.images.empty_camera_0"][start:stop].to(device),
                },
                "image_mask": {
                    "base_0_rgb": torch.ones(width, dtype=torch.bool, device=device),
                    "left_wrist_0_rgb": torch.ones(width, dtype=torch.bool, device=device),
                    "right_wrist_0_rgb": torch.zeros(width, dtype=torch.bool, device=device),
                },
                "state": F_torch_pad(processed_state[start:stop], device=device),
                "tokenized_prompt": processed_tokens_tensor[start:stop].to(device=device, dtype=torch.int32),
                "tokenized_prompt_mask": processed_token_mask_tensor[start:stop].to(device=device),
            }
        )
        noise_torch = torch.from_numpy(np.asarray(arrays["noise"][start:stop], dtype=np.float32)).to(device)
        with torch.inference_mode():
            actions = pytorch_model.sample_actions(
                device, pt_observation, noise=noise_torch, num_steps=10
            )
        return actions.detach().cpu().numpy().astype(np.float32)

    pytorch_actions_np = np.concatenate(
        [sample_pytorch(start, min(start + microbatch_size, batch_size)) for start in range(0, batch_size, microbatch_size)],
        axis=0,
    )
    pytorch_actions_replay_np = None
    if same_process_replay:
        pytorch_actions_replay_np = np.concatenate(
            [sample_pytorch(start, min(start + microbatch_size, batch_size)) for start in range(0, batch_size, microbatch_size)],
            axis=0,
        )
    if not np.isfinite(pytorch_actions_np).all():
        raise FloatingPointError("folded PyTorch action contains NaN/Inf")
    del pytorch_model, model_config
    gc.collect()
    torch.cuda.empty_cache()

    # Folded JAX reference must come from the same checkpoint that produced
    # model.safetensors. Default remains R6 so historical audits stay valid.
    jax_checkpoint = Path(jax_checkpoint) if jax_checkpoint is not None else R6_JAX_CHECKPOINT
    lora_run_dir = Path(lora_run_dir) if lora_run_dir is not None else jax_checkpoint
    if not (jax_checkpoint / "params").is_dir():
        raise FileNotFoundError(jax_checkpoint / "params")
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from tools.axis_pi05.lora_folding import fold_lora_in_mapping

    os.environ.update(
        {
            "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
            "AXIS_PI05_LORA_RUN_DIR": str(lora_run_dir),
            "AXIS_PI05_LORA_NORM_STATS": str(norm_stats),
            "AXIS_PI05_LORA_AXIS_PARAMS": str(
                ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1/params"
            ),
        }
    )
    training_cfg = config_lib.get_config("axis_pi05_lora_screen")
    raw_params = model_lib.restore_params(
        jax_checkpoint / "params",
        restore_type=np.ndarray,
        dtype="float32",
    )
    folded_params = copy.deepcopy(raw_params)
    folded_modules = fold_lora_in_mapping(folded_params)
    dense_config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    dense_template = dense_config.create(jax.random.key(0))
    dense_graphdef, dense_state = nnx.split(dense_template)
    dense_state.replace_by_pure_dict(folded_params)
    dense_model = nnx.merge(dense_graphdef, dense_state)
    dense_model.eval()
    image_arrays = {
        "base_0_rgb": processed_images["observation.images.image"].numpy().transpose(0, 2, 3, 1),
        "left_wrist_0_rgb": processed_images["observation.images.image2"].numpy().transpose(0, 2, 3, 1),
        "right_wrist_0_rgb": processed_images["observation.images.empty_camera_0"].numpy().transpose(0, 2, 3, 1),
    }

    def sample_jax(start: int, stop: int) -> np.ndarray:
        width = stop - start
        jax_observation = model_lib.Observation.from_dict(
            {
                "image": {key: jnp.asarray(value[start:stop]) for key, value in image_arrays.items()},
                "image_mask": {
                    "base_0_rgb": jnp.ones((width,), dtype=bool),
                    "left_wrist_0_rgb": jnp.ones((width,), dtype=bool),
                    "right_wrist_0_rgb": jnp.zeros((width,), dtype=bool),
                },
                "state": jnp.asarray(np.asarray(processed_state[start:stop], dtype=np.float32).copy().tolist()),
                "tokenized_prompt": jnp.asarray(processed_tokens[start:stop]),
                "tokenized_prompt_mask": jnp.asarray(processed_token_mask[start:stop]),
            }
        )
        return np.asarray(
            jax.block_until_ready(
                dense_model.sample_actions(
                    jax.random.key(seed),
                    jax_observation,
                    num_steps=10,
                    noise=jnp.asarray(arrays["noise"][start:stop], dtype=jnp.float32),
                )
            ),
            dtype=np.float32,
        )

    jax_actions = np.concatenate(
        [sample_jax(start, min(start + microbatch_size, batch_size)) for start in range(0, batch_size, microbatch_size)],
        axis=0,
    )
    if not np.isfinite(jax_actions).all():
        raise FloatingPointError("folded JAX action contains NaN/Inf")
    gpu_monitor_samples.append(gpu_snapshot())

    postprocessor = DataProcessorPipeline.from_pretrained(
        model_dir, config_filename="policy_postprocessor.json", local_files_only=True
    )
    postprocessed = postprocessor({"action": torch.from_numpy(pytorch_actions_np[..., :7])})["action"]
    postprocessed_np = postprocessed.detach().cpu().numpy().astype(np.float32)
    direct_external = (
        (pytorch_actions_np[..., :7] + 1.0)
        * (q99_action - q01_action)
        / 2.0
        + q01_action
    ).astype(np.float32)
    jax_external = (
        (jax_actions[..., :7] + 1.0) * (q99_action - q01_action) / 2.0 + q01_action
    ).astype(np.float32)
    pt_jax_diff = _max_abs(pytorch_actions_np, jax_actions)
    post_direct_diff = _max_abs(postprocessed_np, direct_external)
    post_jax_diff = _max_abs(postprocessed_np, jax_external)
    tolerance = {"atol": 1e-3, "rtol": 1e-3, "relative_error_is_diagnostic_only": True}
    model_within = bool(np.allclose(pytorch_actions_np, jax_actions, **{k: tolerance[k] for k in ("atol", "rtol")}))
    processor_within = state_diff <= 1e-6 and token_diff == 0 and token_mask_diff == 0
    post_within = bool(np.allclose(postprocessed_np, jax_external, **{k: tolerance[k] for k in ("atol", "rtol")}))
    same_process_report = None
    same_process_within = True
    if same_process_replay:
        assert pytorch_actions_replay_np is not None
        same_process_report = {
            "internal": _diff_summary(pytorch_actions_np, pytorch_actions_replay_np),
            "external": _diff_summary(
                ((pytorch_actions_np[..., :7] + 1.0) * (q99_action - q01_action) / 2.0 + q01_action).astype(np.float32),
                ((pytorch_actions_replay_np[..., :7] + 1.0) * (q99_action - q01_action) / 2.0 + q01_action).astype(np.float32),
            ),
        }
        same_process_within = bool(same_process_report["internal"]["exact"] and same_process_report["external"]["exact"])

    del dense_model
    gc.collect()
    gpu_monitor_samples.append(gpu_snapshot())
    gpu_evidence = gpu_boundary_evidence(
        gpu_monitor_samples,
        target_pids={os.getpid()},
        allowed_gpu_ids={int(gpu_id)},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "processor_state": processed_state,
        "fixed_state": normalized_state,
        "processor_tokens": processed_tokens,
        "fixed_tokens": fixed_tokens,
        "folded_jax_actions": jax_actions,
        "folded_pytorch_actions": pytorch_actions_np,
        "lerobot_postprocessed_actions": postprocessed_np,
        "folded_jax_external_actions": jax_external,
    }
    if pytorch_actions_replay_np is not None:
        artifact_payload["folded_pytorch_actions_replay"] = pytorch_actions_replay_np
    np.savez_compressed(output_dir / "e2e_action_parity.npz", **artifact_payload)
    report = {
        "schema_version": 1,
        "status": "passed" if processor_within and model_within and post_within and same_process_within else "failed",
        "backend": "lerobot_disk_processors_and_model_safetensors_with_openpi_pytorch_core",
        "model_dir": str(model_dir),
        "jax_checkpoint": str(jax_checkpoint),
        "lora_run_dir": str(lora_run_dir),
        "model_sha256": sha256_file(model_dir / "model.safetensors"),
        "source_config_sha256": sha256_file(config_path),
        "source_config_contract": {
            "type": source_config["type"],
            "dtype": source_config.get("dtype"),
            "chunk_size": source_config["chunk_size"],
            "n_action_steps": source_config["n_action_steps"],
            "max_action_dim": source_config.get("max_action_dim"),
        },
        "processor_hashes": {path.name: sha256_file(path) for path in processor_paths},
        "fixed_input": str(fixed_input),
        "fixed_input_sha256": sha256_file(fixed_input),
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": sha256_file(norm_stats),
        "task": tasks if len(set(tasks)) > 1 else tasks[0],
        "tasks": tasks,
        "batch_size": batch_size,
        "microbatch_size": microbatch_size,
        "gpu_id": str(gpu_id),
        "gpu0_touched": gpu_evidence["gpu0_touched"],
        "gpu_monitor_evidence": {"samples": gpu_monitor_samples, **gpu_evidence},
        "folded_lora_modules": folded_modules,
        "normalization": "QUANTILES",
        "discrete_state_input": False,
        "processor_state_max_abs_diff": state_diff,
        "processor_token_max_abs_diff": token_diff,
        "processor_token_mask_mismatch_count": token_mask_diff,
        "processor_within_tolerance": processor_within,
        "folded_jax": _fingerprint(jax_actions),
        "folded_pytorch": _fingerprint(pytorch_actions_np),
        "folded_pytorch_vs_jax_max_abs_diff": pt_jax_diff,
        "folded_pytorch_vs_jax_within_tolerance": model_within,
        "lerobot_postprocessed": _fingerprint(postprocessed_np),
        "lerobot_postprocessor_vs_direct_quantiles_max_abs_diff": post_direct_diff,
        "lerobot_postprocessor_vs_folded_jax_external_max_abs_diff": post_jax_diff,
        "lerobot_postprocessor_within_tolerance": post_within,
        "same_process_replay_required": same_process_replay,
        "same_process_replay": same_process_report,
        "same_process_replay_within_tolerance": same_process_within,
        "parity_tolerance": tolerance,
        "artifact": str(output_dir / "e2e_action_parity.npz"),
    }
    write_json(output_dir / "e2e_action_parity_report.json", report)
    return report


def F_torch_pad(state: np.ndarray, *, device: Any) -> Any:
    """Pad the eight-dimensional LeRobot state to Pi05's 32-wide contract."""

    import torch

    return torch.nn.functional.pad(torch.from_numpy(np.asarray(state, dtype=np.float32)), (0, 24)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixed-input", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task")
    task_group.add_argument("--tasks-file", type=Path)
    parser.add_argument("--gpu-id", default="1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--same-process-replay", action="store_true")
    parser.add_argument("--jax-checkpoint", type=Path)
    parser.add_argument("--lora-run-dir", type=Path)
    args = parser.parse_args()

    task: str | list[str]
    if args.tasks_file is not None:
        payload = json.loads(resolve_path(args.tasks_file).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            task = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            task = payload["tasks"]
        else:
            raise ValueError("--tasks-file must contain a JSON list or an object with a tasks list")
    else:
        task = args.task
    print(json.dumps(run(
        resolve_path(args.model_dir),
        resolve_path(args.fixed_input),
        resolve_path(args.norm_stats),
        resolve_path(args.output),
        task=task,
        gpu_id=args.gpu_id,
        seed=args.seed,
        microbatch_size=args.microbatch_size,
        same_process_replay=args.same_process_replay,
        jax_checkpoint=None if args.jax_checkpoint is None else resolve_path(args.jax_checkpoint),
        lora_run_dir=None if args.lora_run_dir is None else resolve_path(args.lora_run_dir),
    ), indent=2))


if __name__ == "__main__":
    main()
