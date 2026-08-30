"""Q2 frozen-fixture policy sensitivity audit for the N1 recovery plan."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_http_server import _resize_for_openpi
from .common import ROOT, sha256_file, write_json
from .n1_environment_probe import _array_from_descriptor


FIXTURE_CASES = (
    "C20-task00-init0",
    "C20-task02-init0",
    "C20-task04-init0",
    "C20-task06-init0",
    "C20-task08-init0",
    "C20-task11-init0",
    "C20-task15-init0",
    "C20-task19-init0",
)
SEEDS = ((62000, 63000), (62001, 63001), (62002, 63002), (62003, 63003))
NORM_PATH = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
Q1_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q1_environment_full_c20"
RAW_CHECKPOINT = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
FOLDED_CHECKPOINT = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500"
AXIS_PARAMS = ROOT / "data/models/axisrobotics/Pi0.5-Axis/c342608c489515efa9ba8281400dc648bf3970e1/params"
RUN_DIR = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1"
NESTED_SRC = ROOT / "third_party/axis_openpi/src"


def _hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _fixture_observation(raw: dict[str, Any], norm_stats: Any, prompt: str, policy_seed: int, noise_seed: int) -> dict[str, Any]:
    from openpi.models import model as model_lib
    from openpi.models import tokenizer
    from openpi import transforms
    from openpi.policies import libero_policy

    data: dict[str, Any] = {
        "observation/image": _resize_for_openpi(raw["agentview_image"]),
        "observation/wrist_image": _resize_for_openpi(raw["robot0_eye_in_hand_image"]),
        "observation/state": np.asarray(raw["policy_state"], dtype=np.float32),
        "prompt": prompt,
    }
    data = libero_policy.LiberoInputs(model_type=model_lib.ModelType.PI05)(data)
    data = transforms.Normalize(norm_stats, use_quantiles=True)(data)
    data = transforms.ResizeImages(224, 224)(data)
    data = transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(max_len=200), discrete_state_input=True)(data)
    data = transforms.PadStatesAndActions(32)(data)
    noise = np.random.default_rng(noise_seed).standard_normal((10, 32), dtype=np.float32)
    return {
        "raw_agentview_image": np.asarray(raw["agentview_image"], dtype=np.uint8),
        "raw_wrist_image": np.asarray(raw["robot0_eye_in_hand_image"], dtype=np.uint8),
        "base_0_rgb": np.asarray(data["image"]["base_0_rgb"], dtype=np.float32),
        "left_wrist_0_rgb": np.asarray(data["image"]["left_wrist_0_rgb"], dtype=np.float32),
        "right_wrist_0_rgb": np.asarray(data["image"]["right_wrist_0_rgb"], dtype=np.float32),
        "state": np.asarray(data["state"], dtype=np.float32),
        "tokenized_prompt": np.asarray(data["tokenized_prompt"], dtype=np.int32),
        "tokenized_prompt_mask": np.asarray(data["tokenized_prompt_mask"], dtype=bool),
        "action_is_pad": np.zeros((10,), dtype=bool),
        "noise": noise,
        "policy_seed": np.asarray(policy_seed, dtype=np.int64),
        "noise_seed": np.asarray(noise_seed, dtype=np.int64),
    }


def build_fixtures(output: Path = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q2_fixtures") -> dict[str, Any]:
    import json as _json
    from openpi.shared import normalize

    manifest = [
        _json.loads(line)
        for line in (ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_case = {row["case_id"]: row for row in manifest}
    summary = json.loads((Q1_ROOT / "replica0/summary.json").read_text(encoding="utf-8"))
    records = {case["case_id"]: case for case in summary["records"][0]["cases"]}
    norm_stats = normalize.deserialize_json(NORM_PATH.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    fixtures: list[dict[str, Any]] = []
    for case_id in FIXTURE_CASES:
        row = by_case[case_id]
        case = records[case_id]
        boundary = next(item for item in case["snapshots"] if item.get("stage") == "E3" and item.get("warmup_step") == 10)
        raw = {
            "agentview_image": _array_from_descriptor(boundary["observation"]["agentview_image"]),
            "robot0_eye_in_hand_image": _array_from_descriptor(boundary["observation"]["robot0_eye_in_hand_image"]),
            "policy_state": _array_from_descriptor(boundary["policy_state"]),
        }
        if any(value is None for value in raw.values()):
            raise ValueError(f"incomplete Q1 policy-boundary fixture: {case_id}")
        for seed_index, (policy_seed, noise_seed) in enumerate(SEEDS):
            fixture_id = f"{case_id}-s{seed_index}"
            values = _fixture_observation(raw, norm_stats, str(row["language"]), policy_seed, noise_seed)
            path = output / f"{fixture_id}.npz"
            np.savez_compressed(path, **values)
            fixtures.append({
                "fixture_id": fixture_id,
                "case_id": case_id,
                "task_name": row["task_name"],
                "prompt": row["language"],
                "policy_seed": policy_seed,
                "noise_seed": noise_seed,
                "source_boundary": f"{Q1_ROOT}/replica0/summary.json::{case_id}::E3-10",
                "sha256": sha256_file(path),
                "path": str(path),
                "shapes_dtypes": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in values.items()},
            })
    report = {
        "schema_version": 1,
        "phase": "Q2",
        "fixture_type": "frozen_observation_policy_sensitivity",
        "fixture_count": len(fixtures),
        "case_count": len(FIXTURE_CASES),
        "seeds_per_case": len(SEEDS),
        "cases": list(FIXTURE_CASES),
        "norm_stats": str(NORM_PATH),
        "norm_stats_sha256": sha256_file(NORM_PATH),
        "transform": "production OpenPI LIBERO image transform + QUANTILES state normalization + Pi05 PaligemmaTokenizer + 32D padding",
        "fixtures": fixtures,
    }
    write_json(output / "fixture_manifest.json", report)
    return report


def _load_model(kind: str):
    if str(NESTED_SRC) not in sys.path:
        sys.path.insert(0, str(NESTED_SRC))
    if str(ROOT) not in sys.path:
        sys.path.insert(1, str(ROOT))
    import jax
    import flax.nnx as nnx
    from openpi.models import model as model_lib
    from openpi.models import pi0_config
    from openpi.shared import normalize
    from openpi.training import config as config_lib
    from tools.axis_pi05.lora_folding import fold_lora_in_mapping

    norm_stats = normalize.deserialize_json(NORM_PATH.read_text(encoding="utf-8"))
    if kind == "axis":
        model_config = pi0_config.Pi0Config(pi05=True, dtype="float32", action_dim=32, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
        loaded = model_lib.restore_params(AXIS_PARAMS, restore_type=np.ndarray, dtype="float32")
        return model_config.load(loaded), norm_stats
    os.environ.update({
        "AXIS_PI05_LORA_ARM": "dual_l2",
        "AXIS_PI05_LORA_RUN_DIR": str(RUN_DIR),
        "AXIS_PI05_LORA_NORM_STATS": str(NORM_PATH),
        "AXIS_PI05_LORA_AXIS_PARAMS": str(AXIS_PARAMS),
    })
    config = config_lib.get_config("axis_pi05_lora_screen")
    raw_params = model_lib.restore_params(RAW_CHECKPOINT / "params", restore_type=np.ndarray, dtype="float32")
    if kind == "raw_lora":
        return config.model.load(raw_params), norm_stats
    if kind != "folded_dense":
        raise ValueError(kind)
    folded_params = copy.deepcopy(raw_params)
    fold_lora_in_mapping(folded_params)
    model_config = pi0_config.Pi0Config(pi05=True, dtype="float32", action_dim=32, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    template = model_config.create(jax.random.key(0))
    graphdef, state = nnx.split(template)
    state.replace_by_pure_dict(folded_params)
    return nnx.merge(graphdef, state), norm_stats


def run_model(kind: str, fixtures: Path, output: Path) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi import transforms
    from openpi.shared import normalize

    model, norm_stats = _load_model(kind)
    model.eval()
    manifest = json.loads((fixtures / "fixture_manifest.json").read_text(encoding="utf-8"))
    internal: list[np.ndarray] = []
    external: list[np.ndarray] = []
    input_hashes: list[dict[str, Any]] = []
    for fixture in manifest["fixtures"]:
        values = dict(np.load(fixture["path"], allow_pickle=False))
        observation = model_lib.Observation.from_dict({
            "image": {
                "base_0_rgb": jnp.asarray(values["base_0_rgb"])[None],
                "left_wrist_0_rgb": jnp.asarray(values["left_wrist_0_rgb"])[None],
                "right_wrist_0_rgb": jnp.asarray(values["right_wrist_0_rgb"])[None],
            },
            "image_mask": {"base_0_rgb": jnp.ones((1,), dtype=bool), "left_wrist_0_rgb": jnp.ones((1,), dtype=bool), "right_wrist_0_rgb": jnp.zeros((1,), dtype=bool)},
            "state": jnp.asarray(values["state"])[None],
            "tokenized_prompt": jnp.asarray(values["tokenized_prompt"])[None],
            "tokenized_prompt_mask": jnp.asarray(values["tokenized_prompt_mask"])[None],
            "action_is_pad": jnp.asarray(values["action_is_pad"])[None],
        })
        actions = np.asarray(jax.block_until_ready(model.sample_actions(jax.random.key(int(values["policy_seed"])), observation, num_steps=10, noise=jnp.asarray(values["noise"])[None])))
        unnormalized = transforms.Unnormalize(norm_stats, use_quantiles=True)({"state": values["state"][None], "actions": actions})
        actions_external = np.asarray(unnormalized["actions"][..., :7], dtype=np.float32)
        if not np.isfinite(actions).all() or not np.isfinite(actions_external).all():
            raise FloatingPointError(f"non-finite action for {fixture['fixture_id']}")
        internal.append(actions[0].astype(np.float32, copy=False))
        external.append(actions_external[0])
        input_hashes.append({key: _hash(values[key]) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask", "noise")})
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "policy_outputs.npz"
    np.savez_compressed(artifact, internal_actions=np.stack(internal), external_actions=np.stack(external))
    report = {
        "schema_version": 1,
        "phase": "Q2",
        "kind": kind,
        "status": "passed",
        "fixture_count": len(internal),
        "all_finite": True,
        "input_hashes": input_hashes,
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "norm_stats_sha256": sha256_file(NORM_PATH),
        "gpu_id_physical": os.environ.get("AXIS_Q2_GPU_ID"),
    }
    write_json(output / "policy_report.json", report)
    return report


def compare_outputs(first: Path, second: Path, output: Path, *, label: str) -> dict[str, Any]:
    left, right = np.load(first), np.load(second)
    result: dict[str, Any] = {"schema_version": 1, "phase": "Q2", "comparison": label, "first": str(first), "second": str(second), "status": "passed"}
    for name in ("internal_actions", "external_actions"):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result[name] = {"shape_match": a.shape == b.shape, "dtype_match": a.dtype == b.dtype, "max_abs_diff": float(np.max(difference)), "p99_abs_diff": float(np.percentile(difference, 99)), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()), "exact": bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))}
        if not result[name]["shape_match"] or not result[name]["finite"]:
            result["status"] = "failed"
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-fixtures")
    build.add_argument("--output", type=Path, default=ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q2_fixtures")
    run = sub.add_parser("run-model")
    run.add_argument("--kind", choices=("axis", "raw_lora", "folded_dense"), required=True)
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.command == "build-fixtures":
        print(json.dumps(build_fixtures(args.output), ensure_ascii=False, indent=2))
    elif args.command == "run-model":
        print(json.dumps(run_model(args.kind, args.fixtures, args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compare_outputs(args.first, args.second, args.output, label=args.label), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
