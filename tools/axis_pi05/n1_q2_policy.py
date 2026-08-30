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


def build_snapshot_fixture(case_id: str, warmup_step: int, output: Path, *, seed_index: int = 0) -> dict[str, Any]:
    """Materialize a policy-boundary fixture for a selected real Q1 snapshot."""
    import json as _json
    from openpi.shared import normalize

    manifest = [
        _json.loads(line)
        for line in (ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row = next(item for item in manifest if item["case_id"] == case_id)
    summary = json.loads((Q1_ROOT / "replica0/summary.json").read_text(encoding="utf-8"))
    case = next(item for item in summary["records"][0]["cases"] if item["case_id"] == case_id)
    snapshot = next(item for item in case["snapshots"] if item.get("stage") == "E3" and item.get("warmup_step") == warmup_step)
    raw = {
        "agentview_image": _array_from_descriptor(snapshot["observation"]["agentview_image"]),
        "robot0_eye_in_hand_image": _array_from_descriptor(snapshot["observation"]["robot0_eye_in_hand_image"]),
        "policy_state": _array_from_descriptor(snapshot["policy_state"]),
    }
    policy_seed, noise_seed = SEEDS[seed_index]
    values = _fixture_observation(raw, normalize.deserialize_json(NORM_PATH.read_text(encoding="utf-8")), str(row["language"]), policy_seed, noise_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    report = {
        "schema_version": 1,
        "phase": "Q2",
        "fixture_type": "selected_real_q1_snapshot",
        "case_id": case_id,
        "warmup_step": warmup_step,
        "policy_seed": policy_seed,
        "noise_seed": noise_seed,
        "source_snapshot": f"{Q1_ROOT}/replica0/summary.json::{case_id}::E3-{warmup_step}",
        "fixture": str(output),
        "fixture_sha256": sha256_file(output),
        "norm_stats_sha256": sha256_file(NORM_PATH),
    }
    write_json(output.with_suffix(".json"), report)
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


def _observation_from_values(values: dict[str, np.ndarray], model_lib: Any, jnp: Any) -> Any:
    return model_lib.Observation.from_dict({
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


def _run_one(model: Any, values: dict[str, np.ndarray], model_lib: Any, transforms: Any, norm_stats: Any, jax: Any, jnp: Any) -> tuple[np.ndarray, np.ndarray]:
    observation = _observation_from_values(values, model_lib, jnp)
    actions = np.asarray(jax.block_until_ready(model.sample_actions(jax.random.key(int(values["policy_seed"])), observation, num_steps=10, noise=jnp.asarray(values["noise"])[None])))
    unnormalized = transforms.Unnormalize(norm_stats, use_quantiles=True)({"state": values["state"][None], "actions": actions})
    actions_external = np.asarray(unnormalized["actions"][..., :7], dtype=np.float32)
    if not np.isfinite(actions).all() or not np.isfinite(actions_external).all():
        raise FloatingPointError("non-finite action")
    return actions[0].astype(np.float32, copy=False), actions_external[0]


def _diff_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    difference = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    result: dict[str, Any] = {
        "shape_match": a.shape == b.shape,
        "dtype_match": a.dtype == b.dtype,
        "max_abs_diff": float(np.max(difference)),
        "p99_abs_diff": float(np.percentile(difference, 99)),
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "exact": bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)),
    }
    if difference.ndim and difference.shape[-1]:
        by_dim = difference.reshape(-1, difference.shape[-1])
        result["per_dim_max_abs_diff"] = np.max(by_dim, axis=0).tolist()
        result["per_dim_p99_abs_diff"] = np.percentile(by_dim, 99, axis=0).tolist()
    return result


def _transformed_pair(agentview: np.ndarray, wrist: np.ndarray, norm_stats: Any) -> dict[str, np.ndarray]:
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies import libero_policy

    data: dict[str, Any] = {
        "observation/image": _resize_for_openpi(agentview),
        "observation/wrist_image": _resize_for_openpi(wrist),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "",
    }
    data = libero_policy.LiberoInputs(model_type=model_lib.ModelType.PI05)(data)
    data = transforms.Normalize(norm_stats, use_quantiles=True)(data)
    data = transforms.ResizeImages(224, 224)(data)
    # This helper is used only for image tensors. Tokenization is performed from
    # the base fixture so that the pair changes no text/state input.
    return {
        "base_0_rgb": np.asarray(data["image"]["base_0_rgb"], dtype=np.float32),
        "left_wrist_0_rgb": np.asarray(data["image"]["left_wrist_0_rgb"], dtype=np.float32),
        "right_wrist_0_rgb": np.asarray(data["image"]["right_wrist_0_rgb"], dtype=np.float32),
    }


def run_model(kind: str, fixtures: Path, output: Path, *, same_process_replay: bool = False) -> dict[str, Any]:
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
    replay_internal: list[np.ndarray] = []
    replay_external: list[np.ndarray] = []
    for fixture in manifest["fixtures"]:
        values = dict(np.load(fixture["path"], allow_pickle=False))
        first_internal, first_external = _run_one(model, values, model_lib, transforms, norm_stats, jax, jnp)
        internal.append(first_internal)
        external.append(first_external)
        if same_process_replay:
            second_internal, second_external = _run_one(model, values, model_lib, transforms, norm_stats, jax, jnp)
            replay_internal.append(second_internal)
            replay_external.append(second_external)
        input_hashes.append({key: _hash(values[key]) for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask", "noise")})
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "policy_outputs.npz"
    payload: dict[str, np.ndarray] = {"internal_actions": np.stack(internal), "external_actions": np.stack(external)}
    if same_process_replay:
        payload.update({"internal_actions_replay": np.stack(replay_internal), "external_actions_replay": np.stack(replay_external)})
    np.savez_compressed(artifact, **payload)
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
    if same_process_replay:
        report["same_process_replay"] = {
            "internal": _diff_summary(payload["internal_actions"], payload["internal_actions_replay"]),
            "external": _diff_summary(payload["external_actions"], payload["external_actions_replay"]),
            "gate_pass": bool(
                _diff_summary(payload["internal_actions"], payload["internal_actions_replay"])["exact"]
                and _diff_summary(payload["external_actions"], payload["external_actions_replay"])["exact"]
            ),
        }
    write_json(output / "policy_report.json", report)
    return report


def compare_outputs(first: Path, second: Path, output: Path, *, label: str) -> dict[str, Any]:
    left, right = np.load(first), np.load(second)
    result: dict[str, Any] = {"schema_version": 1, "phase": "Q2", "comparison": label, "first": str(first), "second": str(second), "status": "passed"}
    for name in ("internal_actions", "external_actions"):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result[name] = _diff_summary(a, b)
        if not result[name]["shape_match"] or not result[name]["dtype_match"] or not result[name]["finite"] or result[name]["max_abs_diff"] > 1e-6:
            result["status"] = "failed"
    write_json(output, result)
    return result


def run_real_pair(kind: str, base_fixture: Path, left_agentview: Path, right_agentview: Path, left_wrist: Path, right_wrist: Path, output: Path, *, label: str) -> dict[str, Any]:
    """Run one actual Q1 renderer pair with all non-image inputs held fixed."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_lib
    from openpi import transforms

    model, norm_stats = _load_model(kind)
    model.eval()
    base = dict(np.load(base_fixture, allow_pickle=False))
    left_agent = np.load(left_agentview, allow_pickle=False)
    right_agent = np.load(right_agentview, allow_pickle=False)
    left_wrist_arr = np.load(left_wrist, allow_pickle=False)
    right_wrist_arr = np.load(right_wrist, allow_pickle=False)
    left_images = _transformed_pair(left_agent, left_wrist_arr, norm_stats)
    right_images = _transformed_pair(right_agent, right_wrist_arr, norm_stats)
    left = dict(base)
    right = dict(base)
    left.update(left_images)
    right.update(right_images)
    left_internal, left_external = _run_one(model, left, model_lib, transforms, norm_stats, jax, jnp)
    right_internal, right_external = _run_one(model, right, model_lib, transforms, norm_stats, jax, jnp)
    from .n1_environment_probe import _image_diff_metrics
    internal = _diff_summary(left_internal, right_internal)
    external = _diff_summary(left_external, right_external)
    gripper_left = np.sign(left_external[..., -1])
    gripper_right = np.sign(right_external[..., -1])
    result = {
        "schema_version": 1,
        "phase": "Q2",
        "comparison": label,
        "kind": kind,
        "base_fixture": str(base_fixture),
        "raw_image": {"agentview": _image_diff_metrics(left_agent, right_agent), "wrist": _image_diff_metrics(left_wrist_arr, right_wrist_arr)},
        "transformed_image": {"agentview": _diff_summary(left["base_0_rgb"], right["base_0_rgb"]), "wrist": _diff_summary(left["left_wrist_0_rgb"], right["left_wrist_0_rgb"])},
        "token_mask_mismatch": bool(not np.array_equal(left["tokenized_prompt"], right["tokenized_prompt"]) or not np.array_equal(left["tokenized_prompt_mask"], right["tokenized_prompt_mask"])),
        "state_exact": bool(np.array_equal(left["state"], right["state"])),
        "internal_actions": internal,
        "external_actions": external,
        "first_internal": _diff_summary(left_internal[0], right_internal[0]),
        "first_external": _diff_summary(left_external[0], right_external[0]),
        "execute8_external": _diff_summary(left_external[:8], right_external[:8]),
        "gripper_sign_flip": bool(np.any(gripper_left != gripper_right)),
    }
    result["status"] = "passed" if (
        not result["token_mask_mismatch"] and result["state_exact"] and internal["finite"] and external["finite"]
        and not result["gripper_sign_flip"] and external["max_abs_diff"] <= 1e-6
    ) else "failed"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    np.savez_compressed(output.with_suffix(".npz"), left_internal=left_internal, right_internal=right_internal, left_external=left_external, right_external=right_external)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-fixtures")
    build.add_argument("--output", type=Path, default=ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q2_fixtures")
    snapshot = sub.add_parser("build-snapshot-fixture")
    snapshot.add_argument("--case-id", required=True)
    snapshot.add_argument("--warmup-step", type=int, required=True)
    snapshot.add_argument("--seed-index", type=int, default=0)
    snapshot.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run-model")
    run.add_argument("--kind", choices=("axis", "raw_lora", "folded_dense"), required=True)
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--same-process-replay", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--label", required=True)
    pair = sub.add_parser("run-real-pair")
    pair.add_argument("--kind", choices=("axis", "raw_lora", "folded_dense"), required=True)
    pair.add_argument("--base-fixture", type=Path, required=True)
    pair.add_argument("--left-agentview", type=Path, required=True)
    pair.add_argument("--right-agentview", type=Path, required=True)
    pair.add_argument("--left-wrist", type=Path, required=True)
    pair.add_argument("--right-wrist", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.command == "build-fixtures":
        print(json.dumps(build_fixtures(args.output), ensure_ascii=False, indent=2))
    elif args.command == "build-snapshot-fixture":
        print(json.dumps(build_snapshot_fixture(args.case_id, args.warmup_step, args.output, seed_index=args.seed_index), ensure_ascii=False, indent=2))
    elif args.command == "run-model":
        print(json.dumps(run_model(args.kind, args.fixtures, args.output, same_process_replay=args.same_process_replay), ensure_ascii=False, indent=2))
    elif args.command == "compare":
        print(json.dumps(compare_outputs(args.first, args.second, args.output, label=args.label), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(run_real_pair(args.kind, args.base_fixture, args.left_agentview, args.right_agentview, args.left_wrist, args.right_wrist, args.output, label=args.label), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
