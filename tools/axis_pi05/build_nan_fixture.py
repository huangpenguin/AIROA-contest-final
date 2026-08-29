"""Build the fixed 64-sample replay fixture used by the NaN matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHONPATH = str(ROOT / "third_party/axis_openpi/src") + os.pathsep + str(ROOT)
sys.path[:0] = [str(ROOT / "third_party/axis_openpi/src"), str(ROOT)]


def _rows(metadata: dict[str, Any], count: int) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        row = {}
        for key, value in metadata.items():
            array = np.asarray(value)
            item = array[index] if array.ndim > 0 and array.shape[0] == count else value
            if isinstance(item, np.generic):
                item = item.item()
            elif isinstance(item, np.ndarray):
                item = item.tolist()
            row[str(key)] = item
        result.append(row)
    return result


def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "finite": bool(np.isfinite(array).all()) if array.dtype.kind in "biufc" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/reports/axis_libero_30k/nan_localization/fixed_replay",
    )
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.count != 64:
        raise ValueError("the locked NaN matrix requires exactly 64 samples")

    # These must be set before importing JAX/OpenPI.  The fixture builder is a
    # CPU-only process and therefore cannot reserve any GPU memory.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.update(
        {
            "PYTHONPATH": PYTHONPATH,
            "AXIS_PI05_AUDIT_METADATA": "1",
            "AXIS_PI05_LORA_MICROBATCH": "1",
            "AXIS_PI05_LORA_ACCUM": "1",
            "AXIS_PI05_LORA_EFFECTIVE_BATCH": "1",
            "AXIS_PI05_LORA_ARM": "L2_action_lora_heads",
            "HF_HUB_OFFLINE": "1",
        }
    )

    import jax

    from openpi.training import config as config_lib
    from openpi.training import data_loader
    from openpi.training import sharding

    config = config_lib.get_config("axis_pi05_lora_screen")
    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False, batch_size=1)
    iterator = iter(loader)
    samples = []
    metadata_rows = []
    for _ in range(args.count):
        loaded = next(iterator)
        if not (isinstance(loaded, tuple) and len(loaded) == 3):
            raise RuntimeError("audit metadata was not returned by the transformed data loader")
        observation, actions, metadata = loaded
        samples.append((observation, actions))
        metadata_rows.extend(_rows(metadata, 1))
    if len(metadata_rows) != args.count:
        raise RuntimeError(f"expected {args.count} metadata rows, got {len(metadata_rows)}")

    def stack(path: str) -> np.ndarray:
        values = []
        for observation, actions in samples:
            if path == "actions":
                value = actions
            else:
                value: Any = observation.to_dict()
                for part in path.split("/"):
                    value = value[part]
            values.append(np.asarray(value)[0])
        return np.stack(values, axis=0)

    arrays = {
        "observation__image__base_0_rgb": stack("image/base_0_rgb"),
        "observation__image__left_wrist_0_rgb": stack("image/left_wrist_0_rgb"),
        "observation__image__right_wrist_0_rgb": stack("image/right_wrist_0_rgb"),
        "observation__image_mask__base_0_rgb": stack("image_mask/base_0_rgb"),
        "observation__image_mask__left_wrist_0_rgb": stack("image_mask/left_wrist_0_rgb"),
        "observation__image_mask__right_wrist_0_rgb": stack("image_mask/right_wrist_0_rgb"),
        "observation__state": stack("state"),
        "observation__action_is_pad": stack("action_is_pad"),
        "observation__tokenized_prompt": stack("tokenized_prompt"),
        "observation__tokenized_prompt_mask": stack("tokenized_prompt_mask"),
        "actions": stack("actions"),
    }

    # The override is generated once per sample from a stable JAX key.  It is
    # stored explicitly so single- and dual-device runs consume identical
    # model inputs despite different microbatch grouping.
    base_key = jax.random.key(args.seed)
    noises = []
    times = []
    rng_rows = []
    for sample_index in range(args.count):
        sample_key = jax.random.fold_in(base_key, sample_index)
        _, noise_key, time_key = jax.random.split(sample_key, 3)
        noises.append(np.asarray(jax.random.normal(noise_key, (10, 32)), dtype=np.float32))
        times.append(float(jax.random.beta(time_key, 1.5, 1, ()).item() * 0.999 + 0.001))
        rng_rows.append(
            {
                "sample_index": sample_index,
                "sample_key_data": np.asarray(jax.random.key_data(sample_key)).tolist(),
                "noise_key_data": np.asarray(jax.random.key_data(noise_key)).tolist(),
                "time_key_data": np.asarray(jax.random.key_data(time_key)).tolist(),
            }
        )
    arrays["rng__noise"] = np.stack(noises, axis=0)
    arrays["rng__time"] = np.asarray(times, dtype=np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    replay_path = args.output / "fixed_replay.npz"
    np.savez_compressed(replay_path, **arrays)
    config_hash = hashlib.sha256(
        json.dumps(__import__("dataclasses").asdict(config), sort_keys=True, default=str).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "sample_count": args.count,
        "source": "physical-intelligence/libero",
        "source_revision": os.environ.get(
            "AXIS_PI05_CANONICAL_REVISION", "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
        ),
        "sampling": "shuffle_false_first_64_transformed_samples",
        "transform_config_hash": config_hash,
        "metadata": metadata_rows,
        "rng": {"seed": args.seed, "per_sample": rng_rows},
        "arrays": {key: _fingerprint(value) for key, value in arrays.items()},
        "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
    }
    (args.output / "fixed_replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "passed", "samples": args.count, "replay": str(replay_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
