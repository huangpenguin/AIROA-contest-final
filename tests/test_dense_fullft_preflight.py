from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.axis_pi05.dense_fullft_preflight import (  # noqa: E402
    EXPECTED_PARAM_NUMEL,
    _memory_budget,
    validate_config_contract,
)
from tools.axis_pi05.dense_fullft_allocator_fraction_recovery import (  # noqa: E402
    FORBIDDEN_ALLOCATOR_TOKENS,
    FRACTIONS,
    _validate_fraction_environment,
)


def test_dense_config_matches_canonical_recipe():
    report = validate_config_contract()
    assert report["status"] == "passed", report["mismatches"]
    assert report["config_name"] == "axis_pi05_libero_fullft"


def test_dense_memory_budget_preserves_recipe_and_shows_v100_boundary():
    report = _memory_budget(EXPECTED_PARAM_NUMEL)
    assert report["parameter_numel"] == 3_353_433_872
    assert report["recipe_preserved"] == {
        "ema_decay": 0.999,
        "effective_global_batch": 64,
        "precision": "float32",
        "freeze_filter": "nnx.Nothing",
    }
    assert report["topologies"]["2"]["theoretical_headroom_gib_per_card"] < 2.0
    assert report["topologies"]["4"]["theoretical_headroom_gib_per_card"] > 2.0


def test_dense_openpi_source_has_no_lora_variants_in_config():
    source = (ROOT / "third_party/axis_openpi/src/openpi/training/config.py").read_text(encoding="utf-8")
    start = source.index('name="axis_pi05_libero_fullft"')
    end = source.index('name="axis_pi05_lora_screen"', start)
    block = source[start:end]
    assert 'paligemma_variant="gemma_2b"' in block
    assert 'action_expert_variant="gemma_300m"' in block
    assert "freeze_filter=nnx.Nothing" in block
    assert "lora" not in block.lower()


def test_sharded_init_uses_parameter_subtree_sharding_and_no_donation_claim():
    source = (ROOT / "third_party/axis_openpi/scripts/train.py").read_text(encoding="utf-8")
    assert "def _partial_params_sharding(state_sharding: Any, partial_params: at.Params)" in source
    assert "jax.device_put(partial_params, partial_params_sharding)" in source
    assert "in_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), partial_params_sharding)" in source
    assert "donate_argnums=(1,)" not in source[source.index("def init_train_state"):source.index("def _loss_sum_and_aux")]


def test_four_cpu_toy_reconstructs_from_direct_parameter_shards():
    script = r'''
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from openpi.training import sharding

class Toy(nnx.Module):
    def __init__(self):
        self.large = nnx.Param(jnp.arange(4096 * 4096, dtype=jnp.float32).reshape(4096, 4096))
        self.scalar = nnx.Param(jnp.asarray(3.0, dtype=jnp.float32))

toy = Toy()
host = nnx.state(toy).to_pure_dict()
mesh = sharding.make_mesh(4)
target = sharding.fsdp_sharding(nnx.state(toy), mesh).to_pure_dict()
assert target["large"].spec[1] == "fsdp"
assert target["scalar"].spec == ()
sharded = jax.device_put(host, target)
assert sharded["large"].addressable_shards[0].data.shape == (4096, 1024)
reconstructed = np.asarray(jax.device_get(sharded["large"]))
assert np.array_equal(reconstructed, np.asarray(host["large"]))
assert reconstructed.dtype == np.float32
print(json.dumps({"devices": jax.device_count(), "large_spec": str(target["large"].spec)}))
'''
    environment = os.environ.copy()
    environment.update({
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": str(ROOT / "third_party/axis_openpi/src"),
    })
    result = subprocess.run(
        [str(ROOT / ".venv_axis_pi05/bin/python"), "-c", script],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"devices": 4' in result.stdout


def test_allocator_contract_accepts_only_preallocated_registered_fractions():
    environment = {
        "XLA_CLIENT_MEM_FRACTION": "0.90",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "JAX_PLATFORMS": "cuda",
    }
    _validate_fraction_environment(environment, "0.90")
    assert FRACTIONS == ("0.90", "0.95")
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in environment
    assert "cuda_malloc_async" in FORBIDDEN_ALLOCATOR_TOKENS


def test_allocator_contract_rejects_deprecated_fraction_and_forbidden_flags():
    deprecated = {
        "XLA_CLIENT_MEM_FRACTION": "0.90",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }
    try:
        _validate_fraction_environment(deprecated, "0.90")
    except ValueError as exc:
        assert "deprecated" in str(exc)
    else:
        raise AssertionError("deprecated fraction was accepted")

    forbidden = {
        "XLA_CLIENT_MEM_FRACTION": "0.95",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/tmp/cuda_malloc_async",
    }
    try:
        _validate_fraction_environment(forbidden, "0.95")
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("forbidden allocator flag was accepted")


def test_allocator_observability_is_host_side_and_does_not_change_train_math():
    source = (ROOT / "third_party/axis_openpi/scripts/train.py").read_text(encoding="utf-8")
    assert "def _record_allocator_stats" in source
    assert '"microbatch_complete"' in source
    assert '"optimizer_update_complete"' in source
    assert '"checkpoint_restore"' in source
    assert "memory_analysis" in source
    assert "donate_argnums=(1,)" not in source[source.index("def init_train_state"):source.index("def _loss_sum_and_aux")]


def test_step_zero_resume_requires_explicit_recovery_opt_in():
    source = (ROOT / "third_party/axis_openpi/src/openpi/training/checkpoints.py").read_text(encoding="utf-8")
    assert 'os.environ.get("AXIS_PI05_ALLOW_STEP0_RESUME") != "1"' in source


def test_restored_rng_is_replaced_on_replicated_mesh_before_fold_in():
    source = (ROOT / "third_party/axis_openpi/scripts/train.py").read_text(encoding="utf-8")
    assert "jax.device_put(train_state.rng, replicated_sharding)" in source
    assert "this changes placement, not math" in source
