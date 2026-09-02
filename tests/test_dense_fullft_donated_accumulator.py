from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "third_party/axis_openpi/scripts/train.py"
_SPEC = importlib.util.spec_from_file_location("axis_train_donated_accumulator_test", TRAIN_SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_TRAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TRAIN)


def _batch(batch_size: int = 4):
    return (None, np.zeros((batch_size, 1), dtype=np.float32))


def _components(value: float):
    return {
        "valid_action_elements": jnp.asarray(1, dtype=jnp.int32),
        "seven_d_loss_mass": jnp.asarray(value, dtype=jnp.float32),
        "padded25_loss_mass": jnp.asarray(0.0, dtype=jnp.float32),
        "temporal_padding_loss_mass": jnp.asarray(0.0, dtype=jnp.float32),
    }


def test_donated_tree_add_is_fp32_left_associative_and_invalidates_input():
    left = {
        "a": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        "nested": {"b": jnp.asarray([[3.0]], dtype=jnp.float32)},
    }
    right = {
        "a": jnp.asarray([4.0, 5.0], dtype=jnp.float32),
        "nested": {"b": jnp.asarray([[6.0]], dtype=jnp.float32)},
    }
    result = _TRAIN._add_gradients(left, right)
    jax.block_until_ready(result)
    assert result["a"].dtype == jnp.float32
    assert result["nested"]["b"].dtype == jnp.float32
    assert np.array_equal(np.asarray(result["a"]), np.asarray([5.0, 7.0], dtype=np.float32))
    assert np.array_equal(np.asarray(result["nested"]["b"]), np.asarray([[9.0]], dtype=np.float32))
    assert left["a"].is_deleted()


def test_donated_tree_add_exposes_positive_alias_memory_analysis():
    left = {"a": jnp.ones((8,), dtype=jnp.float32)}
    right = {"a": jnp.ones((8,), dtype=jnp.float32)}
    compiled = _TRAIN._DONATED_GRADIENT_ADD.lower(left, right).compile()
    analysis = compiled.memory_analysis()
    assert int(analysis.alias_size_in_bytes) >= left["a"].size * left["a"].dtype.itemsize


def test_merge_only_uses_two_microbatches_without_optimizer_update():
    config = SimpleNamespace(effective_global_batch=64)
    state = SimpleNamespace(step=jnp.asarray(0, dtype=jnp.int32), rng=jax.random.key(29))
    seen = []
    updates = []

    def microbatch_grad(micro_rng, _state, _batch_value, rng_fixture):
        assert rng_fixture is None
        value = float(len(seen) + 1)
        seen.append(tuple(np.asarray(jax.random.key_data(micro_rng)).tolist()))
        return jnp.asarray(value, dtype=jnp.float32), {"g": jnp.asarray([value], dtype=jnp.float32)}, _components(value)

    def apply_update(*_args):
        updates.append(True)
        raise AssertionError("merge-only canary must not call optimizer update")

    new_state, info = _TRAIN.train_step_accumulated_host(
        config,
        jax.random.key(31),
        state,
        (_batch(), _batch()),
        microbatch_grad,
        apply_update,
        merge_only=True,
    )
    assert len(seen) == 2
    assert not updates
    assert int(new_state.step) == 0
    assert int(np.asarray(info["micro_steps"])) == 2
    assert int(np.asarray(info["examples_seen"])) == 8


def test_transition_probe_accepts_one_microbatch_without_optimizer_update():
    config = SimpleNamespace(effective_global_batch=64)
    state = SimpleNamespace(step=jnp.asarray(2, dtype=jnp.int32), rng=jax.random.key(37))
    updates = []

    def microbatch_grad(_micro_rng, _state, _batch_value, rng_fixture):
        assert rng_fixture is None
        return (
            jnp.asarray(2.0, dtype=jnp.float32),
            {"g": jnp.asarray([2.0], dtype=jnp.float32)},
            _components(2.0),
        )

    def apply_update(*_args):
        updates.append(True)
        raise AssertionError("transition probe must not call optimizer update")

    new_state, info = _TRAIN.train_step_accumulated_host(
        config,
        jax.random.key(41),
        state,
        (_batch(),),
        microbatch_grad,
        apply_update,
        transition_probe=True,
    )
    assert not updates
    assert int(new_state.step) == 2
    assert info["optimizer_step"] == 2
    assert info["micro_steps"] == 1
    assert not any(hasattr(value, "device") for value in info.values())


def test_d1_source_contract_is_explicit():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert "_DONATED_GRADIENT_ADD = jax.jit(_add_gradients_impl, donate_argnums=(0,))" in source
    assert "donate_argnums=(0, 1)" in source
    assert '"donated_gradient_add"' in source
    assert '"merge_only_complete"' in source
    for marker in (
        '"optimizer_returned"',
        '"new_state_ready"',
        '"metrics_host_materialized"',
        '"device_metrics_released"',
        '"next_step_start"',
        '"transition_probe_complete"',
    ):
        assert marker in source
