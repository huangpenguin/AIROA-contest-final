from __future__ import annotations

import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/axis_openpi/src"))

from openpi.models import lora  # noqa: E402
from openpi.training import axis_lora  # noqa: E402
from openpi.training import nan_replay  # noqa: E402


def _einsum_params():
    module = lora.Einsum(
        shape=(4, 5),
        init_fn=nn.initializers.lecun_normal(),
        lora_config=lora.LoRAConfig(rank=2),
    )
    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    variables = module.init(jax.random.key(0), "BD,DO->BO", x)
    return module, x, variables["params"]


def test_lora_is_random_a_zero_b_and_dense_zero_delta():
    module, x, params = _einsum_params()
    assert float(jnp.linalg.norm(params["lora_a"])) > 0
    assert jnp.all(params["lora_b"] == 0)
    dense = params["w"]
    folded = lora.fold_lora_weight(dense, params["lora_a"], params["lora_b"])
    assert jnp.array_equal(folded, dense)
    assert jnp.array_equal(module.apply({"params": params}, "BD,DO->BO", x), module.apply({"params": params}, "BD,DO->BO", x))


def test_step_one_b_gradient_is_finite_and_nonzero():
    module, x, params = _einsum_params()

    def objective(p):
        return jnp.sum(module.apply({"params": p}, "BD,DO->BO", x) ** 2)

    grads = jax.grad(objective)(params)
    assert float(jnp.linalg.norm(grads["lora_b"])) > 0
    assert jnp.all(jnp.isfinite(grads["lora_b"]))
    assert jnp.all(jnp.isfinite(grads["lora_a"]))


def test_attention_and_ffn_fold_parity():
    key = jax.random.key(3)
    attention = lora.Einsum(
        shape=(3, 4, 5, 6),
        init_fn=nn.initializers.lecun_normal(),
        lora_config=lora.LoRAConfig(rank=2),
    )
    x = jax.random.normal(key, (2, 7, 5))
    attention_params = attention.init(key, "BSD,3KDH->3BSKH", x)["params"]
    # B=0 makes the raw/folded attention paths exactly equal at initialization;
    # exercise the general fold formula with nonzero factors too.
    attention_params = dict(attention_params, lora_b=jnp.ones_like(attention_params["lora_b"]) * 0.02)
    raw = attention.apply({"params": attention_params}, "BSD,3KDH->3BSKH", x)
    folded_params = {"w": lora.fold_lora_weight(
        attention_params["w"], attention_params["lora_a"], attention_params["lora_b"], scaling=0.5
    )}
    dense = lora.Einsum(shape=(3, 4, 5, 6), init_fn=nn.initializers.lecun_normal())
    folded = dense.apply({"params": folded_params}, "BSD,3KDH->3BSKH", x)
    assert jnp.allclose(raw, folded, rtol=2e-5, atol=2e-5)

    ffn = lora.FeedForward(features=5, hidden_dim=8, lora_config=lora.LoRAConfig(rank=2))
    ffn_params = ffn.init(key, x)["params"]
    ffn_params = dict(ffn_params, gating_einsum_lora_b=jnp.ones_like(ffn_params["gating_einsum_lora_b"]) * 0.01)
    ffn_params = dict(ffn_params, linear_lora_b=jnp.ones_like(ffn_params["linear_lora_b"]) * 0.01)
    raw_ffn = ffn.apply({"params": ffn_params}, x)
    plain = lora.FeedForward(features=5, hidden_dim=8)
    plain_params = {
        "gating_einsum": lora.fold_lora_weight(
                ffn_params["gating_einsum"], ffn_params["gating_einsum_lora_a"], ffn_params["gating_einsum_lora_b"]
        ),
        "linear": lora.fold_lora_weight(
            ffn_params["linear"], ffn_params["linear_lora_a"], ffn_params["linear_lora_b"]
        ),
    }
    folded_ffn = plain.apply({"params": plain_params}, x)
    assert jnp.allclose(raw_ffn, folded_ffn, rtol=2e-5, atol=2e-5)


def test_exact_scope_does_not_use_broad_expert_regex():
    paths = {
        "action_attn": ("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "lora_a"),
        "action_mlp": ("PaliGemma", "llm", "layers", "mlp_1", "linear_lora_b"),
        "pali_attn": ("PaliGemma", "llm", "layers", "attn", "q_einsum", "lora_a"),
        "pali_mlp": ("PaliGemma", "llm", "layers", "mlp", "linear_lora_b"),
        "action_in": ("action_in_proj", "kernel"),
        "time_out": ("time_mlp_out", "kernel"),
        "vision": ("PaliGemma", "img", "kernel"),
    }
    assert axis_lora.is_declared_trainable(paths["action_attn"], "L1_action_lora")
    assert axis_lora.is_declared_trainable(paths["action_mlp"], "L1_action_lora")
    assert not axis_lora.is_declared_trainable(paths["pali_attn"], "L1_action_lora")
    assert axis_lora.is_declared_trainable(paths["pali_attn"], "L3_dual_lora_heads")
    assert axis_lora.is_declared_trainable(paths["action_in"], "L2_action_lora_heads")
    assert axis_lora.is_declared_trainable(paths["time_out"], "L2_action_lora_heads")
    assert not axis_lora.is_declared_trainable(paths["vision"], "L3_dual_lora_heads")


def test_loss_masks_use_actual_denominator_and_all_pad_is_safe():
    squared = jnp.ones((2, 4, 32), dtype=jnp.float32)
    pad = jnp.asarray([[False, False, True, True], [False, False, False, False]])
    valid7 = lora.masked_loss_components(squared, pad, objective="valid7_temporal")
    padded32 = lora.masked_loss_components(squared, pad, objective="padded32_temporal")
    assert int(valid7["valid_action_elements"]) == 42
    assert int(padded32["valid_action_elements"]) == 192
    assert float(valid7["loss"]) == pytest.approx(1.0)
    assert float(padded32["loss"]) == pytest.approx(1.0)
    assert float(valid7["padded25_loss_mass"]) == pytest.approx(150.0)
    all_pad = lora.masked_loss_components(squared, jnp.ones((2, 4), dtype=bool), objective="valid7_temporal")
    assert int(all_pad["valid_action_elements"]) == 0
    assert float(all_pad["loss"]) == 0.0


def test_accumulation_averages_before_one_clip_and_one_optimizer_step():
    params = jnp.asarray([1.0, -2.0], dtype=jnp.float32)
    grads = (jnp.asarray([2.0, 0.0]), jnp.asarray([0.0, 4.0]))
    summed = sum(grads)
    averaged = summed / 2.0
    tx = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(1.0))
    state = tx.init(params)
    updates, new_state = tx.update(averaged, state, params)
    expected_updates, _ = tx.update(jnp.asarray([1.0, 2.0]), state, params)
    assert jnp.allclose(updates, expected_updates)
    assert int(new_state[1][0].count) == 1


def test_rng_restore_formula_and_v100_bfloat16_rejection():
    key = jax.random.key(11)
    assert jnp.array_equal(jax.random.fold_in(jax.random.fold_in(key, 7), 3), jax.random.fold_in(jax.random.fold_in(key, 7), 3))
    with pytest.raises(ValueError, match="BF16"):
        axis_lora.assert_no_bfloat16({"x": jnp.ones((2,), dtype=jnp.bfloat16)})


def test_nan_replay_fingerprints_capture_nonfinite_inputs():
    value = np.asarray([1.0, np.nan, np.inf], dtype=np.float32)
    fingerprint = nan_replay.array_fingerprint(value)
    assert fingerprint["shape"] == [3]
    assert fingerprint["finite"] is False
    assert fingerprint["nonfinite_count"] == 2
    assert nan_replay.tree_fingerprints({"actions": value})["actions"]["sha256"] == fingerprint["sha256"]
