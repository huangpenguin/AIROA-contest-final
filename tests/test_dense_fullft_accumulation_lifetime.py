from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from tools.axis_pi05 import dense_fullft_accumulation_memory_lifetime_recovery as recovery  # noqa: E402


TRAIN_SOURCE = ROOT / "third_party/axis_openpi/scripts/train.py"
_SPEC = importlib.util.spec_from_file_location("axis_train_accumulation_lifetime_test", TRAIN_SOURCE)
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


def test_sixteen_fp32_microbatches_preserve_ordered_sum_and_single_update():
    config = SimpleNamespace(effective_global_batch=64)
    state = SimpleNamespace(step=jnp.asarray(0, dtype=jnp.int32), rng=jax.random.key(17))
    microbatches = tuple(_batch() for _ in range(16))
    seen_rng_keys = []
    received_gradients = []

    def microbatch_grad(micro_rng, _state, _batch_value, rng_fixture):
        assert rng_fixture is None
        index = len(seen_rng_keys)
        seen_rng_keys.append(tuple(np.asarray(jax.random.key_data(micro_rng)).tolist()))
        value = float(index + 1)
        return jnp.asarray(value, dtype=jnp.float32), {"g": jnp.asarray([value], dtype=jnp.float32)}, _components(value)

    def apply_update(old_state, grads, _next_rng):
        received_gradients.append(np.asarray(grads["g"]))
        return SimpleNamespace(step=old_state.step + 1), {"update_marker": jnp.asarray(1, dtype=jnp.int32)}

    new_state, info = _TRAIN.train_step_accumulated_host(
        config,
        jax.random.key(23),
        state,
        microbatches,
        microbatch_grad,
        apply_update,
    )

    expected = np.asarray([(sum(range(1, 17)) / 16)], dtype=np.float32)
    assert np.array_equal(received_gradients[0], expected)
    assert int(new_state.step) == 1
    assert int(np.asarray(info["micro_steps"])) == 16
    assert int(np.asarray(info["examples_seen"])) == 64
    assert float(np.asarray(info["loss"])) == float(expected[0])
    assert len(seen_rng_keys) == 16
    assert len(set(seen_rng_keys)) == 16


def test_nonfinite_gate_reports_one_value_without_gradient_container():
    finite_components = {"valid_action_elements": jnp.asarray(1, dtype=jnp.int32)}
    assert (
        _TRAIN._first_nonfinite_microbatch_name(
            jnp.asarray(1.0, dtype=jnp.float32),
            finite_components,
            {"g": jnp.asarray([1.0], dtype=jnp.float32)},
        )
        is None
    )
    assert (
        _TRAIN._first_nonfinite_microbatch_name(
            jnp.asarray(1.0, dtype=jnp.float32),
            {"valid_action_elements": jnp.asarray(np.nan, dtype=jnp.float32)},
            {"g": jnp.asarray([1.0], dtype=jnp.float32)},
        )
        == "valid_action_elements"
    )
    assert (
        _TRAIN._first_nonfinite_microbatch_name(
            jnp.asarray(1.0, dtype=jnp.float32),
            finite_components,
            {"g": jnp.asarray([np.nan], dtype=jnp.float32)},
        )
        == "grad:g"
    )


def test_accumulation_source_has_explicit_lifetime_boundaries():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    block = source[source.index("def train_step_accumulated_host"):source.index("\ndef main", source.index("def train_step_accumulated_host"))]
    assert "values = {" not in block
    assert "def _first_nonfinite_microbatch_name" in source
    assert "jax.block_until_ready(next_grad_sum)" in block
    assert "del grads, loss_sum, components" in block
    assert '"microbatch_compute_complete"' in block
    assert '"accumulation_complete"' in block
    assert '"temporaries_released"' in block


def test_memory_plateau_gate_requires_all_microbatch_boundaries():
    rows = []
    for index in range(16):
        devices = [{"memory_stats_available": True, "memory_stats": {"bytes_in_use": 10_000_000 + index}}]
        for phase in ("microbatch_compute_complete", "accumulation_complete", "temporaries_released"):
            rows.append(
                {
                    "phase": phase,
                    "optimizer_step_before": 0,
                    "micro_index": index,
                    "devices": devices,
                }
            )
    result = recovery._plateau_for_rows(rows, expected_count=16)
    assert result["passed"] is True
    assert result["microbatch_compute_complete_indexes"] == list(range(16))
    assert result["temporaries_released_indexes"] == list(range(16))
    assert result["max_absolute_drift_bytes_per_card"] <= recovery.PLATEAU_LIMIT_BYTES

    missing = [row for row in rows if not (row["phase"] == "temporaries_released" and row["micro_index"] == 7)]
    assert recovery._plateau_for_rows(missing, expected_count=16)["passed"] is False


def test_m0_summary_is_fail_closed_and_machine_handoff_matches_execution():
    summary_path = ROOT / "data/reports/axis_libero_30k/dense_fullft_allocator_fraction_recovery/summary.json"
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    for key in ("a2_fraction_090", "a3_fraction_095"):
        run = summary[key]
        assert run["status"] == "allocator_bfc_oom_after_microbatch_1_before_microbatch_2"
        assert run["completed_microbatch_indexes"] == [0, 1]
        assert run["oom_micro_index"] == 2
        assert run["optimizer_updates"] == 0
    assert summary["optimizer_updates"] == 0
    assert summary["tests"]["passed"] == 132
    assert summary["next_action"].startswith("M2 ")

    handoff_path = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
    handoff = __import__("json").loads(handoff_path.read_text(encoding="utf-8"))
    execution = handoff["dense_fullft_allocator_fraction_recovery_execution"]
    assert handoff["next_plan"].endswith("AXIS_PI05_DENSE_FULLFT_ACCUMULATION_MEMORY_LIFETIME_RECOVERY_PLAN.md")
    assert handoff["next_phase_only"] == "accumulation_memory_lifetime_M0_to_M3_max_10_steps"
    assert handoff["tests"]["passed"] == 137
    assert execution["a2_fraction_090"]["optimizer_updates"] == 0
    assert execution["a3_fraction_095"]["optimizer_updates"] == 0
