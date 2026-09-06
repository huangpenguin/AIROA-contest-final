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


def test_m0_summary_is_fail_closed_and_machine_handoff_advances_without_losing_execution():
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
    assert handoff["next_plan"].endswith(
        "AXIS_PI05_TRACK3_TZ_EVIDENCE_AND_EXECUTOR_CONTRACT_REMEDIATION_PLAN.md"
    )
    assert (
        handoff["next_phase_only"]
        == "new_reviewed_plan_required"
    )
    autogen = handoff["track3_public_primitive_autogen_planning"]
    assert autogen["accepted_reverse_episode_count"] == 0
    assert autogen["superseded_plan_executed"] is False
    assert autogen["authorization"]["max_pending_autogen_canaries"] == 3
    assert autogen["authorization"]["human_teleop"] is False
    assert autogen["authorization"]["lora_or_dense_training"] is False
    assert autogen["authorization"]["official_gpu"] is False
    assert handoff["tests"]["passed"] >= 158
    assert execution["a2_fraction_090"]["optimizer_updates"] == 0
    assert execution["a3_fraction_095"]["optimizer_updates"] == 0
    audit = handoff["dense_fullft_m2_donation_audit"]
    assert audit["status"] == "remediation_required"
    assert audit["finding"]["compute_completed_indices"] == [0, 1]
    assert audit["finding"]["failed_accumulation_index"] == 1
    assert audit["finding"]["first_unattempted_compute_index"] == 2
    assert audit["blackwell_authorized"] is False
    interstep = handoff["dense_fullft_interstep_lifetime_audit"]
    assert interstep["status"] == "remediation_required"
    assert interstep["d4_semantic_correction"]["completed_update_transition"] == [1, 2]
    assert interstep["d4_semantic_correction"]["failed_optimizer_step_before"] == 2
    assert interstep["d4_semantic_correction"]["failed_compute_microbatch_index"] == 1
    assert interstep["official_gpu_authorized"] is False
    redirect = handoff["interstep_blocked_track3_redirection_audit"]
    assert redirect["status"] == "blocked_dense_only_project_redirected"
    assert redirect["dense_interstep"]["l1_cpu_code_gate_passed"] is True
    assert redirect["dense_interstep"]["three_gpu_is_exact_recipe_equivalent"] is False
    assert redirect["track3_public_rule"]["hidden_task_details_or_counts_used"] is False
    assert redirect["resource_route"]["training_gpu_ids"] == [1, 2]
    assert redirect["resource_route"]["gpu0_authorized"] is False
    track3 = handoff["track3_t1"]
    assert track3["status"] == "track3_reverse_rules_or_pilot_blocked"
    assert track3["pilot"]["accepted_episode_count"] == 0
    terminal_audit = handoff["track3_t1_terminal_source_audit"]
    assert terminal_audit["status"] == "blocked_old_pilot_public_terminal_recovery_authorized"
    assert terminal_audit["diagnosis"]["lerobot_8d_state_is_full_simulator_state"] is False
    assert terminal_audit["diagnosis"]["reverse_supervision_hypothesis_falsified"] is False
    assert terminal_audit["authorized_terminal_sources"] == [
        "official_hdf5_terminal_state",
        "live_public_forward_success_capture",
    ]
    assert terminal_audit["pilot_gate"]["fresh_process_replay"] == "3/3"
    assert terminal_audit["resource_boundaries"]["gpu0_authorized"] is False
    assert terminal_audit["training_fast_zip_upload_submission_authorized"] is False
    assert (ROOT / handoff["next_plan"]).is_file()
    autogen_audit = handoff["track3_public_autogen_s0_s5_audit"]
    assert autogen_audit["status"] == "remediation_required"
    assert autogen_audit["findings"]["s1_reported_identity_sum"] == 22156
    assert autogen_audit["findings"]["s1_episode_count"] == 21961
    assert autogen_audit["findings"]["s3_dynamic_three_of_three_executed"] is False
    assert autogen_audit["findings"]["support_executor_controls_orientation"] is False
    assert (ROOT / autogen_audit["archived_plan"]).is_file()
    recovery = handoff["track3_support_autogen_se3_recovery_authorization"]
    assert recovery["authorization"]["support_canary_max_attempts"] == 1
    assert recovery["authorization"]["human_teleop"] is False
    assert recovery["authorization"]["containment_articulation"] is False
    assert recovery["authorization"]["lora_or_dense_training"] is False
    assert recovery["authorization"]["official_gpu"] is False
    assert (ROOT / terminal_audit["archived_plan"]).is_file()
    recorder_audit = handoff["track3_p0_p2_recorder_readiness_audit"]
    assert recorder_audit["status"] == "remediation_required"
    assert recorder_audit["preserved_results"]["accepted_demonstration_count"] == 0
    assert recorder_audit["corrections"]["p0_lock_actual_sha256"] == (
        "be12795b3d18d46f774022d817789a14d64a5e16c8e27bee58e256c29acb4ab5"
    )
    assert recorder_audit["recorder_findings"]["no_key_preserves_gripper_contract"] is False
    assert recorder_audit["recorder_findings"]["live_visual_preview_implemented"] is False
    assert recorder_audit["human_input_authorized"] is False
    assert recorder_audit["planner_pilot_training_fast_zip_authorized"] is False
    assert (ROOT / recorder_audit["archived_plan"]).is_file()
    readiness = handoff["track3_teleop_r0_r4_readiness_audit"]
    assert readiness["status"] == "approved_task19_only"
    assert readiness["approved_evidence"]["r4_gpu3_task_order"] == [19, 10, 20]
    assert readiness["approved_evidence"]["accepted_count"] == 0
    assert readiness["residual_risk"]["task19_is_first_true_end_to_end_canary"] is True
    assert readiness["authorization"]["task19_single_human_pending_candidate"] is True
    assert readiness["authorization"]["accepted_promotion"] is False
    assert readiness["authorization"]["task10_task20_recording"] is False
    assert (ROOT / readiness["archived_plan"]).is_file()
