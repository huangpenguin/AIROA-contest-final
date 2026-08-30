from __future__ import annotations

from tools.axis_pi05.r6_release_lock import RAW_Q2_MAX_ABS_DIFF, RAW_Q2_STATUS, validate_lock_record


def _record() -> dict:
    return {
        "research_raw_lora_q2_status": "failed_preserved",
        "research_raw_lora_max_abs_diff": RAW_Q2_MAX_ABS_DIFF,
        "research_raw_lora_q2": {"status": RAW_Q2_STATUS},
        "deployment_candidate": "r6_step500_lerobot_folded",
        "deployment_runtime_status": "not_yet_tested",
        "formal_winner": False,
        "auto_submit_authorized": False,
        "folded_q2_reference": {"backend": "folded_dense_jax"},
        "deployment_source": {
            "backend": "lerobot_folded_pytorch",
            "model_path": "/r6/model.safetensors",
            "model_sha256": "a" * 64,
        },
        "normalization": {"ACTION": "QUANTILES", "STATE": "QUANTILES"},
    }


def test_release_lock_keeps_jax_reference_and_lerobot_deployment_distinct():
    assert validate_lock_record(_record()) == []


def test_release_lock_rejects_unverified_or_legacy_deployment():
    record = _record()
    record["deployment_runtime_status"] = "passed"
    assert any("runtime" in error for error in validate_lock_record(record))
    record = _record()
    record["deployment_source"]["model_path"] = "/old/f2_m1/model.safetensors"
    assert any("legacy" in error for error in validate_lock_record(record))


def test_release_lock_preserves_raw_lora_failure_even_if_deployment_changes():
    record = _record()
    record["research_raw_lora_q2_status"] = "passed"
    assert any("not preserved" in error for error in validate_lock_record(record))
