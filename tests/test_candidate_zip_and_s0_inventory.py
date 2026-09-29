from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.axis_pi05.audit_fast_submission import _validate_allowed_gpu
from tools.axis_pi05.export_fast_submission_weights import (
    R6_MODEL_SHA256,
    inventory_from_model_dir,
)
from tools.axis_pi05.s0_storage_release_inventory import classify


ROOT = Path(__file__).resolve().parents[1]


def test_r6_staging_rejects_a_non_r6_source_hash_by_default(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        (source / name).write_bytes(b"legacy-f2-m1")
    (source / "config.json").write_text(
        json.dumps({"type": "pi05", "chunk_size": 10, "n_action_steps": 8, "normalization_mapping": {"ACTION": "QUANTILES", "STATE": "QUANTILES"}}),
        encoding="utf-8",
    )
    template = tmp_path / "policy_server.py"
    template.write_text("# template\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    with pytest.raises(ValueError, match="R6 folded source model hash"):
        inventory_from_model_dir(
            source,
            tmp_path / "staging",
            tmp_path / "report.json",
            server_template=template,
            vendor_source=vendor,
        )


def test_candidate_identity_accepts_matching_hashes_and_does_not_use_r6_lock(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    blobs = {
        "model.safetensors": b"candidate-model",
        "config.json": json.dumps(
            {"type": "pi05", "chunk_size": 10, "n_action_steps": 8, "normalization_mapping": {"ACTION": "QUANTILES", "STATE": "QUANTILES"}}
        ).encode(),
        "policy_preprocessor.json": b"pre",
        "policy_postprocessor.json": b"post",
        "policy_preprocessor_step_2_normalizer_processor.safetensors": b"pre-t",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors": b"post-t",
    }
    for name, payload in blobs.items():
        (source / name).write_bytes(payload)
    import hashlib

    identity = {
        "source_kind": "on_disk_t1_p20_canary100_folded_pytorch",
        "candidate": "t1_p20_canary100",
        "topology": "L2_action_lora_heads",
        "training_steps": 100,
        "model_sha256": hashlib.sha256(blobs["model.safetensors"]).hexdigest(),
        "norm_stats_sha256": "f49d37dac32533048a14039570ab99ef2e8fd693f0081544807e72a7ceffa50c",
        "processor_sha256": {
            name: hashlib.sha256(blobs[name]).hexdigest()
            for name in (
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "policy_preprocessor_step_2_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
        },
        "lock_mode": "candidate_locked",
    }
    assert identity["model_sha256"] != R6_MODEL_SHA256
    template = tmp_path / "policy_server.py"
    template.write_text("# template\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    report = inventory_from_model_dir(
        source,
        tmp_path / "staging",
        tmp_path / "report.json",
        server_template=template,
        vendor_source=vendor,
        identity=identity,
    )
    assert report["candidate"] == "t1_p20_canary100"
    assert report["lock_mode"] == "candidate_locked"
    assert report["source_model_sha256"] == identity["model_sha256"]
    assert report["safety"]["source_is_current_r6_on_disk_model"] is False


def test_gpu3_is_allowed_and_gpu0_is_not():
    _validate_allowed_gpu("3")
    _validate_allowed_gpu("1")
    with pytest.raises(ValueError, match="GPU0"):
        _validate_allowed_gpu("0")


def test_s0_never_cleanup_roles():
    reports = ROOT / "data/reports"
    assert classify(reports, "reports") == "immutable_keep"
    assert classify(ROOT / "data/manifests_track12", "manifests") == "immutable_keep"
    assert classify(ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2", "fast400_raw") == "immutable_keep"
    assert classify(ROOT / "artifacts/submission_axis_r6_fallback_revalidated_20260906.zip", "r6_fallback_zip") == "keep_local"
    assert classify(ROOT / "outputs/axis_pi05/official_canary100/t1_p20_canary100/100", "canary100") == "keep_local"
    assert classify(ROOT / "artifacts/submission_axis_f2_m1_500_20260826.zip", "old_f2_m1_zip") == "archive_then_delete_source"
    assert classify(ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3/0", "q5_checkpoint") == "archive_then_delete_source"


def test_t3_zip_norm_lock_uses_checkpoint_assets_dump():
    from tools.axis_pi05.export_candidate_fast_zip import TRACKS

    mix_source = "faf3f20f04756c8f85eeea7b72b893cd6a712d37d91cb5c0144004323cc994ab"
    checkpoint_dump = "50b6947a6000f504236f78c3616ad8541145b56b5810553ace9db2007016feff"
    assert TRACKS["T3"]["norm_lock"] == checkpoint_dump
    assert TRACKS["T3"]["eval_source_norm_sha256"] == mix_source
    assert TRACKS["T3"]["norm_lock"] != mix_source


def test_folded_complete_requires_every_processor_file(tmp_path):
    from tools.axis_pi05.export_candidate_fast_zip import FOLDED_REQUIRED, folded_complete

    missing = tmp_path / "folded"
    missing.mkdir()
    for name in FOLDED_REQUIRED[:-1]:
        (missing / name).write_bytes(b"x")
    assert folded_complete(missing) is False
    (missing / FOLDED_REQUIRED[-1]).write_bytes(b"x")
    assert folded_complete(missing) is True


def test_e2e_jax_checkpoint_default_is_r6():
    from tools.axis_pi05.e2e_action_parity import R6_JAX_CHECKPOINT

    assert R6_JAX_CHECKPOINT.name == "500"
    assert "r6_dual_l2_fresh500_r1" in str(R6_JAX_CHECKPOINT)


def test_purge_vendor_sys_path_removes_only_that_entry(tmp_path, monkeypatch):
    from tools.axis_pi05.audit_fast_submission import purge_vendor_sys_path

    vendor = tmp_path / "vendor"
    vendor.mkdir()
    other = str(tmp_path / "other")
    monkeypatch.setattr("sys.path", [str(vendor), other, str(vendor)])
    purge_vendor_sys_path(vendor)
    import sys

    assert str(vendor) not in sys.path
    assert other in sys.path
