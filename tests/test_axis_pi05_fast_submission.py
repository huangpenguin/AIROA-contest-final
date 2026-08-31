from __future__ import annotations

from pathlib import Path

import pytest

from tools.axis_pi05.audit_fast_submission import _policy_static_audit
from tools.axis_pi05.export_fast_submission_weights import (
    R6_MODEL_SHA256,
    R6_NORM_SHA256,
    inventory_from_model_dir,
)


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging"


def test_staged_policy_is_fast_only_and_syntax_valid():
    result = _policy_static_audit(STAGING / "policy_server.py")
    assert result["passed"], result


def test_r6_identity_is_not_the_legacy_f2_lock():
    assert R6_MODEL_SHA256 != "d8a4c67c56b61bf8ca9387ab711d9abb9c2fe480862cf321672cbf59d5c96f00"
    assert R6_NORM_SHA256 != "d895707b3435bb116313b19a1d9f0e462268f04c929f089066dc443e501d2fed"
    assert R6_NORM_SHA256 == "4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5"


def test_r6_staging_rejects_a_non_r6_source_hash(tmp_path):
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
