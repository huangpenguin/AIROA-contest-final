from __future__ import annotations

import json

import pytest

from tools.axis_pi05 import run_track3_official_blackwell_lora as runner


def test_official_runner_uses_independent_namespace():
    assert "q5_local_gpu3" not in str(runner.DEFAULT_OUTPUT_BASE)
    assert "track12" not in str(runner.DEFAULT_OUTPUT_BASE).lower()
    assert runner.EXPECTED_R6_TREE_SHA


def test_strict_json_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"a","status":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.strict_json_load(path)


def test_release_proof_rejects_active_track12(tmp_path):
    path = tmp_path / "release.json"
    path.write_text(json.dumps({"status": "finished", "active": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="active trainer"):
        runner._release_proof(path)


def test_train_code_has_fresh_and_resume_modes():
    fresh = runner._train_code(resume=False, run_id="fresh")
    resumed = runner._train_code(resume=True, run_id="same")
    assert "resume=False" in fresh
    assert "resume=True" in resumed
    assert "axis_pi05_track3_q5" in fresh
    assert "original_r6_params_only_fresh_optimizer" in fresh
