from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import pytest

from tools.axis_pi05 import build_r6_execute4_official_trial as mod


def test_rewrite_policy_source_is_execute4_only() -> None:
    source = (mod.SOURCE_STAGING / "policy_server.py").read_text(encoding="utf-8")
    rewritten = mod.rewrite_policy_source(source)
    ast.parse(rewritten)
    assert 'n_action_steps.strip() != "4"' in rewritten
    assert "N_ACTION_STEPS must be 4" in rewritten
    assert "        n = 4\n" in rewritten
    assert "execute=4 " in rewritten
    assert "n_action_steps=4 action_ema=0.0 rtc=False" in rewritten
    assert 'n_action_steps.strip() != "8"' not in rewritten
    assert "execute=8 " not in rewritten


def test_rewrite_fails_closed_on_unexpected_source() -> None:
    with pytest.raises(ValueError):
        mod.rewrite_policy_source("def get_action():\n    pass\n")


def test_source_config_is_frozen_execute8() -> None:
    config = json.loads(
        (mod.SOURCE_STAGING / "model_weights/config.json").read_text(encoding="utf-8")
    )
    assert config["type"] == "pi05"
    assert config["chunk_size"] == 10
    assert config["n_action_steps"] == 8
    assert config["normalization_mapping"]["ACTION"] == "QUANTILES"
    assert config["normalization_mapping"]["STATE"] == "QUANTILES"


def test_paths_are_isolated_and_no_auto_submit() -> None:
    assert mod.STAGING != mod.SOURCE_STAGING
    assert mod.OUTPUT != mod.SOURCE_ZIP
    assert "execute4" in mod.OUTPUT.name
    script = Path(mod.__file__).read_text(encoding="utf-8")
    assert '"auto_upload": False' in script
    assert '"auto_submit": False' in script


def test_plan_records_failed_gate_and_user_override() -> None:
    plan = mod.PLAN.read_text(encoding="utf-8")
    assert "user risk override" in plan
    assert "+2" in plan
    assert "不把 I2 改写成通过" in plan
    assert "不得自动上传或提交" in plan


def _write_zip(path: Path, values: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in values.items():
            archive.writestr(name, value)


def test_zip_delta_allows_only_three_contract_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    common = b"same"
    _write_zip(
        source,
        {
            "policy_server.py": b"execute8",
            "model_weights/config.json": b'{"n_action_steps":8}',
            "submission_manifest.json": b'{"n_action_steps":8}',
            "requirements.txt": common,
        },
    )
    _write_zip(
        candidate,
        {
            "policy_server.py": b"execute4",
            "model_weights/config.json": b'{"n_action_steps":4}',
            "submission_manifest.json": b'{"n_action_steps":4}',
            "requirements.txt": common,
        },
    )
    monkeypatch.setattr(mod, "SOURCE_ZIP", source)
    result = mod.compare_zip_entries(candidate)
    assert result["status"] == "passed"
    assert set(result["changed_entries"]) == mod.ALLOWED_CHANGED_ENTRIES


def test_zip_delta_rejects_unexpected_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.zip"
    candidate = tmp_path / "candidate.zip"
    _write_zip(
        source,
        {
            "policy_server.py": b"8",
            "model_weights/config.json": b"8",
            "submission_manifest.json": b"8",
            "requirements.txt": b"same",
        },
    )
    _write_zip(
        candidate,
        {
            "policy_server.py": b"4",
            "model_weights/config.json": b"4",
            "submission_manifest.json": b"4",
            "requirements.txt": b"changed",
        },
    )
    monkeypatch.setattr(mod, "SOURCE_ZIP", source)
    result = mod.compare_zip_entries(candidate)
    assert result["status"] == "failed"
    assert result["unchanged_mismatch"] == ["requirements.txt"]
