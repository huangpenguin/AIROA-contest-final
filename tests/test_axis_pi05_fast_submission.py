from __future__ import annotations

import json
from pathlib import Path

from tools.axis_pi05.audit_fast_submission import _policy_static_audit


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging"


def test_staged_policy_is_fast_only_and_syntax_valid():
    result = _policy_static_audit(STAGING / "policy_server.py")
    assert result["passed"], result


def test_recovery_protocol_config_is_locked():
    config = json.loads((ROOT / "data/reports/axis_fast_recovery/f2_m1_500_fast/export_report.json").read_text())
    assert config["source_is_current_submission_model"] is True
    assert config["historical_fast_asset_used"] is False
    assert config["training_performed"] is False
    assert config["key_gate"]["passed"] is True
    assert config["vision_gate"]["passed"] is True
    assert config["reload"]["strict_key_gate"] is True
