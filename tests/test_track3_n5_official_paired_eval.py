from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.axis_pi05.launch_official_blackwell import parse_args, resolve_launch  # noqa: E402
from tools.axis_pi05.download_official_canary100 import CONTRACT, load_contract  # noqa: E402
from tools.axis_pi05.run_track3_n5_official_paired_eval import (  # noqa: E402
    DEFAULT_CANDIDATE,
    Q7_NAMESPACE,
    continuation_gates,
    paired_summary,
    validate_r6_params,
)
from tools.axis_pi05.run_track12_step100_paired_eval import TRACKS, c20_gates  # noqa: E402
from tools.axis_pi05.run_track3_q5_step10_frozen_eval import server_environment  # noqa: E402


def test_canary100_hf_contract_uploads_full_step100_and_prefers_t3():
    contract = load_contract()
    assert contract["upload_order"] == ["T3", "T1", "T2"]
    assert contract["required_members"] == ["params", "train_state", "_CHECKPOINT_METADATA"]
    assert "t3_four_gold_canary100/100" in str(DEFAULT_CANDIDATE)
    assert CONTRACT.is_file()
    for track in ("T3", "T1", "T2"):
        assert contract["tracks"][track]["repo_id"].startswith("Huangpenguin/parc2026-")
        assert contract["tracks"][track]["repo_id"].endswith("step100")


def test_track12_norm_lock_uses_checkpoint_assets_not_pretrain_source_json():
    assert TRACKS["T1"]["expected_norm_sha256"] == "f49d37dac32533048a14039570ab99ef2e8fd693f0081544807e72a7ceffa50c"
    assert TRACKS["T2"]["expected_norm_sha256"] == "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41"


def test_track12_c20_eval_cannot_authorize_step250_without_dedicated_panel():
    pairs = [{"panel": "C20", "baseline_success": True, "candidate_success": True}] * 18
    pairs += [{"panel": "C20", "baseline_success": True, "candidate_success": False}] * 2
    decision = c20_gates(pairs)
    assert decision["c20"]["delta"] == -2
    assert decision["gates"]["c20_drop_at_most_2_of_20"] is True
    assert decision["continue_to_step250_authorized"] is False
    assert decision["continue_blocked_reason"] == "dedicated_plus_panel_not_frozen"


def test_r6_eval_checkpoint_is_step_dir_not_nested_params(tmp_path):
    step = tmp_path / "500"
    params = step / "params"
    params.mkdir(parents=True)
    (params / "_METADATA").write_text("ok", encoding="utf-8")
    from_params = validate_r6_params(params)
    from_step = validate_r6_params(step)
    assert from_params["path"] == str(step.resolve())
    assert from_step["path"] == str(step.resolve())
    assert from_params["path"].endswith("/500")
    assert not from_params["path"].endswith("/params")


def test_n5_namespace_is_not_q7():
    assert "n5_official_t3_step100_paired_eval" in str(ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/n5_official_t3_step100_paired_eval")
    assert "q7_gpu3_frozen_eval" in str(Q7_NAMESPACE)


def test_eval_gpu_override_defaults_to_q7_gpu3(monkeypatch):
    monkeypatch.delenv("AXIS_PI05_EVAL_GPU", raising=False)
    env = server_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    monkeypatch.setenv("AXIS_PI05_EVAL_GPU", "0")
    env0 = server_environment()
    assert env0["CUDA_VISIBLE_DEVICES"] == "0"
    assert env0["MUJOCO_EGL_DEVICE_ID"] == "0"


def test_n5_continue_gate_requires_reverse_plus_one_and_c20_drop_at_most_two():
    pairs = [{"panel": "reverse_dev", "baseline_success": False, "candidate_success": True}] * 1
    pairs += [{"panel": "reverse_dev", "baseline_success": False, "candidate_success": False}] * 3
    pairs += [{"panel": "C20", "baseline_success": True, "candidate_success": True}] * 18
    pairs += [{"panel": "C20", "baseline_success": True, "candidate_success": False}] * 2
    decision = continuation_gates(pairs)
    assert decision["reverse"]["delta"] == 1
    assert decision["c20"]["delta"] == -2
    assert decision["continue_to_step300_authorized"] is True


def test_n5_continue_gate_rejects_reverse_flat_and_c20_drop_of_three():
    pairs = [{"panel": "reverse_dev", "baseline_success": False, "candidate_success": False}] * 4
    pairs += [{"panel": "C20", "baseline_success": True, "candidate_success": True}] * 17
    pairs += [{"panel": "C20", "baseline_success": True, "candidate_success": False}] * 3
    decision = continuation_gates(pairs)
    assert decision["gates"]["reverse_at_least_plus_1_of_4"] is False
    assert decision["gates"]["c20_drop_at_most_2_of_20"] is False
    assert decision["continue_to_step300_authorized"] is False


def test_paired_summary_keeps_training_steps_zero():
    cases = [{"case_id": f"c{i}", "panel": "reverse_dev" if i < 4 else "C20"} for i in range(24)]
    rows = [{"case_id": case["case_id"], "panel": case["panel"], "success": False, "collision": False, "timeout": False} for case in cases]
    summary = paired_summary(rows, rows, cases)
    assert summary["training_steps"] == 0
    assert summary["official_result"] is False
    assert summary["continue_to_step300_authorized"] is False
    assert summary["rollout_count"] == 48


def test_fresh_launcher_still_refuses_step300(tmp_path):
    args = parse_args(["--track", "T3", "--steps", "300", "--run-dir", str(tmp_path / "t3_300"), "--forward-root", str(tmp_path / "p"), "--reverse-root", str(tmp_path / "r")])
    with pytest.raises(ValueError, match="paired behavior evaluation"):
        resolve_launch(args)
