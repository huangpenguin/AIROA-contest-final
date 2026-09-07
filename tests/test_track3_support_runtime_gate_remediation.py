from __future__ import annotations

import subprocess
from pathlib import Path

from tools.axis_pi05 import track3_support_runtime_gate_remediation as sr


ROOT = Path(__file__).resolve().parents[1]


def test_precontact_prefix_does_not_fail_acquisition_window() -> None:
    empty = {"complete": True, "contacts": []}
    grip = {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "robot0_right_hand"}]}
    samples = [{"contact_matrix": empty} for _ in range(6)] + [{"contact_matrix": grip} for _ in range(4)]
    result = sr.required_contact_gate(samples, 9, stable_ticks=4)
    assert result["state"] == "passed"
    assert result["prefix_without_contact_ignored"] is True
    assert result["required_contact_values"] == [True, True, True, True]


def test_source_release_is_not_evaluated_before_lift() -> None:
    dwell = sr.source_released_gate([True, True, True, True], ["grasp_dwell"] * 4)
    assert dwell["state"] == "not_evaluated"
    lifted = sr.source_released_gate([True, True, True, True, False, False, False, False], ["grasp_dwell"] * 4 + ["lift"] * 4)
    assert lifted["state"] == "passed"


def test_bowl_main_table_is_goal_receiver_not_unrelated() -> None:
    result = sr.contact_truth_v3(10, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "main_table"}]})
    assert result["goal_receiver_contact"] is True
    assert result["unrelated_contact_runtime"] is False
    plate = sr.contact_truth_v3(10, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "plate_1"}]})
    assert plate["source_receiver_contact"] is True
    salad = sr.contact_truth_v3(10, {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1", "body2_name": "salad_dressing_1"}]})
    assert salad["unrelated_contact_runtime"] is True


def test_goal_continuous_requires_frozen_window_not_final_frame() -> None:
    final_only = sr.goal_continuous_gate([False, False, False, True], ["place"] * 4, check_success=True)
    window = sr.goal_continuous_gate([True, True, True, True], ["place"] * 4, check_success=True)
    assert final_only["state"] == "failed"
    assert final_only["not_final_frame_only"] is True
    assert window["state"] == "passed"


def test_success_path_fixture_is_reachable() -> None:
    positive = sr.success_path_fixture()
    assert positive["required_contact"]["state"] == "passed"
    assert positive["source_released"]["state"] == "passed"
    assert positive["goal_continuous_true"]["state"] == "passed"
    assert positive["unrelated_any"] is False
    late = sr.success_path_fixture(late_contact_only=True)
    assert late["required_contact"]["state"] == "failed"
    unknown = sr.success_path_fixture(unknown=True)
    assert unknown["required_contact"]["state"] == "infrastructure_failure"


def test_stop_classes_are_scoped() -> None:
    assert sr.classify_stop(environment_created=False, physics_steps=0, error_type="TypeError") == "runtime_lane_stop"
    assert sr.classify_stop(environment_created=None, physics_steps=None, error_type="TypeError") == "global_hard_stop"
    assert sr.classify_stop(environment_created=False, physics_steps=0, error_type=None, identity_failed=True) == "global_hard_stop"
    assert sr.classify_stop(environment_created=True, physics_steps=8, error_type=None, family_gate_failed=True) == "family_outcome"


def test_prepare_only_uses_real_assets_without_control_env() -> None:
    code = (
        "import json,sys\n"
        "from tools.axis_pi05.track3_support_runtime_gate_remediation import PRIMARY_PROFILE, prepare_support_identity, CONTROL_ENV_MODULE\n"
        "result = prepare_support_identity(PRIMARY_PROFILE)\n"
        "payload = {key: result[key] for key in ('environment_created','physics_steps','control_env_imported_after','budget_matches_profile','transport_offset_used_by_stage_target','orientation_error_nonzero','start_state_finite')}\n"
        "payload['control_env_in_sys_modules'] = CONTROL_ENV_MODULE in sys.modules\n"
        "print(json.dumps(payload))\n"
    )
    completed = subprocess.run([str(ROOT / ".venv_axis_pi05/bin/python"), "-c", code], cwd=ROOT, capture_output=True, text=True, check=False, env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu", "MUJOCO_GL": "disable", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}"})
    assert completed.returncode == 0, completed.stderr
    payload = __import__("json").loads(completed.stdout.strip().splitlines()[-1])
    assert payload["environment_created"] is False
    assert payload["physics_steps"] == 0
    assert payload["control_env_imported_after"] is False
    assert payload["control_env_in_sys_modules"] is False
    assert payload["budget_matches_profile"] is True
    assert payload["transport_offset_used_by_stage_target"] is True
    assert payload["orientation_error_nonzero"] is True
    assert payload["start_state_finite"] is True


def test_source_porcelain_excludes_sr_evidence_rows() -> None:
    row = " M data/reports/axis_libero_30k/track3_support_runtime_gate_remediation/current_attempt.json"
    path_field = row[3:].strip()
    roots = (sr.rel(sr.SR_ROOT), sr.rel(sr.SG_ROOT))
    assert any(path_field == root or path_field.startswith(root + "/") for root in roots)
    stripped = row.strip()
    assert stripped[3:].strip().startswith("ata/") is True
    live = sr.source_porcelain()
    assert not any("track3_support_runtime_gate_remediation/current_attempt.json" in item for item in live)


def test_identity_uses_this_module_tool_hash() -> None:
    tool_sha = sr.sha256_file(Path(sr.__file__))
    result = sr.identity_gate(expected_commit="a", expected_tool_sha256=tool_sha, expected_openpi="c", expected_command=["worker"], observed_commit="a", observed_openpi="c", observed_command=["worker"], dirty=False)
    assert result["checks"]["tool_sha256_exact"] is True
    wrong = sr.identity_gate(expected_commit="a", expected_tool_sha256="dead", expected_openpi="c", expected_command=["worker"], observed_commit="a", observed_openpi="c", observed_command=["worker"], dirty=False)
    assert wrong["checks"]["tool_sha256_exact"] is False


def test_quantified_progress_requires_real_gain() -> None:
    no_gain = sr.quantified_progress({"metrics": {"last_stage": "contact_descent", "last_translation_error_m": 0.02, "acquisition_entered": False, "lift_entered": False, "place_entered": False}})
    gain = sr.quantified_progress({"metrics": {"last_stage": "grasp_dwell", "last_translation_error_m": 0.02, "acquisition_entered": True, "lift_entered": False, "place_entered": False}})
    assert no_gain["eligible"] is False
    assert gain["eligible"] is True
    assert "entered_acquisition" in gain["reasons"]


def test_finalizer_uses_relative_paths_and_rejects_late_or_repeat() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sr_finalizer_") as temporary:
        root = Path(temporary)
        sr.write_json(root / "early.json", {"n": 1})
        sr.write_json(root / "nested" / "closure.json", {"producer": True})
        manifest = sr.make_manifest(root)
        assert any(row["relative_path"] == "nested/closure.json" for row in manifest["entries"])
        sr.write_json(root / "late.json", {"n": 2})
        assert sr.verify_complete_set(root, manifest)["exact"] is False
    with tempfile.TemporaryDirectory(prefix="sr_finalizer_ok_") as temporary:
        root = Path(temporary)
        sr.write_json(root / "only.json", {"n": 1})
        sr.make_manifest(root)
        first = sr.finalize_attempt(root)
        second = sr.finalize_attempt(root)
        assert first["status"] == "passed"
        assert second["reason"] == "repeat_finalization"


def test_not_evaluated_is_derived_not_hardcoded_empty() -> None:
    assert sr.derive_not_evaluated(sr.SR_ROOT / "attempts/attempt_0004", success_count=0) == ["SR9", "SR10", "SR11"]
    topology = sr.alternative_topology_is_structurally_distinct()
    assert topology["actual_stage_order_identical"] is True
    assert topology["structurally_distinct"] is False
