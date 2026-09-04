from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.axis_pi05 import track3_matched_terminal_state_release as release


def test_l0_manifest_is_fixed_three_demos_per_family() -> None:
    manifest = json.loads((release.OUT_ROOT / "l0_demo_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["fixed_selection"]
    assert [(row["task_index"], row["demo_index"]) for row in rows] == [(10, 0), (10, 1), (10, 2), (20, 0), (20, 1), (20, 2), (19, 0), (19, 1), (19, 2)]
    assert all(row["selection"]["suffix_length"] == 1 for row in rows)
    assert all(row["predicate_oracle"].startswith("never_hdf5") for row in rows)


def test_renderer_schema_reads_wrapper_and_underlying_flags() -> None:
    class Underlying:
        use_camera_obs = True
        has_offscreen_renderer = True
        has_renderer = False

    class Wrapper:
        env = Underlying()
        use_camera_obs = False
        has_offscreen_renderer = False
        has_renderer = False

    observation = {"agentview_image": np.zeros((4, 5, 3), dtype=np.uint8), "robot0_eye_in_hand_image": np.zeros((4, 5, 3), dtype=np.uint8)}
    schema = release.renderer_schema(Wrapper(), observation)
    assert schema["wrapper_flags"]["use_camera_obs"] is False
    assert schema["underlying_flags"]["use_camera_obs"] is True
    assert schema["schema_complete"] is True


def test_h264_uses_atomic_mp4_partial_suffix(tmp_path: Path) -> None:
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(3)]
    target = tmp_path / "camera.mp4"
    result = release.atomic_h264(target, frames, fps=20)
    assert result["state"] == "passed"
    assert target.is_file()
    assert not (tmp_path / "camera.mp4.partial").exists()
    assert not (tmp_path / "camera.partial.mp4").exists()
    assert result["probe"]["codec_name"] == "h264"


def test_public_drawer_close_midpoint_is_absolute() -> None:
    metadata = release.public_drawer_articulation_metadata()
    assert metadata["open_range"] == [-0.16, -0.14]
    assert metadata["close_range"] == [0.0, 0.005]
    assert release.close_range_midpoint(metadata) == 0.0025


def test_contact_v2_does_not_grandfather_step5_operator_background_contact() -> None:
    matrix = {"complete": True, "contacts": [{"body1_name": "wooden_cabinet_1_cabinet_middle", "body2_name": "akita_black_bowl_1_main", "distance_m": -0.0068}]}
    result = release.contact_contract_v2(19, matrix)
    assert result["state"] == "failed"
    assert result["forbidden_count"] == 1


def test_contact_v2_retains_allowed_support_contact_and_rejects_background() -> None:
    allowed = {"complete": True, "contacts": [{"body1_name": "akita_black_bowl_1_main", "body2_name": "plate_1_main", "distance_m": 0.0}]}
    assert release.contact_contract_v2(10, allowed)["state"] == "passed"
    mixed = {"complete": True, "contacts": allowed["contacts"] + [{"body1_name": "akita_black_bowl_1_main", "body2_name": "wooden_cabinet_1_cabinet_middle", "distance_m": -0.001}]}
    assert release.contact_contract_v2(10, mixed)["state"] == "failed"


def test_fail_closed_worker_contract_and_passed_gate_reason_semantics() -> None:
    assert release.worker_exit_code("worker_exception") != 0
    assert release.worker_exit_code("failed_gate") == 0
    passed = release.gate(executed=True, value=True)
    failed = release.gate(executed=True, value=False, reason="schema_missing")
    assert "reason" not in passed
    assert failed["reason"] == "schema_missing"
