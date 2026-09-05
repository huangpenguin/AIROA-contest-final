from __future__ import annotations

from tools.axis_pi05 import track3_object_frame_executor_and_bounded_autogen_lora as tool


def test_object_frame_composition_uses_object_to_eef_anchor() -> None:
    result = tool.compose_object_to_eef(
        {"pos_m": [1.0, 2.0, 3.0], "quat": [0.0, 0.0, 0.0, 1.0]},
        {"translation_m": [0.1, 0.2, 0.3], "quaternion_xyzw": [0.0, 0.0, 0.0, -1.0]},
    )
    assert result["pos_m"] == [1.1, 2.2, 3.3]
    assert tool.qgeo(result["quat"], [0.0, 0.0, 0.0, 1.0]) == 0.0


def test_progress_window_has_explicit_pass_fail_and_not_evaluated_states() -> None:
    assert tool.progress_window_gate([1.0, 0.8, 0.6, 0.4], window=4, min_improvement=0.5)["state"] == "passed"
    assert tool.progress_window_gate([1.0, 0.99, 0.98, 0.97], window=4, min_improvement=0.5)["state"] == "failed"
    assert tool.progress_window_gate([1.0, 0.9], window=4, min_improvement=0.1)["state"] == "not_evaluated"


def test_unentered_gate_is_not_evaluated() -> None:
    result = tool.gate(False, None, reason="stage_not_entered")
    assert result["executed"] is False
    assert result["state"] == "not_evaluated"
