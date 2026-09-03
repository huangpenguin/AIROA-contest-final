from __future__ import annotations

import numpy as np

from tools.axis_pi05.track3_scripted_autogen import closed_loop_action, parse_atom, self_test


def test_scripted_autogen_self_test() -> None:
    assert self_test()["status"] == "passed"


def test_closed_loop_action_is_bounded_and_has_explicit_gripper() -> None:
    action, error, saturated = closed_loop_action(
        {"robot0_eef_pos": np.zeros(3, dtype=np.float32)},
        np.array([0.10, -0.10, 0.0], dtype=np.float64),
        -1.0,
    )
    assert action.shape == (7,)
    assert np.all(np.abs(action[:6]) <= 1.0)
    assert action[-1] == -1.0
    assert error > 0.0 and saturated


def test_public_predicate_parser_is_strict() -> None:
    assert parse_atom("(Close wooden_cabinet_1_middle_region)") == (
        "Close",
        ("wooden_cabinet_1_middle_region",),
    )
