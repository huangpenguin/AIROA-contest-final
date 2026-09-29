import numpy as np
import pytest

from tools.axis_pi05 import run_r6_t2_light_smoothing_screen as screen


class FakeClient:
    def __init__(self, actions: list[list[float]]) -> None:
        self.actions = [np.asarray(action, dtype=np.float32) for action in actions]
        self.reset_calls: list[dict] = []

    def reset(self, **kwargs) -> None:
        self.reset_calls.append(kwargs)

    def get_action(self, _observation) -> np.ndarray:
        return self.actions.pop(0).copy()


def _actions(count: int) -> list[list[float]]:
    return [list(np.arange(6, dtype=np.float32) + index) + [10.0 + index] for index in range(count)]


def test_seam_blend_changes_only_first_action_of_new_chunk() -> None:
    screen.configure("seam_blend")
    client = FakeClient(_actions(5))
    policy = screen.VariantPolicy(client)
    policy.reset("task", seed=7)
    observed = [policy.get_action({}) for _ in range(5)]

    np.testing.assert_array_equal(observed[0], np.asarray(_actions(1)[0], dtype=np.float32))
    np.testing.assert_array_equal(observed[3], np.asarray(_actions(4)[3], dtype=np.float32))
    expected_arm = 0.9 * (np.arange(6, dtype=np.float32) + 4.0) + 0.1 * (np.arange(6, dtype=np.float32) + 3.0)
    np.testing.assert_allclose(observed[4][:6], expected_arm)
    assert observed[4][6] == pytest.approx(14.0)
    assert all(float(observed[i][6]) == pytest.approx(10.0 + i) for i in range(5))
    assert len(client.reset_calls) == 2
    assert client.reset_calls[1]["reset_kind"] == "chunk_discard"


def test_arm_ema_uses_previous_emitted_arm_and_raw_gripper() -> None:
    screen.configure("arm_ema01")
    client = FakeClient(_actions(2))
    policy = screen.VariantPolicy(client)
    policy.reset("task", seed=8)
    first, second = policy.get_action({}), policy.get_action({})

    np.testing.assert_array_equal(first, np.asarray(_actions(1)[0], dtype=np.float32))
    expected_arm = 0.9 * (np.arange(6, dtype=np.float32) + 1.0) + 0.1 * np.arange(6, dtype=np.float32)
    np.testing.assert_allclose(second[:6], expected_arm)
    assert second[6] == pytest.approx(11.0)


@pytest.mark.parametrize("variant, horizon", [("execute4_reference", 4), ("seam_blend", 4), ("arm_ema01", 4), ("execute5", 5)])
def test_variant_protocol_is_frozen(variant: str, horizon: int) -> None:
    screen.configure(variant)
    assert screen.base.CHUNK_SIZE == 10
    assert screen.base.EXECUTE_ACTIONS == horizon
    assert screen.VARIANTS[variant]["execute_actions"] == horizon
    assert screen.VARIANTS[variant]["mode"] in {"reference", "seam_blend", "arm_ema01", "execute5"}


def test_eef_jerk_uses_actual_position_trace() -> None:
    dt = screen.CONTROL_DT
    t = np.arange(8, dtype=np.float64) * dt
    positions = np.stack([t**3, np.zeros_like(t), np.zeros_like(t)], axis=1)
    assert screen._eef_jerk_rms(positions) == pytest.approx(6.0)


def test_eef_jerk_rejects_short_or_nonfinite_trace() -> None:
    with pytest.raises(ValueError):
        screen._eef_jerk_rms(np.zeros((3, 3)))
    with pytest.raises(FloatingPointError):
        screen._eef_jerk_rms(np.full((4, 3), np.nan))
