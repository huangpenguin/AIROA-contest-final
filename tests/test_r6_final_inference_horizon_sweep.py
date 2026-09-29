from pathlib import Path

import pytest

from tools.axis_pi05 import run_r6_final_inference_horizon_sweep as sweep


def test_configure_binds_horizon_adapter_and_plan() -> None:
    sweep.configure(6)
    assert sweep.base.EXECUTE_ACTIONS == 6
    assert sweep.base.ADAPTER_PATH == Path(sweep.__file__).resolve()
    assert sweep.base.PLAN == sweep.PLAN


@pytest.mark.parametrize("horizon", [4, 6, 8, 10])
def test_variant_roots_are_disjoint(horizon: int) -> None:
    evidence, output = sweep.variant_roots(horizon)
    assert evidence.name == f"execute{horizon}"
    assert output.name == f"execute{horizon}"
    assert evidence != output


def test_invalid_horizon_fails_closed() -> None:
    with pytest.raises(ValueError):
        sweep.configure(5)


def test_screen_threshold_is_not_small_delta() -> None:
    assert sweep.BASELINE_HORIZON == 8
    assert sweep.HORIZONS == (4, 6, 8, 10)

