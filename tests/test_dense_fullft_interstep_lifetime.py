from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.axis_pi05 import dense_fullft_interstep_state_lifetime_recovery as recovery


def _row(phase: str, step: int, index: int, value: int = 20_000_000_000):
    return {
        "phase": phase,
        "optimizer_step_before": step,
        "micro_index": index,
        "devices": [
            {"memory_stats_available": True, "memory_stats": {"bytes_in_use": value + card}}
            for card in range(4)
        ],
    }


def test_step_plateau_requires_ordered_compute_merge_and_release_boundaries():
    rows = []
    for index in range(16):
        rows.extend(
            _row(phase, 1, index)
            for phase in ("microbatch_compute_complete", "accumulation_complete", "temporaries_released")
        )
    result = recovery._step_plateau(rows, 1)
    assert result["passed"] is True
    assert result["compute_completed_indexes"] == list(range(16))
    assert result["temporaries_released_indexes"] == list(range(16))

    missing = [row for row in rows if not (row["phase"] == "temporaries_released" and row["micro_index"] == 7)]
    assert recovery._step_plateau(missing, 1)["passed"] is False


def test_transition_gate_compares_next_microbatch_to_previous_step_release():
    rows = [_row("temporaries_released", 1, 15), _row("temporaries_released", 2, 0)]
    result = recovery._transition_memory_gate(rows)
    assert result["passed"] is True
    assert result["max_absolute_drift_bytes_per_card"] == 0

    missing = [_row("temporaries_released", 1, 15)]
    assert recovery._transition_memory_gate(missing)["passed"] is False


def test_l0_semantic_fields_do_not_use_loop_index_as_post_update_step():
    rows = [
        _row("temporaries_released", 2, 0),
        {"phase": "microbatch_exception", "optimizer_step_before": 2, "micro_index": 1},
    ]
    assert recovery._indexes(rows, "temporaries_released", 2) == [0]
    assert recovery._indexes(rows, "microbatch_exception", 2) == [1]
