from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.axis_pi05.dense_fullft_preflight import (  # noqa: E402
    EXPECTED_PARAM_NUMEL,
    _memory_budget,
    validate_config_contract,
)


def test_dense_config_matches_canonical_recipe():
    report = validate_config_contract()
    assert report["status"] == "passed", report["mismatches"]
    assert report["config_name"] == "axis_pi05_libero_fullft"


def test_dense_memory_budget_preserves_recipe_and_shows_v100_boundary():
    report = _memory_budget(EXPECTED_PARAM_NUMEL)
    assert report["parameter_numel"] == 3_353_433_872
    assert report["recipe_preserved"] == {
        "ema_decay": 0.999,
        "effective_global_batch": 64,
        "precision": "float32",
        "freeze_filter": "nnx.Nothing",
    }
    assert report["topologies"]["2"]["theoretical_headroom_gib_per_card"] < 2.0
    assert report["topologies"]["4"]["theoretical_headroom_gib_per_card"] > 2.0


def test_dense_openpi_source_has_no_lora_variants_in_config():
    source = (ROOT / "third_party/axis_openpi/src/openpi/training/config.py").read_text(encoding="utf-8")
    start = source.index('name="axis_pi05_libero_fullft"')
    end = source.index('name="axis_pi05_lora_screen"', start)
    block = source[start:end]
    assert 'paligemma_variant="gemma_2b"' in block
    assert 'action_expert_variant="gemma_300m"' in block
    assert "freeze_filter=nnx.Nothing" in block
    assert "lora" not in block.lower()
