from __future__ import annotations

from pathlib import Path

from tools.axis_pi05.track3_public_primitive_mining import operator_for_instruction


def test_public_mining_operator_taxonomy_is_conservative() -> None:
    assert operator_for_instruction("pick up the orange juice and place it in the basket") == "containment_remove"
    assert operator_for_instruction("put the bowl on the plate") == "support_remove"
    assert operator_for_instruction("open the middle drawer of the cabinet") == "articulation_close"
    assert operator_for_instruction("push the plate to the front of the stove") is None


def test_source_lock_is_present() -> None:
    path = Path("configs/axis_pi05/track3_public_sources.lock.yaml")
    assert path.is_file()
    assert "libero_infinity" in path.read_text(encoding="utf-8")
