from __future__ import annotations

from tools.axis_pi05 import track3_reverse_semantic_adapter as adapter


def test_canary_order_is_fixed_and_non_training() -> None:
    assert adapter.TASK_ORDER == ((10, "support_remove"), (20, "containment_remove"), (19, "articulation_close"))


def test_denylist_token_reader_does_not_change_manifest() -> None:
    tokens = adapter.denylist_tokens()
    assert isinstance(tokens, set)
    assert adapter.DENYLIST.is_file()
