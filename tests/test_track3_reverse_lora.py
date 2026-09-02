from __future__ import annotations

from pathlib import Path

import pytest

from tools.axis_pi05 import track3_reverse_lora as t3


def test_public_operator_families_are_fixed_and_public_only() -> None:
    assert t3.PUBLIC_OPERATOR_FAMILIES == (
        "containment_remove_restore",
        "support_remove_restore",
        "articulation_inverse",
        "activation_inverse",
        "sequence_reverse",
    )


def test_gpu_gate_rejects_target_busy_but_never_requires_gpu0() -> None:
    snapshot = [
        {"index": 0, "memory_used_mib": 16000, "utilization_gpu_percent": 80},
        {"index": 1, "memory_used_mib": 5, "utilization_gpu_percent": 0},
        {"index": 2, "memory_used_mib": 5, "utilization_gpu_percent": 0},
        {"index": 3, "memory_used_mib": 5, "utilization_gpu_percent": 0},
    ]
    t3.require_track3_targets_free(snapshot)
    with pytest.raises(RuntimeError, match="WAITING_FOR_GPU_TARGETS"):
        t3.require_track3_targets_free([{**row, "memory_used_mib": 100} if row["index"] == 2 else row for row in snapshot])


def test_gpu_gate_rejects_gpu0_as_target() -> None:
    with pytest.raises(ValueError, match="GPU0"):
        t3.require_track3_targets_free([], targets=(0,))


def test_denylist_scanner_is_exact_and_does_not_emit_identifiers(tmp_path) -> None:
    deny = tmp_path / "deny.jsonl"
    train = tmp_path / "train.jsonl"
    deny.write_text('{"task_name":"public_eval","language":"reverse public_eval","bddl_path":"public.bddl"}\n', encoding="utf-8")
    train.write_text('{"task_name":"forward_train","language":"forward train","bddl_path":"forward.bddl"}\n', encoding="utf-8")
    result = t3.scan_evaluation_denylist(denylist_path=deny, training_manifest=train)
    assert result["status"] == "passed"
    assert result["exact_overlap_count"] == 0
    assert result["identifiers_emitted"] is False


def test_t0_validation_rejects_overclaimed_score() -> None:
    record = {
        "status": "track3_reverse_t0_cpu_locked",
        "public_reverse_rule_only": True,
        "official_hidden_details_used": False,
        "gpu_boundary": {"target_idle_gate": "passed"},
        "evaluation_only_denylist": {"exact_overlap_count": 0},
        "scores": {"track1": {"provenance": "official"}},
    }
    with pytest.raises(ValueError, match="score provenance"):
        t3.validate_t0_lock(record)


def test_public_predicate_parser_drops_empty_unary_argument() -> None:
    text = """
    (:init
      (Open drawer_1)
      (On bowl_1 table_bowl_init_region)
    )
    (:goal
      (And (Close drawer_1) (In bowl_1 drawer_1_top_region))
    )
    """
    assert t3.parse_public_predicates(text, "init") == (
        t3.Predicate("Open", ("drawer_1",)),
        t3.Predicate("On", ("bowl_1", "table_bowl_init_region")),
    )
    assert t3.parse_public_predicates(text, "goal") == (
        t3.Predicate("Close", ("drawer_1",)),
        t3.Predicate("In", ("bowl_1", "drawer_1_top_region")),
    )


def test_compile_reverse_spec_preserves_roles_and_flips_state(tmp_path: Path) -> None:
    bddl = tmp_path / "KITCHEN_SCENE1_put_the_bowl_in_the_drawer.bddl"
    init = tmp_path / "KITCHEN_SCENE1_put_the_bowl_in_the_drawer.pruned_init"
    bddl.write_text(
        """(define (problem P)\n(:language put the bowl in the drawer)\n(:init\n (Open drawer_1)\n (On bowl_1 table_bowl_init_region)\n)\n(:goal\n (And (In bowl_1 drawer_1_top_region) (Close drawer_1))\n)\n)\n""",
        encoding="utf-8",
    )
    init.write_bytes(b"public-init")
    spec = t3.compile_reverse_spec(0, "put the bowl in the drawer", bddl, init)
    assert spec["operator_family"] == "sequence_reverse"
    assert "(In bowl_1 drawer_1_top_region)" in spec["reverse_initial_predicates"]
    assert "(Close drawer_1)" in spec["reverse_initial_predicates"]
    assert "(On bowl_1 table_bowl_init_region)" in spec["reverse_goal_predicates"]
    assert "(Open drawer_1)" in spec["reverse_goal_predicates"]
    assert spec["reverse_goal_predicates"][0] == "(On bowl_1 table_bowl_init_region)"
    assert "swap" not in spec["reverse_instruction"].lower()
