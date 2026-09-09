from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
AUDIT = ROOT / "data/reports/axis_libero_30k/track3_o1_2_over_budget_independent_audit_20260909/summary.json"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_MULTITASK_GOLD_AND_R6_SHORT_CONTINUATION_PLAN.md"


def strict_json(path: Path) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def test_o1_2_audit_approves_closeout_but_not_gold() -> None:
    audit = strict_json(AUDIT)
    assert audit["audit_class"] == "approved"
    assert audit["status"] == "approved_closeout_not_gold"
    assert audit["o1_2"]["physical_success"] is True
    assert audit["o1_2"]["frame_count"] == 314
    assert audit["o1_2"]["step_budget"] == 300
    assert audit["o1_2"]["over_budget_by"] == 14
    assert audit["o1_2"]["gold_eligible"] is False
    assert audit["o1_2"]["fresh_replay"] == "not_evaluated"
    assert audit["o1_2"]["npz_object_arrays"] == []
    assert audit["raw_files"]["front.mp4"]["frames"] == 314
    assert audit["raw_files"]["wrist.mp4"]["frames"] == 314


def test_multitask_plan_is_the_only_active_authorization() -> None:
    handoff = strict_json(HANDOFF)
    assert PLAN.is_file()
    assert handoff["status"] == "track3_mg1_red_mug_attempts_exhausted_ready_for_user_decision"
    assert handoff["next_plan"] == "docs/AXIS_PI05_TRACK3_MULTITASK_GOLD_AND_R6_SHORT_CONTINUATION_PLAN.md"
    assert handoff["next_phase_only"] == "MG2_alphabet_soup_requires_explicit_user_takeover"
    authorization = handoff["track3_multitask_gold_and_r6_short_continuation_authorization"]
    assert authorization["status"] == "mg0_passed_ready_for_user_calibration"
    assert authorization["formal_attempt_max_total"] == 7
    assert authorization["task_order"][:3] == [
        "official_t3_red_mug",
        "official_t3_alphabet_soup",
        "official_t3_yellow_book",
    ]
    assert authorization["training_gate"]["accepted_min"] == 3
    assert authorization["training_gate"]["task_id_min"] == 2
    assert authorization["training_gate"]["family_min"] == 1
    assert authorization["training_gate"]["family_target"] == 2
    assert authorization["human_review_required"] is True
    assert authorization["automatic_upload_or_submission"] is False


def test_unexecuted_training_and_new_attempts_remain_not_evaluated() -> None:
    audit = strict_json(AUDIT)
    assert all(value == "not_evaluated" for value in audit["unexecuted"].values())
