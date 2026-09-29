"""Freeze the pre-training reverse-dev identities and the historical C20 panel.

This is a derived evidence writer.  It never edits an episode metadata file or
regenerates simulator state.  The selected reverse identities already exist in
the locked public T3 holdout manifest and are checked against the four reverse
training start-state hashes before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910"
HOLDOUT = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"
C20 = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
Q2_MANIFEST = ROOT / "data/manifests_track3/four_gold_80_20_v1/manifest.json"
Q3_NORM = ROOT / "data/manifests_track3/four_gold_80_20_v1/norm_stats.json"

TASK_TO_INDEX = {
    "official_t3_alphabet_soup": 0,
    "official_t3_red_mug": 1,
    "official_t3_two_mugs": 2,
    "official_t3_yellow_book": 3,
}

# These are explicit, stable selections from the already locked public holdout
# file.  They are not generated from the training episodes.
REVERSE_DEV_INIT_INDEX = {0: 7, 1: 3, 2: 9, 3: 6}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=_strict_object)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVIDENCE / "evaluation_freeze.json")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    accepted = sorted((ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted").glob("*/metadata.json"))
    if len(accepted) != 4:
        raise ValueError(f"expected four accepted reverse metadata files, found {len(accepted)}")
    train_rows = [load_json(path) for path in accepted]
    train_by_task = {str(row["task_id"]): row for row in train_rows}
    if set(train_by_task) != set(TASK_TO_INDEX):
        raise ValueError(f"accepted reverse task set mismatch: {sorted(train_by_task)}")
    train_start_hashes = {str(row["start_state_sha256"]) for row in train_rows}

    holdout_rows = load_jsonl(HOLDOUT)
    selected: list[dict[str, Any]] = []
    for task_index, init_index in sorted(REVERSE_DEV_INIT_INDEX.items()):
        matches = [
            row
            for row in holdout_rows
            if row.get("task_id") == task_index and row.get("init_state_index") == init_index
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one holdout row for task={task_index}, init={init_index}; found {len(matches)}")
        row = matches[0]
        if row.get("panel") != "public_t3_holdout":
            raise ValueError(f"unexpected holdout panel for task={task_index}")
        if row["init_state_sha256"] in train_start_hashes:
            raise ValueError(f"reverse dev start overlaps training start: task={task_index}")
        selected.append(
            {
                "task_index": task_index,
                "task_name": row["task_name"],
                "init_state_index": row["init_state_index"],
                "init_state_sha256": row["init_state_sha256"],
                "bddl_path": row["bddl_path"],
                "bddl_sha256": row["bddl_sha256"],
                "case_id": row["case_id"],
                "env_seed": row["env_seed"],
                "policy_seed": row["policy_seed"],
                "max_steps": row["max_steps"],
                "selection_source": str(HOLDOUT.relative_to(ROOT)),
            }
        )

    c20_rows = load_jsonl(C20)
    case_ids = [row.get("case_id") for row in c20_rows]
    if len(c20_rows) != 20 or len(set(case_ids)) != 20:
        raise ValueError("C20 must contain exactly 20 unique cases")
    if any(row.get("panel") != "C20" for row in c20_rows):
        raise ValueError("C20 manifest contains a non-C20 row")

    train_identities = [
        {
            "task_id": row["task_id"],
            "start_state_sha256": row["start_state_sha256"],
            "trajectory_sha256": row["trajectory"]["sha256"],
            "metadata_path": str(path.relative_to(ROOT)),
        }
        for path, row in zip(accepted, train_rows)
    ]
    payload = {
        "schema_version": "track3_evaluation_freeze_v1",
        "status": "evaluation_set_frozen_before_q5",
        "created_by": "tools/axis_pi05/freeze_track3_eval_v1.py",
        "training_reverse_identities": train_identities,
        "reverse_dev": {
            "status": "frozen",
            "panel": "public_t3_holdout",
            "manifest_path": str(HOLDOUT.relative_to(ROOT)),
            "manifest_sha256": sha256(HOLDOUT),
            "fixed_policy_seed": 2000,
            "fixed_env_seed": 1000,
            "selection_rule": "explicit task_index -> init_state_index mapping",
            "rows": selected,
            "disjoint_start_state_sha256": True,
        },
        "c20_retention": {
            "status": "frozen",
            "manifest_path": str(C20.relative_to(ROOT)),
            "manifest_sha256": sha256(C20),
            "case_count": len(c20_rows),
            "cases": [
                {
                    "case_id": row["case_id"],
                    "official_task_index": row["official_task_index"],
                    "init_state_index": row["init_state_index"],
                    "init_state_sha256": row["init_state_sha256"],
                    "env_seed": row["env_seed"],
                    "policy_seed": row["policy_seed"],
                }
                for row in c20_rows
            ],
        },
        "training_inputs_at_freeze": {
            "q2_manifest_path": str(Q2_MANIFEST.relative_to(ROOT)),
            "q2_manifest_sha256": sha256(Q2_MANIFEST),
            "q3_norm_path": str(Q3_NORM.relative_to(ROOT)),
            "q3_norm_sha256": sha256(Q3_NORM),
        },
        "historical_metadata_unchanged": True,
        "simulator_generation_performed": False,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest, "reverse_dev_count": len(selected), "c20_count": len(c20_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
