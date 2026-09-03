#!/usr/bin/env python3
"""Public-BDDL reverse semantic adapter for the three Track 3 canaries.

This is intentionally a small adapter around the already locked public LIBERO
BDDL.  It does not import the third-party environment, expose evaluation task
identifiers, or claim that a forward demonstration is an exact reverse demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from . import track3_reverse_lora as legacy

ROOT = Path(__file__).resolve().parents[2]
P0_MAPPING = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora/p0_task_mapping.json"
DENYLIST = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_autogen/s3_reverse_specs"
SUMMARY = ROOT / "data/reports/axis_libero_30k/track3_public_autogen/s3_reverse_adapter.json"
TASK_ORDER = ((10, "support_remove"), (20, "containment_remove"), (19, "articulation_close"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(value)
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def denylist_tokens() -> set[str]:
    rows = [json.loads(line) for line in DENYLIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    tokens: set[str] = set()
    for row in rows:
        for key in ("task_name", "language", "instruction", "bddl_path"):
            if row.get(key) not in (None, ""):
                tokens.add(str(row[key]))
    return tokens


def build_specs() -> dict[str, Any]:
    mapping = {int(row["task_index"]): row for row in json.loads(P0_MAPPING.read_text(encoding="utf-8"))}
    denied = denylist_tokens()
    specs: list[dict[str, Any]] = []
    for task_index, expected_family in TASK_ORDER:
        row = mapping[task_index]
        if row.get("base_public_task") in denied or row.get("bddl_path") in denied:
            raise RuntimeError("selected public canary overlaps evaluation-only denylist")
        bddl = Path(row["bddl_path"])
        init = Path(row["official_init_path"])
        compiled = legacy.compile_reverse_spec(task_index, row["base_public_task"], bddl, init)
        family = {"support_remove_restore": "support_remove", "containment_remove_restore": "containment_remove", "articulation_inverse": "articulation_close"}.get(compiled["operator_family"])
        if family != expected_family:
            raise RuntimeError(f"task {task_index} resolved to {family}, expected {expected_family}")
        output = OUT_ROOT / f"canary_{expected_family}_task_{task_index:02d}.bddl"
        text = legacy._render_custom_bddl(bddl.read_text(encoding="utf-8"), compiled["reverse_instruction"], compiled["reverse_initial_predicates"], compiled["reverse_goal_predicates"])
        atomic_write(output, text.encode())
        specs.append({
            "schema_version": "track3_reverse_spec_v1",
            "canary_id": f"{expected_family}_task_{task_index:02d}",
            "task_index": task_index,
            "operator_family": expected_family,
            "base_template_id": row["base_task_template_id"],
            "forward_bddl_path": str(bddl.resolve()),
            "forward_bddl_sha256": sha256_file(bddl),
            "forward_init_path": str(init.resolve()),
            "forward_init_sha256": sha256_file(init),
            "reverse_bddl_path": str(output.resolve()),
            "reverse_bddl_sha256": sha256_file(output),
            "reverse_initial_predicates": compiled["reverse_initial_predicates"],
            "reverse_goal_predicates": compiled["reverse_goal_predicates"],
            "invariants": compiled["invariants"],
            "reverse_instruction": compiled["reverse_instruction"],
            "source_region_invariants": {"forward_goal_predicates": legacy.parse_public_predicates(bddl.read_text(encoding="utf-8"), "goal") and [p.render() for p in legacy.parse_public_predicates(bddl.read_text(encoding="utf-8"), "goal")]},
            "generator_provenance": {"adapter": "track3_reverse_semantic_adapter", "public_rule_source": "KE7/libero-infinity task_reverser plus locked public LIBERO BDDL", "source_revision": "dc5451ff80efa33ef170350188a6c3457fbd7a81", "hidden_track3_details_used": False},
            "official_track3_claim": False,
            "identity": "generator_reference_only",
            "denylist_overlap": False,
        })
    summary = {"schema_version": 1, "status": "s3_reverse_semantic_adapter_passed", "task_order": [row["task_index"] for row in specs], "families": [row["operator_family"] for row in specs], "spec_count": len(specs), "predicate_nontrivial_static": all(row["reverse_initial_predicates"] != row["reverse_goal_predicates"] for row in specs), "official_track3_claim": False, "denylist_overlap_count": 0, "specs": specs}
    write_json(SUMMARY, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("use --build")
    print(json.dumps(build_specs(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
