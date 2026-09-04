from __future__ import annotations

import json
import math
from pathlib import Path

from tools.axis_pi05 import track3_background_predicate_convergence as release


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence"


def test_k0_closes_exactly_the_registered_fifteen_raw_reports() -> None:
    manifest = json.loads((OUT / "k0_raw_manifest.json").read_text(encoding="utf-8"))
    records = manifest["raw_reports"]
    identities = {tuple(row[key] for key in ("task_index", "kind", "run_index")) for row in records}
    assert len(records) == 15
    assert identities == release.expected_identities()
    assert sum(row["bytes"] for row in records) == 16_072_261
    assert manifest["manifest_canonical_sha256"] == release.canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_canonical_sha256"}
    )
    assert all(row["relative_path"].startswith("data/reports/axis_libero_30k/track3_operator_scoped_stability/runs/j2_task") for row in records)


def test_k0_uses_the_explicit_j2_manifest_list_not_a_glob() -> None:
    paths = release.raw_paths_from_prior()
    assert len(paths) == 15
    assert {path.name for path in paths} == {
        f"j2_task{task}_{kind}_run_{run}.json"
        for task, kind, run in release.expected_identities()
    }


def test_k0_preserves_old_i2_and_previous_exact_vector() -> None:
    lock = json.loads((OUT / "k0_input_lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "track3_background_convergence_input_locked"
    assert lock["old_i2_sha256"] == release.OLD_I2_SHA256
    assert lock["prior_j2_exact_vector"]["reverse_pass"] == 8
    assert lock["prior_j2_exact_vector"]["failed_run"] == [
        {"failed_gates": ["truth_vector_no_transition"], "kind": "reverse", "run_index": 1, "task_index": 19}
    ]


def test_finite_tree_rejects_non_finite_numbers() -> None:
    assert release.finite_tree({"ok": [1.0, True]}) == []
    assert release.finite_tree({"bad": float("nan")}) == ["root.bad"]
    assert release.finite_tree({"bad": float("inf")}) == ["root.bad"]
