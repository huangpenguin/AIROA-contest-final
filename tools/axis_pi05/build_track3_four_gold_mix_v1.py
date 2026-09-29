#!/usr/bin/env python3
"""Build the locked LIBERO-plus 80% + reverse-gold 20% Q2 manifest.

The forward source is filtered by its exact LeRobot repository identity before
the existing deterministic v2 split and M1 stratification are applied.  The
reverse source is the locally materialized pre-action repository and receives
explicit family-balanced episode mass.  All writes occur in a new temporary
namespace and the final directory is atomically renamed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .build_mix_v2 import make_weights_v2, split_rows_v2
from .common import ROOT, canonical_json, read_jsonl, sha256_bytes, sha256_file, sha256_tree, write_csv, write_json, write_table


FORWARD_REPO = "lerobot/libero_plus"
FORWARD_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
REVERSE_REPO = "axis-pi05/track3_reverse_gold_v1"
REVERSE_REVISION = "track3-accepted-pre-action-v1"
FORWARD_MASS = 0.8
REVERSE_MASS = 0.2
SPLIT_SEED = 1000
VALIDATION_FRACTION = 0.10
FORWARD_MIX = "M1"

REVERSE_FAMILY_BY_TASK = {
    "official_t3_red_mug": "support_removal",
    "official_t3_yellow_book": "support_removal",
    "official_t3_alphabet_soup": "containment_removal",
    "official_t3_two_mugs": "multi_object_support_removal",
}
REVERSE_EPISODE_ORDER = (
    "official_t3_alphabet_soup",
    "official_t3_red_mug",
    "official_t3_two_mugs",
    "official_t3_yellow_book",
)


def strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_forward(catalog_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(catalog_path)
    plus = [row for row in rows if row.get("source_repo") == FORWARD_REPO]
    _require(len(plus) == 14347, f"expected 14347 Plus episodes, got {len(plus)}")
    _require(sum(int(row["frame_count"]) for row in plus) == 2238036, "Plus frame count mismatch")
    for row in plus:
        _require(row.get("revision") == FORWARD_REVISION, "Plus revision mismatch")
        _require(row.get("source_repo") == FORWARD_REPO, "non-canonical Plus repository")
        _require(int(row.get("source_fps", 0)) == 20, "Plus source fps mismatch")
        _require(int(row.get("state_dim", 0)) == 8 and int(row.get("action_dim", 0)) == 7, "Plus state/action shape mismatch")
        _require(str(row.get("source")) == "libero_plus", "Plus source label mismatch")
    non_plus = [row for row in rows if row.get("source_repo") != FORWARD_REPO]
    return plus, non_plus


def _load_reverse(reverse_root: Path, materializer_path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    info = strict_json(reverse_root / "meta/info.json")
    provenance = strict_json(reverse_root / "provenance.json")
    revision = (reverse_root / ".axis_pi05_revision").read_text(encoding="utf-8").strip()
    _require(revision == REVERSE_REVISION, f"reverse revision mismatch: {revision!r}")
    _require(provenance.get("total_frames") == 1060, "reverse total frame count mismatch")
    _require(info.get("total_episodes") == 4 and info.get("total_frames") == 1060, "reverse info counts mismatch")
    materializer_sha256 = sha256_file(materializer_path)
    _require(materializer_sha256 == provenance.get("materializer_sha256"), "materializer/provenance hash mismatch")

    tasks = pq.read_table(reverse_root / "meta/tasks.parquet").to_pylist()
    episode_table = pq.read_table(reverse_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    _require(len(tasks) == 4 and len(episode_table) == 4, "reverse metadata episode count mismatch")
    provenance_by_task = {str(row["task_id"]): row for row in provenance.get("episodes", [])}
    _require(set(provenance_by_task) == set(REVERSE_EPISODE_ORDER), "reverse provenance task set mismatch")

    rows: list[dict[str, Any]] = []
    for episode_index, task_id in enumerate(REVERSE_EPISODE_ORDER):
        task_row = tasks[episode_index]
        episode_row = episode_table[episode_index]
        _require(int(task_row["task_index"]) == episode_index, "reverse task index mismatch")
        length = int(episode_row["length"])
        data_path = reverse_root / f"data/chunk-000/file-{episode_index:03d}.parquet"
        data = pq.read_table(data_path, columns=["episode_index", "frame_index", "observation.state", "action"])
        _require(data.num_rows == length, f"reverse parquet frame count mismatch for {task_id}")
        _require(np.asarray(data["observation.state"].to_pylist(), dtype=np.float32).shape == (length, 8), f"reverse state shape mismatch for {task_id}")
        _require(np.asarray(data["action"].to_pylist(), dtype=np.float32).shape == (length, 7), f"reverse action shape mismatch for {task_id}")
        provenance_row = provenance_by_task[task_id]
        _require(provenance_row.get("final_success") is True, f"reverse diagnostic success missing for {task_id}")
        prompt = str(task_row["task"])
        rows.append({
            "source_repo": REVERSE_REPO,
            "revision": REVERSE_REVISION,
            "source": "track3_reverse",
            "suite": "track3_reverse",
            "task": prompt,
            "task_id": task_id,
            "task_index": episode_index,
            "source_episode_index": episode_index,
            "source_episode_id": provenance_row["source_episode"],
            "source_file": provenance_row["source_episode"],
            "dataset_root": str(reverse_root.resolve()),
            "frame_count": length,
            "source_fps": 20,
            "state_dim": 8,
            "action_dim": 7,
            "cameras": ["front", "wrist"],
            "perturb_category": REVERSE_FAMILY_BY_TASK[task_id],
            "family": REVERSE_FAMILY_BY_TASK[task_id],
            "identity": "accepted_reverse_pre_action",
            "raw_trajectory_sha256": provenance_row["raw_trajectory_sha256"],
            "raw_metadata_sha256": provenance_row["raw_metadata_sha256"],
            "start_state_sha256": provenance_row["start_state_sha256"],
            "materializer_sha256": materializer_sha256,
        })
    return rows


def _forward_weight_rows(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = make_weights_v2(train_rows, FORWARD_MIX)
    result: list[dict[str, Any]] = []
    for row in base:
        probability = float(row["episode_sampling_probability"]) * FORWARD_MASS
        observed = int(row["observed_frame_count"])
        result.append({
            **row,
            "mix_source": "forward",
            "mass_group": "forward_80",
            "family": "forward_libero_plus",
            "episode_sampling_probability": probability,
            "sampling_probability": probability,
            "sampling_weight": probability,
            "observed_frame_count": observed,
            "frame_weight": probability / observed,
            "effective_exposures": probability * len(train_rows),
        })
    return result


def _reverse_weight_rows(reverse_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_count = Counter(str(row["family"]) for row in reverse_rows)
    result: list[dict[str, Any]] = []
    for row in reverse_rows:
        family = str(row["family"])
        probability = REVERSE_MASS / 3.0 / family_count[family]
        observed = int(row["frame_count"])
        result.append({
            **row,
            "mix_source": "reverse",
            "mass_group": "reverse_20",
            "episode_sampling_probability": probability,
            "sampling_probability": probability,
            "sampling_weight": probability,
            "observed_frame_count": observed,
            "frame_weight": probability / observed,
            "effective_exposures": probability * len(reverse_rows),
        })
    return result


def _write_outputs(
    output_root: Path,
    forward_train: list[dict[str, Any]],
    forward_validation: list[dict[str, Any]],
    reverse_rows: list[dict[str, Any]],
    sampling_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    source_roots: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Q2 output: {output_root}")
    temporary = output_root.with_name(output_root.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to reuse Q2 partial output: {temporary}")
    try:
        temporary.mkdir(parents=True)
        write_table(temporary / "forward_train_episodes.parquet", forward_train)
        write_table(temporary / "forward_validation_episodes.parquet", forward_validation)
        write_table(temporary / "reverse_train_episodes.parquet", reverse_rows)
        write_table(temporary / "sampling_weights.parquet", sampling_rows)
        distribution = [
            {"scope": "forward", "family": "forward_libero_plus", "episode_count": len(forward_train), "frame_count": sum(int(row["frame_count"]) for row in forward_train), "mass": FORWARD_MASS},
        ]
        for family in sorted({str(row["family"]) for row in reverse_rows}):
            family_rows = [row for row in reverse_rows if row["family"] == family]
            distribution.append({"scope": "reverse", "family": family, "episode_count": len(family_rows), "frame_count": sum(int(row["frame_count"]) for row in family_rows), "mass": REVERSE_MASS / 3.0})
        write_csv(temporary / "distribution_before_after.csv", distribution, ["scope", "family", "episode_count", "frame_count", "mass"])
        write_json(temporary / "source_roots.json", source_roots)
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "manifest.sha256", {"sha256": sha256_file(temporary / "manifest.json")})
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build(config_path: Path) -> dict[str, Any]:
    config = strict_json(config_path) if config_path.suffix == ".json" else None
    if config is None:
        import yaml

        config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config_value, dict):
            raise ValueError(f"Q2 config must be a mapping: {config_path}")
        config = config_value
    catalog_path = Path(str(config["catalog_path"]))
    reverse_root = Path(str(config["reverse_root"]))
    output_root = Path(str(config["output_root"]))
    evidence_root = Path(str(config["evidence_root"]))
    materializer_path = ROOT / "tools/axis_pi05/materialize_track3_reverse_gold_v1.py"
    plus, non_plus = _load_forward(catalog_path)
    train, validation, exceptions, grouping_audit = split_rows_v2(plus, SPLIT_SEED, VALIDATION_FRACTION)
    _require(not exceptions, f"unexpected Plus split exceptions: {exceptions}")
    train_ids = {(row["source_repo"], row["revision"], int(row["source_episode_index"])) for row in train}
    validation_ids = {(row["source_repo"], row["revision"], int(row["source_episode_index"])) for row in validation}
    _require(not train_ids & validation_ids, "forward train/validation overlap")
    _require(len(train_ids | validation_ids) == len(plus), "forward split lost episode identities")
    reverse = _load_reverse(reverse_root, materializer_path)
    forward_weights = _forward_weight_rows(train)
    reverse_weights = _reverse_weight_rows(reverse)
    sampling_rows = forward_weights + reverse_weights

    forward_mass = sum(float(row["episode_sampling_probability"]) for row in forward_weights)
    reverse_mass = sum(float(row["episode_sampling_probability"]) for row in reverse_weights)
    frame_mass = sum(float(row["frame_weight"]) * int(row["observed_frame_count"]) for row in sampling_rows)
    _require(math.isclose(forward_mass, FORWARD_MASS, rel_tol=1e-10, abs_tol=1e-10), f"forward mass mismatch: {forward_mass}")
    _require(math.isclose(reverse_mass, REVERSE_MASS, rel_tol=1e-10, abs_tol=1e-10), f"reverse mass mismatch: {reverse_mass}")
    _require(math.isclose(forward_mass + reverse_mass, 1.0, rel_tol=1e-10, abs_tol=1e-10), "total episode mass mismatch")
    _require(math.isclose(frame_mass, 1.0, rel_tol=1e-10, abs_tol=1e-10), f"frame mass mismatch: {frame_mass}")
    family_mass = {family: sum(float(row["episode_sampling_probability"]) for row in reverse_weights if row["family"] == family) for family in sorted({str(row["family"]) for row in reverse_weights})}
    _require(all(math.isclose(value, REVERSE_MASS / 3.0, rel_tol=1e-10, abs_tol=1e-10) for value in family_mass.values()), f"reverse family mass mismatch: {family_mass}")

    q0_path = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q0_input_inventory.json"
    reverse_provenance = reverse_root / "provenance.json"
    forward_info = strict_json(ROOT / "data/raw/hf/lerobot/libero_plus/meta/info.json")
    forward_identity = {
        "root": str((ROOT / "data/raw/hf/lerobot/libero_plus").resolve()),
        "source_repo": FORWARD_REPO,
        "revision": FORWARD_REVISION,
        "episodes": 14347,
        "frames": 2238036,
        "fps": 20,
        "state_dim": 8,
        "action_dim": 7,
        "info_sha256": sha256_file(ROOT / "data/raw/hf/lerobot/libero_plus/meta/info.json"),
        "stats_sha256": sha256_file(ROOT / "data/raw/hf/lerobot/libero_plus/meta/stats.json"),
        "tasks_sha256": sha256_file(ROOT / "data/raw/hf/lerobot/libero_plus/meta/tasks.parquet"),
        "revision_marker_sha256": sha256_file(ROOT / "data/raw/hf/lerobot/libero_plus/.axis_pi05_revision"),
        "readonly": True,
    }
    _require(int(forward_info.get("total_episodes", -1)) == 14347 and int(forward_info.get("total_frames", -1)) == 2238036, "forward info identity mismatch")
    source_roots = {
        "schema_version": "track3_four_gold_source_roots_v1",
        "lerobot/libero_plus": forward_identity,
        REVERSE_REPO: {
            "root": str(reverse_root.resolve()),
            "source_repo": REVERSE_REPO,
            "revision": REVERSE_REVISION,
            "episodes": 4,
            "frames": 1060,
            "fps": 20,
            "state_dim": 8,
            "action_dim": 7,
            "provenance_sha256": sha256_file(reverse_provenance),
            "tree_sha256": sha256_tree(reverse_root),
            "readonly": True,
        },
    }
    source_roots_sha256 = sha256_bytes(canonical_json(source_roots))
    manifest = {
        "schema_version": "track3_four_gold_80_20_manifest_v1",
        "mix_id": "four_gold_80_20_v1",
        "mix_name": "libero_plus_forward_80_reverse_four_gold_20",
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "catalog_source_filter": {"source_repo": FORWARD_REPO, "excluded_catalog_rows": len(non_plus)},
        "forward_source_repo": FORWARD_REPO,
        "forward_revision": FORWARD_REVISION,
        "forward_mix": FORWARD_MIX,
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "split_unit": "source_repo/revision/source_episode_index",
        "grouping_policy": "episode_only",
        "forward_episode_count": len(plus),
        "forward_train_episode_count": len(train),
        "forward_validation_episode_count": len(validation),
        "forward_frame_count": sum(int(row["frame_count"]) for row in plus),
        "forward_train_frame_count": sum(int(row["frame_count"]) for row in train),
        "forward_validation_frame_count": sum(int(row["frame_count"]) for row in validation),
        "reverse_source_repo": REVERSE_REPO,
        "reverse_revision": REVERSE_REVISION,
        "reverse_episode_count": len(reverse),
        "reverse_frame_count": sum(int(row["frame_count"]) for row in reverse),
        "reverse_evaluation_identity_count": 0,
        "reverse_evaluation_overlap": "not_applicable_no_eval_identities_created_in_q2",
        "forward_mass": forward_mass,
        "reverse_mass": reverse_mass,
        "reverse_family_mass": family_mass,
        "episode_probability_sum": forward_mass + reverse_mass,
        "frame_weight_mass_sum": frame_mass,
        "sampling_rows": len(sampling_rows),
        "source_roots_path": "source_roots.json",
        "source_roots_sha256": source_roots_sha256,
        "q0_inventory_path": str(q0_path.relative_to(ROOT)),
        "q0_inventory_sha256": sha256_file(q0_path),
        "reverse_provenance_sha256": sha256_file(reverse_provenance),
        "materializer_sha256": sha256_file(materializer_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "output_files": {
            "forward_train": "forward_train_episodes.parquet",
            "forward_validation": "forward_validation_episodes.parquet",
            "reverse_train": "reverse_train_episodes.parquet",
            "sampling_weights": "sampling_weights.parquet",
            "distribution": "distribution_before_after.csv",
            "source_roots": "source_roots.json",
        },
        "hard_gates": {
            "forward_source_identity_exact": True,
            "forward_train_validation_overlap_zero": True,
            "forward_mass_exact_0_8": math.isclose(forward_mass, FORWARD_MASS, rel_tol=1e-10, abs_tol=1e-10),
            "reverse_mass_exact_0_2": math.isclose(reverse_mass, REVERSE_MASS, rel_tol=1e-10, abs_tol=1e-10),
            "reverse_family_mass_equal": all(math.isclose(value, REVERSE_MASS / 3.0, rel_tol=1e-10, abs_tol=1e-10) for value in family_mass.values()),
            "frame_weight_mass_exact_one": math.isclose(frame_mass, 1.0, rel_tol=1e-10, abs_tol=1e-10),
            "reverse_train_eval_overlap_zero": True,
        },
        "grouping_audit": grouping_audit,
        "status": "q2_manifest_built_ready_for_q3",
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    # The builder intentionally writes Q2 outputs below the requested output
    # root.  Evidence mirrors only immutable identity and result metadata.
    _write_outputs(output_root, train, validation, reverse, sampling_rows, manifest, source_roots)
    manifest["source_roots_sha256"] = sha256_file(output_root / "source_roots.json")
    manifest["manifest_sha256"] = sha256_file(output_root / "manifest.json")
    # Keep the original manifest atomically generated, then expose the final
    # source-root and manifest hashes in a separate evidence record.
    write_json(evidence_root / "q2_summary.json", {
        "schema_version": "track3_four_gold_q2_summary_v1",
        "status": "q2_manifest_built_ready_for_q3",
        "output_root": str(output_root.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_roots_sha256": manifest["source_roots_sha256"],
        "forward_episodes": len(plus),
        "forward_train_episodes": len(train),
        "forward_validation_episodes": len(validation),
        "forward_frames": 2238036,
        "reverse_episodes": len(reverse),
        "reverse_frames": 1060,
        "forward_mass": forward_mass,
        "reverse_mass": reverse_mass,
        "reverse_family_mass": family_mass,
        "sampling_weight_sum": forward_mass + reverse_mass,
        "frame_weight_mass_sum": frame_mass,
        "excluded_catalog_rows": len(non_plus),
        "q2_only": True,
        "q3_started": False,
    })
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/track3_four_gold_mix_v1_20260910.yaml")
    args = parser.parse_args()
    manifest = build(args.config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
