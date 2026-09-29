#!/usr/bin/env python3
"""Freeze the accepted Fast400 Track3 gold into a new 80/20 namespace.

This is deliberately a new namespace.  It reads accepted raw episodes and the
locked forward catalog, materializes a pre-action reverse repository, builds a
weighted manifest, and computes QUANTILES without touching the historical
four_gold_80_20_v1 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import build_track3_four_gold_mix_v1 as legacy_builder
from . import compute_track3_mix_norm_v1 as norm_tool
from . import materialize_track3_reverse_gold_v1 as materializer
from .common import ROOT, canonical_json, sha256_file, sha256_tree, write_json


RAW_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted"
REVERSE_ROOT = ROOT / "data/processed/axis_pi05_track3_reverse_gold_fast400_20260913"
MANIFEST_ROOT = ROOT / "data/manifests_track3/fast400_gold_80_20_20260913"
EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/track3_fast400_manifest_freeze_20260913"
REVERSE_REPO = "axis-pi05/track3_reverse_gold_fast400_20260913"
REVERSE_REVISION = "track3-accepted-pre-action-fast400-20260913"

# Old accepted demos remain eligible, and the newly accepted variants are
# included for diversity.  The raw directories are immutable inputs.
SELECTION = (
    "official_t3_alphabet_soup_1788946170",
    "official_t3_alphabet_soup_1789306058",
    "official_t3_red_mug_1788976069",
    "official_t3_red_mug_1788921471",
    "official_t3_two_mugs_1788976820",
    "official_t3_yellow_book_1788971728",
    "official_t3_yellow_book_1789306339",
    "official_t3_yellow_book_1789306981",
)

BASE_SPECS: dict[str, dict[str, Any]] = {
    "official_t3_alphabet_soup": {
        "prompt": "take the alphabet soup out of the tray and place it on the table in front of the ketchup",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_alphabet_soup_out_of_the_tray_and_place_it_on_the_table.bddl",
        "seed": (1.0, -1.0),
        "family": "containment_removal",
    },
    "official_t3_red_mug": {
        "prompt": "take the red mug off the left plate and put it back on the table behind the other two mugs",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_red_mug_off_the_left_plate_and_put_it_on_the_table.bddl",
        "seed": (1.0, -1.0),
        "family": "support_removal",
    },
    "official_t3_two_mugs": {
        "prompt": "take the white mug off the left plate and the yellow and white mug off the right plate, then place the white mug to the left of the red mug and the yellow and white mug to the right of the red mug",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_two_mugs_off_the_plates_and_place_them_on_the_table.bddl",
        "seed": (-1.0, 1.0),
        "family": "multi_object_support_removal",
    },
    "official_t3_yellow_book": {
        "prompt": "take the yellow book off the top of the shelf and put it back on the table on the left",
        "bddl": "compe/t3/assets/bddl_files/libero_t3/take_the_yellow_book_off_the_shelf_and_put_it_on_the_table.bddl",
        "seed": (1.0, -1.0),
        "family": "support_removal",
    },
}


def base_task(episode: str) -> str:
    for task_id in BASE_SPECS:
        if episode.startswith(task_id + "_"):
            return task_id
    raise ValueError(f"unknown accepted episode: {episode}")


def strict_json(path: Path) -> dict[str, Any]:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            out[key] = value
        return out

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def configure_materializer() -> None:
    if REVERSE_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite reverse freeze: {REVERSE_ROOT}")
    if MANIFEST_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite manifest freeze: {MANIFEST_ROOT}")
    if EVIDENCE_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite freeze evidence: {EVIDENCE_ROOT}")

    task_specs: dict[str, dict[str, Any]] = {}
    seeds: dict[str, tuple[float, float]] = {}
    for task_index, episode in enumerate(SELECTION):
        task_id = base_task(episode)
        spec = BASE_SPECS[task_id]
        task_specs[episode] = {
            "episode": episode,
            "task_index": task_index,
            "prompt": spec["prompt"],
            "bddl": spec["bddl"],
        }
        seeds[episode] = tuple(spec["seed"])
    # The materializer's overlay path is keyed by the semantic task name.
    # Keep aliases for its original configure_replay implementation while
    # using raw episode IDs as the unique output task IDs.
    for task_id, spec in BASE_SPECS.items():
        task_specs[task_id] = {
            "episode": SELECTION[0],
            "task_index": 0,
            "prompt": spec["prompt"],
            "bddl": spec["bddl"],
        }
        seeds[task_id] = tuple(spec["seed"])

    original_configure = materializer.configure_replay
    original_tasks = materializer.TASKS
    materializer.RAW_ROOT = RAW_ROOT
    materializer.TASKS = task_specs
    materializer.GRIPPER_CONTROLLER_SEEDS = seeds
    materializer.SOURCE_REVISION = REVERSE_REVISION

    def configure(recorder: Any, task_id: str) -> tuple[Path | None, list[float]]:
        return original_configure(recorder, base_task(task_id))

    materializer.configure_replay = configure
    results = [materializer.replay_one(episode, output_root=REVERSE_ROOT) for episode in SELECTION]
    materializer.write_repo(results, REVERSE_ROOT)
    materializer.configure_replay = original_configure
    materializer.TASKS = original_tasks


def reverse_rows(reverse_root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    info = strict_json(reverse_root / "meta/info.json")
    provenance = strict_json(reverse_root / "provenance.json")
    revision = (reverse_root / ".axis_pi05_revision").read_text(encoding="utf-8").strip()
    if revision != REVERSE_REVISION:
        raise ValueError(f"reverse revision mismatch: {revision!r}")
    episodes = provenance.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(SELECTION):
        raise ValueError("reverse provenance episode count mismatch")
    if int(info.get("total_episodes", -1)) != len(SELECTION):
        raise ValueError("reverse info episode count mismatch")
    task_rows = pq.read_table(reverse_root / "meta/tasks.parquet").to_pylist()
    episode_rows = pq.read_table(reverse_root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    if len(task_rows) != len(SELECTION) or len(episode_rows) != len(SELECTION):
        raise ValueError("reverse metadata row count mismatch")
    materializer_hash = str(provenance.get("materializer_sha256"))
    rows: list[dict[str, Any]] = []
    for index, (task_row, episode_row, provenance_row) in enumerate(zip(task_rows, episode_rows, episodes, strict=True)):
        task_id = str(provenance_row["task_id"])
        if task_id != SELECTION[index]:
            raise ValueError(f"reverse order mismatch at {index}: {task_id}")
        data_path = reverse_root / f"data/chunk-000/file-{index:03d}.parquet"
        data = pq.read_table(data_path, columns=["episode_index", "frame_index", "observation.state", "action"])
        length = int(episode_row["length"])
        if data.num_rows != length:
            raise ValueError(f"reverse frame count mismatch for {task_id}")
        if np.asarray(data["observation.state"].to_pylist(), dtype=np.float32).shape != (length, 8):
            raise ValueError(f"reverse state shape mismatch for {task_id}")
        if np.asarray(data["action"].to_pylist(), dtype=np.float32).shape != (length, 7):
            raise ValueError(f"reverse action shape mismatch for {task_id}")
        family = BASE_SPECS[base_task(task_id)]["family"]
        rows.append({
            "source_repo": REVERSE_REPO,
            "revision": REVERSE_REVISION,
            "source": "track3_reverse",
            "suite": "track3_reverse",
            "task": str(task_row["task"]),
            "task_id": task_id,
            "task_index": index,
            "source_episode_index": index,
            "source_episode_id": str(provenance_row["source_episode"]),
            "source_file": str(provenance_row["source_episode"]),
            "dataset_root": str(reverse_root.resolve()),
            "frame_count": length,
            "source_fps": 20,
            "state_dim": 8,
            "action_dim": 7,
            "cameras": ["front", "wrist"],
            "perturb_category": family,
            "family": family,
            "identity": "accepted_reverse_pre_action_fast400",
            "raw_trajectory_sha256": str(provenance_row["raw_trajectory_sha256"]),
            "raw_metadata_sha256": str(provenance_row["raw_metadata_sha256"]),
            "start_state_sha256": str(provenance_row["start_state_sha256"]),
            "materializer_sha256": materializer_hash,
        })
    return rows


def build_manifest() -> dict[str, Any]:
    catalog = ROOT / "data/processed/libero_axis_mix_v1/catalog.jsonl"
    plus, non_plus = legacy_builder._load_forward(catalog)
    train, validation, exceptions, grouping_audit = legacy_builder.split_rows_v2(plus, 1000, 0.10)
    if exceptions:
        raise ValueError(f"unexpected forward split exceptions: {exceptions}")
    reverse = reverse_rows(REVERSE_ROOT)
    forward_weights = legacy_builder._forward_weight_rows(train)
    family_counts: dict[str, int] = {}
    for row in reverse:
        family_counts[str(row["family"])] = family_counts.get(str(row["family"]), 0) + 1
    reverse_weights: list[dict[str, Any]] = []
    for row in reverse:
        family = str(row["family"])
        probability = 0.2 / 3.0 / family_counts[family]
        observed = int(row["frame_count"])
        reverse_weights.append({
            **row,
            "mix_source": "reverse",
            "mass_group": "reverse_20",
            "episode_sampling_probability": probability,
            "sampling_probability": probability,
            "sampling_weight": probability,
            "observed_frame_count": observed,
            "frame_weight": probability / observed,
            "effective_exposures": probability * len(reverse),
        })
    sampling = forward_weights + reverse_weights
    forward_mass = sum(float(row["episode_sampling_probability"]) for row in forward_weights)
    reverse_mass = sum(float(row["episode_sampling_probability"]) for row in reverse_weights)
    frame_mass = sum(float(row["frame_weight"]) * int(row["observed_frame_count"]) for row in sampling)
    family_mass = {family: sum(float(row["episode_sampling_probability"]) for row in reverse_weights if row["family"] == family) for family in sorted(family_counts)}
    if not math.isclose(forward_mass, 0.8, abs_tol=1e-10) or not math.isclose(reverse_mass, 0.2, abs_tol=1e-10) or not math.isclose(frame_mass, 1.0, abs_tol=1e-10):
        raise ValueError(f"mass check failed: {forward_mass=} {reverse_mass=} {frame_mass=}")
    forward_root = ROOT / "data/raw/hf/lerobot/libero_plus"
    forward_info = strict_json(forward_root / "meta/info.json")
    forward_identity = {
        "root": str(forward_root.resolve()),
        "source_repo": legacy_builder.FORWARD_REPO,
        "revision": legacy_builder.FORWARD_REVISION,
        "episodes": int(forward_info["total_episodes"]),
        "frames": int(forward_info["total_frames"]),
        "fps": 20,
        "state_dim": 8,
        "action_dim": 7,
        "info_sha256": sha256_file(forward_root / "meta/info.json"),
        "stats_sha256": sha256_file(forward_root / "meta/stats.json"),
        "tasks_sha256": sha256_file(forward_root / "meta/tasks.parquet"),
        "revision_marker_sha256": sha256_file(forward_root / ".axis_pi05_revision"),
        "readonly": True,
    }
    reverse_provenance = REVERSE_ROOT / "provenance.json"
    source_roots = {
        "schema_version": "track3_fast400_gold_source_roots_v1",
        "lerobot/libero_plus": forward_identity,
        REVERSE_REPO: {
            "root": str(REVERSE_ROOT.resolve()),
            "source_repo": REVERSE_REPO,
            "revision": REVERSE_REVISION,
            "episodes": len(reverse),
            "frames": sum(int(row["frame_count"]) for row in reverse),
            "fps": 20,
            "state_dim": 8,
            "action_dim": 7,
            "provenance_sha256": sha256_file(reverse_provenance),
            "tree_sha256": sha256_tree(REVERSE_ROOT),
            "readonly": True,
        },
    }
    source_roots_sha = hashlib.sha256(canonical_json(source_roots)).hexdigest()
    materializer_sha = sha256_file(ROOT / "tools/axis_pi05/materialize_track3_reverse_gold_v1.py")
    driver_sha = sha256_file(Path(__file__))
    q0 = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q0_input_inventory.json"
    manifest = {
        "schema_version": "track3_fast400_gold_80_20_manifest_v1",
        "mix_id": "fast400_gold_80_20_20260913",
        "mix_name": "libero_plus_forward_80_reverse_accepted_fast400_20",
        "catalog_path": str(catalog.resolve()),
        "catalog_sha256": sha256_file(catalog),
        "catalog_source_filter": {"source_repo": legacy_builder.FORWARD_REPO, "excluded_catalog_rows": len(non_plus)},
        "forward_source_repo": legacy_builder.FORWARD_REPO,
        "forward_revision": legacy_builder.FORWARD_REVISION,
        "forward_mix": "M1",
        "split_seed": 1000,
        "validation_fraction": 0.10,
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
        "reverse_task_count": len({base_task(str(row["task_id"])) for row in reverse}),
        "reverse_selection": list(SELECTION),
        "forward_mass": forward_mass,
        "reverse_mass": reverse_mass,
        "reverse_family_mass": family_mass,
        "episode_probability_sum": forward_mass + reverse_mass,
        "frame_weight_mass_sum": frame_mass,
        "sampling_rows": len(sampling),
        "source_roots_path": "source_roots.json",
        "source_roots_sha256": source_roots_sha,
        "q0_inventory_path": str(q0.relative_to(ROOT)),
        "q0_inventory_sha256": sha256_file(q0),
        "materializer_sha256": materializer_sha,
        "freeze_driver_sha256": driver_sha,
        "status": "fast400_manifest_quantiles_frozen_t3_not_started",
    }
    legacy_builder._write_outputs(MANIFEST_ROOT, train, validation, reverse, sampling, manifest, source_roots)
    write_json(EVIDENCE_ROOT / "accepted_selection.json", {
        "schema_version": "track3_fast400_accepted_selection_v1",
        "status": "accepted_8_episodes_4_tasks",
        "operator": "user-explicit-accept",
        "accepted_episode_count": len(SELECTION),
        "accepted_task_count": len({base_task(episode) for episode in SELECTION}),
        "episodes": list(SELECTION),
        "raw_root": str(RAW_ROOT.resolve()),
        "raw_immutable": True,
        "old_demos_retained": True,
    })
    return manifest


def compute_quantiles() -> dict[str, Any]:
    summary = norm_tool.compute(
        manifest_path=MANIFEST_ROOT / "sampling_weights.parquet",
        source_roots_path=MANIFEST_ROOT / "source_roots.json",
        output_root=MANIFEST_ROOT,
        evidence_root=EVIDENCE_ROOT,
    )
    provenance_path = MANIFEST_ROOT / "norm_stats.provenance.json"
    provenance = strict_json(provenance_path)
    provenance["schema_version"] = "track3_fast400_norm_provenance_v1"
    provenance["mix_id"] = "fast400_gold_80_20_20260913"
    provenance["freeze_driver_sha256"] = sha256_file(Path(__file__))
    write_json(provenance_path, provenance)
    audit_path = MANIFEST_ROOT / "sampler_audit.json"
    audit = strict_json(audit_path)
    audit["schema_version"] = "track3_fast400_sampler_audit_v1"
    write_json(audit_path, audit)
    summary["schema_version"] = "track3_fast400_quantiles_summary_v1"
    summary["status"] = "fast400_manifest_quantiles_frozen_t3_not_started"
    summary["mix_id"] = "fast400_gold_80_20_20260913"
    summary["freeze_driver_sha256"] = sha256_file(Path(__file__))
    write_json(EVIDENCE_ROOT / "q3_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--quantiles", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run_all = args.all or not any((args.materialize, args.build, args.quantiles))
    if run_all or args.materialize:
        configure_materializer()
    if run_all or args.build:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        build_manifest()
    if run_all or args.quantiles:
        print(json.dumps(compute_quantiles(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
