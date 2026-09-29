#!/usr/bin/env python3
"""T1 dedicated-panel P0/P1: constructibility, effect tests, and freeze.

Does not score R6 vs canary100, train, upload, or submit. Empty
TRACK_PERTURBATIONS is never treated as lighting/env. Missing Plus visual
assets are recorded as panel_not_constructible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, write_json
from .eval_manifest import case_id


JST = timezone(timedelta(hours=9))
EVIDENCE = ROOT / "data/reports/axis_libero_30k/track12_t1_dedicated_panel_20260911"
CSV_PATH = ROOT / "compe/t1/T1_TASKS.csv"
PLUS_ASSETS = ROOT / "LIBERO-plus/libero/libero/assets"
VENDOR_ASSETS = ROOT / "LIBERO/libero/libero/assets"
SUFFIX_RE = re.compile(r"_light_[^.]*|_table_\d+")
SCENE_XML_RE = re.compile(r'"scene_xml":\s*"([^"]+)"')
STYLE_RE = re.compile(r'"(floor_style|wall_style)":\s*"([^"]+)"')
CSV_FAMILY = {
    "Light Conditions": "lighting",
    "Background Textures": "env",
}


def now_jst() -> str:
    return datetime.now(JST).isoformat(timespec="seconds")


def _load_csv() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        raise ValueError(f"T1 example suite must be 4 rows, got {len(rows)}")
    return rows


def enumerate_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(_load_csv()):
        category = row["category"]
        if category not in CSV_FAMILY:
            raise ValueError(f"unexpected T1 category: {category}")
        family = CSV_FAMILY[category]
        bddl = (
            ROOT
            / "LIBERO-plus/libero/libero/bddl_files"
            / row["suite"]
            / f"{row['task_id']}.bddl"
        )
        base_stem = SUFFIX_RE.sub("", Path(row["task_id"]).name)
        base_bddl = bddl.with_name(f"{base_stem}.bddl")
        cases.append(
            {
                "task_num": int(row["task_num"]),
                "task_id": row["task_id"],
                "instruction": row["instruction"],
                "csv_category": category,
                "perturbation_family": family,
                "family_note": (
                    "mix label recovered-coarse-not-background-texture; example CSV writes Background Textures / _table_*"
                    if family == "env"
                    else "example CSV Light Conditions / _light_*"
                ),
                "suite": row["suite"],
                "bddl_path": str(bddl),
                "base_bddl_path": str(base_bddl),
                "init_state_index": 0,
                "env_seed": 1000 + index,
                "policy_seed": 0,
                "max_steps": 300,
                "control_freq": 20,
                "chunk_size": 10,
                "n_action_steps": 8,
            }
        )
    return cases


def _problem_name(bddl: Path) -> str:
    first = bddl.read_text(encoding="utf-8").splitlines()[0]
    match = re.search(r"\(define \(problem ([^)]+)\)", first)
    if not match:
        raise ValueError(f"cannot parse problem name: {bddl}")
    return match.group(1)


def _asset_candidates(relative: str) -> list[Path]:
    return [PLUS_ASSETS / relative, VENDOR_ASSETS / relative]


def _first_existing(relative: str) -> Path | None:
    for path in _asset_candidates(relative):
        if path.is_file() or path.is_dir():
            return path
    return None


_PLUS_ASSETS_BYTES: int | None = None


def _plus_assets_bytes() -> int:
    global _PLUS_ASSETS_BYTES
    if _PLUS_ASSETS_BYTES is None:
        if not PLUS_ASSETS.exists() or PLUS_ASSETS.is_symlink():
            _PLUS_ASSETS_BYTES = 0
        else:
            _PLUS_ASSETS_BYTES = sum(path.stat().st_size for path in PLUS_ASSETS.rglob("*") if path.is_file())
    return _PLUS_ASSETS_BYTES


def inspect_problem_assets(bddl: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "LIBERO-plus"))
    from libero.libero.envs import problems as _problems  # noqa: F401
    from libero.libero.envs.arenas.style import STYLE_MAPPING
    from libero.libero.envs.bddl_base_domain import TASK_MAPPING

    name = _problem_name(bddl)
    cls = TASK_MAPPING.get(name.lower())
    if cls is None:
        return {
            "problem_name": name,
            "class_found": False,
            "scene_xml": None,
            "scene_xml_exists": False,
            "missing_assets": [f"problem class {name}"],
        }
    source = inspect.getsource(cls.__init__)
    xml_match = SCENE_XML_RE.search(source)
    scene_xml = xml_match.group(1) if xml_match else None
    styles = dict(STYLE_RE.findall(source))
    missing: list[str] = []
    resolved_textures: dict[str, str] = {}
    scene_path = None
    if scene_xml:
        scene_path = _first_existing(scene_xml)
        if scene_path is None:
            missing.append(scene_xml)
    for key, style_name in styles.items():
        mapping_key = "floor" if key.startswith("floor") else "wall" if key.startswith("wall") else None
        filename = STYLE_MAPPING.get(mapping_key, {}).get(style_name) if mapping_key else None
        candidates = []
        if filename:
            candidates.append(f"textures/{filename}")
            resolved_textures[key] = filename
        candidates.extend([f"textures/{style_name}.png", f"textures/{style_name}"])
        if not any(_first_existing(rel) is not None for rel in candidates):
            missing.append(f"{key}={style_name}")
    return {
        "problem_name": name,
        "class_found": True,
        "class_name": cls.__name__,
        "scene_xml": scene_xml,
        "scene_xml_path": str(scene_path) if scene_path else None,
        "scene_xml_exists": scene_path is not None,
        "scene_properties": styles,
        "resolved_textures": resolved_textures,
        "missing_assets": missing,
        "plus_assets_bytes": _plus_assets_bytes(),
    }


def _init_tensor_sha(task_id: str, suite: str) -> tuple[str, int]:
    sys.path.insert(0, str(ROOT / "LIBERO-plus"))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".eval_libero"))
    from compe.t1.register import register_t1

    factory = register_t1()
    benchmark = factory() if inspect.isclass(factory) else factory
    names = [task.name for task in benchmark.tasks]
    if task_id not in names:
        raise ValueError(f"{task_id} not in libero_t1")
    index = names.index(task_id)
    states = benchmark.get_task_init_states(index)
    import numpy as np

    from .eval_manifest import _state_hash

    state = np.asarray(states[0])
    return _state_hash(state), int(len(states))


def _manifest_row(case: dict[str, Any], *, bddl: Path, family: str, init_sha: str) -> dict[str, Any]:
    row = {
        "schema_version": 1,
        "panel": "t1_dedicated",
        "selection_eligible": True,
        "benchmark": "libero_t1",
        "suite": "libero_t1",
        "task_id": int(case["task_num"]) - 1,
        "task_name": case["task_id"],
        "language": case["instruction"],
        "bddl_path": str(bddl),
        "bddl_sha256": sha256_file(bddl) if bddl.is_file() else "missing",
        "init_state_index": 0,
        "init_state_sha256": init_sha,
        "env_seed": int(case["env_seed"]),
        "policy_seed": 0,
        "max_steps": 300,
        "perturbation_id": family,
        "perturbation_family": family,
        "perturbation_params": {"source": "compe/t1/T1_TASKS.csv", "plus_problem_name": True},
        "repeat_id": 0,
    }
    row["case_id"] = case_id(row)
    return row


def effect_test_case(case: dict[str, Any]) -> dict[str, Any]:
    bddl = Path(case["bddl_path"])
    base = Path(case["base_bddl_path"])
    result = {
        **case,
        "bddl_exists": bddl.is_file(),
        "base_bddl_exists": base.is_file(),
        "empty_track_perturbations_not_used": True,
    }
    if not bddl.is_file():
        result.update({"effect_test": "failed", "status": "panel_not_constructible", "reason": f"missing BDDL {bddl}"})
        return result
    perturbed = inspect_problem_assets(bddl)
    base_assets = inspect_problem_assets(base) if base.is_file() else {"missing_assets": ["base bddl"], "scene_xml": None}
    result["perturbed_assets"] = perturbed
    result["base_assets"] = base_assets
    missing = list(perturbed.get("missing_assets") or [])
    if not base.is_file():
        missing.append(str(base))
    xml_changed = perturbed.get("scene_xml") and perturbed.get("scene_xml") != base_assets.get("scene_xml")
    styles_changed = perturbed.get("scene_properties") != base_assets.get("scene_properties")
    declared_delta = bool(xml_changed or styles_changed)
    if missing:
        result.update(
            {
                "effect_test": "failed",
                "status": "panel_not_constructible",
                "reason": "plus visual assets missing; lighting/table scene cannot be reconstructed",
                "missing_assets": missing,
                "declared_scene_delta": declared_delta,
                "simulator_hashes": "not_run",
            }
        )
        return result
    result.update(
        {
            "effect_test": "passed_files_present",
            "status": "assets_present_simulator_not_scored",
            "declared_scene_delta": declared_delta,
            "simulator_hashes": "not_run_until_assets_and_evalvenv_confirmed",
        }
    )
    return result


def run_p0(*, create_env: bool = False) -> dict[str, Any]:
    cases = enumerate_cases()
    reports = []
    families = {"lighting": True, "env": True}
    for case in cases:
        try:
            init_sha, n_init = _init_tensor_sha(case["task_id"], case["suite"])
        except Exception as exc:
            init_sha, n_init = f"unavailable:{type(exc).__name__}", 0
            case["init_error"] = f"{type(exc).__name__}: {exc}"
        row = None
        if Path(case["bddl_path"]).is_file() and not str(init_sha).startswith("unavailable"):
            row = _manifest_row(case, bddl=Path(case["bddl_path"]), family=case["perturbation_family"], init_sha=init_sha)
        report = effect_test_case(case)
        report["init_state_sha256"] = init_sha
        report["init_state_count"] = n_init
        report["manifest_row"] = row
        if report.get("status") != "assets_present_simulator_not_scored":
            families[case["perturbation_family"]] = False
        if create_env and report.get("status") == "assets_present_simulator_not_scored":
            report["simulator"] = _try_env(case)
            if report["simulator"].get("status") == "passed":
                report["effect_test"] = "passed"
                report["status"] = "constructible"
            else:
                report["effect_test"] = "failed"
                report["status"] = "panel_not_constructible"
                families[case["perturbation_family"]] = False
        reports.append(report)

    required_status = "constructible" if create_env else "assets_present_simulator_not_scored"
    constructible = all(families.values()) and all(item.get("status") == required_status for item in reports)
    freeze_status = "frozen_constructible" if constructible else "panel_not_constructible"
    if constructible:
        freeze_reason = None
    else:
        freeze_reason = "; ".join(
            f"{item['task_id']}:{item.get('reason') or item.get('effect_test') or item.get('status')}"
            for item in reports
            if item.get("status") != required_status
        ) or "one or more T1 example cases failed constructibility"
    freeze_cases = []
    for report in reports:
        freeze_cases.append(
            {
                "task_id": report["task_id"],
                "perturbation_family": report["perturbation_family"],
                "init_state_index": 0,
                "env_seed": report["env_seed"],
                "policy_seed": 0,
                "max_steps": 300,
                "bddl_path": report["bddl_path"],
                "bddl_sha256": sha256_file(Path(report["bddl_path"])) if Path(report["bddl_path"]).is_file() else None,
                "init_state_sha256": report.get("init_state_sha256"),
                "case_id": (report.get("manifest_row") or {}).get("case_id"),
                "status": freeze_status if constructible else "panel_not_constructible",
                "effect_test": report.get("effect_test"),
                "missing_assets": report.get("missing_assets") or report.get("perturbed_assets", {}).get("missing_assets"),
            }
        )
    p0 = {
        "schema_version": "t1_dedicated_panel_p0_effect_v1",
        "status": "passed" if constructible else "panel_not_constructible",
        "track": "T1",
        "created_at_jst": now_jst(),
        "selection_rule": "example suite order; init_state_index=0; env_seed=1000+i; policy_seed=0; max_steps=300; 20Hz; chunk10/execute8",
        "families_required": ["lighting", "env"],
        "families_constructible": families,
        "empty_track_perturbations_not_used_as_lighting": True,
        "plus_assets_dir": str(PLUS_ASSETS),
        "plus_assets_is_symlink": PLUS_ASSETS.is_symlink(),
        "plus_assets_symlink_target": os.readlink(PLUS_ASSETS) if PLUS_ASSETS.is_symlink() else None,
        "plus_assets_empty": not PLUS_ASSETS.exists() or PLUS_ASSETS.is_symlink() or not any(PLUS_ASSETS.rglob("*")),
        "do_not_write_fake_success_delta": True,
        "local_plus2_gate": "not_available" if not constructible else "pending_p2",
        "continue_to_step250_authorized": False,
        "cases": reports,
    }
    freeze = {
        "schema_version": "t1_dedicated_panel_p1_freeze_v1",
        "status": freeze_status,
        "track": "T1",
        "created_at_jst": now_jst(),
        "frozen_for_rollout": constructible,
        "reason": freeze_reason,
        "cases": freeze_cases,
        "p2_authorized": constructible,
        "official_result": False,
        "training_steps": 0,
        "continue_to_step250_authorized": False,
        "canary100_external_regression_blocks_long_train": True,
        "p2_note": "P2 may score the frozen lighting/env panel; it cannot unfreeze T1/T2 canary100 long-train after user-reported official scores 0.30/0.31.",
    }
    write_json(EVIDENCE / "p0_effect_tests.json", p0)
    write_json(EVIDENCE / "p1_freeze.json", freeze)
    return {"p0": p0["status"], "freeze": freeze["status"], "families": families}


def _hash_array(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _env_snapshot(bddl: str, seed: int) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128, control_freq=20)
    env.seed(int(seed))
    obs = env.reset()
    sim = env.sim
    image_keys = sorted(
        key
        for key, value in obs.items()
        if hasattr(value, "tobytes") and ("image" in key.lower() or "rgb" in key.lower())
    )
    snap = {
        "light_pos": _hash_array(sim.model.light_pos),
        "light_dir": _hash_array(sim.model.light_dir),
        "light_diffuse": _hash_array(sim.model.light_diffuse),
        "light_ambient": _hash_array(sim.model.light_ambient),
        "geom_rgba": _hash_array(sim.model.geom_rgba),
        "ntex": int(sim.model.ntex),
        "images": {key: _hash_array(obs[key]) for key in image_keys},
    }
    env.close()
    return snap


def _try_env(case: dict[str, Any]) -> dict[str, Any]:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "3"
    sys.path.insert(0, str(ROOT / "LIBERO-plus"))
    try:
        perturbed = _env_snapshot(case["bddl_path"], case["env_seed"])
        repeat = _env_snapshot(case["bddl_path"], case["env_seed"])
        base = _env_snapshot(case["base_bddl_path"], case["env_seed"])
        family = case["perturbation_family"]
        light_keys = ("light_pos", "light_dir", "light_diffuse", "light_ambient")
        if family == "lighting":
            changed = any(perturbed[key] != base[key] for key in light_keys)
            field = "lights"
            reproducible = all(perturbed[key] == repeat[key] for key in light_keys)
        else:
            changed = (
                perturbed["geom_rgba"] != base["geom_rgba"]
                or perturbed["ntex"] != base["ntex"]
                or perturbed["images"] != base["images"]
            )
            field = "textures_or_images"
            reproducible = (
                perturbed["geom_rgba"] == repeat["geom_rgba"]
                and perturbed["ntex"] == repeat["ntex"]
                and perturbed["images"] == repeat["images"]
            )
        image_reproducible = perturbed.get("images") == repeat.get("images")
        passed = bool(changed and reproducible)
        return {
            "status": "passed" if passed else "failed",
            "changed_field": field,
            "base_vs_perturbed_changed": changed,
            "same_seed_reproducible": reproducible,
            "image_hash_reproducible": image_reproducible,
            "renderer_jitter_note": "image hashes may differ at uint8 ±2; lighting effect uses light_* hashes, not pixels",
            "base": base,
            "perturbed": perturbed,
            "repeat_light_or_texture": {key: repeat[key] for key in light_keys}
            | {"geom_rgba": repeat["geom_rgba"], "ntex": repeat["ntex"]},
            "error": None if passed else "effect hashes did not change as required or were not reproducible",
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-env", action="store_true")
    args = parser.parse_args()
    result = run_p0(create_env=args.create_env)
    print(__import__("json").dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
