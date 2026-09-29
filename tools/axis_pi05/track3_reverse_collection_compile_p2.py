#!/usr/bin/env python3
"""Compile P2 reverse-only BDDLs with JSON natural-language prompts. No Reverse prefix."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "configs/axis_pi05/track3_reverse_collection_targets_20260909.json"
OUT_BDDL = ROOT / "compe/t3/assets/bddl_files/reverse_collection"
OUT_REPORT = ROOT / "data/reports/axis_libero_30k/track3_reverse_collection_rc3_p2_compile_20260909"
P2_IDS = {
    "rev_black_bowl_plate_to_table_center",
    "rev_alphabet_soup_basket_to_floor_front",
    "rev_close_middle_drawer",
    "rev_turn_off_stove",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def compile_one(target: dict[str, Any], task_index: int) -> dict[str, Any]:
    from tools.axis_pi05.track3_reverse_lora import _render_custom_bddl, compile_reverse_spec, sha256_file

    source = ROOT / target["source_bddl"]
    if not source.is_file():
        raise FileNotFoundError(source)
    suite = str(target.get("source_suite") or source.parent.name)
    init_path = ROOT / "LIBERO/libero/libero/init_files" / suite / f"{source.stem}.pruned_init"
    if not init_path.is_file():
        raise FileNotFoundError(init_path)
    spec = compile_reverse_spec(task_index, target["instruction"], source, init_path)
    instruction = str(target["instruction"])
    if instruction.lower().startswith("reverse"):
        raise ValueError(f"forbidden reverse prefix in {target['task_id']}")
    expected_source = [f"({atom})" if not str(atom).startswith("(") else str(atom) for atom in target["reverse_source"]]
    expected_goal = [f"({atom})" if not str(atom).startswith("(") else str(atom) for atom in target["reverse_goal"]]
    actual_source_tail = spec["reverse_initial_predicates"][-len(expected_source) :]
    if actual_source_tail != expected_source:
        raise ValueError(f"{target['task_id']} reverse source mismatch: {actual_source_tail} != {expected_source}")
    if spec["reverse_goal_predicates"] != expected_goal:
        raise ValueError(f"{target['task_id']} reverse goal mismatch: {spec['reverse_goal_predicates']} != {expected_goal}")
    OUT_BDDL.mkdir(parents=True, exist_ok=True)
    output = OUT_BDDL / f"{target['task_id']}.bddl"
    output.write_text(
        _render_custom_bddl(source.read_text(encoding="utf-8"), instruction, list(spec["reverse_initial_predicates"]), list(spec["reverse_goal_predicates"])),
        encoding="utf-8",
    )
    rendered = output.read_text(encoding="utf-8")
    if "Reverse the public LIBERO operation:" in rendered:
        raise ValueError(f"compiled BDDL still contains forbidden prefix: {output}")
    return {
        "status": "compiled",
        "task_id": target["task_id"],
        "instruction": instruction,
        "source_bddl": str(source.relative_to(ROOT)),
        "source_bddl_sha256": sha256_file(source),
        "reverse_bddl": str(output.relative_to(ROOT)),
        "reverse_bddl_sha256": sha256_file(output),
        "reverse_initial_predicates": spec["reverse_initial_predicates"],
        "reverse_goal_predicates": spec["reverse_goal_predicates"],
        "operator_family": spec["operator_family"],
        "smoke": {"status": "not_evaluated"},
    }


def _eval_atoms(env: Any, atoms: list[str]) -> bool:
    problem = getattr(env, "env", env)
    values = []
    for atom in atoms:
        text = atom.strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        parts = text.split()
        values.append(bool(problem._eval_predicate([parts[0].lower(), *parts[1:]])))
    return all(values)


def smoke_one(row: dict[str, Any]) -> dict[str, Any]:
    from tools.axis_pi05 import track3_public_teleop as teleop

    teleop.configure_public_libero()
    from libero.libero.envs import OffScreenRenderEnv

    bddl = ROOT / row["reverse_bddl"]
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), control_freq=20, use_camera_obs=False, render_gpu_device_id=0, ignore_done=True)
    try:
        env.reset()
        catalog = json.loads(TARGETS.read_text(encoding="utf-8"))
        target = next(item for item in catalog["targets"] if item["task_id"] == row["task_id"])
        source_atoms = [f"({atom})" if not str(atom).startswith("(") else str(atom) for atom in target["reverse_source"]]
        goal_atoms = [f"({atom})" if not str(atom).startswith("(") else str(atom) for atom in target["reverse_goal"]]
        source_ok = _eval_atoms(env, source_atoms)
        goal_ok = _eval_atoms(env, goal_atoms)
        success = bool(env.check_success())
        status = "passed" if source_ok and (not goal_ok) and (not success) else "needs_spec_fix"
        return {
            "status": status,
            "env_started": True,
            "source_true": source_ok,
            "goal_false": (not goal_ok),
            "check_success_false": (not success),
            "source_atoms": source_atoms,
            "goal_atoms": goal_atoms,
        }
    except Exception as exc:
        return {"status": "needs_spec_fix", "env_started": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def main() -> int:
    catalog = json.loads(TARGETS.read_text(encoding="utf-8"))
    targets = [row for row in catalog["targets"] if row["task_id"] in P2_IDS]
    if len(targets) != 4:
        raise RuntimeError(f"expected 4 P2 targets, got {len(targets)}")
    compiled = []
    for index, target in enumerate(targets, start=2001):
        compiled.append(compile_one(target, task_index=index))
    smoke = os.environ.get("TRACK3_P2_SMOKE", "0") == "1"
    if smoke:
        for row in compiled:
            row["smoke"] = smoke_one(row)
    report = {
        "schema_version": "track3_reverse_collection_rc3_p2_compile_v1",
        "status": "compiled_smoke_pending" if not smoke else ("compiled" if all(row["smoke"].get("status") == "passed" for row in compiled) else "compiled_with_spec_fix"),
        "created_at_utc": utc_now(),
        "forbidden_prefix_present": False,
        "simulator_started": smoke,
        "tasks": compiled,
    }
    write_json(OUT_REPORT / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"compiled", "compiled_smoke_pending"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
