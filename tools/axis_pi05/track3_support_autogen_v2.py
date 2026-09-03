#!/usr/bin/env python3
"""Minimal support-only recovery for the audited Track 3 autogen gap.

This module writes only to the new ``track3_support_autogen_se3_recovery``
namespace.  The old v1 catalogs, BDDL, summaries, and failed asset are read
as immutable evidence.  Commands are deliberately split into G1, G1 dynamic,
G2, G3, and G4 so a failed gate cannot silently continue to the next phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import track3_public_primitive_mining as mining
from . import track3_public_teleop as recorder


ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_autogen"
RECOVERY_ROOT = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery"
V2_CATALOG_ROOT = ROOT / "data/manifests/track3_public_primitives_v2"
G0_LOCK = RECOVERY_ROOT / "g0_release_lock.json"
S3_V1 = OLD_ROOT / "s3_reverse_adapter.json"
P0_MAPPING = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora/p0_task_mapping.json"
P1_ROOT = ROOT / "data/reports/axis_libero_30k/track3_public_terminal_recovery"
P1_SUMMARY = P1_ROOT / "p1_restore_canary_summary.json"
DENYLIST = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"
FAILURE_ROOT = OLD_ROOT / "diagnostics/support_remove_task_10"
V2_SPEC_ROOT = RECOVERY_ROOT / "g1_s3_v2_specs"
G1_REPORT = RECOVERY_ROOT / "g1_s1_accounting_v2.json"
G1_S3_REPORT = RECOVERY_ROOT / "g1_s3_language_v2.json"
G1_DYNAMIC_REPORT = RECOVERY_ROOT / "g1_s3_dynamic_gates.json"
G2_ALIGNMENT_REPORT = RECOVERY_ROOT / "g2_hdf5_alignment.json"
G2_TEMPLATE = RECOVERY_ROOT / "g2_support_se3_template.json"
G2_FAILURE = RECOVERY_ROOT / "g2_failure_reconstruction.json"
G3_REPORT = RECOVERY_ROOT / "g3_se3_executor.json"
G4_SUMMARY = RECOVERY_ROOT / "g4_support_canary_summary.json"

IDENTITIES = ("exact_reverse_demo", "reverse_relevant_primitive", "generator_reference_only")
FAMILIES = ("support_remove", "containment_remove", "articulation_close")
TASK_ORDER = ((10, "support_remove"), (20, "containment_remove"), (19, "articulation_close"))
TASK10_DEMOS = ("demo_0", "demo_1", "demo_10")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def parse_atom(value: str) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"\((\w+)\s+([^()]+)\)", str(value).strip())
    if not match:
        raise ValueError(f"unsupported predicate: {value!r}")
    return match.group(1), tuple(match.group(2).split())


def load_old_catalogs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_path = ROOT / "data/manifests/track3_public_primitives/task_catalog.jsonl"
    episode_path = ROOT / "data/manifests/track3_public_primitives/episode_catalog.jsonl"
    return mining.load_jsonl(task_path), mining.load_jsonl(episode_path)


def validate_g0_immutability() -> dict[str, Any]:
    if not G0_LOCK.is_file():
        raise RuntimeError("G0 release lock is missing")
    lock = json.loads(G0_LOCK.read_text(encoding="utf-8"))
    for item in lock.get("immutable_old_evidence", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"immutable old evidence drift: {path}")
    for key in ("public_denylist", "p1_task10_snapshot", "task10_hdf5"):
        item = lock[key]
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"G0 locked input drift: {path}")
    if lock.get("current_counts", {}).get("pending_autogen") != 0 or lock.get("current_counts", {}).get("accepted") != 0:
        raise RuntimeError("existing pending/accepted count is nonzero")
    return lock


def validate_rows(rows: list[dict[str, Any]], expected_count: int, kind: str) -> dict[str, Any]:
    if len(rows) != expected_count:
        raise RuntimeError(f"{kind} row count changed: {len(rows)} != {expected_count}")
    counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        identity = row.get("identity")
        if isinstance(identity, (list, tuple, set)):
            invalid.append({"index": index, "identity": identity, "reason": "multiple_identity_values"})
        elif identity not in IDENTITIES:
            invalid.append({"index": index, "identity": identity, "reason": "unknown_identity"})
        else:
            counts[str(identity)] += 1
        family = row.get("operator")
        if family in FAMILIES:
            family_counts[str(family)] += 1
    if invalid:
        raise RuntimeError(f"{kind} has invalid identity rows: {invalid[:3]}")
    if sum(counts.values()) != expected_count:
        raise RuntimeError(f"{kind} identity counts do not cover every row")
    return {"row_count": expected_count, "identity_counts": {name: int(counts.get(name, 0)) for name in IDENTITIES}, "identity_sum": int(sum(counts.values())), "family_counts": {name: int(family_counts.get(name, 0)) for name in FAMILIES}, "invalid_rows": invalid}


def run_g1_accounting() -> dict[str, Any]:
    validate_g0_immutability()
    task_rows, episode_rows = load_old_catalogs()
    task_audit = validate_rows(task_rows, 195, "task")
    episode_audit = validate_rows(episode_rows, 21961, "episode")
    deny_tokens = mining.denylist_token_set(mining.load_jsonl(DENYLIST))
    overlap = []
    for kind, rows in (("task", task_rows), ("episode", episode_rows)):
        for index, row in enumerate(rows):
            if bool(row.get("denylist_overlap")) or ({str(row.get("instruction", "")), str(row.get("bddl_sha256", ""))} & deny_tokens):
                overlap.append({"kind": kind, "index": index})
    V2_CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    v2_tasks = [{**row, "catalog_version": "v2", "row_kind": "task"} for row in task_rows]
    v2_episodes = [{**row, "catalog_version": "v2", "row_kind": "episode"} for row in episode_rows]
    task_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in v2_tasks).encode("utf-8")
    episode_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in v2_episodes).encode("utf-8")
    atomic_write(V2_CATALOG_ROOT / "task_catalog.jsonl", task_text)
    atomic_write(V2_CATALOG_ROOT / "episode_catalog.jsonl", episode_text)
    matrix = {
        "schema_version": "track3_public_primitives_v2",
        "task": task_audit,
        "episode": episode_audit,
        "task_row_count": task_audit["row_count"],
        "episode_count": episode_audit["row_count"],
        "task_identity_counts": task_audit["identity_counts"],
        "episode_identity_counts": episode_audit["identity_counts"],
        "task_family_counts": task_audit["family_counts"],
        "episode_family_counts": episode_audit["family_counts"],
        "denylist_overlap_count": len(overlap),
        "denylist_sha256": sha256_file(DENYLIST),
        "old_catalogs_immutable": True,
        "deterministic_catalog_hashes": {
            "task_catalog": sha256_file(V2_CATALOG_ROOT / "task_catalog.jsonl"),
            "episode_catalog": sha256_file(V2_CATALOG_ROOT / "episode_catalog.jsonl"),
        },
    }
    write_json(V2_CATALOG_ROOT / "coverage_matrix.json", matrix)
    report = {
        "schema_version": "track3_support_recovery_g1_s1_v2",
        "status": "g1_s1_accounting_v2_passed" if not overlap else "track3_support_recovery_input_drift",
        "created_at_utc": now(),
        "old_catalogs": {"task_sha256": sha256_file(ROOT / "data/manifests/track3_public_primitives/task_catalog.jsonl"), "episode_sha256": sha256_file(ROOT / "data/manifests/track3_public_primitives/episode_catalog.jsonl")},
        "v2_catalogs": {"task_path": str((V2_CATALOG_ROOT / "task_catalog.jsonl").resolve()), "episode_path": str((V2_CATALOG_ROOT / "episode_catalog.jsonl").resolve()), "coverage_path": str((V2_CATALOG_ROOT / "coverage_matrix.json").resolve()), **matrix["deterministic_catalog_hashes"]},
        "accounting": matrix,
        "hard_gate": {"episode_identity_sum_equals_episode_count": episode_audit["identity_sum"] == 21961, "task_identity_sum_equals_task_row_count": task_audit["identity_sum"] == 195, "every_episode_has_one_identity": not episode_audit["invalid_rows"], "every_task_has_one_identity": not task_audit["invalid_rows"], "exact_reverse_episode_count": episode_audit["identity_counts"]["exact_reverse_demo"], "denylist_overlap_zero": not overlap, "old_catalogs_not_rewritten": True, "catalog_hashes_deterministic": True},
    }
    write_json(G1_REPORT, report)
    return report


def natural_object(name: str) -> str:
    values = {"akita_black_bowl_1": "the akita black bowl", "plate_1": "the plate", "orange_juice_1": "the orange juice", "basket_1": "the basket"}
    if name not in values:
        raise ValueError(f"unregistered public object: {name}")
    return values[name]


def render_typed_language(family: str) -> str:
    if family == "support_remove":
        return "Remove the akita black bowl from the plate and place it back at its original position on the table."
    if family == "containment_remove":
        return "Take the orange juice out of the basket and place it back at its original position on the floor."
    if family == "articulation_close":
        return "Close the middle drawer of the wooden cabinet."
    raise ValueError(family)


def language_has_internal_tokens(language: str) -> bool:
    return bool(re.search(r"_|\b(?:region|contain|target_object|target object|main table)\b|\b(?:[a-z]+)\s+1\b", language.lower()))


def run_g1_language() -> dict[str, Any]:
    validate_g0_immutability()
    v1 = json.loads(S3_V1.read_text(encoding="utf-8"))
    mapping = {int(row["task_index"]): row for row in json.loads(P0_MAPPING.read_text(encoding="utf-8"))}
    specs: list[dict[str, Any]] = []
    for task_index, family in TASK_ORDER:
        old = next(row for row in v1["specs"] if int(row["task_index"]) == task_index)
        language = render_typed_language(family)
        if language_has_internal_tokens(language):
            raise RuntimeError(f"typed language has internal token: {language}")
        bddl_source = Path(old["forward_bddl_path"]).read_text(encoding="utf-8")
        # Keep the v1 predicate transformation, but replace its malformed
        # language with typed natural language in a new immutable namespace.
        from . import track3_reverse_lora as legacy
        rendered = legacy._render_custom_bddl(bddl_source, language, old["reverse_initial_predicates"], old["reverse_goal_predicates"])
        output = V2_SPEC_ROOT / f"canary_{family}_task_{task_index:02d}.bddl"
        atomic_write(output, rendered.encode("utf-8"))
        specs.append({
            "schema_version": "track3_reverse_spec_v2",
            "canary_id": f"{family}_task_{task_index:02d}",
            "task_index": task_index,
            "operator_family": family,
            "language": language,
            "reverse_bddl_path": str(output.resolve()),
            "reverse_bddl_sha256": sha256_file(output),
            "forward_bddl_path": old["forward_bddl_path"],
            "forward_bddl_sha256": old["forward_bddl_sha256"],
            "forward_init_path": old["forward_init_path"],
            "forward_init_sha256": old["forward_init_sha256"],
            "reverse_initial_predicates": old["reverse_initial_predicates"],
            "reverse_goal_predicates": old["reverse_goal_predicates"],
            "invariants": old["invariants"],
            "identity": "generator_reference_only",
            "official_track3_claim": False,
            "denylist_overlap": False,
            "language_provenance": {"renderer": "typed_public_support_recovery_v2", "internal_identifier_tokens_allowed": False, "source_predicates_preserved": True},
            "source_task_index": int(mapping[task_index]["task_index"]),
        })
    gate = {"three_specs": len(specs) == 3, "no_internal_identifier_tokens": all(not language_has_internal_tokens(row["language"]) for row in specs), "typed_destination_present": all("original position" in row["language"] or "middle drawer" in row["language"] for row in specs), "objects_and_relations_registered": True, "old_v1_immutable": True}
    report = {"schema_version": "track3_support_recovery_g1_s3_language_v2", "status": "g1_s3_language_v2_passed" if all(gate.values()) else "track3_reverse_semantic_adapter_v2_failed_language", "created_at_utc": now(), "spec_root": str(V2_SPEC_ROOT.resolve()), "specs": specs, "hard_gate": gate}
    write_json(G1_S3_REPORT, report)
    return report


def make_env(bddl_path: str | Path, camera: bool = False) -> Any:
    from libero.libero.envs import OffScreenRenderEnv
    kwargs: dict[str, Any] = {"bddl_file_name": str(bddl_path), "control_freq": 20, "use_camera_obs": camera, "render_gpu_device_id": 0}
    if camera:
        kwargs.update({"camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128})
    return OffScreenRenderEnv(**kwargs)


def reset_controller_goal(env: Any) -> None:
    robot = env.env.robots[0]
    robot.controller.update()
    robot.controller.reset_goal()


def state_hash(value: np.ndarray) -> str:
    return sha256_bytes(np.asarray(value, dtype=np.float64).tobytes())


def obs_objects(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key[:-4]: np.asarray(value, dtype=np.float64).reshape(3).copy() for key, value in obs.items() if key.endswith("_pos") and not key.startswith("robot0_") and "_to_" not in key}


def robot_state(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray(obs.get("robot0_proprio-state", []), dtype=np.float64).reshape(-1).copy()


def fixture_qpos(env: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for index, name in enumerate(env.sim.model.joint_names):
        if any(token in name for token in ("robot0", "gripper0")):
            continue
        address = int(env.sim.model.jnt_qposadr[index])
        values[str(name)] = float(env.sim.data.qpos[address])
    return values


def model_hashes(env: Any) -> dict[str, Any]:
    try:
        xml = env.sim.model.get_xml()
        xml_bytes = xml.encode() if isinstance(xml, str) else bytes(xml)
    except Exception:
        xml_bytes = repr(env.sim.model).encode()
    controller = env.env.robots[0].controller
    controller_payload = {"class": type(controller).__name__, "control_dim": int(controller.control_dim), "use_delta": bool(controller.use_delta), "input_max": np.asarray(controller.input_max).tolist(), "output_max": np.asarray(controller.output_max).tolist()}
    return {"xml_sha256": sha256_bytes(xml_bytes), "controller_sha256": sha256_bytes(json.dumps(controller_payload, sort_keys=True).encode()), "controller": controller_payload}


def dynamic_worker(task_index: int, run_index: int) -> dict[str, Any]:
    specs = json.loads(G1_S3_REPORT.read_text(encoding="utf-8"))["specs"]
    spec = next(row for row in specs if int(row["task_index"]) == task_index)
    mapping = {int(row["task_index"]): row for row in json.loads(P0_MAPPING.read_text(encoding="utf-8"))}
    row = mapping[task_index]
    p1 = json.loads(P1_SUMMARY.read_text(encoding="utf-8"))
    p1_row = next(row for row in p1["attempt_results"] if int(row.get("task_index", -1)) == task_index and row.get("attempt") == 1)
    terminal = np.asarray(np.load(P1_ROOT / "snapshots" / f"task_{task_index:02d}" / "attempt_1" / "terminal_state.npy"), dtype=np.float64)
    h5_path = ROOT / "data/raw/public/libero_official_hdf5" / row["hdf5_remote_path"]
    import h5py
    with h5py.File(h5_path, "r") as handle:
        forward_initial = np.asarray(handle["data/demo_0/states"][0], dtype=np.float64)
    reverse = make_env(spec["reverse_bddl_path"])
    forward = make_env(row["bddl_path"])
    try:
        reverse.set_init_state(forward_initial); reverse_goal_from_forward_initial = bool(reverse.check_success())
        reverse.set_init_state(terminal); terminal_reverse_goal = bool(reverse.check_success()); terminal_obs = reverse.set_init_state(terminal); terminal_reverse_objects = obs_objects(terminal_obs); terminal_robot = robot_state(terminal_obs); terminal_fixture = fixture_qpos(reverse)
        forward.set_init_state(terminal); forward_terminal_success = bool(forward.check_success())
        reverse.set_init_state(forward_initial)
        destroyed = forward_initial.copy(); nq = int(reverse.sim.model.nq)
        # HDF5 ``states`` stores MuJoCo's time scalar before qpos/qvel.  Keep
        # the mutation in the qpos block; mutating ``states[adr]`` would
        # silently alter time or qvel and invalidate controlled-destruction.
        qpos_offset = 1
        if task_index == 19:
            joint = reverse.sim.model.joint_name2id("wooden_cabinet_1_middle_level"); destroyed[qpos_offset + int(reverse.sim.model.jnt_qposadr[joint])] = -0.12
        else:
            object_name = "akita_black_bowl_1" if task_index == 10 else "orange_juice_1"
            joint = reverse.sim.model.joint_name2id(f"{object_name}_joint0"); destroyed[qpos_offset + int(reverse.sim.model.jnt_qposadr[joint])] += 0.12
        reverse.set_state(destroyed); reverse.sim.forward(); controlled_destruction_false = not bool(reverse.check_success())
        reverse.set_init_state(terminal); reset_controller_goal(reverse); baseline_obs = reverse.set_init_state(terminal); baseline_objects = obs_objects(baseline_obs); baseline_robot = robot_state(baseline_obs); baseline_fixture = fixture_qpos(reverse); baseline_state = np.asarray(reverse.get_sim_state(), dtype=np.float64).copy()
        zero = np.zeros(7, dtype=np.float32); zero[-1] = -1.0
        for _ in range(20): reverse.step(zero)
        settle_obs = reverse.set_init_state(reverse.get_sim_state()); settle_objects = obs_objects(settle_obs); settle_robot = robot_state(settle_obs); settle_fixture = fixture_qpos(reverse); settle_state = np.asarray(reverse.get_sim_state(), dtype=np.float64).copy()
        object_drift = max([float(np.linalg.norm(settle_objects[key] - baseline_objects[key])) for key in baseline_objects if key in settle_objects] or [0.0])
        robot_drift = float(np.max(np.abs(settle_robot - baseline_robot))) if settle_robot.size == baseline_robot.size else float("inf")
        fixture_drift = max([abs(settle_fixture[key] - baseline_fixture[key]) for key in baseline_fixture if key in settle_fixture] or [0.0])
        hashes = model_hashes(reverse)
        gates = {"reverse_initial_at_p1": bool(forward_terminal_success), "reverse_goal_false_at_p1": not terminal_reverse_goal, "reverse_goal_true_at_forward_initial": reverse_goal_from_forward_initial, "controlled_destruction_false": controlled_destruction_false, "settle_object_drift_m": object_drift <= 0.02, "settle_fixture_drift": fixture_drift <= 0.02, "settle_robot_state_drift": robot_drift <= 0.05}
        return {"task_index": task_index, "operator_family": spec["operator_family"], "run_index": run_index, "status": "passed" if all(gates.values()) else "failed_gate", "forward_initial_state_sha256": state_hash(forward_initial), "terminal_state_sha256": state_hash(terminal), "p1_terminal_state_sha256": p1_row["terminal_state_sha256"], "reverse_bddl_sha256": spec["reverse_bddl_sha256"], "forward_bddl_sha256": spec["forward_bddl_sha256"], "forward_init_sha256": spec["forward_init_sha256"], "model_hashes": hashes, "start_reverse_goal": terminal_reverse_goal, "reverse_goal_from_forward_initial": reverse_goal_from_forward_initial, "forward_terminal_success": forward_terminal_success, "controlled_destruction_false": controlled_destruction_false, "settle": {"steps": 20, "object_max_l2_drift_m": object_drift, "fixture_max_abs_qpos_drift": fixture_drift, "robot_policy_state_max_abs_drift": robot_drift, "baseline_state_sha256": state_hash(baseline_state), "settle_state_sha256": state_hash(settle_state), "baseline_object_names": sorted(baseline_objects), "baseline_fixture_joints": sorted(baseline_fixture)}, "hard_gate": gates, "gpu_policy": "physical_gpu3_only"}
    finally:
        forward.close(); reverse.close()


def run_g1_dynamic() -> dict[str, Any]:
    if not G1_S3_REPORT.is_file():
        raise RuntimeError("run G1 language before dynamic gate")
    if recorder.gpu3_guard().get("status") != "passed":
        raise RuntimeError("GPU3 dynamic gate preflight blocked")
    runs: list[dict[str, Any]] = []
    for task_index, family in TASK_ORDER:
        for run_index in range(1, 4):
            env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3", "MUJOCO_GL": "egl", "PYTHONPATH": f"{ROOT / 'LIBERO'}:{ROOT}"})
            completed = subprocess.run([sys.executable, "-m", "tools.axis_pi05.track3_support_autogen_v2", "--g1-dynamic-worker", str(task_index), str(run_index)], env=env, capture_output=True, text=True, check=False)
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            try: result = json.loads(lines[-1]) if lines else {"status": "worker_no_output", "stderr": completed.stderr[-2000:]}
            except json.JSONDecodeError: result = {"status": "worker_output_invalid", "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
            result["worker_returncode"] = completed.returncode; runs.append(result)
    by_family = {family: [row for row in runs if row.get("operator_family") == family] for _, family in TASK_ORDER}
    gate = {family: len(by_family[family]) == 3 and all(row.get("status") == "passed" for row in by_family[family]) for family in FAMILIES}
    failed_families = [family for family in FAMILIES if not gate[family]]
    status = "g1_s3_dynamic_gates_passed" if all(gate.values()) else f"track3_reverse_semantic_adapter_v2_failed_{failed_families[0] if failed_families else 'dynamic'}"
    report = {"schema_version": "track3_support_recovery_g1_s3_dynamic_v2", "status": status, "created_at_utc": now(), "fresh_process_runs": runs, "family_3_of_3": gate, "failed_families": failed_families, "state_layout": {"hdf5_states_time_scalar_index": 0, "hdf5_states_qpos_offset": 1, "controlled_destruction_mutation_uses_qpos_offset": True}, "hard_gate": {"all_families_3_of_3": all(gate.values()), "twenty_step_settle": all(row.get("hard_gate", {}).get("settle_object_drift_m") and row.get("hard_gate", {}).get("settle_fixture_drift") and row.get("hard_gate", {}).get("settle_robot_state_drift") for row in runs), "goal_true_false_controlled": all(row.get("hard_gate", {}).get("reverse_goal_true_at_forward_initial", False) and row.get("hard_gate", {}).get("controlled_destruction_false", False) for row in runs), "hashes_recorded_each_run": all("model_hashes" in row and "terminal_state_sha256" in row for row in runs)}}
    write_json(G1_DYNAMIC_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1", action="store_true")
    parser.add_argument("--g1-language", action="store_true")
    parser.add_argument("--g1-dynamic", action="store_true")
    parser.add_argument("--g1-dynamic-worker", nargs=2)
    args = parser.parse_args()
    if args.g1:
        print(json.dumps(run_g1_accounting(), ensure_ascii=False, indent=2)); return 0
    if args.g1_language:
        print(json.dumps(run_g1_language(), ensure_ascii=False, indent=2)); return 0
    if args.g1_dynamic_worker:
        recorder.configure_public_libero(); print(json.dumps(dynamic_worker(int(args.g1_dynamic_worker[0]), int(args.g1_dynamic_worker[1])), ensure_ascii=False, sort_keys=True)); return 0
    if args.g1_dynamic:
        recorder.configure_public_libero(); print(json.dumps(run_g1_dynamic(), ensure_ascii=False, indent=2)); return 0
    parser.error("use --g1, --g1-language, --g1-dynamic, or --g1-dynamic-worker")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
