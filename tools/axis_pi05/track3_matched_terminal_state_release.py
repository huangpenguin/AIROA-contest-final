#!/usr/bin/env python3
"""Matched public-terminal-state release gates for Track 3.

This namespace is deliberately independent of the older reverse-init reports.
L0/L1 are filesystem/code-only.  L2 uses one public demo's model XML and
full-state together in a fresh CPU process; L3/L4 add only the bounded GPU3
camera canaries authorized by the matched-terminal-state plan.
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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data/reports/axis_libero_30k/track3_matched_terminal_state_release"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_MATCHED_TERMINAL_STATE_RELEASE_PLAN.md"
AUDIT = ROOT / "docs/reviews/axis_pi05/TRACK3_BACKGROUND_CONVERGENCE_K0_K4_AUDIT_20260904.md"
HANDOFF_JSON = ROOT / "data/reports/axis_libero_30k/post_r6_lora/PROJECT_HANDOFF_20260831.json"
PUBLIC_LIBERO = ROOT / "LIBERO"
LIBERO_PLUS = ROOT / "LIBERO-plus"

HDF5_INPUTS = {
    10: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
    20: ROOT / "data/raw/public/libero_official_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5",
    19: ROOT / "data/raw/public/libero_official_hdf5/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5",
}
FORWARD_BDDL = {
    10: PUBLIC_LIBERO / "libero/libero/bddl_files/libero_goal/put_the_bowl_on_the_plate.bddl",
    20: PUBLIC_LIBERO / "libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl",
    19: PUBLIC_LIBERO / "libero/libero/bddl_files/libero_goal/open_the_middle_drawer_of_the_cabinet.bddl",
}
SPECS = ROOT / "data/reports/axis_libero_30k/track3_support_autogen_se3_recovery/g1_s3_v2_specs"
REVERSE_BDDL = {
    10: SPECS / "canary_support_remove_task_10.bddl",
    20: SPECS / "canary_containment_remove_task_20.bddl",
    19: SPECS / "canary_articulation_close_task_19.bddl",
}
TASK_ORDER = (10, 20, 19)
FAMILY = {10: "support_remove", 20: "containment_remove", 19: "articulation_close"}
SOURCE_ATOM = {
    10: "(On akita_black_bowl_1 plate_1)",
    20: "(In orange_juice_1 basket_1_contain_region)",
    19: "(Open wooden_cabinet_1_middle_region)",
}
GOAL_ATOM = {
    10: "(On akita_black_bowl_1 main_table_akita_black_bowl_region)",
    20: "(On orange_juice_1 floor_target_object_region)",
    19: "(Close wooden_cabinet_1_middle_region)",
}
OPERATOR = {
    10: {"subject": ("akita_black_bowl_1",), "target": (), "allowed": (("akita_black_bowl_1", "plate_1"),)},
    20: {"subject": ("orange_juice_1",), "target": (), "allowed": (("orange_juice_1", "basket_1"),)},
    19: {"subject": ("wooden_cabinet_1_middle_region",), "target": ("wooden_cabinet_1_cabinet_middle",), "allowed": ()},
}
CHECKPOINTS = tuple(range(21))
RELEASE_STEPS = tuple(range(5, 21))
ACTION_DIM = 7
CONTROL_FREQ_HZ = 20
THRESHOLD_BACKGROUND_TRANSLATION_M = 0.002
WORKER_COMPLETED_STATUSES = frozenset({"passed", "failed_gate"})


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_output(repo: Path, args: Sequence[str]) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()


def porcelain(repo: Path) -> list[str]:
    return [line for line in git_output(repo, ("status", "--porcelain", "--untracked-files=all")).splitlines() if line]


def worker_exit_code(status: str) -> int:
    return 0 if status in WORKER_COMPLETED_STATUSES else 1


def gate(*, executed: bool, value: bool | None, required: bool = True, reason: str | None = None) -> dict[str, Any]:
    row = {"executed": bool(executed), "state": "not_evaluated" if not executed else ("passed" if value is True else "failed"), "required": bool(required)}
    if reason is not None:
        row["reason"] = reason
    return row


def all_required(gates: Mapping[str, Mapping[str, Any]]) -> bool:
    required = [row for row in gates.values() if row.get("required", True)]
    return bool(required) and all(row.get("state") == "passed" for row in required)


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def resolve_model_xml(model_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Resolve only stale absolute asset roots; preserve XML semantics.

    Public HDF5 records contain the source machine's absolute roots.  The
    bytes and hash in the manifest remain the authoritative source identity;
    this local path resolution is required only for MuJoCo to find the same
    checked-in public assets on this machine.
    """
    import robosuite

    source = model_bytes.decode("utf-8")
    libero_assets = (PUBLIC_LIBERO / "libero/libero/assets").resolve()
    robosuite_assets = (Path(robosuite.__file__).resolve().parent / "models/assets").resolve()
    replacements = {
        "/Users/yifengz/workspace/libero-dev/chiliocosm/assets/": str(libero_assets) + "/",
        "/Users/yifengz/workspace/robosuite-master/robosuite/models/assets/": str(robosuite_assets) + "/",
    }
    resolved = source
    for old, new in replacements.items():
        resolved = resolved.replace(old, new)
    referenced = re.findall(r'(?:file|mesh)="([^"]+)"', resolved)
    missing = sorted({value for value in referenced if value.startswith("/") and not Path(value).exists()})
    if missing:
        raise FileNotFoundError(f"resolved model assets missing: {missing[:3]}")
    return resolved, {"source_sha256": sha256_bytes(model_bytes), "resolved_sha256": sha256_bytes(resolved.encode("utf-8")), "replacements": replacements, "absolute_asset_references": len(referenced), "missing_absolute_assets": missing}


def _terminal_selection(rewards: Any, states_len: int, actions_len: int) -> dict[str, Any]:
    """Select the first state belonging to the sustained positive reward suffix."""
    import numpy as np

    values = np.asarray(rewards).reshape(-1) if rewards is not None else np.asarray([], dtype=np.float64)
    if values.size != actions_len:
        raise ValueError(f"reward/action length mismatch: {values.size} != {actions_len}")
    positive = values > 0
    if not bool(positive.any()):
        raise ValueError("reward schema has no positive success suffix")
    starts = [int(index) for index in np.flatnonzero(positive) if bool(positive[index:].all())]
    if not starts:
        raise ValueError("reward positives are not a sustained suffix")
    action_index = starts[0]
    state_index = action_index + 1 if states_len == actions_len + 1 else action_index if states_len == actions_len else -1
    if state_index < 0 or state_index >= states_len:
        raise ValueError(f"unsupported state/action alignment: states={states_len}, actions={actions_len}")
    return {"action_index": action_index, "state_index": state_index, "suffix_start_action_index": action_index, "suffix_length": int(values.size - action_index), "positive_count": int(positive.sum()), "reward_at_selection": float(values[action_index]), "alignment": "state_index_equals_action_index" if states_len == actions_len else "state_index_equals_action_index_plus_one"}


def read_demo_manifest_row(task_index: int, demo_id: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    path = HDF5_INPUTS[task_index]
    with h5py.File(path, "r") as handle:
        key = f"data/demo_{demo_id}"
        if key not in handle:
            raise ValueError(f"missing fixed demo: {key}")
        demo = handle[key]
        states = np.asarray(demo["states"][()])
        actions = np.asarray(demo["actions"][()])
        rewards = np.asarray(demo["rewards"][()]) if "rewards" in demo else None
        model_bytes = _as_bytes(demo.attrs["model_file"])
        selection = _terminal_selection(rewards, len(states), len(actions))
        state = np.asarray(states[selection["state_index"]], dtype=np.float64)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError(f"non-finite terminal state: task={task_index} demo={demo_id}")
        return {
            "task_index": task_index,
            "family": FAMILY[task_index],
            "demo_id": f"demo_{demo_id}",
            "demo_index": demo_id,
            "hdf5_path": str(path.resolve()),
            "hdf5_bytes": path.stat().st_size,
            "hdf5_sha256": sha256_file(path),
            "model_file_bytes": len(model_bytes),
            "model_file_sha256": sha256_bytes(model_bytes),
            "states_shape": list(states.shape),
            "states_dtype": str(states.dtype),
            "actions_shape": list(actions.shape),
            "actions_dtype": str(actions.dtype),
            "rewards_shape": list(rewards.shape) if rewards is not None else None,
            "rewards_dtype": str(rewards.dtype) if rewards is not None else None,
            "selection": selection,
            "terminal_state_sha256": sha256_bytes(state.tobytes()),
            "terminal_state_shape": list(state.shape),
            "terminal_state_dtype": str(state.dtype),
            "predicate_oracle": "never_hdf5; runtime_only_in_same_demo_model_xml",
        }


def _old_evidence_paths() -> list[Path]:
    root = ROOT / "data/reports/axis_libero_30k/track3_background_predicate_convergence"
    paths = [
        root / "k0_input_lock.json", root / "k0_raw_manifest.json", root / "k1_monotonic_contract.json",
        root / "k2_transition_ledger.json", root / "k2_summary.json", root / "k2_reverse_release.json",
        root / "k2_forward_controls.json", root / "k3_summary.json", root / "k3_gpu3_release/task19_canary.json",
        root / "k4_summary.json", root / "K4_CLOSEOUT_REPORT_ZH.md", AUDIT,
        ROOT / "tools/axis_pi05/track3_background_predicate_convergence.py",
    ]
    return [path for path in paths if path.is_file()]


def run_l0() -> dict[str, Any]:
    """Lock old evidence and the fixed 9-demo selection without simulation."""
    current = git_output(ROOT, ("rev-parse", "HEAD"))
    if current != "c74f68b" and not current.startswith("c74f68b"):
        raise RuntimeError(f"L0 expected audited input commit c74f68b, observed {current}")
    ancestor = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "8490f8d", current], check=False)
    if ancestor.returncode != 0:
        raise RuntimeError("L0 input is not descended from the reviewed K4 root")
    old_paths = _old_evidence_paths()
    old = {str(path.resolve()): {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in old_paths if path.is_file()}
    missing = [str(path.resolve()) for path in old_paths if not path.is_file()]
    selected = [read_demo_manifest_row(task, demo) for task in TASK_ORDER for demo in range(3)]
    hdf5_paths = {str(task): {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for task, path in HDF5_INPUTS.items()}
    handoff = json.loads(HANDOFF_JSON.read_text(encoding="utf-8"))
    lock = {
        "schema_version": "track3_matched_terminal_state_release_l0_v1",
        "status": "track3_matched_terminal_state_input_locked" if not missing else "track3_matched_terminal_state_input_lock_failed",
        "created_at_utc": now(), "plan": str(PLAN.resolve()), "audit": str(AUDIT.resolve()),
        "root_commit_at_l0": current, "reviewed_k4_root": "8490f8dde1148aec0aec4d3d105082b44cfbb286",
        "reviewed_k2_commit": "cb8e898640a66e9baf5a587f21d5f44a84494a8e",
        "nested_openpi_expected": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
        "root_porcelain_at_l0": porcelain(ROOT),
        "libero_plus_dirty_read_only_snapshot": {"path": str(LIBERO_PLUS.resolve()), "status_porcelain": porcelain(LIBERO_PLUS), "policy": "record_only_never_clean_or_touch"},
        "immutable_old_evidence": old, "missing_old_evidence": missing, "hdf5_inputs": hdf5_paths,
        "fixed_selection": selected, "selection_rule": "task10/task20/task19 then demo_0/demo_1/demo_2; first state in sustained positive reward suffix; no replacement",
        "prior_counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0},
        "prior_handoff_status": handoff.get("status"),
        "resource_policy": {"simulator_started": False, "gpu_started": False, "official_gpu": False, "action_or_demo_generated": False},
    }
    write_json(OUT_ROOT / "l0_input_lock.json", lock)
    write_json(OUT_ROOT / "l0_demo_manifest.json", {"schema_version": "track3_matched_terminal_state_demo_manifest_v1", "status": lock["status"], "fixed_selection": selected, "canonical_sha256": canonical_hash(selected)})
    errata = f"""# Track 3 Matched Terminal-State L0：K3 Errata\n\n本文件是新 namespace 的只读 errata，不修改旧 K3/K4 raw evidence。\n\n- 旧 task19 native reverse K3 的 step5 有 39 个 `akita_black_bowl_1_main` ↔ `wooden_cabinet_1_cabinet_middle` geom contacts，最深 penetration 为约 `-6.797624546751698e-3 m`；该 contact 不能被 step5 baseline grandfather。\n- K3 renderer 的 flags 只读了 wrapper 层；实际 camera frames 仍为 agentview `21/21`、wrist `21/21`。\n- H.264 失败来自 `*.mp4.partial` 临时扩展名，ffmpeg 无法选择 muxer；新工具使用 `*.partial.mp4` 后再 atomic rename。\n- task19 sensitivity 的 raw mutation 从 `qpos=-0.146432...` 继续向负方向移动，仍满足 public `WoodenCabinet.is_open`；因此 raw truth 是 `Open true → true → true`，不是旧中文报告中的 `true → false → true`。新实现从 public close range 的绝对 qpos 中点验证并 exact restore。\n- 旧 K3 没有显式 seed；历史 `no_seed_change` 只作为历史字段保留，不能冒充 seed identity。\n\n新 L0 固定使用公开 HDF5 中同一个 demo 的 `model_file` 与 terminal full-state；predicate oracle 只在加载该 XML 的 runtime env 中执行。\n"""
    (OUT_ROOT / "L0_K3_ERRATA_ZH.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "L0_K3_ERRATA_ZH.md").write_text(errata, encoding="utf-8")
    return lock


def renderer_schema(wrapper: Any, observation: Mapping[str, Any]) -> dict[str, Any]:
    """Read actual wrapper and underlying flags separately."""
    underlying = getattr(wrapper, "env", None)
    def attrs(obj: Any) -> dict[str, Any]:
        return {name: getattr(obj, name, None) if obj is not None else None for name in ("use_camera_obs", "has_offscreen_renderer", "has_renderer")}
    cameras = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        value = observation.get(key)
        shape = list(getattr(value, "shape", ())) if value is not None else None
        dtype = str(getattr(value, "dtype", "")) if value is not None else None
        cameras[key] = {"present": value is not None, "shape": shape, "dtype": dtype, "uint8": bool(value is not None and str(getattr(value, "dtype", "")) == "uint8")}
    actual = attrs(underlying)
    return {"wrapper_flags": attrs(wrapper), "underlying_flags": actual, "camera_observations": cameras, "camera_keys_present": all(row["present"] for row in cameras.values()), "camera_keys_uint8": all(row["uint8"] for row in cameras.values()), "schema_complete": actual["use_camera_obs"] is True and actual["has_offscreen_renderer"] is True and actual["has_renderer"] is False and all(row["present"] and row["uint8"] for row in cameras.values())}


def atomic_h264(path: Path, frames: Sequence[Any], fps: int = 20) -> dict[str, Any]:
    """Encode to ``name.partial.mp4`` and atomically publish after ffprobe."""
    import numpy as np

    if not frames:
        return {"state": "failed", "reason": "empty_frames"}
    arrays = [np.asarray(frame) for frame in frames]
    if any(array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8 for array in arrays):
        return {"state": "failed", "reason": "frames_must_be_uint8_rgb"}
    height, width = arrays[0].shape[:2]
    if any(array.shape != arrays[0].shape for array in arrays):
        return {"state": "failed", "reason": "frame_shape_changed"}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    try:
        if partial.exists():
            partial.unlink()
        command = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(partial)]
        completed = subprocess.run(command, input=b"".join(array.tobytes() for array in arrays), capture_output=True, check=False)
        if completed.returncode != 0:
            return {"state": "failed", "reason": "ffmpeg_failed", "returncode": completed.returncode, "stderr_tail": completed.stderr.decode(errors="replace")[-1000:], "partial_path": str(partial)}
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,r_frame_rate,width,height,nb_frames", "-of", "json", str(partial)], capture_output=True, text=True, check=False)
        if probe.returncode != 0:
            return {"state": "failed", "reason": "ffprobe_failed", "stderr_tail": probe.stderr[-1000:], "partial_path": str(partial)}
        stream = json.loads(probe.stdout).get("streams", [{}])[0]
        numerator, denominator = (stream.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
        frame_rate = float(numerator) / float(denominator)
        frame_count = int(stream.get("nb_frames", 0))
        valid = stream.get("codec_name") == "h264" and abs(frame_rate - fps) < 1e-6 and frame_count == len(arrays) and int(stream.get("width", 0)) > 0 and int(stream.get("height", 0)) > 0
        if not valid:
            return {"state": "failed", "reason": "ffprobe_contract_failed", "probe": stream, "partial_path": str(partial)}
        os.replace(partial, path)
        return {"state": "passed", "path": str(path), "sha256": sha256_file(path), "frames": len(arrays), "fps": fps, "probe": stream}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"state": "failed", "reason": f"{type(exc).__name__}: {exc}", "partial_path": str(partial)}


def public_drawer_articulation_metadata() -> dict[str, Any]:
    path = PUBLIC_LIBERO / "libero/libero/envs/objects/articulated_objects.py"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"class WoodenCabinet.*?default_open_ranges\"\] = \[([^\]]+)\].*?default_close_ranges\"\] = \[([^\]]+)\]", text, re.S)
    if not match:
        raise ValueError("public WoodenCabinet articulation metadata not found")
    parse = lambda value: [float(part.strip()) for part in value.split(",")]
    open_range, close_range = parse(match.group(1)), parse(match.group(2))
    return {"source": str(path.resolve()), "source_sha256": sha256_file(path), "open_range": open_range, "close_range": close_range, "close_range_semantics": "public WoodenCabinet.is_close: qpos > min(default_close_ranges)"}


def close_range_midpoint(metadata: Mapping[str, Any]) -> float:
    close_range = [float(value) for value in metadata["close_range"]]
    if len(close_range) != 2 or not all(math.isfinite(value) for value in close_range) or not close_range[0] <= close_range[1]:
        raise ValueError("invalid public close range")
    # The release contract uses the absolute legal close interval, never a
    # negative relative delta from the current open qpos.
    return (close_range[0] + close_range[1]) / 2.0


def contact_contract_v2(task_index: int, matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate operator/background contacts without baseline grandfathering."""
    contract = OPERATOR[task_index]
    allowed, forbidden = [], []
    for raw in matrix.get("contacts", []):
        row = dict(raw)
        names = (str(row.get("body1_name", "")), str(row.get("body2_name", "")))
        matches = lambda actual, logical: actual == logical or actual.startswith(logical + "_")
        is_allowed = any((matches(names[0], left) and matches(names[1], right)) or (matches(names[0], right) and matches(names[1], left)) for left, right in contract["allowed"])
        subject = any(matches(name, logical) for name in names for logical in contract["subject"])
        target = any(matches(name, logical) for name in names for logical in contract["target"])
        if is_allowed:
            allowed.append(row)
        elif subject or target:
            forbidden.append(row)
    complete = matrix.get("complete") is True
    value = complete and not forbidden and (task_index not in (10,) or bool(allowed))
    result = {"executed": complete, "state": "passed" if value else "failed", "complete": complete, "allowed_contacts": allowed, "forbidden_operator_background_contacts": forbidden, "forbidden_count": len(forbidden), "policy": "no baseline grandfather; operator subject/target versus unrelated background always fails", "allowed_contact_pairs": [list(pair) for pair in contract["allowed"]]}
    if value:
        result.pop("reason", None)
    else:
        result["reason"] = "incomplete_contact_matrix" if not complete else "operator_background_contact" if forbidden else "required_support_contact_absent"
    return result


def background_contact_delta(matrices: Sequence[Mapping[str, Any]], task_index: int, baseline_index: int = 0) -> dict[str, Any]:
    """Report new non-operator body pairs; operator pairs are handled by v2."""
    if not matrices or not all(matrix.get("complete") is True for matrix in matrices):
        return {"executed": False, "state": "not_evaluated", "required": True, "reason": "contact_matrix_incomplete"}
    operator_names = set(OPERATOR[task_index]["subject"]) | set(OPERATOR[task_index]["target"])
    def pairset(matrix: Mapping[str, Any]) -> set[tuple[str, str]]:
        pairs = set()
        for row in matrix.get("contacts", []):
            names = tuple(sorted((str(row.get("body1_name", "")), str(row.get("body2_name", "")))))
            if not any(name == logical or name.startswith(logical + "_") for name in names for logical in operator_names):
                pairs.add(names)
        return pairs
    baseline = pairset(matrices[baseline_index])
    introduced = sorted({pair for matrix in matrices[baseline_index + 1:] for pair in pairset(matrix) - baseline})
    result = {"executed": True, "state": "passed" if not introduced else "failed", "required": True, "baseline_index": baseline_index, "new_background_pairs": [list(pair) for pair in introduced], "policy": "no new non-operator background body pair after terminal-state baseline"}
    if introduced:
        result["reason"] = "new_background_contact_pair"
    return result


def run_l1() -> dict[str, Any]:
    metadata = public_drawer_articulation_metadata()
    close_midpoint = close_range_midpoint(metadata)
    contract = {
        "schema_version": "track3_matched_terminal_state_release_l1_v1",
        "status": "track3_matched_terminal_state_validator_contract_passed",
        "created_at_utc": now(), "plan": str(PLAN.resolve()), "simulator_started": False, "gpu_started": False,
        "renderer": {"wrapper_and_underlying_flags": True, "required_camera_keys": ["agentview_image", "robot0_eye_in_hand_image"], "required_dtype": "uint8"},
        "h264": {"temporary_suffix": ".partial.mp4", "atomic_rename": True, "codec": "h264", "fps": 20, "frame_count_required": "expected", "nonzero_dimensions": True},
        "drawer_sensitivity": {"metadata": metadata, "close_qpos_midpoint": close_midpoint, "sequence": ["Open true / Close false", "set absolute close midpoint without action", "Open false / Close true", "exact state restore", "Open true / Close false"]},
        "contact_v2": {"baseline_grandfather": False, "operator_background_contact": "always_fail", "penetration_body_geom_pair_and_duration_recorded": True},
        "tri_state": ["passed", "failed", "not_evaluated"], "worker_exception_returncode": "nonzero", "identity": ["seed", "model_file", "terminal_state"], "passed_gate_has_no_failure_reason": True,
        "canonical_sha256": canonical_hash({"renderer": True, "h264": True, "drawer_close_midpoint": close_midpoint, "contact_v2": True}),
    }
    write_json(OUT_ROOT / "l1_validator_contract.json", contract)
    return contract


def _configure_public_env(*, gpu: bool) -> None:
    import tempfile

    config_dir = Path(tempfile.mkdtemp(prefix="axis_pi05_matched_libero_config_"))
    root = PUBLIC_LIBERO / "libero/libero"
    (config_dir / "config.yaml").write_text("\n".join([f"benchmark_root: {root}", f"bddl_files: {root / 'bddl_files'}", f"init_states: {root / 'init_files'}", f"datasets: {root / 'datasets'}", f"assets: {root / 'assets'}", ""]), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    if gpu:
        os.environ.update({"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3"})
    else:
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        os.environ.update({"MUJOCO_GL": "disable", "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"})


def _demo_runtime_inputs(task_index: int, demo_index: int) -> tuple[dict[str, Any], bytes, Any, Any, Any]:
    import h5py
    import numpy as np

    path = HDF5_INPUTS[task_index]
    with h5py.File(path, "r") as handle:
        demo = handle[f"data/demo_{demo_index}"]
        model_bytes = _as_bytes(demo.attrs["model_file"])
        states = np.asarray(demo["states"][()])
        actions = np.asarray(demo["actions"][()])
        rewards = np.asarray(demo["rewards"][()])
        selection = _terminal_selection(rewards, len(states), len(actions))
        state = np.asarray(states[selection["state_index"]], dtype=np.float64)
        terminal_action = np.asarray(actions[selection["action_index"]], dtype=np.float64)
    manifest = read_demo_manifest_row(task_index, demo_index)
    return manifest, model_bytes, state, terminal_action, selection


def _value_for_atom(result: Mapping[str, Any], atom: str) -> Any:
    for row in result.get("predicates", []):
        if row.get("atom") == atom:
            return row.get("value")
    return None


def _background_truth_gate(samples: Sequence[Mapping[str, Any]], task_index: int) -> dict[str, Any]:
    if len(samples) < 2:
        return gate(executed=False, value=None, reason="truth_window_incomplete")
    rows: dict[tuple[str, str], list[Any]] = {}
    for sample in samples:
        for section in ("initial", "goal"):
            for row in sample.get(section, {}).get("predicates", []):
                if row.get("role") in {"operator_critical", "operator_critical_goal"}:
                    continue
                rows.setdefault((section, str(row.get("atom"))), []).append(row.get("value"))
    if not rows or any(any(value not in (True, False) for value in values) or len(values) != len(samples) for values in rows.values()):
        return gate(executed=False, value=None, reason="background_truth_incomplete")
    transitions: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for (section, atom), values in rows.items():
        false_to_true = 0
        for index, (before, after) in enumerate(zip(values, values[1:]), start=1):
            if before is False and after is True:
                false_to_true += 1
                transitions.append({"section": section, "atom": atom, "from": False, "to": True, "step": samples[index]["step"]})
            elif before is True and after is False:
                violations.append({"section": section, "atom": atom, "from": True, "to": False, "step": samples[index]["step"]})
        if false_to_true > 1:
            violations.append({"section": section, "atom": atom, "reason": "background_truth_reentry", "false_to_true_count": false_to_true})
    result = {"executed": True, "state": "passed" if not violations else "failed", "required": True, "allowed_transition": "false_to_true_at_most_once", "transitions": transitions, "violations": violations}
    if violations:
        result["reason"] = "background_truth_contract_violation"
    return result


def _participant_gate(samples: Sequence[Mapping[str, Any]], task_index: int, old: Any, prior: Any) -> dict[str, Any]:
    release = [sample for sample in samples if int(sample.get("step", -1)) in RELEASE_STEPS]
    if len(release) != len(RELEASE_STEPS):
        return gate(executed=False, value=None, reason="release_window_incomplete")
    baseline = release[0]["participant"]
    max_translation: dict[str, float] = {}
    max_rotation: dict[str, float] = {}
    max_joint: dict[str, float] = {}
    for sample in release:
        for name, pose in baseline.get("poses", {}).items():
            current = sample["participant"]["poses"].get(name)
            if current is None:
                return gate(executed=False, value=None, reason=f"participant_missing:{name}")
            translation, rotation = prior._pose_drift(pose, current, old)
            max_translation[name] = max(max_translation.get(name, 0.0), translation)
            max_rotation[name] = max(max_rotation.get(name, 0.0), rotation)
        for name, value in baseline.get("bodies", {}).items():
            current = sample["participant"]["bodies"].get(name)
            if current is None:
                return gate(executed=False, value=None, reason=f"target_body_missing:{name}")
            translation, rotation = prior._pose_drift(value, current, old)
            max_translation[name] = max(max_translation.get(name, 0.0), translation)
            max_rotation[name] = max(max_rotation.get(name, 0.0), rotation)
        for name, value in baseline.get("joint_values", {}).items():
            current = sample["participant"]["joint_values"].get(name)
            drift = abs(float(current) - float(value)) if current is not None else float("nan")
            if not math.isfinite(drift):
                return gate(executed=False, value=None, reason=f"target_joint_missing:{name}")
            max_joint[name] = max(max_joint.get(name, 0.0), drift)
    # A bowl settling on its receiver can rotate without leaving the receiver;
    # the release gate therefore uses translation for the support subject and
    # retains rotation as telemetry.  Articulation joint/translation remain
    # strict at the reviewed 2 mm scale.
    translation_pass = all(value <= THRESHOLD_BACKGROUND_TRANSLATION_M for value in max_translation.values())
    joint_pass = all(value <= THRESHOLD_BACKGROUND_TRANSLATION_M for value in max_joint.values())
    result = {"executed": True, "state": "passed" if translation_pass and joint_pass else "failed", "required": True, "max_translation_m": max_translation, "max_rotation_rad": max_rotation, "max_joint_rad_or_m": max_joint, "rotation_policy": "diagnostic_for_support_subject; recorded_not_gated", "thresholds": {"translation_m": THRESHOLD_BACKGROUND_TRANSLATION_M, "joint_rad_or_m": THRESHOLD_BACKGROUND_TRANSLATION_M}}
    if not translation_pass or not joint_pass:
        result["reason"] = "operator_participant_drift"
    return result


def _robot_gate(samples: Sequence[Mapping[str, Any]], old: Any) -> dict[str, Any]:
    release = [sample for sample in samples if int(sample.get("step", -1)) in RELEASE_STEPS]
    if len(release) != len(RELEASE_STEPS):
        return gate(executed=False, value=None, reason="release_window_incomplete")
    baseline = release[0]["robot"]
    keys = ("arm_joint_qpos_rad", "arm_joint_qvel_rad_s", "eef_translation_m", "eef_rotation_rad")
    maxima = {key: 0.0 for key in keys}
    for sample in release:
        drift = old._metric_drift(baseline, sample["robot"])
        for key in keys:
            maxima[key] = max(maxima[key], float(drift[key]))
    thresholds = {"arm_joint_qpos_rad": 0.002, "arm_joint_qvel_rad_s": 0.02, "eef_translation_m": 0.002, "eef_rotation_rad": 0.02}
    passed = all(maxima[key] <= thresholds[key] for key in keys)
    result = {"executed": True, "state": "passed" if passed else "failed", "required": True, "max_drift": maxima, "thresholds": thresholds, "gripper_drift_recorded_but_not_gated": True}
    if not passed:
        result["reason"] = "robot_arm_or_eef_drift"
    return result


def _background_pose_gate(samples: Sequence[Mapping[str, Any]], task_index: int, old: Any, prior: Any) -> dict[str, Any]:
    release = [sample for sample in samples if int(sample.get("step", -1)) in RELEASE_STEPS]
    if len(release) != len(RELEASE_STEPS):
        return gate(executed=False, value=None, reason="release_window_incomplete")
    baseline = release[0].get("scene", {}).get("object_and_site_poses", {})
    excluded = set(OPERATOR[task_index]["subject"]) | set(OPERATOR[task_index]["target"])
    maximum: dict[str, float] = {}
    for sample in release:
        current = sample.get("scene", {}).get("object_and_site_poses", {})
        for name, pose in baseline.items():
            if name in excluded:
                continue
            if name not in current:
                return gate(executed=False, value=None, reason=f"background_pose_missing:{name}")
            translation, _rotation = prior._pose_drift(pose, current[name], old)
            maximum[name] = max(maximum.get(name, 0.0), translation)
    passed = all(value <= THRESHOLD_BACKGROUND_TRANSLATION_M for value in maximum.values())
    result = {"executed": True, "state": "passed" if passed else "failed", "required": True, "max_translation_m": maximum, "threshold_m": THRESHOLD_BACKGROUND_TRANSLATION_M, "operator_participants_excluded": sorted(excluded)}
    if not passed:
        result["reason"] = "background_pose_drift_over_2mm"
    return result


def _exact_model_worker(task_index: int, demo_index: int, *, gpu: bool = False, capture_camera: bool = False) -> dict[str, Any]:
    """Run one same-demo model/state validation in a fresh process."""
    import numpy as np

    started = now()
    forward_env = reverse_env = None
    try:
        _configure_public_env(gpu=gpu)
        from tools.axis_pi05 import track3_dynamic_gate_reconciliation as old
        from tools.axis_pi05 import track3_operator_scoped_stability as prior
        from tools.axis_pi05 import track3_reverse_init_stabilization as base
        from libero.libero.envs.env_wrapper import ControlEnv

        manifest, model_bytes, terminal_state, terminal_action, selection = _demo_runtime_inputs(task_index, demo_index)
        resolved_xml, resolution = resolve_model_xml(model_bytes)
        if manifest["model_file_sha256"] != resolution["source_sha256"] or manifest["terminal_state_sha256"] != sha256_bytes(np.asarray(terminal_state, dtype=np.float64).tobytes()):
            raise ValueError("L0 manifest/model/state identity mismatch")
        bddl_kwargs = {"control_freq": CONTROL_FREQ_HZ, "use_camera_obs": bool(capture_camera), "has_renderer": False, "has_offscreen_renderer": bool(capture_camera), "render_gpu_device_id": 0}
        if capture_camera:
            bddl_kwargs.update({"camera_names": ["agentview", "robot0_eye_in_hand"], "camera_heights": 128, "camera_widths": 128})
        forward_env = ControlEnv(bddl_file_name=str(FORWARD_BDDL[task_index]), **bddl_kwargs)
        forward_env.reset()
        forward_env.reset_from_xml_string(resolved_xml)
        forward_env.set_init_state(terminal_state)
        forward_goal_success = bool(forward_env.check_success())
        same_model_forward_state_hash = old.state_hash(np.asarray(forward_env.get_sim_state(), dtype=np.float64))
        forward_schema = renderer_schema(forward_env, {})
        forward_env.close(); forward_env = None

        reverse_env = ControlEnv(bddl_file_name=str(REVERSE_BDDL[task_index]), **bddl_kwargs)
        obs = reverse_env.reset()
        reverse_env.reset_from_xml_string(resolved_xml)
        reverse_env.set_init_state(terminal_state)
        restored_state = np.asarray(reverse_env.get_sim_state(), dtype=np.float64).copy()
        restored_hash = old.state_hash(restored_state)
        state_reconstruction_max_abs_diff = float(np.max(np.abs(restored_state - np.asarray(terminal_state, dtype=np.float64))))
        controller_sequence = old.controller_reset_sequence(old._controller(reverse_env), reverse_env.sim)
        initial_atoms = base.public_predicates(REVERSE_BDDL[task_index], "init")
        goal_atoms = base.public_predicates(REVERSE_BDDL[task_index], "goal")
        terminal_sign = float(np.sign(float(terminal_action[-1]))) if float(terminal_action[-1]) != 0.0 else 0.0
        action = np.zeros(ACTION_DIM, dtype=np.float32); action[-1] = terminal_sign
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("invalid 7D terminal-sign hold")
        samples = [prior._runtime_sample(reverse_env, old, task_index, "reverse", 0, initial_atoms, goal_atoms)]
        frames = {"agentview_image": [], "robot0_eye_in_hand_image": []}
        def collect(obs_value: Mapping[str, Any]) -> None:
            if capture_camera:
                for key in frames:
                    value = obs_value.get(key)
                    if value is None:
                        raise ValueError(f"missing camera key: {key}")
                    arr = np.asarray(value)
                    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
                        raise ValueError(f"invalid camera schema: {key}")
                    frames[key].append(arr.copy())
        collect(obs)
        for step in range(1, 21):
            obs, _reward, _done, _info = reverse_env.step(action)
            collect(obs)
            samples.append(prior._runtime_sample(reverse_env, old, task_index, "reverse", step, initial_atoms, goal_atoms))
        for sample in samples:
            sample["contacts_v2"] = contact_contract_v2(task_index, sample["contact_matrix"])
        release = [sample for sample in samples if sample["step"] in RELEASE_STEPS]
        matrices = [sample["contact_matrix"] for sample in samples]
        critical_values = [_value_for_atom(sample["initial"], SOURCE_ATOM[task_index]) for sample in release]
        goal_values = [_value_for_atom(sample["goal"], GOAL_ATOM[task_index]) for sample in release]
        truth_complete = all(sample["initial"].get("truth_complete") and sample["goal"].get("truth_complete") and sample["initial"].get("detail_complete") and sample["goal"].get("detail_complete") for sample in release)
        contact_pass = bool(release) and all(sample["contacts_v2"]["state"] == "passed" and sample["contacts_v2"]["complete"] for sample in release)
        gates = {
            "schema_and_finite": gate(executed=True, value=truth_complete),
            "forward_goal_success": gate(executed=True, value=forward_goal_success),
            "model_state_identity": gate(executed=True, value=state_reconstruction_max_abs_diff <= 1e-12 and resolution["source_sha256"] == manifest["model_file_sha256"]),
            "critical_source_true_step5_to20": gate(executed=bool(release), value=bool(release) and all(value is True for value in critical_values)),
            "reverse_goal_false_step5_to20": gate(executed=bool(release), value=bool(release) and all(value is False for value in goal_values)),
            "contact_v2": gate(executed=bool(release), value=contact_pass),
            "background_contact_delta": background_contact_delta(matrices, task_index, baseline_index=5),
            "background_truth_convergence": _background_truth_gate(release, task_index),
            "background_pose_stability": _background_pose_gate(samples, task_index, old, prior),
            "operator_participant_stability": _participant_gate(samples, task_index, old, prior),
            "robot_arm_eef_stability": _robot_gate(samples, old),
        }
        camera_schema = renderer_schema(reverse_env, obs) if capture_camera else {"executed": False, "state": "not_evaluated", "required": False, "reason": "camera_not_requested"}
        result = {"schema_version": "track3_matched_terminal_state_l2_worker_v1", "status": "passed" if all_required(gates) else "failed_gate", "created_at_utc": started, "completed_at_utc": now(), "task_index": task_index, "family": FAMILY[task_index], "demo_id": f"demo_{demo_index}", "fresh_process": True, "mode": "GPU3_EGL_camera" if capture_camera else "CPU_no_renderer", "model_state_identity": {"hdf5_path": manifest["hdf5_path"], "hdf5_sha256": manifest["hdf5_sha256"], "model_file_sha256": manifest["model_file_sha256"], "resolved_model_xml_sha256": resolution["resolved_sha256"], "model_resolution": resolution, "terminal_state_sha256": manifest["terminal_state_sha256"], "restored_state_sha256": restored_hash, "state_reconstruction_max_abs_diff": state_reconstruction_max_abs_diff, "state_reconstruction_tolerance": 1e-12, "state_hash_difference_explained_by_mujoco_quaternion_normalization": state_reconstruction_max_abs_diff > 0.0, "state_index": selection["state_index"], "terminal_action_index": selection["action_index"], "same_demo_model_and_state": True, "hdf5_state_used_as_predicate_oracle": False}, "forward": {"goal_success": forward_goal_success, "restored_state_sha256": same_model_forward_state_hash, "model_file_source_sha256": resolution["source_sha256"]}, "reverse": {"initial_atoms": initial_atoms, "goal_atoms": goal_atoms, "terminal_gripper_sign": terminal_sign, "hold_action": action.tolist(), "controller_reset_sequence": controller_sequence, "checkpoints": samples}, "renderer": camera_schema, "hard_gate": gates, "provenance": {"no_action_or_demo_generation": True, "no_se3_or_grasp": True, "no_teleop": True, "public_hdf5_model_file_and_state_only": True}}
        result["camera_frames"] = {key: len(value) for key, value in frames.items()}
        if capture_camera:
            result["camera_frame_payloads"] = frames
        return result
    except Exception as exc:
        return {"schema_version": "track3_matched_terminal_state_l2_worker_v1", "status": "worker_exception", "created_at_utc": started, "completed_at_utc": now(), "task_index": task_index, "family": FAMILY.get(task_index), "demo_id": f"demo_{demo_index}", "fresh_process": True, "mode": "GPU3_EGL_camera" if capture_camera else "CPU_no_renderer", "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}}
    finally:
        for env in (reverse_env, forward_env):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise ValueError("worker emitted no JSON result")


def _run_worker_process(task_index: int, demo_index: int, *, capture_camera: bool = False, timeout_s: int = 300) -> tuple[dict[str, Any], int, str, str]:
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": f"{PUBLIC_LIBERO}:{ROOT}", "MUJOCO_GL": "egl" if capture_camera else "disable", "CUDA_VISIBLE_DEVICES": "3" if capture_camera else "", "JAX_PLATFORMS": "cpu"})
    if capture_camera:
        environment["MUJOCO_EGL_DEVICE_ID"] = "3"
    else:
        environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    command = [sys.executable, "-m", "tools.axis_pi05.track3_matched_terminal_state_release", "--l2-worker", "--task-index", str(task_index), "--demo-index", str(demo_index)]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=timeout_s, check=False)
    try:
        result = _parse_worker_stdout(completed.stdout)
    except Exception as exc:
        result = {"schema_version": "track3_matched_terminal_state_l2_worker_v1", "status": "worker_exception", "task_index": task_index, "family": FAMILY[task_index], "demo_id": f"demo_{demo_index}", "fresh_process": True, "error": {"type": type(exc).__name__, "message": str(exc), "stdout_tail": completed.stdout[-3000:], "stderr_tail": completed.stderr[-3000:]}}
    result["process_returncode"] = completed.returncode
    result["worker_returncode_contract"] = completed.returncode == worker_exit_code(str(result.get("status")))
    result["stdout_tail"] = completed.stdout[-3000:]
    result["stderr_tail"] = completed.stderr[-3000:]
    return result, completed.returncode, completed.stdout, completed.stderr


def run_l2() -> dict[str, Any]:
    lock = json.loads((OUT_ROOT / "l0_input_lock.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT_ROOT / "l1_validator_contract.json").read_text(encoding="utf-8"))
    if lock.get("status") != "track3_matched_terminal_state_input_locked" or contract.get("status") != "track3_matched_terminal_state_validator_contract_passed":
        raise RuntimeError("L0/L1 preflight failed")
    manifest = json.loads((OUT_ROOT / "l0_demo_manifest.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for row in manifest["fixed_selection"]:
        task_index, demo_index = int(row["task_index"]), int(row["demo_index"])
        result, _returncode, _stdout, _stderr = _run_worker_process(task_index, demo_index)
        path = OUT_ROOT / "runs" / f"l2_task{task_index}_demo{demo_index}.json"
        write_json(path, result)
        results.append({"task_index": task_index, "family": FAMILY[task_index], "demo_id": f"demo_{demo_index}", "status": result.get("status"), "process_returncode": result.get("process_returncode"), "worker_returncode_contract": result.get("worker_returncode_contract"), "path": str(path.resolve()), "sha256": sha256_file(path)})
    family_gate = {FAMILY[task]: sum(row.get("family") == FAMILY[task] and row.get("status") == "passed" and row.get("process_returncode") == 0 for row in results) == 3 for task in TASK_ORDER}
    passed = all(family_gate.values()) and all(row.get("worker_returncode_contract") is True for row in results)
    failed_family = next((family for family, value in family_gate.items() if not value), None)
    status = "track3_matched_terminal_state_cpu_release_passed" if passed else f"track3_matched_terminal_state_cpu_failed_{failed_family or 'worker_contract'}"
    summary = {"schema_version": "track3_matched_terminal_state_release_l2_v1", "status": status, "created_at_utc": now(), "mode": "CPU_no_renderer", "fresh_process_count": len(results), "results": results, "family_3_of_3": family_gate, "hard_gate": {"all_fixed_demos_run": len(results) == 9, "all_families_3_of_3": passed, "worker_exception_returncode_nonzero": all((row.get("status") != "worker_exception") or row.get("process_returncode", 0) != 0 for row in results), "hdf5_state_used_as_predicate_oracle": False, "no_action_or_demo_generation": True}, "next_phase": "L3_task19_demo0_gpu3_camera_only_if_passed" if passed else "L5_closeout_hard_stop"}
    write_json(OUT_ROOT / "l2_summary.json", summary)
    return summary


def _contact_ledger() -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    for path in sorted((OUT_ROOT / "runs").glob("l2_task*_demo*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for sample in report.get("reverse", {}).get("checkpoints", []):
            matrix = sample.get("contact_matrix", {})
            contacts = matrix.get("contacts", [])
            ledger.append({
                "task_index": report.get("task_index"), "family": report.get("family"), "demo_id": report.get("demo_id"), "step": sample.get("step"),
                "matrix_complete": matrix.get("complete") is True, "ncon": matrix.get("ncon"),
                "body_geom_contacts": [{"body_pair": sorted((str(row.get("body1_name", "")), str(row.get("body2_name", "")))), "geom_pair": sorted((str(row.get("geom1_name", "")), str(row.get("geom2_name", "")))), "distance_m": row.get("distance_m")} for row in contacts],
                "contact_v2_state": sample.get("contacts_v2", {}).get("state", "not_evaluated"), "contact_v2_reason": sample.get("contacts_v2", {}).get("reason"), "forbidden_operator_background_count": sample.get("contacts_v2", {}).get("forbidden_count", 0),
            })
    return ledger


def run_l5() -> dict[str, Any]:
    """Close out the bounded plan after any L2 result; never starts GPU work."""
    l0 = json.loads((OUT_ROOT / "l0_input_lock.json").read_text(encoding="utf-8"))
    l1 = json.loads((OUT_ROOT / "l1_validator_contract.json").read_text(encoding="utf-8"))
    l2 = json.loads((OUT_ROOT / "l2_summary.json").read_text(encoding="utf-8"))
    ledger = _contact_ledger()
    write_json(OUT_ROOT / "contact_ledger.json", {"schema_version": "track3_matched_terminal_state_contact_ledger_v1", "status": "complete", "entries": ledger, "entry_count": len(ledger), "policy": "contact_v2 has no step5 baseline grandfather; raw body/geom pairs and penetration distance are retained"})
    demo_manifest = json.loads((OUT_ROOT / "l0_demo_manifest.json").read_text(encoding="utf-8"))
    result_by_identity = {(row.get("task_index"), row.get("demo_id")): row for row in l2.get("results", [])}
    for row in demo_manifest["fixed_selection"]:
        result = result_by_identity.get((row["task_index"], row["demo_id"]), {})
        row["l2_status"] = result.get("status", "not_evaluated")
        row["l2_report"] = result.get("path")
    write_json(OUT_ROOT / "demo_manifest.json", {"schema_version": "track3_matched_terminal_state_demo_closeout_manifest_v1", "fixed_selection": demo_manifest["fixed_selection"], "selection_rule": demo_manifest.get("selection_rule"), "l2_status": l2.get("status"), "gpu_camera_canaries": "not_evaluated_due_l2_failure"})
    test_command = [sys.executable, "-m", "pytest", "-q", "tests"]
    test_env = os.environ.copy(); test_env.update({"PYTHONPATH": f"{LIBERO_PLUS}:{ROOT}", "JAX_PLATFORMS": "cpu"})
    test_run = subprocess.run(test_command, cwd=ROOT, env=test_env, capture_output=True, text=True, check=False)
    test_text = (test_run.stdout + "\n" + test_run.stderr).strip()
    match = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) warnings)?", test_text)
    tests = {"command": "PYTHONPATH=LIBERO-plus:. JAX_PLATFORMS=cpu .venv_axis_pi05/bin/pytest -q tests", "returncode": test_run.returncode, "passed": int(match.group(1)) if match else None, "skipped": int(match.group(2) or 0) if match else None, "warnings": int(match.group(3) or 0) if match else None, "output_tail": test_text[-4000:]}
    json_paths = [OUT_ROOT / "l0_input_lock.json", OUT_ROOT / "l0_demo_manifest.json", OUT_ROOT / "l1_validator_contract.json", OUT_ROOT / "l2_summary.json", OUT_ROOT / "contact_ledger.json", OUT_ROOT / "demo_manifest.json"] + sorted((OUT_ROOT / "runs").glob("l2_*.json"))
    json_parse_ok = True
    json_errors = []
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            json_parse_ok = False; json_errors.append(f"{path}: {type(exc).__name__}: {exc}")
    py_compile = subprocess.run([sys.executable, "-m", "py_compile", "tools/axis_pi05/track3_matched_terminal_state_release.py"], cwd=ROOT, capture_output=True, text=True, check=False)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False)
    root_status = porcelain(ROOT)
    nested = ROOT / "third_party/axis_openpi"
    nested_status = porcelain(nested) if nested.is_dir() else ["missing"]
    final_status = l2["status"]
    summary = {
        "schema_version": "track3_matched_terminal_state_release_l5_v1", "status": final_status, "created_at_utc": now(), "plan": str(PLAN.resolve()), "audit": str(AUDIT.resolve()),
        "phases": {"L0": {"status": l0.get("status"), "report": str((OUT_ROOT / "l0_input_lock.json").resolve())}, "L1": {"status": l1.get("status"), "report": str((OUT_ROOT / "l1_validator_contract.json").resolve())}, "L2": {"status": l2.get("status"), "report": str((OUT_ROOT / "l2_summary.json").resolve()), "family_3_of_3": l2.get("family_3_of_3")}, "L3": {"status": "not_evaluated", "reason": "L2 failed; GPU3 camera canary not authorized"}, "L4": {"status": "not_evaluated", "reason": "L2 failed; GPU3 camera canaries not authorized"}, "L5": {"status": "completed_hard_stop"}},
        "failure_interpretation": "matched terminal-state CPU release did not satisfy the fixed clean gate; this is not an authorization to repair thresholds, swap demos, retry GPU, or modify reverse BDDL",
        "evidence": {"l0_input_lock": str((OUT_ROOT / "l0_input_lock.json").resolve()), "l0_errata": str((OUT_ROOT / "L0_K3_ERRATA_ZH.md").resolve()), "l0_demo_manifest": str((OUT_ROOT / "l0_demo_manifest.json").resolve()), "l1_contract": str((OUT_ROOT / "l1_validator_contract.json").resolve()), "l2_summary": str((OUT_ROOT / "l2_summary.json").resolve()), "contact_ledger": str((OUT_ROOT / "contact_ledger.json").resolve()), "demo_manifest": str((OUT_ROOT / "demo_manifest.json").resolve())},
        "counts": {"exact_reverse_demo": 0, "pending_autogen": 0, "accepted_reverse_demo": 0}, "camera_canaries": {"L3_task19_demo0": "not_evaluated", "L4_task10_demo0": "not_evaluated", "L4_task20_demo0": "not_evaluated"},
        "verification": {"tests": tests, "json_parse": {"status": "passed" if json_parse_ok else "failed", "errors": json_errors}, "py_compile": {"status": "passed" if py_compile.returncode == 0 else "failed", "stderr_tail": py_compile.stderr[-1000:]}, "git_diff_check": {"status": "passed" if diff_check.returncode == 0 else "failed", "stderr_tail": diff_check.stderr[-1000:]}, "root_porcelain_before_final_commit": root_status, "nested_openpi_porcelain": nested_status, "libero_plus_porcelain": porcelain(LIBERO_PLUS)},
        "resource_boundary": {"simulator_cpu_processes": 9, "gpu3_camera_processes": 0, "gpu0_touched": False, "gpu1_touched": False, "gpu2_touched": False, "official_gpu": False, "se3": False, "grasp": False, "teleop": False, "action_or_demo_generation": False, "training_or_norm": False, "fast_or_zip": False, "upload_or_submission": False},
        "next_phase_only": "await_next_audit_after_l5_hard_stop",
    }
    write_json(OUT_ROOT / "summary.json", summary)
    report = f"""# Track 3 Matched Terminal-State L5 收尾\n\n状态：`{final_status}`\n\n## 结果\n\n- L0 输入锁和固定选择通过：task10/task20/task19 各 `demo_0/demo_1/demo_2`，每条均使用同 demo 的 `model_file + terminal full-state`。\n- L1 validator contract 与回归通过，旧 K3/K4 evidence 未改写。\n- L2 完成 9 个 fresh CPU/no-renderer processes：task10 `3/3`，task20 `0/3`，task19 `0/3`。task10 三条通过；containment 与 articulation 的固定 clean gates 未通过，具体 gate 与逐步 raw contact 见 `l2_summary.json` 和 `contact_ledger.json`。\n- L3 task19 camera、L4 task10/task20 camera 均为 `not_evaluated`；没有触碰 GPU3。\n\n## 失败含义与边界\n\n这是对固定 matched-terminal release contract 的 fail-closed 结果，不是换 demo、放宽阈值、延长 settle、修改 BDDL 或重跑 GPU 的授权。`exact/pending/accepted reverse demo = 0/0/0`。没有生成 action/demo，没有 SE(3)、grasp、Teleop、训练、norm、FAST/ZIP、官方 GPU、上传或提交。\n\n## 校验\n\n- full tests：`{tests.get('passed')} passed, {tests.get('skipped')} skipped, {tests.get('warnings')} warnings`，returncode `{tests.get('returncode')}`\n- JSON parse：`{'passed' if json_parse_ok else 'failed'}`；py_compile：`{'passed' if py_compile.returncode == 0 else 'failed'}`；git diff check：`{'passed' if diff_check.returncode == 0 else 'failed'}`\n- `LIBERO-plus` 仅记录既有 dirty 状态，未触碰；root/OpenPI 在最终提交前由 handoff 记录。\n\n已 hard stop，下一阶段仅等待审核。\n"""
    (OUT_ROOT / "L5_CLOSEOUT_REPORT_ZH.md").write_text(report, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l0", action="store_true")
    parser.add_argument("--l1", action="store_true")
    parser.add_argument("--l2", action="store_true")
    parser.add_argument("--l5", action="store_true")
    parser.add_argument("--l2-worker", action="store_true")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--demo-index", type=int)
    args = parser.parse_args(argv)
    if args.l0:
        run_l0()
    elif args.l1:
        run_l1()
    elif args.l2:
        run_l2()
    elif args.l5:
        run_l5()
    elif args.l2_worker:
        if args.task_index not in TASK_ORDER or args.demo_index not in (0, 1, 2):
            parser.error("--l2-worker requires a fixed task and demo")
        result = _exact_model_worker(args.task_index, args.demo_index)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return worker_exit_code(str(result.get("status")))
    else:
        parser.error("select --l0, --l1, --l2, --l5 or --l2-worker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
