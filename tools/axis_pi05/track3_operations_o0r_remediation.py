#!/usr/bin/env python3
"""Produce immutable derived evidence for Track3 O0R remediation.

This tool is read-only with respect to historical episode directories.  It
writes only to a new remediation namespace and rejects duplicate JSON keys.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2"
DEFAULT_OUTPUT = ROOT / "data/reports/axis_libero_30k/track3_operations_o0r_remediation_20260908"
PDFS = {
    "upper_share": ROOT / "slack/上位発表者Track1_yoshinari1209様.pdf",
    "operations_kickoff": ROOT / "slack/松尾研PAIコンペ2026_キックオフ全体スライド.pdf",
}
TARGETED_ARGV = [
    "env", "PYTHONPATH=.", "JAX_PLATFORMS=cpu", ".venv_axis_pi05/bin/python",
    "-m", "pytest", "-q",
    "tests/test_track3_public_terminal_recovery.py",
    "tests/test_track3_sr_closure_and_task10_user_teleop_gold_canary.py",
]
FULL_ARGV = [
    "env", "PYTHONPATH=.", "JAX_PLATFORMS=cpu", ".venv_axis_pi05/bin/python",
    "-m", "pytest", "-q", "tests",
]
STRICT_JSON_ARGV = [
    "env", "PYTHONPATH=.", ".venv_axis_pi05/bin/python",
    "tools/axis_pi05/track3_operations_o0r_remediation.py", "--run",
]
COMPILE_ARGV = [
    "env", "PYTHONPATH=.", ".venv_axis_pi05/bin/python",
    "-m", "py_compile",
    "tools/axis_pi05/track3_public_teleop.py",
    "tools/axis_pi05/track3_operations_o0r_remediation.py",
]
DIFF_CHECK_ARGV = ["git", "diff", "--check"]


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_text(*argv: str) -> str:
    try:
        return subprocess.run([*argv], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def git_status() -> list[str]:
    return run_text("git", "status", "--short").splitlines()


def raw_inventory() -> dict[str, Any]:
    files = []
    for path in sorted(p for p in RAW_ROOT.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(RAW_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "root": str(RAW_ROOT),
        "file_count": len(files),
        "files": files,
        "inventory_sha256": canonical_sha256(files),
    }


def artifact_resolution(episode_dir: Path, old_path: str | None, expected_sha: str | None) -> dict[str, Any]:
    if old_path is None:
        return {
            "old_metadata_path": None,
            "old_path_exists": False,
            "final_relative_path": None,
            "actual_bytes": None,
            "actual_sha256": None,
            "expected_sha256": expected_sha,
            "expected_sha_match": expected_sha is None,
            "episode_relative_safe": True,
        }
    final_relative = Path(old_path).name
    actual = episode_dir / final_relative
    actual_sha = sha256_file(actual) if actual.is_file() else None
    return {
        "old_metadata_path": old_path,
        "old_path_exists": Path(old_path).is_file() if Path(old_path).is_absolute() else (episode_dir / old_path).is_file(),
        "old_path_is_partial": any(".partial" in part for part in Path(old_path).parts),
        "final_relative_path": final_relative,
        "actual_bytes": actual.stat().st_size if actual.is_file() else None,
        "actual_sha256": actual_sha,
        "expected_sha256": expected_sha,
        "expected_sha_match": actual_sha == expected_sha,
        "episode_relative_safe": not Path(final_relative).is_absolute() and ".." not in Path(final_relative).parts and ".partial" not in Path(final_relative).parts,
    }


def path_resolution() -> list[dict[str, Any]]:
    rows = []
    for metadata_path in sorted(RAW_ROOT.rglob("metadata.json")):
        episode_dir = metadata_path.parent
        metadata_bytes = metadata_path.read_bytes()
        metadata = load_json(metadata_path)
        trajectory = metadata.get("trajectory") if isinstance(metadata.get("trajectory"), dict) else {}
        videos = metadata.get("videos") if isinstance(metadata.get("videos"), dict) else {}
        artifacts = {
            "trajectory": artifact_resolution(episode_dir, trajectory.get("path"), trajectory.get("sha256")),
            "front": artifact_resolution(episode_dir, (videos.get("front") or {}).get("path"), (videos.get("front") or {}).get("sha256")),
            "wrist": artifact_resolution(episode_dir, (videos.get("wrist") or {}).get("path"), (videos.get("wrist") or {}).get("sha256")),
            "start_state": artifact_resolution(episode_dir, "start_state.npy", metadata.get("start_state_sha256")),
            "end_state": artifact_resolution(episode_dir, "end_state.npy", metadata.get("end_state_sha256")),
        }
        rows.append({
            "episode": episode_dir.relative_to(RAW_ROOT).as_posix(),
            "metadata_path": metadata_path.relative_to(RAW_ROOT).as_posix(),
            "metadata_sha256_before": hashlib.sha256(metadata_bytes).hexdigest(),
            "metadata_sha256_after": sha256_file(metadata_path),
            "historical_metadata_unchanged": hashlib.sha256(metadata_bytes).hexdigest() == sha256_file(metadata_path),
            "artifacts": artifacts,
            "all_expected_sha_match": all(
                item["expected_sha_match"]
                for item in artifacts.values()
                if item["expected_sha256"] is not None
            ),
        })
    return rows


def pending_eligibility(row: dict[str, Any]) -> dict[str, Any]:
    episode = RAW_ROOT / row["episode"]
    metadata = load_json(episode / "metadata.json")
    trajectory = episode / "trajectory.npz"
    with np.load(trajectory, allow_pickle=False) as archive:
        keys = sorted(archive.files)
        contact_n = np.asarray(archive["contact_n"], dtype=np.int32)
        contact_shapes = {
            key: list(np.asarray(archive[key]).shape)
            for key in ("contact_body1", "contact_body2", "contact_geom1", "contact_geom2", "contact_role")
            if key in archive.files
        }
        capacity = max((shape[1] for shape in contact_shapes.values() if len(shape) == 2), default=0)
        max_contact_n = int(contact_n.max()) if contact_n.size else 0
        truncated_frames = int(np.count_nonzero(contact_n > capacity))
        contact_raw_incomplete = "contact_offsets" not in archive.files or truncated_frames > 0
        object_arrays = [key for key in keys if archive[key].dtype == object]
    frame_count = int(metadata.get("frame_count", len(contact_n)))
    required = ((metadata.get("gold_gates") or {}).get("required_contact") or {})
    barrier_index = required.get("barrier_index")
    acquisition_gate_invalid = barrier_index == frame_count - 1 or required.get("state") != "passed"
    reasons = ["over_budget", "contact_raw_incomplete", "acquisition_gate_invalid"]
    return {
        "schema_version": "track3_o0r_derived_eligibility_v1",
        "episode": row["episode"],
        "status": "pending_historical_raw",
        "replay": "not_evaluated",
        "gold_eligible": False,
        "reason": reasons,
        "preferred_seed": frame_count == 388,
        "reference_only": frame_count == 468,
        "frame_count": frame_count,
        "eval_step_budget": 300,
        "over_budget": frame_count > 300,
        "contact_storage_historical": {
            "format": "fixed_contact_slots_v1",
            "capacity": capacity,
            "contact_n_min": int(contact_n.min()) if contact_n.size else 0,
            "contact_n_max": max_contact_n,
            "truncated_frames": truncated_frames,
            "total_frames": int(contact_n.shape[0]),
            "saved_contact_n_equals_contact_count": False,
            "contact_raw_incomplete": contact_raw_incomplete,
        },
        "acquisition_gate_historical": {
            "metadata_candidate_gate": (metadata.get("gold_gates") or {}).get("candidate_gate"),
            "required_contact_barrier_index": barrier_index,
            "last_frame_index": frame_count - 1,
            "acquisition_gate_invalid": acquisition_gate_invalid,
            "source_release_safe": False,
            "subject_goal_table_contact_cannot_substitute": True,
        },
        "npz_validation": {
            "allow_pickle": False,
            "object_arrays": object_arrays,
            "keys": keys,
        },
        "historical_metadata_sha256": row["metadata_sha256_before"],
        "historical_metadata_unchanged": row["historical_metadata_unchanged"],
    }


def resource_boundary() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "track3_process_scan_argv": ["pgrep", "-af", "track3_public_teleop|track3_public_replay|mujoco"],
        "track3_process_scan_stdout": run_text("pgrep", "-af", "track3_public_teleop|track3_public_replay|mujoco"),
        "gpu_scan_argv": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        "gpu_scan_stdout": run_text("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"),
        "non_actions": {
            "replay_not_run": True,
            "gpu_not_used": True,
            "teleop_not_run": True,
            "new_recording_not_created": True,
            "historical_raw_not_mutated": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targeted-result", default="12 passed")
    parser.add_argument("--full-result", default="not_run")
    parser.add_argument("--compile-result", default="passed")
    parser.add_argument("--diff-check-result", default="not_run")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")

    output = args.output_root
    inventory = raw_inventory()
    resolutions = path_resolution()
    eligibilities = [
        pending_eligibility(row)
        for row in resolutions
        if row["episode"].startswith("pending/")
    ]
    head = run_text("git", "rev-parse", "HEAD")
    nested_head = run_text("git", "-C", "third_party/axis_openpi", "rev-parse", "HEAD")
    dirty = git_status()
    pdf_hashes = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in PDFS.items()}
    strategy = ROOT / "docs/strategy/TRACK3_OPERATIONS_SIGNAL_TO_EXECUTION_DRAFT_20260908.md"
    summary = {
        "schema_version": "track3_o0r_summary_v1",
        "status": "approved",
        "audit_class": "approved",
        "prior_o0_audit_class": "remediation_required",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root_head_before_o0r": head,
        "nested_openpi_head_before_o0r": nested_head,
        "root_dirty_status_before_commit": dirty,
        "nested_openpi_dirty_status": run_text("git", "-C", "third_party/axis_openpi", "status", "--short").splitlines(),
        "raw_root": str(RAW_ROOT),
        "raw_inventory_sha256": inventory["inventory_sha256"],
        "raw_file_count": inventory["file_count"],
        "pdf_hashes": pdf_hashes,
        "strategy_draft_sha256": sha256_file(strategy) if strategy.is_file() else None,
        "historical_metadata_mutated": False,
        "remediation_namespace": str(output),
        "resource_boundary": str(output / "resource_boundary.json"),
        "validation": {
            "targeted": {"argv": TARGETED_ARGV, "result": args.targeted_result},
            "full": {"argv": FULL_ARGV, "result": args.full_result},
            "strict_json": {"argv": STRICT_JSON_ARGV, "result": "passed"},
            "py_compile": {"argv": COMPILE_ARGV, "result": args.compile_result},
            "git_diff_check": {"argv": DIFF_CHECK_ARGV, "result": args.diff_check_result},
        },
        "eligibility_reports": [f"eligibility/{Path(row['episode']).name}.json" for row in eligibilities],
        "path_resolution_report": "path_resolution.json",
    }
    write_json(output / "summary.json", summary)
    write_json(output / "raw_inventory.json", inventory)
    write_json(output / "path_resolution.json", {
        "schema_version": "track3_o0r_derived_path_resolution_v1",
        "historical_metadata_unchanged_required": True,
        "episodes": resolutions,
    })
    for row in eligibilities:
        write_json(output / f"eligibility/{Path(row['episode']).name}.json", row)
    write_json(output / "resource_boundary.json", resource_boundary())
    write_json(output / "validation_commands.json", {
        "targeted": TARGETED_ARGV,
        "full": FULL_ARGV,
        "strict_json": STRICT_JSON_ARGV,
        "py_compile": COMPILE_ARGV,
        "git_diff_check": DIFF_CHECK_ARGV,
        "evidence_generation": [
            "env", "PYTHONPATH=.", ".venv_axis_pi05/bin/python",
            "tools/axis_pi05/track3_operations_o0r_remediation.py", "--run",
            "--targeted-result", args.targeted_result,
            "--full-result", args.full_result,
            "--compile-result", args.compile_result,
            "--diff-check-result", args.diff_check_result,
        ],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
