"""Aggregate the Phase 1 gates without authorizing any training run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, write_json


PHASE1_ROOT = ROOT / "data" / "reports" / "axis_libero_30k" / "phase1"
ROLLOUT_ROOT = PHASE1_ROOT / "rollout"
OPENPI_ROOT = ROOT / "third_party" / "axis_openpi"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=OPENPI_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    implementation_status = subprocess.run(
        ["git", "status", "--short"], cwd=OPENPI_ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status,
        "third_party_axis_openpi": {
            "commit": implementation_commit,
            "dirty": bool(implementation_status),
            "status": implementation_status,
        },
    }


def summarize(out_path: Path = PHASE1_ROOT / "summary.json") -> dict[str, Any]:
    norm_manifest = _read_json(PHASE1_ROOT / "norm_manifest.json")
    finite = _read_json(PHASE1_ROOT / "state_action_finite_report.json")
    loader = _read_json(PHASE1_ROOT / "loader_audit.json")
    forward = _read_json(PHASE1_ROOT / "step0" / "fixed_forward_report.json")
    lock = _read_json(ROLLOUT_ROOT / "manifest.lock.json")

    rollout_runs: dict[str, Any] = {}
    for panel in ("C20", "B80"):
        for mode in ("execute8", "full_chunk"):
            run_id = f"{panel}_{mode}"
            summary = _read_json(ROLLOUT_ROOT / run_id / "summary.json")
            expected = int(lock.get("panels", {}).get(panel, {}).get("cases", 0))
            videos = summary.get("videos", []) if summary else []
            partial_video_paths = sorted(
                path for path in (ROLLOUT_ROOT / run_id / "videos").glob("*.h264.mp4")
                if path.is_file()
            )
            video_files = [
                path for path in videos
                if (ROLLOUT_ROOT / run_id / path).is_file()
            ]
            rollout_runs[run_id] = {
                "status": summary.get("status", "missing"),
                "case_count": summary.get("case_count", 0),
                "expected_case_count": expected,
                "success_count": summary.get("success_count", 0),
                "video_count": summary.get("video_count", 0),
                "expected_video_count": expected,
                "video_files_present": len(video_files),
                "partial_video_count": len(partial_video_paths),
                "partial_success_count": sum("__SUCCESS.h264.mp4" in path.name for path in partial_video_paths),
                "stall_rate": summary.get("stall_rate"),
                "gpu_id": summary.get("gpu_id"),
                "gpu0_touched": summary.get("gpu0_touched", True),
                "summary_sha256": sha256_file(ROLLOUT_ROOT / run_id / "summary.json")
                if (ROLLOUT_ROOT / run_id / "summary.json").is_file()
                else None,
            }

    gates = {
        "canonical_data_unique": (
            norm_manifest.get("source", {}).get("repo_id") == "physical-intelligence/libero"
            and norm_manifest.get("source", {}).get("revision")
            == "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
            and norm_manifest.get("sampling", {}).get("fps") == 10
            and norm_manifest.get("sampling", {}).get("resampling") == "none"
            and norm_manifest.get("counts") == {"episodes": 1693, "frames": 273465, "payload_files": 1699, "tasks": 40}
        ),
        "fresh_norm_finite": bool(finite.get("all_finite"))
        and not norm_manifest.get("axis_norm_stats_loaded", True),
        "loader_contract": loader.get("checks", {}).get("official_only") is True
        and all(loader.get("checks", {}).get(key) is True for key in (
            "native_10hz",
            "front_wrist_images",
            "state_8d",
            "external_action_7d",
            "action_horizon_10",
            "padding_mask_preserved",
            "instruction_task_mapping",
            "roundtrip_fp32",
            "all_probe_values_finite",
        )),
        "fixed_forward_reproducible": forward.get("status") == "passed"
        and forward.get("deterministic_bitwise") is True
        and forward.get("finite_external_actions") is True
        and forward.get("finite_internal_actions") is True
        and forward.get("optimizer_created") is False
        and forward.get("optimizer_updates") == 0
        and forward.get("axis_params", {}).get("unchanged") is True,
        "rollout_artifacts_complete": all(
            value["status"] == "passed"
            and value["case_count"] == value["expected_case_count"]
            and value["success_count"] > 0
            and value["video_count"] == value["expected_video_count"]
            and value["video_files_present"] == value["expected_video_count"]
            and value["stall_rate"] is not None
            and value["stall_rate"] <= 0.9
            and value["gpu0_touched"] is False
            for value in rollout_runs.values()
        ) and len(rollout_runs) == 4,
    }
    git_state = _git_state()
    gates["clean_implementation_commit"] = (
        not git_state["dirty"] and not git_state["third_party_axis_openpi"]["dirty"]
    )
    functional_gates = all(gates[key] for key in gates if key != "clean_implementation_commit")
    all_gates = all(gates.values())
    if all_gates:
        status = "phase1_complete_ready_for_phase2"
        next_action = "Phase 2 numerical preflight only: 1/10/50/100 steps; do not start 30K or 20 Hz training."
    elif functional_gates and not gates["clean_implementation_commit"]:
        status = "phase1_complete_pending_clean_commit"
        next_action = "Review and commit the Phase 1 implementation/configuration, then re-run this summary; training remains blocked."
    else:
        status = "phase1_in_progress"
        next_action = "Complete every failed Phase 1 gate; training remains blocked."

    result = {
        "schema_version": 1,
        "phase": "phase1",
        "status": status,
        "phase2_allowed": all_gates,
        "next_action": next_action,
        "protocol_policy": {
            "primary": "execute8",
            "diagnostic": "full_chunk",
            "paired_comparison_required": True,
            "note": "Never select or mix checkpoint results across execution protocols.",
        },
        "gates": gates,
        "rollout_runs": rollout_runs,
        "artifacts": {
            "norm_manifest": str(PHASE1_ROOT / "norm_manifest.json"),
            "norm_stats": str(PHASE1_ROOT / "norm_stats.json"),
            "loader_audit": str(PHASE1_ROOT / "loader_audit.json"),
            "finite_report": str(PHASE1_ROOT / "state_action_finite_report.json"),
            "fixed_forward": str(PHASE1_ROOT / "step0" / "fixed_forward_report.json"),
            "rollout_lock": str(ROLLOUT_ROOT / "manifest.lock.json"),
        },
        "git": git_state,
    }
    write_json(out_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PHASE1_ROOT / "summary.json")
    args = parser.parse_args()
    print(json.dumps(summarize(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
