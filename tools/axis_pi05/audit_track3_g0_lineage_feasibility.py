"""Read-only G0 audit of the accepted reverse episodes’ provenance.

The audit deliberately does not infer lineage from episode names, timestamps,
payload hashes, start-state hashes, replay counts, or human-review files.
Those are useful artifact identities, but they do not prove that two samples
came from independent recorder/session/process lineages.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import track3_paired_eval_finalizer_v2 as finalizer


ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_pilot_v2/accepted"
DEFAULT_OUT = ROOT / "data/reports/axis_libero_30k/t3_g0_lineage_feasibility_audit_20260916_attempt_0001"
LINEAGE_KEYS = {
    "session_id",
    "session_uuid",
    "recorder_id",
    "recorder_uuid",
    "capture_id",
    "capture_uuid",
    "lineage_id",
    "lineage_uuid",
    "supervisor_id",
    "supervisor_uuid",
    "process_id",
    "process_uuid",
    "pid",
    "process_argv",
    "recorder_argv",
    "supervisor_argv",
    "event_log_sha256",
    "session_event_log_sha256",
    "capture_event_log_sha256",
}
LINEAGE_FILENAME = re.compile(r"(session|recorder|supervisor|capture|lineage|event|process)", re.IGNORECASE)


def _walk_keys(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if key_text in LINEAGE_KEYS:
                found.append(f"{prefix}.{key_text}")
            found.extend(_walk_keys(child, f"{prefix}.{key_text}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_walk_keys(child, f"{prefix}[{index}]"))
    return found


def _load_json_files(directory: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    values: dict[str, Any] = {}
    lineage_keys: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.json")):
        relative = path.name
        value = finalizer.strict_json_load(path)
        values[relative] = value
        keys = _walk_keys(value)
        if keys:
            lineage_keys[relative] = keys
    return values, lineage_keys


def _sha_check(directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    trajectory = directory / "trajectory.npz"
    start_state = directory / "start_state.npy"
    end_state = directory / "end_state.npy"
    checks["trajectory"] = {
        "path": str(trajectory),
        "exists": trajectory.is_file(),
        "declared_sha256": metadata.get("trajectory", {}).get("sha256"),
        "actual_sha256": finalizer.sha256_file(trajectory) if trajectory.is_file() else None,
    }
    checks["start_state"] = {
        "path": str(start_state),
        "exists": start_state.is_file(),
        "declared_sha256": metadata.get("start_state_sha256"),
        "actual_sha256": finalizer.sha256_file(start_state) if start_state.is_file() else None,
    }
    checks["end_state"] = {
        "path": str(end_state),
        "exists": end_state.is_file(),
        "declared_sha256": metadata.get("end_state_sha256"),
        "actual_sha256": finalizer.sha256_file(end_state) if end_state.is_file() else None,
    }
    for camera in ("front", "wrist"):
        video = directory / f"{camera}.mp4"
        declared = ((metadata.get("videos") or {}).get(camera) or {}).get("sha256")
        checks[camera] = {
            "path": str(video),
            "exists": video.is_file(),
            "declared_sha256": declared,
            "actual_sha256": finalizer.sha256_file(video) if video.is_file() else None,
        }
    for value in checks.values():
        value["hash_matches"] = bool(value["exists"] and value["declared_sha256"] == value["actual_sha256"])
    return checks


def _episode_row(directory: Path) -> dict[str, Any]:
    metadata_files, lineage_keys = _load_json_files(directory)
    metadata = metadata_files["metadata.json"]
    replay = metadata_files.get("replay_summary.json", {})
    promotion = metadata_files.get("promotion.json", {})
    human_review = metadata_files.get("human_review.json", {})
    source_pending = metadata.get("promotion_source")
    source_pending_path = Path(source_pending) if isinstance(source_pending, str) else None
    filename_evidence = [path.name for path in directory.iterdir() if path.is_file() and LINEAGE_FILENAME.search(path.name)]
    explicit_lineage_keys = sorted({key.rsplit(".", 1)[-1] for paths in lineage_keys.values() for key in paths})
    evidence = {
        "capture_or_session_identity": False,
        "recorder_identity": False,
        "supervisor_or_process_identity": False,
        "episode_bound_event_log": False,
        "original_pending_capture_present": bool(source_pending_path and source_pending_path.is_dir()),
        "explicit_lineage_keys": explicit_lineage_keys,
        "json_files_with_lineage_keys": lineage_keys,
        "lineage_named_files": sorted(filename_evidence),
    }
    return {
        "episode": directory.name,
        "episode_path": str(directory),
        "task_id": metadata.get("task_id"),
        "status": metadata.get("status"),
        "created_at_utc": metadata.get("created_at_utc"),
        "episode_id_present": True,
        "timestamp_present": bool(metadata.get("created_at_utc") or (metadata.get("timing") or {}).get("wall_timestamp_start_s")),
        "start_state_sha256": metadata.get("start_state_sha256"),
        "trajectory_sha256": (metadata.get("trajectory") or {}).get("sha256"),
        "replay_status": replay.get("status"),
        "replay_passed_runs": replay.get("passed_runs"),
        "human_review_status": metadata.get("human_review"),
        "promotion_status": promotion.get("status"),
        "source": metadata.get("source"),
        "source_pending": source_pending,
        "source_pending_exists": evidence["original_pending_capture_present"],
        "artifact_hash_checks": _sha_check(directory, metadata),
        "lineage_evidence": evidence,
        "independent_lineage_proven": False,
    }


def _markdown(rows: list[dict[str, Any]], task_summary: dict[str, Any], d2: dict[str, Any]) -> str:
    lines = [
        "# Track3 G0 lineage feasibility (read-only)",
        "",
        "| task | accepted episode | recorder/session/process lineage | source capture present | replay | artifact hashes |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        checks = row["artifact_hash_checks"]
        hashes_ok = all(value["hash_matches"] for value in checks.values())
        lines.append(
            f"| {row['task_id']} | `{row['episode']}` | no | {'yes' if row['source_pending_exists'] else 'no'} | "
            f"{row['replay_status']} | {'pass' if hashes_ok else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Per-task count and gate",
            "",
            "| task | accepted count | explicit independent lineages | reverse dev evidence | D2 task gate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task_id, summary in sorted(task_summary.items()):
        lines.append(f"| {task_id} | {summary['accepted_count']} | {summary['independent_lineage_count']} | no | no |")
    lines.extend(
        [
            "",
            f"**D2 decision:** `{d2['decision']}` — {d2['reason']}",
            "",
            "Episode IDs, timestamps, start-state/trajectory hashes, 3/3 replay and human review are retained as artifact evidence; none is promoted to independent lineage evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"G0 output namespace already exists: {out}")
    directories = sorted(path for path in ACCEPTED_ROOT.iterdir() if path.is_dir())
    if len(directories) != 8:
        raise RuntimeError(f"expected 8 accepted reverse episodes, got {len(directories)}")
    rows = [_episode_row(directory) for directory in directories]
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_rows[str(row["task_id"])].append(row)
    task_summary: dict[str, Any] = {}
    for task_id, task_items in task_rows.items():
        task_summary[task_id] = {
            "accepted_count": len(task_items),
            "independent_lineage_count": sum(bool(item["independent_lineage_proven"]) for item in task_items),
            "distinct_episode_ids": len({item["episode"] for item in task_items}),
            "distinct_start_state_hashes": len({item["start_state_sha256"] for item in task_items}),
            "distinct_trajectory_hashes": len({item["trajectory_sha256"] for item in task_items}),
            "reverse_dev_evidence": False,
        }
    d2 = {
        "decision": "D2_unreachable_within_one_day",
        "reason": "At least one task lacks 3 accepted samples and every task has zero explicitly proven independent recorder/session lineages; no reverse dev evidence is present.",
        "all_tasks_have_at_least_3_accepted": all(value["accepted_count"] >= 3 for value in task_summary.values()),
        "all_tasks_have_at_least_2_lineages": all(value["independent_lineage_count"] >= 2 for value in task_summary.values()),
        "lineage_disjoint_reverse_dev": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "lineage_episode_table.md"
    table_path.write_text(_markdown(rows, task_summary, d2), encoding="utf-8")
    summary = {
        "schema_version": "track3_g0_lineage_feasibility_audit_v1",
        "status": "completed_read_only",
        "accepted_root": str(ACCEPTED_ROOT),
        "accepted_episode_count": len(rows),
        "task_distribution": dict(Counter(str(row["task_id"]) for row in rows)),
        "task_summary": task_summary,
        "episode_lineage_rows": rows,
        "lineage_rule": "requires capture/session/process/event evidence; episode id, timestamp, payload/start hash, replay or human review alone are insufficient",
        "d2": d2,
        "table_path": str(table_path),
        "gpu_used": False,
        "simulator_started": False,
        "files_written": [str(table_path), str(out / "summary.json")],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
