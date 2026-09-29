"""Run the final four-card T2 inference screen for light action smoothing.

The screen is deliberately a new adapter identity.  It reuses the frozen R6
paired evaluator only for its input/checkpoint/runtime contracts, while
wrapping the emitted action at the simulator boundary.  It never trains,
resumes H300, mutates the existing execute4 ZIP, uploads, or submits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import run_track3_fast400_h300_paired_eval as base


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/AXIS_PI05_R6_T2_LIGHT_SMOOTHING_SCREEN_PLAN_20260917.md"
ADAPTER = Path(__file__).resolve()
EVIDENCE_ROOT = ROOT / "data/reports/axis_libero_30k/r6_t2_light_smoothing_screen_20260917_attempt_0001"
OUTPUT_ROOT = ROOT / "outputs/axis_pi05/r6_t2_light_smoothing_screen_20260917_attempt_0001"

VARIANTS = {
    "execute4_reference": {"execute_actions": 4, "mode": "reference"},
    "seam_blend": {"execute_actions": 4, "mode": "seam_blend"},
    "arm_ema01": {"execute_actions": 4, "mode": "arm_ema01"},
    "execute5": {"execute_actions": 5, "mode": "execute5"},
}
VARIANT_ORDER = tuple(VARIANTS)
BASELINE_VARIANT = "execute4_reference"
CANDIDATE_VARIANTS = tuple(name for name in VARIANT_ORDER if name != BASELINE_VARIANT)
OLD_ACTION_WEIGHT = 0.1
CONTROL_DT = 1.0 / 20.0

ACTIVE_VARIANT = "execute4_reference"
_ORIGINAL_EPISODE_ROW = base._episode_row
_ORIGINAL_PREFLIGHT = base.preflight


def configure(variant: str) -> None:
    """Bind one variant before any evaluator worker state is created."""

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANT_ORDER}")
    global ACTIVE_VARIANT
    ACTIVE_VARIANT = variant
    base.EXECUTE_ACTIONS = int(VARIANTS[variant]["execute_actions"])
    base.ADAPTER_PATH = ADAPTER
    base.PLAN = PLAN
    base.Execute8Policy = VariantPolicy
    base._episode_row = _episode_row_with_eef


class VariantPolicy(base.Execute8Policy):
    """Apply only the requested arm-only postprocessing to emitted actions."""

    def __init__(self, client: Any):
        super().__init__(client)
        self.variant = ACTIVE_VARIANT
        self.previous_emitted: np.ndarray | None = None

    def reset(self, instruction: str = "", seed: int | None = None) -> None:
        super().reset(instruction=instruction, seed=seed)
        self.previous_emitted = None

    def get_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        is_new_chunk = bool(self.needs_chunk_reset)
        raw = np.asarray(super().get_action(observation), dtype=np.float32)
        action = raw.copy()
        if self.variant == "seam_blend" and is_new_chunk and self.previous_emitted is not None:
            action[:6] = (1.0 - OLD_ACTION_WEIGHT) * raw[:6] + OLD_ACTION_WEIGHT * self.previous_emitted[:6]
        elif self.variant == "arm_ema01" and self.previous_emitted is not None:
            action[:6] = (1.0 - OLD_ACTION_WEIGHT) * raw[:6] + OLD_ACTION_WEIGHT * self.previous_emitted[:6]
        # The seventh dimension is intentionally copied from the raw model
        # output for every variant, including the first frame of a new chunk.
        action[6] = raw[6]
        if action.shape != (7,) or not np.isfinite(action).all():
            raise FloatingPointError(f"postprocessed action is invalid: {action.shape}")
        self.previous_emitted = action.copy()
        return action


def _eef_jerk_rms(eef_positions: np.ndarray) -> float:
    """Return discrete EEF translation jerk RMS in metres/sec^3."""

    positions = np.asarray(eef_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 4:
        raise ValueError(f"EEF trace must be N x 3 with N >= 4, got {positions.shape}")
    if not np.isfinite(positions).all():
        raise FloatingPointError("EEF position trace contains NaN or Inf")
    jerk = np.diff(positions, n=3, axis=0) / (CONTROL_DT ** 3)
    return float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))


def _eef_trace_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / "eef_traces" / f"{base._safe_case_id(case_id)}.npy"


def _episode_row_with_eef(
    case: dict[str, Any],
    role: str,
    identity: dict[str, Any],
    episode: Any | None,
    output_dir: Path,
    *,
    error: Exception | None = None,
) -> dict[str, Any]:
    eef: np.ndarray | None = None
    eef_error: Exception | None = None
    if error is None and episode is not None:
        try:
            eef = np.asarray(getattr(episode, "ee_positions", []), dtype=np.float32).reshape((-1, 3))
            # Compute before writing the action row so malformed simulator
            # telemetry becomes a failed case, not a half-written pair.
            eef_jerk = _eef_jerk_rms(eef)
        except Exception as exc:
            eef_error = exc
    effective_error = error or eef_error
    row = _ORIGINAL_EPISODE_ROW(case, role, identity, episode if effective_error is None else None, output_dir, error=effective_error)
    row.update({
        "variant": ACTIVE_VARIANT,
        "variant_mode": VARIANTS[ACTIVE_VARIANT]["mode"],
        "old_action_weight": OLD_ACTION_WEIGHT if ACTIVE_VARIANT in {"seam_blend", "arm_ema01"} else 0.0,
        "smoothing_arm_dimensions": 6 if ACTIVE_VARIANT in {"seam_blend", "arm_ema01"} else 0,
        "smoothing_gripper": False,
    })
    if effective_error is not None or episode is None or eef is None:
        row.update({"eef_trace_path": None, "eef_trace_finite": False, "eef_jerk_rms": None})
        return row
    path = _eef_trace_path(output_dir, str(case["case_id"]))
    if path.exists():
        raise FileExistsError(f"refusing to overwrite EEF trace: {path}")
    base._atomic_npy(path, eef)
    row.update({
        "eef_trace_path": str(path.resolve()),
        "eef_trace_shape": list(eef.shape),
        "eef_trace_finite": bool(eef.size and np.isfinite(eef).all()),
        "eef_jerk_rms": eef_jerk,
        "eef_jerk_definition": "sqrt(mean(norm(diff(eef_positions,n=3)/0.05^3,axis=1)^2))",
    })
    return row


def _variant_preflight(**kwargs: Any) -> dict[str, Any]:
    report = _ORIGINAL_PREFLIGHT(**kwargs)
    report.update({
        "schema_version": "r6_t2_light_smoothing_preflight_v1",
        "variant": ACTIVE_VARIANT,
        "variant_mode": VARIANTS[ACTIVE_VARIANT]["mode"],
        "candidate_screen": ACTIVE_VARIANT != BASELINE_VARIANT,
        "old_action_weight": OLD_ACTION_WEIGHT if ACTIVE_VARIANT in {"seam_blend", "arm_ema01"} else 0.0,
        "smoothing_gripper": False,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    })
    return report


def _screen_process_conflicts(gpu: int) -> list[dict[str, Any]]:
    current = os.getpid()
    conflicts: list[dict[str, Any]] = []
    for pid, command in base._process_rows():
        if pid == current or "run_r6_t2_light_smoothing_screen" not in command:
            continue
        if f"--gpu {gpu}" in command:
            conflicts.append({"pid": pid, "args": command, "reason": "same_screen_gpu"})
    return conflicts


def preflight(variant: str, gpu: int) -> dict[str, Any]:
    configure(variant)
    conflicts = _screen_process_conflicts(gpu)
    if conflicts:
        raise RuntimeError(f"another T2 screen process owns GPU {gpu}: {conflicts}")
    base.preflight = _variant_preflight
    return base.preflight(
        role="r6",
        gpu=gpu,
        evidence_dir=EVIDENCE_ROOT / variant,
        output_dir=OUTPUT_ROOT / variant,
        adapter_source=base.adapter_source_identity(),
    )


def worker(variant: str, gpu: int, *, resume: bool = False) -> dict[str, Any]:
    configure(variant)
    conflicts = _screen_process_conflicts(gpu)
    if conflicts:
        raise RuntimeError(f"another T2 screen process owns GPU {gpu}: {conflicts}")
    base.preflight = _variant_preflight
    summary = base.run_worker(
        role="r6",
        gpu=gpu,
        resume=resume,
        evidence_dir=EVIDENCE_ROOT / variant,
        output_dir=OUTPUT_ROOT / variant,
        python=base.PYTHON,
    )
    summary = {
        **summary,
        "schema_version": "r6_t2_light_smoothing_worker_summary_v1",
        "variant": variant,
        "variant_mode": VARIANTS[variant]["mode"],
        "old_action_weight": OLD_ACTION_WEIGHT if variant in {"seam_blend", "arm_ema01"} else 0.0,
        "smoothing_gripper": False,
        "official_result": False,
        "auto_upload": False,
        "auto_submit": False,
    }
    base.atomic_json(EVIDENCE_ROOT / variant / "r6" / "summary.json", summary)
    return summary


def _load_variant(variant: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    configure(variant)
    root = EVIDENCE_ROOT / variant / "r6"
    summary = base.strict_json_load(root / "summary.json")
    if summary.get("status") != "completed" or summary.get("case_count") != 24:
        raise ValueError(f"{variant} is not a completed 24-case worker")
    if summary.get("variant") != variant:
        raise ValueError(f"{variant} summary has wrong variant identity")
    inputs = base.load_frozen_inputs(adapter_source=base.adapter_source_identity())
    identity = base.validate_checkpoint("r6")
    rows: dict[str, dict[str, Any]] = {}
    for case in inputs["all_cases"]:
        case_id = str(case["case_id"])
        path = base._case_file(root, case_id)
        row = base._load_completed(path, case, identity)
        if row is None:
            raise ValueError(f"missing case row for {variant}/{case_id}")
        if row.get("variant") != variant or row.get("smoothing_gripper") is not False:
            raise ValueError(f"variant identity mismatch for {variant}/{case_id}")
        eef_path = Path(str(row.get("eef_trace_path", "")))
        if not eef_path.is_file():
            raise FileNotFoundError(f"missing EEF trace for {variant}/{case_id}: {eef_path}")
        eef = np.load(eef_path, allow_pickle=False)
        if eef.ndim != 2 or eef.shape[1] != 3 or not np.isfinite(eef).all():
            raise ValueError(f"invalid EEF trace for {variant}/{case_id}: {eef_path}")
        jerk = _eef_jerk_rms(eef)
        if float(row.get("eef_jerk_rms")) != jerk:
            raise ValueError(f"EEF jerk metadata mismatch for {variant}/{case_id}")
        rows[case_id] = row
    if len(rows) != 24:
        raise ValueError(f"{variant} has {len(rows)} cases, expected 24")
    return summary, rows


def _ffprobe_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        return {"path": str(path), "readable": False, "reason": "missing_or_empty"}
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,nb_frames", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout) if result.stdout else {}
    except json.JSONDecodeError:
        payload = {}
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    readable = result.returncode == 0 and bool(streams) and streams[0].get("codec_name")
    return {"path": str(path), "readable": bool(readable), "returncode": result.returncode, "stream": streams[0] if streams else None, "stderr": result.stderr[-500:]}


def _panel(rows: dict[str, dict[str, Any]], panel: str) -> dict[str, Any]:
    selected = [row for row in rows.values() if row.get("panel") == panel]
    return {
        "n": len(selected),
        "success": sum(bool(row.get("success")) for row in selected),
        "collision": sum(bool(row.get("collision")) for row in selected),
        "timeout": sum(bool(row.get("timeout")) for row in selected),
        "steps_mean": float(np.mean([int(row.get("steps", 0)) for row in selected])) if selected else None,
    }


def _metrics(variant: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    videos = {case_id: _ffprobe_video(Path(str(row.get("video_path", "")))) for case_id, row in rows.items()}
    eef_jerks = [float(row["eef_jerk_rms"]) for row in rows.values()]
    steps = [int(row.get("steps", 0)) for row in rows.values()]
    successful_steps = [int(row.get("steps", 0)) for row in rows.values() if row.get("success")]
    return {
        "variant": variant,
        "case_count": len(rows),
        "success": sum(bool(row.get("success")) for row in rows.values()),
        "collision": sum(bool(row.get("collision")) for row in rows.values()),
        "timeout": sum(bool(row.get("timeout")) for row in rows.values()),
        "exception": sum(row.get("failure_class") in {"infrastructure", "model_error", "policy_nonfinite"} for row in rows.values()),
        "mean_eef_jerk_rms": float(np.mean(eef_jerks)),
        "mean_steps_all_cases": float(np.mean(steps)),
        "mean_steps_successes": float(np.mean(successful_steps)) if successful_steps else None,
        "reverse_dev": _panel(rows, "reverse_dev"),
        "c20": _panel(rows, "C20"),
        "video_ffprobe": {"count": len(videos), "passed": sum(bool(item["readable"]) for item in videos.values()), "cases": videos},
    }


def compare() -> dict[str, Any]:
    loaded = {variant: _load_variant(variant) for variant in VARIANT_ORDER}
    baseline_rows = loaded[BASELINE_VARIANT][1]
    baseline_ids = list(baseline_rows)
    baseline_fingerprints = {case_id: row.get("case_fingerprint") for case_id, row in baseline_rows.items()}
    baseline_metrics = _metrics(BASELINE_VARIANT, baseline_rows)
    variants: dict[str, Any] = {BASELINE_VARIANT: {"metrics": baseline_metrics, "passes_screen": False, "is_reference": True}}
    passing: list[str] = []
    for variant in CANDIDATE_VARIANTS:
        rows = loaded[variant][1]
        input_match = list(rows) == baseline_ids and {case_id: row.get("case_fingerprint") for case_id, row in rows.items()} == baseline_fingerprints
        metrics = _metrics(variant, rows)
        c20_wins = sum(not bool(baseline_rows[case_id].get("success")) and bool(rows[case_id].get("success")) for case_id in baseline_ids if rows[case_id].get("panel") == "C20")
        c20_losses = sum(bool(baseline_rows[case_id].get("success")) and not bool(rows[case_id].get("success")) for case_id in baseline_ids if rows[case_id].get("panel") == "C20")
        gates = {
            "exact_24_case_identity": input_match,
            "success_not_lower_all": metrics["success"] >= baseline_metrics["success"],
            "reverse_success_not_lower": metrics["reverse_dev"]["success"] >= baseline_metrics["reverse_dev"]["success"],
            "c20_success_not_lower": metrics["c20"]["success"] >= baseline_metrics["c20"]["success"],
            "collision_not_higher_all": metrics["collision"] <= baseline_metrics["collision"],
            "reverse_collision_not_higher": metrics["reverse_dev"]["collision"] <= baseline_metrics["reverse_dev"]["collision"],
            "c20_collision_not_higher": metrics["c20"]["collision"] <= baseline_metrics["c20"]["collision"],
            "no_exception": metrics["exception"] == 0,
            "all_eef_traces_finite": all(bool(row.get("eef_trace_finite")) for row in rows.values()),
            "all_videos_ffprobe_readable": metrics["video_ffprobe"]["passed"] == 24,
            "eef_jerk_strictly_lower": metrics["mean_eef_jerk_rms"] < baseline_metrics["mean_eef_jerk_rms"],
            "steps_strictly_lower": metrics["mean_steps_all_cases"] < baseline_metrics["mean_steps_all_cases"],
        }
        passed = all(gates.values())
        if passed:
            passing.append(variant)
        variants[variant] = {
            "metrics": metrics,
            "c20_delta": metrics["c20"]["success"] - baseline_metrics["c20"]["success"],
            "reverse_delta": metrics["reverse_dev"]["success"] - baseline_metrics["reverse_dev"]["success"],
            "c20_paired_wins": c20_wins,
            "c20_paired_losses": c20_losses,
            "c20_paired_net_wins": c20_wins - c20_losses,
            "gates": gates,
            "passes_screen": passed,
            "is_reference": False,
        }
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: (
                variants[name]["metrics"]["c20"]["success"],
                variants[name]["metrics"]["reverse_dev"]["success"],
                -variants[name]["metrics"]["collision"],
                -variants[name]["metrics"]["mean_eef_jerk_rms"],
                -variants[name]["metrics"]["mean_steps_all_cases"],
            ),
        )
    result = {
        "schema_version": "r6_t2_light_smoothing_screen_v1",
        "status": "candidate_passed_zip_authorized" if selected else "screen_failed_keep_existing_execute4",
        "adapter": str(ADAPTER.resolve()),
        "adapter_sha256": base.sha256_file(ADAPTER),
        "plan": str(PLAN.resolve()),
        "plan_sha256": base.sha256_file(PLAN),
        "baseline_variant": BASELINE_VARIANT,
        "variants": variants,
        "selected_variant": selected,
        "release_authorized": selected is not None,
        "training_performed": False,
        "fold_performed": False,
        "zip_generated": False,
        "auto_upload": False,
        "auto_submit": False,
        "completed_at_unix": time.time(),
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    base.atomic_json(EVIDENCE_ROOT / "comparison.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "worker"):
        item = sub.add_parser(command)
        item.add_argument("--variant", choices=VARIANT_ORDER, required=True)
        item.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
        if command == "worker":
            item.add_argument("--resume", action="store_true")
    sub.add_parser("compare")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.variant, args.gpu)
    elif args.command == "worker":
        result = worker(args.variant, args.gpu, resume=args.resume)
    else:
        result = compare()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") in {"passed", "completed", "candidate_passed_zip_authorized", "screen_failed_keep_existing_execute4"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
