"""Lock and gate real Q1 EGL image pairs through the R6 folded deployment path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint_http_server import _resize_for_openpi
from .common import ROOT, sha256_file, write_json
from .r6_s1_frozen import DEFAULT_OUTPUT, TOKENIZER_MODEL, _diff_summary


Q1_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q1_environment_full_c20"
S1_ROOT = DEFAULT_OUTPUT
PAIR_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_folded_submission/s1_real_egl_pairs"
BASE_INPUT = S1_ROOT / "frozen_input.npz"
BASE_TASKS = S1_ROOT / "tasks.json"
NORM_STATS = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
C20_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
CAMERA_TO_INPUT = {
    "agentview_image": "base_0_rgb",
    "robot0_eye_in_hand_image": "left_wrist_0_rgb",
}
REPLICA_NAMES = ("replica0", "replica1", "replica2")


def _raw_descriptor(snapshot: dict[str, Any], camera: str) -> dict[str, Any] | None:
    value = snapshot.get("observation", {}).get(camera)
    return value if isinstance(value, dict) and value.get("path") else None


def _snapshot_maps() -> list[dict[tuple[str, str], dict[str, Any]]]:
    maps: list[dict[tuple[str, str], dict[str, Any]]] = []
    for replica in REPLICA_NAMES:
        path = Q1_ROOT / replica / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        result: dict[tuple[str, str], dict[str, Any]] = {}
        for case in summary["records"][0]["cases"]:
            case_id = str(case["case_id"])
            for snapshot in case["snapshots"]:
                stage = str(snapshot.get("stage"))
                if stage == "E3":
                    phase = f"E3-{int(snapshot['warmup_step'])}"
                elif stage in {"E1", "E2"}:
                    phase = stage
                else:
                    continue
                result[(case_id, phase)] = snapshot
        maps.append(result)
    return maps


def _transformed_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(np.load(path, allow_pickle=False))
    transformed_uint8 = _resize_for_openpi(raw)
    transformed_float = transformed_uint8.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)
    return raw, transformed_float


def _pair_metrics(left_path: Path, right_path: Path) -> dict[str, Any]:
    left_raw, left_transformed = _transformed_image(left_path)
    right_raw, right_transformed = _transformed_image(right_path)
    raw_diff = np.abs(left_raw.astype(np.int16) - right_raw.astype(np.int16))
    transformed_diff = np.abs(left_transformed.astype(np.float64) - right_transformed.astype(np.float64))
    return {
        "raw": {
            "max_abs_diff": float(np.max(raw_diff)),
            "changed_element_count": int(np.count_nonzero(raw_diff)),
            "changed_pixel_count": int(np.count_nonzero(np.any(raw_diff != 0, axis=-1))),
            "mean_abs_diff": float(np.mean(raw_diff)),
            "p99_abs_diff": float(np.percentile(raw_diff, 99)),
        },
        "transformed_float32_minus1_plus1": {
            "max_abs_diff": float(np.max(transformed_diff)),
            "changed_element_count": int(np.count_nonzero(transformed_diff != 0)),
            "changed_pixel_count": int(np.count_nonzero(np.any(transformed_diff != 0, axis=-1))),
            "mean_abs_diff": float(np.mean(transformed_diff)),
            "p99_abs_diff": float(np.percentile(transformed_diff, 99)),
        },
    }


def _candidate_pairs(snapshot_maps: list[dict[tuple[str, str], dict[str, Any]]], camera: str) -> list[dict[str, Any]]:
    common = set(snapshot_maps[0]) & set(snapshot_maps[1]) & set(snapshot_maps[2])
    candidates: list[dict[str, Any]] = []
    for case_id, phase in sorted(common):
        for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
            left_descriptor = _raw_descriptor(snapshot_maps[left_index][(case_id, phase)], camera)
            right_descriptor = _raw_descriptor(snapshot_maps[right_index][(case_id, phase)], camera)
            if left_descriptor is None or right_descriptor is None:
                continue
            left_path = Path(str(left_descriptor["path"]))
            right_path = Path(str(right_descriptor["path"]))
            if not left_path.is_file() or not right_path.is_file():
                raise FileNotFoundError(f"Q1 pair image missing: {left_path} / {right_path}")
            metrics = _pair_metrics(left_path, right_path)
            candidates.append(
                {
                    "case_id": case_id,
                    "phase": phase,
                    "camera": camera,
                    "left_replica": REPLICA_NAMES[left_index],
                    "right_replica": REPLICA_NAMES[right_index],
                    "left_path": str(left_path.resolve()),
                    "right_path": str(right_path.resolve()),
                    "left_sha256": sha256_file(left_path),
                    "right_sha256": sha256_file(right_path),
                    "metrics": metrics,
                }
            )
    if not candidates:
        raise ValueError(f"no Q1 candidates for camera {camera}")
    return candidates


def _selection_key(item: dict[str, Any]) -> tuple[float, int, float, str, str, str]:
    metric = item["metrics"]["transformed_float32_minus1_plus1"]
    return (
        float(metric["max_abs_diff"]),
        int(metric["changed_element_count"]),
        float(metric["mean_abs_diff"]),
        str(item["case_id"]),
        str(item["phase"]),
        str(item["left_replica"]),
    )


def _load_base() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not BASE_INPUT.is_file() or not BASE_TASKS.is_file():
        raise FileNotFoundError("S1.2 frozen input/tasks are required before S1.3")
    arrays_file = np.load(BASE_INPUT, allow_pickle=False)
    arrays = {key: np.asarray(arrays_file[key]) for key in arrays_file.files}
    tasks = json.loads(BASE_TASKS.read_text(encoding="utf-8"))
    if len(tasks.get("fixture_ids", [])) != 32 or len(tasks.get("tasks", [])) != 32:
        raise ValueError("S1.2 task fixture is not 32 samples")
    return arrays, tasks


def _deployment_tokens(prompt: str, tokenizer: Any) -> tuple[np.ndarray, np.ndarray]:
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = tokenizer.encode(cleaned, add_bos=True) + tokenizer.encode("\n")
    tokens = tokens[:200]
    padded = np.zeros((200,), dtype=np.int32)
    padded[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
    mask = np.zeros((200,), dtype=np.bool_)
    mask[: len(tokens)] = True
    return padded, mask


def _manifest_prompts() -> dict[str, str]:
    return {
        str(row["case_id"]): str(row["language"])
        for row in (json.loads(line) for line in C20_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _q1_base_fixture(case_id: str, snapshot_maps: list[dict[tuple[str, str], dict[str, Any]]], tokenizer: Any) -> tuple[dict[str, np.ndarray], str]:
    snapshot = snapshot_maps[0].get((case_id, "E3-10"))
    if snapshot is None:
        raise ValueError(f"Q1 has no E3-10 base snapshot for {case_id}")
    prompt = _manifest_prompts()[case_id]
    with NORM_STATS.open(encoding="utf-8") as handle:
        norm = json.load(handle)["norm_stats"]["state"]
    state_raw = np.asarray(np.load(snapshot["policy_state"]["path"], allow_pickle=False), dtype=np.float32)
    state_norm = ((state_raw - np.asarray(norm["q01"], dtype=np.float32)) / (np.asarray(norm["q99"], dtype=np.float32) - np.asarray(norm["q01"], dtype=np.float32) + np.float32(1e-6)) * np.float32(2.0) - np.float32(1.0))
    tokenized, tokenized_mask = _deployment_tokens(prompt, tokenizer)
    base_image = _normalize_raw(Path(str(snapshot["observation"]["agentview_image"]["path"])))
    wrist_image = _normalize_raw(Path(str(snapshot["observation"]["robot0_eye_in_hand_image"]["path"])))
    return {
        "base_0_rgb": base_image,
        "left_wrist_0_rgb": wrist_image,
        "right_wrist_0_rgb": np.full((224, 224, 3), -1.0, dtype=np.float32),
        "state": np.pad(state_norm, (0, 24)).astype(np.float32),
        "tokenized_prompt": tokenized,
        "tokenized_prompt_mask": tokenized_mask,
        "noise": np.random.default_rng(63000).standard_normal((10, 32), dtype=np.float32),
    }, f"{case_id}::Q1-replica0::E3-10"


def _normalize_raw(path: Path) -> np.ndarray:
    raw = np.asarray(np.load(path, allow_pickle=False))
    transformed = _resize_for_openpi(raw)
    return transformed.astype(np.float32) / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)


def build_egl_pairs(output_dir: Path = PAIR_ROOT) -> dict[str, Any]:
    arrays, base_tasks = _load_base()
    snapshot_maps = _snapshot_maps()
    tokenizer = __import__("sentencepiece").SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))
    base_index_by_case = {
        str(fixture_id).rsplit("-s", 1)[0]: index
        for index, fixture_id in enumerate(base_tasks["fixture_ids"])
        if str(fixture_id).endswith("-s0")
    }
    selections: list[dict[str, Any]] = []
    for camera in CAMERAS:
        candidates = _candidate_pairs(snapshot_maps, camera)
        worst = max(candidates, key=_selection_key)
        nonzero = [item for item in candidates if item["metrics"]["transformed_float32_minus1_plus1"]["max_abs_diff"] > 0]
        if not nonzero:
            raise ValueError(f"Q1 has no nonzero transformed pair for {camera}")
        representative = next((item for item in sorted(nonzero, key=lambda value: (value["case_id"], value["phase"], value["left_replica"])) if item != worst), nonzero[0])
        for selection_name, item in (("worst_transformed", worst), ("representative_nonzero", representative)):
            selected = dict(item)
            selected["selection"] = selection_name
            selected["selection_rule"] = "worst=max(transformed max, changed elements, mean; stable case/phase/replica tie-break)" if selection_name == "worst_transformed" else "first stable nonzero transformed pair distinct from worst when available"
            selections.append(selected)

    input_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "state", "tokenized_prompt", "tokenized_prompt_mask", "noise")}
    tasks: list[str] = []
    pair_ids: list[str] = []
    variants: list[str] = []
    pair_rows: list[dict[str, Any]] = []
    for pair_index, selected in enumerate(selections):
        case_id = str(selected["case_id"])
        if case_id in base_index_by_case:
            base_index = base_index_by_case[case_id]
            base_values = {key: arrays[key][base_index] for key in input_arrays}
            base_fixture_id = str(base_tasks["fixture_ids"][base_index])
            base_task = str(base_tasks["tasks"][base_index])
        else:
            base_values, base_fixture_id = _q1_base_fixture(case_id, snapshot_maps, tokenizer)
            base_task = _manifest_prompts()[case_id]
        camera_key = CAMERA_TO_INPUT[str(selected["camera"])]
        pair_id = f"pair{pair_index:02d}_{selected['camera']}_{selected['selection']}"
        left_image = _normalize_raw(Path(str(selected["left_path"])))
        right_image = _normalize_raw(Path(str(selected["right_path"])))
        pair_values: dict[str, np.ndarray] = {}
        for key in input_arrays:
            if key == camera_key:
                pair_values[key] = np.stack((left_image, right_image), axis=0)
            else:
                pair_values[key] = np.repeat(np.asarray(base_values[key])[None], 2, axis=0)
            input_arrays[key].extend(list(pair_values[key]))
        tasks.extend([base_task] * 2)
        pair_ids.extend([pair_id, pair_id])
        variants.extend(["left", "right"])
        pair_rows.append(
            {
                **selected,
                "pair_id": pair_id,
                "input_indices": [2 * pair_index, 2 * pair_index + 1],
                "base_fixture_id": base_fixture_id,
                "held_fixed_camera": next(camera for camera in CAMERAS if camera != selected["camera"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "egl_input.npz"
    np.savez_compressed(input_path, **{key: np.stack(value, axis=0) for key, value in input_arrays.items()})
    tasks_path = output_dir / "tasks.json"
    write_json(tasks_path, {"pair_ids": pair_ids, "variants": variants, "tasks": tasks})
    q1_paths = [Q1_ROOT / replica / "summary.json" for replica in REPLICA_NAMES]
    manifest = {
        "schema_version": 1,
        "phase": "S1.3",
        "status": "egl_pair_input_built",
        "selection_rule": "per camera: transformed maximum pair plus one representative nonzero pair; p99 is not used because Q1 reports sparse differences",
        "camera_count": len(CAMERAS),
        "pair_count": len(pair_rows),
        "sample_count": len(pair_ids),
        "source_q1_summary": str((Q1_ROOT / "q1_environment_full_c20_summary.json").resolve()),
        "source_q1_summary_sha256": sha256_file(Q1_ROOT / "q1_environment_full_c20_summary.json"),
        "source_q1_replica_summary_sha256": {replica: sha256_file(Q1_ROOT / replica / "summary.json") for replica in REPLICA_NAMES},
        "source_s1_frozen_input": str(BASE_INPUT.resolve()),
        "source_s1_frozen_input_sha256": sha256_file(BASE_INPUT),
        "source_q1_replicas": {replica: str(path.resolve()) for replica, path in zip(REPLICA_NAMES, q1_paths)},
        "pairs": pair_rows,
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "tasks": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "deployment_image_convention": "float32 [-1, 1], _resize_for_openpi uint8 / 255 * 2 - 1",
    }
    write_json(output_dir / "pair_manifest.json", manifest)
    write_json(output_dir / "summary.json", manifest)
    return manifest


def compare_egl_pairs(run_dir: Path, manifest_path: Path, output_dir: Path = PAIR_ROOT) -> dict[str, Any]:
    report_path = run_dir / "e2e_action_parity_report.json"
    artifact_path = run_dir / "e2e_action_parity.npz"
    if not report_path.is_file() or not artifact_path.is_file():
        raise FileNotFoundError("S1.3 E2E run report/artifact is incomplete")
    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = np.load(artifact_path, allow_pickle=False)
    failures: list[dict[str, Any]] = []

    def require(condition: bool, gate: str, evidence: Any) -> None:
        if not condition:
            failures.append({"gate": gate, "evidence": evidence})

    require(run_report.get("backend") == "lerobot_disk_processors_and_model_safetensors_with_openpi_pytorch_core", "actual_disk_lerobot_backend", run_report.get("backend"))
    require(run_report.get("batch_size") == 8, "coverage_4_pairs_x_2", run_report.get("batch_size"))
    require(run_report.get("normalization") == "QUANTILES", "quantiles_normalization", run_report.get("normalization"))
    require(run_report.get("processor_token_mask_mismatch_count") == 0, "token_mask_mismatch", run_report.get("processor_token_mask_mismatch_count"))
    require(run_report.get("processor_token_max_abs_diff") == 0, "token_mismatch", run_report.get("processor_token_max_abs_diff"))
    require(run_report.get("processor_state_max_abs_diff", float("inf")) <= 1e-6, "state_processor_diff", run_report.get("processor_state_max_abs_diff"))
    for name in ("folded_jax_external_actions", "lerobot_postprocessed_actions", "folded_jax_actions", "folded_pytorch_actions"):
        require(name in artifact and bool(np.isfinite(artifact[name]).all()), f"finite_{name}", name in artifact)

    pair_results: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        left, right = map(int, pair["input_indices"])
        pt = artifact["lerobot_postprocessed_actions"]
        jax = artifact["folded_jax_external_actions"]
        pt_full = _diff_summary(pt[left], pt[right])
        pt_execute8 = _diff_summary(pt[left, :8], pt[right, :8])
        jax_full = _diff_summary(jax[left], jax[right])
        jax_execute8 = _diff_summary(jax[left, :8], jax[right, :8])
        pt_flip = int(np.count_nonzero(np.sign(pt[left, ..., -1]) != np.sign(pt[right, ..., -1])))
        jax_flip = int(np.count_nonzero(np.sign(jax[left, ..., -1]) != np.sign(jax[right, ..., -1])))
        result = {
            "pair_id": pair["pair_id"],
            "camera": pair["camera"],
            "selection": pair["selection"],
            "case_id": pair["case_id"],
            "phase": pair["phase"],
            "raw_and_transformed_input_metrics": pair["metrics"],
            "deployment_pytorch_external_full_chunk": pt_full,
            "deployment_pytorch_external_execute8": pt_execute8,
            "axis_reference_jax_external_full_chunk": jax_full,
            "axis_reference_jax_external_execute8": jax_execute8,
            "deployment_pytorch_gripper_sign_flip": pt_flip,
            "axis_reference_jax_gripper_sign_flip": jax_flip,
        }
        require(pt_full["finite"] and pt_full["max_abs_diff"] <= 1e-6, f"{pair['pair_id']}_pytorch_full_chunk_le_1e-6", pt_full)
        require(pt_execute8["finite"] and pt_execute8["max_abs_diff"] <= 1e-6, f"{pair['pair_id']}_pytorch_execute8_le_1e-6", pt_execute8)
        require(pt_flip == 0, f"{pair['pair_id']}_pytorch_gripper_sign_flip_zero", pt_flip)
        require(jax_full["finite"] and jax_execute8["finite"] and jax_flip == 0, f"{pair['pair_id']}_jax_reference_finite_no_flip", {"full": jax_full, "execute8": jax_execute8, "gripper_flip": jax_flip})
        pair_results.append(result)

    status = "r6_lerobot_folded_policy_sensitivity_locked" if not failures else "s1_real_egl_pairs_failed"
    result = {
        "schema_version": 1,
        "phase": "S1.3",
        "status": status,
        "deployment_backend": run_report.get("backend"),
        "model_sha256": run_report.get("model_sha256"),
        "norm_stats_sha256": run_report.get("norm_stats_sha256"),
        "run_report": str(report_path.resolve()),
        "run_artifact": str(artifact_path.resolve()),
        "pair_manifest": str(manifest_path.resolve()),
        "pairs": pair_results,
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "affected_artifact": str((run_dir if failures else output_dir).resolve()),
        "not_started_phases": ["S2", "S3", "S4", "S5", "S6", "S7"] if failures else [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)
    (output_dir / "S1_REAL_EGL_REPORT_ZH.md").write_text(
        "# S1.3 R6 folded real EGL pair sensitivity\n\n"
        f"状态：`{status}`\n\n"
        f"- pairs：{len(pair_results)}（每 camera 最差 transformed pair + representative nonzero pair）\n"
        f"- backend：`{result['deployment_backend']}`\n"
        f"- first_failure：{json.dumps(result['first_failure'], ensure_ascii=False)}\n"
        f"- not_started_phases：{', '.join(result['not_started_phases']) or 'none'}\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--output", type=Path, default=PAIR_ROOT)
    compare = sub.add_parser("compare")
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--manifest", type=Path, default=PAIR_ROOT / "pair_manifest.json")
    compare.add_argument("--output", type=Path, default=PAIR_ROOT)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_egl_pairs(args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compare_egl_pairs(args.run_dir, args.manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
