"""Build and gate the R6 on-disk LeRobot folded frozen 8x4 grid.

The source fixtures are the already locked Q2 observations.  This tool only
materializes their deployment-facing image convention and compares outputs
from the real on-disk PyTorch/LeRobot path; it never changes the checkpoint or
relabels the raw-LoRA Q2 failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import ROOT, sha256_file, write_json


Q2_FIXTURES = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary/q2_fixtures"
DEFAULT_OUTPUT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_folded_submission/s1_lerobot_frozen_grid"
TOKENIZER_MODEL = ROOT / "outputs/axis_fast_recovery/f2_m1_500_fast/staging/vendor/paligemma_tokenizer/tokenizer.model"
EXPECTED_CASES = {
    "C20-task00-init0",
    "C20-task02-init0",
    "C20-task04-init0",
    "C20-task06-init0",
    "C20-task08-init0",
    "C20-task11-init0",
    "C20-task15-init0",
    "C20-task19-init0",
}
EXPECTED_SEED_INDICES = {0, 1, 2, 3}
REQUIRED_FIELDS = {
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "noise",
    "policy_seed",
    "noise_seed",
}


def _fingerprint(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": __import__("hashlib").sha256(array.tobytes()).hexdigest(),
    }


def _fixture_rows(fixtures_dir: Path) -> list[dict[str, Any]]:
    manifest_path = fixtures_dir / "fixture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("fixtures")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("S1 requires the locked Q2 8x4 fixture manifest (32 rows)")
    if manifest.get("case_count") != 8 or manifest.get("seeds_per_case") != 4:
        raise ValueError("Q2 fixture manifest is not 8 cases x 4 seeds")
    seen_cases: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    for row in rows:
        fixture_id = str(row.get("fixture_id", ""))
        if "-s" not in fixture_id:
            raise ValueError(f"invalid fixture id: {fixture_id}")
        case_id, seed_text = fixture_id.rsplit("-s", 1)
        seed_index = int(seed_text)
        if case_id not in EXPECTED_CASES or seed_index not in EXPECTED_SEED_INDICES:
            raise ValueError(f"fixture outside locked Q2 8x4 grid: {fixture_id}")
        seen_cases.add(case_id)
        seen_pairs.add((case_id, seed_index))
    if seen_cases != EXPECTED_CASES or len(seen_pairs) != 32:
        raise ValueError("Q2 fixture manifest does not cover every locked case/seed pair")
    return rows


def build_frozen_input(
    fixtures_dir: Path = Q2_FIXTURES,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Materialize a deployment-facing batch without modifying Q2 fixtures."""

    rows = _fixture_rows(fixtures_dir)
    if not TOKENIZER_MODEL.is_file():
        raise FileNotFoundError(TOKENIZER_MODEL)
    # The Q2 policy fixture intentionally used the JAX Pi05 discrete-state
    # tokenizer.  R6's persisted LeRobot config declares discrete_state_input
    # false, so the deployment fixture must use prompt + newline only.  The
    # SentencePiece model is the tokenizer implementation/vendor asset; the
    # R6 processors and normalization files remain the only deployment data.
    import sentencepiece

    sentencepiece_tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER_MODEL))

    def deployment_tokens(prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = sentencepiece_tokenizer.encode(cleaned, add_bos=True) + sentencepiece_tokenizer.encode("\n")
        if len(tokens) > 200:
            tokens = tokens[:200]
        mask = np.zeros((200,), dtype=np.bool_)
        mask[: len(tokens)] = True
        padded = np.zeros((200,), dtype=np.int32)
        padded[: len(tokens)] = np.asarray(tokens, dtype=np.int32)
        return padded, mask

    values: dict[str, list[np.ndarray]] = {
        "base_0_rgb": [],
        "left_wrist_0_rgb": [],
        "right_wrist_0_rgb": [],
        "state": [],
        "tokenized_prompt": [],
        "tokenized_prompt_mask": [],
        "noise": [],
    }
    tasks: list[str] = []
    fixture_ids: list[str] = []
    policy_seeds: list[int] = []
    noise_seeds: list[int] = []
    source_records: list[dict[str, Any]] = []
    for row in rows:
        fixture_path = Path(str(row["path"]))
        if not fixture_path.is_absolute():
            fixture_path = ROOT / fixture_path
        if not fixture_path.is_file():
            raise FileNotFoundError(fixture_path)
        loaded = np.load(fixture_path, allow_pickle=False)
        missing = sorted(REQUIRED_FIELDS.difference(loaded.files))
        if missing:
            raise ValueError(f"{fixture_path} missing fields: {missing}")
        deployment_prompt_tokens, deployment_prompt_mask = deployment_tokens(str(row["prompt"]))
        for key in values:
            value = np.asarray(loaded[key])
            if key.endswith("_rgb"):
                if value.shape != (224, 224, 3) or value.dtype != np.float32:
                    raise ValueError(f"unexpected Q2 image fixture {key}: {value.shape}/{value.dtype}")
                # Q2 stores transformed uint8-equivalent values as float32.
                # The direct PyTorch Pi0.5 core consumes the OpenPI [-1, 1]
                # convention, while the R6 visual processor is IDENTITY.
                value = (value / np.float32(127.5) - np.float32(1.0)).astype(np.float32)
                if not np.isfinite(value).all() or value.min() < -1.0 or value.max() > 1.0:
                    raise ValueError(f"invalid deployment image range for {key}: {value.min()}..{value.max()}")
            elif key == "state" and (value.shape != (32,) or value.dtype != np.float32):
                raise ValueError(f"unexpected state fixture: {value.shape}/{value.dtype}")
            elif key == "tokenized_prompt" and (value.shape != (200,) or value.dtype != np.int32):
                raise ValueError(f"unexpected token fixture: {value.shape}/{value.dtype}")
            elif key == "tokenized_prompt_mask" and (value.shape != (200,) or value.dtype != np.bool_):
                raise ValueError(f"unexpected token mask fixture: {value.shape}/{value.dtype}")
            elif key == "noise" and (value.shape != (10, 32) or value.dtype != np.float32):
                raise ValueError(f"unexpected noise fixture: {value.shape}/{value.dtype}")
            if key == "tokenized_prompt":
                value = deployment_prompt_tokens
            elif key == "tokenized_prompt_mask":
                value = deployment_prompt_mask
            values[key].append(value)
        fixture_ids.append(str(row["fixture_id"]))
        tasks.append(str(row["prompt"]))
        policy_seeds.append(int(row["policy_seed"]))
        noise_seeds.append(int(row["noise_seed"]))
        source_records.append(
            {
                "fixture_id": str(row["fixture_id"]),
                "source_path": str(fixture_path.resolve()),
                "source_sha256": sha256_file(fixture_path),
                "policy_seed": int(row["policy_seed"]),
                "noise_seed": int(row["noise_seed"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "frozen_input.npz"
    np.savez_compressed(
        input_path,
        **{key: np.stack(items, axis=0) for key, items in values.items()},
        policy_seed=np.asarray(policy_seeds, dtype=np.int64),
        noise_seed=np.asarray(noise_seeds, dtype=np.int64),
    )
    tasks_path = output_dir / "tasks.json"
    write_json(tasks_path, {"fixture_ids": fixture_ids, "tasks": tasks})
    result = {
        "schema_version": 1,
        "phase": "S1.2",
        "status": "frozen_input_built",
        "fixture_count": len(rows),
        "case_count": len(EXPECTED_CASES),
        "seeds_per_case": len(EXPECTED_SEED_INDICES),
        "source_fixture_manifest": str((fixtures_dir / "fixture_manifest.json").resolve()),
        "source_fixture_manifest_sha256": sha256_file(fixtures_dir / "fixture_manifest.json"),
        "source_fixtures": source_records,
        "tokenization": {
            "source_q2_contract": "discrete_state_input=true (retained read-only; not used as R6 deployment token input)",
            "deployment_contract": "discrete_state_input=false; cleaned prompt + newline",
            "tokenizer_model": str(TOKENIZER_MODEL.resolve()),
            "tokenizer_model_sha256": sha256_file(TOKENIZER_MODEL),
        },
        "deployment_image_convention": "float32 [-1, 1], R6 VISUAL=IDENTITY",
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "tasks": str(tasks_path.resolve()),
        "tasks_sha256": sha256_file(tasks_path),
        "arrays": {
            key: _fingerprint(np.stack(items, axis=0)) for key, items in values.items()
        },
    }
    write_json(output_dir / "fixture_summary.json", result)
    return result


def _diff_summary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    a = np.asarray(left)
    b = np.asarray(right)
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "shape_match": a.shape == b.shape,
        "dtype_match": a.dtype == b.dtype,
        "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
        "max_abs_diff": float(np.max(difference)),
        "p99_abs_diff": float(np.percentile(difference, 99)),
        "exact": bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)),
    }


def _sign_flip(left: np.ndarray, right: np.ndarray) -> int:
    a = np.sign(np.asarray(left)[..., -1])
    b = np.sign(np.asarray(right)[..., -1])
    return int(np.count_nonzero(a != b))


def compare_frozen_grid(first_dir: Path, second_dir: Path, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Apply the S1.2 gates to two fresh processes and their replay artifacts."""

    first_report_path = first_dir / "e2e_action_parity_report.json"
    second_report_path = second_dir / "e2e_action_parity_report.json"
    first_artifact_path = first_dir / "e2e_action_parity.npz"
    second_artifact_path = second_dir / "e2e_action_parity.npz"
    for path in (first_report_path, second_report_path, first_artifact_path, second_artifact_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    second_report = json.loads(second_report_path.read_text(encoding="utf-8"))
    first = np.load(first_artifact_path, allow_pickle=False)
    second = np.load(second_artifact_path, allow_pickle=False)
    failures: list[dict[str, Any]] = []

    def require(condition: bool, name: str, evidence: Any) -> None:
        if not condition:
            failures.append({"gate": name, "evidence": evidence})

    require(first_report.get("backend") == "lerobot_disk_processors_and_model_safetensors_with_openpi_pytorch_core", "actual_disk_lerobot_backend", first_report.get("backend"))
    require(second_report.get("backend") == first_report.get("backend"), "backend_identity_match", second_report.get("backend"))
    require(first_report.get("batch_size") == 32 and second_report.get("batch_size") == 32, "coverage_32", {"first": first_report.get("batch_size"), "second": second_report.get("batch_size")})
    require(first_report.get("normalization") == "QUANTILES" and second_report.get("normalization") == "QUANTILES", "quantiles_normalization", {"first": first_report.get("normalization"), "second": second_report.get("normalization")})
    require(first_report.get("processor_token_mask_mismatch_count") == 0 and second_report.get("processor_token_mask_mismatch_count") == 0, "token_mask_mismatch", {"first": first_report.get("processor_token_mask_mismatch_count"), "second": second_report.get("processor_token_mask_mismatch_count")})
    require(first_report.get("processor_token_max_abs_diff") == 0 and second_report.get("processor_token_max_abs_diff") == 0, "token_mismatch", {"first": first_report.get("processor_token_max_abs_diff"), "second": second_report.get("processor_token_max_abs_diff")})
    require(first_report.get("processor_state_max_abs_diff", float("inf")) <= 1e-6 and second_report.get("processor_state_max_abs_diff", float("inf")) <= 1e-6, "state_processor_diff", {"first": first_report.get("processor_state_max_abs_diff"), "second": second_report.get("processor_state_max_abs_diff")})
    require(first_report.get("status") == "passed" and second_report.get("status") == "passed", "per_process_report_status", {"first": first_report.get("status"), "second": second_report.get("status")})
    for name in ("folded_jax_actions", "folded_pytorch_actions", "lerobot_postprocessed_actions", "folded_jax_external_actions"):
        require(name in first and name in second, f"artifact_field_{name}", {"first": name in first, "second": name in second})
        if name in first and name in second:
            require(bool(np.isfinite(first[name]).all() and np.isfinite(second[name]).all()), f"finite_{name}", None)

    same_process = first_report.get("same_process_replay") or {}
    require(first_report.get("same_process_replay_required") is True, "same_process_replay_required", first_report.get("same_process_replay_required"))
    require(bool(first_report.get("same_process_replay_within_tolerance")), "same_process_replay_exact", same_process)
    if "folded_pytorch_actions_replay" in first:
        require(bool(_diff_summary(first["folded_pytorch_actions"], first["folded_pytorch_actions_replay"])["exact"]), "same_process_internal_artifact_exact", None)
    else:
        require(False, "same_process_internal_artifact_present", None)

    fresh_internal = _diff_summary(first["folded_pytorch_actions"], second["folded_pytorch_actions"])
    fresh_external = _diff_summary(first["lerobot_postprocessed_actions"], second["lerobot_postprocessed_actions"])
    jax_internal = _diff_summary(first["folded_jax_actions"], first["folded_pytorch_actions"])
    jax_external = _diff_summary(first["folded_jax_external_actions"], first["lerobot_postprocessed_actions"])
    require(fresh_external["shape_match"] and fresh_external["dtype_match"] and fresh_external["finite"] and fresh_external["max_abs_diff"] <= 1e-6, "fresh_process_external_max_abs_le_1e-6", fresh_external)
    require(jax_internal["max_abs_diff"] <= 1e-3 and jax_external["max_abs_diff"] <= 1e-3, "jax_folded_vs_lerobot_folded_le_1e-3", {"internal": jax_internal, "external": jax_external})
    require(_sign_flip(first["lerobot_postprocessed_actions"], second["lerobot_postprocessed_actions"]) == 0, "fresh_gripper_sign_flip", _sign_flip(first["lerobot_postprocessed_actions"], second["lerobot_postprocessed_actions"]))
    require(_sign_flip(first["folded_jax_external_actions"], first["lerobot_postprocessed_actions"]) == 0, "jax_lerobot_gripper_sign_flip", _sign_flip(first["folded_jax_external_actions"], first["lerobot_postprocessed_actions"]))

    status = "r6_lerobot_folded_frozen_grid_passed_real_egl_not_started" if not failures else "s1_frozen_grid_failed"
    result = {
        "schema_version": 1,
        "phase": "S1.2",
        "status": status,
        "coverage": {"expected": 32, "first": first_report.get("batch_size"), "second": second_report.get("batch_size")},
        "backend": first_report.get("backend"),
        "model_sha256": first_report.get("model_sha256"),
        "norm_stats_sha256": first_report.get("norm_stats_sha256"),
        "source_config_sha256": first_report.get("source_config_sha256"),
        "processor_hashes": first_report.get("processor_hashes"),
        "same_process": same_process,
        "fresh_process_internal": fresh_internal,
        "fresh_process_external": fresh_external,
        "jax_vs_lerobot_internal": jax_internal,
        "jax_vs_lerobot_external": jax_external,
        "gripper_sign_flips": {
            "fresh_process": _sign_flip(first["lerobot_postprocessed_actions"], second["lerobot_postprocessed_actions"]),
            "jax_vs_lerobot": _sign_flip(first["folded_jax_external_actions"], first["lerobot_postprocessed_actions"]),
        },
        "replicas": {"first": str(first_dir.resolve()), "second": str(second_dir.resolve())},
        "failures": failures,
        "first_failure": failures[0] if failures else None,
        "affected_artifact": str((first_dir if failures else output_dir).resolve()),
        "not_started_phases": ["S1.3", "S2", "S3", "S4", "S5", "S6", "S7"] if failures else ["S1.3"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)
    report_status = "通过" if not failures else "失败并停止"
    (output_dir / "S1_FROZEN_GRID_REPORT_ZH.md").write_text(
        "# S1.2 R6 folded LeRobot frozen grid\n\n"
        f"状态：`{status}`（{report_status}）\n\n"
        f"- coverage：{result['coverage']}\n"
        f"- backend：`{result['backend']}`\n"
        f"- fresh-process external max abs：{fresh_external['max_abs_diff']}（门禁 `<=1e-6`）\n"
        f"- same-process exact：{same_process}\n"
        f"- gripper sign flips：{result['gripper_sign_flips']}\n"
        f"- first_failure：{json.dumps(result['first_failure'], ensure_ascii=False)}\n"
        f"- not_started_phases：{', '.join(result['not_started_phases'])}\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--fixtures", type=Path, default=Q2_FIXTURES)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    compare = sub.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_frozen_input(args.fixtures, args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(compare_frozen_grid(args.first, args.second, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
