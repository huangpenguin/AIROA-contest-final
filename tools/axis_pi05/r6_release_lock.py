"""Create and validate the fail-closed R6 folded deployment release lock."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .common import ROOT, sha256_file, sha256_tree, write_json


FOLDED_SOURCE = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500"
RAW_CHECKPOINT = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
NORM_STATS = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
C20_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/rollout/C20.jsonl"
REPORT_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/n1_policy_boundary"
OUTPUT_ROOT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_folded_submission"
EXPECTED_MODEL_SHA256 = "5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b"
EXPECTED_NORM_SHA256 = "4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5"
RAW_Q2_MAX_ABS_DIFF = 1.0728836059570312e-6
RAW_Q2_STATUS = "n1_q2_policy_sensitivity_failed_raw_lora_fresh_replay"

REQUIRED_SOURCE_FILES = (
    "model.safetensors",
    "config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_2_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
CODE_FILES = (
    "tools/axis_pi05/r6_release_lock.py",
    "tools/axis_pi05/n1_q2_policy.py",
    "tools/axis_pi05/lora_folding.py",
    "tools/axis_pi05/prepare_lerobot_assets.py",
    "tools/axis_pi05/export_fast_submission_weights.py",
    "tools/axis_pi05/checkpoint_http_server.py",
    "pipeline/environment.py",
)
FORBIDDEN_ASSETS = (
    "outputs/axis_fast_recovery/f2_m1_500_fast/staging/model_weights/model_fast.safetensors",
    "artifacts/submission_axis_f2_m1_500_fast_*.zip",
    "legacy F2/M1 processor or norm",
)


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _normalization_modes(value: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key in ("norm_map", "normalization_mapping"):
            child = value.get(key)
            if isinstance(child, dict):
                for feature_type in ("ACTION", "STATE"):
                    if feature_type in child:
                        found.append({feature_type: str(child[feature_type])})
        for child in value.values():
            found.extend(_normalization_modes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_normalization_modes(child))
    return found


def _source_file_records() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SOURCE_FILES:
        path = FOLDED_SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def validate_lock_record(record: dict[str, Any]) -> list[str]:
    """Validate release-lock semantics without reading heavyweight artifacts."""
    errors: list[str] = []
    if record.get("research_raw_lora_q2_status") != "failed_preserved":
        errors.append("raw-LoRA Q2 failure is not preserved")
    if record.get("research_raw_lora_q2", {}).get("status") != RAW_Q2_STATUS:
        errors.append("raw-LoRA Q2 status detail is missing or overwritten")
    if float(record.get("research_raw_lora_max_abs_diff", 0.0)) != RAW_Q2_MAX_ABS_DIFF:
        errors.append("raw-LoRA Q2 max diff does not match the frozen failure")
    if record.get("deployment_candidate") != "r6_step500_lerobot_folded":
        errors.append("deployment candidate is not R6 LeRobot folded")
    if record.get("deployment_runtime_status") != "not_yet_tested":
        errors.append("deployment runtime was marked tested before S1")
    if record.get("formal_winner") is not False:
        errors.append("formal_winner must be false")
    if record.get("auto_submit_authorized") is not False:
        errors.append("auto_submit_authorized must be false")
    deployment = record.get("deployment_source", {})
    if deployment.get("backend") != "lerobot_folded_pytorch":
        errors.append("deployment source backend is not lerobot_folded_pytorch")
    if not deployment.get("model_path") or not deployment.get("model_sha256"):
        errors.append("deployment source lacks actual model path/hash")
    if record.get("folded_q2_reference", {}).get("backend") != "folded_dense_jax":
        errors.append("JAX folded reference does not have its own backend kind")
    if "model_fast.safetensors" in str(deployment.get("model_path", "")):
        errors.append("release lock points deployment source at model_fast")
    deployment_paths = (deployment.get("model_path", ""), deployment.get("model_dir", ""))
    if any("f2_m1" in str(value).lower() or "/old/" in str(value).lower() for value in deployment_paths):
        errors.append("deployment source references a forbidden legacy asset")
    if record.get("normalization", {}).get("ACTION") != "QUANTILES" or record.get("normalization", {}).get("STATE") != "QUANTILES":
        errors.append("ACTION/STATE normalization is not QUANTILES")
    return errors


def build_lock(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if not FOLDED_SOURCE.is_dir():
        raise FileNotFoundError(FOLDED_SOURCE)
    if not RAW_CHECKPOINT.is_dir():
        raise FileNotFoundError(RAW_CHECKPOINT)
    source_files = _source_file_records()
    model_sha = source_files["model.safetensors"]["sha256"]
    norm_sha = sha256_file(NORM_STATS)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise ValueError(f"R6 folded model hash mismatch: {model_sha}")
    if norm_sha != EXPECTED_NORM_SHA256:
        raise ValueError(f"R6 norm hash mismatch: {norm_sha}")
    config = json.loads((FOLDED_SOURCE / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "pi05" or config.get("chunk_size") != 10 or config.get("n_action_steps") != 8:
        raise ValueError("R6 folded config does not declare pi05/chunk=10/n_action_steps=8")
    processor_values = [
        json.loads((FOLDED_SOURCE / name).read_text(encoding="utf-8"))
        for name in ("policy_preprocessor.json", "policy_postprocessor.json")
    ]
    modes = _normalization_modes(config) + [mode for value in processor_values for mode in _normalization_modes(value)]
    if not modes or any(mode.get(feature) != "QUANTILES" for mode in modes for feature in mode):
        raise ValueError("R6 folded config/processors are not fully QUANTILES")
    if any(path.name == "model_fast.safetensors" for path in FOLDED_SOURCE.rglob("*")):
        raise ValueError("folded source unexpectedly contains model_fast.safetensors")

    nested_root = ROOT / "third_party/axis_openpi"
    q2_report = REPORT_ROOT / "Q2_POLICY_SENSITIVITY_REPORT_ZH.md"
    report_inputs = {
        path.name: {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in (REPORT_ROOT / "Q0_AUDIT_REPORT_ZH.md", REPORT_ROOT / "Q1_ENVIRONMENT_REPORT_ZH.md", q2_report)
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "phase": "S0",
        "status": "r6_folded_release_locked_not_yet_tested",
        "root": str(ROOT.resolve()),
        "root_git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty": bool(_git("status", "--porcelain")),
        },
        "nested_openpi_git": {
            "commit": _git("rev-parse", "HEAD", cwd=nested_root),
            "dirty": bool(_git("status", "--porcelain", cwd=nested_root)),
        },
        "research_raw_lora_q2_status": "failed_preserved",
        "research_raw_lora_max_abs_diff": RAW_Q2_MAX_ABS_DIFF,
        "research_raw_lora_q2": {
            "status": RAW_Q2_STATUS,
            "report": str(q2_report.resolve()),
            "failure_location": {
                "fixture": "C20-task04-init0-s0",
                "horizon_index": 8,
                "action_dim_zero_based": 6,
                "semantic": "gripper",
            },
        },
        "deployment_candidate": "r6_step500_lerobot_folded",
        "deployment_runtime_status": "not_yet_tested",
        "formal_winner": False,
        "auto_submit_authorized": False,
        "folded_q2_reference": {"backend": "folded_dense_jax", "model_path": "Q2 in-memory JAX fold; not a deployment proof"},
        "deployment_source": {
            "backend": "lerobot_folded_pytorch",
            "model_dir": str(FOLDED_SOURCE.resolve()),
            "model_path": str((FOLDED_SOURCE / "model.safetensors").resolve()),
            "model_sha256": model_sha,
            "files": source_files,
        },
        "raw_checkpoint": {"path": str(RAW_CHECKPOINT.resolve()), "tree_sha256": sha256_tree(RAW_CHECKPOINT)},
        "norm_stats": {"path": str(NORM_STATS.resolve()), "sha256": norm_sha},
        "c20_manifest": {"path": str(C20_MANIFEST.resolve()), "sha256": sha256_file(C20_MANIFEST)},
        "normalization": {"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"},
        "contract": {"chunk_size": 10, "n_action_steps": 8, "external_action_dim": 7, "internal_action_dim": 32, "ema": 0.0, "rtc": False},
        "reports": report_inputs,
        "code_sha256": {path: sha256_file(ROOT / path) for path in CODE_FILES},
        "forbidden_legacy_assets": list(FORBIDDEN_ASSETS),
    }
    errors = validate_lock_record(record)
    if errors or record["root_git"]["dirty"] or record["nested_openpi_git"]["dirty"]:
        dirty = ["root dirty"] if record["root_git"]["dirty"] else []
        dirty += ["nested worktree dirty"] if record["nested_openpi_git"]["dirty"] else []
        raise ValueError("release lock validation failed: " + "; ".join(errors + dirty))
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "R6_FOLDED_RELEASE_LOCK.json", record)
    report = f"""# S0 R6 folded release lock audit

状态：`{record['status']}`

- deployment candidate：`r6_step500_lerobot_folded`
- deployment backend：`lerobot_folded_pytorch`
- source model：`{model_sha}`
- R6 fresh norm：`{norm_sha}`，ACTION/STATE=`QUANTILES`
- raw-LoRA research Q2：`{RAW_Q2_STATUS}`，失败最大差 `{RAW_Q2_MAX_ABS_DIFF}`，保留且不可覆盖
- deployment runtime：尚未测试；没有生成 `model_fast.safetensors`
- formal winner：`false`
- auto submit：`false`

S0 只锁定输入与 provenance，不宣称 LeRobot/PyTorch、FAST、server 或 ZIP 通过。旧 F2/M1 FAST、processor、norm 被列入拒绝名单，未作为 R6 输入。S1 将实际从磁盘 source model/processors 加载并验证。

机器锁定：[R6_FOLDED_RELEASE_LOCK.json]({(output_root / 'R6_FOLDED_RELEASE_LOCK.json').resolve()})
"""
    (output_root / "S0_AUDIT_REPORT_ZH.md").write_text(report, encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_lock(args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
