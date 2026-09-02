"""CPU-side locks and public-task utilities for the Track 3 reverse study.

The module deliberately has no simulator, JAX, Torch, or model imports at
module import time.  T0 can therefore be audited without allocating a CUDA
context.  Public evaluation material is handled as a denylist only; no
evaluation task is exposed as training data by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .common import ROOT, canonical_json, load_yaml, sha256_file, sha256_tree, write_json


TRACK3_ROOT = ROOT / "data/reports/axis_libero_30k/track3_reverse_lora"
T0_LOCK = TRACK3_ROOT / "t0_cpu_lock.json"
T0_REPORT = TRACK3_ROOT / "T0_CPU_LOCK_REPORT_ZH.md"
PLAN = ROOT / "docs/AXIS_PI05_TRACK3_REVERSE_LORA_POSITIVE_SCORE_PLAN.md"
HANDOFF = ROOT / "docs/AXIS_PI05_PROJECT_HANDOFF_20260831.md"
INDEX = ROOT / "docs/AXIS_PI05_PLAN_INDEX.md"

RAW_CHECKPOINT = ROOT / "outputs/axis_pi05/axis_libero_lora_screen/L2_action_lora_heads/r6_dual_l2_fresh500_r1/500"
FOLDED_MODEL = ROOT / "outputs/axis_pi05/axis_libero_lora_folded/r6_dual_l2_fresh500_r1_step500/model.safetensors"
FAST_MODEL = ROOT / "outputs/axis_pi05/r6_positive_score_fast/staging/model_weights/model_fast.safetensors"
POSITIVE_ZIP = ROOT / "artifacts/submission_axis_r6_step500_fast_positive_score_20260831.zip"
R6_RELEASE_LOCK = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/R6_FOLDED_RELEASE_LOCK.json"
FAST_AUDIT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/fast_submission_audit.json"
FAST_EXPORT = ROOT / "data/reports/axis_libero_30k/post_r6_lora/r6_positive_score_fast/export_report.json"

NORM_STATS = ROOT / "data/reports/axis_libero_30k/phase1/norm_stats.json"
NORM_MANIFEST = ROOT / "data/reports/axis_libero_30k/phase1/norm_manifest.json"
CANONICAL_PAYLOAD = ROOT / "data/reports/axis_libero_30k/canonical_payload_manifest.json"
CANONICAL_SAMPLING = ROOT / "data/reports/axis_libero_30k/phase1/official_sampling_manifest.jsonl"
CANONICAL_TASKS = ROOT / "data/raw/hf/physical-intelligence/libero/meta/tasks.jsonl"
PLUS_INFO = ROOT / "data/raw/hf/lerobot/libero_plus/meta/info.json"
PLUS_TASKS = ROOT / "data/raw/hf/lerobot/libero_plus/meta/tasks.parquet"
SOURCE_LOCK = ROOT / "configs/axis_pi05/data_sources.lock.yaml"
STANDARD_CONFIG = ROOT / "configs/axis_pi05/standard_libero_30k.yaml"
PLUS_CONFIG = ROOT / "configs/axis_pi05/libero_plus_track3_20hz.yaml"
PUBLIC_T3_HOLDOUT = ROOT / "data/eval_manifests/axis_pi05_eval_v1/public_t3_holdout.jsonl"

EXPECTED_RAW_TREE = "d203635f1eb0b006ec27841db6ce5171f74a37035eca4291c8107c05838c276f"
EXPECTED_FOLDED_SHA = "5c73e1e8217dbb75a934c2b0165208e851f69e66cfe2ea0430590a29901e6d5b"
EXPECTED_FAST_SHA = "e55bf7dcca78b2195993dd7a12b96a4ea13e94761d711c99a82b7411c3f8ceb6"
EXPECTED_ZIP_SHA = "95415a3131588665d1ac742d036b07ed4900466c7b73574e600ef6b36c5c5f5a"
EXPECTED_NORM_SHA = "4c803bf59c52677c36b6b420c614d8a7742e0847708443863e13e67cb80b18f5"
EXPECTED_PLUS_REVISION = "f3f49f426d75030177b18778374005bc12ccd588"
EXPECTED_CANONICAL_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"

PUBLIC_OPERATOR_FAMILIES = (
    "containment_remove_restore",
    "support_remove_restore",
    "articulation_inverse",
    "activation_inverse",
    "sequence_reverse",
)


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def git_lock() -> dict[str, Any]:
    nested = ROOT / "third_party/axis_openpi"
    return {
        "root_commit": _git("rev-parse", "HEAD"),
        "root_tree": _git("rev-parse", "HEAD^{tree}"),
        "root_status": _git("status", "--porcelain=v1").splitlines(),
        "nested_commit": _git("rev-parse", "HEAD", cwd=nested),
        "nested_tree": _git("rev-parse", "HEAD^{tree}", cwd=nested),
        "nested_status": _git("status", "--porcelain=v1", cwd=nested).splitlines(),
    }


def gpu_read_only_snapshot() -> list[dict[str, Any]]:
    """Query GPU state without opening a CUDA/JAX/Torch context."""

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    snapshot: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        index, uuid, used, total, util = [part.strip() for part in line.split(",")]
        snapshot.append({
            "index": int(index),
            "uuid": uuid,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_gpu_percent": int(util),
        })
    return snapshot


def require_track3_targets_free(snapshot: Iterable[dict[str, Any]], targets: tuple[int, ...] = (1, 2, 3)) -> None:
    values = {int(row["index"]): row for row in snapshot}
    if 0 in targets:
        raise ValueError("GPU0 is permanently outside the Track 3 resource boundary")
    missing = [gpu for gpu in targets if gpu not in values]
    if missing:
        raise RuntimeError(f"requested GPU is not visible: {missing}")
    busy = [values[gpu] for gpu in targets if int(values[gpu]["memory_used_mib"]) > 64 or int(values[gpu]["utilization_gpu_percent"]) > 5]
    if busy:
        raise RuntimeError(f"WAITING_FOR_GPU_TARGETS: requested GPUs are not idle: {busy}")


def _asset_record(path: Path, *, expected_sha256: str, independently_hashed: bool = False) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    is_tree = path.is_dir()
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "kind": "tree" if is_tree else "file",
        # The raw Orbax tree is ~95 GiB.  T0 inherits its already-audited
        # tree identity and intentionally does not rescan every byte.
        "bytes": None if is_tree else path.stat().st_size,
        "sha256": expected_sha256,
        "hash_source": "independent_current_tree_or_file" if independently_hashed else "prior_locked_audit",
    }
    if independently_hashed:
        value["sha256"] = sha256_tree(path) if is_tree else sha256_file(path)
    if value["sha256"] != expected_sha256:
        raise ValueError(f"hash mismatch for {path}: {value['sha256']} != {expected_sha256}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _entry_tokens(row: dict[str, Any]) -> set[str]:
    """Return non-empty public identifiers without preserving them in reports."""

    values = (
        row.get("task_name"),
        row.get("language", row.get("instruction", row.get("tasks"))),
        row.get("bddl_path"),
    )
    return {str(value) for value in values if value not in (None, "")}


def scan_evaluation_denylist(*, denylist_path: Path = PUBLIC_T3_HOLDOUT, training_manifest: Path = CANONICAL_SAMPLING) -> dict[str, Any]:
    """Compare public evaluation entries with training rows without exporting IDs."""

    if not denylist_path.is_file() or not training_manifest.is_file():
        raise FileNotFoundError(f"denylist or training manifest missing: {denylist_path}, {training_manifest}")
    deny_rows = _load_jsonl(denylist_path)
    train_rows = _load_jsonl(training_manifest)
    deny_keys = set().union(*(_entry_tokens(row) for row in deny_rows)) if deny_rows else set()
    train_keys = set().union(*(_entry_tokens(row) for row in train_rows)) if train_rows else set()
    exact = deny_keys & train_keys
    deny_digest = hashlib.sha256(canonical_json(sorted(deny_keys))).hexdigest()
    return {
        "status": "passed" if not exact else "failed",
        "denylist_path": str(denylist_path.resolve()),
        "denylist_sha256": sha256_file(denylist_path),
        "denylist_entry_count": len(deny_rows),
        "denylist_keyset_sha256": deny_digest,
        "training_manifest_path": str(training_manifest.resolve()),
        "training_manifest_sha256": sha256_file(training_manifest),
        "training_row_count": len(train_rows),
        "exact_overlap_count": len(exact),
        "overlap_key_types_checked": ["task_name", "language_or_instruction", "bddl_path"],
        "official_result": False,
        "identifiers_emitted": False,
    }


def _source_revisions() -> dict[str, Any]:
    lock = load_yaml(SOURCE_LOCK)
    datasets = {
        str(item["repo_id"]): str(item["revision"])
        for item in lock.get("datasets", [])
        if isinstance(item, dict) and item.get("repo_id") and item.get("revision")
    }
    canonical = load_yaml(STANDARD_CONFIG)["canonical_dataset"]
    plus = load_yaml(PLUS_CONFIG)["data"]["sources"]["libero_plus"]
    if datasets.get("lerobot/libero_plus") != EXPECTED_PLUS_REVISION:
        raise ValueError("libero_plus revision is not locked")
    if str(canonical["revision"]) != EXPECTED_CANONICAL_REVISION:
        raise ValueError("canonical LIBERO revision is not locked")
    return {
        "canonical_libero": {
            "repo_id": str(canonical["repo_id"]),
            "revision": str(canonical["revision"]),
            "fps": 10,
            "resampling": "none",
            "manifest_sha256": sha256_file(CANONICAL_SAMPLING),
            "payload_manifest_sha256": sha256_file(CANONICAL_PAYLOAD),
            "norm_manifest_sha256": sha256_file(NORM_MANIFEST),
        },
        "libero_plus": {
            "repo_id": str(plus["repo_id"]),
            "revision": str(plus["revision"]),
            "fps": 20,
            "resampling": "none",
            "metadata_sha256": sha256_file(PLUS_INFO),
            "tasks_metadata_sha256": sha256_file(PLUS_TASKS),
        },
    }


def build_t0_lock(*, snapshot: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create the CPU-only T0 lock; no model or simulator is loaded."""

    git = git_lock()
    if git["root_status"] or git["nested_status"]:
        raise RuntimeError("T0 requires clean root and nested worktrees before evidence is written")
    snapshot = gpu_read_only_snapshot() if snapshot is None else snapshot
    require_track3_targets_free(snapshot)

    prior = json.loads(R6_RELEASE_LOCK.read_text(encoding="utf-8"))
    if prior.get("raw_checkpoint", {}).get("tree_sha256") != EXPECTED_RAW_TREE:
        raise ValueError("prior R6 raw tree lock is not the expected immutable input")
    if prior.get("deployment_source", {}).get("model_sha256") != EXPECTED_FOLDED_SHA:
        raise ValueError("prior R6 folded model lock is not the expected immutable input")
    fast_audit = json.loads(FAST_AUDIT.read_text(encoding="utf-8"))
    fast_record = fast_audit.get("fast_model", {})
    if fast_record.get("sha256") != EXPECTED_FAST_SHA:
        raise ValueError("prior FAST audit is not the expected immutable input")
    if sha256_file(NORM_STATS) != EXPECTED_NORM_SHA:
        raise ValueError("fresh QUANTILES norm changed")
    if not FOLDED_MODEL.is_file() or FOLDED_MODEL.stat().st_size != 14467165872:
        raise ValueError("R6 folded model is missing or has unexpected size")
    if FAST_MODEL.stat().st_size != 9354050544:
        raise ValueError("R6 FAST model is missing or has unexpected size")
    if POSITIVE_ZIP.stat().st_size != 9383571677:
        raise ValueError("frozen positive-score ZIP is missing or has unexpected size")

    denylist = scan_evaluation_denylist()
    if denylist["exact_overlap_count"] != 0:
        raise RuntimeError("evaluation-only denylist overlaps the canonical training manifest")
    revisions = _source_revisions()
    record: dict[str, Any] = {
        "schema_version": 1,
        "phase": "T0",
        "status": "track3_reverse_t0_cpu_locked",
        "plan": str(PLAN.resolve()),
        "public_reverse_rule_only": True,
        "official_hidden_details_used": False,
        "synthetic_or_privileged_data_rule_authorization": "not_claimed_by_this_lock",
        "operator_families_registered": list(PUBLIC_OPERATOR_FAMILIES),
        "root_and_nested_git": git,
        "gpu_boundary": {
            "targets": [1, 2, 3],
            "target_idle_gate": "passed",
            "snapshot": snapshot,
            "gpu0_policy": "never_use; read-only observation only",
            "gpu0_compute_started_by_t0": False,
        },
        "immutable_assets": {
            "r6_raw_step500": _asset_record(RAW_CHECKPOINT, expected_sha256=EXPECTED_RAW_TREE),
            "r6_folded_model": _asset_record(FOLDED_MODEL, expected_sha256=EXPECTED_FOLDED_SHA),
            "r6_fast_model": _asset_record(FAST_MODEL, expected_sha256=EXPECTED_FAST_SHA),
            "positive_score_zip": _asset_record(POSITIVE_ZIP, expected_sha256=EXPECTED_ZIP_SHA),
            "fresh_quantiles_norm": _asset_record(NORM_STATS, expected_sha256=EXPECTED_NORM_SHA, independently_hashed=True),
        },
        "scores": {
            "track1": {"value": 0.47, "provenance": "user_reported", "official_receipt": False},
            "track2": {"value": 0.33, "provenance": "user_reported", "official_receipt": False},
            "track3": {"value": 0.0, "provenance": "user_reported", "official_receipt": False},
        },
        "source_revisions": revisions,
        "contracts": {
            "canonical_libero": {"source_fps": 10, "state_dim": 8, "action_dim": 7, "action_semantics": "native_libero_delta"},
            "libero_plus": {"source_fps": 20, "state_dim": 8, "action_dim": 7, "action_semantics": "native_libero_delta"},
            "reverse_training_target": {"fps": 20, "state_dim": 8, "action_dim": 7, "gripper_values": [-1, 1], "action_space": "OSC_POSE"},
        },
        "evaluation_only_denylist": denylist,
        "t0_policy": {
            "gpu_or_simulator_data_generation": False,
            "training_started": False,
            "hidden_track3_data_read": False,
            "zip_modified": False,
            "official_result": False,
        },
    }
    TRACK3_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(T0_LOCK, record)
    T0_REPORT.write_text(
        "# T0 CPU-only 输入与资源锁\n\n"
        f"状态：`{record['status']}`\n\n"
        "本阶段只读取已锁定资产、公开元数据和 GPU 状态；没有创建 CUDA context、启动 simulator、生成 demonstration 或启动训练。\n\n"
        f"- root commit：`{git['root_commit']}`；nested OpenPI：`{git['nested_commit']}`\n"
        "- GPU1–3 idle gate：`passed`；GPU0 policy：`never_use`\n"
        f"- evaluation-only denylist overlap：`{denylist['exact_overlap_count']}`\n"
        "- Track1/2/3 分数 provenance：均为 `user_reported`，没有伪造官方 receipt\n"
        "- FAST/ZIP：仅复用既有审核 hash，未修改或重新压缩 ZIP\n\n"
        f"机器锁：[t0_cpu_lock.json]({T0_LOCK.resolve()})\n",
        encoding="utf-8",
    )
    return record


def validate_t0_lock(record: dict[str, Any]) -> None:
    if record.get("status") != "track3_reverse_t0_cpu_locked":
        raise ValueError("unexpected T0 status")
    if record.get("public_reverse_rule_only") is not True:
        raise ValueError("public reverse rule declaration missing")
    if record.get("official_hidden_details_used") is not False:
        raise ValueError("hidden Track 3 details must remain unused")
    if record.get("gpu_boundary", {}).get("target_idle_gate") != "passed":
        raise ValueError("GPU target idle gate did not pass")
    if record.get("evaluation_only_denylist", {}).get("exact_overlap_count") != 0:
        raise ValueError("evaluation denylist overlap is non-zero")
    if any(value.get("provenance") != "user_reported" for value in record.get("scores", {}).values()):
        raise ValueError("score provenance was overclaimed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 3 reverse LoRA CPU-only T0 lock")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        value = json.loads(T0_LOCK.read_text(encoding="utf-8"))
        validate_t0_lock(value)
        print(json.dumps({"status": "passed", "lock": str(T0_LOCK)}, ensure_ascii=False, indent=2))
    else:
        value = build_t0_lock()
        print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
