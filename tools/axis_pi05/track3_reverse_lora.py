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
from dataclasses import dataclass
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

PUBLIC_BDDL_ROOT = ROOT / "LIBERO-plus/libero/libero/bddl_files/libero_mix"
PUBLIC_INIT_ROOT = ROOT / "LIBERO-plus/libero/libero/init_files/libero_mix"
T1_COMPILED_ROOT = TRACK3_ROOT / "t1_compiled_bddl"
T1_MANIFEST = TRACK3_ROOT / "t1_reverse_task_manifest.jsonl"
T1_REPORT = TRACK3_ROOT / "T1_CPU_COMPILER_REPORT_ZH.md"


@dataclass(frozen=True)
class Predicate:
    """One public BDDL predicate, kept as an immutable CPU-side value."""

    name: str
    args: tuple[str, ...]

    def render(self) -> str:
        return f"({self.name} {' '.join(self.args)})"


_ATOM_RE = re.compile(r"\((In|On|Open|Close|Turnon|Turnoff)\s+([^()\s]+)(?:\s+([^()\s]+))?\)")
_SECTION_NAMES = ("regions", "fixtures", "objects", "obj_of_interest", "init", "goal")


def _balanced_section(text: str, section: str) -> tuple[int, int, str]:
    """Return the exact span/body of a balanced public BDDL section."""

    marker = f"(:{section}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing BDDL section: {section}")
    open_index = text.find("(", start)
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1, text[start : index + 1]
    raise ValueError(f"unbalanced BDDL section: {section}")


def parse_public_predicates(text: str, section: str) -> tuple[Predicate, ...]:
    """Parse only executable public predicates from one BDDL section."""

    _, _, body = _balanced_section(text, section)
    return tuple(Predicate(name, tuple(arg for arg in (left, right) if arg)) for name, left, right in _ATOM_RE.findall(body))


def _canonical_public_bddl(language: str) -> Path:
    """Resolve the checked-in canonical LIBERO-mix BDDL for a public language."""

    safe = language.replace(" ", "_")
    direct = PUBLIC_BDDL_ROOT / f"{safe}.bddl"
    if direct.is_file():
        return direct
    candidates = sorted(PUBLIC_BDDL_ROOT.glob(f"*_{safe}*.bddl"))
    excluded = (" copy", "_add_", "_language_", "_tb_", "_view_", "_initstate_", "_pruned", "_aug", "_demo", "_level", "_light", "_table_")
    candidates = [path for path in candidates if not any(token in path.name for token in excluded)]
    if not candidates:
        raise FileNotFoundError(f"no canonical public LIBERO-mix BDDL for {language!r}")
    return candidates[0]


def resolve_public_bddl(language: str) -> tuple[Path, Path]:
    """Resolve a public BDDL and its matching fixed-init state file."""

    bddl = _canonical_public_bddl(language)
    stem = re.sub(r" copy(?: \d+)?$", "", bddl.stem)
    stem = re.sub(r"_(?:language|add|tb|view)_\d+$", "", stem)
    init = PUBLIC_INIT_ROOT / f"{stem}.pruned_init"
    if not init.is_file():
        init = PUBLIC_INIT_ROOT / f"{language.replace(' ', '_')}.pruned_init"
    if not init.is_file():
        raise FileNotFoundError(f"missing canonical public init state for {language!r}: {bddl}")
    return bddl, init


def _flip_state(name: str) -> str:
    pairs = {"Open": "Close", "Close": "Open", "Turnon": "Turnoff", "Turnoff": "Turnon"}
    return pairs[name]


def _same_subject(left: Predicate, right: Predicate) -> bool:
    return bool(left.args) and bool(right.args) and left.args[0] == right.args[0]


def _human_label(value: str) -> str:
    value = re.sub(r"(?:^|_)\d+$", "", value)
    value = value.replace("_", " ")
    value = re.sub(r"^(?:kitchen table|living room table|main table) ", "", value)
    value = value.replace(" init region", " original starting location")
    value = value.replace(" region", "")
    return value


def _reverse_instruction(operations: list[tuple[str, Predicate, Predicate]]) -> str:
    sentences: list[str] = []
    for family, forward_goal, reverse_goal in operations:
        if family == "containment_remove_restore":
            sentences.append(
                f"take the {_human_label(forward_goal.args[0])} out of the {_human_label(forward_goal.args[1])} and return it to its {_human_label(reverse_goal.args[1])}"
            )
        elif family == "support_remove_restore":
            sentences.append(
                f"remove the {_human_label(forward_goal.args[0])} from the {_human_label(forward_goal.args[1])} and put it back at its {_human_label(reverse_goal.args[1])}"
            )
        elif family == "articulation_inverse":
            verb = "open" if reverse_goal.name == "Open" else "close"
            sentences.append(f"{verb} the {_human_label(reverse_goal.args[0])}")
        elif family == "activation_inverse":
            verb = "turn on" if reverse_goal.name == "Turnon" else "turn off"
            sentences.append(f"{verb} the {_human_label(reverse_goal.args[0])}")
    if not sentences:
        raise ValueError("reverse instruction has no executable operation")
    return "Reverse the public LIBERO operation: " + "; then ".join(sentences) + "."


def compile_reverse_spec(task_index: int, language: str, bddl: Path, init_path: Path) -> dict[str, Any]:
    """Compile one public task by swapping forward state/goal, never A/B roles."""

    text = bddl.read_text(encoding="utf-8")
    init = list(parse_public_predicates(text, "init"))
    goal = list(parse_public_predicates(text, "goal"))
    if not init or not goal:
        raise ValueError("public BDDL has no executable init/goal predicates")

    reverse_initial: list[Predicate] = []
    reverse_goal: list[Predicate] = []
    consumed_init: set[int] = set()
    operations: list[tuple[str, Predicate, Predicate]] = []

    relation_goals = [predicate for predicate in goal if predicate.name in {"In", "On"}]
    for forward_goal in relation_goals:
        if len(forward_goal.args) != 2:
            raise ValueError(f"relation goal has unexpected arity: {forward_goal}")
        matches = [
            (index, predicate)
            for index, predicate in enumerate(init)
            if predicate.name in {"In", "On"} and predicate.args[0] == forward_goal.args[0]
        ]
        if len(matches) != 1 or matches[0][1].args[1] == forward_goal.args[1]:
            raise ValueError(f"relation goal is not a reversible move: {forward_goal}")
        init_index, forward_init = matches[0]
        consumed_init.add(init_index)
        family = "containment_remove_restore" if forward_goal.name == "In" else "support_remove_restore"
        reverse_initial.append(forward_goal)
        reverse_goal.append(forward_init)
        operations.append((family, forward_goal, forward_init))

    state_goals = [predicate for predicate in goal if predicate.name in {"Open", "Close", "Turnon", "Turnoff"}]
    for forward_goal in state_goals:
        matches = [
            (index, predicate)
            for index, predicate in enumerate(init)
            if predicate.name == forward_goal.name and predicate.args == forward_goal.args
        ]
        if matches:
            # Some public multi-step tasks repeat a fixture condition (for
            # example, Turnon) in both init and goal.  It is an invariant, not
            # a reverse operator, so leave it in the unchanged init context.
            continue
        opposite = _flip_state(forward_goal.name)
        opposite_matches = [
            (index, predicate)
            for index, predicate in enumerate(init)
            if predicate.name == opposite and predicate.args == forward_goal.args
        ]
        if len(opposite_matches) > 1:
            raise ValueError(f"state init contains duplicate opposite predicates: {forward_goal}")
        if opposite_matches:
            consumed_init.add(opposite_matches[0][0])
        reverse_initial.append(forward_goal)
        reverse_goal_predicate = opposite_matches[0][1] if opposite_matches else Predicate(opposite, forward_goal.args)
        reverse_goal.append(reverse_goal_predicate)
        operations.append(("articulation_inverse" if forward_goal.name in {"Open", "Close"} else "activation_inverse", forward_goal, reverse_goal_predicate))

    if not operations:
        raise ValueError("public task has no supported reverse operator")
    families = {family for family, _, _ in operations}
    family = next(iter(families)) if len(operations) == 1 else "sequence_reverse"
    if family not in PUBLIC_OPERATOR_FAMILIES:
        raise ValueError(f"unsupported reverse family: {family}")

    unchanged_init = [predicate for index, predicate in enumerate(init) if index not in consumed_init]
    reverse_initial = unchanged_init + reverse_initial
    reverse_goal = list(dict.fromkeys(reverse_goal))
    scene_match = re.match(r"((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", bddl.stem)
    scene_id = scene_match.group(1) if scene_match else "public_mix_generic"
    template_id = hashlib.sha256(bddl.read_bytes()).hexdigest()[:16]
    seed = 310000 + int(task_index)
    provenance = hashlib.sha256(canonical_json({
        "task_index": task_index,
        "language": language,
        "bddl_sha256": sha256_file(bddl),
        "init_sha256": sha256_file(init_path),
        "reverse_initial": [predicate.render() for predicate in reverse_initial],
        "reverse_goal": [predicate.render() for predicate in reverse_goal],
        "family": family,
        "seed": seed,
    })).hexdigest()
    return {
        "schema_version": "track3_reverse_pilot_v1",
        "task_index": task_index,
        "base_public_task": language,
        "base_task_template_id": template_id,
        "scene_id": scene_id,
        "operator_family": family,
        "seed": seed,
        "forward_bddl_path": str(bddl.resolve()),
        "forward_bddl_sha256": sha256_file(bddl),
        "forward_init_path": str(init_path.resolve()),
        "forward_init_sha256": sha256_file(init_path),
        "reverse_instruction": _reverse_instruction(operations),
        "reverse_initial_predicates": [predicate.render() for predicate in reverse_initial],
        "reverse_goal_predicates": [predicate.render() for predicate in reverse_goal],
        "invariants": [predicate.render() for predicate in unchanged_init],
        "provenance_sha256": provenance,
        "official_result": False,
        "training_eligible": True,
    }


def _render_custom_bddl(text: str, language: str, initial: list[str], goal: list[str]) -> str:
    """Render a local reverse BDDL while preserving public world geometry."""

    def section(name: str, values: list[str]) -> str:
        if name == "goal":
            expression = values[0] if len(values) == 1 else "(And " + " ".join(values) + ")"
        else:
            expression = "\n".join(f"    {value}" for value in values)
        return f"(:{name}\n    {expression}\n  )"

    replacements = [("language", f"(:language {language})"), ("init", section("init", initial)), ("goal", section("goal", goal))]
    rendered = text
    for name, replacement in replacements:
        start, end, _ = _balanced_section(rendered, name)
        rendered = rendered[:start] + replacement + rendered[end:]
    return rendered


def compile_public_reverse_tasks() -> dict[str, Any]:
    """CPU-only T1 compiler; no simulator, CUDA, model, or hidden task import."""

    validate_t0_lock(json.loads(T0_LOCK.read_text(encoding="utf-8")))
    catalog = [json.loads(line)["task"] for line in CANONICAL_TASKS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(catalog) != 40:
        raise RuntimeError(f"public canonical catalog must have 40 tasks, got {len(catalog)}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for task_index, language in enumerate(catalog):
        try:
            bddl, init_path = resolve_public_bddl(language)
            spec = compile_reverse_spec(task_index, language, bddl, init_path)
            accepted.append(spec)
        except (FileNotFoundError, ValueError) as exc:
            rejected.append({"task_index": task_index, "reason_class": type(exc).__name__, "reason": str(exc)})

    if not accepted:
        raise RuntimeError("public reverse compiler produced no supported task")
    template_order = sorted({row["base_task_template_id"] for row in accepted})
    holdout_count = max(1, (len(template_order) + 4) // 5)
    holdout_templates = set(template_order[:: max(1, len(template_order) // holdout_count)][:holdout_count])
    for row in accepted:
        row["split"] = "template_holdout" if row["base_task_template_id"] in holdout_templates else "train"
        source = Path(row["forward_bddl_path"])
        initial = list(row["reverse_initial_predicates"])
        goal = list(row["reverse_goal_predicates"])
        output = T1_COMPILED_ROOT / f"task_{int(row['task_index']):02d}_{row['base_task_template_id']}.bddl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_render_custom_bddl(source.read_text(encoding="utf-8"), row["reverse_instruction"], initial, goal), encoding="utf-8")
        row["reverse_bddl_path"] = str(output.resolve())
        row["reverse_bddl_sha256"] = sha256_file(output)

    T1_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    T1_MANIFEST.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted), encoding="utf-8")
    split_counts = {split: sum(row["split"] == split for row in accepted) for split in ("train", "template_holdout")}
    family_counts = {family: sum(row["operator_family"] == family for row in accepted) for family in PUBLIC_OPERATOR_FAMILIES}
    report = {
        "status": "t1_cpu_compiled_pending_pilot",
        "schema_version": "track3_reverse_pilot_v1",
        "public_catalog_tasks": len(catalog),
        "accepted_compiled_tasks": len(accepted),
        "rejected_tasks": len(rejected),
        "rejections": rejected,
        "split_counts": split_counts,
        "operator_family_counts": family_counts,
        "template_count": len(template_order),
        "holdout_template_count": len(holdout_templates),
        "holdout_fraction": split_counts["template_holdout"] / len(accepted),
        "public_reverse_rule_only": True,
        "hidden_track3_details_used": False,
        "simulator_or_cuda_started": False,
        "accepted_episodes": 0,
        "pilot_gate": "not_met",
        "training_started": False,
        "official_result": False,
        "manifest_sha256": sha256_file(T1_MANIFEST),
    }
    write_json(TRACK3_ROOT / "t1_cpu_compile_summary.json", report)
    T1_REPORT.write_text(
        "# T1 CPU-only 公共 reverse 编译报告\n\n"
        f"状态：`{report['status']}`\n\n"
        f"公共任务：`{len(catalog)}`；编译通过：`{len(accepted)}`；静态拒绝：`{len(rejected)}`。\n\n"
        f"template split：train `{split_counts['train']}` / holdout `{split_counts['template_holdout']}`（`{report['holdout_fraction']:.1%}`）。\n\n"
        f"operator families：`{json.dumps(family_counts, ensure_ascii=False, sort_keys=True)}`。\n\n"
        "这些只是由公开 BDDL 的 init/goal 交换生成的候选规格，不是 accepted demonstration；尚未启动 simulator、CUDA 或训练。\n\n"
        "下一门禁：GPU3 真实 20Hz pilot，必须达到 30–60 条且每条 3/3 replay predicate success；当前 accepted episodes 为 `0`。\n\n"
        f"manifest SHA-256：`{report['manifest_sha256']}`\n",
        encoding="utf-8",
    )
    return report


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
    parser = argparse.ArgumentParser(description="Track 3 reverse LoRA CPU-side lock/compiler")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compile", action="store_true", help="compile public LIBERO reverse task candidates on CPU")
    args = parser.parse_args()
    if args.compile:
        value = compile_public_reverse_tasks()
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.validate:
        value = json.loads(T0_LOCK.read_text(encoding="utf-8"))
        validate_t0_lock(value)
        print(json.dumps({"status": "passed", "lock": str(T0_LOCK)}, ensure_ascii=False, indent=2))
    else:
        value = build_t0_lock()
        print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
