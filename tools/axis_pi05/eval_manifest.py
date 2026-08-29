"""Build and validate immutable simulator evaluation manifests.

The builder enumerates the benchmark registry and its real init-state tensors.
It never derives cases from training parquet files and never invents cases when
a suite cannot be registered in the current evaluator environment.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common import (
    ROOT,
    canonical_json,
    dump_yaml,
    load_yaml,
    read_jsonl,
    resolve_path,
    sha256_bytes,
    sha256_file,
    write_json,
    write_jsonl,
)


MANIFEST_FIELDS = (
    "schema_version",
    "case_id",
    "panel",
    "selection_eligible",
    "benchmark",
    "suite",
    "task_id",
    "task_name",
    "language",
    "bddl_path",
    "bddl_sha256",
    "init_state_index",
    "init_state_sha256",
    "env_seed",
    "policy_seed",
    "max_steps",
    "perturbation_id",
    "perturbation_family",
    "perturbation_params",
    "repeat_id",
)


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_versions() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ("libero", "robosuite", "bddl"):
        try:
            module = __import__(name)
            modules[name] = {
                "version": getattr(module, "__version__", None),
                "file": str(getattr(module, "__file__", "")),
            }
        except Exception as exc:
            modules[name] = {"version": None, "file": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "libero": _version("libero"),
        "robosuite": _version("robosuite"),
        "bddl": _version("bddl"),
        "numpy": _version("numpy"),
        "modules": modules,
    }


def _state_hash(state: Any) -> str:
    array = np.asarray(state)
    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }
    return sha256_bytes(canonical_json(payload))


def _path_for_record(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _canonical_case_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in MANIFEST_FIELDS if key != "case_id"}


def case_id(row: dict[str, Any]) -> str:
    return "sha256:" + sha256_bytes(canonical_json(_canonical_case_payload(row)))


def validate_case(row: dict[str, Any]) -> None:
    missing = sorted(set(MANIFEST_FIELDS) - set(row))
    if missing:
        raise ValueError(f"manifest row missing fields: {missing}")
    if row["schema_version"] != 1:
        raise ValueError(f"unsupported manifest row schema: {row['schema_version']}")
    if row["case_id"] != case_id(row):
        raise ValueError(f"case_id mismatch: {row.get('case_id')}")
    if int(row["init_state_index"]) < 0 or int(row["max_steps"]) <= 0:
        raise ValueError("init_state_index must be non-negative and max_steps positive")
    if not isinstance(row["perturbation_params"], dict):
        raise ValueError("perturbation_params must be an object")


def validate_manifest_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids: list[str] = []
    for row in rows:
        validate_case(row)
        ids.append(row["case_id"])
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate case ids: {duplicates[:5]}")
    return {
        "case_count": len(rows),
        "unique_case_count": len(set(ids)),
        "complete": len(rows) == len(set(ids)),
        "case_ids_sha256": sha256_bytes(canonical_json(sorted(ids))),
    }


def manifest_sha256(rows: Iterable[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json(row) for row in sorted(rows, key=lambda item: item["case_id"]))
    return sha256_bytes(payload)


_TASK_CACHE: dict[tuple[str, int, tuple[int, ...] | None], tuple[list[Any], str | None]] = {}
_TASK_HEADER_CACHE: dict[tuple[str, int], tuple[list[dict[str, Any]], str | None]] = {}


def _load_registered_task_headers(benchmark: str, task_order_index: int) -> tuple[list[dict[str, Any]], str | None]:
    """Load task metadata without materializing every task's init-state tensor."""

    cache_key = (benchmark, int(task_order_index))
    if cache_key in _TASK_HEADER_CACHE:
        return _TASK_HEADER_CACHE[cache_key]
    try:
        from pipeline.config import EvalConfig
        from pipeline.environment import EnvironmentManager

        with contextlib.redirect_stdout(io.StringIO()):
            manager = EnvironmentManager(EvalConfig(task_order_index=task_order_index))
            suite = manager.get_task_suite(benchmark)
            result = ([{
                "task_id": int(task_id),
                "name": str(suite.get_task(task_id).name),
                "language": str(suite.get_task(task_id).language),
                "bddl_file": str(suite.get_task_bddl_file_path(task_id)),
            } for task_id in range(suite.get_num_tasks())], None)
    except Exception as exc:
        result = ([], f"{type(exc).__name__}: {exc}")
    _TASK_HEADER_CACHE[cache_key] = result
    return result


def _load_registered_tasks(
    benchmark: str,
    task_order_index: int,
    task_ids: Iterable[int] | None = None,
) -> tuple[list[Any], str | None]:
    selected_ids = None if task_ids is None else tuple(sorted({int(task_id) for task_id in task_ids}))
    cache_key = (benchmark, int(task_order_index), selected_ids)
    if cache_key in _TASK_CACHE:
        return _TASK_CACHE[cache_key]
    try:
        from pipeline.config import EvalConfig
        from pipeline.environment import EnvironmentManager, TaskInfo

        # Some pinned registry helpers print a large task-order list at import
        # time.  Keep availability.json useful without hiding exceptions.
        with contextlib.redirect_stdout(io.StringIO()):
            manager = EnvironmentManager(EvalConfig(task_order_index=task_order_index))
            if selected_ids is None:
                result = (manager.get_task_infos(benchmark), None)
            else:
                suite = manager.get_task_suite(benchmark)
                tasks = []
                for task_id in selected_ids:
                    task = suite.get_task(task_id)
                    tasks.append(TaskInfo(
                        task_id=int(task_id),
                        name=task.name,
                        language=task.language,
                        bddl_file=suite.get_task_bddl_file_path(task_id),
                        init_states=suite.get_task_init_states(task_id),
                        benchmark_name=benchmark,
                    ))
                result = (tasks, None)
    except Exception as exc:
        result = ([], f"{type(exc).__name__}: {exc}")
    _TASK_CACHE[cache_key] = result
    return result


def _seed_list(panel: dict[str, Any], key: str, fallback: int) -> list[int]:
    values = panel.get(key)
    if values is None:
        return [fallback]
    return [int(value) for value in values]


def _select_tasks(tasks: list[Any], panel: dict[str, Any]) -> list[Any]:
    count = panel.get("tasks_per_suite")
    if count is None:
        return tasks
    count = int(count)
    if count <= 0:
        return []
    if panel.get("task_selection") == "sha256_name":
        tasks = sorted(
            tasks,
            key=lambda task: hashlib.sha256(
                str(task["name"] if isinstance(task, dict) else task.name).encode()
            ).hexdigest(),
        )
    return tasks[:count]


def build_panel(
    panel_name: str,
    panel: dict[str, Any],
    *,
    task_order_index: int,
    default_max_steps: int,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    suite_status: dict[str, Any] = {}
    perturbations = panel.get("perturbations", [])
    if panel_name == "stress_perturbation" and not perturbations:
        return [], {
            "status": "unsupported",
            "reason": "no perturbation family has an implemented, validated effect test",
            "suites": {},
        }

    env_seeds = _seed_list(panel, "env_seeds", 1000)
    policy_seeds = _seed_list(panel, "policy_seeds", 2000)
    init_limit = int(panel.get("init_states_per_task", 0))
    for benchmark in panel.get("suites", []):
        headers, error = _load_registered_task_headers(str(benchmark), task_order_index)
        if error:
            suite_status[str(benchmark)] = {"status": "unavailable", "error": error, "task_count": 0}
            continue
        selected_headers = _select_tasks(headers, panel)
        tasks, error = _load_registered_tasks(
            str(benchmark), task_order_index,
            task_ids=[header["task_id"] for header in selected_headers],
        )
        if error:
            suite_status[str(benchmark)] = {"status": "unavailable", "error": error, "task_count": 0}
            continue
        suite_status[str(benchmark)] = {"status": "available", "error": None, "task_count": len(tasks)}
        for task in tasks:
            bddl = Path(task.bddl_file)
            if not bddl.is_file():
                suite_status[str(benchmark)] = {
                    "status": "unavailable",
                    "error": f"BDDL file does not exist: {bddl}",
                    "task_count": 0,
                }
                continue
            try:
                init_states = np.asarray(task.init_states)
                bddl_hash = sha256_file(bddl)
            except Exception as exc:
                suite_status[str(benchmark)] = {
                    "status": "unavailable",
                    "error": f"cannot read task assets: {type(exc).__name__}: {exc}",
                    "task_count": 0,
                }
                continue
            count = min(init_limit, int(init_states.shape[0]))
            for init_index in range(count):
                state_hash = _state_hash(init_states[init_index])
                for env_seed in env_seeds:
                    for policy_seed in policy_seeds:
                        if panel_name == "stress_perturbation":
                            perturbation_rows = perturbations
                        else:
                            perturbation_rows = [{
                                "id": "clean",
                                "family": "clean",
                                "params": {},
                            }]
                        for perturbation in perturbation_rows:
                            row = {
                                "schema_version": 1,
                                "case_id": "",
                                "panel": panel_name,
                                "selection_eligible": bool(panel.get("selection_eligible", False)),
                                "benchmark": str(benchmark),
                                "suite": str(benchmark),
                                "task_id": int(task.task_id),
                                "task_name": str(task.name),
                                "language": str(task.language),
                                "bddl_path": _path_for_record(bddl, root),
                                "bddl_sha256": bddl_hash,
                                "init_state_index": int(init_index),
                                "init_state_sha256": state_hash,
                                "env_seed": int(env_seed),
                                "policy_seed": int(policy_seed),
                                "max_steps": int(panel.get("max_steps", default_max_steps)),
                                "perturbation_id": str(perturbation.get("id", "clean")),
                                "perturbation_family": str(perturbation.get("family", "clean")),
                                "perturbation_params": dict(perturbation.get("params", {})),
                                "repeat_id": 0,
                            }
                            row["case_id"] = case_id(row)
                            rows.append(row)
    rows.sort(key=lambda item: item["case_id"])
    status = "available" if rows else "unavailable"
    return rows, {"status": status, "suites": suite_status, "case_count": len(rows)}


def build_manifests(config_path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_root = resolve_path(config.get("output_root", "data/eval_manifests/axis_pi05_eval_v1"), root)
    output_root.mkdir(parents=True, exist_ok=True)
    panels = config.get("panels", {})
    if not isinstance(panels, dict) or not panels:
        raise ValueError("eval_v2 config must define panels")
    availability: dict[str, Any] = {
        "schema_version": 1,
        "manifest_version": config.get("manifest_version", "unknown"),
        "official_result": False,
        "software": software_versions(),
        "panels": {},
    }
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for panel_name, panel_config in panels.items():
        rows, status = build_panel(
            str(panel_name),
            dict(panel_config),
            task_order_index=int(config.get("task_order_index", 0)),
            default_max_steps=int(config.get("default_max_steps", 300)),
            root=root,
        )
        path = output_root / f"{panel_name}.jsonl"
        if panel_name == "public_t3_holdout" and status.get("status") == "unavailable":
            availability["panels"][panel_name] = {**status, "path": None, "optional": True}
            continue
        validate_manifest_rows(rows)
        write_jsonl(path, rows)
        hashes[panel_name] = sha256_file(path)
        counts[panel_name] = len(rows)
        availability["panels"][panel_name] = {
            **status,
            "path": str(path.relative_to(root)),
            "optional": bool(panel_config.get("optional", False)),
            "manifest_sha256": hashes[panel_name],
        }
    availability_path = output_root / "availability.json"
    write_json(availability_path, availability)
    lock = {
        "schema_version": 1,
        "manifest_version": config.get("manifest_version", "unknown"),
        "config_sha256": sha256_file(config_path),
        "availability_sha256": sha256_file(availability_path),
        "panels": hashes,
        "case_counts": counts,
        "official_result": False,
        "locked": True,
    }
    write_json(output_root / "manifest.lock.json", lock)
    return {
        "output_root": str(output_root),
        "availability": availability,
        "lock": lock,
    }


def load_locked_panel(path: Path, lock_path: Path | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    validate_manifest_rows(rows)
    expected = None
    if lock_path is not None:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = lock.get("panels", {}).get(path.stem)
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"manifest hash mismatch for {path}: {actual} != {expected}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/eval_v2.yaml")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = build_manifests(resolve_path(args.config), root=resolve_path(args.root))
    print(json.dumps({"output_root": report["output_root"], "panels": report["lock"]["case_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
