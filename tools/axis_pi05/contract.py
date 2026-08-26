from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

from .common import FULL_SHA_RE, ROOT, load_yaml, resolve_path, sha256_file, write_json


CATEGORIES = {"base", "env", "lighting", "language", "camera_pose", "sensor_noise"}
SOURCES = {"libero_standard", "libero_plus"}
QUALITY = {"exact-source", "recovered-coarse", "unavailable"}


def validate_sample(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "source_fps", "source", "source_repo", "revision", "source_episode_index",
        "suite", "task", "task_index", "instruction", "perturb_category", "perturb_label_quality",
    }
    errors.extend(f"missing:{key}" for key in sorted(required - set(row)))
    if row.get("source_fps") != 20:
        errors.append(f"source_fps={row.get('source_fps')!r}, expected 20")
    if row.get("source") not in SOURCES:
        errors.append(f"invalid source={row.get('source')!r}")
    if row.get("perturb_category") not in CATEGORIES:
        errors.append(f"invalid perturb_category={row.get('perturb_category')!r}")
    if row.get("perturb_label_quality") not in QUALITY:
        errors.append(f"invalid perturb_label_quality={row.get('perturb_label_quality')!r}")
    if not isinstance(row.get("revision"), str) or not re.fullmatch(FULL_SHA_RE, row.get("revision", "")):
        errors.append("revision is not a full SHA")
    for key in ("source_episode_index", "task_index"):
        if not isinstance(row.get(key), int) or row[key] < 0:
            errors.append(f"invalid {key}={row.get(key)!r}")
    for key in ("task", "instruction", "suite", "source_repo"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"invalid {key}")
    return errors


def validate_frame(row: dict[str, Any]) -> list[str]:
    errors = validate_sample(row)
    for key in ("observation.images.front", "observation.images.wrist"):
        if key not in row:
            errors.append(f"missing:{key}")
    errors.extend(validate_numeric_vector(row.get("observation.state"), dim=8, name="observation.state"))
    errors.extend(validate_numeric_vector(row.get("action"), dim=7, name="action"))
    return errors


def validate_numeric_vector(values: Any, *, dim: int, name: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or len(values) != dim:
        return [f"{name}: expected length {dim}"]
    errors: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            errors.append(f"{name}[{index}] is not finite")
    return errors


def validate_catalog(rows: Iterable[dict[str, Any]], *, expected_fps: int = 20) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    count = 0
    for row in rows:
        count += 1
        row_errors = validate_sample(row)
        identity = (str(row.get("source_repo")), str(row.get("revision")), int(row.get("source_episode_index", -1)))
        if identity in seen:
            row_errors.append("duplicate episode identity")
        seen.add(identity)
        if row.get("source_fps") != expected_fps:
            row_errors.append("fps gate failed")
        if row_errors:
            errors.append({"row": count - 1, "errors": row_errors, "identity": identity})
    return {"rows": count, "errors": errors, "ok": not errors, "unique_episode_identities": len(seen)}


def validate_norm_stats(path: Path, *, state_dim: int = 8, action_dim: int = 7) -> dict[str, Any]:
    value = __import__("json").loads(path.read_text(encoding="utf-8"))
    if isinstance(value.get("norm_stats"), dict):
        value = value["norm_stats"]
    errors: list[str] = []
    for key, dim in (("state", state_dim), ("actions", action_dim), ("action", action_dim)):
        if key not in value:
            continue
        stats = value[key]
        if not isinstance(stats, dict):
            errors.append(f"{key}: expected mapping")
            continue
        for stat_name in ("mean", "std", "min", "max"):
            if stat_name in stats:
                errors.extend(validate_numeric_vector(stats[stat_name], dim=dim, name=f"{key}.{stat_name}"))
    if "state" not in value or not ("actions" in value or "action" in value):
        errors.append("must contain state and actions/action stats")
    return {"path": str(path), "sha256": sha256_file(path), "errors": errors, "ok": not errors}


def main() -> None:
    ap = argparse.ArgumentParser(description="validate canonical sample or norm-stat contracts")
    ap.add_argument("--sample-json", type=Path)
    ap.add_argument("--norm-stats", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if bool(args.sample_json) == bool(args.norm_stats):
        raise SystemExit("二选一：--sample-json 或 --norm-stats")
    if args.sample_json:
        result = {"sample_errors": validate_sample(__import__("json").loads(args.sample_json.read_text()))}
        result["ok"] = not result["sample_errors"]
    else:
        result = validate_norm_stats(resolve_path(args.norm_stats))
    if args.json:
        write_json(resolve_path(args.json), result)
    print(result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
