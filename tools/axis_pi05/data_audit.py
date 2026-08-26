from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .catalog import build_catalog, catalog_report
from .common import ROOT, resolve_path, sha256_file, write_csv, write_json
from .lock import load_and_validate


def _plus_join_report(labels: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .catalog import load_plus_labels

    plus = [row for row in rows if row.get("source") == "libero_plus"]
    label_map = load_plus_labels(labels)
    missing = []
    task_mismatch = []
    length_mismatch = []
    for row in plus:
        label = label_map.get(int(row["source_episode_index"]))
        if label is None:
            missing.append(row["source_episode_index"])
            continue
        if str(label.get("task", "")) != str(row.get("task", "")):
            task_mismatch.append(row["source_episode_index"])
        if label.get("length") not in (None, "") and int(float(label["length"])) != int(row.get("frame_count", -1)):
            length_mismatch.append(row["source_episode_index"])
    return {
        "local_episode_count": len(plus),
        "label_row_count": len(label_map),
        "missing": missing,
        "task_mismatch": task_mismatch,
        "length_mismatch": length_mismatch,
        "one_to_one": not (missing or task_mismatch or length_mismatch) and len(plus) == len(label_map),
    }


def run(config_path: Path) -> dict[str, Any]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parents[2]
    lock = load_and_validate(resolve_path(config["lock"], root))
    raw_root = resolve_path(config["raw_root"], root)
    labels = resolve_path(config["plus_category_csv"], root)
    rows, source_report = build_catalog(lock, raw_root, labels)
    report = catalog_report(rows, source_report)
    report["lock_sha256"] = sha256_file(resolve_path(config["lock"], root))
    report["plus_join"] = _plus_join_report(labels, rows)
    report["catalog_hash_input"] = "canonical JSONL sorted by source/revision/episode"
    # Human-review flags are additive: they never delete an episode.
    for row in rows:
        flags: list[str] = []
        if not isinstance(row.get("frame_count"), int) or row.get("frame_count", 0) <= 0:
            flags.append("invalid_length")
        if row.get("state_dim") != 8:
            flags.append("state_dim")
        if row.get("action_dim") != 7:
            flags.append("action_dim")
        if set(row.get("cameras", [])) != {"front", "wrist"}:
            flags.append("camera_keys")
        if row.get("source_fps") != 20:
            flags.append("fps")
        row["review_flags"] = flags
    report["review_flag_count"] = sum(bool(row["review_flags"]) for row in rows)
    report["ok"] = bool(report["contract"]["ok"] and report["plus_join"]["one_to_one"])
    return {"config": config, "rows": rows, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description="audit all six locked sources and write the virtual catalog")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/axis_pi05/data.yaml")
    args = ap.parse_args()
    config_path = resolve_path(args.config)
    try:
        result = run(config_path)
    except Exception as exc:
        print(json.dumps({"ok": False, "blocked": True, "reason": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc
    config = result["config"]
    root = config_path.parents[2]
    catalog_path = resolve_path(config["catalog_path"], root)
    report_path = resolve_path(config["reports_root"], root) / "data_audit.json"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    from .common import write_jsonl

    write_jsonl(catalog_path, result["rows"])
    result["report"]["catalog_sha256"] = sha256_file(catalog_path)
    write_json(report_path, result["report"])
    write_csv(
        resolve_path(config["reports_root"], root) / "manual_review_manifest.csv",
        [
            {"source_repo": row["source_repo"], "episode": row["source_episode_index"], "suite": row["suite"], "task": row["task"], "review_flags": ";".join(row["review_flags"])}
            for row in result["rows"] if row["review_flags"]
        ],
        ["source_repo", "episode", "suite", "task", "review_flags"],
    )
    reports_root = resolve_path(config["reports_root"], root)
    source_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    category_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result["rows"]:
        source_groups[(row["source"], row["suite"])].append(row)
        task_groups[(row["source"], row["suite"], row["task"])].append(row)
        category_groups[(row["source"], row["perturb_category"])].append(row)
    write_csv(reports_root / "source_summary.csv", [
        {"source": source, "suite": suite, "episodes": len(rows), "frames": sum(int(row.get("frame_count", 0)) for row in rows), "review_flags": sum(bool(row["review_flags"]) for row in rows)}
        for (source, suite), rows in sorted(source_groups.items())
    ], ["source", "suite", "episodes", "frames", "review_flags"])
    write_csv(reports_root / "task_summary.csv", [
        {"source": source, "suite": suite, "task": task, "episodes": len(rows), "frames": sum(int(row.get("frame_count", 0)) for row in rows)}
        for (source, suite, task), rows in sorted(task_groups.items())
    ], ["source", "suite", "task", "episodes", "frames"])
    write_csv(reports_root / "category_summary.csv", [
        {"source": source, "category": category, "episodes": len(rows), "frames": sum(int(row.get("frame_count", 0)) for row in rows)}
        for (source, category), rows in sorted(category_groups.items())
    ], ["source", "category", "episodes", "frames"])
    write_csv(reports_root / "action_state_quantiles.csv", [{"status": "requires_frame_scan", "reason": "catalog audit does not invent numeric values from metadata"}], ["status", "reason"])
    write_csv(reports_root / "corrupt_or_unreadable.csv", [
        {"source_repo": row["source_repo"], "episode": row["source_episode_index"], "flags": ";".join(row["review_flags"])}
        for row in result["rows"] if row["review_flags"]
    ], ["source_repo", "episode", "flags"])
    cards = "".join(
        f"<tr data-search='{row['source']} {row['suite']} {row['task']} {row['perturb_category']}'><td>{row['source']}</td><td>{row['suite']}</td><td>{row['task']}</td><td>{row['perturb_category']}</td><td>{';'.join(row['review_flags'])}</td></tr>"
        for row in result["rows"][:5000]
    )
    (reports_root / "eda_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Axis Pi0.5 EDA</title>"
        "<style>body{font:14px sans-serif;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.25rem}</style>"
        "<label>filter <input id='q'></label><table><thead><tr><th>source</th><th>suite</th><th>task</th><th>category</th><th>review flags</th></tr></thead><tbody>"
        + cards + "</tbody></table><script>q.oninput=()=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.dataset.search.toLowerCase().includes(q.value.toLowerCase()))</script>",
        encoding="utf-8",
    )
    print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    if not result["report"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
