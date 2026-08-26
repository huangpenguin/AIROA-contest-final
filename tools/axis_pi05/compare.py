from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, resolve_path, write_json


def _worst_source(summary: dict[str, Any]) -> float:
    values = [value.get("success") for value in summary.get("source_macro", {}).values() if isinstance(value, dict) and isinstance(value.get("success"), (int, float))]
    return min(values) if values else -1.0


def rank(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(summaries, key=lambda row: (float(row.get("success", -1.0)), _worst_source(row), -float(row.get("collision_rate", 1.0))), reverse=True)
    if len(ordered) >= 2:
        first, second = ordered[:2]
        if abs(float(first.get("success", -1)) - float(second.get("success", -1))) < 0.02 and float(second.get("collision_rate", 1)) < float(first.get("collision_rate", 1)):
            ordered[0], ordered[1] = second, first
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser(description="write a stage comparison; never auto-activates a winner")
    ap.add_argument("--summary", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--write-winner", action="store_true", help="explicitly write winner.json after operator approval")
    args = ap.parse_args()
    summaries = []
    for path in args.summary:
        value = json.loads(resolve_path(path).read_text(encoding="utf-8"))
        summaries.extend(value.get("summaries", [value]) if isinstance(value, dict) else [])
    ordered = rank(summaries)
    result = {"schema_version": 1, "selection_policy": "validation task×perturbation macro, worst source, T3, collision; <2pp tie uses lower collision", "candidates": ordered, "winner": ordered[0] if ordered else None, "operator_approved": bool(args.write_winner)}
    out = resolve_path(args.out)
    write_json(out, result)
    if args.write_winner and ordered:
        write_json(out.with_name("winner.json"), {"approved": True, "winner": ordered[0]})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
