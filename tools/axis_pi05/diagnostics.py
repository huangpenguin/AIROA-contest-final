from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, resolve_path, write_json


def failure_tags(episode: dict[str, Any]) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    if episode.get("collision"):
        tags.append({"tag": "collision", "source": "official_fact"})
    if episode.get("timeout"):
        tags.append({"tag": "timeout", "source": "official_fact"})
    if episode.get("policy_error") or episode.get("simulator_error"):
        tags.append({"tag": "environment_or_service_error", "source": "official_fact"})
    if episode.get("goal_reached") and episode.get("collision"):
        tags.append({"tag": "goal_reached_but_collision", "source": "official_fact"})
    if episode.get("action_saturation"):
        tags.append({"tag": "action_saturation", "source": "diagnostic_inference"})
    return tags


def main() -> None:
    ap = argparse.ArgumentParser(description="write timeline JSON and optional PNG for one episode")
    ap.add_argument("--episode", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    episode = json.loads(resolve_path(args.episode).read_text(encoding="utf-8"))
    timeline = {
        "schema_version": 1,
        "episode": episode,
        "failure_tags": failure_tags(episode),
        "series": {key: episode.get(key, []) for key in ("reward", "goal", "collision_displacement", "action", "action_norm", "replan_boundary")},
    }
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out.with_suffix(".json"), timeline)
    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError as exc:
        raise SystemExit(f"timeline JSON 已生成，但 PNG 需要 Pillow: {out.with_suffix('.json')}") from exc
    image = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Axis Pi0.5 episode timeline", fill="black")
    draw.text((20, 50), "; ".join(f"{tag['tag']}[{tag['source']}]" for tag in timeline["failure_tags"]) or "no failure tag", fill="black")
    for index, boundary in enumerate(episode.get("replan_boundary", [])):
        x = 20 + int(boundary) % 1160
        draw.line((x, 100, x, 380), fill="gray", width=1)
    image.save(out, format="PNG")


if __name__ == "__main__":
    main()
