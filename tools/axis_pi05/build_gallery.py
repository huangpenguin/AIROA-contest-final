from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .common import ROOT, resolve_path


def main() -> None:
    ap = argparse.ArgumentParser(description="build a local filterable review gallery")
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    run = resolve_path(args.run)
    review_root = run if (run / "cases").exists() else ROOT / "reviews/axis_pi05" / run.name
    out = resolve_path(args.out) if args.out else review_root / "gallery.html"
    cards = []
    for episode in sorted(review_root.rglob("episode.json")) + sorted(review_root.rglob("*.episode.json")):
        metadata = json.loads(episode.read_text(encoding="utf-8"))
        video = episode.parent / "failure_h264.mp4"
        if not video.is_file():
            continue
        text = html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))
        tags = " ".join(html.escape(str(metadata.get(key, "unavailable"))) for key in ("internal_step", "task", "suite", "source", "perturb_category"))
        video_src = html.escape(video.relative_to(out.parent).as_posix() if video.is_relative_to(out.parent) else str(video))
        media = f'<video controls preload="metadata" src="{video_src}"></video>'
        for image_name in ("contact_sheet.jpg", "timeline.png"):
            image = episode.parent / image_name
            if image.is_file():
                image_src = html.escape(image.relative_to(out.parent).as_posix() if image.is_relative_to(out.parent) else str(image))
                media += f'<img loading="lazy" src="{image_src}" alt="{image_name}">'
        cards.append(f'<article class="card" data-tags="{tags}"><h3>{html.escape(str(metadata.get("task", "unknown")))}</h3>{media}<details><summary>metadata</summary><pre>{text}</pre></details></article>')
    page = """<!doctype html><meta charset="utf-8"><title>Axis Pi0.5 review</title>
<style>body{font:14px sans-serif;margin:2rem;background:#f5f5f5}.toolbar{position:sticky;top:0;background:white;padding:1rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}.card{background:white;padding:1rem;border-radius:8px}.card video{width:100%}pre{white-space:pre-wrap}</style>
<div class="toolbar"><label>filter <input id="filter" placeholder="step task suite source category"></label><span id="count"></span></div><main class="grid">__CARDS__</main>
<script>const input=document.querySelector('#filter'),cards=[...document.querySelectorAll('.card')],count=document.querySelector('#count');function update(){let q=input.value.toLowerCase();let n=0;cards.forEach(c=>{let ok=c.dataset.tags.toLowerCase().includes(q);c.hidden=!ok;if(ok)n++});count.textContent=n+' cases'}input.addEventListener('input',update);update();</script>""".replace("__CARDS__", "\n".join(cards))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"gallery={out} cards={len(cards)}")


if __name__ == "__main__":
    main()
