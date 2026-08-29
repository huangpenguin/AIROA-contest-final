"""Build a case-aligned, filterable local failure review page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import ROOT, resolve_path
from .eval_result import load_episode_rows


def _safe_path(value: str | None, result_dir: Path, gallery_dir: Path) -> str | None:
    if not value:
        return None
    source = (result_dir / value).resolve()
    if not source.exists():
        return None
    return os.path.relpath(source, gallery_dir.resolve()).replace(os.sep, "/")


def build_gallery(results: list[tuple[str, Path]], out: Path) -> dict[str, Any]:
    cards: list[dict[str, Any]] = []
    for candidate, result_dir in results:
        episodes_path = result_dir / "episodes.jsonl"
        if not episodes_path.is_file():
            continue
        for row in load_episode_rows(episodes_path):
            item = dict(row)
            item["candidate"] = candidate
            item["video_url"] = _safe_path(row.get("video_path"), result_dir, out)
            item["trajectory_url"] = _safe_path(row.get("trajectory_path"), result_dir, out)
            item["video_missing_reason"] = None if item["video_url"] else ("no video recorded" if not row.get("video_path") else "video file is missing or outside result directory")
            cards.append(item)
    cards.sort(key=lambda row: (str(row.get("task_name")), str(row.get("perturbation_family")), str(row.get("case_id")), str(row.get("candidate"))))
    out.mkdir(parents=True, exist_ok=True)
    # Keep JSON valid inside the script element and prevent a task instruction
    # containing ``</script>`` from terminating the data block.
    data = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Axis Pi0.5 model selection failure review</title>
<style>
body{{font:14px system-ui,sans-serif;margin:1.5rem;background:#f4f5f7;color:#17202a}}
.toolbar{{position:sticky;top:0;z-index:2;background:#fff;padding:1rem;border:1px solid #ddd;border-radius:8px;display:flex;gap:.7rem;flex-wrap:wrap}}
.toolbar input{{min-width:24rem;padding:.45rem}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:1rem;margin-top:1rem}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1rem}} .bad{{border-left:5px solid #c0392b}} .good{{border-left:5px solid #27ae60}}
video{{width:100%;background:#111;max-height:360px}} .tags{{display:flex;gap:.35rem;flex-wrap:wrap}} .tag{{background:#e9eef5;padding:.15rem .4rem;border-radius:4px;font-size:12px}}
pre{{white-space:pre-wrap;max-height:260px;overflow:auto;font-size:11px}} a{{color:#165dba}}
</style>
<div class="toolbar"><label>filter <input id="filter" placeholder="candidate task suite perturbation failure case id"></label><label><input id="failures" type="checkbox" checked> failures only</label><span id="count"></span></div>
<main id="grid" class="grid"></main>
<script id="episode-data" type="application/json">{data}</script>
<script>
const rows=JSON.parse(document.querySelector('#episode-data').textContent), grid=document.querySelector('#grid'), input=document.querySelector('#filter'), only=document.querySelector('#failures'), count=document.querySelector('#count');
function esc(x){{return String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function card(r){{let tags=[r.candidate,r.task_name,r.suite,r.perturbation_family,r.failure_reason,r.case_id].join(' ');let media=r.video_url?`<video controls preload="metadata" src="${{esc(r.video_url)}}"></video>`:`<p>Video unavailable: ${{esc(r.video_missing_reason)}}</p>`;let links=r.trajectory_url?`<a href="${{esc(r.trajectory_url)}}">trajectory</a>`:'';return `<article class="card ${{r.success?'good':'bad'}}" data-tags="${{esc(tags)}}"><h3>${{esc(r.task_name)}} — ${{r.success?'SUCCESS':'FAILURE'}}</h3>${{media}}<div class="tags"><span class="tag">candidate=${{esc(r.candidate)}}</span><span class="tag">suite=${{esc(r.suite)}}</span><span class="tag">case=${{esc(r.case_id)}}</span><span class="tag">init=${{esc(r.init_state_index)}}</span><span class="tag">env=${{esc(r.env_seed)}}</span><span class="tag">policy=${{esc(r.policy_seed)}}</span><span class="tag">perturb=${{esc(r.perturbation_family)}} ${{esc(JSON.stringify(r.perturbation_params||{{}}))}}</span><span class="tag">collision=${{esc(r.collision)}}</span><span class="tag">timeout=${{esc(r.timeout)}}</span><span class="tag">goal=${{esc(r.goal_reached)}}</span></div><p>${{links}}</p><details><summary>instruction / error</summary><p>${{esc(r.instruction)}}</p><pre>${{esc(JSON.stringify(r,null,2))}}</pre></details></article>`}}
function update(){{let q=input.value.toLowerCase(),visible=rows.filter(r=>(!only.checked||!r.success)&&(!q||[r.candidate,r.task_name,r.suite,r.perturbation_family,r.failure_reason,r.case_id].join(' ').toLowerCase().includes(q)));grid.innerHTML=visible.map(card).join('');count.textContent=visible.length+' cases'}} input.oninput=update;only.onchange=update;update();
</script>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    return {"out": str(out / "index.html"), "case_count": len(cards), "failure_count": sum(not row.get("success", False) for row in cards)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", action="append", required=True, help="candidate_id=result_dir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for value in args.result:
        if "=" not in value:
            raise SystemExit("--result must use candidate_id=result_dir")
        candidate, path = value.split("=", 1)
        results.append((candidate, resolve_path(path)))
    print(json.dumps(build_gallery(results, resolve_path(args.out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
