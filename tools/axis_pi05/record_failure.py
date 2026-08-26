from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="make a browser/VLC-compatible front+wrist H.264 failure clip")
    ap.add_argument("--front", required=True, type=Path)
    ap.add_argument("--wrist", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("缺少 ffmpeg")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("perturb_category") is None:
        metadata["perturb_category"] = "unavailable"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw = "run={run} step={step} task={task} suite={suite} source={source} category={category}".format(
        run=metadata.get("run_id", "unknown"), step=metadata.get("internal_step", "unknown"), task=metadata.get("task", "unknown"), suite=metadata.get("suite", "unknown"), source=metadata.get("source", "unknown"), category=metadata.get("perturb_category", "unavailable"),
    ).replace(":", "\\:").replace("'", "\\'")
    command = ["ffmpeg", "-y", "-i", str(args.front), "-i", str(args.wrist), "-filter_complex", f"[0:v][1:v]hstack=inputs=2,drawtext=text='{draw}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55", "-r", "20", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
    subprocess.run(command, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt,r_frame_rate", "-of", "json", str(args.out)], capture_output=True, text=True, check=True)
    metadata["video_probe"] = json.loads(probe.stdout)
    (args.out.parent / "episode.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
