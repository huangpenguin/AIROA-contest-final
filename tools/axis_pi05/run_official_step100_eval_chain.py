#!/usr/bin/env python3
"""Wait for T3 N5, then run T1 and T2 C20 evals on GPU3.

Downloads T1/T2 checkpoints from private HF if they are not local yet.
Does not start training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .common import ROOT
from . import download_official_canary100 as dl
from . import run_track12_step100_paired_eval as t12
from . import run_track3_n5_official_paired_eval as n5


T3_SUMMARY = n5.DEFAULT_EVIDENCE / "paired_summary.json"
T3_CURSOR = n5.DEFAULT_EVIDENCE / "cursor.json"


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_checkpoint(track: str) -> dict:
    spec = dl.load_contract()["tracks"][track]
    local = ROOT / spec["local_dir"]
    try:
        inventory = dl.verify_checkpoint(local)
        return {"track": track, "status": "local", "inventory": inventory}
    except (FileNotFoundError, ValueError):
        pass
    print(json.dumps({"download": track, "repo_id": spec["repo_id"]}), flush=True)
    return {"track": track, "status": "downloaded", **dl.download_track(track)}


def wait_t3(*, timeout_sec: int = 16 * 3600) -> dict:
    started = time.time()
    while time.time() - started < timeout_sec:
        summary = _load(T3_SUMMARY)
        if summary is not None:
            return summary
        cursor = _load(T3_CURSOR) or {}
        print(json.dumps({"waiting_for_t3": True, "cursor": cursor.get("status"), "elapsed_sec": int(time.time() - started)}), flush=True)
        time.sleep(60)
    raise TimeoutError("T3 N5 eval did not finish")


def main() -> int:
    print(json.dumps({"schema_version": "official_step100_eval_chain_v1", "training_steps": 0, "gpu": 3}), flush=True)
    downloads = []
    for track in ("T1", "T2"):
        try:
            downloads.append(ensure_checkpoint(track))
        except Exception as exc:
            downloads.append({"track": track, "status": "blocked_missing_checkpoint", "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"downloads": [{"track": row["track"], "status": row["status"]} for row in downloads]}), flush=True)
    t3 = wait_t3()
    print(json.dumps({"t3_status": t3.get("status"), "continue_to_step300_authorized": t3.get("continue_to_step300_authorized")}), flush=True)
    results = {"T3": {"status": t3.get("status"), "continue_to_step300_authorized": t3.get("continue_to_step300_authorized")}}
    for track in ("T1", "T2"):
        local = ROOT / dl.load_contract()["tracks"][track]["local_dir"]
        if not (local / "params").is_dir():
            results[track] = {"status": "blocked_missing_checkpoint"}
            print(json.dumps(results[track] | {"track": track}), flush=True)
            continue
        print(json.dumps({"starting": track, "gpu": 3}), flush=True)
        results[track] = t12.run_track(track, gpu=3)
        print(json.dumps({"track": track, "status": results[track].get("status"), "continue_to_step250_authorized": results[track].get("continue_to_step250_authorized")}), flush=True)
    print(json.dumps({"chain_complete": True, "training_steps": 0, "results": {name: {"status": row.get("status"), "continue_to_step300_authorized": row.get("continue_to_step300_authorized"), "continue_to_step250_authorized": row.get("continue_to_step250_authorized")} for name, row in results.items()}}, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
