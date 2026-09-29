"""Build the post-E2 Track3 official-transfer manifest without transferring data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, canonical_json, sha256_file


EVIDENCE = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval"
OUTPUTS = ROOT / "outputs/axis_pi05/track3_four_gold_mix_v1/q7_gpu3_frozen_eval"
Q6_MANIFEST = ROOT / "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q6_official_transfer_manifest.json"
Q7_MANIFEST = EVIDENCE / "q7_official_transfer_manifest.json"
Q7_SIDECAR = EVIDENCE / "q7_official_transfer_manifest.json.sha256"


def _entry(path: Path, role: str) -> dict[str, object]:
    relative = path.relative_to(ROOT).as_posix()
    return {"bytes": path.stat().st_size, "role": role, "sha256": sha256_file(path), "source_path": relative, "target_path": relative}


def build() -> dict[str, object]:
    q6 = json.loads(Q6_MANIFEST.read_text(encoding="utf-8"))
    files = list(q6["files"])
    for item in files:
        source = ROOT / item["source_path"]
        if not source.is_file() or sha256_file(source) != item["sha256"]:
            raise RuntimeError(f"Q6 manifest file mismatch: {source}")

    sources = [
        ROOT / "docs/AXIS_PI05_TRACK3_Q5_STEP10_FROZEN_EVAL_TO_OFFICIAL_TRAINING_HANDOFF_20260910.md",
        ROOT / "tools/axis_pi05/run_track3_q5_step10_frozen_eval.py",
        ROOT / "tools/axis_pi05/libero_asset_bridge.py",
        ROOT / "tools/axis_pi05/checkpoint_http_server.py",
        ROOT / "tools/axis_pi05/build_track3_q7_transfer_manifest.py",
        ROOT / "tests/test_track3_q5_step10_frozen_eval.py",
    ]
    additions = [_entry(path, "q7_frozen_eval_source") for path in sources]
    additions.extend(_entry(path, "q7_frozen_eval_evidence") for path in sorted(EVIDENCE.rglob("*")) if path.is_file() and path not in {Q7_MANIFEST, Q7_SIDECAR} and path.name != "failure.json")
    additions.extend(_entry(path, "q7_frozen_eval_output") for path in sorted(OUTPUTS.rglob("*")) if path.is_file() and ".previous_" not in path.name and ".partial" not in path.name)

    seen: set[str] = set()
    all_files = []
    for item in files + additions:
        if item["source_path"] not in seen:
            all_files.append(item)
            seen.add(item["source_path"])

    paired = json.loads((EVIDENCE / "paired_summary.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "track3_q7_official_transfer_manifest_v1",
        "status": "ready_for_official_transfer_after_track12_release",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transfer_performed": False,
        "official_training_started": False,
        "manifest_sha256_sidecar": "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/q7_official_transfer_manifest.json.sha256",
        "self_hash_rule": "SHA256 is computed over the exact manifest JSON bytes; the manifest is not included in its own files list.",
        "contract": {
            "initialization": "R6 step-500 params-only",
            "q5_step10_diagnostic_only": True,
            "q5_checkpoint_tree_transferred": False,
            "q5_step10_transferred": False,
            "fresh_optimizer_required_for_official_step1": True,
            "track12_server_stop_or_extension": False,
            "track3_official_training_started": False,
        },
        "base_q6_manifest": {"path": "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q6_official_transfer_manifest.json", "sha256": sha256_file(Q6_MANIFEST), "file_count": len(files), "total_bytes": sum(int(item["bytes"]) for item in files)},
        "evaluation": {
            "e1_canary": {"path": "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/e1_canary_summary.json", "sha256": sha256_file(EVIDENCE / "e1_canary_summary.json"), "status": "passed", "cases": 4},
            "e2_paired": {"path": "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/paired_summary.json", "sha256": sha256_file(EVIDENCE / "paired_summary.json"), "status": paired["status"], "cases": paired["case_count"], "gates": paired["gates"]},
            "diagnostic_outputs": "outputs/axis_pi05/track3_four_gold_mix_v1/q7_gpu3_frozen_eval",
            "official_result": False,
        },
        "source": {
            "root_head": "36e8f269ecd30d4a2a6e555e67bd7315011e5dbf",
            "nested_openpi_head": "b58aff0750f7a9ee07f3feb5df951a20ad2c76e1",
            "q6_semantic_signoff": "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q6_semantic_signoff.json",
            "q6_semantic_signoff_sha256": "3693d049c39ab98905571c38504010b4b33e306d310d20f8c04ff051b561cc37",
            "r6_params_tree_sha256": "d9b7e0d15781060fccfd61dc1890ad98993f0224e4aebba1dd6658d7c011931f",
        },
        "excluded": [
            {"path": "outputs/axis_pi05/track3_four_gold_mix_v1/q5_local_gpu3", "reason": "Q5 checkpoint/train-state is diagnostic-only and is not an official initializer."},
            {"path": "outputs/axis_pi05/track3_four_gold_mix_v1/q7_gpu3_frozen_eval/**/.previous_*", "reason": "Historical retry evidence is preserved locally and is not part of the stable transfer payload."},
            {"path": "data/reports/axis_libero_30k/track3_four_gold_mix_v1_20260910/q7_gpu3_frozen_eval/failure.json", "reason": "Historical pre-bridge retry failure is preserved locally; the final q7 attempt and paired summary are passed."},
            {"path": "official Track1/Track2 output, dataset cache, PID/cursor/heartbeat and ports", "reason": "Active official state is out of scope and must not be touched."},
        ],
        "files": all_files,
        "total_files": len(all_files),
        "total_bytes": sum(int(item["bytes"]) for item in all_files),
    }
    payload_bytes = canonical_json(payload)
    Q7_MANIFEST.write_bytes(payload_bytes)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    Q7_SIDECAR.write_text(f"{digest}  {Q7_MANIFEST.name}\n", encoding="utf-8")
    return {"path": str(Q7_MANIFEST), "sha256": digest, "files": len(all_files), "bytes": payload["total_bytes"], "paired_status": paired["status"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(build(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
