"""Merge disjoint frozen-manifest result shards into one result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ROOT, resolve_path, sha256_file, write_json
from .eval_manifest import load_locked_panel
from .eval_result import load_episode_rows, write_result_bundle
from .eval_runner import checkpoint_identity, ensure_protocol


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def merge_shards(
    *,
    manifest: Path,
    checkpoint: Path,
    candidate: str,
    protocol_path: Path,
    shard_dirs: list[Path],
    output_dir: Path,
    root: Path = ROOT,
) -> dict[str, Any]:
    rows = load_locked_panel(manifest, manifest.parent / "manifest.lock.json")
    manifest_hash = sha256_file(manifest)
    protocol = ensure_protocol(protocol_path, manifest_hash)
    protocol_hash = sha256_file(protocol_path)
    identity = checkpoint_identity(checkpoint)
    expected_ids = [str(row["case_id"]) for row in rows]
    expected_set = set(expected_ids)
    collected: dict[str, dict[str, Any]] = {}

    for shard_dir in shard_dirs:
        metadata_path = shard_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"shard metadata missing: {metadata_path}")
        metadata = _read_json(metadata_path)
        if metadata.get("candidate") not in (None, candidate):
            raise ValueError(f"candidate mismatch in shard: {metadata_path}")
        if metadata.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"manifest hash mismatch in shard: {metadata_path}")
        if metadata.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"protocol hash mismatch in shard: {metadata_path}")
        for episode in load_episode_rows(shard_dir / "episodes.jsonl"):
            if episode.get("candidate") != candidate:
                raise ValueError(f"candidate mismatch in episode shard: {shard_dir}")
            case_id = str(episode["case_id"])
            if case_id not in expected_set:
                raise ValueError(f"unknown case in shard {shard_dir}: {case_id}")
            if case_id in collected:
                raise ValueError(f"duplicate case across shards: {case_id}")
            normalized = dict(episode)
            try:
                shard_prefix = shard_dir.resolve().relative_to(output_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"shard must be under output directory for media links: {shard_dir}"
                ) from exc
            for key in ("video_path", "trajectory_path"):
                value = normalized.get(key)
                if value:
                    normalized[key] = (shard_prefix / value).as_posix()
            collected[case_id] = normalized

    missing = [case_id for case_id in expected_ids if case_id not in collected]
    if missing:
        raise ValueError(f"missing {len(missing)} cases while merging shards")
    episodes = [collected[case_id] for case_id in expected_ids]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = write_result_bundle(
        output_dir,
        episodes,
        expected_case_ids=expected_ids,
        summary_kwargs={
            "candidate": candidate,
            "panel": manifest.stem,
            "checkpoint": identity["path"],
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "manifest_sha256": manifest_hash,
            "protocol_sha256": protocol_hash,
            "official_result": False,
            "result_scope": "local_model_selection",
            "bootstrap_seed": protocol["bootstrap_seed"],
            "bootstrap_draws": protocol["bootstrap_draws"],
        },
        run_metadata={
            "schema_version": 2,
            "candidate": candidate,
            "panel": manifest.stem,
            "server": {"external": False, "sharded": True, "shard_count": len(shard_dirs)},
            "checkpoint": identity,
            "manifest_sha256": manifest_hash,
            "protocol_sha256": protocol_hash,
            "official_result": False,
        },
    )
    write_json(output_dir / "merge_metadata.json", {
        "schema_version": 1,
        "candidate": candidate,
        "manifest_sha256": manifest_hash,
        "protocol_sha256": protocol_hash,
        "shards": [str(path) for path in shard_dirs],
        "case_count": len(episodes),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = merge_shards(
        manifest=resolve_path(args.manifest),
        checkpoint=resolve_path(args.checkpoint),
        candidate=args.candidate,
        protocol_path=resolve_path(args.protocol),
        shard_dirs=[resolve_path(path) for path in args.shard],
        output_dir=resolve_path(args.out),
        root=resolve_path(args.root),
    )
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
