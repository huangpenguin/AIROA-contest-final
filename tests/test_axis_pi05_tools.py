from __future__ import annotations

import sys
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.axis_pi05.build_mix import make_weights, split_rows  # noqa: E402
from tools.axis_pi05.check import run as static_check  # noqa: E402
from tools.axis_pi05.contract import validate_catalog, validate_sample  # noqa: E402
from tools.axis_pi05.common import load_yaml  # noqa: E402
from tools.axis_pi05.lock import load_and_validate  # noqa: E402
from tools.axis_pi05.param_audit import audit  # noqa: E402
from tools.axis_pi05.phase0_audit import canonical_payload_manifest, data_source_gate  # noqa: E402


def _row(i: int, source: str = "libero_standard", category: str = "base") -> dict:
    return {
        "source": source, "source_repo": f"org/{source}", "revision": "a" * 40,
        "source_episode_index": i, "suite": "libero_spatial" if source == "libero_standard" else "plus",
        "task": f"task-{i % 2}", "task_index": i % 2, "instruction": f"task-{i % 2}",
        "perturb_category": category, "perturb_label_quality": "exact-source", "source_file": f"f-{i // 2}" if source == "libero_plus" else None,
        "frame_count": 10, "source_fps": 20, "state_dim": 8, "action_dim": 7, "cameras": ["front", "wrist"],
    }


class TestAxisPi05Tools(unittest.TestCase):
    def test_lock_has_full_revisions(self):
        load_and_validate(ROOT / "configs/axis_pi05/data_sources.lock.yaml")

    def test_lock_records_effective_openpi_implementation(self):
        openpi = load_yaml(ROOT / "configs/axis_pi05/data_sources.lock.yaml")["openpi"]
        self.assertEqual(openpi["upstream_base_revision"], openpi["revision"])
        self.assertRegex(openpi["implementation_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(openpi["implementation_git_tree"], r"^[0-9a-f]{40}$")
        self.assertRegex(openpi["implementation_patch_sha256"], r"^[0-9a-f]{64}$")


    def test_canonical_source_supersedes_non_equivalent_ipec(self):
        self.assertTrue(data_source_gate(canonical_ready=True, alternative_equivalent=False))
        self.assertTrue(data_source_gate(canonical_ready=False, alternative_equivalent=True))
        self.assertFalse(data_source_gate(canonical_ready=False, alternative_equivalent=False))


    def test_payload_manifest_excludes_unlisted_cache_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta").mkdir()
            (root / "README.md").write_bytes(b"readme")
            (root / "meta/info.json").write_bytes(b"info")
            (root / ".axis_pi05_revision").write_bytes(b"revision")
            (root / ".cache/huggingface").mkdir(parents=True)
            (root / ".cache/huggingface/cache.bin").write_bytes(b"cache")
            manifest = canonical_payload_manifest(root, ["README.md", "meta/info.json"], expected_count=2)
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual([entry["path"] for entry in manifest["entries"]], ["README.md", "meta/info.json"])


    def test_reproduction_and_competition_lines_have_separate_fps(self):
        reproduction = load_yaml(ROOT / "configs/axis_pi05/standard_libero_30k.yaml")
        competition = load_yaml(ROOT / "configs/axis_pi05/libero_plus_track3_20hz.yaml")
        self.assertEqual(reproduction["canonical_dataset"]["authoritative_fps"], 10)
        self.assertNotIn("plan_declared_fps", reproduction["canonical_dataset"])
        self.assertEqual(competition["data"]["source_fps"], 20)
        self.assertFalse(competition["normalization"]["reuse_axis_norm_stats"])


    def test_sample_contract_and_catalog(self):
        rows = [_row(i) for i in range(6)]
        self.assertEqual(validate_sample(rows[0]), [])
        self.assertTrue(validate_catalog(rows)["ok"])


    def test_split_is_deterministic_and_disjoint(self):
        rows = [_row(i) for i in range(12)]
        train_a, val_a, _ = split_rows(rows, 1000, 0.1)
        train_b, val_b, _ = split_rows(rows, 1000, 0.1)
        self.assertEqual([r["source_episode_index"] for r in val_a], [r["source_episode_index"] for r in val_b])
        self.assertFalse({r["source_episode_index"] for r in train_a} & {r["source_episode_index"] for r in val_a})
        self.assertEqual(len(train_a) + len(val_a), len(rows))


    def test_sampling_weights_sum_to_one(self):
        weights = make_weights([_row(i) for i in range(20)], "M0")
        self.assertTrue(math.isclose(sum(row["sampling_weight"] for row in weights), 1.0, rel_tol=1e-9))


    def test_parameter_audit_scopes_are_nested(self):
        entries = [
            {"path": "PaliGemma/llm/lora_a", "shape": [2, 2]},
            {"path": "action_expert/lora_b", "shape": [2, 2]},
            {"path": "action_in_proj/kernel", "shape": [2, 2]},
            {"path": "action_out_proj/kernel", "shape": [2, 2]},
            {"path": "time_mlp_in/kernel", "shape": [2, 2]},
            {"path": "vision/backbone/kernel", "shape": [2, 2]},
        ]
        config = {"arms": {
            "F0": {"trainable_regex": [r"lora"], "frozen_regex": [r"vision", r"action_(in|out)_proj|time_mlp"]},
            "F1": {"trainable_regex": [r"lora", r"action_(in|out)_proj"], "frozen_regex": [r"vision", r"time_mlp"]},
            "F2": {"trainable_regex": [r"lora", r"action_(in|out)_proj|time_mlp"], "frozen_regex": [r"vision"]},
        }}
        counts = [audit(entries, scope, config)["trainable_parameter_count"] for scope in ("F0", "F1", "F2")]
        self.assertEqual(counts, [8, 16, 20])


    def test_static_gates(self):
        self.assertEqual(static_check(ROOT), [])
