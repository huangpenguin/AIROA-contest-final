from __future__ import annotations

import math
import unittest

from tools.axis_pi05.build_track12_mix import MIXES, eligible_rows, hierarchical_mass, make_weights


def _row(i: int, source: str, category: str, suite: str, task: str, frames: int = 10) -> dict:
    return {
        "source": source,
        "source_repo": f"org/{source}",
        "revision": "a" * 40,
        "source_episode_index": i,
        "suite": suite,
        "task": task,
        "task_index": 0,
        "instruction": task,
        "perturb_category": category,
        "perturb_label_quality": "recovered-coarse",
        "source_file": None,
        "frame_count": frames,
        "source_fps": 20,
        "state_dim": 8,
        "action_dim": 7,
        "cameras": ["front", "wrist"],
        "dataset_root": f"/tmp/{source}",
    }


class TestTrack12Mix(unittest.TestCase):
    def test_t1_mass_is_80_10_10(self):
        rows = []
        n = 0
        for suite in ("libero_spatial", "libero_object"):
            for task in ("t0", "t1"):
                for _ in range(6):
                    rows.append(_row(n, "libero_standard", "base", suite, task, frames=20 + n))
                    n += 1
        for category in ("lighting", "env", "language"):
            for suite in ("libero_spatial", "libero_object"):
                for task in ("t0", "t1"):
                    for _ in range(6):
                        rows.append(_row(n, "libero_plus", category, suite, task, frames=8 + (n % 5)))
                        n += 1
        selected = eligible_rows(rows, "T1-P20")
        self.assertTrue(all(row["perturb_category"] != "language" for row in selected))
        mass = hierarchical_mass(selected, "T1-P20")
        self.assertAlmostEqual(sum(mass.values()), 1.0, places=12)
        weights = make_weights(selected, "T1-P20")
        plus = {"lighting": 0.0, "env": 0.0}
        standard = 0.0
        for row in weights:
            if row["source"] == "libero_standard":
                standard += row["episode_sampling_probability"]
            else:
                plus[row["perturb_category"]] += row["episode_sampling_probability"]
        self.assertAlmostEqual(standard, MIXES["T1-P20"]["standard_mass"], places=10)
        self.assertAlmostEqual(plus["lighting"], 0.10, places=10)
        self.assertAlmostEqual(plus["env"], 0.10, places=10)
        self.assertTrue(math.isclose(sum(w["frame_weight"] * w["observed_frame_count"] for w in weights), 1.0, rel_tol=1e-10))

    def test_t2_mass_is_70_10_10_10(self):
        rows = []
        n = 0
        for suite in ("libero_spatial", "libero_object"):
            for task in ("t0", "t1"):
                for _ in range(6):
                    rows.append(_row(n, "libero_standard", "base", suite, task))
                    n += 1
        for category in ("camera_pose", "sensor_noise", "env"):
            for suite in ("libero_spatial", "libero_object"):
                for task in ("t0", "t1"):
                    for _ in range(6):
                        rows.append(_row(n, "libero_plus", category, suite, task, frames=12))
                        n += 1
        weights = make_weights(eligible_rows(rows, "T2-P30"), "T2-P30")
        observed = {"standard": 0.0, "camera_pose": 0.0, "sensor_noise": 0.0, "env": 0.0}
        for row in weights:
            key = "standard" if row["source"] == "libero_standard" else row["perturb_category"]
            observed[key] += row["episode_sampling_probability"]
        self.assertAlmostEqual(observed["standard"], 0.70, places=10)
        self.assertAlmostEqual(observed["camera_pose"], 0.10, places=10)
        self.assertAlmostEqual(observed["sensor_noise"], 0.10, places=10)
        self.assertAlmostEqual(observed["env"], 0.10, places=10)
