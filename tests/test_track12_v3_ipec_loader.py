from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training.data_loader import (
    AxisPlusV3Dataset,
    _is_lerobot_v3_file_root,
    _lerobot_v3_tasks,
    _lerobot_v3_video_dirs,
)


def _write_tasks_parquet(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    pq.write_table(
        pa.table({"task_index": [0], "task": ["put the mug on the plate"]}),
        root / "meta" / "tasks.parquet",
    )


def _write_frames(root: Path) -> None:
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [7, 7],
                "frame_index": [0, 1],
                "observation.state": [[0.0] * 8, [0.1] * 8],
                "action": [[0.0] * 7, [0.2] * 7],
                "task_index": [0, 0],
            }
        ),
        data_dir / "file-000.parquet",
    )
    (root / "videos" / "observation.images.image" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "observation.images.wrist_image" / "chunk-000").mkdir(parents=True)


class TestTrack12V3IpecLoader(unittest.TestCase):
    def test_v3_root_and_ipec_video_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ipec"
            _write_frames(root)
            self.assertTrue(_is_lerobot_v3_file_root(root))
            self.assertFalse(_is_lerobot_v3_file_root(Path(tmp) / "missing"))
            self.assertEqual(
                _lerobot_v3_video_dirs(root),
                ("observation.images.image", "observation.images.wrist_image"),
            )

    def test_v3_tasks_prefer_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ipec"
            _write_tasks_parquet(root)
            (root / "meta" / "tasks.jsonl").write_text(
                '{"task_index": 0, "task": "wrong jsonl"}\n', encoding="utf-8"
            )
            self.assertEqual(_lerobot_v3_tasks(root)[0], "put the mug on the plate")

    def test_axis_plus_v3_reads_ipec_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ipec"
            _write_tasks_parquet(root)
            _write_frames(root)
            dataset = AxisPlusV3Dataset(str(root), episodes=[7], action_horizon=2)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.tasks[0], "put the mug on the plate")
            self.assertEqual(dataset._front_dir, "observation.images.image")
            self.assertEqual(dataset._wrist_dir, "observation.images.wrist_image")

    def test_plus_video_dirs_still_win_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "plus"
            (root / "videos" / "observation.images.front").mkdir(parents=True)
            (root / "videos" / "observation.images.wrist").mkdir(parents=True)
            self.assertEqual(
                _lerobot_v3_video_dirs(root),
                ("observation.images.front", "observation.images.wrist"),
            )


if __name__ == "__main__":
    unittest.main()
