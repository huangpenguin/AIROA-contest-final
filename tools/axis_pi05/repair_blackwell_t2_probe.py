#!/usr/bin/env python3
"""One-shot official Blackwell T2 probe repair + relaunch.

Copy this file to the official Jupyter home or NVMe code tree, then:

  /opt/dlami/nvme/parc20260911/work/trainvenv/bin/python repair_blackwell_t2_probe.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

os.environ.pop('LD_LIBRARY_PATH', None)

CODE = Path('/opt/dlami/nvme/parc20260911/code/PARC2026_final')
LOADER = CODE / 'third_party/axis_openpi/src/openpi/training/data_loader.py'
PY = Path('/opt/dlami/nvme/parc20260911/work/trainvenv/bin/python')
WRAP = Path('/opt/dlami/nvme/parc20260911/work/trainvenv_python.sh')
PARAMS = Path('/opt/dlami/nvme/parc20260911/work/r6_dual_l2_fresh500_r1/500/params')
RUNS = Path('/opt/dlami/nvme/parc20260911/runs')
RUN_DIR = RUNS / 't2_p30_r6_h10k_throughput_probe_r1'
LOG = RUNS / 't2_p30_r6_h10k_throughput_probe_r1.train.log'

HELPER = 'def _is_lerobot_v3_file_root(root):\n    return any(pathlib.Path(root).joinpath("data").glob("chunk-*/file-*.parquet"))\n'

CREATE_TORCH_DATASET = 'def create_torch_dataset(\n    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig\n) -> Dataset:\n    """Create a dataset for training."""\n    repo_id = data_config.repo_id\n    if repo_id is None:\n        raise ValueError("Repo ID is not set. Cannot create dataset.")\n    if repo_id == "fake":\n        return FakeDataset(model_config, num_samples=1024)\n    if data_config.local_repo_roots:\n        repo_ids = tuple(data_config.local_repo_ids) or (repo_id,)\n        roots = tuple(data_config.local_repo_roots)\n        if len(repo_ids) != len(roots):\n            raise ValueError("local_repo_ids and local_repo_roots must have equal length")\n        source_weights = tuple(data_config.local_sampling_weights) or tuple(1.0 for _ in roots)\n        if len(source_weights) != len(roots):\n            raise ValueError("local_sampling_weights and local_repo_roots must have equal length")\n        episode_selection: dict[str, list[int]] = {}\n        episode_weights: dict[tuple[str, int], float] = {}\n        episode_frame_counts: dict[tuple[str, int], int] = {}\n        train_manifest = os.environ.get("AXIS_PI05_TRAIN_MANIFEST")\n        sampling_manifest = data_config.local_sampling_manifest or os.environ.get("AXIS_PI05_SAMPLING_MANIFEST")\n        if train_manifest or sampling_manifest:\n            manifest_path = sampling_manifest or train_manifest\n            if sampling_manifest and pathlib.Path(manifest_path).suffix == ".jsonl":\n                # The official Phase 1 manifest is a complete frame catalog;\n                # the one-source dataset below already contains those frames.\n                _validate_official_sampling_jsonl(\n                    manifest_path,\n                    expected_count=273465,\n                    expected_revision=os.environ.get(\n                        "AXIS_PI05_CANONICAL_REVISION", "a4336d589d589045d1c56423ffdf3b88a0e19b1f"\n                    ),\n                )\n                sampling_manifest = None\n            else:\n                columns = ["source_repo", "source_episode_index"]\n                if sampling_manifest:\n                    columns.extend(["episode_sampling_probability", "observed_frame_count", "frame_weight"])\n                manifest = pd.read_parquet(manifest_path, columns=columns)\n                for source_repo, rows in manifest.groupby("source_repo", sort=False):\n                    episode_selection[str(source_repo)] = sorted(\n                        {int(value) for value in rows["source_episode_index"].tolist()}\n                    )\n                    if sampling_manifest:\n                        for row in rows.itertuples(index=False):\n                            key = (str(source_repo), int(row.source_episode_index))\n                            episode_weights[key] = float(row.frame_weight)\n                            episode_frame_counts[key] = int(row.observed_frame_count)\n        datasets = []\n        dataset_frame_weights = []\n        for local_repo_id, local_root in zip(repo_ids, roots, strict=True):\n            episodes = episode_selection.get(local_repo_id)\n            if train_manifest and not episodes:\n                raise ValueError(f"train manifest has no episodes for local source {local_repo_id}")\n            if local_repo_id == "physical-intelligence/libero":\n                dataset = AxisOfficialLiberoDataset(local_root, action_horizon, episodes, v3_keys=False)\n                dataset_tasks = dataset.tasks\n            elif local_repo_id in {"lerobot/libero_plus", "axis-pi05/track3_reverse_gold_v1"} or _is_lerobot_v3_file_root(\n                local_root\n            ):\n                dataset = AxisPlusV3Dataset(local_root, episodes, action_horizon)\n                dataset_tasks = dataset.tasks\n            else:\n                dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(local_repo_id, root=local_root)\n                dataset = AxisSelectedLeRobotDataset(\n                    local_repo_id,\n                    root=local_root,\n                    episodes=episodes,\n                    delta_timestamps={\n                        key: [t / dataset_meta.fps for t in range(action_horizon)]\n                        for key in data_config.action_sequence_keys\n                    },\n                )\n                dataset_tasks = dataset_meta.tasks\n            if sampling_manifest:\n                if isinstance(dataset, AxisPlusV3Dataset):\n                    weights = [\n                        episode_weights[(local_repo_id, int(episode))]\n                        for episode, _ in dataset._index\n                    ]\n                    observed = {\n                        int(episode): count\n                        for (source, episode), count in episode_frame_counts.items()\n                        if source == local_repo_id\n                    }\n                    actual = {int(episode): 0 for episode, _ in dataset._index}\n                    for episode, _ in dataset._index:\n                        actual[int(episode)] += 1\n                    for episode, count in actual.items():\n                        if count != observed[episode]:\n                            raise ValueError(f"manifest frame count mismatch for {local_repo_id}:{episode}: {count} != {observed[episode]}")\n                else:\n                    if dataset.episodes is None:\n                        raise ValueError("manifest-aware standard dataset requires explicit episode selection")\n                    weights = []\n                    for position, episode in enumerate(dataset.episodes):\n                        start = int(dataset.episode_data_index["from"][position].item())\n                        end = int(dataset.episode_data_index["to"][position].item())\n                        key = (local_repo_id, int(episode))\n                        expected = episode_frame_counts[key]\n                        if end - start != expected:\n                            raise ValueError(f"manifest frame count mismatch for {local_repo_id}:{episode}: {end - start} != {expected}")\n                        weights.extend([episode_weights[key]] * (end - start))\n                dataset_frame_weights.append(weights)\n            if data_config.prompt_from_task:\n                dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_tasks)])\n            datasets.append(dataset)\n        return WeightedConcatDataset(datasets, source_weights, dataset_frame_weights if sampling_manifest else None)\n\n    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)\n    dataset = lerobot_dataset.LeRobotDataset(\n        data_config.repo_id,\n        delta_timestamps={\n            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys\n        },\n    )\n\n    if data_config.prompt_from_task:\n        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])\n\n    return dataset\n\n\n'


def repair_loader() -> None:
    text = LOADER.read_text(encoding="utf-8")
    if "def _is_lerobot_v3_file_root" not in text:
        marker = "def create_torch_dataset("
        if marker not in text:
            raise SystemExit("create_torch_dataset not found")
        text = text.replace(marker, HELPER + "\n\n" + marker, 1)
        print("inserted _is_lerobot_v3_file_root")
    start = text.find("def create_torch_dataset(")
    end = text.find("def create_rlds_dataset(")
    if start < 0 or end < 0 or end <= start:
        raise SystemExit(f"cannot locate function bounds start={start} end={end}")
    text = text[:start] + CREATE_TORCH_DATASET + text[end:]
    compile(text, str(LOADER), "exec")
    backup = LOADER.with_suffix(".py.bak_t2_probe")
    if not backup.exists():
        shutil.copy2(LOADER, backup)
        print("backup", backup)
    LOADER.write_text(text, encoding="utf-8")
    print("rewrote create_torch_dataset", LOADER)


def archive_probe() -> None:
    fail = RUNS / "t2_p30_r6_h10k_throughput_probe_r1.failed_before_repair"
    pidp = Path(str(LOG) + ".pid")
    for src, dst in (
        (RUN_DIR, fail),
        (LOG, RUNS / "t2_p30_r6_h10k_throughput_probe_r1.failed_before_repair.train.log"),
        (pidp, RUNS / "t2_p30_r6_h10k_throughput_probe_r1.failed_before_repair.train.log.pid"),
    ):
        if src.exists():
            if dst.exists() and dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            src.rename(dst)
            print("archived", src.name)


def launch_probe() -> None:
    if RUN_DIR.exists():
        raise SystemExit(f"run dir still exists: {RUN_DIR}")
    cmd = [
        str(PY),
        "-m",
        "tools.axis_pi05.run_track12_long_horizon_lora",
        "--mode",
        "throughput-probe",
        "--run-id",
        "t2_p30_r6_h10k_throughput_probe_r1",
        "--run-dir",
        str(RUN_DIR),
        "--log",
        str(LOG),
        "--schedule-horizon-steps",
        "10000",
        "--stop-after-step",
        "100",
        "--python",
        str(WRAP),
        "--params",
        str(PARAMS),
        "--microbatch",
        "4",
        "--wandb",
        "disabled",
        "--execute",
        "--detach",
    ]
    result = subprocess.run(cmd, cwd=str(CODE), check=False)
    print("launch_rc", result.returncode)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    repair_loader()
    archive_probe()
    launch_probe()
