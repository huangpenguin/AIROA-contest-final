from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.axis_pi05.launch_official_blackwell import (  # noqa: E402
    build_environment,
    load_launch_config,
    parse_args,
    resolve_launch,
    wandb_credentials_present,
)

CONFIG = ROOT / "configs/axis_pi05/official_blackwell_launch.yaml"


def _source_roots(tmp_path: Path) -> Path:
    path = tmp_path / "source_roots.json"
    path.write_text(
        json.dumps(
            {
                "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot": {"root": "/data/ipec10"},
                "lerobot/libero_plus": {"root": "/data/plus"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_yaml_recipe_matches_canary():
    recipe = load_launch_config(CONFIG)
    assert recipe["wandb"]["project"] == "parc2026-axis-pi05"
    assert recipe["wandb"]["default_mode"] == "disabled"
    assert recipe["shared"]["microbatch"] == 4
    assert recipe["shared"]["accumulation"] == 16
    assert float(recipe["tracks"]["T1-P20"]["peak_lr"]) == 3e-7
    assert float(recipe["tracks"]["T2-P30"]["peak_lr"]) == 3e-7
    assert float(recipe["tracks"]["T3"]["peak_lr"]) == 1e-6
    assert recipe["tracks"]["T3"]["warmup_steps"] == 100
    assert recipe["tracks"]["T1-P20"]["warmup_steps"] == 10


def test_t2_env_uses_t2_mix_and_disabled_wandb(tmp_path):
    recipe = load_launch_config(CONFIG)
    roots = _source_roots(tmp_path)
    env = build_environment(
        track="T2-P30",
        recipe=recipe,
        steps=250,
        run_dir=tmp_path / "runs" / "t2",
        run_id="t2_p30_official250",
        params=tmp_path / "params",
        norm=tmp_path / "norm_stats.json",
        sampling=tmp_path / "sampling_weights.parquet",
        python=tmp_path / "python",
        wandb_mode="disabled",
        wandb_project="parc2026-axis-pi05",
        source_roots=roots,
        forward_root=None,
        reverse_root=None,
        microbatch=4,
        accumulation=16,
    )
    assert env["AXIS_PI05_WANDB_MODE"] == "disabled"
    assert env["WANDB_MODE"] == "disabled"
    assert env["AXIS_PI05_TRACK12_ASSET_ID"] == "T2-P30"
    assert env["AXIS_PI05_TRACK12_STEPS"] == "250"
    assert env["AXIS_PI05_TRACK12_PEAK_LR"] == "3e-7"
    assert env["AXIS_PI05_TRACK12_WARMUP_STEPS"] == "10"
    assert "T2-P30" in env["AXIS_PI05_TRACK12_SAMPLING_MANIFEST"] or env["AXIS_PI05_TRACK12_SAMPLING_MANIFEST"].endswith("sampling_weights.parquet")
    assert env["WANDB_PROJECT"] == "parc2026-axis-pi05"


def test_t3_env_is_not_track12_lr(tmp_path):
    recipe = load_launch_config(CONFIG)
    env = build_environment(
        track="T3",
        recipe=recipe,
        steps=100,
        run_dir=tmp_path / "runs" / "t3",
        run_id="t3_four_gold_official100",
        params=tmp_path / "params",
        norm=tmp_path / "norm_stats.json",
        sampling=tmp_path / "sampling_weights.parquet",
        python=tmp_path / "python",
        wandb_mode="disabled",
        wandb_project="parc2026-axis-pi05",
        source_roots=None,
        forward_root=tmp_path / "plus",
        reverse_root=tmp_path / "reverse",
        microbatch=4,
        accumulation=16,
    )
    assert env["AXIS_PI05_TRACK3_PEAK_LR"] == "1e-6"
    assert env["AXIS_PI05_TRACK3_WARMUP_STEPS"] == "100"
    assert env["AXIS_PI05_TRACK3_SOURCE_WEIGHTS"] == "0.8,0.2"
    assert "AXIS_PI05_TRACK12_PEAK_LR" not in env


def test_online_wandb_without_credentials_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("tools.axis_pi05.launch_official_blackwell.wandb_credentials_present", lambda env: False)
    recipe = load_launch_config(CONFIG)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        build_environment(
            track="T1-P20",
            recipe=recipe,
            steps=100,
            run_dir=tmp_path / "run",
            run_id="t1",
            params=tmp_path / "params",
            norm=tmp_path / "norm.json",
            sampling=tmp_path / "w.parquet",
            python=tmp_path / "python",
            wandb_mode="online",
            wandb_project="parc2026-axis-pi05",
            source_roots=_source_roots(tmp_path),
            forward_root=None,
            reverse_root=None,
            microbatch=4,
            accumulation=16,
        )


def test_print_env_keeps_log_outside_run(tmp_path):
    args = parse_args(
        [
            "--track",
            "T1-P20",
            "--run-dir",
            str(tmp_path / "runs" / "t1_official"),
            "--source-roots",
            str(_source_roots(tmp_path)),
            "--norm",
            str(tmp_path / "norm_stats.json"),
            "--sampling",
            str(tmp_path / "sampling_weights.parquet"),
            "--params",
            str(tmp_path / "params"),
            "--print-env",
        ]
    )
    resolved = resolve_launch(args)
    assert resolved["log"] == tmp_path / "runs" / "t1_official.train.log"
    assert resolved["log"] != resolved["run_dir"]
    assert resolved["wandb_mode"] == "disabled"


def test_fresh_launcher_refuses_post_canary_long_training(tmp_path):
    args = parse_args(
        [
            "--track",
            "T3",
            "--steps",
            "300",
            "--run-dir",
            str(tmp_path / "runs" / "t3_official300"),
            "--forward-root",
            str(tmp_path / "plus"),
            "--reverse-root",
            str(tmp_path / "reverse"),
        ]
    )
    with pytest.raises(ValueError, match="run_track12_long_horizon_lora"):
        resolve_launch(args)


@pytest.mark.parametrize("steps", [0, -1])
def test_fresh_launcher_refuses_nonpositive_steps(tmp_path, steps):
    args = parse_args(
        [
            "--track",
            "T1-P20",
            "--steps",
            str(steps),
            "--run-dir",
            str(tmp_path / "runs" / "invalid"),
            "--source-roots",
            str(_source_roots(tmp_path)),
        ]
    )
    with pytest.raises(ValueError, match="steps must be positive"):
        resolve_launch(args)


def test_wandb_api_key_counts_as_credentials(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    assert wandb_credentials_present(os.environ)
