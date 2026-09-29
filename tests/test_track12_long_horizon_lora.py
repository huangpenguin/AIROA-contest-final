from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/axis_openpi/src"))

from tools.axis_pi05 import run_track12_long_horizon_lora as runner  # noqa: E402


EXPECTED_MANIFEST = "85700f9d85311cdadbacadcecf2b7c9b6617b77e03f51de42a81681ffc770af4"
EXPECTED_NORM = "39ff8a7cc2ff2a0be7d43659e708f9ec260686e876cf603c0460972b49746b41"


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fake_r6_params(tmp_path: Path) -> Path:
    params = tmp_path / "r6_dual_l2_fresh500_r1" / "500" / "params"
    params.mkdir(parents=True)
    (params / "weights.bin").write_bytes(b"r6-params-only")
    (params.parent / "train_state").mkdir()
    (params.parent / "train_state" / "opt.bin").write_bytes(b"do-not-load")
    return params


def _locked_files(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    mix = tmp_path / "T2-P30"
    manifest_bytes = b'{"mix_id":"T2-P30","locked":true}\n'
    norm_bytes = b'{"norm_stats":{"actions":{"q01":[-1],"q99":[1]}}}\n'
    sampling_bytes = b"PAR1fake"
    _write(mix / "manifest.json", manifest_bytes)
    _write(mix / "norm_stats.json", norm_bytes)
    _write(mix / "sampling_weights.parquet", sampling_bytes)
    _write(
        mix / "source_roots.json",
        json.dumps(
            {
                "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot": {"root": "/data/ipec10"},
                "lerobot/libero_plus": {"root": "/data/plus"},
            }
        ).encode(),
    )
    monkeypatch.setattr(runner, "DEFAULT_T2_MIX", mix)
    monkeypatch.setattr(runner, "EXPECTED_T2_MANIFEST_SHA", _sha(manifest_bytes))
    monkeypatch.setattr(runner, "EXPECTED_T2_NORM_SHA", _sha(norm_bytes))
    monkeypatch.setattr(runner, "LOCKED_T2_NORM", mix / "norm_stats.json")
    return {
        "mix": mix,
        "manifest": mix / "manifest.json",
        "norm": mix / "norm_stats.json",
        "sampling": mix / "sampling_weights.parquet",
        "source_roots": mix / "source_roots.json",
        "params": _fake_r6_params(tmp_path),
    }


def _argv(tmp_path: Path, files: dict[str, Path], **overrides) -> list[str]:
    values = {
        "mode": "print-env",
        "run_id": "t2_p30_r6_h10k_r1",
        "run_dir": tmp_path / "runs" / "t2_p30_r6_h10k_r1",
        "schedule": "10000",
        "stop": "10000",
    }
    values.update(overrides)
    return [
        "--mode",
        values["mode"],
        "--run-id",
        values["run_id"],
        "--run-dir",
        str(values["run_dir"]),
        "--schedule-horizon-steps",
        str(values["schedule"]),
        "--stop-after-step",
        str(values["stop"]),
        "--manifest",
        str(files["manifest"]),
        "--norm",
        str(files["norm"]),
        "--sampling",
        str(files["sampling"]),
        "--source-roots",
        str(files["source_roots"]),
        "--params",
        str(files["params"]),
    ]


def test_h10k_schedule_survives_stop100_probe(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    args = runner.parse_args(
        _argv(
            tmp_path,
            files,
            mode="throughput-probe",
            run_id="t2_p30_r6_h10k_throughput_probe_r1",
            run_dir=tmp_path / "runs" / "t2_p30_r6_h10k_throughput_probe_r1",
            schedule="10000",
            stop="100",
        )
    )
    resolved = runner.resolve_launch(args)
    contract = resolved["step_contract"]
    assert contract["schedule_horizon_steps"] == 10000
    assert contract["stop_after_step"] == 100
    assert contract["num_train_steps"] == 100
    assert contract["cosine_decay_steps"] == 10000
    assert contract["legacy_steps"] == 100
    assert resolved["identity"]["mode_family"] == "throughput_probe"
    assert resolved["env"]["AXIS_PI05_TRACK12_RESUME"] == "1"
    assert resolved["env"]["OPENPI_KEEP_STEPS"] == "1,10,100"


def test_stop_after_does_not_change_official_horizon(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    args = runner.parse_args(_argv(tmp_path, files))
    resolved = runner.resolve_launch(args)
    assert resolved["schedule_horizon_steps"] == 10000
    assert resolved["stop_after_step"] == 10000
    with pytest.raises(ValueError, match="complete cosine"):
        runner.resolve_launch(runner.parse_args(_argv(tmp_path, files, stop="3000")))


def test_h3k_complete_cosine_identity(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    args = runner.parse_args(
        _argv(
            tmp_path,
            files,
            run_id="t2_p30_r6_h3k_r1",
            run_dir=tmp_path / "runs" / "t2_p30_r6_h3k_r1",
            schedule="3000",
            stop="3000",
        )
    )
    resolved = runner.resolve_launch(args)
    assert resolved["save_steps"] == (1, 10, 100, 1000, 3000)
    assert resolved["step_contract"]["cosine_decay_steps"] == 3000


def test_horizon_manifest_norm_and_code_drift_fail_closed(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    args = runner.parse_args(_argv(tmp_path, files, mode="new"))
    resolved = runner.resolve_launch(args)
    run_dir = resolved["run_dir"]
    run_dir.mkdir(parents=True)
    runner.atomic_json(runner.identity_path(run_dir), resolved["identity"])

    drifted = dict(resolved["identity"])
    drifted["schedule_horizon_steps"] = 3000
    drifted = runner.identity_payload(drifted)
    with pytest.raises(ValueError, match="identity drift"):
        runner.compare_identity(resolved["identity"], drifted)

    files["manifest"].write_bytes(b'{"mix_id":"T2-P30","tampered":true}\n')
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        runner.resolve_launch(runner.parse_args(_argv(tmp_path, files, mode="resume")))


def test_canary100_resume_source_rejected(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    canary_dir = tmp_path / "t2_p30_canary100" / "100"
    (canary_dir / "params").mkdir(parents=True)
    (canary_dir / "train_state").mkdir()
    (canary_dir / "_CHECKPOINT_METADATA").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="canary100"):
        runner.resolve_launch(
            runner.parse_args(
                _argv(
                    tmp_path,
                    files,
                    mode="resume",
                    run_id="t2_p30_r6_h10k_r1",
                    run_dir=canary_dir,
                )
            )
        )
    with pytest.raises(ValueError, match="canary100"):
        runner.assert_params_are_r6_only(canary_dir / "params")
    with pytest.raises(ValueError, match="train_state"):
        runner.assert_params_are_r6_only(files["params"].parent / "train_state")


def test_existing_dir_requires_explicit_new_or_resume(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "t2_p30_r6_h10k_r1"
    args = runner.parse_args(_argv(tmp_path, files, mode="new", run_dir=run_dir))
    resolved = runner.resolve_launch(args)
    run_dir.mkdir(parents=True)
    runner.atomic_json(runner.identity_path(run_dir), resolved["identity"])
    with pytest.raises(FileExistsError, match="already exists"):
        runner.resolve_launch(runner.parse_args(_argv(tmp_path, files, mode="new", run_dir=run_dir)))
    resumed = runner.resolve_launch(runner.parse_args(_argv(tmp_path, files, mode="resume", run_dir=run_dir)))
    assert resumed["env"]["AXIS_PI05_TRACK12_RESUME"] == "1"
    assert resumed["identity"]["identity_sha256"] == resolved["identity"]["identity_sha256"]


def test_refuses_30k_and_wrong_horizon_pair(tmp_path, monkeypatch):
    files = _locked_files(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="3000 or 10000"):
        runner.validate_contract(
            mode="new",
            track="T2-P30",
            run_id="t2_p30_r6_h10k_r1",
            schedule_horizon_steps=30000,
            stop_after_step=30000,
            save_steps=(1,),
            manifest=files["manifest"],
            norm=files["norm"],
            params=files["params"],
        )


def test_openpi_config_splits_horizon_from_stop(monkeypatch):
    monkeypatch.setenv("AXIS_PI05_TRACK12_SCHEDULE_HORIZON_STEPS", "10000")
    monkeypatch.setenv("AXIS_PI05_TRACK12_STOP_AFTER_STEP", "100")
    monkeypatch.setenv("AXIS_PI05_TRACK12_STEPS", "100")
    monkeypatch.setenv("AXIS_PI05_TRACK12_WARMUP_STEPS", "10")
    import openpi.training.config as config_lib

    config_lib = importlib.reload(config_lib)
    cfg = config_lib.get_config("axis_pi05_track12_lora")
    assert cfg.num_train_steps == 100
    assert cfg.lr_schedule.decay_steps == 10000
    assert cfg.policy_metadata["schedule_horizon_steps"] == 10000
    assert cfg.policy_metadata["stop_after_step"] == 100
    assert cfg.resume is False


def test_canary_env_still_uses_steps_as_both_when_new_vars_absent(monkeypatch):
    monkeypatch.delenv("AXIS_PI05_TRACK12_SCHEDULE_HORIZON_STEPS", raising=False)
    monkeypatch.delenv("AXIS_PI05_TRACK12_STOP_AFTER_STEP", raising=False)
    monkeypatch.setenv("AXIS_PI05_TRACK12_STEPS", "100")
    import openpi.training.config as config_lib

    config_lib = importlib.reload(config_lib)
    cfg = config_lib.get_config("axis_pi05_track12_lora")
    assert cfg.num_train_steps == 100
    assert cfg.lr_schedule.decay_steps == 100


def test_estimate_fit_prefers_10k_when_time_allows():
    from datetime import datetime, timedelta, timezone

    jst = timezone(timedelta(hours=9))
    now = datetime(2026, 9, 12, 19, 0, tzinfo=jst)
    deadline = datetime(2026, 9, 17, 23, 59, tzinfo=jst)
    fast = runner.estimate_fit(seconds_per_optimizer_step=8.0, horizon_steps=10000, deadline=deadline, now=now)
    slow = runner.estimate_fit(seconds_per_optimizer_step=40.0, horizon_steps=10000, deadline=deadline, now=now)
    h3k = runner.estimate_fit(seconds_per_optimizer_step=40.0, horizon_steps=3000, deadline=deadline, now=now)
    assert fast["fits"] is True
    assert slow["fits"] is False
    assert h3k["fits"] is True


def test_locked_constants_match_plan():
    assert runner.EXPECTED_T2_MANIFEST_SHA == EXPECTED_MANIFEST
    assert runner.EXPECTED_T2_NORM_SHA == EXPECTED_NORM
    assert runner.OFFICIAL_RUN_IDS["t2_p30_r6_h10k_r1"]["schedule_horizon_steps"] == 10000
    assert runner.OFFICIAL_RUN_IDS["t2_p30_r6_h10k_r1"]["stop_after_step"] == 10000
