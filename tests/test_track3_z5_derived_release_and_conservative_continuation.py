from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.axis_pi05 import track3_z5_derived_release_and_conservative_continuation as tool


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    try:
        tool.strict_json(path)
    except tool.DuplicateKeyError:
        pass
    else:
        raise AssertionError("duplicate JSON key accepted")


def test_conservative_profile_is_exact_and_no_nominal_cli(tmp_path: Path) -> None:
    assert tool.CONSERVATIVE_PROFILE["name"] == "object_frame_conservative"
    assert tool.CONSERVATIVE_PROFILE["translation_clip_m"] == 0.025
    assert tool.CONSERVATIVE_PROFILE["rotation_clip_rad"] == 0.06
    assert tool.CONSERVATIVE_PROFILE["stage_budgets"] == [200, 160, 50, 200, 260, 160, 50, 120]
    result = subprocess.run([sys.executable, "-m", tool.MODULE, "--aa5-worker", "--profile", "object_frame_nominal", "--result-output", str(tmp_path / "result.json")], cwd=ROOT, capture_output=True, check=False)
    assert result.returncode == 2


def test_aa3_fixture_return_code_classification(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    command = [sys.executable, "-m", tool.MODULE, "--aa3-worker", "--status", "failed_gate", "--result-output", str(result_path)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_gate"
    assert payload["physics_steps"] == 0


def test_aa3_worker_exception_is_nonzero(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result = subprocess.run([sys.executable, "-m", tool.MODULE, "--aa3-worker", "--status", "worker_exception", "--result-output", str(result_path)], cwd=ROOT, capture_output=True, check=False)
    assert result.returncode == 1
    assert json.loads(result_path.read_text(encoding="utf-8"))["status"] == "worker_exception"


def test_identity_gate_rejects_drift() -> None:
    frozen = {"root_commit": "a", "tool_sha256": "b", "openpi_commit": "c", "libero_plus_porcelain": []}
    observed = {**frozen, "command": ["worker"], "cuda_visible_devices": "", "gpu_policy": "cpu_no_renderer", "source_porcelain": [], "libero_plus_porcelain": []}
    assert tool.identity_gate(observed, frozen, command=["worker"])["state"] == "passed"
    assert tool.identity_gate({**observed, "root_commit": "drift"}, frozen, command=["worker"])["state"] == "failed"

