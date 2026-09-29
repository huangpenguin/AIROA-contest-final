from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_public_hdf5_mapping_covers_locked_40_tasks_without_spaces():
    module = _load("track3_public_terminal_recovery", "tools/axis_pi05/track3_public_terminal_recovery.py")
    paths = module.candidate_paths(module.load_t1_rows())
    assert len(paths) == 40
    assert len(set(paths)) == 40
    assert all(" " not in path for path in paths)
    assert sum(path.startswith("libero_10/") for path in paths) == 10
    assert sum(path.startswith("libero_goal/") for path in paths) == 10
    assert sum(path.startswith("libero_object/") for path in paths) == 10
    assert sum(path.startswith("libero_spatial/") for path in paths) == 10


def test_teleop_action_contract_and_cpu_self_test():
    module = _load("track3_public_teleop", "tools/axis_pi05/track3_public_teleop.py")
    machine = module.ActionStateMachine(config=module.RecorderConfig())
    assert module.RecorderConfig().hz == 20.0
    assert module.RecorderConfig().operator_hz == 8.0
    assert module.RecorderConfig().tick_ms() == 125
    assert "tick=${s.tick_ms||125}ms" in module.PREVIEW_HTML
    assert "setInterval(refresh,50)" in module.PREVIEW_HTML
    assert module.RecorderConfig().translation_step >= 0.30
    for _ in range(50):
        action = machine.next_action("")
        module.validate_action(action)
        assert np.allclose(action[:6], 0)
        assert action[-1] == -1
    machine.event("keydown", "h")
    assert machine.next_action("")[-1] == 1
    machine.event("keydown", "g")
    assert machine.next_action("")[-1] == -1
    machine.event("keydown", "p")
    assert np.allclose(machine.next_action("w")[:6], 0)
    machine.event("keydown", "p")
    machine.event("connect")
    machine.event("keydown", "w")
    assert machine.next_action()[0] < 0
    machine.event("keyup", "w")
    assert machine.next_action()[0] == 0
    machine.event("keydown", "s")
    assert machine.next_action()[0] > 0
    machine.event("keydown", "c")
    assert machine.state.camera == "wrist"
    machine.event("keydown", "z")
    assert machine.consume("reset")
    machine.event("keydown", "enter")
    assert machine.consume("save")
    machine.event("keydown", "x")
    assert np.allclose(machine.next_action()[:6], 0)
    with pytest.raises(ValueError):
        module.validate_action(np.zeros(7, dtype=np.float32))


def test_preview_is_loopback_only_and_acceptance_starts_at_zero(tmp_path):
    module = _load("track3_public_teleop_preview", "tools/axis_pi05/track3_public_teleop.py")
    module.OUTPUT_ROOT = tmp_path / "output"
    server = module.PreviewServer(module.ActionStateMachine(), port=0)
    try:
        assert server.httpd.server_address[0] == "127.0.0.1"
        assert not (module.OUTPUT_ROOT / "accepted").exists()
    finally:
        server.stop()


def test_replay_and_human_review_are_both_required_for_atomic_promotion(tmp_path):
    module = _load("track3_public_replay", "tools/axis_pi05/track3_public_replay.py")
    module.OUTPUT_ROOT = tmp_path / "output"
    pending = module.OUTPUT_ROOT / "pending" / "task_19_fixture"
    pending.mkdir(parents=True)
    metadata = {"status": "pending_replay", "task_index": 19}
    (pending / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (pending / "replay_summary.json").write_text(json.dumps({"status": "replay_failed", "passed_runs": 2}), encoding="utf-8")
    (pending / "human_review.json").write_text(json.dumps({"status": "approved"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="3/3"):
        module.promote_pending(pending)
    (pending / "replay_summary.json").write_text(json.dumps({"status": "replay_3_of_3_passed_human_review_pending", "passed_runs": 3}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="human review"):
        module.promote_pending(pending)
    review = {"status": "approved", "operator": "operator-local", "instruction_match": True, "objects_correct": True, "no_teleport_or_failure_trim": True, "front_wrist_confirmed": True, "approved_at_utc": "2026-09-03T00:00:00+00:00"}
    (pending / "human_review.json").write_text(json.dumps(review), encoding="utf-8")
    result = module.promote_pending(pending)
    assert result["status"] == "accepted"
    assert (module.OUTPUT_ROOT / "accepted" / "task_19_fixture" / "promotion.json").is_file()
    assert not pending.exists()


def test_r4_canary_allowlist_and_order_are_fixed():
    module = _load("track3_public_teleop_canary", "tools/axis_pi05/track3_public_teleop_canary.py")
    assert module.TASK_ORDER == (19, 10, 20)
    assert module.OUTPUT_ROOT.name == "track3_reverse_pilot_v2"
