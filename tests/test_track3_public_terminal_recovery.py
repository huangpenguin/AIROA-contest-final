from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


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
    state = module.KeyboardState()
    action = module.key_to_action("w", state, module.RecorderConfig())
    module.validate_action(action)
    assert action.shape == (7,)
    assert action[0] > 0
    assert action[-1] == 1
    state.gripper = -1
    action = module.key_to_action("", state, module.RecorderConfig())
    module.validate_action(action)
    assert np.allclose(action[:6], 0)
    assert action[-1] == -1
