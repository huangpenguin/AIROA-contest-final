"""Run the frozen phase-1 rollout with the local LIBERO-plus asset bridge."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # LIBERO-plus keeps the Python package and asset tree under a different
    # root than the legacy bddl/object loaders expect.  Apply the same bridge
    # used by the validated asset probe before phase1_rollout imports the
    # environment manager.
    from libero.libero.envs import bddl_base_domain

    bddl_base_domain.DIR_PATH = ROOT / "LIBERO/libero/libero/envs"
    for name, module in list(sys.modules.items()):
        if name.startswith("libero.libero.envs.objects") and hasattr(module, "absolute_path"):
            module.absolute_path = ROOT / "LIBERO/libero/libero"

    from . import phase1_rollout

    phase1_rollout.main()


if __name__ == "__main__":
    main()
