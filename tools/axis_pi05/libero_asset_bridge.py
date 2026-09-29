"""Runtime-only bridge from LIBERO-plus task code to vendored LIBERO assets."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def activate(root: Path) -> dict[str, Any]:
    """Install the in-process asset bridge and return immutable metadata."""

    root = Path(root).resolve()
    source_package = root / "LIBERO/libero/libero"
    source_assets = source_package / "assets"
    if not source_assets.is_dir():
        raise FileNotFoundError(f"validated LIBERO assets are missing: {source_assets}")

    from libero.libero.envs import bddl_base_domain
    from libero.libero.envs.arenas import style

    bddl_base_domain.DIR_PATH = source_package / "envs"
    for name, module in list(sys.modules.items()):
        if name.startswith("libero.libero.envs.objects") and hasattr(module, "absolute_path"):
            module.absolute_path = source_package

    texture_root = source_assets / "textures"
    fallbacks = {"floor": "light-gray-floor-tile.png", "wall": "light-gray-plaster.png"}
    for texture_type, mapping in style.STYLE_MAPPING.items():
        fallback = fallbacks[texture_type]
        if not (texture_root / fallback).is_file():
            raise FileNotFoundError(f"validated LIBERO fallback texture is missing: {texture_root / fallback}")
        for style_name, filename in list(mapping.items()):
            if not (texture_root / filename).is_file():
                mapping[style_name] = fallback

    scene_fallbacks = {
        "floor": "scenes/libero_floor_base_style.xml",
        "table": "scenes/libero_tabletop_base_style.xml",
        "kitchen": "scenes/libero_kitchen_tabletop_base_style.xml",
        "living_room": "scenes/libero_living_room_tabletop_base_style.xml",
        "coffee_table": "scenes/libero_coffee_table_base_style.xml",
        "study": "scenes/libero_study_base_style.xml",
    }
    original_init = bddl_base_domain.BDDLBaseDomain.__init__
    if not getattr(original_init, "_axis_pi05_asset_bridge", False):
        def bridged_init(self, bddl_file_name, *args, **kwargs):
            scene_xml = kwargs.get("scene_xml")
            arena_type = kwargs.get("arena_type", "table")
            if scene_xml is not None and not (source_assets / scene_xml).is_file():
                kwargs["scene_xml"] = scene_fallbacks.get(arena_type, scene_fallbacks["table"])
            return original_init(self, bddl_file_name, *args, **kwargs)

        bridged_init._axis_pi05_asset_bridge = True
        bddl_base_domain.BDDLBaseDomain.__init__ = bridged_init

    return {
        "bridge": "axis_pi05_libero_asset_bridge_v1",
        "source_package": str(source_package),
        "source_assets": str(source_assets),
        "source_assets_realpath": str(source_assets.resolve()),
        "scene_fallbacks": scene_fallbacks,
    }
