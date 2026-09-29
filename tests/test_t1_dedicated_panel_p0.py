from __future__ import annotations

from pathlib import Path

from tools.axis_pi05.run_t1_dedicated_panel_p0 import (
    CSV_FAMILY,
    SUFFIX_RE,
    enumerate_cases,
    inspect_problem_assets,
)


ROOT = Path(__file__).resolve().parents[1]


def test_t1_example_suite_maps_two_families_without_using_empty_perturbations():
    cases = enumerate_cases()
    assert [case["perturbation_family"] for case in cases] == ["env", "env", "lighting", "lighting"]
    assert {case["csv_category"] for case in cases} == set(CSV_FAMILY)
    assert {case["env_seed"] for case in cases} == {1000, 1001, 1002, 1003}
    assert all(case["init_state_index"] == 0 for case in cases)
    assert all(case["max_steps"] == 300 for case in cases)


def test_suffix_strip_finds_standard_libero_base_bddl():
    cases = enumerate_cases()
    for case in cases:
        assert Path(case["bddl_path"]).is_file()
        assert Path(case["base_bddl_path"]).is_file()
        assert SUFFIX_RE.search(Path(case["bddl_path"]).stem)
        assert not SUFFIX_RE.search(Path(case["base_bddl_path"]).stem)


def test_lighting_scene_xml_tracks_real_plus_assets():
    lighting = [case for case in enumerate_cases() if case["perturbation_family"] == "lighting"][0]
    assets = inspect_problem_assets(Path(lighting["bddl_path"]))
    assets_dir = ROOT / "LIBERO-plus/libero/libero/assets"
    assert assets["class_found"] is True
    assert assets["scene_xml"]
    if assets_dir.is_symlink() or not (assets_dir / "scenes").is_dir():
        assert assets["scene_xml_exists"] is False
        assert assets["scene_xml"] in assets["missing_assets"]
    else:
        assert assets["scene_xml_exists"] is True
        assert "floor_style=light-gray" not in (assets.get("missing_assets") or [])
        assert assets.get("resolved_textures", {}).get("floor_style") == "light-gray-floor-tile.png"
