"""Manual injection-mask override: subtraction, dilation-safety, re-statusing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_brain_pipeline import candidate_detection as cd  # noqa: E402
from mouse_brain_pipeline import injection_overrides as io  # noqa: E402
from mouse_brain_pipeline import postprocess as pp  # noqa: E402
from mouse_brain_pipeline.config import InjectionExclusionConfig  # noqa: E402


def _bright_stack(h=120, w=120):
    stack = np.zeros((3, h, w), dtype=np.float32)
    stack[:, 50:70, 50:70] = 1000.0  # a compact bright injection core
    return stack


def _auto_cfg(**kw):
    base = dict(enabled=True, automatic=True, downsample_um=1, smoothing_sigma_um=2,
                intensity_percentile=95, minimum_area_um2=50, core_dilation_um=5,
                analysis_exclusion_dilation_um=20)
    base.update(kw)
    return InjectionExclusionConfig(**base)


def test_non_injection_polygon_removes_region():
    stack = _bright_stack()
    voxel = (6.0, 1.0, 1.0)
    poly = [[[80, 80], [110, 80], [110, 110], [80, 110]]]
    _core, analysis, _w, _d = cd.build_injection_masks_with_components(
        stack, voxel, _auto_cfg(analysis_exclusion_dilation_um=40,
                                 manual_non_injection_polygons=poly))
    removed = cd.rasterize_polygons(analysis.shape, poly)
    assert removed.sum() > 0
    # The false region is entirely absent from the final mask.
    assert int((analysis & removed).sum()) == 0


def test_dilation_cannot_add_removed_region_back():
    stack = _bright_stack()
    voxel = (6.0, 1.0, 1.0)
    poly = [[[70, 70], [115, 70], [115, 115], [70, 115]]]
    removed = cd.rasterize_polygons((120, 120), poly)
    # Even with a very large dilation the subtraction is applied last.
    for dil in (20, 60, 150):
        _c, analysis, _w, _d = cd.build_injection_masks_with_components(
            stack, voxel, _auto_cfg(analysis_exclusion_dilation_um=dil,
                                    manual_non_injection_polygons=poly))
        assert int((analysis & removed).sum()) == 0


def test_other_injection_area_survives_subtraction():
    stack = _bright_stack()
    voxel = (6.0, 1.0, 1.0)
    poly = [[[85, 85], [115, 85], [115, 115], [85, 115]]]  # away from the core
    _c, analysis, _w, _d = cd.build_injection_masks_with_components(
        stack, voxel, _auto_cfg(manual_non_injection_polygons=poly))
    # The genuine core region is still masked.
    assert bool(analysis[55, 55])


def test_removed_candidate_returns_to_normal_interpretation():
    row = {
        "current_status": "suspect_injection_mask",
        "preliminary_sampling_category": "preliminary_rule_pass",
        "preliminary_rule_reason": "",
        "invalid_coordinate": "False",
        "original_cellfinder_z_valid": "True",
        "measurement_valid": "True",
        "injection_mask_source": "automatic",
        "injection_mask_validated": "False",
        "injection_mask_qc_failed": "False",
    }
    # Still inside -> stays suspect.
    _old, new = pp.restatus_row(dict(row), inside_analysis=True, inside_core=True)
    assert new == "suspect_injection_mask"
    # Removed from the mask -> back to its preliminary interpretation, not injection.
    _old, new = pp.restatus_row(row, inside_analysis=False, inside_core=False)
    assert new == "preliminary_rule_pass"
    assert row["inside_injection_analysis_exclusion"] is False


def test_artifact_inside_mask_never_becomes_injection():
    row = {
        "current_status": "artifact",
        "preliminary_sampling_category": "preliminary_rule_fail",
        "invalid_coordinate": "False",
        "original_cellfinder_z_valid": "True",
        "measurement_valid": "True",
        "injection_mask_source": "automatic",
        "injection_mask_validated": "False",
        "injection_mask_qc_failed": "False",
    }
    _old, new = pp.restatus_row(row, inside_analysis=True, inside_core=False)
    assert new == "artifact"


def test_override_yaml_roundtrip_and_merge(tmp_path):
    path = tmp_path / "ov.yml"
    inj_poly = [[[1, 1], [5, 1], [5, 5]]]
    non_inj = [[[10, 10], [20, 10], [20, 20]]]
    io.save_channel_polygons(path, "green_signal", inj_poly, non_inj)

    got_inj, got_non = io.read_channel_polygons(path, "green_signal")
    assert got_non == [[[10, 10], [20, 10], [20, 20]]]
    assert got_inj == [[[1, 1], [5, 1], [5, 5]]]

    cfg = InjectionExclusionConfig(injection_seed_points=[[3, 3]])
    io.apply_overrides_to_injection_cfg(cfg, io.load_overrides(path))
    green = cfg.for_channel("green_signal")
    assert green.manual_non_injection_polygons == [[[10, 10], [20, 10], [20, 20]]]
    # An unrelated existing setting is preserved.
    assert green.injection_seed_points == [[3, 3]]
