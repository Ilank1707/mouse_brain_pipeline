"""Radial candidate analysis maths: distance, in-tissue area, density, safety."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_brain_pipeline import radial_analysis as ra  # noqa: E402
from mouse_brain_pipeline.coordinate_exports import is_confirmed_cell  # noqa: E402

XY_UM = 1.004


def test_radial_distance_uses_xy_pixel_size():
    d = ra.radial_distances_um([100.0], [0.0], (0.0, 0.0), (XY_UM, XY_UM))
    assert abs(float(d[0]) - 100.0 * XY_UM) < 1e-6


def test_per_candidate_distance_and_bin():
    cands = [{"x_global_px": "300", "y_global_px": "400", "candidate_id": "c",
              "channel": "green_signal", "section": "70", "current_status": "x"}]
    rows = ra.per_candidate_rows(cands, (0.0, 0.0), (XY_UM, XY_UM), 100.0, n_bins=20)
    expect = ((300.0 ** 2 + 400.0 ** 2) ** 0.5) * XY_UM  # 500 px -> ~502 um
    assert abs(rows[0]["radial_distance_um"] - round(expect, 2)) < 0.01
    assert rows[0]["radial_bin_start_um"] == 500.0


def test_tissue_area_only_inside_mask():
    # A disk of tissue inside a larger frame; ring area must ignore the corners.
    n = 101
    yy, xx = np.mgrid[0:n, 0:n]
    tissue = (xx - 50) ** 2 + (yy - 50) ** 2 <= 40 ** 2
    area = ra.tissue_area_by_bin(tissue, (50, 50), (1.0, 1.0), 10.0, n_bins=12)
    # Every counted pixel is a tissue pixel and all tissue pixels are counted.
    assert int(area.sum()) == int(tissue.sum())
    # Beyond the disk radius (>40 um) there is no tissue area.
    assert int(area[5:].sum()) == 0


def test_density_equals_count_over_valid_tissue_area():
    counts = np.array([3, 10, 0, 4])
    area_px = np.array([100, 500, 0, 200])  # bin 2 has NO tissue
    voxel_area = 1.004 * 1.004
    rows = ra.assemble_series(counts, area_px, 100.0, voxel_area)

    for i in (0, 1, 3):
        area_mm2 = area_px[i] * voxel_area / 1.0e6
        assert abs(rows[i]["density_per_mm2"] - counts[i] / area_mm2) < 1e-3
    # Empty annulus (no tissue) -> density undefined, never a divide-by-zero.
    assert rows[2]["density_per_mm2"] == ""


def test_fraction_and_cumulative_fraction():
    counts = np.array([2, 3, 5])
    area_px = np.array([10, 10, 10])
    rows = ra.assemble_series(counts, area_px, 100.0, 1.0)
    assert abs(rows[0]["fraction"] - 0.2) < 1e-9
    assert abs(rows[-1]["cumulative_fraction"] - 1.0) < 1e-9
    assert rows[-1]["cumulative_count"] == 10


def test_empty_series_has_no_divide_by_zero():
    rows = ra.assemble_series(np.zeros(4), np.zeros(4), 100.0, 1.0)
    assert all(r["fraction"] == 0 for r in rows)
    assert all(r["density_per_mm2"] == "" for r in rows)  # area 0 -> undefined


def test_preliminary_pass_is_not_a_confirmed_cell():
    prelim = {"current_status": "preliminary_rule_pass", "manual_label": "",
              "model_validation_passed": "False"}
    assert not is_confirmed_cell(prelim)
    labelled = {"current_status": "preliminary_rule_pass", "manual_label": "cell"}
    assert is_confirmed_cell(labelled)


def test_injection_core_centroid_uses_crop_origin():
    core = np.zeros((20, 20), dtype=bool)
    core[8:12, 8:12] = True
    centroid = ra.injection_core_centroid(core, crop_origin=(100, 200))
    assert abs(centroid["x_global_px"] - (9.5 + 200)) < 1e-6
    assert abs(centroid["y_global_px"] - (9.5 + 100)) < 1e-6
