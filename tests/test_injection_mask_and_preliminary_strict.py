"""Focused tests for the green injection-mask caps and the stricter preliminary
rule (Part 1 + Part 2). Behaviours 1-12 from the task spec.

Nothing here targets a candidate count; each test asserts a monotonic/structural
property of a named configurable parameter.
"""

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from mouse_brain_pipeline import candidate_detection as cd  # noqa: E402
from mouse_brain_pipeline import run_layout  # noqa: E402
from mouse_brain_pipeline.candidate_detection import (  # noqa: E402
    STATUS_PRELIMINARY_PASS,
    DetectionParams,
    _preliminary_interpretation,
    params_from_config,
    write_candidate_tables,
    SectionDetectionResult,
)
from mouse_brain_pipeline.config import Config, InjectionExclusionConfig  # noqa: E402

VOXEL_LOW = (1.0, 1.0)


def _two_disks(gap):
    """Two radius-15 disks whose nearest edges are ``gap`` px apart (one row)."""
    h, w = 60, 80
    yy, xx = np.ogrid[:h, :w]
    cx_left = 20
    cx_right = cx_left + 30 + gap  # 30 = 2*radius
    left = (yy - 30) ** 2 + (xx - cx_left) ** 2 <= 15 ** 2
    right = (yy - 30) ** 2 + (xx - cx_right) ** 2 <= 15 ** 2
    return left | right, (30, cx_left), (30, cx_right)


# --------------------------------------------------------------------------- #
# 1. Closing cannot bridge a gap wider than maximum_bridge_width_um.
# --------------------------------------------------------------------------- #
def test_closing_cannot_bridge_gap_wider_than_bridge_width():
    from scipy import ndimage as ndi

    mask, _l, _r = _two_disks(gap=10)
    capped = InjectionExclusionConfig(closing_radius_um=20.0, maximum_bridge_width_um=6.0)
    uncapped = InjectionExclusionConfig(closing_radius_um=20.0, maximum_bridge_width_um=0.0)

    out_capped = cd._apply_bright_morphology(mask, capped, VOXEL_LOW)
    out_uncapped = cd._apply_bright_morphology(mask, uncapped, VOXEL_LOW)

    # Bridge-capped: the 10 px gap survives -> still two components.
    assert ndi.label(out_capped)[1] == 2
    # Uncapped closing (radius 20) bridges the 10 px gap -> a single component.
    assert ndi.label(out_uncapped)[1] == 1


# --------------------------------------------------------------------------- #
# 2. A seeded component cannot grow beyond maximum_distance_from_seed_um.
# --------------------------------------------------------------------------- #
def test_seeded_region_capped_at_maximum_distance():
    mask = np.zeros((100, 220), dtype=bool)
    mask[30:70, 10:210] = True  # one long solid bright bar
    seed = (50, 20)  # (y, x) near the left end

    capped_cfg = InjectionExclusionConfig(
        split_min_peak_distance_um=40.0,
        maximum_distance_from_seed_um=30.0,
        seed_distance_metric="geodesic",
    )
    uncapped_cfg = InjectionExclusionConfig(
        split_min_peak_distance_um=40.0, maximum_distance_from_seed_um=None
    )
    kept_capped, diag = cd._split_and_filter_by_seeds(
        mask, [seed], VOXEL_LOW, capped_cfg, 1
    )
    kept_uncapped, _ = cd._split_and_filter_by_seeds(
        mask, [seed], VOXEL_LOW, uncapped_cfg, 1
    )

    assert diag["seed_distance_capped"] is True
    assert diag["maximum_distance_from_seed_um"] == 30.0
    assert kept_capped.sum() < kept_uncapped.sum()
    assert bool(kept_capped[50, 20]) is True             # near the seed: kept
    assert bool(kept_capped[50, 200]) is False           # 180 px away: dropped
    # Every kept pixel is within the cap (+1 px tolerance for the disk boundary).
    ys, xs = np.nonzero(kept_capped)
    assert np.hypot(ys - 50, xs - 20).max() <= 31.0


# --------------------------------------------------------------------------- #
# 3. A seedless post-split lobe is removed.
# --------------------------------------------------------------------------- #
def test_seedless_post_split_lobe_removed():
    # Dumbbell: two lobes joined by a neck = one connected component.
    yy, xx = np.ogrid[:120, :240]
    left = (yy - 60) ** 2 + (xx - 60) ** 2 <= 45 ** 2
    right = (yy - 60) ** 2 + (xx - 180) ** 2 <= 45 ** 2
    neck = (np.abs(yy - 60) <= 12) & (xx >= 60) & (xx <= 180)
    mask = left | right | neck

    cfg = InjectionExclusionConfig(split_min_peak_distance_um=40.0)
    kept, diag = cd._split_and_filter_by_seeds(mask, [(60, 60)], VOXEL_LOW, cfg, 1)

    assert bool(kept[60, 60]) is True     # seeded left lobe kept
    assert bool(kept[60, 180]) is False   # seedless right lobe removed
    removed = [r for r in diag["post_split_subcomponents"] if not r["kept"]]
    assert removed and all(not r["contains_seed"] for r in removed)


# --------------------------------------------------------------------------- #
# 4. Channel-specific parameters do not leak between green and red.
# --------------------------------------------------------------------------- #
def test_channel_specific_mask_params_do_not_leak():
    cfg = InjectionExclusionConfig.from_dict({
        "maximum_distance_from_seed_um": None,          # base default
        "green_signal": {
            "injection_seed_points": [[20, 50]],
            "opening_radius_um": 25.0,
            "maximum_distance_from_seed_um": 30.0,
            "maximum_bridge_width_um": 50.0,
        },
        "channel_2_signal": {"injection_seed_points": [[200, 50]]},
    })
    green = cfg.for_channel("green_signal")
    red = cfg.for_channel("channel_2_signal")

    assert green.maximum_distance_from_seed_um == 30.0
    assert green.opening_radius_um == 25.0
    assert green.maximum_bridge_width_um == 50.0
    # Red inherits the base (None / 0) -- the green caps do NOT leak across.
    assert red.maximum_distance_from_seed_um is None
    assert red.opening_radius_um == 0.0
    assert red.injection_seed_points == [[200, 50]]
    assert green.injection_seed_points != red.injection_seed_points

    # Functional: the same bright bar yields a smaller green base (capped) than
    # the red base (uncapped), on their own seeds.
    mask = np.zeros((100, 260), dtype=bool)
    mask[30:70, 10:250] = True
    green_kept, _ = cd._split_and_filter_by_seeds(mask, [(50, 20)], VOXEL_LOW, green, 1)
    red_kept, _ = cd._split_and_filter_by_seeds(mask, [(50, 200)], VOXEL_LOW, red, 1)
    assert green_kept.sum() < red_kept.sum()


def test_channel_specific_candidate_screening_does_not_leak():
    cfg = Config.from_dict({
        "detection": {
            "minimum_component_xy_area_um2": 28.3,
            "minimum_component_volume_um3": 113.0,
            "minimum_supporting_voxels": 19,
            "minimum_support_planes": 2,
            "minimum_signal_to_background_ratio": 8.0,
            "green_signal": {
                "minimum_component_xy_area_um2": 50.3,
                "minimum_component_volume_um3": 268.1,
                "minimum_supporting_voxels": 45,
                "minimum_support_planes": 3,
                "minimum_signal_to_background_ratio": 10.0,
            },
            "cellfinder": {
                "n_sds_above_mean_thresh": 10,
                "n_sds_above_mean_tiled_thresh": 10,
                "green_signal": {
                    "n_sds_above_mean_thresh": 12,
                    "n_sds_above_mean_tiled_thresh": 12,
                },
            },
        },
    })
    params = params_from_config(cfg)
    green = params.for_channel("green_signal")
    red = params.for_channel("channel_2_signal")

    assert (
        green.min_component_xy_area_um2,
        green.min_component_volume_um3,
        green.min_supporting_voxels,
        green.min_support_planes,
        green.min_signal_to_background_ratio,
    ) == (50.3, 268.1, 45, 3, 10.0)
    assert (
        red.min_component_xy_area_um2,
        red.min_component_volume_um3,
        red.min_supporting_voxels,
        red.min_support_planes,
        red.min_signal_to_background_ratio,
    ) == (28.3, 113.0, 19, 2, 8.0)
    assert params.cellfinder.for_channel("green_signal").n_sds_above_mean_thresh == 12
    assert params.cellfinder.for_channel("channel_2_signal").n_sds_above_mean_thresh == 10


# --------------------------------------------------------------------------- #
# Preliminary-rule helpers (Part 2).
# --------------------------------------------------------------------------- #
def _passing_rec():
    return {
        "inside_tissue": True,
        "invalid_coordinate": False,
        "original_cellfinder_z_valid": True,
        "measurement_valid": True,
        "is_artifact": False,
        "touches_crop_boundary": False,
        "n_consecutive_planes": 3,
        "support_plane_count": 3,
        "equivalent_diameter_um": 10.0,
        "xy_diameter_um": 10.0,
        "xy_area_um2": float(np.pi * 25.0),   # ~78.5
        "volume_um3": 500.0,
        "supporting_voxel_count": 80,
        "elongation": 1.2,
        "xy_centroid_shift_um": 1.0,
        "local_robust_z": 10.0,
    }


def _passes(rec, params):
    status, _ = _preliminary_interpretation(rec, params, has_tissue=True)
    return status == STATUS_PRELIMINARY_PASS


def test_baseline_rec_passes_with_default_params():
    assert _passes(_passing_rec(), DetectionParams()) is True


# 5. Increasing minimum area makes the rule stricter.
def test_increasing_min_area_makes_rule_stricter():
    rec = _passing_rec()
    loose = DetectionParams(min_component_xy_area_um2=0.0)
    strict = DetectionParams(min_component_xy_area_um2=100.0)  # > 78.5
    assert _passes(rec, loose) is True
    assert _passes(rec, strict) is False


# 6. Increasing minimum support planes makes the rule stricter.
def test_increasing_min_support_planes_makes_rule_stricter():
    rec = _passing_rec()
    loose = DetectionParams(min_support_planes=0)
    strict = DetectionParams(min_support_planes=4)  # rec has 3
    assert _passes(rec, loose) is True
    assert _passes(rec, strict) is False


# 7. Increasing minimum support voxels makes the rule stricter.
def test_increasing_min_support_voxels_makes_rule_stricter():
    rec = _passing_rec()
    loose = DetectionParams(min_supporting_voxels=0)
    strict = DetectionParams(min_supporting_voxels=100)  # rec has 80
    assert _passes(rec, loose) is True
    assert _passes(rec, strict) is False


# 8. Increasing minimum signal-to-background ratio makes the rule stricter.
def test_increasing_min_snr_makes_rule_stricter():
    rec = _passing_rec()
    loose = DetectionParams(min_signal_to_background_ratio=0.0)
    strict = DetectionParams(min_signal_to_background_ratio=12.0)  # rec has 10
    assert _passes(rec, loose) is True
    assert _passes(rec, strict) is False


# 9. Edge-clipped candidates with centres inside tissue are not auto-rejected.
def test_edge_clipped_in_tissue_candidate_not_auto_rejected():
    rec = _passing_rec()
    rec["touches_crop_boundary"] = True
    rec["inside_tissue"] = True

    keep = DetectionParams(
        exclude_crop_boundary=True, keep_edge_clipped_if_center_in_tissue=True
    )
    drop = DetectionParams(
        exclude_crop_boundary=True, keep_edge_clipped_if_center_in_tissue=False
    )
    status_keep, reason_keep = _preliminary_interpretation(rec, keep, has_tissue=True)
    status_drop, reason_drop = _preliminary_interpretation(rec, drop, has_tissue=True)

    assert status_keep == STATUS_PRELIMINARY_PASS
    assert reason_keep != cd.REASON_CROP_BOUNDARY
    # Legacy behaviour still available: dropped for the edge.
    assert reason_drop == cd.REASON_CROP_BOUNDARY


# --------------------------------------------------------------------------- #
# 10. No desired final candidate count is encoded anywhere.
# --------------------------------------------------------------------------- #
def test_no_target_candidate_count_encoded_in_source():
    suspicious_counts = {"8083", "8108", "7634", "9315", "11506", "19249", "7743"}
    forbidden_names = re.compile(
        r"target_count|desired_count|expected_count|goal_count|TARGET_COUNT",
        re.IGNORECASE,
    )
    src_dir = ROOT / "src" / "mouse_brain_pipeline"
    scripts_dir = ROOT / "scripts"
    files = list(src_dir.glob("*.py")) + [
        scripts_dir / "debug_injection_mask.py",
        scripts_dir / "audit_preliminary_rules.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not forbidden_names.search(text), f"count-target name in {path.name}"
        for count in suspicious_counts:
            assert count not in text, f"hard-coded run count {count} in {path.name}"


# --------------------------------------------------------------------------- #
# 11. Every detected candidate remains in the audit output (all_candidates.csv).
# --------------------------------------------------------------------------- #
def test_all_candidates_preserved_including_failures(tmp_path):
    import csv

    candidates = [
        {"candidate_id": "g_0", "channel": "green_signal",
         "preliminary_sampling_category": STATUS_PRELIMINARY_PASS,
         "current_status": STATUS_PRELIMINARY_PASS},
        {"candidate_id": "g_1", "channel": "green_signal",
         "preliminary_sampling_category": "preliminary_rule_fail",
         "current_status": "preliminary_rule_fail"},
        {"candidate_id": "g_2", "channel": "green_signal",
         "preliminary_sampling_category": "manual_review",
         "current_status": "manual_review"},
    ]
    result = SectionDetectionResult(channel="green_signal", section=70,
                                    candidates=candidates)
    paths = write_candidate_tables(tmp_path, [result])
    all_rows = list(csv.DictReader((tmp_path / "all_candidates.csv").open(encoding="utf-8")))
    ids = {r["candidate_id"] for r in all_rows}
    assert ids == {"g_0", "g_1", "g_2"}  # failures preserved, not only passes
    pass_rows = list(
        csv.DictReader((tmp_path / "preliminary_pass_candidates.csv").open(encoding="utf-8"))
    )
    assert {r["candidate_id"] for r in pass_rows} == {"g_0"}


# --------------------------------------------------------------------------- #
# 12. Old isolated run folders are not modified / reused.
# --------------------------------------------------------------------------- #
def test_old_isolated_run_folder_is_not_modified(tmp_path):
    run_dir = run_layout.create_run_dir(tmp_path, "section070_strict")
    marker = run_dir / "all_candidates.csv"
    marker.write_text("original-run-data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_layout.create_run_dir(tmp_path, "section070_strict")
    assert marker.read_text(encoding="utf-8") == "original-run-data"
