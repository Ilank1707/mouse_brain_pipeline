"""Regression tests for measurement, review, masks and classifier workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from mouse_brain_pipeline.audit import ChannelIndex  # noqa: E402
from mouse_brain_pipeline.candidate_detection import (  # noqa: E402
    CANDIDATE_COLUMNS,
    STATUS_INVALID_MEASUREMENT,
    STATUS_MANUAL_REVIEW,
    STATUS_PRELIMINARY_PASS,
    STATUS_SUSPECT_INJECTION,
    DetectionParams,
    SectionDetectionResult,
    build_injection_masks,
    detect_candidates_in_stack,
    measure_fixed_xy_profile,
    measure_plane_contrast,
    write_candidate_tables,
)
from mouse_brain_pipeline.candidate_qc import (  # noqa: E402
    REVIEW_BATCH_COLUMNS,
    candidates_outside_analysis_mask,
    write_review_batch,
)
from mouse_brain_pipeline.classifier import (  # noqa: E402
    CandidatePatchDataset,
    binary_training_records,
    candidate_group_key,
    classifier_state,
    grouped_train_validation_split,
    require_minimum_labels,
    validation_metrics,
)
from mouse_brain_pipeline.config import InjectionExclusionConfig, TissueMaskConfig  # noqa: E402
from mouse_brain_pipeline.pilot_stack import section_availability  # noqa: E402
from mouse_brain_pipeline.reference_audit import match_reference_points  # noqa: E402
from mouse_brain_pipeline.review import (  # noqa: E402
    LabelConflictError,
    load_manual_labels,
    save_manual_label,
)

VOXEL = (6.0, 1.0, 1.0)


def params(**overrides):
    result = DetectionParams(backend="cellfinder_candidates")
    result.tissue = TissueMaskConfig(enabled=False)
    result.injection = InjectionExclusionConfig(enabled=False)
    result.exclude_crop_boundary = False
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


class FakeCell:
    def __init__(self, x=40, y=40, z=3, type=1):
        self.x, self.y, self.z, self.type = x, y, z, type


def detect_stub(stack, detection_params=None):
    return detect_candidates_in_stack(
        stack, detection_params or params(), VOXEL,
        channel="green_signal", section=70, first_section=70,
        plane_numbers=list(range(1, 8)),
        injection_cfg=InjectionExclusionConfig(enabled=False),
        backend="cellfinder_candidates",
        cellfinder_detect_main=lambda **_kwargs: [FakeCell()],
    )


def test_zero_mad_never_returns_10000():
    rng = np.random.default_rng(1)
    plane = 100 + rng.normal(0, 2, (81, 81))
    yy, xx = np.ogrid[:81, :81]
    ring = ((yy - 40) ** 2 + (xx - 40) ** 2 >= 8 ** 2) & \
        ((yy - 40) ** 2 + (xx - 40) ** 2 <= 16 ** 2)
    plane[ring] = 100  # local MAD and IQR are exactly zero
    plane[38:43, 38:43] = 150
    measured = measure_plane_contrast(
        plane, np.ones_like(plane, dtype=bool), 40, 40, 3, 8, 16,
        minimum_background_pixels=20, padding_values=(0,),
    )
    assert measured["background_mad"] == 0
    assert measured["background_noise_method"] == "invalid_local_noise"
    assert measured["measurement_valid"] is False
    assert measured["contrast"] != 10000


def test_insufficient_annulus_is_invalid_and_routes_to_manual_review():
    stack = np.full((7, 81, 81), 100.0, dtype=np.float32)
    stack[3, 38:43, 38:43] = 500
    tissue = np.zeros((81, 81), dtype=bool)
    tissue[38:43, 38:43] = True
    result = detect_candidates_in_stack(
        stack, params(), VOXEL, channel="green_signal", section=70, first_section=70,
        plane_numbers=list(range(1, 8)), shared_tissue_mask=tissue,
        injection_cfg=InjectionExclusionConfig(enabled=False),
        backend="cellfinder_candidates",
        cellfinder_detect_main=lambda **_kwargs: [FakeCell()],
    )
    candidate = result.candidates[0]
    assert candidate["measurement_valid"] is False
    assert np.isnan(candidate["local_robust_z"])
    assert candidate["current_status"] == STATUS_INVALID_MEASUREMENT
    assert candidate["preliminary_sampling_category"] == STATUS_MANUAL_REVIEW
    assert candidate["rejection_reason"] == "invalid_local_noise"


def test_fixed_xy_support_ignores_unrelated_bright_object():
    rng = np.random.default_rng(2)
    stack = (100 + rng.normal(0, 2, (7, 101, 101))).astype(np.float32)
    yy, xx = np.ogrid[:101, :101]
    centre = 500 * np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / 8)
    for z in (1, 2, 3):
        stack[z] += centre
    stack[4, 68:73, 68:73] += 5000  # bright, but not at candidate XY
    _measurements, _profile, peak, support = measure_fixed_xy_profile(
        stack, np.ones((101, 101), dtype=bool), 50, 50, params(), voxel_y_um=1.0
    )
    assert peak in {1, 2, 3}
    assert support == [1, 2, 3]
    assert 4 not in support


def test_broad_seven_plane_feature_is_manual_review_not_counted():
    rng = np.random.default_rng(3)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    yy, xx = np.ogrid[:81, :81]
    blob = 500 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 8)
    stack += blob
    result = detect_stub(stack)
    candidate = result.candidates[0]
    assert candidate["support_plane_count"] == 7
    assert candidate["current_status"] == STATUS_MANUAL_REVIEW
    assert candidate["rejection_reason"] == "many_planes_review"
    assert candidate["rejection_reason"] != "too_many_planes"
    assert candidate["included_in_count"] is False


def test_review_label_saves_immediately_and_survives_restart(tmp_path):
    path = tmp_path / "manual_labels.csv"
    candidate = {
        "candidate_id": "c1", "channel": "green_signal", "section": 70,
        "source_crop": "1000:5000,500:4000",
    }
    save_manual_label(path, candidate, "cell", "reviewer-a")
    loaded = load_manual_labels(path)
    assert loaded[("c1", "green_signal")]["manual_label"] == "cell"
    with pytest.raises(LabelConflictError):
        save_manual_label(path, candidate, "artefact", "reviewer-a")
    save_manual_label(
        path, candidate, "artefact", "reviewer-a", allow_overwrite=True
    )
    restarted = load_manual_labels(path)
    assert restarted[("c1", "green_signal")]["manual_label"] == "artefact"
    assert len(restarted) == 1


def test_injection_core_and_analysis_masks_are_distinct_and_channel_specific():
    stack = np.ones((7, 100, 100), dtype=np.float32)
    green_cfg = InjectionExclusionConfig(
        enabled=True, automatic=False, core_dilation_um=2,
        analysis_exclusion_dilation_um=8, manual_rectangles=[[10, 20, 10, 20]],
    )
    ch2_cfg = InjectionExclusionConfig(
        enabled=True, automatic=False, core_dilation_um=2,
        analysis_exclusion_dilation_um=8, manual_rectangles=[[70, 80, 70, 80]],
    )
    green_core, green_analysis, _ = build_injection_masks(stack, VOXEL, green_cfg)
    ch2_core, ch2_analysis, _ = build_injection_masks(stack, VOXEL, ch2_cfg)
    assert green_analysis.sum() > green_core.sum()
    assert ch2_analysis.sum() > ch2_core.sum()
    assert green_analysis[15, 15] and not green_analysis[75, 75]
    assert ch2_analysis[75, 75] and not ch2_analysis[15, 15]


def test_training_refuses_insufficient_labels_and_excludes_nonbinary_labels():
    records = [
        {"manual_label": "cell"},
        {"manual_label": "artefact"},
        {"manual_label": "injection"},
        {"manual_label": "uncertain"},
    ]
    assert [row["manual_label"] for row in binary_training_records(records)] == [
        "cell", "artefact"
    ]
    with pytest.raises(ValueError, match="Insufficient manual labels"):
        require_minimum_labels(records, 50, 50)


def test_classifier_dataset_input_shape_is_n_1_z_y_x(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    files = {}
    for plane in range(1, 8):
        path = tmp_path / f"section_070_{plane:02d}.tif"
        tifffile.imwrite(path, np.full((64, 64), 100 + plane, dtype=np.uint16))
        files[(70, plane)] = path
    index = ChannelIndex("green_signal", tmp_path, files=files)
    dataset = CandidatePatchDataset(
        [{
            "candidate_id": "c1", "section": 70, "x_global_px": 32,
            "y_global_px": 32, "manual_label": "cell",
        }],
        index, patch_size_xy_px=31,
    )
    tensor, label, _candidate_id = dataset[0]
    assert tuple(tensor.shape) == (1, 7, 31, 31)
    assert tuple(tensor.unsqueeze(0).shape) == (1, 1, 7, 31, 31)
    assert label == 1


def test_grouped_split_keeps_spatial_tiles_together():
    records = [
        {"candidate_id": f"a{i}", "section": 70, "x_global_px": 10 + i, "y_global_px": 10}
        for i in range(3)
    ] + [
        {"candidate_id": f"b{i}", "section": 70, "x_global_px": 700 + i, "y_global_px": 10}
        for i in range(3)
    ]
    train, validation = grouped_train_validation_split(
        records, group_by="spatial_tile", spatial_tile_size_px=512,
        validation_fraction=0.5, seed=1,
    )
    train_groups = {candidate_group_key(row, "spatial_tile", 512) for row in train}
    validation_groups = {candidate_group_key(row, "spatial_tile", 512) for row in validation}
    assert train_groups.isdisjoint(validation_groups)


def test_unreviewed_preliminary_candidate_is_never_included():
    rng = np.random.default_rng(4)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    yy, xx = np.ogrid[:81, :81]
    blob = 500 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 8)
    for z in (2, 3, 4):
        stack[z] += blob
    result = detect_stub(stack)
    candidate = result.candidates[0]
    assert candidate["current_status"] == STATUS_PRELIMINARY_PASS
    assert candidate["included_in_count"] is False


def test_missing_section_71_is_reported_as_skipped():
    report = section_availability([70, 71], [70])
    assert report["processed"] == [70]
    assert report["skipped"] == [71]


def test_padding_zeros_are_excluded_from_background():
    rng = np.random.default_rng(11)
    plane = np.zeros((81, 81), dtype=np.float32)
    yy, xx = np.ogrid[:81, :81]
    annulus = ((yy - 40) ** 2 + (xx - 40) ** 2 >= 8 ** 2) & \
        ((yy - 40) ** 2 + (xx - 40) ** 2 <= 16 ** 2)
    valid_positions = np.argwhere(annulus)
    chosen = valid_positions[:30]
    plane[chosen[:, 0], chosen[:, 1]] = 100 + rng.normal(0, 2, len(chosen))
    plane[38:43, 38:43] = 150
    measured = measure_plane_contrast(
        plane, np.ones_like(plane, dtype=bool), 40, 40, 3, 8, 16,
        minimum_background_pixels=20, padding_values=(0,),
    )
    assert measured["background_pixel_count"] == 30
    assert measured["background_median"] > 90


def test_adjacent_plane_pooled_annulus_is_documented_fallback():
    rng = np.random.default_rng(17)
    plane = np.full((81, 81), 100.0, dtype=np.float32)
    plane[38:43, 38:43] = 150
    adjacent_values = 100 + rng.normal(0, 3, 100)
    measured = measure_plane_contrast(
        plane, np.ones_like(plane, dtype=bool), 40, 40, 3, 8, 16,
        minimum_background_pixels=20, padding_values=(0,),
        adjacent_background_values=adjacent_values,
    )
    assert measured["measurement_valid"] is True
    assert measured["background_noise_method"].startswith("adjacent_pooled_annulus_")
    assert np.isfinite(measured["contrast"])


def test_original_cellfinder_and_fixed_xy_coordinates_are_preserved():
    rng = np.random.default_rng(12)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    yy, xx = np.ogrid[:81, :81]
    for z in (2, 3, 4):
        stack[z] += 500 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 8)
    candidate = detect_stub(stack).candidates[0]
    assert int(float(candidate["original_cellfinder_z_index"])) == 3
    assert candidate["fixed_xy_center_x_local_px"] == 40
    assert candidate["fixed_xy_center_y_local_px"] == 40
    assert 0 <= candidate["fixed_xy_peak_z_index"] <= 6


def test_automatic_unvalidated_mask_is_suspect_not_confirmed_injection():
    rng = np.random.default_rng(13)
    stack = (100 + rng.normal(0, 2, (7, 101, 101))).astype(np.float32)
    stack[:, 20:80, 20:80] += 3000
    cfg = InjectionExclusionConfig(
        enabled=True, automatic=True, downsample_um=2,
        smoothing_sigma_um=5, intensity_percentile=80,
        minimum_area_um2=100, core_dilation_um=1,
        analysis_exclusion_dilation_um=3, mask_validated=False,
    )
    result = detect_candidates_in_stack(
        stack, params(), VOXEL, channel="green_signal", section=70, first_section=70,
        plane_numbers=list(range(1, 8)), injection_cfg=cfg,
        backend="cellfinder_candidates",
        cellfinder_detect_main=lambda **_kwargs: [FakeCell(40, 40, 3, 1)],
    )
    candidate = result.candidates[0]
    assert candidate["inside_injection_analysis_exclusion"] is True
    assert candidate["current_status"] == STATUS_SUSPECT_INJECTION
    assert candidate["injection_assignment_source"] != "validated_automatic_mask"


def test_human_injection_label_overrides_automatic_uncertainty():
    status, included = classifier_state(
        {"measurement_valid": "True", "current_status": STATUS_SUSPECT_INJECTION},
        0.99,
        manual_label="injection",
        model_validated=True,
    )
    assert status == "injection_site"
    assert included is False


def test_all_status_counts_reconcile_exactly():
    rng = np.random.default_rng(14)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    result = detect_stub(stack)
    counts = {}
    for candidate in result.candidates:
        status = candidate["current_status"]
        counts[status] = counts.get(status, 0) + 1
    assert sum(counts.values()) == len(result.candidates)


def test_csv_schema_matches_review_interface(tmp_path):
    rng = np.random.default_rng(15)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    result = detect_stub(stack)
    section_result = SectionDetectionResult(
        channel="green_signal", section=70, candidates=result.candidates
    )
    paths = write_candidate_tables(tmp_path, [section_result])
    write_review_batch(tmp_path, result.candidates, {})
    import csv

    with open(paths["all"], newline="", encoding="utf-8") as fh:
        all_header = next(csv.reader(fh))
    with open(tmp_path / "review_batch.csv", newline="", encoding="utf-8") as fh:
        review_header = next(csv.reader(fh))
    required = {
        "measurement_valid", "background_noise_method", "background_pixel_count",
        "fixed_xy_support_z_indices", "fixed_xy_peak_z_index",
        "inside_injection_core", "inside_injection_analysis_exclusion",
        "plane_0_contrast", "plane_6_contrast",
    }
    assert required <= set(all_header)
    assert required <= set(review_header)
    assert set(CANDIDATE_COLUMNS) <= set(REVIEW_BATCH_COLUMNS)


def test_rule_failed_candidates_remain_in_all_csv_and_outside_qc_selection(tmp_path):
    rng = np.random.default_rng(16)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    result = detect_stub(stack, params(min_diameter_um=100))
    assert result.candidates[0]["preliminary_sampling_category"] == "preliminary_rule_fail"
    section_result = SectionDetectionResult(
        channel="green_signal", section=70, candidates=result.candidates
    )
    paths = write_candidate_tables(tmp_path, [section_result])
    assert "preliminary_rule_fail" in paths["all"].read_text(encoding="utf-8")
    assert result.candidates[0] in candidates_outside_analysis_mask(result.candidates)


def test_unvalidated_predicted_cell_is_not_included():
    status, included = classifier_state(
        {"measurement_valid": "True", "current_status": "preliminary_rule_pass"},
        0.99,
        model_validated=False,
    )
    assert status == "predicted_cell"
    assert included is False


def test_validation_metrics_report_missing_class_limitation():
    metrics = validation_metrics([1, 1], [0.9, 0.8])
    assert metrics["pr_auc"] is None
    assert any("both classes" in text for text in metrics["limitations"])


def test_reference_recall_matching_is_one_to_one():
    references = [
        {"reference_id": "r1", "channel": "green_signal", "section": "70",
         "x_global_px": "100", "y_global_px": "100", "z_index": "3"},
        {"reference_id": "r2", "channel": "green_signal", "section": "70",
         "x_global_px": "102", "y_global_px": "100", "z_index": "3"},
    ]
    candidates = [
        {"candidate_id": "c1", "channel": "green_signal", "section": "70",
         "x_global_px": "101", "y_global_px": "100", "cellfinder_z_index": "3"},
    ]
    matches, unmatched_references, unmatched_candidates = match_reference_points(
        references, candidates,
        voxel_size_y_um=1, voxel_size_x_um=1, voxel_size_z_um=6,
        xy_tolerance_um=5, z_tolerance_um=6,
    )
    assert len(matches) == 1
    assert len(unmatched_references) == 1
    assert unmatched_candidates == []


def test_raw_tiff_is_not_written_when_extracting_classifier_patch(tmp_path):
    import hashlib
    import os
    import time

    tifffile = pytest.importorskip("tifffile")
    files = {}
    for plane in range(1, 8):
        path = tmp_path / f"section_070_{plane:02d}.tif"
        tifffile.imwrite(path, np.full((64, 64), 100 + plane, dtype=np.uint16))
        files[(70, plane)] = path
    watched = files[(70, 1)]
    before_hash = hashlib.sha256(watched.read_bytes()).hexdigest()
    before_mtime = os.stat(watched).st_mtime_ns
    time.sleep(0.001)
    index = ChannelIndex("green_signal", tmp_path, files=files)
    dataset = CandidatePatchDataset(
        [{"candidate_id": "c1", "section": 70, "x_global_px": 32,
          "y_global_px": 32, "manual_label": "cell"}],
        index, patch_size_xy_px=31,
    )
    dataset[0]
    assert hashlib.sha256(watched.read_bytes()).hexdigest() == before_hash
    assert os.stat(watched).st_mtime_ns == before_mtime
