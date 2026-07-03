"""Focused tests for the strict candidate refinement (Part B)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")

from mouse_brain_pipeline import strict_refinement as S  # noqa: E402
import refine_candidates_strict as cli  # noqa: E402

VOXEL_ZYX_UM = (6.0, 1.004, 1.004)
REQUIRED_COLUMNS = [
    "candidate_id", "channel", "section", "x_local_px", "y_local_px",
    "x_global_px", "y_global_px", "volume_um3", "xy_diameter_um",
    "support_plane_count", "mean_intensity", "peak_intensity",
    "measurement_valid", "invalid_coordinate", "inside_tissue", "current_status",
]


def _mask_left_half(height=200, width=200, split=100):
    mask = np.zeros((height, width), dtype=bool)
    mask[:, :split] = True
    return mask


def _cand(candidate_id, x, y, *, volume_um3, xy_diameter_um, support_plane_count,
          status="preliminary_rule_pass", channel="green_signal",
          measurement_valid=True, invalid_coordinate=False, mask=None):
    inside = False
    if mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        inside = bool(mask[y, x])
    return {
        "candidate_id": candidate_id, "channel": channel, "section": 70,
        "x_local_px": x, "y_local_px": y, "x_global_px": x, "y_global_px": y,
        "volume_um3": volume_um3, "xy_diameter_um": xy_diameter_um,
        "support_plane_count": support_plane_count, "mean_intensity": 120.0,
        "peak_intensity": 300.0, "measurement_valid": measurement_valid,
        "invalid_coordinate": invalid_coordinate, "inside_tissue": inside,
        "current_status": status,
    }


def _refine(rows, mask, *, mode, thresholds, channel="green_signal"):
    return S.refine_candidates_strict(
        rows, mask, voxel_zyx_um=VOXEL_ZYX_UM, mode=mode, thresholds=thresholds,
        channel=channel, section=70,
    )


# 4 + 5 -------------------------------------------------------------------- #
def test_tiny_weak_candidate_filtered_and_large_multiplane_retained():
    mask = _mask_left_half()
    rows = [
        _cand("tiny", 50, 50, volume_um3=30.0, xy_diameter_um=3.0,
              support_plane_count=1, mask=mask),
        _cand("big", 60, 60, volume_um3=6000.0, xy_diameter_um=22.0,
              support_plane_count=6, mask=mask),
    ]
    thresholds = S.StrictThresholds(
        min_component_volume_um3=100.0, min_support_planes=2
    )
    by_id = {r["candidate_id"]: r for r in _refine(
        rows, mask, mode="apply", thresholds=thresholds).rows}

    assert by_id["tiny"]["refined_candidate_status"] == S.STRICT_FILTERED_SMALL
    assert by_id["big"]["refined_candidate_status"] == S.STRICT_RETAINED


def test_low_support_bucket_is_separate_from_too_small():
    mask = _mask_left_half()
    # Passes size, fails support -> filtered_low_support (distinct bucket).
    rows = [
        _cand("low_support", 50, 50, volume_um3=6000.0, xy_diameter_um=22.0,
              support_plane_count=1, mask=mask)
    ]
    thresholds = S.StrictThresholds(
        min_component_volume_um3=100.0, min_support_planes=3
    )
    row = _refine(rows, mask, mode="apply", thresholds=thresholds).rows[0]
    assert row["refined_candidate_status"] == S.STRICT_FILTERED_LOW_SUPPORT


# 6 ----------------------------------------------------------------------- #
def test_edge_clipped_candidate_measured_with_valid_pixels_only():
    mask = _mask_left_half(height=20, width=20, split=10)  # tissue x < 10
    # Patch at (y=5, x=9), half=3 -> 7x7=49 intended pixels; only in-image,
    # in-tissue pixels counted (x 6..9, y 2..8 = 28).
    patch = S.measure_patch_validity(mask, cy=5, cx=9, half_px=3)
    assert patch["full_count"] == 49
    assert patch["valid_count"] == 28
    assert patch["clipped"] is True
    assert patch["valid_pixel_fraction"] == pytest.approx(28 / 49)

    # Image-boundary clip: only in-bounds pixels are read (never out of range).
    corner = S.measure_patch_validity(mask, cy=1, cx=1, half_px=3)
    assert corner["valid_count"] == 25
    assert corner["clipped"] is True


# 7 ----------------------------------------------------------------------- #
def test_report_mode_does_not_change_status():
    mask = _mask_left_half()
    rows = [
        _cand("t", 50, 50, volume_um3=1.0, xy_diameter_um=1.0,
              support_plane_count=1, mask=mask),
        _cand("b", 60, 60, volume_um3=9000.0, xy_diameter_um=25.0,
              support_plane_count=7, mask=mask),
    ]
    result = _refine(
        rows, mask, mode="report",
        thresholds=S.StrictThresholds(min_support_planes=99),  # ignored in report
    )
    for row in result.rows:
        assert row["refined_candidate_status"] == row["original_status"]
        assert row["strict_filter_status"] == "not_applied"
    refined = {r["refined_candidate_status"] for r in result.rows}
    assert S.STRICT_FILTERED_SMALL not in refined
    assert result.sweep_rows  # threshold sweep produced


# 8 ----------------------------------------------------------------------- #
def test_apply_mode_requires_explicit_threshold():
    mask = _mask_left_half()
    rows = [
        _cand("a", 50, 50, volume_um3=100.0, xy_diameter_um=10.0,
              support_plane_count=3, mask=mask)
    ]
    with pytest.raises(ValueError):
        _refine(rows, mask, mode="apply", thresholds=S.StrictThresholds())
    # An edge flag/distance alone is not a size/support threshold.
    with pytest.raises(ValueError):
        _refine(rows, mask, mode="apply",
                thresholds=S.StrictThresholds(rescue_edge_candidates=True,
                                              max_edge_distance_penalty_um=15.0))


def test_rescue_flag_requires_edge_distance():
    mask = _mask_left_half()
    rows = [
        _cand("a", 50, 50, volume_um3=100.0, xy_diameter_um=10.0,
              support_plane_count=3, mask=mask)
    ]
    with pytest.raises(ValueError):
        _refine(rows, mask, mode="apply",
                thresholds=S.StrictThresholds(min_support_planes=2,
                                              rescue_edge_candidates=True))


# ---- run-level helpers for 9 and 10 ------------------------------------- #
def _write_two_channel_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    channels = ("green_signal", "channel_2_signal")
    mask = _mask_left_half(120, 120, split=80)
    with (run_dir / "all_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        rng = np.random.default_rng(5)
        for channel in channels:
            for index in range(40):
                x = int(rng.integers(5, 110)); y = int(rng.integers(5, 110))
                writer.writerow(_cand(
                    f"{channel}_{index}", x, y,
                    volume_um3=float(rng.uniform(20, 6000)),
                    xy_diameter_um=float(rng.uniform(3, 22)),
                    support_plane_count=int(rng.integers(1, 7)),
                    channel=channel, mask=mask,
                ))
    for channel in channels:
        section_dir = run_dir / "qc" / f"{channel}_section_070"
        section_dir.mkdir(parents=True)
        np.save(section_dir / "tissue_mask.npy", mask)
    return run_dir, channels


def _refine_channel_to_out(run_dir, out_dir, channel, mode="report", thresholds=None):
    rows = cli._read_candidates(run_dir / "all_candidates.csv", channel, 70)
    mask = np.load(run_dir / "qc" / f"{channel}_section_070" / "tissue_mask.npy",
                   mmap_mode="r")
    result = S.refine_candidates_strict(
        rows, mask, voxel_zyx_um=VOXEL_ZYX_UM, mode=mode,
        thresholds=thresholds or S.StrictThresholds(), channel=channel, section=70,
    )
    channel_dir = cli._prepare_output_root(out_dir, run_dir, channel)
    S.write_strict_outputs(channel_dir, result, make_plots=True)
    return channel_dir


# 9 ----------------------------------------------------------------------- #
def test_audit_csv_contains_every_original_candidate(tmp_path):
    run_dir, _channels = _write_two_channel_run(tmp_path)
    out_dir = tmp_path / "strict_out"
    channel_dir = _refine_channel_to_out(run_dir, out_dir, "green_signal")

    original_ids = {
        row["candidate_id"]
        for row in cli._read_candidates(run_dir / "all_candidates.csv", "green_signal", 70)
    }
    with (channel_dir / S.AUDIT_CSV).open(newline="", encoding="utf-8") as handle:
        audit = list(csv.DictReader(handle))
    assert {row["candidate_id"] for row in audit} == original_ids
    assert all(row["original_status"] for row in audit)


# 10 ---------------------------------------------------------------------- #
def test_green_and_red_outputs_remain_separate(tmp_path):
    run_dir, channels = _write_two_channel_run(tmp_path)
    out_dir = tmp_path / "strict_out"
    green_dir = _refine_channel_to_out(run_dir, out_dir, "green_signal")
    red_dir = _refine_channel_to_out(run_dir, out_dir, "channel_2_signal")

    assert green_dir != red_dir
    for channel, channel_dir in (("green_signal", green_dir), ("channel_2_signal", red_dir)):
        for name in (S.AUDIT_CSV, S.SWEEP_CSV, S.PLOT_SIZE, S.PLOT_SUPPORT_PLANES,
                     S.PLOT_SUPPORT_VOXELS, S.PLOT_SIZE_VS_SUPPORT, S.PLOT_EDGE_QC):
            assert (channel_dir / name).is_file()
        with (channel_dir / S.AUDIT_CSV).open(newline="", encoding="utf-8") as handle:
            assert {row["channel"] for row in csv.DictReader(handle)} == {channel}


def test_cli_parser_defaults_and_flags():
    parser = cli.build_parser()
    args = parser.parse_args([
        "--run-dir", "r", "--channel", "green_signal", "--section", "70",
        "--out-dir", "o", "--mode", "report",
    ])
    assert args.min_component_area_um2 is None
    assert args.min_support_voxels is None
    assert args.max_edge_distance_penalty_um is None
    assert args.rescue_edge_candidates is False
