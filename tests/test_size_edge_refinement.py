"""Focused tests for post-detection candidate size + edge refinement."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

np = pytest.importorskip("numpy")

from mouse_brain_pipeline import size_edge_refinement as R  # noqa: E402
import refine_candidate_size_and_edges as cli  # noqa: E402

VOXEL_ZYX_UM = (6.0, 1.004, 1.004)
REQUIRED_COLUMNS = [
    "candidate_id",
    "channel",
    "section",
    "x_local_px",
    "y_local_px",
    "x_global_px",
    "y_global_px",
    "volume_um3",
    "xy_diameter_um",
    "support_plane_count",
    "measurement_valid",
    "invalid_coordinate",
    "inside_tissue",
    "current_status",
]


def _mask_left_half(height=200, width=200, split=100):
    """Tissue on the left (x < split), background on the right."""
    mask = np.zeros((height, width), dtype=bool)
    mask[:, :split] = True
    return mask


def _cand(
    candidate_id,
    x,
    y,
    *,
    volume_um3,
    xy_diameter_um,
    support_plane_count,
    status="preliminary_rule_pass",
    channel="green_signal",
    measurement_valid=True,
    invalid_coordinate=False,
    mask=None,
):
    inside = False
    if mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        inside = bool(mask[y, x])
    return {
        "candidate_id": candidate_id,
        "channel": channel,
        "section": 70,
        "x_local_px": x,
        "y_local_px": y,
        "x_global_px": x,
        "y_global_px": y,
        "volume_um3": volume_um3,
        "xy_diameter_um": xy_diameter_um,
        "support_plane_count": support_plane_count,
        "measurement_valid": measurement_valid,
        "invalid_coordinate": invalid_coordinate,
        "inside_tissue": inside,
        "current_status": status,
    }


def _refine(rows, mask, *, mode, thresholds, channel="green_signal"):
    return R.refine_candidates(
        rows,
        mask,
        voxel_zyx_um=VOXEL_ZYX_UM,
        mode=mode,
        thresholds=thresholds,
        channel=channel,
        section=70,
    )


# 1 + 2 -------------------------------------------------------------------- #
def test_tiny_one_plane_flagged_and_large_multiplane_retained():
    mask = _mask_left_half()
    rows = [
        _cand("tiny", 50, 50, volume_um3=30.0, xy_diameter_um=3.0,
              support_plane_count=1, mask=mask),
        _cand("big", 60, 60, volume_um3=4000.0, xy_diameter_um=20.0,
              support_plane_count=5, mask=mask),
    ]
    thresholds = R.RefinementThresholds(
        min_component_volume_um3=100.0, min_support_planes=2
    )
    result = _refine(rows, mask, mode="apply", thresholds=thresholds)
    by_id = {r["candidate_id"]: r for r in result.rows}

    assert by_id["tiny"]["refined_candidate_status"] == R.REFINED_FILTERED_SMALL
    assert by_id["tiny"]["size_filter_reason"]  # explains which thresholds failed
    assert by_id["big"]["refined_candidate_status"] == R.REFINED_RETAINED


# 3 ------------------------------------------------------------------------ #
def test_inside_tissue_candidate_with_clipped_patch_remains_eligible():
    mask = _mask_left_half()  # tissue x < 100
    # Centre inside tissue at x=98; the component's measurement patch crosses the
    # tissue boundary at x=100 but the candidate must stay eligible.
    rows = [
        _cand("edge_in", 98, 50, volume_um3=4000.0, xy_diameter_um=20.0,
              support_plane_count=5, mask=mask)
    ]
    thresholds = R.RefinementThresholds(
        min_component_volume_um3=100.0, min_support_planes=2
    )
    row = _refine(rows, mask, mode="apply", thresholds=thresholds).rows[0]

    assert row["centre_inside_tissue"] is True
    assert row["measurement_patch_clipped"] is True
    # Not auto-discarded: it meets size thresholds, so it is retained.
    assert row["refined_candidate_status"] == R.REFINED_RETAINED


def test_inside_tissue_small_but_clipped_goes_to_manual_review_edge():
    mask = _mask_left_half()
    rows = [
        _cand("small_edge", 98, 50, volume_um3=30.0, xy_diameter_um=6.0,
              support_plane_count=1, mask=mask)
    ]
    thresholds = R.RefinementThresholds(
        min_component_volume_um3=100.0, min_support_planes=2
    )
    row = _refine(rows, mask, mode="apply", thresholds=thresholds).rows[0]

    assert row["centre_inside_tissue"] is True
    assert row["measurement_patch_clipped"] is True
    # Small AND clipped: not filtered_too_small (would auto-discard a valid edge
    # cell); routed to manual review instead.
    assert row["refined_candidate_status"] == R.REFINED_MANUAL_REVIEW_EDGE


# 4 ------------------------------------------------------------------------ #
def test_edge_clipped_patch_uses_only_valid_pixels():
    mask = _mask_left_half(height=20, width=20, split=10)  # tissue x < 10
    # Patch centred at (y=5, x=9), half=3 -> intended 7x7 = 49 pixels. Only the
    # in-image, in-tissue pixels (x in 6..9, y in 2..8) are counted: 4*7 = 28.
    patch = R.measure_patch_validity(mask, cy=5, cx=9, half_px=3)
    assert patch["full_count"] == 49
    assert patch["valid_count"] == 28
    assert patch["clipped"] is True
    assert patch["valid_pixel_fraction"] == pytest.approx(28 / 49)

    # Image-boundary clipping: patch centred at the corner uses only in-bounds
    # pixels (0..4 in each axis = 25), never reading outside the image.
    corner = R.measure_patch_validity(mask, cy=1, cx=1, half_px=3)
    assert corner["valid_count"] == 25
    assert corner["clipped"] is True


# 5 ------------------------------------------------------------------------ #
def test_candidate_just_outside_mask_is_rescued():
    mask = _mask_left_half()  # tissue x < 100
    rows = [
        _cand("near_out", 103, 50, volume_um3=200.0, xy_diameter_um=8.0,
              support_plane_count=3, status="preliminary_rule_fail", mask=mask)
    ]
    thresholds = R.RefinementThresholds(
        min_support_planes=1, edge_rescue_distance_um=10.0
    )
    row = _refine(rows, mask, mode="apply", thresholds=thresholds).rows[0]

    assert row["centre_inside_tissue"] is False
    assert row["distance_to_tissue_um"] <= 10.0
    assert row["edge_rescued"] is True
    assert row["refined_candidate_status"] == R.REFINED_MANUAL_REVIEW_EDGE


# 6 ------------------------------------------------------------------------ #
def test_candidate_far_outside_mask_is_not_rescued():
    mask = _mask_left_half()
    rows = [
        _cand("far_out", 180, 50, volume_um3=200.0, xy_diameter_um=8.0,
              support_plane_count=3, status="preliminary_rule_fail", mask=mask)
    ]
    thresholds = R.RefinementThresholds(
        min_support_planes=1, edge_rescue_distance_um=10.0
    )
    row = _refine(rows, mask, mode="apply", thresholds=thresholds).rows[0]

    assert row["centre_inside_tissue"] is False
    assert row["edge_rescued"] is False
    # Exclusion preserved: original status is unchanged.
    assert row["refined_candidate_status"] == "preliminary_rule_fail"


# 10 ----------------------------------------------------------------------- #
def test_report_mode_applies_no_threshold():
    mask = _mask_left_half()
    rows = [
        _cand("t", 50, 50, volume_um3=1.0, xy_diameter_um=1.0,
              support_plane_count=1, mask=mask),
        _cand("b", 60, 60, volume_um3=9000.0, xy_diameter_um=25.0,
              support_plane_count=6, mask=mask),
    ]
    # Even with thresholds passed in, report mode must not apply them.
    result = _refine(
        rows,
        mask,
        mode="report",
        thresholds=R.RefinementThresholds(min_support_planes=99),
    )
    for row in result.rows:
        assert row["refined_candidate_status"] == row["original_status"]
        assert row["size_filter_status"] == "not_applied"
    # No candidate was filtered/rescued/invalidated.
    refined = {r["refined_candidate_status"] for r in result.rows}
    assert R.REFINED_FILTERED_SMALL not in refined
    assert "threshold_table" in result.summary


# 11 ----------------------------------------------------------------------- #
def test_apply_mode_requires_explicit_size_threshold():
    mask = _mask_left_half()
    rows = [
        _cand("a", 50, 50, volume_um3=100.0, xy_diameter_um=10.0,
              support_plane_count=3, mask=mask)
    ]
    with pytest.raises(ValueError):
        _refine(rows, mask, mode="apply", thresholds=R.RefinementThresholds())
    # An edge-rescue distance alone is not a size threshold.
    with pytest.raises(ValueError):
        _refine(
            rows,
            mask,
            mode="apply",
            thresholds=R.RefinementThresholds(edge_rescue_distance_um=15.0),
        )


# ---- run-level fixtures for 7, 8, 9 -------------------------------------- #
def _write_two_channel_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    channels = ("green_signal", "channel_2_signal")
    mask = _mask_left_half(120, 120, split=80)

    with (run_dir / "all_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        rng = np.random.default_rng(3)
        for channel in channels:
            for index in range(40):
                x = int(rng.integers(5, 110))
                y = int(rng.integers(5, 110))
                writer.writerow(
                    _cand(
                        f"{channel}_{index}",
                        x,
                        y,
                        volume_um3=float(rng.uniform(20, 5000)),
                        xy_diameter_um=float(rng.uniform(3, 22)),
                        support_plane_count=int(rng.integers(1, 6)),
                        channel=channel,
                        mask=mask,
                    )
                )
    for channel in channels:
        section_dir = run_dir / "qc" / f"{channel}_section_070"
        section_dir.mkdir(parents=True)
        np.save(section_dir / "tissue_mask.npy", mask)

    # A stand-in raw TIFF and an existing run output to prove they stay intact.
    (run_dir / "raw_plane_01.tif").write_bytes(b"RAWTIFF-DO-NOT-TOUCH")
    (run_dir / "candidate_status_summary.csv").write_text(
        "status,count\npreliminary_rule_pass,40\n", encoding="utf-8"
    )
    return run_dir, channels


def _refine_channel_to_out(run_dir, out_dir, channel):
    rows = cli._read_candidates(run_dir / "all_candidates.csv", channel, 70)
    mask = np.load(
        run_dir / "qc" / f"{channel}_section_070" / "tissue_mask.npy", mmap_mode="r"
    )
    result = R.refine_candidates(
        rows,
        mask,
        voxel_zyx_um=VOXEL_ZYX_UM,
        mode="report",
        thresholds=R.RefinementThresholds(),
        channel=channel,
        section=70,
    )
    channel_dir = cli._prepare_output_root(out_dir, run_dir, channel)
    R.write_refinement_outputs(channel_dir, result, make_plots=True)
    return channel_dir


# 7 ------------------------------------------------------------------------ #
def test_audit_csv_contains_every_original_candidate_and_status(tmp_path):
    run_dir, _channels = _write_two_channel_run(tmp_path)
    out_dir = tmp_path / "refine_out"
    channel_dir = _refine_channel_to_out(run_dir, out_dir, "green_signal")

    original_ids = {
        row["candidate_id"]
        for row in cli._read_candidates(run_dir / "all_candidates.csv", "green_signal", 70)
    }
    with (channel_dir / R.AUDIT_CSV).open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    audit_ids = {row["candidate_id"] for row in audit_rows}

    assert audit_ids == original_ids
    assert all("original_status" in row for row in audit_rows)
    assert all(row["original_status"] for row in audit_rows)


# 8 ------------------------------------------------------------------------ #
def test_green_and_red_outputs_remain_separate(tmp_path):
    run_dir, channels = _write_two_channel_run(tmp_path)
    out_dir = tmp_path / "refine_out"
    green_dir = _refine_channel_to_out(run_dir, out_dir, "green_signal")
    red_dir = _refine_channel_to_out(run_dir, out_dir, "channel_2_signal")

    assert green_dir != red_dir
    for channel, channel_dir in (("green_signal", green_dir), ("channel_2_signal", red_dir)):
        for name in (
            R.AUDIT_CSV,
            R.PLOT_SIZE,
            R.PLOT_SUPPORT,
            R.PLOT_SIZE_VS_EDGE,
            R.PLOT_EDGE_QC,
        ):
            assert (channel_dir / name).is_file()
        with (channel_dir / R.AUDIT_CSV).open(newline="", encoding="utf-8") as handle:
            assert {row["channel"] for row in csv.DictReader(handle)} == {channel}


# 9 ------------------------------------------------------------------------ #
def _snapshot(directory: Path) -> dict:
    snapshot = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(directory))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return snapshot


def test_raw_tiffs_and_existing_run_outputs_are_unmodified(tmp_path):
    run_dir, _channels = _write_two_channel_run(tmp_path)
    out_dir = tmp_path / "refine_out"  # separate from run_dir

    before = _snapshot(run_dir)
    _refine_channel_to_out(run_dir, out_dir, "green_signal")
    _refine_channel_to_out(run_dir, out_dir, "channel_2_signal")
    after = _snapshot(run_dir)

    assert before == after  # nothing under run_dir changed
    assert (run_dir / "raw_plane_01.tif").read_bytes() == b"RAWTIFF-DO-NOT-TOUCH"
    assert out_dir.exists() and any(out_dir.iterdir())  # outputs went elsewhere


def test_cli_parser_defaults_thresholds_to_none():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--run-dir", "r", "--channel", "green_signal", "--section", "70",
            "--out-dir", "o", "--mode", "report",
        ]
    )
    assert args.min_component_area_um2 is None
    assert args.min_component_volume_um3 is None
    assert args.min_support_planes is None
    assert args.edge_rescue_distance_um is None
    assert args.mode == "report"
