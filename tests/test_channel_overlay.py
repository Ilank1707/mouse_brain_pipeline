"""Tests for the green/red cross-channel overlay analysis and the QC skip flags.

Covers, per the task:
  * overlay measurements are written (CSV + summary + QC png);
  * ``dominant_channel`` is assigned from MEASURED signal (green blob -> green,
    red blob -> red, flat -> unclear), and red is never forced down;
  * ``--fast-qc`` skips the expensive outputs;
  * default behaviour is unchanged unless a flag is used.

Uses small synthetic 16-bit TIFFs at non-square blob positions so an x/y swap
would be caught.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from mouse_brain_pipeline.channel_overlay import (  # noqa: E402
    BOTH,
    GREEN_DOMINANT,
    RED_DOMINANT,
    UNCLEAR,
    analyze_run,
    classify_dominant,
    red_green_ratio,
    signal_above_background,
    summarize_overlay,
)
from mouse_brain_pipeline.channels import CHANNEL_2_SIGNAL, GREEN_SIGNAL  # noqa: E402
from mouse_brain_pipeline.config import Config  # noqa: E402
from mouse_brain_pipeline.qc_options import resolve_qc_flags  # noqa: E402

HEIGHT, WIDTH = 130, 130


# --------------------------------------------------------------------------- #
# Pure dominant-channel logic
# --------------------------------------------------------------------------- #
def test_classify_dominant_covers_all_four_labels():
    common = dict(snr_threshold=3.0, dominance_ratio=1.5, green_valid=True, red_valid=True)
    # Green present & much stronger -> green.
    assert classify_dominant(10.0, 1.0, 900.0, 100.0, **common) == GREEN_DOMINANT
    # Red present & much stronger -> red (red is NOT penalised).
    assert classify_dominant(1.0, 10.0, 100.0, 900.0, **common) == RED_DOMINANT
    # Both present & comparable -> both.
    assert classify_dominant(9.0, 9.0, 500.0, 500.0, **common) == BOTH
    # Neither present -> unclear.
    assert classify_dominant(1.0, 1.0, 10.0, 10.0, **common) == UNCLEAR


def test_classify_dominant_present_in_only_one_channel():
    common = dict(snr_threshold=3.0, dominance_ratio=1.5, green_valid=True, red_valid=True)
    assert classify_dominant(8.0, 0.5, 400.0, 5.0, **common) == GREEN_DOMINANT
    assert classify_dominant(0.5, 8.0, 5.0, 400.0, **common) == RED_DOMINANT


def test_classify_dominant_both_measurements_invalid_is_unclear():
    assert classify_dominant(
        float("nan"), float("nan"), 0.0, 0.0,
        green_valid=False, red_valid=False, snr_threshold=3.0, dominance_ratio=1.5,
    ) == UNCLEAR


def test_signal_and_ratio_helpers():
    assert signal_above_background(500.0, 100.0) == 400.0
    assert signal_above_background(50.0, 100.0) == 0.0            # clamped
    assert signal_above_background(float("nan"), 100.0) == 0.0
    assert red_green_ratio(400.0, 200.0) == pytest.approx(0.5)
    assert red_green_ratio(0.0, 0.0) is None                     # undefined
    assert red_green_ratio(0.0, 5.0) == float("inf")             # red-only


def test_summarize_overlay_keeps_all_categories_and_groups():
    rows = [
        {"channel": GREEN_SIGNAL, "dominant_channel": GREEN_DOMINANT},
        {"channel": GREEN_SIGNAL, "dominant_channel": RED_DOMINANT},
        {"channel": CHANNEL_2_SIGNAL, "dominant_channel": RED_DOMINANT},
    ]
    summary = summarize_overlay(rows)
    by_key = {(r["detection_channel"], r["dominant_channel"]): r["count"] for r in summary}
    # Every group x every category is present (zeros kept).
    assert len(summary) == 3 * 4
    assert by_key[("all", GREEN_DOMINANT)] == 1
    assert by_key[("all", RED_DOMINANT)] == 2
    assert by_key[("all", UNCLEAR)] == 0
    assert by_key[(GREEN_SIGNAL, RED_DOMINANT)] == 1
    assert by_key[(CHANNEL_2_SIGNAL, RED_DOMINANT)] == 1


# --------------------------------------------------------------------------- #
# QC skip-flag resolution
# --------------------------------------------------------------------------- #
def test_default_qc_flags_unchanged_with_no_flags():
    qc = resolve_qc_flags()
    assert qc.write_fullres_qc is True          # full-res QC on by default
    assert qc.run_channel_overlay is True        # overlay on by default
    assert qc.skip_pair_correlation is False     # pair correlation on by default
    assert qc.render_seven_planes is False       # opt-in, still off
    assert qc.save_review_patches is False        # opt-in, still off


def test_fast_qc_skips_expensive_outputs():
    qc = resolve_qc_flags(fast_qc=True)
    assert qc.write_fullres_qc is False           # full-resolution QC skipped
    assert qc.render_seven_planes is False        # seven-plane skipped
    assert qc.save_review_patches is False        # review patches skipped
    # But NOT the overlay or pair correlation (not in the fast-qc bundle).
    assert qc.run_channel_overlay is True
    assert qc.skip_pair_correlation is False


def test_explicit_request_wins_over_fast_qc():
    qc = resolve_qc_flags(
        fast_qc=True, render_seven_planes=True, save_review_patches=True,
    )
    assert qc.render_seven_planes is True
    assert qc.save_review_patches is True


def test_explicit_skip_flags_always_suppress():
    qc = resolve_qc_flags(
        render_seven_planes=True, save_review_patches=True, fullres_seven_planes=True,
        skip_seven_plane_qc=True, skip_review_patches=True, skip_fullres_qc=True,
        skip_pair_correlation=True, skip_channel_overlay=True,
    )
    assert qc.render_seven_planes is False
    assert qc.save_review_patches is False
    assert qc.write_fullres_qc is False
    assert qc.fullres_seven_planes is False
    assert qc.skip_pair_correlation is True
    assert qc.run_channel_overlay is False


def test_skip_spatial_analysis_alias_skips_pair_correlation():
    assert resolve_qc_flags(skip_spatial_analysis=True).skip_pair_correlation is True


# --------------------------------------------------------------------------- #
# End-to-end overlay on synthetic data
# --------------------------------------------------------------------------- #
def _blob_plane(seed, blobs):
    """Background (non-zero) + optional bright disks at ``blobs`` = [(y, x, val)]."""
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 111, size=(HEIGHT, WIDTH)).astype(np.uint16)
    yy, xx = np.ogrid[:HEIGHT, :WIDTH]
    for cy, cx, val in blobs:
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= 5 ** 2
        img[disk] = val
    return img


def _write_channel(directory, blobs, *, section=70, seed0=0):
    directory.mkdir(parents=True, exist_ok=True)
    for plane in range(1, 8):
        img = _blob_plane(seed0 + plane, blobs)
        tifffile.imwrite(directory / f"section_{section:03d}_{plane:02d}.tif", img)


# Green blob at (y=40, x=30); red blob at (y=80, x=90). Non-square positions catch swaps.
_GREEN_XY = (30, 40)   # (x, y)
_RED_XY = (90, 80)
_FLAT_XY = (65, 65)


def _make_dataset(tmp_path):
    green_dir = tmp_path / "green"
    red_dir = tmp_path / "red"
    _write_channel(green_dir, [(_GREEN_XY[1], _GREEN_XY[0], 5000)], seed0=10)
    _write_channel(red_dir, [(_RED_XY[1], _RED_XY[0], 5000)], seed0=50)
    return green_dir, red_dir


def _write_candidates(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"candidate_id": "g1", "channel": GREEN_SIGNAL, "section": 70,
         "x_global_px": _GREEN_XY[0], "y_global_px": _GREEN_XY[1],
         "z_index": 3, "optical_plane": 4},
        {"candidate_id": "r1", "channel": CHANNEL_2_SIGNAL, "section": 70,
         "x_global_px": _RED_XY[0], "y_global_px": _RED_XY[1],
         "z_index": 3, "optical_plane": 4},
        {"candidate_id": "u1", "channel": GREEN_SIGNAL, "section": 70,
         "x_global_px": _FLAT_XY[0], "y_global_px": _FLAT_XY[1],
         "z_index": 3, "optical_plane": 4},
    ]
    fields = ["candidate_id", "channel", "section", "x_global_px", "y_global_px",
              "z_index", "optical_plane"]
    with open(run_dir / "all_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _config(green_dir, red_dir, work_dir):
    return Config.from_dict({
        "data": {
            "green_signal_dir": str(green_dir),
            "channel_2_signal_dir": str(red_dir),
            "work_dir": str(work_dir),
        },
    })


def test_analyze_run_writes_measurements_and_assigns_dominant_channel(tmp_path):
    green_dir, red_dir = _make_dataset(tmp_path)
    run_dir = tmp_path / "run"
    _write_candidates(run_dir)
    config = _config(green_dir, red_dir, tmp_path / "work")

    result = analyze_run(run_dir, config)

    # Overlay measurements + summary + QC image are written.
    measurements_csv = run_dir / "channel_overlay" / "channel_overlay_candidate_measurements.csv"
    summary_csv = run_dir / "channel_overlay" / "channel_overlay_summary.csv"
    qc_png = run_dir / "channel_overlay" / "green_red_overlay_qc.png"
    assert measurements_csv.is_file()
    assert summary_csv.is_file()
    assert qc_png.is_file()

    with open(measurements_csv, newline="", encoding="utf-8") as fh:
        rows = {r["candidate_id"]: r for r in csv.DictReader(fh)}

    # dominant_channel is assigned from the MEASURED signal.
    assert "dominant_channel" in rows["g1"]
    assert rows["g1"]["dominant_channel"] == GREEN_DOMINANT
    assert rows["r1"]["dominant_channel"] == RED_DOMINANT
    assert rows["u1"]["dominant_channel"] == UNCLEAR

    # Both channels are measured at every candidate (red is not forced down): the
    # red candidate really is strong in red and weak in green.
    assert float(rows["r1"]["red_peak"]) > float(rows["r1"]["green_peak"])
    assert float(rows["g1"]["green_peak"]) > float(rows["g1"]["red_peak"])

    # Summary reconciles with the three candidates.
    total = sum(r["count"] for r in result["summary"] if r["detection_channel"] == "all")
    assert total == 3


def test_analyze_run_respects_section_filter(tmp_path):
    green_dir, red_dir = _make_dataset(tmp_path)
    run_dir = tmp_path / "run"
    _write_candidates(run_dir)
    config = _config(green_dir, red_dir, tmp_path / "work")

    # No candidate is in section 999 -> no measured rows.
    result = analyze_run(run_dir, config, sections=[999], render_qc=False)
    assert result["candidate_count"] == 0
