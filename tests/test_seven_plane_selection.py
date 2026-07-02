"""Tests for candidate-file selection (cropped vs full-section) and the new
marker / display modes of the seven-plane QC renderer.

Proves:

1. A cropped candidate CSV is rejected for a full-section render.
2. The latest valid full-section CSV is selected correctly (by run-metadata
   timestamp, not folder modification time).
3. The renderer reports the exact number of loaded candidates.
4. ``marker-mode all`` displays every candidate on every plane.
5. ``marker-mode support`` displays only candidates supported on that plane.
6. Display scaling does not alter the raw arrays / TIFFs (incl. per-plane-robust).
7. Saved images retain the original TIFF dimensions.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from mouse_brain_pipeline.audit import ChannelIndex  # noqa: E402
from mouse_brain_pipeline.seven_plane_qc import (  # noqa: E402
    classify_run_crop,
    count_mismatch_warning,
    crop_covers_full_image,
    find_latest_full_section_csv,
    ordered_section_planes,
    recorded_candidate_count,
    render_section_seven_planes,
    run_is_full_section,
    select_candidate_rows,
)

HEIGHT, WIDTH = 60, 90


def _fixed():
    return SimpleNamespace(mode="fixed", minimum=0.0, maximum=513.0,
                           lower_percentile=0.5, upper_percentile=99.7)


def _percentile():
    return SimpleNamespace(mode="robust_tissue_percentile", minimum=0.0, maximum=513.0,
                           lower_percentile=0.5, upper_percentile=99.7)


def _write_section(directory, channel="green_signal", section=70, seed=0):
    files = {}
    rng = np.random.default_rng(seed)
    Path(directory).mkdir(parents=True, exist_ok=True)
    for plane in range(1, 8):
        img = rng.integers(50, 400, size=(HEIGHT, WIDTH)).astype(np.uint16)
        img[20:40, 30:60] = 9000  # a bright "injection-like" patch
        path = Path(directory) / f"section_{section:03d}_{plane:02d}.tif"
        tifffile.imwrite(path, img)
        files[(section, plane)] = path
    return ChannelIndex(channel, directory, files=files)


def _candidate(cid="c1", x=45, y=30, peak=3, support="2;3;4",
               status="preliminary_rule_pass", section=70, channel="green_signal"):
    return {
        "candidate_id": cid, "channel": channel, "section": section,
        "x_global_px": x, "y_global_px": y,
        "fixed_xy_peak_z_index": str(peak), "fixed_xy_support_z_indices": support,
        "current_status": status,
    }


def _write_run(root, name, *, crop_mode, crop, processed, channels,
               timestamp, counts=None):
    run = Path(root) / name
    run.mkdir(parents=True)
    (run / "all_candidates.csv").write_text("candidate_id,channel,section\n", encoding="utf-8")
    meta = {
        "run_timestamp_utc": timestamp,
        "crop_mode": crop_mode,
        "crop_x_min_x_max_y_min_y_max": crop,
        "processed_sections": processed,
        "candidate_counts_by_channel": counts or {c: 100 for c in channels},
    }
    (run / "candidate_run_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return run / "all_candidates.csv"


# 1 ------------------------------------------------------------------------- #
def test_cropped_csv_is_rejected_for_full_section_render():
    cropped = {
        "crop_mode": "xy_crop", "crop_x_min_x_max_y_min_y_max": [1000, 5000, 500, 4000],
        "processed_sections": [70], "candidate_counts_by_channel": {"green_signal": 600},
    }
    full = {
        "crop_mode": "full_xy_section", "crop_x_min_x_max_y_min_y_max": None,
        "processed_sections": [70], "candidate_counts_by_channel": {"green_signal": 12508},
    }
    assert classify_run_crop(cropped) == "crop"
    assert classify_run_crop(full) == "full"
    assert run_is_full_section(cropped, 70, "green_signal") == (False, "cropped_run")
    assert run_is_full_section(full, 70, "green_signal")[0] is True
    # A crop that spans the whole image IS acceptable as a full section.
    full_extent = {
        "crop_mode": "xy_crop", "crop_x_min_x_max_y_min_y_max": [0, WIDTH, 0, HEIGHT],
        "processed_sections": [70], "candidate_counts_by_channel": {"green_signal": 1},
    }
    assert crop_covers_full_image([0, WIDTH, 0, HEIGHT], WIDTH, HEIGHT) is True
    assert run_is_full_section(full_extent, 70, "green_signal",
                               image_width=WIDTH, image_height=HEIGHT)[0] is True
    # Right crop, but wrong section or wrong channel are rejected.
    assert run_is_full_section(full, 71, "green_signal")[1] == "section_not_processed"
    assert run_is_full_section(full, 70, "channel_2_signal")[1] == "channel_absent"


# 2 ------------------------------------------------------------------------- #
def test_latest_valid_full_section_csv_is_selected(tmp_path):
    # The correct full run has the NEWEST run timestamp but is created FIRST,
    # so it has the OLDEST folder mtime -- proving selection is not by mtime.
    target = _write_run(tmp_path, "full_new", crop_mode="full_xy_section", crop=None,
                        processed=[70], channels=["green_signal"],
                        timestamp="2026-06-20T10:00:00+00:00", counts={"green_signal": 12508})
    time.sleep(0.01)
    _write_run(tmp_path, "full_old", crop_mode="full_xy_section", crop=None,
               processed=[70], channels=["green_signal"],
               timestamp="2026-06-01T10:00:00+00:00")
    time.sleep(0.01)
    # Newest folder mtime, but cropped -> must be ignored.
    _write_run(tmp_path, "cropped_newest_folder", crop_mode="xy_crop",
               crop=[1000, 5000, 500, 4000], processed=[70], channels=["green_signal"],
               timestamp="2026-06-25T10:00:00+00:00")
    # Full but missing the section / channel -> ignored.
    _write_run(tmp_path, "full_wrong_section", crop_mode="full_xy_section", crop=None,
               processed=[71], channels=["green_signal"],
               timestamp="2026-06-30T10:00:00+00:00")
    _write_run(tmp_path, "full_wrong_channel", crop_mode="full_xy_section", crop=None,
               processed=[70], channels=["channel_2_signal"],
               timestamp="2026-06-30T10:00:00+00:00")

    found = find_latest_full_section_csv(tmp_path, 70, "green_signal",
                                         image_width=WIDTH, image_height=HEIGHT)
    assert found is not None
    csv_path, meta = found
    assert csv_path == target
    assert meta["candidate_counts_by_channel"]["green_signal"] == 12508


def test_count_mismatch_warning():
    assert count_mismatch_warning(600, 12508) is not None     # the bug scenario
    assert count_mismatch_warning(12508, 12508) is None
    assert count_mismatch_warning(12500, 12508) is None        # within tolerance
    assert count_mismatch_warning(600, None) is None
    assert recorded_candidate_count({"candidate_counts_by_channel": {"g": 5}}, "g") == 5


# 3 ------------------------------------------------------------------------- #
def test_renderer_reports_exact_loaded_candidate_count(tmp_path):
    index = _write_section(tmp_path / "g")
    rows = [_candidate(cid=f"c{i}", x=10 + i, y=20) for i in range(9)]
    rows.append(_candidate(cid="other_section", section=71))
    rows.append(_candidate(cid="other_channel", channel="channel_2_signal"))

    selected = select_candidate_rows(rows, "green_signal", 70)
    assert len(selected) == 9

    result = render_section_seven_planes(
        index, 70, rows, tmp_path / "qc",
        channel="green_signal", display_settings=_fixed(), mode="all",
    )
    assert result["candidate_count"] == 9   # only section 70 candidates


# 4 ------------------------------------------------------------------------- #
def test_marker_mode_all_displays_every_candidate_on_every_plane(tmp_path):
    index = _write_section(tmp_path / "g")
    rows = [
        _candidate("c1", x=45, y=30, peak=3, support="3"),
        _candidate("c2", x=20, y=40, peak=5, support="5"),
        _candidate("c3", x=70, y=15, peak=0, support="0"),
    ]
    result = render_section_seven_planes(
        index, 70, rows, tmp_path / "qc",
        channel="green_signal", display_settings=_fixed(), mode="all",
    )
    for row in result["metadata_rows"]:
        # Every candidate is visible on every plane, even unsupported ones.
        assert row["candidates_displayed"] == 3


# 5 ------------------------------------------------------------------------- #
def test_marker_mode_support_displays_only_supported(tmp_path):
    index = _write_section(tmp_path / "g")
    rows = [
        _candidate("c1", x=45, y=30, peak=3, support="2;3;4"),
        _candidate("c2", x=20, y=40, peak=5, support="5"),
    ]
    result = render_section_seven_planes(
        index, 70, rows, tmp_path / "qc",
        channel="green_signal", display_settings=_fixed(), mode="support",
    )
    by_plane = {r["optical_plane"]: r for r in result["metadata_rows"]}
    # plane 04 == z-index 3: c1 supported; plane 06 == z-index 5: c2 supported.
    assert by_plane[4]["candidates_displayed"] == 1
    assert by_plane[6]["candidates_displayed"] == 1
    assert by_plane[1]["candidates_displayed"] == 0   # z-index 0: neither supported
    for r in result["metadata_rows"]:
        assert r["candidates_displayed"] == r["candidates_supported_on_plane"]


# 6 + 7 --------------------------------------------------------------------- #
def test_per_plane_robust_keeps_raw_and_dimensions(tmp_path):
    from PIL import Image

    index = _write_section(tmp_path / "g")
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns)
        for _, path in ordered_section_planes(index, 70)
    }
    time.sleep(0.01)

    result = render_section_seven_planes(
        index, 70, [_candidate()], tmp_path / "qc",
        channel="green_signal", display_settings=_percentile(),
        mode="all", display_mode="per_plane_robust",
    )

    # Raw TIFFs are untouched.
    for path, (digest, mtime) in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert os.stat(path).st_mtime_ns == mtime
    # Per-plane windows were actually computed and recorded; dims preserved.
    assert result["display_mode"] == "per_plane_robust"
    for row, path in zip(result["metadata_rows"], result["plane_files"]):
        assert row["display_mode"] == "per_plane_robust"
        assert row["display_max"] > row["display_min"]
        assert (row["saved_width"], row["saved_height"]) == (WIDTH, HEIGHT)
        assert row["resizing_occurred"] is False
        with Image.open(path) as im:
            assert im.size == (WIDTH, HEIGHT)
