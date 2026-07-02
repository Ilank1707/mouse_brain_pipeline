"""Tests for the whole-section seven-plane candidate QC renderer.

Uses small synthetic 16-bit TIFFs at non-square dimensions (so a width/height
swap would be caught) and proves:

1. Planes load in order from ``_01`` to ``_07``.
2. Each plane uses its original XY dimensions.
3. Candidate coordinates remain identical across planes.
4. Candidate support is matched to the correct plane.
5. Peak-plane highlighting is correct.
6. Rendering does not modify the raw arrays or TIFFs.
7. Full-resolution files are not resized.
8. Both biological channels work independently.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")

from mouse_brain_pipeline.audit import ChannelIndex  # noqa: E402
from mouse_brain_pipeline.seven_plane_qc import (  # noqa: E402
    candidate_draw_xy,
    candidate_plane_state,
    count_supported_on_plane,
    marker_spec,
    ordered_section_planes,
    render_plane_overlay,
    render_section_seven_planes,
)

# Deliberately non-square so a (w, h) swap is detectable.
HEIGHT, WIDTH = 60, 90


def _fixed(minimum=0.0, maximum=513.0):
    return SimpleNamespace(mode="fixed", minimum=minimum, maximum=maximum,
                           lower_percentile=0.5, upper_percentile=99.7)


def _percentile():
    return SimpleNamespace(mode="robust_tissue_percentile", minimum=0.0, maximum=513.0,
                           lower_percentile=0.5, upper_percentile=99.7)


def _write_section(directory, channel="green_signal", section=70, seed=0):
    files = {}
    rng = np.random.default_rng(seed)
    for plane in range(1, 8):
        img = rng.integers(50, 400, size=(HEIGHT, WIDTH)).astype(np.uint16)
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


# 1 ------------------------------------------------------------------------- #
def test_planes_load_in_order_01_to_07(tmp_path):
    index = _write_section(tmp_path)
    # Shuffle the dict so insertion order can't accidentally pass the test.
    index.files = dict(sorted(index.files.items(), key=lambda kv: kv[0][1], reverse=True))
    ordered = ordered_section_planes(index, 70)
    assert [plane for plane, _ in ordered] == [1, 2, 3, 4, 5, 6, 7]
    assert [Path(p).name for _, p in ordered] == [
        f"section_070_{p:02d}.tif" for p in range(1, 8)
    ]


# 2 + 7 --------------------------------------------------------------------- #
def test_each_plane_keeps_original_dimensions_and_is_not_resized(tmp_path):
    from PIL import Image

    index = _write_section(tmp_path)
    out = tmp_path / "qc"
    result = render_section_seven_planes(
        index, 70, [_candidate()], out,
        channel="green_signal", display_settings=_fixed(), mode="all",
    )
    assert len(result["plane_files"]) == 7
    for row, path in zip(result["metadata_rows"], result["plane_files"]):
        assert (row["source_width"], row["source_height"]) == (WIDTH, HEIGHT)
        assert (row["saved_width"], row["saved_height"]) == (WIDTH, HEIGHT)
        assert row["resizing_occurred"] is False
        with Image.open(path) as im:
            assert im.size == (WIDTH, HEIGHT)   # PIL size is (w, h)


# 3 ------------------------------------------------------------------------- #
def test_candidate_coordinates_are_identical_across_planes(tmp_path):
    candidate = _candidate(x=45, y=30)
    # The draw position is purely a function of the global coordinates.
    assert candidate_draw_xy(candidate) == (45, 30)

    index = _write_section(tmp_path)
    out = tmp_path / "qc"
    render_section_seven_planes(
        index, 70, [candidate], out,
        channel="green_signal", display_settings=_fixed(), mode="all",
    )
    # In mode 'all' the marker is drawn on every plane; locate the coloured
    # pixels and confirm their centroid is the same on plane 1 and plane 7.
    from PIL import Image

    def marker_centroid(path):
        arr = np.asarray(Image.open(path).convert("RGB"))
        coloured = (arr[:, :, 0] != arr[:, :, 1]) | (arr[:, :, 1] != arr[:, :, 2])
        ys, xs = np.nonzero(coloured)
        return (round(xs.mean()), round(ys.mean()))

    first = marker_centroid(out / "plane_01_candidates_fullres.png")
    last = marker_centroid(out / "plane_07_candidates_fullres.png")
    assert first == last
    assert abs(first[0] - 45) <= 3 and abs(first[1] - 30) <= 3


# 4 ------------------------------------------------------------------------- #
def test_support_is_matched_to_the_correct_plane():
    candidate = _candidate(peak=3, support="2;3;4")
    states = {z: candidate_plane_state(candidate, z) for z in range(7)}
    assert states[2] == "support"
    assert states[4] == "support"
    assert states[3] == "peak"
    assert states[0] == "unsupported"
    assert states[5] == "unsupported"
    assert count_supported_on_plane([candidate], 2) == 1
    assert count_supported_on_plane([candidate], 0) == 0


# 5 ------------------------------------------------------------------------- #
def test_peak_plane_highlighting_is_correct():
    candidate = _candidate(peak=3, support="2;3;4")
    # Peak wins over support, and only peak draws the strong centre dot.
    assert candidate_plane_state(candidate, 3) == "peak"
    assert marker_spec("peak", "all")["centre_dot"] is True
    assert marker_spec("support", "all")["centre_dot"] is False
    # support_only hides unsupported planes entirely but keeps peak/support.
    assert marker_spec("unsupported", "support_only") is None
    assert marker_spec("peak", "support_only") is not None


def test_support_only_mode_hides_unsupported_candidates(tmp_path):
    index = _write_section(tmp_path)
    out = tmp_path / "qc"
    candidate = _candidate(peak=3, support="3")  # supported on plane index 3 only
    result = render_section_seven_planes(
        index, 70, [candidate], out,
        channel="green_signal", display_settings=_fixed(), mode="support_only",
    )
    by_plane = {r["optical_plane"]: r for r in result["metadata_rows"]}
    assert by_plane[4]["candidates_displayed"] == 1   # plane 04 == z-index 3 (peak)
    assert by_plane[1]["candidates_displayed"] == 0   # unsupported -> hidden
    assert by_plane[1]["candidates_supported_on_plane"] == 0


# 6 ------------------------------------------------------------------------- #
def test_rendering_does_not_modify_raw_arrays_or_tiffs(tmp_path):
    index = _write_section(tmp_path)
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns)
        for _, path in ordered_section_planes(index, 70)
    }
    # Also confirm the in-memory windowing leaves its input array untouched.
    from mouse_brain_pipeline.qc_native import apply_window_uint8

    sample = np.asarray(tifffile.imread(next(iter(before)))).copy()
    sample_guard = sample.copy()
    apply_window_uint8(sample, 0, 513)
    assert np.array_equal(sample, sample_guard)

    render_section_seven_planes(
        index, 70, [_candidate()], tmp_path / "qc",
        channel="green_signal", display_settings=_fixed(), mode="all",
    )
    for path, (digest, mtime) in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert os.stat(path).st_mtime_ns == mtime


# 8 ------------------------------------------------------------------------- #
def test_both_channels_work_independently(tmp_path):
    green_dir = tmp_path / "green"
    ch2_dir = tmp_path / "ch2"
    green_dir.mkdir()
    ch2_dir.mkdir()
    green_index = _write_section(green_dir, channel="green_signal", seed=1)
    ch2_index = _write_section(ch2_dir, channel="channel_2_signal", seed=2)

    green = render_section_seven_planes(
        green_index, 70, [_candidate(channel="green_signal")], tmp_path / "green_qc",
        channel="green_signal", display_settings=_percentile(), mode="all",
    )
    # Channel 2 uses the Fiji-like fixed 0-513 display window.
    ch2 = render_section_seven_planes(
        ch2_index, 70, [_candidate(channel="channel_2_signal")], tmp_path / "ch2_qc",
        channel="channel_2_signal", display_settings=_fixed(0, 513), mode="all",
    )

    assert green["display_mode"] == "robust_tissue_percentile"
    assert (ch2["display_min"], ch2["display_max"]) == (0.0, 513.0)
    assert ch2["display_mode"] == "fixed"
    # Independent output folders, both complete seven-plane sets.
    assert len(green["plane_files"]) == 7 and len(ch2["plane_files"]) == 7
    assert green["plane_files"][0].parent != ch2["plane_files"][0].parent
    assert all(p.exists() for p in green["plane_files"] + ch2["plane_files"])
