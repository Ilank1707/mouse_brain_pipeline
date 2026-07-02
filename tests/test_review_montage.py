"""Tests for the seven-plane candidate reviewer.

These prove the behaviour the manual-review workflow depends on, using small
synthetic 16-bit TIFFs (skipped automatically if numpy/tifffile are absent):

1. The seven planes load in order from ``_01`` to ``_07``.
2. Every patch uses the same global XY centre (no per-plane recentring).
3. The peak plane is highlighted correctly.
4. The maximum projection does not alter the raw stack.
5. A candidate near the image boundary is padded safely.
6. Labels resume correctly from ``manual_labels.csv``.
7. Raw TIFF files are never written to.
8. Both biological channels are reviewed independently.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from mouse_brain_pipeline.audit import ChannelIndex  # noqa: E402
from mouse_brain_pipeline.review import (  # noqa: E402
    MANUAL_LABEL_COLUMNS,
    load_manual_labels,
    previous_label,
    save_manual_label,
    unreviewed_candidates,
)
from mouse_brain_pipeline.review_patches import (  # noqa: E402
    colour_coded_z_projection,
    load_fixed_xy_stack,
    max_intensity_projection,
    ordered_section_planes,
    panel_highlight_class,
    parse_peak_index,
    parse_support_indices,
)


def _write_planes(directory, section=70, size=64, marker=None):
    """Write seven uint16 planes; return {(section, plane): path}.

    ``marker`` is an optional ``(y, x)`` global pixel set bright in every plane.
    """
    files = {}
    for plane in range(1, 8):
        image = np.full((size, size), 100 + plane, dtype=np.uint16)
        if marker is not None:
            image[marker[0], marker[1]] = 60000
        path = Path(directory) / f"section_{section:03d}_{plane:02d}.tif"
        tifffile.imwrite(path, image)
        files[(section, plane)] = path
    return files


# 1 ------------------------------------------------------------------------- #
def test_seven_planes_load_in_order_01_to_07(tmp_path):
    files = _write_planes(tmp_path)
    # Insert keys deliberately out of order so a naive dict iteration would fail.
    shuffled = dict(sorted(files.items(), key=lambda kv: kv[0][1], reverse=True))
    index = ChannelIndex("green_signal", tmp_path, files=shuffled)

    ordered = ordered_section_planes(index, 70)

    assert [plane for plane, _ in ordered] == [1, 2, 3, 4, 5, 6, 7]
    assert [Path(p).name for _, p in ordered] == [
        f"section_070_{plane:02d}.tif" for plane in range(1, 8)
    ]


def test_loaded_stack_preserves_16_bit(tmp_path):
    files = _write_planes(tmp_path)
    index = ChannelIndex("green_signal", tmp_path, files=files)
    stack, _ = load_fixed_xy_stack(ordered_section_planes(index, 70), 32, 32, 8)
    assert stack.dtype == np.uint16
    assert stack.shape == (7, 17, 17)


# 2 ------------------------------------------------------------------------- #
def test_every_patch_uses_the_same_xy_centre(tmp_path):
    # A bright pixel at the same global offset from the candidate in every plane
    # must land at the same patch coordinate -- proving a single shared centre.
    centre_y, centre_x, half = 40, 30, 8
    files = _write_planes(tmp_path, size=80, marker=(centre_y + 3, centre_x - 2))
    index = ChannelIndex("green_signal", tmp_path, files=files)

    stack, (cy, cx) = load_fixed_xy_stack(
        ordered_section_planes(index, 70), centre_x, centre_y, half
    )

    assert (cy, cx) == (half, half)
    # argmax (the bright marker) is at an identical patch location in all planes.
    positions = {tuple(np.argwhere(plane == plane.max())[0]) for plane in stack}
    assert len(positions) == 1
    assert positions.pop() == (half + 3, half - 2)


# 3 ------------------------------------------------------------------------- #
def test_peak_plane_is_highlighted_correctly():
    candidate = {"fixed_xy_peak_z_index": "3", "fixed_xy_support_z_indices": "2;3;4"}
    peak = parse_peak_index(candidate)
    support = parse_support_indices(candidate)

    assert peak == 3
    assert support == {2, 3, 4}
    assert panel_highlight_class(3, peak, support) == "peak"     # peak wins
    assert panel_highlight_class(2, peak, support) == "support"
    assert panel_highlight_class(4, peak, support) == "support"
    assert panel_highlight_class(0, peak, support) == "none"


# 4 ------------------------------------------------------------------------- #
def test_max_projection_does_not_alter_raw_stack():
    rng = np.random.default_rng(0)
    stack = (rng.integers(0, 5000, size=(7, 16, 16))).astype(np.uint16)
    before = stack.copy()

    mip = max_intensity_projection(stack)

    assert np.array_equal(stack, before)              # raw stack untouched
    assert mip is not stack
    assert np.array_equal(mip, stack.max(axis=0))     # correct projection
    # The colour overlay is also display-only and must not mutate the stack.
    colour_coded_z_projection(stack)
    assert np.array_equal(stack, before)


# 5 ------------------------------------------------------------------------- #
def test_boundary_candidate_is_padded_safely(tmp_path):
    files = _write_planes(tmp_path, size=64)
    index = ChannelIndex("green_signal", tmp_path, files=files)
    half = 6

    # Candidate at the very corner: three quadrants of the window fall off-image.
    stack, (cy, cx) = load_fixed_xy_stack(
        ordered_section_planes(index, 70), 0, 0, half
    )

    assert stack.shape == (7, 2 * half + 1, 2 * half + 1)
    assert (cy, cx) == (half, half)
    # Out-of-image region is zero-padded; the in-image quadrant keeps real data.
    assert stack[0, :half, :half].max() == 0                # top-left padding
    assert stack[0, half, half] == 101                      # candidate pixel (plane 1)
    # A candidate fully outside still yields a safe all-pad patch, no exception.
    far, _ = load_fixed_xy_stack(ordered_section_planes(index, 70), 10_000, 10_000, half)
    assert far.shape == (7, 2 * half + 1, 2 * half + 1)
    assert far.max() == 0


# 6 ------------------------------------------------------------------------- #
def test_labels_resume_correctly_from_csv(tmp_path):
    path = tmp_path / "manual_labels.csv"
    candidates = [
        {"candidate_id": f"c{i}", "channel": "green_signal", "section": 70,
         "x_global_px": 100 + i, "y_global_px": 200, "z_index": 3}
        for i in range(4)
    ]
    save_manual_label(path, candidates[0], "cell", "rev")
    save_manual_label(path, candidates[2], "artefact", "rev")

    labels = load_manual_labels(path)
    remaining = unreviewed_candidates(candidates, labels)

    assert [c["candidate_id"] for c in remaining] == ["c1", "c3"]
    assert previous_label(labels, candidates[0]) == "cell"
    assert previous_label(labels, candidates[2]) == "artefact"
    assert previous_label(labels, candidates[1]) is None

    # File holds exactly the required schema, in order, and resumes round-trip.
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert header == MANUAL_LABEL_COLUMNS
    assert header == [
        "candidate_id", "channel", "section", "x_global_px", "y_global_px",
        "z_index", "manual_label", "reviewer", "timestamp",
    ]
    row0 = labels[("c0", "green_signal")]
    assert row0["x_global_px"] == "100" and row0["z_index"] == "3"


# 7 ------------------------------------------------------------------------- #
def test_raw_tiffs_are_never_written(tmp_path):
    files = _write_planes(tmp_path, size=48, marker=(20, 24))
    index = ChannelIndex("green_signal", tmp_path, files=files)
    before = {
        path: (hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns)
        for path in files.values()
    }
    time.sleep(0.01)

    ordered = ordered_section_planes(index, 70)
    stack, _ = load_fixed_xy_stack(ordered, 24, 20, 8)
    max_intensity_projection(stack)
    colour_coded_z_projection(stack)

    for path, (digest, mtime) in before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert os.stat(path).st_mtime_ns == mtime


# 8 ------------------------------------------------------------------------- #
def test_both_channels_are_reviewed_independently(tmp_path):
    path = tmp_path / "manual_labels.csv"
    green = {"candidate_id": "c1", "channel": "green_signal", "section": 70,
             "x_global_px": 10, "y_global_px": 20, "z_index": 3}
    ch2 = {"candidate_id": "c1", "channel": "channel_2_signal", "section": 70,
           "x_global_px": 10, "y_global_px": 20, "z_index": 3}

    save_manual_label(path, green, "cell", "rev")
    labels = load_manual_labels(path)

    # Same candidate_id in the other channel is still unreviewed and unlabelled.
    assert previous_label(labels, green) == "cell"
    assert previous_label(labels, ch2) is None
    assert unreviewed_candidates([green], labels) == []
    assert unreviewed_candidates([ch2], labels) == [ch2]

    # Labelling channel_2 does not disturb the green label: two distinct rows.
    save_manual_label(path, ch2, "artefact", "rev")
    labels = load_manual_labels(path)
    assert labels[("c1", "green_signal")]["manual_label"] == "cell"
    assert labels[("c1", "channel_2_signal")]["manual_label"] == "artefact"
    assert len(labels) == 2
