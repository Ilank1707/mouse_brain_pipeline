"""Relative-Z geometry tests (pure standard-library)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_brain_pipeline.filenames import (  # noqa: E402
    global_plane,
    validate_contiguous,
    z_um,
)


def test_global_plane_first_plane_is_zero():
    assert global_plane(section=70, first_section=70, plane=1, planes_per_section=7) == 0


def test_global_plane_within_first_section():
    # plane 7 of the first section -> global index 6
    assert global_plane(70, 70, 7, 7) == 6


def test_global_plane_second_section():
    # first plane of the next section -> 7
    assert global_plane(71, 70, 1, 7) == 7
    # last plane of the next section -> 13
    assert global_plane(71, 70, 7, 7) == 13


def test_z_um_spacing():
    # 6 um spacing between optical planes
    assert z_um(0, 6.0) == 0.0
    assert z_um(1, 6.0) == 6.0
    assert z_um(7, 6.0) == 42.0  # one full physical cut (7 planes)


def test_z_um_matches_cut_thickness():
    # Crossing into the next section equals the physical cut thickness (42 um).
    gp = global_plane(71, 70, 1, 7)
    assert z_um(gp, 6.0) == 42.0


def test_contiguous_sections():
    ok, missing = validate_contiguous([70, 71, 72])
    assert ok and missing == []


def test_non_contiguous_sections_detected():
    ok, missing = validate_contiguous([70, 72, 73])
    assert not ok and missing == [71]


def test_plane_must_be_positive():
    try:
        global_plane(70, 70, 0, 7)
    except ValueError:
        return
    raise AssertionError("expected ValueError for plane < 1")
