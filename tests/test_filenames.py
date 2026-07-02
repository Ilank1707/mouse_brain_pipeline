"""Filename parsing and sorting tests (pure standard-library)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_brain_pipeline.filenames import (  # noqa: E402
    compile_regex,
    expected_planes,
    numeric_sort_key,
    parse_filename,
)


def test_leading_zero_parse():
    p = parse_filename("section_070_01.tif")
    assert p is not None
    assert p.section == 70  # leading zeros stripped
    assert p.plane == 1


def test_tiff_double_extension():
    assert parse_filename("section_005_07.tiff") is not None


def test_case_insensitive_extension():
    assert parse_filename("SECTION_005_07.TIF") is not None


def test_unparseable_returns_none():
    assert parse_filename("not_a_section.tif") is None
    assert parse_filename("section_07.tif") is None  # missing plane field
    assert parse_filename("random.txt") is None


def test_numeric_sort_not_alphabetical():
    names = ["section_1000_01.tif", "section_099_07.tif", "section_099_01.tif", "section_100_01.tif"]
    parsed = [parse_filename(n) for n in names]
    ordered = sorted(parsed, key=numeric_sort_key)
    ordered_keys = [(p.section, p.plane) for p in ordered]
    # Numerically 99 < 100 < 1000; alphabetical would wrongly put "1000" before "099".
    assert ordered_keys == [(99, 1), (99, 7), (100, 1), (1000, 1)]


def test_seven_planes_per_section():
    assert expected_planes(7) == [1, 2, 3, 4, 5, 6, 7]


def test_missing_plane_detection_logic():
    present = {1, 2, 3, 5, 6, 7}  # plane 4 missing
    missing = sorted(set(expected_planes(7)) - present)
    assert missing == [4]


def test_duplicate_keys_collapse_to_same_section_plane():
    a = parse_filename("section_070_03.tif")
    b = parse_filename("nested/section_070_03.tif".split("/")[-1])
    assert numeric_sort_key(a) == numeric_sort_key(b)  # duplicates share a key


def test_custom_regex_compiles():
    pat = compile_regex(r"sec(?P<section>\d+)_p(?P<plane>\d+)\.tif$")
    p = parse_filename("sec12_p03.tif", pat)
    assert p is not None and p.section == 12 and p.plane == 3
