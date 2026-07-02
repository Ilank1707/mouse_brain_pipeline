"""Filename parsing, numerical sorting and relative-Z geometry.

This module is intentionally pure standard-library (regex + dataclasses) so it
can be imported, tested and run without numpy / tifffile / pyyaml installed.

Filename convention
-------------------
``section_070_01.tif``
    * ``070`` -> major physical cutting cycle (``section``)
    * ``01``  -> optical plane within that cycle (``plane``, valid 1..7)

Relative-Z geometry
-------------------
    global_plane = (section - first_section) * planes_per_section + (plane - 1)
    z_um         = global_plane * voxel_size_z_um

The formula is only meaningful when sections are *contiguous*; always validate
with :func:`validate_contiguous` before relying on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern

# Named groups <section> and <plane>; tolerant of .tif / .tiff.
DEFAULT_FILENAME_REGEX = r"section_(?P<section>\d+)_(?P<plane>\d+)\.tif{1,2}$"


@dataclass(frozen=True)
class ParsedFile:
    """A successfully parsed TIFF filename."""

    section: int
    plane: int
    name: str  # original filename (basename), preserved verbatim


def compile_regex(pattern: str = DEFAULT_FILENAME_REGEX) -> Pattern[str]:
    """Compile the filename regex case-insensitively with a clear error message."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid filename_regex {pattern!r}: {exc}") from exc


def parse_filename(name: str, regex: Pattern[str] | str = DEFAULT_FILENAME_REGEX) -> ParsedFile | None:
    """Parse a basename into :class:`ParsedFile`, or ``None`` if it does not match.

    Leading zeros are handled (``int`` strips them). Only the basename matters;
    callers should pass ``os.path.basename(path)``.
    """
    pat = regex if isinstance(regex, re.Pattern) else compile_regex(regex)
    m = pat.search(name)
    if m is None:
        return None
    try:
        section = int(m.group("section"))
        plane = int(m.group("plane"))
    except (IndexError, ValueError):
        return None
    return ParsedFile(section=section, plane=plane, name=name)


def numeric_sort_key(parsed: ParsedFile) -> tuple[int, int]:
    """Sort key that orders numerically by (section, plane), NOT alphabetically.

    Alphabetical sorting would place ``section_1000`` before ``section_99``;
    integer keys avoid that. 
    """
    return (parsed.section, parsed.plane)


def global_plane(section: int, first_section: int, plane: int, planes_per_section: int = 7) -> int:
    """Return the 0-based global optical-plane index.

    ``plane`` is 1-based (01..07). ``first_section`` anchors the stack so the
    first plane of the first section is global plane 0.
    """
    if plane < 1:
        raise ValueError(f"plane must be >= 1, got {plane}")
    return (section - first_section) * planes_per_section + (plane - 1)


def z_um(global_plane_index: int, voxel_size_z_um: float = 6.0) -> float:
    """Relative Z (micrometres) for a 0-based global plane index."""
    return global_plane_index * voxel_size_z_um


def expected_planes(planes_per_section: int = 7) -> list[int]:
    """The set of plane numbers every section must contain (1..N)."""
    return list(range(1, planes_per_section + 1))


def validate_contiguous(sections: Iterable[int]) -> tuple[bool, list[int]]:
    """Check that a set of section numbers forms a gap-free run.

    Returns ``(is_contiguous, missing_sections)``. ``missing_sections`` lists the
    section numbers that would be required to fill the range min..max.
    """
    uniq = sorted(set(sections))
    if not uniq:
        return True, []
    full = set(range(uniq[0], uniq[-1] + 1))
    missing = sorted(full - set(uniq))
    return (len(missing) == 0), missing
