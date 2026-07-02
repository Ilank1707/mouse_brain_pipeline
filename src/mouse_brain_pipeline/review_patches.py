"""Pure-array helpers for the seven-plane candidate reviewer.

This module deliberately imports **no matplotlib** so that the geometry and I/O
logic stays unit-testable and the raw 16-bit TIFF values are never altered.

Invariants enforced here (and checked by ``tests/test_review_montage.py``):

* The seven optical planes load in ascending plane order ``_01 .. _07`` -- never
  alphabetically or by dict-insertion order.
* Every plane is cropped at the **same** global ``(x, y)`` centre; planes are
  never independently recentred.
* Crops keep the source dtype (16-bit stays 16-bit); display scaling happens
  separately and only for visualisation.
* Out-of-bounds crops (candidates near the image boundary) are zero-padded so a
  full ``(size, size)`` patch is always returned.
* The maximum-intensity projection and colour overlay are **display aids** that
  return new arrays and never mutate the raw ``(z, y, x)`` stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

PLANES_PER_SECTION = 7


# --------------------------------------------------------------------------- #
# Plane ordering / loading
# --------------------------------------------------------------------------- #
def ordered_section_planes(channel_index, section: int) -> list[tuple[int, Path]]:
    """Return ``[(plane_number, path), ...]`` for one section, ascending 01..07.

    ``channel_index`` is a :class:`mouse_brain_pipeline.audit.ChannelIndex` whose
    ``files`` map is keyed by ``(section, plane)``. Sorting by ``plane`` proves
    the seven planes are presented in physical optical order, independent of how
    they were discovered on disk.
    """
    section = int(section)
    return sorted(
        ((int(plane), path) for (sec, plane), path in channel_index.files.items()
         if int(sec) == section),
        key=lambda item: item[0],
    )


def load_fixed_xy_stack(
    ordered_planes: Iterable[tuple[int, Path]],
    x_global,
    y_global,
    half_px: int,
    *,
    pad_value: int = 0,
):
    """Load a fixed-XY crop from every plane in order, preserving 16-bit data.

    ``ordered_planes`` is the ``(plane, path)`` list from
    :func:`ordered_section_planes`. The **same** ``(x_global, y_global)`` centre
    is used for every plane: planes are never independently recentred. Crops that
    run past the image edge are zero-padded so the result is always
    ``(Z, size, size)`` with ``size = 2*half_px + 1`` and the candidate centred
    at ``(half_px, half_px)``.

    The returned stack keeps the source TIFF dtype (``uint16`` stays ``uint16``);
    raw values are copied out read-only and never written back.

    Returns ``(stack, (centre_y, centre_x))``.
    """
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    x = int(round(float(x_global)))
    y = int(round(float(y_global)))
    size = 2 * int(half_px) + 1
    patches: list = []
    dtype = None
    for _plane, path in ordered_planes:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            try:
                image = page.asarray(out="memmap")
            except (TypeError, ValueError):
                image = page.asarray()
            if dtype is None:
                dtype = image.dtype
            height, width = int(image.shape[-2]), int(image.shape[-1])
            target = np.full((size, size), pad_value, dtype=image.dtype)
            y0, y1 = max(0, y - half_px), min(height, y + half_px + 1)
            x0, x1 = max(0, x - half_px), min(width, x + half_px + 1)
            if y1 > y0 and x1 > x0:
                # np.asarray makes a writable copy; the memmap/source is untouched.
                crop = np.array(image[y0:y1, x0:x1], copy=True)
                offset_y = half_px - (y - y0)
                offset_x = half_px - (x - x0)
                target[offset_y:offset_y + crop.shape[0],
                       offset_x:offset_x + crop.shape[1]] = crop
            patches.append(target)
    if not patches:
        return np.zeros((0, size, size), dtype=dtype or np.uint16), (half_px, half_px)
    return np.stack(patches), (half_px, half_px)


# --------------------------------------------------------------------------- #
# Display-only projections (never mutate the raw stack)
# --------------------------------------------------------------------------- #
def max_intensity_projection(stack):
    """Maximum-intensity projection across Z.

    DISPLAY AID ONLY. A bright MIP must never on its own decide that a candidate
    is a cell -- it collapses the very Z structure the reviewer needs. The input
    stack is never mutated; a fresh array is returned.
    """
    import numpy as np  # noqa: PLC0415

    return np.asarray(stack).max(axis=0)


def _hue_to_rgb(hue: float) -> tuple[float, float, float]:
    """Fully-saturated HSV->RGB (s=v=1) without importing matplotlib."""
    sextant = int(hue * 6) % 6
    frac = hue * 6 - int(hue * 6)
    rising, falling = frac, 1.0 - frac
    return (
        (1.0, rising, 0.0),
        (falling, 1.0, 0.0),
        (0.0, 1.0, rising),
        (0.0, falling, 1.0),
        (rising, 0.0, 1.0),
        (1.0, 0.0, falling),
    )[sextant]


def colour_coded_z_projection(stack, *, lo_pct: float = 1.0, hi_pct: float = 99.8):
    """Colour-code a Z stack so each optical plane gets a distinct hue.

    DISPLAY AID ONLY: returns a float RGB image in ``[0, 1]`` for the optional
    aligned overlay. The raw 16-bit values are read but never modified.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(stack, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError("colour_coded_z_projection expects a (z, y, x) stack")
    z = arr.shape[0]
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    scaled = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    rgb = np.zeros(arr.shape[1:] + (3,), dtype=np.float32)
    for zi in range(z):
        colour = np.asarray(_hue_to_rgb(zi / max(1, z)), dtype=np.float32)
        rgb += scaled[zi][..., None] * colour[None, None, :]
    peak = float(rgb.max())
    if peak > 0:
        rgb = rgb / peak
    return np.clip(rgb, 0.0, 1.0)


def display_limits(array, lo_pct: float = 1.0, hi_pct: float = 99.8) -> tuple[float, float]:
    """Robust display window (percentile clip). Visualisation only -- the raw
    data and any measurement are never rescaled."""
    import numpy as np  # noqa: PLC0415

    flat = np.asarray(array)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def fixed_xy_central_intensity(stack, radius_px: int | None = None):
    """Mean intensity of the central disk in each plane, at the fixed XY centre.

    Because the stack is already cropped at one fixed ``(x, y)``, this is a
    genuine fixed-XY profile across Z -- it is always available even when a
    candidate row carries no pre-computed ``plane_*_contrast`` columns.
    """
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    _z, height, width = arr.shape
    cy, cx = height // 2, width // 2
    if radius_px is None:
        radius_px = max(1, min(cy, cx) // 4)
    yy, xx = np.ogrid[:height, :width]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px ** 2
    return arr[:, disk].mean(axis=1)


# --------------------------------------------------------------------------- #
# Peak / support parsing + montage highlighting
# --------------------------------------------------------------------------- #
def parse_peak_index(candidate: dict, default: int = 0) -> int:
    """Best-available fixed-XY peak optical-plane index for a candidate row."""
    for key in ("fixed_xy_peak_z_index", "peak_z_index", "z_index"):
        value = candidate.get(key, "")
        if value not in ("", None):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return default


def parse_support_indices(candidate: dict) -> set[int]:
    """Set of detected support optical-plane indices for a candidate row."""
    raw = candidate.get(
        "fixed_xy_support_z_indices",
        candidate.get("support_z_indices", candidate.get("z_indices", "")),
    )
    out: set[int] = set()
    for token in str(raw).replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            out.add(int(float(token)))
        except (TypeError, ValueError):
            continue
    return out


def panel_highlight_class(z: int, peak: int, support: Iterable[int]) -> str:
    """Classify one montage panel: ``'peak'`` (wins) > ``'support'`` > ``'none'``."""
    if int(z) == int(peak):
        return "peak"
    if int(z) in {int(s) for s in support}:
        return "support"
    return "none"


# Border colours shared by the montage and the single-plane scrubber.
HIGHLIGHT_COLOURS = {"peak": "#FFD400", "support": "#39FF14", "none": "#444444"}
