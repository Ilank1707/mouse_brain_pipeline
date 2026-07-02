"""Radial candidate distance / density analysis around the injection centre.

Pure, array-level maths (no file I/O) so it is unit-testable. Distances use the
acquisition XY pixel size (~1.004 um). Tissue area per annulus is measured from
the tissue MASK only -- never the full mathematical ring area -- so density is
candidates per mm2 of real brain.

Everything here is about PROVISIONAL candidates. A preliminary-rule pass is not a
cell; only genuine confirmed cells are ever labelled as cells.
"""

from __future__ import annotations

import math

RADIAL_COORDINATE_COLUMNS = [
    "candidate_id",
    "channel",
    "section",
    "x_global_px",
    "y_global_px",
    "current_status",
    "inside_injection_analysis_exclusion",
    "dx_px",
    "dy_px",
    "radial_distance_px",
    "radial_distance_um",
    "radial_bin_start_um",
    "radial_bin_end_um",
]

RADIAL_COUNT_COLUMNS = [
    "series",
    "radial_bin_start_um",
    "radial_bin_end_um",
    "count",
    "tissue_area_px",
    "tissue_area_mm2",
    "density_per_mm2",
    "fraction",
    "cumulative_count",
    "cumulative_fraction",
]


def radial_distances_um(xs, ys, center_xy, voxel_yx_um):
    """Per-point distance (um) to ``center_xy`` using the XY pixel size."""
    import numpy as np  # noqa: PLC0415

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    vy, vx = float(voxel_yx_um[0]), float(voxel_yx_um[1])
    return np.hypot((xs - cx) * vx, (ys - cy) * vy)


def bin_index(dist_um, bin_width_um):
    import numpy as np  # noqa: PLC0415

    return np.floor(np.asarray(dist_um, dtype=float) / float(bin_width_um)).astype(int)


def counts_by_bin(dist_um, bin_width_um, n_bins):
    """Histogram distances into ``n_bins`` equal-width annuli (out-of-range dropped)."""
    import numpy as np  # noqa: PLC0415

    idx = bin_index(dist_um, bin_width_um)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    if idx.size == 0:
        return np.zeros(n_bins, dtype=int)
    return np.bincount(idx, minlength=n_bins)[:n_bins]


def tissue_area_by_bin(tissue_mask, center_xy_local, voxel_yx_um, bin_width_um, n_bins):
    """Tissue-pixel count per annulus, measured ONLY inside the tissue mask.

    ``center_xy_local`` is (x, y) in the mask's own (crop-local) pixel frame.
    """
    import numpy as np  # noqa: PLC0415

    ys, xs = np.nonzero(np.asarray(tissue_mask, dtype=bool))
    if ys.size == 0:
        return np.zeros(n_bins, dtype=int)
    dist = radial_distances_um(xs, ys, center_xy_local, voxel_yx_um)
    return counts_by_bin(dist, bin_width_um, n_bins)


def resolve_n_bins(bin_width_um, maximum_radius_um=None, max_distance_um=None):
    """Number of equal-width bins covering the data (or the configured max radius)."""
    if maximum_radius_um is not None:
        span = float(maximum_radius_um)
    elif max_distance_um is not None:
        span = float(max_distance_um)
    else:
        span = float(bin_width_um)
    return max(1, int(math.ceil((span + 1e-9) / float(bin_width_um))))


def assemble_series(counts, tissue_area_px, bin_width_um, voxel_area_um2):
    """Combine per-bin candidate counts + tissue area into the full series rows.

    density = count / tissue_area_mm2 (undefined -> NaN where an annulus has no
    tissue, so an empty annulus never divides by zero).
    """
    import numpy as np  # noqa: PLC0415

    counts = np.asarray(counts, dtype=float)
    area_px = np.asarray(tissue_area_px, dtype=float)
    n = len(counts)
    area_mm2 = area_px * float(voxel_area_um2) / 1.0e6

    density = np.full(n, np.nan)
    has_tissue = area_mm2 > 0
    density[has_tissue] = counts[has_tissue] / area_mm2[has_tissue]

    total = float(counts.sum())
    fraction = counts / total if total > 0 else np.zeros(n)
    cumulative_count = np.cumsum(counts)
    cumulative_fraction = cumulative_count / total if total > 0 else np.zeros(n)

    rows = []
    for i in range(n):
        rows.append({
            "radial_bin_start_um": round(i * bin_width_um, 3),
            "radial_bin_end_um": round((i + 1) * bin_width_um, 3),
            "count": int(counts[i]),
            "tissue_area_px": int(area_px[i]),
            "tissue_area_mm2": round(float(area_mm2[i]), 9),
            "density_per_mm2": (round(float(density[i]), 4)
                                if np.isfinite(density[i]) else ""),
            "fraction": round(float(fraction[i]), 6),
            "cumulative_count": int(cumulative_count[i]),
            "cumulative_fraction": round(float(cumulative_fraction[i]), 6),
        })
    return rows


def injection_core_centroid(core_mask, crop_origin=(0, 0)):
    """Global (x, y) centroid of a boolean injection-core mask, plus pixel count."""
    import numpy as np  # noqa: PLC0415

    ys, xs = np.nonzero(np.asarray(core_mask, dtype=bool))
    if ys.size == 0:
        return None
    oy, ox = crop_origin
    return {
        "x_global_px": float(xs.mean()) + float(ox),
        "y_global_px": float(ys.mean()) + float(oy),
        "n_pixels": int(ys.size),
    }


def per_candidate_rows(candidates, center_xy, voxel_yx_um, bin_width_um, n_bins):
    """Per-candidate radial coordinate rows (dx/dy/distance/bin)."""
    vy, vx = float(voxel_yx_um[0]), float(voxel_yx_um[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    rows = []
    for c in candidates:
        try:
            x = float(c.get("x_global_px"))
            y = float(c.get("y_global_px"))
        except (TypeError, ValueError):
            continue
        dx = x - cx
        dy = y - cy
        dist_px = math.hypot(dx, dy)
        dist_um = math.hypot(dx * vx, dy * vy)
        b = int(dist_um // bin_width_um)
        in_range = 0 <= b < n_bins
        rows.append({
            "candidate_id": c.get("candidate_id", ""),
            "channel": c.get("channel", ""),
            "section": c.get("section", ""),
            "x_global_px": int(round(x)),
            "y_global_px": int(round(y)),
            "current_status": c.get("current_status", ""),
            "inside_injection_analysis_exclusion":
                c.get("inside_injection_analysis_exclusion", ""),
            "dx_px": round(dx, 2),
            "dy_px": round(dy, 2),
            "radial_distance_px": round(dist_px, 2),
            "radial_distance_um": round(dist_um, 2),
            "radial_bin_start_um": round(b * bin_width_um, 3) if in_range else "",
            "radial_bin_end_um": round((b + 1) * bin_width_um, 3) if in_range else "",
        })
    return rows
