"""Provisional spatial overlap between green and channel_2 detections.

IMPORTANT LABELLING RULE: a spatial match here is a PROVISIONAL spatial
classification only. It is NEVER called "double-positive" and never assumed to
represent the same biological population without independent validation.

Matching is one-to-one (greedy nearest-neighbour within a 3D tolerance) and is
DISABLED by default at the CLI level.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .utilities import LOG, ensure_dir

OVERLAP_COLUMNS = ["green_cell_id", "channel_2_cell_id", "distance_um", "provisional_overlap"]


def _coords_um(rows: Sequence[dict]):
    """Build an (N,3) array of (z_um, y_um, x_um) and parallel id list."""
    import numpy as np  # noqa: PLC0415

    pts = []
    ids = []
    for r in rows:
        z = float(r.get("z_um", r.get("section_relative_z_um", 0)) or 0)
        # Prefer micrometre columns; fall back to pixel columns if needed.
        y = float(r.get("y_um", r.get("y_global_px", r.get("y_px", 0))) or 0)
        x = float(r.get("x_um", r.get("x_global_px", r.get("x_px", 0))) or 0)
        pts.append([z, y, x])
        ids.append(r.get("candidate_id") or r.get("cell_id") or "")
    return np.array(pts, dtype=float) if pts else np.empty((0, 3)), ids


def match_one_to_one(
    green_rows: Sequence[dict],
    channel_2_rows: Sequence[dict],
    tolerance_um: float,
) -> list[dict]:
    """Greedy one-to-one nearest-neighbour matching within ``tolerance_um``.

    Returns rows with green_cell_id, channel_2_cell_id, distance_um, provisional_overlap.
    """
    import numpy as np  # noqa: PLC0415

    g_pts, g_ids = _coords_um(green_rows)
    c_pts, c_ids = _coords_um(channel_2_rows)
    if len(g_pts) == 0 or len(c_pts) == 0:
        return []

    # All candidate pairs within tolerance, sorted by distance, assigned greedily.
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        tree = cKDTree(c_pts)
        pairs = []
        for gi, gp in enumerate(g_pts):
            for ci in tree.query_ball_point(gp, tolerance_um):
                d = float(np.linalg.norm(gp - c_pts[ci]))
                pairs.append((d, gi, ci))
    except Exception:  # pragma: no cover - fallback
        pairs = []
        for gi, gp in enumerate(g_pts):
            d = np.linalg.norm(c_pts - gp, axis=1)
            for ci in np.where(d <= tolerance_um)[0]:
                pairs.append((float(d[ci]), gi, int(ci)))

    pairs.sort(key=lambda t: t[0])
    used_g: set[int] = set()
    used_c: set[int] = set()
    out: list[dict] = []
    for d, gi, ci in pairs:
        if gi in used_g or ci in used_c:
            continue
        used_g.add(gi)
        used_c.add(ci)
        out.append(
            {
                "green_cell_id": g_ids[gi],
                "channel_2_cell_id": c_ids[ci],
                "distance_um": round(d, 3),
                "provisional_overlap": True,  # PROVISIONAL spatial classification ONLY
            }
        )
    LOG.warning(
        "Found %d PROVISIONAL spatial overlaps (tolerance %.1f um). This is NOT "
        "'double-positive' and requires biological validation.",
        len(out), tolerance_um,
    )
    return out


def write_overlaps(out_dir: Path, rows: Sequence[dict]) -> Path:
    ensure_dir(out_dir)
    path = out_dir / "provisional_overlaps.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERLAP_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in OVERLAP_COLUMNS})
    return path
