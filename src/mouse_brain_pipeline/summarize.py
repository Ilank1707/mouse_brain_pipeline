"""Export object-level and region-level result tables.

Atlas region columns are only populated when a registration/atlas assignment is
available (i.e. after a real Brainmapper run with a background channel). For pilot
candidate detections those fields stay blank, and ``source_method`` makes clear
the rows are provisional candidates, not validated cells.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from .config import Config
from .utilities import ensure_dir

OBJECT_COLUMNS = [
    "cell_id",
    "channel",
    "x_px",
    "y_px",
    "z_plane",
    "x_um",
    "y_um",
    "z_um",
    "section",
    "optical_plane",
    "atlas_region_id",
    "atlas_region_name",
    "parent_region",
    "signal_intensity",
    "classification_score",
    "source_method",
]

REGION_COLUMNS = [
    "channel",
    "atlas_region_id",
    "atlas_region_name",
    "parent_region",
    "marker_positive_cells",
    "region_volume_mm3",
    "cells_per_mm3",
]


def candidate_to_object_rows(candidate_rows: Iterable[dict], config: Config) -> list[dict]:
    """Map pilot candidate rows into the object-level schema (atlas fields blank)."""
    vz, vy, vx = config.acquisition.voxel_size_zyx
    out: list[dict] = []
    for c in candidate_rows:
        # Accept both the explicit schema (x_global_px, z_index, ...) and the
        # legacy aliases (x_px, z_plane, ...).
        x_px = float(c.get("x_global_px", c.get("x_px", 0)) or 0)
        y_px = float(c.get("y_global_px", c.get("y_px", 0)) or 0)
        out.append(
            {
                "cell_id": c.get("candidate_id", c.get("cell_id", "")),
                "channel": c.get("channel", ""),
                "x_px": x_px,
                "y_px": y_px,
                "z_plane": c.get("z_index", c.get("z_plane", "")),
                "x_um": round(x_px * vx, 3),
                "y_um": round(y_px * vy, 3),
                "z_um": c.get("section_relative_z_um", c.get("z_um", "")),
                "section": c.get("section", ""),
                "optical_plane": c.get("optical_plane", c.get("plane", "")),
                "atlas_region_id": "",
                "atlas_region_name": "",
                "parent_region": "",
                "signal_intensity": c.get("peak_intensity", c.get("intensity", "")),
                "classification_score": c.get("classification_score", c.get("score", "")),
                "source_method": c.get("backend", c.get("method", "pilot_candidate")),
            }
        )
    return out


def write_object_csv(out_dir: Path, rows: Sequence[dict], filename: str = "objects.csv") -> Path:
    ensure_dir(out_dir)
    path = out_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBJECT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in OBJECT_COLUMNS})
    return path


def aggregate_regions(
    object_rows: Sequence[dict],
    region_volumes_mm3: dict[str, float] | None = None,
) -> list[dict]:
    """Aggregate object rows into region-level counts per channel.

    Rows without an ``atlas_region_id`` are grouped under an explicit
    "unassigned" bucket so pilot data still summarises cleanly.
    """
    region_volumes_mm3 = region_volumes_mm3 or {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    names: dict[str, str] = {}
    parents: dict[str, str] = {}
    for r in object_rows:
        rid = str(r.get("atlas_region_id") or "unassigned")
        channel = str(r.get("channel", ""))
        counts[(channel, rid)] += 1
        names[rid] = str(r.get("atlas_region_name", ""))
        parents[rid] = str(r.get("parent_region", ""))

    out: list[dict] = []
    for (channel, rid), n in sorted(counts.items()):
        vol = region_volumes_mm3.get(rid)
        density = round(n / vol, 3) if vol else ""
        out.append(
            {
                "channel": channel,
                "atlas_region_id": rid,
                "atlas_region_name": names.get(rid, ""),
                "parent_region": parents.get(rid, ""),
                "marker_positive_cells": n,
                "region_volume_mm3": vol if vol is not None else "",
                "cells_per_mm3": density,
            }
        )
    return out


def write_region_csv(out_dir: Path, rows: Sequence[dict], filename: str = "regions.csv") -> Path:
    ensure_dir(out_dir)
    path = out_dir / filename
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGION_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in REGION_COLUMNS})
    return path


def read_candidate_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
