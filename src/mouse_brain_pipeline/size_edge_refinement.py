"""Auditable post-detection refinement of candidate size and edge handling.

This module reduces PROVISIONAL small-object detections and rescues valid
edge candidates without deleting anything. It reuses the connected-component
measurements already recorded per candidate (volume, XY diameter, plane
support) rather than re-measuring raw pixels, and it uses the *original*
(non-eroded) saved tissue mask for all edge geometry.

Candidates are never confirmed as cells. Raw TIFFs, candidate coordinates,
detection thresholds, injection masks, candidate statuses, and existing run
outputs are read only; every result is written into a caller-supplied,
isolated output directory.

Two modes:

* ``report`` -- diagnostics only. No candidate status is changed and no size
  threshold is applied. A threshold table shows how many candidates would
  remain across many threshold combinations, without selecting one.
* ``apply``  -- applies ONLY the thresholds explicitly supplied by the caller.
  Refuses to run if no size threshold is supplied.

Arrays are ``z, y, x``; a physical section spans seven optical planes; voxel
sizes are taken from the configuration (default 6.0, 1.004, 1.004 um).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Statuses currently treated as "included" provisional detections that are
# eligible for size-based refinement. Excluded populations (injection, artefact,
# rule-fail, invalid) are preserved unless the edge-rescue rule applies; this
# module never overrides injection or artefact logic.
STATUS_PRELIMINARY_PASS = "preliminary_rule_pass"
STATUS_MANUAL_REVIEW = "manual_review"
SIZE_ELIGIBLE_STATUSES = frozenset({STATUS_PRELIMINARY_PASS, STATUS_MANUAL_REVIEW})

# Refined status vocabulary produced by the refinement.
REFINED_RETAINED = "retained"
REFINED_FILTERED_SMALL = "filtered_too_small"
REFINED_MANUAL_REVIEW_EDGE = "manual_review_edge"
REFINED_INVALID = "invalid_measurement"

MODE_REPORT = "report"
MODE_APPLY = "apply"

# Minimum measurement-patch radius (um) so the valid-pixel fraction is measured
# over a small neighbourhood even for tiny components.
MINIMUM_PATCH_RADIUS_UM = 3.0
# Diagnostic search radius (um) used to report distance-to-tissue for candidate
# centres outside the mask. This is a reporting window, NOT a decision
# threshold; distances beyond it are recorded as capped.
DIAGNOSTIC_EDGE_SEARCH_UM = 60.0

# New columns the refinement adds to every candidate row (originals are kept).
REFINEMENT_FIELDS = [
    "original_status",
    "refined_candidate_status",
    "size_filter_status",
    "size_filter_reason",
    "component_area_um2",
    "component_volume_um3",
    "support_voxel_count",
    "maximum_diameter_um",
    "centre_inside_tissue",
    "distance_to_tissue_um",
    "distance_to_tissue_capped",
    "valid_pixel_fraction",
    "measurement_patch_clipped",
    "measurement_complete",
    "signal_support_overlaps_tissue",
    "edge_rescued",
    "refinement_mode",
]


@dataclass(frozen=True)
class RefinementThresholds:
    """Explicitly supplied thresholds. ``None`` means "not supplied"."""

    min_component_area_um2: Optional[float] = None
    min_component_volume_um3: Optional[float] = None
    min_support_planes: Optional[int] = None
    edge_rescue_distance_um: Optional[float] = None

    def has_any_size_threshold(self) -> bool:
        return any(
            value is not None
            for value in (
                self.min_component_area_um2,
                self.min_component_volume_um3,
                self.min_support_planes,
            )
        )


@dataclass
class RefinementResult:
    channel: str
    section: int
    mode: str
    voxel_zyx_um: tuple[float, float, float]
    thresholds: RefinementThresholds
    rows: list[dict]
    summary: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Type coercion (rows may hold Python natives in-pipeline or CSV strings)
# --------------------------------------------------------------------------- #
def _to_float(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    try:
        text = str(value).strip()
        if text == "" or text.lower() == "nan":
            return math.nan
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _to_int(value, default: int = 0) -> int:
    number = _to_float(value)
    return int(round(number)) if math.isfinite(number) else default


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


# --------------------------------------------------------------------------- #
# Per-candidate measurement reuse / derivation
# --------------------------------------------------------------------------- #
def derive_candidate_metrics(row: dict, voxel_zyx_um: Sequence[float]) -> dict:
    """Reuse or derive connected-component size measurements for one candidate.

    Size is taken from the connected signal component measured during detection
    (``volume_um3``, ``xy_diameter_um``) and its seven-plane support
    (``support_plane_count``). Brightness is never used as a size measurement.
    """
    voxel_z, voxel_y, voxel_x = (float(v) for v in voxel_zyx_um)
    voxel_volume_um3 = voxel_z * voxel_y * voxel_x

    volume_um3 = _to_float(row.get("volume_um3"))
    xy_diameter_um = _to_float(row.get("xy_diameter_um"))
    support_plane_count = _to_int(row.get("support_plane_count"), 0)

    # Connected-component XY area implied by the recorded circular-equivalent XY
    # diameter of the component projection.
    if math.isfinite(xy_diameter_um) and xy_diameter_um > 0:
        component_area_um2 = math.pi * (xy_diameter_um / 2.0) ** 2
        maximum_diameter_um = xy_diameter_um
    else:
        component_area_um2 = math.nan
        maximum_diameter_um = math.nan

    # Total supporting voxels recovered exactly from the component volume.
    if math.isfinite(volume_um3) and volume_um3 > 0 and voxel_volume_um3 > 0:
        support_voxel_count = int(round(volume_um3 / voxel_volume_um3))
    else:
        support_voxel_count = 0

    component_present = (
        math.isfinite(volume_um3)
        and volume_um3 > 0
        and math.isfinite(xy_diameter_um)
        and xy_diameter_um > 0
    )
    measurement_valid = _to_bool(row.get("measurement_valid"))
    invalid_coordinate = _to_bool(row.get("invalid_coordinate"))

    return {
        "component_area_um2": component_area_um2,
        "component_volume_um3": volume_um3,
        "support_plane_count": support_plane_count,
        "support_voxel_count": support_voxel_count,
        "maximum_diameter_um": maximum_diameter_um,
        "component_present": component_present,
        "measurement_valid": measurement_valid,
        "invalid_coordinate": invalid_coordinate,
    }


def size_failure_reasons(metrics: dict, thresholds: RefinementThresholds) -> list[str]:
    """Explicitly-supplied thresholds this candidate fails (never invented)."""
    reasons: list[str] = []
    area = metrics["component_area_um2"]
    volume = metrics["component_volume_um3"]
    support = metrics["support_plane_count"]

    if thresholds.min_component_area_um2 is not None:
        if not math.isfinite(area) or area < thresholds.min_component_area_um2:
            reasons.append(
                f"component_area_um2 {area:.2f} < {thresholds.min_component_area_um2:g}"
            )
    if thresholds.min_component_volume_um3 is not None:
        if not math.isfinite(volume) or volume < thresholds.min_component_volume_um3:
            reasons.append(
                f"component_volume_um3 {volume:.2f} < "
                f"{thresholds.min_component_volume_um3:g}"
            )
    if thresholds.min_support_planes is not None:
        if support < thresholds.min_support_planes:
            reasons.append(
                f"support_plane_count {support} < {thresholds.min_support_planes:g}"
            )
    return reasons


# --------------------------------------------------------------------------- #
# Edge geometry (uses the original, non-eroded tissue mask only)
# --------------------------------------------------------------------------- #
def measure_patch_validity(tissue_mask, cy: int, cx: int, half_px: int) -> dict:
    """Clip a square measurement patch to valid image + tissue pixels.

    Returns the valid-pixel fraction of the intended patch (a patch pixel is
    valid only if it lies inside the image AND inside the original tissue mask),
    whether the patch is clipped, and whether it overlaps any valid tissue.
    """
    import numpy as np  # noqa: PLC0415

    height, width = tissue_mask.shape
    full_count = (2 * half_px + 1) ** 2
    y0, y1 = cy - half_px, cy + half_px + 1
    x0, x1 = cx - half_px, cx + half_px + 1
    yy0, yy1 = max(0, y0), min(height, y1)
    xx0, xx1 = max(0, x0), min(width, x1)
    if yy1 <= yy0 or xx1 <= xx0:
        return {
            "valid_pixel_fraction": 0.0,
            "valid_count": 0,
            "full_count": full_count,
            "clipped": True,
            "overlaps_tissue": False,
        }
    sub = np.asarray(tissue_mask[yy0:yy1, xx0:xx1], dtype=bool)
    valid_count = int(sub.sum())
    fraction = valid_count / full_count if full_count else 0.0
    return {
        "valid_pixel_fraction": fraction,
        "valid_count": valid_count,
        "full_count": full_count,
        # Clipped if any intended patch pixel is out of image OR out of tissue.
        "clipped": valid_count < full_count,
        "overlaps_tissue": valid_count > 0,
    }


def distance_to_tissue_um(
    tissue_mask, cy: int, cx: int, voxel_y_um: float, search_px: int
) -> tuple[float, bool]:
    """Distance (um) from a centre to the nearest original-tissue pixel.

    Zero when the centre is inside tissue. For centres outside tissue only a
    bounded ``search_px`` window is scanned (memory-safe on full-section masks);
    distances beyond the window are returned capped.
    """
    import numpy as np  # noqa: PLC0415

    height, width = tissue_mask.shape
    inside = 0 <= cy < height and 0 <= cx < width and bool(tissue_mask[cy, cx])
    if inside:
        return 0.0, False
    y0, y1 = max(0, cy - search_px), min(height, cy + search_px + 1)
    x0, x1 = max(0, cx - search_px), min(width, cx + search_px + 1)
    sub = np.asarray(tissue_mask[y0:y1, x0:x1], dtype=bool)
    ys, xs = np.nonzero(sub)
    if ys.size == 0:
        return float(search_px * voxel_y_um), True
    dy = (ys.astype(np.float64) + y0) - cy
    dx = (xs.astype(np.float64) + x0) - cx
    nearest_px = float(np.sqrt(dy * dy + dx * dx).min())
    return nearest_px * float(voxel_y_um), False


# --------------------------------------------------------------------------- #
# Per-candidate classification
# --------------------------------------------------------------------------- #
def classify_candidate(
    *,
    original_status: str,
    metrics: dict,
    patch: dict,
    distance_um: float,
    thresholds: RefinementThresholds,
    mode: str,
) -> dict:
    """Decide the refined status for one candidate.

    ``report`` mode never changes status. ``apply`` mode applies only supplied
    thresholds. Rescued edge candidates go to ``manual_review_edge`` and are
    never confirmed as cells.
    """
    centre_inside = _to_bool(metrics["_centre_inside_tissue"])
    measurement_complete = (
        metrics["measurement_valid"]
        and metrics["component_present"]
        and not metrics["invalid_coordinate"]
        and not patch["clipped"]
    )
    # Connected signal support overlapping valid tissue: real z-support AND the
    # measurement patch reaches valid tissue.
    has_signal_support = (
        metrics["measurement_valid"]
        and metrics["support_plane_count"] >= 1
        and patch["overlaps_tissue"]
    )

    base = {
        "measurement_complete": measurement_complete,
        "signal_support_overlaps_tissue": has_signal_support,
        "edge_rescued": False,
    }

    invalid = (
        not metrics["measurement_valid"]
        or not metrics["component_present"]
        or metrics["invalid_coordinate"]
    )

    if mode == MODE_REPORT:
        # Diagnostics only: preserve status, apply nothing.
        return {
            **base,
            "refined_candidate_status": original_status,
            "size_filter_status": "not_applied",
            "size_filter_reason": "report mode: diagnostics only, no threshold applied",
        }

    # ---- apply mode ----
    if invalid:
        return {
            **base,
            "refined_candidate_status": REFINED_INVALID,
            "size_filter_status": "invalid",
            "size_filter_reason": "measurement invalid or connected component missing",
        }

    reasons = size_failure_reasons(metrics, thresholds)
    size_fail = bool(reasons)

    if centre_inside:
        if not size_fail:
            return {
                **base,
                "refined_candidate_status": REFINED_RETAINED,
                "size_filter_status": "adequate",
                "size_filter_reason": "meets all supplied size thresholds",
            }
        if measurement_complete:
            return {
                **base,
                "refined_candidate_status": REFINED_FILTERED_SMALL,
                "size_filter_status": "small",
                "size_filter_reason": "; ".join(reasons),
            }
        # Small but edge-clipped: do not auto-discard a possibly valid edge cell.
        return {
            **base,
            "refined_candidate_status": REFINED_MANUAL_REVIEW_EDGE,
            "size_filter_status": "small_edge_clipped",
            "size_filter_reason": (
                "fails size but measurement clipped by an edge: " + "; ".join(reasons)
            ),
        }

    # Centre outside the original tissue mask: never auto-accept.
    rescue_distance = thresholds.edge_rescue_distance_um
    rescued = (
        rescue_distance is not None
        and distance_um <= rescue_distance
        and has_signal_support
    )
    if rescued:
        return {
            **base,
            "refined_candidate_status": REFINED_MANUAL_REVIEW_EDGE,
            "size_filter_status": "edge_rescue",
            "size_filter_reason": (
                f"centre {distance_um:.2f} um outside tissue, within rescue distance "
                f"{rescue_distance:g} um with signal support overlapping valid tissue"
            ),
            "edge_rescued": True,
            "signal_support_overlaps_tissue": has_signal_support,
            "measurement_complete": measurement_complete,
        }
    return {
        **base,
        "refined_candidate_status": original_status,
        "size_filter_status": "preserved",
        "size_filter_reason": "centre outside tissue and not eligible for edge rescue",
    }


# --------------------------------------------------------------------------- #
# Threshold table (report mode)
# --------------------------------------------------------------------------- #
def _threshold_grids(metrics_list: list[dict]) -> dict:
    import numpy as np  # noqa: PLC0415

    areas = np.array(
        [m["component_area_um2"] for m in metrics_list if math.isfinite(m["component_area_um2"])],
        dtype=float,
    )
    volumes = np.array(
        [m["component_volume_um3"] for m in metrics_list if math.isfinite(m["component_volume_um3"])],
        dtype=float,
    )
    supports = np.array([m["support_plane_count"] for m in metrics_list], dtype=float)

    def percentile_grid(values, points):
        if values.size == 0:
            return [0.0]
        qs = np.percentile(values, points)
        grid = sorted({round(float(v), 2) for v in np.concatenate(([0.0], qs))})
        return grid

    return {
        "area": percentile_grid(areas, [10, 25, 50, 75, 90]),
        "volume": percentile_grid(volumes, [10, 25, 50, 75, 90]),
        "support": sorted({int(v) for v in [1, 2, 3, 4]}),
    }


def build_threshold_table(metrics_list: list[dict]) -> dict:
    """Counts of candidates that would remain across threshold combinations.

    Report-mode only. Does NOT choose a threshold; it enumerates combinations of
    minimum component area, minimum component volume, and minimum support planes
    over the evaluable population and reports how many would remain / be flagged.
    """
    grids = _threshold_grids(metrics_list)
    total = len(metrics_list)
    rows = []
    for min_area in grids["area"]:
        for min_volume in grids["volume"]:
            for min_support in grids["support"]:
                remaining = 0
                for m in metrics_list:
                    area_ok = (
                        math.isfinite(m["component_area_um2"])
                        and m["component_area_um2"] >= min_area
                    )
                    volume_ok = (
                        math.isfinite(m["component_volume_um3"])
                        and m["component_volume_um3"] >= min_volume
                    )
                    support_ok = m["support_plane_count"] >= min_support
                    if area_ok and volume_ok and support_ok:
                        remaining += 1
                rows.append(
                    {
                        "min_component_area_um2": min_area,
                        "min_component_volume_um3": min_volume,
                        "min_support_planes": min_support,
                        "candidates_evaluated": total,
                        "candidates_remaining": remaining,
                        "candidates_flagged_small": total - remaining,
                    }
                )
    return {
        "population": "size-eligible candidates with a valid connected-component "
        "measurement and a centre inside the original tissue mask",
        "candidates_evaluated": total,
        "grids": grids,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Core driver
# --------------------------------------------------------------------------- #
def refine_candidates(
    rows: Iterable[dict],
    tissue_mask,
    *,
    voxel_zyx_um: Sequence[float],
    mode: str,
    thresholds: RefinementThresholds,
    channel: str,
    section: int,
) -> RefinementResult:
    """Compute diagnostics and (in apply mode) refined statuses for candidates.

    Every input candidate is preserved in the returned rows with its original
    status recorded. The tissue mask is the original saved mask (never eroded).
    """
    if mode not in (MODE_REPORT, MODE_APPLY):
        raise ValueError(f"mode must be '{MODE_REPORT}' or '{MODE_APPLY}', got {mode!r}")
    if mode == MODE_APPLY and not thresholds.has_any_size_threshold():
        raise ValueError(
            "apply mode requires at least one explicit size threshold "
            "(--min-component-area-um2, --min-component-volume-um3, or "
            "--min-support-planes)"
        )

    voxel_z, voxel_y, voxel_x = (float(v) for v in voxel_zyx_um)
    height, width = int(tissue_mask.shape[0]), int(tissue_mask.shape[1])

    search_um = DIAGNOSTIC_EDGE_SEARCH_UM
    if thresholds.edge_rescue_distance_um is not None:
        search_um = max(search_um, float(thresholds.edge_rescue_distance_um))
    search_px = max(1, int(math.ceil(search_um / voxel_y)))

    refined_rows: list[dict] = []
    eligible_metrics: list[dict] = []
    for row in rows:
        out = dict(row)  # preserve every original field
        original_status = str(row.get("current_status", ""))

        metrics = derive_candidate_metrics(row, voxel_zyx_um)

        cx = _to_int(row.get("x_local_px", row.get("x_global_px")), -1)
        cy = _to_int(row.get("y_local_px", row.get("y_global_px")), -1)
        in_bounds = 0 <= cy < height and 0 <= cx < width
        centre_inside = bool(in_bounds and _to_bool_mask(tissue_mask, cy, cx))
        metrics["_centre_inside_tissue"] = centre_inside

        half_um = MINIMUM_PATCH_RADIUS_UM
        if math.isfinite(metrics["maximum_diameter_um"]):
            half_um = max(half_um, metrics["maximum_diameter_um"] / 2.0)
        half_px = max(1, int(math.ceil(half_um / voxel_y)))

        if in_bounds:
            patch = measure_patch_validity(tissue_mask, cy, cx, half_px)
            distance, capped = distance_to_tissue_um(
                tissue_mask, cy, cx, voxel_y, search_px
            )
        else:
            patch = {
                "valid_pixel_fraction": 0.0,
                "valid_count": 0,
                "full_count": (2 * half_px + 1) ** 2,
                "clipped": True,
                "overlaps_tissue": False,
            }
            distance, capped = float(search_px * voxel_y), True

        decision = classify_candidate(
            original_status=original_status,
            metrics=metrics,
            patch=patch,
            distance_um=distance,
            thresholds=thresholds,
            mode=mode,
        )

        out.update(
            {
                "original_status": original_status,
                "refined_candidate_status": decision["refined_candidate_status"],
                "size_filter_status": decision["size_filter_status"],
                "size_filter_reason": decision["size_filter_reason"],
                "component_area_um2": _round(metrics["component_area_um2"], 3),
                "component_volume_um3": _round(metrics["component_volume_um3"], 3),
                "support_voxel_count": metrics["support_voxel_count"],
                "maximum_diameter_um": _round(metrics["maximum_diameter_um"], 3),
                "centre_inside_tissue": centre_inside,
                "distance_to_tissue_um": _round(distance, 3),
                "distance_to_tissue_capped": bool(capped),
                "valid_pixel_fraction": _round(patch["valid_pixel_fraction"], 4),
                "measurement_patch_clipped": bool(patch["clipped"]),
                "measurement_complete": bool(decision["measurement_complete"]),
                "signal_support_overlaps_tissue": bool(
                    decision["signal_support_overlaps_tissue"]
                ),
                "edge_rescued": bool(decision["edge_rescued"]),
                "refinement_mode": mode,
            }
        )
        # ``support_plane_count`` is reused verbatim from the original column.
        refined_rows.append(out)

        if (
            original_status in SIZE_ELIGIBLE_STATUSES
            and centre_inside
            and metrics["component_present"]
            and metrics["measurement_valid"]
            and not metrics["invalid_coordinate"]
        ):
            eligible_metrics.append(metrics)

    summary = _build_summary(
        refined_rows,
        eligible_metrics,
        channel=channel,
        section=section,
        mode=mode,
        thresholds=thresholds,
        voxel_zyx_um=(voxel_z, voxel_y, voxel_x),
        mask_shape=(height, width),
    )
    return RefinementResult(
        channel=channel,
        section=section,
        mode=mode,
        voxel_zyx_um=(voxel_z, voxel_y, voxel_x),
        thresholds=thresholds,
        rows=refined_rows,
        summary=summary,
    )


def _to_bool_mask(tissue_mask, cy: int, cx: int) -> bool:
    return bool(tissue_mask[cy, cx])


def _round(value, ndigits):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return round(number, ndigits)


def _build_summary(
    refined_rows,
    eligible_metrics,
    *,
    channel,
    section,
    mode,
    thresholds,
    voxel_zyx_um,
    mask_shape,
):
    from collections import Counter

    original_counts = Counter(r["original_status"] for r in refined_rows)
    refined_counts = Counter(r["refined_candidate_status"] for r in refined_rows)
    size_filter_counts = Counter(r["size_filter_status"] for r in refined_rows)

    summary = {
        "analysis": "post-detection candidate size and edge refinement",
        "provisional_candidates": True,
        "candidates_are_not_cells": True,
        "channel": channel,
        "section": section,
        "mode": mode,
        "voxel_size_zyx_um": list(voxel_zyx_um),
        "tissue_mask_shape_yx": list(mask_shape),
        "tissue_mask_erosion_used": False,
        "size_measurement_source": (
            "reused connected-component volume_um3 and xy_diameter_um and "
            "support_plane_count from detection; brightness never used as size"
        ),
        "size_eligible_statuses": sorted(SIZE_ELIGIBLE_STATUSES),
        "thresholds_supplied": {
            "min_component_area_um2": thresholds.min_component_area_um2,
            "min_component_volume_um3": thresholds.min_component_volume_um3,
            "min_support_planes": thresholds.min_support_planes,
            "edge_rescue_distance_um": thresholds.edge_rescue_distance_um,
        },
        "total_candidates": len(refined_rows),
        "original_status_counts": dict(original_counts),
        "refined_status_counts": dict(refined_counts),
        "size_filter_status_counts": dict(size_filter_counts),
    }
    if mode == MODE_REPORT:
        summary["threshold_table"] = build_threshold_table(eligible_metrics)
        summary["note"] = (
            "report mode: no threshold applied and no candidate status changed"
        )
    else:
        summary["applied"] = {
            "filtered_too_small": refined_counts.get(REFINED_FILTERED_SMALL, 0),
            "manual_review_edge": refined_counts.get(REFINED_MANUAL_REVIEW_EDGE, 0),
            "retained": refined_counts.get(REFINED_RETAINED, 0),
            "invalid_measurement": refined_counts.get(REFINED_INVALID, 0),
            "edge_rescued": sum(1 for r in refined_rows if r["edge_rescued"]),
        }
    return summary


# --------------------------------------------------------------------------- #
# Output writing (CSVs + JSON + plots), shared by the CLI and the pipeline
# --------------------------------------------------------------------------- #
AUDIT_CSV = "refined_candidates.csv"
RETAINED_CSV = "retained_candidates.csv"
FILTERED_CSV = "filtered_too_small.csv"
MANUAL_REVIEW_EDGE_CSV = "manual_review_edge.csv"
INVALID_CSV = "invalid_measurements.csv"
SUMMARY_JSON = "refinement_summary.json"
PLOT_SIZE = "candidate_size_distributions.png"
PLOT_SUPPORT = "support_planes_distribution.png"
PLOT_SIZE_VS_EDGE = "size_vs_edge_distance.png"
PLOT_EDGE_QC = "edge_candidate_qc.png"


def _write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ordered_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def write_refinement_outputs(
    channel_dir: Path,
    result: RefinementResult,
    *,
    make_plots: bool = True,
) -> dict:
    """Write every refinement output into ``channel_dir`` (already isolated)."""
    channel_dir = Path(channel_dir)
    channel_dir.mkdir(parents=True, exist_ok=True)
    rows = result.rows
    fieldnames = _ordered_fieldnames(rows)

    # Audit CSV: every original candidate, its original status, and diagnostics.
    _write_rows_csv(channel_dir / AUDIT_CSV, rows, fieldnames)

    by_refined = {
        RETAINED_CSV: [r for r in rows if r["refined_candidate_status"] == REFINED_RETAINED],
        FILTERED_CSV: [
            r for r in rows if r["refined_candidate_status"] == REFINED_FILTERED_SMALL
        ],
        MANUAL_REVIEW_EDGE_CSV: [
            r for r in rows if r["refined_candidate_status"] == REFINED_MANUAL_REVIEW_EDGE
        ],
        INVALID_CSV: [r for r in rows if r["refined_candidate_status"] == REFINED_INVALID],
    }
    for filename, subset in by_refined.items():
        _write_rows_csv(channel_dir / filename, subset, fieldnames)

    (channel_dir / SUMMARY_JSON).write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )

    outputs = {
        "audit_csv": str(channel_dir / AUDIT_CSV),
        "retained_csv": str(channel_dir / RETAINED_CSV),
        "filtered_too_small_csv": str(channel_dir / FILTERED_CSV),
        "manual_review_edge_csv": str(channel_dir / MANUAL_REVIEW_EDGE_CSV),
        "invalid_measurements_csv": str(channel_dir / INVALID_CSV),
        "summary_json": str(channel_dir / SUMMARY_JSON),
    }
    if make_plots:
        outputs.update(_write_plots(channel_dir, result))
    return outputs


# --------------------------------------------------------------------------- #
# Plots (green and red kept in separate per-channel folders; each figure is
# labelled with its channel)
# --------------------------------------------------------------------------- #
def _finite(values):
    import numpy as np  # noqa: PLC0415

    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _legend_if_labelled(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8)


def _write_plots(channel_dir: Path, result: RefinementResult) -> dict:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    rows = result.rows
    channel = result.channel
    section = result.section
    thresholds = result.thresholds

    area = _finite([r.get("component_area_um2") for r in rows])
    volume = _finite([r.get("component_volume_um3") for r in rows])
    support = [int(r.get("support_plane_count") or 0) for r in rows]
    distance = _finite([r.get("distance_to_tissue_um") for r in rows])

    title_suffix = f"{channel}, section {section:03d}, {result.mode} mode"

    # 1. Size distributions (area + volume).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        if area.size:
            axes[0].hist(area, bins=40, color="#2E7D32" if "green" in channel else "#C62828")
        axes[0].set(xlabel="connected-component XY area (µm²)", ylabel="candidates")
        if thresholds.min_component_area_um2 is not None:
            axes[0].axvline(
                thresholds.min_component_area_um2,
                color="black",
                linestyle="--",
                label=f"min area={thresholds.min_component_area_um2:g}",
            )
            axes[0].legend(fontsize=8)
        if volume.size:
            axes[1].hist(volume, bins=40, color="#2E7D32" if "green" in channel else "#C62828")
        axes[1].set(xlabel="connected-component volume (µm³)", ylabel="candidates")
        if thresholds.min_component_volume_um3 is not None:
            axes[1].axvline(
                thresholds.min_component_volume_um3,
                color="black",
                linestyle="--",
                label=f"min volume={thresholds.min_component_volume_um3:g}",
            )
            axes[1].legend(fontsize=8)
        fig.suptitle(f"PROVISIONAL candidate size distributions — {title_suffix}")
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_SIZE, dpi=150)
    finally:
        plt.close(fig)

    # 2. Support-plane distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        counts = np.bincount(np.clip(support, 0, 7), minlength=8)
        ax.bar(range(8), counts, color="#2E7D32" if "green" in channel else "#C62828")
        if thresholds.min_support_planes is not None:
            ax.axvline(
                thresholds.min_support_planes - 0.5,
                color="black",
                linestyle="--",
                label=f"min support planes={thresholds.min_support_planes:g}",
            )
            ax.legend(fontsize=8)
        ax.set(
            xlabel="planes with valid signal support",
            ylabel="candidates",
            title=f"PROVISIONAL support-plane distribution — {title_suffix}",
        )
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_SUPPORT, dpi=150)
    finally:
        plt.close(fig)

    # 3. Size vs edge distance.
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        xs = [r.get("distance_to_tissue_um") for r in rows]
        ys = [r.get("component_area_um2") for r in rows]
        inside = [bool(r.get("centre_inside_tissue")) for r in rows]
        xs = np.array([_safe(x) for x in xs])
        ys = np.array([_safe(y) for y in ys])
        inside = np.array(inside)
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.any():
            ax.scatter(
                xs[finite & inside], ys[finite & inside], s=6, alpha=0.4,
                color="#1565C0", label="centre inside tissue",
            )
            ax.scatter(
                xs[finite & ~inside], ys[finite & ~inside], s=10, alpha=0.6,
                color="#EF6C00", label="centre outside tissue",
            )
        if thresholds.edge_rescue_distance_um is not None:
            ax.axvline(
                thresholds.edge_rescue_distance_um,
                color="black",
                linestyle="--",
                label=f"edge rescue={thresholds.edge_rescue_distance_um:g} µm",
            )
        ax.set(
            xlabel="distance from centre to tissue boundary (µm)",
            ylabel="connected-component XY area (µm²)",
            title=f"PROVISIONAL size vs edge distance — {title_suffix}",
        )
        _legend_if_labelled(ax)
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_SIZE_VS_EDGE, dpi=150)
    finally:
        plt.close(fig)

    # 4. Edge-candidate QC: valid-pixel fraction, split by centre location.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        frac_inside = _finite(
            [r.get("valid_pixel_fraction") for r in rows if bool(r.get("centre_inside_tissue"))]
        )
        frac_outside = _finite(
            [
                r.get("valid_pixel_fraction")
                for r in rows
                if not bool(r.get("centre_inside_tissue"))
            ]
        )
        if frac_inside.size:
            axes[0].hist(frac_inside, bins=20, range=(0, 1), color="#1565C0")
        axes[0].set(
            xlabel="valid-pixel fraction of measurement patch",
            ylabel="candidates (centre inside tissue)",
            title="patch clipping — inside tissue",
        )
        rescued = [
            r for r in rows if r.get("refined_candidate_status") == REFINED_MANUAL_REVIEW_EDGE
        ]
        rescued_d = _finite([r.get("distance_to_tissue_um") for r in rescued])
        outside_d = _finite(
            [
                r.get("distance_to_tissue_um")
                for r in rows
                if not bool(r.get("centre_inside_tissue"))
            ]
        )
        if outside_d.size:
            axes[1].hist(outside_d, bins=20, color="#EF6C00", alpha=0.6, label="outside tissue")
        if rescued_d.size:
            axes[1].hist(rescued_d, bins=20, color="#2E7D32", alpha=0.7, label="manual_review_edge")
        if thresholds.edge_rescue_distance_um is not None:
            axes[1].axvline(
                thresholds.edge_rescue_distance_um,
                color="black",
                linestyle="--",
                label=f"edge rescue={thresholds.edge_rescue_distance_um:g} µm",
            )
        axes[1].set(
            xlabel="distance from centre to tissue boundary (µm)",
            ylabel="candidates (centre outside tissue)",
            title="edge candidates",
        )
        _legend_if_labelled(axes[1])
        fig.suptitle(f"PROVISIONAL edge-candidate QC — {title_suffix}")
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_EDGE_QC, dpi=150)
    finally:
        plt.close(fig)

    return {
        "candidate_size_distributions_png": str(channel_dir / PLOT_SIZE),
        "support_planes_distribution_png": str(channel_dir / PLOT_SUPPORT),
        "size_vs_edge_distance_png": str(channel_dir / PLOT_SIZE_VS_EDGE),
        "edge_candidate_qc_png": str(channel_dir / PLOT_EDGE_QC),
    }


def _safe(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number
