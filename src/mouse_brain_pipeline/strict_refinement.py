"""Stricter, auditable post-detection refinement of PROVISIONAL candidates.

This is a stricter sibling of :mod:`size_edge_refinement`. It separates
too-small filtering from low-support filtering, adds explicit support-voxel and
edge-penalty thresholds, and emits a full threshold sweep. It reuses the
connected-component and 7-plane fixed-XY support measurements already recorded
per candidate (never re-measuring raw pixels) and the ORIGINAL (non-eroded)
tissue mask for edge geometry.

Provisional candidates are never called cells and nothing is auto-promoted.
Raw TIFFs, candidate coordinates, detection thresholds, injection masks, and
existing run outputs are read only. Two modes:

* ``report`` -- diagnostics + threshold sweep only; no status changed.
* ``apply``  -- applies ONLY explicitly supplied thresholds; refuses to run
  without at least one size/support threshold.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

# Reuse the validated geometry + coercion helpers from the size/edge module.
from .size_edge_refinement import (
    DIAGNOSTIC_EDGE_SEARCH_UM,
    MINIMUM_PATCH_RADIUS_UM,
    SIZE_ELIGIBLE_STATUSES,
    _to_bool,
    _to_float,
    _to_int,
    distance_to_tissue_um,
    measure_patch_validity,
)

MODE_REPORT = "report"
MODE_APPLY = "apply"

STRICT_RETAINED = "retained_strict"
STRICT_FILTERED_SMALL = "filtered_too_small"
STRICT_FILTERED_LOW_SUPPORT = "filtered_low_support"
STRICT_MANUAL_REVIEW_EDGE = "manual_review_edge"
STRICT_INVALID = "invalid_measurement"

REFINEMENT_FIELDS = [
    "original_status",
    "refined_candidate_status",
    "strict_filter_status",
    "strict_filter_reason",
    "component_area_um2",
    "component_volume_um3",
    "support_voxel_count",
    "maximum_diameter_um",
    "reused_mean_intensity",
    "reused_peak_intensity",
    "centre_inside_tissue",
    "edge_distance_um",
    "edge_distance_capped",
    "valid_pixel_fraction",
    "measurement_clipped",
    "signal_support_overlaps_tissue",
    "edge_rescued",
    "refinement_mode",
]


@dataclass(frozen=True)
class StrictThresholds:
    """Explicitly supplied thresholds. ``None`` means "not supplied"."""

    min_component_area_um2: Optional[float] = None
    min_component_volume_um3: Optional[float] = None
    min_support_planes: Optional[int] = None
    min_support_voxels: Optional[int] = None
    max_edge_distance_penalty_um: Optional[float] = None
    rescue_edge_candidates: bool = False

    def has_any_size_or_support_threshold(self) -> bool:
        return any(
            value is not None
            for value in (
                self.min_component_area_um2,
                self.min_component_volume_um3,
                self.min_support_planes,
                self.min_support_voxels,
            )
        )


@dataclass
class StrictResult:
    channel: str
    section: int
    mode: str
    voxel_zyx_um: tuple[float, float, float]
    thresholds: StrictThresholds
    rows: list[dict]
    summary: dict = field(default_factory=dict)
    sweep_rows: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-candidate measurements (reuse / derive)
# --------------------------------------------------------------------------- #
def derive_metrics(row: dict, voxel_zyx_um: Sequence[float]) -> dict:
    voxel_z, voxel_y, voxel_x = (float(v) for v in voxel_zyx_um)
    voxel_volume_um3 = voxel_z * voxel_y * voxel_x

    volume_um3 = _to_float(row.get("volume_um3"))
    xy_diameter_um = _to_float(row.get("xy_diameter_um"))
    support_plane_count = _to_int(row.get("support_plane_count"), 0)
    mean_intensity = _to_float(row.get("mean_intensity"))
    peak_intensity = _to_float(row.get("peak_intensity"))

    if math.isfinite(xy_diameter_um) and xy_diameter_um > 0:
        component_area_um2 = math.pi * (xy_diameter_um / 2.0) ** 2
        maximum_diameter_um = xy_diameter_um
    else:
        component_area_um2 = math.nan
        maximum_diameter_um = math.nan

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
    return {
        "component_area_um2": component_area_um2,
        "component_volume_um3": volume_um3,
        "support_plane_count": support_plane_count,
        "support_voxel_count": support_voxel_count,
        "maximum_diameter_um": maximum_diameter_um,
        "mean_intensity": mean_intensity,
        "peak_intensity": peak_intensity,
        "component_present": component_present,
        "measurement_valid": _to_bool(row.get("measurement_valid")),
        "invalid_coordinate": _to_bool(row.get("invalid_coordinate")),
    }


def _size_failures(metrics: dict, thresholds: StrictThresholds) -> list[str]:
    reasons: list[str] = []
    area = metrics["component_area_um2"]
    volume = metrics["component_volume_um3"]
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
    return reasons


def _support_failures(metrics: dict, thresholds: StrictThresholds) -> list[str]:
    reasons: list[str] = []
    if thresholds.min_support_planes is not None:
        if metrics["support_plane_count"] < thresholds.min_support_planes:
            reasons.append(
                f"support_plane_count {metrics['support_plane_count']} < "
                f"{thresholds.min_support_planes:g}"
            )
    if thresholds.min_support_voxels is not None:
        if metrics["support_voxel_count"] < thresholds.min_support_voxels:
            reasons.append(
                f"support_voxel_count {metrics['support_voxel_count']} < "
                f"{thresholds.min_support_voxels:g}"
            )
    return reasons


# --------------------------------------------------------------------------- #
# Threshold sweep (report mode)
# --------------------------------------------------------------------------- #
def _sweep_grid(values, points):
    import numpy as np  # noqa: PLC0415

    finite = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if finite.size == 0:
        return [0.0]
    percentiles = np.percentile(finite, points)
    return sorted({round(float(v), 2) for v in np.concatenate(([0.0], percentiles))})


def build_threshold_sweep(metrics_list: list[dict]) -> list[dict]:
    """How many size-eligible candidates would remain across threshold combos."""
    area_grid = _sweep_grid([m["component_area_um2"] for m in metrics_list], [25, 50, 75, 90])
    volume_grid = _sweep_grid([m["component_volume_um3"] for m in metrics_list], [25, 50, 75, 90])
    plane_grid = [1, 2, 3, 4]
    voxel_grid = _sweep_grid([float(m["support_voxel_count"]) for m in metrics_list], [25, 50, 75, 90])
    voxel_grid = sorted({int(round(v)) for v in voxel_grid})

    total = len(metrics_list)
    sweep: list[dict] = []
    for min_area in area_grid:
        for min_volume in volume_grid:
            for min_planes in plane_grid:
                for min_voxels in voxel_grid:
                    remaining = 0
                    for m in metrics_list:
                        if (
                            math.isfinite(m["component_area_um2"])
                            and m["component_area_um2"] >= min_area
                            and math.isfinite(m["component_volume_um3"])
                            and m["component_volume_um3"] >= min_volume
                            and m["support_plane_count"] >= min_planes
                            and m["support_voxel_count"] >= min_voxels
                        ):
                            remaining += 1
                    sweep.append(
                        {
                            "min_component_area_um2": min_area,
                            "min_component_volume_um3": min_volume,
                            "min_support_planes": min_planes,
                            "min_support_voxels": min_voxels,
                            "candidates_evaluated": total,
                            "candidates_remaining": remaining,
                            "candidates_filtered": total - remaining,
                        }
                    )
    return sweep


# --------------------------------------------------------------------------- #
# Core driver
# --------------------------------------------------------------------------- #
def refine_candidates_strict(
    rows: Iterable[dict],
    tissue_mask,
    *,
    voxel_zyx_um: Sequence[float],
    mode: str,
    thresholds: StrictThresholds,
    channel: str,
    section: int,
) -> StrictResult:
    if mode not in (MODE_REPORT, MODE_APPLY):
        raise ValueError(f"mode must be '{MODE_REPORT}' or '{MODE_APPLY}', got {mode!r}")
    if mode == MODE_APPLY and not thresholds.has_any_size_or_support_threshold():
        raise ValueError(
            "apply mode requires at least one explicit size/support threshold "
            "(--min-component-area-um2, --min-component-volume-um3, "
            "--min-support-planes, or --min-support-voxels)"
        )
    if (
        mode == MODE_APPLY
        and thresholds.rescue_edge_candidates
        and thresholds.max_edge_distance_penalty_um is None
    ):
        raise ValueError(
            "--rescue-edge-candidates requires --max-edge-distance-penalty-um "
            "in apply mode"
        )

    voxel_z, voxel_y, voxel_x = (float(v) for v in voxel_zyx_um)
    height, width = int(tissue_mask.shape[0]), int(tissue_mask.shape[1])

    search_um = DIAGNOSTIC_EDGE_SEARCH_UM
    if thresholds.max_edge_distance_penalty_um is not None:
        search_um = max(search_um, float(thresholds.max_edge_distance_penalty_um))
    search_px = max(1, int(math.ceil(search_um / voxel_y)))

    refined_rows: list[dict] = []
    eligible_metrics: list[dict] = []
    for row in rows:
        out = dict(row)
        original_status = str(row.get("current_status", ""))
        metrics = derive_metrics(row, voxel_zyx_um)

        cx = _to_int(row.get("x_local_px", row.get("x_global_px")), -1)
        cy = _to_int(row.get("y_local_px", row.get("y_global_px")), -1)
        in_bounds = 0 <= cy < height and 0 <= cx < width
        centre_inside = bool(in_bounds and bool(tissue_mask[cy, cx]))

        half_um = MINIMUM_PATCH_RADIUS_UM
        if math.isfinite(metrics["maximum_diameter_um"]):
            half_um = max(half_um, metrics["maximum_diameter_um"] / 2.0)
        half_px = max(1, int(math.ceil(half_um / voxel_y)))

        if in_bounds:
            patch = measure_patch_validity(tissue_mask, cy, cx, half_px)
            edge_distance, capped = distance_to_tissue_um(
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
            edge_distance, capped = float(search_px * voxel_y), True

        decision = _classify(
            original_status=original_status,
            metrics=metrics,
            patch=patch,
            centre_inside=centre_inside,
            edge_distance_um=edge_distance,
            thresholds=thresholds,
            mode=mode,
        )

        out.update(
            {
                "original_status": original_status,
                "refined_candidate_status": decision["refined_candidate_status"],
                "strict_filter_status": decision["strict_filter_status"],
                "strict_filter_reason": decision["strict_filter_reason"],
                "component_area_um2": _round(metrics["component_area_um2"], 3),
                "component_volume_um3": _round(metrics["component_volume_um3"], 3),
                "support_voxel_count": metrics["support_voxel_count"],
                "maximum_diameter_um": _round(metrics["maximum_diameter_um"], 3),
                "reused_mean_intensity": _round(metrics["mean_intensity"], 3),
                "reused_peak_intensity": _round(metrics["peak_intensity"], 3),
                "centre_inside_tissue": centre_inside,
                "edge_distance_um": _round(edge_distance, 3),
                "edge_distance_capped": bool(capped),
                "valid_pixel_fraction": _round(patch["valid_pixel_fraction"], 4),
                "measurement_clipped": bool(patch["clipped"]),
                "signal_support_overlaps_tissue": bool(
                    decision["signal_support_overlaps_tissue"]
                ),
                "edge_rescued": bool(decision["edge_rescued"]),
                "refinement_mode": mode,
            }
        )
        refined_rows.append(out)

        if (
            original_status in SIZE_ELIGIBLE_STATUSES
            and centre_inside
            and metrics["component_present"]
            and metrics["measurement_valid"]
            and not metrics["invalid_coordinate"]
        ):
            eligible_metrics.append(metrics)

    sweep_rows = build_threshold_sweep(eligible_metrics) if mode == MODE_REPORT else []
    summary = _build_summary(
        refined_rows,
        eligible_metrics,
        channel=channel,
        section=section,
        mode=mode,
        thresholds=thresholds,
        voxel_zyx_um=(voxel_z, voxel_y, voxel_x),
        mask_shape=(height, width),
        sweep_rows=sweep_rows,
    )
    return StrictResult(
        channel=channel,
        section=section,
        mode=mode,
        voxel_zyx_um=(voxel_z, voxel_y, voxel_x),
        thresholds=thresholds,
        rows=refined_rows,
        summary=summary,
        sweep_rows=sweep_rows,
    )


def _classify(
    *,
    original_status,
    metrics,
    patch,
    centre_inside,
    edge_distance_um,
    thresholds: StrictThresholds,
    mode,
):
    has_signal_support = (
        metrics["measurement_valid"]
        and metrics["support_plane_count"] >= 1
        and patch["overlaps_tissue"]
    )
    base = {
        "signal_support_overlaps_tissue": has_signal_support,
        "edge_rescued": False,
    }

    if mode == MODE_REPORT:
        return {
            **base,
            "refined_candidate_status": original_status,
            "strict_filter_status": "not_applied",
            "strict_filter_reason": "report mode: diagnostics only, no threshold applied",
        }

    invalid = (
        not metrics["measurement_valid"]
        or not metrics["component_present"]
        or metrics["invalid_coordinate"]
    )
    if invalid:
        return {
            **base,
            "refined_candidate_status": STRICT_INVALID,
            "strict_filter_status": "invalid",
            "strict_filter_reason": "measurement invalid or connected component missing",
        }

    size_reasons = _size_failures(metrics, thresholds)
    support_reasons = _support_failures(metrics, thresholds)

    if centre_inside:
        if not size_reasons and not support_reasons:
            return {
                **base,
                "refined_candidate_status": STRICT_RETAINED,
                "strict_filter_status": "adequate",
                "strict_filter_reason": "meets all supplied size and support thresholds",
            }
        # A centre inside tissue is never rejected solely for a clipped patch:
        # when rescue is on and the patch is clipped, route it to review instead.
        if thresholds.rescue_edge_candidates and patch["clipped"]:
            return {
                **base,
                "refined_candidate_status": STRICT_MANUAL_REVIEW_EDGE,
                "strict_filter_status": "edge_clipped_review",
                "strict_filter_reason": (
                    "fails a threshold but measurement patch clipped by an edge: "
                    + "; ".join(size_reasons + support_reasons)
                ),
            }
        if size_reasons:
            return {
                **base,
                "refined_candidate_status": STRICT_FILTERED_SMALL,
                "strict_filter_status": "too_small",
                "strict_filter_reason": "; ".join(size_reasons),
            }
        return {
            **base,
            "refined_candidate_status": STRICT_FILTERED_LOW_SUPPORT,
            "strict_filter_status": "low_support",
            "strict_filter_reason": "; ".join(support_reasons),
        }

    # Centre outside the original tissue mask: never auto-accept.
    rescued = (
        thresholds.rescue_edge_candidates
        and thresholds.max_edge_distance_penalty_um is not None
        and edge_distance_um <= thresholds.max_edge_distance_penalty_um
        and has_signal_support
    )
    if rescued:
        return {
            **base,
            "refined_candidate_status": STRICT_MANUAL_REVIEW_EDGE,
            "strict_filter_status": "edge_rescue",
            "strict_filter_reason": (
                f"centre {edge_distance_um:.2f} um outside tissue within edge "
                f"penalty {thresholds.max_edge_distance_penalty_um:g} um with "
                "signal support overlapping valid tissue"
            ),
            "edge_rescued": True,
        }
    return {
        **base,
        "refined_candidate_status": original_status,
        "strict_filter_status": "preserved",
        "strict_filter_reason": "centre outside tissue and not eligible for edge rescue",
    }


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
    sweep_rows,
):
    from collections import Counter

    original_counts = Counter(r["original_status"] for r in refined_rows)
    refined_counts = Counter(r["refined_candidate_status"] for r in refined_rows)
    status_counts = Counter(r["strict_filter_status"] for r in refined_rows)

    summary = {
        "analysis": "strict post-detection candidate refinement",
        "provisional_candidates": True,
        "candidates_are_not_cells": True,
        "channel": channel,
        "section": section,
        "mode": mode,
        "voxel_size_zyx_um": list(voxel_zyx_um),
        "tissue_mask_shape_yx": list(mask_shape),
        "tissue_mask_erosion_used": False,
        "size_measurement_source": (
            "reused connected-component volume_um3, xy_diameter_um, "
            "support_plane_count and derived support_voxel_count from detection; "
            "mean/peak intensity reused for QC, never used as a size measure"
        ),
        "size_eligible_statuses": sorted(SIZE_ELIGIBLE_STATUSES),
        "thresholds_supplied": {
            "min_component_area_um2": thresholds.min_component_area_um2,
            "min_component_volume_um3": thresholds.min_component_volume_um3,
            "min_support_planes": thresholds.min_support_planes,
            "min_support_voxels": thresholds.min_support_voxels,
            "max_edge_distance_penalty_um": thresholds.max_edge_distance_penalty_um,
            "rescue_edge_candidates": thresholds.rescue_edge_candidates,
        },
        "total_candidates": len(refined_rows),
        "original_status_counts": dict(original_counts),
        "refined_status_counts": dict(refined_counts),
        "strict_filter_status_counts": dict(status_counts),
        "threshold_sweep_population": len(eligible_metrics),
        "threshold_sweep_rows": len(sweep_rows),
    }
    if mode == MODE_REPORT:
        summary["note"] = (
            "report mode: no threshold applied and no candidate status changed"
        )
    else:
        summary["applied"] = {
            "retained_strict": refined_counts.get(STRICT_RETAINED, 0),
            "filtered_too_small": refined_counts.get(STRICT_FILTERED_SMALL, 0),
            "filtered_low_support": refined_counts.get(STRICT_FILTERED_LOW_SUPPORT, 0),
            "manual_review_edge": refined_counts.get(STRICT_MANUAL_REVIEW_EDGE, 0),
            "invalid_measurement": refined_counts.get(STRICT_INVALID, 0),
            "edge_rescued": sum(1 for r in refined_rows if r["edge_rescued"]),
        }
    return summary


# --------------------------------------------------------------------------- #
# Output writing (CSVs + JSON + sweep + plots)
# --------------------------------------------------------------------------- #
AUDIT_CSV = "strict_refined_candidates.csv"
RETAINED_CSV = "strict_retained_candidates.csv"
FILTERED_SMALL_CSV = "strict_filtered_too_small.csv"
FILTERED_LOW_SUPPORT_CSV = "strict_filtered_low_support.csv"
MANUAL_REVIEW_EDGE_CSV = "strict_manual_review_edge.csv"
INVALID_CSV = "strict_invalid_measurements.csv"
SUMMARY_JSON = "strict_refinement_summary.json"
SWEEP_CSV = "strict_threshold_sweep.csv"
PLOT_SIZE = "candidate_size_distributions.png"
PLOT_SUPPORT_PLANES = "support_planes_distribution.png"
PLOT_SUPPORT_VOXELS = "support_voxels_distribution.png"
PLOT_SIZE_VS_SUPPORT = "size_vs_support.png"
PLOT_EDGE_QC = "edge_candidate_qc.png"

SWEEP_FIELDS = [
    "min_component_area_um2",
    "min_component_volume_um3",
    "min_support_planes",
    "min_support_voxels",
    "candidates_evaluated",
    "candidates_remaining",
    "candidates_filtered",
]


def _ordered_fieldnames(rows: list[dict]) -> list[str]:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _write_rows_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_strict_outputs(channel_dir, result: StrictResult, *, make_plots: bool = True,
                         plot_size_distributions: bool = True) -> dict:
    channel_dir = Path(channel_dir)
    channel_dir.mkdir(parents=True, exist_ok=True)
    rows = result.rows
    fieldnames = _ordered_fieldnames(rows)

    _write_rows_csv(channel_dir / AUDIT_CSV, rows, fieldnames)
    buckets = {
        RETAINED_CSV: STRICT_RETAINED,
        FILTERED_SMALL_CSV: STRICT_FILTERED_SMALL,
        FILTERED_LOW_SUPPORT_CSV: STRICT_FILTERED_LOW_SUPPORT,
        MANUAL_REVIEW_EDGE_CSV: STRICT_MANUAL_REVIEW_EDGE,
        INVALID_CSV: STRICT_INVALID,
    }
    for filename, status in buckets.items():
        subset = [r for r in rows if r["refined_candidate_status"] == status]
        _write_rows_csv(channel_dir / filename, subset, fieldnames)

    _write_rows_csv(channel_dir / SWEEP_CSV, result.sweep_rows, SWEEP_FIELDS)
    (channel_dir / SUMMARY_JSON).write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )

    outputs = {
        "audit_csv": str(channel_dir / AUDIT_CSV),
        "retained_csv": str(channel_dir / RETAINED_CSV),
        "filtered_too_small_csv": str(channel_dir / FILTERED_SMALL_CSV),
        "filtered_low_support_csv": str(channel_dir / FILTERED_LOW_SUPPORT_CSV),
        "manual_review_edge_csv": str(channel_dir / MANUAL_REVIEW_EDGE_CSV),
        "invalid_measurements_csv": str(channel_dir / INVALID_CSV),
        "threshold_sweep_csv": str(channel_dir / SWEEP_CSV),
        "summary_json": str(channel_dir / SUMMARY_JSON),
    }
    if make_plots:
        outputs.update(_write_plots(
            channel_dir, result, plot_size_distributions=plot_size_distributions))
    return outputs


def _finite(values):
    import numpy as np  # noqa: PLC0415

    arr = np.array(values, dtype=float)
    return arr[np.isfinite(arr)]


def _safe(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _legend_if_labelled(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=8)


def _write_plots(channel_dir: Path, result: StrictResult, *,
                 plot_size_distributions: bool = True) -> dict:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    rows = result.rows
    channel = result.channel
    section = result.section
    thresholds = result.thresholds
    colour = "#2E7D32" if "green" in channel else "#C62828"
    suffix = f"{channel}, section {section:03d}, {result.mode} mode"

    area = _finite([r.get("component_area_um2") for r in rows])
    volume = _finite([r.get("component_volume_um3") for r in rows])
    planes = [int(r.get("support_plane_count") or 0) for r in rows]
    voxels = _finite([r.get("support_voxel_count") for r in rows])

    # 1. Size distributions -- OFF by default; rendered only when
    #    candidate_size_distributions is explicitly enabled in config.
    if plot_size_distributions:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        try:
            if area.size:
                axes[0].hist(area, bins=40, color=colour)
            axes[0].set(xlabel="connected-component XY area (µm²)", ylabel="candidates")
            if thresholds.min_component_area_um2 is not None:
                axes[0].axvline(thresholds.min_component_area_um2, color="black", linestyle="--",
                                label=f"min area={thresholds.min_component_area_um2:g}")
                axes[0].legend(fontsize=8)
            if volume.size:
                axes[1].hist(volume, bins=40, color=colour)
            axes[1].set(xlabel="connected-component volume (µm³)", ylabel="candidates")
            if thresholds.min_component_volume_um3 is not None:
                axes[1].axvline(thresholds.min_component_volume_um3, color="black", linestyle="--",
                                label=f"min volume={thresholds.min_component_volume_um3:g}")
                axes[1].legend(fontsize=8)
            fig.suptitle(f"PROVISIONAL candidate size distributions — {suffix}")
            fig.tight_layout()
            fig.savefig(channel_dir / PLOT_SIZE, dpi=150)
        finally:
            plt.close(fig)

    # 2. Support-plane distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        counts = np.bincount(np.clip(planes, 0, 7), minlength=8)
        ax.bar(range(8), counts, color=colour)
        if thresholds.min_support_planes is not None:
            ax.axvline(thresholds.min_support_planes - 0.5, color="black", linestyle="--",
                       label=f"min planes={thresholds.min_support_planes:g}")
            ax.legend(fontsize=8)
        ax.set(xlabel="planes with valid signal support", ylabel="candidates",
               title=f"PROVISIONAL support-plane distribution — {suffix}")
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_SUPPORT_PLANES, dpi=150)
    finally:
        plt.close(fig)

    # 3. Support-voxel distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if voxels.size:
            ax.hist(voxels, bins=40, color=colour)
        if thresholds.min_support_voxels is not None:
            ax.axvline(thresholds.min_support_voxels, color="black", linestyle="--",
                       label=f"min voxels={thresholds.min_support_voxels:g}")
            ax.legend(fontsize=8)
        ax.set(xlabel="supporting voxels", ylabel="candidates",
               title=f"PROVISIONAL support-voxel distribution — {suffix}")
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_SUPPORT_VOXELS, dpi=150)
    finally:
        plt.close(fig)

    # 4. Size vs support.
    fig, ax = plt.subplots(figsize=(9, 6))
    try:
        xs = np.array([_safe(r.get("support_voxel_count")) for r in rows])
        ys = np.array([_safe(r.get("component_area_um2")) for r in rows])
        finite = np.isfinite(xs) & np.isfinite(ys)
        if finite.any():
            ax.scatter(xs[finite], ys[finite], s=6, alpha=0.4, color=colour)
        if thresholds.min_support_voxels is not None:
            ax.axvline(thresholds.min_support_voxels, color="black", linestyle="--",
                       label=f"min voxels={thresholds.min_support_voxels:g}")
        if thresholds.min_component_area_um2 is not None:
            ax.axhline(thresholds.min_component_area_um2, color="grey", linestyle=":",
                       label=f"min area={thresholds.min_component_area_um2:g}")
        ax.set(xlabel="supporting voxels", ylabel="connected-component XY area (µm²)",
               title=f"PROVISIONAL size vs support — {suffix}")
        _legend_if_labelled(ax)
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_SIZE_VS_SUPPORT, dpi=150)
    finally:
        plt.close(fig)

    # 5. Edge-candidate QC.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        frac_inside = _finite(
            [r.get("valid_pixel_fraction") for r in rows if bool(r.get("centre_inside_tissue"))]
        )
        if frac_inside.size:
            axes[0].hist(frac_inside, bins=20, range=(0, 1), color="#1565C0")
        axes[0].set(xlabel="valid-pixel fraction of measurement patch",
                    ylabel="candidates (centre inside tissue)",
                    title="patch clipping — inside tissue")
        outside_d = _finite(
            [r.get("edge_distance_um") for r in rows if not bool(r.get("centre_inside_tissue"))]
        )
        rescued_d = _finite(
            [r.get("edge_distance_um") for r in rows
             if r.get("refined_candidate_status") == STRICT_MANUAL_REVIEW_EDGE]
        )
        if outside_d.size:
            axes[1].hist(outside_d, bins=20, color="#EF6C00", alpha=0.6, label="outside tissue")
        if rescued_d.size:
            axes[1].hist(rescued_d, bins=20, color="#2E7D32", alpha=0.7, label="manual_review_edge")
        if thresholds.max_edge_distance_penalty_um is not None:
            axes[1].axvline(thresholds.max_edge_distance_penalty_um, color="black", linestyle="--",
                            label=f"edge penalty={thresholds.max_edge_distance_penalty_um:g} µm")
        axes[1].set(xlabel="distance from centre to tissue boundary (µm)",
                    ylabel="candidates (centre outside tissue)", title="edge candidates")
        _legend_if_labelled(axes[1])
        fig.suptitle(f"PROVISIONAL edge-candidate QC — {suffix}")
        fig.tight_layout()
        fig.savefig(channel_dir / PLOT_EDGE_QC, dpi=150)
    finally:
        plt.close(fig)

    outputs = {
        "support_planes_distribution_png": str(channel_dir / PLOT_SUPPORT_PLANES),
        "support_voxels_distribution_png": str(channel_dir / PLOT_SUPPORT_VOXELS),
        "size_vs_support_png": str(channel_dir / PLOT_SIZE_VS_SUPPORT),
        "edge_candidate_qc_png": str(channel_dir / PLOT_EDGE_QC),
    }
    if plot_size_distributions:
        outputs["candidate_size_distributions_png"] = str(channel_dir / PLOT_SIZE)
    return outputs
