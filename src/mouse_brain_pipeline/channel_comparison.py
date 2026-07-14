"""Auditable green/red comparison for PROVISIONAL candidate detections.

Every selected row from ``all_candidates.csv`` is measured in both biological
signal channels at the candidate's immutable full-resolution XY coordinate and
recorded optical plane.  Report mode is diagnostic only.  Apply mode writes a
separate refined decision while preserving the source row and ``current_status``.

This module only reads source TIFFs and run files.  All outputs go to a fresh,
explicit output directory; candidates are partitioned into four comparison
classes but are never deleted from the audit table.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .channels import CHANNEL_2_SIGNAL, GREEN_SIGNAL

MODE_REPORT = "report"
MODE_APPLY = "apply"

GREEN_DOMINANT = "green_dominant"
RED_DOMINANT = "red_dominant"
BOTH = "both"
UNCLEAR = "unclear"
DOMINANT_CHANNELS = (GREEN_DOMINANT, RED_DOMINANT, BOTH, UNCLEAR)

_DOMINANT_RGB = {
    GREEN_DOMINANT: (0, 255, 0),
    RED_DOMINANT: (255, 45, 45),
    BOTH: (255, 225, 0),
    UNCLEAR: (175, 175, 175),
}

AUDIT_CSV = "channel_comparison_candidates.csv"
SUMMARY_CSV = "channel_comparison_summary.csv"
SUMMARY_JSON = "channel_comparison_summary.json"
RATIO_PNG = "green_red_ratio_histograms.png"
SNR_PNG = "green_vs_red_snr_scatter.png"
OVERLAY_PNG = "green_red_overlay_qc.png"

CATEGORY_FILES = {
    GREEN_DOMINANT: "green_dominant_candidates.csv",
    RED_DOMINANT: "red_dominant_candidates.csv",
    BOTH: "both_channel_candidates.csv",
    UNCLEAR: "unclear_candidates.csv",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "channel",
    "section",
    "x_global_px",
    "y_global_px",
    "current_status",
}

COMPARISON_FIELDS = [
    "original_status",
    "refined_candidate_status",
    "channel_comparison_decision",
    "channel_comparison_reason",
    "comparison_mode",
    "comparison_z_index",
    "comparison_optical_plane",
    "comparison_z_mapping",
    "green_peak",
    "red_peak",
    "green_local_background",
    "red_local_background",
    "green_snr",
    "red_snr",
    "green_signal_above_background",
    "red_signal_above_background",
    "red_green_ratio",
    "green_measurement_valid",
    "red_measurement_valid",
    "green_measurement_reason",
    "red_measurement_reason",
    "nearest_opposite_channel_candidate_distance_um",
    "matched_opposite_channel_candidate_id",
    "dominant_channel",
]

SUMMARY_COLUMNS = [
    "section",
    "detection_channel",
    "dominant_channel",
    "candidate_count",
]


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_int(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _validate_thresholds(min_dominance_ratio, min_snr, max_match_distance_um) -> None:
    values = {
        "min_dominance_ratio": min_dominance_ratio,
        "min_snr": min_snr,
        "max_match_distance_um": max_match_distance_um,
    }
    for name, value in values.items():
        if not _finite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if float(min_dominance_ratio) < 1.0:
        raise ValueError("min_dominance_ratio must be >= 1.0")
    if float(min_snr) < 0.0:
        raise ValueError("min_snr must be >= 0.0")
    if float(max_match_distance_um) < 0.0:
        raise ValueError("max_match_distance_um must be >= 0.0")


def signal_above_background(peak, background) -> float:
    """Return a non-negative background-subtracted signal."""
    if not (_finite(peak) and _finite(background)):
        return 0.0
    return max(float(peak) - float(background), 0.0)


def red_green_ratio(green_signal, red_signal):
    """Background-subtracted red/green ratio, ``None`` when both are zero."""
    green = float(green_signal)
    red = float(red_signal)
    if green > 0.0:
        return red / green
    if red > 0.0:
        return math.inf
    return None


def classify_dominant_channel(
    *,
    green_snr,
    red_snr,
    green_signal,
    red_signal,
    green_valid,
    red_valid,
    min_dominance_ratio,
    min_snr,
) -> str:
    """Classify two measurements with an explicit, symmetric rule.

    Both strong channels are labelled ``both`` before applying dominance tests.
    A single strong channel must also exceed the other channel's measured,
    background-subtracted signal by ``min_dominance_ratio``.  Invalid pairs and
    weak or ambiguous pairs are ``unclear``.
    """
    if not (bool(green_valid) and bool(red_valid)):
        return UNCLEAR
    green_strong = _finite(green_snr) and float(green_snr) >= float(min_snr)
    red_strong = _finite(red_snr) and float(red_snr) >= float(min_snr)
    green_signal = float(green_signal) if _finite(green_signal) else 0.0
    red_signal = float(red_signal) if _finite(red_signal) else 0.0

    if green_strong and red_strong:
        return BOTH
    if (
        green_strong
        and green_signal >= float(min_dominance_ratio) * red_signal
    ):
        return GREEN_DOMINANT
    if red_strong and red_signal >= float(min_dominance_ratio) * green_signal:
        return RED_DOMINANT
    return UNCLEAR


def _read_candidates(path: Path, sections: set[int]) -> tuple[list[dict], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate table: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_CANDIDATE_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        selected = []
        for row in reader:
            section = _as_int(row.get("section"))
            if section in sections:
                selected.append(row)

    if not selected:
        raise ValueError(
            f"No candidates for requested section(s) {sorted(sections)} in {path}"
        )
    found_sections = {_as_int(row.get("section")) for row in selected}
    missing_sections = sorted(sections - found_sections)
    if missing_sections:
        raise ValueError(
            f"No candidates for requested section(s) {missing_sections} in {path}"
        )
    ids = [str(row.get("candidate_id", "")) for row in selected]
    if any(not candidate_id for candidate_id in ids):
        raise ValueError("Every candidate must have a non-empty candidate_id")
    duplicates = sorted(
        candidate_id for candidate_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate candidate_id values: {', '.join(duplicates)}")
    invalid_channels = sorted(
        {
            str(row.get("channel"))
            for row in selected
            if row.get("channel") not in (GREEN_SIGNAL, CHANNEL_2_SIGNAL)
        }
    )
    if invalid_channels:
        raise ValueError(
            "Unsupported candidate channel(s): " + ", ".join(invalid_channels)
        )
    return selected, fieldnames


class _PlaneStack:
    """Small lazy ``(z, y, x)`` adapter around read-only TIFF planes."""

    def __init__(self, planes):
        self.planes = list(planes)
        height, width = self.planes[0].shape
        self.shape = (len(self.planes), int(height), int(width))

    def __getitem__(self, z):
        return self.planes[z]


def _load_section_stacks(indexes, section: int, expected_planes: int):
    import numpy as np  # noqa: PLC0415

    from .review_patches import ordered_section_planes
    from .seven_plane_qc import _read_plane

    stacks = {}
    ordered_by_channel = {}
    plane_numbers_by_channel = {}
    shapes = {}
    for channel in (GREEN_SIGNAL, CHANNEL_2_SIGNAL):
        index = indexes[channel]
        duplicate_planes = sorted(
            plane for (candidate_section, plane) in index.duplicates
            if int(candidate_section) == int(section)
        )
        if duplicate_planes:
            raise ValueError(
                f"Duplicate {channel} TIFF planes for section {section}: "
                f"{duplicate_planes}"
            )
        ordered = ordered_section_planes(index, section)
        if len(ordered) != int(expected_planes):
            raise FileNotFoundError(
                f"Expected {expected_planes} {channel} TIFF planes for section "
                f"{section}, found {len(ordered)}"
            )
        plane_numbers = tuple(int(plane) for plane, _path in ordered)
        planes = [_read_plane(path) for _plane, path in ordered]
        if any(np.asarray(plane).ndim != 2 for plane in planes):
            raise ValueError(f"Every {channel} source TIFF plane must be 2-D")
        channel_shapes = {tuple(int(v) for v in plane.shape) for plane in planes}
        if len(channel_shapes) != 1:
            raise ValueError(
                f"Mismatched {channel} TIFF shapes in section {section}: "
                f"{sorted(channel_shapes)}"
            )
        stacks[channel] = _PlaneStack(planes)
        ordered_by_channel[channel] = ordered
        plane_numbers_by_channel[channel] = plane_numbers
        shapes[channel] = next(iter(channel_shapes))

    if plane_numbers_by_channel[GREEN_SIGNAL] != plane_numbers_by_channel[CHANNEL_2_SIGNAL]:
        raise ValueError(
            f"Green/red optical-plane numbers differ for section {section}: "
            f"{plane_numbers_by_channel[GREEN_SIGNAL]} vs "
            f"{plane_numbers_by_channel[CHANNEL_2_SIGNAL]}"
        )
    if shapes[GREEN_SIGNAL] != shapes[CHANNEL_2_SIGNAL]:
        raise ValueError(
            f"Green/red TIFF shapes differ for section {section}: "
            f"{shapes[GREEN_SIGNAL]} vs {shapes[CHANNEL_2_SIGNAL]}"
        )
    return (
        stacks,
        ordered_by_channel,
        plane_numbers_by_channel[GREEN_SIGNAL],
        shapes[GREEN_SIGNAL],
    )


def _candidate_z(row: dict, plane_numbers: Sequence[int]):
    z_index = _as_int(row.get("z_index"))
    optical_plane = _as_int(row.get("optical_plane"))
    optical_index = None
    if optical_plane in plane_numbers:
        optical_index = plane_numbers.index(optical_plane)

    if optical_index is not None:
        mapping = "optical_plane"
        if z_index is not None and z_index != optical_index:
            mapping = "optical_plane_z_index_mismatch"
        return optical_index, int(plane_numbers[optical_index]), mapping
    if z_index is not None and 0 <= z_index < len(plane_numbers):
        return z_index, int(plane_numbers[z_index]), "z_index"
    return None, optical_plane, "invalid_candidate_z"


def _invalid_measurement(reason: str) -> dict:
    return {
        "peak": math.nan,
        "local_background": math.nan,
        "snr": math.nan,
        "valid": False,
        "reason": reason,
    }


def _measure_at_location(stack, x, y, z_index, tissue_plane, params, voxel_y_um):
    from .candidate_detection import measure_fixed_xy_profile  # noqa: PLC0415

    if z_index is None:
        return _invalid_measurement("invalid_candidate_z")
    if not (_finite(x) and _finite(y)):
        return _invalid_measurement("invalid_candidate_xy")
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    _z, height, width = stack.shape
    if not (0 <= cx < width and 0 <= cy < height):
        return _invalid_measurement("candidate_xy_outside_source_tiff")
    measurements, _profile, _peak_z, _support = measure_fixed_xy_profile(
        stack,
        tissue_plane,
        cy,
        cx,
        params,
        voxel_y_um=float(voxel_y_um),
    )
    measurement = measurements[int(z_index)]
    return {
        "peak": measurement.get("central_signal", math.nan),
        "local_background": measurement.get("background_median", math.nan),
        "snr": measurement.get("contrast", math.nan),
        "valid": bool(measurement.get("measurement_valid", False)),
        "reason": measurement.get("measurement_reason", ""),
    }


def _matching_fields(rows: list[dict], voxel_y_um: float, voxel_x_um: float,
                     max_match_distance_um: float) -> dict[int, dict]:
    """Nearest distances plus deterministic one-to-one matches, per section."""
    import numpy as np  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    result = {
        index: {
            "nearest_opposite_channel_candidate_distance_um": math.nan,
            "matched_opposite_channel_candidate_id": "",
        }
        for index in range(len(rows))
    }
    by_section: dict[int, dict[str, list[int]]] = {}
    for index, row in enumerate(rows):
        section = _as_int(row.get("section"))
        by_section.setdefault(
            section, {GREEN_SIGNAL: [], CHANNEL_2_SIGNAL: []}
        )[row["channel"]].append(index)

    def valid_xy(index):
        row = rows[index]
        return _finite(row.get("x_global_px")) and _finite(row.get("y_global_px"))

    def coords(indices):
        return np.asarray(
            [
                (
                    float(rows[index]["x_global_px"]) * float(voxel_x_um),
                    float(rows[index]["y_global_px"]) * float(voxel_y_um),
                )
                for index in indices
            ],
            dtype=float,
        )

    for groups in by_section.values():
        green_indices = [index for index in groups[GREEN_SIGNAL] if valid_xy(index)]
        red_indices = [index for index in groups[CHANNEL_2_SIGNAL] if valid_xy(index)]
        if not green_indices or not red_indices:
            continue
        green_xy = coords(green_indices)
        red_xy = coords(red_indices)
        green_tree = cKDTree(green_xy)
        red_tree = cKDTree(red_xy)

        green_dist, _green_nearest = red_tree.query(green_xy, k=1)
        red_dist, _red_nearest = green_tree.query(red_xy, k=1)
        for local_index, distance in enumerate(green_dist):
            result[green_indices[local_index]][
                "nearest_opposite_channel_candidate_distance_um"
            ] = float(distance)
        for local_index, distance in enumerate(red_dist):
            result[red_indices[local_index]][
                "nearest_opposite_channel_candidate_distance_um"
            ] = float(distance)

        pairs = []
        neighbours = red_tree.query_ball_point(
            green_xy, r=float(max_match_distance_um)
        )
        for green_local, red_locals in enumerate(neighbours):
            for red_local in red_locals:
                distance = float(np.linalg.norm(green_xy[green_local] - red_xy[red_local]))
                pairs.append((distance, green_local, red_local))
        used_green = set()
        used_red = set()
        for _distance, green_local, red_local in sorted(
            pairs,
            key=lambda item: (
                item[0],
                str(rows[green_indices[item[1]]]["candidate_id"]),
                str(rows[red_indices[item[2]]]["candidate_id"]),
            ),
        ):
            if green_local in used_green or red_local in used_red:
                continue
            used_green.add(green_local)
            used_red.add(red_local)
            green_index = green_indices[green_local]
            red_index = red_indices[red_local]
            result[green_index]["matched_opposite_channel_candidate_id"] = str(
                rows[red_index]["candidate_id"]
            )
            result[red_index]["matched_opposite_channel_candidate_id"] = str(
                rows[green_index]["candidate_id"]
            )
    return result


def _apply_decision(row: dict, dominant_channel: str, mode: str):
    original = str(row.get("current_status", ""))
    if mode == MODE_REPORT:
        return {
            "original_status": original,
            "refined_candidate_status": original,
            "channel_comparison_decision": "not_applied",
            "channel_comparison_reason": (
                "report mode: diagnostics only; candidate status unchanged"
            ),
        }

    expected = (
        GREEN_DOMINANT if row.get("channel") == GREEN_SIGNAL else RED_DOMINANT
    )
    if dominant_channel == expected:
        return {
            "original_status": original,
            "refined_candidate_status": f"retained_{expected}",
            "channel_comparison_decision": "retained_expected_channel",
            "channel_comparison_reason": (
                f"measured {expected} at the candidate's recorded XY/Z location"
            ),
        }
    return {
        "original_status": original,
        "refined_candidate_status": "not_retained_channel_comparison",
        "channel_comparison_decision": "not_retained_expected_channel",
        "channel_comparison_reason": (
            f"detected in {row.get('channel')} but measured {dominant_channel}"
        ),
    }


def _measure_rows(rows, *, stacks_by_section, plane_numbers_by_section,
                  params, voxel_y_um, matches, mode, min_dominance_ratio, min_snr):
    import numpy as np  # noqa: PLC0415

    measured_rows = []
    for index, row in enumerate(rows):
        section = _as_int(row.get("section"))
        stacks = stacks_by_section[section]
        plane_numbers = plane_numbers_by_section[section]
        z_index, optical_plane, z_mapping = _candidate_z(row, plane_numbers)
        height, width = stacks[GREEN_SIGNAL].shape[1:]
        tissue_plane = np.broadcast_to(np.bool_(True), (height, width))
        green = _measure_at_location(
            stacks[GREEN_SIGNAL], row.get("x_global_px"), row.get("y_global_px"),
            z_index, tissue_plane, params, voxel_y_um,
        )
        red = _measure_at_location(
            stacks[CHANNEL_2_SIGNAL], row.get("x_global_px"), row.get("y_global_px"),
            z_index, tissue_plane, params, voxel_y_um,
        )
        green_signal = signal_above_background(
            green["peak"], green["local_background"]
        )
        red_signal = signal_above_background(red["peak"], red["local_background"])
        ratio = red_green_ratio(green_signal, red_signal)
        dominant = classify_dominant_channel(
            green_snr=green["snr"],
            red_snr=red["snr"],
            green_signal=green_signal,
            red_signal=red_signal,
            green_valid=green["valid"],
            red_valid=red["valid"],
            min_dominance_ratio=min_dominance_ratio,
            min_snr=min_snr,
        )
        output = dict(row)
        output.update(_apply_decision(row, dominant, mode))
        output.update(
            {
                "comparison_mode": mode,
                "comparison_z_index": "" if z_index is None else int(z_index),
                "comparison_optical_plane": (
                    "" if optical_plane is None else int(optical_plane)
                ),
                "comparison_z_mapping": z_mapping,
                "green_peak": green["peak"],
                "red_peak": red["peak"],
                "green_local_background": green["local_background"],
                "red_local_background": red["local_background"],
                "green_snr": green["snr"],
                "red_snr": red["snr"],
                "green_signal_above_background": green_signal,
                "red_signal_above_background": red_signal,
                "red_green_ratio": ratio,
                "green_measurement_valid": green["valid"],
                "red_measurement_valid": red["valid"],
                "green_measurement_reason": green["reason"],
                "red_measurement_reason": red["reason"],
                **matches[index],
                "dominant_channel": dominant,
            }
        )
        measured_rows.append(output)
    return measured_rows


def summarize_rows(rows: Iterable[dict], sections: Sequence[int]) -> list[dict]:
    rows = list(rows)
    section_groups = [str(section) for section in sorted({int(s) for s in sections})]
    section_groups.append("all")
    channel_groups = (GREEN_SIGNAL, CHANNEL_2_SIGNAL, "all")
    output = []
    for section_group in section_groups:
        for channel_group in channel_groups:
            for dominant in DOMINANT_CHANNELS:
                count = sum(
                    1
                    for row in rows
                    if row.get("dominant_channel") == dominant
                    and (
                        section_group == "all"
                        or str(_as_int(row.get("section"))) == section_group
                    )
                    and (
                        channel_group == "all"
                        or row.get("channel") == channel_group
                    )
                )
                output.append(
                    {
                        "section": section_group,
                        "detection_channel": channel_group,
                        "dominant_channel": dominant,
                        "candidate_count": count,
                    }
                )
    return output


def _fieldnames(input_fieldnames: Sequence[str]) -> list[str]:
    fields = list(input_fieldnames)
    for field in COMPARISON_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return round(value, 6)
    return value


def _write_rows(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fieldnames})


def _write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_ratio_histograms(path: Path, rows: Sequence[dict], min_dominance_ratio: float):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    try:
        for axis, channel, colour in zip(
            axes,
            (GREEN_SIGNAL, CHANNEL_2_SIGNAL),
            ("#19A44B", "#D62F2F"),
        ):
            channel_rows = [row for row in rows if row.get("channel") == channel]
            ratios = []
            zero_count = infinite_count = 0
            for row in channel_rows:
                ratio = row.get("red_green_ratio")
                if ratio is None:
                    continue
                if not _finite(ratio):
                    infinite_count += 1
                elif float(ratio) <= 0.0:
                    zero_count += 1
                else:
                    ratios.append(math.log10(float(ratio)))
            if ratios:
                axis.hist(ratios, bins=min(40, max(5, len(ratios))), color=colour)
            axis.axvline(
                math.log10(1.0 / float(min_dominance_ratio)),
                color="#19A44B", linestyle="--", linewidth=1,
                label="green dominance boundary",
            )
            axis.axvline(
                math.log10(float(min_dominance_ratio)),
                color="#D62F2F", linestyle="--", linewidth=1,
                label="red dominance boundary",
            )
            axis.text(
                0.02, 0.97,
                f"zero ratios: {zero_count}\ninfinite ratios: {infinite_count}",
                transform=axis.transAxes, va="top", fontsize=8,
            )
            axis.set(
                xlabel="log10(background-subtracted red / green)",
                title=f"PROVISIONAL candidates detected in {channel}",
            )
        axes[0].set_ylabel("candidate count")
        axes[1].legend(fontsize=7, loc="lower right")
        fig.suptitle("PROVISIONAL candidates - green/red ratio diagnostics")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _write_snr_scatter(path: Path, rows: Sequence[dict], min_snr: float):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axis = plt.subplots(figsize=(8, 7))
    try:
        for dominant in DOMINANT_CHANNELS:
            points = [
                (float(row["green_snr"]), float(row["red_snr"]))
                for row in rows
                if row.get("dominant_channel") == dominant
                and _finite(row.get("green_snr"))
                and _finite(row.get("red_snr"))
            ]
            if points:
                xs, ys = zip(*points)
                colour = tuple(value / 255.0 for value in _DOMINANT_RGB[dominant])
                axis.scatter(xs, ys, s=18, alpha=0.7, color=colour, label=dominant)
        axis.axvline(float(min_snr), color="black", linestyle="--", linewidth=1)
        axis.axhline(float(min_snr), color="black", linestyle="--", linewidth=1)
        axis.set(
            xlabel="green SNR (local robust contrast)",
            ylabel="red SNR (local robust contrast)",
            title="PROVISIONAL candidates - green vs red SNR",
        )
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _small_projection(ordered_planes, max_dim: int):
    import numpy as np  # noqa: PLC0415

    from .qc_native import apply_window_uint8
    from .seven_plane_qc import _read_plane

    first = _read_plane(ordered_planes[0][1])
    height, width = first.shape
    step = max(1, int(math.ceil(max(height, width) / float(max_dim))))
    projection = None
    for _plane, source in ordered_planes:
        sampled = np.asarray(_read_plane(source)[::step, ::step])
        projection = (
            np.array(sampled, copy=True)
            if projection is None
            else np.maximum(projection, sampled)
        )
    finite = projection[np.isfinite(projection)]
    non_padding = finite[finite != 0]
    display_values = non_padding if non_padding.size else finite
    if display_values.size:
        low, high = np.percentile(display_values, [1.0, 99.7])
    else:
        low, high = 0.0, 1.0
    return apply_window_uint8(projection, float(low), float(high)), step


def _section_overlay(ordered_by_channel, rows: Sequence[dict], section: int,
                     max_dim: int):
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    green, green_step = _small_projection(
        ordered_by_channel[GREEN_SIGNAL], max_dim
    )
    red, red_step = _small_projection(
        ordered_by_channel[CHANNEL_2_SIGNAL], max_dim
    )
    if green_step != red_step or green.shape != red.shape:
        raise ValueError("Green/red QC projection geometry differs")
    height, width = green.shape
    composite = np.zeros((height, width, 3), dtype=np.uint8)
    composite[:, :, 0] = red
    composite[:, :, 1] = green

    title_height = 42
    image = Image.new("RGB", (width, height + title_height), (0, 0, 0))
    image.paste(Image.fromarray(composite, "RGB"), (0, title_height))
    draw = ImageDraw.Draw(image)
    draw.text(
        (5, 4), f"PROVISIONAL candidates - section {int(section):03d}",
        fill=(255, 255, 255),
    )
    draw.text(
        (5, 20), "green=green_dominant red=red_dominant yellow=both gray=unclear",
        fill=(220, 220, 220),
    )

    section_rows = [row for row in rows if _as_int(row.get("section")) == int(section)]
    by_id = {str(row.get("candidate_id")): row for row in section_rows}

    def point(row):
        if not (_finite(row.get("x_global_px")) and _finite(row.get("y_global_px"))):
            return None
        x = int(round(float(row["x_global_px"]) / green_step))
        y = int(round(float(row["y_global_px"]) / green_step)) + title_height
        if 0 <= x < width and title_height <= y < height + title_height:
            return x, y
        return None

    connected = set()
    for row in section_rows:
        matched_id = str(row.get("matched_opposite_channel_candidate_id") or "")
        if not matched_id or matched_id not in by_id:
            continue
        edge = tuple(sorted((str(row.get("candidate_id")), matched_id)))
        if edge in connected:
            continue
        connected.add(edge)
        first = point(row)
        second = point(by_id[matched_id])
        if first is not None and second is not None:
            draw.line([first, second], fill=(255, 255, 255), width=1)
            for x, y in (first, second):
                draw.rectangle([x - 5, y - 5, x + 5, y + 5], outline=(255, 255, 255))

    radius = max(2, width // 350)
    for row in section_rows:
        candidate_point = point(row)
        if candidate_point is None:
            continue
        x, y = candidate_point
        colour = _DOMINANT_RGB.get(row.get("dominant_channel"), _DOMINANT_RGB[UNCLEAR])
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=colour, width=2,
        )
    return image


def _write_overlay(path: Path, ordered_by_section, rows: Sequence[dict], max_dim: int):
    from PIL import Image  # noqa: PLC0415

    panels = [
        _section_overlay(ordered_by_section[section], rows, section, max_dim)
        for section in sorted(ordered_by_section)
    ]
    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels) + max(0, len(panels) - 1) * 4
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    y = 0
    for panel in panels:
        canvas.paste(panel, (0, y))
        y += panel.height + 4
    canvas.save(path, format="PNG")


def _prepare_output_dir(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing channel-comparison outputs: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def run_channel_comparison(
    *,
    config,
    run_dir,
    sections,
    out_dir,
    mode,
    min_dominance_ratio,
    min_snr,
    max_match_distance_um,
    render_plots: bool = True,
) -> dict:
    """Measure, classify, match and write a fresh channel-comparison report."""
    from .audit import index_channel  # noqa: PLC0415
    from .candidate_detection import params_from_config  # noqa: PLC0415

    if mode not in (MODE_REPORT, MODE_APPLY):
        raise ValueError(f"mode must be '{MODE_REPORT}' or '{MODE_APPLY}', got {mode!r}")
    _validate_thresholds(min_dominance_ratio, min_snr, max_match_distance_um)
    section_set = {int(section) for section in sections}
    if not section_set:
        raise ValueError("At least one --section is required")

    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir == run_dir:
        raise ValueError("--out-dir must be separate from --run-dir")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing channel-comparison outputs: {out_dir}"
        )

    candidate_path = run_dir / "all_candidates.csv"
    candidates, input_fieldnames = _read_candidates(candidate_path, section_set)
    indexes = {
        GREEN_SIGNAL: index_channel(
            GREEN_SIGNAL,
            config.data.green_signal_dir,
            config.data.filename_regex,
        ),
        CHANNEL_2_SIGNAL: index_channel(
            CHANNEL_2_SIGNAL,
            config.data.channel_2_signal_dir,
            config.data.filename_regex,
        ),
    }
    expected_planes = int(config.acquisition.planes_per_section)
    stacks_by_section = {}
    ordered_by_section = {}
    plane_numbers_by_section = {}
    for section in sorted(section_set):
        stacks, ordered, plane_numbers, _shape = _load_section_stacks(
            indexes, section, expected_planes
        )
        stacks_by_section[section] = stacks
        ordered_by_section[section] = ordered
        plane_numbers_by_section[section] = plane_numbers

    matches = _matching_fields(
        candidates,
        float(config.acquisition.voxel_size_y_um),
        float(config.acquisition.voxel_size_x_um),
        float(max_match_distance_um),
    )
    rows = _measure_rows(
        candidates,
        stacks_by_section=stacks_by_section,
        plane_numbers_by_section=plane_numbers_by_section,
        params=params_from_config(config),
        voxel_y_um=float(config.acquisition.voxel_size_y_um),
        matches=matches,
        mode=mode,
        min_dominance_ratio=float(min_dominance_ratio),
        min_snr=float(min_snr),
    )
    summary_rows = summarize_rows(rows, sorted(section_set))

    _prepare_output_dir(out_dir)
    fields = _fieldnames(input_fieldnames)
    _write_rows(out_dir / AUDIT_CSV, rows, fields)
    for dominant, filename in CATEGORY_FILES.items():
        _write_rows(
            out_dir / filename,
            [row for row in rows if row.get("dominant_channel") == dominant],
            fields,
        )
    _write_summary_csv(out_dir / SUMMARY_CSV, summary_rows)

    if render_plots:
        _write_ratio_histograms(
            out_dir / RATIO_PNG, rows, float(min_dominance_ratio)
        )
        _write_snr_scatter(out_dir / SNR_PNG, rows, float(min_snr))
        qc_max_dim = int(
            getattr(getattr(config, "channel_overlay", None), "qc_max_dim", 2000)
        )
        _write_overlay(
            out_dir / OVERLAY_PNG, ordered_by_section, rows, max(64, qc_max_dim)
        )

    dominant_counts = Counter(row["dominant_channel"] for row in rows)
    decision_counts = Counter(row["channel_comparison_decision"] for row in rows)
    original_status_counts = Counter(row["original_status"] for row in rows)
    refined_status_counts = Counter(row["refined_candidate_status"] for row in rows)
    summary = {
        "analysis": "green/red PROVISIONAL candidate comparison",
        "provisional_candidates": True,
        "candidates_are_not_cells": True,
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "candidate_table": str(candidate_path),
        "out_dir": str(out_dir),
        "sections": sorted(section_set),
        "candidate_count": len(rows),
        "thresholds": {
            "min_dominance_ratio": float(min_dominance_ratio),
            "min_snr": float(min_snr),
            "max_match_distance_um": float(max_match_distance_um),
        },
        "measurement": {
            "xy_source": "x_global_px/y_global_px at the same source-TIFF pixels",
            "z_source": "recorded optical_plane (z_index fallback)",
            "snr_definition": "local robust contrast from fixed-XY central disk and annulus",
            "ratio_definition": "background-subtracted red / green",
            "raw_tiffs_modified": False,
        },
        "matching": {
            "dimensions": "XY only",
            "same_section_required": True,
            "opposite_detection_channel_required": True,
            "one_to_one": True,
        },
        "dominant_channel_counts": {
            dominant: dominant_counts.get(dominant, 0)
            for dominant in DOMINANT_CHANNELS
        },
        "channel_comparison_decision_counts": dict(decision_counts),
        "original_status_counts": dict(original_status_counts),
        "refined_status_counts": dict(refined_status_counts),
        "statuses_changed_in_source_run": False,
        "all_candidates_preserved_in_audit": len(rows) == len(candidates),
        "outputs": {
            "audit_csv": str(out_dir / AUDIT_CSV),
            "summary_csv": str(out_dir / SUMMARY_CSV),
            "green_dominant_csv": str(out_dir / CATEGORY_FILES[GREEN_DOMINANT]),
            "red_dominant_csv": str(out_dir / CATEGORY_FILES[RED_DOMINANT]),
            "both_csv": str(out_dir / CATEGORY_FILES[BOTH]),
            "unclear_csv": str(out_dir / CATEGORY_FILES[UNCLEAR]),
            "ratio_histograms_png": str(out_dir / RATIO_PNG) if render_plots else None,
            "snr_scatter_png": str(out_dir / SNR_PNG) if render_plots else None,
            "overlay_qc_png": str(out_dir / OVERLAY_PNG) if render_plots else None,
        },
    }
    if mode == MODE_REPORT:
        summary["note"] = "report mode: diagnostics only; candidate statuses unchanged"
    else:
        summary["note"] = (
            "apply mode: refined decisions use only the explicitly supplied CLI thresholds"
        )
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    return {
        "out_dir": out_dir,
        "candidate_count": len(rows),
        "rows": rows,
        "summary": summary,
        "summary_rows": summary_rows,
    }

