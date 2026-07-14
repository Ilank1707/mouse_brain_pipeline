"""Read-only green/red histogram analysis for PROVISIONAL candidates.

Signals are measured in both biological channels at each candidate's recorded
full-resolution XY coordinate and optical plane.  The source candidate table,
candidate statuses, and raw TIFFs are never modified.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from .channels import CHANNEL_2_SIGNAL, GREEN_SIGNAL
from .channel_comparison import (
    BOTH,
    DOMINANT_CHANNELS,
    GREEN_DOMINANT,
    RED_DOMINANT,
    UNCLEAR,
    _candidate_z,
    _load_section_stacks,
    _measure_at_location,
    _read_candidates,
    _section_overlay,
)

MEASUREMENTS_CSV = "channel_overlay_measurements.csv"
SUMMARY_CSV = "channel_overlay_summary.csv"
GREEN_SNR_PNG = "histogram_green_snr.png"
RED_SNR_PNG = "histogram_red_snr.png"
RED_GREEN_RATIO_PNG = "histogram_red_green_ratio.png"
GREEN_RED_RATIO_PNG = "histogram_green_red_ratio.png"
DOMINANT_COUNTS_PNG = "histogram_dominant_channel_counts.png"
BY_ORIGINAL_CHANNEL_PNG = "histogram_by_original_channel.png"
OVERLAY_QC_PNG = "overlay_filter_qc.png"

OUTPUT_FILENAMES = (
    MEASUREMENTS_CSV,
    SUMMARY_CSV,
    GREEN_SNR_PNG,
    RED_SNR_PNG,
    RED_GREEN_RATIO_PNG,
    GREEN_RED_RATIO_PNG,
    DOMINANT_COUNTS_PNG,
    BY_ORIGINAL_CHANNEL_PNG,
    OVERLAY_QC_PNG,
)

MEASUREMENT_FIELDS = [
    "original_channel",
    "original_status",
    "comparison_z_index",
    "comparison_optical_plane",
    "comparison_z_mapping",
    "green_peak",
    "red_peak",
    "green_local_background",
    "red_local_background",
    "green_snr",
    "red_snr",
    "red_green_ratio",
    "green_red_ratio",
    "green_measurement_valid",
    "red_measurement_valid",
    "green_measurement_reason",
    "red_measurement_reason",
    "dominant_channel",
]

SUMMARY_FIELDS = [
    "original_channel",
    "original_status",
    "dominant_channel",
    "candidate_count",
    "percent_of_group",
    "suggested_action",
]

_DOMINANT_COLOURS = {
    GREEN_DOMINANT: "#19A44B",
    RED_DOMINANT: "#D62F2F",
    BOTH: "#E0B400",
    UNCLEAR: "#8A8A8A",
}


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def safe_snr_ratio(numerator, denominator):
    """Return a safe ratio of non-negative SNR support values.

    Negative SNR has no positive signal support and is treated as zero.  A
    positive numerator over zero is infinity; zero over zero is neutral (1.0).
    Near-zero positive denominators remain finite and do not raise.
    """
    if not (_finite(numerator) and _finite(denominator)):
        return math.nan
    numerator = max(float(numerator), 0.0)
    denominator = max(float(denominator), 0.0)
    if denominator == 0.0:
        return math.inf if numerator > 0.0 else 1.0
    return numerator / denominator


def classify_dominant_channel(
    green_snr,
    red_snr,
    *,
    ratio_threshold: float = 2.0,
    snr_threshold: float = 3.0,
) -> str:
    """Assign green/red/both/unclear using measured SNR only."""
    if not (_finite(green_snr) and _finite(red_snr)):
        return UNCLEAR
    green_snr = float(green_snr)
    red_snr = float(red_snr)
    green_strong = green_snr >= float(snr_threshold)
    red_strong = red_snr >= float(snr_threshold)
    green_red = safe_snr_ratio(green_snr, red_snr)
    red_green = safe_snr_ratio(red_snr, green_snr)

    if green_strong and green_red >= float(ratio_threshold):
        return GREEN_DOMINANT
    if red_strong and red_green >= float(ratio_threshold):
        return RED_DOMINANT
    if green_strong and red_strong:
        return BOTH
    return UNCLEAR


def suggested_action(original_channel: str, original_status: str,
                     dominant_channel: str) -> str:
    """Map one measured class to an analysis suggestion, never a status edit."""
    if dominant_channel == BOTH:
        return "possible_duplicate_or_both"
    if dominant_channel == GREEN_DOMINANT:
        return (
            "keep_green_candidate"
            if original_channel == GREEN_SIGNAL
            else "manual_review"
        )
    if dominant_channel == RED_DOMINANT:
        return (
            "keep_red_candidate"
            if original_channel == CHANNEL_2_SIGNAL
            else "manual_review"
        )
    if str(original_status) == "manual_review":
        return "manual_review"
    return "likely_filter_unclear"


def build_suggested_filter_table(rows: Iterable[dict]) -> list[dict]:
    """Group by original channel/status and include all four measured classes."""
    rows = list(rows)
    groups = sorted(
        {
            (str(row.get("original_channel", "")), str(row.get("original_status", "")))
            for row in rows
        }
    )
    output = []
    for original_channel, original_status in groups:
        group_rows = [
            row
            for row in rows
            if row.get("original_channel") == original_channel
            and row.get("original_status") == original_status
        ]
        total = len(group_rows)
        counts = Counter(row.get("dominant_channel") for row in group_rows)
        for dominant_channel in DOMINANT_CHANNELS:
            count = counts.get(dominant_channel, 0)
            output.append(
                {
                    "original_channel": original_channel,
                    "original_status": original_status,
                    "dominant_channel": dominant_channel,
                    "candidate_count": count,
                    "percent_of_group": round(100.0 * count / total, 2) if total else 0.0,
                    "suggested_action": suggested_action(
                        original_channel, original_status, dominant_channel
                    ),
                }
            )
    return output


def _ensure_thresholds(ratio_threshold, snr_threshold) -> None:
    if not _finite(ratio_threshold) or float(ratio_threshold) < 1.0:
        raise ValueError("ratio_threshold must be finite and >= 1.0")
    if not _finite(snr_threshold) or float(snr_threshold) < 0.0:
        raise ValueError("snr_threshold must be finite and >= 0.0")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_location(out_dir: Path, run_dir: Path, config) -> None:
    protected = [
        run_dir,
        Path(config.data.green_signal_dir).expanduser().resolve(),
        Path(config.data.channel_2_signal_dir).expanduser().resolve(),
    ]
    for root in protected:
        if out_dir == root or _is_within(out_dir, root):
            raise ValueError(
                f"--out-dir must be separate from source run/raw data: {root}"
            )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing overlay-histogram outputs: {out_dir}"
        )


def _measure_rows(candidates, stacks, plane_numbers, params, voxel_y_um,
                  ratio_threshold, snr_threshold):
    import numpy as np  # noqa: PLC0415

    height, width = stacks[GREEN_SIGNAL].shape[1:]
    tissue_plane = np.broadcast_to(np.bool_(True), (height, width))
    output = []
    for candidate in candidates:
        z_index, optical_plane, z_mapping = _candidate_z(candidate, plane_numbers)
        green = _measure_at_location(
            stacks[GREEN_SIGNAL],
            candidate.get("x_global_px"),
            candidate.get("y_global_px"),
            z_index,
            tissue_plane,
            params,
            voxel_y_um,
        )
        red = _measure_at_location(
            stacks[CHANNEL_2_SIGNAL],
            candidate.get("x_global_px"),
            candidate.get("y_global_px"),
            z_index,
            tissue_plane,
            params,
            voxel_y_um,
        )
        red_green = safe_snr_ratio(red["snr"], green["snr"])
        green_red = safe_snr_ratio(green["snr"], red["snr"])
        dominant = classify_dominant_channel(
            green["snr"],
            red["snr"],
            ratio_threshold=ratio_threshold,
            snr_threshold=snr_threshold,
        )
        row = dict(candidate)
        row.update(
            {
                "original_channel": candidate.get("channel", ""),
                "original_status": candidate.get("current_status", ""),
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
                "red_green_ratio": red_green,
                "green_red_ratio": green_red,
                "green_measurement_valid": green["valid"],
                "red_measurement_valid": red["valid"],
                "green_measurement_reason": green["reason"],
                "red_measurement_reason": red["reason"],
                "dominant_channel": dominant,
            }
        )
        output.append(row)
    return output


def _fieldnames(input_fields: Sequence[str]) -> list[str]:
    fields = list(input_fields)
    for field in MEASUREMENT_FIELDS:
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


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _finite_values(rows, field):
    return [float(row[field]) for row in rows if _finite(row.get(field))]


def _finite_log2_ratios(rows, field):
    values = []
    omitted = 0
    for row in rows:
        ratio = row.get(field)
        if _finite(ratio) and float(ratio) > 0.0:
            values.append(math.log2(float(ratio)))
        else:
            omitted += 1
    return values, omitted


def _plot_single_histogram(path: Path, values, *, colour, xlabel, title,
                           threshold=None, omitted=0):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axis = plt.subplots(figsize=(8, 5))
    try:
        if values:
            axis.hist(values, bins="auto", color=colour, alpha=0.85)
        if threshold is not None:
            axis.axvline(float(threshold), color="black", linestyle="--", linewidth=1)
        if omitted:
            axis.text(
                0.98, 0.96,
                f"non-finite/zero ratios omitted: {omitted}",
                transform=axis.transAxes, ha="right", va="top", fontsize=8,
            )
        axis.set(xlabel=xlabel, ylabel="PROVISIONAL candidate count", title=title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_dominant_counts(path: Path, rows: Sequence[dict]):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    counts = Counter(row.get("dominant_channel") for row in rows)
    values = [counts.get(category, 0) for category in DOMINANT_CHANNELS]
    fig, axis = plt.subplots(figsize=(9, 5))
    try:
        bars = axis.bar(
            DOMINANT_CHANNELS,
            values,
            color=[_DOMINANT_COLOURS[category] for category in DOMINANT_CHANNELS],
        )
        axis.bar_label(bars)
        axis.set(
            ylabel="PROVISIONAL candidate count",
            title="PROVISIONAL candidates - measured dominant-channel counts",
        )
        axis.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_split_histograms(path: Path, rows: Sequence[dict], ratio_threshold,
                           snr_threshold):
    """Two channel rows; within each row, distributions are split by status."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    channels = (GREEN_SIGNAL, CHANNEL_2_SIGNAL)
    statuses = sorted({str(row.get("original_status", "")) for row in rows})
    cmap = plt.get_cmap("tab10")
    status_colours = {status: cmap(index % 10) for index, status in enumerate(statuses)}
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    try:
        for row_index, channel in enumerate(channels):
            channel_rows = [row for row in rows if row.get("original_channel") == channel]
            for status in statuses:
                subset = [
                    row for row in channel_rows if row.get("original_status") == status
                ]
                colour = status_colours[status]
                for column, field in enumerate(("green_snr", "red_snr")):
                    values = _finite_values(subset, field)
                    if values:
                        axes[row_index, column].hist(
                            values, bins="auto", histtype="step", linewidth=1.5,
                            color=colour, label=status,
                        )
                for column, field in (
                    (2, "red_green_ratio"), (3, "green_red_ratio")
                ):
                    values, _omitted = _finite_log2_ratios(subset, field)
                    if values:
                        axes[row_index, column].hist(
                            values, bins="auto", histtype="step", linewidth=1.5,
                            color=colour, label=status,
                        )

            count_axis = axes[row_index, 4]
            bottom = np.zeros(len(DOMINANT_CHANNELS), dtype=float)
            for status in statuses:
                subset = [
                    row for row in channel_rows if row.get("original_status") == status
                ]
                counts = Counter(row.get("dominant_channel") for row in subset)
                values = np.asarray(
                    [counts.get(category, 0) for category in DOMINANT_CHANNELS],
                    dtype=float,
                )
                count_axis.bar(
                    DOMINANT_CHANNELS, values, bottom=bottom,
                    color=status_colours[status], label=status,
                )
                bottom += values

            axes[row_index, 0].axvline(
                float(snr_threshold), color="black", linestyle="--", linewidth=1
            )
            axes[row_index, 1].axvline(
                float(snr_threshold), color="black", linestyle="--", linewidth=1
            )
            axes[row_index, 2].axvline(
                math.log2(float(ratio_threshold)),
                color="black", linestyle="--", linewidth=1,
            )
            axes[row_index, 3].axvline(
                math.log2(float(ratio_threshold)),
                color="black", linestyle="--", linewidth=1,
            )
            titles = (
                "green SNR", "red SNR", "log2(red/green SNR)",
                "log2(green/red SNR)", "dominant-channel counts",
            )
            for column, title in enumerate(titles):
                axes[row_index, column].set_title(f"{channel}: {title}")
                axes[row_index, column].set_ylabel("PROVISIONAL candidates")
            count_axis.tick_params(axis="x", rotation=20)

        handles, labels = axes[0, 4].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, title="original status", loc="upper center",
                       ncol=min(5, len(labels)))
        fig.suptitle(
            "PROVISIONAL candidate histograms split by original channel and status",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _write_plots(out_dir: Path, rows: Sequence[dict], ordered_by_channel,
                 section: int, ratio_threshold: float, snr_threshold: float,
                 qc_max_dim: int) -> None:
    _plot_single_histogram(
        out_dir / GREEN_SNR_PNG,
        _finite_values(rows, "green_snr"),
        colour="#19A44B",
        xlabel="green SNR (local robust contrast)",
        title="PROVISIONAL candidates - green SNR distribution",
        threshold=snr_threshold,
    )
    _plot_single_histogram(
        out_dir / RED_SNR_PNG,
        _finite_values(rows, "red_snr"),
        colour="#D62F2F",
        xlabel="red SNR (local robust contrast)",
        title="PROVISIONAL candidates - red SNR distribution",
        threshold=snr_threshold,
    )
    red_green, red_green_omitted = _finite_log2_ratios(rows, "red_green_ratio")
    _plot_single_histogram(
        out_dir / RED_GREEN_RATIO_PNG,
        red_green,
        colour="#B3261E",
        xlabel="log2(red SNR / green SNR)",
        title="PROVISIONAL candidates - red/green SNR ratio",
        threshold=math.log2(ratio_threshold),
        omitted=red_green_omitted,
    )
    green_red, green_red_omitted = _finite_log2_ratios(rows, "green_red_ratio")
    _plot_single_histogram(
        out_dir / GREEN_RED_RATIO_PNG,
        green_red,
        colour="#137333",
        xlabel="log2(green SNR / red SNR)",
        title="PROVISIONAL candidates - green/red SNR ratio",
        threshold=math.log2(ratio_threshold),
        omitted=green_red_omitted,
    )
    _plot_dominant_counts(out_dir / DOMINANT_COUNTS_PNG, rows)
    _plot_split_histograms(
        out_dir / BY_ORIGINAL_CHANNEL_PNG,
        rows,
        ratio_threshold,
        snr_threshold,
    )
    overlay = _section_overlay(
        ordered_by_channel,
        rows,
        section,
        max(64, int(qc_max_dim)),
    )
    overlay.save(out_dir / OVERLAY_QC_PNG, format="PNG")


def run_channel_overlay_histograms(
    *,
    config,
    run_dir,
    section: int,
    out_dir,
    green_channel: str = GREEN_SIGNAL,
    red_channel: str = CHANNEL_2_SIGNAL,
    ratio_threshold: float = 2.0,
    snr_threshold: float = 3.0,
) -> dict:
    """Run the read-only analysis and write all requested outputs."""
    from .audit import index_channel  # noqa: PLC0415
    from .candidate_detection import params_from_config  # noqa: PLC0415

    if green_channel != GREEN_SIGNAL or red_channel != CHANNEL_2_SIGNAL:
        raise ValueError(
            "green_channel must be green_signal and red_channel must be channel_2_signal"
        )
    _ensure_thresholds(ratio_threshold, snr_threshold)
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    _validate_output_location(out_dir, run_dir, config)

    candidates, input_fields = _read_candidates(
        run_dir / "all_candidates.csv", {int(section)}
    )
    indexes = {
        GREEN_SIGNAL: index_channel(
            GREEN_SIGNAL, config.data.green_signal_dir, config.data.filename_regex
        ),
        CHANNEL_2_SIGNAL: index_channel(
            CHANNEL_2_SIGNAL,
            config.data.channel_2_signal_dir,
            config.data.filename_regex,
        ),
    }
    stacks, ordered, plane_numbers, _shape = _load_section_stacks(
        indexes, int(section), int(config.acquisition.planes_per_section)
    )
    rows = _measure_rows(
        candidates,
        stacks,
        plane_numbers,
        params_from_config(config),
        float(config.acquisition.voxel_size_y_um),
        float(ratio_threshold),
        float(snr_threshold),
    )
    summary_rows = build_suggested_filter_table(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / MEASUREMENTS_CSV, rows, _fieldnames(input_fields))
    _write_csv(out_dir / SUMMARY_CSV, summary_rows, SUMMARY_FIELDS)
    qc_max_dim = int(
        getattr(getattr(config, "channel_overlay", None), "qc_max_dim", 2000)
    )
    _write_plots(
        out_dir,
        rows,
        ordered,
        int(section),
        float(ratio_threshold),
        float(snr_threshold),
        qc_max_dim,
    )
    return {
        "out_dir": out_dir,
        "candidate_count": len(rows),
        "rows": rows,
        "summary_rows": summary_rows,
        "outputs": {name: out_dir / name for name in OUTPUT_FILENAMES},
    }

