#!/usr/bin/env python
"""Automatic post-run candidate-to-candidate pair-correlation reports.

The normal ``run_candidate_pilot.py`` command calls this module for both signal
channels, every processed section, and the four configured candidate groups. It
is deliberately read-only with respect to detection results: all writes go below
``<run-dir>/spatial_analysis/pair_correlation``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _bootstrap  # noqa: F401,E402

import pair_correlation_analysis as pca  # noqa: E402


CHANNELS = ("green_signal", "channel_2_signal")
POSTRUN_STATUSES = (
    "preliminary_pass",
    "preliminary_fail",
    "all_outside_injection",
    "manual_review",
)

# status -> (stable series index, selector, restrict to outside-injection window)
_SERIES_BY_STATUS = {
    status: (index, selector, outside_only)
    for index, (status, selector, outside_only) in enumerate(pca.SERIES)
}

PAIR_CORRELATION_COLUMNS = [
    "radius_start_um",
    "radius_end_um",
    "radius_mid_um",
    "g_r",
    "g_r_lower_95",
    "g_r_upper_95",
    "number_of_candidates",
    "status",
    "channel",
]

PAIR_DENSITY_COLUMNS = [
    "radius_start_um",
    "radius_end_um",
    "radius_mid_um",
    "observed_pair_count",
    "observed_pair_density_per_mm2",
    "csr_mean_pair_density_per_mm2",
    "csr_lower_95",
    "csr_upper_95",
    "number_of_candidates",
    "status",
    "channel",
]

MANIFEST_COLUMNS = [
    "section",
    "channel",
    "status",
    "outcome",
    "number_of_candidates",
    "reason",
    "error",
    "directory",
]

REPORT_FILENAMES = (
    "pair_correlation_g_r.png",
    "pair_density_per_mm2.png",
    "pair_correlation_values.csv",
    "pair_density_values.csv",
    "metadata.json",
)


def _write_values_csv(path, result, columns, *, status, channel) -> None:
    scalars = {
        "number_of_candidates": result["number_of_candidates"],
        "status": status,
        "channel": channel,
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(len(result["radius_mid_um"])):
            row = {
                column: (
                    scalars[column]
                    if column in scalars
                    else result[column][index]
                )
                for column in columns
            }
            writer.writerow({key: pca._csv_value(value) for key, value in row.items()})


def _channel_masks(run_dir: Path, channel: str, section: int):
    """Return the tissue and outside-injection windows, or a skip reason."""
    section_dir = run_dir / "qc" / f"{channel}_section_{section:03d}"
    tissue_path = section_dir / "tissue_mask.npy"
    exclusion_path = section_dir / "injection_analysis_exclusion_mask.npy"
    if not tissue_path.is_file():
        return None, f"missing saved tissue mask ({tissue_path})"
    if not exclusion_path.is_file():
        return None, f"missing saved injection-exclusion mask ({exclusion_path})"
    tissue_window = pca.MaskWindow.from_npy(tissue_path)
    outside_window = pca.MaskWindow.from_npy(tissue_path, exclusion_path)
    return (tissue_window, outside_window, tissue_path, exclusion_path), None


def _report_entry(
    *,
    section: int,
    channel: str,
    status: str,
    outcome: str,
    number_of_candidates: int = 0,
    reason: str = "",
    error: str = "",
    directory: str = "",
) -> dict:
    return {
        "section": int(section),
        "channel": channel,
        "status": status,
        "outcome": outcome,
        "number_of_candidates": int(number_of_candidates),
        "reason": reason,
        "error": error,
        "directory": directory,
    }


def _record(summary: dict, entry: dict) -> None:
    summary[entry["outcome"]].append(entry)


def _record_all(
    summary: dict,
    sections: Iterable[int],
    statuses: Iterable[str],
    *,
    outcome: str,
    reason: str = "",
    error: str = "",
) -> None:
    for section in sections:
        for channel in CHANNELS:
            for status in statuses:
                _record(
                    summary,
                    _report_entry(
                        section=section,
                        channel=channel,
                        status=status,
                        outcome=outcome,
                        reason=reason,
                        error=error,
                    ),
                )


def _normalise_sections(
    *, section: int | None, sections: Iterable[int] | int | None
) -> list[int]:
    if sections is None:
        values = [] if section is None else [section]
    elif isinstance(sections, int):
        values = [sections]
    else:
        values = list(sections)
    # Preserve processing order while preventing duplicate section folders/reports.
    return list(dict.fromkeys(int(value) for value in values))


def _write_run_outputs(analysis_root: Path, summary: dict) -> None:
    analysis_root.mkdir(parents=True, exist_ok=True)
    entries = summary["completed"] + summary["skipped"] + summary["failed"]
    manifest_path = analysis_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)

    summary["counts"] = {
        "completed": len(summary["completed"]),
        "skipped": len(summary["skipped"]),
        "failed": len(summary["failed"]),
    }
    summary["manifest"] = str(manifest_path)
    summary_path = analysis_root / "summary.json"
    summary["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def generate_postrun_pair_correlation(
    *,
    config,
    run_dir,
    section: int | None = None,
    sections: Iterable[int] | int | None = None,
    statuses=POSTRUN_STATUSES,
    maximum_distance_um=500.0,
    simulations=99,
    random_seed=20260713,
    bin_width_um=5.0,
    minimum_candidates=2,
    include_cropped_runs=False,
) -> dict:
    """Generate pair-correlation reports for all requested sections.

    A completed report always contains exactly the five files named in
    :data:`REPORT_FILENAMES`. Low-count groups are recorded as skipped without
    creating a status folder. Every channel/status report has its own exception
    boundary so one plotting failure cannot stop the remaining reports.
    """
    import numpy as np  # noqa: PLC0415

    run_dir = Path(run_dir).resolve()
    analysis_root = run_dir / "spatial_analysis" / "pair_correlation"
    section_numbers = _normalise_sections(section=section, sections=sections)
    statuses = tuple(statuses)

    if maximum_distance_um <= 0:
        raise ValueError("maximum_distance_um must be greater than zero")
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if bin_width_um <= 0:
        raise ValueError("bin_width_um must be greater than zero")
    if minimum_candidates < 2:
        raise ValueError("minimum_candidates must be at least 2")

    summary = {
        "analysis": "2D candidate-to-candidate pair correlation g(r)",
        "provisional_candidates_not_cell_counts": True,
        "read_only_detection_inputs": True,
        "run_dir": str(run_dir),
        "output_root": str(analysis_root),
        "sections": section_numbers,
        "channels": list(CHANNELS),
        "statuses": list(statuses),
        "maximum_distance_um": float(maximum_distance_um),
        "simulations": int(simulations),
        "random_seed": int(random_seed),
        "bin_width_um": float(bin_width_um),
        "minimum_candidates": int(minimum_candidates),
        "crop_runs_included": bool(include_cropped_runs),
        "completed": [],
        "skipped": [],
        "failed": [],
    }

    try:
        metadata = pca._run_metadata(run_dir)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _record_all(
            summary,
            section_numbers,
            statuses,
            outcome="skipped",
            reason=f"cannot read candidate_run_metadata.json: {exc}",
        )
        _write_run_outputs(analysis_root, summary)
        return summary

    crop = metadata.get("crop_x_min_x_max_y_min_y_max")
    if crop is not None and not include_cropped_runs:
        _record_all(
            summary,
            section_numbers,
            statuses,
            outcome="skipped",
            reason=(
                "cropped run: crop boundaries bias clustering; set "
                "postrun_spatial_analysis.pair_correlation.include_cropped_runs "
                "to true only for a deliberate within-crop analysis"
            ),
        )
        _write_run_outputs(analysis_root, summary)
        return summary

    try:
        all_rows, candidate_columns = pca._read_candidates(
            run_dir / "all_candidates.csv"
        )
    except (FileNotFoundError, ValueError) as exc:
        _record_all(
            summary,
            section_numbers,
            statuses,
            outcome="skipped",
            reason=f"cannot read all_candidates.csv: {exc}",
        )
        _write_run_outputs(analysis_root, summary)
        return summary

    summary["candidate_columns_inspected"] = candidate_columns
    crop_origin_yx = pca._crop_origin_yx(metadata)
    voxel_yx_um = (
        float(config.acquisition.voxel_size_y_um),
        float(config.acquisition.voxel_size_x_um),
    )
    edges_um = pca.distance_edges(bin_width_um, maximum_distance_um)

    for section_number in section_numbers:
        section_root = analysis_root / f"section_{section_number:03d}"
        for channel in CHANNELS:
            section_rows = [
                row
                for row in all_rows
                if row.get("channel") == channel
                and str(row.get("section")) == str(section_number)
            ]
            if not section_rows:
                for status in statuses:
                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="skipped",
                            reason="no candidates for channel/section",
                        ),
                    )
                continue

            candidate_ids = [
                str(row.get("candidate_id", "")).strip() for row in section_rows
            ]
            id_error = ""
            if any(not candidate_id for candidate_id in candidate_ids):
                id_error = "every candidate row must have a non-empty candidate_id"
            elif len(set(candidate_ids)) != len(candidate_ids):
                id_error = "duplicate candidate_id rows would double-count candidates"
            if id_error:
                for status in statuses:
                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="failed",
                            error=id_error,
                        ),
                    )
                continue

            try:
                windows, reason = _channel_masks(run_dir, channel, section_number)
            except Exception as exc:  # noqa: BLE001 - corrupt mask is channel-scoped
                windows, reason = None, ""
                mask_error = f"{type(exc).__name__}: {exc}"
                for status in statuses:
                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="failed",
                            error=mask_error,
                        ),
                    )
                continue
            if windows is None:
                for status in statuses:
                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="skipped",
                            reason=reason,
                        ),
                    )
                continue

            tissue_window, outside_window, tissue_path, exclusion_path = windows
            expected = metadata.get("source_image_dimensions", {}).get(channel, {})
            if crop is None and expected.get("height") is not None and expected.get("width") is not None:
                expected_shape = (int(expected["height"]), int(expected["width"]))
                if tissue_window.tissue.shape != expected_shape:
                    error = (
                        "full-section tissue mask shape does not match run metadata: "
                        f"{tissue_window.tissue.shape} vs {expected_shape}"
                    )
                    for status in statuses:
                        _record(
                            summary,
                            _report_entry(
                                section=section_number,
                                channel=channel,
                                status=status,
                                outcome="failed",
                                error=error,
                            ),
                        )
                    continue

            for status in statuses:
                spec = _SERIES_BY_STATUS.get(status)
                if spec is None:
                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="skipped",
                            reason="unknown status",
                        ),
                    )
                    continue

                series_index, selector, outside_only = spec
                window = outside_window if outside_only else tissue_window
                number_of_candidates = 0
                try:
                    selected = [row for row in section_rows if selector(row)]
                    _kept, points_um, dropped = pca._points_for_rows(
                        selected, window, crop_origin_yx, voxel_yx_um
                    )
                    number_of_candidates = len(points_um)
                    if number_of_candidates < minimum_candidates:
                        _record(
                            summary,
                            _report_entry(
                                section=section_number,
                                channel=channel,
                                status=status,
                                outcome="skipped",
                                number_of_candidates=number_of_candidates,
                                reason=(
                                    f"fewer than {minimum_candidates} candidates in "
                                    "the valid sampling window"
                                ),
                            ),
                        )
                        continue

                    series_seed = int(
                        np.random.SeedSequence([random_seed, series_index])
                        .generate_state(1)[0]
                    )
                    result = pca.analyze_pair_correlation(
                        points_um,
                        window,
                        edges_um,
                        simulations=simulations,
                        random_seed=series_seed,
                        voxel_yx_um=voxel_yx_um,
                    )

                    status_dir = section_root / channel / status
                    status_dir.mkdir(parents=True, exist_ok=True)
                    graph_path = status_dir / "pair_correlation_g_r.png"
                    density_path = status_dir / "pair_density_per_mm2.png"
                    values_path = status_dir / "pair_correlation_values.csv"
                    density_values_path = status_dir / "pair_density_values.csv"
                    metadata_path = status_dir / "metadata.json"

                    pca._plot_g_r(
                        graph_path,
                        result,
                        channel=channel,
                        section=section_number,
                        status=status,
                    )
                    pca._plot_pair_density(
                        density_path,
                        result,
                        channel=channel,
                        section=section_number,
                        status=status,
                    )
                    _write_values_csv(
                        values_path,
                        result,
                        PAIR_CORRELATION_COLUMNS,
                        status=status,
                        channel=channel,
                    )
                    _write_values_csv(
                        density_values_path,
                        result,
                        PAIR_DENSITY_COLUMNS,
                        status=status,
                        channel=channel,
                    )

                    report_metadata = {
                        "analysis": "2D candidate-to-candidate pair correlation g(r)",
                        "provisional_candidates_not_cell_counts": True,
                        "channel": channel,
                        "section": section_number,
                        "status": status,
                        "number_of_candidates": number_of_candidates,
                        "dropped_outside_window_or_invalid": dropped,
                        "maximum_distance_um": float(maximum_distance_um),
                        "bin_width_um": float(bin_width_um),
                        "simulations": int(simulations),
                        "random_seed": int(random_seed),
                        "series_random_seed": series_seed,
                        "csr_reference_line": 1.0,
                        "envelope": "95% pointwise CSR simulation envelope",
                        "coordinates": "XY separation in µm; one row per candidate",
                        "sampling_window": (
                            "tissue_mask_minus_channel_injection_exclusion"
                            if outside_only
                            else "tissue_mask"
                        ),
                        "voxel_size_yx_um": list(voxel_yx_um),
                        "tissue_mask": str(tissue_path),
                        "injection_analysis_exclusion_mask": (
                            str(exclusion_path) if outside_only else None
                        ),
                        "outputs": {
                            "pair_correlation_g_r.png": str(graph_path),
                            "pair_density_per_mm2.png": str(density_path),
                            "pair_correlation_values.csv": str(values_path),
                            "pair_density_values.csv": str(density_values_path),
                            "metadata.json": str(metadata_path),
                        },
                    }
                    metadata_path.write_text(
                        json.dumps(report_metadata, indent=2), encoding="utf-8"
                    )

                    missing = [
                        filename
                        for filename in REPORT_FILENAMES
                        if not (status_dir / filename).is_file()
                    ]
                    if missing:
                        raise RuntimeError(
                            "report did not create required outputs: " + ", ".join(missing)
                        )

                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="completed",
                            number_of_candidates=number_of_candidates,
                            directory=str(status_dir),
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - isolate every report
                    _record(
                        summary,
                        _report_entry(
                            section=section_number,
                            channel=channel,
                            status=status,
                            outcome="failed",
                            number_of_candidates=number_of_candidates,
                            error=f"{type(exc).__name__}: {exc}",
                            directory=str(section_root / channel / status),
                        ),
                    )

            duplicated = section_root / channel / channel
            if duplicated.exists():
                raise RuntimeError(f"duplicated channel folder detected: {duplicated}")

    _write_run_outputs(analysis_root, summary)
    return summary
