"""Run-scoped radial candidate analysis: CSVs + charts around the injection.

Thin I/O + plotting layer over the pure maths in :mod:`radial_analysis`. Reads a
run's ``all_candidates.csv`` and tissue mask, then writes the radial coordinate
table, the per-annulus count/density table, and four charts.

These are PROVISIONAL candidates. Preliminary-rule passes are never called cells;
only genuine confirmed cells get the confirmed-cell series.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .channels import channel_display_name
from .coordinate_exports import is_confirmed_cell
from .radial_analysis import (
    RADIAL_COORDINATE_COLUMNS,
    RADIAL_COUNT_COLUMNS,
    assemble_series,
    counts_by_bin,
    injection_core_centroid,
    per_candidate_rows,
    radial_distances_um,
    resolve_n_bins,
)
from .review import read_csv_rows
from .utilities import LOG, ensure_dir

# Provisional series (name -> current_status filter). "all_outside_injection" is
# the union of everything outside the injection analysis mask.
_STATUS_SERIES = [
    ("preliminary_pass", "preliminary_rule_pass"),
    ("preliminary_fail", "preliminary_rule_fail"),
    ("manual_review", "manual_review"),
    ("invalid_measurement", "invalid_measurement"),
]


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_mask(run_dir: Path, channel: str, section: int, name: str):
    import numpy as np  # noqa: PLC0415

    path = run_dir / "qc" / f"{channel}_section_{int(section):03d}" / name
    if not path.is_file():
        return None
    return np.load(path)


def load_or_build_tissue_mask(run_dir, config, channel, section, crop):
    """Tissue mask for one channel+section: saved npy first, else rebuild it.

    Rebuilding uses BOTH channels (the shared mask the detector used) so legacy
    runs without a saved mask still get the correct in-tissue area.
    """
    saved = _load_mask(Path(run_dir), channel, section, "tissue_mask.npy")
    if saved is not None:
        return saved

    LOG.warning("No saved tissue mask for %s section %d -- rebuilding from TIFFs.",
                channel, section)
    from mouse_brain_pipeline import CHANNEL_2_SIGNAL, GREEN_SIGNAL  # noqa: PLC0415
    from mouse_brain_pipeline.audit import index_channel  # noqa: PLC0415
    from mouse_brain_pipeline.candidate_detection import (  # noqa: PLC0415
        build_shared_tissue_mask,
        params_from_config,
        read_crop_stack,
    )

    regex = config.data.filename_regex
    stacks = []
    for ch, directory in ((GREEN_SIGNAL, config.data.green_signal_dir),
                          (CHANNEL_2_SIGNAL, config.data.channel_2_signal_dir)):
        index = index_channel(ch, directory, regex)
        plane_paths = {pl: path for (s, pl), path in index.files.items() if s == section}
        if plane_paths:
            stack, _pn, _origin, _shape = read_crop_stack(plane_paths, crop)
            stacks.append(stack)
    if not stacks:
        return None
    params = params_from_config(config)
    return build_shared_tissue_mask(stacks, config.acquisition.voxel_size_zyx, params.tissue)


def resolve_center(run_dir, config, channel, section, crop_origin, center_xy=None):
    """Return (center_x_global, center_y_global, source, warning-or-None)."""
    radial_cfg = config.radial_analysis
    if center_xy is not None:
        return float(center_xy[0]), float(center_xy[1]), "cli_manual", None
    if radial_cfg.center_source == "manual" and radial_cfg.manual_center_xy_px:
        cx, cy = radial_cfg.manual_center_xy_px
        return float(cx), float(cy), "config_manual", None

    core = _load_mask(Path(run_dir), channel, section, "injection_core_mask.npy")
    centroid = injection_core_centroid(core, crop_origin) if core is not None else None
    if centroid is None:
        return None, None, "none", "no injection core mask and no manual centre configured"
    # Warn unless the core is a validated / manually-defined mask.
    meta = _run_metadata(run_dir)
    inj = (meta.get("injection_exclusion_by_channel") or {}).get(channel, {})
    validated = bool(inj.get("mask_validated"))
    manual = bool(inj.get("manual_polygons") or inj.get("manual_rectangles"))
    warning = None
    if not (validated or manual):
        warning = ("radial centre is the centroid of an UNVALIDATED automatic "
                   "injection core -- confirm the mask or set a manual centre.")
    return centroid["x_global_px"], centroid["y_global_px"], "injection_core_centroid", warning


def _run_metadata(run_dir) -> dict:
    path = Path(run_dir) / "candidate_run_metadata.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def _plot(series_rows, out_path, *, y_key, title, ylabel, subtitle=None, cumulative=False):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rows in series_rows:
        xs = [(r["radial_bin_start_um"] + r["radial_bin_end_um"]) / 2.0 for r in rows]
        ys = [r[y_key] if r[y_key] != "" else float("nan") for r in rows]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, label=name)
    ax.set_xlabel("radial distance from injection centre (um)")
    ax.set_ylabel(ylabel)
    full_title = title
    if subtitle:
        full_title += f"\n{subtitle}"
    ax.set_title(full_title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.6)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def analyze_run(run_dir, config, *, channel=None, section=None, center_xy=None,
                bin_width_um=None, maximum_radius_um=None, out_dir=None):
    """Compute + write the radial analysis for one run. Returns a paths dict."""
    import numpy as np  # noqa: PLC0415

    run_dir = Path(run_dir)
    radial_cfg = config.radial_analysis
    channel = channel or radial_cfg.channel
    bin_width_um = float(bin_width_um if bin_width_um is not None else radial_cfg.bin_width_um)
    if maximum_radius_um is None:
        maximum_radius_um = radial_cfg.maximum_radius_um
    out_dir = ensure_dir(Path(out_dir) if out_dir else run_dir / "radial_analysis")

    rows = read_csv_rows(run_dir / "all_candidates.csv")
    rows = [r for r in rows if r.get("channel") == channel]
    if section is not None:
        rows = [r for r in rows if str(r.get("section")) == str(section)]
    if not rows:
        raise ValueError(f"No candidates for channel {channel!r}"
                         + (f" section {section}" if section is not None else ""))
    section_used = int(section) if section is not None else int(rows[0].get("section") or 0)

    meta = _run_metadata(run_dir)
    crop = meta.get("crop_x_min_x_max_y_min_y_max")
    crop_origin = (int(crop[2]), int(crop[0])) if crop else (0, 0)
    vy = float(config.acquisition.voxel_size_y_um)
    vx = float(config.acquisition.voxel_size_x_um)

    cx, cy, center_source, warning = resolve_center(
        run_dir, config, channel, section_used, crop_origin, center_xy=center_xy)
    if cx is None:
        raise ValueError("Could not resolve a radial centre: " + (warning or "unknown"))
    if warning:
        LOG.warning(warning)
    center_local = (cx - crop_origin[1], cy - crop_origin[0])

    tissue = load_or_build_tissue_mask(run_dir, config, channel, section_used, crop)
    if tissue is None:
        raise ValueError("No tissue mask available (saved or rebuilt) for radial area.")
    tissue = np.asarray(tissue, dtype=bool)

    # Candidate distances (global frame) + tissue pixel distances (local frame).
    outside = [r for r in rows if not _truthy(r.get("inside_injection_analysis_exclusion"))]
    cand_dist = radial_distances_um(
        [float(r.get("x_global_px") or 0) for r in outside],
        [float(r.get("y_global_px") or 0) for r in outside],
        (cx, cy), (vy, vx))
    ys, xs = np.nonzero(tissue)
    tissue_dist = radial_distances_um(xs, ys, center_local, (vy, vx))

    max_dist = float(max(cand_dist.max() if cand_dist.size else 0.0,
                         tissue_dist.max() if tissue_dist.size else 0.0))
    n_bins = resolve_n_bins(bin_width_um, maximum_radius_um, max_dist)
    tissue_area_px = counts_by_bin(tissue_dist, bin_width_um, n_bins)
    voxel_area_um2 = vy * vx

    # Build each provisional series (all outside-injection, then by status).
    series_defs = [("all_outside_injection", outside)]
    for name, status in _STATUS_SERIES:
        series_defs.append((name, [r for r in outside if r.get("current_status") == status]))
    confirmed = [r for r in rows if is_confirmed_cell(r)]
    if confirmed:
        series_defs.append(("confirmed_cells", confirmed))

    count_rows = []
    series_for_plot = []
    for name, subset in series_defs:
        dist = radial_distances_um(
            [float(r.get("x_global_px") or 0) for r in subset],
            [float(r.get("y_global_px") or 0) for r in subset],
            (cx, cy), (vy, vx))
        counts = counts_by_bin(dist, bin_width_um, n_bins)
        assembled = assemble_series(counts, tissue_area_px, bin_width_um, voxel_area_um2)
        for r in assembled:
            count_rows.append({"series": name, **r})
        series_for_plot.append((name, assembled))

    # ---- write CSVs ----
    coord_path = out_dir / "candidate_radial_coordinates.csv"
    coord_rows = per_candidate_rows(outside, (cx, cy), (vy, vx), bin_width_um, n_bins)
    with open(coord_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RADIAL_COORDINATE_COLUMNS)
        writer.writeheader()
        writer.writerows(coord_rows)

    counts_path = out_dir / "radial_counts_by_status.csv"
    with open(counts_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RADIAL_COUNT_COLUMNS)
        writer.writeheader()
        writer.writerows(count_rows)

    # ---- charts (all provisional candidates) ----
    provisional = [s for s in series_for_plot if s[0] != "confirmed_cells"] or series_for_plot
    density_plot = [s for s in series_for_plot]  # density chart keeps every series
    tag = (f"{channel_display_name(channel)} ({channel}) section {section_used:03d} "
           f"-- centre ({cx:.0f}, {cy:.0f}) [{center_source}]")
    density_path = _plot(
        density_plot, out_dir / "radial_density_vs_distance.png",
        y_key="density_per_mm2",
        title=f"Candidate density vs radial distance -- PROVISIONAL candidates\n{tag}",
        ylabel="candidates per mm2 of tissue")
    count_path = _plot(
        provisional, out_dir / "radial_count_vs_distance.png",
        y_key="count",
        title=f"Candidate count vs radial distance -- PROVISIONAL candidates\n{tag}",
        ylabel="raw candidate count",
        subtitle="raw counts -- affected by annulus area; see density chart")
    fraction_path = _plot(
        provisional, out_dir / "radial_fraction_vs_distance.png",
        y_key="fraction",
        title=f"Candidate fraction per annulus -- PROVISIONAL candidates\n{tag}",
        ylabel="fraction of series candidates")
    cumulative_path = _plot(
        provisional, out_dir / "radial_cumulative_fraction.png",
        y_key="cumulative_fraction",
        title=f"Cumulative candidate fraction -- PROVISIONAL candidates\n{tag}",
        ylabel="cumulative fraction")

    summary = {
        "run_dir": str(run_dir),
        "channel": channel,
        "section": section_used,
        "center_xy_global_px": [cx, cy],
        "center_source": center_source,
        "center_warning": warning,
        "bin_width_um": bin_width_um,
        "n_bins": n_bins,
        "maximum_radius_um": maximum_radius_um,
        "voxel_size_yx_um": [vy, vx],
        "candidate_radial_coordinates": str(coord_path),
        "radial_counts_by_status": str(counts_path),
        "radial_count_vs_distance": str(count_path),
        "radial_density_vs_distance": str(density_path),
        "radial_fraction_vs_distance": str(fraction_path),
        "radial_cumulative_fraction": str(cumulative_path),
    }
    (out_dir / "radial_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    LOG.warning("Wrote radial analysis -> %s", out_dir)
    return summary
