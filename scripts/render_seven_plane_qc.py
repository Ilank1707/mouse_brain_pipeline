#!/usr/bin/env python
"""Whole-section seven-plane candidate QC renderer.

Generates full-resolution candidate overlays of the SAME kind as
``04_candidates_before_injection_exclusion.png`` -- one per optical plane of a
section (``section_070_01.tif`` .. ``section_070_07.tif``) -- plus a combined
montage and a marker-free raw montage. This does NOT replace the interactive
candidate reviewer (``review_candidates.py``); it is a separate, static export.

It also guards against the classic mistake of rendering an **old cropped run**:
it prints the candidate CSV path and counts prominently, detects from the run
metadata whether the CSV came from a crop, refuses to render a full section from
a cropped CSV unless ``--allow-cropped-candidates`` is given, and can
auto-select the newest valid full-section run with ``--find-latest-full-section``.

This task only fixes candidate-file selection and rendering. It never reruns
Cellfinder, never changes candidate coordinates / counts / statuses, and never
modifies the raw TIFFs. Display scaling is visualisation only.

Examples
--------
    # auto-select the latest full-section run for the channel + section:
    python scripts/render_seven_plane_qc.py --config config.yml ^
      --channel green_signal --section 70 --find-latest-full-section ^
      --output "C:\\mouse_brain_work\\candidates\\seven_plane_qc" --marker-mode all

    # explicit CSV (refused if it is a cropped run, unless --allow-cropped-candidates):
    python scripts/render_seven_plane_qc.py --config config.yml ^
      --channel green_signal --section 70 ^
      --candidates "C:\\mouse_brain_work\\candidates\\full_run\\all_candidates.csv" ^
      --output "...\\seven_plane_qc" --marker-mode all --display-mode per-plane-robust
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import index_channel, read_shape_dtype
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.review import read_csv_rows
from mouse_brain_pipeline.review_patches import ordered_section_planes
from mouse_brain_pipeline.seven_plane_qc import (
    classify_run_crop,
    count_mismatch_warning,
    crop_covers_full_image,
    find_latest_full_section_csv,
    read_run_metadata,
    recorded_candidate_count,
    render_section_seven_planes,
    select_candidate_rows,
)

DEFAULT_CANDIDATES_ROOT = r"C:\mouse_brain_work\candidates"


def _channel_dir(config, channel):
    return (
        config.data.green_signal_dir
        if channel == "green_signal"
        else config.data.channel_2_signal_dir
    )


def _render_run_dir(args, config) -> int:
    """Render peak-assigned QC images for exactly one run directory."""
    from pathlib import Path

    from mouse_brain_pipeline.seven_plane_report import RenderRefusedError, render_run

    run_dir = Path(args.run_dir)
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Channel: {args.channel}   Section: {args.section}")
    print(f"Reads only: {run_dir / 'all_candidates.csv'}")
    print(f"            {run_dir / 'candidate_run_metadata.json'}")
    print("=" * 70)
    try:
        out = render_run(
            run_dir, args.channel, args.section, config=config,
            display_mode="per_plane_robust", allow_cropped=args.allow_cropped,
            make_preview=not args.no_preview, write_exports=True, subdir=None,
            planes_per_section=config.acquisition.planes_per_section,
        )
    except RenderRefusedError as exc:
        print(f"REFUSING TO RENDER: {exc}")
        return 2
    rec = out["reconciliation"]
    print(f"unique 3D candidates           : {rec['unique_total']}")
    print(f"assigned across planes 01-07   : {rec['assigned_total']}")
    print(f"unassigned (missing/invalid Z) : {out['unassigned']}")
    print(f"peak-assignment reconciles     : {rec['peak_assignment_reconciles']}")
    print(f"status reconciles              : {rec['status_reconciles']}")
    print("per-plane assigned counts:")
    for plane in range(1, config.acquisition.planes_per_section + 1):
        print(f"  plane {plane:02d}: {rec['per_plane_assigned'].get(plane, 0)}")
    print(f"seven-plane QC images : {out['qc_dir']}")
    print(f"support views         : {out['support_dir']}")
    print(f"coordinate exports    : {run_dir / 'coordinate_exports'}")
    print("Peak-assigned images draw each candidate on exactly ONE plane "
          "(no double counting). Support views may repeat candidates - do not sum them.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render full-resolution seven-plane candidate QC images for one section.",
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--channel", required=True, choices=["green_signal", "channel_2_signal"])
    parser.add_argument("--section", required=True, type=int)
    parser.add_argument("--run-dir", default=None,
                        help="Render the PEAK-ASSIGNED QC images for exactly this run "
                             "directory (reads only <run-dir>/all_candidates.csv and "
                             "<run-dir>/candidate_run_metadata.json).")
    parser.add_argument("--allow-cropped", action="store_true",
                        help="With --run-dir: permit a full-section render of a cropped run.")
    parser.add_argument("--no-preview", action="store_true",
                        help="With --run-dir: skip the downscaled *_preview.png files.")
    parser.add_argument("--candidates", default=None,
                        help="all_candidates.csv (default: work/candidates/all_candidates.csv).")
    parser.add_argument("--find-latest-full-section", action="store_true",
                        help="Auto-select the newest valid full-section run (by run metadata "
                             "timestamp) under --candidates-root.")
    parser.add_argument("--candidates-root", default=DEFAULT_CANDIDATES_ROOT,
                        help="Root searched by --find-latest-full-section.")
    parser.add_argument("--allow-cropped-candidates", action="store_true",
                        help="Permit rendering a full section from a CROPPED candidate CSV.")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: work/candidates/seven_plane_qc).")
    parser.add_argument("--marker-mode", choices=["all", "support"], default=None,
                        help="all: every candidate on every plane (default); "
                             "support: only candidates supported on that plane.")
    parser.add_argument("--mode", choices=["all", "support", "support_only"], default=None,
                        help="Deprecated alias of --marker-mode.")
    parser.add_argument("--display-mode", choices=["configured", "per-plane-robust"],
                        default="configured",
                        help="configured: one section window; per-plane-robust: a robust "
                             "window computed independently for each (dimmer) plane.")
    parser.add_argument("--display-min", type=float, default=None,
                        help="Override display minimum (display only; e.g. 0 for the Fiji view).")
    parser.add_argument("--display-max", type=float, default=None,
                        help="Override display maximum (display only; e.g. 513 for channel 2).")
    parser.add_argument("--montage-columns", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)

    # Preferred path: render the peak-assigned QC images for exactly one run dir.
    if args.run_dir is not None:
        return _render_run_dir(args, config)

    # Resolve the marker mode (default all). --marker-mode wins over --mode.
    if args.marker_mode is not None:
        marker_mode = args.marker_mode
    elif args.mode is not None:
        marker_mode = "support" if args.mode in ("support", "support_only") else "all"
    else:
        marker_mode = "all"
    display_mode = "per_plane_robust" if args.display_mode == "per-plane-robust" else "configured"

    # The channel TIFFs give us the ORIGINAL image dimensions (header only).
    index = index_channel(args.channel, _channel_dir(config, args.channel),
                          config.data.filename_regex)
    ordered = ordered_section_planes(index, args.section)
    if not ordered:
        print(f"ERROR: no TIFF planes for channel {args.channel!r}, section {args.section} "
              f"under {_channel_dir(config, args.channel)!r}.")
        return 1
    shape, _dtype = read_shape_dtype(ordered[0][1])
    image_height = int(shape[0]) if shape else None
    image_width = int(shape[1]) if shape else None

    # Resolve which candidate CSV to use.
    if args.find_latest_full_section:
        found = find_latest_full_section_csv(
            args.candidates_root, args.section, args.channel,
            image_width=image_width, image_height=image_height,
        )
        if found is None:
            print(f"ERROR: no full-section run found under {args.candidates_root!r} with "
                  f"crop=none, section {args.section} processed and channel {args.channel!r} "
                  f"present (matching {image_width}x{image_height}).")
            return 1
        candidates_path, run_meta = found
    else:
        candidates_path = Path(
            args.candidates or config.work_dir / "candidates" / "all_candidates.csv"
        )
        if not candidates_path.is_file():
            print(f"ERROR: candidates CSV not found: {candidates_path}")
            return 1
        run_meta = read_run_metadata(candidates_path)

    all_rows = read_csv_rows(candidates_path)
    channel_rows = [r for r in all_rows if r.get("channel") == args.channel]
    section_rows = select_candidate_rows(all_rows, args.channel, args.section)

    crop_kind = classify_run_crop(run_meta)
    crop = (run_meta or {}).get("crop_x_min_x_max_y_min_y_max")
    full_extent_crop = crop_covers_full_image(crop, image_width, image_height)
    crop_display = "none" if crop_kind == "full" else (
        "full-image crop" if full_extent_crop else (crop if crop else "unknown"))

    # ----- prominent pre-render report -----
    print("=" * 70)
    print(f"Candidate CSV: {candidates_path}")
    print(f"Channel: {args.channel}")
    print(f"Section: {args.section}")
    print(f"Candidates loaded: {len(section_rows)}")
    print(f"Run crop: {crop_display}")
    print(f"Source image dimensions: {image_width} x {image_height}")
    print("=" * 70)

    recorded = recorded_candidate_count(run_meta, args.channel)
    if recorded is not None:
        print(f"Run metadata candidate count for {args.channel}: {recorded} "
              f"(this CSV has {len(channel_rows)} {args.channel} rows across all sections)")
    if run_meta is None:
        print("WARNING: no candidate_run_metadata.json next to this CSV -- cannot verify "
              "whether it is a cropped or full-section run.")
    warning = count_mismatch_warning(len(channel_rows), recorded)
    if warning:
        print(f"WARNING: {warning}")

    # ----- refuse a cropped CSV for a full-section render -----
    if crop_kind == "crop" and not full_extent_crop and not args.allow_cropped_candidates:
        print("-" * 70)
        print("REFUSING: this candidate CSV is from a CROPPED run "
              f"(crop x_min,x_max,y_min,y_max = {crop}), but this renders the FULL section.")
        print("Use --find-latest-full-section to pick the newest full-section run, or pass "
              "--allow-cropped-candidates to override.")
        return 2

    display_override = None
    if (args.display_min is None) != (args.display_max is None):
        print("Provide both --display-min and --display-max, or neither.")
        return 1
    if args.display_min is not None:
        display_override = (args.display_min, args.display_max)

    output_dir = Path(args.output or config.work_dir / "candidates" / "seven_plane_qc")
    settings = config.qc_display.for_channel(args.channel)

    result = render_section_seven_planes(
        index, args.section, channel_rows, output_dir,
        channel=args.channel, display_settings=settings,
        mode=marker_mode, display_mode=display_mode,
        display_override=display_override,
        minimum_pixels=config.qc_display.minimum_pixels,
        padding_values=tuple(config.detection.padding_values),
        planes_per_section=config.acquisition.planes_per_section,
        montage_columns=args.montage_columns,
    )

    print(f"Marker mode: {marker_mode}   Display mode: {result['display_mode']}")
    if result["display_mode"] == "per_plane_robust":
        print("Per-plane robust display windows (display only; raw values unchanged):")
        for row in result["metadata_rows"]:
            print(f"  plane {row['optical_plane']:02d}: "
                  f"min={row['display_min']:.1f} max={row['display_max']:.1f}")
    else:
        print(f"Display window: min={result['display_min']:.2f} max={result['display_max']:.2f}")
    print(f"Output folder: {output_dir}")
    print("Full-resolution planes (each matches its source TIFF dimensions):")
    all_match = True
    for row in result["metadata_rows"]:
        ok = not row["resizing_occurred"] and \
            (row["saved_width"], row["saved_height"]) == (image_width, image_height)
        all_match = all_match and ok
        print(f"  {row['filename']}: {row['saved_width']}x{row['saved_height']} "
              f"(source {row['source_width']}x{row['source_height']}) "
              f"[{'OK' if ok else 'MISMATCH!'}]  "
              f"displayed={row['candidates_displayed']} "
              f"supported={row['candidates_supported_on_plane']}")
    print(f"Montage:     {result['montage'].name}")
    print(f"Raw montage: {result['raw_montage'].name}")
    print(f"Metadata:    {result['metadata_csv'].name}")
    print("All full-resolution outputs match the original TIFF dimensions: "
          f"{'YES' if all_match else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
