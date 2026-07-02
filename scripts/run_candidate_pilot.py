#!/usr/bin/env python
"""Run the EXPERIMENTAL pilot candidate detector on a small section range.

Outputs are PROVISIONAL CANDIDATE detections (pilot), NOT final cell counts.
Automatic injection geometry remains unconfirmed until its mask is explicitly
validated and passes QC; every candidate remains in the output table.

NOTE: ``--crop`` is X_MIN X_MAX Y_MIN Y_MAX (full-resolution pixels).

Examples:
  python scripts/run_candidate_pilot.py --config config.yml --dry-run
  python scripts/run_candidate_pilot.py --config config.yml --first-section 70 --n 1
  python scripts/run_candidate_pilot.py --config config.yml --crop 1000 5000 500 4000
  python scripts/run_candidate_pilot.py --config config.yml --crop 1000 5000 500 4000 --save-review-patches
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone

import _bootstrap  # noqa: F401

from mouse_brain_pipeline import CHANNEL_2_SIGNAL, GREEN_SIGNAL, channel_display_name
from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.candidate_detection import (
    STATUS_ARTIFACT,
    STATUS_INJECTION,
    STATUS_INVALID_MEASUREMENT,
    STATUS_MANUAL_REVIEW,
    STATUS_RULE_FAILED,
    STATUS_RULE_PASSED,
    STATUS_SUSPECT_INJECTION,
    SectionDetectionResult,
    detect_candidates_in_stack,
    detect_section,
    build_shared_tissue_mask,
    params_from_config,
    read_crop_stack,
    write_candidate_tables,
)
from mouse_brain_pipeline.candidate_qc import (
    save_review_patches,
    select_review_batch,
    write_channel_qc,
    write_intensity_diagnostics,
    write_mask_diagnostics,
    write_native_qc,
    write_qc_display_metadata,
    write_qc_image_metadata,
    write_review_batch,
    write_shared_tissue_qc,
    write_status_summary,
)
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.coordinate_exports import (
    write_coordinate_exports,
    write_count_summaries,
)
from mouse_brain_pipeline.injection_overrides import (
    apply_overrides_to_config,
    load_overrides,
    overrides_hash,
)
from mouse_brain_pipeline.timing import StageTimer
from mouse_brain_pipeline.pilot_stack import section_availability
from mouse_brain_pipeline.run_layout import (
    code_version,
    config_hash,
    create_run_dir,
    is_single_section,
    make_run_id,
    write_latest_run,
)
from mouse_brain_pipeline.utilities import ensure_dir, setup_logging

# Fixed order for the terminal status summary so no category is ever omitted.
_SUMMARY_STATUS_ORDER = [
    STATUS_RULE_PASSED,
    STATUS_RULE_FAILED,
    STATUS_MANUAL_REVIEW,
    STATUS_INVALID_MEASUREMENT,
    STATUS_SUSPECT_INJECTION,
    STATUS_INJECTION,
    STATUS_ARTIFACT,
]


def main() -> int:
    p = argparse.ArgumentParser(description="EXPERIMENTAL pilot candidate detector (not final cells).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--first-section", type=int, default=None)
    p.add_argument("--n", type=int, default=None, help="Number of sections (default config pilot value)")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
                   default=None, help="Process only this XY crop (full-res px)")
    p.add_argument("--work-dir", default=None,
                   help="Override data.work_dir (e.g. a full-image output folder) "
                        "without copying the config")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-preview", action="store_true", help="Skip QC images (CSV only)")
    p.add_argument("--save-review-patches", action="store_true",
                   help="Save centred XYZ review patches for the manual-review batch")
    p.add_argument("--run-name", default=None,
                   help="Name this run's isolated output folder (default: timestamp_sectionNNN).")
    p.add_argument("--render-seven-planes", action="store_true",
                   help="After detection, render the seven peak-assigned QC images for this run.")
    p.add_argument("--fullres-seven-planes", action="store_true",
                   help="Write full-resolution seven-plane PNGs (slow). Default is fast previews.")
    p.add_argument("--injection-overrides", default=None,
                   help="Optional YAML with manual injection mask polygons "
                        "(e.g. config_injection_overrides.yml).")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    timer = StageTimer()
    cfg = load_config(args.config)
    # Optional manual injection-mask override (kept separate from config.yml).
    injection_overrides = load_overrides(args.injection_overrides) if args.injection_overrides else {}
    if injection_overrides:
        apply_overrides_to_config(cfg, injection_overrides)
        print(f"injection overrides  : {args.injection_overrides}")
    # Crop is controlled only by --crop; --work-dir lets one config serve both
    # cropped and full-image runs without copying it.
    if args.work_dir is not None:
        cfg.data.work_dir = args.work_dir
    if args.first_section is not None:
        cfg.pilot.first_section = args.first_section
    if args.n is not None:
        cfg.pilot.number_of_sections = args.n

    regex = cfg.data.filename_regex
    green = index_channel(GREEN_SIGNAL, cfg.data.green_signal_dir, regex)
    ch2 = index_channel(CHANNEL_2_SIGNAL, cfg.data.channel_2_signal_dir, regex)
    all_sections = sorted(green.sections | ch2.sections)
    if not all_sections:
        print("No sections discovered. Check channel directories in config.yml.")
        return 1

    first = cfg.pilot.first_section if cfg.pilot.first_section is not None else all_sections[0]
    sections = list(range(first, first + max(1, cfg.pilot.number_of_sections)))
    section_report = section_availability(sections, all_sections)
    available_sections = section_report["available"]
    skipped_sections = section_report["skipped"]
    cfg.pilot.first_section = first  # anchor for z calculations
    params = params_from_config(cfg)
    crop = tuple(args.crop) if args.crop else None
    voxel = cfg.acquisition.voxel_size_zyx

    channels = (GREEN_SIGNAL, green), (CHANNEL_2_SIGNAL, ch2)

    print("=" * 70)
    print("EXPERIMENTAL CANDIDATE DETECTION (pilot) -- results are NOT final cell counts")
    print("=" * 70)
    print(f"config path        : {cfg.source_path}")
    print(f"work directory     : {cfg.work_dir}")
    print(f"crop (X_MIN X_MAX Y_MIN Y_MAX): {crop if crop else 'none (full XY section)'}")
    print(f"mode               : {'XY crop' if crop else 'full XY section'}")
    print(f"backend            : {params.backend}")
    print(f"requested sections : {sections}")
    print(f"available sections : {available_sections}")
    print(f"skipped sections   : {skipped_sections}")
    print(f"diameter           : {params.min_diameter_um}-{params.max_diameter_um} um")
    print(f"consecutive planes : {params.min_consecutive_planes}-{params.max_consecutive_planes}")
    print(f"tissue mask enabled: {params.tissue.enabled}")
    print("-" * 70)
    for channel, _index in channels:
        disp = cfg.qc_display.for_channel(channel)
        inj = params.injection.for_channel(channel)
        cf = params.cellfinder.for_channel(channel)
        two_pass = bool(inj.generation_suppression_enabled
                        and params.backend == "cellfinder_candidates")
        if disp.mode == "fixed":
            disp_text = f"fixed [{disp.minimum}, {disp.maximum}]"
        else:
            disp_text = f"{disp.mode} (p{disp.lower_percentile}-p{disp.upper_percentile})"
        print(f"[{channel_display_name(channel)} ({channel})]")
        print(f"    QC display       : {disp_text}")
        print(f"    two_pass_requested      : {two_pass}")
        print(f"    injection_suppression   : {inj.generation_suppression_enabled}")
        print(f"    cellfinder n_sds thresh : {cf.n_sds_above_mean_thresh} / "
              f"tiled {cf.n_sds_above_mean_tiled_thresh}")
        if params.backend == "cellfinder_candidates" and not two_pass:
            print(f"    WARNING: two-pass injection suppression is OFF for {channel}.")
    for warning in cfg.config_warnings:
        print(f"  [CONFIG WARNING] {warning}")
    print("=" * 70)

    if args.dry_run:
        processed_sections = []
        for channel, idx in channels:
            for section in sections:
                plane_paths = {pl: path for (s, pl), path in idx.files.items() if s == section}
                if plane_paths:
                    if section not in processed_sections:
                        processed_sections.append(section)
                    detect_section(channel, section, plane_paths, cfg, params,
                                   crop=crop, dry_run=True)
        print(f"processed sections : {processed_sections}")
        print(f"skipped sections   : {[s for s in sections if s not in processed_sections]}")
        print("[dry-run] read plan logged; no detection performed.")
        return 0

    # Every real run gets its OWN isolated directory so attempts never mix and a
    # previous run is never silently overwritten.
    run_id = make_run_id(args.run_name, first)
    try:
        run_dir = create_run_dir(cfg.work_dir, run_id)
    except FileExistsError as exc:
        print("=" * 70)
        print(f"ERROR: {exc}")
        print("=" * 70)
        return 4
    out_dir = run_dir
    qc_dir = ensure_dir(out_dir / "qc")
    patch_dir = ensure_dir(out_dir / "review_patches") if args.save_review_patches else None
    print(f"isolated run dir   : {run_dir}")

    results = []
    review_rows = []
    patch_files: dict = {}
    qc_image_rows: list = []
    processed_sections = []

    for section in sections:
        #both channels get read making it more eff 
        loaded = {}
        for channel, idx in channels:
            plane_paths = {pl: path for (s, pl), path in idx.files.items() if s == section}
            if not plane_paths:
                print(f"  {channel} section {section}: no planes, skipping")
                continue
            with timer.stage("tiff_loading"):
                stack, plane_numbers, origin, _ = read_crop_stack(plane_paths, crop)
            loaded[channel] = (stack, plane_numbers, origin, plane_paths)

        if not loaded:
            continue
        processed_sections.append(section)

        shared = None
        if params.tissue.enabled:
            with timer.stage("mask_processing"):
                shared = build_shared_tissue_mask(
                    [v[0] for v in loaded.values()], voxel, params.tissue)

        section_results = []
        for channel, (stack, plane_numbers, origin, plane_paths) in loaded.items():
            try:
                sr = detect_candidates_in_stack(
                    stack, params, voxel, channel=channel, section=section,
                    first_section=first, planes_per_section=cfg.acquisition.planes_per_section,
                    plane_numbers=plane_numbers, crop_origin=origin,
                    shared_tissue_mask=shared,
                    injection_cfg=params.injection.for_channel(channel),
                    backend=params.backend,
                    cellfinder_cfg=params.cellfinder.for_channel(channel),
                    timer=timer,
                )
            except ImportError as exc:
                print("=" * 70)
                print(f"ERROR: {exc}")
                print("=" * 70)
                return 3

            res = SectionDetectionResult(
                channel=channel, section=section, candidates=sr.candidates,
                tissue_mask=sr.tissue_mask, injection_mask=sr.injection_mask,
                injection_core_mask=sr.injection_core_mask,
                injection_analysis_exclusion_mask=sr.injection_analysis_exclusion_mask,
                generation_suppression_mask=sr.generation_suppression_mask,
                generation_suppression_mask_source=sr.generation_suppression_mask_source,
                injection_components=sr.injection_components,
                mask_diagnostics=sr.mask_diagnostics,
                generation_diagnostics=sr.generation_diagnostics,
                projection=sr.projection, suppressed_projection=sr.suppressed_projection,
                corrected=sr.corrected, crop_origin=origin,
                plane_numbers=plane_numbers, plane_paths=plane_paths, backend=params.backend,
                n_invalid=sr.n_invalid, warnings=sr.warnings,
            )
            for w in res.warnings:
                print(f"  [QC WARN] {w}")

            _print_counts(channel, section, res)
            _print_injection_components(channel, section, res)
            if not args.no_preview:
                with timer.stage("qc_rendering"):
                    write_channel_qc(qc_dir, res, qc_display_cfg=cfg.qc_display,
                                     padding_values=tuple(params.padding_values))
                    qc_image_rows.extend(write_native_qc(
                        qc_dir, res, qc_display_cfg=cfg.qc_display,
                        padding_values=tuple(params.padding_values),
                    ))

            batch = select_review_batch(res.candidates, params)
            if patch_dir is not None:
                patch_files.update(save_review_patches(patch_dir, res, batch, params=params))
            review_rows.extend(batch)

            res.corrected = None  # free memory before the next channel/section
            res.suppressed_projection = None  # derived QC input no longer needed
            results.append(res)
            section_results.append(res)

        if not args.no_preview and shared is not None:
            write_shared_tissue_qc(qc_dir, section_results, shared,
                                   qc_display_cfg=cfg.qc_display,
                                   padding_values=tuple(params.padding_values))

    timer.start("csv_writing")
    paths = write_candidate_tables(out_dir, results)
    all_candidates = [c for r in results for c in r.candidates]
    review_path = write_review_batch(out_dir, review_rows, patch_files)
    mask_diagnostics_path = write_mask_diagnostics(out_dir, results)
    status_summary_path = write_status_summary(out_dir, results)
    padding = tuple(params.padding_values)
    intensity_diagnostics_path = write_intensity_diagnostics(
        out_dir, results, qc_display_cfg=cfg.qc_display, padding_values=padding
    )
    qc_display_metadata_path = write_qc_display_metadata(
        out_dir, results, qc_display_cfg=cfg.qc_display, padding_values=padding
    )
    qc_image_metadata_path = (
        write_qc_image_metadata(out_dir, qc_image_rows) if qc_image_rows else None
    )
    one_section = is_single_section(processed_sections)
    overall_status_counts = dict(Counter(c["current_status"] for c in all_candidates))
    # Original TIFF dimensions per channel (header only) so the renderer can
    # refuse a mismatched-resolution candidate table.
    from mouse_brain_pipeline.audit import read_shape_dtype  # noqa: PLC0415
    source_image_dimensions = {}
    for channel, idx in channels:
        plane_path = next(
            (path for (s, pl), path in idx.files.items()
             if s in processed_sections and pl == 1), None)
        if plane_path is not None:
            shape, _dtype = read_shape_dtype(plane_path)
            if shape:
                source_image_dimensions[channel] = {
                    "height": int(shape[0]), "width": int(shape[1])}
    metadata_path = out_dir / "candidate_run_metadata.json"
    metadata_path.write_text(json.dumps({
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_id,
        "run_dir": str(run_dir),
        "code_version": code_version(),
        "config_path": cfg.source_path,
        "config_hash": config_hash(cfg.source_path),
        "work_dir": str(cfg.work_dir),
        "config_warnings": cfg.config_warnings,
        # Human labels: green_signal is the green dye, channel_2_signal the red dye.
        "channel_display_names": {
            channel: channel_display_name(channel) for channel, _index in channels
        },
        "source_image_dimensions": source_image_dimensions,
        "cellfinder_rerun": True,
        "injection_overrides_path": args.injection_overrides,
        "injection_overrides_hash": overrides_hash(args.injection_overrides),
        # Detection runs apply no classifier; predictions never silently change.
        "classifier": {"used": False, "model": None, "validation_state": "none"},
        "status_counts": overall_status_counts,
        "crop_mode": "xy_crop" if crop else "full_xy_section",
        "one_section_not_whole_brain": one_section,
        "backend": params.backend,
        "array_order": "z,y,x",
        "requested_sections": sections,
        "available_sections": available_sections,
        "processed_sections": processed_sections,
        "skipped_sections": [s for s in sections if s not in processed_sections],
        "crop_x_min_x_max_y_min_y_max": crop,
        "acquisition": asdict(cfg.acquisition),
        "cellfinder_parameters": asdict(cfg.detection.cellfinder),
        # The exact EFFECTIVE Cellfinder config used per channel (overrides resolved).
        "effective_cellfinder_by_channel": {
            channel: asdict(params.cellfinder.for_channel(channel))
            for channel, _index in channels
        },
        "injection_exclusion_by_channel": {
            channel: asdict(params.injection.for_channel(channel))
            for channel, _index in channels
        },
        "qc_display": asdict(cfg.qc_display),
        "generation_diagnostics": [
            {"channel": r.channel, "section": r.section, **(r.generation_diagnostics or {})}
            for r in results
        ],
        "candidate_counts_by_channel": {
            channel: sum(1 for c in all_candidates if c["channel"] == channel)
            for channel, _index in channels
        },
        "candidate_generation_source_counts": {
            channel: _generation_source_counts(all_candidates, channel)
            for channel, _index in channels
        },
    }, indent=2), encoding="utf-8")

    # Clean coordinate CSV exports + reconciling count summaries per channel+section
    # (channel-scoped so the two signal channels never collide).
    ppl = cfg.acquisition.planes_per_section
    for res in results:
        scope = ensure_dir(out_dir / "coordinate_exports" / res.channel)
        write_coordinate_exports(scope, res.candidates, channel=res.channel,
                                 section=res.section, planes_per_section=ppl)
        write_count_summaries(scope, res.candidates, channel=res.channel,
                              section=res.section, planes_per_section=ppl)
    timer.stop("csv_writing")

    n_invalid = sum(r.n_invalid for r in results)

    print("-" * 70)
    print(f"all candidates       : {paths['all']}")
    print(f"preliminary-pass CSV : {paths['preliminary_pass']}")
    print(f"review batch CSV     : {review_path}")
    print(f"mask diagnostics CSV : {mask_diagnostics_path}")
    print(f"status summary CSV   : {status_summary_path}")
    print(f"intensity diag CSV   : {intensity_diagnostics_path}")
    print(f"qc display meta CSV  : {qc_display_metadata_path}")
    if qc_image_metadata_path is not None:
        print(f"qc image meta CSV    : {qc_image_metadata_path}")
    print(f"run metadata JSON    : {metadata_path}")
    print(f"QC images            : {qc_dir}")
    if patch_dir is not None:
        print(f"review patches       : {patch_dir}")
    print("-" * 70)
    print(f"requested sections     : {sections}")
    print(f"available sections     : {available_sections}")
    print(f"processed sections     : {processed_sections}")
    print(f"skipped sections       : {[s for s in sections if s not in processed_sections]}")
    print(f"total candidates       : {len(all_candidates)}")
    for channel, _index in channels:
        print(f"  {channel} generation sources: "
              f"{_generation_source_counts(all_candidates, channel)}")
    if n_invalid:
        print(f"*** {n_invalid} INVALID-COORDINATE candidates detected -- "
              f"SCIENTIFIC COUNTS WITHHELD. Investigate before trusting any numbers. ***")
    overall = Counter(c["current_status"] for c in all_candidates)
    # Always show every category, including zeros, in a fixed order.
    for status in _SUMMARY_STATUS_ORDER:
        print(f"{status:28}: {overall.get(status, 0)}")
    for status in sorted(set(overall) - set(_SUMMARY_STATUS_ORDER)):
        print(f"{status:28}: {overall[status]}  (other)")
    print(f"status counts reconcile : {sum(overall.values()) == len(all_candidates)}")
    print("=" * 70)

    # Seven-plane peak-assigned QC rendering from THIS run only (never a search).
    if args.render_seven_planes:
        from mouse_brain_pipeline.seven_plane_report import RenderRefusedError, render_run
        fullres = args.fullres_seven_planes
        print(f"Rendering seven peak-assigned QC images from this run "
              f"({'full-resolution' if fullres else 'fast previews'})...")
        with timer.stage("seven_plane_rendering"):
            for res in results:
                try:
                    out = render_run(
                        run_dir, res.channel, res.section, config=cfg,
                        subdir=res.channel, write_exports=False,
                        allow_cropped=crop is not None, fullres=fullres,
                    )
                except RenderRefusedError as exc:
                    print(f"  [{res.channel} s{res.section:03d}] render refused: {exc}")
                    continue
                rec = out["reconciliation"]
                print(f"  [{res.channel} s{res.section:03d}] unique={rec['unique_total']} "
                      f"assigned={rec['assigned_total']} unassigned={out['unassigned']} "
                      f"reconciles={rec['peak_assignment_reconciles'] and rec['status_reconciles']} "
                      f"-> {out['qc_dir']}")

    # Stage timings for this run (never overwrites another run -- own folder).
    timings_path = timer.write_csv(out_dir / "stage_timings.csv")

    # Record the newest successfully completed run (only after success).
    latest = write_latest_run(cfg.work_dir, run_dir, {
        "run_name": run_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(all_candidates),
        "candidate_counts_by_channel": {
            channel: sum(1 for c in all_candidates if c["channel"] == channel)
            for channel, _index in channels
        },
        "status_counts": overall_status_counts,
        "code_version": code_version(),
        "config_hash": config_hash(cfg.source_path),
    })

    print("-" * 70)
    print(f"isolated run dir     : {run_dir}")
    print(f"coordinate exports   : {out_dir / 'coordinate_exports'}")
    if args.render_seven_planes:
        print(f"seven-plane QC       : {out_dir / 'seven_plane_qc'}")
    print(f"stage timings CSV    : {timings_path}")
    print(f"latest run pointer   : {latest}")
    print("=" * 70)
    if one_section:
        print("This run contains one section and is not a whole-brain count.")
    print("REMINDER: PROVISIONAL candidate detections. No state means 'real cell'.")
    print("Use review_candidates.py for human labels; do not infer labels from rules.")
    return 0


def _print_injection_components(channel, section, res) -> None:
    components = res.injection_components or {}
    if not components.get("seed_filter_applied"):
        return
    print(f"  {channel} section {section}: injection seed filtering "
          f"(kept {components.get('n_kept', 0)} / removed {components.get('n_removed', 0)})")
    for comp in components.get("components", []):
        verdict = "KEPT (seeded)" if comp["kept"] else "REMOVED (no seed)"
        print(f"    component {comp['label']}: area={comp['area_px']} px "
              f"at (x={comp['centroid_x_local']}, y={comp['centroid_y_local']}) -> {verdict}")


def _generation_source_counts(all_candidates, channel) -> dict:
    counts: dict = {}
    for c in all_candidates:
        if c.get("channel") != channel:
            continue
        source = c.get("candidate_generation_source", "raw_stack")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _print_counts(channel, section, res) -> dict:
    counts = Counter(c["current_status"] for c in res.candidates)
    rendered = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(
        f"  {channel} section {section}: {len(res.candidates)} candidates -> "
        f"{rendered}; invalid-coord={res.n_invalid}; "
        f"reconciles={sum(counts.values()) == len(res.candidates)}"
    )
    if res.mask_diagnostics:
        d = res.mask_diagnostics
        print(
            f"    mask diagnostics: area={d['mask_area_px']} px / tissue={d['tissue_area_px']} px "
            f"({d['mask_fraction_of_tissue']:.1%}); candidates inside="
            f"{d['candidates_inside_mask']}/{len(res.candidates)} "
            f"({d['candidate_fraction_inside_mask']:.1%})"
        )
    return dict(counts)


if __name__ == "__main__":
    sys.exit(main())
