#!/usr/bin/env python
"""Post-process an existing candidate run WITHOUT rerunning Cellfinder.

Applies an injection-mask override to already-detected candidates, recomputes
mask membership + statuses, and writes a brand-new isolated run (QC, CSVs and
optionally seven-plane images + radial charts). The source run is never touched.

Example:
  python scripts/postprocess_candidate_run.py `
    --config config.yml `
    --source-run-dir "C:/mouse_brain_work/candidates/7photosrun/section070_test_07" `
    --injection-overrides config_injection_overrides.yml `
    --new-run-name section070_maskfix_01 `
    --render-seven-planes `
    --radial-analysis
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.postprocess import postprocess_run
from mouse_brain_pipeline.utilities import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(
        description="Post-process a run with a new injection override (no Cellfinder rerun).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--source-run-dir", required=True)
    p.add_argument("--injection-overrides", default=None,
                   help="YAML with manual injection polygons (config_injection_overrides.yml).")
    p.add_argument("--new-run-name", required=True)
    p.add_argument("--work-dir", default=None, help="Override data.work_dir.")
    p.add_argument("--render-seven-planes", action="store_true",
                   help="Render seven-plane QC (fast previews unless --fullres-seven-planes).")
    p.add_argument("--fullres-seven-planes", action="store_true",
                   help="Write full-resolution seven-plane PNGs (slow).")
    p.add_argument("--radial-analysis", action="store_true",
                   help="Also write radial candidate charts for the new run.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.config)
    if args.work_dir is not None:
        cfg.data.work_dir = args.work_dir

    print("=" * 70)
    print("POST-PROCESS CANDIDATE RUN (no Cellfinder rerun)")
    print("=" * 70)
    print(f"source run   : {args.source_run_dir}")
    print(f"overrides    : {args.injection_overrides}")
    print(f"new run name : {args.new_run_name}")

    try:
        result = postprocess_run(
            config=cfg, source_run_dir=args.source_run_dir,
            new_run_name=args.new_run_name, work_dir=cfg.work_dir,
            overrides_path=args.injection_overrides,
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        return 4
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    run_dir = result["run_dir"]
    timer = result["timer"]
    print(f"new run dir  : {run_dir}")
    print("-" * 70)
    for change in result["status_changes"]:
        print(f"  [{change['channel']} s{change['section']:03d}] "
              f"status changed for {change['status_changed']}/{change['candidates']} "
              f"candidates; removed {change['removed_polygon_pixels']} px, "
              f"added {change['added_polygon_pixels']} px")
    print(f"status counts: {result['status_counts']}")
    for path in result["mask_qc_paths"]:
        print(f"  mask QC     : {path}")

    # Optional seven-plane render from the NEW run only.
    if args.render_seven_planes:
        from mouse_brain_pipeline.seven_plane_report import RenderRefusedError, render_run
        fullres = args.fullres_seven_planes
        print(f"Rendering seven-plane QC ({'full-res' if fullres else 'fast previews'})...")
        with timer.stage("seven_plane_rendering"):
            for res in result["results"]:
                try:
                    out = render_run(run_dir, res.channel, res.section, config=cfg,
                                     subdir=res.channel, write_exports=False,
                                     allow_cropped=True, fullres=fullres)
                    print(f"  [{res.channel} s{res.section:03d}] -> {out['qc_dir']}")
                except RenderRefusedError as exc:
                    print(f"  [{res.channel} s{res.section:03d}] render refused: {exc}")

    # Optional radial analysis on the NEW run.
    if args.radial_analysis:
        from mouse_brain_pipeline.radial_report import analyze_run
        print("Computing radial candidate analysis...")
        with timer.stage("radial_analysis"):
            for res in result["results"]:
                try:
                    summary = analyze_run(run_dir, cfg, channel=res.channel,
                                          section=res.section)
                    print(f"  [{res.channel} s{res.section:03d}] radial -> "
                          f"{summary['radial_density_vs_distance']}")
                except (ValueError, FileNotFoundError) as exc:
                    print(f"  [{res.channel} s{res.section:03d}] radial skipped: {exc}")

    timings_path = timer.write_csv(run_dir / "stage_timings.csv")
    print("-" * 70)
    print(f"stage timings: {timings_path}")
    print(f"new run dir  : {run_dir}")
    print("REMINDER: PROVISIONAL candidate detections; source run left unchanged.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
