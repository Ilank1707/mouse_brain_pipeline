#!/usr/bin/env python
"""Cross-channel (green vs red) overlay analysis for an existing candidate run.

Measures every candidate in ``<run-dir>/all_candidates.csv`` with the SAME
fixed-XY seven-plane measurement used by the detector, in BOTH biological
channels, and labels each candidate ``green_dominant`` / ``red_dominant`` /
``both`` / ``unclear`` from the MEASURED signal. It never changes a candidate,
status, mask or count, never uses one channel as the other's input, and never
forces the red channel to have fewer detections.

Writes, under ``<run-dir>/channel_overlay/``:
  * channel_overlay_candidate_measurements.csv
  * channel_overlay_summary.csv
  * green_red_overlay_qc.png

This runs automatically inside scripts/run_candidate_pilot.py; use this standalone
tool to (re)build the overlay for an older run.

Example:
  python scripts/overlay_channels.py --config config.yml --run-dir "PATH_TO_RUN"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.channel_overlay import analyze_run
from mouse_brain_pipeline.config import load_config


def main() -> int:
    p = argparse.ArgumentParser(description="Green/red cross-channel overlay analysis.")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--run-dir", required=True, help="Run folder containing all_candidates.csv")
    p.add_argument("--section", type=int, action="append", default=None,
                   help="Limit to this section (repeatable). Default: all sections in the run.")
    p.add_argument("--no-qc", action="store_true", help="Skip the composite QC PNG.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "all_candidates.csv").is_file():
        print(f"ERROR: no all_candidates.csv under {run_dir}")
        return 2

    cfg = load_config(args.config)
    result = analyze_run(run_dir, cfg, sections=args.section, render_qc=not args.no_qc)

    print("=" * 70)
    print("CROSS-CHANNEL OVERLAY (green vs red) -- audit only, not a cell count")
    print("=" * 70)
    print(f"candidates measured  : {result['candidate_count']}")
    for row in result["summary"]:
        if row["detection_channel"] == "all":
            print(f"  {row['dominant_channel']:15}: {row['count']}")
    print(f"measurements CSV     : {result['measurements_csv']}")
    print(f"summary CSV          : {result['summary_csv']}")
    print(f"overlay QC image     : {result['qc_png']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
