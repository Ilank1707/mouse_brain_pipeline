#!/usr/bin/env python
"""INJECTION-CENTRED radial candidate distance/density charts.

This is a DIFFERENT, optional spatial analysis from the candidate-to-candidate
pair correlation. It measures every candidate's distance FROM THE INJECTION
CENTRE and writes the radial CSVs and four charts (count, density, fraction,
cumulative fraction). Results are PROVISIONAL candidates -- never final cell
counts. For candidate-to-candidate clustering use ``run_pair_correlation.py``.

Examples:
  python scripts/injection_centered_radial_analysis.py --config config.yml \
      --run-dir "C:/mouse_brain_work/candidates/runs/section070_maskfix_01"
  python scripts/injection_centered_radial_analysis.py --config config.yml \
      --run-dir RUN --channel green_signal --center-xy 3300 3100 --bin-width-um 100
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.radial_report import analyze_run
from mouse_brain_pipeline.utilities import setup_logging


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(
        description="Injection-centred radial candidate analysis (PROVISIONAL candidates).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--run-dir", required=True, help="Run folder containing all_candidates.csv")
    p.add_argument("--channel", default=None, help="Channel (default: radial_analysis.channel)")
    p.add_argument("--section", type=int, default=None)
    p.add_argument("--center-xy", type=float, nargs=2, metavar=("X", "Y"), default=None,
                   help="Injection centre in full-resolution px (overrides config).")
    p.add_argument("--bin-width-um", type=float, default=None)
    p.add_argument("--maximum-radius-um", type=float, default=None)
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: <run-dir>/radial_analysis).")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.config)
    summary = analyze_run(
        args.run_dir, cfg, channel=args.channel, section=args.section,
        center_xy=args.center_xy, bin_width_um=args.bin_width_um,
        maximum_radius_um=args.maximum_radius_um, out_dir=args.out_dir,
    )
    print("=" * 70)
    print("RADIAL CANDIDATE ANALYSIS (PROVISIONAL candidates)")
    print("=" * 70)
    print(f"channel / section : {summary['channel']} / {summary['section']:03d}")
    print(f"centre (x, y) px  : {summary['center_xy_global_px']} [{summary['center_source']}]")
    if summary.get("center_warning"):
        print(f"WARNING           : {summary['center_warning']}")
    print(f"bins              : {summary['n_bins']} x {summary['bin_width_um']} um")
    print("-" * 70)
    for key in ("candidate_radial_coordinates", "radial_counts_by_status",
                "radial_count_vs_distance", "radial_density_vs_distance",
                "radial_fraction_vs_distance", "radial_cumulative_fraction"):
        print(f"{key:32}: {summary[key]}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
