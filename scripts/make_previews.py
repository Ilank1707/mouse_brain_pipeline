#!/usr/bin/env python
"""Create downsampled PNG previews (single-channel + overlay) for QC.

Examples:
  python scripts/make_previews.py --config config.yml --n 5 --downsample 16
  python scripts/make_previews.py --config config.yml --dry-run
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import run_audit
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.previews import make_preview, select_preview_specs
from mouse_brain_pipeline.utilities import ensure_dir, setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Make downsampled preview PNGs.")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--n", type=int, default=5, help="Number of representative planes")
    p.add_argument("--downsample", type=int, default=16, help="XY downsample factor")
    p.add_argument("--low", type=float, default=1.0, help="Low display percentile")
    p.add_argument("--high", type=float, default=99.5, help="High display percentile")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.config)

    # Lightweight discovery (no metadata, no files written) to get the manifest in memory.
    audit = run_audit(cfg, check_metadata=False, dry_run=True)
    specs = select_preview_specs(audit.manifest_rows, n=args.n)
    if not specs:
        print("No planes available to preview. Check the channel directories in config.yml.")
        return 1

    out_dir = ensure_dir(cfg.work_dir / "previews")
    print(f"Selected {len(specs)} plane(s) for preview -> {out_dir}")
    for spec in specs:
        files = make_preview(
            spec, out_dir, cfg,
            downsample=args.downsample, low=args.low, high=args.high, dry_run=args.dry_run,
        )
        for f in files:
            print(f"  {'(would write) ' if args.dry_run else ''}{f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
