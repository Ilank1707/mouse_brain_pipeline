#!/usr/bin/env python
"""Prepare a small contiguous pilot range (ordered file lists, no data copy).

Examples:
  python scripts/prepare_pilot.py --config config.yml
  python scripts/prepare_pilot.py --config config.yml --first-section 70 --n 2
  python scripts/prepare_pilot.py --config config.yml --symlinks --dry-run
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.pilot_stack import plan_pilot, write_pilot
from mouse_brain_pipeline.utilities import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare a contiguous pilot section range.")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--first-section", type=int, default=None, help="Override pilot.first_section")
    p.add_argument("--n", type=int, default=None, help="Override pilot.number_of_sections")
    p.add_argument("--symlinks", action="store_true", help="Also create read-only symlinks")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.config)
    if args.first_section is not None:
        cfg.pilot.first_section = args.first_section
    if args.n is not None:
        cfg.pilot.number_of_sections = args.n

    plan = plan_pilot(cfg)
    write_pilot(cfg, plan, use_symlinks=args.symlinks, dry_run=args.dry_run)  # raises SystemExit(1) if missing
    return 0


if __name__ == "__main__":
    sys.exit(main())
