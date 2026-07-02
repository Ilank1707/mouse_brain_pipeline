#!/usr/bin/env python
"""Build, display and (optionally) run the Brainmapper command.

Brainmapper is REFUSED unless a real background_dir AND a confirmed orientation
are present. A signal channel is never used as background.

Examples:
  python scripts/run_brainmapper.py --config config.yml --dry-run
  python scripts/run_brainmapper.py --config config.yml --start-plane 0 --end-plane 13 --dry-run
  python scripts/run_brainmapper.py --config config.yml --confirm     # actually run (long job)
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.brainmapper_runner import run
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.utilities import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Brainmapper wrapper (signal-only datasets are blocked).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--start-plane", type=int, default=None)
    p.add_argument("--end-plane", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Show the command and stop")
    p.add_argument("--confirm", action="store_true", help="Actually execute (requires no blockers)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.config)
    # Default to dry-run unless the user explicitly confirms.
    dry = args.dry_run or not args.confirm
    return run(cfg, start_plane=args.start_plane, end_plane=args.end_plane,
               dry_run=dry, confirm=args.confirm)


if __name__ == "__main__":
    sys.exit(main())
