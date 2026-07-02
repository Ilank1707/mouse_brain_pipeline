#!/usr/bin/env python
"""Audit the dataset: discover, parse, pair, validate, manifest.

Examples:
  python scripts/audit_dataset.py --config config.yml
  python scripts/audit_dataset.py --config config.yml --dry-run
  python scripts/audit_dataset.py --check-env
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import run_audit
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.utilities import check_environment, print_environment_report, setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Audit a serial two-photon mouse-brain dataset.")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--dry-run", action="store_true", help="Validate but write nothing")
    p.add_argument("--no-metadata", action="store_true", help="Skip TIFF header shape/dtype checks")
    p.add_argument("--check-env", action="store_true", help="Print the environment report and exit")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    if args.check_env:
        cfg_paths = []
        try:
            cfg = load_config(args.config)
            cfg_paths = [cfg.data.green_signal_dir, cfg.data.channel_2_signal_dir, cfg.data.work_dir]
        except Exception:
            pass
        print_environment_report(check_environment(extra_paths=cfg_paths))
        return 0

    cfg = load_config(args.config)
    result = run_audit(cfg, check_metadata=not args.no_metadata, dry_run=args.dry_run)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
