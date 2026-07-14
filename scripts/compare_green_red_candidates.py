#!/usr/bin/env python
"""Compare green/red signal at identical XY/Z PROVISIONAL candidate locations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.channel_comparison import (
    MODE_APPLY,
    MODE_REPORT,
    run_channel_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Auditable green/red comparison for PROVISIONAL candidates; "
            "source TIFFs and runs are read only."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--section",
        required=True,
        type=int,
        action="append",
        help="Section to compare; repeat for multiple sections.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", required=True, choices=(MODE_REPORT, MODE_APPLY))
    # None means omitted. Report mode resolves omissions from visible config
    # defaults; apply mode requires every value explicitly on this command line.
    parser.add_argument("--min-dominance-ratio", type=float, default=None)
    parser.add_argument("--min-snr", type=float, default=None)
    parser.add_argument("--max-match-distance-um", type=float, default=None)
    return parser


def _require_apply_thresholds(args: argparse.Namespace) -> None:
    if args.mode != MODE_APPLY:
        return
    missing = [
        flag
        for flag, value in (
            ("--min-dominance-ratio", args.min_dominance_ratio),
            ("--min-snr", args.min_snr),
            ("--max-match-distance-um", args.max_match_distance_um),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(
            "apply mode requires all thresholds explicitly: " + ", ".join(missing)
        )


def run_cli(args: argparse.Namespace) -> dict:
    from mouse_brain_pipeline.config import load_config  # noqa: PLC0415

    _require_apply_thresholds(args)
    config = load_config(args.config)
    defaults = config.channel_comparison
    min_dominance_ratio = (
        args.min_dominance_ratio
        if args.min_dominance_ratio is not None
        else defaults.default_min_dominance_ratio
    )
    min_snr = (
        args.min_snr if args.min_snr is not None else defaults.default_min_snr
    )
    max_match_distance_um = (
        args.max_match_distance_um
        if args.max_match_distance_um is not None
        else defaults.default_max_match_distance_um
    )
    result = run_channel_comparison(
        config=config,
        run_dir=Path(args.run_dir),
        sections=args.section,
        out_dir=Path(args.out_dir),
        mode=args.mode,
        min_dominance_ratio=min_dominance_ratio,
        min_snr=min_snr,
        max_match_distance_um=max_match_distance_um,
    )
    print(
        f"{args.mode}: compared {result['candidate_count']} PROVISIONAL candidates"
    )
    print(f"Wrote channel-comparison outputs to {result['out_dir']}")
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
