#!/usr/bin/env python
"""Create read-only green/red histograms for PROVISIONAL candidates."""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.channel_overlay_histograms import (
    CHANNEL_2_SIGNAL,
    GREEN_SIGNAL,
    run_channel_overlay_histograms,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure both signal channels at identical XY/Z locations and create "
            "analysis-only histograms for PROVISIONAL candidates."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--section", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--green-channel", required=True, choices=(GREEN_SIGNAL,)
    )
    parser.add_argument(
        "--red-channel", required=True, choices=(CHANNEL_2_SIGNAL,)
    )
    parser.add_argument("--ratio-threshold", type=float, default=2.0)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    return parser


def run_cli(args: argparse.Namespace) -> dict:
    from mouse_brain_pipeline.config import load_config  # noqa: PLC0415

    result = run_channel_overlay_histograms(
        config=load_config(args.config),
        run_dir=args.run_dir,
        section=args.section,
        out_dir=args.out_dir,
        green_channel=args.green_channel,
        red_channel=args.red_channel,
        ratio_threshold=args.ratio_threshold,
        snr_threshold=args.snr_threshold,
    )
    print(
        f"Compared {result['candidate_count']} PROVISIONAL candidates; "
        f"outputs: {result['out_dir']}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    run_cli(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
