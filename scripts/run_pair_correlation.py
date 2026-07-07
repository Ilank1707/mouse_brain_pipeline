#!/usr/bin/env python
"""Run candidate-to-candidate pair correlation for BOTH channels in one call.

This is the entry point for candidate-to-candidate spatial clustering. It wraps
the existing ``pair_correlation_analysis`` implementation (whose statistics are
unchanged) and removes the two easy mistakes:

  * running the wrong (injection-centred) script -- use
    ``injection_centered_radial_analysis.py`` for that separate analysis; and
  * passing an ``--out-dir`` that already contains a channel name, which produced
    duplicated folders such as ``green_signal\\green_signal``.

One fresh timestamped root is created per invocation. Both ``green_signal`` and
``channel_2_signal`` are analyzed automatically into channel subfolders of that
root, so a channel folder is never nested inside another channel folder. Existing
outputs are never overwritten. A root-level ``spatial_analysis_outputs.csv``
indexes every generated graph.

Output layout::

    pair_correlation_<timestamp>/
      green_signal/
        preliminary_pass/
          pair_correlation_g_r.png
          pair_density_per_mm2.png
          pair_correlation_inhomogeneous_g_r.png
          estimated_intensity_surface.png
        preliminary_fail/
        all_outside_injection/
        manual_review/
      channel_2_signal/
        ... (same four statuses)
      spatial_analysis_outputs.csv

Example (PowerShell)::

    python scripts\\run_pair_correlation.py --config config.yml `
      --run-dir "C:\\mouse_brain_work\\candidates\\runs\\section070_20260706_151305" `
      --section 70 --out-dir "C:\\mouse_brain_work\\candidates\\spatial_analysis"
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import pair_correlation_analysis as pair_correlation

# The four spatial-analysis series (a subset of pair_correlation.SERIES). Order
# here is only cosmetic; each becomes a subfolder per channel when it has data.
SPATIAL_ANALYSIS_STATUSES = (
    "preliminary_pass",
    "preliminary_fail",
    "all_outside_injection",
    "manual_review",
)

CHANNELS = ("green_signal", "channel_2_signal")

# (graph_type, filename) written by pair_correlation for every analyzed status.
GRAPH_FILES = (
    ("pair_correlation_g_r", "pair_correlation_g_r.png"),
    ("pair_density_per_mm2", "pair_density_per_mm2.png"),
    ("pair_correlation_inhomogeneous_g_r", "pair_correlation_inhomogeneous_g_r.png"),
    ("estimated_intensity_surface", "estimated_intensity_surface.png"),
)

SPATIAL_OUTPUTS_COLUMNS = [
    "channel",
    "status",
    "candidate_count",
    "graph_type",
    "graph_path",
]


def build_output_root(out_dir: Path, timestamp: str | None = None) -> Path:
    """One fresh timestamped root under ``out_dir`` (never a channel folder)."""
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(out_dir) / f"pair_correlation_{stamp}"


def _assert_no_duplicated_channel_folder(root: Path, channel: str) -> None:
    """Guard against the ``green_signal/green_signal`` duplication class of bug."""
    nested = root / channel / channel
    if nested.exists():
        raise RuntimeError(f"Duplicated channel folder detected: {nested}")


def generate_pair_correlation(
    *,
    config,
    run_dir: Path,
    section: int,
    out_dir: Path,
    timestamp: str | None = None,
    statuses=SPATIAL_ANALYSIS_STATUSES,
    bin_width_um: float = pair_correlation.DEFAULT_BIN_WIDTH_UM,
    maximum_distance_um: float = pair_correlation.DEFAULT_MAXIMUM_DISTANCE_UM,
    simulations: int = pair_correlation.DEFAULT_SIMULATIONS,
    random_seed: int = pair_correlation.DEFAULT_RANDOM_SEED,
    intensity_bandwidth_um: float = pair_correlation.DEFAULT_INTENSITY_BANDWIDTH_UM,
) -> tuple[Path, list[dict]]:
    """Analyze both channels into one fresh timestamped root; index every graph.

    Returns ``(root, rows)`` where ``rows`` is the ``spatial_analysis_outputs``
    table. Refuses to overwrite an existing, non-empty root.
    """
    run_dir = Path(run_dir)
    root = build_output_root(out_dir, timestamp)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing pair-correlation outputs: {root}"
        )

    graph_rows: list[dict] = []
    analyzed_channels: list[str] = []
    for channel in CHANNELS:
        try:
            manifest = pair_correlation.run_analysis(
                config=config,
                run_dir=run_dir,
                channel=channel,
                section=section,
                out_dir=root,
                statuses=statuses,
                bin_width_um=bin_width_um,
                maximum_distance_um=maximum_distance_um,
                simulations=simulations,
                random_seed=random_seed,
                intensity_bandwidth_um=intensity_bandwidth_um,
            )
        except ValueError as exc:
            if "No candidates" in str(exc):
                print(f"  {channel}: no candidates for section {section}; skipped.")
                continue
            raise
        _assert_no_duplicated_channel_folder(root, channel)
        analyzed_channels.append(channel)

        for status, info in manifest["series"].items():
            directory = info.get("directory")
            if directory is None:  # status had too few candidates to analyze
                continue
            status_dir = Path(directory)
            count = info.get("number_of_candidates", "")
            for graph_type, filename in GRAPH_FILES:
                graph_path = status_dir / filename
                if graph_path.is_file():
                    graph_rows.append({
                        "channel": channel,
                        "status": status,
                        "candidate_count": count,
                        "graph_type": graph_type,
                        "graph_path": str(graph_path),
                    })

    if not analyzed_channels:
        raise ValueError(
            f"No candidates found for section {section} in either channel; "
            "nothing was generated."
        )

    outputs_csv = root / "spatial_analysis_outputs.csv"
    with outputs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SPATIAL_OUTPUTS_COLUMNS)
        writer.writeheader()
        writer.writerows(graph_rows)

    return root, graph_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Candidate-to-candidate pair correlation for BOTH channels "
                    "into one fresh timestamped folder.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--run-dir", required=True,
                        help="Run folder containing all_candidates.csv.")
    parser.add_argument("--section", required=True, type=int)
    parser.add_argument("--out-dir", required=True,
                        help="Parent folder; a fresh pair_correlation_<timestamp> "
                             "root is created inside it.")
    parser.add_argument("--bin-width-um", type=float,
                        default=pair_correlation.DEFAULT_BIN_WIDTH_UM)
    parser.add_argument("--maximum-distance-um", type=float,
                        default=pair_correlation.DEFAULT_MAXIMUM_DISTANCE_UM)
    parser.add_argument("--simulations", type=int,
                        default=pair_correlation.DEFAULT_SIMULATIONS)
    parser.add_argument("--random-seed", type=int,
                        default=pair_correlation.DEFAULT_RANDOM_SEED)
    parser.add_argument("--intensity-bandwidth-um", type=float,
                        default=pair_correlation.DEFAULT_INTENSITY_BANDWIDTH_UM)
    return parser


def main(argv: list[str] | None = None) -> int:
    from mouse_brain_pipeline.config import load_config

    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    root, graph_rows = generate_pair_correlation(
        config=config,
        run_dir=Path(args.run_dir),
        section=args.section,
        out_dir=Path(args.out_dir),
        bin_width_um=args.bin_width_um,
        maximum_distance_um=args.maximum_distance_um,
        simulations=args.simulations,
        random_seed=args.random_seed,
        intensity_bandwidth_um=args.intensity_bandwidth_um,
    )

    print("=" * 72)
    print(f"Pair-correlation root : {root}")
    print(f"Index CSV             : {root / 'spatial_analysis_outputs.csv'}")
    print("Generated graphs (exact folder for every graph):")
    for row in graph_rows:
        print(f"  {row['channel']}/{row['status']}  {row['graph_type']}")
        print(f"      {row['graph_path']}")
    if not graph_rows:
        print("  (no eligible status had >= 2 candidates)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
