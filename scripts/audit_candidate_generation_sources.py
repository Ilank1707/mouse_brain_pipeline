#!/usr/bin/env python
"""Read-only audit of how candidates were generated (Task 4).

Reports, per channel, how many candidates came from the raw Cellfinder pass, the
injection-suppressed pass, or both -- broken down by injection-core/analysis-mask
membership, status and peak optical plane. It also reports the fraction of
OUTSIDE-mask candidates found by both passes and warns (never errors) when that
fraction is below 10%: an intense injection site can shift Cellfinder's
global/tiled thresholding, so suppression-only outside candidates need manual
recall + precision validation.

This script ONLY reads a completed run's all_candidates.csv and writes new files;
it never changes any candidate, status, mask, threshold or raw TIFF, and never
targets a count. Outputs are PROVISIONAL candidates, never final cell counts.

Outputs (under ``--out-dir``):
  candidate_generation_source_audit.csv    counts by the axes above
  candidate_generation_source_summary.json per-channel fractions + warnings
  candidate_generation_source_audit.png    per-channel source + outside-mask chart

Example (PowerShell):
  python scripts/audit_candidate_generation_sources.py `
    --run-dir "C:/mouse_brain_work/candidates/runs/section070_20260706_151305"
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.generation_source_audit import (
    AUDIT_COLUMNS,
    DEFAULT_BOTH_FRACTION_THRESHOLD,
    SOURCE_BOTH,
    SOURCE_RAW,
    SOURCE_SUPPRESSED,
    audit_rows,
    normalize_source,
    outside_mask_candidates,
    source_fractions,
    summarize,
)
from mouse_brain_pipeline.review import read_csv_rows


def _resolve_inputs(args):
    if args.candidates:
        candidates_csv = Path(args.candidates)
        base = candidates_csv.parent
    elif args.run_dir:
        base = Path(args.run_dir)
        candidates_csv = base / "all_candidates.csv"
    else:
        return None, None
    return candidates_csv, base


def _planes_per_section(config_path) -> int:
    if not config_path:
        return 7
    try:
        from mouse_brain_pipeline.config import load_config  # noqa: PLC0415

        return int(load_config(config_path).acquisition.planes_per_section)
    except Exception:  # pragma: no cover - config optional
        return 7


def _plot(out_path, candidates, threshold):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    channels = sorted({c.get("channel", "") for c in candidates}) or [""]
    sources = [SOURCE_RAW, SOURCE_SUPPRESSED, SOURCE_BOTH]
    colours = {SOURCE_RAW: "#1F77B4", SOURCE_SUPPRESSED: "#00D9FF", SOURCE_BOTH: "#2CA02C"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    width = 0.25
    x = np.arange(len(channels))
    for i, source in enumerate(sources):
        counts = [sum(1 for c in candidates
                      if c.get("channel", "") == ch and normalize_source(c) == source)
                  for ch in channels]
        axes[0].bar(x + (i - 1) * width, counts, width, label=source, color=colours[source])
    axes[0].set_xticks(x); axes[0].set_xticklabels(channels, fontsize=8)
    axes[0].set_ylabel("candidate count")
    axes[0].set_title("Candidate-generation source by channel\n(PROVISIONAL candidates)",
                      fontsize=9)
    axes[0].legend(fontsize=8)

    outside_both = []
    for ch in channels:
        outside = outside_mask_candidates(
            [c for c in candidates if c.get("channel", "") == ch])
        outside_both.append(source_fractions(outside)["fraction_both"])
    bars = axes[1].bar(x, outside_both, 0.5, color="#2CA02C")
    axes[1].axhline(threshold, color="#C62828", ls="--", lw=1.4,
                    label=f"warn < {threshold:.0%}")
    axes[1].set_xticks(x); axes[1].set_xticklabels(channels, fontsize=8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("fraction of outside-mask candidates found by BOTH passes")
    axes[1].set_title("Outside-mask 'both' fraction\n(low => suppression sensitive)",
                      fontsize=9)
    for rect, value in zip(bars, outside_both):
        axes[1].text(rect.get_x() + rect.get_width() / 2, value + 0.02,
                     f"{value:.0%}", ha="center", fontsize=8)
    axes[1].legend(fontsize=8)

    fig.suptitle("Candidate-generation source audit -- read-only; no thresholds "
                 "changed; green/red separate", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only candidate-generation source audit "
                    "(PROVISIONAL candidates; changes nothing).")
    p.add_argument("--config", "-c", default=None,
                   help="Optional config.yml (only for planes_per_section).")
    p.add_argument("--run-dir", default=None,
                   help="Completed run folder containing all_candidates.csv.")
    p.add_argument("--candidates", default=None,
                   help="Explicit all_candidates.csv (overrides --run-dir).")
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: <run-dir>/generation_source_audit).")
    p.add_argument("--both-fraction-threshold", type=float,
                   default=DEFAULT_BOTH_FRACTION_THRESHOLD,
                   help="Warn if the outside-mask 'both' fraction is below this "
                        "(default 0.10). Warning only -- never changes thresholds.")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG chart.")
    args = p.parse_args(argv)

    candidates_csv, base = _resolve_inputs(args)
    if candidates_csv is None:
        print("ERROR: provide --run-dir or --candidates.")
        return 2
    if not candidates_csv.is_file():
        print(f"ERROR: all_candidates.csv not found: {candidates_csv}")
        return 2

    candidates = read_csv_rows(candidates_csv)
    if not candidates:
        print(f"ERROR: no candidates in {candidates_csv}")
        return 2

    planes = _planes_per_section(args.config)
    out_dir = Path(args.out_dir) if args.out_dir else base / "generation_source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Candidates : {candidates_csv} ({len(candidates)} rows)")
    print(f"Output     : {out_dir}")
    print("Read-only audit -- no candidate, status, mask, threshold or TIFF is changed.")
    print("=" * 72)

    rows = audit_rows(candidates, planes_per_section=planes)
    audit_csv = out_dir / "candidate_generation_source_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(candidates, planes_per_section=planes,
                        threshold=args.both_fraction_threshold)
    (out_dir / "candidate_generation_source_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_plot:
        try:
            _plot(out_dir / "candidate_generation_source_audit.png", candidates,
                  args.both_fraction_threshold)
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            print(f"WARNING: could not render the chart: {exc}")

    for channel in summary["channels"]:
        chan = summary["by_channel"][channel]
        outside = chan["outside_analysis_mask_source_fractions"]
        print(f"  {channel:16s} total={chan['n']:5d}  outside-mask={outside['n']:5d}  "
              f"outside 'both'={outside['fraction_both']:.1%}  "
              f"raw-only={outside['fraction_raw_stack_only']:.1%}  "
              f"suppressed-only={outside['fraction_injection_suppressed_stack_only']:.1%}")

    warnings = summary["suppression_sensitivity"]["warnings"]
    if warnings:
        print("!" * 72)
        for warning in warnings:
            print("WARNING: " + warning["message"])
        print("!" * 72)
    else:
        print("No suppression-sensitivity warning "
              f"(outside-mask 'both' fraction >= {args.both_fraction_threshold:.0%}).")

    print(f"Wrote audit -> {audit_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
