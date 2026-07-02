#!/usr/bin/env python
"""Export object-level and region-level CSVs (and optional provisional overlap).

Works on pilot candidate CSVs now; atlas region columns populate once a real
Brainmapper run (with a background channel) has produced atlas assignments.

Overlap matching is DISABLED by default. It produces PROVISIONAL spatial
classifications only -- never "double-positive".

Examples:
  python scripts/summarize_results.py --config config.yml \
      --candidates work/candidates/classified_candidates_green_signal.csv
  python scripts/summarize_results.py --config config.yml \
      --candidates work/candidates/classified_candidates_green_signal.csv --overlap
"""

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.overlap import match_one_to_one, write_overlaps
from mouse_brain_pipeline.summarize import (
    aggregate_regions,
    candidate_to_object_rows,
    read_candidate_csv,
    write_object_csv,
    write_region_csv,
)
from mouse_brain_pipeline.utilities import ensure_dir, setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise detections into object/region CSVs.")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--candidates", required=True,
                   help="Path to a manually reviewed or validated classified candidate CSV")
    p.add_argument("--overlap", action="store_true",
                   help="Enable PROVISIONAL spatial overlap matching (disabled by default)")
    p.add_argument("--overlap-distance-um", type=float, default=None,
                   help="Override detection.overlap_distance_um")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    cfg = load_config(args.config)
    cand_path = Path(args.candidates)
    if not cand_path.is_file():
        print(f"Candidate CSV not found: {cand_path}")
        return 1

    candidates = read_candidate_csv(cand_path)
    out_dir = ensure_dir(cfg.work_dir / "summary")

    obj_rows = candidate_to_object_rows(candidates, cfg)
    obj_path = write_object_csv(out_dir, obj_rows)
    region_rows = aggregate_regions(obj_rows)
    region_path = write_region_csv(out_dir, region_rows)

    print("=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"objects : {len(obj_rows)}  -> {obj_path}")
    print(f"regions : {len(region_rows)} -> {region_path}")
    if not any(r.get("atlas_region_id") for r in obj_rows):
        print("NOTE: atlas_region_* columns are blank -- no registration/atlas assignment yet.")
        print("      Region rows are grouped under 'unassigned'.")

    if args.overlap:
        tol = args.overlap_distance_um or cfg.detection.overlap_distance_um
        green = [c for c in candidates if c.get("channel") == "green_signal"]
        ch2 = [c for c in candidates if c.get("channel") == "channel_2_signal"]
        overlaps = match_one_to_one(green, ch2, tolerance_um=tol)
        ov_path = write_overlaps(out_dir, overlaps)
        print("-" * 70)
        print(f"PROVISIONAL spatial overlaps (tol {tol} um): {len(overlaps)} -> {ov_path}")
        print("WARNING: these are provisional spatial classifications, NOT double-positive cells.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
