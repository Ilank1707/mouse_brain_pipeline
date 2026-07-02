#!/usr/bin/env python
"""Evaluate Cellfinder candidate-generation recall against manual reference points."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.reference_audit import (
    REFERENCE_COLUMNS,
    evaluate_recall_by_source,
    match_reference_points,
    read_reference_points,
)
from mouse_brain_pipeline.review import read_csv_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate candidate-generation recall.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--references", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--xy-tolerance-um", type=float, default=None)
    parser.add_argument("--z-tolerance-um", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    references_path = Path(
        args.references
        or config.work_dir / "candidates" / "manual_reference_points.csv"
    )
    candidates_path = Path(
        args.candidates or config.work_dir / "candidates" / "all_candidates.csv"
    )
    output_dir = Path(
        args.output_dir or config.work_dir / "candidates" / "candidate_recall_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    references = read_reference_points(references_path)
    candidates = [
        row for row in read_csv_rows(candidates_path)
        if str(row.get("candidate_exists", "True")).lower() in {"true", "1", "yes"}
    ]
    xy_tolerance = (
        args.xy_tolerance_um
        if args.xy_tolerance_um is not None
        else config.candidate_recall.xy_tolerance_um
    )
    z_tolerance = (
        args.z_tolerance_um
        if args.z_tolerance_um is not None
        else config.candidate_recall.z_tolerance_um
    )
    matches, unmatched_references, unmatched_candidates = match_reference_points(
        references,
        candidates,
        voxel_size_y_um=config.acquisition.voxel_size_y_um,
        voxel_size_x_um=config.acquisition.voxel_size_x_um,
        voxel_size_z_um=config.acquisition.voxel_size_z_um,
        xy_tolerance_um=xy_tolerance,
        z_tolerance_um=z_tolerance,
    )

    matched_columns = REFERENCE_COLUMNS + [
        "matched_candidate_id",
        "candidate_x_global_px",
        "candidate_y_global_px",
        "candidate_cellfinder_z_index",
        "xy_distance_um",
        "z_distance_um",
    ]
    with open(output_dir / "matched_reference_points.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=matched_columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in matched_columns} for row in matches)
    with open(output_dir / "unmatched_reference_points.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in REFERENCE_COLUMNS}
            for row in unmatched_references
        )

    by_source = evaluate_recall_by_source(
        references,
        candidates,
        voxel_size_y_um=config.acquisition.voxel_size_y_um,
        voxel_size_x_um=config.acquisition.voxel_size_x_um,
        voxel_size_z_um=config.acquisition.voxel_size_z_um,
        xy_tolerance_um=xy_tolerance,
        z_tolerance_um=z_tolerance,
    )
    (output_dir / "candidate_recall_summary.json").write_text(
        json.dumps(by_source, indent=2), encoding="utf-8"
    )
    print(f"Manual references: {len(references)}")
    print(f"Matching tolerance: XY={xy_tolerance} um, Z={z_tolerance} um")
    if not by_source.get("has_references"):
        print("Candidate-generation recall: not reported (no manual references).")
        return 1
    for name in ("raw_pass", "suppressed_pass", "union"):
        stats = by_source["by_source"][name]
        print(
            f"{name:16s} recall: {stats['matched_references']}/{len(references)} "
            f"= {stats['recall']:.3f}  "
            f"(candidates considered={stats['candidates_considered']})"
        )
    print("Unmatched Cellfinder candidates were not labelled false positives.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
