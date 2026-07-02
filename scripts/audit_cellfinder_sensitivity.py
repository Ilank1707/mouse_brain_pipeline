#!/usr/bin/env python
"""Run a small, EXPLICIT Cellfinder parameter grid on one section and report.

For each grid point this reports total candidates, outside-core candidates,
outside-analysis-mask candidates, reference recall (only when manual references
exist), candidate density by spatial tile and runtime. It NEVER declares a
"best" setting unless genuine manual reference points exist -- and even then the
extra candidates must still be reviewed for artefacts by a human.

Outputs are PROVISIONAL CANDIDATE detections, never final cell counts.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline import CHANNEL_2_SIGNAL, GREEN_SIGNAL
from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.candidate_detection import (
    detect_candidates_in_stack,
    params_from_config,
    read_crop_stack,
)
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.reference_audit import (
    evaluate_recall_by_source,
    read_reference_points,
)

# Explicit, hand-listed grid (NOT an automatic optimiser). Edit deliberately.
PARAMETER_GRID = [
    {"n_sds_above_mean_thresh": 8, "n_sds_above_mean_tiled_thresh": 8},
    {"n_sds_above_mean_thresh": 10, "n_sds_above_mean_tiled_thresh": 10},
    {"n_sds_above_mean_thresh": 12, "n_sds_above_mean_tiled_thresh": 12},
]

REPORT_COLUMNS = [
    "grid_index",
    "channel",
    "section",
    "n_sds_above_mean_thresh",
    "n_sds_above_mean_tiled_thresh",
    "tiled_thresh_tile_size",
    "soma_diameter_um",
    "total_candidates",
    "outside_core",
    "outside_analysis_mask",
    "raw_pass_recall",
    "suppressed_pass_recall",
    "union_recall",
    "runtime_seconds",
]


def _tile_density(candidates, tile_px) -> dict:
    density: dict[str, int] = {}
    for c in candidates:
        tx = int(c["x_global_px"]) // tile_px
        ty = int(c["y_global_px"]) // tile_px
        density[f"{tx}:{ty}"] = density.get(f"{tx}:{ty}", 0) + 1
    return density


def main() -> int:
    parser = argparse.ArgumentParser(description="Cellfinder parameter sensitivity audit.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--channel", required=True,
                        choices=[GREEN_SIGNAL, CHANNEL_2_SIGNAL])
    parser.add_argument("--section", type=int, default=None)
    parser.add_argument("--crop", type=int, nargs=4,
                        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"), default=None)
    parser.add_argument("--tile-px", type=int, default=1024)
    parser.add_argument("--references", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    params = params_from_config(config)
    if params.backend != "cellfinder_candidates":
        print("This audit requires detection.backend == 'cellfinder_candidates'.")
        return 2
    section = args.section if args.section is not None else config.pilot.first_section
    first = config.pilot.first_section if config.pilot.first_section is not None else section
    directory = (
        config.data.green_signal_dir if args.channel == GREEN_SIGNAL
        else config.data.channel_2_signal_dir
    )
    index = index_channel(args.channel, directory, config.data.filename_regex)
    plane_paths = {pl: path for (s, pl), path in index.files.items() if s == section}
    if not plane_paths:
        print(f"ERROR: no planes for {args.channel} section {section}.")
        return 2

    crop = tuple(args.crop) if args.crop else None
    stack, plane_numbers, origin, _ = read_crop_stack(plane_paths, crop)
    voxel = config.acquisition.voxel_size_zyx
    injection_cfg = params.injection.for_channel(args.channel)
    cellfinder_cfg = params.cellfinder.for_channel(args.channel)

    references = [
        r for r in read_reference_points(
            args.references
            or config.work_dir / "candidates" / "manual_reference_points.csv"
        )
        if r.get("channel") == args.channel and str(r.get("section")) == str(section)
    ]

    output_dir = Path(
        args.output_dir or config.work_dir / "candidates" / "cellfinder_sensitivity_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    density_report = {}

    for grid_index, overrides in enumerate(PARAMETER_GRID):
        grid_cfg = dataclasses.replace(cellfinder_cfg, **overrides)
        start = time.time()
        result = detect_candidates_in_stack(
            stack, params, voxel, channel=args.channel, section=section,
            first_section=first, planes_per_section=config.acquisition.planes_per_section,
            plane_numbers=plane_numbers, crop_origin=origin,
            injection_cfg=injection_cfg, backend=params.backend,
            cellfinder_cfg=grid_cfg,
        )
        runtime = time.time() - start
        candidates = result.candidates
        outside_core = sum(1 for c in candidates if not c.get("inside_injection_core"))
        outside_analysis = sum(
            1 for c in candidates if not c.get("inside_injection_analysis_exclusion")
        )
        recall = evaluate_recall_by_source(
            references, candidates,
            voxel_size_y_um=config.acquisition.voxel_size_y_um,
            voxel_size_x_um=config.acquisition.voxel_size_x_um,
            voxel_size_z_um=config.acquisition.voxel_size_z_um,
            xy_tolerance_um=config.candidate_recall.xy_tolerance_um,
            z_tolerance_um=config.candidate_recall.z_tolerance_um,
        )
        by_source = recall.get("by_source", {})
        rows.append({
            "grid_index": grid_index,
            "channel": args.channel,
            "section": section,
            "n_sds_above_mean_thresh": grid_cfg.n_sds_above_mean_thresh,
            "n_sds_above_mean_tiled_thresh": grid_cfg.n_sds_above_mean_tiled_thresh,
            "tiled_thresh_tile_size": grid_cfg.tiled_thresh_tile_size,
            "soma_diameter_um": grid_cfg.soma_diameter_um,
            "total_candidates": len(candidates),
            "outside_core": outside_core,
            "outside_analysis_mask": outside_analysis,
            "raw_pass_recall": by_source.get("raw_pass", {}).get("recall", ""),
            "suppressed_pass_recall": by_source.get("suppressed_pass", {}).get("recall", ""),
            "union_recall": by_source.get("union", {}).get("recall", ""),
            "runtime_seconds": round(runtime, 2),
        })
        density_report[str(grid_index)] = _tile_density(candidates, args.tile_px)
        print(
            f"[grid {grid_index}] {overrides} -> total={len(candidates)} "
            f"outside_core={outside_core} outside_analysis={outside_analysis} "
            f"runtime={runtime:.1f}s"
        )

    with open(output_dir / "sensitivity_report.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "candidate_density_by_tile.json").write_text(
        json.dumps(density_report, indent=2), encoding="utf-8"
    )

    print("-" * 70)
    if not references:
        print("NO manual reference points for this channel/section: no 'best' setting "
              "is declared. Annotate references first (annotate_reference_cells.py).")
    else:
        print(f"Manual references available: {len(references)}. Compare recall columns, "
              "but still review the EXTRA candidates for artefacts before choosing.")
    print(f"Report: {output_dir / 'sensitivity_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
