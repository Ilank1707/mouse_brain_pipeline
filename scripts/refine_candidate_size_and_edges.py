#!/usr/bin/env python
"""Auditable post-detection refinement of candidate size and edge handling.

Reduces PROVISIONAL small-object detections and rescues valid edge candidates
without deleting anything, using measured connected-component size and explicit
edge handling against the original (non-eroded) tissue mask.

Candidates are NOT cells. Raw TIFFs, candidate coordinates, detection
thresholds, injection masks, candidate statuses, and existing run outputs are
read only. Everything is written into the supplied isolated ``--out-dir``.

Modes:
  --mode report  Diagnostics only; no status change, no threshold applied.
  --mode apply   Applies ONLY the thresholds explicitly supplied on the CLI;
                 refuses to run without at least one size threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  # adds project src/ to sys.path

from mouse_brain_pipeline.size_edge_refinement import (
    MODE_APPLY,
    MODE_REPORT,
    RefinementThresholds,
    refine_candidates,
    write_refinement_outputs,
)

REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "channel",
    "section",
    "x_local_px",
    "y_local_px",
    "current_status",
    "volume_um3",
    "xy_diameter_um",
    "support_plane_count",
    "measurement_valid",
    "invalid_coordinate",
    "inside_tissue",
}


def _read_candidates(path: Path, channel: str, section: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate table: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_CANDIDATE_COLUMNS - fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows = [
            row
            for row in reader
            if row.get("channel") == channel
            and str(row.get("section")) == str(section)
        ]
    if not rows:
        raise ValueError(f"No candidates for {channel}, section {section} in {path}")
    return rows


def _prepare_output_root(out_dir: Path, run_dir: Path, channel: str) -> Path:
    """Bind an output root to one run and reserve a fresh channel subfolder."""
    marker_path = out_dir / "size_edge_refinement_run.json"
    expected_run = str(run_dir.resolve())
    if out_dir.exists():
        entries = list(out_dir.iterdir())
        if entries and not marker_path.is_file():
            raise FileExistsError(
                f"--out-dir is not an isolated refinement folder: {out_dir}"
            )
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if str(Path(marker.get("run_dir", "")).resolve()) != expected_run:
                raise FileExistsError(
                    f"--out-dir belongs to another run: {marker.get('run_dir')}"
                )
    else:
        out_dir.mkdir(parents=True)
    marker_path.write_text(
        json.dumps(
            {"run_dir": expected_run, "analysis": "candidate size and edge refinement"},
            indent=2,
        ),
        encoding="utf-8",
    )
    channel_dir = out_dir / channel
    if channel_dir.exists() and any(channel_dir.iterdir()):
        raise FileExistsError(
            f"Channel output already exists; choose a fresh --out-dir: {channel_dir}"
        )
    channel_dir.mkdir(parents=True, exist_ok=True)
    return channel_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable candidate size + edge refinement (report/apply)."
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--channel", required=True, choices=("green_signal", "channel_2_signal")
    )
    parser.add_argument("--section", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", required=True, choices=(MODE_REPORT, MODE_APPLY))
    # Thresholds default to None so "not supplied" is distinguishable; apply mode
    # applies ONLY those explicitly provided here.
    parser.add_argument("--min-component-area-um2", type=float, default=None)
    parser.add_argument("--min-component-volume-um3", type=float, default=None)
    parser.add_argument("--min-support-planes", type=int, default=None)
    parser.add_argument("--edge-rescue-distance-um", type=float, default=None)
    return parser


def run_cli(args: argparse.Namespace) -> dict:
    import numpy as np  # noqa: PLC0415

    from mouse_brain_pipeline.config import load_config  # noqa: PLC0415

    config = load_config(args.config)
    voxel_zyx_um = (
        float(config.acquisition.voxel_size_z_um),
        float(config.acquisition.voxel_size_y_um),
        float(config.acquisition.voxel_size_x_um),
    )

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    thresholds = RefinementThresholds(
        min_component_area_um2=args.min_component_area_um2,
        min_component_volume_um3=args.min_component_volume_um3,
        min_support_planes=args.min_support_planes,
        edge_rescue_distance_um=args.edge_rescue_distance_um,
    )
    if args.mode == MODE_APPLY and not thresholds.has_any_size_threshold():
        raise SystemExit(
            "apply mode requires at least one explicit size threshold: "
            "--min-component-area-um2, --min-component-volume-um3, or "
            "--min-support-planes"
        )

    candidate_path = run_dir / "all_candidates.csv"
    rows = _read_candidates(candidate_path, args.channel, args.section)

    section_dir = run_dir / "qc" / f"{args.channel}_section_{args.section:03d}"
    tissue_path = section_dir / "tissue_mask.npy"
    if not tissue_path.is_file():
        raise FileNotFoundError(f"Missing saved tissue mask: {tissue_path}")
    tissue_mask = np.load(tissue_path, mmap_mode="r")
    if tissue_mask.ndim != 2:
        raise ValueError(f"Tissue mask must be 2D (y,x), got {tissue_mask.shape}")

    result = refine_candidates(
        rows,
        tissue_mask,
        voxel_zyx_um=voxel_zyx_um,
        mode=args.mode,
        thresholds=thresholds,
        channel=args.channel,
        section=args.section,
    )
    result.summary["run_dir"] = str(run_dir)
    result.summary["candidate_table"] = str(candidate_path)
    result.summary["tissue_mask"] = str(tissue_path)

    channel_dir = _prepare_output_root(out_dir, run_dir, args.channel)
    outputs = write_refinement_outputs(
        channel_dir,
        result,
        make_plots=True,
        plot_size_distributions=(
            config.postrun_spatial_analysis.enabled
            and config.postrun_spatial_analysis.candidate_size_distributions.enabled
        ),
    )
    result.summary["outputs"] = outputs
    (channel_dir / "refinement_summary.json").write_text(
        json.dumps(result.summary, indent=2), encoding="utf-8"
    )

    print(f"{args.channel}/{args.mode}: {len(rows)} candidates")
    print(f"Wrote refinement outputs to {channel_dir}")
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_cli(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
