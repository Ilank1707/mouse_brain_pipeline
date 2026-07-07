#!/usr/bin/env python
"""Calibrate the preliminary-pass rules from HUMAN LABELS (analysis only).

Reads the newest completed run's ``all_candidates.csv`` (for measured features)
and the reviewer's ``validation_review_batch.csv`` (for ``human_label``), joins
them on ``candidate_id``, and evaluates the existing configurable preliminary-pass
thresholds against the labels -- separately for green_signal and channel_2_signal.

It NEVER changes any candidate, status, mask, threshold or raw TIFF; it NEVER
targets a candidate count and NEVER uses pair-correlation g(r). The only objective
is agreement with the human labels. A preliminary-rule pass is a PROVISIONAL
candidate, not a confirmed cell.

Outputs (into ``--out``):
  calibration_results.csv           every evaluated parameter set + P/R/F1/FP/FN
  calibration_pareto_front.csv      non-dominated precision-vs-recall settings
  confusion_matrix_by_channel.csv   baseline confusion, per channel + label
  false_positive_examples.csv       baseline passes labelled artefact
  false_negative_examples.csv       baseline fails labelled cell
  calibration_summary.json          provenance + baseline metrics + guarantees
  precision_recall_tradeoff.png     P/R scatter + Pareto front per channel

Example (PowerShell):
  python scripts/calibrate_candidate_rules.py --config config.yml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.candidate_detection import params_from_config
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.review import read_csv_rows
from mouse_brain_pipeline.rule_calibration import (
    CHANNELS,
    CONFUSION_COLUMNS,
    EXAMPLE_COLUMNS,
    RESULT_COLUMNS,
    VALID_LABELS,
    DupPool,
    calibrate_channel,
    coerce_rec,
    enforce_edge_policy,
    normalize_label,
    predicted_pass,
)


def find_newest_run(runs_root: Path) -> Path | None:
    matches = list(runs_root.glob("*/all_candidates.csv"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime).parent


def _write_csv(path, columns, rows, extra_columns=None):
    cols = list(columns) + list(extra_columns or [])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def _build_dup_pool(channel_rows, base_params):
    """NMS neighbour pool = channel candidates that pass baseline morphology."""
    pool = []
    for row in channel_rows:
        rec = coerce_rec(row)
        if predicted_pass(rec, base_params, dup_pool=None):
            pool.append(rec)
    return DupPool(pool)


def _plot_tradeoff(out_path, per_channel):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, len(per_channel), figsize=(7 * len(per_channel), 6),
                             squeeze=False)
    for ax, result in zip(axes[0], per_channel):
        results = result["results"]
        rec = [(r["recall"], r["precision"]) for r in results if r["tp"] > 0]
        if rec:
            xs, ys = zip(*rec)
            ax.scatter(xs, ys, s=14, c="#BBBBBB", alpha=0.6, label="evaluated settings")
        front = result["pareto"]
        if front:
            fx = [p["recall"] for p in front]
            fy = [p["precision"] for p in front]
            ax.plot(fx, fy, "-o", color="#C62828", lw=1.4, ms=5, label="Pareto front")
        base = result["baseline"]
        ax.scatter([base["recall"]], [base["precision"]], marker="*", s=220,
                   c="#1F77B4", edgecolors="black", zorder=5, label="current (baseline)")
        ax.set_xlabel("recall (labelled cells retained)")
        ax.set_ylabel("precision (retained that are cells)")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{result['channel']}  (PROVISIONAL candidates)\n"
                     f"cells={result['label_counts'].get('cell', 0)} "
                     f"artefacts={result['label_counts'].get('artefact', 0)}",
                     fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Preliminary-rule precision/recall vs human labels -- no count "
                 "targeted, g(r) not used. Choose a setting yourself.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Human-label calibration of the preliminary-pass rules "
                    "(analysis only; changes nothing).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--candidates", default=None,
                   help="all_candidates.csv (default: newest run under the runs root).")
    p.add_argument("--batch", default=None,
                   help="validation_review_batch.csv (default: "
                        "<run>/preliminary_validation/validation_review_batch.csv).")
    p.add_argument("--runs-root", default=None,
                   help="Runs root (default: <work_dir>/candidates/runs).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <run>/rule_calibration).")
    p.add_argument("--no-duplicate-nms", action="store_true",
                   help="Skip duplicate-distance (NMS) evaluation (faster).")
    args = p.parse_args()

    config = load_config(args.config)
    runs_root = Path(args.runs_root or (config.work_dir / "candidates" / "runs"))

    if args.candidates:
        candidates_csv = Path(args.candidates)
        run_dir = candidates_csv.parent
    else:
        run_dir = find_newest_run(runs_root)
        if run_dir is None:
            print(f"ERROR: no all_candidates.csv found under {runs_root}")
            return 1
        candidates_csv = run_dir / "all_candidates.csv"

    batch_csv = Path(args.batch or (run_dir / "preliminary_validation"
                                    / "validation_review_batch.csv"))
    out_dir = Path(args.out or (run_dir / "rule_calibration"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run directory : {run_dir}")
    print(f"Candidates CSV: {candidates_csv}")
    print(f"Label batch   : {batch_csv}")
    print(f"Output        : {out_dir}")
    print("Analysis only -- no candidate, status, mask or threshold is changed.")
    print("=" * 72)

    if not candidates_csv.is_file():
        print(f"ERROR: candidates CSV not found: {candidates_csv}")
        return 1
    if not batch_csv.is_file():
        print(f"ERROR: label batch not found: {batch_csv}")
        return 1

    all_rows = read_csv_rows(candidates_csv)
    batch_rows = read_csv_rows(batch_csv)

    # human_label is required: cell / artefact / uncertain / injection.
    labels_by_id = {}
    unknown = 0
    for row in batch_rows:
        label = normalize_label(row.get("human_label"))
        if label in VALID_LABELS:
            labels_by_id[(row.get("candidate_id"), row.get("channel"))] = row
        elif str(row.get("human_label") or "").strip():
            unknown += 1
    if not labels_by_id:
        print("ERROR: no usable human_label values in the batch. Fill the "
              "'human_label' column with one of: cell, artefact, uncertain, "
              "injection, then re-run. (This tool does not label for you.)")
        return 2
    if unknown:
        print(f"WARNING: {unknown} rows had a non-blank but unrecognised human_label "
              "(ignored).")

    # Join: full measured features from all_candidates + the human label + patch ref.
    features_by_id = {(r.get("candidate_id"), r.get("channel")): r for r in all_rows}
    labeled_rows_by_channel = {ch: [] for ch in CHANNELS}
    missing = 0
    for key, batch_row in labels_by_id.items():
        feat = features_by_id.get(key)
        channel = key[1]
        if channel not in labeled_rows_by_channel:
            continue
        merged = dict(feat) if feat else dict(batch_row)
        if feat is None:
            missing += 1
        merged["human_label"] = batch_row.get("human_label")
        merged["review_patch_file"] = batch_row.get("review_patch_file", "")
        labeled_rows_by_channel[channel].append(merged)
    if missing:
        print(f"WARNING: {missing} labelled candidates were not found in "
              "all_candidates.csv; their batch-row fields were used instead.")

    base_params = params_from_config(config)

    per_channel = []
    all_results, all_pareto, all_confusion = [], [], []
    all_fp, all_fn = [], []
    for channel in CHANNELS:
        rows = labeled_rows_by_channel[channel]
        if not rows:
            print(f"  {channel}: no labelled candidates; skipped.")
            continue
        channel_base = enforce_edge_policy(base_params.for_channel(channel))
        dup_pool = None
        if not args.no_duplicate_nms:
            channel_all = [r for r in all_rows if r.get("channel") == channel]
            dup_pool = _build_dup_pool(channel_all, channel_base)
        result = calibrate_channel(channel, rows, channel_base, dup_pool=dup_pool)
        per_channel.append(result)
        all_results.extend(result["results"])
        for p in result["pareto"]:
            all_pareto.append(p)
        all_confusion.extend(result["confusion"])
        all_fp.extend(result["false_positives"])
        all_fn.extend(result["false_negatives"])
        b = result["baseline"]
        print(f"  {channel:16s} labelled={result['n_labeled']:4d} "
              f"cells={result['label_counts'].get('cell', 0)} "
              f"artefacts={result['label_counts'].get('artefact', 0)}  "
              f"baseline P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}  "
              f"pareto_points={len(result['pareto'])}")

    if not per_channel:
        print("ERROR: no channel had labelled candidates.")
        return 2

    _write_csv(out_dir / "calibration_results.csv", RESULT_COLUMNS, all_results)
    _write_csv(out_dir / "calibration_pareto_front.csv", RESULT_COLUMNS, all_pareto,
               extra_columns=["pareto_role"])
    _write_csv(out_dir / "confusion_matrix_by_channel.csv", CONFUSION_COLUMNS, all_confusion)
    _write_csv(out_dir / "false_positive_examples.csv", EXAMPLE_COLUMNS, all_fp)
    _write_csv(out_dir / "false_negative_examples.csv", EXAMPLE_COLUMNS, all_fn)
    _plot_tradeoff(out_dir / "precision_recall_tradeoff.png", per_channel)

    summary = {
        "analysis": "human-label calibration of preliminary-pass rules "
                    "(PROVISIONAL candidates; NOT cells)",
        "run_directory": str(run_dir),
        "candidates_csv": str(candidates_csv),
        "label_batch_csv": str(batch_csv),
        "voxel_size_zyx_um": list(config.acquisition.voxel_size_zyx),
        "label_mapping": {
            "positive": sorted(["cell"]),
            "negative": sorted(["artefact"]),
            "excluded_from_precision_recall": sorted(["uncertain", "injection"]),
        },
        "duplicate_distance_evaluated": not args.no_duplicate_nms,
        "guarantees": [
            "no candidate, status, mask, threshold or raw TIFF was modified",
            "no candidate count was targeted or optimised",
            "pair-correlation g(r) was not used and was not an objective",
            "green_signal and channel_2_signal were calibrated separately",
            "candidates are never rejected for being near the edge alone "
            "(keep_edge_clipped_if_center_in_tissue enforced True)",
            "the Pareto front is reported; no single setting is auto-selected",
        ],
        "thresholds_evaluated": [
            "component area (min_component_xy_area_um2)",
            "component volume (min_component_volume_um3)",
            "support-plane count (min_support_planes)",
            "support voxels (min_supporting_voxels)",
            "signal-to-background ratio (min_signal_to_background_ratio / robust z)",
            "diameter range (min_diameter_um, max_diameter_um)",
            "elongation (max_elongation)",
            "duplicate distance (min_separation_um)",
        ],
        "channels": {
            r["channel"]: {
                "n_labeled": r["n_labeled"],
                "label_counts": r["label_counts"],
                "baseline_precision": r["baseline"]["precision"],
                "baseline_recall": r["baseline"]["recall"],
                "baseline_f1": r["baseline"]["f1"],
                "baseline_false_positive_count": r["baseline"]["false_positive_count"],
                "baseline_false_negative_count": r["baseline"]["false_negative_count"],
                "baseline_n_retained": r["baseline"]["n_retained"],
                "n_parameter_sets_evaluated": len(r["results"]),
                "n_pareto_points": len(r["pareto"]),
                "pareto_front": [
                    {"pareto_role": p.get("pareto_role", ""),
                     "precision": p["precision"], "recall": p["recall"], "f1": p["f1"],
                     "n_retained": p["n_retained"],
                     "thresholds": {k: p[k] for k in (
                         "min_component_xy_area_um2", "min_component_volume_um3",
                         "min_support_planes", "min_supporting_voxels",
                         "min_signal_to_background_ratio", "min_local_robust_z",
                         "min_diameter_um", "max_diameter_um", "max_elongation",
                         "duplicate_distance_um")}}
                    for p in r["pareto"]
                ],
            }
            for r in per_channel
        },
    }
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("=" * 72)
    print(f"Wrote calibration artifacts to {out_dir}")
    print("Pareto front reported for each channel -- pick a stricter (high "
          "precision) or higher-recall setting yourself; nothing was auto-chosen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
