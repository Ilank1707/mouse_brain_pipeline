#!/usr/bin/env python
"""Calibrate the preliminary-pass rules from HUMAN LABELS (analysis only).

Reads a completed ``validation_review_batch.csv`` (the reviewer's ``human_label``
plus every measured feature) and evaluates the existing configurable
preliminary-pass thresholds against the labels -- separately for green_signal and
channel_2_signal. If the batch is missing any feature column and the run's
``all_candidates.csv`` is available, the missing columns are enriched from it.

It NEVER changes any candidate, status, mask, threshold or raw TIFF; it NEVER
targets a candidate count; it NEVER uses pair-correlation g(r); and it NEVER edits
config.yml. It only proposes threshold options for a human to review. A
preliminary-rule pass is a PROVISIONAL candidate, not a confirmed cell.

Outputs (into ``--out``):
  calibration_results.csv           every evaluated parameter set + P/R/F1/FP/FN
  calibration_pareto_front.csv      non-dominated precision-vs-recall settings
  confusion_matrix_by_channel.csv   baseline confusion, per channel + label
  false_positive_examples.csv       baseline passes labelled artefact
  false_negative_examples.csv       baseline fails labelled cell
  calibration_summary.json          provenance + baseline metrics + guarantees
  precision_recall_tradeoff.png     P/R scatter + Pareto front per channel
  proposed_config_changes.yml       REVIEW-ONLY high-precision / high-recall snippet

Example (PowerShell):
  python scripts/calibrate_candidate_rules.py --config config.yml `
    --run-dir "C:/mouse_brain_work/candidates/runs/section070_20260706_151305" `
    --batch  "C:/mouse_brain_work/candidates/validation_070/validation_review_batch.csv" `
    --out    "C:/mouse_brain_work/candidates/validation_070/calibration"
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

# Map a calibration threshold to its config.yml key. The screening subset is
# per-channel (green overrides via detection.green_signal); the rest are global.
THRESHOLD_TO_CONFIG_KEY = {
    "min_component_xy_area_um2": "minimum_component_xy_area_um2",
    "min_component_volume_um3": "minimum_component_volume_um3",
    "min_supporting_voxels": "minimum_supporting_voxels",
    "min_support_planes": "minimum_support_planes",
    "min_signal_to_background_ratio": "minimum_signal_to_background_ratio",
    "min_local_robust_z": "minimum_local_robust_z",
    "min_diameter_um": "minimum_cell_diameter_um",
    "max_diameter_um": "maximum_cell_diameter_um",
    "max_elongation": "maximum_elongation",
    "duplicate_distance_um": "minimum_candidate_separation_um",
}
PER_CHANNEL_SCREENING_KEYS = {
    "minimum_component_xy_area_um2",
    "minimum_component_volume_um3",
    "minimum_supporting_voxels",
    "minimum_support_planes",
    "minimum_signal_to_background_ratio",
}


def find_newest_run(runs_root: Path):
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
    pool = []
    for row in channel_rows:
        rec = coerce_rec(row)
        if predicted_pass(rec, base_params, dup_pool=None):
            pool.append(rec)
    return DupPool(pool)


def _metrics_block(point):
    return {
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "false_positive_count": point["false_positive_count"],
        "false_negative_count": point["false_negative_count"],
        "n_retained": point["n_retained"],
    }


def _option_block(point, channel):
    """Threshold values for one Pareto option, split by config scope."""
    per_channel, global_shared = {}, {}
    for threshold, config_key in THRESHOLD_TO_CONFIG_KEY.items():
        value = point[threshold]
        if channel == "green_signal" and config_key in PER_CHANNEL_SCREENING_KEYS:
            per_channel[config_key] = value
        else:
            global_shared[config_key] = value
    block = {"metrics": _metrics_block(point)}
    if per_channel:
        block["detection.green_signal (per-channel override)"] = per_channel
    block["detection (global, shared by both channels)"] = global_shared
    return block


def _find_role(pareto, role):
    for point in pareto:
        if role in str(point.get("pareto_role", "")):
            return point
    return None


def _write_proposed_config(path, per_channel):
    """REVIEW-ONLY YAML: high-precision and high-recall options per channel.

    This is never applied and config.yml is never edited. A human reconciles the
    global (shared) keys before changing anything.
    """
    import yaml  # noqa: PLC0415

    proposed = {}
    for result in per_channel:
        channel = result["channel"]
        entry = {"baseline_metrics": _metrics_block(result["baseline"])}
        hp = _find_role(result["pareto"], "high_precision")
        hr = _find_role(result["pareto"], "high_recall")
        if hp is not None:
            entry["high_precision_option"] = _option_block(hp, channel)
        if hr is not None:
            entry["high_recall_option"] = _option_block(hr, channel)
        proposed[channel] = entry

    header = (
        "# PROPOSED preliminary-pass thresholds -- REVIEW ONLY. NOT APPLIED.\n"
        "# Generated by calibrate_candidate_rules.py from human labels. Nothing\n"
        "# here edits config.yml. No candidate count was targeted and\n"
        "# pair-correlation g(r) was not used. Two options per channel from the\n"
        "# precision/recall Pareto front:\n"
        "#   high_precision_option -> fewer false positives (stricter)\n"
        "#   high_recall_option    -> fewer false negatives (looser)\n"
        "# Green screening keys map under detection.green_signal; the 'global,\n"
        "# shared' keys live under detection: and affect BOTH channels -- reconcile\n"
        "# green and red before changing them.\n"
    )
    path.write_text(header + yaml.safe_dump(proposed, sort_keys=False, default_flow_style=False),
                    encoding="utf-8")


def _plot_tradeoff(out_path, per_channel):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, len(per_channel), figsize=(7 * len(per_channel), 6),
                             squeeze=False)
    for ax, result in zip(axes[0], per_channel):
        rec = [(r["recall"], r["precision"]) for r in result["results"] if r["tp"] > 0]
        if rec:
            xs, ys = zip(*rec)
            ax.scatter(xs, ys, s=14, c="#BBBBBB", alpha=0.6, label="evaluated settings")
        front = result["pareto"]
        if front:
            ax.plot([p["recall"] for p in front], [p["precision"] for p in front],
                    "-o", color="#C62828", lw=1.4, ms=5, label="Pareto front")
        base = result["baseline"]
        ax.scatter([base["recall"]], [base["precision"]], marker="*", s=220,
                   c="#1F77B4", edgecolors="black", zorder=5, label="current (baseline)")
        ax.set_xlabel("recall (labelled cells retained)")
        ax.set_ylabel("precision (retained that are cells)")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{result['channel']} (PROVISIONAL candidates)\n"
                     f"cells={result['label_counts'].get('cell', 0)} "
                     f"artefacts={result['label_counts'].get('artefact', 0)}", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Preliminary-rule precision/recall vs human labels -- no count "
                 "targeted, g(r) not used. Choose a setting yourself.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Human-label calibration of the preliminary-pass rules "
                    "(analysis only; changes nothing).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--batch", default=None,
                   help="validation_review_batch.csv (primary input).")
    p.add_argument("--run-dir", default=None,
                   help="Completed run folder; used only to enrich missing feature "
                        "columns and build the duplicate-distance neighbour pool.")
    p.add_argument("--candidates", default=None,
                   help="Explicit all_candidates.csv (overrides --run-dir).")
    p.add_argument("--runs-root", default=None,
                   help="Runs root for auto-discovery (default: <work_dir>/candidates/runs).")
    p.add_argument("--out", default=None, help="Output directory.")
    p.add_argument("--no-duplicate-nms", action="store_true",
                   help="Skip duplicate-distance (NMS) evaluation.")
    args = p.parse_args(argv)

    config = load_config(args.config)
    runs_root = Path(args.runs_root or (config.work_dir / "candidates" / "runs"))

    # Resolve the run (optional: only for feature enrichment + duplicate NMS).
    if args.candidates:
        candidates_csv = Path(args.candidates)
        run_dir = candidates_csv.parent
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        candidates_csv = run_dir / "all_candidates.csv"
    else:
        run_dir = find_newest_run(runs_root)
        candidates_csv = (run_dir / "all_candidates.csv") if run_dir else None

    # The batch is the primary input.
    if args.batch:
        batch_csv = Path(args.batch)
    elif run_dir is not None:
        batch_csv = run_dir / "validation_batch" / "validation_review_batch.csv"
    else:
        print("ERROR: provide --batch (a completed validation_review_batch.csv).")
        return 1
    out_dir = Path(args.out or ((run_dir / "rule_calibration") if run_dir
                                else batch_csv.parent / "rule_calibration"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Label batch   : {batch_csv}")
    print(f"Run (features): {candidates_csv if candidates_csv and candidates_csv.is_file() else 'batch only'}")
    print(f"Output        : {out_dir}")
    print("Analysis only -- no candidate, status, mask, threshold or config is changed.")
    print("=" * 72)

    if not batch_csv.is_file():
        print(f"ERROR: label batch not found: {batch_csv}")
        return 1

    all_rows = (read_csv_rows(candidates_csv)
                if candidates_csv and candidates_csv.is_file() else [])
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
        print(f"WARNING: {unknown} rows had an unrecognised human_label (ignored).")

    # Batch is the primary feature source; enrich only MISSING columns from the run.
    features_by_id = {(r.get("candidate_id"), r.get("channel")): r for r in all_rows}
    labeled_rows_by_channel = {ch: [] for ch in CHANNELS}
    for key, batch_row in labels_by_id.items():
        channel = key[1]
        if channel not in labeled_rows_by_channel:
            continue
        merged = dict(batch_row)
        feat = features_by_id.get(key)
        if feat:
            for column, value in feat.items():
                merged.setdefault(column, value)
        merged["human_label"] = batch_row.get("human_label")
        labeled_rows_by_channel[channel].append(merged)

    base_params = params_from_config(config)

    per_channel = []
    all_results, all_pareto, all_confusion, all_fp, all_fn = [], [], [], [], []
    for channel in CHANNELS:
        rows = labeled_rows_by_channel[channel]
        if not rows:
            print(f"  {channel}: no labelled candidates; skipped.")
            continue
        channel_base = enforce_edge_policy(base_params.for_channel(channel))
        dup_pool = None
        if not args.no_duplicate_nms and all_rows:
            channel_all = [r for r in all_rows if r.get("channel") == channel]
            dup_pool = _build_dup_pool(channel_all, channel_base)
        result = calibrate_channel(channel, rows, channel_base, dup_pool=dup_pool)
        per_channel.append(result)
        all_results.extend(result["results"])
        all_pareto.extend(result["pareto"])
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
    _write_proposed_config(out_dir / "proposed_config_changes.yml", per_channel)

    summary = {
        "analysis": "human-label calibration of preliminary-pass rules "
                    "(PROVISIONAL candidates; NOT cells)",
        "label_batch_csv": str(batch_csv),
        "run_features_csv": (str(candidates_csv)
                             if candidates_csv and candidates_csv.is_file() else None),
        "voxel_size_zyx_um": list(config.acquisition.voxel_size_zyx),
        "label_mapping": {
            "positive": ["cell"],
            "negative": ["artefact"],
            "excluded_from_precision_recall": ["uncertain", "injection"],
        },
        "duplicate_distance_evaluated": (not args.no_duplicate_nms) and bool(all_rows),
        "guarantees": [
            "no candidate, status, mask, threshold, config.yml or raw TIFF was modified",
            "no candidate count was targeted or optimised",
            "pair-correlation g(r) was not used and was not an objective",
            "green_signal and channel_2_signal were calibrated separately",
            "candidates are never rejected for being near the edge alone",
            "the Pareto front is reported; no single setting is auto-selected",
            "proposed_config_changes.yml is REVIEW ONLY and is not applied",
        ],
        "thresholds_evaluated": [
            "component area", "component volume", "support planes",
            "support voxels", "signal-to-background ratio", "diameter range",
            "elongation", "duplicate distance",
        ],
        "channels": {
            r["channel"]: {
                "n_labeled": r["n_labeled"],
                "label_counts": r["label_counts"],
                "baseline_precision": r["baseline"]["precision"],
                "baseline_recall": r["baseline"]["recall"],
                "baseline_f1": r["baseline"]["f1"],
                "n_parameter_sets_evaluated": len(r["results"]),
                "n_pareto_points": len(r["pareto"]),
            }
            for r in per_channel
        },
    }
    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("=" * 72)
    print(f"Wrote calibration artifacts to {out_dir}")
    print("Review proposed_config_changes.yml (high-precision vs high-recall). "
          "Nothing was auto-chosen or applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
