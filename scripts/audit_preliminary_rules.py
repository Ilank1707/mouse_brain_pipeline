#!/usr/bin/env python
"""Audit the preliminary-pass rule: OLD vs NEW configurable gates (Part 2).

Re-applies the preliminary-pass interpretation to an EXISTING all_candidates.csv
under the OLD gates (new gates disabled) and the NEW gates (from config), so the
tightening can be documented without re-running detection. Preliminary passes are
NOT cells; this only re-labels the sampling category using the same code path as
the pipeline (``_preliminary_interpretation``).

It never modifies the input run; every artifact is written into ``--out``.

Outputs:
  preliminary_rule_parameter_comparison.csv
  preliminary_rule_fail_reasons.csv
  preliminary_rule_distributions.png
  preliminary_rule_before_after_qc.png
  preliminary_rule_summary.json

Example (PowerShell):
  python scripts/audit_preliminary_rules.py --config config.yml `
      --candidates <run>/all_candidates.csv --channel green_signal --section 70 `
      --out C:/mouse_brain_work/candidates/preliminary_rule_audit
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from mouse_brain_pipeline import candidate_detection as cd
from mouse_brain_pipeline.candidate_detection import (
    STATUS_PRELIMINARY_PASS,
    params_from_config,
)
from mouse_brain_pipeline.config import load_config

_VOXEL_UM3 = 6.0 * 1.004 * 1.004
_BOOL_TRUE = {"true", "1", "yes"}


def _fnum(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _fbool(value):
    return str(value).strip().lower() in _BOOL_TRUE


def _coerce_rec(row):
    """Rebuild the fields ``_preliminary_interpretation`` reads from a CSV row.

    xy_area_um2 / supporting_voxel_count are derived when absent (older tables).
    """
    xy_diam = _fnum(row.get("xy_diameter_um"))
    volume = _fnum(row.get("volume_um3"))
    xy_area = _fnum(row.get("xy_area_um2"))
    if not math.isfinite(xy_area):
        xy_area = math.pi * (xy_diam / 2.0) ** 2 if math.isfinite(xy_diam) else float("nan")
    vox = _fnum(row.get("supporting_voxel_count"))
    if not math.isfinite(vox):
        vox = volume / _VOXEL_UM3 if math.isfinite(volume) else float("nan")
    support = _fnum(row.get("support_plane_count"))
    if not math.isfinite(support):
        support = _fnum(row.get("n_consecutive_planes"))
    return {
        "inside_tissue": _fbool(row.get("inside_tissue")),
        "invalid_coordinate": _fbool(row.get("invalid_coordinate")),
        "original_cellfinder_z_valid": _fbool(row.get("original_cellfinder_z_valid") or "true"),
        "measurement_valid": _fbool(row.get("measurement_valid")),
        "is_artifact": _fbool(row.get("is_artifact")),
        "touches_crop_boundary": _fbool(row.get("touches_crop_boundary")),
        "n_consecutive_planes": support if math.isfinite(support) else 0,
        "support_plane_count": support if math.isfinite(support) else 0,
        "equivalent_diameter_um": _fnum(row.get("equivalent_diameter_um")),
        "xy_diameter_um": xy_diam,
        "xy_area_um2": xy_area,
        "volume_um3": volume,
        "supporting_voxel_count": vox,
        "elongation": _fnum(row.get("elongation")),
        "xy_centroid_shift_um": _fnum(row.get("xy_centroid_shift_um")),
        "local_robust_z": _fnum(row.get("local_robust_z")),
        "x_global_px": _fnum(row.get("x_global_px")),
        "y_global_px": _fnum(row.get("y_global_px")),
    }


def _old_params(new_params):
    """New gates disabled + legacy edge dropping = the previous behaviour."""
    return dataclasses.replace(
        new_params,
        min_component_xy_area_um2=0.0,
        min_component_volume_um3=0.0,
        min_support_planes=0,
        min_supporting_voxels=0,
        min_signal_to_background_ratio=0.0,
        keep_edge_clipped_if_center_in_tissue=False,
    )


def _incremental_projection(plane_paths):
    import tifffile

    proj = None
    for _pl in sorted(plane_paths):
        with tifffile.TiffFile(str(plane_paths[_pl])) as tf:
            arr = np.asarray(tf.pages[0].asarray())
            proj = arr.copy() if proj is None else np.maximum(proj, arr)
    return proj


def main() -> int:
    p = argparse.ArgumentParser(description="Preliminary-rule OLD vs NEW audit (no counts).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--candidates", required=True, help="Existing all_candidates.csv")
    p.add_argument("--channel", default="green_signal")
    p.add_argument("--section", type=int, default=70)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    new_params = params_from_config(cfg)
    old_params = _old_params(new_params)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        r for r in csv.DictReader(open(args.candidates, encoding="utf-8"))
        if r.get("channel") == args.channel
    ]
    if not rows:
        print(f"No {args.channel} rows in {args.candidates}.")
        return 1
    recs = [_coerce_rec(r) for r in rows]
    has_tissue = True  # detection run had the tissue mask enabled

    old_pass, new_pass = [], []
    new_fail_reason = []
    for rec in recs:
        os_, _ = cd._preliminary_interpretation(rec, old_params, has_tissue)
        ns_, nr_ = cd._preliminary_interpretation(rec, new_params, has_tissue)
        old_pass.append(os_ == STATUS_PRELIMINARY_PASS)
        new_pass.append(ns_ == STATUS_PRELIMINARY_PASS)
        new_fail_reason.append("" if ns_ == STATUS_PRELIMINARY_PASS else nr_)

    n = len(recs)
    n_old = sum(old_pass)
    n_new = sum(new_pass)
    newly_failed = [i for i in range(n) if old_pass[i] and not new_pass[i]]
    newly_passed = [i for i in range(n) if new_pass[i] and not old_pass[i]]

    # ---- individual-rule pass counts (NEW thresholds) ----------------------- #
    def count(cond):
        return int(sum(1 for rec in recs if cond(rec)))

    gates = [
        ("minimum_component_xy_area_um2", 0.0, new_params.min_component_xy_area_um2,
         "min-diameter cross-section pi*(6/2)^2",
         lambda r: r["xy_area_um2"] >= new_params.min_component_xy_area_um2),
        ("minimum_component_volume_um3", 0.0, new_params.min_component_volume_um3,
         "min-diameter sphere (pi/6)*6^3",
         lambda r: r["volume_um3"] >= new_params.min_component_volume_um3),
        ("minimum_supporting_voxels", 0, new_params.min_supporting_voxels,
         "113 um3 / voxel volume",
         lambda r: r["supporting_voxel_count"] >= new_params.min_supporting_voxels),
        ("minimum_support_planes", 0, new_params.min_support_planes,
         "axial support on >= 2 optical planes",
         lambda r: r["support_plane_count"] >= new_params.min_support_planes),
        ("minimum_signal_to_background_ratio", new_params.min_local_robust_z,
         new_params.min_signal_to_background_ratio,
         "robust z anchored to single-plane review level 8.0",
         lambda r: r["local_robust_z"] >= new_params.min_signal_to_background_ratio),
    ]
    comparison_rows = []
    for name, old_v, new_v, reason, cond in gates:
        passing = count(cond)
        comparison_rows.append({
            "parameter": name,
            "old_value": old_v,
            "new_value": new_v,
            "reason": reason,
            "n_passing_individual_rule": passing,
            "n_failing_individual_rule": n - passing,
        })

    with (out_dir / "preliminary_rule_parameter_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
        w.writeheader()
        w.writerows(comparison_rows)

    # ---- fail-reason breakdown --------------------------------------------- #
    from collections import Counter

    reason_counts = Counter(r for r in new_fail_reason if r)
    with (out_dir / "preliminary_rule_fail_reasons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["fail_reason", "count_new_rule"])
        for reason, c in reason_counts.most_common():
            w.writerow([reason, c])
        w.writerow(["__old_preliminary_passes__", n_old])
        w.writerow(["__new_preliminary_passes__", n_new])
        w.writerow(["__newly_failed_vs_old__", len(newly_failed)])
        w.writerow(["__newly_passed_vs_old__", len(newly_passed)])

    # ---- distributions ------------------------------------------------------ #
    _plot_distributions(out_dir, recs, new_params)

    # ---- before/after QC on the SAME image + coordinates -------------------- #
    _plot_before_after(out_dir, cfg, args, rows, old_pass, new_pass)

    summary = {
        "analysis": "preliminary-rule OLD vs NEW audit (PROVISIONAL; not cells)",
        "channel": args.channel,
        "candidates_csv": str(args.candidates),
        "n_candidates": n,
        "old_preliminary_passes": n_old,
        "new_preliminary_passes": n_new,
        "newly_failed_vs_old": len(newly_failed),
        "newly_passed_vs_old": len(newly_passed),
        "note": "thresholds derived from the 6 um minimum-cell-diameter scale and "
                "the existing 8.0 single-plane review level; NOT tuned to a count.",
        "parameter_comparison": comparison_rows,
        "new_fail_reason_counts": dict(reason_counts),
        "combined_rule": "a preliminary pass now requires ALL applicable gates "
                         "(area, volume, supporting voxels, >=2 support planes, "
                         "signal-to-background) in addition to the prior rules.",
    }
    (out_dir / "preliminary_rule_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"old preliminary passes: {n_old}")
    print(f"new preliminary passes: {n_new}  (newly failed {len(newly_failed)}, "
          f"newly passed {len(newly_passed)})")
    print(f"Wrote preliminary-rule audit to {out_dir}")
    return 0


def _plot_distributions(out_dir, recs, params):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = [
        ("xy_area_um2", params.min_component_xy_area_um2, "XY area (um^2)"),
        ("volume_um3", params.min_component_volume_um3, "volume (um^3)"),
        ("supporting_voxel_count", params.min_supporting_voxels, "supporting voxels"),
        ("support_plane_count", params.min_support_planes, "support planes"),
        ("local_robust_z", params.min_signal_to_background_ratio, "signal/background (robust z)"),
        ("elongation", None, "elongation"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, thr, label) in zip(axes.ravel(), fields):
        vals = np.array([rec[key] for rec in recs if math.isfinite(rec.get(key, float("nan")))])
        if vals.size:
            hi = np.percentile(vals, 99)
            ax.hist(vals[vals <= hi], bins=60, color="#4C78A8")
        if thr:
            ax.axvline(thr, color="#C62828", lw=2, label=f"new min = {thr}")
            ax.legend(fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("candidates")
    fig.suptitle("Preliminary-rule input distributions (PROVISIONAL candidates)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "preliminary_rule_distributions.png", dpi=130)
    plt.close(fig)


def _plot_before_after(out_dir, cfg, args, rows, old_pass, new_pass):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mouse_brain_pipeline.audit import index_channel

    dirs = {
        "green_signal": cfg.data.green_signal_dir,
        "channel_2_signal": cfg.data.channel_2_signal_dir,
    }
    background = None
    extent = None
    try:
        idx = index_channel(args.channel, dirs[args.channel], cfg.data.filename_regex)
        plane_paths = {pl: path for (s, pl), path in idx.files.items() if s == args.section}
        if plane_paths:
            proj = _incremental_projection(plane_paths)
            step = max(1, min(proj.shape) // 1200)
            background = proj[::step, ::step]
            extent = (0, proj.shape[1], proj.shape[0], 0)
    except Exception as exc:  # pragma: no cover - background is optional
        print(f"  [before/after QC] projection unavailable: {exc}")

    xs = np.array([_fnum(r.get("x_global_px")) for r in rows])
    ys = np.array([_fnum(r.get("y_global_px")) for r in rows])
    op = np.array(old_pass)
    npass = np.array(new_pass)
    both = op & npass
    dropped = op & ~npass
    added = ~op & npass

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, title in zip(axes, ("OLD preliminary passes", "NEW preliminary passes")):
        if background is not None:
            lo, hi = np.percentile(background, [1, 99.5])
            ax.imshow(background, cmap="gray", extent=extent,
                      vmin=lo, vmax=max(hi, lo + 1), aspect="equal")
        ax.set_title(f"{title} -- {args.channel} section {args.section:03d}\n"
                     "PROVISIONAL (not cells)", fontsize=10)
        ax.set_xlabel("x_global_px")
        ax.set_ylabel("y_global_px")
    sel_old = op
    sel_new = npass
    axes[0].scatter(xs[sel_old], ys[sel_old], s=3, c="#2E7D32", alpha=0.5, label="pass")
    axes[0].scatter(xs[~sel_old], ys[~sel_old], s=1, c="#BBBBBB", alpha=0.25)
    axes[1].scatter(xs[both], ys[both], s=3, c="#2E7D32", alpha=0.5, label="still pass")
    axes[1].scatter(xs[dropped], ys[dropped], s=6, c="#C62828", alpha=0.7,
                    label="newly failed")
    axes[1].scatter(xs[added], ys[added], s=6, c="#1F77B4", alpha=0.7, label="newly passed")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "preliminary_rule_before_after_qc.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
