#!/usr/bin/env python
"""Trace the per-channel injection MASK from image to overlay (audit, no counts).

This reproduces -- stage by stage -- the exact automatic injection-mask pipeline
used by ``candidate_detection`` so the mask can be audited without running a full
detection. It NEVER modifies raw TIFFs and NEVER writes into an existing run
folder (it writes into its own ``--out`` directory).

It answers, for one channel (default the GREEN signal channel):

  1. which seven-plane projection is used;
  2. the injection threshold and how it is computed;
  3. opening / closing / dilation / hole-filling parameters;
  4. connected-component labels before and after morphology;
  5. the configured seeds;
  6. which seed matched each component;
  7. whether multiple seeds retain multiple regions;
  8. whether dilation / closing / bridge / watershed / seed assignment enlarges
     the final mask;
  9. why the final mask contains its specific regions;
 10. how the red channel result compares (compactness).

Outputs (into ``--out``):
  injection_mask_debug_summary.json
  injection_mask_debug_steps.png
  injection_mask_debug_components.csv
  injection_mask_kept_vs_removed.png
  injection_mask_parameter_summary.json
  injection_mask_components_before_split.csv
  injection_mask_components_after_split.csv
  injection_mask_seed_matches.csv

Example (PowerShell):
  python scripts/debug_injection_mask.py --config config.yml --section 70 `
      --channel green_signal --out C:/mouse_brain_work/candidates/injection_mask_debug
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from mouse_brain_pipeline import candidate_detection as cd
from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.injection_mask_diagnostics import (
    write_injection_mask_diagnostics,
)


def _incremental_projection(plane_paths: dict) -> np.ndarray:
    """Max projection over the section's planes, one plane at a time (low memory)."""
    import tifffile

    proj = None
    for _pl in sorted(plane_paths):
        with tifffile.TiffFile(str(plane_paths[_pl])) as tf:
            page = tf.pages[0]
            try:
                arr = page.asarray(out="memmap")
            except (ValueError, TypeError):
                arr = page.asarray()
            arr = np.asarray(arr)
            proj = arr.copy() if proj is None else np.maximum(proj, arr)
    return proj


def _low_res_stages(proj: np.ndarray, cfg, voxel_yx):
    """Reproduce the low-res bright-mask stages exactly like ``_injection_base_mask``."""
    from scipy import ndimage as ndi

    vy, vx = float(voxel_yx[0]), float(voxel_yx[1])
    factor = max(1, int(round(cfg.downsample_um / vy)))
    low = proj[::factor, ::factor].astype(np.float32)
    sigma_px = max(1.0, cfg.smoothing_sigma_um / (vy * factor))
    smoothed = ndi.gaussian_filter(low, sigma=sigma_px)

    thr, thr_method = cd._bright_threshold(smoothed, cfg)
    thresholded = smoothed >= thr                       # mask before morphology
    low_voxel = (vy * factor, vx * factor)

    opening_um = float(getattr(cfg, "opening_radius_um", 0.0) or 0.0)
    closing_um = float(getattr(cfg, "closing_radius_um", 0.0) or 0.0)
    bridge_um = float(getattr(cfg, "maximum_bridge_width_um", 0.0) or 0.0)
    opened = thresholded
    if opening_um > 0:
        r = max(1, int(round(opening_um / low_voxel[0])))
        opened = ndi.binary_opening(thresholded, structure=cd._disk(r))
    effective_close_um = closing_um
    if closing_um > 0 and bridge_um > 0:
        effective_close_um = min(closing_um, bridge_um / 2.0)
    closed = opened
    if closing_um > 0:
        r = max(1, int(round(effective_close_um / low_voxel[0])))
        closed = ndi.binary_closing(opened, structure=cd._disk(r))

    low_area = low_voxel[0] * low_voxel[1]
    small_filtered = cd._remove_small_components(closed, cfg.minimum_area_um2 / low_area)
    max_area_um2 = getattr(cfg, "maximum_component_area_um2", None)
    area_filtered = small_filtered
    if max_area_um2:
        area_filtered = cd._remove_large_components(
            small_filtered, float(max_area_um2) / low_area
        )

    pre_labels, n_pre = ndi.label(area_filtered)
    return {
        "factor": factor,
        "low_voxel": low_voxel,
        "smoothed": smoothed,
        "threshold": float(thr),
        "threshold_method": thr_method,
        "thresholded": thresholded,
        "opened": opened,
        "closed": closed,
        "small_filtered": small_filtered,
        "area_filtered": area_filtered,
        "pre_labels": pre_labels,
        "n_pre": int(n_pre),
        "effective_closing_um": effective_close_um,
    }


def _run_channel(channel, plane_paths, cfg_channel, voxel_zyx, seeds_cfg):
    """Return stage dict + current(no-cap) and fixed(config) seed-split diagnostics."""
    voxel_yx = (voxel_zyx[1], voxel_zyx[2])
    proj = _incremental_projection(plane_paths)
    stages = _low_res_stages(proj, cfg_channel, voxel_yx)
    factor = stages["factor"]
    H, W = proj.shape

    seeds_local = cd._seed_points_local(cfg_channel, (0, 0), (H, W))
    bright_low = stages["area_filtered"]
    filter_by_seeds = bool(cfg_channel.injection_seed_points)

    # Fixed = the caps configured on this channel now.
    _kept_fixed, diag_fixed = cd._split_and_filter_by_seeds(
        bright_low, seeds_local, stages["low_voxel"], cfg_channel, factor,
        filter_by_seeds=filter_by_seeds,
    )
    # Current = the same split with the distance/morphology caps disabled, to show
    # the pre-fix (broken) behaviour on the identical bright mask.
    cfg_nocap = dataclasses.replace(
        cfg_channel,
        maximum_distance_from_seed_um=None,
        opening_radius_um=0.0,
        closing_radius_um=0.0,
        maximum_component_area_um2=None,
        seed_match_radius_um=None,
    )
    _kept_cur, diag_current = cd._split_and_filter_by_seeds(
        bright_low, seeds_local, stages["low_voxel"], cfg_nocap, factor,
        filter_by_seeds=filter_by_seeds,
    )
    return {
        "projection_shape": [int(H), int(W)],
        "stages": stages,
        "seeds_local": seeds_local,
        "diag_current": diag_current,
        "diag_fixed": diag_fixed,
        "kept_current": _kept_cur,
        "kept_fixed": _kept_fixed,
    }


def _mask_stats(mask):
    ys, xs = np.nonzero(mask)
    return {
        "area_lowres_px": int(mask.sum()),
        "bbox_y": [int(ys.min()), int(ys.max())] if ys.size else None,
        "bbox_x": [int(xs.min()), int(xs.max())] if xs.size else None,
    }


def _render_steps(out_dir, channel, section, info):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = info["stages"]
    diag_cur = info["diag_current"]
    diag_fix = info["diag_fixed"]
    factor = st["factor"]
    seeds_low = [(int(y) // factor, int(x) // factor) for (y, x) in info["seeds_local"]]

    def _seed_scatter(ax):
        for (yl, xl) in seeds_low:
            ax.scatter([xl], [yl], marker="x", c="red", s=70, linewidths=2)

    panels = [
        ("1. source max projection (smoothed)", st["smoothed"], "gray", False),
        (f"2. thresholded (>= {st['threshold']:.1f}, {st['threshold_method']})",
         st["thresholded"], "binary", False),
        ("3. mask before morphology", st["thresholded"], "binary", False),
        ("4a. after opening", st["opened"], "binary", False),
        ("4b. after closing (bridge-capped)", st["closed"], "binary", False),
        ("4c. after min/max area filter", st["area_filtered"], "binary", False),
        ("5. connected components (pre-split)", st["pre_labels"], "tab20", True),
        ("6. seed locations", st["area_filtered"], "binary", True),
        ("7. seed-matched components (post-split)", diag_fix["post_labels_lowres"],
         "tab20", True),
        ("8. CURRENT kept (no distance cap)", info["kept_current"], "Reds", True),
        ("9. FIXED kept (distance-capped)", info["kept_fixed"], "Greens", True),
        ("10. FIXED removed", st["area_filtered"] & ~info["kept_fixed"], "Oranges", True),
    ]
    ncols = 4
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, data, cmap, seeds) in zip(axes, panels):
        arr = np.asarray(data)
        if cmap in ("tab20",):
            arr = np.ma.masked_where(arr == 0, arr)
        ax.imshow(arr, cmap=cmap, interpolation="nearest")
        if seeds:
            _seed_scatter(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (downsampled px)", fontsize=7)
        ax.set_ylabel("y (downsampled px)", fontsize=7)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        f"Injection MASK debug -- {channel} section {section:03d} "
        f"(PROVISIONAL injection mask; downsample factor {factor})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = out_dir / "injection_mask_debug_steps.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _write_components_csv(out_dir, info):
    diag_fix = info["diag_fixed"]
    diag_cur = info["diag_current"]
    cur_kept = set(diag_cur.get("kept_subcomponent_labels", []))
    fix_kept = set(diag_fix.get("kept_subcomponent_labels", []))
    rows = []
    for rec in diag_fix.get("post_split_subcomponents", []):
        label = rec["subcomponent_label"]
        rows.append({
            "subcomponent_label": label,
            "parent_pre_label": rec["parent_pre_label"],
            "area_px": rec["area_px"],
            "area_um2": rec["area_um2"],
            "centroid_x_local": rec["centroid_x_local"],
            "centroid_y_local": rec["centroid_y_local"],
            "contains_seed": rec["contains_seed"],
            "kept_current_no_cap": label in cur_kept,
            "kept_fixed_capped": label in fix_kept,
            "reason": rec["reason"],
        })
    path = out_dir / "injection_mask_debug_components.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                                ["subcomponent_label"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _render_kept_vs_removed(out_dir, channel, section, info):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    st = info["stages"]
    factor = st["factor"]
    kept = info["kept_fixed"]
    removed = st["area_filtered"] & ~kept
    status = np.zeros(kept.shape, dtype=np.int8)
    status[removed] = 1
    status[kept] = 2
    seeds_low = [(int(y) // factor, int(x) // factor) for (y, x) in info["seeds_local"]]
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = ListedColormap(["#00000000", "#C62828", "#2E7D32"])
    ax.imshow(status, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    for (yl, xl) in seeds_low:
        ax.scatter([xl], [yl], marker="x", c="white", s=80, linewidths=2)
    ax.set_title(
        f"Injection mask kept (green) vs removed (red) -- {channel} section "
        f"{section:03d} (distance-capped)"
    )
    ax.set_xlabel("x (downsampled px)")
    ax.set_ylabel("y (downsampled px)")
    fig.tight_layout()
    path = out_dir / "injection_mask_kept_vs_removed.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Injection-mask audit trace (no counts).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--section", type=int, default=70)
    p.add_argument("--channel", default="green_signal")
    p.add_argument("--compare-channel", default="channel_2_signal",
                   help="Channel used only as a compactness comparison (item 10).")
    p.add_argument("--out", required=True, help="Output directory (own folder; never a run).")
    args = p.parse_args()

    cfg = load_config(args.config)
    voxel = cfg.acquisition.voxel_size_zyx
    regex = cfg.data.filename_regex
    dirs = {
        "green_signal": cfg.data.green_signal_dir,
        "channel_2_signal": cfg.data.channel_2_signal_dir,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plane_paths(channel):
        idx = index_channel(channel, dirs[channel], regex)
        return {pl: path for (s, pl), path in idx.files.items() if s == args.section}

    channel = args.channel
    inj = cfg.detection.injection_exclusion.for_channel(channel)
    plane_paths = _plane_paths(channel)
    if not plane_paths:
        print(f"No planes found for {channel} section {args.section}.")
        return 1
    print(f"Tracing {channel} section {args.section:03d} ({len(plane_paths)} planes)...")
    info = _run_channel(channel, plane_paths, inj, voxel, inj.injection_seed_points)

    steps_png = _render_steps(out_dir, channel, args.section, info)
    components_csv = _write_components_csv(out_dir, info)
    kept_removed_png = _render_kept_vs_removed(out_dir, channel, args.section, info)

    # Before/after split CSVs + seed matches from the FIXED diagnostics.
    write_injection_mask_diagnostics(
        out_dir, info["diag_fixed"], channel=channel, section=args.section,
        voxel_yx_um=(voxel[1], voxel[2]),
    )

    # Compactness comparison for the other (red) channel.
    compare = {}
    comp_ch = args.compare_channel
    comp_paths = _plane_paths(comp_ch)
    if comp_paths:
        comp_inj = cfg.detection.injection_exclusion.for_channel(comp_ch)
        comp_info = _run_channel(comp_ch, comp_paths, comp_inj, voxel, [])
        cst = comp_info["stages"]
        compare = {
            "channel": comp_ch,
            "n_components": cst["n_pre"],
            "bright_area_lowres_px": int(cst["area_filtered"].sum()),
            "seeds_configured": bool(comp_inj.injection_seed_points),
            "kept_stats": _mask_stats(comp_info["kept_fixed"]),
        }

    st = info["stages"]
    diag_cur = info["diag_current"]
    diag_fix = info["diag_fixed"]
    summary = {
        "analysis": "injection-mask audit trace (PROVISIONAL mask; not cells)",
        "channel": channel,
        "section": args.section,
        "1_projection": {
            "type": "per-plane maximum projection over the seven optical planes",
            "planes": sorted(plane_paths),
            "shape_yx": info["projection_shape"],
        },
        "2_threshold": {
            "method": st["threshold_method"],
            "value_smoothed_units": st["threshold"],
            "intensity_percentile": inj.intensity_percentile,
            "smoothing_sigma_um": inj.smoothing_sigma_um,
            "downsample_um": inj.downsample_um,
            "downsample_factor": st["factor"],
        },
        "3_morphology": {
            "opening_radius_um": getattr(inj, "opening_radius_um", 0.0),
            "closing_radius_um": getattr(inj, "closing_radius_um", 0.0),
            "effective_closing_um_after_bridge_cap": st["effective_closing_um"],
            "maximum_bridge_width_um": getattr(inj, "maximum_bridge_width_um", 0.0),
            "core_dilation_um": inj.core_dilation_um,
            "dilation_radius_um": getattr(inj, "dilation_radius_um", None),
            "analysis_exclusion_dilation_um": inj.analysis_exclusion_dilation_um,
            "hole_filling": "binary_fill_holes applied to the tissue mask only; "
                            "the injection base uses opening/closing above",
        },
        "4_components": {
            "labels_before_morphology": int(
                __import__("scipy.ndimage", fromlist=["label"]).label(st["thresholded"])[1]
            ),
            "labels_after_morphology_and_area_filter": st["n_pre"],
            "subcomponents_after_split": diag_fix["n_subcomponents"],
        },
        "5_seeds": {
            "configured_full_res_xy": inj.injection_seed_points,
            "n_seeds": len(inj.injection_seed_points or []),
        },
        "6_seed_component_matches": diag_fix["seed_matches"],
        "7_multiple_seeds_multiple_regions": {
            "n_seeds": diag_fix["n_seed_points"],
            "n_kept_subcomponents_current": diag_cur["n_kept"],
            "n_kept_subcomponents_fixed": diag_fix["n_kept"],
            "note": "each seed retains one region; with 2 seeds up to 2 regions "
                    "are retained (the upper and lower orange areas).",
        },
        "8_enlargement_sources": {
            "watershed_basin_can_be_large": True,
            "current_kept_area_lowres_px": int(info["kept_current"].sum()),
            "fixed_kept_area_lowres_px": int(info["kept_fixed"].sum()),
            "shrink_factor": (
                round(int(info["kept_fixed"].sum()) / max(int(info["kept_current"].sum()), 1), 4)
            ),
            "note": "without maximum_distance_from_seed_um a single seed keeps the "
                    "whole watershed basin; the core/analysis dilations then grow it "
                    "further. The distance cap trims the kept region to a bounded "
                    "neighbourhood of the seed BEFORE dilation.",
        },
        "9_final_regions": {
            "kept_current_stats": _mask_stats(info["kept_current"]),
            "kept_fixed_stats": _mask_stats(info["kept_fixed"]),
        },
        "10_red_comparison": compare,
        "seed_distance_cap": {
            "maximum_distance_from_seed_um": getattr(inj, "maximum_distance_from_seed_um", None),
            "seed_distance_metric": getattr(inj, "seed_distance_metric", "geodesic"),
            "applied": diag_fix.get("seed_distance_capped", False),
        },
        "outputs": {
            "steps_png": str(steps_png),
            "components_csv": str(components_csv),
            "kept_vs_removed_png": str(kept_removed_png),
        },
    }
    (out_dir / "injection_mask_debug_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Parameter summary: old (no-cap) vs new (capped) configurable criteria.
    param_summary = {
        "channel": channel,
        "section": args.section,
        "parameters": {
            "injection_threshold_method": getattr(inj, "injection_threshold_method", "percentile"),
            "injection_threshold_value": getattr(inj, "injection_threshold_value", None),
            "intensity_percentile": inj.intensity_percentile,
            "minimum_area_um2": inj.minimum_area_um2,
            "maximum_component_area_um2": getattr(inj, "maximum_component_area_um2", None),
            "opening_radius_um": getattr(inj, "opening_radius_um", 0.0),
            "closing_radius_um": getattr(inj, "closing_radius_um", 0.0),
            "maximum_bridge_width_um": getattr(inj, "maximum_bridge_width_um", 0.0),
            "dilation_radius_um": getattr(inj, "dilation_radius_um", None),
            "core_dilation_um": inj.core_dilation_um,
            "analysis_exclusion_dilation_um": inj.analysis_exclusion_dilation_um,
            "seed_match_radius_um": getattr(inj, "seed_match_radius_um", None),
            "maximum_distance_from_seed_um": getattr(inj, "maximum_distance_from_seed_um", None),
            "seed_distance_metric": getattr(inj, "seed_distance_metric", "geodesic"),
            "split_merged_components": inj.split_merged_components,
            "split_min_peak_distance_um": inj.split_min_peak_distance_um,
        },
        "kept_area_lowres_px_current_no_cap": int(info["kept_current"].sum()),
        "kept_area_lowres_px_fixed_capped": int(info["kept_fixed"].sum()),
    }
    (out_dir / "injection_mask_parameter_summary.json").write_text(
        json.dumps(param_summary, indent=2, default=str), encoding="utf-8"
    )

    print(f"Wrote injection-mask debug artifacts to {out_dir}")
    print(f"  current kept (no cap) low-res px : {int(info['kept_current'].sum())}")
    print(f"  fixed kept (capped)   low-res px : {int(info['kept_fixed'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
