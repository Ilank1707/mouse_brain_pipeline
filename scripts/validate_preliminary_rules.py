#!/usr/bin/env python
"""Compact validation workflow for the newest completed section-070 run.

Purpose: help decide whether the current green/red preliminary-pass rules are too
strict or too permissive BEFORE any threshold is changed again. It reads an
existing ``all_candidates.csv`` and writes only NEW artifacts into its own
``--out`` directory. It NEVER changes thresholds, statuses, masks or raw TIFFs,
and it NEVER calls a provisional candidate a cell -- every candidate here is a
PROVISIONAL detection / preliminary-rule pass or fail only.

For each channel (green_signal and channel_2_signal, kept strictly separate) it
draws a balanced, reproducible random review sample:

  * 100 preliminary_rule_pass candidates
  * 100 preliminary_rule_fail candidates

(fewer only when the population is smaller). A fixed random seed makes the sample
reproducible. For every sampled candidate it exports a seven-plane review patch
(the same fixed-XY seven optical planes, raw + background-corrected, crosshair,
peak/support planes highlighted).

Outputs (all under ``--out``):
  validation_review_batch.csv          one row per sampled candidate + human_label
  validation_review_patches/           per-candidate seven-plane PNGs, split by
                                        green_signal/ and channel_2_signal/
  validation_sample_summary.csv        population vs sampled counts per stratum
  preliminary_fail_reason_counts.csv   why fails fail (population + sampled)
  injection_mask_qc/                   green + red injection mask over all seven
                                        planes, to confirm the green mask follows
                                        the bright region and is not a fixed circle

Example (PowerShell):
  python scripts/validate_preliminary_rules.py --config config.yml --section 70
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.candidate_detection import background_correct
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.review import read_csv_rows
from mouse_brain_pipeline.review_patches import (
    HIGHLIGHT_COLOURS,
    display_limits,
    ordered_section_planes,
    panel_highlight_class,
    parse_peak_index,
    parse_support_indices,
)

# Reuse the exact injection-mask stages the pipeline / mask audit use, so the QC
# image shows the SAME mask (never a re-invented one).
from debug_injection_mask import _run_channel  # noqa: E402

CHANNELS = ("green_signal", "channel_2_signal")
PASS_CATEGORY = "preliminary_rule_pass"
FAIL_CATEGORY = "preliminary_rule_fail"

# Columns requested for the review batch. current_status / preliminary_rule_reason
# are copied verbatim from the run (never recomputed); human_label is left blank.
REVIEW_BATCH_COLUMNS = [
    "candidate_id",
    "channel",
    "sampling_stratum",                 # preliminary_rule_pass / _fail (sampled bucket)
    "current_status",
    "preliminary_sampling_category",
    "preliminary_rule_reason",
    "x_global_px",
    "y_global_px",
    "z_index",
    "optical_plane",
    "global_z_um",
    "section",
    "xy_area_um2",                      # area
    "volume_um3",                       # volume
    "support_plane_count",              # support planes
    "supporting_voxel_count",           # support voxels
    "local_robust_z",                   # signal-to-background metric (the gated one)
    "local_contrast_score",             # secondary signal/contrast readout
    "touches_crop_boundary",            # edge / clipping
    "invalid_coordinate",               # edge / clipping
    "inside_tissue",                    # edge / clipping context
    "inside_injection_analysis_exclusion",
    "review_patch_file",
    "human_label",                      # BLANK, for the human reviewer
    "review_notes",                     # BLANK
]


# --------------------------------------------------------------------------- #
# Run / candidate discovery
# --------------------------------------------------------------------------- #
def find_newest_candidates_csv(runs_root: Path) -> Path | None:
    """Newest ``all_candidates.csv`` under ``runs_root`` (by modified time)."""
    matches = list(runs_root.glob("*/all_candidates.csv"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _channel_dir(config, channel):
    return (
        config.data.green_signal_dir
        if channel == "green_signal"
        else config.data.channel_2_signal_dir
    )


def patch_half_px(config) -> int:
    """Half-window in pixels from the configured patch size (>= 8 px)."""
    return max(
        8,
        int(round(config.classifier.patch_size_xy_um
                  / (2 * config.acquisition.voxel_size_y_um))),
    )


# --------------------------------------------------------------------------- #
# Balanced reproducible sampling
# --------------------------------------------------------------------------- #
def stratum_seed(base_seed, channel, stratum):
    """Process-stable seed for one (channel, stratum) sample.

    Uses a hashlib digest -- NOT the builtin ``hash()``, which is salted per
    process -- so the four samples differ but each is reproducible run to run.
    """
    digest = hashlib.sha1(f"{channel}:{stratum}".encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest, 16) % 100000


def sample_stratum(rows, seed, per_stratum):
    """Deterministic random subsample; sorted by candidate_id first so the pick
    is reproducible regardless of CSV row order."""
    ordered = sorted(rows, key=lambda r: str(r.get("candidate_id", "")))
    if len(ordered) <= per_stratum:
        return list(ordered)
    return random.Random(seed).sample(ordered, per_stratum)


# --------------------------------------------------------------------------- #
# Seven-plane review patch (static export; never mutates raw TIFFs)
# --------------------------------------------------------------------------- #
def _open_plane_memmaps(ordered_planes):
    import tifffile  # noqa: PLC0415

    handles, arrays, numbers = [], [], []
    for plane, path in ordered_planes:
        tf = tifffile.TiffFile(str(path))
        handles.append(tf)
        page = tf.pages[0]
        try:
            arrays.append(page.asarray(out="memmap"))
        except (TypeError, ValueError):
            arrays.append(page.asarray())
        numbers.append(plane)
    return handles, arrays, numbers


def _crop_fixed_xy(plane_arrays, x_global, y_global, half_px):
    """Zero-padded (Z, size, size) crop at the SAME (x, y) centre in every plane."""
    x = int(round(float(x_global)))
    y = int(round(float(y_global)))
    size = 2 * half_px + 1
    out = []
    for image in plane_arrays:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        target = np.zeros((size, size), dtype=image.dtype)
        y0, y1 = max(0, y - half_px), min(height, y + half_px + 1)
        x0, x1 = max(0, x - half_px), min(width, x + half_px + 1)
        if y1 > y0 and x1 > x0:
            crop = np.array(image[y0:y1, x0:x1], copy=True)
            oy = half_px - (y - y0)
            ox = half_px - (x - x0)
            target[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        out.append(target)
    return np.stack(out), half_px, half_px


def save_seven_plane_patch(out_path, plt, candidate, raw_stack, corrected_stack,
                           plane_numbers, cy, cx):
    """Static seven-plane contact sheet (raw + bg-corrected) for one candidate."""
    peak = parse_peak_index(candidate)
    support = parse_support_indices(candidate)
    raw_lo, raw_hi = display_limits(raw_stack)
    corr_lo, corr_hi = display_limits(corrected_stack)
    n = raw_stack.shape[0]

    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 4.2), squeeze=False)
    for col in range(n):
        highlight = panel_highlight_class(col, peak, support)
        tag = {"peak": " PEAK", "support": " support", "none": ""}[highlight]
        for row, (stack, lo, hi, name) in enumerate((
            (raw_stack, raw_lo, raw_hi, "raw"),
            (corrected_stack, corr_lo, corr_hi, "bg-corr"),
        )):
            ax = axes[row][col]
            ax.imshow(stack[col], cmap="gray", vmin=lo, vmax=hi, origin="upper")
            ax.axhline(cy, color="#FF2D2D", lw=0.5)
            ax.axvline(cx, color="#FF2D2D", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"p{plane_numbers[col]:02d}{tag}", fontsize=7)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
            colour = HIGHLIGHT_COLOURS[highlight]
            for spine in ax.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(1.8 if highlight != "none" else 0.5)

    def _g(key):
        return candidate.get(key, "")

    fig.suptitle(
        f"{_g('candidate_id')}  |  {_g('channel')}  |  PROVISIONAL "
        f"({_g('preliminary_sampling_category')})\n"
        f"status={_g('current_status')}  reason={_g('preliminary_rule_reason') or '-'}  "
        f"xy=({_g('x_global_px')},{_g('y_global_px')})  z={_g('z_index')}\n"
        f"area={_g('xy_area_um2')}um2  vol={_g('volume_um3')}um3  "
        f"support_planes={_g('support_plane_count')}  voxels={_g('supporting_voxel_count')}  "
        f"S/B(robust_z)={_g('local_robust_z')}  edge={_g('touches_crop_boundary')}",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def export_patches(channel, rows, config, patch_root, section):
    """Export one seven-plane patch per sampled candidate for a single channel.

    Opens the seven section planes once and reuses them for every candidate.
    Returns {candidate_id: relative_patch_path}.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    index = index_channel(channel, _channel_dir(config, channel),
                          config.data.filename_regex)
    ordered = ordered_section_planes(index, section)
    if not ordered:
        print(f"  [{channel}] no TIFF planes for section {section}; patches skipped.")
        return {}

    half_px = patch_half_px(config)
    channel_dir = patch_root / channel
    channel_dir.mkdir(parents=True, exist_ok=True)

    handles, plane_arrays, plane_numbers = _open_plane_memmaps(ordered)
    patch_files: dict[str, str] = {}
    try:
        for candidate in rows:
            raw_stack, cy, cx = _crop_fixed_xy(
                plane_arrays, candidate["x_global_px"], candidate["y_global_px"], half_px,
            )
            raw_f = raw_stack.astype(np.float32)
            corrected = background_correct(
                raw_f, config.acquisition.voxel_size_y_um,
                config.detection.background_sigma_um,
            )
            cid = candidate["candidate_id"]
            stratum = candidate["sampling_stratum"]
            fname = f"{stratum}_{cid}.png"
            save_seven_plane_patch(
                channel_dir / fname, plt, candidate, raw_f, corrected,
                plane_numbers, cy, cx,
            )
            patch_files[cid] = f"{channel}/{fname}"
    finally:
        for tf in handles:
            tf.close()
    print(f"  [{channel}] wrote {len(patch_files)} seven-plane patches -> {channel_dir}")
    return patch_files


# --------------------------------------------------------------------------- #
# Injection-mask QC over the seven planes (green + red)
# --------------------------------------------------------------------------- #
def _low_res_plane(path, factor):
    import tifffile  # noqa: PLC0415

    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        try:
            image = page.asarray(out="memmap")
        except (TypeError, ValueError):
            image = page.asarray()
        return np.asarray(image[::factor, ::factor], dtype=np.float32)


def render_injection_mask_qc(config, section, out_dir):
    """One figure: green mask row + red mask row, each over all seven planes.

    Each panel shows the actual plane (downsampled to the mask grid) with the
    SAME automatic injection mask overlaid (kept region solid, full bright region
    dashed, seeds as x). Overlaying the identical mask on every plane lets the
    reviewer confirm the green mask hugs the bright region across planes and is
    not a fixed circle. This recomputes nothing about the run and writes no run
    files -- it only reads TIFF headers/pixels.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    voxel = config.acquisition.voxel_size_zyx
    regex = config.data.filename_regex
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_info = []
    for channel in CHANNELS:
        index = index_channel(channel, _channel_dir(config, channel), regex)
        plane_paths = {pl: path for (sec, pl), path in index.files.items() if sec == section}
        if not plane_paths:
            print(f"  [mask QC] no planes for {channel} section {section}; skipped.")
            continue
        inj = config.detection.injection_exclusion.for_channel(channel)
        info = _run_channel(channel, plane_paths, inj, voxel, inj.injection_seed_points)
        rows_info.append((channel, plane_paths, info))

    if not rows_info:
        return None

    ncols = 7
    fig, axes = plt.subplots(len(rows_info), ncols,
                             figsize=(2.5 * ncols, 2.7 * len(rows_info)), squeeze=False)
    for r, (channel, plane_paths, info) in enumerate(rows_info):
        stages = info["stages"]
        factor = stages["factor"]
        kept = np.asarray(info["kept_fixed"], dtype=bool)      # final injection mask
        bright = np.asarray(stages["area_filtered"], dtype=bool)  # full bright region
        seeds_low = [(int(y) // factor, int(x) // factor) for (y, x) in info["seeds_local"]]
        planes = sorted(plane_paths)[:ncols]
        for c in range(ncols):
            ax = axes[r][c]
            if c < len(planes):
                low = _low_res_plane(plane_paths[planes[c]], factor)
                lo, hi = np.percentile(low, [1.0, 99.5])
                ax.imshow(low, cmap="gray", vmin=lo, vmax=max(hi, lo + 1),
                          interpolation="nearest")
                ys = np.arange(bright.shape[0]); xs = np.arange(bright.shape[1])
                if bright.any():
                    ax.contour(xs, ys, bright.astype(float), levels=[0.5],
                               colors="#00D9FF", linewidths=0.6, linestyles="dashed")
                if kept.any():
                    colour = "#39FF14" if channel == "green_signal" else "#FF7F0E"
                    ax.contour(xs, ys, kept.astype(float), levels=[0.5],
                               colors=colour, linewidths=1.1)
                for (yl, xl) in seeds_low:
                    ax.scatter([xl], [yl], marker="x", c="red", s=45, linewidths=1.5)
                ax.set_title(f"plane {planes[c]:02d}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                tag = "GREEN mask" if channel == "green_signal" else "RED mask"
                ax.set_ylabel(f"{channel}\n{tag}", fontsize=8)
    fig.suptitle(
        "Injection mask over all seven planes -- solid = kept injection mask, "
        "dashed cyan = full bright region, x = seed.\n"
        "PROVISIONAL mask (not cells): confirm the GREEN mask follows the bright "
        "region across planes and is not a fixed circle.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_path = out_dir / f"green_red_injection_mask_seven_planes_section{section:03d}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  [mask QC] wrote {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def write_sample_summary(out_dir, summary_rows):
    path = out_dir / "validation_sample_summary.csv"
    cols = ["channel", "stratum", "population_count", "requested", "sampled_count",
            "patches_written", "random_seed"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in summary_rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"  wrote {path}")
    return path


def write_fail_reason_counts(out_dir, population_by_channel, sampled_fail_rows):
    """preliminary_fail_reason_counts.csv: for each channel + fail reason, how many
    fails exist in the whole population and how many are in the sampled batch."""
    path = out_dir / "preliminary_fail_reason_counts.csv"
    sampled = Counter(
        (r["channel"], r.get("preliminary_rule_reason", ""))
        for r in sampled_fail_rows
    )
    cols = ["channel", "preliminary_rule_reason", "population_fail_count",
            "sampled_fail_count"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for channel in CHANNELS:
            reasons = population_by_channel.get(channel, Counter())
            for reason, count in reasons.most_common():
                w.writerow({
                    "channel": channel,
                    "preliminary_rule_reason": reason or "(blank)",
                    "population_fail_count": count,
                    "sampled_fail_count": sampled.get((channel, reason), 0),
                })
    print(f"  wrote {path}")
    return path


def write_review_batch(out_dir, batch_rows, patch_files):
    path = out_dir / "validation_review_batch.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_BATCH_COLUMNS)
        w.writeheader()
        for c in batch_rows:
            record = {col: c.get(col, "") for col in REVIEW_BATCH_COLUMNS}
            record["review_patch_file"] = patch_files.get(c["candidate_id"], "")
            record["human_label"] = ""
            record["review_notes"] = ""
            w.writerow(record)
    print(f"  wrote {path} ({len(batch_rows)} rows)")
    return path


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(
        description="Compact validation workflow for the newest section-070 run "
                    "(reviews preliminary-pass strictness; changes nothing).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--section", type=int, default=70)
    p.add_argument("--candidates", default=None,
                   help="all_candidates.csv (default: newest under the runs root).")
    p.add_argument("--runs-root", default=None,
                   help="Runs root searched for the newest run "
                        "(default: <work_dir>/candidates/runs).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <run>/preliminary_validation).")
    p.add_argument("--per-stratum", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260706)
    p.add_argument("--no-patches", action="store_true",
                   help="Skip patch/QC image rendering (CSV summaries only).")
    args = p.parse_args()

    config = load_config(args.config)
    runs_root = Path(args.runs_root or (config.work_dir / "candidates" / "runs"))

    if args.candidates:
        candidates_csv = Path(args.candidates)
    else:
        candidates_csv = find_newest_candidates_csv(runs_root)
        if candidates_csv is None:
            print(f"ERROR: no all_candidates.csv found under {runs_root}")
            return 1

    run_dir = candidates_csv.parent
    out_dir = Path(args.out or (run_dir / "preliminary_validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"Run directory : {run_dir}")
    print(f"Candidates CSV: {candidates_csv}")
    print(f"Section       : {args.section}")
    print(f"Output        : {out_dir}")
    print(f"Sampling      : {args.per_stratum} per stratum, seed {args.seed} "
          "(PROVISIONAL candidates -- never cells)")
    print("=" * 72)

    all_rows = read_csv_rows(candidates_csv)
    if not all_rows:
        print(f"ERROR: {candidates_csv} is empty.")
        return 1

    def category(row):
        return row.get("preliminary_sampling_category") or row.get("current_status")

    batch_rows = []
    summary_rows = []
    population_fail_reasons = {ch: Counter() for ch in CHANNELS}

    # ---- balanced reproducible sample, per channel, kept separate ---- #
    for channel in CHANNELS:
        channel_rows = [r for r in all_rows if r.get("channel") == channel]
        passes = [r for r in channel_rows if category(r) == PASS_CATEGORY]
        fails = [r for r in channel_rows if category(r) == FAIL_CATEGORY]
        population_fail_reasons[channel] = Counter(
            r.get("preliminary_rule_reason", "") for r in fails
        )

        for stratum, pool in ((PASS_CATEGORY, passes), (FAIL_CATEGORY, fails)):
            # Seed varies per (channel, stratum) so the four samples are distinct
            # but each is individually reproducible across processes.
            seed = stratum_seed(args.seed, channel, stratum)
            picked = sample_stratum(pool, seed, args.per_stratum)
            for r in picked:
                r["sampling_stratum"] = stratum
            batch_rows.extend(picked)
            summary_rows.append({
                "channel": channel,
                "stratum": stratum,
                "population_count": len(pool),
                "requested": args.per_stratum,
                "sampled_count": len(picked),
                "patches_written": 0,
                "random_seed": seed,
            })
            print(f"  {channel:16s} {stratum:22s} population={len(pool):5d} "
                  f"sampled={len(picked)}")

    # ---- seven-plane patches (per channel, separate folders) ---- #
    patch_files: dict[str, str] = {}
    patch_root = out_dir / "validation_review_patches"
    if not args.no_patches:
        for channel in CHANNELS:
            rows = [r for r in batch_rows if r["channel"] == channel]
            files = export_patches(channel, rows, config, patch_root, args.section)
            patch_files.update(files)
            for srow in summary_rows:
                if srow["channel"] == channel:
                    srow["patches_written"] = sum(
                        1 for r in rows if r["candidate_id"] in files
                    )

    # ---- CSV artifacts ---- #
    write_review_batch(out_dir, batch_rows, patch_files)
    write_sample_summary(out_dir, summary_rows)
    sampled_fail_rows = [r for r in batch_rows if r["sampling_stratum"] == FAIL_CATEGORY]
    write_fail_reason_counts(out_dir, population_fail_reasons, sampled_fail_rows)

    # ---- injection-mask QC over the seven planes ---- #
    if not args.no_patches:
        render_injection_mask_qc(config, args.section, out_dir / "injection_mask_qc")

    print("=" * 72)
    print(f"Done. Review batch + patches under: {out_dir}")
    print("Reminder: these are PROVISIONAL candidates; a preliminary pass is not a cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
