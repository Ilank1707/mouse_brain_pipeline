#!/usr/bin/env python
"""Build a human-label validation batch for the preliminary-pass rules.

Draws a reproducible, balanced review sample from a completed run's
``all_candidates.csv`` -- green and red sampled SEPARATELY -- and exports
seven-plane review patches plus a ``validation_review_batch.csv`` with a blank
``human_label`` column for a human to fill in. The failure sample is stratified
by ``preliminary_rule_reason`` so every failure mode is represented.

Every candidate here is a PROVISIONAL detection / preliminary-rule pass or fail;
a preliminary pass is never a cell. This script only READS the run and writes new
files under ``--out-dir``; it never modifies the original run, masks or TIFFs.

Every sampled row also carries explicit inverse-probability sampling weights
(``sampling_population_count``, ``sampling_selected_count``, ``sampling_probability``,
``sample_weight``, ``sampling_stratum_id``) plus a descriptive ``spatial_tile`` so
downstream calibration can correctly reweight the balanced stratified sample back
to the full population. The spatial tile is descriptive only and NEVER enters the
inclusion probability (the sampling design is unchanged).

Outputs (under ``--out-dir``):
  validation_review_batch.csv     one row per sampled candidate + blank human_label
  validation_review_patches/      per-candidate seven-plane PNGs, split by channel
  validation_sample_summary.csv    population vs sampled counts per channel/stratum/reason
  validation_coverage.csv          sampled counts by channel/stratum/reason/tile/plane/source

Example (PowerShell):
  python scripts/make_validation_batch.py --config config.yml `
    --run-dir "C:/mouse_brain_work/candidates/runs/section070_20260706_151305" `
    --section 70 --out-dir "C:/mouse_brain_work/candidates/validation_070"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.candidate_detection import background_correct
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.coordinate_exports import peak_optical_plane
from mouse_brain_pipeline.review import read_csv_rows
from mouse_brain_pipeline.review_patches import (
    HIGHLIGHT_COLOURS,
    display_limits,
    ordered_section_planes,
    panel_highlight_class,
    parse_peak_index,
    parse_support_indices,
)
from mouse_brain_pipeline.rule_calibration import (
    WeightVerificationError,
    verify_and_build_weights,
)

CHANNELS = ("green_signal", "channel_2_signal")
PASS_CATEGORY = "preliminary_rule_pass"
FAIL_CATEGORY = "preliminary_rule_fail"
DEFAULT_SAMPLES_PER_STATUS = 100
DEFAULT_RANDOM_SEED = 20260707
DEFAULT_SPATIAL_TILE_SIZE = 1024

# One row per sampled candidate. status / reason / features are copied verbatim
# from the run; human_label and review_notes are left blank for the reviewer.
BATCH_COLUMNS = [
    "candidate_id",
    "channel",
    "sampling_stratum",                    # preliminary_rule_pass / preliminary_rule_fail
    "fail_reason_stratum",                 # the fail reason bucket ("" for passes)
    "sampling_stratum_id",                 # channel|stratum[|reason] -- the full stratum key
    "sampling_population_count",           # candidates in this stratum in the run
    "sampling_selected_count",             # candidates drawn from this stratum
    "sampling_probability",                # selected / population (inclusion probability)
    "sample_weight",                       # population / selected (inverse-probability)
    "spatial_tile",                        # descriptive tile id (NOT part of the probability)
    "peak_optical_plane",                  # 1..7 canonical peak plane (or "")
    "candidate_generation_source",         # raw_stack / injection_suppressed_stack / both
    "current_status",
    "preliminary_sampling_category",
    "preliminary_rule_reason",             # fail reason
    "x_global_px", "y_global_px", "z_index", "optical_plane", "global_z_um", "section",
    "equivalent_diameter_um", "xy_diameter_um", "xy_area_um2",   # size
    "volume_um3",                          # volume
    "support_plane_count",                 # support planes
    "supporting_voxel_count",              # support voxels
    "local_robust_z",                      # intensity / background metric (signal-to-background)
    "local_contrast_score", "peak_intensity",
    "local_background_median", "local_background_noise",
    "elongation",
    "n_consecutive_planes",
    "xy_centroid_shift_um",
    "measurement_valid", "original_cellfinder_z_valid",
    "touches_crop_boundary", "invalid_coordinate", "inside_tissue",   # edge fields
    "inside_injection_analysis_exclusion",
    "review_patch_file",
    "human_label",                         # BLANK -- for the human reviewer
    "review_notes",                        # BLANK
]

SUMMARY_COLUMNS = [
    "channel", "stratum", "preliminary_rule_reason",
    "population_count", "allocated", "sampled_count", "random_seed",
]

# validation_coverage.csv: sampled counts broken down by the descriptive axes the
# reviewer needs to see how the sample spreads (tiles, planes, generation source).
COVERAGE_COLUMNS = [
    "channel", "sampling_stratum", "preliminary_rule_reason", "spatial_tile",
    "peak_optical_plane", "candidate_generation_source", "count",
]


# --------------------------------------------------------------------------- #
# Reproducible, process-stable seeding
# --------------------------------------------------------------------------- #
def _seed(base_seed, *parts):
    """Deterministic seed from ``base_seed`` + string parts (not builtin hash())."""
    digest = hashlib.sha1(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(base_seed) + int(digest, 16) % 1_000_000


def category(row):
    return row.get("preliminary_sampling_category") or row.get("current_status")


# --------------------------------------------------------------------------- #
# Sampling (pure)
# --------------------------------------------------------------------------- #
def sample_passes(passes, n, base_seed, channel):
    """Deterministic random subsample of preliminary passes for one channel."""
    ordered = sorted(passes, key=lambda r: str(r.get("candidate_id", "")))
    if len(ordered) <= n:
        return list(ordered)
    return random.Random(_seed(base_seed, channel, PASS_CATEGORY)).sample(ordered, n)


def allocate_by_reason(sizes, n):
    """Allocate ``n`` failure samples across reason buckets (stratified).

    Every non-empty reason gets at least one slot (representation), then the
    remainder is distributed proportionally to remaining capacity using the
    largest-remainder method. Deterministic. Allocations never exceed a bucket's
    population. If there are more reasons than ``n``, the ``n`` largest get one
    each.
    """
    reasons = sorted(sizes)
    total = sum(sizes.values())
    if total <= n:
        return {r: sizes[r] for r in reasons}

    nonempty = [r for r in reasons if sizes[r] > 0]
    if len(nonempty) > n:
        largest = sorted(nonempty, key=lambda r: (-sizes[r], r))[:n]
        return {r: (1 if r in largest else 0) for r in reasons}

    alloc = {r: (1 if sizes[r] > 0 else 0) for r in reasons}
    remaining = n - sum(alloc.values())
    capacity = {r: sizes[r] - alloc[r] for r in reasons}
    denom = sum(capacity.values())
    if remaining > 0 and denom > 0:
        ideal = {r: remaining * capacity[r] / denom for r in reasons}
        add = {r: min(int(math.floor(ideal[r])), capacity[r]) for r in reasons}
        leftover = remaining - sum(add.values())
        order = sorted(reasons, key=lambda r: (-(ideal[r] - math.floor(ideal[r])), r))
        i = 0
        while leftover > 0 and i < 100_000:
            r = order[i % len(order)]
            if add[r] < capacity[r]:
                add[r] += 1
                leftover -= 1
            i += 1
        for r in reasons:
            alloc[r] += add[r]
    return alloc


def sample_fails(fails, n, base_seed, channel):
    """Reason-stratified deterministic subsample of preliminary fails.

    Returns ``(picked_rows, allocation, sizes)``.
    """
    groups = defaultdict(list)
    for row in fails:
        groups[row.get("preliminary_rule_reason", "") or ""].append(row)
    sizes = {reason: len(rows) for reason, rows in groups.items()}
    allocation = allocate_by_reason(sizes, n)

    picked = []
    for reason in sorted(groups):
        take = allocation.get(reason, 0)
        pool = sorted(groups[reason], key=lambda r: str(r.get("candidate_id", "")))
        if take >= len(pool):
            picked.extend(pool)
        elif take > 0:
            rng = random.Random(_seed(base_seed, channel, FAIL_CATEGORY, reason))
            picked.extend(rng.sample(pool, take))
    return picked, allocation, sizes


def stratum_id(channel, stratum, fail_reason):
    """The full stratum key: ``channel|preliminary_rule_pass`` for passes,
    ``channel|preliminary_rule_fail|<reason>`` for fails (per the task spec)."""
    if stratum == PASS_CATEGORY:
        return f"{channel}|{PASS_CATEGORY}"
    return f"{channel}|{FAIL_CATEGORY}|{fail_reason}"


def spatial_tile(row, tile_size):
    """Descriptive tile id ``tx_ty`` from the global XY pixel coordinates.

    This is a coverage/diagnostic label ONLY -- it never enters the inclusion
    probability (the sampling design is not per-tile). Returns "" if the
    coordinates are missing/unparseable.
    """
    try:
        x = int(round(float(row.get("x_global_px"))))
        y = int(round(float(row.get("y_global_px"))))
    except (TypeError, ValueError):
        return ""
    size = int(tile_size) if int(tile_size) > 0 else DEFAULT_SPATIAL_TILE_SIZE
    return f"{x // size}_{y // size}"


def _peak_optical_plane(row, planes_per_section=7):
    plane = peak_optical_plane(row, planes_per_section)
    return plane if plane is not None else ""


def _batch_record(row, channel, stratum, fail_reason, *, population, selected,
                  tile_size, planes_per_section=7):
    """Build one batch row, annotated with its stratum's inverse-probability weight.

    ``sampling_probability = selected / population`` and
    ``sample_weight = population / selected`` (both 1.0 when the whole stratum was
    taken). The spatial tile is descriptive and is NOT folded into the probability.
    """
    record = {column: row.get(column, "") for column in BATCH_COLUMNS}
    record["sampling_stratum"] = stratum
    record["fail_reason_stratum"] = fail_reason
    record["sampling_stratum_id"] = stratum_id(channel, stratum, fail_reason)
    record["sampling_population_count"] = int(population)
    record["sampling_selected_count"] = int(selected)
    record["sampling_probability"] = (
        round(selected / population, 8) if population > 0 else "")
    record["sample_weight"] = (
        round(population / selected, 8) if selected > 0 else "")
    record["spatial_tile"] = spatial_tile(row, tile_size)
    record["peak_optical_plane"] = _peak_optical_plane(row, planes_per_section)
    record["candidate_generation_source"] = row.get("candidate_generation_source", "")
    record["review_patch_file"] = ""
    record["human_label"] = ""
    record["review_notes"] = ""
    return record


# --------------------------------------------------------------------------- #
# Coverage + validation (pure)
# --------------------------------------------------------------------------- #
def build_coverage_rows(batch_records):
    """Sampled counts by channel/stratum/reason/tile/peak-plane/generation-source."""
    counter = Counter(
        (r["channel"], r["sampling_stratum"], r.get("preliminary_rule_reason", "") or "",
         r["spatial_tile"], str(r["peak_optical_plane"]),
         r.get("candidate_generation_source", "") or "")
        for r in batch_records
    )
    rows = []
    for (channel, stratum, reason, tile, plane, source), count in sorted(counter.items()):
        rows.append({
            "channel": channel, "sampling_stratum": stratum,
            "preliminary_rule_reason": reason, "spatial_tile": tile,
            "peak_optical_plane": plane, "candidate_generation_source": source,
            "count": count,
        })
    return rows


def validate_batch(batch_records, summary_rows):
    """Validate the sampling weights and reconcile the sample with the summary.

    Raises ``ValueError`` if: a candidate_id repeats; any probability is outside
    (0, 1]; any weight is non-finite or < 1; a selected count exceeds its
    population; or the sampled rows cannot be reconciled against
    validation_sample_summary.csv (population/selected counts per stratum).
    """
    ids = [r["candidate_id"] for r in batch_records]
    duplicates = [cid for cid, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise ValueError(f"candidate_id is not unique in the batch: {duplicates[:5]}")

    for r in batch_records:
        cid = r["candidate_id"]
        population = int(r["sampling_population_count"])
        selected = int(r["sampling_selected_count"])
        if selected > population:
            raise ValueError(
                f"{cid}: selected_count ({selected}) exceeds population_count "
                f"({population})")
        prob = float(r["sampling_probability"])
        if not (0.0 < prob <= 1.0):
            raise ValueError(f"{cid}: sampling_probability {prob} not in (0, 1]")
        weight = float(r["sample_weight"])
        if not math.isfinite(weight) or weight < 1.0:
            raise ValueError(f"{cid}: sample_weight {weight} is not finite and >= 1")

    # Every sampled row must reconcile with the summary (population >= selected > 0
    # and the summary's sampled_count equals the batch rows in that stratum). Reuse
    # the calibration verifier so the batch and the calibrator agree exactly.
    try:
        verify_and_build_weights(summary_rows, batch_records)
    except WeightVerificationError as exc:
        raise ValueError(
            f"sampled rows cannot be reconciled with validation_sample_summary.csv: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Seven-plane patch rendering (reads TIFFs only; never mutates them)
# --------------------------------------------------------------------------- #
def _channel_dir(config, channel):
    return (config.data.green_signal_dir if channel == "green_signal"
            else config.data.channel_2_signal_dir)


def _patch_half_px(config):
    return max(8, int(round(config.classifier.patch_size_xy_um
                            / (2 * config.acquisition.voxel_size_y_um))))


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
            target[half_px - (y - y0):half_px - (y - y0) + crop.shape[0],
                   half_px - (x - x0):half_px - (x - x0) + crop.shape[1]] = crop
        out.append(target)
    return np.stack(out), half_px, half_px


def _save_patch(out_path, plt, record, raw_stack, corrected_stack, plane_numbers, cy, cx):
    peak = parse_peak_index(record)
    support = parse_support_indices(record)
    raw_lo, raw_hi = display_limits(raw_stack)
    corr_lo, corr_hi = display_limits(corrected_stack)
    n = raw_stack.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.7 * n, 4.2), squeeze=False)
    for col in range(n):
        highlight = panel_highlight_class(col, peak, support)
        tag = {"peak": " PEAK", "support": " support", "none": ""}[highlight]
        for r, (stack, lo, hi, name) in enumerate((
            (raw_stack, raw_lo, raw_hi, "raw"),
            (corrected_stack, corr_lo, corr_hi, "bg-corr"),
        )):
            ax = axes[r][col]
            ax.imshow(stack[col], cmap="gray", vmin=lo, vmax=hi, origin="upper")
            ax.axhline(cy, color="#FF2D2D", lw=0.5)
            ax.axvline(cx, color="#FF2D2D", lw=0.5)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"p{plane_numbers[col]:02d}{tag}", fontsize=7)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
            colour = HIGHLIGHT_COLOURS[highlight]
            for spine in ax.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(1.8 if highlight != "none" else 0.5)

    def g(key):
        return record.get(key, "")

    fig.suptitle(
        f"{g('candidate_id')} | {g('channel')} | PROVISIONAL "
        f"({g('preliminary_sampling_category')})\n"
        f"status={g('current_status')} reason={g('preliminary_rule_reason') or '-'} "
        f"xy=({g('x_global_px')},{g('y_global_px')}) z={g('z_index')}\n"
        f"area={g('xy_area_um2')}um2 vol={g('volume_um3')}um3 "
        f"planes={g('support_plane_count')} voxels={g('supporting_voxel_count')} "
        f"S/B={g('local_robust_z')} edge={g('touches_crop_boundary')}",
        fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def export_patches(channel, records, config, patch_root, section):
    """Render one seven-plane patch per sampled candidate for one channel."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    index = index_channel(channel, _channel_dir(config, channel), config.data.filename_regex)
    ordered = ordered_section_planes(index, section)
    if not ordered:
        print(f"  [{channel}] no TIFF planes for section {section}; patches skipped.")
        return {}
    half_px = _patch_half_px(config)
    channel_dir = patch_root / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    handles, plane_arrays, plane_numbers = _open_plane_memmaps(ordered)
    patch_files = {}
    try:
        for record in records:
            raw_stack, cy, cx = _crop_fixed_xy(
                plane_arrays, record["x_global_px"], record["y_global_px"], half_px)
            raw_f = raw_stack.astype(np.float32)
            corrected = background_correct(
                raw_f, config.acquisition.voxel_size_y_um,
                config.detection.background_sigma_um)
            fname = f"{record['sampling_stratum']}_{record['candidate_id']}.png"
            _save_patch(channel_dir / fname, plt, record, raw_f, corrected,
                        plane_numbers, cy, cx)
            patch_files[record["candidate_id"]] = f"{channel}/{fname}"
    finally:
        for tf in handles:
            tf.close()
    print(f"  [{channel}] wrote {len(patch_files)} seven-plane patches -> {channel_dir}")
    return patch_files


# --------------------------------------------------------------------------- #
# Orchestration (reads the run; writes only under out_dir)
# --------------------------------------------------------------------------- #
def generate_validation_batch(*, config, run_dir, section, out_dir,
                              samples_per_status=DEFAULT_SAMPLES_PER_STATUS,
                              random_seed=DEFAULT_RANDOM_SEED, render=True,
                              spatial_tile_size=DEFAULT_SPATIAL_TILE_SIZE):
    """Sample, (optionally) render patches, and write batch + summary CSVs.

    Returns a summary dict. The original run is only read, never modified.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    candidates_csv = run_dir / "all_candidates.csv"
    all_rows = read_csv_rows(candidates_csv)
    if not all_rows:
        raise FileNotFoundError(f"No candidates found in {candidates_csv}")
    section_rows = [r for r in all_rows if str(r.get("section")) == str(section)]
    if not section_rows:
        raise ValueError(f"No candidates for section {section} in {candidates_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    planes_per_section = int(getattr(
        getattr(config, "acquisition", None), "planes_per_section", 7) or 7)

    batch_records = []
    summary_rows = []
    for channel in CHANNELS:
        channel_rows = [r for r in section_rows if r.get("channel") == channel]
        passes = [r for r in channel_rows if category(r) == PASS_CATEGORY]
        fails = [r for r in channel_rows if category(r) == FAIL_CATEGORY]

        picked_pass = sample_passes(passes, samples_per_status, random_seed, channel)
        picked_fail, allocation, sizes = sample_fails(
            fails, samples_per_status, random_seed, channel)

        sampled_per_reason = defaultdict(int)
        for row in picked_fail:
            sampled_per_reason[row.get("preliminary_rule_reason", "") or ""] += 1

        for row in picked_pass:
            batch_records.append(_batch_record(
                row, channel, PASS_CATEGORY, "",
                population=len(passes), selected=len(picked_pass),
                tile_size=spatial_tile_size, planes_per_section=planes_per_section))
        for row in picked_fail:
            reason = row.get("preliminary_rule_reason", "") or ""
            batch_records.append(_batch_record(
                row, channel, FAIL_CATEGORY, reason,
                population=sizes.get(reason, 0),
                selected=sampled_per_reason.get(reason, 0),
                tile_size=spatial_tile_size, planes_per_section=planes_per_section))

        summary_rows.append({
            "channel": channel, "stratum": PASS_CATEGORY, "preliminary_rule_reason": "",
            "population_count": len(passes), "allocated": samples_per_status,
            "sampled_count": len(picked_pass),
            "random_seed": _seed(random_seed, channel, PASS_CATEGORY),
        })
        for reason in sorted(sizes):
            summary_rows.append({
                "channel": channel, "stratum": FAIL_CATEGORY,
                "preliminary_rule_reason": reason or "(blank)",
                "population_count": sizes[reason],
                "allocated": allocation.get(reason, 0),
                "sampled_count": sampled_per_reason.get(reason, 0),
                "random_seed": _seed(random_seed, channel, FAIL_CATEGORY, reason),
            })
        print(f"  {channel:16s} passes: pop={len(passes):5d} sampled={len(picked_pass)}  "
              f"fails: pop={len(fails):5d} sampled={len(picked_fail)} "
              f"across {sum(1 for s in sizes.values() if s)} reasons")

    # Validate the sampling weights and reconcile the sample against the summary
    # BEFORE the expensive patch rendering, so an inconsistent batch fails fast.
    validate_batch(batch_records, summary_rows)

    patch_files = {}
    if render:
        patch_root = out_dir / "validation_review_patches"
        for channel in CHANNELS:
            channel_records = [b for b in batch_records if b["channel"] == channel]
            patch_files.update(export_patches(channel, channel_records, config,
                                              patch_root, section))
        for record in batch_records:
            record["review_patch_file"] = patch_files.get(record["candidate_id"], "")

    batch_csv = out_dir / "validation_review_batch.csv"
    with batch_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BATCH_COLUMNS)
        writer.writeheader()
        writer.writerows(batch_records)

    summary_csv = out_dir / "validation_sample_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    coverage_rows = build_coverage_rows(batch_records)
    coverage_csv = out_dir / "validation_coverage.csv"
    with coverage_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COVERAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(coverage_rows)

    return {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "section": section,
        "samples_per_status": samples_per_status,
        "random_seed": random_seed,
        "spatial_tile_size": spatial_tile_size,
        "n_batch_rows": len(batch_records),
        "batch_csv": str(batch_csv),
        "summary_csv": str(summary_csv),
        "coverage_csv": str(coverage_csv),
        "patches_rendered": len(patch_files),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Build a reproducible human-label validation batch "
                    "(PROVISIONAL candidates; reads the run, never modifies it).")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--run-dir", required=True, help="Completed run folder with all_candidates.csv")
    p.add_argument("--section", type=int, default=70)
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: <run-dir>/validation_batch).")
    p.add_argument("--samples-per-status", type=int, default=DEFAULT_SAMPLES_PER_STATUS,
                   help="Passes AND fails sampled per channel (default 100 each).")
    p.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--spatial-tile-size", type=int, default=DEFAULT_SPATIAL_TILE_SIZE,
                   help="Descriptive spatial-tile size in full-res px (default 1024). "
                        "Coverage only -- never part of the inclusion probability.")
    p.add_argument("--no-patches", action="store_true",
                   help="Skip seven-plane patch rendering (CSV only).")
    args = p.parse_args(argv)

    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "validation_batch"

    print("=" * 72)
    print(f"Run directory : {run_dir}")
    print(f"Section       : {args.section}")
    print(f"Output        : {out_dir}")
    print(f"Sampling      : {args.samples_per_status} passes + "
          f"{args.samples_per_status} fails per channel, seed {args.random_seed}")
    print("PROVISIONAL candidates -- never cells. The run is read, never modified.")
    print("=" * 72)

    summary = generate_validation_batch(
        config=config, run_dir=run_dir, section=args.section, out_dir=out_dir,
        samples_per_status=args.samples_per_status, random_seed=args.random_seed,
        render=not args.no_patches, spatial_tile_size=args.spatial_tile_size,
    )
    print("=" * 72)
    print(f"Wrote {summary['n_batch_rows']} rows -> {summary['batch_csv']}")
    print(f"Sample summary  -> {summary['summary_csv']}")
    print(f"Sample coverage -> {summary['coverage_csv']}")
    print("Each row carries sampling_probability + sample_weight (inverse-probability) "
          "so calibration can reweight the balanced sample to the full population.")
    print("Fill the blank 'human_label' column (cell / artefact / uncertain / "
          "injection), then run calibrate_candidate_rules.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
