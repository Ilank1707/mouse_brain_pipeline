#!/usr/bin/env python
"""Compare two coordinate CSVs (e.g. Cellfinder vs. this pipeline) in micrometers.

STANDALONE: this script depends only on numpy + scipy and does NOT import or run
any part of the mouse-brain pipeline. It never modifies pipeline files; it only
reads the two CSVs and writes ``coordinate_comparison_results.csv``.

Usage
-----
    python scripts/compare_coordinate_files.py <cellfinder_csv> <pipeline_csv>

With no paths it uses
``C:\\Users\\saleem_lab\\mouse_brain_pipeline\\cellfinder_candidate_coordinates.csv``
as the Cellfinder file and, for the other file, the most recently modified CSV in
the repository whose filename contains "coordinate" (excluding the Cellfinder file
and this script's own output).

Coordinate columns
------------------
Any one of these schemas is accepted (case-insensitive):
    * ``x`` / ``y`` / ``z``
    * ``x_coord`` / ``y_coord`` / ``z_coord``
    * ``x`` / ``y`` / ``plane``   (plane 1..7 is converted to z 0..6)
    * ``x_global_px`` / ``y_global_px`` / ``fixed_xy_peak_z_index``  (the pipeline
      schema; when ``fixed_xy_peak_z_index`` is unavailable, ``peak_optical_plane``
      minus 1 is used). ``global_z_um`` is deliberately NOT used: it is a global
      physical coordinate, while the Cellfinder CSV uses local section z indices
      0..6.

Coordinates are compared in micrometers using voxel sizes
``x = 1.004``, ``y = 1.004``, ``z = 6.0`` um. Matching is mutual nearest-neighbour
(via ``scipy.spatial.cKDTree``) so two points can never both claim the same match;
a pair counts as similar when within 10 um.

Beyond the primary comparison, four diagnostic tests are reported (each within the
10 um threshold): (1) original coordinates, (2) pipeline z shifted down one plane,
(3) pipeline z shifted up one plane, and (4) XY distance only with the z difference
required to be <= one plane. Counts grouped by the pipeline CSV's
``candidate_generation_source`` column are also printed. The detailed per-coordinate
output CSV is unchanged (it reflects the original 3-D comparison).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# ---- fixed comparison parameters ----------------------------------------- #
VOXEL_X_UM = 1.004
VOXEL_Y_UM = 1.004
VOXEL_Z_UM = 6.0
MATCH_THRESHOLD_UM = 10.0
PLANES_PER_SECTION = 7

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CELLFINDER_CSV = REPO_ROOT / "cellfinder_candidate_coordinates.csv"
OUTPUT_CSV = REPO_ROOT / "coordinate_comparison_results.csv"

# Directories that are never worth walking when auto-discovering a CSV.
_PRUNE_DIRS = {".venv", ".git", ".pytest_cache", "__pycache__", "node_modules", ".mypy_cache"}

# Distance-group bin edges (um). Half-open [lo, hi); the last bin is [20, inf).
_DISTANCE_GROUPS = [
    ("0-3um", 0.0, 3.0),
    ("3-6um", 3.0, 6.0),
    ("6-10um", 6.0, 10.0),
    ("10-20um", 10.0, 20.0),
    ("over_20um", 20.0, float("inf")),
]

DETAIL_COLUMNS = [
    "cellfinder_index",
    "cf_x_um", "cf_y_um", "cf_z_um",
    "nearest_pipeline_index",
    "pipeline_x_um", "pipeline_y_um", "pipeline_z_um",
    "nearest_distance_um",
    "mutual_match",
    "matched_within_10um",
    "distance_group",
]


class CoordinateError(Exception):
    """A user-facing error (missing file, unusable coordinate columns, etc.)."""


# --------------------------------------------------------------------------- #
# CSV loading + column detection
# --------------------------------------------------------------------------- #
def detect_coordinate_columns(fieldnames):
    """Return ``(x_col, y_col, z_spec, schema_label)`` for a supported schema.

    ``z_spec`` is an ordered list of ``(column, offset)`` z sources tried per row
    (first parseable value wins, then ``value + offset`` gives a local section z in
    0..6). The list has more than one entry only for the pipeline schema, whose z
    falls back from ``fixed_xy_peak_z_index`` to ``peak_optical_plane - 1``.

    Raises :class:`CoordinateError` when no supported schema is present, listing
    what was found.
    """
    if not fieldnames:
        raise CoordinateError("file has no header row / no columns.")
    # Map lower-case name -> the actual header spelling, so detection is case-insensitive.
    lookup = {}
    for name in fieldnames:
        key = str(name).strip().lower()
        lookup.setdefault(key, name)

    def col(name):
        return lookup[name]

    def has(*names):
        return all(n in lookup for n in names)

    # Pipeline schema: x_global_px / y_global_px, with z from the LOCAL section
    # peak plane. Prefer fixed_xy_peak_z_index (already 0..6); if it is unavailable
    # fall back to peak_optical_plane - 1 (1..7 -> 0..6). global_z_um is deliberately
    # NOT used: it is a global physical coordinate, whereas the Cellfinder CSV uses
    # local section z indices 0..6.
    if has("x_global_px", "y_global_px") and (
        has("fixed_xy_peak_z_index") or has("peak_optical_plane")
    ):
        z_spec = []
        z_sources = []
        if has("fixed_xy_peak_z_index"):
            z_spec.append((col("fixed_xy_peak_z_index"), 0.0))
            z_sources.append("fixed_xy_peak_z_index")
        if has("peak_optical_plane"):
            z_spec.append((col("peak_optical_plane"), -1.0))
            z_sources.append("peak_optical_plane-1")
        label = f"pipeline x_global_px/y_global_px/z[{' -> '.join(z_sources)}]"
        return col("x_global_px"), col("y_global_px"), z_spec, label

    if has("x_coord", "y_coord", "z_coord"):
        return col("x_coord"), col("y_coord"), [(col("z_coord"), 0.0)], "x_coord/y_coord/z_coord"
    if has("x", "y", "z"):
        return col("x"), col("y"), [(col("z"), 0.0)], "x/y/z"
    if has("x", "y", "plane"):
        return col("x"), col("y"), [(col("plane"), -1.0)], "x/y/plane (plane 1-7 -> z 0-6)"
    raise CoordinateError(
        "no supported coordinate columns found.\n"
        f"    columns present : {list(fieldnames)}\n"
        "    supported schemas: x/y/z, x_coord/y_coord/z_coord, x/y/plane, or "
        "x_global_px/y_global_px/fixed_xy_peak_z_index "
        "(fallback peak_optical_plane-1)"
    )


def _resolve_z(row, z_spec):
    """First parseable ``value + offset`` from the ordered z sources, else ``None``."""
    for column, offset in z_spec:
        raw = row.get(column, "")
        if raw is None:
            continue
        text = str(raw).strip()
        if text == "":
            continue
        try:
            return float(text) + offset
        except ValueError:
            continue
    return None


def load_coordinates(path):
    """Load a CSV into an ``(N, 3)`` float array of ``(x_um, y_um, z_um)``.

    Also returns a short human description of the schema used. Rows with missing
    or non-numeric coordinates are skipped and counted.
    """
    import numpy as np  # noqa: PLC0415

    path = Path(path)
    if not path.is_file():
        raise CoordinateError(f"coordinate file not found: {path}")

    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        x_col, y_col, z_spec, schema = detect_coordinate_columns(fieldnames)
        points = []
        skipped = 0
        for row in reader:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except (TypeError, ValueError, KeyError):
                skipped += 1
                continue
            z = _resolve_z(row, z_spec)   # local section z index 0..6
            if z is None:
                skipped += 1
                continue
            points.append((x * VOXEL_X_UM, y * VOXEL_Y_UM, z * VOXEL_Z_UM))

    array = np.asarray(points, dtype=float).reshape(-1, 3)
    return array, schema, skipped


# --------------------------------------------------------------------------- #
# Auto-discovery of the "other" coordinate CSV
# --------------------------------------------------------------------------- #
def find_latest_coordinate_csv(root, exclude):
    """Most recently modified ``*coordinate*.csv`` under ``root``, or ``None``.

    ``exclude`` is a set of resolved paths to ignore (the Cellfinder file and this
    script's own output). Heavy/irrelevant directories are pruned for speed.
    """
    import os  # noqa: PLC0415

    exclude = {Path(p).resolve() for p in exclude}
    best_path = None
    best_mtime = -1.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        for name in filenames:
            if not name.lower().endswith(".csv") or "coordinate" not in name.lower():
                continue
            candidate = Path(dirpath) / name
            if candidate.resolve() in exclude:
                continue
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if mtime > best_mtime:
                best_mtime, best_path = mtime, candidate
    return best_path


# --------------------------------------------------------------------------- #
# Matching + statistics
# --------------------------------------------------------------------------- #
def mutual_nearest_matches(cellfinder_um, pipeline_um):
    """Mutual nearest-neighbour match between two ``(N, 3)`` um arrays.

    Returns a dict with, per Cellfinder point: nearest pipeline index, nearest
    distance (um) and whether the pair is mutually nearest. A KD-tree is built on
    each side so multiple points can never all collapse onto one shared match.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.spatial import cKDTree  # noqa: PLC0415

    n_cf = len(cellfinder_um)
    if n_cf == 0 or len(pipeline_um) == 0:
        return {
            "nearest_index": np.full(n_cf, -1, dtype=int),
            "nearest_distance": np.full(n_cf, np.inf, dtype=float),
            "mutual": np.zeros(n_cf, dtype=bool),
        }

    tree_pipeline = cKDTree(pipeline_um)
    tree_cellfinder = cKDTree(cellfinder_um)
    dist_cf_to_pipe, idx_cf_to_pipe = tree_pipeline.query(cellfinder_um, k=1)
    _dist_pipe_to_cf, idx_pipe_to_cf = tree_cellfinder.query(pipeline_um, k=1)

    # Mutual: cellfinder i -> pipeline j AND pipeline j -> cellfinder i.
    mutual = idx_pipe_to_cf[idx_cf_to_pipe] == np.arange(n_cf)
    return {
        "nearest_index": np.asarray(idx_cf_to_pipe, dtype=int),
        "nearest_distance": np.asarray(dist_cf_to_pipe, dtype=float),
        "mutual": np.asarray(mutual, dtype=bool),
    }


def distance_group(distance):
    for label, lo, hi in _DISTANCE_GROUPS:
        if lo <= distance < hi:
            return label
    return _DISTANCE_GROUPS[-1][0]


def group_counts(distances):
    import numpy as np  # noqa: PLC0415

    counts = {label: 0 for label, _lo, _hi in _DISTANCE_GROUPS}
    for distance in np.asarray(distances, dtype=float):
        counts[distance_group(float(distance))] += 1
    return counts


# --------------------------------------------------------------------------- #
# Diagnostic comparisons (z-shift + XY-only)
# --------------------------------------------------------------------------- #
def matched_within(cellfinder_space, pipeline_space, *, threshold=MATCH_THRESHOLD_UM,
                   cf_z_um=None, pipe_z_um=None, max_z_diff_um=None):
    """Mutual-NN matches within ``threshold`` in the given coordinate space.

    ``cellfinder_space`` / ``pipeline_space`` are the coordinates the
    nearest-neighbour distance is measured in -- 3-D for the z-shift tests, 2-D XY
    for the XY-only test. When ``cf_z_um`` / ``pipe_z_um`` / ``max_z_diff_um`` are
    given, a match additionally requires ``|z_cf - z_pipe[match]| <= max_z_diff_um``.
    Returns ``(n_matched, matched_distances)``.
    """
    import numpy as np  # noqa: PLC0415

    match = mutual_nearest_matches(cellfinder_space, pipeline_space)
    distance = match["nearest_distance"]
    nearest_index = match["nearest_index"]
    matched = match["mutual"] & (distance <= threshold)
    if max_z_diff_um is not None and cf_z_um is not None and pipe_z_um is not None:
        cf_z_um = np.asarray(cf_z_um, dtype=float)
        pipe_z_um = np.asarray(pipe_z_um, dtype=float)
        z_diff = np.full(len(cf_z_um), np.inf)
        valid = nearest_index >= 0
        z_diff[valid] = np.abs(cf_z_um[valid] - pipe_z_um[nearest_index[valid]])
        matched = matched & (z_diff <= max_z_diff_um + 1e-9)
    return int(matched.sum()), distance[matched]


def compute_diagnostics(cellfinder_um, pipeline_um):
    """Run the four diagnostic match tests (all within the 10 um threshold).

    Returns a list of ``{name, n_matched, distances}`` dicts, in order:
      1. original coordinates (3-D);
      2. pipeline z shifted DOWN one plane (3-D);
      3. pipeline z shifted UP one plane (3-D);
      4. XY distance only, requiring the z difference <= one plane.
    """
    import numpy as np  # noqa: PLC0415

    cellfinder_um = np.asarray(cellfinder_um, dtype=float).reshape(-1, 3)
    pipeline_um = np.asarray(pipeline_um, dtype=float).reshape(-1, 3)

    shifted_down = pipeline_um.copy()
    shifted_down[:, 2] -= VOXEL_Z_UM     # pipeline z one plane lower
    shifted_up = pipeline_um.copy()
    shifted_up[:, 2] += VOXEL_Z_UM       # pipeline z one plane higher

    tests = [
        ("1. original coordinates (3-D)",
         matched_within(cellfinder_um, pipeline_um)),
        ("2. pipeline z shifted DOWN 1 plane (3-D)",
         matched_within(cellfinder_um, shifted_down)),
        ("3. pipeline z shifted UP 1 plane (3-D)",
         matched_within(cellfinder_um, shifted_up)),
        ("4. XY only, z diff <= 1 plane",
         matched_within(cellfinder_um[:, :2], pipeline_um[:, :2],
                        cf_z_um=cellfinder_um[:, 2], pipe_z_um=pipeline_um[:, 2],
                        max_z_diff_um=VOXEL_Z_UM)),
    ]
    return [
        {"name": name, "n_matched": n_matched, "distances": distances}
        for name, (n_matched, distances) in tests
    ]


def count_generation_sources(path):
    """Tally ``candidate_generation_source`` in a CSV, or ``None`` if it is absent."""
    from collections import Counter  # noqa: PLC0415

    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        source_col = None
        for name in (reader.fieldnames or []):
            if str(name).strip().lower() == "candidate_generation_source":
                source_col = name
                break
        if source_col is None:
            return None
        counts = Counter()
        for row in reader:
            value = str(row.get(source_col, "")).strip() or "(unspecified)"
            counts[value] += 1
    return counts


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_detailed_results(path, cellfinder_um, pipeline_um, match):
    import numpy as np  # noqa: PLC0415

    nearest_index = match["nearest_index"]
    nearest_distance = match["nearest_distance"]
    mutual = match["mutual"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        for i in range(len(cellfinder_um)):
            j = int(nearest_index[i])
            has_match = j >= 0 and np.isfinite(nearest_distance[i])
            pipe = pipeline_um[j] if has_match else (None, None, None)
            matched = bool(mutual[i] and nearest_distance[i] <= MATCH_THRESHOLD_UM)
            writer.writerow({
                "cellfinder_index": i,
                "cf_x_um": round(float(cellfinder_um[i][0]), 4),
                "cf_y_um": round(float(cellfinder_um[i][1]), 4),
                "cf_z_um": round(float(cellfinder_um[i][2]), 4),
                "nearest_pipeline_index": j if has_match else "",
                "pipeline_x_um": round(float(pipe[0]), 4) if has_match else "",
                "pipeline_y_um": round(float(pipe[1]), 4) if has_match else "",
                "pipeline_z_um": round(float(pipe[2]), 4) if has_match else "",
                "nearest_distance_um": (round(float(nearest_distance[i]), 4)
                                        if has_match else ""),
                "mutual_match": bool(mutual[i]),
                "matched_within_10um": matched,
                "distance_group": (distance_group(float(nearest_distance[i]))
                                   if has_match else ""),
            })


def _pct(part, whole):
    return (100.0 * part / whole) if whole else 0.0


def run_comparison(cellfinder_path, pipeline_path):
    import numpy as np  # noqa: PLC0415

    cellfinder_um, cf_schema, cf_skipped = load_coordinates(cellfinder_path)
    pipeline_um, pipe_schema, pipe_skipped = load_coordinates(pipeline_path)

    match = mutual_nearest_matches(cellfinder_um, pipeline_um)
    nearest_distance = match["nearest_distance"]
    mutual = match["mutual"]

    matched_mask = mutual & (nearest_distance <= MATCH_THRESHOLD_UM)
    n_matched = int(matched_mask.sum())
    matched_distances = nearest_distance[matched_mask]

    finite_nearest = nearest_distance[np.isfinite(nearest_distance)]
    groups = group_counts(finite_nearest)

    write_detailed_results(OUTPUT_CSV, cellfinder_um, pipeline_um, match)

    # ---- report ----
    print("=" * 70)
    print("COORDINATE COMPARISON (micrometers; mutual nearest-neighbour)")
    print("=" * 70)
    print(f"Cellfinder file : {cellfinder_path}")
    print(f"                  detected schema: {cf_schema}")
    print(f"                  {len(cellfinder_um)} coordinates"
          + (f"; {cf_skipped} rows skipped" if cf_skipped else ""))
    print(f"Pipeline file   : {pipeline_path}")
    print(f"                  detected schema: {pipe_schema}")
    print(f"                  {len(pipeline_um)} coordinates"
          + (f"; {pipe_skipped} rows skipped" if pipe_skipped else ""))
    print("-" * 70)
    print(f"voxel sizes (um): x={VOXEL_X_UM}, y={VOXEL_Y_UM}, z={VOXEL_Z_UM}")
    print(f"match threshold : {MATCH_THRESHOLD_UM:.0f} um (mutual nearest neighbour)")
    print("-" * 70)
    print(f"matched pairs   : {n_matched}")
    print(f"  % of Cellfinder ({len(cellfinder_um)}): {_pct(n_matched, len(cellfinder_um)):.1f}%")
    print(f"  % of pipeline  ({len(pipeline_um)}): {_pct(n_matched, len(pipeline_um)):.1f}%")
    if n_matched:
        print(f"matched distance: median {np.median(matched_distances):.3f} um; "
              f"95th pct {np.percentile(matched_distances, 95):.3f} um")
    else:
        print("matched distance: n/a (no matches within threshold)")
    print("-" * 70)
    print("Nearest-neighbour distance groups (each Cellfinder coord -> nearest pipeline coord):")
    total_nn = len(finite_nearest)
    for label, _lo, _hi in _DISTANCE_GROUPS:
        count = groups[label]
        print(f"  {label:10}: {count:6d}  ({_pct(count, total_nn):5.1f}%)")
    print("-" * 70)

    # Diagnostic comparisons: original, z shifted +/- one plane, and XY-only with a
    # one-plane z tolerance. Each reports matches within the 10 um threshold.
    diagnostics = compute_diagnostics(cellfinder_um, pipeline_um)
    n_cf = len(cellfinder_um)
    print(f"Diagnostic comparisons (mutual nearest neighbour, matched within "
          f"{MATCH_THRESHOLD_UM:.0f} um; % of Cellfinder coords):")
    for result in diagnostics:
        n_hit = result["n_matched"]
        distances = result["distances"]
        line = (f"  {result['name']}\n"
                f"      matched {n_hit}  ({_pct(n_hit, n_cf):.1f}% of Cellfinder)")
        if n_hit:
            line += (f"; median {np.median(distances):.3f} um; "
                     f"95th pct {np.percentile(distances, 95):.3f} um")
        else:
            line += "; median n/a; 95th pct n/a"
        print(line)
    print("-" * 70)

    # Pipeline candidate-generation-source breakdown.
    source_counts = count_generation_sources(pipeline_path)
    print("Pipeline candidate_generation_source counts:")
    if source_counts is None:
        print("  (no candidate_generation_source column in the pipeline CSV)")
    else:
        for source, count in sorted(source_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {source:32}: {count}")
    print("-" * 70)
    print(f"detailed results: {OUTPUT_CSV}")
    print("=" * 70)

    return {
        "n_cellfinder": len(cellfinder_um),
        "n_pipeline": len(pipeline_um),
        "n_matched": n_matched,
        "groups": groups,
        "diagnostics": diagnostics,
        "generation_sources": source_counts,
        "output_csv": OUTPUT_CSV,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_paths(cellfinder_arg, pipeline_arg):
    """Resolve the two CSV paths, applying the documented defaults."""
    cellfinder_path = Path(cellfinder_arg) if cellfinder_arg else DEFAULT_CELLFINDER_CSV
    if not cellfinder_path.is_file():
        raise CoordinateError(f"Cellfinder file not found: {cellfinder_path}")

    if pipeline_arg:
        pipeline_path = Path(pipeline_arg)
        if not pipeline_path.is_file():
            raise CoordinateError(f"pipeline file not found: {pipeline_path}")
        return cellfinder_path, pipeline_path

    pipeline_path = find_latest_coordinate_csv(
        REPO_ROOT, exclude={cellfinder_path, OUTPUT_CSV})
    if pipeline_path is None:
        raise CoordinateError(
            "no other coordinate CSV found in the repository "
            f"({REPO_ROOT}).\n    Provide it explicitly:\n"
            "    python scripts/compare_coordinate_files.py <cellfinder_csv> <pipeline_csv>"
        )
    print(f"Auto-selected pipeline CSV (most recent *coordinate*.csv): {pipeline_path}")
    return cellfinder_path, pipeline_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two coordinate CSVs in micrometers (mutual nearest neighbour).")
    parser.add_argument("cellfinder_csv", nargs="?", default=None,
                        help="Cellfinder coordinate CSV (default: "
                             "cellfinder_candidate_coordinates.csv in the repo root).")
    parser.add_argument("pipeline_csv", nargs="?", default=None,
                        help="Pipeline coordinate CSV (default: most recent *coordinate*.csv "
                             "in the repo).")
    args = parser.parse_args(argv)

    try:
        cellfinder_path, pipeline_path = resolve_paths(args.cellfinder_csv, args.pipeline_csv)
        run_comparison(cellfinder_path, pipeline_path)
    except CoordinateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
