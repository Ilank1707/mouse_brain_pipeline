"""Peak-plane assignment, simplified coordinate exports and count summaries.

Candidates are **3D objects**. A candidate may be visible across several
neighbouring optical planes, but the authoritative count is the number of unique
rows in the run's candidate table -- a candidate is never counted twice.

For the seven main count images each candidate is assigned to exactly **one**
plane: its canonical peak plane ``fixed_xy_peak_z_index`` (0-based, where 0 -> the
1-based ``optical_plane`` 01 and 6 -> 07). Candidates whose peak index is missing
or invalid are placed in ``unassigned_peak_plane.csv`` rather than guessed, and

    sum(assigned to planes 01..07) + unassigned == unique candidate total.

The support visualisation (a candidate on every supported plane) is kept
separately and is explicitly NOT a count.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Statuses (kept as literals so this module stays import-light).
STATUS_PRELIMINARY_PASS = "preliminary_rule_pass"
STATUS_PRELIMINARY_FAIL = "preliminary_rule_fail"
STATUS_MANUAL_REVIEW = "manual_review"
STATUS_INVALID_MEASUREMENT = "invalid_measurement"
STATUS_SUSPECT_INJECTION = "suspect_injection_mask"
STATUS_INJECTION = "injection_site"
STATUS_ARTIFACT = "artifact"
STATUS_PREDICTED_CELL = "predicted_cell"

SIMPLE_COLUMNS = [
    "candidate_id",
    "channel",
    "section",
    "x_global_px",
    "y_global_px",
    "fixed_xy_peak_z_index",
    "peak_optical_plane",
    "global_z_um",
    "current_status",
    "preliminary_rule_reason",
    "rejection_reason",
    "inside_injection_site",
    "measurement_valid",
    "manual_label",
    "classifier_probability",
    "model_validation_passed",
    "included_in_count",
    "candidate_generation_source",
]


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


# --------------------------------------------------------------------------- #
# Peak-plane assignment
# --------------------------------------------------------------------------- #
def peak_zero_based_index(candidate: dict):
    """Validated 0-based peak Z index, or ``None`` if missing/invalid (never guessed)."""
    raw = candidate.get("fixed_xy_peak_z_index", "")
    if raw in ("", None):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != int(value):
        return None
    return int(value)


def peak_optical_plane(candidate: dict, planes_per_section: int = 7):
    """One-based optical plane (1..7) from the 0-based peak Z index, or ``None``.

    Zero-based index 0 maps to optical plane 01; index 6 maps to plane 07.
    """
    index = peak_zero_based_index(candidate)
    if index is None or not (0 <= index < planes_per_section):
        return None
    return index + 1


def assign_peak_planes(candidates, planes_per_section: int = 7):
    """Split candidates by canonical peak plane.

    Returns ``(assignments, unassigned)`` where ``assignments`` maps a one-based
    optical plane (1..7) to the candidates assigned there, and ``unassigned`` is
    the list with a missing/invalid peak Z. Each candidate appears exactly once.
    """
    assignments = {plane: [] for plane in range(1, planes_per_section + 1)}
    unassigned = []
    for candidate in candidates:
        plane = peak_optical_plane(candidate, planes_per_section)
        if plane is None:
            unassigned.append(candidate)
        else:
            assignments[plane].append(candidate)
    return assignments, unassigned


def support_optical_planes(candidate: dict, planes_per_section: int = 7) -> list[int]:
    """One-based optical planes a candidate is supported on (may be several)."""
    raw = candidate.get(
        "fixed_xy_support_z_indices",
        candidate.get("support_z_indices", candidate.get("z_indices", "")),
    )
    planes = []
    for token in str(raw).replace(",", ";").split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            index = int(float(token))
        except (TypeError, ValueError):
            continue
        if 0 <= index < planes_per_section:
            planes.append(index + 1)
    return sorted(set(planes))


def is_confirmed_cell(candidate: dict) -> bool:
    """A countable confirmed cell: a human ``cell`` label, or a validated model
    prediction. A preliminary-rule-pass status is NEVER a confirmation on its own.
    """
    if str(candidate.get("manual_label", "")).strip().lower() == "cell":
        return True
    if _truthy(candidate.get("model_validation_passed")) and \
            candidate.get("current_status") == STATUS_PREDICTED_CELL:
        return True
    return False


# --------------------------------------------------------------------------- #
# Reconciliation
# --------------------------------------------------------------------------- #
def reconcile(candidates, planes_per_section: int = 7) -> dict:
    """Exact peak-plane + status reconciliation for a channel/section."""
    from collections import Counter

    assignments, unassigned = assign_peak_planes(candidates, planes_per_section)
    per_plane = {plane: len(rows) for plane, rows in assignments.items()}
    assigned_total = sum(per_plane.values())
    unique_total = len(candidates)
    status_counts = dict(Counter(c.get("current_status") for c in candidates))
    return {
        "unique_total": unique_total,
        "assigned_total": assigned_total,
        "unassigned_total": len(unassigned),
        "per_plane_assigned": per_plane,
        "status_counts": status_counts,
        "peak_assignment_reconciles": assigned_total + len(unassigned) == unique_total,
        "status_reconciles": sum(status_counts.values()) == unique_total,
    }


# --------------------------------------------------------------------------- #
# Simplified rows
# --------------------------------------------------------------------------- #
def simplify_row(candidate: dict, planes_per_section: int = 7) -> dict:
    plane = peak_optical_plane(candidate, planes_per_section)
    row = {column: candidate.get(column, "") for column in SIMPLE_COLUMNS}
    row["peak_optical_plane"] = plane if plane is not None else ""
    return row


def _write_simple(path: Path, rows, planes_per_section: int) -> int:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SIMPLE_COLUMNS)
        writer.writeheader()
        for candidate in rows:
            writer.writerow(simplify_row(candidate, planes_per_section))
    return len(rows)


def _status_subsets(candidates) -> dict:
    """The status-based coordinate subsets (mutually-defined filters)."""
    def by_status(status):
        return [c for c in candidates if c.get("current_status") == status]

    def by_sampling(category):
        return [c for c in candidates
                if c.get("preliminary_sampling_category") == category]

    return {
        "preliminary_pass_coordinates.csv": by_sampling(STATUS_PRELIMINARY_PASS),
        "preliminary_fail_coordinates.csv": by_sampling(STATUS_PRELIMINARY_FAIL),
        "manual_review_coordinates.csv": by_status(STATUS_MANUAL_REVIEW),
        "invalid_measurement_coordinates.csv": by_status(STATUS_INVALID_MEASUREMENT),
        "suspect_injection_coordinates.csv": by_status(STATUS_SUSPECT_INJECTION),
        "confirmed_injection_coordinates.csv": by_status(STATUS_INJECTION),
        "cellfinder_artifact_coordinates.csv": by_status(STATUS_ARTIFACT),
    }


def write_coordinate_exports(export_dir, candidates, *, channel, section,
                             planes_per_section: int = 7) -> dict:
    """Write the simplified per-category coordinate CSVs + a summary.

    Returns ``{filename: row_count}``. ``confirmed_cell_coordinates.csv`` holds
    only human ``cell`` labels or validated model predictions -- never bare
    preliminary-rule passes.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    _assignments, unassigned = assign_peak_planes(candidates, planes_per_section)
    counts: dict[str, int] = {}

    counts["all_candidate_coordinates.csv"] = _write_simple(
        export_dir / "all_candidate_coordinates.csv", candidates, planes_per_section)
    for filename, rows in _status_subsets(candidates).items():
        counts[filename] = _write_simple(export_dir / filename, rows, planes_per_section)
    counts["confirmed_cell_coordinates.csv"] = _write_simple(
        export_dir / "confirmed_cell_coordinates.csv",
        [c for c in candidates if is_confirmed_cell(c)], planes_per_section)
    counts["unassigned_peak_plane.csv"] = _write_simple(
        export_dir / "unassigned_peak_plane.csv", unassigned, planes_per_section)

    counts["plane_assignment_summary.csv"] = _write_plane_assignment_summary(
        export_dir / "plane_assignment_summary.csv", candidates,
        channel=channel, section=section, planes_per_section=planes_per_section)

    # A manifest of every export file and its row count.
    summary_path = export_dir / "coordinate_export_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "row_count"])
        for filename, count in counts.items():
            writer.writerow([filename, count])
    return counts


def _write_plane_assignment_summary(path, candidates, *, channel, section,
                                    planes_per_section) -> int:
    assignments, unassigned = assign_peak_planes(candidates, planes_per_section)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "section", "optical_plane", "assigned_count"])
        for plane in range(1, planes_per_section + 1):
            writer.writerow([channel, section, f"{plane:02d}", len(assignments[plane])])
            rows += 1
        writer.writerow([channel, section, "unassigned", len(unassigned)])
        rows += 1
    return rows


# --------------------------------------------------------------------------- #
# Count summaries (Section 8)
# --------------------------------------------------------------------------- #
def write_count_summaries(out_dir, candidates, *, channel, section,
                          planes_per_section: int = 7) -> dict:
    """Write the three reconciling count summaries; return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = reconcile(candidates, planes_per_section)
    assignments, unassigned = assign_peak_planes(candidates, planes_per_section)

    # 1. Unique status summary (sums to the unique candidate total).
    stack_path = out_dir / "stack_unique_status_summary.csv"
    with open(stack_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "section", "current_status", "candidate_count",
                         "unique_total", "reconciles"])
        for status, count in sorted(rec["status_counts"].items()):
            writer.writerow([channel, section, status, count,
                             rec["unique_total"], rec["status_reconciles"]])

    # 2. Peak-assignment summary, one row per (plane, status). Sums (with the
    #    explicit unassigned rows) to the unique total -- never double counts.
    peak_path = out_dir / "plane_peak_assignment_summary.csv"
    with open(peak_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "section", "optical_plane", "current_status",
                         "assigned_count", "note"])
        for plane in range(1, planes_per_section + 1):
            from collections import Counter

            for status, count in sorted(Counter(
                c.get("current_status") for c in assignments[plane]
            ).items()):
                writer.writerow([channel, section, f"{plane:02d}", status, count,
                                 "unique_peak_assignment"])
        from collections import Counter

        for status, count in sorted(Counter(
            c.get("current_status") for c in unassigned
        ).items()):
            writer.writerow([channel, section, "unassigned", status, count,
                             "missing_or_invalid_peak_z"])

    # 3. Support-visualisation summary -- MAY sum to more than the unique total.
    support_path = out_dir / "plane_support_visualization_summary.csv"
    with open(support_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel", "section", "optical_plane",
                         "candidates_visible_on_plane", "warning"])
        warning = "support visualisation only -- candidates can appear on multiple " \
                  "planes; DO NOT SUM these counts as unique cells"
        for plane in range(1, planes_per_section + 1):
            visible = sum(
                1 for c in candidates
                if plane in support_optical_planes(c, planes_per_section)
            )
            writer.writerow([channel, section, f"{plane:02d}", visible, warning])

    return {
        "stack_unique_status_summary": stack_path,
        "plane_peak_assignment_summary": peak_path,
        "plane_support_visualization_summary": support_path,
        "reconciliation": rec,
    }
