"""Manual reference-point persistence and candidate-generation recall matching."""

from __future__ import annotations

import csv
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

REFERENCE_COLUMNS = [
    "reference_id",
    "channel",
    "section",
    "x_global_px",
    "y_global_px",
    "z_index",
    "optical_plane",
    "reviewer",
    "annotation_timestamp",
    "source_crop",
]


def read_reference_points(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_reference_points_atomic(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REFERENCE_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in REFERENCE_COLUMNS}
            for row in rows
        )
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)
    return path


def make_reference_point(
    *,
    channel,
    section,
    x_global_px,
    y_global_px,
    z_index,
    optical_plane,
    reviewer,
    source_crop,
) -> dict:
    return {
        "reference_id": uuid.uuid4().hex,
        "channel": channel,
        "section": int(section),
        "x_global_px": int(round(x_global_px)),
        "y_global_px": int(round(y_global_px)),
        "z_index": int(z_index),
        "optical_plane": int(optical_plane),
        "reviewer": reviewer,
        "annotation_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_crop": source_crop,
    }


def _float(row, *keys):
    for key in keys:
        value = row.get(key, "")
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return math.nan


def match_reference_points(
    references: list[dict],
    candidates: list[dict],
    *,
    voxel_size_y_um: float,
    voxel_size_x_um: float,
    voxel_size_z_um: float,
    xy_tolerance_um: float,
    z_tolerance_um: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """One-to-one nearest matching within independent XY and Z tolerances."""
    possible = []
    for reference_index, reference in enumerate(references):
        for candidate_index, candidate in enumerate(candidates):
            if reference.get("channel") != candidate.get("channel"):
                continue
            if str(reference.get("section")) != str(candidate.get("section")):
                continue
            dx_um = (
                _float(reference, "x_global_px") - _float(candidate, "x_global_px")
            ) * voxel_size_x_um
            dy_um = (
                _float(reference, "y_global_px") - _float(candidate, "y_global_px")
            ) * voxel_size_y_um
            dz_um = (
                _float(reference, "z_index")
                - _float(
                    candidate,
                    "cellfinder_z_index",
                    "original_cellfinder_z_index",
                    "cellfinder_z",
                    "z_index",
                )
            ) * voxel_size_z_um
            xy_um = math.hypot(dx_um, dy_um)
            abs_z_um = abs(dz_um)
            if xy_um <= xy_tolerance_um and abs_z_um <= z_tolerance_um:
                normalized = math.hypot(
                    xy_um / max(xy_tolerance_um, 1e-9),
                    abs_z_um / max(z_tolerance_um, 1e-9),
                )
                possible.append((
                    normalized, xy_um, abs_z_um, reference_index, candidate_index
                ))

    used_references = set()
    used_candidates = set()
    matches = []
    for _score, xy_um, z_um, reference_index, candidate_index in sorted(possible):
        if reference_index in used_references or candidate_index in used_candidates:
            continue
        used_references.add(reference_index)
        used_candidates.add(candidate_index)
        reference = references[reference_index]
        candidate = candidates[candidate_index]
        matches.append({
            **reference,
            "matched_candidate_id": candidate.get("candidate_id", ""),
            "candidate_x_global_px": candidate.get("x_global_px", ""),
            "candidate_y_global_px": candidate.get("y_global_px", ""),
            "candidate_cellfinder_z_index": candidate.get(
                "cellfinder_z_index",
                candidate.get("original_cellfinder_z_index", candidate.get("cellfinder_z", "")),
            ),
            "xy_distance_um": round(xy_um, 4),
            "z_distance_um": round(z_um, 4),
        })

    unmatched_references = [
        reference for index, reference in enumerate(references)
        if index not in used_references
    ]
    unmatched_candidates = [
        candidate for index, candidate in enumerate(candidates)
        if index not in used_candidates
    ]
    return matches, unmatched_references, unmatched_candidates


def _truthy(value, default=False) -> bool:
    text = str(value).strip().lower()
    if text == "":
        return default
    return text in {"true", "1", "yes"}


def evaluate_recall_by_source(
    references: list[dict],
    candidates: list[dict],
    *,
    voxel_size_y_um: float,
    voxel_size_x_um: float,
    voxel_size_z_um: float,
    xy_tolerance_um: float,
    z_tolerance_um: float,
) -> dict:
    """Recall against manual references for the raw, suppressed and union passes.

    Recall is ONLY reported when manual reference points exist. Unmatched
    candidates are never called false positives (they were not manually
    reviewed). Legacy candidates with no provenance are treated as raw-pass.
    """
    summary = {
        "has_references": bool(references),
        "n_references": len(references),
        "n_candidates": len(candidates),
        "xy_tolerance_um": xy_tolerance_um,
        "z_tolerance_um": z_tolerance_um,
    }
    if not references:
        summary["limitations"] = (
            "No recall reported because no manual reference annotations exist."
        )
        return summary

    subsets = {
        "raw_pass": [
            c for c in candidates
            if _truthy(c.get("detected_on_raw_stack", ""), default=True)
        ],
        "suppressed_pass": [
            c for c in candidates
            if _truthy(c.get("detected_on_injection_suppressed_stack", ""), default=False)
        ],
        "union": list(candidates),
    }
    summary["by_source"] = {}
    for name, subset in subsets.items():
        matches, unmatched_references, unmatched_candidates = match_reference_points(
            references, subset,
            voxel_size_y_um=voxel_size_y_um,
            voxel_size_x_um=voxel_size_x_um,
            voxel_size_z_um=voxel_size_z_um,
            xy_tolerance_um=xy_tolerance_um,
            z_tolerance_um=z_tolerance_um,
        )
        summary["by_source"][name] = {
            "candidates_considered": len(subset),
            "matched_references": len(matches),
            "unmatched_references": len(unmatched_references),
            "recall": len(matches) / len(references),
            "unmatched_candidates_not_called_false_positives": len(unmatched_candidates),
        }
    summary["limitations"] = (
        "Recall applies only to the manually annotated reference sample; unmatched "
        "candidates were NOT labelled false positives."
    )
    return summary
