"""Manual candidate review helpers with atomic, immediately durable labels."""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path

# Exact on-disk schema for manual_labels.csv. Kept in this fixed order so the
# file is stable across sessions and resumes cleanly (see load_manual_labels).
MANUAL_LABEL_COLUMNS = [
    "candidate_id",
    "channel",
    "section",
    "x_global_px",
    "y_global_px",
    "z_index",
    "manual_label",
    "reviewer",
    "timestamp",
]

VALID_MANUAL_LABELS = {"cell", "artefact", "injection", "uncertain"}

# Keyboard label keys shared by the reviewer UI and tests.
LABEL_KEYS = {"1": "cell", "2": "artefact", "3": "injection", "4": "uncertain"}


class LabelConflictError(RuntimeError):
    """Raised when a reviewer would silently replace an existing label."""


def read_csv_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_manual_labels(path: str | Path) -> dict[tuple[str, str], dict]:
    """Labels keyed by ``(candidate_id, channel)``.

    The channel is part of the key so the two biological signal channels are
    reviewed and resumed completely independently: labelling ``c1`` in
    ``green_signal`` never marks ``c1`` in ``channel_2_signal`` as reviewed.
    """
    return {
        (row.get("candidate_id", ""), row.get("channel", "")): row
        for row in read_csv_rows(path)
    }


def previous_label(labels: dict, candidate: dict) -> str | None:
    """Existing manual label for a candidate (shown when revisiting), or None."""
    row = labels.get((candidate.get("candidate_id", ""), candidate.get("channel", "")))
    return row.get("manual_label") if row else None


def unreviewed_candidates(candidates: list[dict], labels: dict) -> list[dict]:
    """Candidates without a saved label -- used to resume a review session."""
    return [
        candidate for candidate in candidates
        if (candidate.get("candidate_id", ""), candidate.get("channel", "")) not in labels
    ]


def _candidate_z_index(candidate: dict) -> object:
    """Best-available integer Z index for the saved label row."""
    for key in ("z_index", "fixed_xy_peak_z_index", "cellfinder_z_index"):
        value = candidate.get(key, "")
        if value not in ("", None):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                continue
    return ""


def save_manual_label(
    path: str | Path,
    candidate: dict,
    manual_label: str,
    reviewer: str,
    *,
    allow_overwrite: bool = False,
) -> dict:
    """Atomically upsert one label so every keypress survives a restart."""
    if manual_label not in VALID_MANUAL_LABELS:
        raise ValueError(f"Invalid manual label: {manual_label!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(path)
    record = {
        "candidate_id": candidate.get("candidate_id", ""),
        "channel": candidate.get("channel", ""),
        "section": candidate.get("section", candidate.get("source_section", "")),
        "x_global_px": candidate.get("x_global_px", ""),
        "y_global_px": candidate.get("y_global_px", ""),
        "z_index": _candidate_z_index(candidate),
        "manual_label": manual_label,
        "reviewer": reviewer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    key = (record["candidate_id"], record["channel"])
    replaced = False
    for index, row in enumerate(rows):
        if (row.get("candidate_id", ""), row.get("channel", "")) == key:
            previous = row.get("manual_label", "")
            if previous != manual_label and not allow_overwrite:
                raise LabelConflictError(
                    f"{key[0]} already has label {previous!r}; "
                    "rerun with explicit label-change permission to replace it."
                )
            if previous == manual_label and not allow_overwrite:
                return row
            rows[index] = record
            replaced = True
            break
    if not replaced:
        rows.append(record)

    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANUAL_LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in MANUAL_LABEL_COLUMNS}
                         for row in rows)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)
    return record


def filter_review_candidates(
    candidates: list[dict],
    filter_name: str,
    *,
    contrast_threshold: float = 6.0,
    random_seed: int = 20260625,
    limit: int | None = None,
) -> list[dict]:
    """Apply the review strata exposed by ``review_candidates.py``."""

    def as_float(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    def as_int(row, key, default=0):
        try:
            return int(float(row.get(key, default)))
        except (TypeError, ValueError):
            return default

    predicates = {
        "preliminary_rule_pass": lambda c: c.get(
            "preliminary_sampling_category", c.get("current_status")
        ) == "preliminary_rule_pass",
        "preliminary_rule_fail": lambda c: c.get(
            "preliminary_sampling_category", c.get("current_status")
        ) == "preliminary_rule_fail",
        "near_threshold": lambda c: abs(as_float(c, "local_robust_z") - contrast_threshold) <= 1.0,
        "single_plane": lambda c: as_int(c, "support_plane_count",
                                        as_int(c, "n_consecutive_planes")) == 1,
        "many_planes": lambda c: as_int(c, "support_plane_count",
                                       as_int(c, "n_consecutive_planes")) >= 5,
        "outside_injection": lambda c: str(c.get(
            "inside_injection_analysis_exclusion", c.get("inside_injection_site", "")
        )).lower()
        not in {"true", "1", "yes"},
        "inside_injection": lambda c: str(c.get(
            "inside_injection_analysis_exclusion", c.get("inside_injection_site", "")
        )).lower()
        in {"true", "1", "yes"},
        "invalid_measurement": lambda c: str(c.get("measurement_valid", "")).lower()
        not in {"true", "1", "yes"},
        "all": lambda _c: True,
    }
    if filter_name == "random_sample":
        selected = list(candidates)
        random.Random(random_seed).shuffle(selected)
    elif filter_name in predicates:
        selected = [candidate for candidate in candidates if predicates[filter_name](candidate)]
    else:
        raise ValueError(f"Unknown review filter: {filter_name}")
    return selected[:limit] if limit else selected
