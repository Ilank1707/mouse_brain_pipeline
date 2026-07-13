"""Read-only audit of candidate-generation source (Task 4).

Pure, I/O-free logic so it is unit-testable. It only *reads* an already-written
candidate table and reports how candidates were generated -- the raw-stack
Cellfinder pass, the injection-suppressed pass, or both. It NEVER changes any
candidate, status, mask, threshold or raw TIFF, and NEVER targets a count.

Why this matters: an intense injection site can shift Cellfinder's global/tiled
thresholding, so a candidate that only appears on the injection-suppressed stack
("injection_suppressed_stack") outside the injection mask is *suppression
sensitive*. If few of the outside-mask candidates are found by BOTH passes, the
generator is highly suppression-sensitive and those outside candidates need manual
recall and precision validation -- this is a WARNING, never an automatic threshold
change.
"""

from __future__ import annotations

from .coordinate_exports import peak_optical_plane

# Canonical generation sources (mutually exclusive).
SOURCE_BOTH = "both"
SOURCE_RAW = "raw_stack"
SOURCE_SUPPRESSED = "injection_suppressed_stack"

AUDIT_COLUMNS = [
    "channel",
    "candidate_generation_source",
    "inside_injection_core",
    "inside_injection_analysis_exclusion",
    "current_status",
    "peak_optical_plane",
    "count",
]

# Default: warn if fewer than 10% of the outside-mask candidates are found by
# BOTH passes (highly suppression-sensitive generation).
DEFAULT_BOTH_FRACTION_THRESHOLD = 0.10


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize_source(row) -> str:
    """The candidate_generation_source, deriving it from the two per-pass flags
    when the column is missing/blank (older tables)."""
    source = str(row.get("candidate_generation_source", "") or "").strip()
    if source in (SOURCE_BOTH, SOURCE_RAW, SOURCE_SUPPRESSED):
        return source
    on_raw = _truthy(row.get("detected_on_raw_stack"))
    on_supp = _truthy(row.get("detected_on_injection_suppressed_stack"))
    if on_raw and on_supp:
        return SOURCE_BOTH
    if on_supp:
        return SOURCE_SUPPRESSED
    if on_raw:
        return SOURCE_RAW
    return source or "unknown"


def _peak_plane(row, planes_per_section) -> str:
    plane = peak_optical_plane(row, planes_per_section)
    return f"{plane:02d}" if plane is not None else "unassigned"


def audit_rows(candidates, planes_per_section: int = 7) -> list:
    """Counts by channel / source / inside-core / inside-analysis-mask / status /
    peak optical plane. One row per populated combination, deterministically
    ordered."""
    from collections import Counter

    counter = Counter(
        (
            c.get("channel", ""),
            normalize_source(c),
            _truthy(c.get("inside_injection_core")),
            _truthy(c.get("inside_injection_analysis_exclusion")),
            c.get("current_status", ""),
            _peak_plane(c, planes_per_section),
        )
        for c in candidates
    )
    rows = []
    for (channel, source, in_core, in_mask, status, plane), count in sorted(
        counter.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        rows.append({
            "channel": channel,
            "candidate_generation_source": source,
            "inside_injection_core": in_core,
            "inside_injection_analysis_exclusion": in_mask,
            "current_status": status,
            "peak_optical_plane": plane,
            "count": count,
        })
    return rows


def source_fractions(candidates) -> dict:
    """Counts + fractions of ``both`` / ``raw_stack`` only / ``injection_suppressed_stack``
    only over the supplied candidates."""
    n = len(candidates)
    both = sum(1 for c in candidates if normalize_source(c) == SOURCE_BOTH)
    raw_only = sum(1 for c in candidates if normalize_source(c) == SOURCE_RAW)
    supp_only = sum(1 for c in candidates if normalize_source(c) == SOURCE_SUPPRESSED)

    def frac(k):
        return (k / n) if n else 0.0

    return {
        "n": n,
        "both": both,
        "raw_stack_only": raw_only,
        "injection_suppressed_stack_only": supp_only,
        "fraction_both": round(frac(both), 6),
        "fraction_raw_stack_only": round(frac(raw_only), 6),
        "fraction_injection_suppressed_stack_only": round(frac(supp_only), 6),
    }


def outside_mask_candidates(candidates) -> list:
    """Candidates OUTSIDE the injection analysis-exclusion mask (regardless of rules)."""
    return [c for c in candidates
            if not _truthy(c.get("inside_injection_analysis_exclusion"))]


SUPPRESSION_WARNING_TEMPLATE = (
    "SUPPRESSION-SENSITIVE candidate generation for {channel}: only {pct:.1f}% of "
    "the {n_outside} outside-mask candidates were found by BOTH the raw and the "
    "injection-suppressed pass (threshold {threshold:.0f}%). An intense injection "
    "site can shift Cellfinder's global/tiled thresholding, so these "
    "suppression-only outside candidates require MANUAL recall and precision "
    "validation. This is a warning only -- do NOT change any threshold "
    "automatically."
)


def suppression_sensitivity_warning(channel, fractions,
                                    threshold=DEFAULT_BOTH_FRACTION_THRESHOLD):
    """``(triggered, message)`` for one channel's outside-mask fractions.

    Triggered when there is at least one outside-mask candidate and the fraction
    found by BOTH passes is below ``threshold``.
    """
    n_outside = fractions.get("n", 0)
    fraction_both = fractions.get("fraction_both", 0.0)
    if n_outside > 0 and fraction_both < threshold:
        return True, SUPPRESSION_WARNING_TEMPLATE.format(
            channel=channel, pct=100.0 * fraction_both, n_outside=n_outside,
            threshold=100.0 * threshold)
    return False, ""


def summarize(candidates, planes_per_section: int = 7,
              threshold=DEFAULT_BOTH_FRACTION_THRESHOLD) -> dict:
    """Full JSON-serialisable audit summary (green and red kept separate)."""
    from collections import Counter

    channels = sorted({c.get("channel", "") for c in candidates})
    by_channel = {}
    warnings = []
    for channel in channels:
        chan = [c for c in candidates if c.get("channel", "") == channel]
        outside = outside_mask_candidates(chan)
        outside_fractions = source_fractions(outside)
        triggered, message = suppression_sensitivity_warning(
            channel, outside_fractions, threshold)
        if triggered:
            warnings.append({
                "channel": channel,
                "n_outside_mask": outside_fractions["n"],
                "fraction_both": outside_fractions["fraction_both"],
                "message": message,
            })
        by_channel[channel] = {
            "n": len(chan),
            "by_source": dict(Counter(normalize_source(c) for c in chan)),
            "all_candidate_source_fractions": source_fractions(chan),
            "outside_analysis_mask_source_fractions": outside_fractions,
        }

    return {
        "n_candidates": len(candidates),
        "channels": channels,
        "planes_per_section": planes_per_section,
        "by_channel": by_channel,
        "outside_analysis_mask_overall": source_fractions(
            outside_mask_candidates(candidates)),
        "suppression_sensitivity": {
            "both_fraction_threshold": threshold,
            "triggered": bool(warnings),
            "warnings": warnings,
        },
        "guarantees": [
            "read-only audit; no candidate, status, mask, threshold or TIFF was changed",
            "no candidate count was targeted",
            "green_signal and channel_2_signal are reported separately",
            "suppression sensitivity is a WARNING, never an automatic threshold change",
        ],
    }
