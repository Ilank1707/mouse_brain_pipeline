"""Read-only run-consistency audit (Task 5).

Pure, I/O-free checks over a completed run's ``all_candidates.csv`` (and, when
available, its coordinate exports). Nothing here changes a candidate, status,
mask, threshold or raw TIFF. Structural violations are HARD ERRORS; suppression
sensitivity is a WARNING (an intense injection can shift Cellfinder thresholds, so
suppression-only outside candidates need manual validation -- never an automatic
threshold change).

Checks (each returns ``passed`` + ``level``):
  * every candidate has a non-empty, unique candidate_id (no silent collapse)
  * all status counts sum to the candidate total
  * inside + outside injection-mask counts equal the total; core => analysis mask
  * every candidate maps to exactly one peak optical plane (or is flagged)
  * included_in_count is false unless a human 'cell' label / validated model confirms
  * coordinate exports reconcile with all_candidates.csv (no candidate disappears)
  * green and red candidate IDs are never mixed
  * (warning) candidate generation is not highly suppression-sensitive
"""

from __future__ import annotations

from .coordinate_exports import (
    is_confirmed_cell,
    peak_optical_plane,
    peak_zero_based_index,
)
from .generation_source_audit import (
    DEFAULT_BOTH_FRACTION_THRESHOLD,
    outside_mask_candidates,
    source_fractions,
    suppression_sensitivity_warning,
)

CHANNELS = ("green_signal", "channel_2_signal")
ERROR = "error"
WARNING = "warning"


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _check(name, level, passed, detail="") -> dict:
    return {"name": name, "level": level, "passed": bool(passed), "detail": detail}


# --------------------------------------------------------------------------- #
# Individual checks (each returns one check dict)
# --------------------------------------------------------------------------- #
def check_candidate_ids(candidates) -> dict:
    """Non-empty and unique candidate_id (a collapse would silently lose a row)."""
    from collections import Counter

    ids = [str(c.get("candidate_id", "")).strip() for c in candidates]
    blank = sum(1 for i in ids if not i)
    dupes = [i for i, n in Counter(ids).items() if i and n > 1]
    passed = blank == 0 and not dupes
    detail = "all candidate_id values are present and unique"
    if not passed:
        detail = f"{blank} blank id(s); {len(dupes)} duplicated id(s): {dupes[:5]}"
    return _check("candidate_id_unique_and_present", ERROR, passed, detail)


def check_status_counts_sum(candidates) -> dict:
    """Every candidate has a non-empty status and the statuses sum to the total."""
    from collections import Counter

    counts = Counter(str(c.get("current_status", "")).strip() for c in candidates)
    missing = counts.get("", 0)
    total_summed = sum(counts.values())
    passed = missing == 0 and total_summed == len(candidates)
    detail = (f"{total_summed} statuses reconcile to {len(candidates)} candidates: "
              + ", ".join(f"{k or '(blank)'}={v}" for k, v in sorted(counts.items())))
    if missing:
        detail = f"{missing} candidate(s) have a blank current_status"
    return _check("status_counts_sum_to_total", ERROR, passed, detail)


def check_mask_partition(candidates) -> dict:
    """inside + outside == total, and every core candidate is inside the analysis mask."""
    inside = sum(1 for c in candidates
                 if _truthy(c.get("inside_injection_analysis_exclusion")))
    outside = len(candidates) - inside
    core_outside_mask = [
        c.get("candidate_id", "") for c in candidates
        if _truthy(c.get("inside_injection_core"))
        and not _truthy(c.get("inside_injection_analysis_exclusion"))
    ]
    passed = (inside + outside == len(candidates)) and not core_outside_mask
    detail = f"inside={inside} + outside={outside} == total={len(candidates)}"
    if core_outside_mask:
        detail = (f"{len(core_outside_mask)} candidate(s) inside the injection CORE but "
                  f"outside the analysis-exclusion mask: {core_outside_mask[:5]}")
    return _check("mask_membership_partitions_total", ERROR, passed, detail)


def check_peak_planes(candidates, planes_per_section: int = 7) -> dict:
    """Each candidate maps to exactly one peak plane; none has an out-of-range peak.

    A present-but-out-of-range peak Z is a HARD error (a plane outside the stack).
    Candidates with a missing/blank peak Z are 'unassigned' and reported in the
    detail (they are handled by unassigned_peak_plane.csv), not treated as an error.
    """
    out_of_range, unassigned = [], 0
    for c in candidates:
        raw = c.get("fixed_xy_peak_z_index", "")
        zero_based = peak_zero_based_index(c)
        plane = peak_optical_plane(c, planes_per_section)
        if plane is None:
            if zero_based is not None and str(raw).strip() != "":
                out_of_range.append(c.get("candidate_id", ""))   # int but out of range
            else:
                unassigned += 1
    passed = not out_of_range
    detail = (f"every candidate maps to <= 1 peak plane; {unassigned} unassigned "
              f"(missing/blank peak Z)")
    if out_of_range:
        detail = (f"{len(out_of_range)} candidate(s) have an out-of-range peak plane "
                  f"(0..{planes_per_section - 1}): {out_of_range[:5]}")
    return _check("exactly_one_peak_optical_plane", ERROR, passed, detail)


def check_included_in_count(candidates) -> dict:
    """included_in_count is only ever true for a confirmed cell (human/model)."""
    offenders = [
        c.get("candidate_id", "") for c in candidates
        if _truthy(c.get("included_in_count")) and not is_confirmed_cell(c)
    ]
    passed = not offenders
    n_included = sum(1 for c in candidates if _truthy(c.get("included_in_count")))
    detail = (f"{n_included} candidate(s) included_in_count, all human/model confirmed"
              if n_included else "no candidate is included_in_count (as expected)")
    if offenders:
        detail = (f"{len(offenders)} candidate(s) have included_in_count=True WITHOUT a "
                  f"human 'cell' label or validated model confirmation: {offenders[:5]}")
    return _check("included_in_count_requires_confirmation", ERROR, passed, detail)


def check_channel_id_separation(candidates) -> dict:
    """Green and red candidate IDs are never shared or cross-labelled."""
    by_channel = {}
    for c in candidates:
        by_channel.setdefault(c.get("channel", ""), set()).add(
            str(c.get("candidate_id", "")))
    shared = set()
    seen = list(by_channel.items())
    for i, (_ch_a, ids_a) in enumerate(seen):
        for _ch_b, ids_b in seen[i + 1:]:
            shared |= (ids_a & ids_b)

    mislabelled = []
    for c in candidates:
        cid = str(c.get("candidate_id", ""))
        channel = c.get("channel", "")
        for token in CHANNELS:
            if token in cid and channel != token:
                mislabelled.append(cid)
                break
    passed = not shared and not mislabelled
    detail = "green and red candidate IDs are disjoint and channel-consistent"
    if not passed:
        detail = (f"{len(shared)} id(s) shared across channels {list(shared)[:5]}; "
                  f"{len(mislabelled)} id(s) whose embedded channel != channel column "
                  f"{mislabelled[:5]}")
    return _check("green_red_ids_never_mixed", ERROR, passed, detail)


def check_coordinate_exports(candidates, exports) -> dict:
    """Coordinate exports reconcile with all_candidates.csv (no candidate disappears).

    ``exports`` maps a coordinate-export filename to its set of candidate_ids. The
    check is skipped (passed, informational) when no exports are supplied.
    """
    all_ids = {str(c.get("candidate_id", "")) for c in candidates}
    if not exports:
        return _check("coordinate_exports_reconcile", ERROR, True,
                      "no coordinate exports supplied; skipped")

    problems = []
    all_export = exports.get("all_candidate_coordinates.csv")
    if all_export is not None:
        missing = all_ids - all_export       # candidate lost from the export
        extra = all_export - all_ids         # phantom id in the export
        if missing:
            problems.append(f"{len(missing)} candidate(s) missing from "
                            f"all_candidate_coordinates.csv: {list(missing)[:5]}")
        if extra:
            problems.append(f"{len(extra)} phantom id(s) in "
                            f"all_candidate_coordinates.csv: {list(extra)[:5]}")
    # Every per-category export must be a subset of all_candidates.
    for filename, ids in sorted(exports.items()):
        stray = ids - all_ids
        if stray:
            problems.append(f"{len(stray)} id(s) in {filename} not in all_candidates.csv: "
                            f"{list(stray)[:5]}")
    passed = not problems
    detail = "coordinate exports reconcile exactly with all_candidates.csv"
    if problems:
        detail = " | ".join(problems)
    return _check("coordinate_exports_reconcile", ERROR, passed, detail)


def suppression_sensitivity_checks(candidates,
                                   threshold=DEFAULT_BOTH_FRACTION_THRESHOLD) -> list:
    """One WARNING-level check per channel (green/red separate)."""
    checks = []
    for channel in sorted({c.get("channel", "") for c in candidates}):
        chan = [c for c in candidates if c.get("channel", "") == channel]
        fractions = source_fractions(outside_mask_candidates(chan))
        triggered, message = suppression_sensitivity_warning(channel, fractions, threshold)
        detail = message if triggered else (
            f"outside-mask 'both' fraction {fractions['fraction_both']:.1%} "
            f">= {threshold:.0%}")
        checks.append(_check(f"suppression_sensitivity[{channel}]", WARNING,
                             not triggered, detail))
    return checks


# --------------------------------------------------------------------------- #
# Top-level audit
# --------------------------------------------------------------------------- #
def audit_run(candidates, *, exports=None, planes_per_section: int = 7,
              both_fraction_threshold=DEFAULT_BOTH_FRACTION_THRESHOLD) -> dict:
    """Run every consistency check; hard errors gate ``ok``, warnings do not."""
    checks = [
        check_candidate_ids(candidates),
        check_status_counts_sum(candidates),
        check_mask_partition(candidates),
        check_peak_planes(candidates, planes_per_section),
        check_included_in_count(candidates),
        check_channel_id_separation(candidates),
        check_coordinate_exports(candidates, exports),
    ]
    checks.extend(suppression_sensitivity_checks(candidates, both_fraction_threshold))

    errors = [c for c in checks if c["level"] == ERROR and not c["passed"]]
    warnings = [c for c in checks if c["level"] == WARNING and not c["passed"]]
    return {
        "n_candidates": len(candidates),
        "ok": not errors,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
