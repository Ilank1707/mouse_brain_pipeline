"""Human-label calibration of the preliminary-pass rules (analysis only).

Pure, I/O-free logic so it is unit-testable. Nothing here changes any candidate,
status, mask, threshold or raw TIFF -- it only *evaluates* the existing
configurable preliminary-pass gates against human labels and reports the
precision/recall trade-off. It NEVER targets a candidate count and NEVER looks at
pair-correlation g(r); the only objective is agreement with human labels.

A preliminary-rule pass is a PROVISIONAL candidate, never a confirmed cell.

Ground truth (human_label):
  * ``cell``      -> positive (a rule pass is correct)
  * ``artefact``  -> negative (a rule pass is a false positive)
  * ``uncertain`` -> excluded from precision/recall (ambiguous), still counted
  * ``injection`` -> excluded from precision/recall (handled by the injection
                     mask, not the morphological rules), still counted

Each channel (green_signal, channel_2_signal) is calibrated separately: a
channel's sweep only ever sees that channel's labelled records.
"""

from __future__ import annotations

import math
from dataclasses import replace

from .candidate_detection import (
    STATUS_PRELIMINARY_PASS,
    _preliminary_interpretation,
)

VOXEL_ZYX_UM = (6.0, 1.004, 1.004)
_VOXEL_UM3 = VOXEL_ZYX_UM[0] * VOXEL_ZYX_UM[1] * VOXEL_ZYX_UM[2]

CHANNELS = ("green_signal", "channel_2_signal")

# Label handling. 'artifact' is accepted as a spelling of 'artefact'.
POSITIVE_LABELS = {"cell"}
NEGATIVE_LABELS = {"artefact"}
PR_IGNORED_LABELS = {"uncertain", "injection"}
VALID_LABELS = POSITIVE_LABELS | NEGATIVE_LABELS | PR_IGNORED_LABELS

_BOOL_TRUE = {"true", "1", "yes"}

# The configurable thresholds evaluated by the sweep. Each spec:
#   name       -> canonical threshold name (also the output column)
#   attr       -> DetectionParams attribute driven by this threshold
#   feature    -> record feature the gate acts on (None => not distribution-based)
#   direction  -> "min" (reject below) or "max" (reject above)
SWEEP_SPECS = [
    dict(name="min_component_xy_area_um2", attr="min_component_xy_area_um2",
         feature="xy_area_um2", direction="min"),
    dict(name="min_component_volume_um3", attr="min_component_volume_um3",
         feature="volume_um3", direction="min"),
    dict(name="min_support_planes", attr="min_support_planes",
         feature="support_plane_count", direction="min", integer=True),
    dict(name="min_supporting_voxels", attr="min_supporting_voxels",
         feature="supporting_voxel_count", direction="min", integer=True),
    dict(name="min_signal_to_background_ratio", attr="min_signal_to_background_ratio",
         feature="local_robust_z", direction="min"),
    dict(name="min_diameter_um", attr="min_diameter_um",
         feature="xy_diameter_um", direction="min"),
    dict(name="max_diameter_um", attr="max_diameter_um",
         feature="equivalent_diameter_um", direction="max"),
    dict(name="max_elongation", attr="max_elongation",
         feature="elongation", direction="max"),
    dict(name="duplicate_distance_um", attr="min_separation_um",
         feature=None, direction="min"),
]

# Threshold columns reported for every evaluated parameter set.
THRESHOLD_COLUMNS = [
    "min_component_xy_area_um2",
    "min_component_volume_um3",
    "min_support_planes",
    "min_supporting_voxels",
    "min_signal_to_background_ratio",
    "min_local_robust_z",
    "min_diameter_um",
    "max_diameter_um",
    "max_elongation",
    "duplicate_distance_um",
    "keep_edge_clipped_if_center_in_tissue",
]

RESULT_COLUMNS = (
    ["channel", "sweep_type", "swept_parameter", "swept_value"]
    + THRESHOLD_COLUMNS
    + ["n_labeled_cell", "n_labeled_artefact", "n_labeled_uncertain",
       "n_labeled_injection", "n_retained", "tp", "fp", "fn", "tn",
       "false_positive_count", "false_negative_count",
       "precision", "recall", "f1"]
)


# --------------------------------------------------------------------------- #
# Small coercion helpers
# --------------------------------------------------------------------------- #
def _fnum(value):
    try:
        f = float(value)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _fbool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _BOOL_TRUE


def normalize_label(value):
    """Lower-case human label; map the 'artifact' spelling to 'artefact'."""
    label = str(value or "").strip().lower()
    if label == "artifact":
        label = "artefact"
    return label


def coerce_rec(row):
    """Rebuild the fields ``_preliminary_interpretation`` reads from a data row.

    Missing area / voxels are derived from diameter / volume so older tables
    still evaluate. Edge handling is enforced by the swept params, not here.
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
    n_consec = _fnum(row.get("n_consecutive_planes"))
    if not math.isfinite(n_consec):
        n_consec = support
    return {
        "candidate_id": row.get("candidate_id", ""),
        "channel": row.get("channel", ""),
        "inside_tissue": _fbool(row.get("inside_tissue")),
        "invalid_coordinate": _fbool(row.get("invalid_coordinate")),
        "original_cellfinder_z_valid": _fbool(row.get("original_cellfinder_z_valid") or "true"),
        "measurement_valid": _fbool(row.get("measurement_valid") or "true"),
        "is_artifact": _fbool(row.get("is_artifact")),
        "touches_crop_boundary": _fbool(row.get("touches_crop_boundary")),
        "n_consecutive_planes": n_consec if math.isfinite(n_consec) else 0,
        "support_plane_count": support if math.isfinite(support) else 0,
        "equivalent_diameter_um": _fnum(row.get("equivalent_diameter_um")),
        "xy_diameter_um": xy_diam,
        "xy_area_um2": xy_area,
        "volume_um3": volume,
        "supporting_voxel_count": vox,
        "elongation": _fnum(row.get("elongation")),
        "xy_centroid_shift_um": _fnum(row.get("xy_centroid_shift_um")),
        "local_robust_z": _fnum(row.get("local_robust_z")),
        "z_index": _fnum(row.get("z_index")),
        "x_global_px": _fnum(row.get("x_global_px")),
        "y_global_px": _fnum(row.get("y_global_px")),
    }


# --------------------------------------------------------------------------- #
# Duplicate-distance (NMS) evaluation
# --------------------------------------------------------------------------- #
class DupPool:
    """Neighbour pool for duplicate-distance evaluation (NMS winner = higher z).

    Mirrors ``candidate_detection._apply_nms``: a pass is suppressed when a
    STRONGER pass (higher ``local_robust_z``) exists within the distance.
    """

    def __init__(self, recs, voxel_zyx=VOXEL_ZYX_UM):
        import numpy as np  # noqa: PLC0415

        vz, vy, vx = voxel_zyx
        self.ids = [r.get("candidate_id", "") for r in recs]
        self.z = np.array([_fnum(r.get("local_robust_z")) for r in recs], dtype=float)
        self.pts = np.array(
            [[_fnum(r.get("z_index")) * vz, _fnum(r.get("y_global_px")) * vy,
              _fnum(r.get("x_global_px")) * vx] for r in recs],
            dtype=float,
        ).reshape(-1, 3)
        self._id_pos = {cid: i for i, cid in enumerate(self.ids)}
        self._tree = None
        if len(self.ids):
            try:
                from scipy.spatial import cKDTree  # noqa: PLC0415

                self._tree = cKDTree(self.pts)
            except Exception:  # pragma: no cover - fallback to brute force
                self._tree = None

    def is_duplicate(self, rec, distance_um, voxel_zyx=VOXEL_ZYX_UM):
        import numpy as np  # noqa: PLC0415

        if not len(self.ids) or not distance_um or distance_um <= 0:
            return False
        vz, vy, vx = voxel_zyx
        z_here = _fnum(rec.get("local_robust_z"))
        cid = rec.get("candidate_id", "")
        p = np.array([_fnum(rec.get("z_index")) * vz,
                      _fnum(rec.get("y_global_px")) * vy,
                      _fnum(rec.get("x_global_px")) * vx], dtype=float)
        if self._tree is not None:
            neighbours = self._tree.query_ball_point(p, distance_um)
        else:  # pragma: no cover
            d = np.linalg.norm(self.pts - p, axis=1)
            neighbours = list(np.nonzero(d <= distance_um)[0])
        for j in neighbours:
            if self.ids[j] == cid:
                continue
            # Stronger neighbour, or equal strength with a lower id (deterministic
            # tie-break) -> this candidate loses and is a duplicate.
            if self.z[j] > z_here or (self.z[j] == z_here and self.ids[j] < cid):
                return True
        return False


# --------------------------------------------------------------------------- #
# Prediction + metrics
# --------------------------------------------------------------------------- #
def predicted_pass(rec, params, dup_pool=None, voxel_zyx=VOXEL_ZYX_UM):
    """True when ``rec`` passes the preliminary rule under ``params``.

    Edge handling: ``params`` must keep edge-clipped candidates whose centre is in
    tissue (never rejected for the edge alone). Duplicate suppression is applied
    only after a morphological pass, like the pipeline's NMS.
    """
    status, _reason = _preliminary_interpretation(rec, params, True)
    if status != STATUS_PRELIMINARY_PASS:
        return False
    if dup_pool is not None and getattr(params, "min_separation_um", 0) > 0:
        if dup_pool.is_duplicate(rec, params.min_separation_um, voxel_zyx):
            return False
    return True


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate_params(params, labeled, *, dup_pool=None, voxel_zyx=VOXEL_ZYX_UM):
    """Confusion counts + precision/recall/F1 for one parameter set on one
    channel's labelled records. ``labeled`` = ``[(rec, normalized_label), ...]``."""
    tp = fp = fn = tn = 0
    retained = 0
    n_cell = n_art = n_unc = n_inj = 0
    for rec, label in labeled:
        passed = predicted_pass(rec, params, dup_pool=dup_pool, voxel_zyx=voxel_zyx)
        retained += int(passed)
        if label in POSITIVE_LABELS:
            n_cell += 1
            tp += int(passed)
            fn += int(not passed)
        elif label in NEGATIVE_LABELS:
            n_art += 1
            fp += int(passed)
            tn += int(not passed)
        elif label == "uncertain":
            n_unc += 1
        elif label == "injection":
            n_inj += 1
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    return {
        "n_labeled_cell": n_cell,
        "n_labeled_artefact": n_art,
        "n_labeled_uncertain": n_unc,
        "n_labeled_injection": n_inj,
        "n_retained": retained,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def threshold_values(params):
    """The reported threshold columns for a DetectionParams instance."""
    return {
        "min_component_xy_area_um2": round(float(params.min_component_xy_area_um2), 4),
        "min_component_volume_um3": round(float(params.min_component_volume_um3), 4),
        "min_support_planes": int(params.min_support_planes),
        "min_supporting_voxels": int(params.min_supporting_voxels),
        "min_signal_to_background_ratio": round(float(params.min_signal_to_background_ratio), 4),
        "min_local_robust_z": round(float(params.min_local_robust_z), 4),
        "min_diameter_um": round(float(params.min_diameter_um), 4),
        "max_diameter_um": round(float(params.max_diameter_um), 4),
        "max_elongation": round(float(params.max_elongation), 4),
        "duplicate_distance_um": round(float(params.min_separation_um), 4),
        "keep_edge_clipped_if_center_in_tissue":
            bool(params.keep_edge_clipped_if_center_in_tissue),
    }


# --------------------------------------------------------------------------- #
# Parameter enumeration
# --------------------------------------------------------------------------- #
def enforce_edge_policy(params):
    """Never reject a candidate for being near the edge alone."""
    return replace(params, keep_edge_clipped_if_center_in_tissue=True)


def _apply_threshold(params, spec, value):
    """Return a copy of ``params`` with ``spec`` set to ``value``.

    The signal-to-background sweep also lowers ``min_local_robust_z`` in tandem
    so the *effective* robust-z gate (``max`` of the two) equals the swept value
    even below the base contrast floor.
    """
    attr = spec["attr"]
    if spec.get("integer"):
        value = int(round(value))
    updated = replace(params, **{attr: value})
    if spec["name"] == "min_signal_to_background_ratio":
        updated = replace(updated, min_local_robust_z=min(params.min_local_robust_z, value))
    return updated


def feature_levels(recs, base_params, spec):
    """Distribution-anchored candidate values for one threshold.

    Anchored to the base (current) value and to percentiles of the CELL feature
    distribution -- i.e. where real cells sit -- NOT to any candidate count.
    """
    import numpy as np  # noqa: PLC0415

    name = spec["name"]
    base = getattr(base_params, spec["attr"])
    if name == "min_support_planes":
        return [1, 2, 3, 4]
    if name == "duplicate_distance_um":
        return sorted({0.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, round(float(base), 3)})
    feature = spec["feature"]
    values = np.array(
        [rec[feature] for rec, label in recs
         if label in POSITIVE_LABELS and math.isfinite(rec.get(feature, float("nan")))],
        dtype=float,
    )
    levels = {round(float(base), 4)}
    if spec["direction"] == "min":
        levels.add(0.0)  # disabled
        if values.size:
            for pct in (2, 5, 10, 25, 40):
                levels.add(round(float(np.percentile(values, pct)), 4))
    else:  # max gate
        big = round(float(base) * 3.0, 4) if base else 1e6
        levels.add(big)
        if values.size:
            for pct in (60, 75, 90, 95, 98):
                levels.add(round(float(np.percentile(values, pct)), 4))
    if spec.get("integer"):
        levels = {int(round(v)) for v in levels}
    return sorted(levels)


def _grid_levels(recs, base_params, spec, n=3):
    """A reduced 3-level {loose, current, strict} set for the joint grid."""
    import numpy as np  # noqa: PLC0415

    name = spec["name"]
    base = getattr(base_params, spec["attr"])
    if name == "min_support_planes":
        return sorted({1, int(round(base)), 3})
    feature = spec["feature"]
    values = np.array(
        [rec[feature] for rec, label in recs
         if label in POSITIVE_LABELS and math.isfinite(rec.get(feature, float("nan")))],
        dtype=float,
    )
    if spec["direction"] == "min":
        loose = 0.0
        strict = round(float(np.percentile(values, 25)), 4) if values.size else float(base)
    else:
        loose = round(float(base) * 2.0, 4) if base else 1e6
        strict = round(float(np.percentile(values, 90)), 4) if values.size else float(base)
    levels = {loose, round(float(base), 4), strict}
    if spec.get("integer"):
        levels = {int(round(v)) for v in levels}
    return sorted(levels)


# Joint grid spans the gates that most drive the precision/recall trade-off.
_GRID_SPEC_NAMES = (
    "min_signal_to_background_ratio",
    "min_support_planes",
    "min_component_xy_area_um2",
    "min_supporting_voxels",
)


def enumerate_parameter_sets(recs, base_params):
    """All (sweep_type, swept_parameter, swept_value, params) to evaluate.

    Deterministic given ``recs`` + ``base_params`` (reproducible). Includes the
    baseline, one single-axis sweep per configurable threshold, and a bounded
    joint grid over the most impactful gates.
    """
    base = enforce_edge_policy(base_params)
    out = [("baseline", "", "", base)]

    specs_by_name = {s["name"]: s for s in SWEEP_SPECS}
    for spec in SWEEP_SPECS:
        for value in feature_levels(recs, base, spec):
            params = enforce_edge_policy(_apply_threshold(base, spec, value))
            out.append(("single", spec["name"], value, params))

    import itertools

    grid_specs = [specs_by_name[n] for n in _GRID_SPEC_NAMES]
    grid_levels = [_grid_levels(recs, base, s) for s in grid_specs]
    for combo in itertools.product(*grid_levels):
        params = base
        for spec, value in zip(grid_specs, combo):
            params = _apply_threshold(params, spec, value)
        out.append(("grid", "+".join(_GRID_SPEC_NAMES), "|".join(str(c) for c in combo),
                    enforce_edge_policy(params)))
    return out


# --------------------------------------------------------------------------- #
# Pareto front (precision vs recall ONLY -- never a count)
# --------------------------------------------------------------------------- #
def pareto_front(points):
    """Indices of the non-dominated (precision, recall) points, best first.

    A point dominates another when it is >= on both precision and recall and >
    on at least one. Only precision and recall are read -- never n_retained or any
    count -- so the front cannot be steered toward a target count. Degenerate
    all-fail points (no true positives) are dropped.
    """
    usable = [i for i, p in enumerate(points) if p.get("tp", 0) > 0]
    front = []
    for i in usable:
        pi, ri = points[i]["precision"], points[i]["recall"]
        dominated = False
        for j in usable:
            if j == i:
                continue
            pj, rj = points[j]["precision"], points[j]["recall"]
            if pj >= pi and rj >= ri and (pj > pi or rj > ri):
                dominated = True
                break
        if not dominated:
            front.append(i)
    # Stable, human-readable order: high precision first, then higher recall.
    front.sort(key=lambda i: (-points[i]["precision"], -points[i]["recall"]))
    return front


def annotate_pareto_roles(front_points):
    """Descriptive-only tags; this NEVER selects a single recommended setting."""
    if not front_points:
        return front_points
    hp = max(range(len(front_points)), key=lambda i: (front_points[i]["precision"],
                                                      front_points[i]["recall"]))
    hr = max(range(len(front_points)), key=lambda i: (front_points[i]["recall"],
                                                      front_points[i]["precision"]))
    bf = max(range(len(front_points)), key=lambda i: front_points[i]["f1"])
    for k, p in enumerate(front_points):
        roles = []
        if k == hp:
            roles.append("high_precision")
        if k == hr:
            roles.append("high_recall")
        if k == bf:
            roles.append("best_f1")
        p["pareto_role"] = "+".join(roles) if roles else "frontier"
    return front_points


# --------------------------------------------------------------------------- #
# Confusion matrix + examples at the current (baseline) settings
# --------------------------------------------------------------------------- #
def confusion_rows(channel, labeled, base_params, *, dup_pool=None, voxel_zyx=VOXEL_ZYX_UM):
    """Per (human_label, predicted_pass) counts at the baseline settings."""
    from collections import Counter

    counts = Counter()
    for rec, label in labeled:
        passed = predicted_pass(rec, base_params, dup_pool=dup_pool, voxel_zyx=voxel_zyx)
        counts[(label, passed)] += 1
    rows = []
    for label in ("cell", "artefact", "uncertain", "injection"):
        for passed in (True, False):
            rows.append({
                "channel": channel,
                "config": "baseline_current",
                "human_label": label,
                "predicted_preliminary_pass": passed,
                "count": counts.get((label, passed), 0),
            })
    return rows


def _example_row(channel, rec, label, source_row, kind):
    return {
        "channel": channel,
        "candidate_id": rec.get("candidate_id", ""),
        "human_label": label,
        "error_kind": kind,
        "current_status": source_row.get("current_status", ""),
        "preliminary_rule_reason": source_row.get("preliminary_rule_reason", ""),
        "x_global_px": source_row.get("x_global_px", ""),
        "y_global_px": source_row.get("y_global_px", ""),
        "z_index": source_row.get("z_index", ""),
        "xy_area_um2": rec.get("xy_area_um2", ""),
        "volume_um3": rec.get("volume_um3", ""),
        "support_plane_count": rec.get("support_plane_count", ""),
        "supporting_voxel_count": rec.get("supporting_voxel_count", ""),
        "local_robust_z": rec.get("local_robust_z", ""),
        "equivalent_diameter_um": rec.get("equivalent_diameter_um", ""),
        "xy_diameter_um": rec.get("xy_diameter_um", ""),
        "elongation": rec.get("elongation", ""),
        "touches_crop_boundary": rec.get("touches_crop_boundary", ""),
        "inside_tissue": rec.get("inside_tissue", ""),
        "review_patch_file": source_row.get("review_patch_file", ""),
    }


EXAMPLE_COLUMNS = [
    "channel", "candidate_id", "human_label", "error_kind", "current_status",
    "preliminary_rule_reason", "x_global_px", "y_global_px", "z_index",
    "xy_area_um2", "volume_um3", "support_plane_count", "supporting_voxel_count",
    "local_robust_z", "equivalent_diameter_um", "xy_diameter_um", "elongation",
    "touches_crop_boundary", "inside_tissue", "review_patch_file",
]

CONFUSION_COLUMNS = [
    "channel", "config", "human_label", "predicted_preliminary_pass", "count",
]


# --------------------------------------------------------------------------- #
# Top-level calibration (still pure: takes rows, returns dicts)
# --------------------------------------------------------------------------- #
def build_labeled(rows):
    """``[(coerce_rec(row), normalized_label, row), ...]`` for rows with a valid
    human_label. Rows with a blank / unknown label are dropped."""
    out = []
    for row in rows:
        label = normalize_label(row.get("human_label"))
        if label in VALID_LABELS:
            out.append((coerce_rec(row), label, row))
    return out


def calibrate_channel(channel, rows, base_params, *, dup_pool=None,
                      voxel_zyx=VOXEL_ZYX_UM):
    """Calibrate ONE channel using ONLY that channel's rows (kept separate)."""
    labeled_full = build_labeled(rows)
    labeled = [(rec, label) for rec, label, _row in labeled_full]

    param_sets = enumerate_parameter_sets(labeled, base_params)
    results = []
    for sweep_type, swept_param, swept_value, params in param_sets:
        metrics = evaluate_params(params, labeled, dup_pool=dup_pool, voxel_zyx=voxel_zyx)
        row = {"channel": channel, "sweep_type": sweep_type,
               "swept_parameter": swept_param, "swept_value": swept_value}
        row.update(threshold_values(params))
        row.update(metrics)
        results.append(row)

    front_idx = pareto_front(results)
    # Collapse exact-duplicate settings (same thresholds reached by a single-axis
    # sweep and the grid). This removes clutter only; it never picks among
    # genuinely distinct settings.
    seen_signatures = set()
    front = []
    for i in front_idx:
        signature = tuple(results[i][c] for c in THRESHOLD_COLUMNS)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        front.append(dict(results[i]))
    front = annotate_pareto_roles(front)

    base = enforce_edge_policy(base_params)
    baseline_metrics = results[0]  # baseline is always first
    confusion = confusion_rows(channel, labeled, base, dup_pool=dup_pool,
                               voxel_zyx=voxel_zyx)

    fps, fns = [], []
    for rec, label, source_row in labeled_full:
        passed = predicted_pass(rec, base, dup_pool=dup_pool, voxel_zyx=voxel_zyx)
        if label in NEGATIVE_LABELS and passed:
            fps.append(_example_row(channel, rec, label, source_row, "false_positive"))
        elif label in POSITIVE_LABELS and not passed:
            fns.append(_example_row(channel, rec, label, source_row, "false_negative"))

    from collections import Counter

    label_counts = Counter(label for _rec, label in labeled)
    return {
        "channel": channel,
        "n_labeled": len(labeled),
        "label_counts": dict(label_counts),
        "baseline": baseline_metrics,
        "results": results,
        "pareto": front,
        "confusion": confusion,
        "false_positives": fps,
        "false_negatives": fns,
    }
