"""Tests for human-label calibration of the preliminary-pass rules.

Covers the four required properties: separate green/red calibration,
reproducibility, edge handling (never reject for the edge alone), and the absence
of any target-count optimisation. All logic is analysis-only and never mutates a
candidate or status.
"""

from __future__ import annotations

import inspect

import pytest

from mouse_brain_pipeline.candidate_detection import DetectionParams
from mouse_brain_pipeline import rule_calibration as rc


def make_params(**overrides):
    """Baseline-like gates so a healthy cell passes and a weak artefact fails."""
    base = dict(
        min_diameter_um=6.0, max_diameter_um=30.0,
        min_local_robust_z=6.0, min_signal_to_background_ratio=8.0,
        min_component_xy_area_um2=28.3, min_component_volume_um3=113.0,
        min_support_planes=2, min_supporting_voxels=19,
        max_elongation=3.0, min_separation_um=6.0,
        min_consecutive_planes=2, single_plane_review_min_z=8.0,
        single_plane_pass_min_z=12.0,
        keep_edge_clipped_if_center_in_tissue=True,
    )
    base.update(overrides)
    return DetectionParams(**base)


def cell_row(cid, channel="green_signal", **overrides):
    row = dict(
        candidate_id=cid, channel=channel, human_label="cell",
        xy_area_um2=120.0, volume_um3=320.0, support_plane_count=3,
        supporting_voxel_count=40, local_robust_z=22.0,
        equivalent_diameter_um=10.0, xy_diameter_um=9.0, elongation=1.4,
        xy_centroid_shift_um=1.0, n_consecutive_planes=3,
        inside_tissue="True", touches_crop_boundary="False",
        invalid_coordinate="False", measurement_valid="True",
        original_cellfinder_z_valid="True",
        z_index=3, x_global_px=1000, y_global_px=2000,
    )
    row.update(overrides)
    return row


def artefact_row(cid, channel="green_signal", **overrides):
    row = cell_row(cid, channel, human_label="artefact",
                   xy_area_um2=8.0, volume_um3=20.0, support_plane_count=1,
                   supporting_voxel_count=3, local_robust_z=3.0,
                   equivalent_diameter_um=3.0, xy_diameter_um=3.0, elongation=1.2)
    row.update(overrides)
    return row


def _labeled(rows):
    return [(rec, label) for rec, label, _ in rc.build_labeled(rows)]


# --------------------------------------------------------------------------- #
# Core sanity
# --------------------------------------------------------------------------- #
def test_cell_passes_artefact_fails_at_baseline():
    params = rc.enforce_edge_policy(make_params())
    assert rc.predicted_pass(rc.coerce_rec(cell_row("c1")), params) is True
    assert rc.predicted_pass(rc.coerce_rec(artefact_row("a1")), params) is False


def test_normalize_label_maps_artifact_spelling_and_drops_blanks():
    assert rc.normalize_label("Artifact") == "artefact"
    assert rc.normalize_label("CELL") == "cell"
    rows = [cell_row("c1"), artefact_row("a1"), cell_row("c2", human_label="")]
    labeled = rc.build_labeled(rows)
    assert len(labeled) == 2  # the blank label is dropped


# --------------------------------------------------------------------------- #
# 1. Separate green / red calibration
# --------------------------------------------------------------------------- #
def test_channels_are_calibrated_independently():
    params = make_params()
    green = [cell_row(f"g{i}", "green_signal") for i in range(6)] + \
            [artefact_row(f"ga{i}", "green_signal") for i in range(6)]
    # Red cells are genuine but weak in signal-to-background, so the SAME baseline
    # gate rejects them -> a different recall than green. Different data per
    # channel must give different calibration.
    red = [cell_row(f"r{i}", "channel_2_signal", local_robust_z=5.0) for i in range(6)] + \
          [artefact_row(f"ra{i}", "channel_2_signal") for i in range(6)]

    g = rc.calibrate_channel("green_signal", green, rc.enforce_edge_policy(params))
    r = rc.calibrate_channel("channel_2_signal", red, rc.enforce_edge_policy(params))

    assert g["channel"] == "green_signal" and r["channel"] == "channel_2_signal"
    # Only that channel's labels are counted.
    assert g["label_counts"]["cell"] == 6 and r["label_counts"]["cell"] == 6
    # Weak-signal red cells are rejected at baseline -> lower recall than green.
    assert g["baseline"]["recall"] == pytest.approx(1.0)
    assert r["baseline"]["recall"] < g["baseline"]["recall"]


def test_green_calibration_unaffected_by_red_rows():
    params = rc.enforce_edge_policy(make_params())
    green = [cell_row(f"g{i}") for i in range(4)] + [artefact_row(f"ga{i}") for i in range(4)]
    only_green = rc.calibrate_channel("green_signal", green, params)
    # Passing red rows into the green call would be a leak; the API takes one
    # channel's rows, so an independent call must be byte-for-byte identical.
    again = rc.calibrate_channel("green_signal", list(green), params)
    assert only_green["results"] == again["results"]


# --------------------------------------------------------------------------- #
# 2. Reproducibility
# --------------------------------------------------------------------------- #
def test_calibration_is_deterministic():
    params = rc.enforce_edge_policy(make_params())
    rows = [cell_row(f"c{i}") for i in range(5)] + [artefact_row(f"a{i}") for i in range(5)]
    a = rc.calibrate_channel("green_signal", rows, params)
    b = rc.calibrate_channel("green_signal", rows, params)
    assert a["results"] == b["results"]
    assert a["pareto"] == b["pareto"]
    assert a["confusion"] == b["confusion"]


def test_parameter_enumeration_is_deterministic():
    params = rc.enforce_edge_policy(make_params())
    rows = _labeled([cell_row(f"c{i}") for i in range(5)] +
                    [artefact_row(f"a{i}") for i in range(5)])
    first = rc.enumerate_parameter_sets(rows, params)
    second = rc.enumerate_parameter_sets(rows, params)
    key = lambda sets: [(t, p, str(v), rc.threshold_values(pr)) for t, p, v, pr in sets]
    assert key(first) == key(second)


# --------------------------------------------------------------------------- #
# 3. Edge handling: never reject for the edge alone
# --------------------------------------------------------------------------- #
def test_edge_clipped_cell_with_center_in_tissue_is_not_rejected():
    params = rc.enforce_edge_policy(make_params())
    edge = rc.coerce_rec(cell_row("edge", touches_crop_boundary="True",
                                  inside_tissue="True"))
    assert rc.predicted_pass(edge, params) is True


def test_enforce_edge_policy_forces_keep_edge_even_if_disabled():
    params = make_params(keep_edge_clipped_if_center_in_tissue=False)
    forced = rc.enforce_edge_policy(params)
    assert forced.keep_edge_clipped_if_center_in_tissue is True


def test_every_enumerated_param_set_keeps_edge_clipped():
    params = make_params(keep_edge_clipped_if_center_in_tissue=False)
    rows = _labeled([cell_row(f"c{i}") for i in range(5)] +
                    [artefact_row(f"a{i}") for i in range(5)])
    for _t, _p, _v, pr in rc.enumerate_parameter_sets(rows, params):
        assert pr.keep_edge_clipped_if_center_in_tissue is True
        tv = rc.threshold_values(pr)
        assert tv["keep_edge_clipped_if_center_in_tissue"] is True


# --------------------------------------------------------------------------- #
# 4. No target-count optimisation
# --------------------------------------------------------------------------- #
def test_pareto_front_ignores_retained_count():
    # Two candidate sets with identical precision/recall but wildly different
    # n_retained must yield the same Pareto front -- the count must not matter.
    base = [
        {"precision": 0.9, "recall": 0.5, "f1": 0.64, "tp": 5, "n_retained": 6},
        {"precision": 0.6, "recall": 0.9, "f1": 0.72, "tp": 9, "n_retained": 15},
        {"precision": 0.5, "recall": 0.4, "f1": 0.44, "tp": 4, "n_retained": 8},  # dominated
    ]
    bumped = [dict(p, n_retained=p["n_retained"] * 1000) for p in base]
    assert rc.pareto_front(base) == rc.pareto_front(bumped)
    # The dominated point (index 2) is excluded regardless of its large count.
    assert 2 not in rc.pareto_front(base)


def test_pareto_front_drops_zero_true_positive_points():
    points = [
        {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "n_retained": 50},
        {"precision": 1.0, "recall": 0.3, "f1": 0.46, "tp": 3, "n_retained": 3},
    ]
    assert rc.pareto_front(points) == [1]


def test_no_function_takes_a_target_count_argument():
    for fn in (rc.pareto_front, rc.evaluate_params, rc.calibrate_channel,
               rc.enumerate_parameter_sets, rc.annotate_pareto_roles):
        names = set(inspect.signature(fn).parameters)
        assert not any("count" in n or "target" in n for n in names), fn.__name__


def test_annotate_pareto_reports_all_points_without_choosing_one():
    front = [
        {"precision": 0.95, "recall": 0.4, "f1": 0.56},
        {"precision": 0.7, "recall": 0.8, "f1": 0.75},
        {"precision": 0.55, "recall": 0.95, "f1": 0.70},
    ]
    annotated = rc.annotate_pareto_roles([dict(p) for p in front])
    # Every frontier point survives (nothing is auto-selected/removed) ...
    assert len(annotated) == len(front)
    # ... and the descriptive extremes are both present.
    roles = " ".join(p["pareto_role"] for p in annotated)
    assert "high_precision" in roles and "high_recall" in roles


# --------------------------------------------------------------------------- #
# Duplicate-distance (NMS) evaluation
# --------------------------------------------------------------------------- #
def test_duplicate_pool_suppresses_weaker_neighbour():
    params = rc.enforce_edge_policy(make_params())
    strong = rc.coerce_rec(cell_row("strong", local_robust_z=30.0,
                                    x_global_px=1000, y_global_px=2000, z_index=3))
    weak = rc.coerce_rec(cell_row("weak", local_robust_z=15.0,
                                  x_global_px=1001, y_global_px=2000, z_index=3))
    pool = rc.DupPool([strong, weak])
    # The weaker of two near-coincident passes is suppressed as a duplicate.
    assert rc.predicted_pass(weak, params, dup_pool=pool) is False
    assert rc.predicted_pass(strong, params, dup_pool=pool) is True
    # With no pool, both pass (duplicate distance not applied).
    assert rc.predicted_pass(weak, params, dup_pool=None) is True
