"""Focused tests for inverse-probability weighted calibration.

Covers weight verification/refusal, the weighting maths, weighted Pareto ranking,
deterministic stratified bootstrap CIs, green/red separation, and the absence of
any target-count optimisation.
"""

from __future__ import annotations

import inspect

import pytest

from mouse_brain_pipeline.candidate_detection import DetectionParams
from mouse_brain_pipeline import rule_calibration as rc


def make_params(**overrides):
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


def _feat(strong):
    if strong:
        return dict(xy_area_um2=120.0, volume_um3=320.0, support_plane_count=3,
                    supporting_voxel_count=40, local_robust_z=22.0,
                    equivalent_diameter_um=10.0, xy_diameter_um=9.0, elongation=1.4,
                    n_consecutive_planes=3)
    return dict(xy_area_um2=8.0, volume_um3=20.0, support_plane_count=1,
                supporting_voxel_count=3, local_robust_z=3.0,
                equivalent_diameter_um=3.0, xy_diameter_um=3.0, elongation=1.2,
                n_consecutive_planes=1)


def row(cid, channel, stratum, reason, label, *, strong):
    r = dict(candidate_id=cid, channel=channel, human_label=label,
             sampling_stratum=stratum, fail_reason_stratum=reason,
             touches_crop_boundary="False", inside_tissue="True",
             invalid_coordinate="False", measurement_valid="True",
             original_cellfinder_z_valid="True", xy_centroid_shift_um=1.0,
             z_index=3, x_global_px=1000 + hash(cid) % 500, y_global_px=2000)
    r.update(_feat(strong))
    return r


def summary_row(channel, stratum, reason, population, sampled):
    return {"channel": channel, "stratum": stratum, "preliminary_rule_reason": reason,
            "population_count": population, "allocated": sampled, "sampled_count": sampled,
            "random_seed": 1}


# --------------------------------------------------------------------------- #
# Weight verification / refusal
# --------------------------------------------------------------------------- #
def test_weights_are_inverse_probability_per_stratum():
    batch = ([row(f"g_p{i}", "green_signal", "preliminary_rule_pass", "", "cell", strong=True)
              for i in range(4)]
             + [row(f"g_f{i}", "green_signal", "preliminary_rule_fail", "too_large",
                    "artefact", strong=True) for i in range(2)])
    summary = [
        summary_row("green_signal", "preliminary_rule_pass", "", 1000, 4),
        summary_row("green_signal", "preliminary_rule_fail", "too_large", 100, 2),
    ]
    weights = rc.verify_and_build_weights(summary, batch)
    assert weights["green_signal"][("preliminary_rule_pass", "")] == pytest.approx(250.0)
    assert weights["green_signal"][("preliminary_rule_fail", "too_large")] == pytest.approx(50.0)


def test_verification_refuses_on_sampled_count_mismatch():
    # Summary says 4 sampled but only 3 batch rows exist -> cannot verify.
    batch = [row(f"g_p{i}", "green_signal", "preliminary_rule_pass", "", "cell", strong=True)
             for i in range(3)]
    summary = [summary_row("green_signal", "preliminary_rule_pass", "", 1000, 4)]
    with pytest.raises(rc.WeightVerificationError):
        rc.verify_and_build_weights(summary, batch)


def test_verification_refuses_on_missing_summary_row():
    batch = [row("g_f0", "green_signal", "preliminary_rule_fail", "too_large",
                 "artefact", strong=True)]
    with pytest.raises(rc.WeightVerificationError):
        rc.verify_and_build_weights([], batch)


def test_verification_refuses_when_population_below_sample():
    batch = [row(f"g_p{i}", "green_signal", "preliminary_rule_pass", "", "cell", strong=True)
             for i in range(4)]
    summary = [summary_row("green_signal", "preliminary_rule_pass", "", 2, 4)]
    with pytest.raises(rc.WeightVerificationError):
        rc.verify_and_build_weights(summary, batch)


# --------------------------------------------------------------------------- #
# Weighting maths
# --------------------------------------------------------------------------- #
def test_weighted_confusion_reweights_metrics():
    preds = [True, True, False]
    labels = ["cell", "artefact", "cell"]
    unweighted = rc.weighted_confusion(preds, labels, None)
    assert unweighted["precision"] == pytest.approx(0.5)
    weighted = rc.weighted_confusion(preds, labels, [1.0, 10.0, 1.0])
    # The upweighted artefact false positive drags precision down.
    assert weighted["fp"] == pytest.approx(10.0)
    assert weighted["precision"] == pytest.approx(1.0 / 11.0)
    assert weighted["precision"] < unweighted["precision"]


def test_channels_get_independent_weights():
    batch = ([row(f"g{i}", "green_signal", "preliminary_rule_pass", "", "cell", strong=True)
              for i in range(2)]
             + [row(f"r{i}", "channel_2_signal", "preliminary_rule_pass", "", "cell",
                    strong=True) for i in range(5)])
    summary = [
        summary_row("green_signal", "preliminary_rule_pass", "", 800, 2),
        summary_row("channel_2_signal", "preliminary_rule_pass", "", 100, 5),
    ]
    weights = rc.verify_and_build_weights(summary, batch)
    assert weights["green_signal"][("preliminary_rule_pass", "")] == pytest.approx(400.0)
    assert weights["channel_2_signal"][("preliminary_rule_pass", "")] == pytest.approx(20.0)
    assert set(weights) == {"green_signal", "channel_2_signal"}


# --------------------------------------------------------------------------- #
# Weighted calibration end to end
# --------------------------------------------------------------------------- #
def _weighted_scenario_rows():
    rows = []
    rows += [row(f"c{i}", "green_signal", "preliminary_rule_pass", "", "cell", strong=True)
             for i in range(4)]                      # TP, weight 1
    rows += [row(f"a{i}", "green_signal", "preliminary_rule_fail", "too_large",
                 "artefact", strong=True) for i in range(2)]   # FP, weight 10
    rows += [row(f"m{i}", "green_signal", "preliminary_rule_fail", "too_large",
                 "cell", strong=False) for i in range(2)]      # FN, weight 10
    return rows


def test_calibrate_channel_weighted_applies_weights_at_baseline():
    rows = _weighted_scenario_rows()
    weights = {("preliminary_rule_pass", ""): 1.0, ("preliminary_rule_fail", "too_large"): 10.0}
    result = rc.calibrate_channel_weighted(
        "green_signal", rows, make_params(), weights, dup_pool=None, n_bootstrap=200)
    base = result["baseline"]
    assert (base["tp"], base["fp"], base["fn"]) == (4, 2, 2)
    assert base["precision"] == pytest.approx(4 / 6, abs=1e-3)
    # Weighting: TP=4*1, FP=2*10, FN=2*10 -> precision 4/24.
    assert base["weighted_tp"] == pytest.approx(4.0)
    assert base["weighted_fp"] == pytest.approx(20.0)
    assert base["weighted_precision"] == pytest.approx(4 / 24, abs=1e-3)
    assert base["weighted_precision"] < base["precision"]
    # Confusion matrix carries weighted counts; CIs cover the point estimate.
    assert any(r["weighted_count"] for r in result["confusion"])
    ci = next(c for c in result["confidence_intervals"] if c["setting"] == "baseline_current")
    assert ci["weighted_precision_ci_low"] <= base["weighted_precision"] <= \
           ci["weighted_precision_ci_high"] + 1e-9


def test_weighted_calibration_is_reproducible():
    rows = _weighted_scenario_rows()
    weights = {("preliminary_rule_pass", ""): 1.0, ("preliminary_rule_fail", "too_large"): 10.0}
    a = rc.calibrate_channel_weighted("green_signal", rows, make_params(), weights,
                                      dup_pool=None, n_bootstrap=200, bootstrap_seed=5)
    b = rc.calibrate_channel_weighted("green_signal", rows, make_params(), weights,
                                      dup_pool=None, n_bootstrap=200, bootstrap_seed=5)
    assert a["results"] == b["results"]
    assert a["confidence_intervals"] == b["confidence_intervals"]  # deterministic bootstrap


def test_weighted_pareto_ranks_by_weighted_metrics():
    # Two distinct settings; unweighted would prefer A, weighted prefers B.
    def rowset(area, prec_w, rec_w, prec_u, rec_u):
        r = {c: 0 for c in rc.THRESHOLD_COLUMNS}
        r["min_component_xy_area_um2"] = area
        r.update(weighted_precision=prec_w, weighted_recall=rec_w, weighted_f1=0.5,
                 weighted_tp=5.0, precision=prec_u, recall=rec_u, tp=5)
        return r
    results = [
        rowset(1, prec_w=0.4, rec_w=0.9, prec_u=0.95, rec_u=0.4),
        rowset(2, prec_w=0.95, rec_w=0.4, prec_u=0.4, rec_u=0.9),
    ]
    idx = rc._weighted_pareto_indices(results)
    # Both are non-dominated in weighted (precision, recall) space.
    assert set(idx) == {0, 1}
    # Neither would be dropped for its unweighted values; ranking used WEIGHTED.
    front = [dict(results[i]) for i in idx]
    rc._annotate_weighted_roles(front)
    roles = " ".join(p["pareto_role"] for p in front)
    assert "high_precision" in roles and "high_recall" in roles


def test_bootstrap_ci_is_deterministic_and_bounded():
    preds = [True, False, True, False, True, True]
    labels = ["cell", "cell", "artefact", "artefact", "cell", "artefact"]
    weights = [1.0, 1.0, 10.0, 10.0, 1.0, 10.0]
    strata = [("preliminary_rule_pass", "")] * 2 + [("preliminary_rule_fail", "too_large")] * 4
    a = rc.bootstrap_weighted_ci(preds, labels, weights, strata, n_bootstrap=300, seed=42)
    b = rc.bootstrap_weighted_ci(preds, labels, weights, strata, n_bootstrap=300, seed=42)
    assert a == b
    for key in ("precision_ci", "recall_ci", "f1_ci"):
        low, high = a[key]
        assert 0.0 <= low <= high <= 1.0


# --------------------------------------------------------------------------- #
# Inverse-probability weighting recovers the known full-population precision
# --------------------------------------------------------------------------- #
def test_ipw_recovers_population_precision_from_balanced_sample():
    # Full population: 990 retained cells (TP) + 10 retained artefacts (FP) -> the
    # TRUE precision is 0.99. A BALANCED review sample takes 10 from each stratum,
    # so the unweighted sample precision is a MISLEADING 0.5. Inverse-probability
    # weights (990/10 for the cell stratum, 10/10 for the artefact stratum) recover
    # the known 0.99.
    preds = [True] * 20                        # every sampled candidate is retained
    labels = ["cell"] * 10 + ["artefact"] * 10
    weights = [99.0] * 10 + [1.0] * 10         # population / sample, per stratum
    unweighted = rc.weighted_confusion(preds, labels, None)
    weighted = rc.weighted_confusion(preds, labels, weights)
    assert unweighted["precision"] == pytest.approx(0.5)             # misleading
    assert weighted["precision"] == pytest.approx(0.99, abs=1e-6)    # recovered


def test_calibrate_weighted_recovers_population_precision_end_to_end():
    batch = ([row(f"c{i}", "green_signal", "preliminary_rule_pass", "", "cell", strong=True)
              for i in range(10)]
             + [row(f"a{i}", "green_signal", "preliminary_rule_fail", "too_large",
                    "artefact", strong=True) for i in range(10)])
    # The cell stratum is 99x larger in the population than the sample suggests.
    summary = [
        summary_row("green_signal", "preliminary_rule_pass", "", 990, 10),
        summary_row("green_signal", "preliminary_rule_fail", "too_large", 10, 10),
    ]
    weights = rc.verify_and_build_weights(summary, batch)
    result = rc.calibrate_channel_weighted(
        "green_signal", batch, make_params(), weights["green_signal"],
        dup_pool=None, n_bootstrap=50)
    base = result["baseline"]
    # Every 'strong' candidate passes the rule -> all predicted pass. Unweighted
    # precision is the balanced 10/20 = 0.5; weighting recovers the true ~0.99.
    assert base["precision"] == pytest.approx(0.5, abs=1e-6)
    assert base["weighted_precision"] == pytest.approx(0.99, abs=1e-3)
    assert base["metrics_weighting"] == rc.METRICS_WEIGHTING_IPW
    assert float(base["weighted_retained_population_estimate"]) == \
        pytest.approx(1000.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Legacy (unweighted) batches still run, clearly flagged
# --------------------------------------------------------------------------- #
def test_legacy_batch_runs_unweighted_and_is_flagged():
    rows = _weighted_scenario_rows()
    result = rc.calibrate_channel_weighted(
        "green_signal", rows, make_params(), {}, dup_pool=None, n_bootstrap=50,
        metrics_weighting=rc.METRICS_WEIGHTING_LEGACY)
    base = result["baseline"]
    assert result["metrics_weighting"] == "unweighted_legacy_batch"
    assert base["metrics_weighting"] == "unweighted_legacy_batch"
    # With every weight defaulting to 1.0, weighted == unweighted ...
    assert base["weighted_precision"] == pytest.approx(base["precision"])
    # ... and the population estimate is withheld (never a legacy population number).
    assert base["weighted_retained_population_estimate"] == ""


def test_weights_from_batch_column_reads_valid_sample_weight():
    batch = [dict(channel="green_signal", sampling_stratum="preliminary_rule_pass",
                  fail_reason_stratum="", sample_weight="8.0"),
             dict(channel="green_signal", sampling_stratum="preliminary_rule_fail",
                  fail_reason_stratum="too_large", sample_weight="4.0")]
    w = rc.weights_from_batch_column(batch)
    assert w["green_signal"][("preliminary_rule_pass", "")] == pytest.approx(8.0)
    assert w["green_signal"][("preliminary_rule_fail", "too_large")] == pytest.approx(4.0)


def test_weights_from_batch_column_rejects_missing_bad_or_inconsistent():
    missing = [{"channel": "green_signal", "sampling_stratum": "preliminary_rule_pass",
                "fail_reason_stratum": ""}]
    assert rc.weights_from_batch_column(missing) is None
    below_one = [{"channel": "green_signal", "sampling_stratum": "preliminary_rule_pass",
                  "fail_reason_stratum": "", "sample_weight": "0.5"}]
    assert rc.weights_from_batch_column(below_one) is None
    inconsistent = [
        dict(channel="green_signal", sampling_stratum="preliminary_rule_pass",
             fail_reason_stratum="", sample_weight="8.0"),
        dict(channel="green_signal", sampling_stratum="preliminary_rule_pass",
             fail_reason_stratum="", sample_weight="9.0"),
    ]
    assert rc.weights_from_batch_column(inconsistent) is None


# --------------------------------------------------------------------------- #
# No target-count optimisation
# --------------------------------------------------------------------------- #
def test_no_weighting_function_takes_a_target_count():
    for fn in (rc.verify_and_build_weights, rc.calibrate_channel_weighted,
               rc.weighted_confusion, rc.bootstrap_weighted_ci,
               rc._weighted_pareto_indices):
        names = set(inspect.signature(fn).parameters)
        assert not any("target" in n or "count" in n for n in names), fn.__name__
