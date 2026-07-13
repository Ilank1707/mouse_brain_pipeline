"""Focused tests for the human-label validation batch builder.

Covers reproducible sampling, separate green/red channels, failure stratification
by ``preliminary_rule_reason``, absence of any target-count optimisation, and that
the original run is never modified.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_validation_batch as mvb  # noqa: E402
from mouse_brain_pipeline import rule_calibration as rc  # noqa: E402

FAIL_REASONS = (
    "too_large",
    "insufficient_support_planes",
    "component_xy_area_too_small",
    "insufficient_signal_to_background",
    "too_elongated",
)

CANDIDATE_COLUMNS = [
    "candidate_id", "channel", "section", "preliminary_sampling_category",
    "current_status", "preliminary_rule_reason", "x_global_px", "y_global_px",
    "z_index", "xy_area_um2", "volume_um3", "support_plane_count",
    "supporting_voxel_count", "local_robust_z", "equivalent_diameter_um",
    "xy_diameter_um", "elongation", "touches_crop_boundary", "inside_tissue",
]


def _write_run(tmp_path, *, passes_per_channel=400, fails_per_reason=40):
    """Synthetic run with green+red passes and fails across several reasons."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    rows = []
    for channel in ("green_signal", "channel_2_signal"):
        idx = 0
        for _ in range(passes_per_channel):
            rows.append(_row(channel, idx, "preliminary_rule_pass", ""))
            idx += 1
        for reason in FAIL_REASONS:
            for _ in range(fails_per_reason):
                rows.append(_row(channel, idx, "preliminary_rule_fail", reason))
                idx += 1
    with (run_dir / "all_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return run_dir


def _row(channel, idx, category, reason):
    return {
        "candidate_id": f"{channel}_s070_{idx:05d}",
        "channel": channel,
        "section": 70,
        "preliminary_sampling_category": category,
        "current_status": category,
        "preliminary_rule_reason": reason,
        "x_global_px": 1000 + idx,
        "y_global_px": 2000 + idx,
        "z_index": 3,
        "xy_area_um2": 100.0,
        "volume_um3": 300.0,
        "support_plane_count": 3,
        "supporting_voxel_count": 40,
        "local_robust_z": 12.0,
        "equivalent_diameter_um": 9.0,
        "xy_diameter_um": 9.0,
        "elongation": 1.5,
        "touches_crop_boundary": "False",
        "inside_tissue": "True",
    }


def _generate(run_dir, out_dir, **kwargs):
    return mvb.generate_validation_batch(
        config=None, run_dir=run_dir, section=70, out_dir=out_dir, render=False, **kwargs)


def _read_batch(out_dir):
    with (out_dir / "validation_review_batch.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# Reproducible sampling
# --------------------------------------------------------------------------- #
def test_sampling_is_reproducible_across_calls(tmp_path):
    run_dir = _write_run(tmp_path)
    _generate(run_dir, tmp_path / "a", samples_per_status=50, random_seed=99)
    _generate(run_dir, tmp_path / "b", samples_per_status=50, random_seed=99)
    ids_a = [r["candidate_id"] for r in _read_batch(tmp_path / "a")]
    ids_b = [r["candidate_id"] for r in _read_batch(tmp_path / "b")]
    assert ids_a == ids_b and len(ids_a) > 0


def test_sample_fails_helper_is_deterministic(tmp_path):
    run_dir = _write_run(tmp_path)
    fails = [r for r in _read_all(run_dir)
             if r["channel"] == "green_signal"
             and r["preliminary_sampling_category"] == "preliminary_rule_fail"]
    a, alloc_a, _ = mvb.sample_fails(fails, 100, 7, "green_signal")
    b, alloc_b, _ = mvb.sample_fails(fails, 100, 7, "green_signal")
    assert [r["candidate_id"] for r in a] == [r["candidate_id"] for r in b]
    assert alloc_a == alloc_b


# --------------------------------------------------------------------------- #
# Separate green / red channels
# --------------------------------------------------------------------------- #
def test_channels_are_sampled_separately(tmp_path):
    run_dir = _write_run(tmp_path)
    _generate(run_dir, tmp_path / "out", samples_per_status=50, random_seed=1)
    rows = _read_batch(tmp_path / "out")
    green = {r["candidate_id"] for r in rows if r["channel"] == "green_signal"}
    red = {r["candidate_id"] for r in rows if r["channel"] == "channel_2_signal"}
    assert green and red
    assert green.isdisjoint(red)
    assert all(cid.startswith("green_signal") for cid in green)
    assert all(cid.startswith("channel_2_signal") for cid in red)
    # 50 passes + 50 fails per channel.
    assert len(green) == 100 and len(red) == 100


def test_green_sample_unaffected_by_red_rows(tmp_path):
    run_dir = _write_run(tmp_path)
    all_rows = _read_all(run_dir)
    green_pass = [r for r in all_rows if r["channel"] == "green_signal"
                  and r["preliminary_sampling_category"] == "preliminary_rule_pass"]
    picked_alone = mvb.sample_passes(green_pass, 30, 5, "green_signal")
    # Adding red rows to the pool the function never sees must not matter: the
    # helper only receives green rows, so the pick is a pure function of them.
    picked_again = mvb.sample_passes(list(green_pass), 30, 5, "green_signal")
    assert [r["candidate_id"] for r in picked_alone] == \
           [r["candidate_id"] for r in picked_again]


# --------------------------------------------------------------------------- #
# Stratification by preliminary_rule_reason
# --------------------------------------------------------------------------- #
def test_failures_are_stratified_across_reasons(tmp_path):
    run_dir = _write_run(tmp_path)
    fails = [r for r in _read_all(run_dir)
             if r["channel"] == "green_signal"
             and r["preliminary_sampling_category"] == "preliminary_rule_fail"]
    picked, allocation, sizes = mvb.sample_fails(fails, 100, 3, "green_signal")

    assert sum(allocation.values()) == 100  # every requested slot allocated
    reasons_present = {r["preliminary_rule_reason"] for r in picked}
    # Every non-empty failure reason is represented in the sample.
    assert reasons_present == set(FAIL_REASONS)
    for reason in FAIL_REASONS:
        assert allocation[reason] >= 1
        assert allocation[reason] <= sizes[reason]


def test_allocation_takes_all_when_population_below_request():
    sizes = {"a": 10, "b": 5, "c": 3}
    alloc = mvb.allocate_by_reason(sizes, 100)
    assert alloc == sizes  # cannot invent candidates; take all


def test_summary_csv_reports_per_reason_rows(tmp_path):
    run_dir = _write_run(tmp_path)
    _generate(run_dir, tmp_path / "out", samples_per_status=100, random_seed=1)
    with (tmp_path / "out" / "validation_sample_summary.csv").open(
            newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fail_reasons_green = {r["preliminary_rule_reason"] for r in rows
                          if r["channel"] == "green_signal" and r["stratum"].endswith("fail")}
    assert set(FAIL_REASONS) <= fail_reasons_green


# --------------------------------------------------------------------------- #
# No target-count optimisation
# --------------------------------------------------------------------------- #
def test_sampling_uses_a_fixed_count_not_a_target(tmp_path):
    # Doubling the available population must NOT change how many are sampled: the
    # count is a fixed request, never tuned toward a desired output candidate total.
    small = _write_run(tmp_path / "s", passes_per_channel=120, fails_per_reason=40)
    big = _write_run(tmp_path / "b", passes_per_channel=800, fails_per_reason=200)
    _generate(small, tmp_path / "so", samples_per_status=60, random_seed=1)
    _generate(big, tmp_path / "bo", samples_per_status=60, random_seed=1)
    per_channel_small = len([r for r in _read_batch(tmp_path / "so")
                             if r["channel"] == "green_signal"])
    per_channel_big = len([r for r in _read_batch(tmp_path / "bo")
                           if r["channel"] == "green_signal"])
    assert per_channel_small == per_channel_big == 120  # 60 pass + 60 fail, both


def test_no_sampling_or_calibration_function_takes_a_target_count():
    for fn in (mvb.generate_validation_batch, mvb.sample_passes, mvb.sample_fails,
               mvb.allocate_by_reason, rc.pareto_front, rc.evaluate_params):
        names = set(inspect.signature(fn).parameters)
        assert not any("target" in n for n in names), fn.__name__
    # And calibration's Pareto choice is invariant to retained counts.
    base = [
        {"precision": 0.9, "recall": 0.5, "f1": 0.64, "tp": 5, "n_retained": 6},
        {"precision": 0.6, "recall": 0.9, "f1": 0.72, "tp": 9, "n_retained": 15},
    ]
    bumped = [dict(p, n_retained=p["n_retained"] * 1000) for p in base]
    assert rc.pareto_front(base) == rc.pareto_front(bumped)


# --------------------------------------------------------------------------- #
# Task 2 -- explicit inverse-probability sampling weights
# --------------------------------------------------------------------------- #
def test_sampling_weight_columns_present_and_correct(tmp_path):
    run_dir = _write_run(tmp_path, passes_per_channel=400, fails_per_reason=40)
    _generate(run_dir, tmp_path / "out", samples_per_status=50, random_seed=99)
    rows = _read_batch(tmp_path / "out")
    for col in ("sampling_population_count", "sampling_selected_count",
                "sampling_probability", "sample_weight", "sampling_stratum_id",
                "spatial_tile"):
        assert col in rows[0]

    passes = [r for r in rows if r["sampling_stratum"] == "preliminary_rule_pass"
              and r["channel"] == "green_signal"]
    assert passes
    p = passes[0]
    assert int(p["sampling_population_count"]) == 400
    assert int(p["sampling_selected_count"]) == 50
    assert float(p["sampling_probability"]) == pytest.approx(50 / 400)
    assert float(p["sample_weight"]) == pytest.approx(400 / 50)  # = 8.0
    assert p["sampling_stratum_id"] == "green_signal|preliminary_rule_pass"

    fails = [r for r in rows if r["sampling_stratum"] == "preliminary_rule_fail"
             and r["channel"] == "green_signal"]
    f = fails[0]
    # probability = selected/population, weight = population/selected, per stratum.
    assert float(f["sample_weight"]) == pytest.approx(
        int(f["sampling_population_count"]) / int(f["sampling_selected_count"]))
    assert f["sampling_stratum_id"] == (
        f"green_signal|preliminary_rule_fail|{f['preliminary_rule_reason']}")


def test_weight_is_one_when_entire_stratum_selected(tmp_path):
    # Request more than the population -> the whole stratum is taken -> weight 1.0.
    run_dir = _write_run(tmp_path, passes_per_channel=30, fails_per_reason=8)
    _generate(run_dir, tmp_path / "out", samples_per_status=1000, random_seed=1)
    rows = _read_batch(tmp_path / "out")
    assert rows
    for r in rows:
        assert float(r["sampling_probability"]) == pytest.approx(1.0)
        assert float(r["sample_weight"]) == pytest.approx(1.0)


def test_validation_coverage_csv_axes_and_totals(tmp_path):
    run_dir = _write_run(tmp_path)
    _generate(run_dir, tmp_path / "out", samples_per_status=50, random_seed=1)
    with (tmp_path / "out" / "validation_coverage.csv").open(
            newline="", encoding="utf-8") as fh:
        cov = list(csv.DictReader(fh))
    for col in ("channel", "sampling_stratum", "preliminary_rule_reason",
                "spatial_tile", "peak_optical_plane", "candidate_generation_source",
                "count"):
        assert col in cov[0]
    batch = _read_batch(tmp_path / "out")
    assert sum(int(r["count"]) for r in cov) == len(batch)  # coverage covers the sample


def test_spatial_tile_is_configurable_and_not_in_probability(tmp_path):
    run_dir = _write_run(tmp_path)
    _generate(run_dir, tmp_path / "big", samples_per_status=50, random_seed=1,
              spatial_tile_size=1024)
    _generate(run_dir, tmp_path / "small", samples_per_status=50, random_seed=1,
              spatial_tile_size=64)
    big = {r["candidate_id"]: r for r in _read_batch(tmp_path / "big")}
    small = {r["candidate_id"]: r for r in _read_batch(tmp_path / "small")}
    # A finer tiling changes the tile ids ...
    assert any(big[c]["spatial_tile"] != small[c]["spatial_tile"] for c in big)
    # ... but the inclusion probability and weight are identical (design unchanged).
    for c in big:
        assert big[c]["sample_weight"] == small[c]["sample_weight"]
        assert big[c]["sampling_probability"] == small[c]["sampling_probability"]


def test_stratum_id_and_spatial_tile_helpers():
    assert mvb.stratum_id("green_signal", "preliminary_rule_pass", "") == \
        "green_signal|preliminary_rule_pass"
    assert mvb.stratum_id("channel_2_signal", "preliminary_rule_fail", "too_large") == \
        "channel_2_signal|preliminary_rule_fail|too_large"
    assert mvb.spatial_tile({"x_global_px": 2050, "y_global_px": 1024}, 1024) == "2_1"
    assert mvb.spatial_tile({"x_global_px": "n/a", "y_global_px": 5}, 1024) == ""


# --------------------------------------------------------------------------- #
# Task 2 -- batch validation
# --------------------------------------------------------------------------- #
def _wrec(cid, channel, stratum, reason, population, selected):
    return {
        "candidate_id": cid, "channel": channel,
        "sampling_stratum": stratum, "fail_reason_stratum": reason,
        "preliminary_rule_reason": reason,
        "sampling_population_count": population, "sampling_selected_count": selected,
        "sampling_probability": selected / population,
        "sample_weight": population / selected,
    }


def _wsummary(channel, stratum, reason, population, sampled):
    return {"channel": channel, "stratum": stratum, "preliminary_rule_reason": reason,
            "population_count": population, "sampled_count": sampled}


def test_validate_batch_accepts_a_consistent_batch():
    batch = [_wrec(f"g{i}", "green_signal", "preliminary_rule_pass", "", 100, 2)
             for i in range(2)]
    summary = [_wsummary("green_signal", "preliminary_rule_pass", "", 100, 2)]
    mvb.validate_batch(batch, summary)  # does not raise


def test_validate_batch_rejects_duplicate_ids():
    batch = [_wrec("dup", "green_signal", "preliminary_rule_pass", "", 100, 2),
             _wrec("dup", "green_signal", "preliminary_rule_pass", "", 100, 2)]
    summary = [_wsummary("green_signal", "preliminary_rule_pass", "", 100, 2)]
    with pytest.raises(ValueError, match="not unique"):
        mvb.validate_batch(batch, summary)


def test_validate_batch_rejects_selected_over_population():
    bad = _wrec("g0", "green_signal", "preliminary_rule_pass", "", 2, 5)
    with pytest.raises(ValueError, match="exceeds population"):
        mvb.validate_batch([bad], [_wsummary("green_signal", "preliminary_rule_pass", "", 2, 5)])


def test_validate_batch_rejects_out_of_range_probability_or_weight():
    bad_prob = _wrec("g0", "green_signal", "preliminary_rule_pass", "", 100, 2)
    bad_prob["sampling_probability"] = 1.5
    with pytest.raises(ValueError, match="sampling_probability"):
        mvb.validate_batch([bad_prob], [_wsummary("green_signal", "preliminary_rule_pass", "", 100, 2)])
    bad_weight = _wrec("g0", "green_signal", "preliminary_rule_pass", "", 100, 2)
    bad_weight["sample_weight"] = 0.5
    with pytest.raises(ValueError, match="sample_weight"):
        mvb.validate_batch([bad_weight], [_wsummary("green_signal", "preliminary_rule_pass", "", 100, 2)])


def test_validate_batch_rejects_summary_mismatch():
    # Three sampled rows but the summary claims only two -> cannot reconcile.
    batch = [_wrec(f"g{i}", "green_signal", "preliminary_rule_pass", "", 100, 2)
             for i in range(3)]
    summary = [_wsummary("green_signal", "preliminary_rule_pass", "", 100, 2)]
    with pytest.raises(ValueError, match="reconcile"):
        mvb.validate_batch(batch, summary)


# --------------------------------------------------------------------------- #
# Original run is never modified
# --------------------------------------------------------------------------- #
def test_original_run_is_not_modified(tmp_path):
    run_dir = _write_run(tmp_path)
    before = _snapshot(run_dir)
    out_dir = tmp_path / "elsewhere" / "validation_070"  # outside the run
    _generate(run_dir, out_dir, samples_per_status=40, random_seed=1)
    assert _snapshot(run_dir) == before  # run bytes + mtimes unchanged
    assert (out_dir / "validation_review_batch.csv").is_file()
    assert (out_dir / "validation_sample_summary.csv").is_file()
    # Nothing new was written inside the run directory.
    assert {p.name for p in run_dir.iterdir()} == {"all_candidates.csv"}


def test_human_label_column_is_blank_and_columns_present(tmp_path):
    run_dir = _write_run(tmp_path)
    _generate(run_dir, tmp_path / "out", samples_per_status=20, random_seed=1)
    rows = _read_batch(tmp_path / "out")
    assert rows
    assert all(r["human_label"] == "" for r in rows)
    for required in ("candidate_id", "channel", "current_status",
                     "preliminary_rule_reason", "x_global_px", "xy_area_um2",
                     "volume_um3", "support_plane_count", "supporting_voxel_count",
                     "local_robust_z", "touches_crop_boundary", "human_label"):
        assert required in rows[0]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_all(run_dir):
    with (run_dir / "all_candidates.csv").open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _snapshot(directory):
    snapshot = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            snapshot[str(path.relative_to(directory))] = (
                len(data), path.stat().st_mtime_ns, hashlib.sha1(data).hexdigest())
    return snapshot
