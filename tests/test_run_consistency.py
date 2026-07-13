"""Tests for the read-only run-consistency audit (Task 5)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mouse_brain_pipeline import run_consistency as rcs  # noqa: E402
import audit_run_consistency as audit_script  # noqa: E402


def crow(cid, channel="green_signal", *, status="preliminary_rule_pass",
         in_mask=False, in_core=False, peak=3, included=False, manual_label="",
         model_ok=False, source="both", **extra):
    row = {
        "candidate_id": cid, "channel": channel, "current_status": status,
        "inside_injection_analysis_exclusion": str(in_mask),
        "inside_injection_core": str(in_core),
        "fixed_xy_peak_z_index": str(peak),
        "included_in_count": str(included),
        "manual_label": manual_label,
        "model_validation_passed": str(model_ok),
        "candidate_generation_source": source,
    }
    row.update(extra)
    return row


def clean_run(n=6):
    return ([crow(f"green_signal_{i}", "green_signal") for i in range(n)]
            + [crow(f"channel_2_signal_{i}", "channel_2_signal") for i in range(n)])


def _pass(check):
    return check["passed"]


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def test_candidate_ids_unique_and_present():
    assert _pass(rcs.check_candidate_ids(clean_run()))
    assert not _pass(rcs.check_candidate_ids(
        [crow("dup"), crow("dup")]))
    assert not _pass(rcs.check_candidate_ids([crow("")]))


def test_status_counts_sum():
    assert _pass(rcs.check_status_counts_sum(clean_run()))
    assert not _pass(rcs.check_status_counts_sum([crow("x", status="")]))


def test_mask_partition_and_core_subset():
    assert _pass(rcs.check_mask_partition(clean_run()))
    # core True but analysis-exclusion False is geometrically impossible.
    bad = [crow("x", in_core=True, in_mask=False)]
    assert not _pass(rcs.check_mask_partition(bad))
    # core inside the analysis mask is fine.
    ok = [crow("x", in_core=True, in_mask=True)]
    assert _pass(rcs.check_mask_partition(ok))


def test_peak_plane_out_of_range_is_error_but_blank_is_not():
    assert _pass(rcs.check_peak_planes(clean_run()))
    assert not _pass(rcs.check_peak_planes([crow("x", peak=9)]))       # 9 >= 7 planes
    # A blank/missing peak Z is 'unassigned', not a hard error.
    assert _pass(rcs.check_peak_planes([crow("x", peak="")]))


def test_included_in_count_requires_confirmation():
    assert _pass(rcs.check_included_in_count(clean_run()))
    # included with no confirmation -> hard error.
    assert not _pass(rcs.check_included_in_count([crow("x", included=True)]))
    # included WITH a human 'cell' label -> allowed.
    assert _pass(rcs.check_included_in_count(
        [crow("x", included=True, manual_label="cell")]))
    # included WITH a validated model prediction -> allowed.
    assert _pass(rcs.check_included_in_count(
        [crow("x", included=True, status="predicted_cell", model_ok=True)]))


def test_green_red_ids_never_mixed():
    assert _pass(rcs.check_channel_id_separation(clean_run()))
    # An id embedding the wrong channel token is a mismatch.
    assert not _pass(rcs.check_channel_id_separation(
        [crow("green_signal_1", channel="channel_2_signal")]))
    # The same id under two channels is a hard error.
    shared = [crow("shared", channel="green_signal"),
              crow("shared", channel="channel_2_signal")]
    assert not _pass(rcs.check_channel_id_separation(shared))


def test_coordinate_exports_reconcile():
    cands = clean_run(3)
    all_ids = {c["candidate_id"] for c in cands}
    good = {"all_candidate_coordinates.csv": set(all_ids),
            "preliminary_pass_coordinates.csv": set(list(all_ids)[:2])}
    assert _pass(rcs.check_coordinate_exports(cands, good))
    # A candidate missing from the all-coordinates export -> disappearance -> error.
    missing = {"all_candidate_coordinates.csv": set(list(all_ids)[:-1])}
    assert not _pass(rcs.check_coordinate_exports(cands, missing))
    # A phantom id in a subset export not present in all_candidates -> error.
    phantom = {"all_candidate_coordinates.csv": set(all_ids),
               "manual_review_coordinates.csv": {"ghost"}}
    assert not _pass(rcs.check_coordinate_exports(cands, phantom))
    # No exports supplied -> skipped (passes, informational).
    assert _pass(rcs.check_coordinate_exports(cands, None))


# --------------------------------------------------------------------------- #
# Suppression sensitivity is a WARNING, not an error
# --------------------------------------------------------------------------- #
def test_suppression_sensitivity_is_warning_not_error():
    # green: all outside-mask candidates from a single pass -> 0% both -> warning.
    green = [crow(f"green_signal_{i}", "green_signal", source="raw_stack")
             for i in range(10)]
    report = rcs.audit_run(green)
    assert report["ok"] is True                       # warning does NOT fail the run
    assert report["n_warnings"] >= 1
    warned = {w["name"] for w in report["warnings"]}
    assert any("suppression_sensitivity" in n for n in warned)


# --------------------------------------------------------------------------- #
# Top-level audit_run
# --------------------------------------------------------------------------- #
def test_audit_run_ok_on_clean_run():
    report = rcs.audit_run(clean_run())
    assert report["ok"] is True and report["n_errors"] == 0


def test_audit_run_flags_hard_error():
    bad = clean_run() + [crow("bad", included=True)]   # unconfirmed inclusion
    report = rcs.audit_run(bad)
    assert report["ok"] is False and report["n_errors"] >= 1


# --------------------------------------------------------------------------- #
# Script main()
# --------------------------------------------------------------------------- #
def _write_run(tmp_path, candidates, *, with_exports=True):
    run_dir = tmp_path / "run"
    (run_dir / "coordinate_exports").mkdir(parents=True)
    with (run_dir / "all_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(candidates[0].keys()))
        writer.writeheader(); writer.writerows(candidates)
    if with_exports:
        ids = [{"candidate_id": c["candidate_id"]} for c in candidates]
        with (run_dir / "coordinate_exports" / "all_candidate_coordinates.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["candidate_id"])
            writer.writeheader(); writer.writerows(ids)
    return run_dir


def test_script_main_returns_zero_on_clean_run(tmp_path):
    run_dir = _write_run(tmp_path, clean_run())
    rc = audit_script.main(["--run-dir", str(run_dir)])
    assert rc == 0
    report = json.loads((run_dir / "run_consistency_report.json").read_text(encoding="utf-8"))
    assert report["ok"] is True


def test_script_main_returns_one_on_hard_error(tmp_path):
    run_dir = _write_run(tmp_path, clean_run() + [crow("bad", included=True)])
    before = (run_dir / "all_candidates.csv").read_bytes()
    rc = audit_script.main(["--run-dir", str(run_dir)])
    assert rc == 1
    # The audit never modifies the run's candidate table.
    assert (run_dir / "all_candidates.csv").read_bytes() == before
