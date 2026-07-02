"""Run isolation, peak-plane assignment and coordinate-export tests.

Proves (pure-logic subset of the required proofs):

1.  Every run receives a new isolated output directory.
3.  ``latest_run.json`` is only written by the explicit success call.
5.  Every valid candidate is assigned to exactly one peak plane.
6.  Seven peak-plane counts + unassigned == unique total.
13. Simplified coordinate CSVs contain the expected categories.
14. Confirmed-cell CSV excludes preliminary rule passes.
15. One-based ``peak_optical_plane`` is derived from zero-based Z.
16. Missing/invalid peak Z is reported (unassigned), never guessed.
17. Status counts reconcile exactly.
18. One section is reported as a single section, not a whole brain.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mouse_brain_pipeline import coordinate_exports as ce  # noqa: E402
from mouse_brain_pipeline import run_layout as rl  # noqa: E402


def _cand(cid, peak, status="preliminary_rule_pass", **extra):
    row = {
        "candidate_id": cid, "channel": "green_signal", "section": 70,
        "x_global_px": 10, "y_global_px": 20, "fixed_xy_peak_z_index": peak,
        "fixed_xy_support_z_indices": extra.pop("support", ""),
        "current_status": status, "preliminary_sampling_category": status,
    }
    row.update(extra)
    return row


# 1 ------------------------------------------------------------------------- #
def test_every_run_gets_a_new_isolated_directory(tmp_path):
    a = rl.create_run_dir(tmp_path, "run_a")
    b = rl.create_run_dir(tmp_path, "run_b")
    assert a != b
    assert a.exists() and b.exists()
    for sub in rl.RUN_SUBDIRS:
        assert (a / sub).is_dir() and (b / sub).is_dir()
    # A non-empty existing run is never silently overwritten.
    (a / "all_candidates.csv").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        rl.create_run_dir(tmp_path, "run_a")


def test_make_run_id_uses_name_or_timestamp():
    assert rl.make_run_id("section070_test_07", 70) == "section070_test_07"
    assert rl.make_run_id("has spaces/slash", 70) == "has_spaces_slash"
    from datetime import datetime, timezone
    auto = rl.make_run_id(None, 70, now=datetime(2026, 6, 30, 13, 20, 0, tzinfo=timezone.utc))
    assert auto == "20260630_132000_section070"


# 3 ------------------------------------------------------------------------- #
def test_latest_run_json_only_written_on_explicit_success(tmp_path):
    rl.create_run_dir(tmp_path, "run_a")
    # Creating the run dir alone must NOT write latest_run.json.
    assert rl.read_latest_run(tmp_path) is None
    run_dir = rl.runs_root(tmp_path) / "run_a"
    rl.write_latest_run(tmp_path, run_dir, {"candidate_count": 5})
    latest = rl.read_latest_run(tmp_path)
    assert latest["candidate_count"] == 5
    assert Path(latest["run_dir"]).name == "run_a"


# 5 + 6 --------------------------------------------------------------------- #
def test_every_valid_candidate_assigned_to_exactly_one_peak_plane():
    candidates = [_cand("c1", "0"), _cand("c2", "3"), _cand("c3", "6"),
                  _cand("c4", "3"), _cand("c5", "")]  # c5 unassigned
    assignments, unassigned = ce.assign_peak_planes(candidates)
    seen = [c["candidate_id"] for rows in assignments.values() for c in rows]
    seen += [c["candidate_id"] for c in unassigned]
    assert sorted(seen) == ["c1", "c2", "c3", "c4", "c5"]   # each appears once
    assert len(seen) == len(set(seen))                      # no duplicates
    rec = ce.reconcile(candidates)
    assert rec["unique_total"] == 5
    assert rec["assigned_total"] == 4 and rec["unassigned_total"] == 1
    assert rec["assigned_total"] + rec["unassigned_total"] == rec["unique_total"]
    assert rec["peak_assignment_reconciles"] is True


# 15 + 16 ------------------------------------------------------------------- #
def test_one_based_plane_and_invalid_peak_is_unassigned():
    assert ce.peak_optical_plane(_cand("c", "0")) == 1     # z-index 0 -> plane 01
    assert ce.peak_optical_plane(_cand("c", "6")) == 7     # z-index 6 -> plane 07
    for bad in ("", "abc", "3.5", "7", "-1", None):
        assert ce.peak_optical_plane(_cand("c", bad)) is None   # never guessed
    _assignments, unassigned = ce.assign_peak_planes([_cand("c", "9")])
    assert [c["candidate_id"] for c in unassigned] == ["c"]


# 17 ------------------------------------------------------------------------ #
def test_status_counts_reconcile_exactly():
    candidates = [_cand("c1", "1", "preliminary_rule_pass"),
                  _cand("c2", "2", "manual_review"),
                  _cand("c3", "3", "artifact")]
    rec = ce.reconcile(candidates)
    assert sum(rec["status_counts"].values()) == rec["unique_total"] == 3
    assert rec["status_reconciles"] is True


# 13 + 14 ------------------------------------------------------------------- #
def test_coordinate_exports_categories_and_confirmed_cell_excludes_pass(tmp_path):
    candidates = [
        _cand("pass1", "1", "preliminary_rule_pass"),
        _cand("fail1", "2", "preliminary_rule_fail", preliminary_sampling_category="preliminary_rule_fail"),
        _cand("review1", "3", "manual_review"),
        _cand("invalid1", "4", "invalid_measurement"),
        _cand("suspect1", "5", "suspect_injection_mask"),
        _cand("inj1", "6", "injection_site"),
        _cand("artefact1", "0", "artifact"),
        _cand("human_cell", "2", "preliminary_rule_pass", manual_label="cell"),
        _cand("bad_z", "", "preliminary_rule_pass"),
    ]
    counts = ce.write_coordinate_exports(
        tmp_path, candidates, channel="green_signal", section=70)

    expected = {
        "all_candidate_coordinates.csv", "preliminary_pass_coordinates.csv",
        "preliminary_fail_coordinates.csv", "manual_review_coordinates.csv",
        "invalid_measurement_coordinates.csv", "suspect_injection_coordinates.csv",
        "confirmed_injection_coordinates.csv", "cellfinder_artifact_coordinates.csv",
        "confirmed_cell_coordinates.csv", "unassigned_peak_plane.csv",
        "plane_assignment_summary.csv", "coordinate_export_summary.csv",
    }
    for name in expected:
        assert (tmp_path / name).is_file(), name
    assert counts["all_candidate_coordinates.csv"] == 9
    assert counts["unassigned_peak_plane.csv"] == 1   # bad_z

    # Confirmed cells: only the human "cell" label -- NOT bare preliminary passes.
    confirmed = list(csv.DictReader(open(tmp_path / "confirmed_cell_coordinates.csv")))
    assert [r["candidate_id"] for r in confirmed] == ["human_cell"]
    assert "pass1" not in {r["candidate_id"] for r in confirmed}

    # Simplified files carry the agreed columns incl. one-based peak_optical_plane.
    rows = list(csv.DictReader(open(tmp_path / "all_candidate_coordinates.csv")))
    assert set(ce.SIMPLE_COLUMNS) == set(rows[0].keys())
    by_id = {r["candidate_id"]: r for r in rows}
    assert by_id["pass1"]["peak_optical_plane"] == "2"   # z-index 1 -> plane 02
    assert by_id["bad_z"]["peak_optical_plane"] == ""    # invalid -> blank, reported


def test_count_summaries_reconcile(tmp_path):
    candidates = [_cand("c1", "1", "preliminary_rule_pass", support="0;1;2"),
                  _cand("c2", "1", "manual_review", support="1"),
                  _cand("c3", "", "preliminary_rule_pass")]
    paths = ce.write_count_summaries(tmp_path, candidates, channel="green_signal", section=70)
    stack = list(csv.DictReader(open(paths["stack_unique_status_summary"])))
    assert sum(int(r["candidate_count"]) for r in stack) == 3   # sums to unique total

    support = list(csv.DictReader(open(paths["plane_support_visualization_summary"])))
    # Support visualisation may exceed the unique total and must warn.
    assert sum(int(r["candidates_visible_on_plane"]) for r in support) >= 3
    assert all("DO NOT SUM" in r["warning"] for r in support)


# 18 ------------------------------------------------------------------------ #
def test_single_section_is_not_a_whole_brain():
    assert rl.is_single_section([70]) is True
    assert rl.is_single_section([70, 71]) is False
