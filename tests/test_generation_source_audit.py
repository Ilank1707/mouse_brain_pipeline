"""Tests for the read-only candidate-generation source audit (Task 4)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mouse_brain_pipeline import generation_source_audit as gsa  # noqa: E402
import audit_candidate_generation_sources as audit_script  # noqa: E402


def cand(channel, source, *, in_mask=False, in_core=False,
         status="preliminary_rule_pass", peak=3, **extra):
    row = {
        "candidate_id": f"{channel}_{extra.get('i', 0)}",
        "channel": channel,
        "candidate_generation_source": source,
        "inside_injection_analysis_exclusion": str(in_mask),
        "inside_injection_core": str(in_core),
        "current_status": status,
        "fixed_xy_peak_z_index": str(peak),
    }
    row.update({k: v for k, v in extra.items() if k != "i"})
    return row


# --------------------------------------------------------------------------- #
# normalize_source
# --------------------------------------------------------------------------- #
def test_normalize_source_prefers_column_then_falls_back_to_flags():
    assert gsa.normalize_source({"candidate_generation_source": "both"}) == "both"
    assert gsa.normalize_source(
        {"candidate_generation_source": "", "detected_on_raw_stack": "True",
         "detected_on_injection_suppressed_stack": "True"}) == "both"
    assert gsa.normalize_source(
        {"detected_on_injection_suppressed_stack": "True"}) == "injection_suppressed_stack"
    assert gsa.normalize_source({"detected_on_raw_stack": "True"}) == "raw_stack"


# --------------------------------------------------------------------------- #
# audit_rows
# --------------------------------------------------------------------------- #
def test_audit_rows_counts_and_reconcile_to_total():
    cands = [cand("green_signal", "raw_stack", i=i) for i in range(3)]
    cands += [cand("green_signal", "both", in_mask=True, in_core=True, i=99)]
    rows = gsa.audit_rows(cands)
    assert list(rows[0].keys()) == gsa.AUDIT_COLUMNS
    assert sum(r["count"] for r in rows) == len(cands)   # every candidate counted once
    # The raw_stack, outside-mask, plane-04 (0-based 3 -> plane 04) bucket has 3.
    raw = [r for r in rows if r["candidate_generation_source"] == "raw_stack"][0]
    assert raw["count"] == 3
    assert raw["peak_optical_plane"] == "04"


# --------------------------------------------------------------------------- #
# source_fractions + outside mask
# --------------------------------------------------------------------------- #
def test_source_fractions_over_outside_mask_only():
    cands = [cand("green_signal", "both", in_mask=True)]          # inside -> excluded
    cands += [cand("green_signal", "raw_stack") for _ in range(4)]   # outside
    cands += [cand("green_signal", "both")]                          # outside
    outside = gsa.outside_mask_candidates(cands)
    assert len(outside) == 5
    fr = gsa.source_fractions(outside)
    assert fr["both"] == 1 and fr["raw_stack_only"] == 4
    assert fr["fraction_both"] == pytest.approx(1 / 5)


# --------------------------------------------------------------------------- #
# suppression-sensitivity warning
# --------------------------------------------------------------------------- #
def test_warning_triggers_below_threshold_only():
    low = {"n": 20, "fraction_both": 0.05}
    triggered, message = gsa.suppression_sensitivity_warning("green_signal", low)
    assert triggered
    assert "manual" in message.lower() and "threshold" in message.lower()

    high = {"n": 20, "fraction_both": 0.5}
    assert gsa.suppression_sensitivity_warning("green_signal", high)[0] is False
    # No outside-mask candidates -> never a warning (nothing to validate).
    assert gsa.suppression_sensitivity_warning(
        "green_signal", {"n": 0, "fraction_both": 0.0})[0] is False


def test_summarize_keeps_channels_separate_and_flags_only_sensitive_channel():
    # green: 0% outside-'both' -> warning; red: 80% -> no warning.
    green = [cand("green_signal", "raw_stack") for _ in range(10)]
    red = ([cand("channel_2_signal", "both") for _ in range(8)]
           + [cand("channel_2_signal", "raw_stack") for _ in range(2)])
    summary = gsa.summarize(green + red)
    assert set(summary["channels"]) == {"green_signal", "channel_2_signal"}
    warned = {w["channel"] for w in summary["suppression_sensitivity"]["warnings"]}
    assert warned == {"green_signal"}
    assert summary["by_channel"]["channel_2_signal"][
        "outside_analysis_mask_source_fractions"]["fraction_both"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Script main() end to end (read-only; writes CSV + JSON)
# --------------------------------------------------------------------------- #
def test_audit_script_writes_outputs_and_warns(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cands = [cand("green_signal", "raw_stack", i=i) for i in range(10)]     # 0% both
    cands += [cand("channel_2_signal", "both", i=i) for i in range(9)]      # 100% both
    cands += [cand("channel_2_signal", "raw_stack", i=99)]
    with (run_dir / "all_candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cands[0].keys()))
        writer.writeheader(); writer.writerows(cands)

    before = (run_dir / "all_candidates.csv").read_bytes()
    out_dir = tmp_path / "audit"
    rc = audit_script.main(["--run-dir", str(run_dir), "--out-dir", str(out_dir),
                            "--no-plot"])
    assert rc == 0
    assert (out_dir / "candidate_generation_source_audit.csv").is_file()
    summary = json.loads(
        (out_dir / "candidate_generation_source_summary.json").read_text(encoding="utf-8"))
    warned = {w["channel"] for w in summary["suppression_sensitivity"]["warnings"]}
    assert warned == {"green_signal"}
    # The audit never touches the input candidates file.
    assert (run_dir / "all_candidates.csv").read_bytes() == before
