"""Regression test for the candidate QC writer.

``write_channel_qc`` builds the per-channel QC figures and references numpy as
``np`` (e.g. when passing it to ``_scatter_local`` and downsampling the
projection). A missing local ``import numpy as np`` made it crash with
``NameError: name 'np' is not defined`` while writing
``04_candidates_before_injection_exclusion.png``. This test drives the writer on
a tiny synthetic section and fails if that import regresses.

It uses only display/rendering; no scientific threshold, mask, Cellfinder logic
or candidate status is exercised or changed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from mouse_brain_pipeline.candidate_detection import (  # noqa: E402
    STATUS_MANUAL_REVIEW,
    STATUS_RULE_PASSED,
    SectionDetectionResult,
)
from mouse_brain_pipeline.candidate_qc import write_channel_qc  # noqa: E402


def _candidate(cid, x, y, status=STATUS_RULE_PASSED):
    return {
        "candidate_id": cid,
        "current_status": status,
        "x_global_px": x,
        "y_global_px": y,
        "inside_injection_analysis_exclusion": False,
        "detected_on_raw_stack": True,
        "detected_on_injection_suppressed_stack": False,
    }


def _result():
    # Non-square projection so a width/height swap would show up.
    rng = np.random.default_rng(0)
    projection = rng.integers(50, 400, size=(48, 72)).astype(np.uint16)
    candidates = [
        _candidate("c1", 20, 15),
        _candidate("c2", 60, 40),
        _candidate("c3", 30, 30, status=STATUS_MANUAL_REVIEW),
    ]
    return SectionDetectionResult(
        channel="green_signal", section=70, candidates=candidates, projection=projection,
    )


def test_write_channel_qc_does_not_crash_on_numpy_use(tmp_path):
    # Before the fix this raised NameError: name 'np' is not defined at figure 04.
    section_dir = write_channel_qc(tmp_path, _result(), qc_display_cfg=None)

    # The figure whose scatter call referenced np must now exist.
    assert (Path(section_dir) / "04_candidates_before_injection_exclusion.png").is_file()
    # And the projection-comparison figure (also uses np.asarray) is written.
    assert (Path(section_dir) / "10_display_scaling_comparison.png").is_file()


def test_write_channel_qc_handles_empty_candidates(tmp_path):
    # np is still evaluated as an argument even with no candidates -> must not crash.
    res = _result()
    res.candidates = []
    section_dir = write_channel_qc(tmp_path, res, qc_display_cfg=None)
    assert (Path(section_dir) / "04_candidates_before_injection_exclusion.png").is_file()
