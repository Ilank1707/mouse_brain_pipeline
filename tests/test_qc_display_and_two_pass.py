"""Tests for Part 1 (QC display) and Part 2/3 (two-pass generation + recall).

These are array-only / CSV-only tests: no real TIFFs, no Cellfinder install. The
Cellfinder stage is stubbed with an injected ``detect_main`` so the two-pass
provenance and merge logic can be exercised deterministically.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from mouse_brain_pipeline.candidate_detection import (  # noqa: E402
    SectionDetectionResult,
    DetectionParams,
    _merge_two_pass_objects,
    _tag_object_source,
    detect_candidates_in_stack,
    write_candidate_tables,
)
from mouse_brain_pipeline.candidate_qc import write_qc_display_metadata  # noqa: E402
from mouse_brain_pipeline.classifier import classifier_state  # noqa: E402
from mouse_brain_pipeline.config import (  # noqa: E402
    CellfinderConfig,
    Config,
    InjectionExclusionConfig,
    QcDisplayConfig,
    QcDisplaySettings,
    TissueMaskConfig,
)
from mouse_brain_pipeline.injection_suppression import suppress_injection_core  # noqa: E402
from mouse_brain_pipeline.qc_display import (  # noqa: E402
    apply_display,
    compute_display_limits,
)
from mouse_brain_pipeline.reference_audit import evaluate_recall_by_source  # noqa: E402

VOXEL = (6.0, 1.0, 1.0)


class FakeCell:
    def __init__(self, x=50, y=50, z=3, type=1):
        self.x, self.y, self.z, self.type = x, y, z, type


def two_pass_params():
    p = DetectionParams(backend="cellfinder_candidates")
    p.tissue = TissueMaskConfig(enabled=False)
    p.exclude_crop_boundary = False
    p.merge_distance_xy_um = 8.0
    p.merge_distance_z_um = 12.0
    return p


def two_pass_injection_cfg():
    # Manual rectangle injection so the conservative core is well defined; the
    # generation-suppression pass is enabled.
    return InjectionExclusionConfig(
        enabled=True, automatic=False,
        manual_rectangles=[[40, 60, 40, 60]],
        core_dilation_um=2.0, analysis_exclusion_dilation_um=8.0,
        generation_suppression_enabled=True,
    )


def make_two_pass_stack():
    rng = np.random.default_rng(7)
    stack = (100 + rng.normal(0, 2, (7, 100, 100))).astype(np.float32)
    stack[:, 40:61, 40:61] += 4000.0  # bright injection block inside the core
    return stack


# --------------------------------------------------------------------------- #
# Part 1 -- QC display
# --------------------------------------------------------------------------- #
def test_1_fixed_display_limits_do_not_alter_raw_array():
    proj = (1000 * np.random.default_rng(1).random((50, 50))).astype(np.float32)
    before = proj.copy()
    settings = QcDisplaySettings(mode="fixed", minimum=0, maximum=513)
    info = compute_display_limits(proj, settings)
    scaled = apply_display(proj, info["display_min"], info["display_max"])
    assert info["display_min"] == 0 and info["display_max"] == 513
    assert np.array_equal(proj, before)            # raw projection untouched
    assert scaled is not proj and scaled.max() <= 1.0


def test_2_channel_2_fixed_display_range_is_0_513():
    cfg = Config.from_dict({
        "qc_display": {
            "channel_2_signal": {"mode": "fixed", "minimum": 0, "maximum": 513},
        }
    }).qc_display
    settings = cfg.for_channel("channel_2_signal")
    assert settings.mode == "fixed"
    proj = np.full((40, 40), 12345.0, dtype=np.float32)
    info = compute_display_limits(proj, settings)
    assert (info["display_min"], info["display_max"]) == (0.0, 513.0)


def test_3_robust_limits_exclude_black_background_and_padding():
    rng = np.random.default_rng(3)
    proj = np.zeros((100, 100), dtype=np.float32)        # black background = 0
    proj[20:80, 20:80] = 200 + rng.normal(0, 4, (60, 60))
    proj[20:25, 20:25] = 9999                            # padding sentinel
    tissue = np.zeros((100, 100), dtype=bool)
    tissue[20:80, 20:80] = True
    settings = QcDisplaySettings(
        mode="robust_tissue_percentile", lower_percentile=0.5, upper_percentile=99.7
    )
    info = compute_display_limits(
        proj, settings, tissue_mask=tissue, padding_values=(0.0, 9999.0)
    )
    assert info["display_min"] > 100      # not pulled to 0 by the background
    assert info["display_max"] < 1000     # padding 9999 excluded from the window


def test_4_robust_limits_optionally_exclude_injection_core():
    rng = np.random.default_rng(4)
    proj = 200 + rng.normal(0, 4, (100, 100)).astype(np.float32)
    core = np.zeros((100, 100), dtype=bool)
    core[40:60, 40:60] = True
    proj[core] = 6000                     # saturated injection core
    tissue = np.ones((100, 100), dtype=bool)
    settings = QcDisplaySettings(
        mode="robust_tissue_percentile", lower_percentile=0.5, upper_percentile=99.7
    )
    excluded = compute_display_limits(
        proj, settings, tissue_mask=tissue, injection_core_mask=core,
        exclude_injection_core=True,
    )
    included = compute_display_limits(
        proj, settings, tissue_mask=tissue, injection_core_mask=core,
        exclude_injection_core=False,
    )
    assert excluded["injection_core_excluded"] is True
    assert excluded["display_max"] < included["display_max"]
    assert excluded["display_max"] < 1000


def test_5_saved_qc_metadata_records_actual_display_limits(tmp_path):
    proj = np.full((40, 40), 300.0, dtype=np.float32)
    res = SectionDetectionResult(channel="channel_2_signal", section=70, projection=proj)
    cfg = QcDisplayConfig(
        channel_2_signal=QcDisplaySettings(mode="fixed", minimum=0, maximum=513)
    )
    path = write_qc_display_metadata(tmp_path, [res], qc_display_cfg=cfg)
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["display_mode"] == "fixed"
    assert float(rows[0]["display_min"]) == 0.0
    assert float(rows[0]["display_max"]) == 513.0


# --------------------------------------------------------------------------- #
# Part 2 -- injection suppression + two-pass generation
# --------------------------------------------------------------------------- #
def test_6_injection_suppression_operates_on_in_memory_copy_only():
    rng = np.random.default_rng(6)
    stack = (100 + rng.normal(0, 2, (3, 60, 60))).astype(np.float32)
    stack[:, 25:35, 25:35] += 5000
    before = stack.copy()
    core = np.zeros((60, 60), dtype=bool)
    core[25:35, 25:35] = True
    out = suppress_injection_core(stack, core, VOXEL)
    assert out is not stack
    assert np.array_equal(stack, before)            # input array unchanged
    assert out.shape == stack.shape and out.dtype == np.float32


def test_8_suppression_fill_has_no_hard_zero_border():
    rng = np.random.default_rng(8)
    stack = (100 + rng.normal(0, 2, (3, 60, 60))).astype(np.float32)
    stack[:, 25:35, 25:35] += 5000
    core = np.zeros((60, 60), dtype=bool)
    core[25:35, 25:35] = True
    out = suppress_injection_core(stack, core, VOXEL)
    core3d = np.broadcast_to(core, out.shape)
    assert out[core3d].min() > 10               # core is not zero-filled
    # The filled core should sit near the surrounding ~100 tissue, not at 0.
    assert abs(float(out[core3d].mean()) - 100.0) < 30.0


def test_9_cellfinder_receives_zyx_array_for_both_passes():
    captured = []

    def stub(**kwargs):
        captured.append(kwargs["signal_array"])
        return [FakeCell(50, 50, 3)] if len(captured) == 1 else [FakeCell(10, 10, 3)]

    stack = make_two_pass_stack()
    detect_candidates_in_stack(
        stack, two_pass_params(), VOXEL, channel="channel_2_signal", section=70,
        first_section=70, plane_numbers=list(range(1, 8)),
        injection_cfg=two_pass_injection_cfg(), backend="cellfinder_candidates",
        cellfinder_detect_main=stub,
    )
    assert len(captured) == 2                    # raw pass + suppressed pass
    for arr in captured:
        assert arr.ndim == 3 and arr.shape[0] == 7
        assert arr.dtype == np.uint16
    assert not np.array_equal(captured[0], captured[1])  # suppressed differs


def _run_two_pass():
    def stub(**kwargs):
        # Raw pass (call 1) seeds inside the core; suppressed pass (call 2) seeds
        # one duplicate inside + one new candidate outside the analysis mask.
        if not getattr(stub, "called", False):
            stub.called = True
            return [FakeCell(50, 50, 3)]
        return [FakeCell(50, 50, 3), FakeCell(10, 10, 3)]

    stack = make_two_pass_stack()
    before = stack.copy()
    result = detect_candidates_in_stack(
        stack, two_pass_params(), VOXEL, channel="channel_2_signal", section=70,
        first_section=70, plane_numbers=list(range(1, 8)),
        injection_cfg=two_pass_injection_cfg(), backend="cellfinder_candidates",
        cellfinder_detect_main=stub,
    )
    return result, stack, before


def test_7_raw_input_array_unchanged_during_two_pass():
    _result, stack, before = _run_two_pass()
    assert np.array_equal(stack, before)


def test_10_and_11_raw_and_suppressed_outside_candidates_are_retained(tmp_path):
    result, _stack, _before = _run_two_pass()
    raw = [c for c in result.candidates if c["candidate_generation_source"] == "raw_stack"]
    suppressed_only = [
        c for c in result.candidates
        if c["candidate_generation_source"] == "injection_suppressed_stack"
    ]
    assert raw, "raw-stack candidate must be preserved"
    assert suppressed_only, "suppressed outside-mask candidate must be retained"
    assert all(
        not c["inside_injection_analysis_exclusion"] for c in suppressed_only
    )
    # Raw candidate survives into the complete audit table.
    section_result = SectionDetectionResult(
        channel="channel_2_signal", section=70, candidates=result.candidates
    )
    paths = write_candidate_tables(tmp_path, [section_result])
    text = paths["all"].read_text(encoding="utf-8")
    assert "raw_stack" in text
    assert "injection_suppressed_stack" in text


def test_12_duplicate_candidates_from_both_passes_merge_in_3d():
    def raw_obj(z, y, x):
        o = {"z_index": z, "y_local_px": y, "x_local_px": x}
        return _tag_object_source(o, on_raw=True, on_suppressed=False,
                                  mask_used=True, mask_source="injection_core")

    def supp_obj(z, y, x):
        o = {"z_index": z, "y_local_px": y, "x_local_px": x}
        return _tag_object_source(o, on_raw=False, on_suppressed=True,
                                  mask_used=True, mask_source="injection_core")

    raw_objects = [raw_obj(3, 50, 50)]
    suppressed = [
        supp_obj(3, 51, 51),   # within XY+Z tolerance -> merges to "both"
        supp_obj(3, 50, 90),   # far in XY, shares the plane -> NOT merged
        supp_obj(0, 50, 50),   # same XY, far in Z -> NOT merged
    ]
    merged = _merge_two_pass_objects(
        raw_objects, suppressed, VOXEL, merge_xy_um=8.0, merge_z_um=12.0
    )
    assert len(merged) == 3                       # 1 merged-both + 2 appended
    both = [o for o in merged
            if o["candidate_generation_source"] == "both"]
    assert len(both) == 1
    assert both[0]["detected_on_raw_stack"] and both[0]["detected_on_injection_suppressed_stack"]


def test_13_candidate_generation_provenance_written_to_csv(tmp_path):
    result, _stack, _before = _run_two_pass()
    section_result = SectionDetectionResult(
        channel="channel_2_signal", section=70, candidates=result.candidates
    )
    paths = write_candidate_tables(tmp_path, [section_result])
    with open(paths["all"], newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)
    for column in (
        "candidate_generation_source", "detected_on_raw_stack",
        "detected_on_injection_suppressed_stack", "generation_suppression_mask_used",
        "generation_suppression_mask_source",
    ):
        assert column in header
    assert any(r["candidate_generation_source"] == "raw_stack" for r in rows)


def test_14_suppressed_only_candidate_is_not_automatically_a_cell():
    result, _stack, _before = _run_two_pass()
    suppressed_only = [
        c for c in result.candidates
        if c["candidate_generation_source"] == "injection_suppressed_stack"
    ]
    assert suppressed_only
    assert all(c["included_in_count"] is False for c in suppressed_only)


def test_15_automatic_mask_flag_is_distinct_from_human_injection_label():
    # A suspect AUTOMATIC mask never confirms injection on its own ...
    status, included = classifier_state(
        {"measurement_valid": "True", "current_status": "suspect_injection_mask"}, 0.99,
    )
    assert status == "manual_review" and included is False
    # ... only a human label does.
    status, included = classifier_state(
        {"measurement_valid": "True", "current_status": "suspect_injection_mask"}, 0.99,
        manual_label="injection",
    )
    assert status == "injection_site" and included is False


def test_16_channel_specific_cellfinder_settings_are_applied():
    cfg = CellfinderConfig.from_dict({
        "n_sds_above_mean_thresh": 10,
        "channel_2_signal": {"n_sds_above_mean_thresh": 99},
    })
    assert cfg.for_channel("green_signal").n_sds_above_mean_thresh == 10
    assert cfg.for_channel("channel_2_signal").n_sds_above_mean_thresh == 99

    captured = {}

    def stub(**kwargs):
        captured.update(kwargs)
        return [FakeCell(50, 50, 3)]

    params = two_pass_params()
    detect_candidates_in_stack(
        make_two_pass_stack(), params, VOXEL, channel="channel_2_signal", section=70,
        first_section=70, plane_numbers=list(range(1, 8)),
        injection_cfg=InjectionExclusionConfig(enabled=False),
        backend="cellfinder_candidates", cellfinder_detect_main=stub,
        cellfinder_cfg=cfg.for_channel("channel_2_signal"),
    )
    assert captured["n_sds_above_mean_thresh"] == 99


def test_17_two_pass_status_counts_reconcile_exactly():
    result, _stack, _before = _run_two_pass()
    counts: dict = {}
    for c in result.candidates:
        counts[c["current_status"]] = counts.get(c["current_status"], 0) + 1
    assert sum(counts.values()) == len(result.candidates)


# --------------------------------------------------------------------------- #
# Part 3 -- recall validation
# --------------------------------------------------------------------------- #
def _candidate(cid, x, y, *, raw=True, suppressed=False):
    return {
        "candidate_id": cid, "channel": "green_signal", "section": "70",
        "x_global_px": str(x), "y_global_px": str(y), "cellfinder_z_index": "3",
        "detected_on_raw_stack": str(raw),
        "detected_on_injection_suppressed_stack": str(suppressed),
    }


def test_18_recall_withheld_when_no_manual_references():
    summary = evaluate_recall_by_source(
        [], [_candidate("c1", 100, 100)],
        voxel_size_y_um=1, voxel_size_x_um=1, voxel_size_z_um=6,
        xy_tolerance_um=8, z_tolerance_um=12,
    )
    assert summary["has_references"] is False
    assert "by_source" not in summary


def test_19_raw_suppressed_and_union_recall_reported_separately():
    references = [
        {"reference_id": "r1", "channel": "green_signal", "section": "70",
         "x_global_px": "100", "y_global_px": "100", "z_index": "3"},
        {"reference_id": "r2", "channel": "green_signal", "section": "70",
         "x_global_px": "300", "y_global_px": "300", "z_index": "3"},
    ]
    candidates = [
        _candidate("c1", 100, 100, raw=True, suppressed=False),    # matches r1 (raw)
        _candidate("c2", 300, 300, raw=False, suppressed=True),    # matches r2 (suppressed)
    ]
    summary = evaluate_recall_by_source(
        references, candidates,
        voxel_size_y_um=1, voxel_size_x_um=1, voxel_size_z_um=6,
        xy_tolerance_um=8, z_tolerance_um=12,
    )
    assert summary["has_references"] is True
    by_source = summary["by_source"]
    assert set(by_source) == {"raw_pass", "suppressed_pass", "union"}
    assert by_source["raw_pass"]["recall"] == 0.5         # only r1
    assert by_source["suppressed_pass"]["recall"] == 0.5  # only r2
    assert by_source["union"]["recall"] == 1.0            # both
