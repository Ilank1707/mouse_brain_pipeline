"""Tests for config hardening, native-resolution QC and Cellfinder z-mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
yaml = pytest.importorskip("yaml")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from mouse_brain_pipeline.candidate_detection import (  # noqa: E402
    DetectionParams,
    SectionDetectionResult,
    detect_candidates_in_stack,
)
from mouse_brain_pipeline.candidate_qc import write_native_qc  # noqa: E402
from mouse_brain_pipeline.cellfinder_adapter import map_cellfinder_z  # noqa: E402
from mouse_brain_pipeline.config import (  # noqa: E402
    QcDisplayConfig,
    QcDisplaySettings,
    load_config,
    schema_drift_warnings,
    unknown_config_keys,
)
from mouse_brain_pipeline.qc_native import (  # noqa: E402
    apply_window_uint8,
    native_max_projection,
    save_native_projection_tiff,
    save_png_fullres,
    save_preview_png,
)

VOXEL = (6.0, 1.0, 1.0)


class FakeCell:
    def __init__(self, x=40, y=40, z=3, type=1):
        self.x, self.y, self.z, self.type = x, y, z, type


def _params():
    from mouse_brain_pipeline.config import InjectionExclusionConfig, TissueMaskConfig

    p = DetectionParams(backend="cellfinder_candidates")
    p.tissue = TissueMaskConfig(enabled=False)
    p.injection = InjectionExclusionConfig(enabled=False)
    p.exclude_crop_boundary = False
    return p


def _detect(cells, injection=None):
    from mouse_brain_pipeline.config import InjectionExclusionConfig

    rng = np.random.default_rng(1)
    stack = (100 + rng.normal(0, 2, (7, 81, 81))).astype(np.float32)
    return detect_candidates_in_stack(
        stack, _params(), VOXEL, channel="channel_2_signal", section=70,
        first_section=70, plane_numbers=list(range(1, 8)),
        injection_cfg=injection or InjectionExclusionConfig(enabled=False),
        backend="cellfinder_candidates",
        cellfinder_detect_main=lambda **_kw: cells,
    )


MINIMAL_CONFIG = {
    "data": {"green_signal_dir": "g", "channel_2_signal_dir": "c", "work_dir": "W"},
    "detection": {
        "backend": "cellfinder_candidates",
        "injection_exclusion": {"generation_suppression_enabled": True},
    },
    "qc_display": {"channel_2_signal": {"mode": "fixed", "minimum": 0, "maximum": 513}},
}


def _write_config(tmp_path, data) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Config hardening
# --------------------------------------------------------------------------- #
def test_1_config_path_and_effective_values_recorded(tmp_path):
    path = _write_config(tmp_path, MINIMAL_CONFIG)
    cfg = load_config(path)
    assert cfg.source_path == str(path)
    settings = cfg.qc_display.for_channel("channel_2_signal")
    assert (settings.mode, settings.minimum, settings.maximum) == ("fixed", 0, 513)
    assert cfg.detection.injection_exclusion.generation_suppression_enabled is True
    assert cfg.config_warnings == []


def test_2_unknown_config_fields_are_not_silently_ignored(tmp_path):
    data = {**MINIMAL_CONFIG}
    data["detection"] = {**MINIMAL_CONFIG["detection"], "bogus_field": 7}
    assert "detection.bogus_field" in unknown_config_keys(data)
    cfg = load_config(_write_config(tmp_path, data))
    assert any("detection.bogus_field" in w for w in cfg.config_warnings)


def test_3_stale_copied_config_warns_about_missing_fields(tmp_path):
    stale = {
        "data": {"green_signal_dir": "g", "channel_2_signal_dir": "c", "work_dir": "W"},
        "detection": {"backend": "cellfinder_candidates", "injection_exclusion": {}},
    }
    warnings = schema_drift_warnings(stale)
    assert any("qc_display" in w for w in warnings)
    assert any("generation_suppression_enabled" in w for w in warnings)
    cfg = load_config(_write_config(tmp_path, stale))
    assert len(cfg.config_warnings) >= 2


def test_4_work_dir_override_works_without_copying_config(tmp_path):
    cfg = load_config(_write_config(tmp_path, MINIMAL_CONFIG))
    cfg.data.work_dir = str(tmp_path / "full_image_out")   # what --work-dir does
    assert cfg.work_dir == Path(tmp_path / "full_image_out")


# --------------------------------------------------------------------------- #
# Native + full-resolution QC
# --------------------------------------------------------------------------- #
def test_12_native_projection_preserves_source_resolution():
    rng = np.random.default_rng(2)
    stack = rng.integers(0, 5000, (7, 40, 50)).astype(np.uint16)
    native, source_dtype, method, upcast = native_max_projection(stack)
    assert native.shape == (40, 50)
    assert native.dtype == np.uint16
    assert np.array_equal(native, stack.max(axis=0))
    assert upcast is False and "max_intensity_projection" in method


def test_13_native_image_uses_lossless_tiff(tmp_path):
    native = np.arange(40 * 50, dtype=np.uint16).reshape(40, 50)
    path = tmp_path / "01_raw_projection_native_16bit.tif"
    save_native_projection_tiff(path, native, channel="c", section=70,
                                projection_method="max", source_dtype="uint16")
    assert path.suffix == ".tif"
    assert np.array_equal(tifffile.imread(path), native)   # exact round-trip


def test_14_fullres_png_dimensions_equal_source(tmp_path):
    proj = (np.arange(60 * 70, dtype=np.float32).reshape(60, 70) % 600)
    display8 = apply_window_uint8(proj, 0, 513)
    path = tmp_path / "02_raw_projection_display_fullres.png"
    save_png_fullres(path, display8)
    back = np.array(Image.open(path))
    assert back.shape == proj.shape


def test_15_preview_is_smaller_and_labelled(tmp_path):
    image = np.zeros((10, 2400), dtype=np.uint8)
    path = tmp_path / "07_candidate_interpretation_audit_preview.png"
    _p, width, height = save_preview_png(path, image, max_dim=2000)
    assert "preview" in path.name
    assert width < 2400


def test_16_rendered_background_is_not_interpolated(tmp_path):
    proj = (np.arange(60 * 70, dtype=np.float32).reshape(60, 70) % 600)
    display8 = apply_window_uint8(proj, 0, 513)
    path = tmp_path / "02_raw_projection_display_fullres.png"
    save_png_fullres(path, display8)
    back = np.array(Image.open(path))
    assert np.array_equal(back, display8)   # no resampling/blur applied


def test_11_and_17_fixed_window_only_affects_display_and_metadata(tmp_path):
    proj = (np.arange(80 * 90, dtype=np.float32).reshape(80, 90) % 600)
    before = proj.copy()
    res = SectionDetectionResult(channel="channel_2_signal", section=70,
                                 candidates=[], projection=proj)
    cfg = QcDisplayConfig(
        channel_2_signal=QcDisplaySettings(mode="fixed", minimum=0, maximum=513)
    )
    rows = write_native_qc(tmp_path, res, qc_display_cfg=cfg)
    assert np.array_equal(proj, before)   # the projection array is untouched
    fullres = [r for r in rows if r["filename"].endswith("display_fullres.png")][0]
    assert fullres["source_width"] == 90 and fullres["source_height"] == 80
    assert fullres["saved_width"] == 90 and fullres["saved_height"] == 80
    assert fullres["resizing_occurred"] is False
    assert (fullres["display_min"], fullres["display_max"]) == (0.0, 513.0)
    native = [r for r in rows if r["filename"].endswith(".tif")][0]
    assert native["file_format"] == "tiff" and native["resizing_occurred"] is False


# --------------------------------------------------------------------------- #
# Cellfinder z-mapping
# --------------------------------------------------------------------------- #
def test_18_invalid_cellfinder_z_is_not_silently_clamped():
    mapped, method, valid = map_cellfinder_z(9, 7)
    assert (method, valid) == ("out_of_range_unmapped", False)
    result = _detect([FakeCell(40, 40, 9)])
    candidate = result.candidates[0]
    assert str(candidate["cellfinder_returned_z_raw"]) == "9"
    assert candidate["original_cellfinder_z_valid"] is False
    assert candidate["current_status"] == "manual_review"
    assert candidate["rejection_reason"] == "cellfinder_z_out_of_range"
    assert candidate["included_in_count"] is False


def test_19_padded_coordinate_conversion():
    mapped, method, valid = map_cellfinder_z(8, 7, padding_offset=2)
    assert (mapped, method, valid) == (6, "padding_offset_corrected", True)


def test_20_status_counts_reconcile_with_z_invalid_candidate():
    result = _detect([FakeCell(40, 40, 3), FakeCell(20, 20, 9)])
    counts: dict = {}
    for c in result.candidates:
        counts[c["current_status"]] = counts.get(c["current_status"], 0) + 1
    assert sum(counts.values()) == len(result.candidates)


def test_21_preliminary_status_cannot_set_included_in_count():
    result = _detect([FakeCell(40, 40, 3)])
    assert all(c["included_in_count"] is False for c in result.candidates)


# --------------------------------------------------------------------------- #
# Full section vs full brain
# --------------------------------------------------------------------------- #
def test_22_one_section_is_never_called_a_full_brain():
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "run_candidate_pilot.py")
    text = runner.read_text(encoding="utf-8")
    assert "This run contains one section and is not a whole-brain count." in text
    assert "full XY section" in text
    assert "full brain" not in text.lower()
