"""Task 1 -- the QC display window is display-only.

Proves that switching channel_2_signal (red) from the old fixed 0-513 window to
robust_tissue_percentile is a *display* change that can never alter candidate
coordinates, statuses, injection-mask membership, measurements or candidate
totals, and that the robust window is no longer saturated by the injection while
the injection core is excluded from the upper limit only. Also locks the config
default so channel_2_signal ships as robust_tissue_percentile.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")

from mouse_brain_pipeline.candidate_detection import SectionDetectionResult  # noqa: E402
from mouse_brain_pipeline.candidate_qc import section_display_info  # noqa: E402
from mouse_brain_pipeline.config import (  # noqa: E402
    Config,
    QcDisplayConfig,
    QcDisplaySettings,
)
from mouse_brain_pipeline.qc_display import apply_display, compute_display_limits  # noqa: E402


def _candidates():
    """Two red candidates carrying coordinates, status, mask membership + measures."""
    return [
        {
            "candidate_id": "channel_2_signal_s070_00001",
            "channel": "channel_2_signal",
            "current_status": "preliminary_rule_pass",
            "x_global_px": 30, "y_global_px": 30, "z_index": 3,
            "inside_injection_core": False,
            "inside_injection_analysis_exclusion": False,
            "xy_area_um2": 120.0, "volume_um3": 320.0, "local_robust_z": 21.0,
            "included_in_count": False,
        },
        {
            "candidate_id": "channel_2_signal_s070_00002",
            "channel": "channel_2_signal",
            "current_status": "preliminary_rule_fail",
            "x_global_px": 50, "y_global_px": 50, "z_index": 3,
            "inside_injection_core": True,
            "inside_injection_analysis_exclusion": True,
            "xy_area_um2": 8.0, "volume_um3": 20.0, "local_robust_z": 3.0,
            "included_in_count": False,
        },
    ]


def _result():
    """A red section result with a bright, saturated injection core."""
    rng = np.random.default_rng(0)
    proj = (200 + rng.normal(0, 4, (100, 100))).astype(np.float32)
    core = np.zeros((100, 100), dtype=bool)
    core[40:60, 40:60] = True
    proj[core] = 6000.0                       # saturated injection core
    tissue = np.ones((100, 100), dtype=bool)
    return SectionDetectionResult(
        channel="channel_2_signal", section=70, candidates=_candidates(),
        projection=proj, tissue_mask=tissue, injection_core_mask=core,
    )


def test_display_setting_never_changes_candidates_or_measurements():
    fixed = QcDisplayConfig(
        channel_2_signal=QcDisplaySettings(mode="fixed", minimum=0, maximum=513))
    robust = QcDisplayConfig(
        channel_2_signal=QcDisplaySettings(
            mode="robust_tissue_percentile", lower_percentile=0.5, upper_percentile=99.7))

    res_fixed = _result()
    res_robust = _result()
    before = copy.deepcopy(res_fixed.candidates)
    proj_before = res_fixed.projection.copy()

    info_fixed = section_display_info(res_fixed, fixed)
    info_robust = section_display_info(res_robust, robust)

    # The setting genuinely changes the *window* (otherwise the test is vacuous).
    assert (info_fixed["display_min"], info_fixed["display_max"]) != \
           (info_robust["display_min"], info_robust["display_max"])

    # ... but coordinates, statuses, mask membership, measurements are untouched,
    for res in (res_fixed, res_robust):
        assert res.candidates == before                       # nothing mutated
        assert len(res.candidates) == len(before)             # totals unchanged
        for got, exp in zip(res.candidates, before):
            assert got["x_global_px"] == exp["x_global_px"]
            assert got["y_global_px"] == exp["y_global_px"]
            assert got["current_status"] == exp["current_status"]
            assert got["inside_injection_core"] == exp["inside_injection_core"]
            assert got["inside_injection_analysis_exclusion"] == \
                exp["inside_injection_analysis_exclusion"]
            assert got["xy_area_um2"] == exp["xy_area_um2"]
            assert got["local_robust_z"] == exp["local_robust_z"]
    # ... and the raw projection array itself is never written to.
    assert np.array_equal(res_fixed.projection, proj_before)


def test_apply_display_returns_a_new_array_and_leaves_raw_untouched():
    proj = (1000 * np.random.default_rng(1).random((40, 40))).astype(np.float32)
    before = proj.copy()
    scaled = apply_display(proj, 0.0, 513.0)
    assert scaled is not proj
    assert np.array_equal(proj, before)
    assert 0.0 <= float(scaled.min()) and float(scaled.max()) <= 1.0


def test_red_robust_window_excludes_saturated_core_from_upper_limit():
    res = _result()
    settings = QcDisplaySettings(
        mode="robust_tissue_percentile", lower_percentile=0.5, upper_percentile=99.7)
    info = compute_display_limits(
        res.projection, settings, tissue_mask=res.tissue_mask,
        injection_core_mask=res.injection_core_mask, exclude_injection_core=True,
    )
    assert info["injection_core_excluded"] is True
    # Upper limit tracks tissue (~200), not the saturated 6000 injection core.
    assert info["display_max"] < 1000
    # Fixed 0-513 vs robust give different windows -> the red QC is no longer
    # clipped to the old saturated range.
    assert info["display_max"] != 513.0


def _load_yaml_config_text(path: Path) -> dict:
    import yaml

    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() not in ("```yaml", "```")]
    return yaml.safe_load("\n".join(lines))


@pytest.mark.parametrize("name", ["config.example.yml", "config_full_image.yml"])
def test_shipped_config_defaults_red_to_robust_percentile(name):
    data = _load_yaml_config_text(ROOT / name)
    cfg = Config.from_dict(data).qc_display
    red = cfg.for_channel("channel_2_signal")
    assert red.mode == "robust_tissue_percentile"
    assert red.lower_percentile == pytest.approx(0.5)
    assert red.upper_percentile == pytest.approx(99.7)
    # Green is unchanged and still robust; the two channels stay independent.
    assert cfg.for_channel("green_signal").mode == "robust_tissue_percentile"
