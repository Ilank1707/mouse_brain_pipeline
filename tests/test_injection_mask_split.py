"""Focused tests for the seeded injection-mask watershed split (Part A)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from mouse_brain_pipeline import candidate_detection as cd  # noqa: E402
from mouse_brain_pipeline.config import InjectionExclusionConfig  # noqa: E402

VOXEL = (6.0, 1.0, 1.0)


def _dumbbell(height=120, width=240, radius=45, neck_half=12):
    """One connected component: two disks joined by a neck (a merged region)."""
    yy, xx = np.ogrid[:height, :width]
    left = (yy - height // 2) ** 2 + (xx - 60) ** 2 <= radius ** 2
    right = (yy - height // 2) ** 2 + (xx - (width - 60)) ** 2 <= radius ** 2
    neck = (np.abs(yy - height // 2) <= neck_half) & (xx >= 60) & (xx <= width - 60)
    return left | right | neck


def _cfg(**kwargs):
    defaults = dict(
        split_merged_components=True,
        split_min_peak_distance_um=40.0,
        split_min_subcomponent_area_um2=50.0,  # both lobes are far above this
    )
    defaults.update(kwargs)
    return InjectionExclusionConfig(**defaults)


def test_merged_region_splits_into_seeded_and_non_seeded_subcomponents():
    from scipy import ndimage as ndi

    mask = _dumbbell()
    assert ndi.label(mask)[1] == 1  # genuinely ONE connected component

    # Seed in the LEFT lobe only.
    kept, diag = cd._split_and_filter_by_seeds(mask, [(60, 60)], (1.0, 1.0), _cfg(), 1)

    assert diag["split_applied"] is True
    assert diag["n_components"] == 1  # one pre-split component
    assert diag["n_subcomponents"] == 2  # split into two lobes
    contains_seed = [s["contains_seed"] for s in diag["post_split_subcomponents"]]
    assert sorted(contains_seed) == [False, True]


def test_non_seeded_lobe_touching_seeded_lobe_is_removed():
    mask = _dumbbell()
    kept, diag = cd._split_and_filter_by_seeds(mask, [(60, 60)], (1.0, 1.0), _cfg(), 1)

    # Seeded left lobe kept; touching non-seeded right lobe removed.
    assert bool(kept[60, 60]) is True
    assert bool(kept[60, 180]) is False
    assert diag["n_kept"] == 1
    assert diag["n_removed"] == 1
    removed = [s for s in diag["post_split_subcomponents"] if not s["kept"]]
    assert removed and removed[0]["reason"].startswith("non_seeded")


def test_seed_matching_keeps_only_correct_subcomponent():
    mask = _dumbbell()
    kept, diag = cd._split_and_filter_by_seeds(mask, [(60, 60)], (1.0, 1.0), _cfg(), 1)

    matches = diag["seed_matches"]
    assert len(matches) == 1
    match = matches[0]
    assert match["kept"] is True
    kept_labels = set(diag["kept_subcomponent_labels"])
    assert kept_labels == {match["subcomponent_label"]}
    # Exactly the seeded subcomponent is kept.
    seeded_records = [s for s in diag["post_split_subcomponents"] if s["contains_seed"]]
    assert {s["subcomponent_label"] for s in seeded_records} == kept_labels


def test_full_build_path_splits_single_bright_region_and_drops_non_seeded_lobe():
    # A solid bright vertical dumbbell that forms ONE bright component; only the
    # TOP lobe is seeded.
    height, width = 220, 130
    stack = (100.0 + np.random.default_rng(7).normal(0, 1, size=(7, height, width))).astype(
        np.float32
    )
    yy, xx = np.ogrid[:height, :width]
    top = (yy - 60) ** 2 + (xx - 65) ** 2 <= 38 ** 2
    bottom = (yy - 160) ** 2 + (xx - 65) ** 2 <= 38 ** 2
    neck = (xx - 65) ** 2 <= 22 ** 2
    neck = neck & (yy >= 60) & (yy <= 160)
    region = top | bottom | neck
    stack[:, region] += 3000.0

    inj = InjectionExclusionConfig(
        enabled=True, automatic=True, downsample_um=1.0, smoothing_sigma_um=6.0,
        intensity_percentile=88.0, minimum_area_um2=500.0,
        core_dilation_um=0.0, analysis_exclusion_dilation_um=0.0,
        split_merged_components=True, split_min_peak_distance_um=50.0,
        split_min_subcomponent_area_um2=500.0,
        injection_seed_points=[[65, 60]],  # [x, y] in the TOP lobe
    )
    core, analysis, warnings, diag = cd.build_injection_masks_with_components(
        stack, VOXEL, inj
    )

    assert diag["seed_filter_applied"] is True
    assert diag["split_applied"] is True
    assert diag["n_subcomponents"] >= 2
    assert bool(core[60, 65]) is True  # seeded top lobe kept
    assert bool(core[160, 65]) is False  # non-seeded bottom lobe removed
