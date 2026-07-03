"""Focused tests for the seeded injection-mask watershed split (Part A)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("skimage")

from mouse_brain_pipeline import candidate_detection as cd  # noqa: E402
from mouse_brain_pipeline import run_layout  # noqa: E402
from mouse_brain_pipeline.config import InjectionExclusionConfig  # noqa: E402
from mouse_brain_pipeline.injection_mask_diagnostics import (  # noqa: E402
    COMPONENTS_AFTER_CSV,
    COMPONENTS_BEFORE_CSV,
    KEPT_VS_REMOVED_PNG,
    SEED_MATCHES_CSV,
    SPLIT_QC_PNG,
    SUMMARY_JSON,
    write_injection_mask_diagnostics,
)

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


def test_multiple_seeds_keep_multiple_valid_subcomponents():
    mask = _dumbbell()
    seeds = [(60, 60), (60, 180)]
    kept, diag = cd._split_and_filter_by_seeds(mask, seeds, (1.0, 1.0), _cfg(), 1)

    assert bool(kept[60, 60]) is True
    assert bool(kept[60, 180]) is True
    assert diag["n_kept"] == 2
    assert diag["n_removed"] == 0
    assert {m["subcomponent_label"] for m in diag["seed_matches"]} == set(
        diag["kept_subcomponent_labels"]
    )


def test_thin_bridge_does_not_keep_unrelated_lobe():
    mask = _dumbbell(neck_half=1)
    kept, diag = cd._split_and_filter_by_seeds(
        mask, [(60, 60)], (1.0, 1.0), _cfg(), 1
    )

    assert bool(kept[60, 60]) is True
    assert bool(kept[60, 180]) is False
    assert diag["n_components"] == 1
    assert diag["n_removed"] >= 1


def test_seedless_nearby_lobe_is_removed():
    yy, xx = np.ogrid[:140, :220]
    seeded = (yy - 70) ** 2 + (xx - 70) ** 2 <= 38 ** 2
    nearby_seedless = (yy - 70) ** 2 + (xx - 158) ** 2 <= 32 ** 2
    mask = seeded | nearby_seedless

    kept, diag = cd._split_and_filter_by_seeds(
        mask, [(70, 70)], (1.0, 1.0), _cfg(), 1
    )

    assert bool(kept[70, 70]) is True
    assert bool(kept[70, 158]) is False
    removed = [row for row in diag["post_split_subcomponents"] if not row["kept"]]
    assert removed and all(not row["contains_seed"] for row in removed)


def test_green_and_red_seed_configs_and_masks_remain_separate():
    cfg = InjectionExclusionConfig.from_dict({
        "split_min_peak_distance_um": 40.0,
        "green_signal": {"injection_seed_points": [[60, 60]]},
        "channel_2_signal": {"injection_seed_points": [[180, 60]]},
    })
    mask = _dumbbell()
    green_cfg = cfg.for_channel("green_signal")
    red_cfg = cfg.for_channel("channel_2_signal")
    green, _ = cd._split_and_filter_by_seeds(
        mask, [(60, 60)], (1.0, 1.0), green_cfg, 1
    )
    red, _ = cd._split_and_filter_by_seeds(
        mask, [(60, 180)], (1.0, 1.0), red_cfg, 1
    )

    assert green_cfg.injection_seed_points != red_cfg.injection_seed_points
    assert bool(green[60, 60]) and not bool(green[60, 180])
    assert bool(red[60, 180]) and not bool(red[60, 60])


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


def test_raw_tiffs_are_not_modified_by_mask_build(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    plane_paths = {}
    source = np.zeros((120, 240), dtype=np.uint16)
    source[_dumbbell()] = 3000
    before = {}
    for plane in range(1, 8):
        path = tmp_path / f"section_070_{plane:02d}.tif"
        tifffile.imwrite(path, source + plane)
        plane_paths[plane] = path
        before[path] = hashlib.sha256(path.read_bytes()).hexdigest()

    stack, plane_numbers, origin, shape = cd.read_crop_stack(plane_paths, crop=None)
    cfg = InjectionExclusionConfig(
        enabled=True, automatic=True, downsample_um=1.0, smoothing_sigma_um=2.0,
        intensity_percentile=80.0, minimum_area_um2=100.0,
        core_dilation_um=0.0, analysis_exclusion_dilation_um=0.0,
        split_min_peak_distance_um=40.0, injection_seed_points=[[60, 60]],
    )
    cd.build_injection_masks_with_components(stack, VOXEL, cfg)

    assert stack.shape[0] == 7 and plane_numbers == list(range(1, 8))
    assert origin == (0, 0) and shape == source.shape
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in plane_paths.values()
    } == before


def test_old_run_folder_is_not_reused(tmp_path):
    run_dir = run_layout.create_run_dir(tmp_path, "section070_seeded_split")
    (run_dir / "all_candidates.csv").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_layout.create_run_dir(tmp_path, "section070_seeded_split")
    assert (run_dir / "all_candidates.csv").read_text(encoding="utf-8") == "existing"


def test_required_diagnostics_include_channel_section_and_plots(tmp_path):
    mask = _dumbbell()
    _kept, diag = cd._split_and_filter_by_seeds(
        mask, [(60, 60)], (1.0, 1.0), _cfg(), 1
    )
    written = write_injection_mask_diagnostics(
        tmp_path, diag, channel="green_signal", section=70
    )

    expected = {
        COMPONENTS_BEFORE_CSV, COMPONENTS_AFTER_CSV, SEED_MATCHES_CSV,
        SUMMARY_JSON, SPLIT_QC_PNG, KEPT_VS_REMOVED_PNG,
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    for name in (COMPONENTS_BEFORE_CSV, COMPONENTS_AFTER_CSV, SEED_MATCHES_CSV):
        rows = list(csv.DictReader((tmp_path / name).open(encoding="utf-8")))
        assert rows
        assert all(row["channel"] == "green_signal" and row["section"] == "70" for row in rows)
    summary = json.loads((tmp_path / SUMMARY_JSON).read_text(encoding="utf-8"))
    assert summary["channel"] == "green_signal" and summary["section"] == 70
    assert set(written) == {
        "components_before_split_csv", "components_after_split_csv",
        "seed_matches_csv", "summary_json", "split_qc_png", "kept_vs_removed_png",
    }
