"""Synthetic tests for the candidate detector (coordinate integrity, masks,
consecutive-plane support, review patches and the Cellfinder adapter).

These build small in-memory 3D stacks (no TIFF I/O, no real data) and skip
cleanly without numpy/scipy.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from mouse_brain_pipeline.candidate_detection import (  # noqa: E402
    STATUS_ARTIFACT,
    STATUS_INJECTION,
    STATUS_MANUAL_REVIEW,
    STATUS_RULE_FAILED,
    STATUS_RULE_PASSED,
    STATUS_SUSPECT_INJECTION,
    DetectionParams,
    build_injection_masks_with_components,
    build_shared_tissue_mask,
    detect_candidates_in_stack,
)
from mouse_brain_pipeline.candidate_qc import aligned_patches  # noqa: E402
from mouse_brain_pipeline.config import (  # noqa: E402
    InjectionExclusionConfig,
    TissueMaskConfig,
)

VOXEL = (6.0, 1.004, 1.004)
RNG = np.random.default_rng(20260625)
ALLOWED_Z_UM = {0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0}


def make_stack(z=7, h=120, w=120, base=200.0, noise=8.0):
    return (base + RNG.normal(0.0, noise, size=(z, h, w))).astype(np.float32)


def add_blob(stack, z_planes, cy, cx, amp=1500.0, sigma_xy=3.0):
    z, h, w = stack.shape
    yy, xx = np.ogrid[:h, :w]
    g = amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_xy ** 2))
    for zp in z_planes:
        stack[zp] += g
    return stack


def base_params(**overrides):
    p = DetectionParams(backend="pilot_log3d")
    p.tissue = TissueMaskConfig(enabled=False)
    p.injection = InjectionExclusionConfig(enabled=False)
    for k, v in overrides.items():
        if not hasattr(p, k):
            raise AttributeError(f"DetectionParams has no field {k!r}")
        setattr(p, k, v)
    return p


def run(stack, params=None, *, crop_origin=(0, 0), shared=None, injection=None,
        backend=None, detect_main=None):
    params = params or base_params()
    return detect_candidates_in_stack(
        stack, params, VOXEL, channel="green_signal", section=70, first_section=70,
        planes_per_section=7, plane_numbers=list(range(1, 8)), crop_origin=crop_origin,
        shared_tissue_mask=shared, injection_cfg=injection, backend=backend,
        cellfinder_detect_main=detect_main,
    )


# 1 -------------------------------------------------------------------------- #
def test_seven_planes_only_valid_z_values():
    stack = make_stack()
    add_blob(stack, (0, 1), 40, 40)
    add_blob(stack, (4, 5), 40, 90)
    add_blob(stack, (2, 3, 4), 90, 60)
    res = run(stack)
    assert res.candidates
    for c in res.candidates:
        assert 0 <= c["z_index"] <= 6
        assert c["section_relative_z_um"] in ALLOWED_Z_UM
        assert c["section_relative_z_um"] == c["z_index"] * 6.0
        assert not c["invalid_coordinate"]


# 2 -------------------------------------------------------------------------- #
def test_no_nan_inf_or_billion_scale_coordinates():
    # Bright uniform plateau (annulus MAD ~ 0) used to make robust-z explode.
    stack = make_stack(h=160, w=160)
    stack[:, 40:120, 40:120] += 5000.0          # flat bright plateau
    add_blob(stack, (2, 3, 4), 80, 80, amp=3000)  # blob on the plateau
    res = run(stack)
    assert res.candidates
    for c in res.candidates:
        assert c["local_robust_z"] != 10000
        assert np.isfinite(c["section_relative_z_um"]) and 0 <= c["section_relative_z_um"] <= 36
        assert np.isfinite(c["global_z_um"])
        assert c["invalid_coordinate"] is False


# 3 -------------------------------------------------------------------------- #
def test_crop_local_to_global_coordinates():
    stack = make_stack()
    add_blob(stack, (2, 3, 4), 60, 60)
    res = run(stack, crop_origin=(100, 200))  # (y0, x0)
    accepted = [c for c in res.candidates if c["current_status"] == STATUS_RULE_PASSED]
    assert len(accepted) == 1
    c = accepted[0]
    assert abs(c["x_local_px"] - 60) <= 3 and abs(c["y_local_px"] - 60) <= 3
    assert c["x_global_px"] == c["x_local_px"] + 200
    assert c["y_global_px"] == c["y_local_px"] + 100


# 4 -------------------------------------------------------------------------- #
def test_full_image_manual_rectangle_maps_into_crop():
    stack = make_stack()
    add_blob(stack, (2, 3, 4), 60, 60)
    origin = (100, 200)  # y0, x0
    # Full-res rect [x_min, x_max, y_min, y_max] covering crop-local x30..90, y30..90.
    inj = InjectionExclusionConfig(
        enabled=True, automatic=False,
        manual_rectangles=[[200 + 30, 200 + 90, 100 + 30, 100 + 90]],
    )
    res = run(stack, crop_origin=origin, injection=inj)
    assert res.injection_mask[60, 60]      # inside the mapped rectangle
    assert res.injection_core_mask[60, 60]
    assert res.injection_analysis_exclusion_mask.sum() >= res.injection_core_mask.sum()
    blob = [c for c in res.candidates if abs(c["x_local_px"] - 60) <= 3]
    assert blob and all(c["current_status"] == STATUS_INJECTION for c in blob)


# 5 + 6 ---------------------------------------------------------------------- #
def _injection_params_for_synthetic():
    return InjectionExclusionConfig(
        enabled=True, automatic=True, downsample_um=2.0, smoothing_sigma_um=20.0,
        intensity_percentile=90.0, minimum_area_um2=4000.0,
        core_dilation_um=5.0, analysis_exclusion_dilation_um=10.0,
    )


def test_broad_injection_region_makes_nonempty_mask():
    stack = make_stack(h=220, w=220)
    stack[:, 30:190, 30:190] += 3000.0
    res = run(stack, injection=_injection_params_for_synthetic())
    assert res.injection_mask is not None and res.injection_mask.any()
    assert res.injection_mask[110, 110]


def test_candidates_inside_injection_are_excluded():
    stack = make_stack(h=220, w=220)
    stack[:, 30:190, 30:190] += 3000.0
    centres = [(80, 80), (110, 110), (150, 150)]
    for (cy, cx) in centres:
        add_blob(stack, (2, 3, 4), cy, cx, amp=2500)
    res = run(stack, injection=_injection_params_for_synthetic())
    accepted = [c for c in res.candidates if c["included_in_count"]]
    assert accepted == []
    assert any(c["current_status"] == STATUS_SUSPECT_INJECTION for c in res.candidates)
    for (cy, cx) in centres:
        assert res.injection_mask[cy, cx]


def test_seed_points_keep_two_injection_components_and_drop_unseeded_component():
    rng = np.random.default_rng(20260626)
    stack = (100.0 + rng.normal(0.0, 2.0, size=(7, 220, 260))).astype(np.float32)
    seeded_top = (70, 140)
    seeded_bottom = (150, 140)
    unseeded = (110, 45)
    for cy, cx in (seeded_top, seeded_bottom, unseeded):
        stack[:, cy - 20:cy + 21, cx - 20:cx + 21] += 2500.0
        add_blob(stack, (2, 3, 4), cy, cx, amp=1200.0, sigma_xy=3.0)

    inj = InjectionExclusionConfig(
        enabled=True, automatic=True, downsample_um=1.0, smoothing_sigma_um=3.0,
        intensity_percentile=94.0, minimum_area_um2=400.0,
        core_dilation_um=0.0, analysis_exclusion_dilation_um=0.0,
        injection_seed_points=[[seeded_top[1], seeded_top[0]],
                               [seeded_bottom[1], seeded_bottom[0]]],
    )
    core, analysis, warnings, diag = build_injection_masks_with_components(stack, VOXEL, inj)

    assert warnings == []
    assert diag["seed_filter_applied"] is True
    assert diag["n_components"] == 3
    assert diag["n_kept"] == 2
    assert diag["n_removed"] == 1
    assert core[seeded_top] and analysis[seeded_top]
    assert core[seeded_bottom] and analysis[seeded_bottom]
    assert not core[unseeded] and not analysis[unseeded]

    class FakeCell:
        def __init__(self, x, y, z=3, type=1):
            self.x, self.y, self.z, self.type = x, y, z, type

    p = base_params(
        max_diameter_um=200.0,
        background_annulus_inner_um=25.0,
        background_annulus_outer_um=35.0,
        min_local_robust_z=1.0,
        minimum_background_pixels=10,
    )
    result = detect_candidates_in_stack(
        stack, p, VOXEL, channel="green_signal", section=70, first_section=70,
        plane_numbers=list(range(1, 8)), injection_cfg=inj,
        backend="cellfinder_candidates",
        cellfinder_detect_main=lambda **_kwargs: [
            FakeCell(seeded_top[1], seeded_top[0]),
            FakeCell(seeded_bottom[1], seeded_bottom[0]),
            FakeCell(unseeded[1], unseeded[0]),
        ],
    )
    by_xy = {(c["x_local_px"], c["y_local_px"]): c for c in result.candidates}
    top_candidate = by_xy[(seeded_top[1], seeded_top[0])]
    bottom_candidate = by_xy[(seeded_bottom[1], seeded_bottom[0])]
    removed_candidate = by_xy[(unseeded[1], unseeded[0])]

    assert top_candidate["current_status"] == STATUS_SUSPECT_INJECTION
    assert bottom_candidate["current_status"] == STATUS_SUSPECT_INJECTION
    assert removed_candidate["inside_injection_analysis_exclusion"] is False
    assert removed_candidate["current_status"] != STATUS_SUSPECT_INJECTION
    assert removed_candidate["current_status"] == removed_candidate[
        "preliminary_sampling_category"
    ]
    assert removed_candidate["rejection_reason"] == removed_candidate[
        "preliminary_rule_reason"
    ]
    assert removed_candidate["injection_assignment_source"] == "none"


# 7 -------------------------------------------------------------------------- #
def test_shared_tissue_mask_covers_dim_tissue_not_only_injection():
    h, w = 120, 200
    dim = (RNG.normal(0, 8, size=(7, h, w))).astype(np.float32)
    dim[:, :, :100] += 300.0                       # DIM, low-signal tissue (left)
    bright = (RNG.normal(0, 8, size=(7, h, w))).astype(np.float32)
    bright[:, 40:160, 120:160] += 5000.0           # bright injection block (other channel)
    cfg = TissueMaskConfig(enabled=True, downsample_um=2.0, smoothing_sigma_um=4.0,
                           threshold_fraction=0.08, closing_um=4.0, minimum_area_um2=2000.0)
    mask = build_shared_tissue_mask([dim, bright], VOXEL, cfg)
    assert mask is not None
    assert mask[60, 40]    # dim tissue (only the low-signal channel) is covered
    assert mask[60, 140]   # injection block is also covered
    assert not mask[60, 185]  # black in BOTH channels stays background


# 8 -------------------------------------------------------------------------- #
def test_compact_object_in_planes_1_and_2_is_retained():
    stack = make_stack()
    add_blob(stack, (0, 1), 60, 60)  # planes 1-2 (z_index 0,1), blank elsewhere
    res = run(stack)
    real = [c for c in res.candidates
            if abs(c["x_local_px"] - 60) <= 4 and abs(c["y_local_px"] - 60) <= 4
            and c["equivalent_diameter_um"] >= 6]
    assert len(real) == 1
    assert real[0]["current_status"] in (STATUS_RULE_PASSED, STATUS_MANUAL_REVIEW)
    assert real[0]["n_consecutive_planes"] >= 2


# 9 -------------------------------------------------------------------------- #
def test_compact_object_in_planes_2_to_4_retained_when_5_to_7_blank():
    stack = make_stack()
    add_blob(stack, (1, 2, 3), 60, 60)  # planes 2-4, planes 5-7 blank
    res = run(stack)
    near = [c for c in res.candidates if abs(c["x_local_px"] - 60) <= 4]
    assert len(near) == 1
    assert near[0]["current_status"] == STATUS_RULE_PASSED
    assert near[0]["n_consecutive_planes"] == 3


# 10 ------------------------------------------------------------------------- #
def test_spatially_jumping_object_is_rejected():
    stack = make_stack(h=140, w=140)
    # One connected 3D object whose XY centroid drifts >5 um across planes.
    add_blob(stack, (2,), 60, 50, amp=1500, sigma_xy=5)
    add_blob(stack, (3,), 60, 58, amp=1500, sigma_xy=5)
    add_blob(stack, (4,), 60, 66, amp=1500, sigma_xy=5)
    res = run(stack, base_params(max_diameter_um=60.0, max_elongation=5.0))
    assert not any(c["current_status"] == STATUS_RULE_PASSED for c in res.candidates)
    assert any(c["rejection_reason"] == "xy_jump" for c in res.candidates)


# 11 ------------------------------------------------------------------------- #
def test_review_patches_use_same_xy_centre_in_every_plane():
    # A unique bright pixel at the SAME (y, x) in every plane must land on the
    # SAME patch coordinate after windowing.
    planes = [np.zeros((120, 120), dtype=np.float32) for _ in range(7)]
    yg, xg = 70, 55
    for i, p in enumerate(planes):
        p[yg, xg] = 1000 + i  # distinct max per plane, same location
    patches, cy, cx = aligned_patches(planes, yg, xg, half_px=18)
    assert len({p.shape for p in patches}) == 1  # identical windows
    for p in patches:
        yy, xx = np.unravel_index(int(np.argmax(p)), p.shape)
        assert (yy, xx) == (cy, cx)


# 12 ------------------------------------------------------------------------- #
def test_cellfinder_adapter_receives_zyx_array_and_voxels():
    captured = {}

    class FakeCell:
        def __init__(self, x, y, z, type):
            self.x, self.y, self.z, self.type = x, y, z, type

    def stub_detect_main(**kwargs):
        captured.update(kwargs)
        # one candidate (type 1=UNKNOWN) and one artifact (type -1).
        return [FakeCell(60, 60, 3, 1), FakeCell(15, 15, 3, -1)]

    stack = make_stack()
    add_blob(stack, (2, 3, 4), 60, 60)
    res = run(stack, backend="cellfinder_candidates", detect_main=stub_detect_main)

    arr = captured["signal_array"]
    assert arr.ndim == 3 and arr.shape[0] == 7          # z, y, x order
    assert arr.dtype == np.uint16                       # source 16-bit preserved
    assert tuple(captured["voxel_sizes"]) == (6.0, 1.004, 1.004)
    assert any(c["current_status"] == STATUS_ARTIFACT for c in res.candidates)
    for c in res.candidates:
        assert 0 <= c["z_index"] <= 6
        assert c["section_relative_z_um"] in ALLOWED_Z_UM
