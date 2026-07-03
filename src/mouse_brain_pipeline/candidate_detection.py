"""EXPERIMENTAL pilot 3D candidate detector -- "candidate detections" ONLY.

Two backends generate candidate seeds; both share identical post-processing
(coordinate bookkeeping, shared tissue mask, per-channel injection-site
exclusion, consecutive-plane support, classification and auditing):

  * ``backend: "cellfinder_candidates"`` -- thin adapter around the real
    ``cellfinder.core.detect.detect.main`` candidate-detection stage (no
    background channel required). Preferred when Cellfinder is installed.
  * ``backend: "pilot_log3d"`` -- a self-contained rule-based 3D detector used
    when Cellfinder is unavailable (must be selected explicitly).

NOTHING here is a validated cell count. Display states include preliminary
sampling categories, retained Cellfinder artefacts, invalid measurements,
suspect automatic injection masks and confirmed injection assignments.

Coordinate integrity rules (see PR history for the original bug -- a robust
z-score was being written into a field labelled ``z``):
  * never reuse bare ``x``/``y``/``z`` after array ops,
  * carry explicit ``z_index`` / ``section_relative_z_um`` / ``global_z_um`` /
    ``x_local_px`` / ``y_local_px`` / ``x_global_px`` / ``y_global_px``,
  * validate every candidate before it is saved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .config import (
    CellfinderConfig,
    Config,
    InjectionExclusionConfig,
    TissueMaskConfig,
)
from .filenames import global_plane
from .utilities import LOG, ensure_dir

# ----- statuses (final pilot states) -------------------------------------- #
STATUS_PRELIMINARY_PASS = "preliminary_rule_pass"
STATUS_PRELIMINARY_FAIL = "preliminary_rule_fail"
# Compatibility aliases for internal imports. Their values deliberately use the
# corrected terminology.
STATUS_RULE_PASSED = STATUS_PRELIMINARY_PASS
STATUS_RULE_FAILED = STATUS_PRELIMINARY_FAIL
STATUS_INJECTION = "injection_site"
STATUS_SUSPECT_INJECTION = "suspect_injection_mask"
STATUS_ARTIFACT = "artifact"
STATUS_MANUAL_REVIEW = "manual_review"
STATUS_INVALID_MEASUREMENT = "invalid_measurement"

# ----- explicit rejection reasons ----------------------------------------- #
REASON_OUTSIDE_TISSUE = "outside_tissue"
REASON_INJECTION = "injection_site"
REASON_SUSPECT_INJECTION = "suspect_injection_mask"
REASON_TOO_SMALL = "too_small"
REASON_TOO_LARGE = "too_large"
REASON_LOW_CONTRAST = "insufficient_local_contrast"
REASON_SINGLE_PLANE = "single_plane"
REASON_MANY_PLANES_REVIEW = "many_planes_review"
REASON_XY_JUMP = "xy_jump"
REASON_ELONGATED = "too_elongated"
REASON_CROP_BOUNDARY = "crop_boundary"
REASON_DUPLICATE = "duplicate"
REASON_ARTIFACT = "artifact"
REASON_INVALID_COORD = "invalid_coordinate"
REASON_CF_Z_INVALID = "cellfinder_z_out_of_range"

CANDIDATE_COLUMNS = [
    "candidate_id",
    "candidate_exists",
    "candidate_generation_index",
    "channel",
    "backend",
    "candidate_generation_source",
    "detected_on_raw_stack",
    "detected_on_injection_suppressed_stack",
    "generation_suppression_used",
    "generation_suppression_mask_used",
    "generation_suppression_mask_source",
    "x_local_px",
    "y_local_px",
    "x_global_px",
    "y_global_px",
    "z_index",
    "optical_plane",
    "section",
    "section_relative_z_um",
    "global_z_um",
    "original_cellfinder_x_local_px",
    "original_cellfinder_y_local_px",
    "original_cellfinder_z_index",
    "cellfinder_z_index",
    "cellfinder_returned_z_raw",
    "cellfinder_z_mapping_method",
    "mapped_stack_z_index",
    "original_cellfinder_z_valid",
    "fixed_xy_center_x_local_px",
    "fixed_xy_center_y_local_px",
    "fixed_xy_peak_z_index",
    "fixed_xy_support_z_indices",
    "fixed_xy_n_consecutive_planes",
    "fixed_xy_z_extent_um",
    "peak_plane",
    "z_indices",
    "n_consecutive_planes",
    "support_z_indices",
    "support_plane_count",
    "support_start",
    "support_end",
    "peak_z_index",
    "plane_0_contrast",
    "plane_1_contrast",
    "plane_2_contrast",
    "plane_3_contrast",
    "plane_4_contrast",
    "plane_5_contrast",
    "plane_6_contrast",
    "xy_centroid_shift_um",
    "peak_intensity",
    "mean_intensity",
    "local_background_median",
    "local_background_mad",
    "local_background_noise",
    "local_robust_z",
    "local_contrast_score",
    "measurement_valid",
    "measurement_reason",
    "background_pixel_count",
    "background_noise_method",
    "equivalent_diameter_um",
    "xy_diameter_um",
    "z_extent_um",
    "volume_um3",
    "elongation",
    "inside_tissue",
    "inside_injection_site",
    "inside_injection_core",
    "inside_injection_analysis_exclusion",
    "injection_mask_source",
    "injection_mask_validated",
    "injection_mask_qc_failed",
    "injection_assignment_source",
    "touches_crop_boundary",
    "invalid_coordinate",
    "classification_score",
    "cellfinder_x",
    "cellfinder_y",
    "cellfinder_z",
    "cellfinder_type",
    "cellfinder_type_name",
    "preliminary_sampling_category",
    "preliminary_rule_reason",
    "manual_label",
    "classifier_probability",
    "classifier_model",
    "classifier_version",
    "model_validation_passed",
    "final_decision",
    "current_status",
    "included_in_count",
    "rejection_reason",
    "source_crop",
]

_EPS = 1e-6
_FOREGROUND_SIGMA = 3.0


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class DetectionParams:
    """Plain (I/O-free) parameters so the pipeline is unit-testable on arrays."""

    backend: str = "pilot_log3d"

    min_diameter_um: float = 6.0
    max_diameter_um: float = 30.0

    background_sigma_um: float = 40.0
    log_sigma_min_um: float = 2.0
    log_sigma_max_um: float = 8.0

    min_local_robust_z: float = 6.0
    z_support_min_contrast: float = 3.0
    central_region_radius_um: float = 3.0
    background_annulus_inner_um: float = 8.0
    background_annulus_outer_um: float = 16.0
    minimum_background_pixels: int = 20
    padding_values: tuple[float, ...] = (0.0,)

    min_consecutive_planes: int = 2
    max_consecutive_planes: int = 6
    max_xy_shift_um: float = 5.0
    merge_distance_xy_um: float = 8.0
    merge_distance_z_um: float = 12.0
    min_separation_um: float = 6.0

    max_elongation: float = 3.0

    exclude_crop_boundary: bool = True
    crop_boundary_margin_um: float = 15.0

    single_plane_manual_review: bool = True
    single_plane_review_min_z: float = 8.0
    # A single-plane object at or above this robust-z is accepted directly as a
    # preliminary pass instead of being routed to manual review. Must be >=
    # single_plane_review_min_z; the band between the two stays in review.
    single_plane_pass_min_z: float = 12.0

    foreground_sigma_factor: float = _FOREGROUND_SIGMA
    planes_per_section: int = 7

    tissue: TissueMaskConfig = field(default_factory=TissueMaskConfig)
    cellfinder: CellfinderConfig = field(default_factory=CellfinderConfig)
    injection: InjectionExclusionConfig = field(default_factory=InjectionExclusionConfig)


def params_from_config(config: Config) -> DetectionParams:
    d = config.detection
    return DetectionParams(
        backend=str(d.backend),
        min_diameter_um=float(d.minimum_cell_diameter_um),
        max_diameter_um=float(d.maximum_cell_diameter_um),
        background_sigma_um=float(d.background_sigma_um),
        log_sigma_min_um=float(d.log_sigma_min_um),
        log_sigma_max_um=float(d.log_sigma_max_um),
        min_local_robust_z=float(d.minimum_local_robust_z),
        z_support_min_contrast=float(d.z_support_min_contrast),
        central_region_radius_um=float(d.central_region_radius_um),
        background_annulus_inner_um=float(d.background_annulus_inner_um),
        background_annulus_outer_um=float(d.background_annulus_outer_um),
        minimum_background_pixels=int(d.minimum_background_pixels),
        padding_values=tuple(float(v) for v in d.padding_values),
        min_consecutive_planes=int(d.minimum_consecutive_planes),
        max_consecutive_planes=int(d.maximum_consecutive_planes),
        max_xy_shift_um=float(d.maximum_xy_centroid_shift_um),
        merge_distance_xy_um=float(d.merge_distance_xy_um),
        merge_distance_z_um=float(d.merge_distance_z_um),
        min_separation_um=float(d.minimum_candidate_separation_um),
        max_elongation=float(d.maximum_elongation),
        exclude_crop_boundary=bool(d.exclude_crop_boundary_objects),
        crop_boundary_margin_um=float(d.crop_boundary_margin_um),
        single_plane_manual_review=bool(d.single_plane_manual_review),
        single_plane_review_min_z=float(d.single_plane_review_min_robust_z),
        single_plane_pass_min_z=float(d.single_plane_pass_min_robust_z),
        planes_per_section=int(config.acquisition.planes_per_section),
        tissue=d.tissue_mask,
        cellfinder=d.cellfinder,
        injection=d.injection_exclusion,
    )


def diameter_um_to_sigma_px(diameter_um: float, voxel_yx_um: float) -> float:
    """Approximate LoG sigma (XY pixels) for a given blob diameter in micrometres."""
    radius_px = (diameter_um / 2.0) / voxel_yx_um
    return radius_px / (2.0 ** 0.5)


# --------------------------------------------------------------------------- #
# Small numerical / morphological helpers
# --------------------------------------------------------------------------- #
def _mad(values) -> float:
    import numpy as np  # noqa: PLC0415

    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _disk(radius_px: float):
    import numpy as np  # noqa: PLC0415

    r = max(1, int(round(radius_px)))
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return (yy * yy + xx * xx) <= (r * r)


def _remove_small_components(mask, min_area_px: float):
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    areas = np.bincount(lab.ravel())
    keep = areas >= min_area_px
    keep[0] = False
    return keep[lab]


def _upsample_nearest(mask_lowres, out_shape):
    """Nearest-neighbour upsample a boolean low-res mask to ``out_shape``."""
    import numpy as np  # noqa: PLC0415

    H, W = out_shape
    h, w = mask_lowres.shape
    if (h, w) == (H, W):
        return mask_lowres
    yi = (np.arange(H) * h // max(H, 1)).clip(0, h - 1)
    xi = (np.arange(W) * w // max(W, 1)).clip(0, w - 1)
    return mask_lowres[yi][:, xi]


def _robust_unit_scale(plane):
    """Scale a 2D plane to ~[0,1] using robust 1st/99th percentiles."""
    import numpy as np  # noqa: PLC0415

    lo, hi = np.percentile(plane, [1.0, 99.0])
    return np.clip((plane - lo) / max(hi - lo, _EPS), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Stage: shared tissue mask (built once from BOTH channels)
# --------------------------------------------------------------------------- #
def build_shared_tissue_mask(stacks: Sequence, voxel_zyx, cfg: TissueMaskConfig):
    """Permissive foreground mask from the combined channels.

    Removes only the clearly-black background outside the specimen. Returns a
    full-crop-resolution boolean array, or ``None`` when disabled.
    """
    if not cfg.enabled or not stacks:
        return None
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    vz, vy, vx = voxel_zyx
    H, W = stacks[0].shape[1:]
    factor = max(1, int(round(cfg.downsample_um / vy)))

    shared = None
    for stack in stacks:
        proj = stack.max(axis=0).astype(np.float32)
        low = proj[::factor, ::factor]
        scaled = _robust_unit_scale(low)
        shared = scaled if shared is None else np.maximum(shared, scaled)

    sigma_px = max(1.0, cfg.smoothing_sigma_um / (vy * factor))
    shared = ndi.gaussian_filter(shared, sigma=sigma_px)
    thr = cfg.threshold_fraction  # shared is already ~[0,1]
    mask = shared > thr

    close_px = max(1, int(round(cfg.closing_um / (vy * factor))))
    mask = ndi.binary_closing(mask, structure=_disk(close_px))
    mask = ndi.binary_fill_holes(mask)
    min_area_px = cfg.minimum_area_um2 / ((vy * factor) * (vx * factor))
    mask = _remove_small_components(mask, min_area_px)

    full = _upsample_nearest(mask, (H, W))
    if not full.any():
        LOG.warning("Shared tissue mask is empty -- treating the whole crop as tissue.")
        return None
    return full


# --------------------------------------------------------------------------- #
# Stage: per-channel injection-site mask
# --------------------------------------------------------------------------- #
def rasterize_polygons(shape, polygons, crop_origin=(0, 0)):
    """Boolean mask (crop resolution) of pixels inside any full-res px polygon.

    ``polygons`` is a list of ``[[x0,y0],[x1,y1],...]`` in full-resolution pixels;
    ``crop_origin`` is (oy, ox) so a crop maps back to full-res coordinates.
    """
    import numpy as np  # noqa: PLC0415

    H, W = shape
    out = np.zeros((H, W), dtype=bool)
    polys = polygons or []
    if not polys:
        return out
    oy, ox = crop_origin
    try:
        from matplotlib.path import Path as MplPath  # noqa: PLC0415

        yy, xx = np.mgrid[0:H, 0:W]
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        for poly in polys:
            if poly is None or len(poly) < 3:
                LOG.warning("Ignoring polygon with < 3 points: %r", poly)
                continue
            local = [(px - ox, py - oy) for (px, py) in poly]
            out |= MplPath(local).contains_points(pts).reshape(H, W)
    except Exception as exc:  # pragma: no cover - matplotlib optional here
        LOG.warning("Could not rasterise polygon(s): %s", exc)
    return out


def _manual_regions_into_mask(mask, cfg: InjectionExclusionConfig, crop_origin, voxel_yx):
    """OR manual rectangles/polygons (full-res px) into a crop-resolution mask."""
    H, W = mask.shape
    oy, ox = crop_origin
    for rect in (cfg.manual_rectangles or []):
        try:
            x0, x1, y0, y1 = (float(v) for v in rect)
        except (TypeError, ValueError):
            LOG.warning("Ignoring malformed manual injection rectangle: %r", rect)
            continue
        ry0 = max(0, int(round(min(y0, y1) - oy)))
        ry1 = min(H, int(round(max(y0, y1) - oy)) + 1)
        rx0 = max(0, int(round(min(x0, x1) - ox)))
        rx1 = min(W, int(round(max(x0, x1) - ox)) + 1)
        if ry1 > ry0 and rx1 > rx0:
            mask[ry0:ry1, rx0:rx1] = True

    mask |= rasterize_polygons(mask.shape, cfg.manual_polygons, crop_origin)
    return mask


def _subtract_non_injection(mask, cfg: InjectionExclusionConfig, crop_origin):
    """Remove any manual non-injection polygon from ``mask`` (returns a new mask).

    This is the LAST mask step so a falsely-included region cannot be re-added by
    a preceding dilation.
    """
    remove = rasterize_polygons(mask.shape, cfg.manual_non_injection_polygons, crop_origin)
    if remove.any():
        return mask & ~remove
    return mask


def _seed_points_local(cfg: InjectionExclusionConfig, crop_origin, shape):
    """Config seed points (full-res [x, y]) -> crop-local (y, x) inside the crop."""
    H, W = shape
    oy, ox = crop_origin
    local = []
    for point in (cfg.injection_seed_points or []):
        try:
            x, y = (float(v) for v in point)
        except (TypeError, ValueError):
            LOG.warning("Ignoring malformed injection seed point: %r", point)
            continue
        yl = int(round(y - oy))
        xl = int(round(x - ox))
        if 0 <= yl < H and 0 <= xl < W:
            local.append((yl, xl))
    return local


def _filter_components_by_seeds(auto_mask, seeds_local):
    """Keep only auto components containing a seed point; report each component.

    Returns (kept_mask, diagnostics). diagnostics carries the all/kept/removed
    masks plus a per-component list of areas and kept/removed flags.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    labels, n = ndi.label(auto_mask)
    seed_labels = set()
    for (yl, xl) in seeds_local:
        label = int(labels[yl, xl])
        if label > 0:
            seed_labels.add(label)

    components = []
    keep_labels = []
    for label in range(1, n + 1):
        member = labels == label
        area = int(member.sum())
        ys, xs = np.nonzero(member)
        kept = label in seed_labels
        components.append({
            "label": label,
            "area_px": area,
            "centroid_x_local": int(round(xs.mean())) if xs.size else 0,
            "centroid_y_local": int(round(ys.mean())) if ys.size else 0,
            "contains_seed": kept,
            "kept": kept,
        })
        if kept:
            keep_labels.append(label)

    kept_mask = np.isin(labels, keep_labels) if keep_labels else np.zeros_like(auto_mask)
    removed_mask = auto_mask & ~kept_mask
    diag = {
        "seed_filter_applied": True,
        "n_seed_points": len(seeds_local),
        "n_components": n,
        "n_kept": len(keep_labels),
        "n_removed": n - len(keep_labels),
        "components": components,
        "seed_points_local": list(seeds_local),
        "all_auto_mask": auto_mask.copy(),
        "kept_auto_mask": kept_mask,
        "removed_auto_mask": removed_mask,
        "warnings": [],
    }
    if n and not keep_labels:
        diag["warnings"].append(
            "no automatic injection component contained a seed point -- all "
            "automatic components removed; check injection_seed_points."
        )
    return kept_mask, diag


def _watershed_split_component(comp_mask, min_peak_distance_px, voxel_yx,
                               seed_markers=()):
    """Split one bright component with seed and distance-peak markers.

    Configured seeds are explicit watershed markers. Additional distance maxima
    mark non-seeded lobes, so a weak bridge cannot make an entire multi-lobed
    component inherit a seed. Automatic peaks within ``min_peak_distance_px`` of
    a seed are suppressed to avoid fragmenting the immediate seeded region.

    Returns ``(labels, marker_records)``. ``labels`` is >= 1 throughout the
    component; each marker record identifies whether the marker came from a
    configured seed or an automatic distance peak.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415
    from skimage.feature import peak_local_max  # noqa: PLC0415
    from skimage.segmentation import watershed  # noqa: PLC0415

    distance = ndi.distance_transform_edt(comp_mask, sampling=voxel_yx)
    automatic_peaks = peak_local_max(
        distance,
        min_distance=max(1, int(round(min_peak_distance_px))),
        labels=comp_mask,
        exclude_border=False,
    )

    markers = np.zeros(comp_mask.shape, dtype=np.int32)
    marker_records: list[dict] = []
    marker_positions: list[tuple[int, int]] = []

    # Seed markers are installed first and can never be displaced by an
    # automatically detected distance maximum.
    for seed_index, yy, xx in seed_markers:
        yy, xx = int(yy), int(xx)
        marker_label = int(markers[yy, xx])
        if marker_label == 0:
            marker_label = len(marker_positions) + 1
            markers[yy, xx] = marker_label
            marker_positions.append((yy, xx))
        marker_records.append({
            "marker_label": marker_label,
            "marker_source": "configured_seed",
            "seed_index": int(seed_index),
            "y_lowres": yy,
            "x_lowres": xx,
        })

    min_peak_distance_sq = float(min_peak_distance_px) ** 2
    for yy, xx in automatic_peaks:
        yy, xx = int(yy), int(xx)
        if any(
            (yy - sy) ** 2 + (xx - sx) ** 2 < min_peak_distance_sq
            for sy, sx in marker_positions
        ):
            continue
        marker_label = len(marker_positions) + 1
        markers[yy, xx] = marker_label
        marker_positions.append((yy, xx))
        marker_records.append({
            "marker_label": marker_label,
            "marker_source": "distance_peak",
            "seed_index": None,
            "y_lowres": yy,
            "x_lowres": xx,
        })

    # A non-empty component always needs at least one marker. This fallback is
    # deterministic and corresponds to the deepest point in the component.
    if not marker_positions:
        yy, xx = np.unravel_index(int(np.argmax(distance)), distance.shape)
        markers[yy, xx] = 1
        marker_positions.append((int(yy), int(xx)))
        marker_records.append({
            "marker_label": 1,
            "marker_source": "distance_global_maximum",
            "seed_index": None,
            "y_lowres": int(yy),
            "x_lowres": int(xx),
        })

    if len(marker_positions) == 1:
        labels = comp_mask.astype(np.int32)
    else:
        labels = watershed(-distance, markers, mask=comp_mask).astype(np.int32)
    return labels, marker_records


def _split_and_filter_by_seeds(mask, seeds_local, voxel_yx, cfg, factor,
                               *, filter_by_seeds=True):
    """Watershed-split merged bright lobes, then keep only seeded subcomponents.

    ``mask`` is the downsampled bright candidate mask; ``seeds_local`` contains
    full-resolution crop-local ``(y, x)`` coordinates. Seed markers and automatic
    distance-peak markers divide neck-connected lobes. When ``filter_by_seeds``
    is true, only subcomponents matched to configured seeds are kept -- no
    touching unseeded fragment is added back. When false, splitting is diagnostic
    only and the original mask is preserved (the no-seed red-channel path).

    Returns ``(kept_mask, diag)`` at ``mask`` resolution. Reported coordinates
    and pixel areas use full-resolution local pixels; physical areas use um2.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    vy_low, vx_low = float(voxel_yx[0]), float(voxel_yx[1])
    min_peak_px = max(1.0, float(cfg.split_min_peak_distance_um) / vy_low)
    half = factor // 2

    pre_labels, n_pre = ndi.label(mask)
    # Match each full-resolution seed to the low-resolution candidate mask.
    # A seed may move by at most one coarse-pixel diagonal to compensate for
    # sampling-grid quantisation; this is recorded explicitly in the seed CSV.
    nearest_indices = None
    nearest_distance = None
    if mask.any():
        nearest_distance, nearest_indices = ndi.distance_transform_edt(
            ~mask, sampling=(vy_low, vx_low), return_indices=True
        )
    max_direct_match_um = float(np.hypot(vy_low, vx_low))
    matched_seeds: list[dict] = []
    for seed_index, (y_full, x_full) in enumerate(seeds_local):
        y_low = min(mask.shape[0] - 1, max(0, int(y_full) // factor))
        x_low = min(mask.shape[1] - 1, max(0, int(x_full) // factor))
        matched_y, matched_x = y_low, x_low
        match_distance_um = 0.0
        match_method = "contained"
        if not mask[y_low, x_low]:
            match_distance_um = (
                float(nearest_distance[y_low, x_low])
                if nearest_distance is not None else float("inf")
            )
            if nearest_indices is not None and match_distance_um <= max_direct_match_um:
                matched_y = int(nearest_indices[0, y_low, x_low])
                matched_x = int(nearest_indices[1, y_low, x_low])
                match_method = "direct_nearest_candidate"
            else:
                matched_y = matched_x = -1
                match_method = "no_candidate_within_direct_match_radius"
        matched_seeds.append({
            "seed_index": seed_index,
            "seed_x_local": int(x_full),
            "seed_y_local": int(y_full),
            "seed_x_lowres": x_low,
            "seed_y_lowres": y_low,
            "matched_x_lowres": matched_x,
            "matched_y_lowres": matched_y,
            "match_method": match_method,
            "match_distance_um": round(match_distance_um, 3),
        })

    post_labels = np.zeros(mask.shape, dtype=np.int32)
    pre_records: list[dict] = []
    sub_records: list[dict] = []
    sub_meta: dict[int, dict] = {}
    split_markers: list[dict] = []
    next_label = 1

    for pre_label in range(1, n_pre + 1):
        comp = pre_labels == pre_label
        ys, xs = np.nonzero(comp)
        pre_area_low = int(comp.sum())
        component_seed_markers = [
            (seed["seed_index"], seed["matched_y_lowres"], seed["matched_x_lowres"])
            for seed in matched_seeds
            if seed["matched_y_lowres"] >= 0
            and int(pre_labels[
                seed["matched_y_lowres"], seed["matched_x_lowres"]
            ]) == pre_label
        ]
        sub_local, marker_records = _watershed_split_component(
            comp, min_peak_px, (vy_low, vx_low), component_seed_markers
        )
        local_ids = [k for k in range(1, int(sub_local.max()) + 1) if np.any(sub_local == k)]
        pre_records.append({
            "pre_label": pre_label,
            "area_px": pre_area_low * factor * factor,
            "area_um2": round(pre_area_low * vy_low * vx_low, 2),
            "centroid_x_local": int(round(xs.mean())) * factor + half if xs.size else 0,
            "centroid_y_local": int(round(ys.mean())) * factor + half if ys.size else 0,
            "n_subcomponents": len(local_ids),
        })
        local_to_global: dict[int, int] = {}
        for local_id in local_ids:
            submask = sub_local == local_id
            global_label = next_label
            next_label += 1
            local_to_global[local_id] = global_label
            post_labels[submask] = global_label
            sy, sx = np.nonzero(submask)
            area_low = int(submask.sum())
            sub_meta[global_label] = {
                "pre_label": pre_label,
                "area_low": area_low,
                "mask_low": submask,
            }
            sub_records.append({
                "label": global_label,
                "subcomponent_label": global_label,
                "parent_pre_label": pre_label,
                "area_px": area_low * factor * factor,
                "area_um2": round(area_low * vy_low * vx_low, 2),
                "centroid_x_local": int(round(sx.mean())) * factor + half if sx.size else 0,
                "centroid_y_local": int(round(sy.mean())) * factor + half if sy.size else 0,
                "contains_seed": False,
                "kept": False,
                "reason": "",
            })
        for marker in marker_records:
            local_label = int(sub_local[marker["y_lowres"], marker["x_lowres"]])
            split_markers.append({
                **marker,
                "pre_label": pre_label,
                "subcomponent_label": local_to_global.get(local_label, 0),
            })

    record_by_label = {rec["label"]: rec for rec in sub_records}

    # Seed matching: each matched seed keeps exactly the basin it lands in.
    seed_matches: list[dict] = []
    seeded_labels: set[int] = set()
    for seed in matched_seeds:
        yl, xl = seed["matched_y_lowres"], seed["matched_x_lowres"]
        label = int(post_labels[yl, xl]) if yl >= 0 and xl >= 0 else 0
        if label > 0:
            seeded_labels.add(label)
            record_by_label[label]["contains_seed"] = True
        seed_matches.append({
            **seed,
            "pre_label": int(sub_meta[label]["pre_label"]) if label > 0 else 0,
            "subcomponent_label": label,
            "kept": bool(filter_by_seeds and label > 0),
        })

    kept_labels = set(seeded_labels) if filter_by_seeds else set(sub_meta)
    for label, meta in sub_meta.items():
        if not filter_by_seeds:
            record_by_label[label]["kept"] = True
            record_by_label[label]["reason"] = "no_seeds_configured_preserved"
        elif label in seeded_labels:
            record_by_label[label]["kept"] = True
            record_by_label[label]["reason"] = "contains_or_directly_matches_seed"
        else:
            record_by_label[label]["reason"] = "non_seeded_subcomponent"

    kept_mask = np.isin(post_labels, list(kept_labels)) if kept_labels else np.zeros_like(mask)

    warnings: list[str] = []
    if filter_by_seeds and n_pre and not seeded_labels:
        warnings.append(
            "no post-split injection subcomponent contained a seed point -- all "
            "automatic components removed; check injection_seed_points."
        )

    diag = {
        "seed_filter_applied": bool(filter_by_seeds),
        "split_applied": True,
        "split_method": "seed_marker_distance_transform_watershed",
        "n_seed_points": len(seeds_local),
        "n_components": n_pre,
        "n_subcomponents": len(sub_records),
        "n_kept": len(kept_labels),
        "n_removed": len(sub_records) - len(kept_labels),
        "factor": factor,
        "low_shape": list(mask.shape),
        "split_min_peak_distance_um": float(cfg.split_min_peak_distance_um),
        "split_min_subcomponent_area_um2": float(cfg.split_min_subcomponent_area_um2),
        "seed_direct_match_radius_um": round(max_direct_match_um, 3),
        "pre_split_components": pre_records,
        "post_split_subcomponents": sub_records,
        "seed_matches": seed_matches,
        "split_markers": split_markers,
        "components": sub_records,  # back-compat: per-subcomponent log records
        "pre_labels_lowres": pre_labels.astype(np.int32),
        "post_labels_lowres": post_labels,
        "kept_subcomponent_labels": sorted(kept_labels),
        "warnings": warnings,
    }
    return kept_mask, diag


def _injection_base_mask(stack_zyx, voxel_zyx, cfg: InjectionExclusionConfig,
                         crop_origin=(0, 0)):
    """Bright broad injection BASE mask (pre-dilation), warnings and component diag.

    When injection_seed_points are configured, automatic components are labelled
    and only seeded components are kept BEFORE any dilation. Merged (touching)
    bright lobes are first split with a distance-transform watershed so a seeded
    lobe can be kept while a touching non-seeded lobe is removed. Manual regions
    are always kept.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    vz, vy, vx = voxel_zyx
    H, W = stack_zyx.shape[1:]
    auto = np.zeros((H, W), dtype=bool)
    warnings: list[str] = []
    bright_low = None
    factor = 1

    if cfg.enabled and cfg.automatic:
        factor = max(1, int(round(cfg.downsample_um / vy)))
        proj = stack_zyx.max(axis=0).astype(np.float32)
        low = proj[::factor, ::factor]
        sigma_px = max(1.0, cfg.smoothing_sigma_um / (vy * factor))
        sm = ndi.gaussian_filter(low, sigma=sigma_px)

        thr = float(np.percentile(sm, cfg.intensity_percentile))
        med = float(np.median(sm))
        bright = sm >= thr
        min_area_px = cfg.minimum_area_um2 / ((vy * factor) * (vx * factor))
        bright = _remove_small_components(bright, min_area_px)

        if bright.any():
            bright_low = bright
        elif (sm.max() - med) > 5.0 * (_mad(sm.ravel()) * 1.4826 + _EPS):
            # An obviously bright broad region exists but no region survived the
            # area filter -- flag it instead of silently producing no mask.
            warnings.append(
                "injection auto-mask is empty although a bright broad region is "
                "present; lower minimum_area_um2 or set a manual_rectangle."
            )

    # Split candidate components for diagnostics in both channels. Seed filtering
    # is applied only where channel-specific seeds are configured; with no seeds,
    # the split is diagnostic and the original automatic mask is preserved.
    diag: dict = {"seed_filter_applied": False, "components": []}
    seeds_configured = bool(cfg.injection_seed_points)
    seeds_local = _seed_points_local(cfg, crop_origin, (H, W))
    if bright_low is not None and bright_low.any():
        use_split = bool(getattr(cfg, "split_merged_components", True)) and _watershed_available()
        if use_split:
            kept_low, diag = _split_and_filter_by_seeds(
                bright_low, seeds_local, (vy * factor, vx * factor), cfg, factor,
                filter_by_seeds=seeds_configured,
            )
            auto = _upsample_nearest(kept_low, (H, W))
            diag["seed_points_local"] = list(seeds_local)
            if seeds_configured:
                all_full = _upsample_nearest(bright_low, (H, W))
                diag["all_auto_mask"] = all_full
                diag["kept_auto_mask"] = auto.copy()
                diag["removed_auto_mask"] = all_full & ~auto
            warnings.extend(diag.get("warnings", []))
        elif seeds_configured:
            auto_full = _upsample_nearest(bright_low, (H, W))
            auto, diag = _filter_components_by_seeds(auto_full, seeds_local)
            warnings.extend(diag.get("warnings", []))
        else:
            auto = _upsample_nearest(bright_low, (H, W))

    base = _manual_regions_into_mask(auto, cfg, crop_origin, vy)
    return base, warnings, diag


def _watershed_available() -> bool:
    try:
        import skimage.feature  # noqa: F401,PLC0415
        import skimage.segmentation  # noqa: F401,PLC0415
    except Exception:  # pragma: no cover - optional dependency guard
        return False
    return True


def _dilate_um(base, voxel_zyx, dilation_um):
    """Boolean mask of all pixels within ``dilation_um`` of ``base`` (XY EDT)."""
    from scipy import ndimage as ndi  # noqa: PLC0415

    _vz, vy, vx = voxel_zyx
    if not base.any():
        return base.copy()
    distance_um = ndi.distance_transform_edt(~base, sampling=(vy, vx))
    return distance_um <= float(dilation_um)


def build_injection_masks_with_components(stack_zyx, voxel_zyx,
                                          cfg: InjectionExclusionConfig, crop_origin=(0, 0)):
    """Build core/analysis masks and return the seed-filter component diagnostics."""
    base, warnings, diag = _injection_base_mask(stack_zyx, voxel_zyx, cfg, crop_origin)
    if base.any():
        core = _dilate_um(base, voxel_zyx, cfg.core_dilation_um)
        analysis = _dilate_um(base, voxel_zyx, cfg.analysis_exclusion_dilation_um)
    else:
        core = base.copy()
        analysis = base.copy()

    # Subtract manual non-injection polygons LAST so dilation cannot add them back.
    core = _subtract_non_injection(core, cfg, crop_origin)
    analysis = _subtract_non_injection(analysis, cfg, crop_origin)

    if cfg.enabled and not analysis.any():
        warnings.append("injection mask is EMPTY for this channel (no red boundary will show).")
    return core, analysis, warnings, diag


def build_injection_masks(stack_zyx, voxel_zyx, cfg: InjectionExclusionConfig,
                          crop_origin=(0, 0)):
    """Build separate core and analysis-exclusion masks for one channel."""
    core, analysis, warnings, _diag = build_injection_masks_with_components(
        stack_zyx, voxel_zyx, cfg, crop_origin
    )
    return core, analysis, warnings


def build_generation_suppression_mask(stack_zyx, voxel_zyx,
                                      cfg: InjectionExclusionConfig, crop_origin=(0, 0)):
    """Smallest defensible bright mask used ONLY to suppress generation domination.

    Returns ``(mask, source)``. By default this is the conservative injection
    CORE (NOT the larger analysis-exclusion dilation). A separate
    ``generation_suppression_dilation_um`` may be configured; it must stay well
    below ``analysis_exclusion_dilation_um``. Seed filtering (if configured) has
    already removed unseeded components from the base.
    """
    import numpy as np  # noqa: PLC0415

    base, _warnings, _diag = _injection_base_mask(stack_zyx, voxel_zyx, cfg, crop_origin)
    if not base.any():
        return np.zeros(stack_zyx.shape[1:], dtype=bool), "empty"
    dilation = cfg.generation_suppression_dilation_um
    if dilation is None:
        mask, source = _dilate_um(base, voxel_zyx, cfg.core_dilation_um), "injection_core"
    else:
        mask = _dilate_um(base, voxel_zyx, dilation)
        source = "injection_generation_suppression_mask"
    # A non-injection region must not be suppressed during generation either.
    return _subtract_non_injection(mask, cfg, crop_origin), source


def build_injection_mask(stack_zyx, voxel_zyx, cfg: InjectionExclusionConfig,
                         crop_origin=(0, 0), candidate_xy=None):
    """Compatibility wrapper returning the larger analysis-exclusion mask."""
    _core, analysis, warnings = build_injection_masks(
        stack_zyx, voxel_zyx, cfg, crop_origin=crop_origin
    )
    return analysis, warnings


# --------------------------------------------------------------------------- #
# Stage: background correction
# --------------------------------------------------------------------------- #
def background_correct(stack_zyx, voxel_yx_um: float, background_sigma_um: float):
    """Float background-flattened stack: per-plane large-Gaussian subtract + clip.

    The source TIFF is never altered; this works on a float copy.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    bg_sigma_px = max(1.0, background_sigma_um / voxel_yx_um)
    corrected = np.empty_like(stack_zyx, dtype=np.float32)
    for z in range(stack_zyx.shape[0]):
        plane = stack_zyx[z].astype(np.float32, copy=False)
        corrected[z] = np.clip(plane - ndi.gaussian_filter(plane, sigma=bg_sigma_px), 0.0, None)
    return corrected


# --------------------------------------------------------------------------- #
# Coordinate construction + validation (explicit, never reuse x/y/z)
# --------------------------------------------------------------------------- #
def make_coordinate_fields(z_index, y_local_px, x_local_px, crop_origin, section,
                           first_section, planes_per_section, voxel_zyx, optical_plane):
    vz, vy, vx = voxel_zyx
    oy, ox = crop_origin
    x_global_px = int(x_local_px) + int(ox)
    y_global_px = int(y_local_px) + int(oy)
    section_relative_z_um = float(z_index) * float(vz)
    gp = global_plane(section, first_section, optical_plane, planes_per_section)
    global_z_um = float(gp) * float(vz)
    return {
        "x_local_px": int(x_local_px),
        "y_local_px": int(y_local_px),
        "x_global_px": x_global_px,
        "y_global_px": y_global_px,
        "z_index": int(z_index),
        "optical_plane": int(optical_plane),
        "section": int(section),
        "section_relative_z_um": round(section_relative_z_um, 3),
        "global_z_um": round(global_z_um, 3),
    }


def validate_candidate_coords(rec, planes_per_section, crop_shape):
    """Return (ok, reason). Hard checks; never silently continue on bad coords."""
    import numpy as np  # noqa: PLC0415

    H, W = crop_shape
    zi = rec.get("z_index")
    zrel = rec.get("section_relative_z_um")
    max_zrel = (planes_per_section - 1) * 6.0
    if zi is None or not (0 <= int(zi) <= planes_per_section - 1):
        return False, f"z_index out of range: {zi}"
    if zrel is None or not np.isfinite(zrel) or zrel < 0 or zrel > max_zrel + _EPS:
        return False, f"section_relative_z_um out of [0,{max_zrel}]: {zrel}"
    for key in ("x_local_px", "y_local_px", "x_global_px", "y_global_px",
                "global_z_um", "equivalent_diameter_um"):
        v = rec.get(key)
        if v is None or not np.isfinite(v):
            return False, f"non-finite {key}: {v}"
    if not (0 <= rec["x_local_px"] < W and 0 <= rec["y_local_px"] < H):
        return False, "local pixel outside crop"
    return True, ""


# --------------------------------------------------------------------------- #
# Local contrast + axial (consecutive-plane) support
# --------------------------------------------------------------------------- #
def _valid_intensity_mask(values, tissue, padding_values):
    import numpy as np  # noqa: PLC0415

    valid = np.asarray(tissue, dtype=bool) & np.isfinite(values)
    for padding in padding_values:
        valid &= values != padding
    return valid


def _robust_clipped_std(values) -> float:
    """Standard deviation after clipping to the central 5th-95th percentile."""
    import numpy as np  # noqa: PLC0415

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    low, high = np.percentile(values, [5.0, 95.0])
    clipped = values[(values >= low) & (values <= high)]
    if clipped.size < 2:
        return np.nan
    value = float(np.std(clipped, ddof=1))
    return value if value > _EPS else np.nan


def _noise_from_values(values, method_prefix):
    """MAD -> IQR -> robust clipped SD for one defensible pixel pool."""
    import numpy as np  # noqa: PLC0415

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, "invalid", np.nan, np.nan
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    noise = 1.4826 * mad
    if noise > _EPS:
        return noise, f"{method_prefix}_mad", med, mad
    q25, q75 = np.percentile(values, [25.0, 75.0])
    iqr_noise = float(q75 - q25) / 1.349
    if iqr_noise > _EPS:
        return iqr_noise, f"{method_prefix}_iqr", med, mad
    clipped_std = _robust_clipped_std(values)
    if np.isfinite(clipped_std):
        return clipped_std, f"{method_prefix}_clipped_std", med, mad
    return np.nan, "invalid", med, mad


def _measurement_samples(
    plane,
    tissue_plane,
    cy,
    cx,
    central_r_px,
    inner_r_px,
    outer_r_px,
    padding_values,
):
    """Extract fixed-centre central-disk and annulus values from one plane."""
    import numpy as np  # noqa: PLC0415

    H, W = plane.shape
    r = int(np.ceil(outer_r_px))
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    win = np.asarray(plane[y0:y1, x0:x1], dtype=float)
    twin = tissue_plane[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0 - cy:y1 - cy, x0 - cx:x1 - cx]
    dist = np.sqrt(yy * yy + xx * xx)
    valid = _valid_intensity_mask(win, twin, padding_values)
    central = (dist <= central_r_px) & valid
    annulus = (dist >= inner_r_px) & (dist <= outer_r_px) & valid
    return win[central], win[annulus]


def measure_plane_contrast(
    plane,
    tissue_plane,
    cy,
    cx,
    central_r_px,
    inner_r_px,
    outer_r_px,
    *,
    minimum_background_pixels=20,
    padding_values=(0.0,),
    adjacent_background_values=None,
):
    """Measure a fixed-XY central disk against a local annulus.

    Noise fallback order is local MAD, local IQR, local robustly-clipped SD,
    then the same hierarchy on pooled annuli from adjacent optical planes.
    No whole-plane or magic-number substitution is used.
    """
    import numpy as np  # noqa: PLC0415

    central_values, background_values = _measurement_samples(
        plane, tissue_plane, cy, cx, central_r_px, inner_r_px,
        outer_r_px, padding_values,
    )
    background_count = int(background_values.size)

    if central_values.size == 0:
        return {
            "central_signal": np.nan,
            "background_median": np.nan,
            "background_mad": np.nan,
            "background_noise": np.nan,
            "contrast": np.nan,
            "measurement_valid": False,
            "background_pixel_count": background_count,
            "background_noise_method": "invalid_no_central_pixels",
            "measurement_reason": "invalid_no_central_pixels",
        }

    central_signal = float(np.median(central_values))
    noise = bg_median = bg_mad = np.nan
    method = "invalid"
    if background_count >= int(minimum_background_pixels):
        noise, method, bg_median, bg_mad = _noise_from_values(
            background_values, "annulus"
        )

    if not np.isfinite(noise) and adjacent_background_values is not None:
        adjacent = np.asarray(adjacent_background_values, dtype=float)
        pooled = np.concatenate([background_values, adjacent])
        if pooled.size >= int(minimum_background_pixels):
            noise, method, bg_median, bg_mad = _noise_from_values(
                pooled, "adjacent_pooled_annulus"
            )
            background_count = int(pooled.size)

    if not np.isfinite(noise):
        method = (
            "invalid_insufficient_annulus_pixels"
            if background_count < int(minimum_background_pixels)
            else "invalid_local_noise"
        )

    measurement_valid = bool(
        np.isfinite(central_signal) and np.isfinite(bg_median)
        and np.isfinite(noise) and noise > _EPS
    )
    contrast = float((central_signal - bg_median) / noise) if measurement_valid else np.nan
    return {
        "central_signal": central_signal,
        "background_median": float(bg_median) if np.isfinite(bg_median) else np.nan,
        "background_mad": float(bg_mad) if np.isfinite(bg_mad) else np.nan,
        "background_noise": float(noise) if np.isfinite(noise) else np.nan,
        "contrast": contrast,
        "measurement_valid": measurement_valid,
        "background_pixel_count": background_count,
        "background_noise_method": method,
        "measurement_reason": "" if measurement_valid else "invalid_local_noise",
    }


def measure_fixed_xy_profile(stack_zyx, tissue_plane, cy, cx, params, voxel_y_um=1.0):
    """Measure all planes at one immutable candidate XY and derive Z support."""
    import numpy as np  # noqa: PLC0415

    vy = float(voxel_y_um)
    central_r = max(1.0, params.central_region_radius_um / vy)
    inner_r = max(central_r + 1.0, params.background_annulus_inner_um / vy)
    outer_r = max(inner_r + 1.0, params.background_annulus_outer_um / vy)
    annuli = [
        _measurement_samples(
            stack_zyx[z], tissue_plane, cy, cx, central_r, inner_r,
            outer_r, params.padding_values,
        )[1]
        for z in range(stack_zyx.shape[0])
    ]
    measurements = []
    for z in range(stack_zyx.shape[0]):
        adjacent = [
            annuli[adjacent_z]
            for adjacent_z in (z - 1, z + 1)
            if 0 <= adjacent_z < stack_zyx.shape[0]
        ]
        adjacent_values = np.concatenate(adjacent) if adjacent else np.asarray([])
        measurements.append(measure_plane_contrast(
            stack_zyx[z], tissue_plane, cy, cx, central_r, inner_r, outer_r,
            minimum_background_pixels=params.minimum_background_pixels,
            padding_values=params.padding_values,
            adjacent_background_values=adjacent_values,
        ))
    profile = np.asarray([m["contrast"] for m in measurements], dtype=float)
    finite = np.isfinite(profile)
    peak_z = int(np.nanargmax(profile)) if finite.any() else 0
    present = finite & (profile >= float(params.z_support_min_contrast))
    if present[peak_z]:
        support, _ = _consecutive_support(present, peak_z)
    else:
        support = []
    return measurements, profile, peak_z, support


def local_robust_z(plane_corr, tissue_plane, cy, cx, inner_r_px, outer_r_px, mad_floor=None):
    """Compatibility wrapper returning NaN for invalid contrast measurements."""
    result = measure_plane_contrast(
        plane_corr, tissue_plane, cy, cx, max(1.0, inner_r_px / 2.0),
        inner_r_px, outer_r_px, minimum_background_pixels=5, padding_values=(),
    )
    return (
        result["central_signal"],
        result["background_median"],
        result["background_mad"],
        result["contrast"],
    )


def _consecutive_support(present, peak_plane):
    """Length and indices of the consecutive run of ``present`` planes that
    contains ``peak_plane`` (boolean 1D array)."""
    import numpy as np  # noqa: PLC0415

    if not present[peak_plane]:
        return [int(peak_plane)], 1
    z0 = peak_plane
    while z0 - 1 >= 0 and present[z0 - 1]:
        z0 -= 1
    z1 = peak_plane
    while z1 + 1 < len(present) and present[z1 + 1]:
        z1 += 1
    idx = list(range(z0, z1 + 1))
    return [int(i) for i in idx], len(idx)


# --------------------------------------------------------------------------- #
# Backend 1: pilot_log3d (rule-based 3D connected components)
# --------------------------------------------------------------------------- #
def _multiscale_log(corrected, voxel_yx_um, s_min_um, s_max_um, n_scales=3):
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    s_min = max(0.6, s_min_um / voxel_yx_um)
    s_max = max(s_min, s_max_um / voxel_yx_um)
    sigmas = np.unique(np.linspace(s_min, s_max, max(1, n_scales)))
    resp = np.zeros_like(corrected, dtype=np.float32)
    for z in range(corrected.shape[0]):
        best = None
        for s in sigmas:
            r = -(s ** 2) * ndi.gaussian_laplace(corrected[z], sigma=s)
            best = r if best is None else np.maximum(best, r)
        resp[z] = best
    return resp


def _xy_elongation(proj_xy) -> float:
    import numpy as np  # noqa: PLC0415

    ys, xs = np.nonzero(proj_xy)
    if ys.size < 2:
        return 1.0
    cov = np.cov(np.stack([ys.astype(float) - ys.mean(), xs.astype(float) - xs.mean()]))
    if not np.all(np.isfinite(cov)):
        return 1.0
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    return float(np.sqrt(eig[1] / max(eig[0], _EPS)))


def _xy_bbox_overlap(a, b) -> bool:
    return a[2] < b[3] and b[2] < a[3] and a[0] < b[1] and b[0] < a[1]


def _union_find_merge(peaks_um, bboxes, merge_xy_um, merge_z_um):
    import numpy as np  # noqa: PLC0415

    n = len(peaks_um)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pts = np.asarray(peaks_um, dtype=float)

    def maybe(i, j):
        dxy = float(np.hypot(pts[i, 1] - pts[j, 1], pts[i, 2] - pts[j, 2]))
        dz = abs(pts[i, 0] - pts[j, 0])
        if dxy <= merge_xy_um and dz <= merge_z_um and _xy_bbox_overlap(bboxes[i], bboxes[j]):
            union(i, j)

    radius = max(merge_xy_um, merge_z_um)
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        tree = cKDTree(pts)
        for i, j in tree.query_pairs(radius):
            maybe(i, j)
    except Exception:  # pragma: no cover
        for i in range(n):
            for j in range(i + 1, n):
                maybe(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def pilot_log3d_objects(corrected, raw, tissue3d, params, voxel_zyx):
    """Generate raw object metrics from the rule-based 3D detector."""
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    vz, vy, vx = voxel_zyx
    voxel_um3 = vz * vy * vx
    Z, H, W = corrected.shape

    tissue_vals = corrected[tissue3d]
    if tissue_vals.size == 0:
        return []
    med = float(np.median(tissue_vals))
    mad = _mad(tissue_vals)
    fg_thr = med + params.foreground_sigma_factor * 1.4826 * mad + _EPS
    if mad <= _EPS:
        fg_thr = max(fg_thr, float(np.percentile(tissue_vals, 99.0)))
    fg = (corrected > fg_thr) & tissue3d
    if not fg.any():
        return []

    min_cell_vox = (np.pi / 6.0) * (params.min_diameter_um ** 3) / max(voxel_um3, _EPS)
    min_object_vox = max(2, int(0.1 * min_cell_vox))

    log_resp = _multiscale_log(corrected, vy, params.log_sigma_min_um, params.log_sigma_max_um)
    labels, n_lab = ndi.label(fg, structure=np.ones((3, 3, 3), dtype=int))
    if n_lab == 0:
        return []
    slices = ndi.find_objects(labels)
    sizes = np.bincount(labels.ravel(), minlength=n_lab + 1)

    comp = {}
    for lab in range(1, n_lab + 1):
        sl = slices[lab - 1]
        if sl is None or sizes[lab] < min_object_vox:
            continue
        submask = labels[sl] == lab
        sublog = np.where(submask, log_resp[sl], -np.inf)
        lz, ly, lx = np.unravel_index(int(np.argmax(sublog)), sublog.shape)
        peak = (lz + sl[0].start, ly + sl[1].start, lx + sl[2].start)
        comp[lab] = {"sl": sl, "peak": peak,
                     "peak_um": (peak[0] * vz, peak[1] * vy, peak[2] * vx)}
    if not comp:
        return []

    lab_ids = list(comp)
    peaks_um = [comp[l]["peak_um"] for l in lab_ids]
    bboxes = [(comp[l]["sl"][1].start, comp[l]["sl"][1].stop,
               comp[l]["sl"][2].start, comp[l]["sl"][2].stop) for l in lab_ids]
    groups = _union_find_merge(peaks_um, bboxes, params.merge_distance_xy_um,
                               params.merge_distance_z_um)

    margin_px = params.crop_boundary_margin_um / vy
    objects = []
    for grp in groups:
        glabels = [lab_ids[i] for i in grp]
        zs = [comp[l]["sl"][0].start for l in glabels] + [comp[l]["sl"][0].stop for l in glabels]
        ys = [comp[l]["sl"][1].start for l in glabels] + [comp[l]["sl"][1].stop for l in glabels]
        xs = [comp[l]["sl"][2].start for l in glabels] + [comp[l]["sl"][2].stop for l in glabels]
        zmin, zmax = min(zs), max(zs)
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)
        sub = np.isin(labels[zmin:zmax, ymin:ymax, xmin:xmax], glabels)
        n_vox = int(sub.sum())
        if n_vox == 0:
            continue

        best_lab = max(glabels, key=lambda l: log_resp[comp[l]["peak"]])
        pz, py, px = comp[best_lab]["peak"]

        volume_um3 = n_vox * voxel_um3
        equiv_diam_um = (6.0 * volume_um3 / np.pi) ** (1.0 / 3.0)
        proj_xy = sub.any(axis=0)
        area_xy = int(proj_xy.sum())
        xy_diam_um = 2.0 * np.sqrt(max(area_xy, 1) / np.pi) * vy
        elongation = _xy_elongation(proj_xy)

        # Consecutive-plane support + per-plane XY centroid shift (within bbox).
        per_plane = sub.reshape(sub.shape[0], -1).sum(axis=1)
        present = per_plane >= max(2, int(0.1 * per_plane.max()))
        if not present.any():
            present = per_plane > 0
        peak_local_z = pz - zmin
        z_run_local, n_consec = _consecutive_support(present, peak_local_z)
        z_indices = [zi + zmin for zi in z_run_local]
        z_extent_um = (max(z_indices) - min(z_indices) + 1) * vz

        centroids = []
        for zl in z_run_local:
            yy, xx = np.nonzero(sub[zl])
            if yy.size:
                centroids.append((yy.mean() + ymin, xx.mean() + xmin))
        xy_shift_um = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                d = np.hypot((centroids[i][0] - centroids[j][0]) * vy,
                             (centroids[i][1] - centroids[j][1]) * vx)
                xy_shift_um = max(xy_shift_um, float(d))

        peak_intensity = float(raw[pz, py, px])
        mean_intensity = float(raw[zmin:zmax, ymin:ymax, xmin:xmax][sub].mean())
        touches = bool(ymin <= margin_px or xmin <= margin_px
                       or ymax >= H - margin_px or xmax >= W - margin_px)

        objects.append({
            "z_index": int(pz), "y_local_px": int(py), "x_local_px": int(px),
            "peak_plane": int(pz), "z_indices": z_indices,
            "n_consecutive_planes": int(n_consec), "xy_centroid_shift_um": round(xy_shift_um, 3),
            "equivalent_diameter_um": round(float(equiv_diam_um), 3),
            "xy_diameter_um": round(float(xy_diam_um), 3),
            "z_extent_um": round(float(z_extent_um), 3),
            "volume_um3": round(float(volume_um3), 3),
            "elongation": round(float(elongation), 3),
            "peak_intensity": round(peak_intensity, 2),
            "mean_intensity": round(mean_intensity, 2),
            "touches_crop_boundary": touches,
            "is_artifact": False,
        })
    return objects


# --------------------------------------------------------------------------- #
# Backend 2: cellfinder adapter
# --------------------------------------------------------------------------- #
def cellfinder_objects(raw, corrected, voxel_zyx, params, *, detect_main=None,
                       cellfinder_cfg=None, detection_array=None):
    """Generate raw object metrics from Cellfinder candidate detection.

    Cellfinder DETECTS on ``detection_array`` (the raw stack, or an
    injection-suppressed copy); per-candidate morphology is always measured on
    the unmodified ``raw``/``corrected`` stacks so suppression never contaminates
    measurements.
    """
    from .cellfinder_adapter import run_cellfinder_detection  # noqa: PLC0415

    return run_cellfinder_detection(
        raw, corrected, voxel_zyx, params, detect_main=detect_main,
        cellfinder_cfg=cellfinder_cfg, detection_array=detection_array,
    )


# --------------------------------------------------------------------------- #
# Two-pass candidate-generation provenance + 3D merge
# --------------------------------------------------------------------------- #
def _tag_object_source(obj, *, on_raw, on_suppressed, mask_used, mask_source):
    obj["detected_on_raw_stack"] = bool(on_raw)
    obj["detected_on_injection_suppressed_stack"] = bool(on_suppressed)
    obj["generation_suppression_used"] = bool(mask_used)
    obj["generation_suppression_mask_used"] = bool(mask_used)
    obj["generation_suppression_mask_source"] = mask_source
    if on_raw and on_suppressed:
        obj["candidate_generation_source"] = "both"
    elif on_suppressed:
        obj["candidate_generation_source"] = "injection_suppressed_stack"
    else:
        obj["candidate_generation_source"] = "raw_stack"
    return obj


def _merge_two_pass_objects(objects_raw, objects_suppressed, voxel_zyx,
                            merge_xy_um, merge_z_um):
    """Union of the two passes, deduplicated in 3D (never merge on a shared plane).

    ``objects_raw`` are kept in full. Each suppressed object that lies within the
    XY and Z tolerances of an existing object updates that object's provenance to
    ``both``; otherwise it is appended as a suppressed-only candidate.
    """
    import numpy as np  # noqa: PLC0415

    vz, vy, vx = voxel_zyx
    merged = list(objects_raw)

    def coords_um(o):
        return (
            float(o["z_index"]) * vz,
            float(o["y_local_px"]) * vy,
            float(o["x_local_px"]) * vx,
        )

    for s in objects_suppressed:
        sz, sy, sx = coords_um(s)
        duplicate = None
        for m in merged:
            mz, my, mx = coords_um(m)
            dxy = float(np.hypot(my - sy, mx - sx))
            dz = abs(mz - sz)
            # Require BOTH tolerances -> a shared plane alone never merges.
            if dxy <= merge_xy_um and dz <= merge_z_um:
                duplicate = m
                break
        if duplicate is not None:
            _tag_object_source(
                duplicate,
                on_raw=duplicate.get("detected_on_raw_stack", False),
                on_suppressed=True,
                mask_used=duplicate.get("generation_suppression_mask_used", True),
                mask_source=duplicate.get("generation_suppression_mask_source", ""),
            )
        else:
            merged.append(s)
    return merged


# --------------------------------------------------------------------------- #
# Shared finalisation: coords, masks, classification, NMS, validation
# --------------------------------------------------------------------------- #
@dataclass
class StackResult:
    candidates: list = field(default_factory=list)
    tissue_mask: object = None
    injection_mask: object = None
    injection_core_mask: object = None
    injection_analysis_exclusion_mask: object = None
    generation_suppression_mask: object = None
    generation_suppression_mask_source: str = "none"
    injection_components: dict = field(default_factory=dict)
    mask_diagnostics: dict = field(default_factory=dict)
    projection: object = None
    suppressed_projection: object = None
    corrected: object = None
    shape: tuple = (0, 0, 0)
    n_invalid: int = 0
    warnings: list = field(default_factory=list)
    generation_diagnostics: dict = field(default_factory=dict)


def _preliminary_interpretation(rec, params, has_tissue):
    """Assign a review-sampling category without deciding biological identity."""
    inside_tissue = rec["inside_tissue"]
    if rec.get("invalid_coordinate"):
        return STATUS_MANUAL_REVIEW, REASON_INVALID_COORD
    # An out-of-range Cellfinder z goes to review even if the fixed-XY peak is fine.
    if not rec.get("original_cellfinder_z_valid", True):
        return STATUS_MANUAL_REVIEW, REASON_CF_Z_INVALID
    if not rec.get("measurement_valid", False):
        return STATUS_MANUAL_REVIEW, "invalid_local_noise"
    if rec.get("is_artifact"):
        return STATUS_ARTIFACT, REASON_ARTIFACT
    if has_tissue and not inside_tissue:
        return STATUS_RULE_FAILED, REASON_OUTSIDE_TISSUE
    if params.exclude_crop_boundary and rec["touches_crop_boundary"]:
        return STATUS_RULE_FAILED, REASON_CROP_BOUNDARY
    n_consec = rec["n_consecutive_planes"]
    if n_consec > params.max_consecutive_planes:
        return STATUS_MANUAL_REVIEW, REASON_MANY_PLANES_REVIEW
    if rec["equivalent_diameter_um"] < params.min_diameter_um or \
            rec["xy_diameter_um"] < params.min_diameter_um:
        return STATUS_RULE_FAILED, REASON_TOO_SMALL
    if rec["equivalent_diameter_um"] > params.max_diameter_um:
        return STATUS_RULE_FAILED, REASON_TOO_LARGE
    if rec["elongation"] > params.max_elongation:
        return STATUS_RULE_FAILED, REASON_ELONGATED
    if rec["xy_centroid_shift_um"] > params.max_xy_shift_um:
        return STATUS_RULE_FAILED, REASON_XY_JUMP
    if n_consec < params.min_consecutive_planes:
        # A very strong single-plane object passes outright: these were
        # consistently confirmed as cells in review, so only the weaker band
        # [single_plane_review_min_z, single_plane_pass_min_z) still needs eyes.
        if rec["local_robust_z"] >= params.single_plane_pass_min_z:
            return STATUS_RULE_PASSED, ""
        if params.single_plane_manual_review and \
                rec["local_robust_z"] >= params.single_plane_review_min_z:
            return STATUS_MANUAL_REVIEW, REASON_SINGLE_PLANE
        return STATUS_RULE_FAILED, REASON_SINGLE_PLANE
    if rec["local_robust_z"] < params.min_local_robust_z:
        return STATUS_RULE_FAILED, REASON_LOW_CONTRAST
    return STATUS_RULE_PASSED, ""


def _injection_mask_provenance(cfg: InjectionExclusionConfig) -> tuple[str, bool]:
    manual = bool(cfg.manual_rectangles or cfg.manual_polygons)
    if not cfg.enabled:
        return "disabled", False
    if cfg.automatic and manual:
        return "automatic_plus_manual_geometry", bool(cfg.mask_validated)
    if cfg.automatic:
        return "automatic", bool(cfg.mask_validated)
    if manual:
        return "manual_geometry", True
    return "empty", False


def _assign_current_status(rec):
    """Combine independent flags into one backward-compatible display status."""
    if rec.get("invalid_coordinate"):
        return STATUS_ARTIFACT, REASON_INVALID_COORD
    # Surface a bad Cellfinder z for review before any injection-mask labelling.
    if not rec.get("original_cellfinder_z_valid", True):
        return STATUS_MANUAL_REVIEW, REASON_CF_Z_INVALID
    if not rec.get("measurement_valid", False):
        return STATUS_INVALID_MEASUREMENT, "invalid_local_noise"
    if rec.get("is_artifact"):
        return STATUS_ARTIFACT, REASON_ARTIFACT
    if rec.get("inside_injection_analysis_exclusion"):
        source = rec.get("injection_mask_source")
        manually_defined = source == "manual_geometry"
        validated = bool(rec.get("injection_mask_validated"))
        qc_failed = bool(rec.get("injection_mask_qc_failed"))
        if manually_defined or (validated and not qc_failed):
            return STATUS_INJECTION, REASON_INJECTION
        return STATUS_SUSPECT_INJECTION, REASON_SUSPECT_INJECTION
    return rec["preliminary_sampling_category"], rec["preliminary_rule_reason"]


def _apply_nms(candidates, min_separation_um, voxel_zyx):
    import numpy as np  # noqa: PLC0415

    vz, vy, vx = voxel_zyx
    survivors = [
        c for c in candidates
        if c["preliminary_sampling_category"] == STATUS_RULE_PASSED
    ]
    if len(survivors) < 2:
        return
    order = sorted(survivors, key=lambda c: -c["local_robust_z"])
    pts = np.array([[c["z_index"] * vz, c["y_global_px"] * vy, c["x_global_px"] * vx]
                    for c in order])
    suppressed = np.zeros(len(order), dtype=bool)
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        tree = cKDTree(pts)
        for i in range(len(order)):
            if suppressed[i]:
                continue
            for j in tree.query_ball_point(pts[i], min_separation_um):
                if j != i and not suppressed[j]:
                    suppressed[j] = True
                    order[j].update(
                        preliminary_sampling_category=STATUS_RULE_FAILED,
                        preliminary_rule_reason=REASON_DUPLICATE,
                    )
    except Exception:  # pragma: no cover
        for i in range(len(order)):
            if suppressed[i]:
                continue
            for j in range(len(order)):
                if j != i and not suppressed[j] and \
                        float(np.linalg.norm(pts[i] - pts[j])) < min_separation_um:
                    suppressed[j] = True
                    order[j].update(
                        preliminary_sampling_category=STATUS_RULE_FAILED,
                        preliminary_rule_reason=REASON_DUPLICATE,
                    )


def detect_candidates_in_stack(
    stack_zyx,
    params: DetectionParams,
    voxel_zyx,
    *,
    channel: str = "",
    section: int = 0,
    first_section: int | None = None,
    planes_per_section: int = 7,
    plane_numbers: Sequence[int] | None = None,
    crop_origin=(0, 0),
    shared_tissue_mask=None,
    injection_cfg: InjectionExclusionConfig | None = None,
    backend: str | None = None,
    cellfinder_detect_main=None,
    cellfinder_cfg=None,
    timer=None,
) -> StackResult:
    """Backend-agnostic candidate pipeline on an in-memory (Z, Y, X) stack.

    When ``injection_cfg.generation_suppression_enabled`` is set and the backend
    is Cellfinder, a SECOND detection pass runs on an in-memory copy whose
    conservative injection core is replaced by a smooth background estimate (see
    ``injection_suppression``); the two passes are merged with full provenance.

    ``timer`` (optional) records per-stage durations; anything with a
    ``.stage(name)`` context manager works (see ``timing.StageTimer``).
    """
    from contextlib import nullcontext  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    def _t(name):
        return timer.stage(name) if timer is not None else nullcontext()

    raw = np.ascontiguousarray(stack_zyx, dtype=np.float32)
    Z, H, W = raw.shape
    vz, vy, vx = voxel_zyx
    if first_section is None:
        first_section = section
    if plane_numbers is None:
        plane_numbers = list(range(1, Z + 1))
    backend = backend or params.backend
    injection_cfg = injection_cfg if injection_cfg is not None else params.injection
    injection_mask_source, injection_mask_validated = _injection_mask_provenance(
        injection_cfg
    )

    result = StackResult(shape=(Z, H, W))
    result.projection = raw.max(axis=0)

    with _t("mask_processing"):
        # Tissue mask: shared if provided, else build from this stack, else all-tissue.
        if shared_tissue_mask is not None:
            tissue = shared_tissue_mask
        elif params.tissue.enabled:
            tissue = build_shared_tissue_mask([raw], voxel_zyx, params.tissue)
        else:
            tissue = None
        has_tissue = tissue is not None
        result.tissue_mask = tissue
        tissue_bool = tissue if has_tissue else np.ones((H, W), dtype=bool)
        tissue3d = np.broadcast_to(tissue_bool, raw.shape)

        corrected = background_correct(raw, vy, params.background_sigma_um)
        result.corrected = corrected

        injection_core, injection, inj_warn, inj_components = (
            build_injection_masks_with_components(
                raw, voxel_zyx, injection_cfg, crop_origin=crop_origin
            )
        )
    result.injection_core_mask = injection_core
    result.injection_analysis_exclusion_mask = injection
    result.injection_mask = injection  # compatibility alias
    result.injection_components = inj_components
    result.warnings.extend(f"[{channel}] {w}" for w in inj_warn)
    # Log each automatic component's area and whether the seed filter kept it.
    if inj_components.get("seed_filter_applied"):
        for comp in inj_components.get("components", []):
            LOG.warning(
                "[%s] injection component label=%d area=%d px at (x=%d,y=%d): %s",
                channel, comp["label"], comp["area_px"],
                comp["centroid_x_local"], comp["centroid_y_local"],
                "KEPT (seeded)" if comp["kept"] else "REMOVED (no seed)",
            )
    if has_tissue and injection.any() and np.array_equal(injection, tissue_bool):
        result.warnings.append(f"[{channel}] injection mask is identical to the tissue mask.")

    cf_cfg = cellfinder_cfg if cellfinder_cfg is not None else params.cellfinder

    # Generate raw objects via the chosen backend (Pass A: original raw stack).
    with _t("cellfinder"):
        if backend == "cellfinder_candidates":
            objects = cellfinder_objects(
                raw, corrected, voxel_zyx, params,
                detect_main=cellfinder_detect_main, cellfinder_cfg=cf_cfg,
                detection_array=raw,
            )
        elif backend == "pilot_log3d":
            objects = pilot_log3d_objects(corrected, raw, tissue3d, params, voxel_zyx)
        else:
            raise ValueError(f"Unknown detection backend: {backend!r}")

    # Default provenance (single raw pass).
    suppression_used = False
    suppression_source = "none"
    for obj in objects:
        _tag_object_source(
            obj, on_raw=True, on_suppressed=False,
            mask_used=False, mask_source="none",
        )

    # Pass B: injection-suppressed detection pass (Cellfinder only).
    two_pass = bool(
        backend == "cellfinder_candidates"
        and getattr(injection_cfg, "generation_suppression_enabled", False)
    )
    if two_pass:
        from .injection_suppression import suppress_injection_core  # noqa: PLC0415

        suppression_mask, suppression_source = build_generation_suppression_mask(
            raw, voxel_zyx, injection_cfg, crop_origin=crop_origin
        )
        result.generation_suppression_mask = suppression_mask
        result.generation_suppression_mask_source = suppression_source
        if suppression_mask.any():
            suppression_used = True
            suppressed_stack = suppress_injection_core(
                raw, suppression_mask, voxel_zyx, tissue_mask=tissue_bool,
            )
            result.suppressed_projection = suppressed_stack.max(axis=0)
            suppressed_corrected = background_correct(
                suppressed_stack, vy, params.background_sigma_um
            )
            with _t("cellfinder"):
                suppressed_objects = cellfinder_objects(
                    raw, suppressed_corrected, voxel_zyx, params,
                    detect_main=cellfinder_detect_main, cellfinder_cfg=cf_cfg,
                    detection_array=suppressed_stack,
                )
            # Keep raw provenance on the originals; tag suppressed pass.
            for obj in objects:
                obj["generation_suppression_mask_used"] = True
                obj["generation_suppression_mask_source"] = suppression_source
            # Only suppressed candidates OUTSIDE the analysis-exclusion region are
            # retained; those inside it would re-introduce injection domination.
            kept_suppressed = []
            for obj in suppressed_objects:
                yy = max(0, min(H - 1, int(obj["y_local_px"])))
                xx = max(0, min(W - 1, int(obj["x_local_px"])))
                if injection.any() and injection[yy, xx]:
                    continue
                _tag_object_source(
                    obj, on_raw=False, on_suppressed=True,
                    mask_used=True, mask_source=suppression_source,
                )
                kept_suppressed.append(obj)
            objects = _merge_two_pass_objects(
                objects, kept_suppressed, voxel_zyx,
                params.merge_distance_xy_um, params.merge_distance_z_um,
            )
        else:
            result.warnings.append(
                f"[{channel}] generation-suppression requested but the bright "
                "injection core is empty; only the raw pass ran."
            )

    result.generation_diagnostics = {
        "two_pass_requested": two_pass,
        "generation_suppression_used": suppression_used,
        "generation_suppression_mask_source": suppression_source,
        "raw_pass_objects": sum(1 for o in objects if o.get("detected_on_raw_stack")),
        "suppressed_pass_objects": sum(
            1 for o in objects if o.get("detected_on_injection_suppressed_stack")
        ),
    }

    if timer is not None:
        timer.start("candidate_measurements")
    margin_px = params.crop_boundary_margin_um / vy
    candidates = []
    for i, obj in enumerate(objects):
        seed_z = int(obj["z_index"])
        py = int(obj["y_local_px"])
        px = int(obj["x_local_px"])
        seed_z = max(0, min(Z - 1, seed_z))
        py = max(0, min(H - 1, py))
        px = max(0, min(W - 1, px))

        plane_measurements, contrast_profile, pz, support = measure_fixed_xy_profile(
            raw, tissue_bool, py, px, params, voxel_y_um=vy
        )
        optical_plane = int(plane_numbers[pz])

        rec = make_coordinate_fields(pz, py, px, crop_origin, section, first_section,
                                     planes_per_section, voxel_zyx, optical_plane)
        rec["candidate_exists"] = True
        rec["candidate_generation_index"] = i
        rec["channel"] = channel
        rec["backend"] = backend
        rec["candidate_generation_source"] = obj.get(
            "candidate_generation_source", "raw_stack"
        )
        rec["detected_on_raw_stack"] = bool(obj.get("detected_on_raw_stack", True))
        rec["detected_on_injection_suppressed_stack"] = bool(
            obj.get("detected_on_injection_suppressed_stack", False)
        )
        rec["generation_suppression_used"] = bool(
            obj.get("generation_suppression_used",
                    obj.get("generation_suppression_mask_used", False))
        )
        rec["generation_suppression_mask_used"] = bool(
            obj.get("generation_suppression_mask_used", False)
        )
        rec["generation_suppression_mask_source"] = obj.get(
            "generation_suppression_mask_source", "none"
        )

        peak_measurement = plane_measurements[pz]
        bg_med = peak_measurement["background_median"]
        bg_mad = peak_measurement["background_mad"]
        bg_noise = peak_measurement["background_noise"]
        robust_z = peak_measurement["contrast"]
        support_text = ";".join(str(z) for z in support)
        support_count = len(support)

        rec.update({
            "original_cellfinder_x_local_px": obj.get("cellfinder_x", px),
            "original_cellfinder_y_local_px": obj.get("cellfinder_y", py),
            "original_cellfinder_z_index": obj.get("cellfinder_z", seed_z),
            "cellfinder_z_index": obj.get("cellfinder_z", seed_z),
            "cellfinder_returned_z_raw": obj.get("cellfinder_returned_z_raw", ""),
            "cellfinder_z_mapping_method": obj.get(
                "cellfinder_z_mapping_method", "rule_based_no_cellfinder_z"
            ),
            "mapped_stack_z_index": obj.get("mapped_stack_z_index", seed_z),
            "original_cellfinder_z_valid": bool(obj.get("original_cellfinder_z_valid", True)),
            "fixed_xy_center_x_local_px": px,
            "fixed_xy_center_y_local_px": py,
            "fixed_xy_peak_z_index": int(pz),
            "fixed_xy_support_z_indices": support_text,
            "fixed_xy_n_consecutive_planes": support_count,
            "fixed_xy_z_extent_um": float(support_count * vz),
            "peak_plane": int(pz),
            "z_indices": support_text,
            "n_consecutive_planes": support_count,
            "support_z_indices": support_text,
            "support_plane_count": support_count,
            "support_start": min(support) if support else "",
            "support_end": max(support) if support else "",
            "peak_z_index": int(pz),
            "xy_centroid_shift_um": float(obj["xy_centroid_shift_um"]),
            "peak_intensity": float(peak_measurement["central_signal"]),
            "mean_intensity": float(obj["mean_intensity"]),
            "local_background_median": round(float(bg_med), 3) if np.isfinite(bg_med) else np.nan,
            "local_background_mad": round(float(bg_mad), 3) if np.isfinite(bg_mad) else np.nan,
            "local_background_noise": (
                round(float(bg_noise), 3) if np.isfinite(bg_noise) else np.nan
            ),
            "local_robust_z": round(float(robust_z), 3) if np.isfinite(robust_z) else np.nan,
            "local_contrast_score": (
                round(float(robust_z), 3) if np.isfinite(robust_z) else np.nan
            ),
            "measurement_valid": bool(peak_measurement["measurement_valid"]),
            "measurement_reason": peak_measurement["measurement_reason"],
            "background_pixel_count": int(peak_measurement["background_pixel_count"]),
            "background_noise_method": peak_measurement["background_noise_method"],
            "equivalent_diameter_um": float(obj["equivalent_diameter_um"]),
            "xy_diameter_um": float(obj["xy_diameter_um"]),
            "z_extent_um": float(support_count * vz),
            "volume_um3": float(obj["volume_um3"]),
            "elongation": float(obj["elongation"]),
            "is_artifact": bool(obj.get("is_artifact", False)),
            "cellfinder_x": obj.get("cellfinder_x", ""),
            "cellfinder_y": obj.get("cellfinder_y", ""),
            "cellfinder_z": obj.get("cellfinder_z", seed_z),
            "cellfinder_type": obj.get("cellfinder_type", ""),
            "cellfinder_type_name": obj.get("cellfinder_type_name", ""),
            "manual_label": "",
            "classifier_probability": "",
            "classifier_model": "",
            "classifier_version": "",
            "model_validation_passed": False,
            "final_decision": "",
            "source_crop": (
                f"{int(crop_origin[1])}:{int(crop_origin[1] + W)},"
                f"{int(crop_origin[0])}:{int(crop_origin[0] + H)}"
            ),
        })
        for z in range(7):
            value = contrast_profile[z] if z < len(contrast_profile) else np.nan
            rec[f"plane_{z}_contrast"] = round(float(value), 3) if np.isfinite(value) else np.nan

        rec["inside_tissue"] = bool(tissue_bool[py, px])
        rec["inside_injection_site"] = bool(injection[py, px])
        rec["inside_injection_core"] = bool(injection_core[py, px])
        rec["inside_injection_analysis_exclusion"] = bool(injection[py, px])
        rec["injection_mask_source"] = injection_mask_source
        rec["injection_mask_validated"] = injection_mask_validated
        rec["injection_mask_qc_failed"] = False
        rec["injection_assignment_source"] = "none"
        rec["touches_crop_boundary"] = bool(
            obj.get("touches_crop_boundary")
            or py <= margin_px or px <= margin_px
            or py >= H - margin_px or px >= W - margin_px
        )

        ok, why = validate_candidate_coords(rec, planes_per_section, (H, W))
        rec["invalid_coordinate"] = not ok
        if not ok:
            LOG.warning("Invalid candidate coordinate rejected (%s): %s", channel, why)

        denom = 2.0 * max(params.min_local_robust_z, _EPS)
        rec["classification_score"] = (
            round(min(1.0, max(0.0, float(robust_z)) / denom), 4)
            if np.isfinite(robust_z) else np.nan
        )
        sampling_category, preliminary_reason = _preliminary_interpretation(
            rec, params, has_tissue
        )
        rec["preliminary_sampling_category"] = sampling_category
        rec["preliminary_rule_reason"] = preliminary_reason
        rec["current_status"] = ""
        rec["included_in_count"] = False
        rec["rejection_reason"] = preliminary_reason
        candidates.append(rec)

    _apply_nms(candidates, params.min_separation_um, voxel_zyx)

    tissue_area = int(np.count_nonzero(tissue_bool))
    mask_area = int(np.count_nonzero(injection & tissue_bool))
    inside_count = sum(1 for c in candidates if c["inside_injection_site"])
    total_count = len(candidates)
    mask_fraction = (mask_area / tissue_area) if tissue_area else 0.0
    candidate_fraction = (inside_count / total_count) if total_count else 0.0
    mask_qc_failed = bool(
        mask_fraction > float(injection_cfg.maximum_mask_fraction_of_tissue)
        or candidate_fraction > float(
            injection_cfg.maximum_candidate_fraction_inside_mask
        )
    )
    result.mask_diagnostics = {
        "mask_area_px": mask_area,
        "tissue_area_px": tissue_area,
        "mask_fraction_of_tissue": mask_fraction,
        "candidates_inside_mask": inside_count,
        "candidates_outside_mask": total_count - inside_count,
        "candidate_fraction_inside_mask": candidate_fraction,
        "injection_mask_source": injection_mask_source,
        "injection_mask_validated": injection_mask_validated,
        "injection_mask_qc_failed": mask_qc_failed,
    }
    if mask_fraction > float(injection_cfg.maximum_mask_fraction_of_tissue):
        result.warnings.append(
            f"[{channel}] injection analysis mask covers "
            f"{mask_fraction:.1%} of tissue "
            f"(>{injection_cfg.maximum_mask_fraction_of_tissue:.0%})."
        )
    if candidate_fraction > float(
        injection_cfg.maximum_candidate_fraction_inside_mask
    ):
        result.warnings.append(
            f"[{channel}] {candidate_fraction:.1%} of candidates are inside the "
            "injection analysis mask "
            f"(>{injection_cfg.maximum_candidate_fraction_inside_mask:.0%})."
        )

    for i, c in enumerate(candidates):
        c["injection_mask_qc_failed"] = mask_qc_failed
        status, reason = _assign_current_status(c)
        c["current_status"] = status
        c["rejection_reason"] = reason
        if status == STATUS_INJECTION:
            c["injection_assignment_source"] = (
                "manual_geometry"
                if injection_mask_source == "manual_geometry"
                else "validated_automatic_mask"
            )
        elif status == STATUS_SUSPECT_INJECTION:
            c["injection_assignment_source"] = (
                "automatic_mask_qc_failed"
                if mask_qc_failed else "automatic_mask_unvalidated"
            )
        c["candidate_id"] = f"{channel}_s{section:03d}_{i:06d}"
        # summarize.py compatibility aliases (not part of CANDIDATE_COLUMNS).
        c["x_px"] = c["x_global_px"]
        c["y_px"] = c["y_global_px"]
        c["z_plane"] = c["z_index"]
        c["z_um"] = c["section_relative_z_um"]
        c["plane"] = c["optical_plane"]
        c["intensity"] = c["peak_intensity"]
        c["score"] = c["classification_score"]
        c["method"] = backend

    if timer is not None:
        timer.stop("candidate_measurements")
    result.candidates = candidates
    result.n_invalid = sum(1 for c in candidates if c["invalid_coordinate"])
    return result


# --------------------------------------------------------------------------- #
# Back-compat greedy NMS (used by the existing tile-overlap test)
# --------------------------------------------------------------------------- #
def _nms_merge(candidates, nms_distance_um, voxel_zyx):
    """Greedy distance NMS to merge duplicates (kept for back-compat)."""
    if not candidates:
        return []
    import numpy as np  # noqa: PLC0415

    vz, vy, vx = voxel_zyx
    pts = np.array([[c["z_plane"] * vz, c["y_px"] * vy, c["x_px"] * vx] for c in candidates])
    order = np.argsort([-c["score"] for c in candidates])
    keep: list[int] = []
    suppressed = np.zeros(len(candidates), dtype=bool)
    try:
        from scipy.spatial import cKDTree  # noqa: PLC0415

        tree = cKDTree(pts)
        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            for j in tree.query_ball_point(pts[i], nms_distance_um):
                if j != i:
                    suppressed[j] = True
    except Exception:  # pragma: no cover
        for i in order:
            if suppressed[i]:
                continue
            keep.append(i)
            d = np.linalg.norm(pts - pts[i], axis=1)
            suppressed |= (d < nms_distance_um)
            suppressed[i] = False
    return [candidates[i] for i in sorted(keep)]


# --------------------------------------------------------------------------- #
# Section-level driver
# --------------------------------------------------------------------------- #
@dataclass
class SectionDetectionResult:
    channel: str
    section: int
    candidates: list = field(default_factory=list)
    tissue_mask: object = None
    injection_mask: object = None
    injection_core_mask: object = None
    injection_analysis_exclusion_mask: object = None
    generation_suppression_mask: object = None
    generation_suppression_mask_source: str = "none"
    injection_components: dict = field(default_factory=dict)
    mask_diagnostics: dict = field(default_factory=dict)
    generation_diagnostics: dict = field(default_factory=dict)
    projection: object = None
    suppressed_projection: object = None
    corrected: object = None
    crop_origin: tuple = (0, 0)
    plane_numbers: list = field(default_factory=list)
    plane_paths: dict = field(default_factory=dict)
    backend: str = "pilot_log3d"
    n_invalid: int = 0
    warnings: list = field(default_factory=list)
    display_info: dict = field(default_factory=dict)

    @property
    def preliminary_pass(self) -> list:
        return [
            c for c in self.candidates
            if c.get("current_status") == STATUS_PRELIMINARY_PASS
        ]


def read_crop_stack(plane_paths: dict, crop):
    """Read the crop region from each plane into a (Z, Y, X) float32 stack.

    ``crop`` is (x_min, x_max, y_min, y_max) in FULL-RESOLUTION pixels.
    Returns (stack, plane_numbers, crop_origin=(y0, x0), full_shape=(H, W)).
    """
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    plane_numbers = sorted(plane_paths)
    with tifffile.TiffFile(str(plane_paths[plane_numbers[0]])) as tf:
        H, W = (int(s) for s in tf.pages[0].shape[:2])
    if crop:
        x_min, x_max, y_min, y_max = crop
        x0, x1 = max(0, int(x_min)), min(W, int(x_max))
        y0, y1 = max(0, int(y_min)), min(H, int(y_max))
    else:
        x0, x1, y0, y1 = 0, W, 0, H

    planes = []
    for pl in plane_numbers:
        with tifffile.TiffFile(str(plane_paths[pl])) as tf:
            page = tf.pages[0]
            try:
                arr = page.asarray(out="memmap")
            except (ValueError, TypeError):
                arr = page.asarray()
            planes.append(np.asarray(arr[y0:y1, x0:x1], dtype=np.float32))
    return np.stack(planes, axis=0), plane_numbers, (y0, x0), (H, W)


def detect_section(
    channel: str,
    section: int,
    plane_paths: dict,
    config: Config,
    params: DetectionParams,
    crop=None,
    dry_run: bool = False,
    shared_tissue_mask=None,
    cellfinder_detect_main=None,
) -> SectionDetectionResult:
    """Run the candidate pipeline on one section (its planes) of one channel."""
    res = SectionDetectionResult(channel=channel, section=section,
                                 plane_paths=dict(plane_paths), backend=params.backend)
    if not plane_paths:
        LOG.warning("%s section %d: no planes available", channel, section)
        return res

    first_section = (
        config.pilot.first_section if config.pilot.first_section is not None else section
    )

    if dry_run:
        import tifffile  # noqa: PLC0415

        plane_numbers = sorted(plane_paths)
        with tifffile.TiffFile(str(plane_paths[plane_numbers[0]])) as tf:
            h, w = (int(s) for s in tf.pages[0].shape[:2])
        if crop:
            xa, xb, ya, yb = crop
            region = f"x {xa}:{xb}, y {ya}:{yb}"
        else:
            region = "full plane"
        LOG.info("[dry-run] %s section %d: %d planes, full %dx%d (HxW), region (%s), backend=%s",
                 channel, section, len(plane_numbers), h, w, region, params.backend)
        return res

    stack, plane_numbers, crop_origin, _ = read_crop_stack(plane_paths, crop)

    stack_res = detect_candidates_in_stack(
        stack, params, config.acquisition.voxel_size_zyx,
        channel=channel, section=section, first_section=first_section,
        planes_per_section=config.acquisition.planes_per_section,
        plane_numbers=plane_numbers, crop_origin=crop_origin,
        shared_tissue_mask=shared_tissue_mask,
        injection_cfg=params.injection.for_channel(channel),
        backend=params.backend, cellfinder_detect_main=cellfinder_detect_main,
        cellfinder_cfg=params.cellfinder.for_channel(channel),
    )

    res.candidates = stack_res.candidates
    res.tissue_mask = stack_res.tissue_mask
    res.injection_mask = stack_res.injection_mask
    res.injection_core_mask = stack_res.injection_core_mask
    res.injection_analysis_exclusion_mask = stack_res.injection_analysis_exclusion_mask
    res.generation_suppression_mask = stack_res.generation_suppression_mask
    res.generation_suppression_mask_source = stack_res.generation_suppression_mask_source
    res.injection_components = stack_res.injection_components
    res.mask_diagnostics = stack_res.mask_diagnostics
    res.generation_diagnostics = stack_res.generation_diagnostics
    res.projection = stack_res.projection
    res.suppressed_projection = stack_res.suppressed_projection
    res.corrected = stack_res.corrected
    res.crop_origin = crop_origin
    res.plane_numbers = plane_numbers
    res.n_invalid = stack_res.n_invalid
    res.warnings = stack_res.warnings

    for w in res.warnings:
        LOG.warning("QC: %s", w)
    counts = _status_counts(res.candidates)
    LOG.info("%s section %d [%s]: %d candidates -> %s",
             channel, section, params.backend, len(res.candidates), counts)
    return res


def _status_counts(candidates) -> dict:
    out: dict[str, int] = {}
    for c in candidates:
        out[c["current_status"]] = out.get(c["current_status"], 0) + 1
    return out


# --------------------------------------------------------------------------- #
# CSV output -- clearly preliminary, never an accepted-cell table
# --------------------------------------------------------------------------- #
def _write_rows(path: Path, rows) -> Path:
    import csv
    import math

    def csv_value(value):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return value

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: csv_value(r.get(k, "")) for k in CANDIDATE_COLUMNS})
    return path


def write_candidate_tables(out_dir: Path, results: Sequence[SectionDetectionResult]) -> dict:
    """Write the full candidate table plus the preliminary-pass subset.

    all_candidates.csv holds every candidate. preliminary_pass_candidates.csv is
    the documented subset that only passed the preliminary rules -- still not
    cells, just the pass stratum. The old preliminary_candidates.csv (a duplicate
    of all_candidates.csv) is no longer written.
    """
    ensure_dir(out_dir)
    all_rows = [c for r in results for c in r.candidates]
    pass_rows = [
        c for c in all_rows
        if c.get("preliminary_sampling_category") == STATUS_PRELIMINARY_PASS
    ]
    paths = {
        "all": _write_rows(out_dir / "all_candidates.csv", all_rows),
        "preliminary_pass": _write_rows(
            out_dir / "preliminary_pass_candidates.csv", pass_rows
        ),
    }
    LOG.warning(
        "Wrote %d candidates (all) and %d preliminary-pass (NOT final cells) -> %s",
        len(all_rows), len(pass_rows), out_dir,
    )
    return paths


def write_candidates(out_dir: Path, results: Sequence[SectionDetectionResult]) -> Path:
    """Back-compat: write the tables and return the all-candidates CSV path."""
    return write_candidate_tables(out_dir, results)["all"]
