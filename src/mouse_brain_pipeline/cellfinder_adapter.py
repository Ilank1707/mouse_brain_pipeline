"""Thin adapter around Cellfinder's real candidate-detection stage.

We do NOT vendor Cellfinder. This calls ``cellfinder.core.detect.detect.main``
(which takes a 3D ``z, y, x`` signal array and does NOT require a background
channel for candidate detection) and converts the returned ``Cell`` objects into
the project's backend-agnostic object-metric dicts.

If Cellfinder is not installed, the ``cellfinder_candidates`` backend raises a
clear installation error -- it never silently falls back to the rule-based
detector (the user must explicitly select ``backend: "pilot_log3d"`` for that).
"""

from __future__ import annotations

from .utilities import LOG

_INSTALL_MSG = (
    "backend='cellfinder_candidates' requires Cellfinder, which is not installed.\n"
    "Install it in a clean Python 3.11 environment, e.g.:\n"
    "    pip install cellfinder\n"
    "(GPU strongly recommended; set detection.cellfinder.torch_device: 'cpu' to run on CPU).\n"
    "To use the self-contained rule-based detector instead, set\n"
    "    detection:\n        backend: \"pilot_log3d\"\nin your config."
)


def _artifact_type():
    """Cellfinder/brainglobe artifact type code (falls back to -1)."""
    try:
        from brainglobe_utils.cells.cells import Cell  # noqa: PLC0415

        return Cell.ARTIFACT
    except Exception:
        return -1


# Known brainglobe Cell type codes -> readable names.
_TYPE_NAMES = {-1: "artifact", 1: "unknown", 2: "cell"}


def _type_name(type_code, fallback=""):
    """Readable name for a Cellfinder type code; keep the numeric code separately."""
    if fallback:
        return fallback
    try:
        return _TYPE_NAMES.get(int(type_code), str(type_code))
    except (TypeError, ValueError):
        return ""


def map_cellfinder_z(raw_z, n_planes, padding_offset=None):
    """Map a returned Cellfinder z to a 0..(n_planes-1) stack index.

    Returns (mapped_stack_z, method, valid). An in-range z maps directly. If a
    padding offset is given and (z - offset) lands in range, we use it. Otherwise
    the value is kept but flagged invalid (the caller clamps only the seed and
    sends the candidate to review). We never silently clamp away a bad z.
    """
    try:
        rz = float(raw_z)
    except (TypeError, ValueError):
        return 0, "unparseable_z", False
    direct = int(round(rz))
    if 0 <= direct <= n_planes - 1:
        return direct, "direct_in_range", True
    if padding_offset is not None:
        corrected = int(round(rz - float(padding_offset)))
        if 0 <= corrected <= n_planes - 1:
            return corrected, "padding_offset_corrected", True
    clamped = min(max(direct, 0), n_planes - 1)
    return clamped, "out_of_range_unmapped", False


def measure_local_morphology(raw, corrected, z_seed, y_local, x_local, voxel_zyx, params):
    """Characterise one candidate centroid from the corrected/raw stacks.

    Returns the same object-metric dict shape as the rule-based backend so the
    shared finalisation (classification, coords, masks) is identical.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    from .candidate_detection import _consecutive_support, _xy_elongation, _EPS

    vz, vy, vx = voxel_zyx
    Z, H, W = corrected.shape
    rad_px = int(np.ceil(params.max_diameter_um / vy)) + 3
    y0, y1 = max(0, y_local - rad_px), min(H, y_local + rad_px + 1)
    x0, x1 = max(0, x_local - rad_px), min(W, x_local + rad_px + 1)
    cyl, cxl = y_local - y0, x_local - x0
    sub_corr = corrected[:, y0:y1, x0:x1]

    # Peak plane = the plane with the strongest signal at the centroid (3x3 max).
    center_vals = np.array([
        float(sub_corr[z, max(0, cyl - 1):cyl + 2, max(0, cxl - 1):cxl + 2].max())
        for z in range(Z)
    ])
    peak_z = int(np.argmax(center_vals))

    plane = sub_corr[peak_z]
    med = float(np.median(plane))
    mad = float(np.median(np.abs(plane - med)))
    thr = med + 3.0 * 1.4826 * mad + _EPS

    fgp = plane > thr
    footprint = np.zeros_like(fgp)
    if fgp[cyl, cxl]:
        lab, n = ndi.label(fgp)
        if n:
            footprint = lab == lab[cyl, cxl]
    if not footprint.any():
        footprint[cyl, cxl] = True

    area_xy = int(footprint.sum())
    xy_diam_um = 2.0 * np.sqrt(max(area_xy, 1) / np.pi) * vy
    elongation = _xy_elongation(footprint)

    present = center_vals > thr
    present[peak_z] = True
    z_run, n_consec = _consecutive_support(present, peak_z)
    z_extent_um = (max(z_run) - min(z_run) + 1) * vz

    volume_um3 = area_xy * n_consec * (vz * vy * vx)
    equiv_diam_um = (6.0 * volume_um3 / np.pi) ** (1.0 / 3.0)

    centroids = []
    for z in z_run:
        member = (sub_corr[z] > thr) & footprint
        ys, xs = np.nonzero(member)
        if ys.size:
            centroids.append((ys.mean(), xs.mean()))
    xy_shift_um = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            d = np.hypot((centroids[i][0] - centroids[j][0]) * vy,
                         (centroids[i][1] - centroids[j][1]) * vx)
            xy_shift_um = max(xy_shift_um, float(d))

    peak_intensity = float(raw[peak_z, y_local, x_local])
    raw_plane = raw[peak_z, y0:y1, x0:x1]
    mean_intensity = float(raw_plane[footprint].mean()) if footprint.any() else peak_intensity

    return {
        "z_index": peak_z, "y_local_px": int(y_local), "x_local_px": int(x_local),
        "peak_plane": peak_z, "z_indices": z_run,
        "n_consecutive_planes": int(n_consec), "xy_centroid_shift_um": round(xy_shift_um, 3),
        "equivalent_diameter_um": round(float(equiv_diam_um), 3),
        "xy_diameter_um": round(float(xy_diam_um), 3),
        "z_extent_um": round(float(z_extent_um), 3),
        "volume_um3": round(float(volume_um3), 3),
        "elongation": round(float(elongation), 3),
        "peak_intensity": round(peak_intensity, 2),
        "mean_intensity": round(mean_intensity, 2),
        "touches_crop_boundary": False,
    }


def run_cellfinder_detection(raw, corrected, voxel_zyx, params, *, detect_main=None,
                             cellfinder_cfg=None, detection_array=None):
    """Call Cellfinder candidate detection and return object-metric dicts.

    Cellfinder detects on ``detection_array`` (defaults to ``raw``; may be an
    injection-suppressed in-memory copy). Per-candidate morphology is always
    measured on ``raw``/``corrected`` so suppression never alters measurements.
    ``cellfinder_cfg`` overrides ``params.cellfinder`` (per-channel settings).
    ``detect_main`` may be injected for testing; otherwise the real
    ``cellfinder.core.detect.detect.main`` is imported lazily.
    """
    import numpy as np  # noqa: PLC0415

    cf = cellfinder_cfg if cellfinder_cfg is not None else params.cellfinder
    if detect_main is None:
        try:
            from cellfinder.core.detect.detect import main as detect_main  # noqa: PLC0415
        except Exception as exc:  # ImportError or downstream torch import errors
            raise ImportError(_INSTALL_MSG) from exc

    # Preserve the source 16-bit values; pass z, y, x order and (z, y, x) voxels.
    detection_source = detection_array if detection_array is not None else raw
    signal_array = np.ascontiguousarray(detection_source).astype(np.uint16)
    voxel_sizes = tuple(float(v) for v in voxel_zyx)
    LOG.info("Cellfinder candidate detection: array shape %s (z,y,x), voxels %s, device=%s",
             signal_array.shape, voxel_sizes, cf.torch_device)

    cells = detect_main(
        signal_array=signal_array,
        voxel_sizes=voxel_sizes,
        soma_diameter=cf.soma_diameter_um,
        max_cluster_size=cf.max_cluster_size_um3,
        ball_xy_size=cf.ball_xy_size_um,
        ball_z_size=cf.ball_z_size_um,
        ball_overlap_fraction=cf.ball_overlap_fraction,
        soma_spread_factor=cf.soma_spread_factor,
        log_sigma_size=cf.log_sigma_size,
        n_sds_above_mean_thresh=cf.n_sds_above_mean_thresh,
        n_sds_above_mean_tiled_thresh=cf.n_sds_above_mean_tiled_thresh,
        tiled_thresh_tile_size=cf.tiled_thresh_tile_size,
        outlier_keep=cf.outlier_keep,
        artifact_keep=cf.artifact_keep,
        batch_size=cf.batch_size,
        torch_device=cf.torch_device,
    )

    artifact_type = _artifact_type()
    padding_offset = getattr(cf, "cellfinder_z_padding_offset", None)
    Z, H, W = corrected.shape
    objects = []
    for cell in cells:
        # Cell coordinates: x = column, y = row, z = plane index (0-based).
        x_local_px = int(round(getattr(cell, "x")))
        y_local_px = int(round(getattr(cell, "y")))
        raw_z = getattr(cell, "z")
        x_local_px = min(max(x_local_px, 0), W - 1)
        y_local_px = min(max(y_local_px, 0), H - 1)
        # Map the returned z explicitly; flag (not hide) anything out of range.
        mapped_z, z_method, z_valid = map_cellfinder_z(raw_z, Z, padding_offset)
        if not z_valid:
            LOG.warning("Cellfinder returned z=%s outside 0..%d (%s) -- kept and "
                        "sent to review.", raw_z, Z - 1, z_method)
        metrics = measure_local_morphology(
            raw, corrected, mapped_z, y_local_px, x_local_px, voxel_zyx, params
        )
        cell_type = getattr(cell, "type", None)
        metrics["is_artifact"] = bool(cell_type == artifact_type)
        metrics["cellfinder_x"] = getattr(cell, "x", "")
        metrics["cellfinder_y"] = getattr(cell, "y", "")
        metrics["cellfinder_z"] = raw_z
        metrics["cellfinder_returned_z_raw"] = raw_z
        metrics["cellfinder_z_mapping_method"] = z_method
        metrics["mapped_stack_z_index"] = mapped_z
        metrics["original_cellfinder_z_valid"] = z_valid
        metrics["cellfinder_type"] = cell_type if cell_type is not None else ""
        metrics["cellfinder_type_name"] = _type_name(
            cell_type, getattr(cell, "type_name", "")
        )
        objects.append(metrics)
    LOG.info("Cellfinder returned %d candidate(s)", len(objects))
    return objects
