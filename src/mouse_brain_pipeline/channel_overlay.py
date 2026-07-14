"""Cross-channel (green vs red) overlay analysis for provisional candidates.

After candidate detection, every candidate in ``all_candidates.csv`` is measured
with the SAME fixed-XY seven-plane measurement used by the detector
(:func:`candidate_detection.measure_fixed_xy_profile`) in BOTH biological
channels:

  * ``green_signal``     -- the green dye,
  * ``channel_2_signal`` -- the red dye.

The two measurements are compared so a candidate can be labelled
``green_dominant`` / ``red_dominant`` / ``both`` / ``unclear`` from the *actual
measured signal*. This is DISPLAY / AUDIT only -- it never changes a candidate,
its status, a mask, or any count, it never uses one channel as the other's input,
and it NEVER forces the red channel to have fewer detections.

Outputs (written under ``<run_dir>/channel_overlay/``):

  * ``channel_overlay_candidate_measurements.csv`` -- one row per candidate;
  * ``channel_overlay_summary.csv``                -- dominant-channel tallies;
  * ``green_red_overlay_qc.png``                   -- green/red composite with the
    candidate markers coloured by ``dominant_channel``.

Nothing here modifies the raw TIFFs, and the whole step is guarded so it can
never abort a detection run.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .channels import CHANNEL_2_SIGNAL, GREEN_SIGNAL

# Dominant-channel labels (the only four values the column ever takes).
GREEN_DOMINANT = "green_dominant"
RED_DOMINANT = "red_dominant"
BOTH = "both"
UNCLEAR = "unclear"
DOMINANT_CHANNELS = (GREEN_DOMINANT, RED_DOMINANT, BOTH, UNCLEAR)

# Marker colours for the composite QC (RGB), keyed by dominant_channel.
_DOMINANT_RGB = {
    GREEN_DOMINANT: (0, 255, 0),
    RED_DOMINANT: (255, 0, 0),
    BOTH: (255, 255, 0),
    UNCLEAR: (170, 170, 170),
}

OVERLAY_MEASUREMENT_COLUMNS = [
    "candidate_id",
    "channel",                 # channel the candidate was DETECTED in
    "section",
    "x_global_px",
    "y_global_px",
    "z_index",                 # candidate's own optical-plane z index
    "optical_plane",
    "green_peak",
    "green_local_background",
    "green_snr",
    "green_peak_plane",        # 0-based z index of the green measurement peak
    "green_measurement_valid",
    "red_peak",
    "red_local_background",
    "red_snr",
    "red_peak_plane",
    "red_measurement_valid",
    "green_signal_above_background",
    "red_signal_above_background",
    "red_green_ratio",
    "dominant_channel",
]

OVERLAY_SUMMARY_COLUMNS = ["detection_channel", "dominant_channel", "count"]


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O -- unit-testable on plain numbers)
# --------------------------------------------------------------------------- #
def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def signal_above_background(peak, local_background) -> float:
    """Background-subtracted peak, clamped at 0 (0.0 when either is not finite)."""
    if not (_finite(peak) and _finite(local_background)):
        return 0.0
    return max(float(peak) - float(local_background), 0.0)


def red_green_ratio(green_signal_above_bg, red_signal_above_bg):
    """Red/green background-subtracted ratio.

    ``None`` when it is undefined (green signal is zero) so nothing spurious is
    written; ``math.inf`` when only red has signal.
    """
    g = float(green_signal_above_bg)
    r = float(red_signal_above_bg)
    if g > 0.0:
        return r / g
    if r > 0.0:
        return math.inf
    return None


def classify_dominant(
    green_snr,
    red_snr,
    green_signal_above_bg,
    red_signal_above_bg,
    *,
    green_valid,
    red_valid,
    snr_threshold,
    dominance_ratio,
) -> str:
    """Label a candidate from the two MEASURED channel signals.

    ``green_dominant`` / ``red_dominant`` when one channel is present and either
    the other is absent or its background-subtracted peak is at least
    ``dominance_ratio`` times weaker; ``both`` when both are present and
    comparable; ``unclear`` when neither channel is present (or both
    measurements are invalid). Red is never penalised -- only measured signal is
    used.
    """
    g_present = bool(green_valid) and _finite(green_snr) and float(green_snr) >= snr_threshold
    r_present = bool(red_valid) and _finite(red_snr) and float(red_snr) >= snr_threshold
    gs = float(green_signal_above_bg) if _finite(green_signal_above_bg) else 0.0
    rs = float(red_signal_above_bg) if _finite(red_signal_above_bg) else 0.0

    if not (green_valid or red_valid):
        return UNCLEAR
    if g_present and r_present:
        if gs <= 0.0 and rs <= 0.0:
            return BOTH
        if gs >= rs * dominance_ratio:
            return GREEN_DOMINANT
        if rs >= gs * dominance_ratio:
            return RED_DOMINANT
        return BOTH
    if g_present:
        return GREEN_DOMINANT
    if r_present:
        return RED_DOMINANT
    return UNCLEAR


def summarize_overlay(rows) -> list[dict]:
    """Tally ``dominant_channel`` per detection channel and overall (all zeros kept)."""
    groups = (GREEN_SIGNAL, CHANNEL_2_SIGNAL, "all")
    tally = {g: {d: 0 for d in DOMINANT_CHANNELS} for g in groups}
    for row in rows:
        dom = row.get("dominant_channel", UNCLEAR)
        if dom not in tally["all"]:
            continue
        tally["all"][dom] += 1
        channel = row.get("channel")
        if channel in tally:
            tally[channel][dom] += 1
    out = []
    for group in groups:
        for dom in DOMINANT_CHANNELS:
            out.append({
                "detection_channel": group,
                "dominant_channel": dom,
                "count": tally[group][dom],
            })
    return out


# --------------------------------------------------------------------------- #
# Measurement (reuses the detector's fixed-XY seven-plane measurement)
# --------------------------------------------------------------------------- #
class _PlaneStack:
    """Minimal ``(z, y, x)`` view over a list of 2-D planes (memmaps or arrays).

    Exposes just the ``shape`` / ``__getitem__`` that
    :func:`candidate_detection.measure_fixed_xy_profile` needs, so memmapped
    planes stay lazy and only small measurement windows are read into memory.
    """

    def __init__(self, planes):
        self._planes = list(planes)
        height, width = self._planes[0].shape[:2]
        self.shape = (len(self._planes), int(height), int(width))

    def __getitem__(self, z):
        return self._planes[z]


def channel_peak_measurement(stack, tissue_plane, cx, cy, params, voxel_y_um):
    """Fixed-XY seven-plane peak measurement of one channel at ``(cx, cy)``.

    Returns ``dict(peak, local_background, snr, peak_plane, valid)`` where the
    peak plane is the profile's maximum-contrast optical plane -- exactly how the
    detector picks a candidate's peak.
    """
    from .candidate_detection import measure_fixed_xy_profile  # noqa: PLC0415

    measurements, _profile, peak_z, _support = measure_fixed_xy_profile(
        stack, tissue_plane, int(round(cy)), int(round(cx)), params, voxel_y_um=voxel_y_um,
    )
    m = measurements[int(peak_z)]
    return {
        "peak": m["central_signal"],
        "local_background": m["background_median"],
        "snr": m["contrast"],
        "peak_plane": int(peak_z),
        "valid": bool(m["measurement_valid"]),
    }


def _read_plane_stack(ordered_planes):
    """Build a lazy ``_PlaneStack`` from ``[(plane_number, path), ...]``."""
    from .seven_plane_qc import _read_plane  # noqa: PLC0415

    return _PlaneStack([_read_plane(path) for _plane, path in ordered_planes])


def _all_true_plane(shape):
    """Zero-memory all-in-tissue mask (validity then only excludes padding/NaN)."""
    import numpy as np  # noqa: PLC0415

    return np.broadcast_to(np.bool_(True), (int(shape[0]), int(shape[1])))


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf"
        return round(value, 4)
    return value


def write_overlay_measurements(out_dir, rows) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "channel_overlay_candidate_measurements.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERLAY_MEASUREMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _fmt(row.get(c, "")) for c in OVERLAY_MEASUREMENT_COLUMNS})
    return path


def write_overlay_summary(out_dir, rows) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "channel_overlay_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OVERLAY_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summarize_overlay(rows))
    return path


def _downsampled_window(projection, max_dim):
    """Strided (cheap, memory-light) downsample of a 2-D projection to <= max_dim."""
    import numpy as np  # noqa: PLC0415

    proj = np.asarray(projection)
    height, width = proj.shape[:2]
    step = max(1, int(math.ceil(max(height, width) / float(max_dim))))
    small = np.ascontiguousarray(proj[::step, ::step])
    return small, step


def _channel_projection_small(ordered_planes, max_dim):
    """Downscaled max-projection (uint8-windowed) for one channel, memory-light."""
    import numpy as np  # noqa: PLC0415

    from .qc_native import apply_window_uint8
    from .seven_plane_qc import _read_plane

    proj = None
    for _plane, path in ordered_planes:
        arr = _read_plane(path)
        proj = np.array(arr, copy=True) if proj is None else np.maximum(proj, arr)
    small, step = _downsampled_window(proj, max_dim)
    finite = small[np.isfinite(small)] if small.size else small
    if finite.size:
        lo, hi = np.percentile(finite, [1.0, 99.7])
    else:
        lo, hi = 0.0, 1.0
    return apply_window_uint8(small, float(lo), float(hi)), step


def render_overlay_qc(out_dir, ordered_by_channel, rows, *, max_dim=2000):
    """Write ``green_red_overlay_qc.png``: green/red composite + dominance markers.

    ``ordered_by_channel`` maps each channel to its ``[(plane_number, path), ...]``.
    Markers are coloured by ``dominant_channel``. Returns the PNG path (or ``None``
    when neither channel has planes to render).
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    green_ordered = ordered_by_channel.get(GREEN_SIGNAL) or []
    red_ordered = ordered_by_channel.get(CHANNEL_2_SIGNAL) or []
    reference = green_ordered or red_ordered
    if not reference:
        return None

    from .seven_plane_qc import _read_plane
    full_h, full_w = (int(s) for s in _read_plane(reference[0][1]).shape[:2])

    step = None
    green8 = red8 = None
    if green_ordered:
        green8, step = _channel_projection_small(green_ordered, max_dim)
    if red_ordered:
        red8, step_r = _channel_projection_small(red_ordered, max_dim)
        step = step if step is not None else step_r
    small_h = green8.shape[0] if green8 is not None else red8.shape[0]
    small_w = green8.shape[1] if green8 is not None else red8.shape[1]
    if green8 is None:
        green8 = np.zeros((small_h, small_w), dtype=np.uint8)
    if red8 is None:
        red8 = np.zeros((small_h, small_w), dtype=np.uint8)

    composite = np.zeros((small_h, small_w, 3), dtype=np.uint8)
    composite[:, :, 0] = red8       # red dye -> red
    composite[:, :, 1] = green8     # green dye -> green
    image = Image.fromarray(composite, "RGB")
    draw = ImageDraw.Draw(image)
    radius = max(2, small_w // 300)
    for row in rows:
        try:
            x = int(round(float(row["x_global_px"]) / step))
            y = int(round(float(row["y_global_px"]) / step))
        except (TypeError, ValueError, KeyError):
            continue
        if not (0 <= x < small_w and 0 <= y < small_h):
            continue
        colour = _DOMINANT_RGB.get(row.get("dominant_channel"), _DOMINANT_RGB[UNCLEAR])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=colour, width=1)

    path = out_dir / "green_red_overlay_qc.png"
    image.save(str(path), format="PNG")
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_run(run_dir, config, *, sections=None, render_qc=True):
    """Measure both channels at every candidate of a run and write the outputs.

    Reads ONLY ``<run_dir>/all_candidates.csv`` and the configured channel TIFFs.
    Candidate ``(x_global_px, y_global_px)`` are full-resolution coordinates, so
    the full section planes are read and measured at those pixels -- this is
    correct for both full-section and cropped runs. Returns a dict with the
    output paths, the per-candidate rows and the dominant-channel summary.
    """
    from .audit import index_channel  # noqa: PLC0415
    from .candidate_detection import params_from_config  # noqa: PLC0415
    from .review import read_csv_rows  # noqa: PLC0415
    from .seven_plane_qc import ordered_section_planes  # noqa: PLC0415

    run_dir = Path(run_dir)
    csv_path = run_dir / "all_candidates.csv"
    candidates = read_csv_rows(csv_path)

    params = params_from_config(config)
    voxel_y_um = float(config.acquisition.voxel_size_y_um)
    overlay_cfg = getattr(config, "channel_overlay", None)
    snr_threshold = float(getattr(overlay_cfg, "snr_threshold", 3.0))
    dominance_ratio = float(getattr(overlay_cfg, "dominance_ratio", 1.5))
    qc_max_dim = int(getattr(overlay_cfg, "qc_max_dim", 2000))

    channel_dirs = {
        GREEN_SIGNAL: config.data.green_signal_dir,
        CHANNEL_2_SIGNAL: config.data.channel_2_signal_dir,
    }
    indexes = {
        ch: index_channel(ch, channel_dirs[ch], config.data.filename_regex)
        for ch in (GREEN_SIGNAL, CHANNEL_2_SIGNAL)
    }

    # Group candidates by section (all channels' candidates measured together).
    by_section: dict[int, list[dict]] = {}
    for cand in candidates:
        try:
            section = int(cand.get("section"))
        except (TypeError, ValueError):
            continue
        if sections is not None and section not in {int(s) for s in sections}:
            continue
        by_section.setdefault(section, []).append(cand)

    measurements_by_id: dict[str, dict] = {}
    ordered_by_channel: dict[str, list] = {}
    for section, section_candidates in sorted(by_section.items()):
        per_channel_measured = {}
        for channel in (GREEN_SIGNAL, CHANNEL_2_SIGNAL):
            ordered = ordered_section_planes(indexes[channel], section)
            if not ordered:
                per_channel_measured[channel] = {}
                continue
            ordered_by_channel.setdefault(channel, ordered)  # first section for the QC
            stack = _read_plane_stack(ordered)
            tissue_plane = _all_true_plane(stack.shape[1:])
            measured = {}
            for cand in section_candidates:
                cid = cand.get("candidate_id")
                try:
                    cx = float(cand["x_global_px"])
                    cy = float(cand["y_global_px"])
                except (TypeError, ValueError, KeyError):
                    continue
                measured[cid] = channel_peak_measurement(
                    stack, tissue_plane, cx, cy, params, voxel_y_um,
                )
            per_channel_measured[channel] = measured

        green_measured = per_channel_measured.get(GREEN_SIGNAL, {})
        red_measured = per_channel_measured.get(CHANNEL_2_SIGNAL, {})
        for cand in section_candidates:
            cid = cand.get("candidate_id")
            g = green_measured.get(cid)
            r = red_measured.get(cid)
            measurements_by_id[cid] = _build_measurement_row(
                cand, g, r, snr_threshold=snr_threshold, dominance_ratio=dominance_ratio,
            )

    rows = [measurements_by_id[c.get("candidate_id")]
            for c in candidates if c.get("candidate_id") in measurements_by_id]

    out_dir = run_dir / "channel_overlay"
    measurements_path = write_overlay_measurements(out_dir, rows)
    summary_path = write_overlay_summary(out_dir, rows)
    qc_path = None
    if render_qc:
        try:
            qc_path = render_overlay_qc(out_dir, ordered_by_channel, rows, max_dim=qc_max_dim)
        except Exception:  # pragma: no cover - QC image must never abort the run
            qc_path = None

    return {
        "out_dir": out_dir,
        "measurements_csv": measurements_path,
        "summary_csv": summary_path,
        "qc_png": qc_path,
        "rows": rows,
        "summary": summarize_overlay(rows),
        "candidate_count": len(rows),
    }


def _build_measurement_row(cand, green, red, *, snr_threshold, dominance_ratio) -> dict:
    """Assemble one overlay row from a candidate + its two channel measurements."""
    green = green or {}
    red = red or {}
    green_peak = green.get("peak", float("nan"))
    green_bg = green.get("local_background", float("nan"))
    red_peak = red.get("peak", float("nan"))
    red_bg = red.get("local_background", float("nan"))
    green_valid = bool(green.get("valid", False))
    red_valid = bool(red.get("valid", False))
    green_snr = green.get("snr", float("nan"))
    red_snr = red.get("snr", float("nan"))

    green_sig = signal_above_background(green_peak, green_bg)
    red_sig = signal_above_background(red_peak, red_bg)
    dominant = classify_dominant(
        green_snr, red_snr, green_sig, red_sig,
        green_valid=green_valid, red_valid=red_valid,
        snr_threshold=snr_threshold, dominance_ratio=dominance_ratio,
    )
    return {
        "candidate_id": cand.get("candidate_id"),
        "channel": cand.get("channel"),
        "section": cand.get("section"),
        "x_global_px": cand.get("x_global_px"),
        "y_global_px": cand.get("y_global_px"),
        "z_index": cand.get("z_index"),
        "optical_plane": cand.get("optical_plane"),
        "green_peak": green_peak,
        "green_local_background": green_bg,
        "green_snr": green_snr,
        "green_peak_plane": green.get("peak_plane", ""),
        "green_measurement_valid": green_valid,
        "red_peak": red_peak,
        "red_local_background": red_bg,
        "red_snr": red_snr,
        "red_peak_plane": red.get("peak_plane", ""),
        "red_measurement_valid": red_valid,
        "green_signal_above_background": green_sig,
        "red_signal_above_background": red_sig,
        "red_green_ratio": red_green_ratio(green_sig, red_sig),
        "dominant_channel": dominant,
    }
