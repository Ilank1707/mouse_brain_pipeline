"""QC display-window estimation and intensity diagnostics (Part 1).

These helpers compute a brightness/contrast *display* window for the saved QC
figures and a per-channel/section intensity-diagnostics record. They are pure
(NumPy in, dicts/arrays out) and READ-ONLY:

  * the source projection is never modified in place;
  * the chosen window NEVER changes raw values, Cellfinder input, background or
    contrast measurements, classifier patches or any CSV measurement.

A "fixed" window reproduces a Fiji 0-513 view; "robust_tissue_percentile" sets
limits from finite, in-tissue, non-background pixels and can exclude the
injection core when estimating the upper limit so the saturated injection does
not blow out the tissue contrast.
"""

from __future__ import annotations

_EPS = 1e-6

# Percentiles recorded in the intensity-diagnostics CSV (Part 1).
DIAGNOSTIC_PERCENTILES = (0.1, 0.5, 1.0, 5.0, 50.0, 95.0, 99.0, 99.5, 99.7, 99.9)

# Column order for the intensity-diagnostics CSV.
INTENSITY_DIAGNOSTIC_COLUMNS = [
    "channel",
    "section",
    "raw_min",
    "raw_max",
    "tissue_pixel_count",
    "background_excluded_pixel_count",
    "p0_1",
    "p0_5",
    "p1",
    "p5",
    "p50",
    "p95",
    "p99",
    "p99_5",
    "p99_7",
    "p99_9",
    "display_mode",
    "display_min",
    "display_max",
    "injection_core_excluded",
]

# Column order for the per-figure QC display-provenance CSV.
QC_DISPLAY_METADATA_COLUMNS = [
    "channel",
    "section",
    "display_mode",
    "display_min",
    "display_max",
    "injection_core_excluded",
    "pixel_pool",
    "fallback_reason",
]

_PERCENTILE_KEYS = {
    0.1: "p0_1",
    0.5: "p0_5",
    1.0: "p1",
    5.0: "p5",
    50.0: "p50",
    95.0: "p95",
    99.0: "p99",
    99.5: "p99_5",
    99.7: "p99_7",
    99.9: "p99_9",
}


def _as_2d_float(projection):
    import numpy as np  # noqa: PLC0415

    return np.asarray(projection, dtype=np.float64)


def _background_excluded_pool(values, tissue_mask, padding_values):
    """Finite, in-tissue, non-padding, non-zero (black-background) pixel pool."""
    import numpy as np  # noqa: PLC0415

    valid = np.isfinite(values)
    if tissue_mask is not None:
        valid &= np.asarray(tissue_mask, dtype=bool)
    # Exclude explicit padding values and the clearly-black background (0).
    for padding in tuple(padding_values) + (0.0,):
        valid &= values != padding
    return valid


def intensity_diagnostics(projection, *, tissue_mask=None, padding_values=(0.0,)) -> dict:
    """Per-channel/section raw range, pixel counts and robust percentiles.

    Percentiles are computed over the background-excluded in-tissue pool so they
    describe genuine tissue rather than the black background.
    """
    import numpy as np  # noqa: PLC0415

    values = _as_2d_float(projection)
    finite = np.isfinite(values)
    finite_vals = values[finite]
    raw_min = float(finite_vals.min()) if finite_vals.size else float("nan")
    raw_max = float(finite_vals.max()) if finite_vals.size else float("nan")

    if tissue_mask is not None:
        tissue_pixel_count = int(np.count_nonzero(finite & np.asarray(tissue_mask, dtype=bool)))
    else:
        tissue_pixel_count = int(finite_vals.size)

    pool_mask = _background_excluded_pool(values, tissue_mask, padding_values)
    pool = values[pool_mask]
    background_excluded_pixel_count = int(pool.size)

    diag = {
        "raw_min": raw_min,
        "raw_max": raw_max,
        "tissue_pixel_count": tissue_pixel_count,
        "background_excluded_pixel_count": background_excluded_pixel_count,
    }
    source = pool if pool.size else finite_vals
    for percentile in DIAGNOSTIC_PERCENTILES:
        key = _PERCENTILE_KEYS[percentile]
        diag[key] = float(np.percentile(source, percentile)) if source.size else float("nan")
    return diag


def compute_display_limits(
    projection,
    settings,
    *,
    tissue_mask=None,
    injection_core_mask=None,
    padding_values=(0.0,),
    minimum_pixels: int = 50,
    exclude_injection_core: bool = True,
) -> dict:
    """Compute a display window for ONE channel's projection (read-only).

    Returns a dict with ``display_min``/``display_max``/``display_mode``,
    ``injection_core_excluded``, ``pixel_pool``, ``fallback_reason`` and the full
    intensity diagnostics. The input projection is never modified.
    """
    import numpy as np  # noqa: PLC0415

    values = _as_2d_float(projection)
    diag = intensity_diagnostics(
        values, tissue_mask=tissue_mask, padding_values=padding_values
    )
    mode = str(getattr(settings, "mode", "robust_tissue_percentile"))
    info = {
        "display_mode": mode,
        "injection_core_excluded": False,
        "pixel_pool": "n/a",
        "fallback_reason": "",
        **diag,
    }

    if mode == "fixed":
        info["display_min"] = float(settings.minimum)
        info["display_max"] = float(settings.maximum)
        info["pixel_pool"] = "fixed"
        return _finalise(info)

    lower_p = float(getattr(settings, "lower_percentile", 0.5))
    upper_p = float(getattr(settings, "upper_percentile", 99.7))

    if mode == "full_data_percentile":
        finite = values[np.isfinite(values)]
        if finite.size >= max(2, minimum_pixels):
            info["display_min"] = float(np.percentile(finite, lower_p))
            info["display_max"] = float(np.percentile(finite, upper_p))
            info["pixel_pool"] = "full_finite"
            return _finalise(info)
        return _finalise(_raw_minmax_fallback(info, finite, "too_few_finite_pixels"))

    # robust_tissue_percentile (default).
    pool_mask = _background_excluded_pool(values, tissue_mask, padding_values)
    pool = values[pool_mask]
    if pool.size < max(2, minimum_pixels):
        # Fall back to full-data percentiles, then raw min/max.
        finite = values[np.isfinite(values)]
        if finite.size >= max(2, minimum_pixels):
            info["display_min"] = float(np.percentile(finite, lower_p))
            info["display_max"] = float(np.percentile(finite, upper_p))
            info["pixel_pool"] = "full_finite_fallback"
            info["fallback_reason"] = "too_few_tissue_pixels"
            return _finalise(info)
        return _finalise(_raw_minmax_fallback(info, finite, "too_few_tissue_pixels"))

    info["display_min"] = float(np.percentile(pool, lower_p))

    upper_pool = pool
    pool_label = "robust_tissue"
    if (
        exclude_injection_core
        and injection_core_mask is not None
        and np.asarray(injection_core_mask, dtype=bool).any()
    ):
        core = np.asarray(injection_core_mask, dtype=bool)
        upper_mask = pool_mask & ~core
        candidate_upper = values[upper_mask]
        if candidate_upper.size >= max(2, minimum_pixels):
            upper_pool = candidate_upper
            info["injection_core_excluded"] = True
            pool_label = "robust_tissue_excl_core"
        else:
            info["fallback_reason"] = "core_exclusion_left_too_few_pixels"
    info["display_max"] = float(np.percentile(upper_pool, upper_p))
    info["pixel_pool"] = pool_label
    return _finalise(info)


def _raw_minmax_fallback(info, finite, reason):
    import numpy as np  # noqa: PLC0415

    if finite.size:
        info["display_min"] = float(np.min(finite))
        info["display_max"] = float(np.max(finite))
    else:
        info["display_min"] = 0.0
        info["display_max"] = 1.0
    info["pixel_pool"] = "raw_minmax_fallback"
    info["fallback_reason"] = reason
    return info


def _finalise(info):
    """Guarantee a strictly increasing, finite window."""
    import math  # noqa: PLC0415

    lo = float(info.get("display_min", 0.0))
    hi = float(info.get("display_max", 1.0))
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    info["display_min"] = lo
    info["display_max"] = hi
    return info


def apply_display(projection, display_min, display_max):
    """Scale a projection to [0, 1] for imshow WITHOUT modifying the input.

    Returns a NEW array; ``projection`` itself (and therefore the raw stack) is
    left untouched.
    """
    import numpy as np  # noqa: PLC0415

    values = np.asarray(projection, dtype=np.float64)
    span = max(float(display_max) - float(display_min), _EPS)
    return np.clip((values - float(display_min)) / span, 0.0, 1.0)


def diagnostics_row(channel, section, info) -> dict:
    """Flatten a ``compute_display_limits`` result into a diagnostics CSV row."""
    row = {"channel": channel, "section": section}
    for column in INTENSITY_DIAGNOSTIC_COLUMNS:
        if column in ("channel", "section"):
            continue
        if column == "display_mode":
            row[column] = info.get("display_mode", "")
        elif column == "display_min":
            row[column] = info.get("display_min", "")
        elif column == "display_max":
            row[column] = info.get("display_max", "")
        elif column == "injection_core_excluded":
            row[column] = bool(info.get("injection_core_excluded", False))
        else:
            row[column] = info.get(column, "")
    return row


def metadata_row(channel, section, info) -> dict:
    """Flatten a ``compute_display_limits`` result into a QC-metadata CSV row."""
    return {
        "channel": channel,
        "section": section,
        "display_mode": info.get("display_mode", ""),
        "display_min": info.get("display_min", ""),
        "display_max": info.get("display_max", ""),
        "injection_core_excluded": bool(info.get("injection_core_excluded", False)),
        "pixel_pool": info.get("pixel_pool", ""),
        "fallback_reason": info.get("fallback_reason", ""),
    }
