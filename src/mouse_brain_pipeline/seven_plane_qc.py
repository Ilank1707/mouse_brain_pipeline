"""Whole-section seven-plane candidate QC rendering.

This produces the same *kind* of full-brain candidate overlay as
``04_candidates_before_injection_exclusion.png``, but separately for each of the
seven optical planes of a section (``section_070_01.tif`` .. ``section_070_07``).

Because the seven planes are already registered, a candidate's
``(x_global_px, y_global_px)`` is drawn at the **same** pixel position on every
plane -- candidates are never independently moved or recentred. What changes per
plane is only how strongly the marker is drawn: a candidate is shown strongly on
its peak / support planes and faintly (mode ``all``) or not at all (mode
``support_only``) on the planes that do not support it.

Design guarantees (checked by ``tests/test_seven_plane_qc.py``):

* planes load in order ``_01`` .. ``_07``;
* each plane is rendered and saved at its **original** TIFF width/height with no
  resize and lossless PNG;
* the raw TIFF and the loaded 16-bit array are never modified -- only a windowed
  *copy* is drawn on;
* the display window is for visualisation only and never feeds detection or any
  CSV measurement;
* status colours come from :data:`qc_native._STATUS_RGB` and symbols mirror
  ``candidate_qc._STATUS_STYLE`` so this view matches the existing QC figures.

Markers are drawn directly onto a native-resolution image with Pillow (no
low-resolution matplotlib screenshot that is later enlarged). The combined
montages are explicit downscaled overviews and never replace the full-res files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .qc_native import _STATUS_RGB, apply_window_uint8, save_png_fullres
from .review_patches import ordered_section_planes, parse_peak_index, parse_support_indices

_DEFAULT_RGB = (255, 255, 255)

# status -> plotting symbol, mirroring candidate_qc._STATUS_STYLE (single source
# of truth for colours is qc_native._STATUS_RGB). Kept here as small literals so
# this module imports without the heavy detection stack.
STATUS_SYMBOLS = {
    "preliminary_rule_pass": "o",   # green circle
    "preliminary_rule_fail": ".",   # grey dot
    "manual_review": "s",           # yellow square
    "invalid_measurement": "D",     # cyan diamond
    "suspect_injection_mask": "^",  # orange triangle
    "injection_site": "x",          # red cross (confirmed injection)
    "artifact": "+",                # purple plus (Cellfinder artefact/outlier)
}

# Per-plane marker emphasis. ``None`` means "do not draw on this plane".
#
# marker-mode "all":     every candidate is shown on every plane; a candidate is
#                        never hidden because the plane is not in its support
#                        list. Unsupported planes use a thinner, fainter marker;
#                        support planes and the peak plane use a stronger marker.
# marker-mode "support": only candidates supported on that plane are drawn.
_MARKER_SPECS = {
    "all": {
        "peak": {"alpha": 255, "width": 3, "scale": 1.7, "centre_dot": True},
        "support": {"alpha": 255, "width": 2, "scale": 1.0, "centre_dot": False},
        "unsupported": {"alpha": 90, "width": 1, "scale": 0.8, "centre_dot": False},
    },
    "support_only": {
        "peak": {"alpha": 255, "width": 3, "scale": 1.7, "centre_dot": True},
        "support": {"alpha": 255, "width": 2, "scale": 1.0, "centre_dot": False},
        "unsupported": None,
    },
}

# CLI exposes "all"/"support"; "support" maps to the internal "support_only".
_MODE_ALIASES = {"support": "support_only"}


def normalize_marker_mode(mode: str) -> str:
    return _MODE_ALIASES.get(mode, mode)

SEVEN_PLANE_QC_METADATA_COLUMNS = [
    "filename",
    "source_tiff_path",
    "channel",
    "section",
    "optical_plane",
    "source_width",
    "source_height",
    "saved_width",
    "saved_height",
    "display_mode",
    "display_min",
    "display_max",
    "render_mode",
    "candidates_displayed",
    "candidates_supported_on_plane",
    "file_format",
    "resizing_occurred",
]


# --------------------------------------------------------------------------- #
# Run metadata / candidate-file selection (cropped vs full-section)
# --------------------------------------------------------------------------- #
def run_metadata_path(csv_path) -> Path:
    """Sibling ``candidate_run_metadata.json`` for an ``all_candidates.csv``."""
    return Path(csv_path).with_name("candidate_run_metadata.json")


def read_run_metadata(csv_path) -> dict | None:
    """Load the run metadata next to a candidate CSV, or ``None`` if absent."""
    import json  # noqa: PLC0415

    path = run_metadata_path(csv_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def classify_run_crop(metadata: dict | None) -> str:
    """Classify a run as ``'full'``, ``'crop'`` or ``'unknown'`` from metadata."""
    if not metadata:
        return "unknown"
    mode = metadata.get("crop_mode")
    crop = metadata.get("crop_x_min_x_max_y_min_y_max")
    if mode == "full_xy_section" or crop in (None, [], ""):
        return "full"
    if mode == "xy_crop" or crop:
        return "crop"
    return "unknown"


def crop_covers_full_image(crop, image_width, image_height) -> bool:
    """True if an ``[x_min, x_max, y_min, y_max]`` crop spans the whole image."""
    if not crop or len(crop) != 4 or not image_width or not image_height:
        return False
    x_min, x_max, y_min, y_max = crop
    return (
        (x_min is None or x_min <= 0)
        and (y_min is None or y_min <= 0)
        and (x_max is None or x_max >= image_width)
        and (y_max is None or y_max >= image_height)
    )


def recorded_candidate_count(metadata: dict | None, channel: str) -> int | None:
    """Candidate count recorded for one channel in the run metadata, if any."""
    counts = (metadata or {}).get("candidate_counts_by_channel") or {}
    try:
        return int(counts[channel])
    except (KeyError, TypeError, ValueError):
        return None


def run_is_full_section(
    metadata: dict | None, section: int, channel: str, *,
    image_width=None, image_height=None,
) -> tuple[bool, str]:
    """Is a run a usable FULL-section run for ``(section, channel)``?

    Requires: crop is none (or a crop that spans the whole image -- which needs
    the original TIFF dimensions); the section was processed; and the channel is
    present in the run. Returns ``(ok, reason)``.
    """
    if not metadata:
        return False, "no_run_metadata"
    crop_kind = classify_run_crop(metadata)
    if crop_kind == "crop":
        crop = metadata.get("crop_x_min_x_max_y_min_y_max")
        if not crop_covers_full_image(crop, image_width, image_height):
            return False, "cropped_run"
    elif crop_kind == "unknown":
        return False, "crop_mode_unknown"
    processed = [int(s) for s in (metadata.get("processed_sections") or [])]
    if int(section) not in processed:
        return False, "section_not_processed"
    channels = set((metadata.get("candidate_counts_by_channel") or {}))
    channels |= set((metadata.get("effective_cellfinder_by_channel") or {}))
    if channel not in channels:
        return False, "channel_absent"
    return True, "ok"


def find_latest_full_section_csv(
    search_root, section: int, channel: str, *,
    image_width=None, image_height=None,
):
    """Newest valid full-section ``all_candidates.csv`` under ``search_root``.

    Selection is by the run metadata's ``run_timestamp_utc`` -- never by folder
    modification time. Only runs that pass :func:`run_is_full_section` are
    considered. Returns ``(csv_path, metadata)`` or ``None``.
    """
    root = Path(search_root)
    if not root.is_dir():
        return None
    matches = []
    for csv_path in root.rglob("all_candidates.csv"):
        metadata = read_run_metadata(csv_path)
        ok, _reason = run_is_full_section(
            metadata, section, channel,
            image_width=image_width, image_height=image_height,
        )
        if ok:
            timestamp = str(metadata.get("run_timestamp_utc", ""))
            matches.append((timestamp, str(csv_path), csv_path, metadata))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1]))   # newest run timestamp wins
    return matches[-1][2], matches[-1][3]


def select_candidate_rows(rows, channel: str, section) -> list[dict]:
    """Rows for one channel + section (exact reported 'candidates loaded')."""
    return [
        r for r in rows
        if r.get("channel") == channel and str(r.get("section")) == str(section)
    ]


def count_mismatch_warning(loaded: int, recorded: int | None, *,
                           tolerance: float = 0.05, min_abs: int = 25) -> str | None:
    """Warn when the loaded count diverges sharply from the recorded count."""
    if recorded is None:
        return None
    diff = abs(int(loaded) - int(recorded))
    if diff == 0:
        return None
    relative = diff / max(int(recorded), 1)
    if diff >= min_abs and (recorded == 0 or relative > tolerance):
        return (
            f"loaded candidate count ({loaded}) differs sharply from the run "
            f"metadata count ({recorded}) for this channel "
            f"(|diff|={diff}, {relative:.0%}). You may be reading the wrong "
            f"all_candidates.csv (e.g. an older cropped run)."
        )
    return None


# --------------------------------------------------------------------------- #
# Candidate <-> plane logic (pure)
# --------------------------------------------------------------------------- #
def candidate_plane_state(candidate: dict, plane_index: int) -> str:
    """Return ``'peak'`` | ``'support'`` | ``'unsupported'`` for a 0-based plane.

    ``plane_index`` is the optical-plane Z index (plane ``01`` -> 0). The peak
    wins when a plane is both the peak and in the support set.
    """
    peak = parse_peak_index(candidate, default=-1)
    support = parse_support_indices(candidate)
    if int(plane_index) == int(peak):
        return "peak"
    if int(plane_index) in support:
        return "support"
    return "unsupported"


def marker_spec(state: str, mode: str):
    """Emphasis dict for a candidate state under a render mode, or ``None``."""
    try:
        specs = _MARKER_SPECS[normalize_marker_mode(mode)]
    except KeyError:
        raise ValueError(f"Unknown render mode: {mode!r}") from None
    return specs[state]


def status_rgb(status) -> tuple[int, int, int]:
    return _STATUS_RGB.get(status, _DEFAULT_RGB)


def status_symbol(status) -> str:
    return STATUS_SYMBOLS.get(status, "o")


def candidate_draw_xy(candidate: dict, origin=(0, 0)) -> tuple[int, int] | None:
    """Integer draw position from the global coordinates (plane-independent).

    The same ``(x_global_px, y_global_px)`` is used for every plane, so this is
    deliberately independent of the optical plane: registered planes share XY.
    """
    try:
        x = int(round(float(candidate.get("x_global_px")))) - int(origin[1])
        y = int(round(float(candidate.get("y_global_px")))) - int(origin[0])
    except (TypeError, ValueError):
        return None
    return x, y


def count_supported_on_plane(candidates: Iterable[dict], plane_index: int) -> int:
    return sum(
        1 for c in candidates
        if candidate_plane_state(c, plane_index) in ("peak", "support")
    )


# --------------------------------------------------------------------------- #
# Native-resolution drawing (Pillow)
# --------------------------------------------------------------------------- #
def _marker_radius(shape) -> int:
    longest = max(shape[:2])
    return max(4, int(round(longest / 600)))


def _draw_symbol(draw, symbol, x, y, r, colour, width):
    """Draw one status symbol at native resolution."""
    box = [x - r, y - r, x + r, y + r]
    if symbol == "o":
        draw.ellipse(box, outline=colour, width=width)
    elif symbol == "s":
        draw.rectangle(box, outline=colour, width=width)
    elif symbol == "D":
        draw.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)],
                     outline=colour, width=width)
    elif symbol == "^":
        draw.polygon([(x, y - r), (x + r, y + r), (x - r, y + r)],
                     outline=colour, width=width)
    elif symbol == "x":
        draw.line([(x - r, y - r), (x + r, y + r)], fill=colour, width=width)
        draw.line([(x - r, y + r), (x + r, y - r)], fill=colour, width=width)
    elif symbol == "+":
        draw.line([(x - r, y), (x + r, y)], fill=colour, width=width)
        draw.line([(x, y - r), (x, y + r)], fill=colour, width=width)
    elif symbol == ".":
        dot = max(1, r // 2)
        draw.ellipse([x - dot, y - dot, x + dot, y + dot], fill=colour)
    else:  # pragma: no cover - defensive
        draw.ellipse(box, outline=colour, width=width)


def render_plane_overlay(display8, candidates, plane_index, mode, *, origin=(0, 0)):
    """Draw status markers for one plane onto a full-res RGB copy.

    ``display8`` is a 2-D uint8 windowed image (already at native resolution).
    Returns ``(rgb_uint8, candidates_displayed, candidates_supported)``. The
    input ``display8`` is not modified.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    height, width = display8.shape[:2]
    base = Image.fromarray(np.repeat(display8[:, :, None], 3, axis=2), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    base_radius = _marker_radius(display8.shape)

    displayed = 0
    supported = 0
    for candidate in candidates:
        state = candidate_plane_state(candidate, plane_index)
        if state in ("peak", "support"):
            supported += 1
        spec = marker_spec(state, mode)
        if spec is None:
            continue
        position = candidate_draw_xy(candidate, origin)
        if position is None:
            continue
        x, y = position
        if not (0 <= x < width and 0 <= y < height):
            continue
        rgb = status_rgb(candidate.get("current_status"))
        colour = (rgb[0], rgb[1], rgb[2], int(spec["alpha"]))
        radius = max(2, int(round(base_radius * spec["scale"])))
        _draw_symbol(draw, status_symbol(candidate.get("current_status")),
                     x, y, radius, colour, int(spec["width"]))
        if spec.get("centre_dot"):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=colour)
        displayed += 1

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(out), displayed, supported


# --------------------------------------------------------------------------- #
# Montage (explicit downscaled overview -- never replaces the full-res planes)
# --------------------------------------------------------------------------- #
def build_montage(tiles, labels, *, columns=4, tile_px=1000, background=(20, 20, 20)):
    """Compose labelled, downscaled tiles into a grid overview image (RGB array)."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    images = []
    for tile in tiles:
        arr = np.asarray(tile)
        img = Image.fromarray(arr, mode="RGB" if arr.ndim == 3 else "L").convert("RGB")
        w, h = img.size
        scale = min(1.0, tile_px / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             resample=Image.LANCZOS)
        images.append(img)

    cell_w = max(img.size[0] for img in images)
    cell_h = max(img.size[1] for img in images)
    pad, label_h = 6, 18
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * pad,
         rows * (cell_h + label_h) + (rows + 1) * pad),
        background,
    )
    draw = ImageDraw.Draw(canvas)
    for i, img in enumerate(images):
        col, row = i % columns, i // columns
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + label_h + pad)
        draw.text((x, y), str(labels[i]), fill=(255, 255, 255))
        canvas.paste(img, (x, y + label_h))
    return np.asarray(canvas)


# --------------------------------------------------------------------------- #
# Display window for the whole section (memory-safe, per-plane reads)
# --------------------------------------------------------------------------- #
def _read_plane(path):
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        try:
            image = page.asarray(out="memmap")
        except (TypeError, ValueError):
            image = page.asarray()
        return np.asarray(image)


def section_display_window(ordered_planes, settings, *, minimum_pixels=50,
                           padding_values=(0.0,)):
    """Choose a single display window for the whole section (read-only).

    ``fixed`` mode (e.g. the Fiji 0-513 channel-2 view) ignores the pixels. Any
    percentile mode accumulates a running max projection across planes -- never
    loading the whole stack at once -- and reuses the existing
    :func:`qc_display.compute_display_limits`.
    """
    import numpy as np  # noqa: PLC0415

    from .qc_display import compute_display_limits

    mode = str(getattr(settings, "mode", "robust_tissue_percentile"))
    if mode == "fixed":
        return {
            "display_mode": "fixed",
            "display_min": float(settings.minimum),
            "display_max": float(settings.maximum),
        }
    running_max = None
    for _plane, path in ordered_planes:
        plane = _read_plane(path)
        running_max = plane.copy() if running_max is None else np.maximum(running_max, plane)
    info = compute_display_limits(
        running_max, settings, padding_values=padding_values,
        minimum_pixels=minimum_pixels,
    )
    return {
        "display_mode": info.get("display_mode", mode),
        "display_min": float(info["display_min"]),
        "display_max": float(info["display_max"]),
    }


def plane_display_window(plane, settings, *, minimum_pixels=50, padding_values=(0.0,),
                         tissue_mask=None, injection_core_mask=None,
                         exclude_injection_core=True):
    """Robust display window computed independently for ONE plane (read-only).

    Individual planes are dimmer than the section max projection, so this uses a
    robust per-plane window: percentiles over finite, in-tissue, non-black-
    background pixels, with the injection core excluded from the upper estimate
    (the robust upper percentile also drops the saturated injection tail when no
    core mask is available). It never alters the raw plane or any measurement.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    from .qc_display import compute_display_limits

    robust = SimpleNamespace(
        mode="robust_tissue_percentile",
        lower_percentile=float(getattr(settings, "lower_percentile", 0.5)),
        upper_percentile=float(getattr(settings, "upper_percentile", 99.7)),
        minimum=float(getattr(settings, "minimum", 0.0)),
        maximum=float(getattr(settings, "maximum", 513.0)),
    )
    info = compute_display_limits(
        plane, robust, tissue_mask=tissue_mask,
        injection_core_mask=injection_core_mask, padding_values=padding_values,
        minimum_pixels=minimum_pixels, exclude_injection_core=exclude_injection_core,
    )
    return {
        "display_mode": "per_plane_robust",
        "display_min": float(info["display_min"]),
        "display_max": float(info["display_max"]),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def render_section_seven_planes(
    channel_index, section, candidates, output_dir, *,
    channel, display_settings, mode="all", display_mode="configured",
    display_override=None, minimum_pixels=50, padding_values=(0.0,),
    planes_per_section=7, montage_columns=4, montage_tile_px=1000,
):
    """Render the seven full-resolution plane PNGs + the two montages.

    ``mode`` is the marker mode (``all`` or ``support``/``support_only``).
    ``display_mode`` is ``configured`` (one section window from the channel's QC
    display config) or ``per_plane_robust`` (a robust window per plane). A
    ``display_override`` ``(min, max)`` forces a fixed window on every plane.

    Returns a dict with ``plane_files``, ``montage``, ``raw_montage``,
    ``metadata_rows`` and the chosen ``display_min``/``display_max``.
    """
    import csv  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    mode = normalize_marker_mode(mode)
    if mode not in _MARKER_SPECS:
        raise ValueError(f"Unknown marker mode: {mode!r}")
    if display_mode not in ("configured", "per_plane_robust"):
        raise ValueError(f"Unknown display mode: {display_mode!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = ordered_section_planes(channel_index, section)
    if not ordered:
        raise FileNotFoundError(
            f"No TIFF planes found for channel {channel!r}, section {section}."
        )
    found = [plane for plane, _ in ordered]
    missing = [p for p in range(1, planes_per_section + 1) if p not in found]
    if missing:
        raise FileNotFoundError(
            f"Section {section} ({channel}) is missing optical planes {missing}; "
            f"refusing to render an incomplete seven-plane set."
        )

    section_candidates = select_candidate_rows(candidates, channel, section)

    # A single section window is reused for every plane unless a robust per-plane
    # window is requested. ``per_plane`` windows are computed inside the loop.
    per_plane = display_mode == "per_plane_robust" and display_override is None
    if display_override is not None:
        section_window = {
            "display_mode": "fixed_override",
            "display_min": float(display_override[0]),
            "display_max": float(display_override[1]),
        }
    elif per_plane:
        section_window = None
    else:
        section_window = section_display_window(
            ordered, display_settings,
            minimum_pixels=minimum_pixels, padding_values=padding_values,
        )

    plane_files = []
    candidate_tiles = []
    raw_tiles = []
    tile_labels = []
    metadata_rows = []
    window = section_window

    for plane_number, path in ordered:
        plane = _read_plane(path)                      # native res, original dtype
        source_h, source_w = int(plane.shape[0]), int(plane.shape[1])
        if per_plane:
            window = plane_display_window(
                plane, display_settings,
                minimum_pixels=minimum_pixels, padding_values=padding_values,
            )
        dmin, dmax = window["display_min"], window["display_max"]
        display8 = apply_window_uint8(plane, dmin, dmax)   # copy; plane untouched
        rgb, displayed, supported = render_plane_overlay(
            display8, section_candidates, plane_number - 1, mode,
        )
        filename = f"plane_{plane_number:02d}_candidates_fullres.png"
        save_png_fullres(output_dir / filename, rgb)        # lossless, no resize

        saved_h, saved_w = int(rgb.shape[0]), int(rgb.shape[1])
        candidate_tiles.append(rgb)
        raw_tiles.append(np.repeat(display8[:, :, None], 3, axis=2))
        tile_labels.append(f"plane {plane_number:02d}")
        plane_files.append(output_dir / filename)
        metadata_rows.append({
            "filename": filename,
            "source_tiff_path": str(path),
            "channel": channel,
            "section": int(section),
            "optical_plane": plane_number,
            "source_width": source_w,
            "source_height": source_h,
            "saved_width": saved_w,
            "saved_height": saved_h,
            "display_mode": window["display_mode"],
            "display_min": dmin,
            "display_max": dmax,
            "render_mode": mode,
            "candidates_displayed": displayed,
            "candidates_supported_on_plane": supported,
            "file_format": "PNG",
            "resizing_occurred": (saved_w != source_w) or (saved_h != source_h),
        })

    montage_path = output_dir / "seven_plane_candidate_montage.png"
    raw_montage_path = output_dir / "seven_plane_raw_montage.png"
    save_png_fullres(montage_path, build_montage(
        candidate_tiles, tile_labels, columns=montage_columns, tile_px=montage_tile_px))
    save_png_fullres(raw_montage_path, build_montage(
        raw_tiles, tile_labels, columns=montage_columns, tile_px=montage_tile_px))

    metadata_csv = output_dir / "seven_plane_qc_metadata.csv"
    with open(metadata_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SEVEN_PLANE_QC_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    if per_plane:
        mins = [row["display_min"] for row in metadata_rows]
        maxs = [row["display_max"] for row in metadata_rows]
        report = {
            "display_mode": "per_plane_robust",
            "display_min": min(mins) if mins else 0.0,
            "display_max": max(maxs) if maxs else 1.0,
        }
    else:
        report = section_window
    return {
        "plane_files": plane_files,
        "montage": montage_path,
        "raw_montage": raw_montage_path,
        "metadata_csv": metadata_csv,
        "metadata_rows": metadata_rows,
        "display_min": report["display_min"],
        "display_max": report["display_max"],
        "display_mode": report["display_mode"],
        "candidate_count": len(section_candidates),
    }
