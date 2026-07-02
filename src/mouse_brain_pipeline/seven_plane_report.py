"""Peak-assigned seven-plane QC reports in the style of ``04_candidates_*.png``.

Each candidate is a 3D object and is drawn on **exactly one** of the seven main
images -- its canonical peak plane (``fixed_xy_peak_z_index``) -- so the seven
plane counts never double-count. A separate, clearly-labelled *support
visualisation* may draw a candidate on every supported plane and must not be
summed.

For every plane two images are written:

* ``plane_0X_peak_assigned_native.png`` -- the brain at the **original** TIFF
  width/height with markers drawn at native resolution (no resize, lossless);
* ``plane_0X_peak_assigned_qc.png`` -- the native brain with **white header /
  footer** bands carrying the title, the count summary and a legend with
  per-plane counts (matching the existing QC figure's status colours/symbols).

Nothing here modifies the raw TIFFs, candidate coordinates, counts or statuses.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .coordinate_exports import (
    assign_peak_planes,
    reconcile,
    support_optical_planes,
    write_coordinate_exports,
    write_count_summaries,
)
from .qc_native import apply_window_uint8, save_png_fullres
from .review import read_csv_rows
from .seven_plane_qc import (
    _draw_symbol,
    _marker_radius,
    _read_plane,
    candidate_draw_xy,
    classify_run_crop,
    crop_covers_full_image,
    ordered_section_planes,
    plane_display_window,
    section_display_window,
    status_rgb,
    status_symbol,
)

# Status -> human label, matching candidate_qc._STATUS_STYLE legend labels.
STATUS_LABELS = {
    "preliminary_rule_pass": "preliminary rule pass",
    "preliminary_rule_fail": "preliminary rule fail",
    "manual_review": "manual review",
    "invalid_measurement": "invalid measurement",
    "suspect_injection_mask": "suspect automatic injection mask",
    "injection_site": "confirmed injection",
    "artifact": "Cellfinder artefact/outlier",
}

SEVEN_PLANE_REPORT_METADATA_COLUMNS = [
    "filename", "kind", "channel", "section", "optical_plane",
    "source_tiff_path", "source_width", "source_height",
    "native_saved_width", "native_saved_height",
    "display_mode", "display_min", "display_max",
    "candidates_assigned", "candidates_supported", "resizing_occurred",
]


class RenderRefusedError(RuntimeError):
    """Raised when a render is refused (bad/mismatched run inputs)."""


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


# --------------------------------------------------------------------------- #
# Header / legend text (pure, testable)
# --------------------------------------------------------------------------- #
def assigned_count_breakdown(assigned) -> dict:
    """Counts among the candidates assigned to one plane."""
    inside = sum(1 for c in assigned if _truthy(c.get("inside_injection_analysis_exclusion")))
    return {
        "assigned": len(assigned),
        "inside_injection": inside,
        "outside_injection": len(assigned) - inside,
        "manual_review": sum(1 for c in assigned if c.get("current_status") == "manual_review"),
        "invalid_measurement": sum(
            1 for c in assigned if c.get("current_status") == "invalid_measurement"),
        "invalid_coordinate": sum(1 for c in assigned if _truthy(c.get("invalid_coordinate"))),
    }


def peak_plane_header_lines(channel, section, optical_plane, assigned,
                            unique_total, window) -> list[str]:
    """The white-header text for one peak-assigned plane image."""
    b = assigned_count_breakdown(assigned)
    return [
        f"{channel} section {int(section):03d} - optical plane {int(optical_plane):02d}",
        "PROVISIONAL 3D candidates assigned by fixed-XY peak plane",
        f"unique candidates in full 7-plane stack: {unique_total}",
        f"assigned to this plane: {b['assigned']}",
        f"inside injection among assigned: {b['inside_injection']}",
        f"outside injection among assigned: {b['outside_injection']}",
        f"manual review among assigned: {b['manual_review']}",
        f"invalid measurement among assigned: {b['invalid_measurement']}",
        f"invalid coordinate among assigned: {b['invalid_coordinate']}",
        f"display: {window['display_mode']} window "
        f"[{window['display_min']:.0f}, {window['display_max']:.0f}]",
    ]


def peak_plane_legend_entries(assigned) -> list[tuple]:
    """Legend rows ``(label, rgb, symbol, count)`` for statuses on this plane."""
    from collections import Counter

    counts = Counter(c.get("current_status") for c in assigned)
    entries = []
    for status, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        label = STATUS_LABELS.get(status, str(status))
        entries.append((label, status_rgb(status), status_symbol(status), count))
    return entries


def support_header_lines(channel, section, optical_plane, visible_count) -> list[str]:
    return [
        f"{channel} section {int(section):03d} - optical plane {int(optical_plane):02d} "
        f"- SUPPORT VISUALIZATION",
        "SUPPORT VISUALIZATION - candidates can appear on multiple planes.",
        "DO NOT SUM THESE PLANE COUNTS.",
        f"candidates visible/supported on this plane: {visible_count}",
    ]


# --------------------------------------------------------------------------- #
# Native-resolution drawing + QC-report composition (Pillow)
# --------------------------------------------------------------------------- #
def render_assigned_overlay(display8, candidates, *, origin=(0, 0)):
    """Draw each given candidate strongly (status colour + symbol) at native res.

    Returns ``(rgb_uint8, n_drawn)``. The input ``display8`` is not modified.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    height, width = display8.shape[:2]
    base = Image.fromarray(np.repeat(display8[:, :, None], 3, axis=2), "RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(3, int(round(_marker_radius(display8.shape) * 1.2)))
    drawn = 0
    for candidate in candidates:
        position = candidate_draw_xy(candidate, origin)
        if position is None:
            continue
        x, y = position
        if not (0 <= x < width and 0 <= y < height):
            continue
        rgb = status_rgb(candidate.get("current_status"))
        colour = (rgb[0], rgb[1], rgb[2], 255)
        _draw_symbol(draw, status_symbol(candidate.get("current_status")),
                     x, y, radius, colour, 2)
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=colour)
        drawn += 1
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB")), drawn


def _load_font(size):
    from PIL import ImageFont  # noqa: PLC0415

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def compose_qc_report(native_rgb, header_lines, legend_entries, footer_lines):
    """Native brain + white header/footer bands (title, counts, legend).

    The brain image is kept at native resolution; white space is added around it
    rather than shrinking it. Returns an RGB uint8 array.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    arr = np.asarray(native_rgb)
    height, width = arr.shape[:2]
    font_size = max(14, width // 90)
    title_size = int(font_size * 1.35)
    pad = max(6, font_size // 2)
    line_h = font_size + pad
    font = _load_font(font_size)
    title_font = _load_font(title_size)

    header_h = pad + title_size + pad + line_h * max(0, len(header_lines) - 1) + pad
    footer_rows = 1 + len(legend_entries) + len(footer_lines)
    footer_h = pad + line_h * footer_rows + pad

    canvas = Image.new("RGB", (width, header_h + height + footer_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = pad
    if header_lines:
        draw.text((pad, y), header_lines[0], fill=(0, 0, 0), font=title_font)
        y += title_size + pad
    for line in header_lines[1:]:
        draw.text((pad, y), line, fill=(25, 25, 25), font=font)
        y += line_h

    canvas.paste(Image.fromarray(arr, "RGB"), (0, header_h))

    y = header_h + height + pad
    draw.text((pad, y), "Legend (counts assigned to THIS plane):", fill=(0, 0, 0), font=font)
    y += line_h
    glyph_r = max(4, font_size // 3)
    for label, rgb, symbol, count in legend_entries:
        cx, cy = pad + glyph_r + 2, y + font_size // 2
        _draw_symbol(draw, symbol, cx, cy, glyph_r, (rgb[0], rgb[1], rgb[2]), 2)
        draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=(rgb[0], rgb[1], rgb[2]))
        draw.text((pad + 2 * glyph_r + 10, y), f"{label} ({count})", fill=(20, 20, 20), font=font)
        y += line_h
    for line in footer_lines:
        draw.text((pad, y), line, fill=(90, 90, 90), font=font)
        y += line_h
    return np.asarray(canvas)


def _save_preview(path, rgb, max_dim=2200):
    from PIL import Image  # noqa: PLC0415

    img = Image.fromarray(rgb, "RGB")
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    img.save(str(path), format="PNG")
    return path


# --------------------------------------------------------------------------- #
# Run-scoped orchestration + validation
# --------------------------------------------------------------------------- #
def _channel_dir(config, channel):
    return (config.data.green_signal_dir if channel == "green_signal"
            else config.data.channel_2_signal_dir)


def render_run(run_dir, channel, section, *, config, display_mode="per_plane_robust",
               display_override=None, allow_cropped=False, make_preview=True,
               planes_per_section=7, subdir=None, write_exports=True):
    """Render the seven peak-assigned QC images for ONE run directory.

    ``subdir`` (e.g. a channel name) nests the image / export output one level
    deeper so a two-channel run does not collide; ``None`` writes the flat layout
    the standalone single-channel CLI uses. ``write_exports`` controls whether
    the coordinate exports + count summaries are (re)written here.

    Reads ONLY ``<run_dir>/all_candidates.csv`` and
    ``<run_dir>/candidate_run_metadata.json`` -- never another folder. Refuses
    (raising :class:`RenderRefusedError`) when the metadata is missing, the CSV
    is from a crop while a full section was requested, the section/channel do not
    match, the TIFF dimensions disagree with the metadata, or the counts do not
    reconcile.
    """
    from mouse_brain_pipeline.audit import index_channel, read_shape_dtype  # noqa: PLC0415

    run_dir = Path(run_dir)
    csv_path = run_dir / "all_candidates.csv"
    meta_path = run_dir / "candidate_run_metadata.json"
    if not meta_path.is_file():
        raise RenderRefusedError(f"run metadata missing: {meta_path}")
    if not csv_path.is_file():
        raise RenderRefusedError(f"candidate table missing: {csv_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    rows = read_csv_rows(csv_path)
    section_rows = [
        r for r in rows
        if r.get("channel") == channel and str(r.get("section")) == str(section)
    ]

    index = index_channel(channel, _channel_dir(config, channel), config.data.filename_regex)
    ordered = ordered_section_planes(index, section)
    if not ordered:
        raise RenderRefusedError(
            f"no TIFF planes for channel {channel!r}, section {section} "
            f"under {_channel_dir(config, channel)!r}")
    found = [p for p, _ in ordered]
    missing = [p for p in range(1, planes_per_section + 1) if p not in found]
    if missing:
        raise RenderRefusedError(f"section {section} ({channel}) missing planes {missing}")
    shape, _dtype = read_shape_dtype(ordered[0][1])
    image_height = int(shape[0]) if shape else None
    image_width = int(shape[1]) if shape else None

    # ---- refusal checks ----
    crop = metadata.get("crop_x_min_x_max_y_min_y_max")
    if classify_run_crop(metadata) == "crop" and \
            not crop_covers_full_image(crop, image_width, image_height) and not allow_cropped:
        raise RenderRefusedError(
            f"candidate CSV is from a CROPPED run (crop={crop}); refusing a "
            f"full-section render. Pass allow_cropped=True to override.")
    processed = [int(s) for s in (metadata.get("processed_sections") or [])]
    if processed and int(section) not in processed:
        raise RenderRefusedError(
            f"section {section} not in run's processed_sections {processed}")
    known_channels = set(metadata.get("candidate_counts_by_channel") or {})
    known_channels |= set(metadata.get("effective_cellfinder_by_channel") or {})
    if known_channels and channel not in known_channels:
        raise RenderRefusedError(f"channel {channel!r} not present in this run")
    recorded_dims = (metadata.get("source_image_dimensions") or {}).get(channel)
    if recorded_dims and image_width and image_height:
        rec_h, rec_w = int(recorded_dims["height"]), int(recorded_dims["width"])
        if (rec_h, rec_w) != (image_height, image_width):
            raise RenderRefusedError(
                f"TIFF dimensions {image_width}x{image_height} do not match run "
                f"metadata {rec_w}x{rec_h}")

    rec = reconcile(section_rows, planes_per_section)
    if not (rec["status_reconciles"] and rec["peak_assignment_reconciles"]):
        raise RenderRefusedError(
            f"candidate counts do not reconcile: {rec}")

    # ---- display windows ----
    settings = config.qc_display.for_channel(channel)
    minimum_pixels = getattr(config.qc_display, "minimum_pixels", 50)
    padding_values = tuple(config.detection.padding_values)
    per_plane = display_mode == "per_plane_robust" and display_override is None
    if display_override is not None:
        section_window = {"display_mode": "fixed_override",
                          "display_min": float(display_override[0]),
                          "display_max": float(display_override[1])}
    elif per_plane:
        section_window = None
    else:
        section_window = section_display_window(
            ordered, settings, minimum_pixels=minimum_pixels, padding_values=padding_values)

    qc_dir = run_dir / "seven_plane_qc" / subdir if subdir else run_dir / "seven_plane_qc"
    support_dir = qc_dir / "support_views"
    qc_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)

    assignments, unassigned = assign_peak_planes(section_rows, planes_per_section)
    unique_total = len(section_rows)
    metadata_rows = []
    main_files = {}

    for plane_number, path in ordered:
        plane = _read_plane(path)
        source_h, source_w = int(plane.shape[0]), int(plane.shape[1])
        window = (plane_display_window(plane, settings, minimum_pixels=minimum_pixels,
                                       padding_values=padding_values)
                  if per_plane else section_window)
        display8 = apply_window_uint8(plane, window["display_min"], window["display_max"])

        assigned = assignments.get(plane_number, [])
        native_rgb, n_drawn = render_assigned_overlay(display8, assigned)
        native_name = f"plane_{plane_number:02d}_peak_assigned_native.png"
        save_png_fullres(qc_dir / native_name, native_rgb)

        header = peak_plane_header_lines(channel, section, plane_number, assigned,
                                         unique_total, window)
        legend = peak_plane_legend_entries(assigned)
        footer = [f"unique 3D candidates across all seven planes: {unique_total}",
                  "PROVISIONAL candidate detections - NOT final cells."]
        report = compose_qc_report(native_rgb, header, legend, footer)
        qc_name = f"plane_{plane_number:02d}_peak_assigned_qc.png"
        save_png_fullres(qc_dir / qc_name, report)
        if make_preview:
            _save_preview(qc_dir / f"plane_{plane_number:02d}_peak_assigned_qc_preview.png", report)
        main_files[plane_number] = qc_dir / native_name

        # Support visualisation (may repeat candidates across planes).
        support = [c for c in section_rows
                   if plane_number in support_optical_planes(c, planes_per_section)]
        support_rgb, n_support = render_assigned_overlay(display8, support)
        save_png_fullres(support_dir / f"plane_{plane_number:02d}_support_native.png", support_rgb)
        support_report = compose_qc_report(
            support_rgb, support_header_lines(channel, section, plane_number, n_support),
            peak_plane_legend_entries(support),
            ["SUPPORT VISUALIZATION - DO NOT SUM these plane counts as unique cells."])
        save_png_fullres(support_dir / f"plane_{plane_number:02d}_support_qc.png", support_report)

        metadata_rows.append({
            "filename": native_name, "kind": "peak_assigned_native",
            "channel": channel, "section": int(section), "optical_plane": plane_number,
            "source_tiff_path": str(path), "source_width": source_w, "source_height": source_h,
            "native_saved_width": int(native_rgb.shape[1]),
            "native_saved_height": int(native_rgb.shape[0]),
            "display_mode": window["display_mode"],
            "display_min": window["display_min"], "display_max": window["display_max"],
            "candidates_assigned": n_drawn, "candidates_supported": n_support,
            "resizing_occurred": (int(native_rgb.shape[1]) != source_w)
            or (int(native_rgb.shape[0]) != source_h),
        })

    meta_csv = qc_dir / "seven_plane_qc_metadata.csv"
    with open(meta_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SEVEN_PLANE_REPORT_METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    export_counts = summaries = None
    if write_exports:
        export_dir = (run_dir / "coordinate_exports" / subdir if subdir
                      else run_dir / "coordinate_exports")
        export_counts = write_coordinate_exports(
            export_dir, section_rows,
            channel=channel, section=section, planes_per_section=planes_per_section)
        summaries = write_count_summaries(
            export_dir if subdir else run_dir, section_rows,
            channel=channel, section=section, planes_per_section=planes_per_section)

    return {
        "run_dir": run_dir,
        "qc_dir": qc_dir,
        "support_dir": support_dir,
        "metadata_csv": meta_csv,
        "metadata_rows": metadata_rows,
        "reconciliation": rec,
        "export_counts": export_counts,
        "summaries": summaries,
        "image_width": image_width,
        "image_height": image_height,
        "unique_total": unique_total,
        "unassigned": len(unassigned),
        "main_files": main_files,
    }
