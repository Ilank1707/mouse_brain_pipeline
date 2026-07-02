"""Native-resolution QC outputs (Part 4).

The source TIFFs are high quality, so the QC here is kept separate from the small
matplotlib preview figures:

  * a lossless native 16-bit max projection (no stretch, no markers);
  * a full-resolution display PNG (the chosen window applied to the full array);
  * full-resolution candidate overlays drawn at the source pixel size;
  * small previews (clearly labelled), which never replace the full-res files;
  * a metadata row per saved file (real dims, dtype, window, resizing, interp).

Nothing here touches the raw TIFFs. Display files only ever apply a window to a
copy of the projection.
"""

from __future__ import annotations

QC_IMAGE_METADATA_COLUMNS = [
    "filename",
    "channel",
    "section",
    "source_width",
    "source_height",
    "saved_width",
    "saved_height",
    "source_dtype",
    "saved_dtype",
    "projection_method",
    "display_mode",
    "display_min",
    "display_max",
    "file_format",
    "resizing_occurred",
    "interpolation",
]

# Marker colours per status (RGB), matching the preview figures.
_STATUS_RGB = {
    "preliminary_rule_pass": (57, 255, 20),
    "manual_review": (255, 225, 0),
    "invalid_measurement": (0, 217, 255),
    "suspect_injection_mask": (255, 127, 14),
    "injection_site": (255, 45, 45),
    "preliminary_rule_fail": (158, 158, 158),
    "artifact": (192, 76, 255),
}
_DEFAULT_RGB = (255, 255, 255)
_PREVIEW_MAX_DIM = 2000


def native_max_projection(stack_zyx):
    """Max projection over z in the source dtype, upcast only if it would clip.

    Max projection can't exceed the input range, so uint16 stays uint16. We still
    guard and upcast (with a flag) rather than clip silently.
    """
    import numpy as np  # noqa: PLC0415

    stack = np.asarray(stack_zyx)
    proj = stack.max(axis=0)
    source_dtype = str(stack.dtype)
    method = "max_intensity_projection_over_z"
    if np.issubdtype(stack.dtype, np.integer):
        out = proj.astype(stack.dtype)
        return out, source_dtype, method, False
    # Float source: keep integer precision if it fits uint16, else preserve float.
    finite_max = float(np.nanmax(proj)) if proj.size else 0.0
    if finite_max <= 65535 and np.all(proj == np.round(proj)):
        return proj.astype(np.uint16), source_dtype, method, False
    if finite_max > 65535:
        return proj.astype(np.uint32), source_dtype, method + "_upcast_uint32", True
    return proj.astype(np.float32), source_dtype, method, False


def apply_window_uint8(projection, display_min, display_max):
    """Apply a display window to the full-res array and return a uint8 image.

    Operates on a copy; the input projection is not modified.
    """
    import numpy as np  # noqa: PLC0415

    values = np.asarray(projection, dtype=np.float64)
    span = max(float(display_max) - float(display_min), 1e-6)
    scaled = np.clip((values - float(display_min)) / span, 0.0, 1.0)
    return (scaled * 255.0 + 0.5).astype(np.uint8)


def save_native_projection_tiff(path, projection, *, channel, section,
                                projection_method, source_dtype):
    """Save the native projection losslessly with a short description."""
    import tifffile  # noqa: PLC0415

    description = (
        f"channel={channel}; section={section}; projection={projection_method}; "
        f"source_dtype={source_dtype}; no display stretch; not a cell count"
    )
    # zlib is lossless; keeps the 16-bit data exact.
    tifffile.imwrite(str(path), projection, compression="zlib", description=description)
    return path


def _to_rgb(display8):
    import numpy as np  # noqa: PLC0415

    return np.repeat(display8[:, :, None], 3, axis=2)


def _marker_radius(projection_shape):
    # Scale markers to the image so they stay visible at full res but not huge.
    longest = max(projection_shape)
    return max(4, int(round(longest / 600)))


def draw_candidate_overlay(display8, candidates, crop_origin, masks=()):
    """Draw status-coloured candidate rings on a full-res RGB copy.

    Markers are hollow rings so the underlying cell stays visible. Mask contours
    are drawn as thin coloured lines. Nothing is resized or smoothed.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    rgb = _to_rgb(display8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    height, width = display8.shape[:2]
    oy, ox = crop_origin

    for mask, colour in masks:
        if mask is None:
            continue
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        from scipy import ndimage as ndi  # noqa: PLC0415

        edge = mask & ~ndi.binary_erosion(mask)
        ys, xs = np.nonzero(edge)
        for y, x in zip(ys.tolist(), xs.tolist()):
            image.putpixel((x, y), colour)

    radius = _marker_radius(display8.shape)
    for c in candidates:
        try:
            x = int(float(c.get("x_global_px"))) - int(ox)
            y = int(float(c.get("y_global_px"))) - int(oy)
        except (TypeError, ValueError):
            continue
        if not (0 <= x < width and 0 <= y < height):
            continue
        colour = _STATUS_RGB.get(c.get("current_status"), _DEFAULT_RGB)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     outline=colour, width=2)
    return np.asarray(image)


def save_png_fullres(path, image_array):
    """Save a full-resolution lossless PNG with no resampling."""
    from PIL import Image  # noqa: PLC0415

    mode = "RGB" if image_array.ndim == 3 else "L"
    Image.fromarray(image_array, mode=mode).save(str(path), format="PNG")
    return path


def save_preview_png(path, image_array, max_dim=_PREVIEW_MAX_DIM):
    """Save a downscaled preview PNG. Returns (path, saved_w, saved_h)."""
    from PIL import Image  # noqa: PLC0415

    mode = "RGB" if image_array.ndim == 3 else "L"
    image = Image.fromarray(image_array, mode=mode)
    width, height = image.size
    scale = min(1.0, max_dim / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resample=Image.LANCZOS,
        )
    image.save(str(path), format="PNG")
    return path, image.size[0], image.size[1]


def metadata_row(*, filename, channel, section, source_shape, saved_shape,
                 source_dtype, saved_dtype, projection_method, display_mode,
                 display_min, display_max, file_format, resizing, interpolation):
    """Build one QC image metadata row."""
    sh, sw = int(source_shape[0]), int(source_shape[1])
    vh, vw = int(saved_shape[0]), int(saved_shape[1])
    return {
        "filename": filename,
        "channel": channel,
        "section": section,
        "source_width": sw,
        "source_height": sh,
        "saved_width": vw,
        "saved_height": vh,
        "source_dtype": source_dtype,
        "saved_dtype": saved_dtype,
        "projection_method": projection_method,
        "display_mode": display_mode,
        "display_min": display_min,
        "display_max": display_max,
        "file_format": file_format,
        "resizing_occurred": bool(resizing),
        "interpolation": interpolation,
    }
