"""Downsampled PNG previews for visual QC.

* Reads planes lazily (memory-mapped via tifffile) and immediately downsamples.
* Percentile-based display scaling -- the SOURCE 16-bit data is never altered.
* Produces single-channel images plus a two-channel overlay
  (green_signal -> green, channel_2_signal -> magenta).
* A low-frequency "anatomical-like" preview can be made from a signal channel
  for visual inspection only -- it is NOT a validated background channel.

Heavy dependencies (numpy, tifffile, Pillow) are imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import Config
from .filenames import global_plane, z_um
from .utilities import LOG, ensure_dir


@dataclass
class PreviewSpec:
    section: int
    plane: int
    green_path: Path | None
    channel_2_path: Path | None


def _lazy_read_downsampled(path: Path, factor: int):
    """Memory-map a TIFF and return a strided (downsampled) numpy array.

    Strided slicing on a memmap avoids reading every pixel into RAM.
    """
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    with tifffile.TiffFile(str(path)) as tf:
        try:
            arr = tf.pages[0].asarray(out="memmap")
        except (ValueError, TypeError):  # some codecs do not support memmap
            arr = tf.pages[0].asarray()
        sub = arr[::factor, ::factor]
        return np.asarray(sub)


def _percentile_scale(arr, low: float = 1.0, high: float = 99.5):
    """Scale to 0..255 uint8 using display percentiles (source untouched)."""
    import numpy as np  # noqa: PLC0415

    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def make_preview(
    spec: PreviewSpec,
    out_dir: Path,
    config: Config,
    downsample: int = 16,
    low: float = 1.0,
    high: float = 99.5,
    dry_run: bool = False,
) -> list[Path]:
    """Write single-channel + overlay PNGs for one (section, plane)."""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    ensure_dir(out_dir)
    gp = global_plane(spec.section, config.pilot.first_section or spec.section, spec.plane,
                      config.acquisition.planes_per_section)
    zz = z_um(gp, config.acquisition.voxel_size_z_um)
    tag = f"section_{spec.section:03d}_plane_{spec.plane:02d}_z{zz:.0f}um"
    written: list[Path] = []

    if dry_run:
        LOG.info("[dry-run] would write previews for %s (downsample=%d)", tag, downsample)
        return [out_dir / f"{tag}_green.png", out_dir / f"{tag}_channel2.png", out_dir / f"{tag}_overlay.png"]

    green_u8 = ch2_u8 = None
    shape = None
    if spec.green_path is not None:
        g = _lazy_read_downsampled(spec.green_path, downsample)
        green_u8 = _percentile_scale(g, low, high)
        shape = green_u8.shape
        p = out_dir / f"{tag}_green.png"
        Image.fromarray(green_u8).save(p)
        written.append(p)
    if spec.channel_2_path is not None:
        c = _lazy_read_downsampled(spec.channel_2_path, downsample)
        ch2_u8 = _percentile_scale(c, low, high)
        shape = shape or ch2_u8.shape
        p = out_dir / f"{tag}_channel2.png"
        Image.fromarray(ch2_u8).save(p)
        written.append(p)

    # Two-channel overlay: green channel in G, channel_2 in R+B (magenta).
    if shape is not None:
        h, w = shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        if green_u8 is not None:
            rgb[..., 1] = green_u8[:h, :w]
        if ch2_u8 is not None:
            rgb[..., 0] = ch2_u8[:h, :w]
            rgb[..., 2] = ch2_u8[:h, :w]
        p = out_dir / f"{tag}_overlay.png"
        Image.fromarray(rgb).save(p)
        written.append(p)
    LOG.info("Wrote %d preview(s) for %s", len(written), tag)
    return written


def select_preview_specs(
    manifest_rows: Sequence[dict], n: int = 5
) -> list[PreviewSpec]:
    """Pick ~n evenly spaced planes across the manifest for representative previews."""
    rows = [r for r in manifest_rows if r.get("green_signal_file") or r.get("channel_2_signal_file")]
    if not rows:
        return []
    if len(rows) <= n:
        chosen = rows
    else:
        step = len(rows) / n
        chosen = [rows[int(i * step)] for i in range(n)]
    specs = []
    for r in chosen:
        specs.append(
            PreviewSpec(
                section=int(r["section"]),
                plane=int(r["plane"]),
                green_path=Path(r["green_signal_file"]) if r.get("green_signal_file") else None,
                channel_2_path=Path(r["channel_2_signal_file"]) if r.get("channel_2_signal_file") else None,
            )
        )
    return specs


def make_anatomical_like_preview(path: Path, out_dir: Path, downsample: int = 16, sigma: float = 8.0,
                                 dry_run: bool = False) -> Path:
    """Low-frequency blur of a SIGNAL channel, for visual inspection ONLY.

    WARNING: this is *not* a scientifically validated anatomical/background
    channel and must never be passed to Brainmapper as the background input.
    """
    ensure_dir(out_dir)
    out = out_dir / (path.stem + "_anatomical_like_NOT_background.png")
    if dry_run:
        LOG.info("[dry-run] would write anatomical-like preview %s", out.name)
        return out
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    arr = _lazy_read_downsampled(path, downsample).astype(np.float32)
    blurred = gaussian_filter(arr, sigma=sigma)
    Image.fromarray(_percentile_scale(blurred)).save(out)
    LOG.warning("Anatomical-like preview is for VISUAL INSPECTION ONLY, not a background channel: %s", out)
    return out
