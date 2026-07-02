"""Manual injection-mask overrides loaded from a small separate YAML file.

The override file is kept apart from ``config.yml`` so the automatic mask config
is never edited in place. It carries per-channel manual polygons -- most
importantly ``manual_non_injection_polygons`` (areas the automatic mask wrongly
includes). Overrides are merged onto the loaded config's injection settings; the
subtraction itself happens last in the mask builder so dilation cannot add a
removed region back.

Layout::

    detection:
      injection_exclusion:
        green_signal:
          manual_non_injection_polygons:
            - [[x0, y0], [x1, y1], ...]     # full-resolution px
          manual_polygons: []               # confirmed-injection additions
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import fields as _dc_fields
from pathlib import Path

from .config import InjectionExclusionConfig

# Fields an override may set on the (per-channel) injection config.
_OVERRIDABLE = {
    "manual_polygons",
    "manual_non_injection_polygons",
    "manual_rectangles",
    "injection_seed_points",
    "core_dilation_um",
    "analysis_exclusion_dilation_um",
}
_CHANNEL_KEYS = ("green_signal", "channel_2_signal")


def load_overrides(path: str | Path | None) -> dict:
    """Read the override YAML into a dict (empty dict when path is None/missing)."""
    if not path:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Injection override file not found: {path}")
    import yaml  # noqa: PLC0415

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def overrides_hash(path: str | Path | None) -> str | None:
    """SHA-256 of the override file so a run records exactly which one it used."""
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def injection_exclusion_dict(overrides: dict) -> dict:
    return ((overrides or {}).get("detection") or {}).get("injection_exclusion") or {}


def _apply_fields(cfg: InjectionExclusionConfig, values: dict) -> None:
    for key, value in (values or {}).items():
        if key in _OVERRIDABLE:
            setattr(cfg, key, value)


def apply_overrides_to_injection_cfg(
    inj: InjectionExclusionConfig, overrides: dict
) -> InjectionExclusionConfig:
    """Merge override polygons onto an injection config (returns the same object).

    Base-level keys apply to the shared config; per-channel blocks apply to (and
    create if needed) that channel's override so the two sites stay independent.
    """
    block = injection_exclusion_dict(overrides)
    if not block:
        return inj

    base_values = {k: v for k, v in block.items() if k not in _CHANNEL_KEYS}
    _apply_fields(inj, base_values)

    for channel in _CHANNEL_KEYS:
        channel_block = block.get(channel)
        if not channel_block:
            continue
        sub = getattr(inj, channel, None)
        if not isinstance(sub, InjectionExclusionConfig):
            sub = copy.deepcopy(inj)
            sub.green_signal = None
            sub.channel_2_signal = None
            setattr(inj, channel, sub)
        _apply_fields(sub, channel_block)
    return inj


def apply_overrides_to_config(config, overrides: dict):
    """Apply overrides to ``config.detection.injection_exclusion`` in place."""
    apply_overrides_to_injection_cfg(config.detection.injection_exclusion, overrides)
    return config


# --------------------------------------------------------------------------- #
# Read / write from the interactive editor
# --------------------------------------------------------------------------- #
def read_channel_polygons(path: str | Path | None, channel: str) -> tuple[list, list]:
    """Return ``(injection_polygons, non_injection_polygons)`` for one channel."""
    block = injection_exclusion_dict(load_overrides(path)) if path else {}
    channel_block = block.get(channel, {}) if isinstance(block, dict) else {}
    injection = list(channel_block.get("manual_polygons") or [])
    non_injection = list(channel_block.get("manual_non_injection_polygons") or [])
    return injection, non_injection


def save_channel_polygons(path: str | Path, channel: str,
                          injection_polygons, non_injection_polygons) -> Path:
    """Write/merge one channel's polygons into the override file.

    Existing content for other channels is preserved; the target channel's
    polygon lists are replaced with the supplied ones.
    """
    import yaml  # noqa: PLC0415

    path = Path(path)
    data = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    detection = data.setdefault("detection", {})
    injection = detection.setdefault("injection_exclusion", {})
    channel_block = injection.setdefault(channel, {})
    channel_block["manual_polygons"] = [
        [[int(round(x)), int(round(y))] for x, y in poly] for poly in injection_polygons
    ]
    channel_block["manual_non_injection_polygons"] = [
        [[int(round(x)), int(round(y))] for x, y in poly] for poly in non_injection_polygons
    ]
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return path


# --------------------------------------------------------------------------- #
# Before/after mask QC
# --------------------------------------------------------------------------- #
def save_mask_comparison_png(out_path, display_image, before_mask, after_mask, *,
                             title, removed_mask=None, added_mask=None):
    """Save one figure: display image with the before + after mask boundaries.

    ``removed_mask`` (red fill) shows what the non-injection polygon took out;
    ``added_mask`` (green fill) shows manual injection additions.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(display_image, cmap="gray", origin="upper")
    if before_mask is not None and np.asarray(before_mask).any():
        ax.contour(before_mask, levels=[0.5], colors="#FFB000", linewidths=1.1)
    if after_mask is not None and np.asarray(after_mask).any():
        ax.contour(after_mask, levels=[0.5], colors="#FF2D2D", linewidths=1.3)
    if removed_mask is not None and np.asarray(removed_mask).any():
        ax.contourf(removed_mask, levels=[0.5, 1.0], colors=["#FF2D2D"], alpha=0.25)
    if added_mask is not None and np.asarray(added_mask).any():
        ax.contourf(added_mask, levels=[0.5, 1.0], colors=["#39FF14"], alpha=0.25)
    handles = [
        Line2D([0], [0], color="#FFB000", lw=2, label="before (automatic)"),
        Line2D([0], [0], color="#FF2D2D", lw=2, label="after override"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _all_injection_fields() -> set:
    return {f.name for f in _dc_fields(InjectionExclusionConfig)}
