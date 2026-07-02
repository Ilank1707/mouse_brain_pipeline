"""QC imagery, review patches and the manual-review batch for the pilot detector.

Per channel+section (all clearly titled as PROVISIONAL candidates, never cells):

  01_raw_projection.png
  02_shared_tissue_mask.png                 -- shared tissue boundary (green)
  03_injection_mask.png                     -- injection-site boundary (red)
  04_candidates_before_injection_exclusion.png
  05_candidates_after_injection_exclusion.png
  06_manual_review_sample.png
  07_candidate_interpretation_audit.png

Plus a shared ``shared_tissue_mask.png`` over both channels, contact-sheet review
patches (centred, crosshair, shared scaling, raw + background-corrected) and a
stratified ``review_batch.csv``.

If ANY invalid coordinate exists, a prominent warning is printed and scientific
summary counts are withheld.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from .candidate_detection import (
    CANDIDATE_COLUMNS,
    STATUS_ARTIFACT,
    STATUS_INJECTION,
    STATUS_INVALID_MEASUREMENT,
    STATUS_MANUAL_REVIEW,
    STATUS_RULE_FAILED,
    STATUS_RULE_PASSED,
    STATUS_SUSPECT_INJECTION,
    SectionDetectionResult,
)
from .channels import channel_display_name
from .qc_display import (
    INTENSITY_DIAGNOSTIC_COLUMNS,
    QC_DISPLAY_METADATA_COLUMNS,
    apply_display,
    compute_display_limits,
    diagnostics_row,
    metadata_row,
)
from .qc_native import (
    QC_IMAGE_METADATA_COLUMNS,
    apply_window_uint8,
    draw_candidate_overlay,
    native_max_projection,
    save_native_projection_tiff,
    save_png_fullres,
    save_preview_png,
)
from .qc_native import metadata_row as _image_metadata_row
from .utilities import LOG, ensure_dir

_MAX_MARKERS = 3000
_TISSUE_COLOR = "#39FF14"      # green
_INJECTION_COLOR = "#FF2D2D"   # red

# status -> (label, colour, marker, filled)
_STATUS_STYLE = {
    STATUS_RULE_PASSED: ("preliminary rule pass", "#39FF14", "o", False),
    STATUS_MANUAL_REVIEW: ("manual review", "#FFE100", "s", False),
    STATUS_INVALID_MEASUREMENT: ("invalid measurement", "#00D9FF", "D", False),
    STATUS_SUSPECT_INJECTION: ("suspect automatic injection mask", "#FF7F0E", "^", False),
    STATUS_INJECTION: ("confirmed injection", "#FF2D2D", "x", True),
    STATUS_RULE_FAILED: ("preliminary rule fail", "#9E9E9E", ".", True),
    STATUS_ARTIFACT: ("Cellfinder artefact/outlier", "#C04CFF", "+", True),
}

REVIEW_BATCH_COLUMNS = list(dict.fromkeys(
    CANDIDATE_COLUMNS
    + ["review_sampling_category", "review_patch_file", "review_notes"]
))

MASK_DIAGNOSTIC_COLUMNS = [
    "channel",
    "section",
    "mask_area_px",
    "tissue_area_px",
    "mask_fraction_of_tissue",
    "candidates_inside_mask",
    "candidates_outside_mask",
    "candidate_fraction_inside_mask",
    "injection_mask_source",
    "injection_mask_validated",
    "injection_mask_qc_failed",
]

STATUS_SUMMARY_COLUMNS = [
    "channel",
    "section",
    "current_status",
    "candidate_count",
    "section_total",
    "counts_reconcile",
]


# --------------------------------------------------------------------------- #
def _display(proj):
    """PREVIOUS global percentile stretch (kept only for the QC comparison panel)."""
    import numpy as np  # noqa: PLC0415

    lo, hi = np.percentile(proj, [1.0, 99.5])
    return np.clip((proj - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def _default_qc_display_cfg():
    from .config import QcDisplayConfig  # noqa: PLC0415

    return QcDisplayConfig()


def section_display_info(res, qc_display_cfg=None, padding_values=(0.0,)) -> dict:
    """Compute (and cache) the chosen display window + diagnostics for a section.

    READ-ONLY: never alters ``res.projection`` or any measurement.
    """
    if getattr(res, "display_info", None):
        return res.display_info
    if res.projection is None:
        return {}
    cfg = qc_display_cfg or _default_qc_display_cfg()
    settings = cfg.for_channel(res.channel)
    info = compute_display_limits(
        res.projection,
        settings,
        tissue_mask=res.tissue_mask,
        injection_core_mask=res.injection_core_mask,
        padding_values=padding_values,
        minimum_pixels=getattr(cfg, "minimum_pixels", 50),
        exclude_injection_core=getattr(cfg, "exclude_injection_core", True),
    )
    try:
        res.display_info = info
    except Exception:  # pragma: no cover - res may be a frozen/foreign object
        pass
    return info


def _truthy(value) -> bool:
    """Accept both in-memory bools and CSV strings ("True"/"1"/"yes")."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _counts(candidates) -> dict:
    from collections import Counter

    statuses = Counter(c["current_status"] for c in candidates)
    return {
        "total": len(candidates),
        "inside_injection": sum(
            1 for c in candidates if c.get("inside_injection_analysis_exclusion")
        ),
        "outside_injection": sum(
            1 for c in candidates if not c.get("inside_injection_analysis_exclusion")
        ),
        "manual_review": statuses[STATUS_MANUAL_REVIEW],
        "invalid": sum(1 for c in candidates if c.get("invalid_coordinate")),
        "rule_passed": statuses[STATUS_RULE_PASSED],
        "statuses": statuses,
        "counts_reconcile": sum(statuses.values()) == len(candidates),
    }


def _summary_title(channel, section, backend, counts) -> str:
    warn = "  *** INVALID COORDINATES PRESENT -- COUNTS WITHHELD ***" if counts["invalid"] else ""
    return (
        f"{channel_display_name(channel)} ({channel}) section {section:03d} "
        f"[backend={backend}] -- PROVISIONAL candidates\n"
        f"total candidates: {counts['total']} | inside injection: {counts['inside_injection']} | "
        f"outside injection: {counts['outside_injection']} | manual-review: {counts['manual_review']} | "
        f"invalid-coord: {counts['invalid']}{warn}"
    )


def _scatter_local(ax, candidates, origin, styles_for, np):
    """Plot candidates in crop-local pixels, grouped by status, marker-capped."""
    from matplotlib.lines import Line2D  # noqa: PLC0415

    oy, ox = origin
    grouped: dict[str, list] = {}
    for c in candidates:
        grouped.setdefault(c["current_status"], []).append(c)
    handles = []
    for status in styles_for:
        pts = grouped.get(status, [])
        if not pts:
            continue
        label, color, marker, filled = _STATUS_STYLE[status]
        shown = pts if len(pts) <= _MAX_MARKERS else pts[:: int(np.ceil(len(pts) / _MAX_MARKERS))]
        xs = [c["x_global_px"] - ox for c in shown]
        ys = [c["y_global_px"] - oy for c in shown]
        kw = dict(s=20, marker=marker, linewidths=0.7)
        kw.update(c=color) if filled else kw.update(facecolors="none", edgecolors=color)
        ax.scatter(xs, ys, **kw)
        handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=color,
                              markeredgecolor=color, markersize=7, label=f"{label} ({len(pts)})"))
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.6)


def all_display_statuses(candidates) -> list[str]:
    """Return all known statuses present, in a stable visual order."""
    ordered = [
        STATUS_RULE_PASSED,
        STATUS_RULE_FAILED,
        STATUS_MANUAL_REVIEW,
        STATUS_INVALID_MEASUREMENT,
        STATUS_SUSPECT_INJECTION,
        STATUS_INJECTION,
        STATUS_ARTIFACT,
    ]
    present = {candidate["current_status"] for candidate in candidates}
    return [status for status in ordered if status in present]


def candidates_outside_analysis_mask(candidates) -> list:
    """All candidates outside the geometric analysis mask, regardless of rules."""
    return [
        candidate for candidate in candidates
        if not candidate.get("inside_injection_analysis_exclusion")
    ]


def write_shared_tissue_qc(qc_dir: Path, results: Sequence[SectionDetectionResult],
                           shared_mask, *, qc_display_cfg=None,
                           padding_values=(0.0,)) -> Path | None:
    """Save shared_tissue_mask.png with the shared boundary over both channels."""
    if shared_mask is None or not any(r.projection is not None for r in results):
        return None
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ensure_dir(qc_dir)
    usable = [r for r in results if r.projection is not None]
    fig, axes = plt.subplots(1, len(usable), figsize=(7 * len(usable), 6), squeeze=False)
    for ax, r in zip(axes[0], usable):
        info = section_display_info(r, qc_display_cfg, padding_values)
        ax.imshow(
            apply_display(r.projection, info["display_min"], info["display_max"]),
            cmap="gray", origin="upper", vmin=0.0, vmax=1.0,
        )
        ax.contour(shared_mask, levels=[0.5], colors=_TISSUE_COLOR, linewidths=0.9)
        ax.set_title(f"{channel_display_name(r.channel)} ({r.channel}) section "
                     f"{r.section:03d}\nshared tissue boundary (green)", fontsize=9)
        ax.axis("off")
    fig.suptitle("SHARED tissue mask over both channels -- PROVISIONAL (not cell counts)",
                 fontsize=10)
    fig.tight_layout()
    out = qc_dir / "shared_tissue_mask.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    LOG.warning("Wrote shared tissue QC -> %s", out)
    return out


def write_channel_qc(qc_dir: Path, res: SectionDetectionResult, *,
                     qc_display_cfg=None, padding_values=(0.0,)) -> Path:
    """Write the per-channel QC images using the chosen (reproducible) display.

    The display window is a brightness/contrast setting ONLY -- it never changes
    raw values, Cellfinder input or any measurement. Every figure records the
    display provenance (mode, min, max, whether the injection core was excluded).
    """
    if res.projection is None:
        return qc_dir
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    section_dir = ensure_dir(qc_dir / f"{res.channel}_section_{res.section:03d}")
    clabel = channel_display_name(res.channel)  # human label; dir name stays raw
    info = section_display_info(res, qc_display_cfg, padding_values)
    display_min = float(info.get("display_min", 0.0))
    display_max = float(info.get("display_max", 1.0))
    disp = apply_display(res.projection, display_min, display_max)
    origin = res.crop_origin
    counts = _counts(res.candidates)
    backend = res.backend
    provenance = (
        f"display: {info.get('display_mode', 'n/a')} window "
        f"[{display_min:.0f}, {display_max:.0f}]  "
        f"injection-core excluded: {bool(info.get('injection_core_excluded', False))}"
    )

    # Exact boolean boundaries for downstream inspection/reuse.
    if res.tissue_mask is not None:
        np.save(section_dir / "tissue_mask.npy", res.tissue_mask.astype(bool))
    if res.injection_core_mask is not None:
        np.save(section_dir / "injection_core_mask.npy", res.injection_core_mask.astype(bool))
    if res.injection_analysis_exclusion_mask is not None:
        np.save(
            section_dir / "injection_analysis_exclusion_mask.npy",
            res.injection_analysis_exclusion_mask.astype(bool),
        )
    if res.generation_suppression_mask is not None:
        np.save(
            section_dir / "generation_suppression_mask.npy",
            res.generation_suppression_mask.astype(bool),
        )

    def base(title):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(disp, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.01, 0.995, provenance, transform=ax.transAxes, va="top", ha="left",
                color="#FFE100", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.4, pad=1.5, edgecolor="none"))
        return fig, ax

    # 01 clean projection (no candidate overlays) using the selected display window.
    fig, ax = base(f"{clabel} ({res.channel}) section {res.section:03d} -- max projection "
                   "(clean, no overlays)\nPROVISIONAL candidates")
    fig.tight_layout(); fig.savefig(section_dir / "01_raw_projection.png", dpi=130); plt.close(fig)

    # 02 shared tissue mask boundary (green)
    fig, ax = base(f"{clabel} ({res.channel}) section {res.section:03d} -- shared tissue boundary (green)")
    if res.tissue_mask is not None and res.tissue_mask.any() and not res.tissue_mask.all():
        ax.contour(res.tissue_mask, levels=[0.5], colors=_TISSUE_COLOR, linewidths=0.9)
    else:
        ax.text(0.5, 0.02, "tissue mask disabled / whole-crop tissue", color="w",
                ha="center", transform=ax.transAxes, fontsize=8)
    fig.tight_layout(); fig.savefig(section_dir / "02_shared_tissue_mask.png", dpi=130); plt.close(fig)

    # 03 injection core and larger analysis-exclusion boundaries.
    fig, ax = base(
        f"{clabel} ({res.channel}) section {res.section:03d} -- injection core (orange) / "
        "analysis exclusion (red)"
    )
    if res.injection_analysis_exclusion_mask is not None and \
            res.injection_analysis_exclusion_mask.any():
        ax.contour(
            res.injection_analysis_exclusion_mask, levels=[0.5],
            colors=_INJECTION_COLOR, linewidths=1.3,
        )
        if res.injection_core_mask is not None and res.injection_core_mask.any():
            ax.contour(res.injection_core_mask, levels=[0.5], colors="#FF9F1C", linewidths=1.1)
        diag = res.mask_diagnostics or {}
        ax.text(
            0.01, 0.01,
            f"mask/tissue={diag.get('mask_fraction_of_tissue', 0):.1%}; "
            f"candidates inside={diag.get('candidate_fraction_inside_mask', 0):.1%}",
            color="white", fontsize=7, transform=ax.transAxes,
        )
        ax.legend(handles=[
            Line2D([0], [0], color="#FF9F1C", lw=2, label="core"),
            Line2D([0], [0], color=_INJECTION_COLOR, lw=2, label="analysis exclusion"),
        ],
                  loc="lower right", fontsize=7, framealpha=0.6)
    else:
        ax.text(0.5, 0.02, "NO injection mask (empty) -- check QC warnings", color=_INJECTION_COLOR,
                ha="center", transform=ax.transAxes, fontsize=9)
    fig.tight_layout(); fig.savefig(section_dir / "03_injection_mask.png", dpi=130); plt.close(fig)

    styles = all_display_statuses(res.candidates)

    # 04 all Cellfinder candidates before interpretation.
    fig, ax = base(_summary_title(res.channel, res.section, backend, counts)
                   + "\nALL Cellfinder candidates before interpretation")
    if res.injection_mask is not None and res.injection_mask.any():
        ax.contour(res.injection_mask, levels=[0.5], colors=_INJECTION_COLOR, linewidths=1.0)
    _scatter_local(ax, res.candidates, origin, styles, np)
    fig.tight_layout()
    fig.savefig(section_dir / "04_candidates_before_injection_exclusion.png", dpi=140)
    plt.close(fig)

    # 05 every candidate outside the geometric analysis mask.
    fig, ax = base(_summary_title(res.channel, res.section, backend, counts)
                   + "\nALL candidates outside the injection analysis mask")
    if res.injection_mask is not None and res.injection_mask.any():
        ax.contour(res.injection_mask, levels=[0.5], colors=_INJECTION_COLOR, linewidths=1.0)
    kept = candidates_outside_analysis_mask(res.candidates)
    _scatter_local(ax, kept, origin, all_display_statuses(kept), np)
    fig.tight_layout()
    fig.savefig(section_dir / "05_candidates_after_injection_exclusion.png", dpi=140)
    plt.close(fig)

    # 06 manual-review sample
    fig, ax = base(f"{clabel} ({res.channel}) section {res.section:03d} -- manual-review sample: "
                   f"{counts['manual_review']}\nPROVISIONAL")
    review = [c for c in res.candidates if c["current_status"] == STATUS_MANUAL_REVIEW]
    _scatter_local(ax, review, origin, [STATUS_MANUAL_REVIEW], np)
    fig.tight_layout(); fig.savefig(section_dir / "06_manual_review_sample.png", dpi=140); plt.close(fig)

    # 07 four-way interpretation audit, explicitly showing what each stage hides.
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    panels = [
        ("All Cellfinder candidates", res.candidates),
        (
            "Geometrically inside automatic/manual mask",
            [c for c in res.candidates if c.get("inside_injection_analysis_exclusion")],
        ),
        ("All candidates outside mask", kept),
        (
            "Outside-mask candidates hidden by preliminary rules",
            [c for c in kept if c["current_status"] != STATUS_RULE_PASSED],
        ),
    ]
    for ax, (title, rows) in zip(axes.ravel(), panels):
        ax.imshow(disp, cmap="gray", origin="upper")
        if res.injection_mask is not None and res.injection_mask.any():
            ax.contour(res.injection_mask, levels=[0.5], colors=_INJECTION_COLOR, linewidths=0.8)
        _scatter_local(ax, rows, origin, all_display_statuses(rows), np)
        ax.set_title(f"{title}\n{len(rows)} candidates")
        ax.axis("off")
    fig.suptitle(
        f"{clabel} ({res.channel}) section {res.section:03d} candidate interpretation audit "
        "(not cell counts)"
    )
    fig.tight_layout()
    fig.savefig(section_dir / "07_candidate_interpretation_audit.png", dpi=140)
    plt.close(fig)

    # 08 raw-pass / suppressed-pass / union candidate-generation distributions.
    raw_pass = [c for c in res.candidates if _truthy(c.get("detected_on_raw_stack"))]
    suppressed_only = [
        c for c in res.candidates
        if _truthy(c.get("detected_on_injection_suppressed_stack"))
        and not _truthy(c.get("detected_on_raw_stack"))
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    gen_panels = [
        (f"Raw-stack pass ({len(raw_pass)})", raw_pass),
        (f"Injection-suppressed-only ({len(suppressed_only)})", suppressed_only),
        (f"Union of both passes ({len(res.candidates)})", res.candidates),
    ]
    for ax, (title, rows) in zip(axes, gen_panels):
        ax.imshow(disp, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
        if res.injection_analysis_exclusion_mask is not None and \
                res.injection_analysis_exclusion_mask.any():
            ax.contour(res.injection_analysis_exclusion_mask, levels=[0.5],
                       colors=_INJECTION_COLOR, linewidths=0.8)
        if res.generation_suppression_mask is not None and \
                res.generation_suppression_mask.any():
            ax.contour(res.generation_suppression_mask, levels=[0.5],
                       colors="#00D9FF", linewidths=0.8)
        _scatter_local(ax, rows, origin, all_display_statuses(rows), np)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{clabel} ({res.channel}) section {res.section:03d} candidate-generation provenance "
        f"({res.generation_diagnostics.get('generation_suppression_mask_source', 'none')}) "
        "-- PROVISIONAL candidates, not cell counts"
    )
    fig.tight_layout()
    fig.savefig(section_dir / "08_candidate_generation_source.png", dpi=140)
    plt.close(fig)

    # 09 derived detection input (injection core suppressed) -- NOT raw data.
    if res.suppressed_projection is not None:
        supp_info = compute_display_limits(
            res.suppressed_projection,
            (qc_display_cfg or _default_qc_display_cfg()).for_channel(res.channel),
            tissue_mask=res.tissue_mask, padding_values=padding_values,
        )
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(
            apply_display(res.suppressed_projection,
                          supp_info["display_min"], supp_info["display_max"]),
            cmap="gray", origin="upper", vmin=0.0, vmax=1.0,
        )
        if res.generation_suppression_mask is not None and \
                res.generation_suppression_mask.any():
            ax.contour(res.generation_suppression_mask, levels=[0.5],
                       colors="#00D9FF", linewidths=1.0)
        ax.set_title(
            f"{clabel} ({res.channel}) section {res.section:03d} -- DERIVED DETECTION INPUT\n"
            "injection core suppressed in-memory (NOT raw data; raw TIFFs untouched)"
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(section_dir / "09_injection_suppressed_detection_input.png", dpi=130)
        plt.close(fig)

    # 10 optional side-by-side: previous global scaling vs the chosen scaling.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes[0].imshow(_display(res.projection), cmap="gray", origin="upper")
    axes[0].set_title("Previous global percentile scaling (1, 99.5)")
    axes[1].imshow(disp, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    axes[1].set_title(provenance)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{clabel} ({res.channel}) section {res.section:03d} -- display scaling comparison "
        "(QC display only; raw data unchanged)"
    )
    fig.tight_layout()
    fig.savefig(section_dir / "10_display_scaling_comparison.png", dpi=130)
    plt.close(fig)

    # 11 injection seed filtering: all auto components, kept (seeded), removed.
    components = res.injection_components or {}
    if components.get("seed_filter_applied"):
        fig, ax = base(
            f"{clabel} ({res.channel}) section {res.section:03d} -- injection seed filtering\n"
            f"kept {components.get('n_kept', 0)} / removed "
            f"{components.get('n_removed', 0)} automatic components"
        )
        all_mask = components.get("all_auto_mask")
        kept_mask = components.get("kept_auto_mask")
        removed_mask = components.get("removed_auto_mask")
        if all_mask is not None and all_mask.any():
            ax.contour(all_mask, levels=[0.5], colors="#FFFFFF", linewidths=0.6)
        if kept_mask is not None and kept_mask.any():
            ax.contour(kept_mask, levels=[0.5], colors=_TISSUE_COLOR, linewidths=1.3)
        if removed_mask is not None and removed_mask.any():
            ax.contour(removed_mask, levels=[0.5], colors=_INJECTION_COLOR, linewidths=1.3)
        for point in (components.get("seed_points_local") or []):
            ax.plot(point[1], point[0], marker="x", color="#00D9FF", markersize=9)
        ax.legend(handles=[
            Line2D([0], [0], color="#FFFFFF", lw=2, label="all auto components"),
            Line2D([0], [0], color=_TISSUE_COLOR, lw=2, label="kept (seeded)"),
            Line2D([0], [0], color=_INJECTION_COLOR, lw=2, label="removed (no seed)"),
            Line2D([0], [0], marker="x", color="#00D9FF", lw=0, label="seed points"),
        ], loc="lower right", fontsize=7, framealpha=0.6)
        fig.tight_layout()
        fig.savefig(section_dir / "11_injection_seed_filtering.png", dpi=140)
        plt.close(fig)

    if counts["invalid"]:
        LOG.error("%s section %d: %d INVALID-COORDINATE candidates -- counts withheld.",
                  res.channel, res.section, counts["invalid"])
    LOG.warning("Wrote QC images -> %s", section_dir)
    return section_dir


# --------------------------------------------------------------------------- #
# Review patches: centred, crosshair, shared scaling, raw + bg-corrected
# --------------------------------------------------------------------------- #
def aligned_patches(plane_arrays, y_center, x_center, half_px):
    """Slice the SAME window from every plane around one (y, x) centre.

    Returns (patches, cy, cx) where the crosshair (cy, cx) is identical for
    every plane -- this is what guarantees each plane is centred on the same XY.
    """
    import numpy as np  # noqa: PLC0415

    H, W = plane_arrays[0].shape[:2]
    y0, y1 = max(0, y_center - half_px), min(H, y_center + half_px + 1)
    x0, x1 = max(0, x_center - half_px), min(W, x_center + half_px + 1)
    patches = [np.asarray(p[y0:y1, x0:x1], dtype=np.float32) for p in plane_arrays]
    return patches, int(y_center - y0), int(x_center - x0)


def save_review_patches(patch_dir: Path, res: SectionDetectionResult, candidates,
                        half_px: int = 18, params=None) -> dict:
    """Save a contact sheet per supplied candidate. Returns {candidate_id: filename}."""
    if res.corrected is None or not res.plane_paths or not candidates:
        return {}
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import tifffile  # noqa: PLC0415

    ensure_dir(patch_dir)
    plane_numbers = sorted(res.plane_paths)
    oy, ox = res.crop_origin

    handles = []
    memmaps = {}
    for pl in plane_numbers:
        tf = tifffile.TiffFile(str(res.plane_paths[pl]))
        handles.append(tf)
        page = tf.pages[0]
        try:
            memmaps[pl] = page.asarray(out="memmap")
        except (ValueError, TypeError):
            memmaps[pl] = page.asarray()

    patch_files: dict[str, str] = {}
    try:
        Zc = res.corrected.shape[0]
        raw_plane_arrays = [memmaps[pl] for pl in plane_numbers]
        for c in candidates:
            xg, yg = int(c["x_global_px"]), int(c["y_global_px"])
            xl, yl = int(c["x_local_px"]), int(c["y_local_px"])
            z_support = {
                int(z) for z in str(c.get("support_z_indices", c.get("z_indices", ""))).split(";")
                if z != ""
            }
            peak_plane = int(c.get("peak_plane", c["z_index"]))

            # Identical window for every plane -> same XY centre (raw at global
            # coords, background-corrected at crop-local coords).
            raw_patches, cy, cx = aligned_patches(raw_plane_arrays, yg, xg, half_px)
            corr_plane_arrays = [res.corrected[z] for z in range(Zc)]
            corr_patches, _, _ = aligned_patches(corr_plane_arrays, yl, xl, half_px)

            raw_vmax = max((p.max() for p in raw_patches if p.size), default=1.0) or 1.0
            corr_vmax = max((p.max() for p in corr_patches if p.size), default=1.0) or 1.0

            n = len(plane_numbers)
            fig = plt.figure(figsize=(1.55 * n, 4.5))
            gs = fig.add_gridspec(3, n, height_ratios=[1, 1, 0.65])
            axes = [[fig.add_subplot(gs[row, col]) for col in range(n)] for row in range(2)]
            central_r = (
                params.central_region_radius_um / 1.004 if params is not None else 3.0
            )
            annulus_inner = (
                params.background_annulus_inner_um / 1.004 if params is not None else 8.0
            )
            annulus_outer = (
                params.background_annulus_outer_um / 1.004 if params is not None else 16.0
            )
            for col, pl in enumerate(plane_numbers):
                z_stack = col  # stack index (== record z_index for this column)
                for row, (patches, vmax, tag) in enumerate(
                    ((raw_patches, raw_vmax, "raw"), (corr_patches, corr_vmax, "bg"))
                ):
                    ax = axes[row][col]
                    ax.imshow(patches[col], cmap="gray", vmin=0, vmax=vmax, origin="upper")
                    ax.axhline(cy, color="#FF2D2D", lw=0.5)
                    ax.axvline(cx, color="#FF2D2D", lw=0.5)
                    for radius, color, style in (
                        (central_r, "#00D9FF", "-"),
                        (annulus_inner, "#FF4DFF", "--"),
                        (annulus_outer, "#FF4DFF", "--"),
                    ):
                        ax.add_patch(plt.Circle(
                            (cx, cy), radius, fill=False, color=color,
                            linestyle=style, linewidth=0.55,
                        ))
                    border = "#FFE100" if z_stack == peak_plane else (
                        "#39FF14" if z_stack in z_support else "none")
                    if border != "none":
                        for s in ax.spines.values():
                            s.set_color(border); s.set_linewidth(1.6)
                    ax.set_xticks([]); ax.set_yticks([])
                    if row == 0:
                        ax.set_title(f"p{pl}", fontsize=6)
                    if col == 0:
                        ax.set_ylabel(tag, fontsize=6)
            profile_ax = fig.add_subplot(gs[2, :])
            contrasts = [c.get(f"plane_{z}_contrast", np.nan) for z in range(n)]
            contrasts = np.asarray(contrasts, dtype=float)
            profile_ax.plot(range(n), contrasts, marker="o", color="#00A6FB", lw=1.2)
            if params is not None:
                profile_ax.axhline(
                    params.z_support_min_contrast, color="#777777", ls="--", lw=0.8,
                    label="support threshold",
                )
            for z in z_support:
                profile_ax.axvspan(z - 0.35, z + 0.35, color="#39FF14", alpha=0.15)
            profile_ax.axvline(peak_plane, color="#FFE100F9", lw=1.2)
            profile_ax.set(
                xlabel="stack z index", ylabel="local contrast",
                xticks=range(n), title="Fixed-XY central-disk / local-annulus contrast profile",
            )
            profile_ax.grid(alpha=0.2)
            fig.suptitle(
                f"{c['candidate_id']}  global_xy=({xg},{yg})  z_indices={sorted(z_support)}  "
                f"peak_plane={peak_plane}  relative_z_um={c['section_relative_z_um']}  "
                f"status={c['current_status']}  injection={c.get('inside_injection_site', False)}",
                fontsize=7,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fname = f"{c['current_status']}_{c['candidate_id']}.png"
            fig.savefig(patch_dir / fname, dpi=110)
            plt.close(fig)
            patch_files[c["candidate_id"]] = fname
    finally:
        for tf in handles:
            tf.close()
    LOG.warning("Wrote %d review patches -> %s", len(patch_files), patch_dir)
    return patch_files


# --------------------------------------------------------------------------- #
# Stratified manual-review batch
# --------------------------------------------------------------------------- #
def select_review_batch(candidates, params, per_stratum: int = 25) -> list:
    """Pick a stratified sample across the key strata for manual labelling.

    Each picked candidate records the stratum it came from in
    review_sampling_category. These are sampling buckets, not labels.
    """
    def by_status(status):
        return [c for c in candidates if c["current_status"] == status]

    def by_reason(reason):
        return [c for c in candidates if c.get("rejection_reason") == reason]

    passed = by_status(STATUS_RULE_PASSED)
    manual = by_status(STATUS_MANUAL_REVIEW)

    thr = params.min_local_robust_z
    near_threshold = sorted(
        by_reason("insufficient_local_contrast"),
        key=lambda c: abs(c["local_robust_z"] - thr),
    )
    single_plane = [c for c in candidates
                    if c["n_consecutive_planes"] < params.min_consecutive_planes
                    and c["local_robust_z"] >= params.single_plane_review_min_z]

    # Provenance strata so both passes get human eyes.
    raw_only = [c for c in candidates
                if _truthy(c.get("detected_on_raw_stack"))
                and not _truthy(c.get("detected_on_injection_suppressed_stack"))]
    suppressed_only = [c for c in candidates
                       if _truthy(c.get("detected_on_injection_suppressed_stack"))
                       and not _truthy(c.get("detected_on_raw_stack"))]
    both = [c for c in candidates
            if _truthy(c.get("detected_on_raw_stack"))
            and _truthy(c.get("detected_on_injection_suppressed_stack"))]

    # (name, candidates) in sampling order; name is recorded on each pick.
    strata = [
        ("outside_mask", [c for c in (passed + manual) if not c["inside_injection_site"]]),
        ("preliminary_pass", passed),
        ("near_threshold", near_threshold),
        ("too_large", by_reason("too_large")),
        ("too_small", by_reason("too_small")),
        ("too_elongated", by_reason("too_elongated")),
        ("xy_jump", by_reason("xy_jump")),
        ("single_plane", single_plane),
        ("many_planes", [c for c in candidates
                         if c["n_consecutive_planes"] > params.max_consecutive_planes]),
        ("raw_only", raw_only),
        ("suppressed_only", suppressed_only),
        ("detected_by_both", both),
        ("inside_suspect_mask", by_status(STATUS_SUSPECT_INJECTION)),
        ("invalid_measurement", by_status(STATUS_INVALID_MEASUREMENT)),
        ("confirmed_injection", by_status(STATUS_INJECTION)),
        ("manual_review", manual),
        ("preliminary_fail", by_status(STATUS_RULE_FAILED)),
    ]

    seen: set[str] = set()
    batch: list = []
    for name, stratum in strata:
        for c in stratum[:per_stratum]:
            cid = c["candidate_id"]
            if cid not in seen:
                seen.add(cid)
                c["review_sampling_category"] = name
                batch.append(c)
    return batch


def write_review_batch(out_dir: Path, rows, patch_files: dict) -> Path:
    """Write review_batch.csv (already-selected rows) with a blank manual_label."""
    ensure_dir(out_dir)
    path = out_dir / "review_batch.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_BATCH_COLUMNS)
        writer.writeheader()
        for c in rows:
            record = {column: c.get(column, "") for column in REVIEW_BATCH_COLUMNS}
            record["review_patch_file"] = patch_files.get(c["candidate_id"], "")
            record["manual_label"] = ""
            record["review_notes"] = ""
            writer.writerow(record)
    LOG.warning("Wrote stratified review batch (%d rows) -> %s", len(rows), path)
    return path


def write_mask_diagnostics(out_dir: Path, results: Sequence[SectionDetectionResult]) -> Path:
    """Persist the per-channel injection-mask diagnostics used for warnings."""
    ensure_dir(out_dir)
    path = out_dir / "injection_mask_diagnostics.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASK_DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for result in results:
            row = {
                "channel": result.channel,
                "section": result.section,
                **(result.mask_diagnostics or {}),
            }
            writer.writerow({column: row.get(column, "") for column in MASK_DIAGNOSTIC_COLUMNS})
    LOG.warning("Wrote injection-mask diagnostics -> %s", path)
    return path


def write_status_summary(out_dir: Path, results: Sequence[SectionDetectionResult]) -> Path:
    """Write mutually exclusive status counts that reconcile to every candidate."""
    from collections import Counter

    ensure_dir(out_dir)
    path = out_dir / "candidate_status_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS_SUMMARY_COLUMNS)
        writer.writeheader()
        for result in results:
            counts = Counter(c["current_status"] for c in result.candidates)
            total = len(result.candidates)
            reconciles = sum(counts.values()) == total
            for status, count in sorted(counts.items()):
                writer.writerow({
                    "channel": result.channel,
                    "section": result.section,
                    "current_status": status,
                    "candidate_count": count,
                    "section_total": total,
                    "counts_reconcile": reconciles,
                })
    LOG.warning("Wrote reconciled candidate status summary -> %s", path)
    return path


def write_intensity_diagnostics(out_dir: Path, results: Sequence[SectionDetectionResult],
                                *, qc_display_cfg=None, padding_values=(0.0,)) -> Path:
    """Per channel+section raw range, pixel counts, percentiles and the chosen
    display window (Part 1 intensity-diagnostics CSV)."""
    ensure_dir(out_dir)
    path = out_dir / "intensity_diagnostics.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INTENSITY_DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        for result in results:
            if result.projection is None:
                continue
            info = section_display_info(result, qc_display_cfg, padding_values)
            writer.writerow(diagnostics_row(result.channel, result.section, info))
    LOG.warning("Wrote intensity diagnostics -> %s", path)
    return path


def write_qc_display_metadata(out_dir: Path, results: Sequence[SectionDetectionResult],
                              *, qc_display_cfg=None, padding_values=(0.0,)) -> Path:
    """Record the actual display provenance for every QC figure (Part 1)."""
    ensure_dir(out_dir)
    path = out_dir / "qc_display_metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QC_DISPLAY_METADATA_COLUMNS)
        writer.writeheader()
        for result in results:
            if result.projection is None:
                continue
            info = section_display_info(result, qc_display_cfg, padding_values)
            writer.writerow(metadata_row(result.channel, result.section, info))
    LOG.warning("Wrote QC display metadata -> %s", path)
    return path


def write_native_qc(qc_dir: Path, res: SectionDetectionResult, *,
                    qc_display_cfg=None, padding_values=(0.0,)) -> list:
    """Write native 16-bit + full-resolution QC images (Part 4).

    Returns the per-file metadata rows. The raw TIFFs are never touched; display
    files only apply the chosen window to a copy of the projection.
    """
    if res.projection is None:
        return []
    import numpy as np  # noqa: PLC0415

    section_dir = ensure_dir(qc_dir / f"{res.channel}_section_{res.section:03d}")
    proj = np.asarray(res.projection)
    source_shape = proj.shape
    info = section_display_info(res, qc_display_cfg, padding_values)
    dmin = float(info.get("display_min", 0.0))
    dmax = float(info.get("display_max", 1.0))
    mode = info.get("display_mode", "n/a")

    rows = []

    def row(filename, saved_shape, saved_dtype, *, projection_method, display_mode,
            display_min, display_max, file_format, resizing, interpolation,
            source_dtype):
        rows.append(_image_metadata_row(
            filename=filename, channel=res.channel, section=res.section,
            source_shape=source_shape, saved_shape=saved_shape,
            source_dtype=source_dtype, saved_dtype=saved_dtype,
            projection_method=projection_method, display_mode=display_mode,
            display_min=display_min, display_max=display_max,
            file_format=file_format, resizing=resizing, interpolation=interpolation,
        ))

    # A. native scientific projection (lossless, no stretch, no markers).
    native, _src, method, _upcast = native_max_projection(proj[None])
    native_name = "01_raw_projection_native_16bit.tif"
    save_native_projection_tiff(
        section_dir / native_name, native, channel=res.channel, section=res.section,
        projection_method=method, source_dtype=str(native.dtype),
    )
    row(native_name, native.shape, str(native.dtype), projection_method=method,
        display_mode="none_native", display_min="", display_max="",
        file_format="tiff", resizing=False, interpolation="none",
        source_dtype=str(native.dtype))

    # B. full-resolution display PNG (window applied to the full array).
    display8 = apply_window_uint8(proj, dmin, dmax)
    display_name = "02_raw_projection_display_fullres.png"
    save_png_fullres(section_dir / display_name, display8)
    row(display_name, display8.shape, "uint8", projection_method=method,
        display_mode=mode, display_min=dmin, display_max=dmax,
        file_format="png", resizing=False, interpolation="none",
        source_dtype=str(native.dtype))

    masks = [
        (res.injection_analysis_exclusion_mask, (255, 45, 45)),
        (res.injection_core_mask, (255, 159, 28)),
        (res.generation_suppression_mask, (0, 217, 255)),
    ]

    # C. full-resolution candidate overlays at the source pixel size.
    overlay_all = draw_candidate_overlay(display8, res.candidates, res.crop_origin, masks)
    before_name = "04_candidates_before_interpretation_fullres.png"
    save_png_fullres(section_dir / before_name, overlay_all)
    row(before_name, overlay_all.shape[:2], "uint8", projection_method=method,
        display_mode=mode, display_min=dmin, display_max=dmax,
        file_format="png", resizing=False, interpolation="none",
        source_dtype=str(native.dtype))

    outside = candidates_outside_analysis_mask(res.candidates)
    overlay_audit = draw_candidate_overlay(display8, outside, res.crop_origin, masks)
    audit_name = "07_candidate_interpretation_audit_fullres.png"
    save_png_fullres(section_dir / audit_name, overlay_audit)
    row(audit_name, overlay_audit.shape[:2], "uint8", projection_method=method,
        display_mode=mode, display_min=dmin, display_max=dmax,
        file_format="png", resizing=False, interpolation="none",
        source_dtype=str(native.dtype))

    # D. small preview (clearly labelled; never replaces the full-res file).
    preview_name = "07_candidate_interpretation_audit_preview.png"
    _path, pw, ph = save_preview_png(section_dir / preview_name, overlay_audit)
    row(preview_name, (ph, pw), "uint8", projection_method=method,
        display_mode=mode, display_min=dmin, display_max=dmax,
        file_format="png", resizing=(pw, ph) != (source_shape[1], source_shape[0]),
        interpolation="lanczos", source_dtype=str(native.dtype))

    del display8, overlay_all, overlay_audit, native
    LOG.warning("Wrote native + full-resolution QC -> %s", section_dir)
    return rows


def write_qc_image_metadata(out_dir: Path, rows) -> Path:
    """Write the QC image metadata CSV (real dims, dtype, window, resizing)."""
    ensure_dir(out_dir)
    path = out_dir / "qc_image_metadata.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QC_IMAGE_METADATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in QC_IMAGE_METADATA_COLUMNS})
    LOG.warning("Wrote QC image metadata (%d files) -> %s", len(rows), path)
    return path
