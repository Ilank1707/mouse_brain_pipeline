"""Auditable diagnostics for the seeded injection-mask split.

Given the seed-filter/split diagnostics dict produced by
``candidate_detection._split_and_filter_by_seeds`` (carried on
``SectionDetectionResult.injection_components``), write per-channel CSV/PNG/JSON
artifacts documenting which pre-split components existed, how they were split,
which seed matched which kept subcomponent, and what was removed.

Raw TIFFs are never touched. Everything is written into the caller-supplied
channel/section directory. green_signal and channel_2_signal stay separate
because each channel has its own directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

COMPONENTS_BEFORE_CSV = "injection_mask_components_before_split.csv"
COMPONENTS_AFTER_CSV = "injection_mask_components_after_split.csv"
SEED_MATCHES_CSV = "injection_mask_seed_matches.csv"
SPLIT_QC_PNG = "injection_mask_split_qc.png"
KEPT_VS_REMOVED_PNG = "injection_mask_kept_vs_removed.png"
SUMMARY_JSON = "injection_mask_summary.json"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_injection_mask_diagnostics(
    out_dir, diag: dict, *, channel: str, section: int, voxel_yx_um=(1.004, 1.004)
) -> dict:
    """Write the six injection-mask split diagnostic files into ``out_dir``.

    Returns a dict of written paths. Safe to call for channels without seeds; it
    then records ``seed_filter_applied: False`` and writes empty component tables.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_filter_applied = bool(diag.get("seed_filter_applied"))
    split_applied = bool(diag.get("split_applied"))
    pre_components = diag.get("pre_split_components", [])
    post_subcomponents = diag.get("post_split_subcomponents", [])
    seed_matches = diag.get("seed_matches", [])

    _write_csv(
        out_dir / COMPONENTS_BEFORE_CSV,
        pre_components,
        ["pre_label", "area_px", "area_um2", "centroid_x_local",
         "centroid_y_local", "n_subcomponents"],
    )
    _write_csv(
        out_dir / COMPONENTS_AFTER_CSV,
        post_subcomponents,
        ["subcomponent_label", "parent_pre_label", "area_px", "area_um2",
         "centroid_x_local", "centroid_y_local", "contains_seed", "kept", "reason"],
    )
    _write_csv(
        out_dir / SEED_MATCHES_CSV,
        seed_matches,
        ["seed_index", "seed_x_local", "seed_y_local", "pre_label",
         "subcomponent_label", "kept"],
    )

    summary = {
        "analysis": "seeded injection-mask split diagnostics",
        "channel": channel,
        "section": section,
        "seed_filter_applied": seed_filter_applied,
        "split_applied": split_applied,
        "split_method": diag.get("split_method"),
        "n_seed_points": diag.get("n_seed_points", 0),
        "n_components_before_split": diag.get("n_components", 0),
        "n_subcomponents_after_split": diag.get("n_subcomponents", len(post_subcomponents)),
        "n_kept": diag.get("n_kept", 0),
        "n_removed": diag.get("n_removed", 0),
        "split_min_peak_distance_um": diag.get("split_min_peak_distance_um"),
        "split_min_subcomponent_area_um2": diag.get("split_min_subcomponent_area_um2"),
        "kept_subcomponent_labels": diag.get("kept_subcomponent_labels", []),
        "voxel_size_yx_um": list(voxel_yx_um),
        "downsample_factor": diag.get("factor"),
        "warnings": diag.get("warnings", []),
        "outputs": {
            "components_before_split_csv": str(out_dir / COMPONENTS_BEFORE_CSV),
            "components_after_split_csv": str(out_dir / COMPONENTS_AFTER_CSV),
            "seed_matches_csv": str(out_dir / SEED_MATCHES_CSV),
        },
    }

    plotted = _write_plots(out_dir, diag, channel=channel, section=section)
    summary["outputs"].update(plotted)
    (out_dir / SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    written = {
        "components_before_split_csv": str(out_dir / COMPONENTS_BEFORE_CSV),
        "components_after_split_csv": str(out_dir / COMPONENTS_AFTER_CSV),
        "seed_matches_csv": str(out_dir / SEED_MATCHES_CSV),
        "summary_json": str(out_dir / SUMMARY_JSON),
    }
    written.update(plotted)
    return written


def _write_plots(out_dir: Path, diag: dict, *, channel: str, section: int) -> dict:
    pre_labels = diag.get("pre_labels_lowres")
    post_labels = diag.get("post_labels_lowres")
    if pre_labels is None or post_labels is None:
        return {}

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    factor = int(diag.get("factor", 1)) or 1
    seeds = diag.get("seed_matches", [])
    kept_labels = set(diag.get("kept_subcomponent_labels", []))

    def _seed_lowres_xy():
        xs = [s["seed_x_local"] / factor for s in seeds]
        ys = [s["seed_y_local"] / factor for s in seeds]
        return xs, ys

    # 1. Pre-split labels vs post-split subcomponent labels.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    try:
        pre_display = np.ma.masked_where(pre_labels == 0, pre_labels)
        post_display = np.ma.masked_where(post_labels == 0, post_labels)
        axes[0].imshow(pre_display, cmap="tab20", interpolation="nearest")
        axes[0].set_title("pre-split components")
        axes[1].imshow(post_display, cmap="tab20", interpolation="nearest")
        axes[1].set_title("post-split subcomponents")
        seed_xs, seed_ys = _seed_lowres_xy()
        for ax in axes:
            if seed_xs:
                ax.scatter(seed_xs, seed_ys, marker="x", c="red", s=60, label="seed")
                ax.legend(fontsize=8, loc="upper right")
            ax.set_xlabel("x (downsampled px)")
            ax.set_ylabel("y (downsampled px)")
        fig.suptitle(
            f"Injection-mask split QC — {channel}, section {section:03d} "
            "(PROVISIONAL injection mask)"
        )
        fig.tight_layout()
        fig.savefig(out_dir / SPLIT_QC_PNG, dpi=150)
    finally:
        plt.close(fig)

    # 2. Kept vs removed subcomponents.
    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        status = np.zeros(post_labels.shape, dtype=np.int8)  # 0 background
        for label in range(1, int(post_labels.max()) + 1):
            member = post_labels == label
            if not member.any():
                continue
            status[member] = 2 if label in kept_labels else 1  # 1 removed, 2 kept
        cmap = plt.matplotlib.colors.ListedColormap(["#00000000", "#C62828", "#2E7D32"])
        ax.imshow(status, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        seed_xs, seed_ys = _seed_lowres_xy()
        if seed_xs:
            ax.scatter(seed_xs, seed_ys, marker="x", c="white", s=70, label="seed")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(
            f"Injection mask kept (green) vs removed (red) — {channel}, "
            f"section {section:03d}"
        )
        ax.set_xlabel("x (downsampled px)")
        ax.set_ylabel("y (downsampled px)")
        fig.tight_layout()
        fig.savefig(out_dir / KEPT_VS_REMOVED_PNG, dpi=150)
    finally:
        plt.close(fig)

    return {
        "split_qc_png": str(out_dir / SPLIT_QC_PNG),
        "kept_vs_removed_png": str(out_dir / KEPT_VS_REMOVED_PNG),
    }
