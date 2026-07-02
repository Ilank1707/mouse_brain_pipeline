#!/usr/bin/env python
"""Keyboard-driven review of a candidate across all seven aligned optical planes.

For every candidate the reviewer loads the matching seven TIFF planes for the
section, crops the **same** global ``(x, y)`` window from every plane (planes are
never independently recentred) and presents:

* a seven-panel montage (plane 1..7, identical display size + XY centre,
  crosshairs on the candidate, peak plane and detected support planes
  highlighted);
* a larger single-plane view with a Z slider and keyboard scrubbing;
* a maximum-intensity projection, clearly labelled as a DISPLAY AID ONLY;
* an optional colour-coded Z overlay (each plane a distinct hue);
* a fixed-XY intensity / local-contrast plot across the seven planes;
* both raw and background-corrected views.

The two folders are both biological signal channels; neither is treated as
background/autofluorescence. Review one channel at a time with ``--channel`` and
labels are stored independently per channel. Raw TIFFs are only ever read.

Keyboard controls
-----------------
    1 = cell        2 = artefact    3 = injection   4 = uncertain   5 = skip
    left / b = previous candidate   right / n = next candidate
    up / ] = next plane (Z+)        down / [ = previous plane (Z-)
    r = toggle raw / background-corrected single-plane view
    o = toggle colour-coded Z overlay
    q = quit
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.candidate_detection import background_correct
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.review import (
    LABEL_KEYS,
    LabelConflictError,
    filter_review_candidates,
    load_manual_labels,
    previous_label,
    read_csv_rows,
    save_manual_label,
    unreviewed_candidates,
)
from mouse_brain_pipeline.review_patches import (
    HIGHLIGHT_COLOURS,
    colour_coded_z_projection,
    display_limits,
    fixed_xy_central_intensity,
    load_fixed_xy_stack,
    max_intensity_projection,
    ordered_section_planes,
    panel_highlight_class,
    parse_peak_index,
    parse_support_indices,
)

MIP_CAPTION = "MAX PROJECTION - display aid only (do NOT call a cell from this alone)"


def patch_half_px(config) -> int:
    """Half-window in pixels from the configured patch size (>= 8 px)."""
    return max(
        8,
        int(round(
            config.classifier.patch_size_xy_um
            / (2 * config.acquisition.voxel_size_y_um)
        )),
    )


class Reviewer:
    """Matplotlib reviewer with a persistent figure rebuilt per candidate.

    Z scrubbing, the raw/corrected toggle and the overlay toggle update only the
    affected axes; moving between candidates does a full rebuild so the Z slider
    always matches the new plane count.
    """

    def __init__(
        self, candidates, indexes, config, labels_path, reviewer,
        allow_label_changes=False,
    ):
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.widgets import Slider  # noqa: PLC0415

        self._plt = plt
        self._Slider = Slider
        self.candidates = candidates
        self.indexes = indexes
        self.config = config
        self.labels_path = labels_path
        self.reviewer = reviewer
        self.allow_label_changes = allow_label_changes
        self.labels = load_manual_labels(labels_path)
        self.half_px = patch_half_px(config)

        self.index = 0
        self.z_current = 0
        self.big_mode = "raw"          # 'raw' or 'corrected'
        self.overlay_on = True
        self._slider_guard = False

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.draw_candidate()

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    def _load_current(self):
        import numpy as np  # noqa: PLC0415

        candidate = self.candidates[self.index]
        section = int(float(candidate["section"]))
        channel = candidate["channel"]
        ordered = ordered_section_planes(self.indexes[channel], section)
        raw, (cy, cx) = load_fixed_xy_stack(
            ordered, candidate["x_global_px"], candidate["y_global_px"], self.half_px,
        )
        raw_f = raw.astype(np.float32)
        corrected = background_correct(
            raw_f,
            self.config.acquisition.voxel_size_y_um,
            self.config.detection.background_sigma_um,
        )
        self.candidate = candidate
        self.section = section
        self.channel = channel
        self.plane_numbers = [plane for plane, _ in ordered]
        self.raw = raw_f
        self.corrected = corrected
        self.centre = (cy, cx)
        self.raw_limits = display_limits(raw_f)
        self.corr_limits = display_limits(corrected)
        self.peak = parse_peak_index(candidate)
        self.support = parse_support_indices(candidate)
        self.z_count = raw_f.shape[0]
        self.z_current = min(max(self.peak, 0), max(self.z_count - 1, 0))

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw_candidate(self):
        self._load_current()
        self.fig.clear()
        z = self.z_count
        gs = self.fig.add_gridspec(
            4, max(z, 7), height_ratios=[1.0, 1.0, 1.7, 0.9],
            hspace=0.45, wspace=0.25,
        )
        self.montage_raw_axes = [self.fig.add_subplot(gs[0, i]) for i in range(z)]
        self.montage_corr_axes = [self.fig.add_subplot(gs[1, i]) for i in range(z)]
        third = max(z, 7)
        self.ax_big = self.fig.add_subplot(gs[2, 0:third // 3 or 1])
        self.ax_mip = self.fig.add_subplot(gs[2, third // 3:2 * third // 3])
        self.ax_overlay = self.fig.add_subplot(gs[2, 2 * third // 3:])
        self.ax_profile = self.fig.add_subplot(gs[3, 0:third // 2 or 1])
        self.ax_info = self.fig.add_subplot(gs[3, third // 2:])
        self.ax_slider = self.fig.add_axes([0.30, 0.005, 0.40, 0.02])

        self._render_montage()
        self._render_big()
        self._render_mip()
        self._render_overlay()
        self._render_profile()
        self._render_info()

        self.slider = self._Slider(
            self.ax_slider, "Z plane", 0, max(self.z_count - 1, 0),
            valinit=self.z_current, valstep=1,
        )
        self.slider.on_changed(self._on_slider)

        self.fig.suptitle(
            f"{self.channel}  |  section {self.section}  |  fixed XY centre in every plane "
            f"(x={self.candidate['x_global_px']}, y={self.candidate['y_global_px']})",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def _draw_panel(self, ax, image, limits, *, title, highlight="none", crosshair=True):
        cy, cx = self.centre
        ax.clear()
        ax.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1])
        if crosshair:
            ax.axhline(cy, color="red", lw=0.6)
            ax.axvline(cx, color="red", lw=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8)
        colour = HIGHLIGHT_COLOURS[highlight]
        width = 2.4 if highlight != "none" else 0.6
        for spine in ax.spines.values():
            spine.set_color(colour)
            spine.set_linewidth(width)

    def _render_montage(self):
        for i, ax in enumerate(self.montage_raw_axes):
            highlight = panel_highlight_class(i, self.peak, self.support)
            tag = {"peak": " PEAK", "support": " support", "none": ""}[highlight]
            here = "  <Z" if i == self.z_current else ""
            self._draw_panel(
                ax, self.raw[i], self.raw_limits,
                title=f"plane {self.plane_numbers[i]:02d}{tag}{here}",
                highlight=highlight,
            )
        for i, ax in enumerate(self.montage_corr_axes):
            highlight = panel_highlight_class(i, self.peak, self.support)
            self._draw_panel(
                ax, self.corrected[i], self.corr_limits,
                title="bg-corrected" if i == 0 else "",
                highlight=highlight,
            )
        if self.montage_raw_axes:
            self.montage_raw_axes[0].set_ylabel("raw", fontsize=8)
        if self.montage_corr_axes:
            self.montage_corr_axes[0].set_ylabel("bg-corr", fontsize=8)

    def _render_big(self):
        stack = self.raw if self.big_mode == "raw" else self.corrected
        limits = self.raw_limits if self.big_mode == "raw" else self.corr_limits
        z = self.z_current
        highlight = panel_highlight_class(z, self.peak, self.support)
        tag = {"peak": "PEAK plane", "support": "support plane", "none": "plane"}[highlight]
        self._draw_panel(
            self.ax_big, stack[z], limits,
            title=f"SINGLE PLANE {self.plane_numbers[z]:02d} ({self.big_mode}, {tag})",
            highlight=highlight,
        )

    def _render_mip(self):
        stack = self.raw if self.big_mode == "raw" else self.corrected
        limits = self.raw_limits if self.big_mode == "raw" else self.corr_limits
        mip = max_intensity_projection(stack)
        self._draw_panel(self.ax_mip, mip, limits, title=MIP_CAPTION, highlight="none")
        self.ax_mip.title.set_color("#B00020")
        self.ax_mip.title.set_fontsize(7)

    def _render_overlay(self):
        self.ax_overlay.clear()
        self.ax_overlay.set_xticks([])
        self.ax_overlay.set_yticks([])
        if self.overlay_on:
            self.ax_overlay.imshow(colour_coded_z_projection(self.raw))
            self.ax_overlay.set_title(
                "Z-COLOUR overlay - display only (o: toggle)", fontsize=7,
            )
            cy, cx = self.centre
            self.ax_overlay.axhline(cy, color="white", lw=0.4)
            self.ax_overlay.axvline(cx, color="white", lw=0.4)
        else:
            self.ax_overlay.text(
                0.5, 0.5, "overlay off\n(press o)", ha="center", va="center",
                fontsize=9, color="#888888",
            )

    def _render_profile(self):
        import numpy as np  # noqa: PLC0415

        ax = self.ax_profile
        ax.clear()
        z_range = range(self.z_count)
        intensity = fixed_xy_central_intensity(self.raw)
        ax.plot(z_range, intensity, "o-", color="#0077B6", label="central intensity")
        contrast = np.asarray([
            float(self.candidate.get(f"plane_{i}_contrast", "nan") or "nan")
            for i in z_range
        ])
        if np.isfinite(contrast).any():
            twin = ax.twinx()
            twin.plot(z_range, contrast, "s--", color="#D4A900", label="local contrast")
            twin.axhline(
                self.config.detection.z_support_min_contrast,
                color="gray", ls=":", lw=0.8,
            )
            twin.set_ylabel("local contrast", fontsize=8, color="#D4A900")
        ax.axvline(self.peak, color="#FFD400", lw=1.5)
        for s in self.support:
            ax.axvline(s, color="#39FF14", lw=0.8, alpha=0.5)
        ax.set_xlabel("optical plane index (fixed XY)", fontsize=8)
        ax.set_ylabel("central intensity", fontsize=8, color="#0077B6")
        ax.set_xticks(list(z_range))
        ax.grid(alpha=0.2)

    def _render_info(self):
        ax = self.ax_info
        ax.clear()
        ax.axis("off")
        candidate = self.candidate
        prev = previous_label(self.labels, candidate)
        info = [
            f"candidate {candidate['candidate_id']}  "
            f"({self.index + 1}/{len(self.candidates)})",
            f"PREVIOUS LABEL: {prev or 'none'}"
            + ("   (will not be overwritten silently)" if prev else ""),
            f"status: {candidate.get('current_status', '?')}   "
            f"measurement valid: {candidate.get('measurement_valid', '?')}",
            f"peak plane: {self.peak}   support planes: {sorted(self.support)}",
            f"contrast (robust z): {candidate.get('local_robust_z', '?')}   "
            f"injection excl: {candidate.get('inside_injection_analysis_exclusion', '?')}",
            "",
            "1 cell  2 artefact  3 injection  4 uncertain  5 skip",
            "left/b prev   right/n next   up/down scrub Z   r raw/bg   o overlay   q quit",
            "",
            "NOTE: a candidate is not a cell until a human label or a validated",
            "classifier prediction. The max projection is a display aid only.",
        ]
        ax.text(0, 1, "\n".join(info), va="top", family="monospace", fontsize=8.5)

    # ------------------------------------------------------------------ #
    # Lightweight updates (no full rebuild)
    # ------------------------------------------------------------------ #
    def _refresh_z_dependent(self):
        self._render_big()
        self._render_montage()  # current-plane marker moves
        self.fig.canvas.draw_idle()

    def _on_slider(self, value):
        if self._slider_guard:
            return
        self.z_current = int(round(value))
        self._refresh_z_dependent()

    def _set_z(self, new_z):
        new_z = max(0, min(self.z_count - 1, new_z))
        if new_z == self.z_current:
            return
        self.z_current = new_z
        self._slider_guard = True
        self.slider.set_val(new_z)
        self._slider_guard = False
        self._refresh_z_dependent()

    # ------------------------------------------------------------------ #
    # Navigation + labelling
    # ------------------------------------------------------------------ #
    def _advance(self, step):
        self.index = max(0, min(len(self.candidates) - 1, self.index + step))
        self.draw_candidate()

    def on_key(self, event):
        key = event.key
        if key in LABEL_KEYS:
            candidate = self.candidates[self.index]
            try:
                record = save_manual_label(
                    self.labels_path, candidate, LABEL_KEYS[key], self.reviewer,
                    allow_overwrite=self.allow_label_changes,
                )
            except LabelConflictError as exc:
                print(f"LABEL NOT CHANGED: {exc}")
                return
            self.labels[(candidate["candidate_id"], candidate["channel"])] = record
            self._advance(+1)
        elif key == "5":
            self._advance(+1)
        elif key in ("right", "n"):
            self._advance(+1)
        elif key in ("left", "b"):
            self._advance(-1)
        elif key in ("up", "]"):
            self._set_z(self.z_current + 1)
        elif key in ("down", "["):
            self._set_z(self.z_current - 1)
        elif key == "r":
            self.big_mode = "corrected" if self.big_mode == "raw" else "raw"
            self._render_big()
            self._render_mip()
            self.fig.canvas.draw_idle()
        elif key == "o":
            self.overlay_on = not self.overlay_on
            self._render_overlay()
            self.fig.canvas.draw_idle()
        elif key == "q":
            self._plt.close(self.fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Review candidates across seven aligned planes.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--channel", required=True, choices=["green_signal", "channel_2_signal"])
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--reviewer", default=getpass.getuser())
    parser.add_argument(
        "--filter", default="random_sample",
        choices=[
            "preliminary_rule_pass", "preliminary_rule_fail", "near_threshold",
            "single_plane", "many_planes", "outside_injection", "inside_injection",
            "invalid_measurement", "random_sample", "all",
        ],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--include-labelled", action="store_true",
        help="Include already labelled candidates; default resumes with unlabelled candidates.",
    )
    parser.add_argument(
        "--allow-label-changes", action="store_true",
        help="Explicitly permit replacing an existing different manual label.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    candidates_path = Path(args.candidates or config.work_dir / "candidates" / "all_candidates.csv")
    labels_path = Path(args.labels or config.work_dir / "candidates" / "manual_labels.csv")
    rows = [row for row in read_csv_rows(candidates_path) if row.get("channel") == args.channel]
    existing_labels = load_manual_labels(labels_path)
    if not args.include_labelled:
        rows = unreviewed_candidates(rows, existing_labels)
    rows = filter_review_candidates(
        rows, args.filter,
        contrast_threshold=config.detection.minimum_local_robust_z,
        random_seed=config.classifier.random_seed,
        limit=args.limit,
    )
    if not rows:
        print(f"No candidates match channel={args.channel!r}, filter={args.filter!r}.")
        return 1

    indexes = {
        "green_signal": index_channel(
            "green_signal", config.data.green_signal_dir, config.data.filename_regex
        ),
        "channel_2_signal": index_channel(
            "channel_2_signal", config.data.channel_2_signal_dir, config.data.filename_regex
        ),
    }
    Reviewer(
        rows, indexes, config, labels_path, args.reviewer,
        allow_label_changes=args.allow_label_changes,
    )
    import matplotlib.pyplot as plt
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
