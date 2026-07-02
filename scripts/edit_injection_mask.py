#!/usr/bin/env python
"""Interactively draw injection / non-injection polygons for the mask override.

Displays one section's full-resolution max projection and lets you click points
to draw polygons. The most important one is the NON-INJECTION polygon: an area
the automatic mask wrongly includes, subtracted from the final mask so dilation
cannot add it back. Coordinates are saved (full-resolution px) to a separate
override YAML -- config.yml and the raw TIFFs are never modified.

Controls:
  left click : add a point to the current polygon
  e          : draw a NON-INJECTION (exclusion) polygon   [default]
  i          : draw an INJECTION (add) polygon
  u          : undo the last point (or last finished polygon if none pending)
  n / enter  : finish the current polygon, start a new one
  s          : save all polygons to the override YAML
  r          : reload polygons from the override YAML
  q          : quit (finishes the current polygon first)

Example:
  python scripts/edit_injection_mask.py --config config.yml \
      --channel green_signal --section 70 \
      --overrides config_injection_overrides.yml
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from mouse_brain_pipeline import CHANNEL_2_SIGNAL, GREEN_SIGNAL
from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.injection_overrides import (
    read_channel_polygons,
    save_channel_polygons,
)
from mouse_brain_pipeline.utilities import setup_logging


def _max_projection(plane_paths):
    """Full-resolution max projection read plane-by-plane (never holds the stack)."""
    import numpy as np
    import tifffile

    running = None
    for pl in sorted(plane_paths):
        with tifffile.TiffFile(str(plane_paths[pl])) as tf:
            page = tf.pages[0]
            try:
                arr = page.asarray(out="memmap")
            except (ValueError, TypeError):
                arr = page.asarray()
            plane = np.asarray(arr, dtype=np.float32)
        running = plane if running is None else np.maximum(running, plane)
    return running


class MaskEditor:
    def __init__(self, display, channel, section, overrides_path):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.channel = channel
        self.section = section
        self.overrides_path = overrides_path
        self.mode = "non_injection"  # start on the exclusion polygon
        self.injection = []
        self.non_injection = []
        self.current = []

        injection, non_injection = read_channel_polygons(overrides_path, channel)
        self.injection = [list(map(list, poly)) for poly in injection]
        self.non_injection = [list(map(list, poly)) for poly in non_injection]

        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.ax.imshow(display, cmap="gray", origin="upper")
        self.ax.set_title(self._title())
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._redraw()

    def _title(self):
        return (f"{self.channel} section {self.section:03d} -- drawing "
                f"{'NON-INJECTION (exclusion)' if self.mode == 'non_injection' else 'INJECTION'}"
                f"  |  injection={len(self.injection)} non_injection={len(self.non_injection)}"
                f"  |  e/i mode, u undo, n new, s save, r reload, q quit")

    def _target(self):
        return self.non_injection if self.mode == "non_injection" else self.injection

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        self.current.append([float(event.xdata), float(event.ydata)])
        self._redraw()

    def _finish_current(self):
        if len(self.current) >= 3:
            self._target().append(self.current)
        elif self.current:
            print(f"Discarded polygon with only {len(self.current)} point(s).")
        self.current = []

    def on_key(self, event):
        key = (event.key or "").lower()
        if key == "e":
            self._finish_current(); self.mode = "non_injection"
        elif key == "i":
            self._finish_current(); self.mode = "injection"
        elif key == "u":
            if self.current:
                self.current.pop()
            elif self._target():
                self._target().pop()
        elif key in ("n", "enter"):
            self._finish_current()
        elif key == "s":
            self._finish_current(); self._save()
        elif key == "r":
            injection, non_injection = read_channel_polygons(self.overrides_path, self.channel)
            self.injection = [list(map(list, poly)) for poly in injection]
            self.non_injection = [list(map(list, poly)) for poly in non_injection]
            self.current = []
            print(f"Reloaded from {self.overrides_path}")
        elif key == "q":
            self._finish_current()
            self.plt.close(self.fig)
            return
        self._redraw()

    def _save(self):
        path = save_channel_polygons(
            self.overrides_path, self.channel, self.injection, self.non_injection)
        print(f"Saved {len(self.injection)} injection + {len(self.non_injection)} "
              f"non-injection polygon(s) -> {path}")

    def _redraw(self):
        # Clear previous vector overlays, keep the image (first artist).
        for artist in list(self.ax.lines) + list(self.ax.patches):
            artist.remove()
        for poly in self.injection:
            self._draw_poly(poly, "#39FF14")
        for poly in self.non_injection:
            self._draw_poly(poly, "#FF2D2D")
        color = "#FF2D2D" if self.mode == "non_injection" else "#39FF14"
        if self.current:
            xs = [p[0] for p in self.current]
            ys = [p[1] for p in self.current]
            self.ax.plot(xs, ys, "o-", color=color, markersize=4, linewidth=1.0)
        self.ax.set_title(self._title(), fontsize=9)
        self.fig.canvas.draw_idle()

    def _draw_poly(self, poly, color):
        xs = [p[0] for p in poly] + [poly[0][0]]
        ys = [p[1] for p in poly] + [poly[0][1]]
        self.ax.plot(xs, ys, "-", color=color, linewidth=1.4)

    def show(self):
        self.plt.show()


def main() -> int:
    p = argparse.ArgumentParser(description="Draw injection/non-injection mask override polygons.")
    p.add_argument("--config", "-c", default="config.yml")
    p.add_argument("--channel", default=GREEN_SIGNAL,
                   choices=[GREEN_SIGNAL, CHANNEL_2_SIGNAL])
    p.add_argument("--section", type=int, required=True)
    p.add_argument("--overrides", default="config_injection_overrides.yml")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    setup_logging(None, verbose=args.verbose)
    import numpy as np

    cfg = load_config(args.config)
    directory = (cfg.data.green_signal_dir if args.channel == GREEN_SIGNAL
                 else cfg.data.channel_2_signal_dir)
    index = index_channel(args.channel, directory, cfg.data.filename_regex)
    plane_paths = {pl: path for (s, pl), path in index.files.items() if s == args.section}
    if not plane_paths:
        print(f"No planes found for {args.channel} section {args.section} under {directory}")
        return 1

    print(f"Loading full-resolution projection for {args.channel} section {args.section}...")
    proj = _max_projection(plane_paths)
    lo, hi = np.percentile(proj, [1.0, 99.5])
    display = np.clip((proj - lo) / max(hi - lo, 1.0), 0.0, 1.0)

    print("Draw the NON-INJECTION polygon around the false region, then press 's' to save.")
    editor = MaskEditor(display, args.channel, args.section, args.overrides)
    editor.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
