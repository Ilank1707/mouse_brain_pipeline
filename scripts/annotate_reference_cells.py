#!/usr/bin/env python
"""Click manual marker-positive reference points for candidate-recall auditing."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from mouse_brain_pipeline.audit import index_channel
from mouse_brain_pipeline.config import load_config
from mouse_brain_pipeline.reference_audit import (
    make_reference_point,
    read_reference_points,
    write_reference_points_atomic,
)


class ReferenceAnnotator:
    def __init__(self, plane_paths, channel, section, crop, output, reviewer):
        import matplotlib.pyplot as plt
        import tifffile

        self.channel = channel
        self.section = section
        self.output = output
        self.reviewer = reviewer
        self.plane_numbers = sorted(plane_paths)
        self.handles = [tifffile.TiffFile(str(plane_paths[p])) for p in self.plane_numbers]
        self.arrays = []
        for handle in self.handles:
            page = handle.pages[0]
            try:
                self.arrays.append(page.asarray(out="memmap"))
            except (TypeError, ValueError):
                self.arrays.append(page.asarray())
        height, width = self.arrays[0].shape[:2]
        if crop:
            x0, x1, y0, y1 = crop
            self.x0, self.x1 = max(0, x0), min(width, x1)
            self.y0, self.y1 = max(0, y0), min(height, y1)
        else:
            self.x0, self.x1, self.y0, self.y1 = 0, width, 0, height
        self.source_crop = f"{self.x0}:{self.x1},{self.y0}:{self.y1}"
        self.rows = read_reference_points(output)
        self.session_ids = []
        self.z_index = 0
        # Display-only brightness control; never alters the underlying data.
        self.display_upper_percentile = 99.8
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.draw()

    def current_rows(self):
        return [
            row for row in self.rows
            if row.get("channel") == self.channel
            and int(float(row.get("section", -1))) == self.section
            and int(float(row.get("z_index", -1))) == self.z_index
        ]

    def draw(self):
        import numpy as np

        self.ax.clear()
        image = self.arrays[self.z_index][self.y0:self.y1, self.x0:self.x1]
        finite = np.asarray(image)[np.isfinite(image)]
        low, high = (
            np.percentile(finite, [1.0, self.display_upper_percentile])
            if finite.size else (0, 1)
        )
        self.ax.imshow(image, cmap="gray", vmin=low, vmax=max(high, low + 1), origin="upper")
        rows = self.current_rows()
        if rows:
            self.ax.scatter(
                [float(row["x_global_px"]) - self.x0 for row in rows],
                [float(row["y_global_px"]) - self.y0 for row in rows],
                facecolors="none", edgecolors="#00FFFF", s=70, linewidths=1.2,
            )
        self.ax.set_title(
            f"{self.channel} section {self.section}, z_index={self.z_index}, "
            f"optical plane={self.plane_numbers[self.z_index]}  "
            f"[display p{self.display_upper_percentile:.1f}, data unchanged]\n"
            "Click reference object centres | left/right change plane | "
            "+/- brightness | u undo | q quit\n"
            "Reference annotations only — not final cells or counts"
        )
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        row = make_reference_point(
            channel=self.channel,
            section=self.section,
            x_global_px=self.x0 + event.xdata,
            y_global_px=self.y0 + event.ydata,
            z_index=self.z_index,
            optical_plane=self.plane_numbers[self.z_index],
            reviewer=self.reviewer,
            source_crop=self.source_crop,
        )
        self.rows.append(row)
        self.session_ids.append(row["reference_id"])
        write_reference_points_atomic(self.output, self.rows)
        self.draw()

    def on_key(self, event):
        import matplotlib.pyplot as plt

        if event.key in {"right", "]"}:
            self.z_index = min(len(self.plane_numbers) - 1, self.z_index + 1)
            self.draw()
        elif event.key in {"left", "["}:
            self.z_index = max(0, self.z_index - 1)
            self.draw()
        elif event.key in {"+", "="}:
            # Display only: brighten by lowering the upper percentile.
            self.display_upper_percentile = max(90.0, self.display_upper_percentile - 0.5)
            self.draw()
        elif event.key in {"-", "_"}:
            self.display_upper_percentile = min(100.0, self.display_upper_percentile + 0.5)
            self.draw()
        elif event.key == "u" and self.session_ids:
            reference_id = self.session_ids.pop()
            self.rows = [
                row for row in self.rows if row.get("reference_id") != reference_id
            ]
            write_reference_points_atomic(self.output, self.rows)
            self.draw()
        elif event.key == "q":
            plt.close(self.fig)

    def on_close(self, _event):
        for handle in self.handles:
            handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate manual candidate-recall references.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--channel", required=True, choices=["green_signal", "channel_2_signal"])
    parser.add_argument("--section", type=int, default=None)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--reviewer", default=getpass.getuser())
    args = parser.parse_args()

    config = load_config(args.config)
    section = args.section if args.section is not None else config.pilot.first_section
    directory = (
        config.data.green_signal_dir if args.channel == "green_signal"
        else config.data.channel_2_signal_dir
    )
    index = index_channel(args.channel, directory, config.data.filename_regex)
    plane_paths = {
        plane: path for (candidate_section, plane), path in index.files.items()
        if candidate_section == section
    }
    if len(plane_paths) != config.acquisition.planes_per_section:
        print(
            f"ERROR: section {section} has {len(plane_paths)} available planes; "
            f"expected {config.acquisition.planes_per_section}.",
            file=sys.stderr,
        )
        return 2
    output = Path(
        args.output
        or config.work_dir / "candidates" / "manual_reference_points.csv"
    )
    ReferenceAnnotator(
        plane_paths, args.channel, section, args.crop, output, args.reviewer
    )
    import matplotlib.pyplot as plt
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
