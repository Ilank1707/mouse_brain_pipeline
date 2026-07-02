"""Tiny stage timer -- accumulate wall-clock time per named stage.

Used to see where a detection / post-processing run spends its time and to write
``stage_timings.csv`` next to the run outputs. Pure standard library.
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path

STAGE_TIMING_COLUMNS = ["stage", "seconds", "calls"]


class StageTimer:
    """Accumulate durations by stage name across many calls.

    Two styles are supported and can be mixed:

        with timer.stage("cellfinder"):
            ...

        timer.start("candidate_measurements")
        ...
        timer.stop("candidate_measurements")
    """

    def __init__(self):
        self.durations: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.order: list[str] = []
        self._open: dict[str, float] = {}

    def _record(self, name: str, seconds: float) -> None:
        if name not in self.durations:
            self.durations[name] = 0.0
            self.calls[name] = 0
            self.order.append(name)
        self.durations[name] += float(seconds)
        self.calls[name] += 1

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, time.perf_counter() - t0)

    def start(self, name: str) -> None:
        self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        t0 = self._open.pop(name, None)
        if t0 is not None:
            self._record(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        """Fold in an externally-measured duration (e.g. from a sub-timer)."""
        self._record(name, seconds)

    def merge(self, other: "StageTimer") -> None:
        for name in other.order:
            if name not in self.durations:
                self.durations[name] = 0.0
                self.calls[name] = 0
                self.order.append(name)
            self.durations[name] += other.durations[name]
            self.calls[name] += other.calls[name]

    def as_rows(self) -> list[dict]:
        return [
            {"stage": name, "seconds": round(self.durations[name], 3),
             "calls": self.calls[name]}
            for name in self.order
        ]

    def total_seconds(self) -> float:
        return sum(self.durations.values())

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=STAGE_TIMING_COLUMNS)
            writer.writeheader()
            for row in self.as_rows():
                writer.writerow(row)
            writer.writerow({"stage": "TOTAL",
                             "seconds": round(self.total_seconds(), 3), "calls": ""})
        return path
