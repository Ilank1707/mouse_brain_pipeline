#!/usr/bin/env python
"""Two-dimensional candidate-to-candidate pair-correlation analysis.

The analysis uses one XY coordinate per row of ``all_candidates.csv``. Pair
counts are computed with ``scipy.spatial.cKDTree`` and normalized against
complete-spatial-randomness (CSR) simulations in the saved tissue mask. Raw
TIFFs, candidate records, masks, classifications, and measurements are read
only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable

import _bootstrap  # noqa: F401  # adds project src/ to sys.path

from mouse_brain_pipeline.coordinate_exports import is_confirmed_cell


DEFAULT_BIN_WIDTH_UM = 5.0
DEFAULT_MAXIMUM_DISTANCE_UM = 500.0
DEFAULT_SIMULATIONS = 99
DEFAULT_RANDOM_SEED = 12345
MINIMUM_CANDIDATES = 2

REQUIRED_CANDIDATE_COLUMNS = {
    "candidate_id",
    "channel",
    "section",
    "x_global_px",
    "y_global_px",
    "current_status",
}

CSV_COLUMNS = [
    "radius_start_um",
    "radius_end_um",
    "radius_mid_um",
    "observed_pair_count",
    "observed_pair_density_per_mm2",
    "csr_mean_pair_density_per_mm2",
    "csr_lower_95",
    "csr_upper_95",
    "g_r",
    "g_r_lower_95",
    "g_r_upper_95",
    "number_of_candidates",
    "status",
    "channel",
]


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _manual_label(row: dict) -> str:
    label = str(row.get("manual_label", "")).strip().lower()
    return "artefact" if label == "artifact" else label


def _is_artefact(row: dict) -> bool:
    return _manual_label(row) == "artefact" or row.get("current_status") in {
        "artifact",
        "manual_artifact",
        "predicted_artifact",
    }


def _is_uncertain(row: dict) -> bool:
    return _manual_label(row) == "uncertain"


def _is_injection(row: dict) -> bool:
    return _manual_label(row) == "injection" or row.get("current_status") == "injection_site"


# (output status, selector, restrict to tissue minus injection exclusion)
SERIES: tuple[tuple[str, Callable[[dict], bool], bool], ...] = (
    ("all_outside_injection", lambda _row: True, True),
    ("preliminary_pass", lambda row: row.get("current_status") == "preliminary_rule_pass", True),
    ("preliminary_fail", lambda row: row.get("current_status") == "preliminary_rule_fail", True),
    ("manual_review", lambda row: row.get("current_status") == "manual_review", True),
    (
        "invalid_measurement",
        lambda row: row.get("current_status") == "invalid_measurement",
        True,
    ),
    ("confirmed_cell", is_confirmed_cell, True),
    ("artefact", _is_artefact, True),
    ("uncertain", _is_uncertain, True),
    # Confirmed/human injection candidates are intentionally evaluated in the
    # tissue window: applying the injection exclusion would remove the requested
    # injection-labelled population itself.
    ("injection", _is_injection, False),
)


class MaskWindow:
    """Read-only 2D sampling window backed by arrays or memory-mapped NPY files."""

    def __init__(self, tissue_mask, exclusion_mask=None):
        import numpy as np

        self.tissue = np.asanyarray(tissue_mask)
        self.exclusion = (
            None if exclusion_mask is None else np.asanyarray(exclusion_mask)
        )
        if self.tissue.ndim != 2:
            raise ValueError(f"Tissue mask must be 2D (y,x), got {self.tissue.shape}")
        if self.exclusion is not None and self.exclusion.shape != self.tissue.shape:
            raise ValueError(
                "Injection exclusion mask shape does not match tissue mask: "
                f"{self.exclusion.shape} vs {self.tissue.shape}"
            )
        self.height, self.width = map(int, self.tissue.shape)
        self._area_pixels: int | None = None

    @classmethod
    def from_npy(cls, tissue_path: Path, exclusion_path: Path | None = None):
        import numpy as np

        tissue = np.load(tissue_path, mmap_mode="r")
        exclusion = (
            np.load(exclusion_path, mmap_mode="r")
            if exclusion_path is not None
            else None
        )
        return cls(tissue, exclusion)

    def contains(self, ys, xs):
        """Return membership for integer local ``(y,x)`` pixel coordinates."""
        import numpy as np

        ys = np.asarray(ys, dtype=np.int64)
        xs = np.asarray(xs, dtype=np.int64)
        if ys.shape != xs.shape:
            raise ValueError("ys and xs must have identical shapes")

        in_bounds = (
            (ys >= 0)
            & (ys < self.height)
            & (xs >= 0)
            & (xs < self.width)
        )
        result = np.zeros(ys.shape, dtype=bool)
        indices = np.flatnonzero(in_bounds)
        if not indices.size:
            return result

        allowed = np.asarray(self.tissue[ys[indices], xs[indices]], dtype=bool)
        if self.exclusion is not None:
            excluded = np.asarray(
                self.exclusion[ys[indices], xs[indices]], dtype=bool
            )
            allowed &= ~excluded
        result[indices] = allowed
        return result

    def area_pixels(self, rows_per_chunk: int = 256) -> int:
        """Count valid pixels without materializing a full-resolution copy."""
        import numpy as np

        if self._area_pixels is not None:
            return self._area_pixels
        total = 0
        for y0 in range(0, self.height, rows_per_chunk):
            y1 = min(self.height, y0 + rows_per_chunk)
            tissue = np.asarray(self.tissue[y0:y1], dtype=bool)
            if self.exclusion is None:
                total += int(np.count_nonzero(tissue))
            else:
                exclusion = np.asarray(self.exclusion[y0:y1], dtype=bool)
                total += int(np.count_nonzero(tissue & ~exclusion))
        self._area_pixels = total
        return total

    def random_points_um(
        self,
        number_of_points: int,
        rng,
        voxel_yx_um: tuple[float, float],
    ):
        """Uniform continuous CSR points within valid mask pixels."""
        import numpy as np

        if number_of_points < 0:
            raise ValueError("number_of_points cannot be negative")
        if number_of_points == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.area_pixels() == 0:
            raise ValueError("The valid sampling mask is empty")

        voxel_y_um, voxel_x_um = map(float, voxel_yx_um)
        points = np.empty((number_of_points, 2), dtype=np.float64)
        filled = 0
        attempted = 0
        maximum_attempts = max(1_000_000, number_of_points * 100_000)

        while filled < number_of_points:
            needed = number_of_points - filled
            draw = max(4096, min(1_000_000, needed * 4))
            ys = rng.integers(0, self.height, size=draw, dtype=np.int64)
            xs = rng.integers(0, self.width, size=draw, dtype=np.int64)
            accepted = self.contains(ys, xs)
            take = min(needed, int(accepted.sum()))
            if take:
                accepted_ys = ys[accepted][:take]
                accepted_xs = xs[accepted][:take]
                # Integer image coordinates denote pixel centres. Jittering by
                # [-0.5, 0.5) samples uniformly within each accepted mask pixel.
                points[filled : filled + take, 0] = (
                    accepted_xs + rng.random(take) - 0.5
                ) * voxel_x_um
                points[filled : filled + take, 1] = (
                    accepted_ys + rng.random(take) - 0.5
                ) * voxel_y_um
                filled += take
            attempted += draw
            if attempted > maximum_attempts and filled < number_of_points:
                raise RuntimeError(
                    "CSR rejection sampling did not converge; the tissue mask "
                    "may be extremely sparse"
                )
        return points


def pair_histogram(points_um, edges_um):
    """Count unique unordered non-self pairs in distance annuli.

    ``cKDTree.count_neighbors`` returns self-pairs and counts each non-self pair
    in both directions when a tree is compared with itself. Subtracting ``N``
    and dividing by two gives unique unordered pairs without constructing an
    ``N x N`` distance matrix or storing individual pair records.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    points = np.asarray(points_um, dtype=np.float64)
    edges = np.asarray(edges_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_um must have shape (N, 2)")
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0:
        raise ValueError("edges_um must be a 1D array beginning at zero")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges_um must be strictly increasing")

    number_of_points = len(points)
    if number_of_points < 2:
        return np.zeros(len(edges) - 1, dtype=np.int64)

    tree = cKDTree(points)
    cumulative_ordered = np.asarray(
        tree.count_neighbors(tree, edges[1:], cumulative=True),
        dtype=np.int64,
    )
    cumulative_unordered = (cumulative_ordered - number_of_points) // 2
    return np.diff(
        np.concatenate((np.zeros(1, dtype=np.int64), cumulative_unordered))
    )


def _pair_density_per_mm2(pair_counts, number_of_points: int, annulus_area_um2):
    """Mean neighbouring-candidate density represented by unordered pairs."""
    import numpy as np

    counts = np.asarray(pair_counts, dtype=np.float64)
    area = np.asarray(annulus_area_um2, dtype=np.float64)
    if number_of_points < 1:
        return np.full(area.shape, np.nan)
    return (2.0 * counts / number_of_points) / area * 1.0e6


def analyze_pair_correlation(
    points_um,
    window: MaskWindow,
    edges_um,
    *,
    simulations: int,
    random_seed: int,
    voxel_yx_um: tuple[float, float],
):
    """Calculate observed g(r), CSR density, and pointwise 95% envelopes."""
    import numpy as np

    points = np.asarray(points_um, dtype=np.float64)
    edges = np.asarray(edges_um, dtype=np.float64)
    number_of_points = len(points)
    if number_of_points < MINIMUM_CANDIDATES:
        raise ValueError("At least two candidate coordinates are required")
    if simulations < 1:
        raise ValueError("simulations must be at least 1")

    observed_counts = pair_histogram(points, edges)
    annulus_area_um2 = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    observed_density = _pair_density_per_mm2(
        observed_counts, number_of_points, annulus_area_um2
    )

    rng = np.random.default_rng(random_seed)
    csr_counts = np.empty((simulations, len(edges) - 1), dtype=np.int64)
    for simulation_index in range(simulations):
        csr_points = window.random_points_um(number_of_points, rng, voxel_yx_um)
        csr_counts[simulation_index] = pair_histogram(csr_points, edges)

    csr_density = _pair_density_per_mm2(
        csr_counts, number_of_points, annulus_area_um2
    )
    csr_mean_density = csr_density.mean(axis=0)
    csr_lower = np.percentile(csr_density, 2.5, axis=0)
    csr_upper = np.percentile(csr_density, 97.5, axis=0)

    valid = csr_mean_density > 0
    g_r = np.full(csr_mean_density.shape, np.nan)
    g_r[valid] = observed_density[valid] / csr_mean_density[valid]

    csr_g = np.full(csr_density.shape, np.nan)
    csr_g[:, valid] = csr_density[:, valid] / csr_mean_density[valid]
    g_lower = np.full(csr_mean_density.shape, np.nan)
    g_upper = np.full(csr_mean_density.shape, np.nan)
    if valid.any():
        g_lower[valid] = np.percentile(csr_g[:, valid], 2.5, axis=0)
        g_upper[valid] = np.percentile(csr_g[:, valid], 97.5, axis=0)

    return {
        "radius_start_um": edges[:-1],
        "radius_end_um": edges[1:],
        "radius_mid_um": (edges[:-1] + edges[1:]) / 2.0,
        "observed_pair_count": observed_counts,
        "observed_pair_density_per_mm2": observed_density,
        "csr_mean_pair_density_per_mm2": csr_mean_density,
        "csr_lower_95": csr_lower,
        "csr_upper_95": csr_upper,
        "g_r": g_r,
        "g_r_lower_95": g_lower,
        "g_r_upper_95": g_upper,
        "number_of_candidates": number_of_points,
    }


def distance_edges(bin_width_um: float, maximum_distance_um: float):
    """Bin edges ending exactly at ``maximum_distance_um``."""
    import numpy as np

    if bin_width_um <= 0:
        raise ValueError("--bin-width-um must be greater than zero")
    if maximum_distance_um <= 0:
        raise ValueError("--maximum-distance-um must be greater than zero")
    edges = np.arange(0.0, maximum_distance_um, bin_width_um, dtype=float)
    if not len(edges) or edges[0] != 0:
        edges = np.insert(edges, 0, 0.0)
    if edges[-1] < maximum_distance_um:
        edges = np.append(edges, float(maximum_distance_um))
    return edges


def _read_candidates(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing candidate table: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_CANDIDATE_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )
        return list(reader), fieldnames


def _run_metadata(run_dir: Path) -> dict:
    path = run_dir / "candidate_run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing run metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _crop_origin_yx(metadata: dict) -> tuple[int, int]:
    crop = metadata.get("crop_x_min_x_max_y_min_y_max")
    if crop is None:
        return 0, 0
    if not isinstance(crop, list) or len(crop) != 4:
        raise ValueError("Invalid crop_x_min_x_max_y_min_y_max in run metadata")
    return int(crop[2]), int(crop[0])


def _points_for_rows(
    rows: list[dict],
    window: MaskWindow,
    crop_origin_yx: tuple[int, int],
    voxel_yx_um: tuple[float, float],
):
    """Return rows/XY positions whose rounded local pixel lies in ``window``."""
    import numpy as np

    if not rows:
        return [], np.empty((0, 2), dtype=np.float64), 0

    origin_y, origin_x = crop_origin_yx
    voxel_y_um, voxel_x_um = map(float, voxel_yx_um)
    parsed_rows = []
    local_x = []
    local_y = []
    invalid_coordinates = 0

    for row in rows:
        try:
            x = float(row["x_global_px"]) - origin_x
            y = float(row["y_global_px"]) - origin_y
        except (KeyError, TypeError, ValueError):
            invalid_coordinates += 1
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            invalid_coordinates += 1
            continue
        parsed_rows.append(row)
        local_x.append(x)
        local_y.append(y)

    xs = np.asarray(local_x, dtype=np.float64)
    ys = np.asarray(local_y, dtype=np.float64)
    inside = window.contains(
        np.rint(ys).astype(np.int64),
        np.rint(xs).astype(np.int64),
    )
    kept_rows = [row for row, keep in zip(parsed_rows, inside) if keep]
    points_um = np.column_stack((xs[inside] * voxel_x_um, ys[inside] * voxel_y_um))
    dropped_by_window = int((~inside).sum())
    return kept_rows, points_um, invalid_coordinates + dropped_by_window


def _csv_value(value):
    import numpy as np

    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "NaN"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_result_csv(
    path: Path,
    result: dict,
    *,
    status: str,
    channel: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index in range(len(result["radius_mid_um"])):
            row = {
                column: result[column][index]
                for column in CSV_COLUMNS
                if column in result and column != "number_of_candidates"
            }
            row.update(
                {
                    "number_of_candidates": result["number_of_candidates"],
                    "status": status,
                    "channel": channel,
                }
            )
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _plot_g_r(
    path: Path,
    result: dict,
    *,
    channel: str,
    section: int,
    status: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        radius = result["radius_mid_um"]
        ax.fill_between(
            radius,
            result["g_r_lower_95"],
            result["g_r_upper_95"],
            alpha=0.25,
            label="95% pointwise CSR envelope",
        )
        ax.plot(radius, result["g_r"], linewidth=1.5, label="observed g(r)")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="CSR g(r)=1")
        ax.set(
            xlabel="candidate-to-candidate XY separation distance (µm)",
            ylabel="normalized pair-correlation g(r)",
            title=(
                f"Pair correlation of PROVISIONAL candidates\n"
                f"{channel}, section {section:03d}, {status}, "
                f"n={result['number_of_candidates']}"
            ),
        )
        ax.set_xlim(0.0, float(result["radius_end_um"][-1]))
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _plot_pair_density(
    path: Path,
    result: dict,
    *,
    channel: str,
    section: int,
    status: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        radius = result["radius_mid_um"]
        ax.fill_between(
            radius,
            result["csr_lower_95"],
            result["csr_upper_95"],
            alpha=0.25,
            label="95% pointwise CSR envelope",
        )
        ax.plot(
            radius,
            result["observed_pair_density_per_mm2"],
            linewidth=1.4,
            label="observed neighbouring-candidate density",
        )
        ax.plot(
            radius,
            result["csr_mean_pair_density_per_mm2"],
            linestyle="--",
            linewidth=1.2,
            label="mean CSR density",
        )
        ax.set(
            xlabel="candidate-to-candidate XY separation distance (µm)",
            ylabel="neighbouring-candidate density per mm²",
            title=(
                f"Pair density of PROVISIONAL candidates\n"
                f"{channel}, section {section:03d}, {status}"
            ),
        )
        ax.set_xlim(0.0, float(result["radius_end_um"][-1]))
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _prepare_output_root(out_dir: Path, run_dir: Path, channel: str) -> Path:
    """Bind an output root to one run and reserve a fresh channel subfolder."""
    marker_path = out_dir / "pair_correlation_run.json"
    expected_run = str(run_dir.resolve())

    if out_dir.exists():
        entries = list(out_dir.iterdir())
        if entries and not marker_path.is_file():
            raise FileExistsError(
                f"--out-dir is not an isolated pair-correlation folder: {out_dir}"
            )
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if str(Path(marker.get("run_dir", "")).resolve()) != expected_run:
                raise FileExistsError(
                    f"--out-dir belongs to another run: {marker.get('run_dir')}"
                )
    else:
        out_dir.mkdir(parents=True)

    marker_path.write_text(
        json.dumps(
            {
                "run_dir": expected_run,
                "analysis": "2D candidate-to-candidate pair correlation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    channel_dir = out_dir / channel
    if channel_dir.exists() and any(channel_dir.iterdir()):
        raise FileExistsError(
            f"Channel output already exists; choose a fresh --out-dir: {channel_dir}"
        )
    channel_dir.mkdir(parents=True, exist_ok=True)
    return channel_dir


def run_analysis(
    *,
    config,
    run_dir: Path,
    channel: str,
    section: int,
    out_dir: Path,
    bin_width_um: float = DEFAULT_BIN_WIDTH_UM,
    maximum_distance_um: float = DEFAULT_MAXIMUM_DISTANCE_UM,
    simulations: int = DEFAULT_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict:
    """Analyze one channel and section, writing only beneath ``out_dir``."""
    import numpy as np

    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if simulations < 1:
        raise ValueError("--simulations must be at least 1")
    edges_um = distance_edges(bin_width_um, maximum_distance_um)

    candidate_path = run_dir / "all_candidates.csv"
    all_rows, candidate_columns = _read_candidates(candidate_path)
    rows = [
        row
        for row in all_rows
        if row.get("channel") == channel and str(row.get("section")) == str(section)
    ]
    if not rows:
        raise ValueError(f"No candidates for {channel}, section {section}")

    candidate_ids = [str(row.get("candidate_id", "")).strip() for row in rows]
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("Every candidate row must have a non-empty candidate_id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(
            "Duplicate candidate_id rows found; pair correlation requires one row "
            "per unique candidate"
        )

    metadata = _run_metadata(run_dir)
    crop_origin_yx = _crop_origin_yx(metadata)
    voxel_yx_um = (
        float(config.acquisition.voxel_size_y_um),
        float(config.acquisition.voxel_size_x_um),
    )

    section_dir = run_dir / "qc" / f"{channel}_section_{section:03d}"
    tissue_path = section_dir / "tissue_mask.npy"
    exclusion_path = section_dir / "injection_analysis_exclusion_mask.npy"
    if not tissue_path.is_file():
        raise FileNotFoundError(f"Missing saved tissue mask: {tissue_path}")
    if not exclusion_path.is_file():
        raise FileNotFoundError(
            f"Missing channel-specific injection exclusion mask: {exclusion_path}"
        )

    tissue_window = MaskWindow.from_npy(tissue_path)
    outside_window = MaskWindow.from_npy(tissue_path, exclusion_path)
    if tissue_window.tissue.shape != outside_window.tissue.shape:
        raise ValueError("Inconsistent tissue-mask shapes")

    source_dimensions = metadata.get("source_image_dimensions", {}).get(channel, {})
    expected_shape = (
        source_dimensions.get("height"),
        source_dimensions.get("width"),
    )
    crop = metadata.get("crop_x_min_x_max_y_min_y_max")
    if crop is None and all(value is not None for value in expected_shape):
        if tuple(map(int, expected_shape)) != tissue_window.tissue.shape:
            raise ValueError(
                "Full-section tissue mask shape does not match run metadata: "
                f"{tissue_window.tissue.shape} vs {expected_shape}"
            )

    prepared: list[tuple[str, dict, MaskWindow, int]] = []
    series_summary: dict[str, dict] = {}
    for series_index, (status, selector, outside_only) in enumerate(SERIES):
        selected = [row for row in rows if selector(row)]
        window = outside_window if outside_only else tissue_window
        kept_rows, points_um, dropped = _points_for_rows(
            selected, window, crop_origin_yx, voxel_yx_um
        )
        if len(points_um) < MINIMUM_CANDIDATES:
            series_summary[status] = {
                "selected_rows": len(selected),
                "number_of_candidates": len(points_um),
                "dropped_outside_sampling_window_or_invalid": dropped,
                "skipped": "fewer than two candidates in the valid sampling window",
            }
            continue
        series_seed = int(
            np.random.SeedSequence([random_seed, series_index]).generate_state(1)[0]
        )
        prepared.append(
            (
                status,
                {
                    "rows": kept_rows,
                    "points_um": points_um,
                    "dropped": dropped,
                    "outside_only": outside_only,
                },
                window,
                series_seed,
            )
        )

    channel_dir = _prepare_output_root(out_dir, run_dir, channel)
    for status, population, window, series_seed in prepared:
        print(
            f"{channel}/{status}: n={len(population['points_um'])}; "
            f"{simulations} CSR simulations"
        )
        result = analyze_pair_correlation(
            population["points_um"],
            window,
            edges_um,
            simulations=simulations,
            random_seed=series_seed,
            voxel_yx_um=voxel_yx_um,
        )

        status_dir = channel_dir / status
        status_dir.mkdir()
        csv_path = status_dir / "pair_correlation.csv"
        graph_path = status_dir / "pair_correlation_g_r.png"
        density_path = status_dir / "pair_density_per_mm2.png"
        summary_path = status_dir / "analysis_summary.json"

        _write_result_csv(csv_path, result, status=status, channel=channel)
        _plot_g_r(
            graph_path,
            result,
            channel=channel,
            section=section,
            status=status,
        )
        _plot_pair_density(
            density_path,
            result,
            channel=channel,
            section=section,
            status=status,
        )

        status_summary = {
            "analysis": "2D candidate-to-candidate pair correlation g(r)",
            "channel": channel,
            "section": section,
            "status": status,
            "provisional_candidates": True,
            "number_of_candidates": result["number_of_candidates"],
            "one_csv_row_per_candidate_input": True,
            "xy_only": True,
            "array_order": metadata.get("array_order", "z,y,x"),
            "planes_per_physical_section": metadata.get("acquisition", {}).get(
                "planes_per_section"
            ),
            "voxel_size_yx_um": list(voxel_yx_um),
            "crop_origin_yx_px": list(crop_origin_yx),
            "sampling_window": (
                "tissue_mask_minus_channel_injection_exclusion"
                if population["outside_only"]
                else "tissue_mask"
            ),
            "tissue_mask": str(tissue_path),
            "injection_analysis_exclusion_mask": (
                str(exclusion_path) if population["outside_only"] else None
            ),
            "sampling_area_pixels": window.area_pixels(),
            "sampling_area_mm2": (
                window.area_pixels() * voxel_yx_um[0] * voxel_yx_um[1] / 1.0e6
            ),
            "dropped_outside_sampling_window_or_invalid": population["dropped"],
            "bin_width_um": bin_width_um,
            "maximum_distance_um": maximum_distance_um,
            "simulations": simulations,
            "random_seed": random_seed,
            "series_random_seed": series_seed,
            "pair_counting": "cKDTree unordered non-self pairs",
            "normalization": "observed pair density / mean CSR pair density",
            "invalid_expected_bins": "NaN",
            "outputs": {
                "pair_correlation_csv": str(csv_path),
                "pair_correlation_g_r_png": str(graph_path),
                "pair_density_per_mm2_png": str(density_path),
            },
        }
        summary_path.write_text(
            json.dumps(status_summary, indent=2), encoding="utf-8"
        )
        series_summary[status] = {
            "number_of_candidates": result["number_of_candidates"],
            "directory": str(status_dir),
        }

    manifest = {
        "analysis": "2D candidate-to-candidate pair correlation g(r)",
        "run_dir": str(run_dir),
        "candidate_table": str(candidate_path),
        "candidate_columns_inspected": candidate_columns,
        "channel": channel,
        "section": section,
        "output_directory": str(channel_dir),
        "bin_width_um": bin_width_um,
        "maximum_distance_um": maximum_distance_um,
        "simulations": simulations,
        "random_seed": random_seed,
        "series": series_summary,
    }
    manifest_path = channel_dir / "analysis_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote pair-correlation outputs to {channel_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2D pair-correlation g(r) among unique candidate XY coordinates."
    )
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--channel",
        required=True,
        choices=("green_signal", "channel_2_signal"),
    )
    parser.add_argument("--section", required=True, type=int)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bin-width-um", type=float, default=DEFAULT_BIN_WIDTH_UM)
    parser.add_argument(
        "--maximum-distance-um",
        type=float,
        default=DEFAULT_MAXIMUM_DISTANCE_UM,
    )
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from mouse_brain_pipeline.config import load_config

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_analysis(
        config=config,
        run_dir=Path(args.run_dir),
        channel=args.channel,
        section=args.section,
        out_dir=Path(args.out_dir),
        bin_width_um=args.bin_width_um,
        maximum_distance_um=args.maximum_distance_um,
        simulations=args.simulations,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
