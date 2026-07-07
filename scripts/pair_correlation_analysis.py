#!/usr/bin/env python
"""Two-dimensional candidate-to-candidate pair-correlation analysis.

The analysis uses one XY coordinate per row of ``all_candidates.csv``. Pair
counts are computed with ``scipy.spatial.cKDTree`` and normalized against
complete-spatial-randomness (CSR) simulations in the saved tissue mask.

A second, inhomogeneous variant additionally estimates the local candidate
intensity with a fixed-bandwidth 2D Gaussian kernel density estimate, reweights
pair contributions by ``1 / (lambda_i * lambda_j)``, and normalizes against
simulations drawn from that estimated intensity surface rather than uniform CSR.
This accounts for large-scale spatial variation in candidate density. Both
variants are written to separate files for every analyzed channel and status.

Raw TIFFs, candidate records, masks, classifications, and measurements are read
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

# Inhomogeneous analysis: a 2D Gaussian kernel density estimate of the local
# candidate intensity is used to reweight pair contributions and to generate the
# null simulations, replacing the uniform-CSR assumption. The bandwidth is a
# fixed spatial scale chosen up front (never tuned against the final g(r) curve).
DEFAULT_INTENSITY_BANDWIDTH_UM = 200.0
# Grid cells per KDE bandwidth for the estimated intensity surface. Eight cells
# per bandwidth resolves the smoothing scale without an unnecessarily fine grid.
GRID_CELLS_PER_BANDWIDTH = 8.0
MINIMUM_GRID_STEP_UM = 5.0
# Upper bound on estimated-intensity grid cells to keep memory bounded; the grid
# step is increased if a section would otherwise exceed this.
MAXIMUM_GRID_CELLS = 4_000_000

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

INHOMOGENEOUS_CSV_COLUMNS = [
    "radius_start_um",
    "radius_end_um",
    "radius_mid_um",
    "observed_weighted_pair_sum",
    "simulated_mean_weighted_pair_sum",
    "simulated_lower_95",
    "simulated_upper_95",
    "g_inhom_r",
    "g_inhom_lower_95",
    "g_inhom_upper_95",
    "number_of_candidates",
    "intensity_bandwidth_um",
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

    def downsampled_valid_counts(self, step_px: int, rows_per_chunk: int = 256):
        """Count valid pixels per ``step_px`` x ``step_px`` grid cell.

        Returns ``(counts, grid_height, grid_width)`` where ``counts[cy, cx]`` is
        the number of valid (in-tissue, non-excluded) pixels inside grid cell
        ``(cy, cx)``. The mask is streamed in step-aligned row bands so a
        full-resolution copy is never materialized.
        """
        import numpy as np

        step = int(step_px)
        if step < 1:
            raise ValueError("step_px must be at least 1")
        grid_height = -(-self.height // step)  # ceil division
        grid_width = -(-self.width // step)
        counts = np.zeros((grid_height, grid_width), dtype=np.int64)

        band_rows = max(step, (max(step, rows_per_chunk) // step) * step)
        pad_cols = (-self.width) % step
        for y0 in range(0, self.height, band_rows):
            y1 = min(self.height, y0 + band_rows)
            tissue = np.asarray(self.tissue[y0:y1], dtype=bool)
            if self.exclusion is not None:
                exclusion = np.asarray(self.exclusion[y0:y1], dtype=bool)
                valid = tissue & ~exclusion
            else:
                valid = tissue
            pad_rows = (-valid.shape[0]) % step
            if pad_rows or pad_cols:
                valid = np.pad(valid, ((0, pad_rows), (0, pad_cols)))
            cell_rows = valid.shape[0] // step
            reshaped = valid.reshape(cell_rows, step, grid_width, step)
            band_counts = reshaped.sum(axis=(1, 3)).astype(np.int64)
            cy0 = y0 // step
            counts[cy0 : cy0 + cell_rows] += band_counts
        return counts, grid_height, grid_width

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


def weighted_pair_histogram(points_um, weights, edges_um):
    """Sum unordered non-self pair weights ``w_i * w_j`` in distance annuli.

    ``cKDTree.count_neighbors`` with a weight tuple returns
    ``sum_{i,j} w_i * w_j * 1[d_ij <= r]`` including self-pairs (``i == j``) and
    counting each unordered pair in both directions. Subtracting ``sum_i w_i**2``
    and halving gives the unique unordered weighted sum without building an
    ``N x N`` distance matrix or storing individual pair records.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    points = np.asarray(points_um, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    edges = np.asarray(edges_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_um must have shape (N, 2)")
    if weights.shape != (len(points),):
        raise ValueError("weights must be one value per point")
    if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0:
        raise ValueError("edges_um must be a 1D array beginning at zero")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges_um must be strictly increasing")

    number_of_points = len(points)
    if number_of_points < 2:
        return np.zeros(len(edges) - 1, dtype=np.float64)

    tree = cKDTree(points)
    cumulative_ordered = np.asarray(
        tree.count_neighbors(
            tree, edges[1:], weights=(weights, weights), cumulative=True
        ),
        dtype=np.float64,
    )
    self_term = float(np.sum(weights**2))
    cumulative_unordered = (cumulative_ordered - self_term) / 2.0
    return np.diff(
        np.concatenate((np.zeros(1, dtype=np.float64), cumulative_unordered))
    )


class IntensitySurface:
    """2D Gaussian KDE estimate of candidate intensity inside a mask window.

    Candidate positions are binned onto a regular grid, smoothed with a Gaussian
    kernel of the requested bandwidth (edge-corrected by normalized convolution
    against the valid-mask coverage), and expressed as candidates per um**2. The
    surface supports evaluating the local intensity at arbitrary points and
    drawing simulated candidate locations proportional to the surface, both
    restricted to the valid tissue mask.
    """

    def __init__(
        self,
        *,
        intensity_per_um2,
        valid,
        sample_weight,
        step_px: int,
        voxel_yx_um: tuple[float, float],
        bandwidth_um: float,
        grid_step_um: float,
        cell_area_um2: float,
    ):
        import numpy as np

        self.intensity = np.asarray(intensity_per_um2, dtype=np.float64)
        self.valid = np.asarray(valid, dtype=bool)
        self.step_px = int(step_px)
        self.voxel_y_um, self.voxel_x_um = map(float, voxel_yx_um)
        self.bandwidth_um = float(bandwidth_um)
        self.grid_step_um = float(grid_step_um)
        self.cell_area_um2 = float(cell_area_um2)
        self.grid_height, self.grid_width = self.intensity.shape

        weight = np.asarray(sample_weight, dtype=np.float64).ravel()
        weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 0.0)
        total = float(weight.sum())
        if total <= 0.0:
            raise ValueError("Estimated intensity surface has no positive mass")
        self._cell_index = np.flatnonzero(weight)
        self._cell_prob = weight[self._cell_index] / total

        finite_positive = self.intensity[
            np.isfinite(self.intensity) & (self.intensity > 0.0)
        ]
        # A strictly positive floor guards the 1 / lambda reweighting against
        # zero or near-zero local intensities without discarding candidates.
        self.intensity_floor = (
            float(finite_positive.min()) * 1.0e-3 if finite_positive.size else 1.0
        )

    @classmethod
    def build(
        cls,
        window: MaskWindow,
        points_um,
        *,
        voxel_yx_um: tuple[float, float],
        bandwidth_um: float,
        grid_step_um: float,
    ):
        import numpy as np
        from scipy.ndimage import gaussian_filter

        if bandwidth_um <= 0:
            raise ValueError("--intensity-bandwidth-um must be greater than zero")
        voxel_y_um, voxel_x_um = map(float, voxel_yx_um)
        step_px = max(1, int(round(grid_step_um / voxel_x_um)))
        # Keep the grid bounded so a small bandwidth on a large section cannot
        # allocate an oversized surface; enlarging the step only coarsens it.
        while (
            (-(-window.height // step_px)) * (-(-window.width // step_px))
            > MAXIMUM_GRID_CELLS
        ):
            step_px += 1

        valid_counts, grid_height, grid_width = window.downsampled_valid_counts(step_px)
        cell_area_um2 = (step_px * voxel_x_um) * (step_px * voxel_y_um)
        # Effective grid step after any bounding adjustment, reported verbatim.
        effective_grid_step_um = step_px * (voxel_x_um + voxel_y_um) / 2.0

        points = np.asarray(points_um, dtype=np.float64)
        column_px = points[:, 0] / voxel_x_um
        row_px = points[:, 1] / voxel_y_um
        cell_col = np.clip(
            np.floor(column_px / step_px).astype(np.int64), 0, grid_width - 1
        )
        cell_row = np.clip(
            np.floor(row_px / step_px).astype(np.int64), 0, grid_height - 1
        )
        candidate_counts = np.zeros((grid_height, grid_width), dtype=np.float64)
        np.add.at(candidate_counts, (cell_row, cell_col), 1.0)

        coverage = valid_counts.astype(np.float64) / float(step_px * step_px)
        # Cell-size in um for the Gaussian sigma; X and Y voxels are near-isotropic.
        cell_size_um = step_px * (voxel_x_um + voxel_y_um) / 2.0
        sigma_cells = float(bandwidth_um) / cell_size_um

        smoothed_counts = gaussian_filter(
            candidate_counts, sigma_cells, mode="constant", cval=0.0
        )
        smoothed_coverage = gaussian_filter(
            coverage, sigma_cells, mode="constant", cval=0.0
        )

        valid = valid_counts > 0
        intensity = np.full((grid_height, grid_width), np.nan, dtype=np.float64)
        usable = valid & (smoothed_coverage > 1.0e-6)
        intensity[usable] = (
            smoothed_counts[usable] / smoothed_coverage[usable] / cell_area_um2
        )

        sample_weight = np.where(
            valid & np.isfinite(intensity) & (intensity > 0.0), intensity, 0.0
        )
        return cls(
            intensity_per_um2=intensity,
            valid=valid,
            sample_weight=sample_weight,
            step_px=step_px,
            voxel_yx_um=(voxel_y_um, voxel_x_um),
            bandwidth_um=bandwidth_um,
            grid_step_um=effective_grid_step_um,
            cell_area_um2=cell_area_um2,
        )

    def evaluate(self, points_um):
        """Local intensity (candidates per um**2) at ``points_um`` via cell lookup.

        Non-finite or non-positive cell intensities are floored to a strictly
        positive value so the ``1 / lambda`` reweighting never divides by zero.
        """
        import numpy as np

        points = np.asarray(points_um, dtype=np.float64)
        if points.size == 0:
            return np.empty(0, dtype=np.float64)
        column_px = points[:, 0] / self.voxel_x_um
        row_px = points[:, 1] / self.voxel_y_um
        cell_col = np.clip(
            np.floor(column_px / self.step_px).astype(np.int64),
            0,
            self.grid_width - 1,
        )
        cell_row = np.clip(
            np.floor(row_px / self.step_px).astype(np.int64),
            0,
            self.grid_height - 1,
        )
        lam = self.intensity[cell_row, cell_col]
        lam = np.where(np.isfinite(lam) & (lam > 0.0), lam, self.intensity_floor)
        return np.maximum(lam, self.intensity_floor)

    def sample(self, number_of_points: int, rng, window: MaskWindow):
        """Draw locations proportional to the surface, inside the tissue mask.

        Grid cells are chosen with probability proportional to their intensity;
        a uniform jitter within the chosen cell places the point, which is
        accepted only if its pixel lies in the valid mask. Because each cell's
        acceptance probability equals its valid fraction, the realized density is
        proportional to ``intensity * valid_area`` per cell.
        """
        import numpy as np

        if number_of_points < 0:
            raise ValueError("number_of_points cannot be negative")
        if number_of_points == 0:
            return np.empty((0, 2), dtype=np.float64)

        points = np.empty((number_of_points, 2), dtype=np.float64)
        filled = 0
        attempted = 0
        maximum_attempts = max(1_000_000, number_of_points * 100_000)
        while filled < number_of_points:
            needed = number_of_points - filled
            draw = max(4096, min(1_000_000, needed * 4))
            picks = rng.choice(self._cell_index.size, size=draw, p=self._cell_prob)
            flat = self._cell_index[picks]
            cell_row = flat // self.grid_width
            cell_col = flat % self.grid_width
            column_px = (cell_col + rng.random(draw)) * self.step_px
            row_px = (cell_row + rng.random(draw)) * self.step_px
            accepted = window.contains(
                np.rint(row_px).astype(np.int64),
                np.rint(column_px).astype(np.int64),
            )
            take = min(needed, int(accepted.sum()))
            if take:
                accepted_x = column_px[accepted][:take]
                accepted_y = row_px[accepted][:take]
                points[filled : filled + take, 0] = accepted_x * self.voxel_x_um
                points[filled : filled + take, 1] = accepted_y * self.voxel_y_um
                filled += take
            attempted += draw
            if attempted > maximum_attempts and filled < number_of_points:
                raise RuntimeError(
                    "Intensity-surface sampling did not converge; the valid mask "
                    "may be extremely sparse"
                )
        return points


def resolve_grid_step_um(bandwidth_um: float) -> float:
    """Grid step for the estimated intensity surface (independent of g(r))."""
    return max(float(bandwidth_um) / GRID_CELLS_PER_BANDWIDTH, MINIMUM_GRID_STEP_UM)


def analyze_inhomogeneous_pair_correlation(
    points_um,
    window: MaskWindow,
    edges_um,
    *,
    simulations: int,
    random_seed: int,
    voxel_yx_um: tuple[float, float],
    bandwidth_um: float,
    grid_step_um: float | None = None,
):
    """Inhomogeneous g(r): intensity-reweighted pairs vs. surface simulations.

    A 2D Gaussian KDE with the requested (fixed) ``bandwidth_um`` estimates the
    local candidate intensity ``lambda``. Pair contributions are weighted by
    ``1 / (lambda_i * lambda_j)`` and normalized against simulations drawn from
    the estimated intensity surface rather than uniform CSR.
    """
    import numpy as np

    points = np.asarray(points_um, dtype=np.float64)
    edges = np.asarray(edges_um, dtype=np.float64)
    number_of_points = len(points)
    if number_of_points < MINIMUM_CANDIDATES:
        raise ValueError("At least two candidate coordinates are required")
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if grid_step_um is None:
        grid_step_um = resolve_grid_step_um(bandwidth_um)

    surface = IntensitySurface.build(
        window,
        points,
        voxel_yx_um=voxel_yx_um,
        bandwidth_um=bandwidth_um,
        grid_step_um=grid_step_um,
    )

    observed_lambda = surface.evaluate(points)
    observed_weights = 1.0 / observed_lambda
    observed_weighted = weighted_pair_histogram(points, observed_weights, edges)

    rng = np.random.default_rng(random_seed)
    simulated = np.empty((simulations, len(edges) - 1), dtype=np.float64)
    for simulation_index in range(simulations):
        simulated_points = surface.sample(number_of_points, rng, window)
        simulated_lambda = surface.evaluate(simulated_points)
        simulated_weights = 1.0 / simulated_lambda
        simulated[simulation_index] = weighted_pair_histogram(
            simulated_points, simulated_weights, edges
        )

    simulated_mean = simulated.mean(axis=0)
    simulated_lower = np.percentile(simulated, 2.5, axis=0)
    simulated_upper = np.percentile(simulated, 97.5, axis=0)

    valid = np.isfinite(simulated_mean) & (simulated_mean > 0.0)
    g_inhom = np.full(simulated_mean.shape, np.nan)
    g_inhom[valid] = observed_weighted[valid] / simulated_mean[valid]

    simulated_g = np.full(simulated.shape, np.nan)
    simulated_g[:, valid] = simulated[:, valid] / simulated_mean[valid]
    g_lower = np.full(simulated_mean.shape, np.nan)
    g_upper = np.full(simulated_mean.shape, np.nan)
    if valid.any():
        g_lower[valid] = np.percentile(simulated_g[:, valid], 2.5, axis=0)
        g_upper[valid] = np.percentile(simulated_g[:, valid], 97.5, axis=0)

    return {
        "radius_start_um": edges[:-1],
        "radius_end_um": edges[1:],
        "radius_mid_um": (edges[:-1] + edges[1:]) / 2.0,
        "observed_weighted_pair_sum": observed_weighted,
        "simulated_mean_weighted_pair_sum": simulated_mean,
        "simulated_lower_95": simulated_lower,
        "simulated_upper_95": simulated_upper,
        "g_inhom_r": g_inhom,
        "g_inhom_lower_95": g_lower,
        "g_inhom_upper_95": g_upper,
        "number_of_candidates": number_of_points,
        "intensity_bandwidth_um": float(bandwidth_um),
        "grid_step_um": float(surface.grid_step_um),
        "surface": surface,
    }


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


def _write_inhomogeneous_csv(
    path: Path,
    result: dict,
    *,
    status: str,
    channel: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INHOMOGENEOUS_CSV_COLUMNS)
        writer.writeheader()
        scalar_columns = {
            "number_of_candidates": result["number_of_candidates"],
            "intensity_bandwidth_um": result["intensity_bandwidth_um"],
            "status": status,
            "channel": channel,
        }
        for index in range(len(result["radius_mid_um"])):
            row = {
                column: result[column][index]
                for column in INHOMOGENEOUS_CSV_COLUMNS
                if column in result and column not in scalar_columns
            }
            row.update(scalar_columns)
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _plot_inhomogeneous_g_r(
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
            result["g_inhom_lower_95"],
            result["g_inhom_upper_95"],
            alpha=0.25,
            label="95% intensity-surface simulation envelope",
        )
        ax.plot(
            radius,
            result["g_inhom_r"],
            linewidth=1.5,
            label="observed g_inhom(r)",
        )
        ax.axhline(
            1.0, color="black", linestyle="--", linewidth=1.0, label="g_inhom(r)=1"
        )
        ax.set(
            xlabel="candidate-to-candidate XY separation distance (µm)",
            ylabel="inhomogeneous pair-correlation g_inhom(r)",
            title=(
                f"Inhomogeneous pair correlation of PROVISIONAL candidates\n"
                f"{channel}, section {section:03d}, {status}, "
                f"n={result['number_of_candidates']}, "
                f"KDE bandwidth={result['intensity_bandwidth_um']:.0f} µm"
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


def _plot_intensity_surface(
    path: Path,
    surface: "IntensitySurface",
    *,
    channel: str,
    section: int,
    status: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 7))
    try:
        # Display as candidates per mm**2; invalid cells are left transparent.
        density_per_mm2 = np.where(
            surface.valid & np.isfinite(surface.intensity),
            surface.intensity * 1.0e6,
            np.nan,
        )
        extent_um = (
            0.0,
            surface.grid_width * surface.step_px * surface.voxel_x_um,
            surface.grid_height * surface.step_px * surface.voxel_y_um,
            0.0,
        )
        image = ax.imshow(
            np.ma.masked_invalid(density_per_mm2),
            origin="upper",
            extent=extent_um,
            aspect="equal",
            interpolation="nearest",
        )
        fig.colorbar(image, ax=ax, label="estimated candidate intensity (per mm²)")
        ax.set(
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=(
                f"Estimated candidate intensity surface (PROVISIONAL candidates)\n"
                f"{channel}, section {section:03d}, {status}, "
                f"KDE bandwidth={surface.bandwidth_um:.0f} µm"
            ),
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


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
    intensity_bandwidth_um: float = DEFAULT_INTENSITY_BANDWIDTH_UM,
    statuses: "list[str] | tuple[str, ...] | None" = None,
) -> dict:
    """Analyze one channel and section, writing only beneath ``out_dir``.

    ``statuses`` optionally restricts which status series in :data:`SERIES` are
    analyzed (e.g. only the four spatial-analysis series). It changes NEITHER the
    statistics NOR the per-series random seeding: the original index of each
    status in :data:`SERIES` still drives the seed, so a restricted run produces
    byte-identical results to selecting the same statuses from a full run.
    """
    import numpy as np

    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if simulations < 1:
        raise ValueError("--simulations must be at least 1")
    if intensity_bandwidth_um <= 0:
        raise ValueError("--intensity-bandwidth-um must be greater than zero")
    edges_um = distance_edges(bin_width_um, maximum_distance_um)
    grid_step_um = resolve_grid_step_um(intensity_bandwidth_um)

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
    requested_statuses = None if statuses is None else set(statuses)
    for series_index, (status, selector, outside_only) in enumerate(SERIES):
        # Skip unrequested series but keep ``series_index`` at the status's
        # position in SERIES so the random seed (and therefore the result) is
        # unchanged by the restriction.
        if requested_statuses is not None and status not in requested_statuses:
            continue
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

        # Inhomogeneous analysis: a 2D Gaussian KDE of local candidate intensity
        # reweights pair contributions and generates the null simulations,
        # replacing the uniform-CSR assumption. Written to distinct filenames so
        # the homogeneous outputs above are untouched.
        print(
            f"{channel}/{status}: inhomogeneous ({simulations} intensity-surface "
            f"simulations, KDE bandwidth {intensity_bandwidth_um:g} µm)"
        )
        inhomogeneous_seed = int(
            np.random.SeedSequence([series_seed, 1]).generate_state(1)[0]
        )
        inhomogeneous_result = analyze_inhomogeneous_pair_correlation(
            population["points_um"],
            window,
            edges_um,
            simulations=simulations,
            random_seed=inhomogeneous_seed,
            voxel_yx_um=voxel_yx_um,
            bandwidth_um=intensity_bandwidth_um,
            grid_step_um=grid_step_um,
        )
        surface = inhomogeneous_result.pop("surface")

        inhomogeneous_csv_path = status_dir / "pair_correlation_inhomogeneous.csv"
        inhomogeneous_graph_path = (
            status_dir / "pair_correlation_inhomogeneous_g_r.png"
        )
        intensity_surface_path = status_dir / "estimated_intensity_surface.png"
        inhomogeneous_summary_path = (
            status_dir / "inhomogeneous_analysis_summary.json"
        )

        _write_inhomogeneous_csv(
            inhomogeneous_csv_path,
            inhomogeneous_result,
            status=status,
            channel=channel,
        )
        _plot_inhomogeneous_g_r(
            inhomogeneous_graph_path,
            inhomogeneous_result,
            channel=channel,
            section=section,
            status=status,
        )
        _plot_intensity_surface(
            intensity_surface_path,
            surface,
            channel=channel,
            section=section,
            status=status,
        )

        inhomogeneous_summary = {
            "analysis": "2D candidate-to-candidate inhomogeneous pair correlation "
            "g_inhom(r)",
            "channel": channel,
            "section": section,
            "status": status,
            "provisional_candidates": True,
            "number_of_candidates": inhomogeneous_result["number_of_candidates"],
            "one_csv_row_per_candidate_input": True,
            "xy_only": True,
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
            "bin_width_um": bin_width_um,
            "maximum_distance_um": maximum_distance_um,
            "simulations": simulations,
            "random_seed": inhomogeneous_seed,
            "intensity_estimator": "2D Gaussian kernel density estimate of "
            "candidate XY positions inside the valid tissue mask",
            "intensity_bandwidth_um": inhomogeneous_result["intensity_bandwidth_um"],
            "intensity_grid_step_um": inhomogeneous_result["grid_step_um"],
            "bandwidth_selection": "fixed spatial scale, not tuned against g_inhom(r)",
            "pair_weighting": "1 / (lambda_i * lambda_j)",
            "pair_counting": "cKDTree weighted unordered non-self pairs",
            "normalization": "observed intensity-reweighted pair sum / mean of "
            "simulations sampled from the estimated intensity surface",
            "invalid_expected_bins": "NaN",
            "outputs": {
                "pair_correlation_inhomogeneous_csv": str(inhomogeneous_csv_path),
                "pair_correlation_inhomogeneous_g_r_png": str(
                    inhomogeneous_graph_path
                ),
                "estimated_intensity_surface_png": str(intensity_surface_path),
            },
        }
        inhomogeneous_summary_path.write_text(
            json.dumps(inhomogeneous_summary, indent=2), encoding="utf-8"
        )

        series_summary[status] = {
            "number_of_candidates": result["number_of_candidates"],
            "directory": str(status_dir),
            "intensity_bandwidth_um": inhomogeneous_result["intensity_bandwidth_um"],
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
        "intensity_bandwidth_um": intensity_bandwidth_um,
        "requested_statuses": (
            "all" if requested_statuses is None else sorted(requested_statuses)
        ),
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
    parser.add_argument(
        "--intensity-bandwidth-um",
        type=float,
        default=DEFAULT_INTENSITY_BANDWIDTH_UM,
        help=(
            "Gaussian KDE bandwidth (µm) for the inhomogeneous intensity surface; "
            "a fixed spatial scale, not tuned against g_inhom(r)."
        ),
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
        intensity_bandwidth_um=args.intensity_bandwidth_um,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
