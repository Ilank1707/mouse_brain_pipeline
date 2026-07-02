"""Memory-safety regression tests for Matplotlib QC rendering."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mouse_brain_pipeline.candidate_qc import (  # noqa: E402
    _prepare_qc_display,
    _show_display,
)


def test_full_size_projection_is_downsampled_to_uint8_before_imshow():
    """A full-section shape must never reach imshow as float64 full-res RGBA."""
    source_shape = (9906, 13912)
    # Zero-stride broadcast: it has the real production shape but owns only two
    # bytes, so the test itself cannot hide a multi-gigabyte allocation.
    projection = np.broadcast_to(np.array(100, dtype=np.uint16), source_shape)

    display8, step, extent = _prepare_qc_display(
        projection, display_min=0, display_max=200,
    )

    assert step > 1
    assert display8.dtype == np.uint8
    assert display8.ndim == 2
    assert max(display8.shape) <= 2000
    assert display8.shape != source_shape
    assert display8.nbytes <= 2000 * 2000

    fig, ax = plt.subplots()
    try:
        artist = _show_display(ax, display8, extent, source_shape)
        imshow_array = np.asarray(artist.get_array())
        assert imshow_array.dtype == np.uint8
        assert imshow_array.shape == display8.shape
        assert imshow_array.shape != (*source_shape, 4)

        # Sampled image pixel centres remain at 0, step, 2*step, ... in source
        # coordinates, and the displayed axes use the original source bounds.
        left, right, bottom, top = artist.get_extent()
        assert left + step / 2 == pytest.approx(0)
        assert right - step / 2 == pytest.approx((display8.shape[1] - 1) * step)
        assert top + step / 2 == pytest.approx(0)
        assert bottom - step / 2 == pytest.approx((display8.shape[0] - 1) * step)
        assert ax.get_xlim() == pytest.approx((-0.5, source_shape[1] - 0.5))
        assert ax.get_ylim() == pytest.approx((source_shape[0] - 0.5, -0.5))
    finally:
        plt.close(fig)
