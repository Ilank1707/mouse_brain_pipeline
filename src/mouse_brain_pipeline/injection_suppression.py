"""In-memory injection-core suppression for the second candidate pass (Part 2).

The bright, textured injection blob dominates Cellfinder candidate generation.
To run a second detection pass that can see the rest of the brain, the
conservative injection CORE is replaced -- on a TEMPORARY in-memory copy only --
by a smoothly varying background estimate plus matched noise. This:

  * never modifies the raw TIFFs or the input array;
  * does NOT fill the core with zeros (which would create artificial
    high-contrast edges Cellfinder would re-detect);
  * derives the estimate from a surrounding tissue RING of the SAME channel
    (never the other fluorescence channel);
  * leaves every pixel outside the core untouched;
  * preserves array shape, (z, y, x) ordering and a Cellfinder-safe dtype.
"""

from __future__ import annotations

_EPS = 1e-6


def _ring_width_px(voxel_yx_um: float) -> int:
    # ~30 um ring around the core gives a stable local background estimate.
    return max(3, int(round(30.0 / max(voxel_yx_um, _EPS))))


def suppress_injection_core(
    stack_zyx,
    core_mask,
    voxel_zyx,
    *,
    tissue_mask=None,
    ring_width_px: int | None = None,
    seed: int = 20260625,
):
    """Return a NEW stack copy whose ``core_mask`` pixels are inpainted.

    Plane-by-plane the masked core is replaced by a heavily smoothed estimate
    grown from the surrounding non-core tissue (nearest-tissue fill -> Gaussian
    smoothing) plus Gaussian noise matched to the surrounding ring. The boundary
    is continuous (no hard zero border) because the fill is anchored to the
    immediately adjacent tissue values.

    ``stack_zyx`` is not modified. If ``core_mask`` is empty the input is copied
    through unchanged.
    """
    import numpy as np  # noqa: PLC0415
    from scipy import ndimage as ndi  # noqa: PLC0415

    source = np.asarray(stack_zyx)
    out = source.astype(np.float32, copy=True)
    core = np.asarray(core_mask, dtype=bool)
    if not core.any():
        return out

    _vz, vy, _vx = voxel_zyx
    if ring_width_px is None:
        ring_width_px = _ring_width_px(vy)

    # A ring of genuine tissue immediately around the core supplies the local
    # background level and noise. Stay inside the tissue mask if one is given.
    dilated = ndi.binary_dilation(core, iterations=int(ring_width_px))
    ring = dilated & ~core
    if tissue_mask is not None:
        ring &= np.asarray(tissue_mask, dtype=bool)
    if not ring.any():
        ring = dilated & ~core  # fall back to the geometric ring

    # Smoothing scale ~ the ring width so the estimate varies smoothly across the
    # core rather than reproducing fine texture.
    sigma_px = max(2.0, float(ring_width_px))
    rng = np.random.default_rng(seed)
    Z = out.shape[0]
    for z in range(Z):
        plane = source[z].astype(np.float32, copy=False)

        # Nearest non-core tissue value for every pixel -> continuous fill that
        # matches the adjacent tissue exactly at the boundary.
        known = ~core
        if tissue_mask is not None:
            known &= np.asarray(tissue_mask, dtype=bool)
        if not known.any():
            known = ~core
        nearest_index = ndi.distance_transform_edt(
            ~known, return_distances=False, return_indices=True
        )
        filled = plane[tuple(nearest_index)]

        # Smooth the filled plane so the core estimate varies slowly.
        estimate = ndi.gaussian_filter(filled, sigma=sigma_px)

        ring_vals = plane[ring]
        if ring_vals.size:
            med = float(np.median(ring_vals))
            mad = float(np.median(np.abs(ring_vals - med)))
            noise_sigma = 1.4826 * mad
            if noise_sigma <= _EPS:
                noise_sigma = float(np.std(ring_vals)) if ring_vals.size > 1 else 0.0
        else:
            noise_sigma = 0.0

        noise = rng.normal(0.0, noise_sigma, size=plane.shape).astype(np.float32) \
            if noise_sigma > _EPS else np.zeros_like(plane)
        replacement = estimate + noise
        out[z] = np.where(core, replacement, plane)

    # Keep values non-negative (fluorescence); preserve a Cellfinder-safe dtype.
    np.clip(out, 0.0, None, out=out)
    return out
