"""Resolve which optional (slow) QC outputs a candidate run should produce.

The pilot writes several expensive QC artefacts (seven-plane images, native
full-resolution QC, review patches). These are optional: ``--fast-qc`` turns on
a bundle of skips for quick iteration, and each output also has its own skip
switch. This resolution is factored out here so it is unit-testable and the rule
"an explicit request always wins over ``--fast-qc``" is enforced in one place.

Nothing here changes any candidate, status, mask, coordinate or count -- it only
decides which QC files get written.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedQcFlags:
    render_seven_planes: bool
    fullres_seven_planes: bool
    write_fullres_qc: bool
    save_review_patches: bool
    skip_pair_correlation: bool
    run_channel_overlay: bool


def resolve_qc_flags(
    *,
    fast_qc: bool = False,
    no_preview: bool = False,
    render_seven_planes: bool = False,
    fullres_seven_planes: bool = False,
    save_review_patches: bool = False,
    skip_seven_plane_qc: bool = False,
    skip_fullres_qc: bool = False,
    skip_review_patches: bool = False,
    skip_pair_correlation: bool = False,
    skip_spatial_analysis: bool = False,
    skip_channel_overlay: bool = False,
) -> ResolvedQcFlags:
    """Resolve the effective optional-QC switches from the raw CLI flags.

    Rules:
      * an explicit per-output skip flag always suppresses that output;
      * ``--fast-qc`` skips seven-plane QC, full-resolution QC and review patches,
        but yields to an EXPLICIT opt-in request (``--render-seven-planes`` /
        ``--save-review-patches`` / ``--fullres-seven-planes``);
      * with no flags, behaviour is unchanged (opt-in outputs stay off, default-on
        outputs stay on).
    """
    fast = bool(fast_qc)
    # Full-resolution native QC has no explicit opt-in flag, so --fast-qc always
    # skips it (the fast preview QC is still written).
    skip_fullres = bool(skip_fullres_qc) or fast
    return ResolvedQcFlags(
        render_seven_planes=bool(render_seven_planes) and not skip_seven_plane_qc,
        fullres_seven_planes=bool(fullres_seven_planes) and not skip_fullres_qc,
        write_fullres_qc=(not bool(no_preview)) and not skip_fullres,
        save_review_patches=bool(save_review_patches) and not skip_review_patches,
        skip_pair_correlation=bool(skip_pair_correlation) or bool(skip_spatial_analysis),
        run_channel_overlay=not bool(skip_channel_overlay),
    )
