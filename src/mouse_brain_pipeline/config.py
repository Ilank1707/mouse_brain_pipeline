"""Typed configuration loaded from ``config.yml``.

Dataclasses mirror ``config.example.yml``. ``Config.from_dict`` lets tests build
a config without writing a YAML file (and without importing PyYAML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Any

from .filenames import DEFAULT_FILENAME_REGEX


def _filtered(cls, d: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only keys that are real dataclass fields of ``cls``.

    Unknown YAML keys are ignored (with the others still applied) instead of
    raising ``TypeError``, so adding a stray/legacy key never crashes loading.
    """
    if not d:
        return {}
    valid = {f.name for f in _dc_fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


@dataclass
class DataConfig:
    green_signal_dir: str = ""       # GREEN biological signal channel
    channel_2_signal_dir: str = ""   # RED biological signal channel (internal name kept)
    background_dir: str | None = None
    work_dir: str = "./work"
    filename_regex: str = DEFAULT_FILENAME_REGEX

    @property
    def has_background(self) -> bool:
        """True only when a real, non-empty background directory is configured."""
        return bool(self.background_dir and str(self.background_dir).strip())


@dataclass
class AcquisitionConfig:
    planes_per_section: int = 7
    voxel_size_z_um: float = 6.0
    voxel_size_y_um: float = 1.004
    voxel_size_x_um: float = 1.004
    cut_thickness_um: float = 42.0

    @property
    def voxel_size_zyx(self) -> tuple[float, float, float]:
        return (self.voxel_size_z_um, self.voxel_size_y_um, self.voxel_size_x_um)

    def validate(self) -> list[str]:
        """Return a list of human-readable consistency warnings (never raises)."""
        warnings: list[str] = []
        expected_cut = self.planes_per_section * self.voxel_size_z_um
        if abs(expected_cut - self.cut_thickness_um) > 1e-6:
            warnings.append(
                f"cut_thickness_um={self.cut_thickness_um} != planes_per_section*"
                f"voxel_size_z_um={expected_cut}. Check acquisition values."
            )
        return warnings


@dataclass
class PilotConfig:
    first_section: int | None = 70
    number_of_sections: int = 1
    tile_size: int = 2048
    tile_overlap: int = 128


@dataclass
class RegistrationConfig:
    orientation: str | None = None  # MUST be confirmed by the user before registration
    atlas: str = "allen_mouse_25um"


@dataclass
class InjectionExclusionConfig:
    """Settings for excluding the broad, densely-labelled injection site.

    The injection site is a large saturated region; it must NOT be counted as
    thousands of cells. A separate mask is built per biological channel because
    the two injection sites may be on different sides. These thresholds are pilot
    defaults and require tuning.
    """

    enabled: bool = True
    automatic: bool = True
    downsample_um: float = 25.0            # work on a low-res projection for speed
    smoothing_sigma_um: float = 100.0      # heavy spatial smoothing scale (~75-150 um)
    # Threshold on the smoothed low-res projection.
    #   "percentile" -> threshold = percentile(intensity_percentile) of the image
    #   "absolute"   -> threshold = injection_threshold_value (raw smoothed units)
    injection_threshold_method: str = "percentile"
    injection_threshold_value: float | None = None  # used only for "absolute"
    intensity_percentile: float = 99.0     # robust high-intensity percentile
    minimum_area_um2: float = 50000.0      # keep only large connected bright regions
    # Drop any connected bright component LARGER than this (a component covering a
    # whole hemisphere is not a plausible injection lobe). None -> no upper bound.
    maximum_component_area_um2: float | None = None
    # Binary opening radius (um) applied to the bright mask BEFORE component
    # labelling. Removes thin spurs and breaks narrow necks so a seed cannot leak
    # across a thin bridge into an unrelated region. 0 -> disabled.
    opening_radius_um: float = 0.0
    # Binary closing radius (um) to fill small holes in the bright mask. It is
    # capped so it can NEVER bridge a gap wider than maximum_bridge_width_um.
    closing_radius_um: float = 0.0
    # Hard limit on how wide a gap morphological closing/dilation may bridge (um).
    # The effective closing radius is min(closing_radius_um, maximum_bridge_width_um/2)
    # so two separate regions closer than this are never merged into one. 0 ->
    # legacy behaviour (closing not capped).
    maximum_bridge_width_um: float = 0.0
    core_dilation_um: float = 50.0
    # Explicit dilation radius (um) applied to the seed-supported base when growing
    # the injection region. None -> use core_dilation_um (back-compatible).
    dilation_radius_um: float | None = None
    analysis_exclusion_dilation_um: float = 150.0
    mask_validated: bool = False
    maximum_mask_fraction_of_tissue: float = 0.25
    maximum_candidate_fraction_inside_mask: float = 0.90
    # Two-pass candidate generation (Part 2). When enabled, a SECOND Cellfinder
    # pass runs on an in-memory copy whose conservative injection CORE is replaced
    # by a smooth background estimate, so the bright textured injection cannot
    # dominate candidate generation. This never touches the raw TIFFs and is
    # SEPARATE from the analysis-exclusion dilation used for interpretation.
    generation_suppression_enabled: bool = False
    # Dilation (um) of the bright base used for the generation-suppression mask.
    # ``None`` -> reuse the conservative injection ``core`` (core_dilation_um).
    # Must stay << analysis_exclusion_dilation_um.
    generation_suppression_dilation_um: float | None = None
    # Deprecated compatibility field. New configurations should use the two
    # explicit boundaries above.
    dilation_um: float | None = None
    # Rectangles in FULL-RESOLUTION pixels: [x_min, x_max, y_min, y_max].
    manual_rectangles: list = field(default_factory=list)
    # Polygons in FULL-RESOLUTION pixels: [[x0,y0],[x1,y1],...]. These ADD area to
    # the injection mask (confirmed injection).
    manual_polygons: list = field(default_factory=list)
    # Polygons in FULL-RESOLUTION pixels that are NOT injection. Subtracted from
    # both the core and analysis masks as the LAST step (after dilation), so a
    # falsely-included nearby region cannot be added back by dilation.
    manual_non_injection_polygons: list = field(default_factory=list)
    # Seed points [x, y] in FULL-RESOLUTION pixels. When set, only automatic
    # injection components containing a seed are kept; disconnected bright regions
    # that are not the injection (e.g. a far-left blob) are dropped. Multiple
    # points allow a real injection made of several disconnected components.
    injection_seed_points: list = field(default_factory=list)
    # Split MERGED bright components (touching / neck-connected lobes) into
    # subcomponents with a distance-transform watershed BEFORE seed filtering, so
    # a seed-containing lobe can be kept while a touching non-seeded lobe is
    # removed. Only affects channels that configure injection_seed_points.
    split_merged_components: bool = True
    # Minimum distance (um) between watershed peak markers; controls how strong a
    # neck must be to split two lobes. Larger -> fewer splits.
    split_min_peak_distance_um: float = 100.0
    # Legacy audit value retained in run metadata for compatibility. It is never
    # used to re-keep a non-seeded watershed subcomponent: only subcomponents
    # containing or directly matched to configured seeds are retained.
    split_min_subcomponent_area_um2: float = 20000.0
    # Radius (um) within which a configured seed may be matched to a nearby bright
    # component/basin when the seed does not land exactly on a bright pixel. None
    # -> legacy behaviour (one low-res-pixel diagonal, ~downsample_um*sqrt(2)).
    seed_match_radius_um: float | None = None
    # HARD CAP on how far the kept region may extend from each seed. The kept mask
    # for a seed is intersected with all pixels within this distance of the seed,
    # so a single seed can NEVER retain an unbounded region (this is the primary
    # control that stops a seeded lobe from swallowing a whole component). Distance
    # is measured with ``seed_distance_metric``. None -> no cap (legacy).
    maximum_distance_from_seed_um: float | None = None
    # "geodesic" -> distance measured WITHIN the bright region (recommended: a seed
    # cannot reach across a thin isthmus). "euclidean" -> straight-line distance.
    seed_distance_metric: str = "geodesic"
    # Optional per-channel overrides (instances of this same config).
    green_signal: "InjectionExclusionConfig | None" = None
    channel_2_signal: "InjectionExclusionConfig | None" = None

    def for_channel(self, channel: str) -> "InjectionExclusionConfig":
        """Return the effective config for a channel (override if present)."""
        override = getattr(self, channel, None)
        return override if isinstance(override, InjectionExclusionConfig) else self

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "InjectionExclusionConfig":
        d = dict(d or {})
        green = d.pop("green_signal", None)
        ch2 = d.pop("channel_2_signal", None)
        base_values = _filtered(cls, d)
        cfg = cls(**base_values)
        if green is not None:
            cfg.green_signal = cls(**{**base_values, **_filtered(cls, green)})
        if ch2 is not None:
            cfg.channel_2_signal = cls(**{**base_values, **_filtered(cls, ch2)})
        return cfg


@dataclass
class TissueMaskConfig:
    """Shared, permissive foreground mask built from BOTH signal channels.

    Its only job is to remove the clearly-black background outside the specimen.
    It must NOT require tissue to be fluorescently labelled. For a crop wholly
    inside the brain, ``enabled: false`` is preferable to mis-labelling tissue.
    """

    enabled: bool = True
    downsample_um: float = 25.0
    smoothing_sigma_um: float = 20.0
    threshold_fraction: float = 0.08      # permissive: low/high robust fraction
    closing_um: float = 40.0
    minimum_area_um2: float = 200000.0


@dataclass
class CellfinderConfig:
    """Parameters forwarded to ``cellfinder.core.detect.detect.main``.

    Initial Cellfinder-like values, NOT validated final settings. Each channel
    may supply its own override (e.g. the red channel, ``channel_2_signal``, is
    weaker/more photobleached and its injection is far brighter, so it may need
    different thresholds). Any threshold change must be justified by manual-review
    or reference-cell recall -- never tuned toward an expected count. Defaults are
    preserved until tested.
    """

    soma_diameter_um: float = 16
    max_cluster_size_um3: float = 100000
    ball_xy_size_um: float = 6
    ball_z_size_um: float = 15
    ball_overlap_fraction: float = 0.6
    soma_spread_factor: float = 1.4
    log_sigma_size: float = 0.2
    n_sds_above_mean_thresh: float = 10
    n_sds_above_mean_tiled_thresh: float = 10
    tiled_thresh_tile_size: float | None = None
    artifact_keep: bool = True
    outlier_keep: bool = True
    batch_size: int = 1
    torch_device: str = "cuda"
    # If Cellfinder returns padded-volume z, subtract this to map back to the
    # 0..(planes-1) stack. Leave null unless padded coords are confirmed; an
    # out-of-range z is otherwise kept but flagged invalid and sent to review.
    cellfinder_z_padding_offset: float | None = None
    # Optional per-channel overrides (instances of this same config).
    green_signal: "CellfinderConfig | None" = None
    channel_2_signal: "CellfinderConfig | None" = None

    def for_channel(self, channel: str) -> "CellfinderConfig":
        """Return the effective Cellfinder config for a channel (override if any)."""
        override = getattr(self, channel, None)
        return override if isinstance(override, CellfinderConfig) else self

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CellfinderConfig":
        d = dict(d or {})
        green = d.pop("green_signal", None)
        ch2 = d.pop("channel_2_signal", None)
        base_values = _filtered(cls, d)
        cfg = cls(**base_values)
        if green is not None:
            cfg.green_signal = cls(**{**base_values, **_filtered(cls, green)})
        if ch2 is not None:
            cfg.channel_2_signal = cls(**{**base_values, **_filtered(cls, ch2)})
        return cfg


@dataclass
class CandidateScreeningConfig:
    """Channel-specific preliminary candidate-screening thresholds."""

    minimum_component_xy_area_um2: float = 0.0
    minimum_component_volume_um3: float = 0.0
    minimum_support_planes: int = 0
    minimum_supporting_voxels: int = 0
    minimum_signal_to_background_ratio: float = 0.0


@dataclass
class DetectionConfig:
    backend: str = "pilot_log3d"  # "pilot_log3d" or "cellfinder_candidates"
    minimum_cell_diameter_um: float = 6
    maximum_cell_diameter_um: float = 30

    # Background flattening / blob scales (pilot_log3d backend).
    background_sigma_um: float = 40.0
    log_sigma_min_um: float = 2.0
    log_sigma_max_um: float = 8.0

    # Local significance.
    minimum_local_robust_z: float = 6.0
    z_support_min_contrast: float = 3.0
    central_region_radius_um: float = 3.0
    background_annulus_inner_um: float = 8.0
    background_annulus_outer_um: float = 16.0
    minimum_background_pixels: int = 20
    padding_values: list[float] = field(default_factory=lambda: [0.0])

    # --- Preliminary-pass post-detection requirements (Part 2) ----------------
    # Every applicable rule below must hold for a candidate to be a preliminary
    # pass. These are NOT cell criteria and are NOT tuned toward a target count;
    # they are conservative post-detection gates on measured morphology/support.
    # 0 / None disables the individual gate (back-compatible default).
    minimum_component_xy_area_um2: float = 0.0     # min fixed-XY footprint area
    minimum_component_volume_um3: float = 0.0      # min 3D volume
    minimum_support_planes: int = 0                # min fixed-XY supported planes
    minimum_supporting_voxels: int = 0             # min supporting voxel count
    # Named signal-to-local-background gate (robust z = (peak-bg)/noise). Applied
    # in ADDITION to minimum_local_robust_z; the stricter of the two governs.
    minimum_signal_to_background_ratio: float = 0.0
    # A candidate clipped by the crop/tissue edge is NOT rejected for the edge
    # alone when its centre is inside tissue and its measurement is valid (it is
    # measured from the clipped valid pixels). Set false for legacy edge dropping.
    keep_edge_clipped_if_center_in_tissue: bool = True

    # 3D / consecutive-plane support.
    minimum_consecutive_planes: int = 2
    maximum_consecutive_planes: int = 6
    maximum_xy_centroid_shift_um: float = 5.0
    merge_distance_xy_um: float = 8.0
    merge_distance_z_um: float = 12.0
    minimum_candidate_separation_um: float = 6.0

    # Morphology.
    maximum_elongation: float = 3.0

    # Crop boundary handling.
    exclude_crop_boundary_objects: bool = True
    crop_boundary_margin_um: float = 15.0

    # Strong single-plane objects go to manual review rather than being dropped;
    # the strongest of them (>= single_plane_pass_min_robust_z) pass outright so
    # they never reach review.
    single_plane_manual_review: bool = True
    single_plane_review_min_robust_z: float = 8.0
    single_plane_pass_min_robust_z: float = 12.0

    # Sub-configs.
    tissue_mask: TissueMaskConfig = field(default_factory=TissueMaskConfig)
    injection_exclusion: InjectionExclusionConfig = field(default_factory=InjectionExclusionConfig)
    cellfinder: CellfinderConfig = field(default_factory=CellfinderConfig)
    # Optional per-channel overrides for the five preliminary screening gates.
    green_signal: CandidateScreeningConfig | None = None
    channel_2_signal: CandidateScreeningConfig | None = None

    # Retained for the provisional cross-channel overlap step (summarize).
    overlap_distance_um: float = 8

    def screening_for_channel(self, channel: str) -> CandidateScreeningConfig:
        """Return effective preliminary-screening settings for ``channel``."""
        override = getattr(self, channel, None)
        if isinstance(override, CandidateScreeningConfig):
            return override
        return CandidateScreeningConfig(
            minimum_component_xy_area_um2=self.minimum_component_xy_area_um2,
            minimum_component_volume_um3=self.minimum_component_volume_um3,
            minimum_support_planes=self.minimum_support_planes,
            minimum_supporting_voxels=self.minimum_supporting_voxels,
            minimum_signal_to_background_ratio=self.minimum_signal_to_background_ratio,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DetectionConfig":
        d = dict(d or {})
        inj = d.pop("injection_exclusion", None)
        tissue = d.pop("tissue_mask", None)
        cf = d.pop("cellfinder", None)
        green = d.pop("green_signal", None)
        ch2 = d.pop("channel_2_signal", None)
        cfg = cls(**_filtered(cls, d))
        if inj is not None:
            cfg.injection_exclusion = InjectionExclusionConfig.from_dict(inj)
        if tissue is not None:
            cfg.tissue_mask = TissueMaskConfig(**_filtered(TissueMaskConfig, tissue))
        if cf is not None:
            cfg.cellfinder = CellfinderConfig.from_dict(cf)
        base = cfg.screening_for_channel("")
        base_values = {
            f.name: getattr(base, f.name) for f in _dc_fields(CandidateScreeningConfig)
        }
        if green is not None:
            cfg.green_signal = CandidateScreeningConfig(
                **{**base_values, **_filtered(CandidateScreeningConfig, green)}
            )
        if ch2 is not None:
            cfg.channel_2_signal = CandidateScreeningConfig(
                **{**base_values, **_filtered(CandidateScreeningConfig, ch2)}
            )
        return cfg


@dataclass
class ClassifierConfig:
    patch_size_xy_um: float = 50.0
    minimum_cells: int = 50
    minimum_artifacts: int = 50
    cell_probability_threshold: float = 0.80
    artifact_probability_threshold: float = 0.20
    validated: bool = False
    group_by: str = "spatial_tile"
    spatial_tile_size_px: int = 512
    validation_fraction: float = 0.20
    random_seed: int = 20260625
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 0.001
    num_workers: int = 0


@dataclass
class QcDisplaySettings:
    """Display (brightness/contrast) window for ONE channel's QC figures.

    This is a *display* window only. It NEVER alters raw image values, Cellfinder
    input, background/contrast measurements, classifier patches or CSV numbers.

    mode:
      * ``fixed``                    -- use ``minimum``/``maximum`` verbatim
                                        (reproduces a Fiji 0-513 view).
      * ``robust_tissue_percentile`` -- percentiles of finite, in-tissue,
                                        non-background pixels (injection core
                                        optionally excluded for the upper limit).
      * ``full_data_percentile``     -- percentiles of every finite pixel.
    """

    mode: str = "robust_tissue_percentile"
    lower_percentile: float = 0.5
    upper_percentile: float = 99.7
    minimum: float = 0.0
    maximum: float = 513.0


@dataclass
class QcDisplayConfig:
    enabled: bool = True
    # Exclude the conservative injection core when estimating the robust UPPER
    # display limit so the saturated injection cannot blow out tissue contrast.
    exclude_injection_core: bool = True
    # Minimum usable pixel pool before a robust window is trusted; below this the
    # estimate safely falls back to full-data percentiles, then raw min/max.
    minimum_pixels: int = 50
    default: QcDisplaySettings = field(default_factory=QcDisplaySettings)
    green_signal: "QcDisplaySettings | None" = None
    channel_2_signal: "QcDisplaySettings | None" = None

    def for_channel(self, channel: str) -> QcDisplaySettings:
        override = getattr(self, channel, None)
        return override if isinstance(override, QcDisplaySettings) else self.default

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "QcDisplayConfig":
        d = dict(d or {})
        default = d.pop("default", None)
        green = d.pop("green_signal", None)
        ch2 = d.pop("channel_2_signal", None)
        cfg = cls(**_filtered(cls, d))
        if default is not None:
            cfg.default = QcDisplaySettings(**_filtered(QcDisplaySettings, default))
        if green is not None:
            cfg.green_signal = QcDisplaySettings(**_filtered(QcDisplaySettings, green))
        if ch2 is not None:
            cfg.channel_2_signal = QcDisplaySettings(**_filtered(QcDisplaySettings, ch2))
        return cfg


@dataclass
class CandidateRecallConfig:
    xy_tolerance_um: float = 8.0
    z_tolerance_um: float = 12.0


@dataclass
class RadialAnalysisConfig:
    """Radial candidate distance/density analysis around the injection centre.

    ``center_source``:
      * ``manual``                  -- use ``manual_center_xy_px`` [x, y] (full-res px).
      * ``injection_core_centroid`` -- centroid of the validated injection core.
    A failed / unvalidated automatic core centroid is used only with a warning.
    """

    enabled: bool = False
    center_source: str = "manual"           # "manual" or "injection_core_centroid"
    manual_center_xy_px: list | None = None  # [x, y] full-res px
    bin_width_um: float = 100.0
    maximum_radius_um: float | None = None
    channel: str = "green_signal"


@dataclass
class PairCorrelationConfig:
    """Automatic post-run candidate-to-candidate pair-correlation reports.

    A candidate is never counted twice: one XY coordinate per candidate row. This
    is display/analysis only -- it never changes a candidate, status, mask or count.
    """

    enabled: bool = True
    maximum_distance_um: float = 500.0
    simulations: int = 99
    random_seed: int = 20260713
    bin_width_um: float = 5.0
    minimum_candidates: int = 2
    # Crop runs are skipped by default: the crop boundary artificially truncates
    # neighbour distances and biases clustering. Set true only for a deliberate
    # within-crop analysis.
    include_cropped_runs: bool = False


@dataclass
class CandidateSizeDistributionsConfig:
    """Legacy candidate size-distribution plots. OFF by default (superseded by the
    pair-correlation reports); enable only if you explicitly want them again."""

    enabled: bool = False


@dataclass
class PostrunSpatialAnalysisConfig:
    """Post-run spatial analysis, run automatically by run_candidate_pilot.py."""

    enabled: bool = True
    pair_correlation: PairCorrelationConfig = field(default_factory=PairCorrelationConfig)
    candidate_size_distributions: CandidateSizeDistributionsConfig = field(
        default_factory=CandidateSizeDistributionsConfig
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PostrunSpatialAnalysisConfig":
        d = dict(d or {})
        pair = d.pop("pair_correlation", None)
        sizes = d.pop("candidate_size_distributions", None)
        cfg = cls(**_filtered(cls, d))
        if pair is not None:
            cfg.pair_correlation = PairCorrelationConfig(
                **_filtered(PairCorrelationConfig, pair)
            )
        if sizes is not None:
            cfg.candidate_size_distributions = CandidateSizeDistributionsConfig(
                **_filtered(CandidateSizeDistributionsConfig, sizes)
            )
        return cfg


@dataclass
class ChannelOverlayConfig:
    """Cross-channel (green vs red) overlay analysis run after candidate detection.

    For every candidate the SAME fixed-XY seven-plane measurement is taken in BOTH
    biological channels (green ``green_signal`` and red ``channel_2_signal``) so a
    candidate can be labelled green-stronger, red-stronger, present in both, or
    unclear. This is DISPLAY/AUDIT only: it never changes a candidate, status,
    mask, count, or the raw images, and it never forces the red channel to have
    fewer detections -- the label comes from actual measured signal.
    """

    enabled: bool = True
    # A channel is "present" at a candidate when its robust-z signal-to-background
    # meets this floor (mirrors detection.z_support_min_contrast, "supported plane").
    snr_threshold: float = 3.0
    # One channel is "dominant" only when its background-subtracted peak exceeds the
    # other channel's by at least this factor; otherwise (both present) -> "both".
    dominance_ratio: float = 1.5
    # Longest side (px) of the downscaled green/red composite QC thumbnail.
    qc_max_dim: int = 2000


@dataclass
class ChannelComparisonConfig:
    """Defaults for opt-in green/red provisional-candidate comparison reports.

    These values are passed explicitly by ``run_candidate_pilot.py`` when the
    optional comparison stage is requested.  Standalone apply mode still
    requires thresholds to be supplied explicitly on its command line.
    """

    default_min_dominance_ratio: float = 1.5
    default_min_snr: float = 3.0
    default_max_match_distance_um: float = 10.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    candidate_recall: CandidateRecallConfig = field(default_factory=CandidateRecallConfig)
    qc_display: QcDisplayConfig = field(default_factory=QcDisplayConfig)
    radial_analysis: RadialAnalysisConfig = field(default_factory=RadialAnalysisConfig)
    postrun_spatial_analysis: PostrunSpatialAnalysisConfig = field(
        default_factory=PostrunSpatialAnalysisConfig
    )
    channel_overlay: ChannelOverlayConfig = field(default_factory=ChannelOverlayConfig)
    channel_comparison: ChannelComparisonConfig = field(
        default_factory=ChannelComparisonConfig
    )
    source_path: str | None = None
    # Warnings about the loaded YAML (unknown keys, stale/older copies). Filled in
    # by load_config; printed at startup and written to the run metadata.
    config_warnings: list = field(default_factory=list)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict[str, Any], source_path: str | None = None) -> "Config":
        d = d or {}
        return cls(
            data=DataConfig(**_filtered(DataConfig, d.get("data"))),
            acquisition=AcquisitionConfig(**_filtered(AcquisitionConfig, d.get("acquisition"))),
            pilot=PilotConfig(**_filtered(PilotConfig, d.get("pilot"))),
            registration=RegistrationConfig(**_filtered(RegistrationConfig, d.get("registration"))),
            detection=DetectionConfig.from_dict(d.get("detection")),
            classifier=ClassifierConfig(**_filtered(ClassifierConfig, d.get("classifier"))),
            candidate_recall=CandidateRecallConfig(
                **_filtered(CandidateRecallConfig, d.get("candidate_recall"))
            ),
            qc_display=QcDisplayConfig.from_dict(d.get("qc_display")),
            radial_analysis=RadialAnalysisConfig(
                **_filtered(RadialAnalysisConfig, d.get("radial_analysis"))
            ),
            postrun_spatial_analysis=PostrunSpatialAnalysisConfig.from_dict(
                d.get("postrun_spatial_analysis")
            ),
            channel_overlay=ChannelOverlayConfig(
                **_filtered(ChannelOverlayConfig, d.get("channel_overlay"))
            ),
            channel_comparison=ChannelComparisonConfig(
                **_filtered(ChannelComparisonConfig, d.get("channel_comparison"))
            ),
            source_path=source_path,
        )

    @property
    def work_dir(self) -> Path:
        return Path(self.data.work_dir).expanduser()


# Which sections have sub-sections that are themselves config dicts. Used to
# walk the raw YAML and flag keys that the parser would silently drop.
_NESTED_SCHEMA = {
    Config: {
        "data": DataConfig, "acquisition": AcquisitionConfig, "pilot": PilotConfig,
        "registration": RegistrationConfig, "detection": DetectionConfig,
        "classifier": ClassifierConfig, "candidate_recall": CandidateRecallConfig,
        "qc_display": QcDisplayConfig, "radial_analysis": RadialAnalysisConfig,
        "postrun_spatial_analysis": PostrunSpatialAnalysisConfig,
        "channel_overlay": ChannelOverlayConfig,
        "channel_comparison": ChannelComparisonConfig,
    },
    PostrunSpatialAnalysisConfig: {
        "pair_correlation": PairCorrelationConfig,
        "candidate_size_distributions": CandidateSizeDistributionsConfig,
    },
    DetectionConfig: {
        "tissue_mask": TissueMaskConfig,
        "injection_exclusion": InjectionExclusionConfig,
        "cellfinder": CellfinderConfig,
        "green_signal": CandidateScreeningConfig,
        "channel_2_signal": CandidateScreeningConfig,
    },
    InjectionExclusionConfig: {
        "green_signal": InjectionExclusionConfig,
        "channel_2_signal": InjectionExclusionConfig,
    },
    CellfinderConfig: {
        "green_signal": CellfinderConfig, "channel_2_signal": CellfinderConfig,
    },
    QcDisplayConfig: {
        "default": QcDisplaySettings, "green_signal": QcDisplaySettings,
        "channel_2_signal": QcDisplaySettings,
    },
}


def _walk_unknown(cls, raw, prefix, out):
    """Collect YAML keys that are not real fields of the matching dataclass."""
    if not isinstance(raw, dict):
        return
    allowed = {f.name for f in _dc_fields(cls)}
    nested = _NESTED_SCHEMA.get(cls, {})
    for key, value in raw.items():
        if key not in allowed:
            out.append(f"{prefix}{key}")
        elif key in nested:
            _walk_unknown(nested[key], value, f"{prefix}{key}.", out)


def unknown_config_keys(raw: dict) -> list[str]:
    """Keys present in the YAML that the parser would ignore (typo / wrong place)."""
    out: list[str] = []
    _walk_unknown(Config, raw or {}, "", out)
    return out


def schema_drift_warnings(raw: dict) -> list[str]:
    """Flag a config that looks like an older copy missing newer fields."""
    raw = raw or {}
    warnings: list[str] = []
    if "qc_display" not in raw:
        warnings.append(
            "config has no 'qc_display' section -- the red channel "
            "(channel_2_signal) will NOT use the fixed 0-513 window (falling back "
            "to robust percentiles)."
        )
    injection = (raw.get("detection") or {}).get("injection_exclusion") or {}
    if "generation_suppression_enabled" not in injection:
        warnings.append(
            "detection.injection_exclusion has no 'generation_suppression_enabled' "
            "-- the injection-suppressed second pass will be OFF (older config?)."
        )
    return warnings


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file.

    PyYAML is imported lazily so the pure-stdlib modules stay importable without it.
    Unknown keys and stale-copy drift are recorded on ``config.config_warnings``;
    the caller is expected to print them (we never silently drop settings).
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Copy config.example.yml to config.yml and edit the placeholders."
        )
    try:
        import yaml  # noqa: PLC0415  (lazy import by design)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to read the config file. Install the project "
            "dependencies (pip install -e .) or `pip install pyyaml`."
        ) from exc

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    config = Config.from_dict(raw, source_path=str(path))
    config.config_warnings = [
        f"unknown config key ignored: {key}" for key in unknown_config_keys(raw)
    ] + schema_drift_warnings(raw)
    return config
