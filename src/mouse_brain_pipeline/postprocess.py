"""Post-process an existing candidate run without rerunning Cellfinder.

Cellfinder detection is the slow (~tens of minutes) stage. When only the
injection-mask override, QC, coordinate exports or radial charts change, we can
reuse the already-detected candidates: recompute mask membership + statuses,
re-render, and write a brand-new isolated run. The source run is never touched.

Nothing here imports or calls the Cellfinder detector.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .candidate_detection import (
    REASON_INJECTION,
    REASON_SUSPECT_INJECTION,
    STATUS_ARTIFACT,
    STATUS_INJECTION,
    STATUS_PRELIMINARY_FAIL,
    STATUS_SUSPECT_INJECTION,
    SectionDetectionResult,
    rasterize_polygons,
    write_candidate_tables,
)
from .candidate_qc import write_mask_diagnostics, write_status_summary
from .channels import channel_display_name
from .coordinate_exports import write_coordinate_exports, write_count_summaries
from .injection_overrides import (
    apply_overrides_to_injection_cfg,
    load_overrides,
    overrides_hash,
    save_mask_comparison_png,
)
from .review import read_csv_rows
from .run_layout import code_version, config_hash, create_run_dir, write_latest_run
from .timing import StageTimer
from .utilities import LOG, ensure_dir


def _row_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"true", "1", "yes"}


def _eligible_for_injection(row) -> bool:
    """A candidate can only be assigned to the injection mask if it passed the
    same earlier gates the detector applies (valid coord/z/measurement, not an
    artefact)."""
    if _row_bool(row.get("invalid_coordinate")):
        return False
    if not _row_bool(row.get("original_cellfinder_z_valid"), default=True):
        return False
    if not _row_bool(row.get("measurement_valid")):
        return False
    return row.get("current_status") != STATUS_ARTIFACT


def restatus_row(row, inside_analysis: bool, inside_core: bool) -> tuple[str, str]:
    """Recompute one candidate's mask membership + status after a mask change.

    Returns (old_status, new_status). A candidate that leaves the injection mask
    returns to its preliminary interpretation; artefacts / invalid measurements
    are never turned into injection.
    """
    old = row.get("current_status", "")
    row["inside_injection_core"] = bool(inside_core)
    row["inside_injection_site"] = bool(inside_analysis)
    row["inside_injection_analysis_exclusion"] = bool(inside_analysis)
    was_injection = old in (STATUS_INJECTION, STATUS_SUSPECT_INJECTION)

    if inside_analysis:
        if _eligible_for_injection(row):
            source = row.get("injection_mask_source", "")
            validated = _row_bool(row.get("injection_mask_validated"))
            qc_failed = _row_bool(row.get("injection_mask_qc_failed"))
            if source == "manual_geometry" or (validated and not qc_failed):
                row["current_status"] = STATUS_INJECTION
                row["rejection_reason"] = REASON_INJECTION
            else:
                row["current_status"] = STATUS_SUSPECT_INJECTION
                row["rejection_reason"] = REASON_SUSPECT_INJECTION
        # else: keep the existing (artefact / invalid) status.
    elif was_injection:
        # Left the mask -> back to normal candidate interpretation.
        row["current_status"] = row.get("preliminary_sampling_category") or STATUS_PRELIMINARY_FAIL
        row["rejection_reason"] = row.get("preliminary_rule_reason", "")
    return old, row.get("current_status", "")


# --------------------------------------------------------------------------- #
def _load_mask(run_dir: Path, channel: str, section: int, name: str):
    import numpy as np  # noqa: PLC0415

    path = run_dir / "qc" / f"{channel}_section_{int(section):03d}" / name
    if not path.is_file():
        return None
    return np.load(path)


def _display_background(run_dir: Path, channel: str, section: int, shape):
    """A grey display background for the before/after QC (from the saved PNG)."""
    import numpy as np  # noqa: PLC0415

    png = (run_dir / "qc" / f"{channel}_section_{int(section):03d}"
           / "02_raw_projection_display_fullres.png")
    if png.is_file():
        try:
            from PIL import Image  # noqa: PLC0415

            return np.asarray(Image.open(png).convert("L"))
        except Exception:  # pragma: no cover
            pass
    return np.zeros(shape, dtype="uint8")


def apply_override_to_masks(core, analysis, inj_cfg, crop_origin):
    """Apply manual additions then non-injection subtraction to saved masks.

    Additions are the polygon as-is (no re-dilation); subtraction is applied
    LAST so it cannot be dilated back. Returns (core2, analysis2, added, removed).
    """
    import numpy as np  # noqa: PLC0415

    shape = analysis.shape
    added = rasterize_polygons(shape, getattr(inj_cfg, "manual_polygons", None), crop_origin)
    removed = rasterize_polygons(
        shape, getattr(inj_cfg, "manual_non_injection_polygons", None), crop_origin)
    core2 = (np.asarray(core, dtype=bool) | added) & ~removed
    analysis2 = (np.asarray(analysis, dtype=bool) | added) & ~removed
    return core2, analysis2, added, removed


# --------------------------------------------------------------------------- #
def postprocess_run(*, config, source_run_dir, new_run_name, work_dir=None,
                    overrides_path=None, timer=None):
    """Create a new run from an existing one with an updated injection override.

    Recomputes mask membership + statuses and writes fresh CSVs / mask QC. Does
    NOT rerun Cellfinder and never edits the source run. Returns a result dict
    (new run dir, per-group status changes, mask QC paths, timer).
    """
    import numpy as np  # noqa: PLC0415

    timer = timer or StageTimer()
    source_run_dir = Path(source_run_dir)
    work_dir = Path(work_dir) if work_dir is not None else config.work_dir

    meta_path = source_run_dir / "candidate_run_metadata.json"
    csv_path = source_run_dir / "all_candidates.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(
            f"Source run must contain candidate_run_metadata.json and "
            f"all_candidates.csv: {source_run_dir}")
    source_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    with timer.stage("load_source_candidates"):
        rows = read_csv_rows(csv_path)

    overrides = load_overrides(overrides_path)
    inj_cfg = apply_overrides_to_injection_cfg(
        config.detection.injection_exclusion, overrides)

    # Refuse to overwrite an existing non-empty run.
    run_dir = create_run_dir(work_dir, new_run_name)
    qc_dir = ensure_dir(run_dir / "qc")

    crop = source_meta.get("crop_x_min_x_max_y_min_y_max")
    crop_origin = (int(crop[2]), int(crop[0])) if crop else (0, 0)

    # Group candidate rows by (channel, section).
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row.get("channel", ""), str(row.get("section", ""))), []).append(row)

    status_changes: list[dict] = []
    mask_qc_paths: list[str] = []
    with timer.stage("mask_processing_and_restatus"):
        for (channel, section_str), grp in groups.items():
            try:
                section = int(section_str)
            except (TypeError, ValueError):
                section = 0
            channel_cfg = inj_cfg.for_channel(channel)
            core = _load_mask(source_run_dir, channel, section, "injection_core_mask.npy")
            analysis = _load_mask(
                source_run_dir, channel, section, "injection_analysis_exclusion_mask.npy")
            if analysis is None:
                LOG.warning("No saved analysis mask for %s section %d -- statuses "
                            "left unchanged for this group.", channel, section)
                continue
            if core is None:
                core = np.zeros_like(analysis)

            core2, analysis2, added, removed = apply_override_to_masks(
                core, analysis, channel_cfg, crop_origin)

            # Save corrected masks into the new run so downstream steps reuse them.
            section_dir = ensure_dir(qc_dir / f"{channel}_section_{section:03d}")
            np.save(section_dir / "injection_core_mask.npy", core2.astype(bool))
            np.save(section_dir / "injection_analysis_exclusion_mask.npy", analysis2.astype(bool))
            tissue = _load_mask(source_run_dir, channel, section, "tissue_mask.npy")
            if tissue is not None:
                np.save(section_dir / "tissue_mask.npy", np.asarray(tissue, dtype=bool))

            H, W = analysis2.shape
            changed = 0
            for row in grp:
                try:
                    yl = int(float(row.get("y_local_px")))
                    xl = int(float(row.get("x_local_px")))
                except (TypeError, ValueError):
                    continue
                if not (0 <= yl < H and 0 <= xl < W):
                    continue
                old, new = restatus_row(row, bool(analysis2[yl, xl]), bool(core2[yl, xl]))
                if old != new:
                    changed += 1
            status_changes.append({
                "channel": channel, "section": section,
                "candidates": len(grp), "status_changed": changed,
                "removed_polygon_pixels": int(removed.sum()),
                "added_polygon_pixels": int(added.sum()),
            })

            # Before/after mask QC.
            background = _display_background(source_run_dir, channel, section, analysis2.shape)
            before_after = save_mask_comparison_png(
                section_dir / "injection_mask_before_after.png",
                background, analysis, analysis2,
                removed_mask=removed, added_mask=added,
                title=(f"{channel} section {section:03d} -- injection analysis mask "
                       "before (orange) vs after override (red)\nPROVISIONAL candidates"))
            mask_qc_paths.append(str(before_after))
            save_mask_comparison_png(
                section_dir / "injection_core_before_after.png",
                background, core, core2,
                removed_mask=removed, added_mask=added,
                title=(f"{channel} section {section:03d} -- injection CORE before vs "
                       "after override\nPROVISIONAL candidates"))

    # Write corrected tables / summaries (new run only).
    results = [
        SectionDetectionResult(channel=ch, section=int(sec) if sec.isdigit() else 0,
                               candidates=grp)
        for (ch, sec), grp in groups.items()
    ]
    with timer.stage("csv_writing"):
        paths = write_candidate_tables(run_dir, results)
        write_status_summary(run_dir, results)
        write_mask_diagnostics(run_dir, results)
        ppl = int(source_meta.get("acquisition", {}).get("planes_per_section", 7))
        for res in results:
            scope = ensure_dir(run_dir / "coordinate_exports" / res.channel)
            write_coordinate_exports(scope, res.candidates, channel=res.channel,
                                     section=res.section, planes_per_section=ppl)
            write_count_summaries(scope, res.candidates, channel=res.channel,
                                  section=res.section, planes_per_section=ppl)

    all_candidates = [c for r in results for c in r.candidates]
    status_counts = dict(Counter(c.get("current_status") for c in all_candidates))

    # New metadata: carry the source fields the renderer checks, add provenance.
    metadata = dict(source_meta)
    metadata.update({
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": new_run_name,
        "run_dir": str(run_dir),
        "postprocessed": True,
        "cellfinder_rerun": False,
        "source_run_dir": str(source_run_dir),
        "injection_overrides_path": str(overrides_path) if overrides_path else None,
        "injection_overrides_hash": overrides_hash(overrides_path),
        "code_version": code_version(),
        "config_path": config.source_path,
        "config_hash": config_hash(config.source_path),
        "status_counts": status_counts,
        "status_changes_by_group": status_changes,
        "channel_display_names": {
            channel: channel_display_name(channel)
            for channel in {r.channel for r in results}
        },
    })
    (run_dir / "candidate_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    write_latest_run(work_dir, run_dir, {
        "run_name": new_run_name,
        "run_timestamp_utc": metadata["run_timestamp_utc"],
        "candidate_count": len(all_candidates),
        "status_counts": status_counts,
        "postprocessed": True,
        "source_run_dir": str(source_run_dir),
    })

    return {
        "run_dir": run_dir,
        "all_candidates_csv": paths["all"],
        "status_changes": status_changes,
        "mask_qc_paths": mask_qc_paths,
        "status_counts": status_counts,
        "results": results,
        "metadata": metadata,
        "timer": timer,
    }
