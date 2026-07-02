"""Dataset audit: discover, parse, pair, validate and manifest the TIFFs.

Outputs (written into ``work_dir/audit``):
  * ``manifest.csv``        -- one row per (section, plane)
  * ``missing_files.csv``   -- missing planes / unpaired / duplicate problems
  * ``dataset_summary.json``-- machine-readable summary
  * ``audit.log``           -- full log

The audit NEVER opens pixel data wholesale: TIFF shape/dtype come from headers
only (``tifffile`` page metadata). It is read-only with respect to the dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import BACKGROUND, CHANNEL_2_SIGNAL, GREEN_SIGNAL
from .config import Config, load_config
from .filenames import (
    compile_regex,
    expected_planes,
    global_plane,
    parse_filename,
    validate_contiguous,
    z_um,
)
from .utilities import LOG, ensure_dir, has_space_in_path, setup_logging

MANIFEST_COLUMNS = [
    "section",
    "plane",
    "global_plane",
    "z_um",
    "green_signal_file",
    "channel_2_signal_file",
    "background_file",
    "green_shape",
    "channel_2_shape",
    "dtype",
    "pair_valid",
    "notes",
]


@dataclass
class ChannelFile:
    section: int
    plane: int
    path: Path


@dataclass
class ChannelIndex:
    """Parsed view of one channel directory."""

    name: str
    directory: Path | None
    files: dict[tuple[int, int], Path] = field(default_factory=dict)
    duplicates: dict[tuple[int, int], list[Path]] = field(default_factory=dict)
    unparseable: list[Path] = field(default_factory=list)

    @property
    def sections(self) -> set[int]:
        return {s for (s, _p) in self.files}


# --------------------------------------------------------------------------- #
# Lazy TIFF header reading
# --------------------------------------------------------------------------- #
def read_shape_dtype(path: Path) -> tuple[tuple[int, ...] | None, str | None]:
    """Return (shape, dtype) from the TIFF header only -- no pixel data loaded.

    Returns (None, None) if tifffile is unavailable or the header cannot be read.
    """
    try:
        import tifffile  # noqa: PLC0415
    except ImportError:
        return None, None
    try:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            return tuple(int(x) for x in page.shape), str(page.dtype)
    except Exception as exc:  # pragma: no cover - corrupt/unreadable file
        LOG.warning("Could not read TIFF header for %s: %s", path, exc)
        return None, None


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def index_channel(name: str, directory: str | Path | None, regex_pattern: str) -> ChannelIndex:
    """Recursively discover and parse TIFFs in one channel directory."""
    idx = ChannelIndex(name=name, directory=Path(directory) if directory else None)
    if not directory:
        return idx
    root = Path(directory).expanduser()
    if not root.is_dir():
        LOG.error("Channel %s directory does not exist: %s", name, root)
        return idx

    pat = compile_regex(regex_pattern)
    seen: dict[tuple[int, int], Path] = {}
    for dirpath, _dirs, fnames in os.walk(root):
        for fname in fnames:
            if not fname.lower().endswith((".tif", ".tiff")):
                continue
            full = Path(dirpath) / fname
            parsed = parse_filename(fname, pat)
            if parsed is None:
                idx.unparseable.append(full)
                continue
            key = (parsed.section, parsed.plane)
            if key in seen:
                idx.duplicates.setdefault(key, [seen[key]]).append(full)
            else:
                seen[key] = full
    idx.files = seen
    return idx


# --------------------------------------------------------------------------- #
# Audit result
# --------------------------------------------------------------------------- #
@dataclass
class AuditResult:
    summary: dict
    manifest_rows: list[dict]
    missing_rows: list[dict]
    errors: list[str]
    warnings: list[str]

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0


def _section_plane_keys(*indexes: ChannelIndex) -> list[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for idx in indexes:
        keys.update(idx.files.keys())
    return sorted(keys)


def run_audit(config: Config, check_metadata: bool = True, dry_run: bool = False) -> AuditResult:
    """Run the full audit and (unless dry_run) write output files."""
    out_dir = ensure_dir(config.work_dir / "audit")
    if not dry_run:
        setup_logging(out_dir / "audit.log", verbose=False)
    LOG.info("Starting dataset audit (dry_run=%s)", dry_run)

    regex = config.data.filename_regex
    ppl = config.acquisition.planes_per_section

    green = index_channel(GREEN_SIGNAL, config.data.green_signal_dir, regex)
    ch2 = index_channel(CHANNEL_2_SIGNAL, config.data.channel_2_signal_dir, regex)
    bg = index_channel(BACKGROUND, config.data.background_dir, regex)

    errors: list[str] = []
    warnings: list[str] = []

    for idx in (green, ch2, bg):
        if idx.directory is None:
            continue
        if not idx.directory.is_dir():
            if idx.name != BACKGROUND:
                errors.append(f"{idx.name} directory missing: {idx.directory}")
            continue
        if not idx.files:
            warnings.append(f"{idx.name}: no parseable TIFFs found in {idx.directory}")
        if idx.duplicates:
            for key, paths in idx.duplicates.items():
                errors.append(
                    f"{idx.name}: duplicate plane section={key[0]} plane={key[1]} -> "
                    + ", ".join(str(p.name) for p in paths)
                )
        if idx.unparseable:
            warnings.append(
                f"{idx.name}: {len(idx.unparseable)} unparseable TIFF filename(s) "
                f"(e.g. {idx.unparseable[0].name})"
            )

    # Section contiguity (only meaningful for the primary signal channels).
    all_sections = sorted(green.sections | ch2.sections)
    contiguous, missing_sections = validate_contiguous(all_sections)
    if not contiguous:
        warnings.append(
            f"Sections are NOT contiguous; missing section numbers: {missing_sections}. "
            "Relative-Z is only valid within contiguous runs."
        )
    first_section = all_sections[0] if all_sections else 0

    # Per-section plane completeness.
    missing_rows: list[dict] = []
    wanted = set(expected_planes(ppl))
    for section in all_sections:
        for chan in (green, ch2):
            if chan.directory is None:
                continue
            present = {p for (s, p) in chan.files if s == section}
            missing = sorted(wanted - present)
            extra = sorted(present - wanted)
            for p in missing:
                missing_rows.append(
                    {"section": section, "plane": p, "channel": chan.name, "problem": "missing_plane"}
                )
                errors.append(f"{chan.name}: section {section} missing plane {p:02d}")
            for p in extra:
                missing_rows.append(
                    {"section": section, "plane": p, "channel": chan.name, "problem": "unexpected_plane"}
                )
                warnings.append(
                    f"{chan.name}: section {section} has unexpected plane {p:02d} "
                    f"(outside 1..{ppl})"
                )

    # Build manifest, pair channels, compare shapes/dtypes.
    manifest_rows: list[dict] = []
    keys = _section_plane_keys(green, ch2)
    pairs_checked = 0
    pairs_valid = 0
    for (section, plane) in keys:
        gp = global_plane(section, first_section, plane, ppl)
        zz = z_um(gp, config.acquisition.voxel_size_z_um)
        gfile = green.files.get((section, plane))
        cfile = ch2.files.get((section, plane))
        bfile = bg.files.get((section, plane))

        notes: list[str] = []
        gshape = gdtype = cshape = cdtype = None
        if check_metadata:
            if gfile is not None:
                gshape, gdtype = read_shape_dtype(gfile)
            if cfile is not None:
                cshape, cdtype = read_shape_dtype(cfile)

        pair_valid = True
        if gfile is None or cfile is None:
            pair_valid = False
            which = CHANNEL_2_SIGNAL if gfile is not None else GREEN_SIGNAL
            notes.append(f"unpaired: missing in {which}")
            errors.append(
                f"Unpaired plane section={section} plane={plane:02d}: "
                f"green={'yes' if gfile else 'NO'} channel_2={'yes' if cfile else 'NO'}"
            )
        else:
            pairs_checked += 1
            if check_metadata and gshape is not None and cshape is not None:
                if gshape != cshape:
                    pair_valid = False
                    notes.append(f"shape mismatch {gshape} vs {cshape}")
                    errors.append(
                        f"Shape mismatch section={section} plane={plane:02d}: "
                        f"green={gshape} channel_2={cshape}"
                    )
                if gdtype != cdtype:
                    pair_valid = False
                    notes.append(f"dtype mismatch {gdtype} vs {cdtype}")
                    errors.append(
                        f"dtype mismatch section={section} plane={plane:02d}: "
                        f"green={gdtype} channel_2={cdtype}"
                    )
            if pair_valid:
                pairs_valid += 1

        dtype = gdtype or cdtype
        manifest_rows.append(
            {
                "section": section,
                "plane": plane,
                "global_plane": gp,
                "z_um": round(zz, 4),
                "green_signal_file": str(gfile) if gfile else "",
                "channel_2_signal_file": str(cfile) if cfile else "",
                "background_file": str(bfile) if bfile else "",
                "green_shape": "x".join(map(str, gshape)) if gshape else "",
                "channel_2_shape": "x".join(map(str, cshape)) if cshape else "",
                "dtype": dtype or "",
                "pair_valid": pair_valid,
                "notes": "; ".join(notes),
            }
        )

    z_span = (manifest_rows[-1]["z_um"] - manifest_rows[0]["z_um"]) if manifest_rows else 0.0

    summary = {
        "channels": {
            GREEN_SIGNAL: {
                "directory": str(green.directory) if green.directory else None,
                "files": len(green.files),
                "duplicates": len(green.duplicates),
                "unparseable": len(green.unparseable),
            },
            CHANNEL_2_SIGNAL: {
                "directory": str(ch2.directory) if ch2.directory else None,
                "files": len(ch2.files),
                "duplicates": len(ch2.duplicates),
                "unparseable": len(ch2.unparseable),
            },
            BACKGROUND: {
                "directory": str(bg.directory) if bg.directory else None,
                "files": len(bg.files),
                "configured": config.data.has_background,
            },
        },
        "sections": {
            "count": len(all_sections),
            "first": first_section,
            "last": all_sections[-1] if all_sections else None,
            "contiguous": contiguous,
            "missing_sections": missing_sections,
        },
        "planes_per_section": ppl,
        "pairs": {"checked": pairs_checked, "valid": pairs_valid},
        "z_um_span": round(z_span, 3),
        "voxel_size_zyx_um": list(config.acquisition.voxel_size_zyx),
        "metadata_checked": check_metadata,
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "config_source": config.source_path,
    }

    if has_space_in_path(config.data.work_dir):
        warnings.append(
            f"work_dir contains a space: {config.data.work_dir!r}. Brainmapper/Cellfinder "
            "dislike spaces -- use a space-free path for the working directory."
        )

    result = AuditResult(
        summary=summary,
        manifest_rows=manifest_rows,
        missing_rows=missing_rows,
        errors=errors,
        warnings=warnings,
    )

    if not dry_run:
        _write_outputs(out_dir, result)
    _print_summary(result, out_dir, dry_run)
    return result


def _write_csv(path: Path, columns: Iterable[str], rows: list[dict]) -> None:
    cols = list(columns)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def _write_outputs(out_dir: Path, result: AuditResult) -> None:
    _write_csv(out_dir / "manifest.csv", MANIFEST_COLUMNS, result.manifest_rows)
    _write_csv(
        out_dir / "missing_files.csv",
        ["section", "plane", "channel", "problem"],
        result.missing_rows,
    )
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(result.summary, fh, indent=2)
    LOG.info("Wrote manifest.csv, missing_files.csv, dataset_summary.json to %s", out_dir)


def _print_summary(result: AuditResult, out_dir: Path, dry_run: bool) -> None:
    s = result.summary
    print("=" * 70)
    print("DATASET AUDIT SUMMARY" + ("  [DRY RUN - no files written]" if dry_run else ""))
    print("=" * 70)
    g = s["channels"][GREEN_SIGNAL]
    c = s["channels"][CHANNEL_2_SIGNAL]
    b = s["channels"][BACKGROUND]
    print(f"green_signal     : {g['files']:>5} files  ({g['directory']})")
    print(f"channel_2_signal : {c['files']:>5} files  ({c['directory']})")
    print(f"background       : {'configured' if b['configured'] else 'NOT configured (signal-only)'}")
    sec = s["sections"]
    print(
        f"sections         : {sec['count']} "
        f"({sec['first']}..{sec['last']}), contiguous={sec['contiguous']}"
    )
    print(f"planes/section   : {s['planes_per_section']}")
    print(f"pairs valid      : {s['pairs']['valid']} / {s['pairs']['checked']}")
    print(f"relative Z span  : {s['z_um_span']} um")
    print("-" * 70)
    print(f"errors: {len(result.errors)}   warnings: {len(result.warnings)}")
    for w in result.warnings[:20]:
        print(f"  [warn]  {w}")
    if len(result.warnings) > 20:
        print(f"  ... and {len(result.warnings) - 20} more warnings")
    for e in result.errors[:20]:
        print(f"  [ERROR] {e}")
    if len(result.errors) > 20:
        print(f"  ... and {len(result.errors) - 20} more errors")
    if not dry_run:
        print("-" * 70)
        print(f"Outputs written to: {out_dir}")
    print("=" * 70)
    if result.errors:
        print("RESULT: FAILED -- serious structure/pairing errors (exit 1).")
    else:
        print("RESULT: OK")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit a serial two-photon mouse-brain dataset.")
    p.add_argument("--config", "-c", default="config.yml", help="Path to config.yml")
    p.add_argument("--dry-run", action="store_true", help="Discover/validate but write nothing")
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip reading TIFF headers (faster; no shape/dtype comparison)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    setup_logging(None, verbose=args.verbose)
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    result = run_audit(config, check_metadata=not args.no_metadata, dry_run=args.dry_run)
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
